"""Composite flexural members of JTG 3362-2018 8.1 (组合式受弯构件): a precast girder that carries a cast-in-place
topping while it hardens, then acts with it (precast T girders, small box girders with a deck slab).

Units as femotion.codes.jtg.rc (mm, mm², MPa; kN, kN·m, sagging +). Clauses: docs/codes/JTG_3362-2018.md.

Stage 1 (8.1.2 item 1): the precast girder alone carries its own weight, the topping's and construction loads (M1).
Stage 2: the composite section carries the deck finishes and the live loads (M2). The precast girder is a section built
with jtg.rect / tee / box (tendons with σp0 and σpe as usual); jtg.composite adds the topping:

    from femotion.codes import jtg
    pre = jtg.tee(200, 1600, 1600, 160, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1450, 1150, 90)],
                  sigma_pe=1050)
    comp = jtg.composite(pre, b=2200, t=180, concrete="C40", top=[(14, 12, 50)])
    M1d, Md = jtg.composite_design_moments(M1Gk=1800, M1Qk=150, M2Gk=600, M2Qd=2600)        # 8.1.4
    jtg.flexure(comp.precast, M1d); jtg.flexure(comp.ultimate, Md, gamma0=1.1)
    jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="A")   # 8.1.8

The topping enters three ways (Composite): `ultimate` with its actual width and, for the ultimate limit state, the
topping's concrete (8.1.4: the cast-in-place class when the classes differ); `elastic` for stresses, cracking and
stiffness, the topping's width scaled by its modulus ratio to the precast concrete (8.1.8: W0 on the section transformed
to the precast concrete); `precast` alone for stage 1.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .materials import Concrete, Rebar, concrete as _concrete, rebar as _rebar
from .rc import Check, Circle, RCSection, Rect, _bars, flexure, shear
from .sls import _Ate, _bounds, _b_at, _cover, _diam, _pc_class, crack_width_limit, stiffness


def _shift(p, dz):
    if isinstance(p, Rect):
        return Rect(p.b, p.top + dz, p.bottom + dz, p.sign)
    return Circle(p.D, p.zc + dz, p.sign)


@dataclass
class Composite:
    precast: RCSection
    topping: Concrete
    b: float  # topping width (effective, 4.3.3)
    t: float  # topping thickness
    top_bars: np.ndarray  # (n, 3) area, depth from the composite top, diameter

    @property
    def h1(self) -> float:
        return self.precast.h

    @property
    def h(self) -> float:
        return self.precast.h + self.t

    @property
    def n(self) -> float:
        """modulus ratio of the topping to the precast concrete"""
        return self.topping.Ec / self.precast.concrete.Ec

    def _section(self, width: float, concrete: Concrete) -> RCSection:
        p = self.precast
        pieces = [Rect(width, 0.0, self.t)] + [_shift(q, self.t) for q in p.pieces]
        bars = np.r_[self.top_bars[:, :2], p.bars + [0.0, self.t]] if len(self.top_bars) else p.bars + [0.0, self.t]
        bar_d = np.r_[self.top_bars[:, 2], p.bar_d] if len(self.top_bars) else p.bar_d
        td = p.tendons.copy()
        td[:, 1] += self.t
        return dataclasses.replace(p, concrete=concrete, pieces=pieces, bars=bars, h=self.h, kind="composite", tendons=td,
                                   bar_d=bar_d, rules_concrete=p.concrete)

    @property
    def ultimate(self) -> RCSection:
        """the composite section for the ultimate limit state: actual widths, the topping's concrete (8.1.4); the minimum
        classes of 3.1.2 checked on the precast girder's concrete"""
        c = self.topping if self.topping.grade != self.precast.concrete.grade else self.precast.concrete
        return self._section(self.b, c)

    @property
    def elastic(self) -> RCSection:
        """the composite section for stresses and stiffness, transformed to the precast concrete (topping width × n)"""
        return self._section(self.b * self.n, self.precast.concrete)


def composite(precast: RCSection, *, b: float, t: float, concrete, top=(), bars=()) -> Composite:
    """the precast section with a cast-in-place topping of width b (effective width, 4.3.3) and thickness t on top, of
    `concrete`; top: rows of (count, diameter, depth of the bar centre from the topping's top); bars: (area, depth from
    the topping's top[, diameter])"""
    if precast.has_external:
        raise NotImplementedError("8.1 has no provisions for composite members with external tendons")
    c = concrete if isinstance(concrete, Concrete) else _concrete(concrete)
    pts = list(_bars(top)) + [(q[0], q[1], q[2] if len(q) > 2 else np.nan) for q in bars or ()]
    return Composite(precast, c, b, t, np.array(pts, float).reshape(-1, 3))


# ── 8.1.4, 8.1.5 two-stage design effects ────────────────────────────────────
def composite_design_moments(M1Gk: float, M1Qk: float, M2Gk: float, M2Qd: float) -> tuple[float, float]:
    """(M1d, Md) (8.1.4): the precast girder M1d = M1Gd + M1Qd (8.1.4-1), the composite member Md = M1Gd + M2Gd + M2Qd
    (8.1.4-2), with M1Gd = 1.2 M1Gk (precast and topping self-weight), M1Qd = 1.4 M1Qk (construction loads of stage 1),
    M2Gd = 1.2 M2Gk (deck finishes), M2Qd the design value of the stage-2 variable actions (D60 basic combination).
    Shears alike (8.1.5). γ0 applies to the composite member (8.1.4)."""
    M1Gd = 1.2 * M1Gk
    return M1Gd + 1.4 * M1Qk, M1Gd + 1.2 * M2Gk + M2Qd


def composite_shear(comp: Composite, V1d: float, Vd: float, stirrups, *, gamma0: float = 1.0,
                    gamma0_precast: float = 1.0, pc_class: str | None = None, **kw) -> list[Check]:
    """shear (8.1.5): the precast girder under V1d (5.2.9 on its own section and concrete; gamma0_precast default 1.0,
    program reading of 8.1.4, γ0 named for the composite member only; pass γ0 to apply it to stage 1 as well, as 5.1.2
    and GB 50010 would) and the composite member under Vd: Vcs with the lower of the two concrete classes, α2 = 1.0 for
    prestressed composite members, and no less than the precast girder's resistance. Further arguments as jtg.shear."""
    pre = shear(comp.precast, V1d, stirrups, gamma0=gamma0_precast, pc_class=pc_class, **kw)
    pre.name = "shear (precast girder, stage 1)"
    low = min(comp.topping, comp.precast.concrete, key=lambda c: c.grade)
    sec = comp.ultimate.replace(concrete=low)
    a2 = 1.0 if comp.precast.prestressed else None
    cmp = shear(sec, Vd, stirrups, gamma0=gamma0, pc_class=pc_class, alpha2=a2, **kw)
    cmp.name = "shear (composite member)"
    cmp.clause = "JTG 3362-2018 8.1.5, 5.2.9"
    R_pre = shear(comp.precast, 0.0, stirrups, pc_class=pc_class, **kw).resistance
    if R_pre > cmp.resistance:
        cmp.notes.append(f"Vcs of the composite ({cmp.resistance:.0f} kN) below the precast girder's ({R_pre:.0f} kN): "
                         "the latter taken (8.1.5)")
        cmp.resistance = R_pre
    cmp.details["concrete_used"] = f"C{low.grade}"
    return [pre, cmp]


def interface_shear(comp: Composite, Vd: float, stirrups, *, gamma0: float = 1.0, steel=None) -> Check:
    """shear across the joint of a composite beam (8.1.6), detailing per 9.3.16, 9.3.17 assumed:
    γ0 Vd ≤ 0.12 fcd b h0 + 0.85 fsv Asv h0 / sv (8.1.6); fcd the lower of the two concretes, b the joint width (the
    precast top), h0 of the composite member, stirrups = (legs crossing the joint, diameter, spacing)"""
    sec = comp.ultimate
    fcd = min(comp.topping.fcd, comp.precast.concrete.fcd)
    st = sec.steel if steel is None else (steel if isinstance(steel, Rebar) else _rebar(steel))
    b = comp.precast.width(1e-6 * comp.precast.h)
    h0 = sec.h0()
    legs, dia, sv = stirrups
    Asv = legs * math.pi * dia * dia / 4
    R = (0.12 * fcd * b * h0 + 0.85 * st.fsv * Asv / sv * h0) / 1e3
    return Check("composite joint shear (beam)", gamma0 * Vd, R, "kN", "JTG 3362-2018 8.1.6",
                 {"b": b, "h0": h0, "fcd": fcd, "Asv": Asv},
                 ["8.1.6 holds with the detailing of 9.3.16, 9.3.17: check it with "
                  "jtg.detailing.composite_beam_detailing (warnings)"])


def interface_shear_slab(Vd: float, *, b: float, h0: float, gamma0: float = 1.0, s: float | None = None,
                         Asv: float | None = None, steel="HRB400") -> Check:
    """shear across the joint of a composite slab (8.1.7): γ0 Vd / (b h0) ≤ 2.00 MPa (8.1.7-1); up to 0.45 MPa (8.1.7-2)
    detailing joint steel per 9.2.7 suffices; above it Asv ≥ 0.3 b s / fsd per vertical section (8.1.7-3), checked when
    Asv and its spacing s are given. Vd in kN, b the precast slab's joint width, h0 of the composite slab (mm)."""
    st = steel if isinstance(steel, Rebar) else _rebar(steel)
    tau = gamma0 * Vd * 1e3 / (b * h0)
    chk = Check("composite joint shear (slab)", tau, 2.00, "MPa", "JTG 3362-2018 8.1.7", {"tau": tau},
                limit_check=True, symbol="γ0Vd/(bh0)")
    if tau <= 0.45:
        chk.notes.append(f"γ0Vd/(bh0) = {tau:.2f} ≤ 0.45 MPa: joint steel by detailing (9.2.7)")
    elif Asv is not None and s is not None:
        need = 0.3 * b * s / st.fsd
        chk.details["Asv_min"] = need
        if Asv < need:
            chk.failed.append(f"joint steel {Asv:.0f} mm² < 0.3 b s / fsd = {need:.0f} mm² (8.1.7-3)")
    else:
        chk.notes.append("γ0Vd/(bh0) > 0.45 MPa: joint steel Asv ≥ 0.3 b s / fsd (8.1.7-3; pass Asv and s to check)")
    return chk


# ── 8.1.8, 8.1.9 crack resistance of prestressed composite members ───────────
def _at(v, z: float) -> float:
    """a number, or a function of the depth z (mm from the composite top)"""
    return float(v(z)) if callable(v) else float(v)


def composite_prestress_stress(comp: Composite, tendons, *, M_p2: float = 0.0):
    """concrete stress (MPa, compression +) at depth z (mm from the composite top) from tendons stressed on the
    composite member, after the topping hardened (e.g. negative-moment tendons in the deck or the precast top flange of
    girders made continuous): rows (Ap mm², depth mm from the composite top, σpe MPa, duct diameter mm, as the
    tendon rows of jtg.rect). Their force
    Σ σpe Ap acts on the composite section transformed to the precast concrete (comp.elastic, the precast girder's ducts
    grouted) less their own ducts (× n in the topping), the net section of a post-tensioned member (6.1.5 item 2,
    6.1.6-4), with their secondary moment M_p2 (kN·m, sagging +); in the topping the stress is × n (comp.n). Returns
    σ(z), for sigma_pc2 of composite_crack_resistance and composite_service_stresses and sigma_cx of
    composite_principal_stresses."""
    rows = np.asarray(tendons, float).reshape(-1, 4)
    A, zc, I = comp.elastic.props("transformed")
    Q, It = A * zc, I + A * zc * zc
    for _, d, _, dd in rows:
        a = math.pi * dd * dd / 4 * (comp.n if d < comp.t else 1.0)
        A, Q, It = A - a, Q - a * d, It - a * d * d
    zc = Q / A
    I = It - A * zc * zc
    F = rows[:, 0] * rows[:, 2]
    N, zN = F.sum(), (F * rows[:, 1]).sum() / F.sum()

    def sigma(z):
        z = np.asarray(z, float)
        s = (N / A + N * (zN - zc) * (z - zc) / I - M_p2 * 1e6 * (z - zc) / I) * np.where(z < comp.t, comp.n, 1.0)
        return float(s) if s.ndim == 0 else s

    return sigma


def _edge_check(title: str, edges: dict, k_: str, segmental: bool, clause: str) -> Check:
    """6.3.1 item 1 at every edge (as jtg.crack_resistance): edges name → sigma_st, sigma_lt (None: not checked),
    sigma_pc, ftk[, sigma_G]; the governing edge reported"""
    f = 0.80 if segmental else 0.85
    for e in edges.values():
        if k_ == "FULL":
            e["demand"], e["limit"] = e["sigma_st"], (f if e["sigma_pc"] > 0 else 1.0) * e["sigma_pc"]
        elif k_ == "A":
            e["demand"], e["limit"] = e["sigma_st"], e["sigma_pc"] + 0.7 * e["ftk"]
        else:
            e["demand"], e["limit"] = e["sigma_G"], e["sigma_pc"]
    cand = list(edges)
    if k_ == "B":
        cand = [n for n in edges if edges[n]["sigma_st"] > edges[n]["sigma_pc"]] or \
            [max(edges, key=lambda n: edges[n]["sigma_st"] - edges[n]["sigma_pc"])]
    gov = max(cand, key=lambda n: edges[n]["demand"] - edges[n]["limit"])
    g = edges[gov]
    d = {k: g.get(k) for k in ("sigma_st", "sigma_lt", "sigma_pc", "ftk", "W01", "W0")} | {"edge": gov, "edges": edges}
    kind = {"FULL": "full prestress", "A": "class A", "B": "class B under self-weight"}[k_]
    sym = "σG" if k_ == "B" else "σst"
    if k_ == "B":
        d["sigma_G"] = g["sigma_G"]
    chk = Check(f"crack resistance ({title}, {kind}, {gov})", g["demand"], g["limit"], "MPa", clause, d,
                limit_check=True, symbol=sym)
    for n in cand:
        e = edges[n]
        if n != gov and e["demand"] > e["limit"] + 1e-9:
            chk.failed.append(f"{n}: {sym} = {e['demand']:.2f} > {e['limit']:.2f} MPa")
    if k_ == "A":
        for n, e in edges.items():
            if e["sigma_lt"] is not None and e["sigma_lt"] > e["sigma_pc"] + 1e-9:
                chk.failed.append(f"{n}: σlt = {e['sigma_lt']:.2f} > σpc = {e['sigma_pc']:.2f} MPa (6.3.1-4)")
    return chk


def composite_crack_resistance(comp: Composite, *, M1k: float, M1Gk: float, M2s: float, M2l: float, pc_class: str,
                               segmental: bool = False, sigma_extra=0.0, sigma_pc2=0.0,
                               M2Gk: float | None = None) -> list[Check]:
    """normal-section crack resistance of a prestressed composite member (8.1.8) at every edge, moments sagging +
    (hogging moments of negative-moment regions put the topping and the precast top in tension):
    the precast girder alone (stage 1) at its top and bottom: σst = M1k / W01 (8.1.8-1);
    the composite member at the precast top and bottom: σst = M1Gk / W01 + M2s / W0 (8.1.8-2), σlt = M1Gk / W01 + M2l / W0
    (8.1.8-3); at the topping's top: σst = n M2s / W0, σlt = n M2l / W0 (the topping carried nothing in stage 1; n =
    comp.n, W0 on the section transformed to the precast concrete).
    M1k = M1Gk + M1Qk (stage 1); M2s, M2l: stage-2 frequent and quasi-permanent moments (deck finishes + ψ live loads).
    σpc: at the precast edges the precast girder's prestress (8.1.8); at the topping 0 (the precast tendons were stressed
    before it was cast); sigma_pc2 adds the prestress applied to the composite member at every composite edge (MPa,
    compression +, actual stress, a number or a function of the depth z from the composite top; tendons stressed after
    the topping hardened: composite_prestress_stress). ftk: the precast concrete's at the precast edges (8.1.8), the
    topping's at its own edge (program convention: 8.1.8 places the edge in the precast girder).
    Limits of 6.3.1 at each edge, the governing one reported (details hold all): full σst − 0.85 σpc ≤ 0 (0.80 segmental;
    1.0 where σpc is a tension); A σst − σpc ≤ 0.7 ftk and, for the composite member, σlt − σpc ≤ 0.
    Class B (program convention: 8.1.8 covers members not allowed to crack; here 6.3.1 item 1-3 in the two-stage form of
    8.1.8): under the self-weight alone the edges in tension under the frequent combination stay compressed: precast
    girder M1Gk / W01 ≤ σpc, composite member M1Gk / W01 + M2Gk / W0 ≤ σpc (n M2Gk / W0 at the topping); M2Gk the stage-2
    self-weight (deck finishes).
    sigma_extra: a further tensile stress (MPa, tension +) of the composite member in the frequent combination, e.g. ψ ×
    the differential shrinkage or temperature gradient (thermal_stresses on comp.elastic, × n in the topping) with
    their secondary moments; a number (the same at every edge) or a function of the depth z; added to σst only."""
    k_ = _pc_class(pc_class)
    if not comp.precast.prestressed:
        raise ValueError("8.1.8 covers prestressed composite members")
    if k_ == "B" and M2Gk is None:
        raise ValueError("class B: pass M2Gk, the stage-2 self-weight moment (6.3.1 item 1-3)")
    p, e = comp.precast, comp.elastic
    _, z01, I01 = p.props("transformed")
    _, z0, I0 = e.props("transformed")
    t, n = comp.t, comp.n
    t1 = lambda zp: (zp - z01) / I01 * 1e6  # tension per kN·m of sagging moment, stage 1 at precast depth zp
    t2 = lambda z: (z - z0) / I0 * 1e6 * (n if z < t else 1.0)  # stage 2 at composite depth z, actual stress
    pre, cmp = {}, {}
    for name, zp in (("precast top", 0.0), ("precast bottom", p.h)):
        spc = float(p.prestress_stress(zp))
        pre[name] = {"sigma_st": M1k * t1(zp), "sigma_lt": None, "sigma_pc": spc, "ftk": p.concrete.ftk,
                     "W01": I01 / abs(zp - z01), "sigma_G": M1Gk * t1(zp)}
        z = zp + t
        cmp[name] = {"sigma_st": M1Gk * t1(zp) + M2s * t2(z) + _at(sigma_extra, z), "sigma_lt": M1Gk * t1(zp) + M2l * t2(z),
                     "sigma_pc": spc + _at(sigma_pc2, z), "ftk": p.concrete.ftk, "W01": I01 / abs(zp - z01),
                     "W0": I0 / abs(z - z0), "sigma_G": M1Gk * t1(zp) + (M2Gk or 0.0) * t2(z)}
    cmp["topping top"] = {"sigma_st": M2s * t2(0.0) + _at(sigma_extra, 0.0), "sigma_lt": M2l * t2(0.0),
                          "sigma_pc": _at(sigma_pc2, 0.0), "ftk": comp.topping.ftk, "W01": None, "W0": I0 / z0,
                          "sigma_G": (M2Gk or 0.0) * t2(0.0)}
    clause = "JTG 3362-2018 8.1.8, 6.3.1" + (" item 1-3" if k_ == "B" else "")
    return [_edge_check("precast girder", pre, k_, segmental, clause),
            _edge_check("composite member", cmp, k_, segmental, clause)]


def composite_principal_stresses(comp: Composite, M1: float, V1: float, M2: float, V2: float, *, depths=None,
                                 bent_tendons=(), sigma_cy: float = 0.0, sigma_cx=0.0) -> list[dict]:
    """principal stresses of a prestressed composite member (8.1.9, 6.3.3) at points of the precast girder (depths from
    the composite top, within the precast girder): σcx = σpc + M1 y01 / I01 + M2 y0 / I0, τ = V1 S01 / (b I01) +
    V2 S0 / (b I0) − Σ σ''pe Apb sin θp Sn / (b In) (stage 1 on the precast transformed section, stage 2 on the composite
    one, the bent tendons on the precast net section, pretensioned: transformed), b the precast width at the point.
    Default points: the centroids of the precast and composite sections (where inside the precast girder) and the precast
    web–flange junctions. bent_tendons: rows (Apb, θp°[, σ''pe]) as jtg.principal_stresses; sigma_cy: vertical
    compression (MPa); sigma_cx: a further longitudinal stress (MPa, compression +), one value or a function of the depth,
    e.g. ψ × differential shrinkage or temperature gradient. Signs as there."""
    p, e = comp.precast, comp.elastic
    t = comp.t
    _, z01, I01 = p.props("transformed")
    _, z0, I0 = e.props("transformed")
    kind_p = "transformed" if p.pretensioned else "net"
    _, _, Ip = p.props(kind_p)
    if depths is None:
        zs = [t + z01] + ([z0] if t <= z0 <= comp.h else []) + [
            t + z for z in _bounds(p) if 0 < z < p.h and abs(p.width(z - 1e-6 * p.h) - p.width(z + 1e-6 * p.h)) > 1e-9]
    else:
        zs = list(depths)
    Pv = 0.0
    for bt in bent_tendons:
        sp = bt[2] if len(bt) > 2 else float(np.mean(p._sigma_pe()))
        Pv += sp * bt[0] * math.sin(math.radians(bt[1]))
    out = []
    for z in sorted(set(float(v) for v in zs)):
        if not t <= z <= comp.h:
            raise ValueError(f"depth {z:g} mm is not in the precast girder ({t:g} … {comp.h:g} mm)")
        zp = z - t
        b = _b_at(p, zp)
        scx = float(p.prestress_stress(zp)) + M1 * 1e6 * (z01 - zp) / I01 + M2 * 1e6 * (z0 - z) / I0 + (
            float(sigma_cx(z)) if callable(sigma_cx) else sigma_cx)
        tau = (abs(V1) * 1e3 * p.first_moment(zp) / (b * I01) + abs(V2) * 1e3 * e.first_moment(z) / (b * I0)
               - Pv * p.first_moment(zp, kind_p) / (b * Ip))
        m, r = (scx + sigma_cy) / 2, math.hypot((scx - sigma_cy) / 2, tau)
        out.append({"z": z, "b": b, "sigma_cx": scx, "sigma_cy": sigma_cy, "tau": tau, "sigma_tp": r - m,
                    "sigma_cp": m + r})
    return out


def composite_inclined_crack_resistance(comp: Composite, M1: float, V1: float, M2: float, V2: float, *, pc_class: str,
                                        precast: bool = True, **kw) -> Check:
    """inclined-section crack resistance of a prestressed composite member (8.1.9): the largest σtp of
    composite_principal_stresses (frequent combination for stage 2) against 6.3.1 for full (0.6 / 0.4 ftk) and class A
    (0.7 / 0.5 ftk) members; ftk of the precast concrete. Class B members take the class A limits, which 6.3.1-7, -8
    give for both classes (8.1.9 names the full and class A requirements). Stage-2 prestress (composite_prestress_stress)
    goes in sigma_cx."""
    k_ = _pc_class(pc_class)
    pts = composite_principal_stresses(comp, M1, V1, M2, V2, **kw)
    w = max(pts, key=lambda q: q["sigma_tp"])
    f = (0.6 if precast else 0.4) if k_ == "FULL" else (0.7 if precast else 0.5)
    return Check("crack resistance (inclined section, composite)", w["sigma_tp"], f * comp.precast.concrete.ftk, "MPa",
                 "JTG 3362-2018 8.1.9, 6.3.1, 6.3.3", {"at": w, "points": pts}, limit_check=True, symbol="σtp")


def composite_principal_compression(comp: Composite, M1: float, V1: float, M2: float, V2: float, *, stirrups=None,
                                    stirrup_steel=None, **kw) -> Check:
    """principal stresses in service of a prestressed composite member (8.1.17, 7.1.6) under the characteristic moments
    and shears of the two stages (composite_principal_stresses, further arguments as there): σcp ≤ 0.6 fck of the precast
    concrete (7.1.6-1); where σtp > 0.5 ftk the stirrup spacing must not exceed fsk Asv / (σtp b) (7.1.6-2), checked when
    stirrups = (legs, diameter, spacing) are given"""
    pts = composite_principal_stresses(comp, M1, V1, M2, V2, **kw)
    c = comp.precast.concrete
    wc = max(pts, key=lambda q: q["sigma_cp"])
    wt = max(pts, key=lambda q: q["sigma_tp"])
    chk = Check("principal compressive stress (service, composite)", wc["sigma_cp"], 0.6 * c.fck, "MPa",
                "JTG 3362-2018 8.1.17, 7.1.6", {"at": wc, "sigma_tp": wt["sigma_tp"], "points": pts}, limit_check=True,
                symbol="σcp")
    if wt["sigma_tp"] <= 0.5 * c.ftk:
        chk.notes.append(f"σtp = {wt['sigma_tp']:.2f} ≤ 0.5 ftk = {0.5 * c.ftk:.2f} MPa: detailing stirrups (7.1.6)")
    elif stirrups is not None:
        st = comp.precast.steel if stirrup_steel is None else (
            stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
        legs, dia, sv = stirrups
        sv_max = st.fsk * legs * math.pi * dia * dia / 4 / (wt["sigma_tp"] * wt["b"])
        chk.details["sv_max"] = sv_max
        if sv > sv_max:
            chk.failed.append(f"stirrup spacing {sv:g} mm > fsk Asv / (σtp b) = {sv_max:.0f} mm (7.1.6-2)")
    else:
        chk.notes.append(f"σtp = {wt['sigma_tp']:.2f} > 0.5 ftk: stirrup spacing by 7.1.6-2 (pass stirrups to check)")
    return chk


# ── 8.1.10, 8.1.11 crack width of reinforced concrete composite members ──────
def composite_crack_width(comp: Composite, *, M1Gk: float, M2s: float, M2l: float, env: str = "I",
                          slab: bool = False, epoxy: bool = False, M1u: float | None = None, d: float | None = None,
                          c: float | None = None, C1: float | None = None) -> Check:
    """largest crack width of a reinforced concrete composite member as a whole (8.1.10, 8.1.11), sagging: 6.4.3 with
    C2 = 1 + 0.5 (M1Gk + M2l) / (M1Gk + M2s) (8.1.11-1) and σss = σs1 + σs2 = M1Gk / (0.87 As h01) + 0.5 (1 + h1/h) M2s /
    (0.87 As h0) ≤ 0.75 fsk (8.1.11-2); h1 = h when M1Gk < 0.35 M1u (M1u: the precast girder's flexural resistance, by
    default from jtg.flexure). As: the precast girder's tension rows; c, d, ρte, C1 as jtg.crack_width; C3 1.0 (1.15 for
    slabs). M2l: the quasi-permanent moment of stage 2 (all variable actions × ψq, 8.1.11)."""
    p = comp.precast
    if p.prestressed:
        raise ValueError("8.1.10, 8.1.11 cover reinforced concrete composite members")
    bot, _ = p.edge_masks()
    As, h01 = p.bars[bot, 0].sum(), p.h0()
    h0 = comp.ultimate.h0()
    if M1u is None:
        M1u = flexure(p, 1.0).resistance
    h1 = comp.h if M1Gk < 0.35 * M1u else comp.h1
    s1 = M1Gk * 1e6 / (0.87 * As * h01)
    s2 = 0.5 * (1 + h1 / comp.h) * M2s * 1e6 / (0.87 * As * h0)
    sss = s1 + s2
    C2 = 1 + 0.5 * (M1Gk + M2l) / (M1Gk + M2s)
    C3 = 1.15 if slab else 1.0
    if C1 is None:
        C1 = 1.4 if p.steel.name == "HPB300" else (1.15 if epoxy else 1.0)
    a_s = p.h - h01
    rho_te = As / _Ate(p, a_s)
    rho_c = min(max(rho_te, 0.01), 0.1)
    d = _diam(p, bot) if d is None else d
    c = min(_cover(p, bot, "bottom") if c is None else c, 50.0)
    W = C1 * C2 * C3 * sss / p.steel.Es * (c + d) / (0.36 + 1.7 * rho_c)
    lim = crack_width_limit(env)
    chk = Check("crack width (composite)", W, lim, "mm", "JTG 3362-2018 8.1.11, 6.4.3",
                {"sigma_s1": s1, "sigma_s2": s2, "sigma_ss": sss, "C1": C1, "C2": C2, "C3": C3, "c": c, "d": d,
                 "rho_te": rho_te, "h1_used": h1, "M1u": M1u}, limit_check=True, symbol="Wcr")
    if sss > 0.75 * p.steel.fsk:
        chk.failed.append(f"σss = {sss:.0f} > 0.75 fsk = {0.75 * p.steel.fsk:.0f} MPa (8.1.11-2)")
    return chk


# ── 8.1.12 – 8.1.16 stiffness, long-term deflection, camber ──────────────────
COMPOSITE_CAMBER_LONG_TERM = 1.75  # long-term factor on the camber of prestressed composite members (8.1.15, Ec1 I01)


def composite_stiffness(comp: Composite, Ms: float, *, pc_class: str | None = None) -> tuple[float, dict]:
    """stiffness of a composite member as a whole under the frequent moment Ms (8.1.13): reinforced concrete 0.9 × B of
    6.5.2-1 with B0 = 0.95 Ec1 I0, Bcr = Ec1 Icr (Ec1: the precast concrete; I0, Icr of the composite section transformed
    to it); full and class A prestressed B0 = 0.80 Ec1 I0. Returns (B N·mm², details)."""
    e = comp.elastic
    if e.prestressed:
        k_ = _pc_class(pc_class)
        if k_ == "B":
            raise NotImplementedError("8.1.13 gives the stiffness of full and class A prestressed composite members")
        _, _, I0 = e.props("transformed")
        return 0.80 * e.concrete.Ec * I0, {"I0": I0}
    B, d = stiffness(e, Ms)
    d["B_monolithic"] = B
    return 0.9 * B, d


def composite_long_term_factor(precast_concrete) -> float:
    """ηθ of composite members (8.1.14), by the precast concrete's class: 1.80 below C40; 1.65 at C40 to 1.55 at C80,
    linear between"""
    g = (precast_concrete if isinstance(precast_concrete, Concrete) else _concrete(precast_concrete)).grade
    return 1.80 if g < 40 else 1.65 - 0.10 * (min(g, 80) - 40) / 40


# ── 8.1.17 stresses in service ───────────────────────────────────────────────
def composite_service_stresses(comp: Composite, *, M1k: float, M2k: float, pc_class: str, sigma_pc2=0.0) -> Check:
    """normal stresses of a full or class A prestressed composite member in service (8.1.17, 7.1.3, 7.1.5): stage 1 on the
    precast transformed section, stage 2 on the composite one: at the precast top and bottom σ = σpc + M1k y01 / I01 +
    M2k y0 / I0 ≤ 0.5 fck (precast), at the topping's top n M2k y0 / I0 ≤ 0.5 fck (topping, n its modulus ratio); tendons
    σpe + αEP (σkt1 + σkt2) ≤ 0.65 fpk (0.75 fpk threaded bars). M1k, M2k: characteristic moments of the two stages in
    service (stage 1: the precast girder's and the topping's own weight, the construction loads being gone; stage 2 with
    impact, 7.1.1). Moments sagging +. sigma_pc2: the prestress applied to the composite member (MPa, compression +, a
    number or a function of the depth from the composite top: composite_prestress_stress) added at the three edges; its
    effect on the precast tendons' σpe is the user's (losses). Principal stresses in service:
    composite_principal_compression."""
    k_ = _pc_class(pc_class)
    if k_ == "B" or not comp.precast.prestressed:
        raise ValueError("composite_service_stresses: full and class A prestressed composite members")
    p, e = comp.precast, comp.elastic
    _, z01, I01 = p.props("transformed")
    _, z0, I0 = e.props("transformed")
    t = comp.t
    edges = {}
    for name, zp in (("precast top", 0.0), ("precast bottom", p.h)):
        edges[name] = (float(p.prestress_stress(zp)) + M1k * 1e6 * (z01 - zp) / I01 + M2k * 1e6 * (z0 - (zp + t)) / I0
                       + _at(sigma_pc2, zp + t), 0.5 * p.concrete.fck)
    edges["topping top"] = (comp.n * M2k * 1e6 * z0 / I0 + _at(sigma_pc2, 0.0), 0.5 * comp.topping.fck)
    gov = max(edges, key=lambda k: edges[k][0] / edges[k][1])
    tn = p.tendon
    dp = p.tendons[:, 1]
    skt = -M1k * 1e6 * (z01 - dp) / I01 - M2k * 1e6 * (z0 - (dp + t)) / I0
    sp = p._sigma_pe() + tn.Ep / p.concrete.Ec * skt
    lim_p = (0.75 if tn.kind == "bar" else 0.65) * tn.fpk
    chk = Check(f"concrete compression in service (composite, {gov})", edges[gov][0], edges[gov][1], "MPa",
                "JTG 3362-2018 8.1.17, 7.1.3, 7.1.5", {"edges": edges, "tendon_stress": sp, "tendon_limit": lim_p},
                limit_check=True, symbol="σ")
    for k, (v, lim) in edges.items():
        if k != gov and v > lim + 1e-9:
            chk.failed.append(f"{k}: {v:.2f} > {lim:.2f} MPa (7.1.5-1)")
    if sp.max() > lim_p:
        chk.failed.append(f"tendon stress {sp.max():.0f} > {lim_p:.0f} MPa (7.1.5)")
    return chk


def shrinkage_difference_profile(comp: Composite, dT: float):
    """8.1.3: the shrinkage difference between the topping and the older precast girder taken as a temperature drop dT
    (°C, negative) in the topping only (commentary: as a reverse temperature difference); use with
    jtg.thermal_stresses(comp.elastic, profile) and multiply the stresses found in the topping by comp.n"""
    from .temperature import Profile
    return Profile((0.0, comp.t, comp.t), (dT, dT, 0.0))
