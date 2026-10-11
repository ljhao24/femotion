"""Serviceability limit state and stress checks of reinforced and prestressed concrete sections, JTG 3362-2018 chapters
6 and 7.

Units as femotion.codes.jtg.rc: dimensions mm, steel areas mm², stresses MPa, crack widths and deflections mm; moments
kN·m (sagging +), forces kN (axial compression +). Clauses, formulas and tables: docs/codes/JTG_3362-2018.md.

Effects come from the combinations of JTG D60-2015: Ms, Ns, Vs from jtg.frequent_combination, Ml from
jtg.quasi_permanent_combination (vehicle load without impact, 6.1.1), Mk, Vk from jtg.characteristic_combination (vehicle
load with impact, 7.1.1). The prestress is a permanent action (D60 4.2.2) carried by the section: σp0 per tendon
(`tendons`), σpe per tendon (`sigma_pe`), σl6 at the bars (`sigma_l6`). The secondary moment Mp2 of post-tensioned
statically indeterminate members is passed separately (M_p2, kN·m, sagging +): it enters σpc through 6.1.6-4 and the
terms Ms ± Mp2 (6.4.4-11), Mk ± Mp2 (7.1.4-3), so it must not be part of Ms, Ml or Mk. The secondary shear goes in V_p2.

    from femotion.codes import jtg
    rc = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    jtg.crack_width(rc, Ms=820.0, Ml=640.0, env="I")              # 6.4: Wcr against table 6.4.2
    B, d = jtg.stiffness(rc, Ms=820.0)                            # 6.5.2: N·mm², for a frame with EI = B
    pc = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)],
                 tendons=[(2780, 1470, 1150, 90)], sigma_pe=1080.0)   # σp0 and σpe per tendon
    jtg.crack_resistance(pc, Ms=3600.0, Ml=3100.0, pc_class="A")   # 6.3.1 item 1
    jtg.inclined_crack_resistance(pc, Ms=900.0, Vs=650.0, pc_class="A")   # 6.3.1 item 2 with 6.3.3
    jtg.service_stresses(pc, Mk=4100.0, pc_class="A")               # 7.1.3–7.1.5
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .materials import Rebar, rebar as _rebar, strength_at
from .rc import Check, RCSection, Rect, cracking_moment

# table 6.4.2: environment class → (reinforced concrete and class B with threaded bars, class B with wire or strand) mm;
# None: class B members with wire or strand not allowed
CRACK_WIDTH_LIMITS = {"I": (0.20, 0.10), "II": (0.20, 0.10), "III": (0.15, 0.10), "IV": (0.15, 0.10), "V": (0.10, None),
                      "VI": (0.15, 0.10), "VII": (0.20, 0.10)}
CAMBER_LONG_TERM = 2.0  # long-term factor on the camber from the prestress after all losses (6.5.4)


def _orient(sec: RCSection, M: float):
    """the section with the face in tension under M at the bottom, and the sign that turns M sagging"""
    return (sec, 1.0) if M >= 0 else (sec.flipped(), -1.0)


def _pc_class(pc_class) -> str:
    k = str(pc_class or "").upper()
    if k not in ("FULL", "A", "B"):
        raise ValueError("pc_class: 'full', 'A' or 'B' (6.1.2)")
    return k


def _b_at(s: RCSection, z: float) -> float:
    """width at depth z, the narrower side where the width changes"""
    e = 1e-6 * s.h
    return min(s.width(z - e), s.width(z + e)) if 0 < z < s.h else s.width(min(max(z, e), s.h - e))


def _bounds(s: RCSection):
    return sorted({p.top for p in s.pieces if isinstance(p, Rect)} | {p.bottom for p in s.pieces if isinstance(p, Rect)})


# ── 6.3 crack resistance of prestressed members ───────────────────────────────
def crack_resistance(sec: RCSection, Ms: float, Ml: float, *, pc_class: str, segmental: bool = False,
                     M_p2: float = 0.0, M_G: float | None = None) -> Check:
    """normal-section crack resistance of a prestressed flexural member (6.3.1 item 1, 6.3.2) at both edges. At each
    edge σst = Ms / W0 and σlt = Ml / W0 are the tensile stresses from the frequent and quasi-permanent moments (W0 of
    the transformed section at that edge; negative where the loads compress it) and σpc is the stress from the prestress
    after all losses (6.1.6 with Mp2, RCSection.prestress_stress; compression +, negative on a side the prestress puts
    in tension). The governing edge is reported; details hold both ("top", "bottom").
    pc_class "full": σst − 0.85 σpc ≤ 0, precast (6.3.1-1); 0.80 for members cast in segments or with mortar joints,
    segmental=True (6.3.1-2); where σpc is a tension the factor is taken as 1.0 (program convention: no tension under the
    frequent combination, 6.1.2). "A": σst − σpc ≤ 0.7 ftk (6.3.1-3) and σlt − σpc ≤ 0 (6.3.1-4). "B": the self-weight
    M_G (kN·m) must not decompress the edges in tension under the frequent combination (the most tensioned one when
    neither is): σG − σpc ≤ 0 (6.3.1 item 1-3)."""
    k_ = _pc_class(pc_class)
    if not sec.prestressed:
        raise ValueError("crack resistance (6.3) is checked for prestressed members")
    if k_ == "B" and M_G is None:
        raise ValueError("class B: pass M_G, the moment from the structure's self-weight (6.3.1 item 1-3)")
    A0, z0, I0 = sec.props("transformed")
    ftk = sec.concrete.ftk
    edges = {}
    for name, z in (("top", 0.0), ("bottom", sec.h)):
        t = (z - z0) / I0 * 1e6  # tension per kN·m of sagging moment at this fibre
        e = {"W0": I0 / abs(z - z0), "sigma_st": Ms * t, "sigma_lt": Ml * t,
             "sigma_pc": float(sec.prestress_stress(z, M_p2))}
        if k_ == "FULL":
            f = (0.80 if segmental else 0.85) if e["sigma_pc"] > 0 else 1.0
            e["demand"], e["limit"] = e["sigma_st"], f * e["sigma_pc"]
        elif k_ == "A":
            e["demand"], e["limit"] = e["sigma_st"], e["sigma_pc"] + 0.7 * ftk
        else:
            e["sigma_G"] = M_G * t
            e["demand"], e["limit"] = e["sigma_G"], e["sigma_pc"]
        edges[name] = e
    cand = list(edges)
    if k_ == "B":
        cand = [n for n in edges if edges[n]["sigma_st"] > edges[n]["sigma_pc"]] or \
            [max(edges, key=lambda n: edges[n]["sigma_st"] - edges[n]["sigma_pc"])]
    gov = max(cand, key=lambda n: edges[n]["demand"] - edges[n]["limit"])
    g = edges[gov]
    d = {k: g[k] for k in ("sigma_st", "sigma_lt", "sigma_pc", "W0")} | {"edge": gov, **edges}
    if k_ == "FULL":
        chk = Check(f"crack resistance (normal section, full prestress, {gov} edge)", g["demand"], g["limit"], "MPa",
                    f"JTG 3362-2018 6.3.1-{2 if segmental else 1}", d, limit_check=True, symbol="σst")
    elif k_ == "A":
        chk = Check(f"crack resistance (normal section, class A, {gov} edge)", g["demand"], g["limit"], "MPa",
                    "JTG 3362-2018 6.3.1-3, -4", d, limit_check=True, symbol="σst")
        for n, e in edges.items():
            if e["sigma_lt"] > e["sigma_pc"] + 1e-9:
                chk.failed.append(f"{n}: σlt = {e['sigma_lt']:.2f} > σpc = {e['sigma_pc']:.2f} MPa (6.3.1-4)")
    else:
        d["sigma_G"] = g["sigma_G"]
        chk = Check(f"crack resistance (normal section, class B under self-weight, {gov} edge)", g["demand"], g["limit"], "MPa",
                    "JTG 3362-2018 6.3.1 item 1-3", d, limit_check=True, symbol="σG")
    for n in cand:
        e = edges[n]
        if n != gov and e["demand"] > e["limit"] + 1e-9:
            chk.failed.append(f"{n}: {chk.symbol} = {e['demand']:.2f} > {e['limit']:.2f} MPa")
    return chk


def principal_stresses(sec: RCSection, M: float, V: float, *, N: float = 0.0, M_p2: float = 0.0, V_p2: float = 0.0,
                       depths=None, bent_tendons=(), vertical=None, sigma_cx: float = 0.0, sigma_cy: float = 0.0,
                       tau_t: float = 0.0) -> list[dict]:
    """principal stresses of 6.3.3 at points of the depth, for the frequent Ms, Vs (6.3.1 item 2) or the characteristic
    Mk, Vk (7.1.6):
        σcx = σpc + M y0 / I0 (6.3.3-2) + N / A0 + sigma_cx, σcy = 0.6 n σ'pe Apv / (b sp) + sigma_cy (6.3.3-3, -4),
        τ = (V + Vp2) S0 / (b I0) − Σ σ''pe Apb sin θp · Sn / (b In) + tau_t (6.3.3-5),
        σtp, σcp = (σcx + σcy)/2 ∓ √(((σcx − σcy)/2)² + τ²) (6.3.3-1).
    Compression + for σcx, σcy, σcp; σtp is returned as a tension (positive). M sagging +; V and the bent tendons by
    magnitude, the tendons' vertical force opposing the shear (θp negative where it adds to it). σpc from the section's
    prestress (6.1.6 with Mp2); M and V on the transformed section, the bent tendons on the net section of post-tensioned
    members (the transformed one of pretensioned members, 6.1.5).
    depths: depths from the top (mm); default the centroid of the transformed section and the web–flange junctions, b
    taken on the narrower side. bent_tendons: rows of (Apb mm², θp degrees[, σ''pe MPa]) in the same bent-up plane;
    σ''pe defaults to the mean σpe of the section. vertical: (n Apv mm² per section, spacing sp mm, σ'pe MPa) of vertical
    prestressing bars. N: axial force of the same combination (kN, compression +, e.g. restraint forces in a rigid frame),
    on the transformed section; sigma_cx: any further normal stress (MPa) computed separately, as the commentary to 6.3.3
    asks for actions that cause axial force as well as moment. sigma_cy: σcy,ph + σcy,t + σcy,l (MPa) from transverse
    prestress, transverse temperature gradient and vehicle loads; tau_t: shear stress from torsion (MPa)."""
    A0, z0, I0 = sec.props("transformed")
    kind_p = "transformed" if sec.pretensioned else "net"
    _, _, Ip = sec.props(kind_p)
    if depths is None:
        zs = [z0] + [z for z in _bounds(sec) if 0 < z < sec.h and
                     abs(sec.width(z - 1e-6 * sec.h) - sec.width(z + 1e-6 * sec.h)) > 1e-9]
    else:
        zs = list(depths)
    rows = []
    for bt in bent_tendons:
        sp = bt[2] if len(bt) > 2 else float(np.mean(sec._sigma_pe()))
        rows.append(sp * bt[0] * math.sin(math.radians(bt[1])))
    Pv = sum(rows)
    out = []
    for z in sorted(set(float(v) for v in zs)):
        b = _b_at(sec, z)
        scx = float(sec.prestress_stress(z, M_p2)) + M * 1e6 * (z0 - z) / I0 + N * 1e3 / A0 + sigma_cx
        scy = sigma_cy
        if vertical is not None:
            Apv, sp_v, spe_v = vertical
            scy += 0.6 * spe_v * Apv / (b * sp_v)
        tau = abs(V + V_p2) * 1e3 * sec.first_moment(z) / (b * I0) - Pv * sec.first_moment(z, kind_p) / (b * Ip) + tau_t
        m, r = (scx + scy) / 2, math.hypot((scx - scy) / 2, tau)
        out.append({"z": z, "b": b, "sigma_cx": scx, "sigma_cy": scy, "tau": tau, "sigma_tp": r - m, "sigma_cp": m + r})
    return out


def inclined_crack_resistance(sec: RCSection, Ms: float, Vs: float, *, pc_class: str, precast: bool = True,
                              **kw) -> Check:
    """inclined-section crack resistance of a prestressed flexural member (6.3.1 item 2): the largest principal tensile
    stress σtp of 6.3.3 (principal_stresses, frequent Ms, Vs; further arguments as there) against: full prestress
    0.6 ftk precast (6.3.1-5), 0.4 ftk cast in place, including precast segmental members (precast=False, 6.3.1-6);
    classes A and B 0.7 ftk precast (6.3.1-7), 0.5 ftk cast in place (6.3.1-8)."""
    k_ = _pc_class(pc_class)
    pts = principal_stresses(sec, Ms, Vs, **kw)
    w = max(pts, key=lambda p: p["sigma_tp"])
    f = (0.6 if precast else 0.4) if k_ == "FULL" else (0.7 if precast else 0.5)
    eq = {("FULL", True): 5, ("FULL", False): 6}.get((k_, precast), 7 if precast else 8)
    return Check("crack resistance (inclined section)", w["sigma_tp"], f * sec.concrete.ftk, "MPa",
                 f"JTG 3362-2018 6.3.1-{eq}, 6.3.3", {"at": w, "points": pts}, limit_check=True, symbol="σtp")


def principal_compression(sec: RCSection, Mk: float, Vk: float, *, stirrups=None, stirrup_steel=None, **kw) -> Check:
    """principal stresses in service (7.1.6) under the characteristic Mk, Vk and the prestress (principal_stresses,
    further arguments as there): σcp ≤ 0.6 fck (7.1.6-1). Stirrups: where σtp ≤ 0.5 ftk detailing suffices; above it
    the spacing must not exceed sv = fsk Asv / (σtp b) (7.1.6-2), checked when stirrups = (legs, diameter mm, spacing
    mm) are given (b: web width at the point). The stirrups needed for shear resistance (5.2.9) govern if more."""
    c = sec.concrete
    pts = principal_stresses(sec, Mk, Vk, **kw)
    wc = max(pts, key=lambda p: p["sigma_cp"])
    wt = max(pts, key=lambda p: p["sigma_tp"])
    chk = Check("principal compressive stress (service)", wc["sigma_cp"], 0.6 * c.fck, "MPa", "JTG 3362-2018 7.1.6",
                {"at": wc, "sigma_tp": wt["sigma_tp"], "points": pts}, limit_check=True, symbol="σcp")
    if wt["sigma_tp"] <= 0.5 * c.ftk:
        chk.notes.append(f"σtp = {wt['sigma_tp']:.2f} ≤ 0.5 ftk = {0.5 * c.ftk:.2f} MPa: detailing stirrups (7.1.6)")
    elif stirrups is not None:
        st = sec.steel if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
        legs, dia, sv = stirrups
        sv_max = st.fsk * legs * math.pi * dia * dia / 4 / (wt["sigma_tp"] * wt["b"])
        chk.details["sv_max"] = sv_max
        if sv > sv_max:
            chk.failed.append(f"stirrup spacing {sv:g} mm > fsk Asv / (σtp b) = {sv_max:.0f} mm (7.1.6-2)")
    else:
        chk.notes.append(f"σtp = {wt['sigma_tp']:.2f} > 0.5 ftk: stirrup spacing by 7.1.6-2 (pass stirrups to check)")
    return chk


# ── 6.4 crack width ───────────────────────────────────────────────────────────
def _gamma_f(s: RCSection, h0: float) -> float:
    """γ'f = (b'f − b) h'f / (b h0) of the flange on the compressed (top) face, h'f ≤ 0.2 h0 (6.4.4-7)"""
    b, bf = s.web(), s.width(1e-6 * s.h)
    if bf <= b * (1 + 1e-9):
        return 0.0
    hf = next((z for z in _bounds(s) if z > 0 and s.width(z + 1e-6 * s.h) < bf * (1 - 1e-9)), s.h)
    return (bf - b) * min(hf, 0.2 * h0) / (b * h0)


def _z_arm(gf: float, h0: float, e: float) -> float:
    """z = [0.87 − 0.12 (1 − γ'f)(h0/e)²] h0 ≤ 0.87 h0 (6.4.4-5)"""
    return min((0.87 - 0.12 * (1 - gf) * (h0 / e) ** 2) * h0, 0.87 * h0)


def _Ate(s: RCSection, a_s: float) -> float:
    """effective tension area Ate = 2 as b (6.4.5-1), b the width at the tension (bottom) face: the section width of a
    rectangle, the effective flange width of a T or I section with its flange in tension; at most the whole section"""
    return min(2 * a_s * s.width(s.h * (1 - 1e-6)), s.area)


def _diam(s: RCSection, mask, extra=()):
    """equivalent diameter de = Σ ni di² / Σ ni di of the bars in mask and extra (n, d) pairs (6.4.3)"""
    dia = s.bar_d[mask]
    if np.isnan(dia).any():
        raise ValueError("bar diameters unknown (points given without a diameter): pass d=… (6.4.3)")
    n = s.bars[mask, 0] / (math.pi * dia * dia / 4)
    nd = [(a, b) for a, b in zip(n, dia)] + list(extra)
    return sum(a * b * b for a, b in nd) / sum(a * b for a, b in nd)


def _cover(s: RCSection, mask, side: str = "bottom") -> float:
    """concrete cover of the outermost row of the bars in mask (to the bar surface), on the bottom or top face"""
    d, dia = s.bars[mask, 1], s.bar_d[mask]
    k = int(np.argmax(d)) if side == "bottom" else int(np.argmin(d))
    if np.isnan(dia[k]):
        raise ValueError("bar diameters unknown: pass c=… (cover of the outermost tension bars, 6.4.3)")
    return (s.h - d[k] if side == "bottom" else d[k]) - dia[k] / 2


def crack_width_limit(env: str, sec: RCSection | None = None) -> float | None:
    """limit of table 6.4.2 (mm) for an environment class "I" … "VII": reinforced concrete and class B members with
    prestressing threaded bars, or class B members with wire or strand (sec prestressed with such tendons); None where
    the latter are not allowed (class V)"""
    e = str(env).upper().replace("CLASS", "").strip()
    if e not in CRACK_WIDTH_LIMITS:
        raise ValueError(f"environment class {env!r}: one of {', '.join(CRACK_WIDTH_LIMITS)} (table 6.4.2)")
    col = 1 if sec is not None and sec.prestressed and sec.tendon.kind in ("strand", "wire") else 0
    return CRACK_WIDTH_LIMITS[e][col]


def crack_width(sec: RCSection, Ms: float, Ml: float, *, N: float = 0.0, Nl: float | None = None, env: str = "I",
                l0: float | None = None, slab: bool = False, epoxy: bool = False, pc_class: str | None = None,
                M_p2: float = 0.0, tendon_bundles=None, d: float | None = None, c: float | None = None,
                welded_cage: bool = False, C1: float | None = None) -> Check:
    """maximum crack width (6.4.3) Wcr = C1 C2 C3 σss/Es · (c + d)/(0.36 + 1.7 ρte) against table 6.4.2, under the
    frequent combination (Ms, N) with the long-term factor C2 = 1 + 0.5 Ml/Ms (Ml quasi-permanent; Nl/N for an axial
    tension member; for eccentric members with Nl given the larger of the two ratios; C2 ≥ 1). Reinforced concrete
    (6.4.4 item 1, 2; 6.4.5):
        N = 0: flexure, σss = Ms / (0.87 As h0), C3 = 1.15 for slabs (slab=True), else 1.0;
        N > 0: eccentric compression (l0 for ηs): rectangles, T, I (and box) sections σss = Ns (es − z)/(As z), C3 0.9,
               no check needed for e0/h ≤ 0.55; solid circles 6.4.4-9, ρte 6.4.5-2, C3 0.75, none for e0/r ≤ 0.55;
        N < 0: axial tension (Ms = 0) σss = Ns / As, all bars, Ate = A, C3 1.2; eccentric tension σss = Ns e's /
               (As (h0 − a's)), C3 1.1.
    Class B prestressed flexural members (pc_class="B", 6.4.4 item 3): σss = (Ms ± Mp2 − Np0 (z − ep)) / ((Ap + As) z),
    Ep in place of Es (commentary to 6.4.3); tendon_bundles: rows of (number of tendons, strands or wires per tendon, dp)
    of the tension-zone tendons, each a bundle of dpe = √n dp in de (table 6.4.3).
    As: reinforced concrete the tension-face rows (RCSection.edge_masks); class B every bar and tendon below the gross
    centroid (of the side in tension). d: the equivalent diameter of those bars (override d=…, e.g. for bundled bars
    dse = √n d; × 1.3 for welded cages, 9.3.11); c: cover of the outermost tension row, at most 50 (override c=…);
    ρte = As / Ate within 0.01 … 0.1, Ate = 2 as b with b the width at the tension face. C1 from the bars' steel: 1.0
    ribbed, 1.4 plain (HPB300), 1.15 epoxy-coated ribbed (epoxy=True); class B the same (override C1=…).
    Circular sections: eccentric compression only."""
    k_ = None
    if sec.prestressed:
        k_ = _pc_class(pc_class)
        if k_ != "B":
            raise ValueError("crack width is checked for reinforced concrete and class B members (6.4.1)")
        if N != 0:
            raise NotImplementedError("class B members: flexure only (6.4.4 item 3)")
    lim = crack_width_limit(env, sec)
    E = sec.steel.Es
    notes, extra_d = [], []
    rho_te = None

    def skip(why):
        chk = Check("crack width", 0.0, lim if lim is not None else 0.0, "mm", "JTG 3362-2018 6.4.3", {}, [why],
                    limit_check=True, symbol="Wcr")
        if lim is None:
            chk.failed.append(f"class B members with {sec.tendon.kind} are not allowed in environment V (table 6.4.2)")
        return chk

    if k_ == "B":
        s, k = _orient(sec, Ms + M_p2)
        M = k * (Ms + M_p2) * 1e6
        Np0, zN = s.prestress_resultant("p0")
        bot, pb = s.bars[:, 1] > s.zg, s.tendons[:, 1] > s.zg  # all the steel of the tension zone (6.4.4-11)
        As, Ap = s.bars[bot, 0].sum(), s.tendons[pb, 0].sum()
        At = As + Ap
        h0 = (np.r_[s.bars[bot, 0], s.tendons[pb, 0]] * np.r_[s.bars[bot, 1], s.tendons[pb, 1]]).sum() / At
        ep = h0 - zN
        e = ep + M / Np0
        gf = _gamma_f(s, h0)
        z = _z_arm(gf, h0, e)
        sss = (M - Np0 * (z - ep)) / (At * z)
        A0, z0, I0 = s.props("transformed")
        sst = k * Ms * 1e6 / (I0 / (s.h - z0))
        spc = float(s.prestress_stress(s.h, k * M_p2))
        dd = {"Np0": Np0 / 1e3, "ep": ep, "e": e, "z": z, "h0": h0, "gamma_f": gf, "sigma_st": sst, "sigma_pc": spc}
        if sst <= spc:
            return skip(f"σst = {sst:.2f} ≤ σpc = {spc:.2f} MPa: the tension face is not decompressed, no crack")
        if z <= 0 or sss <= 0:
            return skip(f"σss = {sss:.1f} MPa ≤ 0 (z = {z:.0f} mm): no crack opening under Ms")
        E = s.tendon.Ep
        C3 = 1.0
        As_te, a_s = At, s.h - h0
        if tendon_bundles is None and d is None:
            raise ValueError("class B: pass tendon_bundles=[(tendons, strands per tendon, dp)] or d=… (table 6.4.3)")
        extra_d = [(n, math.sqrt(m) * dp) for n, m, dp in (tendon_bundles or ())]
        mask = bot
        side = "bottom"
        if c is None and not bot.any():
            raise ValueError("class B without bars on the tension face: pass c=… (cover, 6.4.3)")
    elif sec.kind in ("circle", "annulus") and N <= 0:
        raise NotImplementedError("6.4.4 gives σss of circular sections for eccentric compression only")
    elif N == 0:
        s, k = _orient(sec, Ms)
        bot, _ = s.edge_masks()
        if not bot.any():
            raise ValueError("no tension reinforcement")
        As, h0 = s.bars[bot, 0].sum(), s.h0()
        sss = k * Ms * 1e6 / (0.87 * As * h0)
        C3 = 1.15 if slab else 1.0
        As_te, a_s, mask, side = As, s.h - h0, bot, "bottom"
        dd = {"h0": h0}
    elif N > 0:
        s, k = _orient(sec, Ms)
        e0 = abs(Ms) / N * 1e3
        if s.kind == "annulus":
            raise NotImplementedError("crack width of hollow circular sections is not covered by 6.4.4")
        if s.kind == "circle":
            r, rs = s.h / 2, s.ring
            a_s = r - rs
            if e0 / r <= 0.55:
                return skip(f"e0/r = {e0 / r:.3f} ≤ 0.55: crack width need not be checked (6.4.3)")
            eta = 1.0 if not l0 or l0 / (2 * r) <= 14 else 1 + (l0 / (2 * r)) ** 2 / (4000 * e0 / (2 * r - a_s))
            t = eta * e0 / r
            As = s.bars[:, 0].sum()
            sss = 0.6 * (t - 0.1) ** 3 / ((0.45 + 0.26 * rs / r) * (t + 0.2) ** 2) * N * 1e3 / As
            rho = As / (math.pi * r * r)
            beta = (0.4 + 2.5 * rho) * (1 + 0.353 * t ** -2)
            r1 = r - 2 * a_s
            rho_te = beta * As / (math.pi * (r * r - r1 * r1))
            C3 = 0.75
            mask = np.ones(len(s.bars), bool)
            dd = {"e0": e0, "eta_s": eta, "beta": beta, "rho": rho}
            if c is None:
                c = a_s - s.bar_d[0] / 2
        else:
            if e0 / s.h <= 0.55:
                return skip(f"e0/h = {e0 / s.h:.3f} ≤ 0.55: crack width need not be checked (6.4.3)")
            bot, _ = s.edge_masks()
            As, h0 = s.bars[bot, 0].sum(), s.h0()
            eta = 1.0 if not l0 or l0 / s.h <= 14 else 1 + (l0 / s.h) ** 2 / (4000 * e0 / h0)
            es = eta * e0 + (h0 - s.zg)
            gf = _gamma_f(s, h0)
            z = _z_arm(gf, h0, es)
            sss = N * 1e3 * (es - z) / (As * z)
            C3 = 0.9
            As_te, a_s, mask, side = As, s.h - h0, bot, "bottom"
            dd = {"e0": e0, "eta_s": eta, "es": es, "z": z, "gamma_f": gf, "h0": h0}
    else:
        Nt = -N
        if Ms == 0:
            s = sec
            mask = np.ones(len(s.bars), bool)
            As = s.bars[:, 0].sum()
            sss = Nt * 1e3 / As
            C3 = 1.2
            rho_te = As / s.area
            dd = {}
            if c is None:
                c = min(_cover(s, mask, "bottom"), _cover(s, mask, "top"))
        else:
            s, k = _orient(sec, Ms)
            e0 = abs(Ms) / Nt * 1e3
            bot, top = s.edge_masks()
            if not top.any():
                raise ValueError("eccentric tension: no bars on the less tensioned face (A's of 6.4.4-3)")
            As, h0 = s.bars[bot, 0].sum(), s.h0()
            a_c = float(np.average(s.bars[top, 1], weights=s.bars[top, 0]))
            es_ = s.zg + e0 - a_c  # the tension resultant lies e0 below the centroid
            sss = Nt * 1e3 * es_ / (As * (h0 - a_c))
            C3 = 1.1
            As_te, a_s, mask, side = As, s.h - h0, bot, "bottom"
            dd = {"e0": e0, "e's": es_, "h0": h0}
    if rho_te is None:
        Ate = _Ate(s, a_s)
        rho_te = As_te / Ate
        dd["Ate"] = Ate
    rho_c = min(max(rho_te, 0.01), 0.1)
    if d is None:
        d = _diam(s, mask, extra_d)
    if welded_cage:
        d *= 1.3
    if c is None:
        c = _cover(s, mask, side)
    c = min(c, 50.0)
    if C1 is None:
        C1 = 1.4 if s.steel.name == "HPB300" else (1.15 if epoxy else 1.0)
    if Ms != 0:
        r = Ml / Ms
        if N != 0 and Nl is not None:
            r = max(r, Nl / N)  # eccentric members: the larger of the moment and axial-force ratios
    else:
        if Nl is None:
            raise ValueError("axial tension: pass Nl, the quasi-permanent axial force, for C2 (6.4.3)")
        r = Nl / N
    C2 = 1 + 0.5 * max(r, 0.0)
    W = C1 * C2 * C3 * sss / E * (c + d) / (0.36 + 1.7 * rho_c)
    dd.update(sigma_ss=sss, C1=C1, C2=C2, C3=C3, c=c, d=d, rho_te=rho_te, rho_te_used=rho_c, E=E)
    chk = Check("crack width", W, lim if lim is not None else 0.0, "mm", "JTG 3362-2018 6.4.3, table 6.4.2", dd, notes,
                limit_check=True, symbol="Wcr")
    if lim is None:
        chk.failed.append(f"class B members with {sec.tendon.kind} are not allowed in environment V (table 6.4.2)")
    return chk


# ── cracked transformed section (7.2.4, 7.1.4, appendix J) ─────────────────────
def _cracked(s: RCSection, N: float = 0.0, z_R: float | None = None, M: float | None = None, limits: bool = True):
    """elastic cracked section: concrete in compression above the neutral axis x (no tension), every bar at αES As and
    every tendon at αEP Ap. Pure bending (z_R None: first moment of the transformed area about the axis zero, 7.2.4) or a
    compressive resultant N (N) acting at depth z_R from the top (negative above it, appendix J). With limits, the notes
    of J.0.1: compression bars with αES σcc > f'sd count as f'sd/(αES σcc) A's (also 7.2.4 commentary, then M (N·mm) is
    needed for pure bending); A'p, the tendons on the compression side of the gross centroid, count with a negative
    sign while above the axis with αEP σcc − σ'p0 < 0 (note 2).
    Returns dict x (inf: the whole section in compression), A, zc (centroid depth), I (about it), I_na (about the axis),
    sigma(z) concrete stress (compression +) for N or M, weights."""
    c = s.concrete
    aS, aP = s.steel.Es / c.Ec, (s.tendon.Ep / c.Ec if s.prestressed else 0.0)
    wb, wp = np.ones(len(s.bars)), np.ones(len(s.tendons))
    comp_p = s.tendons[:, 1] <= s.zg  # A'p

    def steel():
        return [(aS * a * w, d) for (a, d), w in zip(s.bars, wb)] + [(aP * a * w, d) for (a, d, _, _), w in zip(s.tendons, wp)]

    def QI(x, st):
        xc = min(x, s.h)
        a, q = s.part(xc)
        i2 = s.part2(xc)
        Q = x * a - q + sum(at * (x - d) for at, d in st)
        I = x * x * a - 2 * x * q + i2 + sum(at * (x - d) ** 2 for at, d in st)
        return Q, I

    def props(x, st):
        xc = min(x, s.h)
        a, q = s.part(xc)
        i2 = s.part2(xc)
        A = a + sum(at for at, _ in st)
        Q1 = q + sum(at * d for at, d in st)
        zc = Q1 / A
        return A, zc, i2 + sum(at * d * d for at, d in st) - A * zc * zc

    def first_root(f, xs):
        """the first sign change of f from − to + over the grid xs, refined; None when there is none"""
        v = [f(x) for x in xs]
        for i in range(len(xs) - 1):
            if v[i] < 0 <= v[i + 1]:
                return brentq(f, xs[i], xs[i + 1], xtol=1e-10 * s.h)
        return None

    for _ in range(50):
        st = steel()
        if not any(at for at, _ in st):
            raise ValueError("no steel: the cracked section is undefined")
        x0 = first_root(lambda x: QI(x, st)[0], s.h * np.geomspace(1e-9, 1.0, 160))  # pure-bending axis
        if x0 is None:
            raise RuntimeError("cracked section: no neutral axis in bending")
        if z_R is None:
            x = x0
        else:
            A, zc_full, _ = props(math.inf, st)
            x = None
            if z_R < zc_full:  # else the resultant is at or below the centroid of the whole section: no tension face
                x = first_root(lambda x: x - QI(x, st)[1] / QI(x, st)[0] - z_R,
                               x0 + s.h * np.geomspace(1e-9, 1e4, 320))
            x = math.inf if x is None else x
        A, zc, I = props(x, st)
        I_na = QI(x, st)[1] if x < math.inf else None
        if z_R is None:
            sig = (lambda z, x=x, I_na=I_na: M * (x - z) / I_na) if M is not None else None
        else:
            e = zc - z_R  # resultant above the centroid
            sig = lambda z, A=A, zc=zc, I=I, e=e: N / A + N * e * (zc - z) / I
        if not limits or sig is None:
            break
        nb = np.array([s.steel.fsd_c / (aS * sig(d)) if (d < x and aS * sig(d) > s.steel.fsd_c) else 1.0
                       for _, d in s.bars])
        npw = np.array([-1.0 if (cp and x < math.inf and d < x and aP * sig(d) - sp0 < 0) else 1.0
                        for (_, d, sp0, _), cp in zip(s.tendons, comp_p)])
        if np.allclose(nb, wb, rtol=1e-6) and np.array_equal(npw, wp):
            break
        wb, wp = nb, npw
    else:
        raise RuntimeError("cracked section: the notes of J.0.1 (compression steel) did not converge")
    return {"x": x, "A": A, "zc": zc, "I": I, "I_na": I_na, "sigma": sig, "w_bars": wb, "w_tendons": wp}


def cracked_section(sec: RCSection, hogging: bool = False) -> dict:
    """neutral-axis depth x0 and moment of inertia Icr of the cracked transformed section in bending (7.2.4; 6.5.2):
    concrete in compression above x0, every bar αES As and every tendon αEP Ap, x0 where the first moments of the
    compressed and tensioned transformed areas about the axis are equal, Icr about that axis. Returns {"x0", "Icr"}
    (mm, mm⁴) for the sagging (or hogging) side."""
    s = sec.flipped() if hogging else sec
    r = _cracked(s, limits=False)
    return {"x0": r["x"], "Icr": r["I_na"]}


# ── 6.5 deflection ────────────────────────────────────────────────────────────
def stiffness(sec: RCSection, Ms: float, *, pc_class: str | None = None, M_p2: float = 0.0) -> tuple[float, dict]:
    """flexural stiffness B (N·mm²) of a section under the frequent moment Ms for deflections by structural mechanics
    (6.5.1, 6.5.2): B0 = 0.95 Ec I0 (I0 transformed), Bcr = Ec Icr (cracked_section).
    Reinforced concrete: Mcr = γ ftk W0 (6.5.2-3, γ = 2 S0 / W0); B = B0 below Mcr, else B0 / ((Mcr/Ms)² + (1 −
    (Mcr/Ms)²) B0/Bcr) (6.5.2-1, -2). Prestressed, full and class A: B0 (6.5.2-4). Class B: B0 up to Mcr = (σpc + γ ftk)
    W0 (6.5.2-7, jtg.cracking_moment) and Bcr for M − Mcr (6.5.2-5, -6), M = Ms + Mp2; returned as the secant stiffness
    B = M / (Mcr/B0 + (M − Mcr)/Bcr), which gives the same curvature."""
    c = sec.concrete
    k_ = _pc_class(pc_class) if sec.prestressed else None
    Mt = Ms + (M_p2 if k_ == "B" else 0.0)
    s, k = _orient(sec, Mt)
    A0, z0, I0 = s.props("transformed")
    B0 = 0.95 * c.Ec * I0
    d = {"B0": B0, "I0": I0}
    if k_ in ("FULL", "A"):
        return B0, d
    cr = cracked_section(s)
    Bcr = c.Ec * cr["Icr"]
    M = k * Mt * 1e6
    if k_ is None:
        W0 = I0 / (s.h - z0)
        gamma = 2 * s.first_moment(z0) / W0
        Mcr = gamma * c.ftk * W0
        B = B0 if M < Mcr else B0 / ((Mcr / M) ** 2 + (1 - (Mcr / M) ** 2) * B0 / Bcr)
    else:
        Mcr = cracking_moment(s)[0] * 1e6
        B = B0 if M <= Mcr else M / (Mcr / B0 + (M - Mcr) / Bcr)
    d.update(Bcr=Bcr, Icr=cr["Icr"], x0=cr["x0"], Mcr=Mcr / 1e6, cracked=M >= Mcr)
    return B, d


def long_term_factor(concrete) -> float:
    """ηθ (6.5.3): 1.60 below C40; 1.45 at C40 to 1.35 at C80, linear between"""
    g = concrete.grade
    return 1.60 if g < 40 else 1.45 - 0.10 * (min(g, 80) - 40) / 40


def deflection_limit(L: float, cantilever: bool = False) -> float:
    """largest long-term deflection from the frequent vehicle (without impact) and crowd loads (6.5.3): L/600 of the
    computed span, L/300 of a cantilever's length (same units as L)"""
    return L / (300 if cantilever else 600)


def deflection_check(f: float, L: float, cantilever: bool = False) -> Check:
    """6.5.3: f = ηθ × the deflection under the frequent vehicle (without impact) and crowd loads (mm), L in mm"""
    return Check("deflection (long term, live load)", f, deflection_limit(L, cantilever), "mm", "JTG 3362-2018 6.5.3",
                 limit_check=True, symbol="f")


def precamber_rc(f_G: float, f_Q: float, L: float) -> float:
    """pre-camber of a reinforced concrete flexural member (6.5.5 item 1), from long-term deflections (× ηθ): none when
    f_G + f_Q (frequent combination) ≤ L/1600, else f_G + f_Q / 2 (self-weight plus half the frequent variable loads)"""
    return 0.0 if f_G + f_Q <= L / 1600 else f_G + f_Q / 2


def precamber_pc(f_s: float, f_pe: float) -> float:
    """pre-camber of a prestressed flexural member (6.5.5 item 2): f_s long-term deflection under the frequent
    combination (× ηθ), f_pe long-term camber from the prestress after all losses (EcI0, × 2.0, 6.5.4). None when the
    camber exceeds the deflection (then mind an excessive camber), else the difference."""
    return max(f_s - f_pe, 0.0)


# ── 7.1 stresses in service of prestressed members ─────────────────────────────
def service_stresses(sec: RCSection, Mk: float, *, pc_class: str, M_p2: float = 0.0) -> Check:
    """normal stresses of a prestressed flexural member in service under the characteristic Mk (vehicle load with
    impact, 7.1.1) and the prestress after all losses (7.1.3–7.1.5):
    full and class A: σkc = Mk y0 / I0 on the transformed section (7.1.3-1) plus the prestress stress σpt / σpc of
    6.1.6 (with Mp2) at each edge; the larger edge compression ≤ 0.5 fck (7.1.5-1). Tendons σpe + σp, σp = αEP σkt with
    σkt the concrete stress from Mk at the tendon (7.1.3-2), ≤ 0.65 fpk for wire and strand (7.1.5-2), 0.75 fpk for
    threaded bars (7.1.5-4); every tendon row is checked (the code asks for the outermost layer).
    class B: cracked section under Np0 at eN = (Mk ± Mp2)/Np0 − hps (7.1.4, appendix J): σcc ≤ 0.5 fck (the larger of
    the two edges when the whole section stays compressed); tendons σp0 + σp with σp = αEP [Np0/Acr − Np0 e0N (hp − c)
    /Icr] (7.1.4-5, a tension increment where negative)."""
    k_ = _pc_class(pc_class)
    if not sec.prestressed:
        raise ValueError("7.1 covers prestressed members")
    c, t = sec.concrete, sec.tendon
    lim_p = (0.75 if t.kind == "bar" else 0.65) * t.fpk
    aEP = t.Ep / c.Ec
    if k_ in ("FULL", "A"):
        A0, z0, I0 = sec.props("transformed")
        edge = {}
        for name, z in (("top", 0.0), ("bottom", sec.h)):
            edge[name] = Mk * 1e6 * (z0 - z) / I0 + float(sec.prestress_stress(z, M_p2))
        gov = max(edge, key=edge.get)
        skt = -Mk * 1e6 * (z0 - sec.tendons[:, 1]) / I0  # tension +
        sp = sec._sigma_pe() + aEP * skt
        d = {"edge_stress": edge, "tendon_stress": sp, "sigma_kt": skt}
        chk = Check(f"concrete compression in service ({gov} edge)", edge[gov], 0.5 * c.fck, "MPa",
                    "JTG 3362-2018 7.1.3, 7.1.5", d, limit_check=True, symbol="σkc + σpt")
        i = int(np.argmax(sp))
    else:
        Mt = Mk + M_p2
        s, k = _orient(sec, Mt)
        Np0, hps = s.prestress_resultant("p0")
        eN = k * Mt * 1e6 / Np0 - hps
        r = _cracked(s, N=Np0, z_R=-eN)
        scc = r["sigma"](0.0)
        edge = "top"
        if r["x"] == math.inf and r["sigma"](s.h) > scc:  # no tension face: the larger compression of the two edges
            scc, edge = r["sigma"](s.h), "bottom"
        sig_p = aEP * np.array([r["sigma"](dp) for dp in s.tendons[:, 1]])  # 7.1.4-5, negative: tension increment
        sp = s.tendons[:, 2] - sig_p
        d = {"x": r["x"], "Acr": r["A"], "Icr": r["I"], "c": r["zc"], "eN": eN, "e0N": eN + r["zc"], "Np0": Np0 / 1e3,
             "hps": hps, "sigma_p": sig_p, "tendon_stress": sp, "edge": edge if k > 0 else {"top": "bottom", "bottom": "top"}[edge]}
        chk = Check("concrete compression in service (cracked section)", scc, 0.5 * c.fck, "MPa",
                    "JTG 3362-2018 7.1.4, 7.1.5", d, limit_check=True, symbol="σcc")
        if r["x"] == math.inf:
            chk.notes.append("Np0 with Mk ± Mp2 leaves the whole section in compression (no crack): the larger edge "
                             "compression is checked")
        i = int(np.argmax(sp))
    if sp[i] > lim_p:
        chk.failed.append(f"tendon stress {sp[i]:.0f} > {lim_p / t.fpk:.2f} fpk = {lim_p:.0f} MPa "
                          f"(7.1.5-{4 if t.kind == 'bar' else 2})")
    chk.details["tendon_limit"] = lim_p
    return chk


# ── 7.2 transient situations ──────────────────────────────────────────────────
def transient_rc(sec: RCSection, Mk: float, Vk: float | None = None, *, fcu: float, stirrups=None,
                 stirrup_steel=None) -> Check:
    """reinforced concrete flexural member in a construction stage (7.2.4–7.2.6), under characteristic construction
    loads Mk, Vk (kN·m, kN; self-weight × the dynamic factor of D60 4.1.10 when lifting or transporting) with concrete
    of cube strength fcu at that stage (f'ck, f'tk interpolated in table 3.1.3):
    σtcc = Mk x0 / Icr ≤ 0.80 f'ck (7.2.4-1); every tension bar row σtsi = αES Mk (h0i − x0)/Icr ≤ 0.75 fsk (7.2.4-2);
    cracked section with compression bars limited to f'sd (commentary). Shear stress at the neutral axis
    σttp = Vk / (b z0) ≤ f'tk (7.2.5), z0 from a triangular compression block; ≤ 0.25 f'tk: carried by the concrete,
    detailing stirrups (7.2.6-1); above it stirrups and bent bars carry it all (7.2.6-2, -3): with stirrups = (legs,
    diameter, spacing) τv = n Asv1 [σs] / (b sv), [σs] = 0.75 fsk, is reported."""
    if sec.prestressed:
        raise ValueError("7.2.4–7.2.6 are for reinforced concrete; prestressed members: transient_pc (7.2.8)")
    fck_t, ftk_t = strength_at(fcu)
    s, k = _orient(sec, Mk)
    M = k * Mk * 1e6
    r = _cracked(s, M=M)
    x0, Icr = r["x"], r["I_na"]
    aS = s.steel.Es / s.concrete.Ec
    scc = M * x0 / Icr
    ten = s.bars[:, 1] > x0
    ss = aS * M * (s.bars[ten, 1] - x0) / Icr
    d = {"x0": x0, "Icr": Icr, "fck_t": fck_t, "ftk_t": ftk_t, "sigma_s": ss}
    chk = Check("transient stresses (reinforced concrete)", scc, 0.80 * fck_t, "MPa", "JTG 3362-2018 7.2.4", d,
                limit_check=True, symbol="σtcc")
    if len(ss) and ss.max() > 0.75 * s.steel.fsk:
        chk.failed.append(f"bar stress {ss.max():.0f} > 0.75 fsk = {0.75 * s.steel.fsk:.0f} MPa (7.2.4-2)")
    if Vk is not None:
        a, q = s.part(x0)
        zc = (x0 * q - s.part2(x0)) / (x0 * a - q)  # resultant of the triangular block
        wt = s.bars[ten, 0] * (s.bars[ten, 1] - x0)
        zt = float((wt * s.bars[ten, 1]).sum() / wt.sum())
        b = s.web()
        tau = abs(Vk) * 1e3 / (b * (zt - zc))
        d.update(z0=zt - zc, sigma_tp=tau)
        if tau > ftk_t:
            chk.failed.append(f"σttp = {tau:.2f} > f'tk = {ftk_t:.2f} MPa (7.2.5)")
        elif tau <= 0.25 * ftk_t:
            chk.notes.append(f"σttp = {tau:.2f} ≤ 0.25 f'tk: carried by the concrete, detailing stirrups (7.2.6-1)")
        else:
            chk.notes.append(f"σttp = {tau:.2f} > 0.25 f'tk: stirrups and bent bars carry the shear stress (7.2.6)")
            if stirrups is not None:
                st = s.steel if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
                legs, dia, sv = stirrups
                d["tau_v"] = legs * math.pi * dia * dia / 4 * 0.75 * st.fsk / (b * sv)
    return chk


def transient_pc(sec: RCSection, M: float, *, fcu: float, grouted: bool = False, M_p2: float = 0.0) -> Check:
    """edge stresses of a prestressed flexural member in a construction stage (7.2.7, 7.2.8): the section carries the
    stage's prestress (σp0 of pretensioned tendons, sigma_pe of post-tensioned ones, after the losses of that stage);
    M: moment of the construction loads (kN·m, sagging +, e.g. the self-weight at transfer, × the dynamic factor of D60
    4.1.10 when lifting); concrete of cube strength fcu at that stage (f'ck, f'tk from table 3.1.3). Loads on the net
    section of post-tensioned members before grouting (grouted=False), else the transformed section (6.1.5).
    σtcc ≤ 0.70 f'ck (7.2.8); σtct ≤ 1.15 f'tk, with (A's + A'p)/A in the pretensioned zone at least 0.2 % up to
    σtct = 0.70 f'tk, 0.4 % at 1.15 f'tk, linear between (A'p only for pretensioned members; A gross; the zone from the
    tension edge to the zero-stress depth)."""
    if not sec.prestressed:
        raise ValueError("7.2.8 is for prestressed members; reinforced concrete: transient_rc")
    fck_t, ftk_t = strength_at(fcu)
    _, zc, I = sec.props("transformed" if (sec.pretensioned or grouted) else "net")
    sig = lambda z: float(sec.prestress_stress(z, M_p2)) + M * 1e6 * (zc - z) / I
    st, sb = sig(0.0), sig(sec.h)
    gov = "top" if st >= sb else "bottom"
    d = {"top": st, "bottom": sb, "fck_t": fck_t, "ftk_t": ftk_t}
    chk = Check(f"transient stresses (prestressed, {gov} edge in compression)", max(st, sb), 0.70 * fck_t, "MPa",
                "JTG 3362-2018 7.2.8", d, limit_check=True, symbol="σtcc")
    sct = -min(st, sb)
    if sct > 0:
        d["sigma_tct"] = sct
        if sct > 1.15 * ftk_t:
            chk.failed.append(f"σtct = {sct:.2f} > 1.15 f'tk = {1.15 * ftk_t:.2f} MPa (7.2.8)")
        need = 0.002 if sct <= 0.70 * ftk_t else 0.002 + 0.002 * (sct - 0.70 * ftk_t) / (0.45 * ftk_t)
        zn = st / (st - sb) * sec.h  # zero-stress depth
        zone = (lambda z: z < zn) if st < 0 else (lambda z: z > zn)
        As_ = sec.bars[[zone(z) for z in sec.bars[:, 1]], 0].sum()
        Ap_ = sec.tendons[[zone(z) for z in sec.tendons[:, 1]], 0].sum() if sec.pretensioned else 0.0
        rho = (As_ + Ap_) / sec.area
        d.update(rho_pretension_zone=rho, rho_needed=min(need, 0.004))
        if rho < min(need, 0.004):
            chk.failed.append(f"(A's + A'p)/A = {rho:.3%} < {min(need, 0.004):.3%} in the pretensioned zone (7.2.8)")
    return chk
