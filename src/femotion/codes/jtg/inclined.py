"""Shear along a beam, JTG 3362-2018 5.2.8–5.2.13: the inclined sections to check (5.2.8), their horizontal projection
(5.2.10), the search for the governing one with the section limit over its range (5.2.11), and the design of stirrups
and bent-up bars from the shear envelope (5.2.13, 9.3.10).

Units: positions along the beam in m (as femotion.frame), section dimensions in mm, forces in kN and kN·m. Clauses,
formulas and the program conventions: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    res = jtg.shear_search(x, Vd, Md, section, [(0, 4, (2, 12, 100)), (4, 16, (2, 10, 200))],
                           supports=[(0.0, "end"), (16.0, "interior", 0.6)], gamma0=1.0)
    print(res.governing, res.ok)
    st = jtg.stirrup_design(section, Vd=620.0, legs=2, dia=10, xi=0.6)          # 5.2.13-1: the spacing
    bb = jtg.bent_bar_design(x, Vd, support=0.0, h=1100, z=900, xi=0.6)         # 5.2.13-2: the rows

An inclined section is fixed by its compression end x_c, the normal section of its shear-compression zone: Vd, Md, b
and h0 are taken there (5.2.9), m = Md / (Vd h0) ≤ 3 and C = 0.6 m h0 (5.2.10). The crack opens on the tension face at
x_t = x_c ∓ C, on the side of the smaller moment: towards the support near an end support (figure 5.2.9a), away from it
near an interior support, where the compression end is at the support (figure 5.2.9b). The positions of 5.2.8 fix
either end: h/2 from an end support, the bend points of bent-up bars, changes of stirrups and of the web are tension
ends (figure 5.2.8a; x_c then solves |x_c − x_t| = C(x_c)); the edge of the diaphragm over an interior support and a
change of depth are compression ends (figure 5.2.8b).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .materials import Rebar, rebar as _rebar
from .rc import Check, RCSection, shear as _shear

STIRRUP_DIAMETERS = (8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32)


# ── inputs along the beam ────────────────────────────────────────────────────
def _zones(v, name: str, const):
    """a constant (`const(v)` true), a callable of x, or rows of (x0, x1, value) → callable of x (the first row that
    contains x); the rows are kept on the callable as .rows"""
    if const(v):
        return lambda x: v
    if callable(v):
        return v
    rows = sorted((float(a), float(b), val) for a, b, val in v)

    def f(x):
        for a, b, val in rows:
            if a - 1e-9 <= x <= b + 1e-9:
                return val
        raise ValueError(f"{name}: no zone contains x = {x:.3f} m")
    f.rows = rows
    return f


def _zone_rows(v):
    return getattr(v, "rows", None)


def _span_data(x, V, M, a, b, tol):
    """(x, |V|, M, duplicates) of the data points within [a, b]; at a duplicated position the side belonging to this
    span is kept at its ends (both sides of a support); inside the span the left value is kept and its position is
    returned in `duplicates` (a jump inside a span, e.g. under a point load: split the data there for both sides)"""
    idx = [k for k in range(len(x)) if a - tol <= x[k] <= b + tol]
    keep, dup = [], []
    for k in idx:
        if keep and abs(x[k] - x[keep[-1]]) <= tol:
            if abs(x[k] - a) <= tol:  # at the left end of the span keep the last (right side of the support)
                keep[-1] = k
            elif abs(x[k] - b) > tol:
                dup.append(float(x[k]))
            continue  # elsewhere the first (left side)
        keep.append(k)
    k = np.array(keep, int)
    return x[k], np.abs(V[k]), M[k], dup


# ── results ─────────────────────────────────────────────────────────────────
@dataclass
class InclinedSection:
    """one inclined section: compression end x_c (where Vd, Md, b, h0 are taken, 5.2.9), tension end x_t, horizontal
    projection C = |x_c − x_t| (m), m (5.2.10), the design forces at x_c and the check (jtg.shear with the stirrups and
    bent-up bars it crosses; 5.2.11 with the least b, h0 and the largest Vd over its range)"""
    x_c: float
    x_t: float
    C: float
    m: float
    Vd: float
    Md: float
    check: Check
    source: str = "grid"

    @property
    def ratio(self) -> float:
        return self.check.ratio

    @property
    def ok(self) -> bool:
        return self.check.ok

    def __str__(self):
        return (f"[{self.source}] x_c = {self.x_c:.3f} m, x_t = {self.x_t:.3f} m, C = {self.C:.3f} m, m = {self.m:.2f}: "
                f"{self.check}")


@dataclass
class ShearSearch:
    """the inclined sections checked (`sections`), the section limit of 5.2.11 at the end supports (`limits`), the
    stirrup rule of 9.3.12 item 3 near the supports (`warnings`, detailing: does not change `ok`) and notes"""
    sections: list = field(default_factory=list)
    limits: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def governing(self) -> InclinedSection:
        """the most unfavourable inclined section: a failed one first, then the highest ratio"""
        return max(self.sections, key=lambda s: (not s.ok, s.ratio))

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.sections) and all(c.ok for c in self.limits)

    def at(self, source: str) -> list:
        """the inclined sections built from the 5.2.8 positions whose label contains `source`"""
        return [s for s in self.sections if source in s.source]

    def __str__(self):
        lines = [str(s) for s in self.sections if s.source != "grid"]
        lines += [str(c) for c in self.limits]
        lines.append(f"governing of {len(self.sections)}: {self.governing}")
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# ── the search ──────────────────────────────────────────────────────────────
def shear_search(x, Vd, Md, section, stirrups, *, gamma0: float = 1.0, supports=(), bent=(), positions=(),
                 alpha1="auto", stirrup_steel: str | Rebar | None = None, grid: bool = True, shear_kw=None,
                 support_limit: bool = True) -> ShearSearch:
    """checks inclined sections along a beam (5.2.8–5.2.11) and finds the governing one.
    x, Vd, Md: positions (m, non-decreasing; a position may appear twice at a support, the two sides of the shear jump),
    the design shear (kN, the sign is not used) and the moment that goes with it (kN·m, sagging +; the moment envelope of
    the sign that governs there is the usual stand-in), without γ0.
    section: an RCSection, a callable of x, or rows (x0, x1, RCSection). stirrups: (legs, diameter, spacing), a callable
    of x, or rows (x0, x1, (legs, diameter, spacing)).
    supports: (x, "end") for end supports (5.2.8 item 1: the tension end h/2 from the support centre; the region within
    h/2 is not searched); (x, "interior"[, diaphragm width m]) for interior supports (item 2: the compression end at the
    edge of the diaphragm, default at the support); (x, "free") for the tip of a cantilever. The beam between supports
    is searched span by span.
    support_limit: also the section limit of 5.2.11 at each end support, with the shear there (V0d) and b, h0 of the
    support section (`limits`; the inclined sections start at h/2 and do not reach it).
    bent: rows (x at the bottom bend, x at the top bend, area mm², angle °) of bent-up bars; one counts when the crack,
    drawn straight between the steel levels from x_t to x_c, crosses it with the opposite inclination (5.2.9-3).
    positions: further 5.2.8 positions as (x, "tension" or "compression"[, label]) (e.g. where bars anchored in the
    tension zone stop being needed). Stirrup and section zone boundaries and the bend points of `bent` are added as
    tension ends (the top bends beyond 5.2.8). grid: every data point inside the spans is also tried as a compression
    end.
    alpha1: "auto" by the segment (5.2.9): 0.9 for compression ends in the half of a span next to an interior support
    and on a cantilever arm (between a support and a "free" tip), 1.0 elsewhere (the halves of figure 5.2.13a, b); a
    number or a callable of x.
    shear_kw: a dict of further arguments of jtg.shear (pc_class, alpha3, alpha2, vertical, bent_tendons, …), or a
    callable of x returning one. The stirrup ratio is the mean Asv/sv over the projection (the stirrups the crack
    crosses; a callable is averaged over 40 points) divided by b at x_c; the 9.3.12 rules take the stirrups at x_c.
    9.3.12 item 3 (within one beam depth of a support centre the stirrup spacing should not exceed 100 mm) goes to
    `warnings`."""
    x = np.asarray(x, float)
    V = np.asarray(Vd, float)
    M = np.asarray(Md, float)
    if not (len(x) == len(V) == len(M)) or np.any(np.diff(x) < -1e-12):
        raise ValueError("x, Vd, Md: arrays of one length, x non-decreasing")
    sec_at = _zones(section, "section", lambda v: isinstance(v, RCSection))
    stir_at = _zones(stirrups, "stirrups", lambda v: isinstance(v, (tuple, list)) and len(v) == 3
                     and all(isinstance(e, (int, float)) for e in v))
    kw_at = shear_kw if callable(shear_kw) else (lambda _x: dict(shear_kw or {}))
    a1_at = alpha1 if callable(alpha1) else None
    if not (a1_at or alpha1 == "auto"):
        alpha1 = float(alpha1)
    tol = 1e-9 * max(1.0, float(x[-1] - x[0]))
    sup = sorted([(float(s[0]), str(s[1]), float(s[2]) if len(s) > 2 else 0.0) for s in supports])
    for s in sup:
        if s[1] not in ("end", "interior", "free"):
            raise ValueError("supports: (x, 'end'), (x, 'interior'[, diaphragm width]) or (x, 'free')")
    bounds = sorted({float(x[0]), float(x[-1])} | {s[0] for s in sup if x[0] - tol <= s[0] <= x[-1] + tol})
    res = ShearSearch()
    sst = None if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
    sec_rows = _zone_rows(sec_at)

    # tension-end positions common to all spans
    t_pos = [(float(p[0]), p[2] if len(p) > 2 else f"{p[1]} end x = {p[0]:g}") for p in positions if p[1] == "tension"]
    c_pos = [(float(p[0]), p[2] if len(p) > 2 else f"{p[1]} end x = {p[0]:g}") for p in positions if p[1] == "compression"]
    for rows, what in ((_zone_rows(stir_at), "stirrup change"), (sec_rows, "section change")):
        for r in (rows or [])[:-1]:
            t_pos.append((r[1], f"{what} x = {r[1]:g}"))
    for xb_, xt_, A, th in bent:
        t_pos.append((float(xb_), f"bend x = {xb_:g}"))
        t_pos.append((float(xt_), f"bend x = {xt_:g}"))
    seen, uniq = set(), []
    for p in t_pos:
        key = round(p[0], 9)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    t_pos = uniq

    def q_of(st):
        n_, d_, s_ = st
        return n_ * math.pi * d_ * d_ / 4 / s_

    # 9.3.12 item 3: stirrup spacing within one beam depth of a support centre
    for s in sup:
        if s[1] == "free" or not (x[0] - tol <= s[0] <= x[-1] + tol):
            continue
        h_s = sec_at(s[0]).h / 1e3
        us = [u for u in np.r_[np.linspace(s[0] - h_s, s[0], 21), np.linspace(s[0], s[0] + h_s, 21)]
              if x[0] - tol <= u <= x[-1] + tol]
        worst = max(stir_at(u)[2] for u in us)
        if worst > 100 + 1e-9:
            res.warnings.append(f"stirrups at {worst:g} mm within one beam depth ({h_s:.2f} m) of the support at x = "
                                f"{s[0]:g}: at most 100 mm there (9.3.12 item 3)")

    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= tol:
            continue
        xs, Vs, Ms, dup = _span_data(x, V, M, a, b, tol)
        if dup:
            res.notes.append(f"positions given twice inside the span {a:g}–{b:g} m ({', '.join(f'{d:g}' for d in dup)}): "
                             "the first value is used (split the data there to keep both sides of a jump)")
        if len(xs) < 2:
            continue
        Vf = lambda u: float(np.interp(u, xs, Vs))  # noqa: E731
        Mf = lambda u: float(np.interp(u, xs, Ms))  # noqa: E731
        m_tol = 1e-9 * max(1.0, float(np.abs(Ms).max()))
        hog_at = lambda u: Mf(u) < -m_tol  # noqa: E731  (round-off at a zero moment is not hogging)
        delta = 1e-4 * (b - a)

        def offset(xs_):
            for s in sup:
                if abs(s[0] - xs_) <= tol:
                    if s[1] == "end":
                        return sec_at(xs_).h / 2e3, ("end", s[0], 0.0)
                    if s[1] == "interior":
                        return s[2] / 2, ("interior", s[0], s[2])
                    return 0.0, ("free", s[0], 0.0)
            return 0.0, (None, None, 0.0)

        off_a, kind_a = offset(a)
        off_b, kind_b = offset(b)
        lo, hi = a + off_a, b - off_b
        if hi <= lo:
            continue

        def a1_auto(u, ka=kind_a[0], kb=kind_b[0], mid=(a + b) / 2):
            """5.2.9: 0.9 near interior supports (the half span next to one; a cantilever arm), else 1.0"""
            if "free" in (ka, kb) and ({ka, kb} & {"end", "interior"}):
                return 0.9
            if (ka == "interior" and u <= mid + 1e-9) or (kb == "interior" and u >= mid - 1e-9):
                return 0.9
            return 1.0

        # 5.2.11 at an end support: the shear there (V0d), b and h0 of the support section
        if support_limit:
            for xs_, kind in ((a, kind_a), (b, kind_b)):
                if kind[0] != "end":
                    continue
                s0 = sec_at(xs_)
                hog = hog_at(xs_)
                sb = s0.flipped() if hog else s0
                bw, h0 = sb.web(), sb.h0(external=False)
                V0 = Vf(xs_)
                lim = 0.51e-3 * math.sqrt(s0.concrete.fcuk) * bw * h0
                res.limits.append(Check(f"section limit at the end support x = {xs_:g}", gamma0 * V0, lim, "kN",
                                        "JTG 3362-2018 5.2.11", {"b": bw, "h0": h0, "V0d": V0, "x": xs_},
                                        ["the shear at the support centre (V0d), b and h0 of the support section "
                                         "(program convention, the near-end-support segment of 5.2.11)"],
                                        section=s0, hogging=hog))

        def slope(u):
            """sign of d|M|/dx at u (the crack runs towards the larger moment)"""
            u0, u1 = max(u - delta, a), min(u + delta, b)
            d = abs(Mf(u1)) - abs(Mf(u0))
            return 0 if abs(d) <= 1e-12 * max(1.0, abs(Mf(u))) else (1 if d > 0 else -1)

        def C_of(u):
            hog = hog_at(u)
            s = sec_at(u)
            s = s.flipped() if hog else s
            h0 = s.h0(external=False) / 1e3
            Vu = Vf(u)
            m = 3.0 if Vu <= 0 else min(abs(Mf(u)) / (Vu * h0), 3.0)
            return 0.6 * m * h0, m, h0

        def build(xc, xt, source):
            C, m, _ = C_of(xc)
            hog = hog_at(xc)
            s0 = sec_at(xc)
            stir = stir_at(xc)
            lo_, hi_ = min(xc, xt), max(xc, xt)
            sv_rows = _zone_rows(stir_at)
            sb = s0.flipped() if hog else s0
            bw = sb.web()
            if hi_ - lo_ <= 1e-9:
                q = q_of(stir)
            elif sv_rows:
                q = sum(max(0.0, min(hi_, z1) - max(lo_, z0)) * q_of(st) for z0, z1, st in sv_rows) / (hi_ - lo_)
            else:  # a constant or a callable: the mean over the projection (40 midpoints)
                q = float(np.mean([q_of(stir_at(u)) for u in lo_ + (np.arange(40) + 0.5) / 40 * (hi_ - lo_)]))
            rho = q / bw
            # bent-up bars crossed (crack from the tension face at x_t to the compression face at x_c)
            cb, ct = (xc, xt) if hog else (xt, xc)  # x of the crack at the bottom and at the top
            crossing = []
            for xb_, xt_, A, th in bent:
                if (xb_ - cb) * (xt_ - ct) <= 0 and np.sign(xt_ - xb_) == -np.sign(ct - cb) and ct != cb:
                    crossing.append((A, th))
            a1 = a1_at(xc) if a1_at else (a1_auto(xc) if alpha1 == "auto" else alpha1)
            kw = dict(kw_at(xc))
            chk = _shear(s0, Vf(xc), stir, gamma0=gamma0, hogging=hog, alpha1=a1, bent=crossing, stirrup_steel=sst,
                         rho_sv=rho, **kw)
            # 5.2.11 over the range of the inclined section: least b and h0, largest Vd
            # every section zone overlapping the range is sampled (its middle within the range)
            zone_mid = [(max(lo_, z0) + min(hi_, z1)) / 2 for z0, z1, _ in (sec_rows or []) if z0 < hi_ and z1 > lo_]
            us = np.unique(np.r_[np.linspace(lo_, hi_, 9), xs[(xs >= lo_) & (xs <= hi_)], zone_mid])
            bmin = h0min = math.inf
            for u in us:
                su = sec_at(u)
                su = su.flipped() if hog else su
                bmin, h0min = min(bmin, su.web()), min(h0min, su.h0(external=False))
            Vmax = max(Vf(u) for u in us)
            vlim = 0.51e-3 * math.sqrt(s0.concrete.fcuk) * bmin * h0min
            chk.failed = [f for f in chk.failed if "(5.2.11)" not in f]
            if gamma0 * Vmax > vlim:
                chk.failed.append(f"section too small: γ0Vd,max = {gamma0 * Vmax:.0f} > 0.51e-3 √fcu,k b h0 = "
                                  f"{vlim:.0f} kN over the inclined section (5.2.11)")
            chk.details.update(V_section_max=vlim, b_min=bmin, h0_min=h0min, Vd_max_range=Vmax, rho_sv_mean=rho,
                               bent_crossed=crossing, C=C, m=m, x_c=xc, x_t=xt)
            chk.clause = "JTG 3362-2018 5.2.9–5.2.11"
            res.sections.append(InclinedSection(xc, xt, abs(xc - xt), m, Vf(xc), Mf(xc), chk, source))

        def from_compression(xc, source, forced=None):
            C, _, _ = C_of(xc)
            sg = slope(xc) if forced is None else forced
            if sg == 0:  # flat |M|: the tension end on the side of the smaller moment
                sg = 1 if abs(Mf(max(xc - C, a))) <= abs(Mf(min(xc + C, b))) else -1
            xt = xc - sg * C
            if lo - 1e-9 <= xt <= hi + 1e-9:
                build(xc, xt, source)
                return True
            return False

        def from_tension(xt, source):
            if not lo - 1e-9 <= xt <= hi + 1e-9:
                return
            sg = slope(xt)
            for direction in ([sg] if sg else [1, -1]):
                end = hi if direction > 0 else lo
                if abs(end - xt) <= 1e-9:
                    continue
                us = np.linspace(xt, end, 241)
                g = np.array([abs(u - xt) - C_of(u)[0] for u in us])
                k = np.nonzero((g[:-1] < 0) & (g[1:] >= 0))[0]
                if not len(k):
                    res.notes.append(f"{source}: no compression end within the span (C exceeds the distance)")
                    continue
                k = k[0]
                xc = brentq(lambda u: abs(u - xt) - C_of(u)[0], us[k], us[k + 1], xtol=1e-10)
                build(xc, xt, source)
                return

        # 5.2.8 positions of this span
        for xs_, kind in ((a, kind_a), (b, kind_b)):
            if kind[0] == "end":
                from_tension(lo if xs_ == a else hi, f"h/2 from end support x = {kind[1]:g}")
            elif kind[0] == "interior":
                xc = lo if xs_ == a else hi
                from_compression(xc, f"diaphragm edge, support x = {kind[1]:g}" if kind[2] > 0 else
                                 f"interior support x = {kind[1]:g}", forced=-1 if xs_ == a else 1)
        for p, label in t_pos:
            if lo - 1e-9 <= p <= hi + 1e-9:
                from_tension(p, label)
        for p, label in c_pos:
            if lo - 1e-9 <= p <= hi + 1e-9:
                from_compression(p, label)
        if grid:
            for u in xs:
                if lo - 1e-9 <= u <= hi + 1e-9:
                    from_compression(float(u), "grid")
    if not res.sections:
        raise ValueError("no inclined section fits within the spans")
    return res


# ── design (5.2.13) ─────────────────────────────────────────────────────────
@dataclass
class StirrupDesign:
    """stirrups from 5.2.13-1 (or, for prestressed members, the commentary's formula 5-9): legs, adopted diameter and
    spacing (mm); the required spacing or diameter; `check` = jtg.shear with the adopted stirrups on the share the
    concrete and stirrups carry (ξ Vd; prestressed: Vd with the bent tendons)"""
    legs: int
    dia: float
    spacing: float
    spacing_required: float | None
    dia_required: float | None
    check: Check
    details: dict = field(default_factory=dict)

    def __str__(self):
        req = (f"required spacing {self.spacing_required:.0f} mm" if self.spacing_required is not None
               else f"required diameter {self.dia_required:.1f} mm")
        return f"stirrups {self.legs} legs Φ{self.dia:g} @ {self.spacing:g} mm ({req}); back-substituted: {self.check}"


def stirrup_design(sec: RCSection, Vd: float, *, legs: int, dia: float | None = None, spacing: float | None = None,
                   xi: float = 0.6, gamma0: float = 1.0, alpha1: float | None = None, alpha3: float | None = None,
                   hogging: bool = False, stirrup_steel: str | Rebar | None = None, pc_class: str | None = None,
                   alpha2: float | None = None, bent_tendons=(), step: float | None = 10.0,
                   diameters=STIRRUP_DIAMETERS) -> StirrupDesign:
    """design of the stirrups (5.2.13 item 2). Vd: the design shear for stirrup design (kN, without γ0): V'd at h/2 from
    the support centre of simply supported beams and near end supports, at the diaphragm edge near interior supports of
    constant-depth beams, V0d at the end of a haunch (item 1). Give either the diameter (the spacing is designed) or the
    spacing (the diameter is designed). alpha1: default 0.9 with hogging=True (the segment next to an interior support,
    where V'd is taken at the diaphragm edge), else 1.0 (5.2.9).
    Reinforced concrete (5.2.13-1): sv = 0.2e-6 α1² α3² (2 + 0.6P) √fcu,k Asv fsv b h0² / (ξ γ0 Vd)², ξ ≥ 0.6 the share of
    the concrete and stirrups (the rest by bent-up bars, bent_bar_design). Prestressed (commentary to 5.2.13, formula
    5-9): sv = 0.2e-6 α1² α2² α3² (2 + 0.6P) √fcu,k Asv fsv b h0² / (γ0 Vd − Vpb)², Vpb from bent_tendons (5.2.9-4);
    pc_class or alpha2 as jtg.shear. b, h0, P as jtg.shear.
    Adopted: the spacing rounded down to `step` (None: not rounded) and within 9.3.12 (≤ h/2, ≤ 400 mm, ρsv ≥ the
    minimum); the diameter the next of `diameters` at or above the need and 8 mm. 5.2.13-1 writes 0.2e-6 for
    0.45² × 1e-6 = 0.2025e-6: the spacing it gives is 1.2 % shorter than 5.2.9-2 needs, and back-substituted
    Vcs = √(0.2025/0.2) ξ γ0 Vd. Detailing (warnings): 9.3.12 item 3, at most 100 mm within one beam depth of the
    support centre, where V'd is taken; item 1, a diameter of at least a quarter of the largest main bar (the tension
    edge rows, when the section knows the bar diameters)."""
    if (dia is None) == (spacing is None):
        raise ValueError("give either dia (the spacing is designed) or spacing (the diameter is designed)")
    s = sec.flipped() if hogging else sec
    c, st = s.concrete, s.steel
    sst = st if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
    a1_note = None
    if alpha1 is None:
        alpha1 = 0.9 if hogging else 1.0
        a1_note = f"α1 = {alpha1:g} ({'near an interior support' if hogging else 'near an end support'}, 5.2.9)"
    probe = _shear(sec, 0.0, (legs, dia or 10, spacing or 100), hogging=hogging, alpha1=alpha1, alpha3=alpha3,
                   stirrup_steel=sst, pc_class=pc_class, alpha2=alpha2, bent_tendons=bent_tendons)
    d = probe.details
    b, h0, P, a2, a3 = d["b"], d["h0"], d["P"], d["alpha2"], d["alpha3"]
    pre = s.prestressed
    if pre:
        demand = gamma0 * abs(Vd) - d["Vpb"]
        coef = 0.2e-6 * alpha1 ** 2 * a2 ** 2 * a3 ** 2
        notes = ["prestressed: commentary to 5.2.13, formula 5-9 (the stirrups carry γ0Vd − Vpb; ξ not used)"]
    else:
        if xi < 0.6 - 1e-12 or xi > 1.0:
            raise ValueError("ξ: the share of the concrete and stirrups, 0.6 ≤ ξ ≤ 1 (5.2.13 item 1)")
        demand = xi * gamma0 * abs(Vd)
        coef = 0.2e-6 * alpha1 ** 2 * a3 ** 2
        notes = [f"5.2.13-1 with ξ = {xi:g}"]
    if demand <= 0:
        raise ValueError("nothing left for the stirrups to carry")
    K = coef * (2 + 0.6 * P) * math.sqrt(c.fcuk) * sst.fsv * b * h0 ** 2  # sv = K Asv / demand²
    smax = min(s.h / 2, 400.0)
    details = {"b": b, "h0": h0, "P": P, "alpha1": alpha1, "alpha2": a2, "alpha3": a3, "fsv": sst.fsv,
               "demand": demand, "rho_sv_min": sst.rho_sv_min, "s_max": smax}
    if spacing is None:
        Asv = legs * math.pi * dia * dia / 4
        s_req = K * Asv / demand ** 2
        s_r = s_req if step is None else math.floor(s_req / step + 1e-9) * step
        s_ad = min(s_r, smax, Asv / (sst.rho_sv_min * b))
        if s_ad <= 0:
            raise ValueError(f"the required spacing {s_req:.1f} mm is below the step {step:g} mm: more legs or a "
                             "larger diameter")
        d_ad, d_req = float(dia), None
        details.update(spacing_exact_5_2_9=s_req * 0.2025 / 0.2)
    else:
        Asv_req = max(demand ** 2 * spacing / K, sst.rho_sv_min * b * spacing)
        d_req = math.sqrt(4 * Asv_req / legs / math.pi)
        fit = [dd for dd in diameters if dd >= max(d_req, 8.0) - 1e-9]
        if not fit:
            raise ValueError(f"a stirrup diameter of {d_req:.1f} mm is needed: more legs or a closer spacing")
        d_ad, s_ad, s_req = float(fit[0]), float(spacing), None
        if spacing > smax:
            notes.append(f"spacing {spacing:g} mm > {smax:.0f} mm (9.3.12)")
        details.update(Asv_required=Asv_req)
    if pre:
        chk = _shear(sec, Vd, (legs, d_ad, s_ad), gamma0=gamma0, hogging=hogging, alpha1=alpha1, alpha3=alpha3,
                     stirrup_steel=sst, pc_class=pc_class, alpha2=alpha2, bent_tendons=bent_tendons)
    else:
        chk = _shear(sec, xi * Vd, (legs, d_ad, s_ad), gamma0=gamma0, hogging=hogging, alpha1=alpha1, alpha3=alpha3,
                     stirrup_steel=sst)
        chk.name = f"shear share ξ = {xi:g} of the concrete and stirrups"
    chk.clause = "JTG 3362-2018 5.2.13, 5.2.9"
    chk.notes = notes + ([a1_note] if a1_note else []) + chk.notes
    if s_ad > 100 + 1e-9:
        chk.warnings.append(f"spacing {s_ad:g} mm: within one beam depth ({s.h:.0f} mm) of the support centre at most "
                            "100 mm (9.3.12 item 3)")
    bot, _ = s.edge_masks()
    dmain = s.bar_d[bot]
    dmain = dmain[~np.isnan(dmain)]
    if len(dmain) and d_ad < dmain.max() / 4 - 1e-9:
        chk.warnings.append(f"stirrup diameter {d_ad:g} mm < 1/4 of the main bars ({dmain.max():g} mm) (9.3.12 item 1)")
    return StirrupDesign(legs, d_ad, s_ad, s_req, d_req, chk, details)


@dataclass
class BentBarRow:
    x_top: float  # m, the bend at the top steel level
    x_bottom: float  # m, the bend at the bottom steel level
    x_read: float  # m, where its share Vsb was read off the envelope
    Vsb: float  # kN
    Asb: float  # mm², 5.2.13-2


@dataclass
class BentBarDesign:
    """bent-up bars from the shear envelope (5.2.13 items 1, 3, 4, 6; 9.3.10): the line ξ V'd splits the envelope
    (below: concrete and stirrups; above: the bent-up bars)"""
    rows: list
    Vd_design: float  # V'd (kN)
    x_design: float  # m
    V_line: float  # ξ V'd (kN)
    theta: float
    details: dict = field(default_factory=dict)

    def bent(self):
        """rows for jtg.shear_search(bent=…): (x at the bottom bend, x at the top bend, area, angle)"""
        return [(r.x_bottom, r.x_top, r.Asb, self.theta) for r in self.rows]

    def __str__(self):
        lines = [f"bent-up bars: V'd = {self.Vd_design:.1f} kN at x = {self.x_design:.3f} m, line ξV'd = "
                 f"{self.V_line:.1f} kN"]
        for k, r in enumerate(self.rows, 1):
            lines.append(f"  row {k}: top bend x = {r.x_top:.3f} m, bottom bend x = {r.x_bottom:.3f} m, "
                         f"Vsb = {r.Vsb:.1f} kN (at x = {r.x_read:.3f} m), Asb = {r.Asb:.0f} mm²")
        return "\n".join(lines)


def bent_bar_design(x, Vd, *, support: float, h: float, z: float, kind: str = "end", diaphragm: float = 0.0,
                    theta: float = 45.0, xi: float = 0.6, gamma0: float = 1.0, steel: str | Rebar = "HRB400",
                    rows=None, max_rows: int = 50) -> BentBarDesign:
    """bent-up bars of a constant-depth reinforced concrete beam (5.2.13, figures 5.2.13a, b). x, Vd: the shear envelope
    (m, kN without γ0) from the support into the span. support: x of the support centre; kind "end" (simply supported
    beams, continuous beams at end supports: V'd at h/2 from the support centre, the first row's top bend at the support
    centre, 9.3.10) or "interior" (continuous beams and cantilevers at interior supports: V'd and the first top bend at
    the diaphragm edge, diaphragm width in m). h: depth of the beam (mm); z: vertical distance between the two bends of
    a bar (mm); theta: the bend angle (45° preferred, 9.3.10).
    The line ξ V'd (ξ ≥ 0.6, at most 40 % for the bars) splits the envelope; row 1 takes the part above the line at the
    design section (item 3), each later row the part at the bottom bend of the row before (item 4), whose top bend it
    starts at (9.3.10: at or inside it); rows stop once the part above the line is used up. Asb = γ0 Vsb /
    (0.75e-3 fsd sin θs) (5.2.13-2). rows: your own layout instead, (x_top, x_bottom[, x where Vsb is read]) each, e.g. for
    haunched beams (figure 5.2.13c: in the haunch at the row's own bottom bend, across the junction the peak there, in
    the constant-depth part at the row's top bend)."""
    if xi < 0.6 - 1e-12 or xi > 1.0:
        raise ValueError("ξ: 0.6 ≤ ξ ≤ 1 (5.2.13 item 1: at most 40 % by bent-up bars)")
    if kind not in ("end", "interior"):
        raise ValueError("kind: 'end' or 'interior'")
    st = steel if isinstance(steel, Rebar) else _rebar(steel)
    x = np.asarray(x, float)
    V = np.abs(np.asarray(Vd, float))
    sgn = 1.0 if float(np.mean(x)) >= support else -1.0
    order = np.argsort(sgn * (x - support), kind="stable")
    xs, Vs = x[order], V[order]
    u = sgn * (xs - support)  # distance from the support into the span
    keep = u >= -1e-12
    u, Vs = u[keep], Vs[keep]
    Vf = lambda xx: float(np.interp(sgn * (xx - support), u, Vs))  # noqa: E731
    x_des = support + sgn * (h / 2e3 if kind == "end" else diaphragm / 2)
    Vdp = Vf(x_des)
    line = xi * Vdp
    run = z / 1e3 / math.tan(math.radians(theta))
    sin = math.sin(math.radians(theta))
    out = []
    if rows is None:
        top = support if kind == "end" else x_des
        read = x_des
        for _ in range(max_rows):
            Vsb = Vf(read) - line
            if Vsb <= 1e-9 * max(Vdp, 1.0):
                break
            bot = top + sgn * run
            out.append(BentBarRow(top, bot, read, Vsb, gamma0 * Vsb / (0.75e-3 * st.fsd * sin)))
            top, read = bot, bot
        else:
            raise RuntimeError("bent_bar_design: more rows than max_rows")
    else:
        prev = None
        for k, r in enumerate(rows):
            xt_, xb_ = float(r[0]), float(r[1])
            read = float(r[2]) if len(r) > 2 else (x_des if k == 0 else prev)
            Vsb = max(Vf(read) - line, 0.0)
            out.append(BentBarRow(xt_, xb_, read, Vsb, gamma0 * Vsb / (0.75e-3 * st.fsd * sin)))
            prev = xb_
    return BentBarDesign(out, Vdp, x_des, line, theta, {"run": run, "fsd": st.fsd, "xi": xi})
