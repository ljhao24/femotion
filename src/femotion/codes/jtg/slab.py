"""Deck slabs and punching, JTG 3362-2018: one-way and two-way slabs (4.2.1), calculation spans and the moments of slabs
monolithic with their ribs (4.2.2), the distribution of wheel loads on one-way slabs (4.2.3) and cantilevers (4.2.5,
with the Sawko formula of the commentary for longer ones), skew slabs (4.2.4), haunches (4.2.6); punching of slabs
(5.6.1, 5.6.2) and of spread footings (5.6.3); slab detailing (9.2.2–9.2.4, 9.2.9). The wheels are the vehicle load
(车辆荷载) of JTG D60-2015 4.3.1 with the impact factor μ = 0.3 of local loading (4.3.2 item 6).

Units. Slab strips (4.2): m, kN, per metre of slab width: loads kN/m², line loads kN/m, moments kN·m/m, shears kN/m;
sagging moments positive. Punching (5.6) and detailing (9.2): mm, mm², MPa and kN like the rest of femotion.codes.jtg.
Clauses, formulas and the program's conventions: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    from femotion.codes.jtg import slab
    s = slab.deck_slab(clear=1.6, t=0.18, h=0.10, h_rib=1.3, surfacing=2.4)          # slab between two T ribs
    s.M_support, s.M_mid, s.V                                                        # (dead, vehicle) per metre
    c = slab.cantilever_slab(0.9, t_root=0.20, t_tip=0.14, h=0.10, curb=0.9, surfacing=2.4)
    sec = jtg.rect(1000, 180, concrete="C40", bottom=[(8, 12, 30)], top=[(8, 12, 30)])  # one metre of slab
    jtg.flexure(sec, 1.2 * s.M_mid[0] + 1.8 * s.M_mid[1])                           # D60 4.1.5: vehicle 1.8
    slab.punching(Fld=650, Um=slab.punching_perimeter(420, bx=600, by=600), h0=420, h=480, concrete="C40")

Box girders as closed frames (table 6.1.3): femotion.codes.jtg.transverse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .actions import VEHICLE_AXLES, VEHICLE_SPACINGS, transverse_reduction
from .materials import Concrete, Rebar, concrete as _concrete, rebar as _rebar
from .rc import Check

# ── vehicle load of JTG D60-2015 4.3.1 (table 4.3.1-3, figures 4.3.1-2, 4.3.1-3) as wheels ──
WHEEL_CONTACT = ((0.3, 0.2), (0.6, 0.2), (0.6, 0.2), (0.6, 0.2), (0.6, 0.2))  # per axle: (width across, length along
#                                                                              the bridge) of one wheel's contact, m
WHEEL_TRACK = 1.8  # m between the wheels of an axle
WHEEL_GAP = 1.3  # m between the inner wheels of two vehicles side by side (0.6 m between the 2.5 m bodies)
CURB_CLEARANCE = 0.5  # m from an outer wheel to the curb
LOCAL_IMPACT = 0.3  # impact factor of local loading and of the cantilever slabs of T and box girders (D60 4.3.2 item 6)
GAMMA_RC = 26.0  # kN/m³, reinforced or prestressed concrete with ≥ 2 % steel (D60 table 4.2.1 and its commentary: 25 below 2 %)

# commentary to 4.2.5, table 4-3: A' of the Sawko formula against the load position lc / a0
SAWKO_A = ((0.25, 1.07), (0.50, 1.17), (0.75, 1.30), (1.00, 1.53))


# ── 4.2.1, 4.2.2, 4.2.4, 4.2.6: spans and moments ────────────────────────────
def slab_kind(l_long: float, l_short: float) -> str:
    """4.2.1: a slab supported on four sides is one-way ("one-way", spanning the short side) when the long side is at
    least twice the short side, else two-way ("two-way")"""
    lo, sh = max(l_long, l_short), min(l_long, l_short)
    return "one-way" if lo >= 2 * sh else "two-way"


def two_way_shares(l_long: float, l_short: float) -> tuple[float, float]:
    """shares of a two-way slab's load carried by the short span, (uniform load, concentrated load): l1⁴ / (l1⁴ + l2⁴)
    and l1³ / (l1³ + l2³) with l1 the long and l2 the short span (commentary to 4.2.1, quoting AASHTO 14th ed. 3.24.6;
    the code itself asks for a two-way analysis by elastic theory)"""
    l1, l2 = max(l_long, l_short), min(l_long, l_short)
    return l1 ** 4 / (l1 ** 4 + l2 ** 4), l1 ** 3 / (l1 ** 3 + l2 ** 3)


def slab_span(*, clear: float | None = None, centres: float | None = None, t: float | None = None,
              monolithic: bool = True, shear: bool = False) -> float:
    """calculation span (m) of a one-way slab (4.2.2): simply supported slabs the distance between the support centres
    (`centres`); slabs monolithic with their ribs: for moments the clear distance plus the slab thickness, at most the
    rib centre distance; for shears the clear distance"""
    if not monolithic:
        if centres is None:
            raise ValueError("a simply supported slab spans between its support centres: pass centres=…")
        return centres
    if clear is None:
        raise ValueError("a slab monolithic with its ribs: pass clear=… (the clear distance between the ribs)")
    if shear:
        return clear
    if t is None:
        raise ValueError("the moment span of a monolithic slab is clear + t: pass t=…")
    return clear + t if centres is None else min(clear + t, centres)


def continuous_moments(M0: float, t: float, h_rib: float) -> tuple[float, float]:
    """(support moment, mid-span moment) of a slab monolithic with its ribs from M0, the mid-span moment of a simply
    supported slab of the same calculation span (4.2.2): −0.7 M0 at the supports (4.2.2-1); +0.7 M0 at mid-span when
    the slab is at least a quarter of the rib depth (t / h_rib ≥ 1/4, 4.2.2-2), else +0.5 M0 (4.2.2-3). The commentary
    takes the support as fixed (−q l²/12 = −0.67 M0) and the mid-span as half fixed (+q l²/16 = +0.5 M0)."""
    return -0.7 * M0, (0.7 if t / h_rib >= 0.25 else 0.5) * M0


def skew_slab(l: float, b: float, phi: float, *, precast: bool = False) -> tuple[float, str]:
    """calculation span (m) of a skew slab (4.2.4) and how it is found: l the skew span (along the bridge axis), b the
    slab width square to the bridge axis, phi the skew angle (°, between the normal to the support axis and the bridge
    axis). Integral slabs with φ ≤ 15° as right slabs: l/b ≤ 1.3 the square distance between the support axes l cos φ,
    l/b > 1.3 the skew span. Precast hinged planks: the skew span (their width square between the plank edges).
    Integral slabs with 15° < φ < 40°: table 4-2 of the commentary (informative); larger angles need a skew-plate
    analysis (ValueError)."""
    lb = l / b
    right = l * math.cos(math.radians(phi))
    if precast:
        return l, "precast hinged planks: skew span (4.2.4)"
    if phi <= 15:
        return (right, "φ ≤ 15°, l/b ≤ 1.3: square span between the support axes (4.2.4)") if lb <= 1.3 else \
            (l, "φ ≤ 15°, l/b > 1.3: skew span (4.2.4)")
    if phi < 40:  # commentary table 4-2 (Olsen's tests)
        if lb >= 1.3:
            return l, "table 4-2 of the commentary (l/b ≥ 1.3, φ ≤ 40°): skew span"
        if lb >= 0.7:
            return (l + right) / 2, "table 4-2 of the commentary (l/b 1.3 … 0.7, 15° < φ < 40°): half of skew + square span"
        return right, "table 4-2 of the commentary (l/b < 0.7, φ < 40°): square span"
    raise ValueError(f"skew angle {phi:g}° > 15°: 4.2.4 does not cover it (table 4-2 of the commentary goes to 40°); "
                     "analyse the skew plate")


def haunch_depth(hf: float, s: float, tan_alpha: float) -> float:
    """calculation depth he = h'f + s tan α of a slab inside its haunch (4.2.6): hf the slab without the haunch, s the
    distance from the start of the haunch towards the rib centre line, tan α the haunch slope, at most 1/3 (the
    commentary: steeper haunches do not carry stress beyond the 1:3 line)"""
    return hf + s * min(tan_alpha, 1 / 3)


# ── 4.2.3, 4.2.5: wheel load distribution ───────────────────────────────────
def spread(b1: float, h: float) -> float:
    """distribution width parallel to the slab span b = b1 + 2h (4.2.3-1): the contact plus 45° through the surfacing"""
    return b1 + 2 * h


def width_mid(a1: float, h: float, l: float, d: float = 0.0) -> float:
    """distribution width square to the span of a wheel at mid-span (4.2.3-2): a = (a1 + 2h) + l/3 ≥ 2l/3; several
    equal wheels whose widths overlap (4.2.3-3): + d, the distance between the outer wheels"""
    return max(a1 + 2 * h + l / 3, 2 * l / 3) + d


def width_support(a1: float, h: float, t: float, x: float = 0.0, a_mid: float | None = None) -> float:
    """distribution width of a wheel at the support (4.2.3-4) a = (a1 + 2h) + t, or x from it (4.2.3-5) + 2x, at most
    the mid-span width a_mid; t the slab's mid-span thickness"""
    a = a1 + 2 * h + t + 2 * max(x, 0.0)
    return a if a_mid is None else min(a, a_mid)


def width_cantilever(a1: float, h: float, lc: float) -> float:
    """distribution width of a wheel on a cantilever slab (4.2.5) a = (a1 + 2h) + 2 lc, lc from the web's outer face to
    the outer edge of the load spread at 45° through the surfacing (figure 4.2.5); for lc ≤ 2.5 m"""
    return a1 + 2 * h + 2 * lc


def sawko_A(lc: float, a0: float) -> float:
    """A' of table 4-3 (commentary to 4.2.5) for the load position lc / a0, linear between 0.25 and 1.00"""
    r = lc / a0
    xs, ys = zip(*SAWKO_A)
    if not xs[0] - 1e-9 <= r <= xs[-1] + 1e-9:
        raise ValueError(f"lc / a0 = {r:.3f}: table 4-3 gives A' for 0.25 … 1.00 only")
    return float(np.interp(r, xs, ys))


def sawko_moment(P: float, lc: float, a0: float, y: float = 0.0) -> float:
    """root moment per metre (kN·m/m, hogging negative) of a long cantilever slab under a concentrated load P (kN) at lc
    from the root, at y (m) along the root from the load (commentary to 4.2.5, formula 4-5, for lc > 2.5 m):
    mx = −(P/π) A' / ch(A' y / lc). Its integral along the root is −P lc for any A' (statics)."""
    A = sawko_A(lc, a0)
    return -P / math.pi * A / math.cosh(A * y / lc)


@dataclass(frozen=True)
class AxleGroup:
    """the wheels of consecutive axles in one wheel line, loading a transverse slab strip together (4.2.3-3)"""
    P: float  # kN, the sum of one wheel of each axle
    d: float  # m, between the outer axles (0 for one axle)
    gaps: tuple  # m, between consecutive axles (for the overlap test)
    a1: float  # m, contact length along the bridge (square to a transversely spanning slab)
    b1: float  # m, contact width across the bridge (parallel to the span)


def axle_groups(axles=VEHICLE_AXLES, spacings=VEHICLE_SPACINGS, contact=WHEEL_CONTACT) -> list[AxleGroup]:
    """every run of consecutive axles of the vehicle as one wheel line on a slab spanning across the bridge: the wheel
    loads (half the axle loads), the distance d between the outer axles, the contact (the widest of the run)"""
    out = []
    n = len(axles)
    for i in range(n):
        for j in range(i, n):
            gaps = tuple(spacings[i:j])
            out.append(AxleGroup(sum(axles[i:j + 1]) / 2, float(sum(gaps)), gaps,
                                 max(c[1] for c in contact[i:j + 1]), max(c[0] for c in contact[i:j + 1])))
    return out


def group_widths(a: float, groups, cap: float | None = None) -> list[tuple[float, float, AxleGroup]]:
    """(load per metre along the bridge kN/m, distribution width m, group) of every group whose single-wheel widths
    overlap (every gap < a, 4.2.3-3): width a + d, at most `cap` (4.2.3 items 5, 6: the slab's full width or a precast
    slab's width). One wheel always qualifies. Program convention: the overlap rule of 4.2.3-3 (written for wheels at
    mid-span) is applied wherever a is found (4.2.3-4, -5, 4.2.5)."""
    out = []
    for g in groups:
        if all(s < a for s in g.gaps):
            w = a + g.d
            if cap is not None:
                w = min(w, cap)
            out.append((g.P / w, w, g))
    return out


@dataclass
class DeckLayout:
    """where a deck slab is supported across the bridge, for the distribution widths of 4.2.3 and 4.2.5.
    faces: (left, right) faces of each rib or web, left to right (m, across the bridge); beyond the first and the last
    the slab is a cantilever. t: mid-span thickness of the interior spans (m; one value or one per span). h: surfacing
    thickness (m). centres: rib centre lines (default the middle of the faces). full_width, precast_width: caps of
    4.2.3 items 5, 6 (m)."""
    faces: list
    t: float | list
    h: float
    centres: list | None = None
    full_width: float | None = None
    precast_width: float | None = None

    def __post_init__(self):
        self.faces = [tuple(map(float, f)) for f in self.faces]
        n = len(self.faces) - 1
        self._t = list(np.broadcast_to(np.asarray(self.t, float), (max(n, 1),)))
        self._c = [0.5 * (a + b) for a, b in self.faces] if self.centres is None else list(self.centres)

    def span(self, k: int) -> float:
        """calculation span (m) of interior span k for the widths (4.2.2: clear + t, at most the centre distance)"""
        clear = self.faces[k + 1][0] - self.faces[k][1]
        return min(clear + self._t[k], self._c[k + 1] - self._c[k])

    def width(self, x: float, a1: float, b1: float) -> tuple[float, str, float]:
        """(distribution width a of one wheel, clause, lc or the distance to the support) for a wheel centred at x with
        contact a1 × b1 (a1 square to the span). Outboard of the outer ribs' centre lines, as long as the spread b
        reaches past the outer face: the cantilever, 4.2.5 with lc from the face to the outer edge of the spread. Over a
        rib otherwise: 4.2.3-4. Interior span: 4.2.3-5 with x the distance from the nearer face to the near edge of the
        spread (program convention), at most the mid-span width 4.2.3-2."""
        b = spread(b1, self.h)
        F, C = self.faces, self._c
        cap = min(v for v in (self.full_width, self.precast_width, math.inf) if v is not None)
        lc = F[0][0] - x + b / 2 if x < C[0] else (x - F[-1][1] + b / 2 if x > C[-1] else 0.0)
        if lc > 0:
            return min(width_cantilever(a1, self.h, lc), cap), "4.2.5", lc
        for k, (fl, fr) in enumerate(F):
            if fl <= x <= fr or (k == 0 and x < C[0]) or (k == len(F) - 1 and x > C[-1]):
                t = self._t[0 if k == 0 else k - 1]  # the span on the rib's left (the first span for the first rib)
                return min(width_support(a1, self.h, t), cap), "4.2.3-4", 0.0
        k = max(i for i in range(len(F) - 1) if F[i][1] < x)
        l, t = self.span(k), self._t[k]
        am = width_mid(a1, self.h, l)
        xs = max(min(x - b / 2 - F[k][1], F[k + 1][0] - (x + b / 2)), 0.0)
        a = width_support(a1, self.h, t, xs, am)
        return min(a, cap), ("4.2.3-2" if a >= am else "4.2.3-5"), xs


# ── vehicles across a strip ─────────────────────────────────────────────────
def best_row(xs: np.ndarray, vals: np.ndarray, *, kmax: int, lane_factors: bool = True, sense: int = 1):
    """the row of vehicles across the bridge that maximises (sense=+1) or minimises (−1) a response. Vehicle i has its
    left wheel at xs[i] (uniform grid) and its right wheel 1.8 m to the right; the next one's left wheel is at least
    1.8 + 1.3 m further (D60 figure 4.3.1-3). vals[i]: the response to one vehicle at xs[i]. With lane_factors the sum
    of k vehicles is multiplied by table 4.3.1-5 (1.20 for one, 1.00 for two, …), and k ≤ kmax. Returns (response, k,
    left-wheel positions); (0, 0, []) when no vehicle is unfavourable (D60 4.1.4)."""
    n = len(xs)
    if n == 0 or kmax < 1:
        return 0.0, 0, []
    dx = xs[1] - xs[0] if n > 1 else 1.0
    gap = int(math.ceil((WHEEL_TRACK + WHEEL_GAP) / dx - 1e-9))
    v = sense * np.asarray(vals, float)
    f = [np.zeros(n)]
    hs = []
    for k in range(1, kmax + 1):
        prev = np.full(n, -np.inf)
        if k == 1:
            prev[:] = 0.0
        elif gap < n:
            prev[gap:] = f[-1][:-gap] if gap > 0 else f[-1]
        h = v + prev
        hs.append(h)
        f.append(np.maximum.accumulate(h))
    best, kbest = 0.0, 0
    for k in range(1, kmax + 1):
        val = f[k][-1]
        if not np.isfinite(val):
            break
        fac = transverse_reduction(k) if lane_factors else 1.0
        if fac * val > best + 1e-12:
            best, kbest = fac * val, k
    pos, lim = [], n - 1
    for k in range(kbest, 0, -1):
        i = int(np.argmax(hs[k - 1][:lim + 1]))
        pos.append(float(xs[i]))
        lim = i - gap
    return sense * best, kbest, sorted(pos)


def _ss_moment_int(lo, hi, l, xi):
    """∫ from lo to hi of the moment influence line at xi of a simply supported span 0 … l (lo, hi within the span)"""
    def F(x):  # antiderivative of x (l − xi)/l (x ≤ xi) and xi (l − x)/l (x ≥ xi)
        if x <= xi:
            return (l - xi) / l * x * x / 2
        return (l - xi) / l * xi * xi / 2 + xi / l * (l * (x - xi) - (x * x - xi * xi) / 2)
    return F(hi) - F(lo)


def _ss_reaction_int(lo, hi, l):
    """∫ from lo to hi of the left reaction's influence line (1 − x/l) of a span 0 … l"""
    return (hi - lo) - (hi * hi - lo * lo) / (2 * l)


@dataclass
class SlabEffects:
    """per-metre effects of a deck slab: pairs (permanent, vehicle); vehicle effects include (1 + μ) and the transverse
    lane factor, not the partial factors (D60 4.1.5: γQ = 1.8 for the vehicle load)"""
    clause: str
    spans: dict
    M0: tuple = (0.0, 0.0)  # kN·m/m, simply supported mid-span moment (4.2.2)
    M_support: tuple = (0.0, 0.0)
    M_mid: tuple = (0.0, 0.0)
    V: tuple = (0.0, 0.0)  # kN/m at the support (root)
    details: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _wheel_values(xs, width_of, effect, h, mu, cap=None):
    """response to one wheel centred at each x of xs, × (1 + μ): of the axle groups whose wheels overlap (group_widths;
    the single-wheel width width_of(x, a1, b1) taken with each group's contact), the one with the largest response,
    its load P/a per metre spread over b1 + 2h. effect(lo, hi): ∫ of the influence line from lo to hi (m)."""
    groups = axle_groups()
    contacts = sorted({(g.a1, g.b1) for g in groups})
    out = np.zeros(len(xs))
    for i, x in enumerate(xs):
        best = 0.0
        for a1, b1 in contacts:
            a = width_of(float(x), a1, b1)
            bb = spread(b1, h)
            e = effect(x - bb / 2, x + bb / 2)
            if e == 0.0:
                continue
            for w, _, gr in group_widths(a, [g for g in groups if (g.a1, g.b1) == (a1, b1)], cap):
                v = w / bb * e
                if abs(v) > abs(best):
                    best = v
        out[i] = best
    return (1 + mu) * out


def deck_slab(clear: float, t: float, *, h: float, h_rib: float | None = None, centres: float | None = None,
              monolithic: bool = True, surfacing: float = 0.0, g: float = 0.0, gamma_c: float = GAMMA_RC,
              mu: float = LOCAL_IMPACT, lanes: int = 4, lane_factors: bool = True, full_width: float | None = None,
              precast_width: float | None = None, moment_width: str = "mid", dx: float = 0.01) -> SlabEffects:
    """one-way deck slab between two ribs, spanning across the bridge, by 4.2.2 and 4.2.3 (per metre of slab):
    M0 of the simply supported slab of the moment span (clear + t ≤ rib centres; `centres` for the supports of a
    simply supported slab, monolithic=False) at mid-span; support −0.7 M0, mid-span +0.7 M0 or +0.5 M0 (t / h_rib);
    shear at the support of the simply supported slab of the clear span (4.2.2). Permanent loads: γc t + surfacing + g
    (kN/m²) over the span. Vehicle: the wheels of D60's vehicle (a1 = 0.2 m along the bridge, b1 = 0.6 m across; the
    heaviest overlapping axle group by 4.2.3-3), each spread over b = b1 + 2h and the width a of DeckLayout.width at its
    position (4.2.3-2, -4, -5, items 5, 6), × (1 + μ), only the part on the clear span loading it; vehicles side by side
    (1.8 m track, 1.3 m between vehicles, up to `lanes`) placed for the largest effect, × table 4.3.1-5 with
    lane_factors. h: surfacing thickness (m).
    moment_width: "mid" (default) spreads every wheel over the mid-span width 4.2.3-2 for M0, the mid-span moment, as
    the textbooks do; "position" uses the width at the wheel's position as for the shear (program convention)."""
    lM = slab_span(clear=clear, centres=centres, t=t, monolithic=monolithic)
    lV = slab_span(clear=clear, centres=centres, monolithic=monolithic, shear=True)
    if monolithic and h_rib is None:
        raise ValueError("a slab monolithic with its ribs needs h_rib for 4.2.2-2 / -3")
    q = gamma_c * t + surfacing + g
    lay = DeckLayout([(-clear / 2 - 10.0, -clear / 2), (clear / 2, clear / 2 + 10.0)], t, h,
                     centres=None if centres is None else [-centres / 2, centres / 2],
                     full_width=full_width, precast_width=precast_width)
    width_of = lambda x, a1, b1: lay.width(x, a1, b1)[0]
    cap = min(v for v in (full_width, precast_width, math.inf) if v is not None)
    if moment_width == "mid":
        width_M = lambda x, a1, b1: min(width_mid(a1, h, lM), cap)
    elif moment_width == "position":
        width_M = width_of
    else:
        raise ValueError("moment_width: 'mid' or 'position'")
    c0, c1 = -clear / 2, clear / 2

    def on_span(f):  # the part of a spread that lies on the clear span, in the strip's own coordinates
        def eff(lo, hi):
            lo, hi = max(lo, c0), min(hi, c1)
            return 0.0 if hi <= lo else f(lo, hi)
        return eff

    eM = on_span(lambda lo, hi: _ss_moment_int(lo + lM / 2, hi + lM / 2, lM, lM / 2))
    eV = on_span(lambda lo, hi: _ss_reaction_int(lo + lV / 2, hi + lV / 2, lV))
    b = spread(max(g_.b1 for g_ in axle_groups()), h)
    xs = np.arange(c0 - b / 2 - WHEEL_TRACK - dx, c1 + b / 2 + dx, dx)  # left wheels that can touch the span
    rM = _wheel_values(np.r_[xs, xs[-1] + dx * np.arange(1, int(round(WHEEL_TRACK / dx)) + 1)], width_M, eM, h, mu)
    rV = _wheel_values(np.r_[xs, xs[-1] + dx * np.arange(1, int(round(WHEEL_TRACK / dx)) + 1)], width_of, eV, h, mu)
    k = int(round(WHEEL_TRACK / dx))
    vM = rM[:len(xs)] + rM[k:k + len(xs)]
    vV = rV[:len(xs)] + rV[k:k + len(xs)]
    M0Q, kM, posM = best_row(xs, vM, kmax=lanes, lane_factors=lane_factors)
    VQ, kV, posV = best_row(xs, vV, kmax=lanes, lane_factors=lane_factors)
    M0 = (q * lM ** 2 / 8, M0Q)
    out = SlabEffects("JTG 3362-2018 4.2.2, 4.2.3", {"moment": lM, "shear": lV}, M0=M0, V=(q * lV / 2, VQ),
                      details={"q": q, "b": b, "a_mid": width_mid(axle_groups()[0].a1, h, lM), "vehicles_M": posM,
                               "vehicles_V": posV, "mu": mu})
    if monolithic:
        sG, mG = continuous_moments(M0[0], t, h_rib)
        sQ, mQ = continuous_moments(M0[1], t, h_rib)
        out.M_support, out.M_mid = (sG, sQ), (mG, mQ)
        out.notes.append(f"t / h_rib = {t / h_rib:.3f} {'≥' if t / h_rib >= 0.25 else '<'} 1/4: mid-span "
                         f"{0.7 if t / h_rib >= 0.25 else 0.5} M0 (4.2.2)")
    else:
        out.M_mid = M0
    return out


def cantilever_slab(length: float, t_root: float, t_tip: float, *, h: float, curb: float | None = None,
                    surfacing: float | tuple = 0.0, line_loads=(), gamma_c: float = GAMMA_RC, mu: float = LOCAL_IMPACT,
                    lanes: int = 4, lane_factors: bool = True, crowd: tuple | None = None,
                    dx: float = 0.01) -> SlabEffects:
    """root moment and shear (per metre; hogging negative) of a cantilever slab of `length` (m, from the web's outer
    face, s outwards) tapering from t_root to t_tip (m). surfacing: kN/m² over the whole length, or (q, s1, s2) over
    s1 … s2; line_loads: (P kN/m, s) such as barriers or railings; crowd: (q kN/m², s1, s2) for a sidewalk (a variable
    action, reported separately in details["crowd"] as (M, V)). Vehicle: wheels by 4.2.5 (a = a1 + 2h + 2lc, lc to the
    outer edge of the spread), the heaviest overlapping axle group (program convention, 4.2.3-3 applied to the
    cantilever), × (1 + μ); the outer wheel at least 0.5 m from the curb at s = `curb` (default the tip), the vehicle's
    other wheel 1.8 m inboard; vehicles side by side as in deck_slab."""
    L = length
    MG = -gamma_c * L * L * (t_root / 6 + t_tip / 3)
    VG = gamma_c * L * (t_root + t_tip) / 2
    q, s1, s2 = (surfacing, 0.0, L) if np.ndim(surfacing) == 0 else tuple(surfacing)
    MG -= q * (s2 * s2 - s1 * s1) / 2
    VG += q * (s2 - s1)
    for P, s in line_loads:
        MG -= P * s
        VG += P
    out = SlabEffects("JTG 3362-2018 4.2.5", {"length": L}, M_support=(MG, 0.0), V=(VG, 0.0))
    if crowd is not None:
        qc, c1, c2 = crowd
        out.details["crowd"] = (-qc * (c2 * c2 - c1 * c1) / 2, qc * (c2 - c1))
    smax = (L if curb is None else curb) - CURB_CLEARANCE
    def width_of(s, a1, b1):
        return width_cantilever(a1, h, max(s + spread(b1, h) / 2, 0.0))

    def on(f):
        def eff(lo, hi):
            lo, hi = max(lo, 0.0), min(hi, L)
            return 0.0 if hi <= lo else f(lo, hi)
        return eff

    eM = on(lambda lo, hi: -(hi * hi - lo * lo) / 2)
    eV = on(lambda lo, hi: hi - lo)
    k = int(round(WHEEL_TRACK / dx))
    ss = smax - dx * np.arange(int(math.ceil((smax + lanes * (WHEEL_TRACK + WHEEL_GAP) + 1.0) / dx)) + k)
    rM = _wheel_values(ss, width_of, eM, h, mu)
    rV = _wheel_values(ss, width_of, eV, h, mu)
    n = len(ss) - k
    xs = -ss[:n]  # the outer wheel as the vehicle's left wheel, x = −s growing inboard
    MQ, kM, posM = best_row(xs, rM[:n] + rM[k:k + n], kmax=lanes, lane_factors=lane_factors, sense=-1)
    VQ, kV, posV = best_row(xs, rV[:n] + rV[k:k + n], kmax=lanes, lane_factors=lane_factors)
    out.M_support, out.V = (MG, MQ), (VG, VQ)
    lcmax = max((s + spread(WHEEL_CONTACT[-1][0], h) / 2 for s, r in zip(ss, rM) if r != 0), default=0.0)
    out.details.update({"wheel_s_M": [-p for p in posM], "wheel_s_V": [-p for p in posV], "lc_max": lcmax, "mu": mu})
    if lcmax > 2.5:
        out.notes.append(f"lc up to {lcmax:.2f} m > 2.5 m: 4.2.5 does not apply; the commentary gives the Sawko formula "
                         "(sawko_moment)")
    return out


# ── 5.6 punching ────────────────────────────────────────────────────────────
def _c(x) -> Concrete:
    return x if isinstance(x, Concrete) else _concrete(x)


def _s(x) -> Rebar:
    return x if isinstance(x, Rebar) else _rebar(x)


def beta_h(h: float) -> float:
    """size factor βh of 5.6.1 from the slab depth h (mm): 1.0 up to 300 mm, 0.85 from 800 mm, linear between"""
    return float(np.interp(h, (300.0, 800.0), (1.0, 0.85)))


def punching_perimeter(h0: float, *, bx: float | None = None, by: float | None = None, d: float | None = None) -> float:
    """Um (mm) of 5.6.1: the perimeter of the punching cone's section h0/2 from the loaded area, for an interior
    rectangle bx × by: 2 (bx + h0) + 2 (by + h0) (figure 5.6.1, square corners); a circular column of diameter d as a
    square of 0.8 d. Near free edges or openings pass Um directly (the code does not cover them)."""
    if d is not None:
        bx = by = 0.8 * d
    if bx is None:
        raise ValueError("pass bx (and by) or d")
    by = bx if by is None else by
    return 2 * (bx + h0) + 2 * (by + h0)


def punching(Fld: float, *, Um: float, h0: float, h: float, concrete, sigma_pc: float = 0.0, stirrups=None,
             bent=None, steel="HRB400", Um_outer: float | None = None, detailing: dict | None = None,
             gamma0: float = 1.0) -> Check:
    """punching of a slab under a concentrated force (5.6). Fld (kN): the largest concentrated design force; for a slab
    on a column the column force less the loads inside the cone. Um (mm): punching_perimeter; h0, h (mm); sigma_pc:
    σpc,m, the mean effective precompression of a prestressed slab (MPa; the code advises 1.0 … 3.5).
        no punching steel (5.6.1):  γ0 Fld ≤ (0.7 βh ftd + 0.15 σpc,m) Um h0
        stirrups=Asvu (mm², all legs crossing the cone's inclined face) (5.6.2-2) or bent=(Asbu mm², θ °) (5.6.2-3):
            γ0 Fld ≤ (0.35 βh ftd + 0.15 σpc,m) Um h0 + 0.75 fsv Asvu   or   + 0.75 fsd Asbu sin θ
        with punching steel the section limit γ0 Fld ≤ 1.05 βh ftd Um h0 (5.6.2-1), and outside the reinforced zone
        5.6.1 again with Um_outer, the least favourable perimeter 0.5 h0 beyond it (5.6.2).
    detailing (9.2.9): {"stirrup_d": mm, "stirrup_s": mm} and/or {"bent_d": mm, "bent_n": per direction}; the slab
    must be at least 150 mm thick; bent bars at 30°…45°. Steel: `steel` for stirrups (fsv) and bent bars (fsd)."""
    c = _c(concrete)
    st = _s(steel)
    bh = beta_h(h)
    if stirrups is not None and bent is not None:
        raise ValueError("5.6.2 gives one formula for stirrups (5.6.2-2) and one for bent bars (5.6.2-3): pass one")
    d = {"beta_h": bh, "Um": Um, "h0": h0, "sigma_pc_m": sigma_pc, "ftd": c.ftd}
    notes = []
    if sigma_pc and not 1.0 <= sigma_pc <= 3.5:
        notes.append(f"σpc,m = {sigma_pc:.2f} MPa outside 1.0 … 3.5 MPa advised by 5.6.1")
    if stirrups is None and bent is None:
        R = (0.7 * bh * c.ftd + 0.15 * sigma_pc) * Um * h0 / 1e3
        return Check("punching (slab, 5.6.1)", gamma0 * Fld, R, "kN", "JTG 3362-2018 5.6.1", d, notes)
    conc = (0.35 * bh * c.ftd + 0.15 * sigma_pc) * Um * h0 / 1e3
    if stirrups is not None:
        steel_part = 0.75 * st.fsv * stirrups / 1e3
        clause, d["Asvu"], d["fsv"] = "JTG 3362-2018 5.6.2-2", stirrups, st.fsv
    else:
        Asbu, th = bent
        steel_part = 0.75 * st.fsd * Asbu * math.sin(math.radians(th)) / 1e3
        clause, d["Asbu"], d["theta"] = "JTG 3362-2018 5.6.2-3", Asbu, th
    chk = Check("punching (slab with punching steel, 5.6.2)", gamma0 * Fld, conc + steel_part, "kN", clause,
                d | {"V_concrete": conc, "V_steel": steel_part}, notes)
    vmax = 1.05 * bh * c.ftd * Um * h0 / 1e3
    chk.details["V_section_max"] = vmax
    if gamma0 * Fld > vmax:
        chk.failed.append(f"section too small: γ0Fld > 1.05 βh ftd Um h0 = {vmax:.0f} kN (5.6.2-1)")
    if Um_outer is None:
        chk.notes.append("outside the punching steel 5.6.1 must hold on the perimeter 0.5 h0 beyond it: pass Um_outer")
    else:
        out = (0.7 * bh * c.ftd + 0.15 * sigma_pc) * Um_outer * h0 / 1e3
        chk.details["V_outer"] = out
        if gamma0 * Fld > out:
            chk.failed.append(f"outside the punching steel: γ0Fld > {out:.0f} kN (5.6.1 on Um = {Um_outer:.0f} mm, 5.6.2)")
    if h < 150:
        chk.failed.append(f"slab {h:.0f} mm < 150 mm with punching steel (9.2.9)")
    dt = detailing or {}
    if stirrups is not None:
        if "stirrup_d" in dt and dt["stirrup_d"] < 8:
            chk.failed.append(f"punching stirrups {dt['stirrup_d']} mm < 8 mm (9.2.9)")
        if "stirrup_s" in dt and dt["stirrup_s"] > h0 / 3:
            chk.failed.append(f"punching stirrup spacing {dt['stirrup_s']} mm > h0/3 = {h0 / 3:.0f} mm (9.2.9)")
    else:
        if not 30 <= bent[1] <= 45:
            chk.failed.append(f"bent bars at {bent[1]}°: 9.2.9 asks for 30° … 45°")
        if "bent_d" in dt and dt["bent_d"] < 12:
            chk.failed.append(f"bent bars {dt['bent_d']} mm < 12 mm (9.2.9)")
        if "bent_n" in dt and dt["bent_n"] < 5:
            chk.failed.append(f"{dt['bent_n']} bent bars per direction < 5 (9.2.9)")
    return chk


def footing_punching_area(B: float, L: float, bt: float, bc: float, h0: float) -> float:
    """area A (mm²) of figure 5.6.3 (ABCDEF) on one side of a rectangular footing B × L under a column bt × bc: B and bt
    along the checked face, L and bc square to it; the base strip beyond the cone's base line (bc/2 + h0 from the
    centre) less the corners cut by the 45° lines from the cone's base corners. A step: the upper step as the column."""
    w = L / 2 - bc / 2 - h0  # strip depth beyond the cone
    if w <= 0:
        return 0.0
    c = B / 2 - bt / 2 - h0  # from the cone's base corner to the footing edge, along the face
    # each corner removes ∫0^w max(c − s, 0) ds
    cut = 0.0 if c <= 0 else (c * c / 2 if c <= w else c * w - w * w / 2)
    return w * B - 2 * cut


def footing_punching(*, ps: float, A: float, bt: float, h0: float, h: float, concrete, bb: float | None = None,
                     gamma0: float = 1.0) -> Check:
    """punching of a spread footing under a rectangular column, at the column or at a step (5.6.3):
    γ0 Fld ≤ 0.7 βh ftd bm h0 (5.6.3-1), Fld = ps A (5.6.3-2), bm = (bt + bb)/2 (5.6.3-3). ps (MPa): the design
    net ground pressure (the largest under eccentric load); A (mm²): footing_punching_area; bt (mm): the column (or
    upper step) width on the critical side, bb = bt + 2 h0 by default (mm); h (mm) for βh (5.6.1)."""
    c = _c(concrete)
    bb = bt + 2 * h0 if bb is None else bb
    bm = (bt + bb) / 2
    bh = beta_h(h)
    Fld = ps * A / 1e3
    R = 0.7 * bh * c.ftd * bm * h0 / 1e3
    return Check("punching (footing, 5.6.3)", gamma0 * Fld, R, "kN", "JTG 3362-2018 5.6.3",
                 {"Fld": Fld, "bm": bm, "bb": bb, "beta_h": bh, "A": A})


# ── 9.2 detailing ───────────────────────────────────────────────────────────
def slab_detailing(t: float, *, main: tuple, distribution: tuple | None = None, sidewalk: bool = False,
                   precast: bool = False, hollow: bool = False, bent_at_support: tuple | None = None) -> Check:
    """detailing of a deck slab per metre (9.2.1, 9.2.2, 9.2.3, 9.2.4). t (mm); main, distribution: (bar diameter mm,
    spacing mm). Roadway slabs: main bars ≥ 10 mm (sidewalks 8 mm), spacing ≤ 200 mm at the critical sections;
    distribution bars ≥ 8 mm (sidewalks 6 mm), spacing ≤ 200 mm, area preferably ≥ 0.1 % of the slab (roadway).
    Thickness: hollow slabs' top and bottom ≥ 80 mm; sidewalk slabs ≥ 80 mm cast in place, 60 mm precast.
    bent_at_support: (bars through the support per metre, their area / the main bars' area) ≥ (3, 1/4) (9.2.3).
    Returns a Check (resistance 1, demand 0) whose `failed` lists the unmet rules."""
    chk = Check("slab detailing (9.2)", 0.0, 1.0, "-", "JTG 3362-2018 9.2.1–9.2.4", {"t": t})
    d, s = main
    dmin = 8 if sidewalk else 10
    if d < dmin:
        chk.failed.append(f"main bars {d} mm < {dmin} mm (9.2.2)")
    if s > 200:
        chk.failed.append(f"main bar spacing {s} mm > 200 mm (9.2.2)")
    if distribution is not None:
        dd, ds = distribution
        dmin = 6 if sidewalk else 8
        if dd < dmin:
            chk.failed.append(f"distribution bars {dd} mm < {dmin} mm (9.2.4)")
        if ds > 200:
            chk.failed.append(f"distribution bar spacing {ds} mm > 200 mm (9.2.4)")
        if not sidewalk:
            r = math.pi * dd * dd / 4 * 1000 / ds / (t * 1000)
            chk.details["distribution_ratio"] = r
            if r < 0.001:
                chk.notes.append(f"distribution bars {r:.3%} < 0.1 % of the slab (9.2.4, advisory)")
    if hollow and t < 80:
        chk.failed.append(f"hollow slab top / bottom {t} mm < 80 mm (9.2.1)")
    if sidewalk and t < (60 if precast else 80):
        chk.failed.append(f"sidewalk slab {t} mm < {60 if precast else 80} mm (9.2.1)")
    if bent_at_support is not None:
        n, share = bent_at_support
        if n < 3 or share < 0.25:
            chk.failed.append(f"straight bars through the support: {n} per metre, {share:.0%} of the main bars; "
                              "9.2.3 asks for ≥ 3 and ≥ 1/4")
    return chk
