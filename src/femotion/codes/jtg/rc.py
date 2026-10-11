"""Ultimate-limit-state resistance of reinforced-concrete sections, JTG 3362-2018 (no prestress).

Units: section dimensions and bar positions in mm, bar areas in mm², stresses in MPa; design forces in kN and kN·m (as
they come out of the frame solver). Every check takes the design effect Sd from the combination and γ0, and verifies
γ0 Sd ≤ R. Clauses and the review against the official text: docs/jtg.md.

A section is a stack of pieces along its depth (z measured down from the top fibre): rectangles (solid or holes) and
circles, so rectangles, T, I and box girders, solid and hollow piers are all covered; bars are (area, depth) points.
Bending is about the horizontal axis; sagging puts the top in compression. A hogging moment is checked on the section
turned upside down.

    from femotion.codes import jtg
    sec = jtg.tee(b=200, h=1300, bf=1600, hf=150, concrete="C40", steel="HRB400",
                  bottom=[(6, 32, 90)], top=[(4, 16, 40)])      # rows of (count, diameter, depth to bar centre)
    r = jtg.flexure(sec, Md=2950.0, gamma0=1.0)                # kN·m
    v = jtg.shear(sec, Vd=780.0, stirrups=(2, 10, 150))         # legs, diameter, spacing
    c = jtg.eccentric(jtg.circle(D=1500, n=28, d=28, cover=60, concrete="C35"), Nd=9800, Md=3100, l0=18000)
    print(r, v, c)                                              # one line each: demand, resistance, ratio, verdict

Behind the code's simplified formulas sits one rule set, used here for every shape: plane sections; concrete tension
ignored; a rectangular block of fcd over x = β·xc; bar stress σi = εcu·Es·(β·h0i/x − 1) within −fsd' … fsd. With the usual
reinforcement (tension bars yield when ξ ≤ ξb, compression bars yield when x ≥ 2as') this gives exactly the code's
formulas for rectangular and T sections, and the code's general formula for bars that do not yield. The special rule for
x < 2as' (moments about the compression bars) is applied as the code requires.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .materials import Concrete, Rebar, concrete as _concrete, rebar as _rebar, xi_b


# ── shape ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Rect:
    b: float
    top: float
    bottom: float
    sign: int = 1  # −1: a hole

    def part(self, x):
        """(area, first moment about the top fibre) of the part above depth x"""
        z2 = min(max(x, self.top), self.bottom)
        a = self.b * (z2 - self.top)
        return self.sign * a, self.sign * self.b * (z2 * z2 - self.top ** 2) / 2

    def second_moment_top(self):
        return self.sign * self.b * (self.bottom ** 3 - self.top ** 3) / 3

    def width(self, z):
        return self.sign * self.b if self.top < z < self.bottom else 0.0

    def flipped(self, h):
        return Rect(self.b, h - self.bottom, h - self.top, self.sign)


@dataclass(frozen=True)
class Circle:
    D: float
    zc: float  # depth of the centre
    sign: int = 1

    def part(self, x):
        r = self.D / 2
        t = min(max(x - (self.zc - r), 0.0), 2 * r)  # height of the segment above x
        if t <= 0:
            return 0.0, 0.0
        a = r * r * math.acos((r - t) / r) - (r - t) * math.sqrt(max(2 * r * t - t * t, 0.0))
        c = math.sqrt(max(2 * r * t - t * t, 0.0))  # half chord
        d = 2 * c ** 3 / (3 * a) if a > 0 else r  # centroid of the segment above the centre
        return self.sign * a, self.sign * a * (self.zc - d)

    def second_moment_top(self):
        r = self.D / 2
        return self.sign * (math.pi * r ** 4 / 4 + math.pi * r * r * self.zc ** 2)

    def width(self, z):
        r = self.D / 2
        u = r * r - (z - self.zc) ** 2
        return self.sign * 2 * math.sqrt(u) if u > 0 else 0.0

    def flipped(self, h):
        return Circle(self.D, h - self.zc, self.sign)


@dataclass
class RCSection:
    concrete: Concrete
    steel: Rebar
    pieces: list
    bars: np.ndarray  # (n, 2): area mm², depth mm
    h: float
    ring: float | None = None  # circular sections: radius of the bar circle rs (h0 = r + rs for the η formula)
    kind: str = "general"  # "rect", "tee", "box", "circle", "general"

    def __post_init__(self):
        self.bars = np.asarray(self.bars, float).reshape(-1, 2)

    # geometry
    def part(self, x):
        a = q = 0.0
        for p in self.pieces:
            pa, pq = p.part(x)
            a += pa
            q += pq
        return a, q

    @property
    def area(self):
        return self.part(self.h)[0]

    @property
    def zg(self):
        """depth of the gross concrete centroid"""
        a, q = self.part(self.h)
        return q / a

    @property
    def I(self):
        """second moment of the gross concrete about its centroidal horizontal axis (mm⁴)"""
        return sum(p.second_moment_top() for p in self.pieces) - self.area * self.zg ** 2

    @property
    def i(self):
        return math.sqrt(self.I / self.area)

    def width(self, z):
        return sum(p.width(z) for p in self.pieces)

    def web(self):
        """web width b: the narrowest total width over the depth (the web of T, I and box sections, b of a rectangle)"""
        zs = np.linspace(0.02 * self.h, 0.98 * self.h, 197)
        return min(self.width(z) for z in zs)

    def flipped(self):
        """the section turned upside down (for hogging moments)"""
        bars = self.bars.copy()
        bars[:, 1] = self.h - bars[:, 1]
        return RCSection(self.concrete, self.steel, [p.flipped(self.h) for p in self.pieces], bars, self.h, self.ring, self.kind)

    def groups(self):
        """(tension bars, compression bars) for sagging: bars below / above the gross centroid"""
        lo = self.bars[:, 1] > self.zg
        return self.bars[lo], self.bars[~lo]

    def h0(self):
        t, _ = self.groups()
        if not len(t):
            raise ValueError("no bars below the centroid")
        return float(np.average(t[:, 1], weights=t[:, 0]))

    # forces for a block depth x (compression +, moment about the gross centroid, sagging +), in N and N·mm
    def stresses(self, x):
        c, s = self.concrete, self.steel
        return np.clip(c.eps_cu * s.Es * (c.beta * self.bars[:, 1] / x - 1), -s.fsd_c, s.fsd)

    def forces(self, x):
        a, q = self.part(min(x, self.h))
        zg = self.zg
        C = self.concrete.fcd * a
        Mc = C * (zg - q / a) if a > 0 else 0.0
        T = self.stresses(x) * self.bars[:, 0]
        return C - T.sum(), Mc + (T * (self.bars[:, 1] - zg)).sum()


def _bars(rows):
    """rows of (count, diameter mm, depth mm) → (area, depth) points (each row lumped at its depth)"""
    out = []
    for n, d, z in rows or ():
        out.append((n * math.pi * d * d / 4, z))
    return out


def _mats(concrete, steel):
    c = concrete if isinstance(concrete, Concrete) else _concrete(concrete)
    s = steel if isinstance(steel, Rebar) else _rebar(steel)
    return c, s


def rect(b, h, *, concrete="C30", steel="HRB400", bottom=(), top=(), bars=()):
    """rectangle b × h; bottom / top: rows of (count, diameter, depth of the bar centre from that face); bars: extra
    (area, depth from the top)"""
    c, s = _mats(concrete, steel)
    bb = [(a, h - z) for a, z in _bars(bottom)] + _bars(top) + list(bars)
    return RCSection(c, s, [Rect(b, 0, h)], bb, h, kind="rect")


def tee(b, h, bf, hf, *, concrete="C30", steel="HRB400", bottom=(), top=(), bars=(), bf_bottom=0.0, hf_bottom=0.0):
    """T section: web b, depth h, top flange bf × hf (the effective width, per JTG 3362-2018 4.3.3); with bf_bottom and
    hf_bottom an I section"""
    c, s = _mats(concrete, steel)
    p = [Rect(bf, 0, hf), Rect(b, hf, h - hf_bottom)]
    if hf_bottom:
        p.append(Rect(bf_bottom, h - hf_bottom, h))
    bb = [(a, h - z) for a, z in _bars(bottom)] + _bars(top) + list(bars)
    return RCSection(c, s, p, bb, h, kind="tee")


def box(B, h, t_top, t_bottom, webs, *, b_top=None, concrete="C40", steel="HRB400", bottom=(), top=(), bars=()):
    """single-cell or multi-cell box: bottom width B, top flange width b_top (default B, i.e. no cantilevers), slab
    thicknesses t_top and t_bottom, total web thickness webs (sum of all webs, as the code takes b for a box)"""
    c, s = _mats(concrete, steel)
    bt = B if b_top is None else b_top
    p = [Rect(bt, 0, t_top), Rect(webs, t_top, h - t_bottom), Rect(B, h - t_bottom, h)]
    bb = [(a, h - z) for a, z in _bars(bottom)] + _bars(top) + list(bars)
    return RCSection(c, s, p, bb, h, kind="box")


def circle(D, n, d, cover, *, concrete="C30", steel="HRB400", hollow: float = 0.0):
    """circular pier or pile: diameter D, n bars of diameter d evenly on a circle, cover to the bar centres;
    hollow: inner diameter of a hollow pier"""
    c, s = _mats(concrete, steel)
    r, rs = D / 2, D / 2 - cover
    th = 2 * math.pi * (np.arange(n) + 0.5) / n
    a = math.pi * d * d / 4
    bars = [(a, r - rs * math.cos(t)) for t in th]
    p = [Circle(D, r)] + ([Circle(hollow, r, -1)] if hollow else [])
    return RCSection(c, s, p, bars, D, ring=rs, kind="circle" if not hollow else "annulus")


# ── results ─────────────────────────────────────────────────────────────────
@dataclass
class Check:
    name: str
    demand: float  # γ0 Sd
    resistance: float
    unit: str
    clause: str
    details: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    failed: list = field(default_factory=list)  # conditions not met (besides demand ≤ resistance)

    @property
    def ratio(self):
        return abs(self.demand) / self.resistance if self.resistance > 0 else math.inf

    @property
    def ok(self):
        return self.ratio <= 1.0 and not self.failed

    def __str__(self):
        v = "OK" if self.ok else "NOT OK"
        extra = ("; " + "; ".join(self.failed)) if self.failed else ""
        return f"{self.name}: γ0Sd = {self.demand:.1f} {self.unit}, R = {self.resistance:.1f} {self.unit}, ratio {self.ratio:.3f} → {v}{extra}"


# ── flexure ─────────────────────────────────────────────────────────────────
def flexure(sec: RCSection, Md: float, gamma0: float = 1.0) -> Check:
    """normal-section flexural resistance of a member without axial force (5.2.2 rectangle, 5.2.3 T / I / box, 5.2.4
    x < 2as') plus the conditions x ≤ ξb h0 and the minimum tension reinforcement (9.1.12)"""
    s = sec if Md >= 0 else sec.flipped()
    c, st = s.concrete, s.steel
    h0 = s.h0()
    t, comp = s.groups()
    lo, hi = 1e-6 * s.h, 20 * s.h
    x = brentq(lambda x: s.forces(x)[0], lo, hi, xtol=1e-9 * s.h)
    xb = xi_b(c, st)
    chk = Check("flexure" + (" (hogging)" if Md < 0 else ""), gamma0 * Md, 0.0, "kN·m", "JTG 3362-2018 5.2.2 / 5.2.3",
                {"x": x, "h0": h0, "xi": x / h0, "xi_b": xb})
    if x > xb * h0:
        chk.failed.append(f"over-reinforced: x = {x:.0f} > ξb h0 = {xb * h0:.0f} mm")
    ring = s.kind in ("circle", "annulus")  # bars all round: no separate compression reinforcement
    a_c = float(np.average(comp[:, 1], weights=comp[:, 0])) if len(comp) and not ring else None
    if a_c is not None and x < 2 * a_c:
        # 5.2.4: compression bars counted, moments about them, concrete ignored. 5.2.4 applies "when the compression
        # bars are counted", so the section may also be taken as singly reinforced (with its own x ≤ ξb h0): the
        # larger governs.
        M_about = (st.fsd * t[:, 0] * (t[:, 1] - a_c)).sum()
        single = RCSection(c, st, s.pieces, t, s.h, s.ring, s.kind)
        x1 = brentq(lambda x: single.forces(x)[0], lo, hi, xtol=1e-9 * s.h)
        M_single = single.forces(x1)[1]
        if x1 > xb * h0:  # singly reinforced is over-reinforced: not a valid alternative (5.2.2-3)
            M_single = 0.0
        Mu = max(M_about, M_single)
        how = "moments about the compression bars" if M_about >= M_single else "compression bars ignored (singly reinforced)"
        chk.notes.append(f"x = {x:.0f} < 2as' = {2 * a_c:.0f} mm: {how} governs")
        chk.details.update(as_c=a_c, M_about_compression_bars=M_about / 1e6, M_singly_reinforced=M_single / 1e6)
    else:
        Mu = s.forces(x)[1]
    chk.resistance = Mu / 1e6
    if not ring:
        b = s.web()
        rho = t[:, 0].sum() / (b * h0)
        rmin = max(0.45 * c.ftd / st.fsd, 0.002)
        chk.details.update(b=b, rho=rho, rho_min=rmin)
        if rho < rmin:
            chk.failed.append(f"tension reinforcement ratio {rho:.4%} < minimum {rmin:.4%}")
    if gamma0 * abs(Md) > 0 and chk.resistance <= 0:
        chk.failed.append("no flexural resistance")
    return chk


# ── shear ───────────────────────────────────────────────────────────────────
def shear(sec: RCSection, Vd: float, stirrups, *, gamma0: float = 1.0, hogging: bool = False, alpha1: float = 1.0,
          alpha3: float | None = None, bent=(), slab: bool = False) -> Check:
    """inclined-section shear resistance of a beam (5.2.9), with the upper limit on the section (5.2.11), the threshold
    below which detailing stirrups suffice (5.2.12) and the minimum stirrup ratio (9.3.12).
    stirrups: (legs, diameter mm, spacing mm); bent: rows of (total area mm², angle in degrees) of bent-up bars in the
    section. alpha1: 1.0 for simply supported beams and positive moment regions, 0.9 next to the intermediate supports of
    continuous beams and cantilevers. alpha3: 1.1 when a flange is in compression (default for T / I / box sections in
    sagging), else 1.0. hogging: the tension face is the top (h0 and ρ from the top bars). slab: 5.2.12 threshold × 1.25."""
    s = sec.flipped() if hogging else sec
    if s.kind in ("circle", "annulus"):
        raise NotImplementedError("shear of circular sections is not implemented")
    c, st = s.concrete, s.steel
    b, h0 = s.web(), s.h0()
    t, _ = s.groups()
    legs, dia, sv = stirrups
    Asv = legs * math.pi * dia * dia / 4
    rho_sv = Asv / (sv * b)
    P = min(100 * t[:, 0].sum() / (b * h0), 2.5)
    a3 = alpha3 if alpha3 is not None else (1.1 if (s.kind in ("tee", "box") and s.width(0.01 * s.h) > b * 1.01) else 1.0)
    alpha2 = 1.0  # no prestress
    Vcs = alpha1 * alpha2 * a3 * 0.45e-3 * b * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(c.fcuk) * rho_sv * st.fsv)
    Vsb = sum(0.75e-3 * st.fsd * A * math.sin(math.radians(th)) for A, th in bent)
    chk = Check("shear" + (" (hogging side)" if hogging else ""), gamma0 * Vd, Vcs + Vsb, "kN", "JTG 3362-2018 5.2.9",
                {"b": b, "h0": h0, "P": P, "rho_sv": rho_sv, "alpha1": alpha1, "alpha3": a3, "Vcs": Vcs, "Vsb": Vsb})
    vmax = 0.51e-3 * math.sqrt(c.fcuk) * b * h0
    chk.details["V_section_max"] = vmax
    if gamma0 * abs(Vd) > vmax:
        chk.failed.append(f"section too small: γ0Vd > 0.51e-3 √fcu,k b h0 = {vmax:.0f} kN (5.2.11)")
    vmin = 0.50e-3 * alpha2 * c.ftd * b * h0 * (1.25 if slab else 1.0)
    chk.details["V_detailing"] = vmin
    if gamma0 * abs(Vd) <= vmin:
        chk.notes.append(f"γ0Vd ≤ {vmin:.0f} kN: detailing stirrups suffice (5.2.12)")
    if rho_sv < st.rho_sv_min:
        chk.failed.append(f"stirrup ratio {rho_sv:.4%} < minimum {st.rho_sv_min:.2%} (9.3.12)")
    return chk


# ── compression ─────────────────────────────────────────────────────────────
def _min_compression_steel(sec: RCSection, chk: Check):
    """9.1.12 item 1: all longitudinal bars ≥ 0.5 % of the gross section (0.6 % from C50), one side ≥ 0.2 %"""
    A = sec.area
    tot = sec.bars[:, 0].sum() / A
    need = 0.006 if sec.concrete.grade >= 50 else 0.005
    chk.details["rho_total"] = tot
    if tot < need:
        chk.failed.append(f"longitudinal reinforcement {tot:.3%} < minimum {need:.1%} (9.1.12)")
    if sec.kind not in ("circle", "annulus"):
        for side, g in zip(("tension side", "compression side"), sec.groups()):
            r = g[:, 0].sum() / A
            if r < 0.002:
                chk.failed.append(f"{side} reinforcement {r:.3%} < minimum 0.2 % (9.1.12)")



_PHI = np.array([1.0, 0.98, 0.95, 0.92, 0.87, 0.81, 0.75, 0.70, 0.65, 0.60, 0.56, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32,
                 0.29, 0.26, 0.23, 0.21, 0.19])
_L0_B = np.arange(8, 52, 2.0)
_L0_D = np.array([7, 8.5, 10.5, 12, 14, 15.5, 17, 19, 21, 22.5, 24, 26, 28, 29.5, 31, 33, 34.5, 36.5, 38, 40, 41.5, 43])
_L0_I = np.array([28, 35, 42, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111, 118, 125, 132, 139, 146, 153, 160, 167, 174.0])


def stability_factor(l0: float, *, b: float | None = None, d: float | None = None, i: float | None = None) -> float:
    """stability factor φ of an axially loaded member (table 5.3.1), linear between the tabulated values, from l0/b
    (rectangle, b = the smaller side), l0/d (circle) or l0/i (any section)"""
    if b is not None:
        x, xs = l0 / b, _L0_B
    elif d is not None:
        x, xs = l0 / d, _L0_D
    elif i is not None:
        x, xs = l0 / i, _L0_I
    else:
        raise ValueError("give b, d or i")
    if x > xs[-1]:
        raise ValueError(f"slenderness {x:.1f} beyond table 5.3.1")
    return float(np.interp(x, xs, _PHI))


def _phi_of(sec: RCSection, l0, b_min, i_min):
    if l0 is None or l0 == 0:
        return 1.0
    if b_min is not None:
        return stability_factor(l0, b=b_min)
    if i_min is not None:
        return stability_factor(l0, i=i_min)
    if sec.kind == "circle":
        return stability_factor(l0, d=sec.h)
    if sec.kind == "rect":
        return stability_factor(l0, b=min(sec.pieces[0].b, sec.h))
    if sec.kind == "annulus":
        return stability_factor(l0, i=sec.i)  # symmetric: i is the same about every axis
    raise ValueError("table 5.3.1 needs the least radius of gyration: pass i_min (mm) or b_min for this section")


def axial(sec: RCSection, Nd: float, l0: float | None = None, *, gamma0: float = 1.0, b_min: float | None = None,
          i_min: float | None = None) -> Check:
    """axially loaded member with tied stirrups (5.3.1): γ0 Nd ≤ 0.90 φ (fcd A + fsd' As'); with ρ' > 3 % the bar area
    is deducted from A. Nd compression positive (kN). φ uses the weakest axis: rectangles and circles find it themselves;
    T, I and box sections need i_min (least radius of gyration, mm) or b_min. Also the out-of-plane check of eccentric
    compression members (5.3.10). Minimum longitudinal reinforcement per 9.1.12."""
    c, st = sec.concrete, sec.steel
    A = sec.area
    As = sec.bars[:, 0].sum()
    if As / A > 0.03:
        A -= As
    phi = _phi_of(sec, l0, b_min, i_min)
    Nu = 0.9 * phi * (c.fcd * A + st.fsd_c * As) / 1e3
    chk = Check("axial compression", gamma0 * Nd, Nu, "kN", "JTG 3362-2018 5.3.1", {"phi": phi, "rho": As / sec.area})
    _min_compression_steel(sec, chk)
    return chk


def eta_factor(sec: RCSection, e0: float, l0: float, h0: float) -> float:
    """moment magnifier η = 1 + (l0/h)² ζ1 ζ2 / (1300 e0/h0), ζ1 = 0.2 + 2.7 e0/h0 ≤ 1, ζ2 = 1.15 − 0.01 l0/h ≤ 1;
    1 when l0/i ≤ 17.5 (e0, l0, h0 in mm)"""
    if not l0 or l0 / sec.i <= 17.5 or e0 <= 0:
        return 1.0
    z1 = min(0.2 + 2.7 * e0 / h0, 1.0)
    z2 = min(1.15 - 0.01 * l0 / sec.h, 1.0)
    return 1 + (l0 / sec.h) ** 2 * z1 * z2 / (1300 * e0 / h0)


def _circle_code(sec: RCSection, e: float):
    """5.3.8 formula for solid circular sections with at least 8 bars evenly on a circle: (Nu N, α)"""
    c, st = sec.concrete, sec.steel
    r, rs = sec.h / 2, sec.ring
    A = math.pi * r * r
    As = sec.bars[:, 0].sum()

    def NM(a):
        at = max(1.25 - 2 * a, 0.0)
        N = a * c.fcd * A * (1 - math.sin(2 * math.pi * a) / (2 * math.pi * a)) + (a - at) * st.fsd * As
        M = 2 / 3 * c.fcd * A * r * math.sin(math.pi * a) ** 3 / math.pi + st.fsd * As * rs * (math.sin(math.pi * a) + math.sin(math.pi * at)) / math.pi
        return N, M

    a = brentq(lambda a: NM(a)[1] - e * NM(a)[0], 1e-4, 1.0 - 1e-9, xtol=1e-12)
    return NM(a)[0], a


def _solve_x(s: RCSection, e: float):
    """block depth x with M(x) = e N(x) and N > 0 (e from the gross centroid, positive towards the top), or None"""
    xs = np.geomspace(1e-4 * s.h, 50 * s.h, 800)
    FN = np.array([s.forces(x) for x in xs])
    f = FN[:, 1] - e * FN[:, 0]
    roots = [k for k in range(len(xs) - 1) if f[k] * f[k + 1] <= 0 and FN[k + 1, 0] > 0]
    if not roots:
        return None
    k = roots[0]
    return brentq(lambda x: s.forces(x)[1] - e * s.forces(x)[0], xs[k], xs[k + 1], xtol=1e-9 * s.h)


def eccentric(sec: RCSection, Nd: float, Md: float, l0: float | None = None, *, gamma0: float = 1.0,
              method: str = "auto") -> Check:
    """eccentrically loaded compression member (5.3.4–5.3.9): finds the axial resistance Nu at the design eccentricity
    e = η e0 and checks γ0 Nd ≤ Nu. e0 = Md / Nd, at least max(20 mm, h/30) (5.3.9; 3362-2018 has no separate
    accidental eccentricity), η from eta_factor. Nd compression positive (kN), Md in kN·m (the sign picks the compressed
    face). method: "code" uses the closed-form formula of 5.3.8 for solid circles with at least 8 bars; "strain" the
    rule set above (any shape); "auto": "code" where it applies, else "strain".
    Small eccentricity with the load between the two bar groups also gets the reverse-failure check (5.3.4-4,
    5.3.5-3/4). Minimum longitudinal reinforcement per 9.1.12. Check `axial` out of plane (5.3.10) separately."""
    if Nd <= 0:
        raise ValueError("Nd must be a compressive force (> 0)")
    s = sec if Md >= 0 else sec.flipped()
    c, st = s.concrete, s.steel
    e0_raw = abs(Md) / Nd * 1e3  # mm
    e0 = max(e0_raw, 20.0, s.h / 30)
    ring = s.kind in ("circle", "annulus")
    h0 = s.h / 2 + s.ring if ring else s.h0()
    eta = eta_factor(s, e0, l0, h0)
    e = eta * e0
    chk = Check("eccentric compression", gamma0 * Nd, 0.0, "kN", "JTG 3362-2018 5.3.4–5.3.9",
                {"e0_from_forces": e0_raw, "e0": e0, "eta": eta, "e": e, "h0": h0})
    if e0 > e0_raw:
        chk.notes.append(f"e0 = {e0_raw:.1f} mm raised to the minimum {e0:.1f} mm (5.3.9)")
    _min_compression_steel(s, chk)
    n_bars = len(s.bars)
    code_ok = s.kind == "circle" and n_bars >= 8
    m = method if method != "auto" else ("code" if code_ok else "strain")
    if m == "code":
        if not code_ok:
            raise ValueError("method='code' (5.3.8) is for solid circular sections with at least 8 bars")
        Nu, a = _circle_code(s, e)
        chk.details["alpha"] = a
        chk.clause = "JTG 3362-2018 5.3.8, 5.3.9"
        chk.resistance = Nu / 1e3
        return chk
    # strain method: the block depth x with M(x) = e N(x), N > 0. With unequal bars the plastic centroid is off the gross
    # centroid; a load between the two is resisted with the far face more compressed: solve on the flipped section.
    s0 = s
    x = _solve_x(s, e)
    flipped = x is None
    if flipped:
        s = s.flipped()
        x = _solve_x(s, -e)
        if x is None:
            raise RuntimeError("no equilibrium at this eccentricity")
        chk.notes.append("load between the gross and the plastic centroid: the far face is the more compressed")
        if not ring:
            h0 = s.h0()
    Nu = s.forces(x)[0]
    xb = xi_b(c, st)
    small = flipped or x > xb * h0
    chk.details.update(x=x, xi=x / h0, xi_b=xb, case="small eccentricity" if small else "large eccentricity")
    t0, c0 = s0.groups()
    if ring or not len(c0) or not len(t0):
        chk.resistance = Nu / 1e3
        return chk
    a_c = float(np.average(c0[:, 1], weights=c0[:, 0]))  # A's: bars on the face nearer the load
    if not small:
        e_c = e - s0.zg + a_c  # from the load to the compression bars
        if x < 2 * a_c and e_c > 0:  # 5.3.6 → 5.2.4
            Nu = (st.fsd * t0[:, 0] * (t0[:, 1] - a_c)).sum() / e_c
            chk.notes.append(f"x < 2as': moments about the compression bars (e' = {e_c:.0f} mm, 5.3.6)")
    else:
        # reverse failure (5.3.4-4, 5.3.5-3/4): the load between A's and As, failure starting at the far face. Moments
        # about A's of the whole concrete section at fcd and of As at fsd'; e' = distance from the load (e0 without η)
        # to A's. e0 is taken here without the 5.3.9 minimum, the more unfavourable reading.
        d_t = float(np.average(t0[:, 1], weights=t0[:, 0]))
        z_load = s0.zg - e0_raw
        if a_c < z_load < d_t:
            e_r = z_load - a_c
            M_r = c.fcd * s0.area * (s0.zg - a_c) + st.fsd_c * (t0[:, 0] * (t0[:, 1] - a_c)).sum()
            N_r = M_r / e_r
            chk.details["N_reverse"] = N_r / 1e3
            if N_r < Nu:
                Nu = N_r
                chk.notes.append(f"reverse failure governs (e' = {e_r:.0f} mm, 5.3.4-4)")
    chk.resistance = Nu / 1e3
    return chk
