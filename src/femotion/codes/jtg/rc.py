"""Ultimate-limit-state resistance of reinforced and prestressed concrete sections, JTG 3362-2018.

Units: section dimensions and steel positions in mm, steel areas in mm², stresses in MPa; design forces in kN and kN·m
(as they come out of the frame solver). Every check takes the design effect Sd from the combination and γ0, and
verifies γ0 Sd ≤ R. Clauses, formulas and tables: docs/codes/JTG_3362-2018.md.

A section is a stack of pieces along its depth (z measured down from the top fibre): rectangles (solid or holes) and
circles, so rectangles, T, I and box girders, solid and hollow piers are all covered. Reinforcing bars are (area, depth)
points; bonded internal tendons are (area, depth, σp0, duct diameter) points, σp0 being the tendon stress when the
concrete stress at the tendon is zero (6.1.6, femotion.codes.jtg.prestress). Bending is about the horizontal axis;
sagging puts the top in compression. A hogging moment is checked on the section turned upside down.

    from femotion.codes import jtg
    sec = jtg.tee(b=200, h=1300, bf=1600, hf=150, concrete="C40", steel="HRB400",
                  bottom=[(6, 32, 90)], top=[(4, 16, 40)])      # rows of (count, diameter, depth to bar centre)
    r = jtg.flexure(sec, Md=2950.0, gamma0=1.0)                # kN·m
    v = jtg.shear(sec, Vd=780.0, stirrups=(2, 10, 150))         # legs, diameter, spacing
    c = jtg.eccentric(jtg.circle(D=1500, n=28, d=28, cover=60, concrete="C35"), Nd=9800, Md=3100, l0=18000)
    pc = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)],
                 tendons=[(2780, 1470, 1150, 90)], tendon="strand1860")   # Ap, depth, σp0, duct diameter
    print(r, v, c, jtg.flexure(pc, 4800.0, pc_class="A"))

Behind the code's simplified formulas sits one rule set, used for every shape: plane sections; concrete tension
ignored; a rectangular block of fcd over x = β·xc; steel stresses from 5.1.5 within the limits of 5.1.3; the edge rows
(As, Ap on the tension face, A's, A'p on the compressed face) at the design values the code's formulas use. For
rectangular and T sections this gives exactly the formulas of 5.2.2–5.2.4 and 5.3.4–5.3.6.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .materials import Concrete, Rebar, Tendon, concrete as _concrete, rebar as _rebar, tendon as _tendon, xi_b


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

    def part2(self, x):
        """second moment about the top fibre of the part above depth x"""
        z2 = min(max(x, self.top), self.bottom)
        return self.sign * self.b * (z2 ** 3 - self.top ** 3) / 3

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

    def part2(self, x):
        """second moment about the top fibre of the part above depth x (Gauss–Legendre in the angle, exact to 1e-12)"""
        r = self.D / 2
        t = min(max(x - (self.zc - r), 0.0), 2 * r)
        if t <= 0:
            return 0.0
        a = math.acos((r - t) / r)  # half angle of the segment, from the top of the circle
        g, w = np.polynomial.legendre.leggauss(48)
        th = a * (g + 1) / 2  # z = zc − r cos θ, width 2 r sin θ, dz = r sin θ dθ
        z = self.zc - r * np.cos(th)
        return self.sign * float((a / 2) * (w * z * z * 2 * r * np.sin(th) * r * np.sin(th)).sum())

    def width(self, z):
        r = self.D / 2
        u = r * r - (z - self.zc) ** 2
        return self.sign * 2 * math.sqrt(u) if u > 0 else 0.0

    def flipped(self, h):
        return Circle(self.D, h - self.zc, self.sign)


def _edge(d, zg, tol):
    """(bottom, top) masks: points within tol of the outermost point on each side of the centroid"""
    below, above = d > zg, d <= zg
    bot = below & (d >= d[below].max() - tol) if below.any() else below
    top = above & (d <= d[above].min() + tol) if above.any() else above
    return bot, top


@dataclass
class RCSection:
    concrete: Concrete
    steel: Rebar
    pieces: list
    bars: np.ndarray  # (n, 2): area mm², depth mm
    h: float
    ring: float | None = None  # circular sections: radius of the bar circle rs (h0 = r + rs for the η formula)
    kind: str = "general"  # "rect", "tee", "box", "circle", "annulus", "composite", "general"
    tendons: np.ndarray | None = None  # (n, 4): area mm², depth mm, σp0 MPa, duct area mm² (bonded internal tendons)
    tendon: Tendon | None = None
    pretensioned: bool = False
    sigma_l6: float | tuple = 0.0  # σl6 at the bars below / above the centroid (6.1.7, for Np0 and Np)
    bar_d: np.ndarray | None = None  # (n,): bar diameter of each bar point (mm), NaN when unknown (crack width, 6.4.3)
    sigma_pe: np.ndarray | float | None = None  # effective prestress σpe of each tendon (MPa, 6.1.6-3, -6), for 6.3, 7.1
    rules_concrete: Concrete | None = None  # concrete for the minimum classes of 3.1.2 when it differs (composite members)

    def __post_init__(self):
        self.bars = np.asarray(self.bars, float).reshape(-1, 2)
        self.tendons = np.zeros((0, 4)) if self.tendons is None else np.asarray(self.tendons, float).reshape(-1, 4)
        if not isinstance(self.sigma_l6, tuple):
            self.sigma_l6 = (float(self.sigma_l6), float(self.sigma_l6))
        if len(self.tendons) and self.tendon is None:
            raise ValueError("tendons need a prestressing steel (tendon=…)")
        self.bar_d = (np.full(len(self.bars), np.nan) if self.bar_d is None
                      else np.asarray(self.bar_d, float).reshape(-1))
        if len(self.bar_d) != len(self.bars):
            raise ValueError("bar_d: one diameter per bar point")
        if self.sigma_pe is not None:
            self.sigma_pe = np.broadcast_to(np.asarray(self.sigma_pe, float), (len(self.tendons),)).copy()

    @property
    def prestressed(self) -> bool:
        return len(self.tendons) > 0

    def material_rules(self) -> list[str]:
        """3.1.2: reinforced concrete at least C25 (jtg.concrete enforces it), at least C30 with bars of characteristic
        strength ≥ 400 MPa; prestressed concrete at least C40"""
        out = []
        c = self.rules_concrete or self.concrete
        if self.steel.fsk >= 400 and c.grade < 30 and len(self.bars):
            out.append(f"C{c.grade} with {self.steel.name}: at least C30 for bars of 400 MPa and above (3.1.2)")
        if self.prestressed and c.grade < 40:
            out.append(f"prestressed concrete in C{c.grade}: at least C40 (3.1.2)")
        return out

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

    def replace(self, **kw) -> "RCSection":
        return dataclasses.replace(self, **kw)

    def flipped(self):
        """the section turned upside down (for hogging moments)"""
        bars = self.bars.copy()
        bars[:, 1] = self.h - bars[:, 1]
        td = self.tendons.copy()
        td[:, 1] = self.h - td[:, 1]
        return self.replace(pieces=[p.flipped(self.h) for p in self.pieces], bars=bars, tendons=td,
                            sigma_l6=self.sigma_l6[::-1])

    def _steel_points(self, kind: str):
        """(area added to the concrete, depth) of the steel in the net or transformed section"""
        c = self.concrete
        pts = [((self.steel.Es / c.Ec - 1) * a, d) for a, d in self.bars]
        for a, d, _, duct in self.tendons:
            pts.append((-(a if self.pretensioned else duct), d))
            if kind == "transformed":
                pts.append((self.tendon.Ep / c.Ec * a, d))
        return pts

    def props(self, kind: str = "transformed"):
        """(A, depth of the centroid, I) of the net section An (concrete less the ducts, bars as αEs·As) or the
        transformed section A0 = An + αEp·Ap (6.1.5, 6.1.6). Pretensioned tendons occupy their own area."""
        A, Q = self.part(self.h)
        Itop = sum(p.second_moment_top() for p in self.pieces)
        for a, d in self._steel_points(kind):
            A += a
            Q += a * d
            Itop += a * d * d
        z = Q / A
        return A, z, Itop - A * z * z

    def first_moment(self, z: float, kind: str = "transformed") -> float:
        """first moment (mm³) about the centroid of the net or transformed section of its part above depth z: S0, Sn of
        6.3.3 at a fibre z; at the centroid S0 of 6.5.2-8 (the parts above and below the centroid have equal moments)"""
        _, zc, _ = self.props(kind)
        a, q = self.part(z)
        S = a * zc - q
        for at, d in self._steel_points(kind):
            if d < z:
                S += at * (zc - d)
        return S

    def part2(self, x):
        """second moment about the top fibre of the concrete above depth x"""
        return sum(p.part2(x) for p in self.pieces)

    # prestress (6.1.6, 6.1.7)
    def _sigma_pe(self):
        if self.sigma_pe is None:
            raise ValueError("this check needs the effective prestress of each tendon: pass sigma_pe=… (σcon − σl, "
                             "6.1.6-3 / -6) to the section")
        return self.sigma_pe

    def _l6(self):
        """σl6 at every bar: the bars on each side of the gross centroid take that side's σl6 when there are tendons on
        that side, else 0 (6.1.7: σl6 at As with Ap; σ'l6 = 0 when A'p = 0). The same whether or not the section is
        turned over."""
        zg = self.zg
        below = self.bars[:, 1] > zg
        l6_b = self.sigma_l6[0] if (self.tendons[:, 1] > zg).any() else 0.0
        l6_t = self.sigma_l6[1] if (self.tendons[:, 1] <= zg).any() else 0.0
        return np.where(below, l6_b, l6_t)

    def prestress_resultant(self, stress: str = "p0"):
        """(force N, depth of its line of action mm) of the prestressing and reinforcing steel: Np0 = Σ σp0 Ap − Σ σl6 As
        (6.1.7-1, -2) with stress="p0"; Np = Σ σpe Ap − Σ σl6 As (6.1.7-3, -4) with stress="pe" """
        td = self.tendons
        sp = td[:, 2] if stress == "p0" else self._sigma_pe()
        F = np.r_[sp * td[:, 0], -self._l6() * self.bars[:, 0]]
        d = np.r_[td[:, 1], self.bars[:, 1]]
        N = F.sum()
        return N, float((F * d).sum() / N)

    def prestress_stress(self, z, M_p2: float = 0.0):
        """normal concrete stress (MPa, compression +) at depth z from the prestress after the losses carried by the
        section (6.1.6): pretensioned members Np0 on the transformed section (6.1.6-1); post-tensioned members Np on the
        net section with the secondary moment Mp2 (kN·m, sagging +) (6.1.6-4). 0 for reinforced concrete."""
        z = np.asarray(z, float)
        if not self.prestressed:
            return 0.0 * z
        N, zN = self.prestress_resultant("p0" if self.pretensioned else "pe")
        A, zc, I = self.props("transformed" if self.pretensioned else "net")
        y = z - zc
        return N / A + N * (zN - zc) * y / I - M_p2 * 1e6 * y / I

    # edge rows (program convention, docs/codes/JTG_3362-2018.md §10)
    def edge_masks(self):
        """(As, A's) as boolean masks over the bars: the rows next to the bottom (tension) face and the top (compressed)
        face, i.e. bars within min(0.1 h, 150 mm) of the outermost bar on that side of the gross centroid. Bars between
        them (side bars, web bars) belong to neither. Program convention for "one side" (9.1.12) and As, A's (5.2, 5.3)."""
        return _edge(self.bars[:, 1], self.zg, min(0.1 * self.h, 150.0))

    def tendon_masks(self):
        """(Ap, A'p) as boolean masks over the tendons, by the same rule as edge_masks"""
        return _edge(self.tendons[:, 1], self.zg, min(0.1 * self.h, 150.0))

    def groups(self):
        """(As, A's): bars of the tension-face rows and of the compressed-face rows for sagging (edge_masks)"""
        bot, top = self.edge_masks()
        return self.bars[bot], self.bars[top]

    def h0(self):
        """depth of the resultant of the tension steel As (at fsd) and Ap (at fpd): h0 = h − a (5.2.2)"""
        bot, _ = self.edge_masks()
        pb, _ = self.tendon_masks()
        F = np.r_[self.steel.fsd * self.bars[bot, 0], self.tendon.fpd * self.tendons[pb, 0] if self.prestressed else []]
        d = np.r_[self.bars[bot, 1], self.tendons[pb, 1]]
        if not len(F):
            raise ValueError("no steel below the centroid")
        return float(np.average(d, weights=F))

    def comp_ref(self):
        """(a', A'p compressed): depth of the compression-steel resultant used in x ≥ 2a' (5.2.2-4 / -5). With A'p in
        compression ((f'pd − σ'p0) > 0) it is the resultant of A's and A'p, else that of A's; None when there is none"""
        _, top = self.edge_masks()
        _, pt = self.tendon_masks()
        Fs = self.steel.fsd_c * self.bars[top, 0]
        Fp = (self.tendon.fpd_c - self.tendons[pt, 2]) * self.tendons[pt, 0] if self.prestressed else np.zeros(0)
        if len(Fp) and Fp.sum() > 0:
            F = np.r_[Fs, Fp]
            return float(np.average(np.r_[self.bars[top, 1], self.tendons[pt, 1]], weights=F)), True
        if len(Fs):
            return float(np.average(self.bars[top, 1], weights=Fs)), False
        return None, False

    def xi_b(self, eccentric: bool = False) -> float:
        """ξb of the tension steel: table 5.2.1 for bars and, for flexure, tendons (the smallest when both are present,
        note 1); for eccentric compression of prestressed members the tendons follow 5.3.3-1 / -2"""
        bot, _ = self.edge_masks()
        pb, _ = self.tendon_masks()
        v = []
        if bot.any() or not pb.any():
            v.append(xi_b(self.concrete, self.steel))
        if pb.any():
            if eccentric:
                c, t = self.concrete, self.tendon
                sp0 = float(np.average(self.tendons[pb, 2], weights=self.tendons[pb, 0]))
                extra = 0.0 if t.kind == "bar" else 0.002 / c.eps_cu
                v.append(c.beta / (1 + extra + (t.fpd - sp0) / (t.Ep * c.eps_cu)))
            else:
                v.append(xi_b(self.concrete, self.tendon))
        return min(v)

    # stresses (tension +) and forces for a block depth x: N compression +, M about the gross centroid, sagging +
    def stresses(self, x, xb: float | None = None):
        """(bar stresses, tendon stresses) for a block depth x. Every bar by 5.1.5-1 within −f'sd … fsd (5.1.3-1), every
        tendon by 5.1.5-2 within −(f'pd − σp0) … fpd (5.1.3-2). For sections with steel on two faces (not circles) the
        edge rows then take the values of the code's formulas: As at fsd and Ap at fpd while ξ = x/h0 ≤ ξb (5.2.2-2,
        5.3.4); A's at −f'sd and A'p at −(f'pd − σ'p0) once x ≥ 2a' (5.2.2-1, 5.3.4-1). xb: ξb (default for flexure)."""
        c, s = self.concrete, self.steel
        sig = np.clip(c.eps_cu * s.Es * (c.beta * self.bars[:, 1] / x - 1), -s.fsd_c, s.fsd)
        if self.prestressed:
            t, sp0 = self.tendon, self.tendons[:, 2]
            sigp = np.clip(c.eps_cu * t.Ep * (c.beta * self.tendons[:, 1] / x - 1) + sp0, -(t.fpd_c - sp0), t.fpd)
        else:
            sigp = np.zeros(0)
        if self.kind in ("circle", "annulus"):
            return sig, sigp
        bot, top = self.edge_masks()
        pb, pt = self.tendon_masks()
        aref, _ = self.comp_ref()
        if aref is None or x >= 2 * aref:
            sig[top] = -s.fsd_c
            if self.prestressed:
                sigp[pt] = -(self.tendon.fpd_c - self.tendons[pt, 2])
        if bot.any() or pb.any():
            try:
                if x <= (self.xi_b() if xb is None else xb) * self.h0():
                    sig[bot] = s.fsd
                    if self.prestressed:
                        sigp[pb] = self.tendon.fpd
            except ValueError:  # no ξb in table 5.2.1 (C75, C80 with bars): the checks themselves raise
                pass
        return sig, sigp

    def forces(self, x, xb: float | None = None):
        a, q = self.part(min(x, self.h))
        zg = self.zg
        C = self.concrete.fcd * a
        Mc = C * (zg - q / a) if a > 0 else 0.0
        sig, sigp = self.stresses(x, xb)
        T = sig * self.bars[:, 0]
        Tp = sigp * self.tendons[:, 0]
        return (C - T.sum() - Tp.sum(),
                Mc + (T * (self.bars[:, 1] - zg)).sum() + (Tp * (self.tendons[:, 1] - zg)).sum())

    def design_tension_moment(self, aref: float, compressed_p: bool):
        """5.2.4: moments about the compression-steel resultant a' of the tension steel at design strength, concrete
        ignored: fpd Ap (dp − a') + fsd As (ds − a'), and with A'p in tension − (f'pd − σ'p0) A'p (a'p − a's) (N·mm)"""
        bot, _ = self.edge_masks()
        pb, pt = self.tendon_masks()
        M = (self.steel.fsd * self.bars[bot, 0] * (self.bars[bot, 1] - aref)).sum()
        if self.prestressed:
            t, td = self.tendon, self.tendons
            M += (t.fpd * td[pb, 0] * (td[pb, 1] - aref)).sum()
            if not compressed_p:
                M -= ((t.fpd_c - td[pt, 2]) * td[pt, 0] * (td[pt, 1] - aref)).sum()
        return M


def _bars(rows):
    """rows of (count, diameter mm, depth mm) → (area, depth, diameter) points (each row lumped at its depth)"""
    out = []
    for n, d, z in rows or ():
        out.append((n * math.pi * d * d / 4, z, d))
    return out


def _all_bars(h, bottom, top, bars):
    """bottom rows (depth from the bottom face), top rows, extra points (area, depth from the top[, diameter])"""
    pts = [(a, h - z, d) for a, z, d in _bars(bottom)] + _bars(top)
    pts += [(b[0], b[1], b[2] if len(b) > 2 else np.nan) for b in bars or ()]
    return np.array(pts, float).reshape(-1, 3)


def _mats(concrete, steel):
    c = concrete if isinstance(concrete, Concrete) else _concrete(concrete)
    s = steel if isinstance(steel, Rebar) else _rebar(steel)
    return c, s


def _tendons(rows, tendon, pretensioned):
    """rows of (Ap mm², depth from the top mm, σp0 MPa[, duct diameter mm]) → (n, 4) array with the duct area"""
    out = []
    for r in rows or ():
        a, d, sp0 = r[:3]
        duct = r[3] if len(r) > 3 else 0.0
        if not pretensioned and duct <= 0:
            raise ValueError("post-tensioned tendons need a duct diameter (4th value), or pass pretensioned=True")
        out.append((a, d, sp0, math.pi * duct * duct / 4))
    if not out:
        return None, None
    t = tendon if isinstance(tendon, Tendon) else _tendon(tendon)
    return np.array(out), t


def _make(c, s, pieces, h, bars, kind, tendons, tendon, pretensioned, sigma_l6, sigma_pe, **kw):
    td, t = _tendons(tendons, tendon, pretensioned)
    return RCSection(c, s, pieces, bars[:, :2], h, kind=kind, tendons=td, tendon=t, pretensioned=pretensioned,
                     sigma_l6=sigma_l6, bar_d=bars[:, 2], sigma_pe=sigma_pe, **kw)


def rect(b, h, *, concrete="C30", steel="HRB400", bottom=(), top=(), bars=(), tendons=(), tendon="strand1860",
         pretensioned=False, sigma_l6=0.0, sigma_pe=None):
    """rectangle b × h; bottom / top: rows of (count, diameter, depth of the bar centre from that face); bars: extra
    (area, depth from the top[, bar diameter]). tendons: rows of (Ap mm², depth from the top mm, σp0 MPa, duct diameter
    mm) of bonded internal tendons, tendon: their steel (jtg.tendon or "strand1860", "wire1570", "bar930"); pretensioned:
    no ducts. sigma_l6: creep and shrinkage loss σl6 at the bars, one value or (bars below, bars above the centroid),
    for Np0 and Np (6.1.7; σ'l6 counts only with tendons in that zone; 0 is on the safe side for the cracking moment).
    sigma_pe: effective prestress σpe of each tendon (one value or one per row), needed by the serviceability and
    stress checks (femotion.codes.jtg.sls). Pretensioned members near their ends: reduce fpd inside the anchorage zone
    (5.1.6, prestress.fpd_in_anchorage)."""
    c, s = _mats(concrete, steel)
    bb = _all_bars(h, bottom, top, bars)
    return _make(c, s, [Rect(b, 0, h)], h, bb, "rect", tendons, tendon, pretensioned, sigma_l6, sigma_pe)


def tee(b, h, bf, hf, *, concrete="C30", steel="HRB400", bottom=(), top=(), bars=(), bf_bottom=0.0, hf_bottom=0.0,
        tendons=(), tendon="strand1860", pretensioned=False, sigma_l6=0.0, sigma_pe=None):
    """T section: web b, depth h, top flange bf × hf (the effective width, per JTG 3362-2018 4.3.3); with bf_bottom and
    hf_bottom an I section. Bars and tendons as for rect."""
    c, s = _mats(concrete, steel)
    p = [Rect(bf, 0, hf), Rect(b, hf, h - hf_bottom)]
    if hf_bottom:
        p.append(Rect(bf_bottom, h - hf_bottom, h))
    bb = _all_bars(h, bottom, top, bars)
    return _make(c, s, p, h, bb, "tee", tendons, tendon, pretensioned, sigma_l6, sigma_pe)


def box(B, h, t_top, t_bottom, webs, *, b_top=None, concrete="C40", steel="HRB400", bottom=(), top=(), bars=(),
        tendons=(), tendon="strand1860", pretensioned=False, sigma_l6=0.0, sigma_pe=None):
    """single-cell or multi-cell box: bottom width B, top flange width b_top (default B, i.e. no cantilevers), slab
    thicknesses t_top and t_bottom, total web thickness webs (sum of all webs, as the code takes b for a box).
    Bars and tendons as for rect."""
    c, s = _mats(concrete, steel)
    bt = B if b_top is None else b_top
    p = [Rect(bt, 0, t_top), Rect(webs, t_top, h - t_bottom), Rect(B, h - t_bottom, h)]
    bb = _all_bars(h, bottom, top, bars)
    return _make(c, s, p, h, bb, "box", tendons, tendon, pretensioned, sigma_l6, sigma_pe)


def circle(D, n, d, cover, *, concrete="C30", steel="HRB400", hollow: float = 0.0):
    """circular pier or pile: diameter D, n bars of diameter d evenly on a circle, cover to the bar centres;
    hollow: inner diameter of a hollow pier"""
    c, s = _mats(concrete, steel)
    r, rs = D / 2, D / 2 - cover
    th = 2 * math.pi * (np.arange(n) + 0.5) / n
    a = math.pi * d * d / 4
    bars = [(a, r - rs * math.cos(t)) for t in th]
    p = [Circle(D, r)] + ([Circle(hollow, r, -1)] if hollow else [])
    return RCSection(c, s, p, bars, D, ring=rs, kind="circle" if not hollow else "annulus", bar_d=np.full(n, float(d)))


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
    # serviceability and stress checks: a signed quantity against a limit (demand ≤ limit), e.g. σst ≤ 0.85 σpc
    limit_check: bool = False
    symbol: str = "γ0Sd"

    @property
    def ratio(self):
        if self.limit_check:
            if self.resistance > 0:
                return self.demand / self.resistance
            return 0.0 if self.demand <= self.resistance else math.inf
        return abs(self.demand) / self.resistance if self.resistance > 0 else math.inf

    @property
    def ok(self):
        if self.limit_check:
            return self.demand <= self.resistance + 1e-9 * max(abs(self.resistance), 1.0) and not self.failed
        return self.ratio <= 1.0 and not self.failed

    def __str__(self):
        v = "OK" if self.ok else "NOT OK"
        extra = ("; " + "; ".join(self.failed)) if self.failed else ""
        if self.limit_check:
            p = 3 if self.unit == "mm" else 2
            r = (f"ratio {self.ratio:.3f}" if self.resistance > 0
                 else f"margin {self.resistance - self.demand:.{p}f} {self.unit}")
            return (f"{self.name}: {self.symbol} = {self.demand:.{p}f} {self.unit}, limit {self.resistance:.{p}f} "
                    f"{self.unit}, {r} → {v}{extra}")
        return f"{self.name}: γ0Sd = {self.demand:.1f} {self.unit}, R = {self.resistance:.1f} {self.unit}, ratio {self.ratio:.3f} → {v}{extra}"


# ── cracking moment ─────────────────────────────────────────────────────────
def cracking_moment(sec: RCSection, hogging: bool = False) -> tuple[float, dict]:
    """cracking moment Mcr = (σpc + γ ftk) W0, γ = 2 S0 / W0 (6.5.2-7, -8), for the minimum-reinforcement rule 9.1.13
    and the stiffness of class B members. σpc: precompression at the tension edge from Np0 = Σ σp0 Ap − σl6 As (6.1.7-1)
    by 6.1.6-1, on the transformed section for pretensioned members and on the net section for post-tensioned ones;
    W0, S0 on the transformed section. Returns (Mcr kN·m, details)."""
    s = sec.flipped() if hogging else sec
    c = s.concrete
    A0, z0, I0 = s.props("transformed")
    A, zc, I = (A0, z0, I0) if s.pretensioned else s.props("net")
    N, zN = s.prestress_resultant("p0")
    sig_pc = N / A + N * (zN - zc) / I * (s.h - zc)  # compression + at the bottom (tension) edge
    W0 = I0 / (s.h - z0)
    S0 = s.first_moment(z0)
    gamma = 2 * S0 / W0
    Mcr = (sig_pc + gamma * c.ftk) * W0
    return Mcr / 1e6, {"sigma_pc": sig_pc, "W0": W0, "S0": S0, "gamma": gamma, "Np0": N / 1e3}


# ── flexure ─────────────────────────────────────────────────────────────────
def flexure(sec: RCSection, Md: float, gamma0: float = 1.0, *, pc_class: str | None = None) -> Check:
    """normal-section flexural resistance of a member without axial force (5.2.2 rectangle, 5.2.3 T / I / box, 5.2.4
    x < 2a') plus the condition x ≤ ξb h0 and the minimum reinforcement: 9.1.12 for reinforced concrete; 9.1.13 for
    prestressed concrete (Mud / Mcr ≥ 1.0, and As ≥ 0.003 b h0 for partially prestressed members: pc_class "A" or "B";
    "full" for fully prestressed)."""
    s = sec if Md >= 0 else sec.flipped()
    c, st = s.concrete, s.steel
    h0 = s.h0()
    t, comp = s.groups()
    lo, hi = 1e-6 * s.h, 20 * s.h
    x = brentq(lambda x: s.forces(x)[0], lo, hi, xtol=1e-9 * s.h)
    xb = s.xi_b()
    chk = Check("flexure" + (" (hogging)" if Md < 0 else ""), gamma0 * Md, 0.0, "kN·m", "JTG 3362-2018 5.2.2 / 5.2.3",
                {"x": x, "h0": h0, "xi": x / h0, "xi_b": xb})
    if x > xb * h0:
        chk.failed.append(f"over-reinforced: x = {x:.0f} > ξb h0 = {xb * h0:.0f} mm")
    ring = s.kind in ("circle", "annulus")  # bars all round: no separate compression reinforcement
    aref, compressed_p = (None, False) if ring else s.comp_ref()
    if aref is not None and x < 2 * aref:
        # 5.2.4: compression steel counted, moments about it, concrete ignored. 5.2.4 applies "when the compression
        # bars are counted"; without compression tendons the section may also be taken as singly reinforced (with its
        # own x ≤ ξb h0): the larger governs.
        M_about = s.design_tension_moment(aref, compressed_p)
        M_single = 0.0
        _, pt = s.tendon_masks()
        if not pt.any():
            bot, top = s.edge_masks()
            single = s.replace(bars=s.bars[~top], bar_d=s.bar_d[~top])
            x1 = brentq(lambda x: single.forces(x)[0], lo, hi, xtol=1e-9 * s.h)
            if x1 <= xb * single.h0():  # an over-reinforced single section is not a valid alternative (5.2.2-3)
                M_single = single.forces(x1)[1]
        Mu = max(M_about, M_single)
        how = "moments about the compression steel" if M_about >= M_single else "compression bars ignored (singly reinforced)"
        chk.notes.append(f"x = {x:.0f} < 2a' = {2 * aref:.0f} mm: {how} governs (5.2.4)")
        chk.details.update(a_ref=aref, M_about_compression_steel=M_about / 1e6, M_singly_reinforced=M_single / 1e6)
    else:
        Mu = s.forces(x)[1]
    chk.resistance = Mu / 1e6
    if not ring:
        b = s.web()
        chk.details["b"] = b
        if s.prestressed:
            Mcr, d = cracking_moment(s)
            chk.details.update(Mcr=Mcr, cracking=d)
            if chk.resistance < Mcr:
                chk.failed.append(f"Mud / Mcr = {chk.resistance / Mcr:.3f} < 1.0 (9.1.13)")
            if pc_class is None:
                chk.notes.append("pc_class not given: As ≥ 0.003 b h0 of partially prestressed members not checked (9.1.13)")
            elif pc_class.upper() in ("A", "B"):
                As_min = 0.003 * b * h0
                if t[:, 0].sum() < As_min:
                    chk.failed.append(f"As = {t[:, 0].sum():.0f} mm² < 0.003 b h0 = {As_min:.0f} mm² (9.1.13)")
        else:
            rho = t[:, 0].sum() / (b * h0)
            rmin = max(0.45 * c.ftd / st.fsd, 0.002)
            chk.details.update(rho=rho, rho_min=rmin)
            if rho < rmin:
                chk.failed.append(f"tension reinforcement ratio {rho:.4%} < minimum {rmin:.4%} (9.1.12)")
    if gamma0 * abs(Md) > 0 and chk.resistance <= 0:
        chk.failed.append("no flexural resistance")
    chk.failed += s.material_rules()
    return chk


# ── shear ───────────────────────────────────────────────────────────────────
def shear(sec: RCSection, Vd: float, stirrups=None, *, gamma0: float = 1.0, hogging: bool = False, alpha1: float = 1.0,
          alpha3: float | None = None, bent=(), slab: bool = False, stirrup_steel: str | Rebar | None = None,
          pc_class: str | None = None, bent_tendons=(), vertical=None, alpha2: float | None = None) -> Check:
    """inclined-section shear resistance of a beam (5.2.9), with the upper limit on the section (5.2.11), the threshold
    below which detailing stirrups suffice (5.2.12) and the stirrup rules of 9.3.12 (ratio, diameter ≥ 8 mm, spacing
    ≤ h/2 and ≤ 400 mm). Take Vd at the sections listed in 5.2.8 (e.g. h/2 from the support centre).
    stirrups: (legs, diameter mm, spacing mm); bent: rows of (total area mm², angle in degrees) of bent-up bars in the
    section. alpha1: 1.0 for simply supported beams and near end supports of continuous beams, 0.9 near intermediate
    supports of continuous beams and cantilevers. alpha3: 1.1 when a flange is in compression (default for T / I / box
    sections whose compressed face has a flange: sagging for T, I and box, hogging for I and box), else 1.0.
    hogging: the tension face is the top (h0 and ρ from the top steel). stirrup_steel: grade of the stirrups when it
    differs from the longitudinal bars (fsv and the minimum stirrup ratio follow it).
    slab=True with stirrups=None: a slab without stirrups, checked against the 5.2.12 threshold × 1.25 only.
    Prestressed sections: pc_class ("full", "A" or "B") is required; α2 = 1.25, but 1.0 for class B (cracking allowed)
    or when the moment of the prestressing force (Np0 from the tendons' σp0) acts in the same direction as the external
    moment. bent_tendons: rows of (area mm², angle °) of bent-up tendons (Vpb, 5.2.9-4); vertical: (area per section mm²,
    spacing mm, steel) of vertical prestressing bars (ρpv fpv in 5.2.9-2). alpha2: override of α2 (e.g. 1.0 for
    prestressed composite members, 8.1.5)."""
    s = sec.flipped() if hogging else sec
    if s.kind in ("circle", "annulus"):
        raise NotImplementedError("shear of circular sections is not implemented")
    c, st = s.concrete, s.steel
    b, h0 = s.web(), s.h0()
    t, _ = s.groups()
    alpha2_given, alpha2 = alpha2, 1.0
    notes = []
    if s.prestressed and alpha2_given is None:
        if pc_class is None or pc_class.upper() not in ("FULL", "A", "B"):
            raise ValueError("prestressed sections need pc_class='full', 'A' or 'B' (α2, 5.2.9)")
        Mp = -(s.tendons[:, 2] * s.tendons[:, 0] * (s.tendons[:, 1] - s.zg)).sum()  # on the concrete, sagging +
        if pc_class.upper() == "B":
            notes.append("class B (cracking allowed): α2 = 1.0")
        elif Mp > 0:
            notes.append("prestressing moment in the direction of the external moment: α2 = 1.0")
        else:
            alpha2 = 1.25
    if alpha2_given is not None:
        alpha2 = alpha2_given
        notes.append(f"α2 = {alpha2:g} given")
    vmin = 0.50e-3 * alpha2 * c.ftd * b * h0
    if stirrups is None:
        if not slab:
            raise ValueError("beams need stirrups (9.3.12); stirrups=None only for slabs (slab=True)")
        vslab = 1.25 * vmin
        chk = Check("shear (slab without stirrups)", gamma0 * Vd, vslab, "kN", "JTG 3362-2018 5.2.12",
                    {"b": b, "h0": h0, "alpha2": alpha2, "V_detailing": vslab}, notes)
        chk.failed += s.material_rules()
        return chk
    sst = st if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
    legs, dia, sv = stirrups
    Asv = legs * math.pi * dia * dia / 4
    rho_sv = Asv / (sv * b)
    pb, _ = s.tendon_masks()
    Ap = s.tendons[pb, 0].sum()
    P = min(100 * (t[:, 0].sum() + Ap) / (b * h0), 2.5)
    rho_pv = fpv = 0.0
    if vertical is not None:
        Apv, sp, vt = vertical
        vt = vt if isinstance(vt, Tendon) else _tendon(vt)
        rho_pv, fpv = Apv / (sp * b), vt.fpd
    a3 = alpha3 if alpha3 is not None else (1.1 if s.width(0.01 * s.h) > b * 1.01 else 1.0)
    Vcs = alpha1 * alpha2 * a3 * 0.45e-3 * b * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(c.fcuk) * (rho_sv * sst.fsv + 0.6 * rho_pv * fpv))
    Vsb = sum(0.75e-3 * st.fsd * A * math.sin(math.radians(th)) for A, th in bent)
    Vpb = 0.0
    if bent_tendons:
        if not s.prestressed:
            raise ValueError("bent_tendons need a prestressed section (its tendon steel gives fpd)")
        Vpb = sum(0.75e-3 * s.tendon.fpd * A * math.sin(math.radians(th)) for A, th in bent_tendons)
    chk = Check("shear" + (" (hogging side)" if hogging else ""), gamma0 * Vd, Vcs + Vsb + Vpb, "kN", "JTG 3362-2018 5.2.9",
                {"b": b, "h0": h0, "P": P, "rho_sv": rho_sv, "alpha1": alpha1, "alpha2": alpha2, "alpha3": a3,
                 "Vcs": Vcs, "Vsb": Vsb, "Vpb": Vpb, "fsv": sst.fsv, "rho_pv": rho_pv}, notes)
    vmax = 0.51e-3 * math.sqrt(c.fcuk) * b * h0
    chk.details.update(V_section_max=vmax, V_detailing=vmin)
    if gamma0 * abs(Vd) > vmax:
        chk.failed.append(f"section too small: γ0Vd > 0.51e-3 √fcu,k b h0 = {vmax:.0f} kN (5.2.11)")
    if gamma0 * abs(Vd) <= vmin:
        chk.notes.append(f"γ0Vd ≤ {vmin:.0f} kN: detailing stirrups per 9.3.12 suffice (5.2.12)")
    if rho_sv < sst.rho_sv_min:
        chk.failed.append(f"stirrup ratio {rho_sv:.4%} < minimum {sst.rho_sv_min:.2%} (9.3.12)")
    if dia < 8:
        chk.failed.append(f"stirrup diameter {dia} mm < 8 mm (9.3.12)")
    smax = min(s.h / 2, 400.0)
    if sv > smax:
        chk.failed.append(f"stirrup spacing {sv} mm > {smax:.0f} mm (h/2 and 400 mm, 9.3.12)")
    chk.failed += s.material_rules()
    return chk


# ── compression ─────────────────────────────────────────────────────────────
def _min_compression_steel(sec: RCSection, chk: Check):
    """9.1.12 item 1: all longitudinal bars ≥ 0.5 % of the gross section (0.6 % from C50), one side (the edge rows,
    RCSection.edge_masks) ≥ 0.2 %"""
    if sec.prestressed:
        chk.notes.append("9.1.12 (reinforced concrete members) not applied to a prestressed section")
        return
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
    compression members (5.3.10). Minimum longitudinal reinforcement per 9.1.12. Reinforced concrete only."""
    if sec.prestressed:
        raise NotImplementedError("5.3.1 covers reinforced-concrete members; prestressed sections are not supported here")
    c, st = sec.concrete, sec.steel
    A = sec.area
    As = sec.bars[:, 0].sum()
    if As / A > 0.03:
        A -= As
    phi = _phi_of(sec, l0, b_min, i_min)
    Nu = 0.9 * phi * (c.fcd * A + st.fsd_c * As) / 1e3
    chk = Check("axial compression", gamma0 * Nd, Nu, "kN", "JTG 3362-2018 5.3.1", {"phi": phi, "rho": As / sec.area})
    _min_compression_steel(sec, chk)
    chk.failed += sec.material_rules()
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


def _solve_x(s: RCSection, e: float, xb: float | None):
    """block depth x with M(x) = e N(x) and N > 0 (e from the gross centroid, positive towards the top), or None"""
    xs = np.geomspace(1e-4 * s.h, 50 * s.h, 800)
    FN = np.array([s.forces(x, xb) for x in xs])
    f = FN[:, 1] - e * FN[:, 0]
    roots = [k for k in range(len(xs) - 1) if f[k] * f[k + 1] <= 0 and FN[k + 1, 0] > 0]
    if not roots:
        return None
    k = roots[0]
    return brentq(lambda x: s.forces(x, xb)[1] - e * s.forces(x, xb)[0], xs[k], xs[k + 1], xtol=1e-9 * s.h)


def eccentric(sec: RCSection, Nd: float, Md: float, l0: float | None = None, *, gamma0: float = 1.0,
              method: str = "auto") -> Check:
    """eccentrically loaded compression member (5.3.4–5.3.9): finds the axial resistance Nu at the design eccentricity
    e = η e0 and checks γ0 Nd ≤ Nu. e0 = Md / Nd, at least max(20 mm, h/30) (5.3.9; 3362-2018 has no separate
    accidental eccentricity), η from eta_factor. Nd compression positive (kN), Md in kN·m (the sign picks the compressed
    face). method: "code" uses the closed-form formula of 5.3.8 for solid circles with at least 8 bars; "strain" the
    rule set above (any shape); "auto": "code" where it applies, else "strain".
    Small eccentricity with the load between the two steel groups also gets the reverse-failure check (5.3.4-4,
    5.3.5-3/4). Minimum longitudinal reinforcement per 9.1.12 (reinforced concrete). Prestressed sections: tendons in
    every formula, ξb by 5.3.3. Check `axial` out of plane (5.3.10) separately."""
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
    chk.failed += s.material_rules()
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
    # strain method: the block depth x with M(x) = e N(x), N > 0. With unequal steel the plastic centroid is off the
    # gross centroid; a load between the two is resisted with the far face more compressed: solve on the flipped section.
    s0 = s
    xb = s.xi_b(eccentric=True)
    x = _solve_x(s, e, xb)
    flipped = x is None
    if flipped:
        s = s.flipped()
        xb = s.xi_b(eccentric=True)
        x = _solve_x(s, -e, xb)
        if x is None:
            raise RuntimeError("no equilibrium at this eccentricity")
        chk.notes.append("load between the gross and the plastic centroid: the far face is the more compressed")
        if not ring:
            h0 = s.h0()
    Nu = s.forces(x, xb)[0]
    small = flipped or x > xb * h0
    chk.details.update(x=x, xi=x / h0, xi_b=xb, case="small eccentricity" if small else "large eccentricity")
    if ring:
        chk.resistance = Nu / 1e3
        return chk
    aref, compressed_p = s0.comp_ref()
    bot, _ = s0.edge_masks()
    pb, _ = s0.tendon_masks()
    if aref is None or not (bot.any() or pb.any()):
        chk.resistance = Nu / 1e3
        return chk
    if not small:
        e_c = e - s0.zg + aref  # from the load to the compression-steel resultant
        if x < 2 * aref and e_c > 0:  # 5.3.6 → 5.2.4
            Nu = s0.design_tension_moment(aref, compressed_p) / e_c
            chk.notes.append(f"x < 2a': moments about the compression steel (e' = {e_c:.0f} mm, 5.3.6)")
    else:
        # reverse failure (5.3.4-4, 5.3.5-3/4): the load between A's/A'p and As/Ap, failure starting at the far face.
        # Moments about the near-side steel resultant a' of the whole concrete section at fcd, of As at f'sd and of Ap at
        # (f'pd − σp0); e' = distance from the load (e0 without η) to a'. e0 is taken here without the 5.3.9 minimum,
        # the more unfavourable reading.
        F = np.r_[st.fsd * s0.bars[bot, 0], s0.tendon.fpd * s0.tendons[pb, 0] if s0.prestressed else []]
        d_t = float(np.average(np.r_[s0.bars[bot, 1], s0.tendons[pb, 1]], weights=F))
        z_load = s0.zg - e0_raw
        if aref < z_load < d_t:
            e_r = z_load - aref
            M_r = c.fcd * s0.area * (s0.zg - aref) + st.fsd_c * (s0.bars[bot, 0] * (s0.bars[bot, 1] - aref)).sum()
            if s0.prestressed:
                td = s0.tendons
                M_r += ((s0.tendon.fpd_c - td[pb, 2]) * td[pb, 0] * (td[pb, 1] - aref)).sum()
            N_r = M_r / e_r
            chk.details["N_reverse"] = N_r / 1e3
            if N_r < Nu:
                Nu = N_r
                chk.notes.append(f"reverse failure governs (e' = {e_r:.0f} mm, 5.3.4-4)")
    chk.resistance = Nu / 1e3
    return chk
