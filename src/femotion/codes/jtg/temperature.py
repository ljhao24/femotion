"""Temperature actions of JTG D60-2015 4.3.12 and the temperature stresses of JTG 3362-2018 4.3.10 and appendix D.

Units: °C, mm, MPa; forces kN, moments kN·m (sagging +), curvatures 1/m. Clauses: docs/codes/JTG_D60-2015.md (4.3.12)
and docs/codes/JTG_3362-2018.md (4.3.10, appendix D).

The vertical gradient of D60 figure 4.3.12 is a piecewise-linear temperature over the depth (Profile). Over a section
it gives (appendix D) the restraint resultants Nt, Mt0 and the self-equilibrating stresses σt = −Nt/A0 + Mt y/I0 +
ty αc Ec; on a statically indeterminate structure the free strain and curvature add the secondary moment Mt′
(D.0.2), found on the frame with femotion.frame.thermal:

    from femotion.codes import jtg
    from femotion.frame.thermal import effects
    grad = jtg.temperature_gradient(H=2000, surfacing="asphalt", asphalt=100)     # T1 = 14, T2 = 5.5 °C
    th = jtg.thermal_stresses(sec, grad)                                            # positive gradient, primary
    eps, kappa = th.free_strain()
    Mt2 = effects(sv, deck_beams, secs, "My", curvature=kappa)                       # secondary moments (kN·m)
    th_k = jtg.thermal_stresses(sec, grad, M_sec=Mt2[k])                             # at section k
    jtg.crack_resistance(sec, Ms[k], Ml[k], pc_class="A", temperature=th_k)          # ψ = 0.8 in the frequent combination

ThermalStresses keeps the positive gradient; .negative() is the negative one (× −0.5, figure 4.3.12 note). The checks
of femotion.codes.jtg.sls take, at each point, whichever of the two (or neither) is unfavourable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .rc import RCSection

# table 4.3.12-1: linear expansion coefficient (1/°C)
EXPANSION = {"steel": 12e-6, "concrete": 10e-6, "concrete block masonry": 9e-6, "masonry": 8e-6}

# table 4.3.12-2: effective temperature (°C), (highest, lowest), by climate zone (appendix A) and structure: steel bridges
# with steel decks, steel bridges with concrete decks, concrete and masonry bridges. In brackets in the table: lowest
# values for Kunming, Nanning, Guangzhou and Fuzhou (south=True).
EFFECTIVE_TEMPERATURE = {
    "severe cold": {"steel deck": (46, -43), "concrete deck": (39, -32), "concrete": (34, -23)},  # 严寒地区
    "cold": {"steel deck": (46, -21), "concrete deck": (39, -15), "concrete": (34, -10)},  # 寒冷地区
    "mild": {"steel deck": (46, -9), "concrete deck": (39, -6), "concrete": (34, -3)},  # 温热地区
}
_SOUTH_LOWEST = {"steel deck": -3, "concrete deck": -1, "concrete": 0}

# table 4.3.12-3: T1, T2 (°C) of the positive vertical gradient by deck surfacing
GRADIENT_TEMPERATURES = {"concrete": (25.0, 6.7), "asphalt 50": (20.0, 6.7), "asphalt 100": (14.0, 5.5)}
NEGATIVE_GRADIENT = -0.5  # negative (reverse) gradient = positive × −0.5 (figure 4.3.12 note)


def effective_temperature(zone: str, structure: str = "concrete", south: bool = False) -> tuple[float, float]:
    """(highest, lowest) effective temperature of table 4.3.12-2 (°C). zone: "severe cold", "cold", "mild" (appendix A
    of D60); structure: "steel deck" (steel bridges with steel decks), "concrete deck" (steel bridges with concrete
    decks), "concrete" (concrete and masonry bridges); south: the bracketed lowest values (Kunming, Nanning, Guangzhou,
    Fuzhou; mild zone only). The uniform temperature action runs from the structure's temperature when it is restrained
    (closure) to these two values (4.3.12 item 2): strain α (T − T_closure), femotion.frame.thermal."""
    if zone not in EFFECTIVE_TEMPERATURE:
        raise ValueError(f"climate zone {zone!r}: one of {', '.join(EFFECTIVE_TEMPERATURE)}")
    if structure not in EFFECTIVE_TEMPERATURE[zone]:
        raise ValueError(f"structure {structure!r}: one of {', '.join(EFFECTIVE_TEMPERATURE[zone])}")
    hi, lo = EFFECTIVE_TEMPERATURE[zone][structure]
    if south:
        if zone != "mild":
            raise ValueError("the bracketed values of table 4.3.12-2 belong to the mild zone")
        lo = _SOUTH_LOWEST[structure]
    return float(hi), float(lo)


@dataclass(frozen=True)
class Profile:
    """temperature over the depth (°C), piecewise linear through the points (z mm from the top, t), zero outside them.
    A step is two points at the same depth; exactly at the step the value below it is returned (the integrals do not
    depend on it)."""
    z: tuple
    t: tuple

    def __post_init__(self):
        if len(self.z) != len(self.t) or len(self.z) < 2 or np.any(np.diff(self.z) < 0):
            raise ValueError("Profile: at least two points, depths not decreasing")

    def __call__(self, z):
        z = np.asarray(z, float)
        v = np.interp(z, self.z, self.t)
        return np.where((z < self.z[0]) | (z > self.z[-1]), 0.0, v)

    def scaled(self, k: float) -> "Profile":
        return Profile(self.z, tuple(k * v for v in self.t))


def temperature_gradient(H: float, *, surfacing: str = "concrete", asphalt: float | None = None, steel: bool = False,
                         t: float | None = None, T1: float | None = None, T2: float | None = None) -> Profile:
    """positive vertical temperature gradient of D60 4.3.12 item 3 (figure 4.3.12): T1 at the top of the deck (pavement
    not included), T2 at 100 mm, zero at 100 + A; A = H − 100 for concrete girders shallower than 400 mm, else 300 mm.
    Steel girders with a concrete deck (steel=True, t the deck thickness): A = 300 mm, and below the deck the steel
    keeps the temperature found at depth t down to the girder's bottom H (the vertical line "for steel girders only" of
    the figure). T1, T2 from table 4.3.12-3 by surfacing: "concrete" (cement concrete pavement, 25 / 6.7 °C), "asphalt"
    with its thickness asphalt = 50 … 100 mm (20 / 6.7 °C at 50 mm, 14 / 5.5 °C at 100 mm; between them T1 linearly (D60
    commentary) and T2 likewise (program convention)); asphalt over a concrete levelling layer: the asphalt thickness only
    (commentary). T1, T2 may be given directly. The negative gradient is this × −0.5 (NEGATIVE_GRADIENT);
    thermal_stresses handles both (for concrete sections: a steel girder's own α and E are not handled there)."""
    if T1 is None or T2 is None:
        if surfacing == "concrete":
            t1, t2 = GRADIENT_TEMPERATURES["concrete"]
        elif surfacing == "asphalt":
            if asphalt is None or not 50 <= asphalt <= 100:
                raise ValueError("table 4.3.12-3 gives asphalt surfacing of 50 … 100 mm; pass asphalt=… or T1, T2")
            a, b = GRADIENT_TEMPERATURES["asphalt 50"], GRADIENT_TEMPERATURES["asphalt 100"]
            k = (asphalt - 50) / 50
            t1, t2 = a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1])
        else:
            raise ValueError("surfacing: 'concrete' or 'asphalt'")
        T1 = t1 if T1 is None else T1
        T2 = t2 if T2 is None else T2
    A = 300.0 if (H >= 400 or steel) else H - 100.0
    if A <= 0:
        raise ValueError("figure 4.3.12 needs H > 100 mm")
    concrete_part = Profile((0.0, 100.0, 100.0 + A), (float(T1), float(T2), 0.0))
    if not steel:
        return concrete_part
    if t is None or not 0 < t < H:
        raise ValueError("steel=True: pass t, the concrete deck's thickness (figure 4.3.12)")
    Tt = float(concrete_part(t))
    pts = [(z, v) for z, v in zip(concrete_part.z, concrete_part.t) if z < t] + [(t, Tt), (float(H), Tt)]
    return Profile(tuple(z for z, _ in pts), tuple(v for _, v in pts))


# ── appendix D: temperature stresses ─────────────────────────────────────────
@dataclass
class ThermalStresses:
    """appendix D for one temperature profile on one section. Lengths mm, forces N, stresses MPa (compression +)"""
    profile: Profile
    alpha: float
    Ec: float
    A: float  # area of the section used (mm²)
    zc: float  # depth of its centroid (mm)
    I: float  # second moment about it (mm⁴)
    Nt: float  # N, D.0.1-1
    Mt0: float  # N·mm, D.0.1-2 (sagging +, y up)
    M_sec: float = 0.0  # N·mm, secondary moment Mt′ (D.0.2), sagging +
    N_sec: float = 0.0  # N, secondary axial force (tension +; frames restraining the axial strain, program extension)
    z_max: float = math.inf  # cracked sections: the neutral axis; the gradient on its tension side is ignored
    hogging: bool = False  # True: the tension side (cracked) is above z_max, else below it
    kind: str = "transformed"

    def t(self, z):
        """temperature at depth z, zero where the gradient is not counted (below the neutral axis of a cracked section)"""
        z = np.asarray(z, float)
        live = (z >= self.z_max) if self.hogging else (z <= self.z_max)
        return np.where(live, self.profile(z), 0.0)

    def sigma(self, z):
        """concrete stress at depth z (MPa, compression +): −Nt/A0 + (Mt0 + Mt′) y/I0 + ty αc Ec (D.0.1-3, D.0.2), y up
        from the centroid; − N′/A0 for a secondary axial force"""
        z = np.asarray(z, float)
        y = self.zc - z
        s = -self.Nt / self.A + (self.Mt0 + self.M_sec) * y / self.I + self.t(z) * self.alpha * self.Ec \
            - self.N_sec / self.A
        return float(s) if s.ndim == 0 else s

    def negative(self) -> "ThermalStresses":
        """the negative gradient: every temperature × −0.5 (secondary effects too)"""
        k = NEGATIVE_GRADIENT
        return replace(self, profile=self.profile.scaled(k), Nt=k * self.Nt, Mt0=k * self.Mt0, M_sec=k * self.M_sec,
                       N_sec=k * self.N_sec)

    def unfavourable(self, z, sense: str = "tension") -> float:
        """the more unfavourable of the positive gradient, the negative gradient and none at depth z: the smallest
        compression (most tension) for sense="tension", the largest compression for "compression" (MPa, compression +;
        a variable action counts only where unfavourable, D60 4.1.4)"""
        a, b = self.sigma(z), self.negative().sigma(z)
        return min(0.0, a, b) if sense == "tension" else max(0.0, a, b)

    @property
    def N_t(self) -> float:
        """Nt (kN)"""
        return self.Nt / 1e3

    @property
    def M_t0(self) -> float:
        """Mt0 (kN·m, sagging +)"""
        return self.Mt0 / 1e6

    def free_strain(self) -> tuple[float, float]:
        """(ε, κ) the section takes when free: axial strain Nt/(Ec A0) and curvature Mt0/(Ec I0) (1/m, sagging +), the
        imposed strains for femotion.frame.thermal (statically indeterminate structures, D.0.2). From an uncracked
        section (transformed, net, gross or literal); the frame's own stiffness follows 4.3.1 (Ec I, 0.8 Ec I for members
        allowed to crack)."""
        if self.kind == "cracked":
            raise ValueError("free_strain: use an uncracked section for the frame's imposed curvature (4.3.1, D.0.2)")
        return self.Nt / (self.Ec * self.A), self.Mt0 / (self.Ec * self.I) * 1e3


def _gauss(f, a, b, n=16):
    if b <= a:
        return 0.0
    x, w = np.polynomial.legendre.leggauss(n)
    z = (b - a) / 2 * x + (a + b) / 2
    return (b - a) / 2 * float(np.dot(w, [f(v) for v in z]))


def _concrete_integrals(sec: RCSection, t, z_lo: float, z_hi: float, zc: float):
    """(∫ t b dz, ∫ t b (zc − z) dz) over the concrete between z_lo and z_hi, split at every piece boundary and profile
    point (Gauss–Legendre on each part: exact for rectangles, converged for circles)"""
    br = {0.0, sec.h, z_lo, z_hi}
    for p in sec.pieces:
        if hasattr(p, "top"):
            br |= {p.top, p.bottom}
        else:
            br |= {p.zc - p.D / 2, p.zc + p.D / 2}
    br |= set(t.profile.z)
    pts = sorted(v for v in br if z_lo <= v <= z_hi)
    a0 = a1 = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        e = 1e-9 * (b - a)
        f0 = lambda z: float(t.t(min(max(z, a + e), b - e))) * sec.width(z)
        a0 += _gauss(f0, a, b)
        a1 += _gauss(lambda z: f0(z) * (zc - z), a, b)
    return a0, a1


def thermal_stresses(sec: RCSection, profile: Profile, *, M_sec: float = 0.0, N_sec: float = 0.0,
                     alpha: float = EXPANSION["concrete"], kind: str = "transformed", cracked: bool = False,
                     hogging: bool = False, M: float | None = None, M_p2: float = 0.0) -> ThermalStresses:
    """appendix D for a temperature profile (normally temperature_gradient, the positive gradient) on a section:
    Nt = Σ Ay ty αc Ec (D.0.1-1), Mt0 = −Σ Ay ty αc Ec ey (D.0.1-2), σt = −Nt/A0 + Mt0 y/I0 + ty αc Ec (D.0.1-3); with the
    secondary moment of a continuous beam M_sec (kN·m, sagging +, femotion.frame.thermal for the free curvature) Mt =
    Mt′ + Mt0 (D.0.2). N_sec: secondary axial force (kN, tension +) where the frame restrains the axial strain (rigid
    frames; program extension). alpha: αc, table 4.3.12-1.
    Section: kind="transformed" (A0, I0, default), "net" or "gross"; the area units Ay are those of the same section
    (concrete, and the bars and tendons as transformed areas at their depths: program convention, so that the
    self-equilibrating stresses balance on that section and a uniform temperature gives none). kind="literal": appendix
    D as printed, Ay the concrete only, A0, I0 of the transformed section (a few per cent more at the top edge).
    cracked=True (appendix D commentary): the cracked transformed section, the gradient on its tension side ignored
    (sagging: below the neutral axis; hogging=True, top in tension: above it, program reading). Reinforced concrete: the
    neutral axis in bending (7.2.4). Class B prestressed members: under Np0 and the moment M (kN·m, the combination's,
    sagging +) with M_p2, by appendix J (as jtg.service_stresses); M is required."""
    c = sec.concrete
    if cracked:
        from .sls import _cracked
        s = sec.flipped() if hogging else sec
        if sec.prestressed:
            if M is None:
                raise ValueError("cracked prestressed section: pass M (kN·m) for the neutral axis under Np0 (appendix J)")
            k = -1.0 if hogging else 1.0
            Np0, hps = s.prestress_resultant("p0")
            eN = k * (M + M_p2) * 1e6 / Np0 - hps
            r = _cracked(s, N=Np0, z_R=-eN)
            if r["x"] == math.inf:
                raise ValueError("the section stays in compression under Np0 and M: not cracked, use cracked=False")
        else:
            r = _cracked(s, limits=False)
        x, A, zc_s, I = r["x"], r["A"], r["zc"], r["I"]
        zc = sec.h - zc_s if hogging else zc_s
        z_cut = sec.h - x if hogging else x
        aS = s.steel.Es / c.Ec
        aP = s.tendon.Ep / c.Ec if s.has_internal else 0.0
        wb, wp = r["w_bars"], r["w_tendons"]  # appendix J notes (compression steel limits) as in the cracked section
        steel = [(aS * a * w, d) for (a, d), w in zip(sec.bars, wb)] + \
            [(aP * a * w, d) for (a, d, _, _), w in zip(sec.tendons, wp)]
        th = ThermalStresses(profile, alpha, c.Ec, A, zc, I, 0.0, 0.0, z_max=z_cut, hogging=hogging, kind="cracked")
        z_lo, z_hi = (z_cut, sec.h) if hogging else (0.0, z_cut)
    else:
        if kind == "gross":
            A, zc, I, steel = sec.area, sec.zg, sec.I, []
        elif kind in ("net", "transformed"):
            A, zc, I = sec.props(kind)
            steel = sec._steel_points(kind)
        elif kind == "literal":
            A, zc, I = sec.props("transformed")
            steel = []
        else:
            raise ValueError("kind: 'transformed', 'net', 'gross' or 'literal'")
        th = ThermalStresses(profile, alpha, c.Ec, A, zc, I, 0.0, 0.0, kind=kind)
        z_lo, z_hi = 0.0, sec.h
    q0, q1 = _concrete_integrals(sec, th, z_lo, z_hi, zc)
    for a, d in steel:
        td = float(th.t(d))
        q0 += a * td
        q1 += a * td * (zc - d)
    th.Nt = alpha * c.Ec * q0
    th.Mt0 = -alpha * c.Ec * q1
    th.M_sec = M_sec * 1e6
    th.N_sec = N_sec * 1e3
    return th
