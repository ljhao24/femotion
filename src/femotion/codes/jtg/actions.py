"""Highway traffic actions and the ultimate-limit-state combination of JTG D60-2015.

Units: kN, m, Hz. Clauses (docs/jtg.md): 4.1.5 (basic combination), 4.3.1 (lane and vehicle loads, design lanes, transverse
and longitudinal reduction), 4.3.2 (impact factor); γ0 by design safety class (4.1.5).

    from femotion.codes import jtg
    q, P = jtg.lane_load(L0=20.0)                        # Class I: 10.5 kN/m, 300 kN
    mu = jtg.impact_factor(jtg.simple_beam_frequency(20.0, EI, m))
    live = jtg.LiveLoad(L0=20.0, lanes=2, mu=mu)         # lane load × (1 + μ) × lanes × transverse reduction
    Qmax, Qmin = live.effects(il, "My")                  # il: femotion.frame.influence.InfluenceLines
    Sd = jtg.basic_combination(G, (Qmax, Qmin))          # (max, min) design effects, without γ0
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# vehicle load (车辆荷载), 4.3.1: axle loads (kN) front to rear and axle spacings (m); 550 kN in all
VEHICLE_AXLES = (30.0, 120.0, 120.0, 140.0, 140.0)
VEHICLE_SPACINGS = (3.0, 1.4, 7.0, 1.4)

GAMMA_0 = {1: 1.1, 2: 1.0, 3: 0.9}  # structural importance factor by design safety class


def lane_load(L0: float, grade: str = "I", shear: bool = False) -> tuple[float, float]:
    """lane load (车道荷载) of highway Class I or II: (qk kN/m, Pk kN). Class I: qk = 10.5 kN/m; Pk = 270 kN for a computed
    span L0 ≤ 5 m, 360 kN for L0 ≥ 50 m, linear between, i.e. 2 (L0 + 130). Class II: 0.75 × Class I.
    shear: Pk × 1.2 when computing shear effects."""
    q = 10.5
    P = 270.0 if L0 <= 5 else 360.0 if L0 >= 50 else 2.0 * (L0 + 130.0)
    if shear:
        P *= 1.2
    g = str(grade).upper().replace("CLASS", "").strip()
    if g in ("II", "2"):
        q, P = 0.75 * q, 0.75 * P
    elif g not in ("I", "1"):
        raise ValueError("grade 'I' or 'II'")
    return q, P


def impact_factor(f: float) -> float:
    """impact factor μ from the fundamental frequency f (Hz): 0.05 below 1.5 Hz, 0.45 above 14 Hz,
    0.1767 ln f − 0.0157 between"""
    if f < 1.5:
        return 0.05
    if f > 14:
        return 0.45
    return 0.1767 * math.log(f) - 0.0157


def simple_beam_frequency(L: float, EI: float, m: float) -> float:
    """fundamental frequency of a simply supported beam, f = π / (2 L²) √(EI / m) (commentary to 4.3.2, eq. 4-3):
    L m, EI kN·m², m t/m (mass per length)"""
    return math.pi / (2 * L * L) * math.sqrt(EI / m)


def design_lanes(width: float, two_way: bool = True) -> int:
    """number of design lanes from the carriageway width (m), table 4.3.1-4 (two-way below 6.0 m the table is blank;
    1 is returned)"""
    if two_way:
        edges = [(6.0, 1), (14.0, 2), (21.0, 4), (28.0, 6), (35.0, 8)]
    else:
        edges = [(7.0, 1), (10.5, 2), (14.0, 3), (17.5, 4), (21.0, 5), (24.5, 6), (28.0, 7), (31.5, 8)]
    for w, n in edges:
        if width < w:
            return n
    raise ValueError(f"carriageway {width} m is wider than table 4.3.1-4")


def transverse_reduction(lanes: int) -> float:
    """multi-lane transverse reduction factor, table 4.3.1-5 (1.20 for a single lane)"""
    tab = {1: 1.20, 2: 1.00, 3: 0.78, 4: 0.67, 5: 0.60, 6: 0.55, 7: 0.52, 8: 0.50}
    if lanes not in tab:
        raise ValueError("1 … 8 lanes")
    return tab[lanes]


def longitudinal_reduction(L0: float) -> float:
    """longitudinal reduction factor for long spans, table 4.3.1-6: none up to 150 m, then 0.97 / 0.96 / 0.95 / 0.94 /
    0.93 from 150 (exclusive) / 400 / 600 / 800 / 1000 m. Continuous structures: the largest computed span."""
    if L0 <= 150:
        return 1.0
    for lim, v in [(400, 0.97), (600, 0.96), (800, 0.95), (1000, 0.94)]:
        if L0 < lim:
            return v
    return 0.93


@dataclass
class LiveLoad:
    """traffic on a spine (single-girder) model: lane load × (1 + μ) × lanes × transverse reduction × longitudinal
    reduction × distribution. For one girder of a multi-girder deck pass lanes=1, transverse=1 and the girder's lateral
    distribution factor as `distribution` (the factor already counts the lanes)."""
    L0: float
    grade: str = "I"
    lanes: int = 1
    mu: float = 0.0
    distribution: float = 1.0
    transverse: float | None = None  # default: table 4.3.1-5 for `lanes`
    longitudinal: float | None = None  # default: table 4.3.1-6 for L0

    def factor(self) -> float:
        """multiplier on one lane's effect. With more than two lanes the loaded lanes are never taken below two lanes'
        effect (4.3.1 item 7), when the default transverse reduction is used."""
        if self.transverse is None:
            lanes = self.lanes * transverse_reduction(self.lanes)
            if self.lanes > 2:
                lanes = max(lanes, 2 * transverse_reduction(2))
        else:
            lanes = self.lanes * self.transverse
        lo = longitudinal_reduction(self.L0) if self.longitudinal is None else self.longitudinal
        return (1 + self.mu) * lanes * lo * self.distribution

    def effects(self, il, comp: str):
        """(max, min) effects of every section of `il` (femotion.frame.influence.InfluenceLines); Pk × 1.2 for shear"""
        q, P = lane_load(self.L0, self.grade, shear=comp in ("Vy", "Vz"))
        mx, mn = il.lane(comp, q, P)
        f = self.factor()
        return mx * f, mn * f

    def vehicle_effects(self, il, comp: str):
        """(max, min) effects of the 550 kN vehicle (车辆荷载) per lane, for local members, abutments and culverts;
        combine with gamma_Q=1.8 (basic_combination(..., vehicle=True))"""
        mx, mn = il.axles(comp, VEHICLE_AXLES, VEHICLE_SPACINGS)
        f = self.factor()
        return mx * f, mn * f


def basic_combination(G, Q, others=(), *, gamma_G: tuple[float, float] = (1.2, 1.0), gamma_Q: float | None = None,
                      vehicle: bool = False, gamma_others=1.4, psi_c: float = 0.75):
    """basic combination (基本组合) of the ultimate limit state, without γ0 (apply γ0 in the check: γ0 Sd ≤ R).

    G: permanent effects (array, or a list of arrays summed: self-weight, superimposed dead load);
    Q: (max, min) of the leading variable action, normally traffic (when another variable action's effect exceeds the
    traffic effect it replaces traffic as the leading action, 4.1.5: pass it as Q and traffic in others);
    others: list of (max, min) of the other variable actions (crowd, …), each × γQj × ψc. gamma_G: (unfavourable, favourable) = (1.2, 1.0) for concrete self-weight;
    the favourable factor is used where the permanent effect has the opposite sign to the extreme sought.
    gamma_Q: 1.4 for the lane load, 1.8 for the vehicle load (vehicle=True). gamma_others: one value or one per action;
    1.4 in general, 1.1 for wind.
    Returns (max, min) design effects."""
    if isinstance(G, (list, tuple)) and len(G) and np.ndim(G[0]) > 0:
        G = np.sum([np.asarray(g, float) for g in G], axis=0)
    G = np.asarray(G, float)
    gq = gamma_Q if gamma_Q is not None else (1.8 if vehicle else 1.4)
    gu, gf = gamma_G
    go = np.broadcast_to(np.asarray(gamma_others, float), (len(others),))
    hi = np.where(G > 0, gu, gf) * G + gq * np.asarray(Q[0], float)
    lo = np.where(G < 0, gu, gf) * G + gq * np.asarray(Q[1], float)
    for g_j, (qmax, qmin) in zip(go, others):
        hi = hi + psi_c * g_j * np.asarray(qmax, float)
        lo = lo + psi_c * g_j * np.asarray(qmin, float)
    return hi, lo
