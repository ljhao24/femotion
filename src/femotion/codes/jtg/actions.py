"""Highway traffic actions, crowd load and the combinations of JTG D60-2015.

Units: kN, m, Hz. Clauses (docs/codes/JTG_D60-2015.md): 4.1.5 (basic combination), 4.1.6 (frequent and
quasi-permanent combinations), 4.1.8 (characteristic values for elastic stresses), 4.3.1 (lane and vehicle loads,
design lanes, transverse and longitudinal reduction), 4.3.2 (impact factor), 4.3.6 (crowd load); γ0 by design safety
class (4.1.5). Temperature actions (4.3.12): femotion.codes.jtg.temperature.

    from femotion.codes import jtg
    q, P = jtg.lane_load(L0=20.0)                        # Class I: 10.5 kN/m, 300 kN
    mu = jtg.impact_factor(jtg.simple_beam_frequency(20.0, EI, m))
    live = jtg.LiveLoad(L0=20.0, lanes=2, mu=mu)         # lane load × (1 + μ) × lanes × transverse reduction
    Qmax, Qmin = live.effects(il, "My")                  # il: femotion.frame.influence.InfluenceLines
    Sd = jtg.basic_combination(G, (Qmax, Qmin))          # (max, min) design effects, without γ0
    Ss = jtg.frequent_combination(G, LiveLoad(L0=20.0, lanes=2).effects(il, "My"))   # serviceability: no impact
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


def continuous_beam_frequencies(L: float, EI: float, m: float) -> tuple[float, float]:
    """estimated frequencies of a continuous beam (commentary to 4.3.2, eqs. 4-5, 4-6): f1 = 13.616 / (2π L²) √(EI / m)
    for positive moments and shear, f2 = 23.651 / (2π L²) √(EI / m) for negative moments. L: computed span (m)"""
    s = math.sqrt(EI / m) / (2 * math.pi * L * L)
    return 13.616 * s, 23.651 * s


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
    distribution factor as `distribution` (the factor already counts the lanes).
    L0: computed span for Pk (table 4.3.1-2); L_max: largest computed span of the structure for the longitudinal
    reduction (4.3.1 item 8), default L0. The two-lane floor (4.3.1 item 7) applies with the default transverse factor
    only. vehicle_effects use the same factor; for local loading pass lanes=1, transverse=1."""
    L0: float
    grade: str = "I"
    lanes: int = 1
    mu: float = 0.0
    distribution: float = 1.0
    transverse: float | None = None  # default: table 4.3.1-5 for `lanes`
    longitudinal: float | None = None  # default: table 4.3.1-6 for L_max
    L_max: float | None = None

    def factor(self) -> float:
        """multiplier on one lane's effect. With more than two lanes the loaded lanes are never taken below two lanes'
        effect (4.3.1 item 7), when the default transverse reduction is used."""
        if self.transverse is None:
            lanes = self.lanes * transverse_reduction(self.lanes)
            if self.lanes > 2:
                lanes = max(lanes, 2 * transverse_reduction(2))
        else:
            lanes = self.lanes * self.transverse
        L = self.L0 if self.L_max is None else self.L_max
        lo = longitudinal_reduction(L) if self.longitudinal is None else self.longitudinal
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


# ── crowd load (人群荷载), 4.3.6 ──
SIDEWALK_SLAB_LOAD = 4.0  # kN/m², sidewalk slabs as local members (4.3.6 item 3)
RAILING_LOADS = (0.75, 1.0)  # kN/m: horizontal on the post tops, vertical on the handrail (4.3.6 item 4)


def crowd_load(L0: float, *, dense: bool = False, footbridge: bool = False) -> float:
    """crowd load (kN/m², table 4.3.6) from the computed span L0 (m; continuous structures with unequal spans: the largest
    computed span): 3.0 up to 50 m, 2.5 from 150 m, 3.25 − 0.005 L0 between. dense: bridges with crowded pedestrian and
    non-motor traffic, × 1.15 (item 1-1); footbridge: 3.5 kN/m² (item 1-2)"""
    if footbridge:
        return 3.5
    w = 3.0 if L0 <= 50 else 2.5 if L0 >= 150 else 3.25 - 0.005 * L0
    return 1.15 * w if dense else w


@dataclass
class CrowdLoad:
    """crowd on the sidewalks of a spine model (4.3.6): the intensity × the loaded clear sidewalk width (m, both
    sidewalks together when both load the girder) × distribution, laid as a uniform load over the parts of the influence
    line of one sign (item 2), no point load. For one girder of a multi-girder deck pass width=1 and the girder's lateral
    distribution factor (per m² of crowd) as `distribution`.
    L0: computed span (m); with unequal continuous spans the largest one (item 1)."""
    L0: float
    width: float
    distribution: float = 1.0
    dense: bool = False
    footbridge: bool = False

    def intensity(self) -> float:
        return crowd_load(self.L0, dense=self.dense, footbridge=self.footbridge)

    def effects(self, il, comp: str):
        """(max, min) effects of every section of `il` (femotion.frame.influence.InfluenceLines); combine as another
        variable action: basic_combination(..., others=[crowd]), frequent_combination(..., kinds=["crowd"])"""
        return il.lane(comp, self.intensity() * self.width * self.distribution, 0.0)


def _is_pair(it) -> bool:
    """(effects, (γ unfavourable, γ favourable))"""
    return isinstance(it, tuple) and len(it) == 2 and isinstance(it[1], tuple) and len(it[1]) == 2 \
        and np.ndim(it[1][0]) == 0


def _items(G) -> list:
    """the permanent actions of G: one array, one (array, factors) pair, or a list of either"""
    if _is_pair(G) or not isinstance(G, (list, tuple)) or not len(G):
        return [G]
    if np.ndim(G[0]) > 0 or _is_pair(G[0]):
        return list(G)
    return [G]  # a plain sequence of numbers: one action


def basic_combination(G, Q, others=(), *, gamma_G: tuple[float, float] = (1.2, 1.0), gamma_Q: float | None = None,
                      vehicle: bool = False, gamma_others=1.4, psi_c: float = 0.75):
    """basic combination (基本组合, 4.1.5-1) of the ultimate limit state, effects added linearly, without γ0 (apply γ0 in
    the check: γ0 Sd ≤ R). Returns (max, min) design effects.

    G: permanent effects. An array; or a list whose items are arrays (all with gamma_G) or (array, (γ unfavourable,
    γ favourable)) pairs with their own factors from table 4.1.5-2 (e.g. earth pressure (1.4, 1.0)). For each permanent
    action and each extreme the unfavourable factor applies where its effect has the sign of that extreme, else the
    favourable one. gamma_G default (1.2, 1.0): concrete and masonry self-weight.
    Q: (max, min) of the leading variable action, normally traffic (when another variable action's effect exceeds the
    traffic effect it replaces traffic as the leading action, 4.1.5: pass it as Q and traffic in others).
    others: list of (max, min) of the other variable actions (crowd, …), each × γQj × ψc.
    Variable actions only count where unfavourable (4.1.4 item 2): positive parts of the max effects, negative parts
    of the min effects.
    gamma_Q: 1.4 for the lane load, 1.8 for the vehicle load (vehicle=True). gamma_others: one value or one per action;
    1.4 in general, 1.1 for wind. psi_c 0.75. γL (design-life adjustment) is 1.0 throughout."""
    hi = lo = 0.0
    items = _items(G)
    for it in items:
        g, (gu, gf) = (it if _is_pair(it) else (it, gamma_G))
        g = np.asarray(g, float)
        hi = hi + np.where(g > 0, gu, gf) * g
        lo = lo + np.where(g < 0, gu, gf) * g
    gq = gamma_Q if gamma_Q is not None else (1.8 if vehicle else 1.4)
    go = np.broadcast_to(np.asarray(gamma_others, float), (len(others),))
    hi = hi + gq * np.maximum(np.asarray(Q[0], float), 0.0)
    lo = lo + gq * np.minimum(np.asarray(Q[1], float), 0.0)
    for g_j, (qmax, qmin) in zip(go, others):
        hi = hi + psi_c * g_j * np.maximum(np.asarray(qmax, float), 0.0)
        lo = lo + psi_c * g_j * np.minimum(np.asarray(qmin, float), 0.0)
    return hi, lo


# ── serviceability combinations (4.1.6) and characteristic values (4.1.8) ──
# frequent and quasi-permanent value factors (ψf, ψq) by action, 4.1.5 item 2 and 4.1.6 (vehicle load without impact)
PSI = {"vehicle": (0.7, 0.4), "crowd": (1.0, 0.4), "wind": (0.75, 0.75), "temperature gradient": (0.8, 0.8),
       "other": (1.0, 1.0)}

DYNAMIC_FACTOR = (1.2, 0.85)  # self-weight of members being lifted or transported (4.1.10): unfavourable, favourable


def _permanent(G):
    """sum of permanent effects at their characteristic values (items may be arrays or (array, factors) pairs as for
    basic_combination; the factors are not used)"""
    items = _items(G)
    out = 0.0
    for it in items:
        out = out + np.asarray(it[0] if _is_pair(it) else it, float)
    return out


def _variable(hi, lo, Q, psi):
    """add ψ × the unfavourable parts of (max, min) of a variable action (4.1.4 item 2)"""
    return hi + psi * np.maximum(np.asarray(Q[0], float), 0.0), lo + psi * np.minimum(np.asarray(Q[1], float), 0.0)


def _pair(hi, lo):
    hi, lo = np.broadcast_arrays(np.asarray(hi, float), np.asarray(lo, float))
    return hi.copy(), lo.copy()


def _psi_of(kind, which):
    if isinstance(kind, (int, float)):
        return float(kind)
    if kind not in PSI:
        raise ValueError(f"action kind {kind!r}: one of {', '.join(PSI)} or a number")
    return PSI[kind][which]


def frequent_combination(G, Q=None, others=(), *, leading: str | float = "vehicle", kinds=()):
    """frequent combination (作用频遇组合, 4.1.6-1): Sfd = Σ Gik + ψf1 Q1k + Σ ψqj Qjk, effects added linearly.
    Returns (max, min). G: permanent effects at characteristic values (an array or a list, as for basic_combination;
    partial factors are 1.0 here). Q: (max, min) of the leading variable action, normally the vehicle load without impact
    (LiveLoad(mu=0), 6.1.1 of JTG 3362-2018), × ψf1 (0.7). others: list of (max, min) of the other variable actions, each
    × its quasi-permanent factor ψqj; kinds: one name of PSI (or a number) per action of others, default "other" (1.0).
    leading: name in PSI or a number for ψf1. Variable actions only count where unfavourable (4.1.4 item 2).
    Prestress (primary and secondary effects) is a permanent action (4.2.2)."""
    G = _permanent(G)
    hi, lo = G, G
    if Q is not None:
        hi, lo = _variable(hi, lo, Q, _psi_of(leading, 0))
    ks = list(kinds) + ["other"] * (len(others) - len(kinds))
    for k, q in zip(ks, others):
        hi, lo = _variable(hi, lo, q, _psi_of(k, 1))
    return _pair(hi, lo)


def quasi_permanent_combination(G, Q=None, others=(), *, leading: str | float = "vehicle", kinds=()):
    """quasi-permanent combination (作用准永久组合, 4.1.6-2): Sqd = Σ Gik + Σ ψqj Qjk (j from 1). Returns (max, min).
    Q: (max, min) of the vehicle load without impact (× ψq 0.4), others with kinds as frequent_combination."""
    G = _permanent(G)
    hi, lo = G, G
    if Q is not None:
        hi, lo = _variable(hi, lo, Q, _psi_of(leading, 1))
    ks = list(kinds) + ["other"] * (len(others) - len(kinds))
    for k, q in zip(ks, others):
        hi, lo = _variable(hi, lo, q, _psi_of(k, 1))
    return _pair(hi, lo)


def characteristic_combination(G, Q=None, others=()):
    """effects at characteristic values with all partial factors 1.0 (4.1.8; JTG 3362-2018 7.1.1: the vehicle load with
    impact): Σ Gik + Q1k + Σ Qjk, variable actions where unfavourable. Returns (max, min)."""
    G = _permanent(G)
    hi, lo = G, G
    for q in ([Q] if Q is not None else []) + list(others):
        hi, lo = _variable(hi, lo, q, 1.0)
    return _pair(hi, lo)
