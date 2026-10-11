"""Prestress losses and effective prestress, JTG 3362-2018 6.1.4–6.1.7, 6.2 and appendices C, G, H.

Units: MPa, mm, days, °C. Each function is one clause; femotion.codes.jtg.rc takes the resulting σp0 per tendon, and
femotion.frame.prestress turns the effective force into frame effects (secondary moments).

    from femotion.codes import jtg
    from femotion.codes.jtg import prestress as ps
    t = jtg.tendon("strand", 1860)
    s_con = t.sigma_con_max()                                     # 0.75 fpk = 1395
    l1 = ps.friction_loss(s_con, mu=0.25, theta=0.21, k=0.0015, x=15.0)
    l5 = ps.relaxation_loss(t, s_con - l1 - l2 - l4)
    eps = ps.shrinkage_strain(jtg.concrete(50), RH=75, h=250, t=3650, t0=7)
    phi = ps.creep_coefficient(jtg.concrete(50), RH=75, h=250, t=3650, t0=7)
    l6 = ps.creep_shrinkage_loss(t.Ep, eps, alpha_EP, sigma_pc, phi, rho, rho_ps)
    lI, lII = ps.loss_stages(post=True, l1=l1, l2=l2, l4=l4, l5=l5, l6=l6)
    sp0 = ps.sigma_p0(s_con, lI + lII, post=True, alpha_EP=alpha_EP, sigma_pc=sigma_pc_eff)

Clauses, formulas and tables: docs/codes/JTG_3362-2018.md.
"""
from __future__ import annotations

import math

import numpy as np

from .materials import Concrete, Tendon

# ── 5.1.6, 6.1.8 pretensioned tendons near the member ends ─────────────────
# table 5.1.6: anchorage length la / d, columns C40, C45, C50, C55, C60, ≥C65, for 1×7 strand (fpd 1260) and
# helical-rib wire (fpd 1200)
_LA = {"strand": ((130, 125, 120, 115, 110, 105), 1260.0), "wire": ((95, 90, 85, 83, 80, 80), 1200.0)}
# table 6.1.8: transfer length ltr / d at σpe = 1000 MPa, columns C40, C45, C50, ≥C55 (f'cu at release)
_LTR = {"strand": (67, 64, 60, 58), "wire": (58, 56, 53, 51)}


def anchorage_length(tendon: Tendon, c: Concrete, d: float) -> float:
    """anchorage length la (mm) of a pretensioned tendon, table 5.1.6: d nominal diameter (mm); the table values are
    for 1×7 strand with fpd = 1260 MPa and helical-rib wire with fpd = 1200 MPa, other fpd in proportion (note 2)"""
    if tendon.kind not in _LA:
        raise ValueError("table 5.1.6 covers 1×7 strand and helical-rib wire")
    if c.grade < 40:
        raise ValueError("table 5.1.6 starts at C40")
    vals, fpd0 = _LA[tendon.kind]
    return vals[min((c.grade - 40) // 5, 5)] * d * tendon.fpd / fpd0


def transfer_length(tendon: Tendon, fcu_release: float, d: float, sigma_pe: float) -> float:
    """transfer length ltr (mm) of a pretensioned tendon, table 6.1.8: by the cube strength f'cu at release (linear
    between the tabulated classes, note 1), d nominal diameter (mm); table values at σpe = 1000 MPa, other σpe in
    proportion (note 2)"""
    if tendon.kind not in _LTR:
        raise ValueError("table 6.1.8 covers 1×7 strand and helical-rib wire")
    if fcu_release < 40:
        raise ValueError("table 6.1.8 starts at C40")
    v = float(np.interp(fcu_release, [40, 45, 50, 55], _LTR[tendon.kind]))
    return v * d * sigma_pe / 1000.0


def fpd_in_anchorage(fpd: float, x: float, la: float, start: float = 0.0) -> float:
    """design tensile strength of a pretensioned tendon at distance x (mm) from the member end inside its anchorage zone
    (5.1.6, for the normal- and inclined-section flexural resistance there): 0 at the start of anchorage, fpd at its end
    la further, linear between. Sudden release: start = 0.25 ltr (table 5.1.6 note 1). Use the result through
    dataclasses.replace(tendon, fpd=…) in the section."""
    return fpd * min(max((x - start) / la, 0.0), 1.0)


# ── 6.2.2 friction ──────────────────────────────────────────────────────────
# table 6.2.2: duct → (k, μ for strand and wire (range), μ for threaded bars or None)
DUCT_FRICTION = {
    "metal corrugated": (0.0015, (0.20, 0.25), 0.50),   # 预埋金属波纹管
    "plastic corrugated": (0.0015, (0.15, 0.20), None),  # 预埋塑料波纹管
    "iron sheet": (0.0030, (0.35, 0.35), 0.40),          # 预埋铁皮管
    "steel pipe": (0.0010, (0.25, 0.25), None),          # 预埋钢管
    "cored": (0.0015, (0.55, 0.55), 0.60),               # 抽芯成型
}
# external tendons (k = 0): μ for strand; in brackets for unbonded (greased) strand
DUCT_FRICTION_EXTERNAL = {
    "steel pipe": ((0.20, 0.30), (0.08, 0.10)),          # 钢管
    "HDPE": ((0.12, 0.15), (0.08, 0.10)),                # 高密度聚乙烯管
}


def friction_loss(sigma_con: float, mu: float, theta: float, k: float, x: float) -> float:
    """σl1 = σcon [1 − e^−(μθ + kx)] (6.2.2): θ sum of the angles turned from the jacking end (rad), x duct length from
    the jacking end (m, may be its projection on the member axis), k and μ from table 6.2.2 (DUCT_FRICTION; external
    tendons: friction_loss_external)"""
    return sigma_con * (1 - math.exp(-(mu * theta + k * x)))


def friction_loss_external(sigma_con: float, mu: float, theta) -> float:
    """σl1 of an external tendon (6.2.2 with the note to table 6.2.2): only the ducts in the deviators and anchorages
    count and k = 0, so σl1 = σcon [1 − e^(−μ Σθ)]; theta: the angle (rad) or the angles the tendon turns through in the
    deviators and anchorage ducts from the jacking end to the section. μ from DUCT_FRICTION_EXTERNAL: bare strand the
    first range, unbonded (greased, sheathed) strand the second (the values in brackets); measured values preferred."""
    return friction_loss(sigma_con, mu, float(np.sum(theta)), 0.0, 0.0)


# ── 6.2.3 anchorage set, Appendix G ─────────────────────────────────────────
# table 6.2.3, Δl (mm) per anchorage or joint
ANCHOR_SET = {
    "wire cone": 6.0,               # 钢丝束的钢制锥形锚具
    "wedge with top pressure": 4.0,  # 夹片式锚具 有顶压时
    "wedge": 6.0,                    # 夹片式锚具 无顶压时
    "nut gap": (1.0, 3.0),           # 带螺帽锚具的螺帽缝隙（一次张拉 2～3，二次张拉 1）
    "button head": 1.0,              # 镦头锚具
    "shim gap": 2.0,                 # 每块后加垫板的缝隙
    "mortar joint": 1.0,             # 水泥砂浆接缝
    "epoxy joint": 1.0,              # 环氧树脂砂浆接缝
}


def anchorage_loss_straight(dl: float, l: float, Ep: float) -> float:
    """σl2 = ΣΔl / l · Ep (6.2.3) for straight tendons: dl = ΣΔl (mm), l from the jacking end to the fixed end (mm)"""
    return dl / l * Ep


def anchorage_loss_reverse(sigma_con: float, sigma_end: float, l: float, dl: float, Ep: float, x):
    """σl2 at distance x (mm) from the jacking end of a curved tendon, with reverse friction (Appendix G, G.0.2).
    sigma_end: stress at the fixed end after friction; l: jacking end to fixed end (mm); dl = ΣΔl (mm).
    Δσd = (σcon − σend)/l (G.0.2-2); lf = √(ΣΔl·Ep/Δσd) (G.0.2-1). lf ≤ l: σl2 = Δσ (lf − x)/lf within lf, Δσ = 2Δσd·lf
    (G.0.2-3, -4). lf > l: σl2 = Δσ' − 2x·Δσd (G.0.2-5) with Δσ' from the trapezoid area ΣΔl·Ep, which for the
    straight-line stress distributions of figure G.0.2 is Δσ' = ΣΔl·Ep / l + Δσd·l.
    Returns (σl2 at x, lf)."""
    dsd = (sigma_con - sigma_end) / l
    if dsd <= 0:
        raise ValueError("no friction loss along the tendon: use anchorage_loss_straight")
    lf = math.sqrt(dl * Ep / dsd)
    x = np.asarray(x, float)
    if (x < 0).any() or (x > l).any():
        raise ValueError("Appendix G covers 0 ≤ x ≤ l")
    if lf <= l:
        loss = np.where(x < lf, 2 * dsd * lf * (lf - x) / lf, 0.0)
    else:
        loss = dl * Ep / l + dsd * l - 2 * x * dsd
    return (float(loss) if loss.ndim == 0 else loss), lf


# ── 6.2.4 temperature difference (pretensioning) ────────────────────────────
def temperature_loss(t2: float, t1: float) -> float:
    """σl3 = 2 (t2 − t1) (6.2.4): t2 highest temperature of the tensioned steel during heat curing, t1 temperature of the
    bed when tensioning (°C)"""
    return 2.0 * (t2 - t1)


# ── 6.2.5 elastic shortening, Appendix H ────────────────────────────────────
def elastic_shortening_post(alpha_EP: float, d_sigma_pc) -> float:
    """σl4 = αEP ΣΔσpc (6.2.5-1), post-tensioning in batches: Δσpc are the concrete stresses at the centroid of the
    tendons already anchored, from each later batch (compression +)"""
    return alpha_EP * float(np.sum(d_sigma_pc))


def elastic_shortening_pre(alpha_EP: float, sigma_pc: float) -> float:
    """σl4 = αEP σpc (6.2.5-2), pretensioning: σpc concrete stress at the steel centroid from the whole prestress"""
    return alpha_EP * sigma_pc


def elastic_shortening_simplified(m: int, alpha_EP: float, d_sigma_pc: float) -> float:
    """σl4 = (m − 1)/2 · αEP Δσpc (H.0.1), post-tensioning one tendon after another: m number of tendons (or of batches,
    e.g. segments of a balanced cantilever), Δσpc mean concrete stress at the centroid of all tendons from one tendon"""
    return (m - 1) / 2 * alpha_EP * d_sigma_pc


# ── 6.2.6 relaxation, table C.3.1 ───────────────────────────────────────────
def relaxation_loss(tendon: Tendon, sigma_pe: float, *, overstress: bool = False, low_relaxation: bool = True,
                    sigma_con: float | None = None) -> float:
    """final relaxation loss σl5 (6.2.6). Wire and strand: Ψ ζ (0.52 σpe/fpk − 0.26) σpe (6.2.6-1), Ψ = 1.0 single
    stressing / 0.9 overstressing, ζ = 1.0 class I (normal) / 0.3 class II (low relaxation); σpe the stress at
    anchoring: σcon − σl1 − σl2 − σl4 (post-tensioned), σcon − σl2 (pretensioned). Threaded bars: 0.05 σcon single
    stressing, 0.035 σcon overstressing (6.2.6-2, -3). A negative value of 6.2.6-1 (σpe below 0.5 fpk) is taken as 0
    (program convention)."""
    if tendon.kind == "bar":
        if sigma_con is None:
            raise ValueError("threaded bars: pass sigma_con")
        return (0.035 if overstress else 0.05) * sigma_con
    psi = 0.9 if overstress else 1.0
    zeta = 0.3 if low_relaxation else 1.0
    return max(psi * zeta * (0.52 * sigma_pe / tendon.fpk - 0.26) * sigma_pe, 0.0)


def relaxation_ratio(days: float) -> float:
    """ratio of the relaxation loss at a given time to its final value (table C.3.1: 2 d 0.50, 10 d 0.61, 20 d 0.74,
    30 d 0.87, 40 d 1.00), linear between the tabulated times (program convention), 1.0 from 40 days"""
    if days < 2:
        raise ValueError("table C.3.1 starts at 2 days")
    return float(np.interp(days, [2, 10, 20, 30, 40], [0.50, 0.61, 0.74, 0.87, 1.00]))


# ── Appendix C: shrinkage and creep ─────────────────────────────────────────
# method="table" (default) follows the design procedure of C.1.3 and C.2.3 (εcs0 from table C.1.2, φ0 from table C.2.2,
# βH with RH = 55 % or 80 %); method="formula" evaluates C.1.1 and C.2.1 with the class's own fcm (C25–C50).
_RH0, _H0, _T1, _FCM0, _BETA_SC = 100.0, 100.0, 1.0, 10.0, 5.0


def _fcm(c: Concrete) -> float:
    """mean cylinder strength at 28 days, fcm = 0.8 fcu,k + 8 MPa (C.1.1, for C25–C50)"""
    return 0.8 * c.fcuk + 8.0


def _rh_check(RH):
    if not 40 <= RH < 99:
        raise ValueError("Appendix C covers 40 % ≤ RH < 99 %")


def beta_s(t_minus_ts: float, h: float) -> float:
    """βs(t − ts) = [((t − ts)/t1) / (350 (h/h0)² + (t − ts)/t1)]^0.5 (C.1.1-5); h notional size 2A/u (mm)"""
    d = max(t_minus_ts, 0.0) / _T1
    return math.sqrt(d / (350 * (h / _H0) ** 2 + d))


def nominal_shrinkage(c: Concrete, RH: float, method: str = "table") -> float:
    """εcs0. "formula": εs(fcm)·βRH with εs = [160 + 10 βsc (9 − fcm/fcm0)]·10⁻⁶, βRH = 1.55 [1 − (RH/RH0)³]
    (C.1.1-2…-4), for C25–C50. "table": table C.1.2, 0.529·10⁻³ (40 % ≤ RH < 70 %) or 0.310·10⁻³ (70 % ≤ RH < 99 %),
    × √(32.4/fck) from C50 up (note 3). Above C50 the formula's fcm is not defined, so the table is used."""
    _rh_check(RH)
    if method == "formula" and c.grade <= 50:
        eps_s = (160 + 10 * _BETA_SC * (9 - _fcm(c) / _FCM0)) * 1e-6
        return eps_s * 1.55 * (1 - (RH / _RH0) ** 3)
    v = 0.529e-3 if RH < 70 else 0.310e-3
    return v * (math.sqrt(32.4 / c.fck) if c.grade >= 50 else 1.0)


def shrinkage_strain(c: Concrete, RH: float, h: float, t: float, t0: float, ts: float = 3.0,
                     method: str = "table") -> float:
    """εcs(t, t0) = εcs0 [βs(t − ts) − βs(t0 − ts)] (C.1.3): shrinkage from age t0 (start of the effect, e.g. anchoring)
    to t (days); ts age at the end of curing (3–7 d); h notional size (mm)"""
    if not t > t0 >= ts:
        raise ValueError("need t > t0 ≥ ts")
    return nominal_shrinkage(c, RH, method) * (beta_s(t - ts, h) - beta_s(t0 - ts, h))


def beta_H(RH: float, h: float) -> float:
    """βH = 150 [1 + (1.2 RH/RH0)^18] h/h0 + 250 ≤ 1500 (C.2.1-7)"""
    return min(150 * (1 + (1.2 * RH / _RH0) ** 18) * h / _H0 + 250, 1500.0)


def beta_c(t_minus_t0: float, RH: float, h: float) -> float:
    """βc(t − t0) = [((t − t0)/t1) / (βH + (t − t0)/t1)]^0.3 (C.2.1-6)"""
    d = max(t_minus_t0, 0.0) / _T1
    return (d / (beta_H(RH, h) + d)) ** 0.3


def beta_t0(t0: float) -> float:
    """β(t0) = 1 / (0.1 + (t0/t1)^0.2) (C.2.1-5)"""
    return 1 / (0.1 + (t0 / _T1) ** 0.2)


# table C.2.2: φ0 for 40 % ≤ RH < 70 % and 70 % ≤ RH < 99 %; rows t0 = 3, 7, 14, 28, 60, 90 d; columns h = 100, 200,
# 300, ≥ 600 mm
_T0_TAB = [3, 7, 14, 28, 60, 90]
_H_TAB = [100, 200, 300, 600]
_PHI0_TAB = {
    "dry": [[3.90, 3.50, 3.31, 3.03], [3.33, 3.00, 2.82, 2.59], [2.92, 2.62, 2.48, 2.27],
            [2.56, 2.30, 2.17, 1.99], [2.21, 1.99, 1.88, 1.72], [2.05, 1.84, 1.74, 1.59]],
    "humid": [[2.83, 2.65, 2.56, 2.44], [2.41, 2.26, 2.19, 2.08], [2.12, 1.99, 1.92, 1.83],
              [1.86, 1.74, 1.69, 1.60], [1.61, 1.51, 1.46, 1.39], [1.49, 1.39, 1.35, 1.28]],
}
# table C.2.4: fly-ash correction φ(α, t0); rows t0 = 7, 14, 28, 60, 90; columns α = 10, 20, 30 %
_FA_T0 = [7, 14, 28, 60, 90]
_FA_A = [10, 20, 30]
_FA_TAB = [[0.80, 0.65, 0.53], [0.70, 0.55, 0.45], [0.64, 0.50, 0.41], [0.60, 0.47, 0.38], [0.58, 0.46, 0.37]]


def _interp2(xs, ys, tab, x, y):
    """bilinear interpolation in a table (rows ys, columns xs), clamped at the edges"""
    tab = np.asarray(tab, float)
    col = np.array([np.interp(x, xs, row) for row in tab])
    return float(np.interp(y, ys, col))


def nominal_creep(c: Concrete, RH: float, h: float, t0: float, method: str = "table") -> float:
    """φ0. "formula": φRH β(fcm) β(t0), φRH = 1 + (1 − RH/RH0)/(0.46 (h/h0)^(1/3)), β(fcm) = 5.3/(fcm/fcm0)^0.5
    (C.2.1-2…-5), for C25–C50. "table": table C.2.2 (dry 40–70 %, humid 70–99 %), linear between the tabulated h and
    t0 (note 4), h ≥ 600 mm as 600; t0 beyond 90 d: φ0(90)·β(t0)/β(90) (C.2.3 note); × √(32.4/fck) from C50 up
    (note 3). Above C50 the formula's fcm is not defined, so the table is used."""
    _rh_check(RH)
    if method == "formula" and c.grade <= 50:
        phi_rh = 1 + (1 - RH / _RH0) / (0.46 * (h / _H0) ** (1 / 3))
        return phi_rh * 5.3 / math.sqrt(_fcm(c) / _FCM0) * beta_t0(t0)
    tab = _PHI0_TAB["dry" if RH < 70 else "humid"]
    if t0 < 3:
        raise ValueError("table C.2.2 starts at 3 days")
    if h < 100:
        raise ValueError("table C.2.2 starts at h = 100 mm; use method='formula' (C25–C50)")
    if t0 > 90:
        v = _interp2(_H_TAB, _T0_TAB, tab, min(h, 600), 90) * beta_t0(t0) / beta_t0(90)
    else:
        v = _interp2(_H_TAB, _T0_TAB, tab, min(h, 600), t0)
    return v * (math.sqrt(32.4 / c.fck) if c.grade >= 50 else 1.0)


def creep_coefficient(c: Concrete, RH: float, h: float, t: float, t0: float, method: str = "table",
                      fly_ash: float | None = None) -> float:
    """φ(t, t0) = φ0 βc(t − t0) (C.2.1-1): loading age t0, age t (days), notional size h = 2A/u (mm). With the table
    method, βH uses RH = 55 % or 80 % for the two humidity bands (C.2.3). fly_ash: fly-ash content α (%, by mass)
    for the correction φ(α, t0) of table C.2.4 (C.2.4)"""
    if t <= t0:
        raise ValueError("need t > t0")
    rh_b = RH if method == "formula" and c.grade <= 50 else (55.0 if RH < 70 else 80.0)
    phi = nominal_creep(c, RH, h, t0, method) * beta_c(t - t0, rh_b, h)
    if fly_ash:
        if not (10 <= fly_ash <= 30 and 7 <= t0 <= 90):
            raise ValueError("table C.2.4 covers fly-ash contents 10–30 % and loading ages 7–90 d")
        phi *= _interp2(_FA_A, _FA_T0, _FA_TAB, fly_ash, t0)
    return phi


# ── 6.2.7 shrinkage and creep loss ──────────────────────────────────────────
def steel_ratios(A: float, I: float, Ap: float, ep: float, As: float = 0.0, es: float = 0.0):
    """(ρ, ρps) of one zone (6.2.7-3…-5): ρ = (Ap + As)/A, eps = (Ap ep + As es)/(Ap + As), ρps = 1 + eps²/i², i² = I/A;
    A, I of the transformed section (pretensioned) or the net section (post-tensioned); ep, es distances of the steel
    centroids from the section centroid (mm)"""
    eps = (Ap * ep + As * es) / (Ap + As)
    return (Ap + As) / A, 1 + eps * eps / (I / A)


def creep_shrinkage_loss(Ep: float, eps_cs: float, alpha_EP: float, sigma_pc: float, phi: float, rho: float,
                         rho_ps: float, fcu_transfer: float | None = None) -> float:
    """σl6(t) = 0.9 [Ep εcs(t,t0) + αEP σpc φ(t,t0)] / (1 + 15 ρ ρps) (6.2.7-1, -2). σpc: concrete stress at the steel
    centroid of the zone from the prestress with the first-batch losses only (σl6 of the bars taken as 0), self-weight
    may be included; σpc must not exceed 0.5 f'cu (fcu_transfer: cube strength at transfer). The code sets a tensile
    σ'pc to 0; the program does the same for σpc of either zone (program convention: one function for both zones)."""
    if fcu_transfer is not None and sigma_pc > 0.5 * fcu_transfer:
        raise ValueError(f"σpc = {sigma_pc:.1f} MPa > 0.5 f'cu = {0.5 * fcu_transfer:.1f} MPa (6.2.7)")
    sigma_pc = max(sigma_pc, 0.0)
    return 0.9 * (Ep * eps_cs + alpha_EP * sigma_pc * phi) / (1 + 15 * rho * rho_ps)


# ── 6.2.8 loss stages, 6.1.6 effective prestress ────────────────────────────
def loss_stages(*, post: bool, l1: float = 0.0, l2: float = 0.0, l3: float = 0.0, l4: float = 0.0, l5: float = 0.0,
                l6: float = 0.0) -> tuple[float, float]:
    """(σlI, σlII) of table 6.2.8. Pretensioned: σlI = σl2 + σl3 + σl4 + 0.5 σl5, σlII = 0.5 σl5 + σl6. Post-tensioned
    (internal, and both tendon types of mixed internal / external members): σlI = σl1 + σl2 + σl4, σlII = σl5 + σl6."""
    if post:
        return l1 + l2 + l4, l5 + l6
    return l2 + l3 + l4 + 0.5 * l5, 0.5 * l5 + l6


def sigma_pe(sigma_con: float, sigma_l: float) -> float:
    """effective prestress σpe = σcon − σl (6.1.6-3, -6, -8)"""
    return sigma_con - sigma_l


def sigma_p0(sigma_con: float, sigma_l: float, *, post: bool, sigma_l4: float = 0.0, alpha_EP: float = 0.0,
             sigma_pc: float = 0.0) -> float:
    """tendon stress when the concrete stress at the tendon is zero (6.1.6): pretensioned σp0 = σcon − σl + σl4
    (6.1.6-2); post-tensioned σp0 = σcon − σl + αEP σpc (6.1.6-5), σpc the concrete stress at the tendon centroid from
    Np (compression +, tension −)"""
    if post:
        return sigma_con - sigma_l + alpha_EP * sigma_pc
    return sigma_con - sigma_l + sigma_l4


def concrete_stress(N: float, e: float, y: float, A: float, I: float, M: float = 0.0) -> float:
    """σ = N/A + N e y / I − M y / I (6.1.6-1, -4; compression +): normal stress at a fibre y from the centroid, from the
    prestress resultant N (N) at eccentricity e and a moment M (N·mm, e.g. the secondary moment Mp2). e and y are
    measured downwards from the centroid (positive below); M is sagging-positive, so it compresses fibres above"""
    return N / A + N * e * y / I - M * y / I
