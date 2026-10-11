"""Members of JTG 3362-2018 beyond plain bending and uniaxial compression: tension members (5.4), spiral columns
(5.3.2), sections with bars spread over the web (5.3.7), biaxial eccentric compression (5.3.11), effective lengths
(appendix E), the circular-section table of appendix F, and shear of circular sections (not in 3362: GB 50010-2010
6.3.15, program convention).

Units as in femotion.codes.jtg.rc: mm, mm², MPa; forces kN, moments kN·m. Clauses: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    l0 = jtg.effective_length(9000, K_F=2.0e4, K_theta=8.0e12, EI=jtg.circle(1500, 28, 28, 60).I * 3.15e4)  # E.0.2
    col = jtg.circle(1500, 28, 28, 60, concrete="C35")
    jtg.eccentric(col, 9800, 3100, l0, method="table")                  # appendix F
    jtg.axial_spiral(col, 30000, 6000, spiral=(12, 60))                  # 5.3.2: spiral φ12 at 60 mm
    jtg.shear_circle(col, 900, (2, 12, 100), pile=True)                  # one hoop or spiral = 2 legs
    tie = jtg.rect(400, 800, top=[(4, 22, 50)], bottom=[(4, 22, 50)])
    jtg.tension(tie, 600, 120)                                           # 5.4.2
    sx, sy = jtg.rect_biaxial(800, 1200, layout=(5, 7, 28, 60), concrete="C40")
    jtg.biaxial(sx, sy, 8000, 1500, 900, l0x=12000, l0y=12000)           # 5.3.11
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .bearing import k_indirect
from .materials import Rebar, rebar as _rebar
from .rc import Check, RCSection, Rect, _circle_code, _circle_NM, _mats, _reverse_failure, axial, eccentric, flexure


# ── appendix E: effective length ────────────────────────────────────────────
LENGTH_FACTORS = {"fixed-fixed": 0.5, "fixed-pinned": 0.7, "pinned-pinned": 1.0, "fixed-free": 2.0}  # E.0.1


def length_factor(*, ends: str | None = None, k_theta: float | None = None, k_F: float | None = None) -> float:
    """effective-length factor k of appendix E. ends: one of LENGTH_FACTORS (E.0.1: both ends fixed 0.5, fixed and
    pinned 0.7, both pinned 1.0, fixed and free 2.0). Otherwise a column fixed at the base whose top is held by springs,
    given by the relative stiffnesses kθ = Kθ l / EI and kF = KF l³ / EI: E.0.2-1 with both, E.0.3 when there is no
    rotational spring (k_theta None or 0). math.inf is accepted for a rigid restraint."""
    if ends is not None:
        if ends not in LENGTH_FACTORS:
            raise ValueError(f"ends {ends!r}: one of {', '.join(LENGTH_FACTORS)}")
        return LENGTH_FACTORS[ends]
    if k_F is None or k_F < 0 or (k_theta is not None and k_theta < 0):
        raise ValueError("give ends=…, or the relative spring stiffnesses k_F (and k_theta), ≥ 0")
    if not k_theta:  # E.0.3: lateral spring only
        if math.isinf(k_F):
            return 0.7
        p = k_F ** 1.5
        return 2 - 1.3 * p / (9.5 + p)
    return 0.5 * math.exp(0.35 / (1 + 0.6 * k_theta) + 0.7 / (1 + 0.01 * k_F ** 2)
                          + 0.35 / ((1 + 0.75 * k_theta) * (1 + 1.15 * k_F)))  # E.0.2-1


def effective_length(l: float, *, ends: str | None = None, k: float | None = None, K_theta: float | None = None,
                     K_F: float | None = None, EI: float | None = None) -> float:
    """computed length l0 = k l (E.0.1) of a compression member of length l between supports (mm). k given, or from
    ends (E.0.1), or from the springs at the top of a column fixed at the base: K_F lateral stiffness (N/mm), K_theta
    rotational stiffness (N·mm/rad, None for E.0.3), EI the member's flexural stiffness (N·mm²); kθ = Kθ l / EI (E.0.2-2),
    kF = KF l³ / EI (E.0.2-3). Pass the result as l0 to axial, eccentric, axial_spiral, biaxial."""
    if k is None:
        if ends is None:
            if K_F is None or EI is None:
                raise ValueError("give k, ends, or K_F (and K_theta) with EI")
            kt = None if K_theta is None else K_theta * l / EI
            k = length_factor(k_theta=kt, k_F=K_F * l ** 3 / EI)
        else:
            k = length_factor(ends=ends)
    return k * l


# ── 5.3.2: spiral (or welded hoop) columns ──────────────────────────────────
def _d_long(sec: RCSection, default: float) -> float:
    """the largest longitudinal bar diameter, or default when the section does not know it"""
    return float(np.nanmax(sec.bar_d)) if np.isfinite(sec.bar_d).any() else default


def _i_min(sec: RCSection, b_min, i_min):
    if i_min is not None:
        return i_min
    if b_min is not None:
        return b_min / math.sqrt(12)
    if sec.kind in ("circle", "annulus"):
        return sec.i
    if sec.kind == "rect":
        return min(sec.pieces[0].b, sec.h) / math.sqrt(12)
    raise ValueError("5.3.2 needs the least radius of gyration: pass i_min (mm) for this section")


def axial_spiral(sec: RCSection, Nd: float, l0: float | None = None, *, spiral, dcor: float | None = None,
                 gamma0: float = 1.0, spiral_steel: str | Rebar | None = None, b_min: float | None = None,
                 i_min: float | None = None) -> Check:
    """axially loaded column with a spiral or welded hoops as indirect reinforcement (5.3.2):
    γ0 Nd ≤ 0.9 (fcd Acor + f'sd A's + k fsd Aso), Aso = π dcor Aso1 / s, k as in 5.7 (bearing.k_indirect).
    spiral: (bar diameter mm, pitch s mm); dcor: core diameter inside the spiral (default for jtg.circle sections: the
    bar circle plus one longitudinal bar diameter, the spiral resting on the bars). The spiral counts only when
    l0/i ≤ 48, Aso ≥ 0.25 A's and s ≤ 80 mm and ≤ dcor/5, and only when 5.3.2-1 gives more than 5.3.1; then at most
    1.5 × 5.3.1. Otherwise the result is that of 5.3.1 (jtg.axial). With the spiral counted, the detailing of 9.6.2:
    longitudinal bars ≥ 0.5 % of Acor, Acor ≥ 2/3 A, s ≥ 40 mm, spiral diameter ≥ 8 mm and ≥ 1/4 of the longitudinal
    bars (besides 9.1.12 as for any compression member). Reinforced concrete, solid sections only: 5.3.2 and figure
    5.3.2 are about a solid core confined by the spiral; a hollow section has no confined core inside (its inner face is
    free), so hollow circles and other sections with holes raise."""
    if sec.prestressed:
        raise NotImplementedError("5.3.2 covers reinforced-concrete members")
    if sec.kind in ("annulus", "box") or any(getattr(p, "sign", 1) < 0 for p in sec.pieces):
        raise NotImplementedError("5.3.2 is for solid sections: a hollow section has no confined core (use jtg.axial)")
    c, st = sec.concrete, sec.steel
    sst = st if spiral_steel is None else (spiral_steel if isinstance(spiral_steel, Rebar) else _rebar(spiral_steel))
    ds, s = spiral
    if dcor is None:
        if sec.ring is None:
            raise ValueError("give dcor (core diameter inside the spiral) for this section")
        dcor = 2 * sec.ring + _d_long(sec, 0.0)
    dcor = float(dcor)
    Acor = math.pi * dcor ** 2 / 4
    Ass1 = math.pi * ds ** 2 / 4
    Aso = math.pi * dcor * Ass1 / s
    As = float(sec.bars[:, 0].sum())
    k = k_indirect(c)
    base = axial(sec, Nd, l0, gamma0=gamma0, b_min=b_min, i_min=i_min)  # 5.3.1, with its minimum steel (9.1.12)
    N531 = base.resistance
    N532 = 0.9 * (c.fcd * Acor + st.fsd_c * As + k * sst.fsd * Aso) / 1e3
    i = _i_min(sec, b_min, i_min)
    lam = (l0 or 0.0) / i
    d = {"Acor": Acor, "dcor": dcor, "Aso": Aso, "k": k, "l0_over_i": lam, "N_5_3_1": N531, "N_5_3_2": N532,
         "phi": base.details["phi"]}
    chk = Check("axial compression (spiral column)", gamma0 * Nd, N531, "kN", "JTG 3362-2018 5.3.1, 5.3.2", d)
    why = []
    if lam > 48:
        why.append(f"l0/i = {lam:.1f} > 48")
    if Aso < 0.25 * As:
        why.append(f"Aso = {Aso:.0f} mm² < 0.25 A's = {0.25 * As:.0f} mm²")
    if s > 80 or s > dcor / 5:
        why.append(f"pitch {s:g} mm > min(80 mm, dcor/5 = {dcor / 5:.0f} mm)")
    if why:
        chk.notes.append("spiral not counted (" + "; ".join(why) + "): 5.3.1 governs (5.3.2)")
        chk.failed += base.failed
        chk.warnings += base.warnings
        return chk
    if N532 < N531:
        chk.notes.append("5.3.2-1 below 5.3.1: the spiral's confinement is not counted (5.3.2)")
        chk.failed += base.failed
        chk.warnings += base.warnings
        return chk
    chk.resistance = min(N532, 1.5 * N531)
    if N532 > 1.5 * N531:
        chk.notes.append(f"5.3.2-1 = {N532:.0f} kN limited to 1.5 × 5.3.1 = {1.5 * N531:.0f} kN (5.3.2)")
    chk.clause = "JTG 3362-2018 5.3.2, 9.6.2"
    chk.failed += base.failed
    chk.warnings += base.warnings  # 9.1.12 and 3.1.2 apply as well
    chk.warnings += spiral_rules(As, Acor, sec.area, ds, s, dcor, _d_long(sec, 0.0))
    return chk


def spiral_rules(As: float, Acor: float, A: float, ds: float, s: float, dcor: float, d_long: float) -> list[str]:
    """9.6.2 items 1, 2, 4 (the detailing of spiral columns), the unmet ones: longitudinal bars As ≥ 0.5 % of the core
    Acor, Acor ≥ 2/3 of the section A; pitch s ≤ dcor/5 and ≤ 80 mm, ≥ 40 mm; spiral diameter ds ≥ 8 mm and ≥ 1/4 of the
    longitudinal bars (d_long, the largest; 0 when unknown). Item 3 (the bars' embedment): pier_detailing."""
    out = []
    if As < 0.005 * Acor:
        out.append(f"longitudinal bars {As / Acor:.3%} of Acor < 0.5 % (9.6.2)")
    if Acor < 2 / 3 * A:
        out.append(f"core area {Acor / A:.1%} of the section < 2/3 (9.6.2)")
    if s > min(80.0, dcor / 5):
        out.append(f"pitch {s:g} mm > min(80 mm, dcor/5 = {dcor / 5:.0f} mm) (9.6.2)")
    if s < 40:
        out.append(f"pitch {s:g} mm < 40 mm (9.6.2)")
    if ds < max(8.0, d_long / 4):
        out.append(f"spiral diameter {ds:g} mm < max(8 mm, d/4 = {d_long / 4:.1f} mm) (9.6.2)")
    return out


# ── 5.3.7: bars spread evenly over the web ──────────────────────────────────
def _eccentric_web(s: RCSection, chk: Check, e: float, e0_raw: float) -> Check:
    """5.3.7 for eccentric(method="web"): s already turned so that the top is the more compressed face, chk carries e0,
    η, the minimum reinforcement. As, A's: the edge rows (RCSection.edge_masks); Asw: all other bars, at least 4 rows
    (program convention: "at least 4 bars in each row" read as at least 4 bars down the web on each side, i.e. 4
    distinct depths)."""
    if s.prestressed:
        raise NotImplementedError("5.3.7 covers reinforced-concrete members")
    if s.kind in ("circle", "annulus"):
        raise ValueError("method='web' (5.3.7) is for rectangles, T, I and box sections")
    c, st = s.concrete, s.steel
    bot, top = s.edge_masks()
    web = ~(bot | top)
    rows = np.unique(np.round(s.bars[web, 1], 6))
    if not bot.any() or not top.any() or len(rows) < 4:
        raise ValueError("method='web' (5.3.7) needs As, A's and bars over the web in at least 4 rows "
                         f"(found {len(rows)} web rows)")
    As, Asc, Asw = s.bars[bot, 0].sum(), s.bars[top, 0].sum(), s.bars[web, 0].sum()
    h0 = s.h0()
    asc, _ = s.comp_ref()
    hsw = h0 - asc
    om = hsw / h0
    fsw = min(st.fsd, st.fsd_c)  # program convention: one value for tension and compression, the smaller (HRB500: 400 MPa)
    beta, xb = c.beta, s.xi_b(eccentric=True)
    gaps = np.diff(rows)
    if len(gaps) > 1 and gaps.max() > 1.5 * gaps.min():
        chk.notes.append("web bars not at equal spacing: 5.3.7 assumes them spread evenly over hsw")
    g = hsw / (len(rows) + 1)  # spacing of the same number of rows spread evenly between A's and As
    if rows[-1] - rows[0] < 0.75 * (len(rows) - 1) * g or abs(rows.mean() - (asc + h0) / 2) > 0.1 * hsw:
        chk.notes.append(f"web bars over {rows[0]:.0f}–{rows[-1]:.0f} mm only: 5.3.7 spreads them over hsw = "
                         f"{asc:.0f}–{h0:.0f} mm")
    if Asw > As + Asc:
        chk.notes.append(f"Asw = {Asw / (As + Asc):.1f} (As + A's): with this much web steel 5.3.7 can exceed the "
                         "bar-by-bar rule set (method='strain') by several per cent (up to about 7 %)")

    def NM(x):
        """(N, moment about As) of 5.3.7-1, -2 for a block depth x (N, N·mm)"""
        xi = x / h0
        a, q = s.part(min(x, s.h))  # ξ ≤ h/h0 for the concrete (5.3.7 note)
        sig = st.fsd if xi <= xb else float(np.clip(c.eps_cu * st.Es * (beta / xi - 1), -st.fsd_c, st.fsd))
        t = max((min(xi, s.h / h0) - beta) / (beta * om), -1.0)  # −1: all web bars in tension (program convention, x < β a's)
        if t > 0:  # ξ > β
            Nsw, Msw = fsw * Asw, 0.5 * fsw * Asw * hsw
        else:
            Nsw, Msw = (1 + 2 * t) * fsw * Asw, (0.5 - t * t) * fsw * Asw * hsw  # 5.3.7-3, -4
        N = c.fcd * a + st.fsd_c * Asc - sig * As + Nsw
        M = c.fcd * (a * h0 - q) + st.fsd_c * Asc * (h0 - asc) + Msw
        return N, M, Nsw, Msw

    z_load = s.zg - e
    e_s = h0 - z_load  # load to As (5.3.4-3 with the gross centroid, §5.9)
    xs = np.geomspace(1e-4 * s.h, 50 * s.h, 800)
    F = np.array([NM(x)[:2] for x in xs])
    f = F[:, 1] - e_s * F[:, 0]
    roots = [k for k in range(len(xs) - 1) if f[k] * f[k + 1] <= 0 and F[k + 1, 0] > 0]
    if not roots:
        raise RuntimeError("5.3.7: no equilibrium at this eccentricity (load beyond the plastic centroid?): use "
                           "method='strain'")
    k = roots[0]
    x = brentq(lambda x: NM(x)[1] - e_s * NM(x)[0], xs[k], xs[k + 1], xtol=1e-9 * s.h)
    Nu, _, Nsw, Msw = NM(x)
    small = x > xb * h0
    chk.clause = "JTG 3362-2018 5.3.7, 5.3.9"
    chk.details.update(x=x, xi=x / h0, xi_b=xb, h0=h0, Asw=Asw, hsw=hsw, omega=om, fsw=fsw, N_sw=Nsw / 1e3,
                       M_sw=Msw / 1e6, case="small eccentricity" if small else "large eccentricity")
    if not small and x < 2 * asc and asc - z_load > 0:  # 5.3.7-5, -6: moments about A's, concrete ignored
        e_c = asc - z_load
        Nu = (st.fsd * As * (h0 - asc) + 0.5 * fsw * Asw * hsw) / e_c
        chk.notes.append(f"x < 2a's: moments about A's (e' = {e_c:.0f} mm, 5.3.7-5)")
    elif small:
        rev = _reverse_failure(s, e0_raw)  # program convention: 5.3.4-4 / 5.3.5-3, -4 without the web bars (safe side)
        if rev is not None:
            chk.details["N_reverse"] = rev[0] / 1e3
            if rev[0] < Nu:
                Nu = rev[0]
                chk.notes.append(f"reverse failure governs (e' = {rev[1]:.0f} mm, 5.3.4-4 without the web bars)")
    chk.resistance = Nu / 1e3
    return chk


# ── appendix F: table F.0.1 ─────────────────────────────────────────────────
F01_E = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                  0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 3.50,
                  4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00])  # η e0 / r
F01_RHO = np.array([0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10,
                    1.20])  # ρ fsd / fcd
# table F.0.1 (p134–136): relative compressive resistance nu = Nu / (A fcd); rows η e0 / r, columns ρ fsd / fcd
TABLE_F01 = np.array([
    (1.0487, 1.0783, 1.1079, 1.1375, 1.1671, 1.1968, 1.2264, 1.2561, 1.2857,  # 0.01
     1.3846, 1.4835, 1.5824, 1.6813, 1.7802, 1.8791, 1.9780, 2.0769, 2.1758),
    (1.0031, 1.0316, 1.0601, 1.0885, 1.1169, 1.1454, 1.1738, 1.2022, 1.2306,  # 0.05
     1.3254, 1.4201, 1.5148, 1.6095, 1.7042, 1.7989, 1.8937, 1.9884, 2.0831),
    (0.9438, 0.9711, 0.9984, 1.0257, 1.0529, 1.0802, 1.1074, 1.1345, 1.1617,  # 0.10
     1.2521, 1.3423, 1.4325, 1.5226, 1.6127, 1.7027, 1.7927, 1.8826, 1.9726),
    (0.8827, 0.9090, 0.9352, 0.9614, 0.9875, 1.0136, 1.0396, 1.0656, 1.0916,  # 0.15
     1.1781, 1.2643, 1.3503, 1.4362, 1.5220, 1.6077, 1.6934, 1.7790, 1.8646),
    (0.8206, 0.8458, 0.8709, 0.8960, 0.9210, 0.9460, 0.9709, 0.9958, 1.0206,  # 0.20
     1.1033, 1.1856, 1.2677, 1.3496, 1.4313, 1.5130, 1.5945, 1.6760, 1.7574),
    (0.7589, 0.7829, 0.8067, 0.8302, 0.8540, 0.8778, 0.9016, 0.9254, 0.9491,  # 0.25
     1.0279, 1.1063, 1.1845, 1.2625, 1.3404, 1.4180, 1.4956, 1.5731, 1.6504),
    (0.7003, 0.7247, 0.7486, 0.7721, 0.7953, 0.8181, 0.8408, 0.8632, 0.8855,  # 0.30
     0.9590, 1.0316, 1.1036, 1.1752, 1.2491, 1.3228, 1.3964, 1.4699, 1.5433),
    (0.6432, 0.6684, 0.6928, 0.7165, 0.7397, 0.7625, 0.7849, 0.8070, 0.8290,  # 0.35
     0.9008, 0.9712, 1.0408, 1.1097, 1.1783, 1.2465, 1.3145, 1.3824, 1.4500),
    (0.5878, 0.6142, 0.6393, 0.6635, 0.6869, 0.7097, 0.7320, 0.7540, 0.7757,  # 0.40
     0.8461, 0.9147, 0.9822, 1.0489, 1.1150, 1.1807, 1.2461, 1.3113, 1.3762),
    (0.5346, 0.5624, 0.5884, 0.6132, 0.6369, 0.6599, 0.6822, 0.7041, 0.7255,  # 0.45
     0.7949, 0.8619, 0.9275, 0.9921, 1.0561, 1.1195, 1.1825, 1.2452, 1.3077),
    (0.4839, 0.5133, 0.5403, 0.5657, 0.5898, 0.6130, 0.6354, 0.6573, 0.6786,  # 0.50
     0.7470, 0.8126, 0.8765, 0.9393, 1.0012, 1.0625, 1.1233, 1.1838, 1.2441),
    (0.4359, 0.4670, 0.4951, 0.5212, 0.5458, 0.5692, 0.5917, 0.6135, 0.6347,  # 0.55
     0.7022, 0.7666, 0.8289, 0.8899, 0.9500, 1.0094, 1.0682, 1.1266, 1.1848),
    (0.3910, 0.4238, 0.4530, 0.4798, 0.5047, 0.5283, 0.5509, 0.5727, 0.5938,  # 0.60
     0.6605, 0.7237, 0.7846, 0.8440, 0.9023, 0.9598, 1.0168, 1.0733, 1.1295),
    (0.3495, 0.3840, 0.4141, 0.4414, 0.4667, 0.4905, 0.5131, 0.5348, 0.5558,  # 0.65
     0.6217, 0.6837, 0.7432, 0.8011, 0.8578, 0.9136, 0.9689, 1.0236, 1.0779),
    (0.3116, 0.3475, 0.3784, 0.4062, 0.4317, 0.4556, 0.4782, 0.4998, 0.5206,  # 0.70
     0.5857, 0.6466, 0.7047, 0.7611, 0.8163, 0.8705, 0.9241, 0.9771, 1.0297),
    (0.2773, 0.3143, 0.3459, 0.3739, 0.3996, 0.4235, 0.4460, 0.4674, 0.4881,  # 0.75
     0.5523, 0.6120, 0.6689, 0.7239, 0.7776, 0.8303, 0.8823, 0.9337, 0.9847),
    (0.2468, 0.2845, 0.3164, 0.3446, 0.3702, 0.3940, 0.4164, 0.4377, 0.4581,  # 0.80
     0.5214, 0.5799, 0.6356, 0.6892, 0.7415, 0.7927, 0.8432, 0.8931, 0.9426),
    (0.2199, 0.2579, 0.2899, 0.3180, 0.3436, 0.3672, 0.3893, 0.4104, 0.4305,  # 0.85
     0.4928, 0.5502, 0.6045, 0.6569, 0.7078, 0.7577, 0.8067, 0.8552, 0.9032),
    (0.1963, 0.2343, 0.2661, 0.2940, 0.3193, 0.3427, 0.3646, 0.3853, 0.4051,  # 0.90
     0.4663, 0.5225, 0.5757, 0.6267, 0.6763, 0.7249, 0.7726, 0.8197, 0.8663),
    (0.1759, 0.2134, 0.2448, 0.2724, 0.2974, 0.3204, 0.3420, 0.3624, 0.3818,  # 0.95
     0.4419, 0.4969, 0.5488, 0.5986, 0.6470, 0.6942, 0.7406, 0.7864, 0.8317),
    (0.1582, 0.1950, 0.2259, 0.2530, 0.2775, 0.3001, 0.3213, 0.3413, 0.3604,  # 1.00
     0.4193, 0.4731, 0.5238, 0.5724, 0.6195, 0.6655, 0.7107, 0.7553, 0.7993),
    (0.1299, 0.1646, 0.1939, 0.2198, 0.2433, 0.2649, 0.2852, 0.3044, 0.3227,  # 1.10
     0.3791, 0.4305, 0.4789, 0.5251, 0.5699, 0.6136, 0.6564, 0.6986, 0.7402),
    (0.1087, 0.1410, 0.1685, 0.1929, 0.2152, 0.2358, 0.2551, 0.2734, 0.2909,  # 1.20
     0.3446, 0.3937, 0.4398, 0.4838, 0.5264, 0.5679, 0.6086, 0.6486, 0.6881),
    (0.0927, 0.1224, 0.1481, 0.1710, 0.1920, 0.2115, 0.2299, 0.2472, 0.2639,  # 1.30
     0.3150, 0.3618, 0.4057, 0.4476, 0.4882, 0.5276, 0.5663, 0.6043, 0.6418),
    (0.0804, 0.1077, 0.1316, 0.1531, 0.1728, 0.1912, 0.2086, 0.2250, 0.2408,  # 1.40
     0.2895, 0.3340, 0.3759, 0.4158, 0.4544, 0.4920, 0.5288, 0.5649, 0.6006),
    (0.0708, 0.0959, 0.1180, 0.1381, 0.1567, 0.1741, 0.1905, 0.2061, 0.2210,  # 1.50
     0.2673, 0.3097, 0.3496, 0.3877, 0.4245, 0.4603, 0.4954, 0.5298, 0.5638),
    (0.0630, 0.0862, 0.1068, 0.1256, 0.1431, 0.1595, 0.1750, 0.1897, 0.2039,  # 1.60
     0.2479, 0.2884, 0.3264, 0.3628, 0.3979, 0.4321, 0.4655, 0.4984, 0.5309),
    (0.0567, 0.0782, 0.0974, 0.1150, 0.1315, 0.1469, 0.1616, 0.1756, 0.1891,  # 1.70
     0.2310, 0.2695, 0.3058, 0.3405, 0.3741, 0.4068, 0.4387, 0.4702, 0.5012),
    (0.0515, 0.0714, 0.0894, 0.1060, 0.1215, 0.1361, 0.1500, 0.1633, 0.1761,  # 1.80
     0.2160, 0.2528, 0.2875, 0.3207, 0.3528, 0.3840, 0.4146, 0.4447, 0.4743),
    (0.0472, 0.0657, 0.0826, 0.0982, 0.1128, 0.1266, 0.1398, 0.1525, 0.1646,  # 1.90
     0.2027, 0.2378, 0.2710, 0.3028, 0.3335, 0.3635, 0.3928, 0.4216, 0.4500),
    (0.0435, 0.0608, 0.0767, 0.0914, 0.1052, 0.1183, 0.1309, 0.1429, 0.1545,  # 2.00
     0.1908, 0.2244, 0.2562, 0.2867, 0.3162, 0.3449, 0.3730, 0.4007, 0.4279),
    (0.0311, 0.0441, 0.0562, 0.0676, 0.0784, 0.0888, 0.0987, 0.1083, 0.1176,  # 2.50
     0.1470, 0.1744, 0.2005, 0.2255, 0.2498, 0.2735, 0.2968, 0.3197, 0.3422),
    (0.0241, 0.0345, 0.0442, 0.0535, 0.0623, 0.0707, 0.0789, 0.0869, 0.0946,  # 3.00
     0.1191, 0.1421, 0.1640, 0.1852, 0.2057, 0.2258, 0.2456, 0.2650, 0.2841),
    (0.0197, 0.0283, 0.0364, 0.0441, 0.0516, 0.0587, 0.0657, 0.0724, 0.0790,  # 3.50
     0.0999, 0.1196, 0.1385, 0.1568, 0.1746, 0.1919, 0.2090, 0.2258, 0.2425),
    (0.0166, 0.0240, 0.0309, 0.0376, 0.0440, 0.0502, 0.0562, 0.0620, 0.0677,  # 4.00
     0.0859, 0.1032, 0.1198, 0.1358, 0.1514, 0.1667, 0.1818, 0.1966, 0.2112),
    (0.0144, 0.0208, 0.0269, 0.0327, 0.0383, 0.0437, 0.0490, 0.0542, 0.0592,  # 4.50
     0.0754, 0.0907, 0.1054, 0.1197, 0.1336, 0.1473, 0.1607, 0.1740, 0.1870),
    (0.0127, 0.0183, 0.0237, 0.0289, 0.0339, 0.0388, 0.0435, 0.0481, 0.0526,  # 5.00
     0.0671, 0.0809, 0.0941, 0.1070, 0.1195, 0.1319, 0.1440, 0.1559, 0.1677),
    (0.0113, 0.0164, 0.0213, 0.0259, 0.0304, 0.0348, 0.0391, 0.0433, 0.0474,  # 5.50
     0.0605, 0.0729, 0.0850, 0.0967, 0.1081, 0.1193, 0.1304, 0.1412, 0.1520),
    (0.0102, 0.0149, 0.0193, 0.0235, 0.0276, 0.0316, 0.0355, 0.0393, 0.0430,  # 6.00
     0.0550, 0.0664, 0.0775, 0.0882, 0.0987, 0.1089, 0.1191, 0.1291, 0.1390),
    (0.0093, 0.0136, 0.0176, 0.0215, 0.0252, 0.0289, 0.0325, 0.0360, 0.0394,  # 6.50
     0.0504, 0.0610, 0.0711, 0.0810, 0.0907, 0.1002, 0.1096, 0.1188, 0.1280),
    (0.0086, 0.0125, 0.0162, 0.0198, 0.0233, 0.0266, 0.0300, 0.0332, 0.0364,  # 7.00
     0.0466, 0.0563, 0.0658, 0.0750, 0.0840, 0.0928, 0.1015, 0.1101, 0.1186),
    (0.0080, 0.0116, 0.0150, 0.0183, 0.0216, 0.0247, 0.0278, 0.0308, 0.0338,  # 7.50
     0.0433, 0.0524, 0.0612, 0.0697, 0.0781, 0.0864, 0.0945, 0.1025, 0.1104),
    (0.0074, 0.0108, 0.0140, 0.0171, 0.0201, 0.0230, 0.0259, 0.0287, 0.0315,  # 8.00
     0.0404, 0.0489, 0.0572, 0.0652, 0.0730, 0.0808, 0.0884, 0.0959, 0.1034),
    (0.0069, 0.0101, 0.0131, 0.0160, 0.0188, 0.0216, 0.0243, 0.0269, 0.0295,  # 8.50
     0.0379, 0.0459, 0.0536, 0.0612, 0.0686, 0.0759, 0.0830, 0.0901, 0.0971),
    (0.0065, 0.0094, 0.0123, 0.0150, 0.0177, 0.0203, 0.0228, 0.0253, 0.0278,  # 9.00
     0.0356, 0.0432, 0.0505, 0.0577, 0.0646, 0.0715, 0.0783, 0.0850, 0.0916),
    (0.0061, 0.0089, 0.0116, 0.0142, 0.0167, 0.0191, 0.0215, 0.0239, 0.0262,  # 9.50
     0.0337, 0.0408, 0.0477, 0.0545, 0.0611, 0.0676, 0.0740, 0.0804, 0.0867),
    (0.0058, 0.0084, 0.0110, 0.0134, 0.0158, 0.0181, 0.0204, 0.0226, 0.0248,  # 10.00
     0.0319, 0.0387, 0.0453, 0.0517, 0.0580, 0.0641, 0.0702, 0.0763, 0.0822),
])


def circle_nu(eta_e0_r: float, rho_fsd_fcd: float) -> float:
    """nu of table F.0.1 (appendix F) between the tabulated values (program convention, the code gives no rule):
    1/nu linear in η e0/r, then nu linear in ρ fsd/fcd. nu falls with η e0/r along a convex curve, so a straight line
    between two rows lies above it (up to +2.7 % in the wide rows 2.0–2.5); 1/nu is nearly linear in η e0/r and stays
    within the table's own rounding (−0.8 % … +0.5 % against the 5.3.8 formula with rs/r = 0.9; the +0.5 % where
    nu < 0.01, as at the rows themselves). Outside the table (η e0/r 0.01–10, ρ fsd/fcd 0.06–1.20) a ValueError."""
    if not (F01_E[0] <= eta_e0_r <= F01_E[-1] and F01_RHO[0] <= rho_fsd_fcd <= F01_RHO[-1]):
        raise ValueError(f"η e0/r = {eta_e0_r:.3f}, ρ fsd/fcd = {rho_fsd_fcd:.3f}: outside table F.0.1 "
                         f"({F01_E[0]}–{F01_E[-1]}, {F01_RHO[0]}–{F01_RHO[-1]})")
    i = min(int(np.searchsorted(F01_E, eta_e0_r, side="right")) - 1, len(F01_E) - 2)
    t = (eta_e0_r - F01_E[i]) / (F01_E[i + 1] - F01_E[i])
    row = 1 / ((1 - t) / TABLE_F01[i] + t / TABLE_F01[i + 1])
    return float(np.interp(rho_fsd_fcd, F01_RHO, row))


def _eccentric_table(s: RCSection, chk: Check, e: float) -> Check:
    """appendix F for eccentric(method="table"): γ0 Nd ≤ nu A fcd (F.0.1); solid circle, C30–C50, ρ 0.5 %–4 %, at least
    8 bars (5.3.8 note). The table was computed with rs/r = 0.9 (commentary p259): with the bars nearer the centre
    (rs/r < 0.9) it overestimates, and a note gives the 5.3.8 formula's shortfall for this section."""
    if s.kind != "circle" or len(s.bars) < 8:
        raise ValueError("method='table' (appendix F) is for solid circular sections with at least 8 bars")
    c, st = s.concrete, s.steel
    if not 30 <= c.grade <= 50:
        raise ValueError(f"appendix F covers C30–C50 (C{c.grade})")
    r = s.h / 2
    A = math.pi * r * r
    rho = s.bars[:, 0].sum() / A
    if not 0.005 <= rho <= 0.04:
        raise ValueError(f"appendix F covers ρ = 0.5 %–4 % (ρ = {rho:.2%})")
    lam = rho * st.fsd / c.fcd
    nu = circle_nu(e / r, lam)
    chk.clause = "JTG 3362-2018 appendix F, 5.3.9"
    chk.details.update(nu=nu, eta_e0_r=e / r, rho_fsd_fcd=lam, rs_over_r=s.ring / r)
    if s.ring / r < 0.9 - 5e-3:  # the table (rs/r = 0.9) then overestimates: say by how much
        N_own, N_09 = _circle_code(s, e)[0], _circle_code(s.replace(ring=0.9 * r), e)[0]
        chk.details["formula_own_rs_over_rs_0_9"] = N_own / N_09
        chk.notes.append(f"table F.0.1 assumes rs/r = 0.9 (commentary); with rs/r = {s.ring / r:.3f} the 5.3.8 formula "
                         f"gives {1 - N_own / N_09:.1%} less: method='code' is the closer check")
    chk.resistance = nu * A * c.fcd / 1e3
    return chk


# ── shear of circular sections: GB 50010-2010 6.3.15 with the formulas of 3362 ──
def shear_circle(sec: RCSection, Vd: float, stirrups, *, gamma0: float = 1.0, alpha1: float = 1.0,
                 stirrup_steel: str | Rebar | None = None, bent=(), pile: bool = False, Nd: float = 0.0) -> Check:
    """shear of a solid circular pier, column or pile. 3362 gives no rule for circles (5.2.9 is for rectangles, T and I
    sections); program convention: GB 50010-2010 6.3.15, b = 1.76 r and h0 = 1.6 r in the formulas for rectangles, here
    those of 3362: 5.2.9 (α2 = 1, α3 = 1), the upper limit 5.2.11 and the threshold 5.2.12. The longitudinal tension
    steel for P (5.2.9) is the bars in the half of the circle beyond the centroid. stirrups: (legs, diameter mm, spacing
    mm), one circular hoop or spiral counting as 2 legs (GB 6.3.15 reads the stirrup area of the equivalent rectangle
    as that of the hoops; ACI 318 likewise takes Av = 2 × the bar area of a hoop or spiral with d = 0.8 D), cross-ties
    adding legs. Nd: axial force acting with Vd (kN, compression +, tension −). Tension, as in GB 50010 6.3.14 (the
    clause 6.3.15 sends tension members to): Vcs − 0.2 γ0 |Nd|, but not below the stirrups alone fsv Asv h0 / s, which
    must be at least 0.36 ftd b h0. Compression is not counted (5.2.9 has none; safe side).
    Detailing of the hoops: columns (pile=False) 9.6.1: diameter ≥ 8 mm and ≥ d/4, spacing ≤ 15 d, 0.8 D and 400 mm,
    with more than 3 % longitudinal steel ≤ 10 d and 200 mm (d: the largest longitudinal bar for the diameter, the
    smallest for the spacing, as GB 50010-2010 9.3.2 reads it; program convention, piles alike); bored piles (pile=True, 9.6.1 excludes piles): JTG
    3363-2019 6.2.2 item 3 4): diameter ≥ 8 mm and ≥ d/4, spacing ≤ 15 d and 300 mm. The beam rules of 9.3.12 are not
    applied. Hollow circles: no code basis, not implemented."""
    if sec.kind == "annulus":
        raise NotImplementedError("shear of hollow circular sections: neither 3362 nor GB 50010 6.3.15 covers them")
    if sec.kind != "circle":
        raise ValueError("shear_circle is for jtg.circle sections")
    if stirrups is None:
        raise ValueError("circular members need hoops or a spiral: stirrups=(legs, diameter, spacing)")
    if bent:
        raise ValueError("bent-up bars do not apply to circular sections")
    c, st = sec.concrete, sec.steel
    sst = st if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
    r = sec.h / 2
    b, h0 = 1.76 * r, 1.6 * r
    As = float(sec.bars[sec.bars[:, 1] > sec.zg, 0].sum())
    P = min(100 * As / (b * h0), 2.5)
    legs, dia, sv = stirrups
    Asv = legs * math.pi * dia * dia / 4
    rho_sv = Asv / (sv * b)
    Vcs = alpha1 * 0.45e-3 * b * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(c.fcuk) * rho_sv * sst.fsv)
    vmax = 0.51e-3 * math.sqrt(c.fcuk) * b * h0
    vmin = 0.50e-3 * c.ftd * b * h0
    chk = Check("shear (circular section)", gamma0 * Vd, Vcs, "kN", "JTG 3362-2018 5.2.9, 5.2.11, 5.2.12 with "
                "GB 50010-2010 6.3.15 (b = 1.76 r, h0 = 1.6 r)",
                {"b": b, "h0": h0, "P": P, "rho_sv": rho_sv, "alpha1": alpha1, "alpha2": 1.0, "alpha3": 1.0,
                 "Vcs": Vcs, "fsv": sst.fsv, "V_section_max": vmax, "V_detailing": vmin, "Nd": Nd},
                ["3362 has no shear rule for circular sections: b, h0 of GB 50010-2010 6.3.15 (program convention)"])
    if Nd < 0:
        Vs = sst.fsv * Asv * h0 / sv / 1e3  # the stirrups alone (GB 6.3.14 lower limit)
        Vt = Vcs - 0.2 * gamma0 * abs(Nd)
        chk.resistance = max(Vt, Vs)
        chk.clause += ", GB 50010-2010 6.3.14 (axial tension)"
        chk.details.update(V_tension=Vt, V_stirrups=Vs)
        chk.notes.append(f"axial tension {abs(Nd):g} kN: Vcs − 0.2 γ0 N = {Vt:.0f} kN, not below fsv Asv h0/s = {Vs:.0f} kN "
                         "(GB 50010 6.3.14, program convention)")
        if Vs < 0.36e-3 * c.ftd * b * h0:
            chk.warnings.append(f"fsv Asv h0/s = {Vs:.0f} kN < 0.36 ftd b h0 = {0.36e-3 * c.ftd * b * h0:.0f} kN "
                                "(GB 50010 6.3.14)")
    elif Nd > 0:
        chk.notes.append("axial compression not counted (5.2.9 has no such term; safe side)")
    if gamma0 * abs(Vd) > vmax:
        chk.failed.append(f"section too small: γ0Vd > 0.51e-3 √fcu,k b h0 = {vmax:.0f} kN (5.2.11)")
    if gamma0 * abs(Vd) <= vmin and Nd >= 0:
        chk.notes.append(f"γ0Vd ≤ {vmin:.0f} kN: detailing hoops suffice (5.2.12)")
    d_long = _d_long(sec, 0.0)
    rule = "JTG 3363-2019 6.2.2" if pile else "9.6.1"
    if dia < max(8.0, d_long / 4):
        chk.warnings.append(f"hoop diameter {dia:g} mm < max(8 mm, d/4 = {d_long / 4:.1f} mm) ({rule})")
    d_min = float(np.nanmin(sec.bar_d)) if np.isfinite(sec.bar_d).any() else 0.0  # spacing: the thinnest bar
    d15 = 15 * d_min if d_min else math.inf
    if pile:
        smax, why = min(d15, 300.0), "15 d, 300 mm"
    elif sec.bars[:, 0].sum() > 0.03 * sec.area:
        smax, why = min(10 * d_min if d_min else math.inf, 200.0), "longitudinal steel > 3 %: 10 d, 200 mm"
    else:
        smax, why = min(d15, 0.8 * sec.h, 400.0), "15 d, 0.8 D, 400 mm"
    if sv > smax:
        chk.warnings.append(f"hoop spacing {sv:g} mm > {smax:.0f} mm ({why}; {rule})")
    chk.warnings += sec.material_rules()
    return chk


# ── 5.4: tension members ────────────────────────────────────────────────────
FSD_TENSION_MAX = 330.0  # table 3.2.3-1 note 1: axial and small-eccentricity tension members of reinforced concrete


def fsd_tension(s: RCSection) -> float:
    """design tensile strength of the bars of an axially or slightly eccentrically loaded tension member: at most
    330 MPa in reinforced concrete (table 3.2.3-1 note 1, which names reinforced-concrete members; prestressed members
    keep fsd)"""
    return s.steel.fsd if s.prestressed else min(s.steel.fsd, FSD_TENSION_MAX)


def _steel_forces(s: RCSection, fs: float):
    """(design forces N, depths mm) of every bar at fs and every tendon at fpd"""
    F = np.r_[fs * s.bars[:, 0], s.tendon.fpd * s.tendons[:, 0] if s.has_internal else []]
    d = np.r_[s.bars[:, 1], s.tendons[:, 1]]
    return F, d


def _resultant(s: RCSection, bars_mask, tendon_mask, fs: float):
    F = np.r_[fs * s.bars[bars_mask, 0], s.tendon.fpd * s.tendons[tendon_mask, 0] if s.has_internal else []]
    d = np.r_[s.bars[bars_mask, 1], s.tendons[tendon_mask, 1]]
    return (float(np.average(d, weights=F)), F.sum()) if len(F) and F.sum() > 0 else (None, 0.0)


def _circle_mud(s: RCSection):
    """flexural resistance without axial force (N·mm): 5.3.8 with Nud = 0 for solid circles with ≥ 8 bars (5.4.4), else
    the strain method (program convention)"""
    if s.kind == "circle" and len(s.bars) >= 8:
        a = brentq(lambda a: _circle_NM(s, a)[0], 1e-4, 0.625, xtol=1e-13)
        return _circle_NM(s, a)[1], "5.3.8 with N = 0"
    return flexure(s, 1.0).resistance * 1e6, "strain method"


def _tension_large(s: RCSection, e0: float):
    """large-eccentricity tension (5.4.2-3, -4) of s with the load e0 below the gross centroid, by the rule set of
    RCSection.forces: block depth x with M(x) = −e0 N(x), N < 0. Returns (Nu N, x, ξb h0) or None."""
    xb = s.xi_b()
    lo, hi = 1e-6 * s.h, 2 * s.h
    xs = np.geomspace(lo, hi, 600)
    FN = np.array([s.forces(x, xb) for x in xs])
    f = FN[:, 1] + e0 * FN[:, 0]
    roots = [k for k in range(len(xs) - 1) if f[k] * f[k + 1] <= 0 and FN[k, 0] < 0]
    if not roots:
        return None
    k = roots[0]
    x = brentq(lambda x: s.forces(x, xb)[1] + e0 * s.forces(x, xb)[0], xs[k], xs[k + 1], xtol=1e-10 * s.h)
    return -s.forces(x, xb)[0], x, xb * s.h0()


def _tension_large_full(s: RCSection, e0: float):
    """5.4.2 large eccentricity with 5.2.4 when x < 2a' (Md → Nd e'). As for flexure (§3.6), the section without its
    compression bars is the alternative: the larger governs. Returns (Nu N, details, notes)."""
    res = _tension_large(s, e0)
    if res is None:
        raise RuntimeError("no equilibrium for this eccentric tension")
    Nu, x, xbh0 = res
    det = {"x": x, "xi_b_h0": xbh0}
    notes = []
    aref, compressed_p = s.comp_ref()
    if aref is not None and x < 2 * aref:
        e_c = s.zg + e0 - aref  # load to the compression-steel resultant
        N_about = s.design_tension_moment(aref, compressed_p) / e_c
        N_single = 0.0
        _, pt = s.tendon_masks()
        if not pt.any():
            _, top = s.edge_masks()
            single = s.replace(bars=s.bars[~top], bar_d=s.bar_d[~top])
            r1 = _tension_large(single, e0)
            if r1 is not None and r1[1] <= r1[2]:
                N_single = r1[0]
        Nu = max(N_about, N_single)
        how = "moments about the compression steel" if N_about >= N_single else "compression bars ignored"
        notes.append(f"x = {x:.0f} < 2a' = {2 * aref:.0f} mm: {how} governs (5.4.2 → 5.2.4, e' = {e_c:.0f} mm)")
        det.update(a_ref=aref, N_about_compression_steel=N_about / 1e3, N_singly_reinforced=N_single / 1e3)
    return Nu, det, notes


def tension(sec: RCSection, Nd: float, Md: float = 0.0, *, gamma0: float = 1.0) -> Check:
    """tension member (5.4). Nd tensile force (kN, > 0) acting with the moment Md (kN·m) about the gross centroid;
    e0 = |Md| / Nd, the sign of Md picks the side (sagging: the load below the centroid, nearer As). No η (tension).
    Reinforced concrete: the bars count at most 330 MPa in axial and small-eccentricity tension and in Nud of 5.4.4
    (table 3.2.3-1 note 1, fsd_tension); large eccentricity and the flexural Mud keep fsd.
    - Md = 0: axial tension 5.4.1, Nud = fsd ΣAs + fpd ΣAp (all the steel).
    - circles (5.4.4): γ0 Nd ≤ 1 / (1/Nud + e0/Mud), Mud by 5.3.8 with N = 0 (solid circles with ≥ 8 bars; otherwise
      the strain method; hollow circles, outside 5.4.4: program convention, a note).
    - other sections (5.4.2, written for rectangles): small eccentricity when the load lies between the resultants of
      As/Ap and A's/A'p: γ0 Nd e ≤ moment about As/Ap of the other steel at fsd / fpd, γ0 Nd e' ≤ moment about A's/A'p
      (5.4.2-1, -2; program convention: side and web bars are counted too, each at its design strength; the result stays
      continuous with 5.4.1 and does not exceed the strain-compatibility solution of 5.1.3–5.1.5, whose steel strain is
      not capped); large eccentricity otherwise: 5.4.2-3, -4 by the rule set of RCSection.forces (T, I and box sections
      included), x ≤ ξb h0 (5.2.2-3), and 5.2.4 with Md → Nd e' when x < 2a'.
    Minimum reinforcement (9.1.12, reinforced concrete): one side ≥ max(45 ftd/fsd, 0.2) % — of the gross area for axial
    and small-eccentricity tension (both faces, being both in tension; fsd as capped above), of b h0 for large
    eccentricity; compression bars of large eccentricity ≥ 0.2 % of the gross area when the resistance needs them.
    Circles: half the bars (program convention)."""
    if Nd <= 0:
        raise ValueError("Nd must be a tensile force (> 0)")
    if sec.has_external:
        raise NotImplementedError("5.4 has no provisions for tension members with external tendons")
    s = sec if Md >= 0 else sec.flipped()
    c, st = s.concrete, s.steel
    ft = fsd_tension(s)
    F, _ = _steel_forces(s, ft)
    Nud = F.sum()
    e0 = abs(Md) / Nd * 1e3
    ring = s.kind in ("circle", "annulus")
    chk = Check("tension" + (" (hogging side)" if Md < 0 else ""), gamma0 * Nd, 0.0, "kN", "JTG 3362-2018 5.4.1",
                {"e0": e0, "Nud": Nud / 1e3, "fsd_tension": ft})
    capped = (f"bars at {ft:g} MPa instead of fsd = {st.fsd:g} MPa (table 3.2.3-1 note 1)" if ft < st.fsd else None)
    rho_min = max(0.45 * c.ftd / ft, 0.002)
    A = s.area
    rc_ = not s.prestressed
    if not rc_:
        chk.notes.append("9.1.12 (reinforced concrete members) not applied to a prestressed section")
    bot, top = s.edge_masks()

    def sides_gross(what):
        if not rc_:
            return
        if ring:
            r = s.bars[:, 0].sum() / 2 / A
            chk.details["rho_one_side"] = r
            if r < rho_min:
                chk.warnings.append(f"{what}: half the bars {r:.3%} of the gross area < {rho_min:.3%} (9.1.12)")
            return
        for side, m in (("tension face", bot), ("far face", top)):
            r = s.bars[m, 0].sum() / A
            chk.details[f"rho_{side.split()[0]}"] = r
            if r < rho_min:
                chk.warnings.append(f"{what}: {side} reinforcement {r:.3%} of the gross area < {rho_min:.3%} (9.1.12)")

    if e0 == 0:
        chk.resistance = Nud / 1e3
        chk.details["case"] = "axial tension"
        chk.notes += [capped] if capped else []
        sides_gross("axial tension")
    elif ring:
        Mud, how = _circle_mud(s)
        chk.resistance = 1 / (1 / Nud + e0 / Mud) / 1e3
        chk.clause = "JTG 3362-2018 5.4.4"
        chk.details.update(Mud=Mud / 1e6, Mud_from=how, case="eccentric tension (circle)")
        chk.notes += [capped + " in Nud; Mud with fsd"] if capped else []
        if s.kind == "annulus":
            chk.notes.append("hollow circle: 5.4.4 is written for solid circles, applied here with Mud from the strain "
                             "method (program convention)")
        sides_gross("eccentric tension")
    else:
        pb, pt = s.tendon_masks()
        d_t, _ = _resultant(s, bot, pb, ft)
        d_c, _ = _resultant(s, top, pt, ft)
        if d_t is None:
            raise ValueError("no steel on the tension face")
        z = s.zg + e0  # depth of the load
        Fs, ds = _steel_forces(s, ft)
        if d_c is not None and z <= d_t:
            e, e_ = d_t - z, z - d_c
            Mt, Mc = (Fs * (d_t - ds)).sum(), (Fs * (ds - d_c)).sum()
            N1 = Mt / e if e > 0 else math.inf
            N2 = Mc / e_ if e_ > 0 else math.inf
            chk.resistance = min(N1, N2) / 1e3
            chk.clause = "JTG 3362-2018 5.4.2-1, -2"
            chk.details.update(case="small eccentricity", e=e, e_prime=e_, N_5_4_2_1=N1 / 1e3, N_5_4_2_2=N2 / 1e3)
            chk.notes += [capped] if capped else []
            sides_gross("small-eccentricity tension")
        else:
            Nu, det, notes = _tension_large_full(s, e0)
            chk.resistance = Nu / 1e3
            chk.clause = "JTG 3362-2018 5.4.2-3, -4"
            chk.details.update(det, case="large eccentricity")
            chk.notes += notes
            if det["x"] > det["xi_b_h0"]:
                chk.failed.append(f"x = {det['x']:.0f} > ξb h0 = {det['xi_b_h0']:.0f} mm (5.2.2-3)")
            if rc_:
                b, h0 = s.web(), s.h0()
                rho = s.bars[bot, 0].sum() / (b * h0)
                rho_min = max(0.45 * c.ftd / st.fsd, 0.002)  # large eccentricity: fsd not capped
                chk.details.update(rho=rho, rho_min=rho_min)
                if rho < rho_min:
                    chk.warnings.append(f"tension reinforcement ratio {rho:.4%} < minimum {rho_min:.4%} (9.1.12)")
                if top.any():
                    single = s.replace(bars=s.bars[~top], bar_d=s.bar_d[~top])
                    try:
                        N_wo = _tension_large_full(single, e0)[0]
                    except RuntimeError:
                        N_wo = 0.0
                    rc1 = s.bars[top, 0].sum() / A
                    chk.details.update(N_without_compression_bars=N_wo / 1e3, rho_compression=rc1)
                    if gamma0 * Nd * 1e3 > N_wo and rc1 < 0.002:
                        chk.warnings.append(f"compression bars needed by the calculation: {rc1:.3%} of the gross area "
                                            "< 0.2 % (9.1.12)")
    chk.warnings += s.material_rules()
    return chk


# ── 5.3.11, 5.4.3: biaxial eccentricity ─────────────────────────────────────
def rect_biaxial(b: float, h: float, *, bars=(), layout=None, concrete="C30", steel="HRB400"):
    """a rectangle b (across, along Y of figure 5.3.11) × h (along X) with bars at (u, v, diameter): u from the left face
    across b, v from the top face down h. layout = (nb, nh, d, cover): nb bars on each face of width b and nh on each
    face of height h, corners included, bar centres `cover` from the faces. Returns (sec_x, sec_y): sec_x bends along X
    (depth h, width b; positive moment compresses the top face), sec_y along Y (depth b, width h; positive moment
    compresses the left face)."""
    c, s = _mats(concrete, steel)
    pts = [tuple(p) for p in bars]
    if layout is not None:
        nb, nh, d, cover = layout
        us = np.linspace(cover, b - cover, nb)
        vs = np.linspace(cover, h - cover, nh)
        grid = {(round(u, 6), round(v, 6)) for u in us for v in (cover, h - cover)}
        grid |= {(round(u, 6), round(v, 6)) for v in vs for u in (cover, b - cover)}
        pts += [(u, v, d) for u, v in sorted(grid)]
    if not pts:
        raise ValueError("no bars")
    P = np.array(pts, float)
    a = math.pi * P[:, 2] ** 2 / 4
    sx = RCSection(c, s, [Rect(b, 0, h)], np.c_[a, P[:, 1]], h, kind="rect", bar_d=P[:, 2])
    sy = RCSection(c, s, [Rect(h, 0, b)], np.c_[a, P[:, 0]], b, kind="rect", bar_d=P[:, 2])
    return sx, sy


def _mirror_symmetric(s: RCSection) -> bool:
    """concrete and bars symmetric about the section's mid-depth"""
    a, d = s.bars[:, 0], s.bars[:, 1]
    k1, k2 = np.lexsort((a, np.round(d, 6))), np.lexsort((a, np.round(s.h - d, 6)))
    return (abs(s.zg - s.h / 2) <= 1e-9 * s.h and bool(np.allclose(a[k1], a[k2]))
            and bool(np.allclose(d[k1], s.h - d[k2])))


def biaxial(sec_x: RCSection, sec_y: RCSection | None, Nd: float, Mdx: float, Mdy: float, *, l0x: float | None = None,
            l0y: float | None = None, gamma0: float = 1.0, method: str = "auto") -> Check:
    """biaxial eccentric compression (5.3.11), reinforced concrete sections with two axes of symmetry:
    γ0 Nd ≤ 1 / (1/Nux + 1/Nuy − 1/Nu0). Nux = jtg.eccentric(sec_x, Nd, Mdx, l0x) (η, the minimum e0 of 5.3.9 and the
    method as there, all bars), Nuy likewise; Nu0 = 0.9 (fcd A + f'sd A's), 5.3.1 without φ. Mdx = Nd e0x: the moment
    whose eccentricity runs along X (the depth of sec_x); sec_x, sec_y from rect_biaxial or any pair describing the same
    section in the two directions. With one moment zero the member is uniaxial (5.3.4–5.3.10): this check then still
    applies the minimum e0 in the other direction (a note). Program convention: Nu ≤ min(Nux, Nuy). The commentary derives
    5.3.11 with Nu0 the true axial resistance, above Nux and Nuy; the 0.9 of 5.3.1 in Nu0 (absent from Nux, Nuy) can put
    Nu0 below them at small eccentricities, and the formula would then give more than the uniaxial resistance, which a
    second eccentricity can only lower. Circles (program convention): symmetric about every axis, checked with the
    resultant moment √(Mdx² + Mdy²) by jtg.eccentric, l0 the larger of l0x, l0y; sec_y is not needed."""
    if sec_x.kind in ("circle", "annulus"):
        l0s = [v for v in (l0x, l0y) if v]
        chk = eccentric(sec_x, Nd, math.hypot(Mdx, Mdy), max(l0s) if l0s else None, gamma0=gamma0, method=method)
        chk.name = "biaxial eccentric compression (circle)"
        chk.notes.append("circle: resultant moment √(Mdx² + Mdy²) (5.3.11 not needed; program convention)")
        return chk
    if sec_y is None:
        raise ValueError("non-circular sections need sec_y (jtg.rect_biaxial gives both)")
    if sec_x.prestressed or sec_y.prestressed:
        raise NotImplementedError("5.3.11 covers reinforced-concrete members")
    if abs(sec_x.area - sec_y.area) > 1e-6 * sec_x.area or abs(sec_x.bars[:, 0].sum() - sec_y.bars[:, 0].sum()) > 1e-6 * sec_x.bars[:, 0].sum():
        raise ValueError("sec_x and sec_y must describe the same section (area and bars)")
    if not (_mirror_symmetric(sec_x) and _mirror_symmetric(sec_y)):
        raise ValueError("5.3.11 is for sections with two axes of symmetry (concrete and bars)")
    cx = eccentric(sec_x, Nd, Mdx, l0x, gamma0=gamma0, method=method)
    cy = eccentric(sec_y, Nd, Mdy, l0y, gamma0=gamma0, method=method)
    c, st = sec_x.concrete, sec_x.steel
    A, As = sec_x.area, sec_x.bars[:, 0].sum()
    if As / A > 0.03:
        A -= As
    Nu0 = 0.9 * (c.fcd * A + st.fsd_c * As) / 1e3
    Nux, Nuy = cx.resistance, cy.resistance
    N_formula = 1 / (1 / Nux + 1 / Nuy - 1 / Nu0)
    Nu = min(N_formula, Nux, Nuy)
    chk = Check("biaxial eccentric compression", gamma0 * Nd, Nu, "kN", "JTG 3362-2018 5.3.11",
                {"Nux": Nux, "Nuy": Nuy, "Nu0": Nu0, "N_5_3_11": N_formula, "x": cx.details, "y": cy.details})
    if N_formula > Nu:
        chk.notes.append(f"5.3.11 gives {N_formula:.0f} kN > min(Nux, Nuy): limited to {Nu:.0f} kN (Nu0 with 0.9 below "
                         "Nux or Nuy; program convention)")
    if Mdx == 0 or Mdy == 0:
        chk.notes.append("one moment is zero: a uniaxial member (5.3.4–5.3.10); 5.3.11 here keeps the minimum e0 of "
                         "5.3.9 in that direction")
    for f_ in cx.failed + cy.failed:
        if f_ not in chk.failed:
            chk.failed.append(f_)
    for w in cx.warnings + cy.warnings:
        if w not in chk.warnings:
            chk.warnings.append(w)
    return chk


def biaxial_tension(sec_x: RCSection, sec_y: RCSection | None, Nd: float, Mdx: float, Mdy: float, *,
                    gamma0: float = 1.0) -> Check:
    """biaxial eccentric tension of a symmetrically reinforced rectangle (5.4.3):
    γ0 Nd ≤ 1 / (1/Nud + √((e0x/Mux)² + (e0y/Muy)²)), Nud by 5.4.1 (bars at most 330 MPa, table 3.2.3-1 note 1),
    Mux, Muy by 5.2 (jtg.flexure of sec_x, sec_y),
    e0x = Mdx/Nd along X, e0y = Mdy/Nd along Y. The minimum reinforcement comes from jtg.tension in each direction.
    Circles: 5.4.4 with e0 = √(e0x² + e0y²) (the commentary derives 5.4.3 from 5.4.4); sec_y is not needed."""
    if Nd <= 0:
        raise ValueError("Nd must be a tensile force (> 0)")
    if sec_x.kind in ("circle", "annulus"):
        chk = tension(sec_x, Nd, math.hypot(Mdx, Mdy), gamma0=gamma0)
        chk.name = "biaxial eccentric tension (circle)"
        return chk
    if sec_y is None:
        raise ValueError("non-circular sections need sec_y (jtg.rect_biaxial gives both)")
    if sec_x.prestressed or sec_y.prestressed:
        raise NotImplementedError("5.4.3 covers reinforced-concrete members")
    for s in (sec_x, sec_y):
        if s.kind != "rect" or not _mirror_symmetric(s):
            raise ValueError("5.4.3 is for symmetrically reinforced rectangles")
    Nud = fsd_tension(sec_x) * sec_x.bars[:, 0].sum()  # 5.4.1 with table 3.2.3-1 note 1
    Mux, Muy = flexure(sec_x, 1.0).resistance * 1e6, flexure(sec_y, 1.0).resistance * 1e6
    e0x, e0y = abs(Mdx) / Nd * 1e3, abs(Mdy) / Nd * 1e3
    Nu = 1 / (1 / Nud + math.hypot(e0x / Mux, e0y / Muy))
    chk = Check("biaxial eccentric tension", gamma0 * Nd, Nu / 1e3, "kN", "JTG 3362-2018 5.4.3",
                {"Nud": Nud / 1e3, "Mux": Mux / 1e6, "Muy": Muy / 1e6, "e0x": e0x, "e0y": e0y})
    for s, M in ((sec_x, Mdx), (sec_y, Mdy)):
        t = tension(s, Nd, M, gamma0=gamma0)
        for f_ in t.failed:
            if f_ not in chk.failed:
                chk.failed.append(f_)
        for w in t.warnings:
            if w not in chk.warnings:
                chk.warnings.append(w)
    return chk
