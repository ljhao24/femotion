"""External prestressing (JTG 3362-2018 5.2.5, 5.2.6, 5.2.9-5, 6.1.4-2, 6.1.6-7, 6.1.7-5, -6, table 6.2.2, 7.1.5-3,
9.4.21–9.4.27).

The code has no worked example for external tendons, so most checks come from somewhere else than the formula they
test: another clause (an external tendon is a constant force on the bonded section, so 5.3.4 for eccentric compression
gives the same flexural resistance as 5.2.5), a physical equivalence (an external tendon at σpe,ex is a yielded bar of
the same force; an internal tendon of vanishing stiffness carrying the same force), an independent implementation
(the box section integrated slice by slice; the cracked section's equilibrium integrated by hand), a closed form (the
force method for the secondary moment of a polygonal tendon; the capstan equation for the deviators) or a limit
(σpe,ex → 0 gives back reinforced concrete). Hand calculations with the printed formulas are marked as such."""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import prestress as ps
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import chain_sections
from femotion.frame.prestress import Tendon, effects

A16, A20, A25, A28 = (math.pi * d * d / 4 for d in (16, 20, 25, 28))
FCD, FSD, FPD = 22.4, 330.0, 1260.0  # C50, HRB400, strand 1860


# ── 6.1.4-2, 7.1.5-3, table 6.2.2 ──────────────────────────────────────────
def test_jacking_and_service_limits():
    """printed values: σcon ≤ 0.70 fpk for external wire and strand (6.1.4-2, p52), + 0.05 fpk when overstressing;
    σpe,ex ≤ 0.60 fpk in service (7.1.5-3, p70). The commentary (p194) derives the 0.70 from the 0.60: with the losses and
    the live-load increment of 30–75 m girders the jacking stress came out at 0.68–0.71 fpk."""
    t = jtg.tendon("strand", 1860)
    assert t.sigma_con_max(external=True) == pytest.approx(0.70 * 1860)
    assert t.sigma_con_max(external=True, overstress=True) == pytest.approx(0.75 * 1860)
    assert jtg.tendon("wire", 1570).sigma_con_max(external=True) == pytest.approx(0.70 * 1570)
    with pytest.raises(ValueError):
        jtg.tendon("bar", 930).sigma_con_max(external=True)

    def sec(spe):
        return jtg.rect(600, 1500, concrete="C50", bottom=[(6, 25, 60)], top=[(4, 16, 45)],
                        external=[(2780, 1300, spe)])
    ok = jtg.service_stresses(sec(0.59 * 1860), 500.0, pc_class="A")
    bad = jtg.service_stresses(sec(0.61 * 1860), 500.0, pc_class="A")
    assert ok.details["external_limit"] == pytest.approx(1116.0)
    assert not any("7.1.5-3" in f for f in ok.failed) and any("7.1.5-3" in f for f in bad.failed)


def test_friction_at_deviators():
    """table 6.2.2 (p56) as printed; k = 0 and only the deviator and anchorage ducts count (table note). Consistency
    with the capstan (Euler–Eytelwein) law behind 6.2.2: deviator by deviator, the factor e^(−μθ) at each one (the same
    identity as e^(−μΣθ), so a check of the summing, not an independent source)."""
    assert ps.DUCT_FRICTION_EXTERNAL == {"steel pipe": ((0.20, 0.30), (0.08, 0.10)), "HDPE": ((0.12, 0.15), (0.08, 0.10))}
    s, mu, th = 1302.0, 0.12, [0.10, 0.06, 0.12]
    after = s
    for t in th:
        after *= math.exp(-mu * t)
    assert ps.friction_loss_external(s, mu, th) == pytest.approx(s - after, rel=1e-12)
    assert ps.friction_loss_external(s, mu, 0.28) == pytest.approx(ps.friction_loss(s, mu, 0.28, 0.0015, 0.0))


# ── 5.2.5, 5.2.6 flexure ───────────────────────────────────────────────────
def _rect_ex(sig_ex=1000.0, dex=1300.0, Aex=2780.0):
    return jtg.rect(600, 1500, concrete="C50", bottom=[(6, 25, 60)], top=[(4, 16, 45)], external=[(Aex, dex, sig_ex)])


def test_flexure_rectangle_as_eccentric_compression_5_3_4():
    """independent clause: the external tendon is a constant force P = σpe,ex Aex (5.2.5), i.e. the bonded section
    carries an eccentric compression P (the commentary to 7.1.4, p205, treats Np0 the same way). 5.3.4-1 gives x from
    P = fcd b x + f'sd A's − fsd As, and 5.3.4-2 the moments about As: P e = fcd b x (h0s − x/2) + f'sd A's (h0s − a's).
    The applied moment about the gross centroid is then M = P e − P (h0s − dex), against the program's 5.2.5."""
    s = _rect_ex()
    As, Asc, P = 6 * A25, 4 * A16, 1000.0 * 2780
    x = (P + FSD * As - FSD * Asc) / (FCD * 600)  # 5.3.4-1 (large eccentricity, A's at f'sd: x ≥ 2a's)
    h0s = 1440.0
    Pe = FCD * 600 * x * (h0s - x / 2) + FSD * Asc * (h0s - 45)  # 5.3.4-2
    M = Pe - P * (h0s - 1300)
    r = jtg.flexure(s, 1000.0, pc_class="A")
    assert x > 90 and r.details["x"] == pytest.approx(x, rel=1e-9)
    assert r.resistance == pytest.approx(M / 1e6, rel=1e-9)
    # h0 of figure 5.2.5: the resultant of As (fsd) and Aex (σpe,ex); ξb by table 5.2.1 note 1: the strand's 0.40
    assert r.details["h0"] == pytest.approx((FSD * As * 1440 + P * 1300) / (FSD * As + P), rel=1e-12)
    assert r.details["xi_b"] == 0.40 and r.clause.endswith("5.2.5 / 5.2.6")
    assert r.details["xi_b_alternative"]["xi_b"] == 0.53  # bonded steel only (note 2)


def test_tee_web_5_2_5_5():
    """hand calculation with 5.2.5-5, -6 (p26): flange in compression, x in the web; h0 to the resultant of As, Ap and Aex
    (figure 5.2.5 a)"""
    As, Ap, Aex = 8 * A25, 1668.0, 2502.0
    s = jtg.tee(200, 1800, 1600, 150, concrete="C50", bottom=[(8, 25, 60)], tendons=[(Ap, 1650, 1100, 80)],
                external=[(Aex, 1500, 1050)])
    F = FSD * As + FPD * Ap + 1050 * Aex
    assert F > FCD * 1600 * 150  # 5.2.5 item 2-2)
    x = (F / FCD - 1400 * 150) / 200  # 5.2.5-6
    h0 = (FSD * As * 1740 + FPD * Ap * 1650 + 1050 * Aex * 1500) / F
    Mu = FCD * (200 * x * (h0 - x / 2) + 1400 * 150 * (h0 - 75))  # 5.2.5-5
    r = jtg.flexure(s, 1000.0, pc_class="A")
    assert r.details["h0"] == pytest.approx(h0, rel=1e-12) and r.details["x"] == pytest.approx(x, rel=1e-9)
    assert r.resistance == pytest.approx(Mu / 1e6, rel=1e-9)
    assert r.details["xi_b"] == 0.40  # the internal strand (table 5.2.1 note 1); the external tendon does not enter


def test_flexure_equivalences_and_limits():
    """physics: (1) an external tendon at σpe,ex = fsd at the bars' depth is the same force as that much more yielded
    bar; (2) σpe,ex → 0 gives back the reinforced concrete section; (3) a hogging moment on the section equals a sagging
    one on the same section built upside down."""
    ex = jtg.rect(600, 1500, concrete="C50", bottom=[(6, 25, 60)], top=[(4, 16, 45)], external=[(1500, 1440, FSD)])
    bars = jtg.rect(600, 1500, concrete="C50", bottom=[(6, 25, 60)], top=[(4, 16, 45)], bars=[(1500, 1440)])
    assert jtg.flexure(ex, 1.0, pc_class="A").resistance == pytest.approx(jtg.flexure(bars, 1.0).resistance, rel=1e-9)
    rc = jtg.flexure(jtg.rect(600, 1500, concrete="C50", bottom=[(6, 25, 60)], top=[(4, 16, 45)]), 1.0)
    assert jtg.flexure(_rect_ex(sig_ex=1e-9), 1.0, pc_class="A").resistance == pytest.approx(rc.resistance, rel=1e-9)
    up = jtg.tee(250, 1600, 1800, 180, concrete="C50", bottom=[(8, 20, 50)], top=[(12, 25, 60)],
                 tendons=[(1112, 150, 1150, 80)], external=[(2502, 250, 1000)])
    # the same tee turned over by hand: flange at the bottom (an inverted T = pieces of an I with no top flange)
    down = jtg.tee(250, 1600, 250, 0.0, bf_bottom=1800, hf_bottom=180, concrete="C50", bottom=[(12, 25, 60)],
                   top=[(8, 20, 50)], tendons=[(1112, 1450, 1150, 80)], external=[(2502, 1350, 1000)])
    a, b = jtg.flexure(up, -1.0, pc_class="A"), jtg.flexure(down, 1.0, pc_class="A")
    assert a.resistance == pytest.approx(b.resistance, rel=1e-9) and a.details["x"] == pytest.approx(b.details["x"])


def test_x_below_2a_5_2_6():
    """hand calculations with 5.2.6-1 and -2 (p27), and the cross-path: the same external force as a bonded bar
    (fsd A = σpe,ex Aex at the same depth) goes through 5.2.4 and gives the same moment about the compression steel"""
    # 5.2.6-2: only A's in the compression zone
    s = jtg.rect(1000, 800, concrete="C50", bottom=[(5, 20, 50)], top=[(10, 28, 60)], external=[(1000, 700, 1000)])
    r = jtg.flexure(s, 1.0, pc_class="A")
    M2 = FSD * 5 * A20 * (800 - 50 - 60) + 1000 * 1000 * (800 - 100 - 60)
    assert r.details["x"] < 120 and r.details["M_about_compression_steel"] == pytest.approx(M2 / 1e6, rel=1e-12)
    eq = jtg.rect(1000, 800, concrete="C50", bottom=[(5, 20, 50)], top=[(10, 28, 60)], bars=[(1e6 / FSD, 700)])
    assert jtg.flexure(eq, 1.0).details["M_about_compression_steel"] == pytest.approx(M2 / 1e6, rel=1e-12)
    # 5.2.6-1: A's and an internal A'p in compression (f'pd − σ'p0 > 0): moments about their resultant a'
    s = jtg.rect(1000, 900, concrete="C50", bottom=[(5, 20, 50)], top=[(6, 20, 50)],
                 tendons=[(556, 800, 1100, 60), (556, 80, 200, 60)], external=[(1112, 820, 1000)])
    Fs, Fp = FSD * 6 * A20, (390 - 200) * 556
    a1 = (Fs * 50 + Fp * 80) / (Fs + Fp)
    M1 = FPD * 556 * (800 - a1) + FSD * 5 * A20 * (850 - a1) + 1000 * 1112 * (820 - a1)
    r = jtg.flexure(s, 1.0, pc_class="A")
    assert r.details["x"] < 2 * a1 and r.details["a_ref"] == pytest.approx(a1)
    assert r.resistance == pytest.approx(M1 / 1e6, rel=1e-12)


def test_box_against_slice_integration():
    """independent implementation: the box of figure 5.2.5's "may be calculated in accordance with this clause",
    integrated in thin slices with its own width function and solved by bisection (As, Ap at fsd, fpd; A's at f'sd with
    x ≥ 2a's; Aex at σpe,ex), against the program's piecewise section (the midpoint rule misses up to one slice at the
    flange edge: 1e-5)"""
    B, h, tt, tb, webs = 3000.0, 2200.0, 150.0, 200.0, 600.0
    s = jtg.box(B, h, tt, tb, webs, concrete="C50", bottom=[(30, 16, 60)], top=[(60, 16, 50)],
                tendons=[(3336, 2050, 1150, 90)], external=[(8340, 1900, 1000)])
    width = lambda z: np.where(z < tt, B, np.where(z < h - tb, webs, B))
    T = [(FSD * 30 * A16, h - 60), (FPD * 3336, 2050.0), (1000.0 * 8340, 1900.0)]
    Cs = (FSD * 60 * A16, 50.0)
    n = 400000

    def block(x):
        z = (np.arange(n) + 0.5) * x / n
        dA = width(z) * x / n
        return FCD * dA.sum(), FCD * (dA * z).sum()
    lo, hi = 100.0, 1000.0
    for _ in range(80):
        x = (lo + hi) / 2
        lo, hi = (x, hi) if block(x)[0] + Cs[0] < sum(f for f, _ in T) else (lo, x)
    C, Cz = block(x)
    Mu = sum(f * d for f, d in T) - Cz - Cs[0] * Cs[1]  # moments about the top fibre, N = 0
    r = jtg.flexure(s, 1.0, pc_class="A")
    assert x > tt and r.details["x"] == pytest.approx(x, rel=1e-5)
    assert r.resistance == pytest.approx(Mu / 1e6, rel=1e-7)


# ── 5.2.9 shear ───────────────────────────────────────────────────────────
def _tee_shear(dex=1400.0, internal=True):
    return jtg.tee(250, 1600, 2000, 180, concrete="C50", bottom=[(8, 20, 50)],
                   tendons=[(1668, 1500, 1100, 90)] if internal else (), external=[(1668, dex, 1000)])


def test_shear_vpb_ex_5_2_9_5():
    """hand calculation with 5.2.9-2, -5 (p28–29): Vpb,ex = 0.75e-3 Σ σpe,ex Aex sin θex, with h0 and P from the bonded
    As, Ap (ρ = (Ap + As)/(b h0), program convention); cross-path: the same σ·A·sinθ as a bent internal tendon (5.2.9-4)
    gives the same resistance"""
    s = _tee_shear()
    r = jtg.shear(s, 900.0, (2, 12, 150), pc_class="A", bent_external=[(1668, 6.0)])
    As = 8 * A20
    h0 = (FSD * As * 1550 + FPD * 1668 * 1500) / (FSD * As + FPD * 1668)
    P = 100 * (As + 1668) / (250 * h0)
    rsv = 2 * 113.097 / (150 * 250)
    Vcs = 0.45e-3 * 1.0 * 1.25 * 1.1 * 250 * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(50) * rsv * 330)
    Vex = 0.75e-3 * 1000 * 1668 * math.sin(math.radians(6))
    assert r.details["h0"] == pytest.approx(h0, rel=1e-12) and r.details["P"] == pytest.approx(P, rel=1e-12)
    assert r.details["Vpb_ex"] == pytest.approx(Vex, rel=1e-12)
    assert r.resistance == pytest.approx(Vcs + Vex, rel=1e-5)
    same = jtg.shear(s, 900.0, (2, 12, 150), pc_class="A", bent_tendons=[(1000 * 1668 / FPD, 6.0)])
    assert same.resistance == pytest.approx(r.resistance, rel=1e-12)


def test_shear_alpha2_direction_of_external_force():
    """5.2.9 α2 (p28): 1.25, but 1.0 when the steel's resultant bends the section the same way as the load. An external
    tendon above the centroid of a girder without internal tendons (e.g. near a deviator over a support) pushes the
    concrete into sagging, so α2 = 1.0 under a sagging moment; below the centroid α2 = 1.25"""
    assert jtg.shear(_tee_shear(1400.0, False), 500.0, (2, 12, 150), pc_class="A").details["alpha2"] == 1.25
    assert jtg.shear(_tee_shear(200.0, False), 500.0, (2, 12, 150), pc_class="A").details["alpha2"] == 1.0


# ── 6.1.6-7, 6.1.7-5, -6 ──────────────────────────────────────────────────
def _mixed():
    return jtg.rect(500, 1200, concrete="C50", bottom=[(4, 20, 50)], top=[(4, 16, 40)], tendons=[(1112, 1050, 1200, 70)],
                    sigma_pe=1100.0, sigma_l6=(60.0, 40.0), external=[(834, 1000, 1000), (278, 200, 900)])


def _rect_props(b, h, pts):
    """area, centroid depth, I of a rectangle plus point areas (area, depth): closed form"""
    A = b * h + sum(a for a, _ in pts)
    z = (b * h * h / 2 + sum(a * d for a, d in pts)) / A
    I = b * h ** 3 / 12 + b * h * (h / 2 - z) ** 2 + sum(a * (d - z) ** 2 for a, d in pts)
    return A, z, I


def test_mixed_prestress_stress_6_1_6_7():
    """hand calculation on closed-form sections: before grouting 6.1.7-5, -6 and 6.1.6-7 on the net section (6.1.7 note
    3); after grouting Np,ex split by 6.1.5 item 2 (p52): the internal tendons and the bars' σl6 on the net section with
    Mp2, the external tendons on the transformed one; σ'l6 counts since A'p,ex ≠ 0 (6.1.7 note 2)"""
    s = _mixed()
    Ec, Es, Ep = 3.45e4, 2.0e5, 1.95e5
    As, Asc, duct = 4 * A20, 4 * A16, math.pi * 70 ** 2 / 4
    net = [((Es / Ec - 1) * As, 1150), ((Es / Ec - 1) * Asc, 40), (-duct, 1050)]
    A0, z0, I0 = _rect_props(500, 1200, net + [(Ep / Ec * 1112, 1050)])
    F = [(1100 * 1112, 1050), (1000 * 834, 1000), (900 * 278, 200), (-60 * As, 1150), (-40 * Asc, 40)]
    N = sum(f for f, _ in F)  # 6.1.7-5
    e = sum(f * (d - z0) for f, d in F) / N  # 6.1.7-6, below the centroid +
    assert s.prestress_resultant("pe")[0] == pytest.approx(N, rel=1e-12)
    An, zn, In = _rect_props(500, 1200, net)
    en = sum(f * (d - zn) for f, d in F) / N
    assert float(s.prestress_stress(1200.0, grouted=False)) == pytest.approx(N / An + N * en * (1200 - zn) / In, rel=1e-10)
    Mp2 = 150.0
    Fi, Fx = [F[0], F[3], F[4]], [F[1], F[2]]
    for z in (0.0, 600.0, 1200.0):
        hand = (sum(f for f, _ in Fi) / An + sum(f * (d - zn) for f, d in Fi) * (z - zn) / In - Mp2 * 1e6 * (z - zn) / In
                + sum(f for f, _ in Fx) / A0 + sum(f * (d - z0) for f, d in Fx) * (z - z0) / I0)
        assert float(s.prestress_stress(z, Mp2)) == pytest.approx(hand, rel=1e-10)
    # the literal single-section reading of 6.1.6-7 (everything on A0, I0) differs by a few per cent at the edges
    lit = N / A0 + N * e * (1200 - z0) / I0
    assert abs(float(s.prestress_stress(1200.0)) / lit - 1) < 0.05
    # 6.1.7 note 2: without A'p and A'p,ex the top bars take no σ'l6
    s2 = s.replace(external=s.external[:1])
    assert s2.prestress_resultant("pe")[0] == pytest.approx(N - 900 * 278 + 40 * Asc, rel=1e-12)


def test_prestress_stress_equilibrium():
    """physics: before grouting the stress field of 6.1.6-7 integrated over its own (net) section, slice by slice for
    the concrete plus the bars and the duct, gives back Np,ex and its moment about the top fibre"""
    s = _mixed()
    n = 200000
    z = (np.arange(n) + 0.5) * 1200 / n
    sig = s.prestress_stress(z, grouted=False)
    F = (sig * 500 * 1200 / n).sum()
    M = (sig * z * 500 * 1200 / n).sum()
    for a, d in s._steel_points("net"):
        F += float(s.prestress_stress(d, grouted=False)) * a
        M += float(s.prestress_stress(d, grouted=False)) * a * d
    N, zN = s.prestress_resultant("pe")
    assert F == pytest.approx(N, rel=1e-6) and M / F == pytest.approx(zN, rel=1e-6)
    ex_only = jtg.rect(500, 1200, concrete="C50", bottom=[(4, 20, 50)], external=[(834, 1000, 1000)])
    assert float(ex_only.prestress_stress(0.0, grouted=False)) == pytest.approx(float(ex_only.prestress_stress(0.0)))


def test_no_jump_when_an_external_tendon_is_added():
    """continuity: an external tendon of vanishing force changes neither the prestress stresses (6.1.5 item 2: the
    internal tendons stay on the net section) nor the cracking moment (6.5.2-7: net section)"""
    kw = dict(concrete="C50", bottom=[(4, 20, 50)], tendons=[(1112, 1050, 1200, 70)], sigma_pe=1100.0)
    a, b = jtg.rect(500, 1200, **kw), jtg.rect(500, 1200, external=[(1e-9, 1000, 1000)], **kw)
    for z in (0.0, 1200.0):
        assert float(b.prestress_stress(z, 80.0)) == pytest.approx(float(a.prestress_stress(z, 80.0)), rel=1e-9)
    assert jtg.cracking_moment(b)[0] == pytest.approx(jtg.cracking_moment(a)[0], rel=1e-9)


def test_cracking_moment_mixed():
    """hand calculation with 6.5.2-7, -8 (p66): σpc as printed, from Np0 on the net section for post-tensioned members
    (the external tendon at σpe,ex in Np0, program convention); W0 and S0 of the transformed section"""
    s = jtg.rect(500, 1200, concrete="C50", bottom=[(4, 20, 50)], tendons=[(1112, 1050, 1200, 70)], sigma_pe=1100.0,
                 external=[(834, 1000, 1000)])
    Ec = 3.45e4
    pts = [((2e5 / Ec - 1) * 4 * A20, 1150), (-math.pi * 35 ** 2, 1050), (1.95e5 / Ec * 1112, 1050)]
    A0, z0, I0 = _rect_props(500, 1200, pts)
    An, zn, In = _rect_props(500, 1200, pts[:2])
    N = 1200 * 1112 + 1000 * 834
    e = (1200 * 1112 * 1050 + 1000 * 834 * 1000) / N - zn
    spc = N / An + N * e * (1200 - zn) / In
    W0 = I0 / (1200 - z0)
    S0 = 500 * z0 * z0 / 2  # the part above the centroid: concrete only
    Mcr = (spc + 2 * S0 / W0 * 2.65) * W0
    assert jtg.cracking_moment(s)[0] == pytest.approx(Mcr / 1e6, rel=1e-9)


# ── chapter 6, 7 ──────────────────────────────────────────────────────────
def test_crack_resistance_external_only_as_a_load():
    """independent path: an external-only member is the reinforced concrete section under a load P at the tendon
    (σ = P/A0 + P e y / I0 with prestress.concrete_stress on the RC section's transformed properties)"""
    rc = jtg.rect(600, 1500, concrete="C50", bottom=[(6, 25, 60)], top=[(4, 16, 45)])
    s = _rect_ex()
    A0, z0, I0 = rc.props("transformed")
    P = 1000.0 * 2780
    r = jtg.crack_resistance(s, 900.0, 700.0, pc_class="A")
    for name, z in (("top", 0.0), ("bottom", 1500.0)):
        assert r.details[name]["sigma_pc"] == pytest.approx(ps.concrete_stress(P, 1300 - z0, z - z0, A0, I0), rel=1e-12)


def test_service_class_B_cracked_equilibrium():
    """physics: the cracked section of 7.1.4 / appendix J under Np0 (internal at σp0, external at σpe,ex) and Mk:
    the concrete compression (linear, zero at x) and the bonded steel at αE σ, integrated by hand, balance Np0 and its
    moment Mk + Np0 hps about the top (commentary eq. 7-1, p205); the external tendon is in the load, not in the section"""
    s = jtg.rect(500, 1200, concrete="C50", bottom=[(6, 25, 50)], top=[(4, 12, 40)], tendons=[(834, 1100, 1250, 70)],
                 sigma_pe=1120.0, external=[(834, 1050, 1000)])
    Mk = 1500.0
    r = jtg.service_stresses(s, Mk, pc_class="B")
    d = r.details
    x, Acr, Icr, c = d["x"], d["Acr"], d["Icr"], d["c"]
    N = 1250 * 834 + 1000 * 834
    assert d["Np0"] == pytest.approx(N / 1e3, rel=1e-12)
    sig = lambda z: N / Acr + N * d["e0N"] * (c - z) / Icr
    assert sig(x) == pytest.approx(0.0, abs=1e-9 * sig(0.0)) and r.demand == pytest.approx(sig(0.0))
    Fc = 500 * sig(0.0) * x / 2
    Mc = 500 * sig(0.0) * x * x / 6  # about the top fibre
    aS, aP = 2e5 / 3.45e4, 1.95e5 / 3.45e4
    steel = [(aS * 6 * A25, 1150), (aS * 4 * A12, 40), (aP * 834, 1100)]
    F = Fc + sum(a * sig(dd) for a, dd in steel)
    M = Mc + sum(a * sig(dd) * dd for a, dd in steel)
    assert F == pytest.approx(N, rel=1e-9)
    assert -M == pytest.approx(Mk * 1e6 - N * d["hps"], rel=1e-9)  # tension + below: Mk = Np0 (hps + eN)
    assert d["hps"] == pytest.approx((1250 * 834 * 1100 + 1000 * 834 * 1050) / N, rel=1e-12)


A12 = math.pi * 12 ** 2 / 4


def test_class_B_external_as_stiffless_internal_tendon():
    """physics: an external tendon is a force without stiffness, i.e. the limit of an internal tendon of vanishing area
    a carrying the same force (σp0 = σpe = P/a). Crack width (6.4.4-11) and the cracked-section stresses (7.1.4) of the
    mixed member and of that limit agree."""
    kw = dict(concrete="C50", bottom=[(6, 25, 50)], top=[(4, 12, 40)])
    P = 1000.0 * 834
    ex = jtg.rect(500, 1200, tendons=[(834, 1100, 1250, 70)], sigma_pe=1120.0, external=[(834, 1050, 1000)], **kw)
    a = 1e-6
    lim = jtg.rect(500, 1200, tendons=[(834, 1100, 1250, 70), (a, 1050, P / a, 1e-6)], sigma_pe=[1120.0, P / a], **kw)
    # Ms = 1600: the cracked section's axis (appendix J) above the bonded steel (at 1300 it lies below it: Wcr = 0)
    w1 = jtg.crack_width(ex, 1600.0, 900.0, pc_class="B", tendon_bundles=[(1, 6, 15.2)])
    w2 = jtg.crack_width(lim, 1600.0, 900.0, pc_class="B", tendon_bundles=[(1, 6, 15.2)])
    assert w1.demand > 0 and w1.demand == pytest.approx(w2.demand, rel=1e-6)
    assert w1.details["sigma_ss"] == pytest.approx(w2.details["sigma_ss"], rel=1e-6)
    s1, s2 = jtg.service_stresses(ex, 1500.0, pc_class="B"), jtg.service_stresses(lim, 1500.0, pc_class="B")
    assert s1.demand == pytest.approx(s2.demand, rel=1e-6)
    assert s1.details["tendon_stress"][0] == pytest.approx(s2.details["tendon_stress"][0], rel=1e-6)


def test_external_only_class_B_conventions():
    """program conventions: without internal tendons, E = Es (σss is the bars' increment) and table 6.4.2 takes the
    strand column (0.10 mm in environment I) since the member is prestressed with strand"""
    s = jtg.rect(500, 1200, concrete="C50", bottom=[(6, 25, 50)], top=[(4, 12, 40)], external=[(1668, 1050, 1000)])
    w = jtg.crack_width(s, 1400.0, 900.0, pc_class="B")
    assert w.details["E"] == 2.0e5 and w.resistance == 0.10 and w.demand > 0


def test_principal_stress_bent_external():
    """hand calculation with 6.3.3-5 (p61) at the centroid: τ = V S0/(b I0) − σ″pe Apb sinθp Sn/(b In) for a bent
    internal tendon (net section, as printed) and the same term on the transformed section for a bent external tendon
    (its section after grouting, 6.1.5 item 2; program convention)"""
    s = _mixed()
    _, z0, I0 = s.props("transformed")
    _, _, In = s.props("net")
    S0, Sn = s.first_moment(z0), s.first_moment(z0, "net")
    V, Pv = 600.0e3, 1000 * 834 * math.sin(math.radians(5))
    pts = jtg.principal_stresses(s, 0.0, 600.0, depths=[z0], bent_external=[(834, 5.0, 1000.0)])
    assert pts[0]["tau"] == pytest.approx(V * S0 / (500 * I0) - Pv * S0 / (500 * I0), rel=1e-12)
    internal = jtg.principal_stresses(s, 0.0, 600.0, depths=[z0], bent_tendons=[(834, 5.0, 1000.0)])
    assert internal[0]["tau"] == pytest.approx(V * S0 / (500 * I0) - Pv * Sn / (500 * In), rel=1e-12)
    mean = (1000 * 834 + 900 * 278) / (834 + 278)  # σpe,ex by default: the section's external tendons, by area
    dflt = jtg.principal_stresses(s, 0.0, 600.0, depths=[z0], bent_external=[(834, 5.0)])
    assert dflt[0]["tau"] == pytest.approx((V - Pv * mean / 1000) * S0 / (500 * I0), rel=1e-12)


def test_transient_before_grouting():
    """hand calculation with 7.2.8 and 6.1.6-7 before grouting (6.1.7 note 3): prestress and load on the net section"""
    s = _mixed()
    An, zn, In = s.props("net")
    N, zN = s.prestress_resultant("pe")
    M = 300.0
    top = N / An + N * (zN - zn) * (0 - zn) / In + M * 1e6 * zn / In
    r = jtg.transient_pc(s, M, fcu=45, grouted=False)
    assert r.details["top"] == pytest.approx(top, rel=1e-12)
    g = jtg.transient_pc(s, M, fcu=45, grouted=True)  # after grouting: 6.1.5 item 2, internal net, external transformed
    A0, z0, I0 = s.props("transformed")
    Nx = 1000 * 834 + 900 * 278
    zx = (1000 * 834 * 1000 + 900 * 278 * 200) / Nx
    Ni = N - Nx
    zi = (N * zN - Nx * zx) / Ni
    top = Ni / An + Ni * (zi - zn) * (0 - zn) / In + Nx / A0 + Nx * (zx - z0) * (0 - z0) / I0 + M * 1e6 * z0 / I0
    assert g.details["top"] == pytest.approx(top, rel=1e-12)


def test_xi_b_readings():
    """table 5.2.1 (p23): note 1 takes the smallest ξb of the kinds of steel in the tension zone, so external strand
    there gives 0.40 (default); note 2 defines ξb by the tension steel reaching its design strength, which the external
    tendon at σpe,ex never does: xi_b_bonded_only=True keeps the bars' 0.53. With x/h0 = 0.455 the two disagree."""
    s = jtg.rect(400, 1000, concrete="C50", bottom=[(6, 28, 60)], external=[(2500, 900, 1000)])
    T = FSD * 6 * A28 + 1000 * 2500
    x, h0 = T / (FCD * 400), (FSD * 6 * A28 * 940 + 1000 * 2500 * 900) / T
    lit, bonded = jtg.flexure(s, 1.0, pc_class="A"), jtg.flexure(s, 1.0, pc_class="A", xi_b_bonded_only=True)
    assert lit.details["x"] == pytest.approx(x, rel=1e-9) and 0.40 < x / h0 < 0.53
    assert lit.details["xi_b"] == 0.40 and any("over-reinforced" in f for f in lit.failed)
    assert bonded.details["xi_b"] == 0.53 and not any("over-reinforced" in f for f in bonded.failed)
    assert lit.details["xi_b_alternative"] == {"reading": "bonded steel only (note 2)", "xi_b": 0.53,
                                               "x": pytest.approx(x), "over_reinforced": False}
    assert bonded.details["xi_b_alternative"]["over_reinforced"] is True
    assert lit.resistance == pytest.approx(bonded.resistance, rel=1e-12)


def test_minimum_bars_9_1_13_bonded_h0():
    """9.1.13 (p102): bars ≥ 0.003 b h0 in partially prestressed members, h0 of the bonded steel (program convention):
    0.003 × 600 × 1440 = 2592 mm² > As = 2454 mm², although 0.003 b h0 with the external tendon in h0 would pass"""
    s = jtg.rect(600, 1500, concrete="C50", bottom=[(5, 25, 60)], external=[(2780, 1000, 1000)])
    r = jtg.flexure(s, 1.0, pc_class="A")
    assert r.details["h0"] < 1100 and any("0.003 b h0 = 2592" in f for f in r.failed)


def _stiffless_pair(d_in=1100.0, d_ex=1050.0):
    """a mixed section and its limit with the external tendon as an internal one of vanishing area and the same force"""
    kw = dict(concrete="C50", bottom=[(6, 25, 50)], top=[(4, 12, 40)])
    P, a = 1000.0 * 834, 1e-6
    ex = jtg.rect(500, 1200, tendons=[(834, d_in, 1250, 70)], sigma_pe=1120.0, external=[(834, d_ex, 1000)], **kw)
    lim = jtg.rect(500, 1200, tendons=[(834, d_in, 1250, 70), (a, d_ex, P / a, 1e-6)], sigma_pe=[1120.0, P / a], **kw)
    return ex, lim


def test_temperature_and_torsion_with_external_tendons():
    """physics, as above: a force without stiffness. Class B crack width and cracked-section stresses with the
    temperature gradient (D60 4.3.12, 3362 appendix D), and the prestress term of torsion (5.5.1, Np0 with the external
    tendon at σpe,ex) agree with the internal tendon of vanishing area"""
    ex, lim = _stiffless_pair()
    grad = jtg.temperature_gradient(1200)
    w = [jtg.crack_width(s, 1300.0, 900.0, pc_class="B", tendon_bundles=[(1, 6, 15.2)],
                         temperature=jtg.thermal_stresses(s, grad)) for s in (ex, lim)]
    assert w[0].demand > 0 and w[0].demand == pytest.approx(w[1].demand, rel=1e-6)
    v = [jtg.service_stresses(s, 1500.0, pc_class="B", temperature=jtg.thermal_stresses(s, grad)) for s in (ex, lim)]
    assert v[0].demand == pytest.approx(v[1].demand, rel=1e-6)
    ex, lim = _stiffless_pair(700.0, 650.0)  # near the centroid: ep0 ≤ h/6, the prestress term counts
    t = [jtg.torsion(s, 150.0, 300.0, stirrups=(2, 12, 100), Ast=6000, core=(400, 1100), pc_class="A") for s in (ex, lim)]
    assert t[0].resistance == pytest.approx(t[1].resistance, rel=1e-6)
    assert any("0.05 Np0" in n for n in t[0].notes)


def test_crack_width_limit_column():
    """table 6.4.2 (p62): members using wire or strand take the second column, internal or external (program reading
    of the column heading); threaded bars alone and reinforced concrete the first"""
    bar = dict(concrete="C50", bottom=[(6, 25, 50)], tendons=[(1000, 1100, 600, 60)], tendon="bar930", sigma_pe=550.0)
    assert jtg.crack_width_limit("I", jtg.rect(500, 1200, **bar)) == 0.20
    assert jtg.crack_width_limit("I", jtg.rect(500, 1200, external=[(834, 1050, 1000)],
                                               external_tendon="strand1860", **bar)) == 0.10
    assert jtg.crack_width_limit("I", jtg.rect(500, 1200, concrete="C50", external=[(834, 1050, 1000)])) == 0.10
    assert jtg.crack_width_limit("I", jtg.rect(500, 1200, concrete="C50", bottom=[(6, 25, 50)])) == 0.20


# ── 9.4 detailing ─────────────────────────────────────────────────────────
def test_detailing_9_4():
    """printed values: radius ≥ 4 m (wires ≤ 5 mm) or 6 m (9.4.23 → 9.4.10, p111, p114); hoops ≥ 20 mm at ≤ 100 mm
    (9.4.25); free length ≤ 8.0 m (9.4.26); anchorage crossbeam ≥ 1000 mm (9.4.27)"""
    assert jtg.deviation_radius_min(5.0) == 4000 and jtg.deviation_radius_min(5.2) == 6000
    ok = jtg.external_detailing(wire_d=5.0, radius=4000, free_length=8000, hoop_d=20, hoop_spacing=100, crossbeam=1000)
    assert ok.ok and ok.demand == pytest.approx(1.0)
    for kw in (dict(wire_d=5.2, radius=5999), dict(free_length=8001), dict(hoop_d=18), dict(hoop_spacing=110),
               dict(crossbeam=950)):
        assert not jtg.external_detailing(**kw).ok
    with pytest.raises(ValueError):
        jtg.external_detailing(radius=5000)


# ── secondary effects of a polygonal external tendon ──────────────────────
def test_secondary_moment_polygonal_tendon_force_method():
    """closed form: the force method on two equal spans with the middle support released. Primary moment M1 = −P cos θ e
    (the tendon straight between deviators at 1/3 points and over the pier, anchored at the centroid); the redundant
    X = ∫M1 m dx / ∫m² dx with m the moment of a unit load at mid-length; secondary moment at the pier −X L/2"""
    L, n, P = 30.0, 30, 5000.0
    m = Model()
    xs = np.linspace(0, 2 * L, 2 * n + 1)
    ns = [m.node((x, 0, 0)) for x in xs]
    bs = [m.beam(a, b, Section(E=3e7, A=1.0, Iy=0.1, Iz=0.1, J=0.1, mass=2.0)) for a, b in zip(ns[:-1], ns[1:])]
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    m.fix(ns[n], (0, 1, 1, 1, 0, 0))
    m.fix(ns[-1], (0, 1, 1, 1, 0, 0))
    kink_x, kink_e = [0, 10, 20, 30, 40, 50, 60], [0.0, 1.0, 1.0, -0.5, 1.0, 1.0, 0.0]
    e = np.interp(xs, kink_x, kink_e)
    tot, pri, sec = effects(Solver(m), Tendon(bs, e, P), chain_sections(bs), "My")
    num = den = 0.0
    for k in range(2 * n):  # Simpson on each element: exact for the products of linear functions
        x0, x1 = xs[k], xs[k + 1]
        cos = math.cos(math.atan((e[k + 1] - e[k]) / (x1 - x0)))
        M1 = lambda x: -P * cos * np.interp(x, xs, e)
        mu = lambda x: min(x, 2 * L - x) / 2
        xm = (x0 + x1) / 2
        num += (x1 - x0) / 6 * (M1(x0) * mu(x0) + 4 * M1(xm) * mu(xm) + M1(x1) * mu(x1))
        den += (x1 - x0) / 6 * (mu(x0) ** 2 + 4 * mu(xm) ** 2 + mu(x1) ** 2)
    X = num / den
    assert sec[2 * n - 1] == pytest.approx(-X * L / 2, rel=1e-6)
    assert pri[2 * n - 1] == pytest.approx(0.5 * P * math.cos(math.atan(0.15)), rel=1e-9)


# ── not covered ───────────────────────────────────────────────────────────
def test_not_covered():
    with pytest.raises(NotImplementedError):
        jtg.eccentric(_rect_ex(), 3000.0, 500.0)
    with pytest.raises(NotImplementedError):
        jtg.rect(500, 1000, concrete="C50", tendons=[(500, 900, 1100)], pretensioned=True, external=[(500, 800, 1000)])
    with pytest.raises(ValueError):
        jtg.rect(500, 1000, concrete="C50", external=[(500, 800, 600)], external_tendon="bar930")
    with pytest.raises(NotImplementedError):
        jtg.composite(_rect_ex(), b=2000, t=200, concrete="C40")
    with pytest.raises(NotImplementedError):
        jtg.axial(_rect_ex(), 1000.0)
