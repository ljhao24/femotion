"""Prestressed concrete (JTG 3362-2018) and tendon effects on frames, against hand calculations and closed forms."""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import prestress as ps
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import chain_sections
from femotion.frame.prestress import Tendon, effects

A16 = math.pi * 16 ** 2 / 4


# ── materials ───────────────────────────────────────────────────────────────
def test_tendon_materials():
    t = jtg.tendon("strand", 1860)
    assert (t.fpd, t.fpd_c, t.Ep) == (1260.0, 390.0, 1.95e5)
    assert jtg.tendon("wire1570").fpd == 1070.0 and jtg.tendon("wire", 1860).Ep == 2.05e5
    assert jtg.tendon("bar", 930).fpd == 770.0 and jtg.tendon("bar930").fpd_c == 400.0
    assert t.sigma_con_max() == pytest.approx(0.75 * 1860)
    assert t.sigma_con_max(external=True) == pytest.approx(0.70 * 1860)
    assert jtg.tendon("bar", 785).sigma_con_max(overstress=True) == pytest.approx(0.90 * 785)
    for g, v in [(50, 0.40), (60, 0.38), (70, 0.36), (80, 0.35)]:
        assert jtg.xi_b(jtg.concrete(g), t) == v
    with pytest.raises(ValueError):
        jtg.xi_b(jtg.concrete(80), jtg.tendon("bar", 930))


# ── flexure ─────────────────────────────────────────────────────────────────
def test_rectangle_tendons_only():
    s = jtg.rect(500, 1000, concrete="C50", tendons=[(1400, 900, 1100, 70)])
    x = 1260 * 1400 / (22.4 * 500)
    r = jtg.flexure(s, 100.0, pc_class="full")
    assert r.details["x"] == pytest.approx(x, rel=1e-9) and r.details["xi_b"] == 0.40
    assert r.resistance == pytest.approx(1260 * 1400 * (900 - x / 2) / 1e6, rel=1e-9)


def test_tee_bars_and_tendons_in_web_5_2_3():
    As, Ap = 6 * A16, 2780.0
    s = jtg.tee(200, 1600, 1600, 80, concrete="C50", bottom=[(6, 16, 50)], tendons=[(Ap, 1470, 1150, 90)])
    F = 330 * As + 1260 * Ap
    h0 = (330 * As * 1550 + 1260 * Ap * 1470) / F  # resultant of As and Ap (5.2.2: a)
    assert F > 22.4 * 1600 * 80  # 5.2.3-1 not met: x in the web
    x = (F / 22.4 - 1400 * 80) / 200  # 5.2.3-3
    Mu = 22.4 * (200 * x * (h0 - x / 2) + 1400 * 80 * (h0 - 40))  # 5.2.3-2
    r = jtg.flexure(s, 100.0, pc_class="A")
    assert r.details["h0"] == pytest.approx(h0, rel=1e-12)
    assert r.details["xi_b"] == 0.40  # the smaller of 0.53 (HRB400) and 0.40 (strand), table 5.2.1 note 1
    assert r.details["x"] == pytest.approx(x, rel=1e-9)
    assert r.resistance == pytest.approx(Mu / 1e6, rel=1e-9)


def _rect_with_top_tendon(Ap, sp0_top):
    return jtg.rect(500, 1200, concrete="C50", top=[(4, 16, 50)], tendons=[(Ap, 1100, 1100, 90), (400, 100, sp0_top, 60)])


def test_compression_zone_tendon_in_tension_5_2_2():
    """(f'pd − σ'p0) < 0: A'p carries tension (σ'p0 − f'pd) in 5.2.2-1/-2; x ≥ 2a's applies"""
    s = _rect_with_top_tendon(2000, 900)
    k = (390 - 900) * 400  # (f'pd − σ'p0) A'p
    x = (1260 * 2000 - 330 * 4 * A16 - k) / (22.4 * 500)
    assert x >= 100
    Mu = 22.4 * 500 * x * (1100 - x / 2) + 330 * 4 * A16 * (1100 - 50) + k * (1100 - 100)
    r = jtg.flexure(s, 100.0, pc_class="full")
    assert r.details["x"] == pytest.approx(x, rel=1e-9)
    assert r.resistance == pytest.approx(Mu / 1e6, rel=1e-9)


def test_x_below_2a_with_tendon_in_tension_5_2_4_2():
    s = _rect_with_top_tendon(300, 900)
    r = jtg.flexure(s, 10.0, pc_class="full")
    assert r.details["x"] < 100 and r.details["a_ref"] == pytest.approx(50)
    Mu = 1260 * 300 * (1200 - 100 - 50) - (390 - 900) * 400 * (100 - 50)  # 5.2.4-2
    assert r.resistance == pytest.approx(Mu / 1e6, rel=1e-9)
    assert r.details["M_singly_reinforced"] == 0.0  # no singly reinforced alternative with compression-zone tendons


def test_x_below_2a_with_tendon_in_compression_5_2_4_1():
    s = _rect_with_top_tendon(300, 200)  # f'pd − σ'p0 = 190 > 0
    Fs, Fp = 330 * 4 * A16, 190 * 400
    a1 = (Fs * 50 + Fp * 100) / (Fs + Fp)  # a': resultant of A's and A'p
    r = jtg.flexure(s, 10.0, pc_class="full")
    assert r.details["a_ref"] == pytest.approx(a1)
    assert r.resistance == pytest.approx(1260 * 300 * (1200 - 100 - a1) / 1e6, rel=1e-9)  # 5.2.4-1


def test_cracking_moment_6_5_2():
    """pretensioned rectangle: Mcr = (σpc + γ ftk) W0, γ = 2 S0/W0, transformed section"""
    b, h, Ap, dp, sp0 = 400.0, 800.0, 1000.0, 700.0, 1200.0
    s = jtg.rect(b, h, concrete="C50", tendons=[(Ap, dp, sp0)], pretensioned=True)
    aE = 1.95e5 / 3.45e4
    A0 = b * h + (aE - 1) * Ap
    z0 = (b * h * h / 2 + (aE - 1) * Ap * dp) / A0
    I0 = b * h ** 3 / 12 + b * h * (z0 - h / 2) ** 2 + (aE - 1) * Ap * (dp - z0) ** 2
    N = sp0 * Ap
    sig = N / A0 + N * (dp - z0) * (h - z0) / I0
    W0 = I0 / (h - z0)
    S0 = b * (h - z0) ** 2 / 2 + (aE - 1) * Ap * (dp - z0)
    Mcr = (sig + 2 * S0 / W0 * 2.65) * W0
    got, d = jtg.cracking_moment(s)
    assert got == pytest.approx(Mcr / 1e6, rel=1e-9)
    r = jtg.flexure(s, 100.0, pc_class="full")
    assert r.details["Mcr"] == pytest.approx(got) and (r.resistance >= got) == (not any("9.1.13" in f for f in r.warnings))


def test_partial_prestress_minimum_bars_9_1_13():
    s = jtg.tee(200, 1600, 1600, 150, concrete="C50", bottom=[(2, 12, 50)], tendons=[(2780, 1470, 1150, 90)])
    assert any("0.003" in f for f in jtg.flexure(s, 100.0, pc_class="A").warnings)
    assert not any("0.003" in f for f in jtg.flexure(s, 100.0, pc_class="full").warnings)
    assert any("pc_class" in n for n in jtg.flexure(s, 100.0).notes)


def test_prestressed_concrete_class_3_1_2():
    s = jtg.rect(500, 1000, concrete="C35", tendons=[(1400, 900, 1100, 70)])
    assert any("C40" in f for f in jtg.flexure(s, 10.0, pc_class="full").warnings)


# ── shear ───────────────────────────────────────────────────────────────────
def test_shear_alpha2_and_tendon_ratio():
    s = jtg.tee(200, 1600, 1600, 150, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1470, 1150, 90)])
    r = jtg.shear(s, 500.0, (2, 10, 200), pc_class="A")
    h0 = r.details["h0"]
    P = min(100 * (6 * A16 + 2780) / (200 * h0), 2.5)
    Vcs = 1.25 * 1.1 * 0.45e-3 * 200 * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(50) * (2 * math.pi * 25 / (200 * 200)) * 330)
    assert r.details["alpha2"] == 1.25 and r.resistance == pytest.approx(Vcs, rel=1e-9)
    assert jtg.shear(s, 500.0, (2, 10, 200), pc_class="B").details["alpha2"] == 1.0
    # hogging: tendons below the axis now act in the direction of the (hogging) external moment
    sh = jtg.tee(200, 1600, 1600, 150, concrete="C50", bottom=[(6, 16, 50)], top=[(10, 16, 50)], tendons=[(2780, 1470, 1150, 90)])
    assert jtg.shear(sh, 500.0, (2, 10, 200), pc_class="A", hogging=True, alpha1=0.9).details["alpha2"] == 1.0
    with pytest.raises(ValueError):
        jtg.shear(s, 500.0, (2, 10, 200))
    v = jtg.shear(s, 500.0, (2, 10, 200), pc_class="A", bent_tendons=[(1390, 8.0)],
                  vertical=(2 * 804.2, 500, jtg.tendon("bar", 785)))
    assert v.details["Vpb"] == pytest.approx(0.75e-3 * 1260 * 1390 * math.sin(math.radians(8)))
    assert v.details["rho_pv"] == pytest.approx(2 * 804.2 / (500 * 200))


def test_eccentric_xi_b_5_3_3():
    s = jtg.rect(800, 1000, concrete="C50", tendons=[(1000, 900, 1000, 70), (1000, 100, 1000, 70)])
    c = jtg.concrete(50)
    xb = 0.8 / (1 + 0.002 / 0.0033 + (1260 - 1000) / (1.95e5 * 0.0033))  # 5.3.3-2
    assert s.xi_b(eccentric=True) == pytest.approx(xb)
    r = jtg.eccentric(s, 8000.0, 2000.0)
    assert r.details["xi_b"] == pytest.approx(xb) and r.resistance > 0


# ── losses (6.2, appendices C, G, H) ────────────────────────────────────────
def test_friction_and_anchorage():
    assert ps.friction_loss(1395, 0.25, 0.2, 0.0015, 20) == pytest.approx(1395 * (1 - math.exp(-(0.05 + 0.03))))
    assert ps.anchorage_loss_straight(6, 20000, 1.95e5) == pytest.approx(58.5)
    # Appendix G: the area between the stress lines before and after anchoring equals ΣΔl·Ep in both cases
    for l, s_end in [(30000, 1250.0), (6000, 1370.0)]:
        x = np.linspace(0, l, 200001)
        loss, lf = ps.anchorage_loss_reverse(1395, s_end, l, 6, 1.95e5, x)
        area = float(((loss[1:] + loss[:-1]) / 2 * np.diff(x)).sum())  # trapezoid rule (np.trapezoid needs NumPy 2)
        assert area == pytest.approx(6 * 1.95e5, rel=1e-6)
        assert (lf <= l) == (l == 30000)


def test_other_losses():
    t = jtg.tendon("strand", 1860)
    assert ps.temperature_loss(60, 20) == 80
    assert ps.elastic_shortening_simplified(6, 5.65, 2.0) == pytest.approx(2.5 * 5.65 * 2.0)
    assert ps.elastic_shortening_post(5.65, [1.0, 2.0]) == pytest.approx(16.95)
    assert ps.relaxation_loss(t, 1300) == pytest.approx(0.3 * (0.52 * 1300 / 1860 - 0.26) * 1300)
    assert ps.relaxation_loss(t, 1300, low_relaxation=False, overstress=True) == pytest.approx(0.9 * (0.52 * 1300 / 1860 - 0.26) * 1300)
    assert ps.relaxation_loss(t, 900) == 0.0
    assert ps.relaxation_loss(jtg.tendon("bar", 930), 0, sigma_con=790) == pytest.approx(39.5)
    assert ps.relaxation_ratio(10) == 0.61 and ps.relaxation_ratio(50) == 1.0
    assert ps.loss_stages(post=True, l1=1, l2=2, l4=4, l5=5, l6=6) == (7, 11)
    assert ps.loss_stages(post=False, l2=2, l3=3, l4=4, l5=5, l6=6) == (11.5, 8.5)
    assert ps.sigma_p0(1395, 250, post=True, alpha_EP=5.65, sigma_pc=10) == pytest.approx(1201.5)
    assert ps.sigma_p0(1395, 250, post=False, sigma_l4=30) == pytest.approx(1175)


def test_appendix_c_formulas_reproduce_tables():
    """at C40 the formulas of C.1.1 and C.2.1 give tables C.1.2 and C.2.2 (RH = 55 %, 80 %); table C.1.2 truncates
    (0.52977 → 0.529), and three cells of table C.2.2 are 0.01 above the rounded formula value"""
    c = jtg.concrete(40)
    assert ps.nominal_shrinkage(c, 55, "formula") == pytest.approx(0.529e-3, abs=1e-6)
    assert ps.nominal_shrinkage(c, 80, "formula") == pytest.approx(0.310e-3, abs=1e-6)
    off = {(55, 3, 300), (55, 7, 200), (80, 28, 300)}
    for RH in (55, 80):
        for t0 in (3, 7, 14, 28, 60, 90):
            for h in (100, 200, 300, 600):
                f = round(ps.nominal_creep(c, RH, h, t0, "formula"), 2)
                tab = ps.nominal_creep(c, RH, h, t0, "table")
                assert tab == pytest.approx(f + (0.01 if (RH, t0, h) in off else 0.0), abs=1e-9), (RH, t0, h)
    assert ps.nominal_shrinkage(jtg.concrete(55), 80) == pytest.approx(0.310e-3 * math.sqrt(32.4 / 35.5))
    with pytest.raises(ValueError):
        ps.nominal_creep(c, 55, 80, 28)  # table below h = 100 mm
    with pytest.raises(ValueError):
        ps.creep_coefficient(c, 55, 200, 3650, 28, fly_ash=5)  # table C.2.4 starts at 10 %


def test_shrinkage_creep_and_loss():
    c = jtg.concrete(50)
    bs = lambda d: math.sqrt(d / (350 * 2.5 ** 2 + d))
    # table method (C.1.3, C.2.3; the default): εcs0 from table C.1.2, φ0 from table C.2.2, βH with RH = 80 %
    e = ps.shrinkage_strain(c, 75, 250, 3650, 7)
    assert e == pytest.approx(0.310e-3 * (bs(3647) - bs(4)))
    phi = ps.creep_coefficient(c, 75, 250, 3650, 7)
    phi0 = (2.26 + 2.19) / 2  # table C.2.2, 70–99 %, t0 = 7 d, h = 250 mm
    bH = min(150 * (1 + (1.2 * 0.80) ** 18) * 2.5 + 250, 1500)
    assert phi == pytest.approx(phi0 * (3643 / (bH + 3643)) ** 0.3)
    # formula method (C.1.1, C.2.1) with the class's own fcm and the actual RH
    ef = ps.shrinkage_strain(c, 75, 250, 3650, 7, method="formula")
    assert ef == pytest.approx(ps.nominal_shrinkage(c, 75, "formula") * (bs(3647) - bs(4)))
    pf = ps.creep_coefficient(c, 75, 250, 3650, 7, method="formula")
    bHf = min(150 * (1 + (1.2 * 0.75) ** 18) * 2.5 + 250, 1500)
    assert pf == pytest.approx(ps.nominal_creep(c, 75, 250, 7, "formula") * (3643 / (bHf + 3643)) ** 0.3)
    rho, rps = ps.steel_ratios(6e5, 2e11, 3000, 500, 1000, 550)
    assert rho == pytest.approx(4000 / 6e5)
    l6 = ps.creep_shrinkage_loss(1.95e5, e, 5.65, 12.0, phi, rho, rps)
    assert l6 == pytest.approx(0.9 * (1.95e5 * e + 5.65 * 12 * phi) / (1 + 15 * rho * rps))
    with pytest.raises(ValueError):
        ps.creep_shrinkage_loss(1.95e5, e, 5.65, 30.0, phi, rho, rps, fcu_transfer=50)


def test_pretensioned_ends_5_1_6_and_6_1_8():
    t = jtg.tendon("strand", 1860)
    assert ps.anchorage_length(t, jtg.concrete(50), 15.2) == pytest.approx(120 * 15.2)
    assert ps.anchorage_length(t, jtg.concrete(70), 15.2) == pytest.approx(105 * 15.2)
    assert ps.anchorage_length(jtg.tendon("strand", 1720), jtg.concrete(40), 15.2) == pytest.approx(130 * 15.2 * 1170 / 1260)
    assert ps.transfer_length(t, 47.5, 15.2, 1100) == pytest.approx(62 * 15.2 * 1.1)
    la = ps.anchorage_length(t, jtg.concrete(50), 15.2)
    assert ps.fpd_in_anchorage(1260, la / 2, la) == pytest.approx(630)
    assert ps.fpd_in_anchorage(1260, 10, la, start=100) == 0.0
    assert t.with_fpd(630).fpd == 630 and t.with_fpd(630).fpk == 1860
    with pytest.raises(ValueError):
        ps.anchorage_length(jtg.tendon("bar", 930), jtg.concrete(50), 25)


def test_cracking_moment_sigma_l6_zones():
    """6.1.7-1: σl6 on the tension-zone bars; σ'l6 on the compression-zone bars only with tendons there (note 1)"""
    base = dict(concrete="C50", bottom=[(4, 20, 50)], top=[(4, 20, 50)], tendons=[(2000, 1100, 1100, 90)])
    s0 = jtg.rect(500, 1200, **base)
    s1 = jtg.rect(500, 1200, sigma_l6=(100.0, 80.0), **base)
    s2 = jtg.rect(500, 1200, sigma_l6=(100.0, 0.0), **base)
    m0, m1, m2 = (jtg.cracking_moment(s)[1]["Np0"] for s in (s0, s1, s2))
    assert m1 == pytest.approx(m2) and m1 == pytest.approx(m0 - 100 * 4 * math.pi * 100 / 1e3)  # A'p = 0: σ'l6 ignored
    s3 = jtg.rect(500, 1200, concrete="C50", bottom=[(4, 20, 50)], top=[(4, 20, 50)], sigma_l6=(100.0, 80.0),
                  tendons=[(2000, 1100, 1100, 90), (400, 100, 1000, 60)])
    assert jtg.cracking_moment(s3)[1]["Np0"] == pytest.approx((1100 * 2000 + 1000 * 400 - 180 * 4 * math.pi * 100) / 1e3)


def test_anchorage_x_range():
    with pytest.raises(ValueError):
        ps.anchorage_loss_reverse(1395, 1250, 30000, 6, 1.95e5, 31000)


# ── secondary effects on frames ─────────────────────────────────────────────
def _girder(spans, n):
    m = Model()
    xs = [0.0]
    for L in spans:
        xs += list(np.linspace(xs[-1], xs[-1] + L, n + 1)[1:])
    ns = [m.node((x, 0, 0)) for x in xs]
    bs = [m.beam(a, b, Section(E=3e7, A=1.0, Iy=0.1, Iz=0.1, J=0.1, mass=2.0)) for a, b in zip(ns[:-1], ns[1:])]
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    k = 0
    for _ in spans:
        k += n
        m.fix(ns[k], (0, 1, 1, 1, 0, 0))
    return m, bs, np.array(xs)


def test_secondary_moments():
    P, e = 1000.0, 0.3
    m, bs, xs = _girder([20.0], 20)
    sv = Solver(m)
    tot, pri, sec = effects(sv, Tendon(bs, np.full(len(xs), e), P), chain_sections(bs), "My")
    assert np.allclose(tot, -P * e) and np.allclose(sec, 0, atol=1e-8)  # statically determinate: no secondary effect
    m, bs, xs = _girder([20.0, 20.0], 20)
    tot, pri, sec = effects(Solver(m), Tendon(bs, np.full(len(xs), e), P), chain_sections(bs), "My")
    assert sec[39] == pytest.approx(1.5 * P * e, rel=1e-9)  # straight tendon, two equal spans: 1.5 P e at the pier
    m, bs, xs = _girder([20.0], 40)
    tot, pri, sec = effects(Solver(m), Tendon(bs, 2.0 * xs * (20 - xs) / 400, P), chain_sections(bs), "My")
    assert tot[39] == pytest.approx(-P * 0.5, rel=1e-4) and np.abs(sec).max() < 1e-6


def test_concrete_stress_signs():
    """y and e downwards from the centroid, M sagging-positive, compression positive"""
    A, I = 1e6, 1e11
    assert ps.concrete_stress(1e6, 400, 500, A, I) == pytest.approx(1 + 2.0)  # tendon below: bottom compressed
    assert ps.concrete_stress(1e6, 400, -500, A, I) == pytest.approx(1 - 2.0)  # top fibre
    assert ps.concrete_stress(0, 0, -500, A, I, M=1e8) == pytest.approx(0.5)  # sagging moment compresses the top
