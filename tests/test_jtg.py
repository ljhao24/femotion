"""JTG D60-2015 actions and JTG 3362-2018 reinforced-concrete resistance against hand calculations."""
import math

import numpy as np
import pytest

from femotion.codes import jtg

A20, A25 = math.pi * 20 ** 2 / 4, math.pi * 25 ** 2 / 4


# ── materials and actions ───────────────────────────────────────────────────
def test_materials():
    c = jtg.concrete("C30")
    assert (c.fcd, c.ftd, c.fcuk, c.beta, c.eps_cu) == (13.8, 1.39, 30.0, 0.8, 0.0033)
    assert jtg.concrete(80).beta == pytest.approx(0.74) and jtg.concrete(80).eps_cu == pytest.approx(0.003)
    assert jtg.rebar("HRB500").fsv == 330.0 and jtg.rebar("HRB500").fsd_c == 400.0
    # table 5.2.1 up to C50
    for s, v in [("HPB300", 0.58), ("HRB400", 0.53), ("HRB500", 0.49)]:
        assert jtg.xi_b(jtg.concrete(50), jtg.rebar(s)) == v


def test_lane_load_and_factors():
    assert jtg.lane_load(5) == (10.5, 270.0)
    assert jtg.lane_load(50) == (10.5, 360.0)
    assert jtg.lane_load(20) == (10.5, 300.0)
    assert jtg.lane_load(20, shear=True)[1] == pytest.approx(360.0)
    assert jtg.lane_load(20, "II") == pytest.approx((7.875, 225.0))
    assert jtg.impact_factor(1.0) == 0.05 and jtg.impact_factor(20) == 0.45
    assert jtg.impact_factor(5.0) == pytest.approx(0.1767 * math.log(5) - 0.0157)
    assert jtg.transverse_reduction(1) == 1.2 and jtg.transverse_reduction(3) == 0.78
    assert jtg.longitudinal_reduction(100) == 1.0 and jtg.longitudinal_reduction(500) == 0.96
    assert jtg.design_lanes(12.0) == 2 and jtg.design_lanes(15.0, two_way=False) == 4
    assert sum(jtg.VEHICLE_AXLES) == 550


def test_basic_combination():
    hi, lo = jtg.basic_combination(np.array([100.0, -100.0]), (np.array([50.0, 50.0]), np.array([-20.0, -20.0])))
    assert np.allclose(hi, [1.2 * 100 + 1.4 * 50, -100 + 1.4 * 50])
    assert np.allclose(lo, [100 - 1.4 * 20, -1.2 * 100 - 1.4 * 20])
    hi, _ = jtg.basic_combination(np.array([0.0]), (np.array([10.0]), np.array([0.0])), others=[(np.array([4.0]), np.array([0.0]))])
    assert hi[0] == pytest.approx(14 + 0.75 * 1.4 * 4)


# ── flexure ─────────────────────────────────────────────────────────────────
def test_singly_reinforced_rectangle():
    s = jtg.rect(250, 500, concrete="C30", steel="HRB400", bottom=[(3, 20, 45)])
    r = jtg.flexure(s, 100.0)
    x = 330 * 3 * A20 / (13.8 * 250)
    assert r.details["x"] == pytest.approx(x, rel=1e-6)
    assert r.resistance == pytest.approx(330 * 3 * A20 * (455 - x / 2) / 1e6, rel=1e-6)
    assert r.ok and r.ratio == pytest.approx(100 / r.resistance)


def test_tee_neutral_axis_in_web():
    As = 6 * math.pi * 28 ** 2 / 4
    s = jtg.tee(200, 800, 600, 100, concrete="C30", steel="HRB400", bottom=[(6, 28, 70)])
    r = jtg.flexure(s, 500.0)
    x = (330 * As / 13.8 - 400 * 100) / 200
    Mu = 13.8 * (200 * x * (730 - x / 2) + 400 * 100 * (730 - 50))
    assert r.details["x"] == pytest.approx(x, rel=1e-6)
    assert r.resistance == pytest.approx(Mu / 1e6, rel=1e-6)


def test_tee_neutral_axis_in_flange():
    As = 4 * math.pi * 25 ** 2 / 4
    s = jtg.tee(200, 800, 1200, 120, concrete="C30", steel="HRB400", bottom=[(4, 25, 60)])
    r = jtg.flexure(s, 300.0)
    x = 330 * As / (13.8 * 1200)
    assert x < 120 and r.resistance == pytest.approx(330 * As * (740 - x / 2) / 1e6, rel=1e-6)


def test_compression_bars_below_2as():
    s = jtg.rect(250, 500, bottom=[(3, 20, 45)], top=[(3, 20, 40)])
    r = jtg.flexure(s, 100.0)
    assert r.details["x"] < 80 and r.notes
    about = 330 * 3 * A20 * (455 - 40)  # moments about the compression bars
    x1 = 330 * 3 * A20 / (13.8 * 250)
    single = 330 * 3 * A20 * (455 - x1 / 2)  # compression bars ignored
    assert r.resistance == pytest.approx(max(about, single) / 1e6, rel=1e-6)
    # a wide flange: x tiny, ignoring the top bars gives more
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(8, 28, 110)], top=[(8, 20, 50)])
    As = 8 * math.pi * 28 ** 2 / 4
    x = 330 * As / (18.4 * 1600)
    assert jtg.flexure(t, 1.0).resistance == pytest.approx(330 * As * (990 - x / 2) / 1e6, rel=1e-6)


def test_over_reinforced_and_minimum_steel():
    heavy = jtg.rect(250, 500, bottom=[(8, 32, 60)])
    assert any("over-reinforced" in f for f in jtg.flexure(heavy, 10.0).failed)
    light = jtg.rect(400, 1000, bottom=[(2, 12, 40)])
    r = jtg.flexure(light, 1.0)
    assert any("minimum" in f for f in r.warnings) and r.ok  # 9.1.12 warns (detailing does not fail the check)


def test_hogging_uses_top_bars():
    s = jtg.rect(250, 500, top=[(3, 20, 45)], bottom=[(2, 12, 40)])
    r = jtg.flexure(s, -100.0)
    assert r.details["h0"] == pytest.approx(455)
    assert "hogging" in r.name and r.resistance > 100


# ── shear ───────────────────────────────────────────────────────────────────
def test_shear_formula():
    s = jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 90)])
    r = jtg.shear(s, 600.0, stirrups=(2, 10, 150))
    As, h0 = 6 * math.pi * 32 ** 2 / 4, 1210.0
    P = 100 * As / (200 * h0)
    rho = 2 * math.pi * 25 / (150 * 200)
    Vcs = 1.1 * 0.45e-3 * 200 * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(40) * rho * 330)
    assert r.resistance == pytest.approx(Vcs, rel=1e-9) and r.details["alpha3"] == 1.1
    assert r.details["V_section_max"] == pytest.approx(0.51e-3 * math.sqrt(40) * 200 * h0)
    assert jtg.shear(s, 600.0, stirrups=(2, 6, 300)).warnings  # 0.094 % < 0.11 %


# ── compression ─────────────────────────────────────────────────────────────
def test_stability_factor():
    assert jtg.stability_factor(12 * 400, b=400) == 0.95
    assert jtg.stability_factor(9 * 400, b=400) == pytest.approx(0.99)
    assert jtg.stability_factor(5 * 1000, d=1000) == 1.0


def test_axial():
    s = jtg.rect(400, 400, bottom=[(2, 20, 40)], top=[(2, 20, 40)])
    r = jtg.axial(s, 1000.0, l0=4800)
    assert r.resistance == pytest.approx(0.9 * 0.95 * (13.8 * 160000 + 330 * 4 * A20) / 1e3)


def test_large_eccentricity_symmetric_rectangle():
    """symmetric bars, both yield: N = fcd b x and N e_s = fcd b x (h0 − x/2) + fsd' As' (h0 − as')"""
    As = 4 * A25
    s = jtg.rect(400, 600, bottom=[(4, 25, 40)], top=[(4, 25, 40)])
    e0 = 500.0
    es = e0 + 300 - 40
    a, b, K = 13.8 * 400 / 2, 13.8 * 400 * (es - 560), 330 * As * 520
    x = (-b + math.sqrt(b * b + 4 * a * K)) / (2 * a)
    r = jtg.eccentric(s, Nd=1000.0, Md=1000.0 * e0 / 1e3)  # no l0: η = 1
    assert r.details["case"] == "large eccentricity"
    assert r.resistance == pytest.approx(13.8 * 400 * x / 1e3, rel=1e-6)


def test_eta():
    s = jtg.rect(400, 600, bottom=[(4, 25, 40)], top=[(4, 25, 40)])
    e0, l0, h0 = 300.0, 9000.0, 560.0
    z1, z2 = min(0.2 + 2.7 * e0 / h0, 1), min(1.15 - 0.01 * l0 / 600, 1)
    assert jtg.eta_factor(s, e0, l0, h0) == pytest.approx(1 + (l0 / 600) ** 2 * z1 * z2 / (1300 * e0 / h0))
    assert jtg.eta_factor(s, e0, 2000.0, h0) == 1.0  # l0/i ≤ 17.5


def test_circle_code_formula_close_to_strain_method():
    s = jtg.circle(1200, 24, 25, 60, concrete="C30")
    for e0 in (100.0, 300.0, 800.0):
        Nd = 5000.0
        a = jtg.eccentric(s, Nd, Nd * e0 / 1e3, method="code").resistance
        b = jtg.eccentric(s, Nd, Nd * e0 / 1e3, method="strain").resistance
        assert a == pytest.approx(b, rel=0.06)


def test_circle_geometry():
    s = jtg.circle(1000, 12, 20, 50)
    assert s.area == pytest.approx(math.pi * 500 ** 2, rel=1e-12)
    assert s.zg == pytest.approx(500)
    assert s.I == pytest.approx(math.pi * 1000 ** 4 / 64, rel=1e-12)
    hol = jtg.circle(2000, 30, 25, 60, hollow=1400)
    assert hol.area == pytest.approx(math.pi * (1000 ** 2 - 700 ** 2), rel=1e-12)


# ── items from the review against the official text (shared review, 2026-10-11) ──
def test_xi_b_table_5_2_1():
    tab = {"HPB300": (0.58, 0.56, 0.54), "HRB400": (0.53, 0.51, 0.49), "HRB500": (0.49, 0.47, 0.46)}
    for s, (a, b, c) in tab.items():
        for g, v in [(30, a), (50, a), (55, b), (60, b), (65, c), (70, c)]:
            assert jtg.xi_b(jtg.concrete(g), jtg.rebar(s)) == v
    with pytest.raises(ValueError):
        jtg.xi_b(jtg.concrete(80), jtg.rebar("HRB400"))  # "—" in the table


def test_e0_minimum():
    s = jtg.rect(500, 500, bottom=[(4, 20, 40)], top=[(4, 20, 40)])
    r = jtg.eccentric(s, 2000.0, 0.0, l0=6000)
    assert r.details["e0"] == 20.0 and r.notes  # max(20, 500/30 = 16.7)
    assert r.resistance == pytest.approx(jtg.eccentric(s, 2000.0, 2000.0 * 0.020, l0=6000).resistance)
    big = jtg.rect(800, 900, bottom=[(6, 25, 50)], top=[(6, 25, 50)])
    assert jtg.eccentric(big, 3000.0, 0.0).details["e0"] == pytest.approx(30.0)  # h/30


def test_reverse_failure():
    """bars mostly on the compressed side, load close to the centroid: failure from the far face governs (5.3.4-4)"""
    As_c, As_t = 4 * A25, 2 * math.pi * 16 ** 2 / 4
    s = jtg.rect(400, 600, top=[(4, 25, 40)], bottom=[(2, 16, 40)])
    r = jtg.eccentric(s, 3000.0, 3000.0 * 0.025)
    e1 = 300 - 25 - 40
    Nr = (13.8 * 400 * 600 * (300 - 40) + 330 * As_t * (560 - 40)) / e1 / 1e3
    assert r.details["case"] == "small eccentricity"
    assert r.details["N_reverse"] == pytest.approx(Nr, rel=1e-9)
    assert r.resistance == pytest.approx(min(Nr, r.resistance), rel=1e-9) and "reverse" in " ".join(r.notes)


def test_compression_member_minimum_steel():
    s = jtg.rect(500, 500, bottom=[(2, 16, 40)], top=[(2, 16, 40)])  # 0.32 %
    assert any("0.5" in f for f in jtg.axial(s, 1000.0, 3000).warnings)
    assert any("0.5" in f for f in jtg.eccentric(s, 1000.0, 100.0).warnings)
    assert jtg.axial(s, 1000.0, 3000).ok


def test_axial_weak_axis_required_for_tee():
    t = jtg.tee(400, 1200, 1600, 200, bottom=[(6, 25, 60)], top=[(6, 25, 60)])
    with pytest.raises(ValueError):
        jtg.axial(t, 1000.0, l0=10000)
    assert jtg.axial(t, 1000.0, l0=10000, i_min=150.0).details["phi"] == pytest.approx(np.interp(10000 / 150, [62, 69], [0.81, 0.75]))


def test_circle_formula_needs_8_bars():
    s6 = jtg.circle(1000, 6, 28, 60)
    assert jtg.eccentric(s6, 3000.0, 300.0).clause == "JTG 3362-2018 5.3.4–5.3.9"  # auto falls back to the rule set
    with pytest.raises(ValueError):
        jtg.eccentric(s6, 3000.0, 300.0, method="code")


def test_reductions_and_lanes_review():
    assert jtg.longitudinal_reduction(150) == 1.0 and jtg.longitudinal_reduction(150.1) == 0.97
    # 3 lanes: 3 × 0.78 = 2.34 ≥ 2; 4 lanes 2.68; the two-lane floor only bites if a table value drops below
    assert jtg.LiveLoad(L0=30, lanes=3).factor() == pytest.approx(2.34)
    assert jtg.LiveLoad(L0=30, lanes=1).factor() == pytest.approx(1.2)


# ── items from writing docs/codes/JTG_3362-2018.md against the original pages ──
def test_compression_bars_take_fsd_prime_hrb500():
    """5.2.2-1 / 5.2.2-2: with x ≥ 2as' the compression bars carry f'sd (400 MPa for HRB500), not the 5.1.5-1 stress"""
    As, As_c = 6 * math.pi * 25 ** 2 / 4, 2 * math.pi * 16 ** 2 / 4
    s = jtg.rect(300, 600, concrete="C40", steel="HRB500", bottom=[(6, 25, 60)], top=[(2, 16, 40)])
    x = (415 * As - 400 * As_c) / (18.4 * 300)
    assert x >= 80
    r = jtg.flexure(s, 100.0)
    assert r.details["x"] == pytest.approx(x, rel=1e-9)
    assert r.resistance == pytest.approx((18.4 * 300 * x * (540 - x / 2) + 400 * As_c * (540 - 40)) / 1e6, rel=1e-9)


def test_stirrup_detailing_and_slab():
    s = jtg.rect(300, 700, concrete="C30", bottom=[(4, 25, 50)])
    assert any("diameter" in f for f in jtg.shear(s, 100.0, (4, 6, 100)).warnings)
    assert any("spacing" in f for f in jtg.shear(s, 100.0, (2, 10, 360)).warnings)  # h/2 = 350
    assert jtg.shear(s, 100.0, (2, 10, 360)).ok
    assert not jtg.shear(s, 100.0, (2, 10, 300)).warnings
    slab = jtg.rect(1000, 300, concrete="C30", bottom=[(8, 16, 40)])
    r = jtg.shear(slab, 200.0, slab=True)
    assert r.resistance == pytest.approx(1.25 * 0.50e-3 * 1.39 * 1000 * 260)
    with pytest.raises(ValueError):
        jtg.shear(s, 100.0)  # a beam without stirrups


def test_concrete_class_with_400_mpa_bars():
    s = jtg.rect(300, 600, concrete="C25", steel="HRB400", bottom=[(3, 20, 45)])
    assert any("3.1.2" in f for f in jtg.flexure(s, 10.0).warnings)
    ok = jtg.rect(300, 600, concrete="C25", steel="HPB300", bottom=[(3, 20, 45)])
    assert not any("3.1.2" in f for f in jtg.flexure(ok, 10.0).warnings)


def test_continuous_beam_frequencies():
    f1, f2 = jtg.continuous_beam_frequencies(20.0, 1e7, 10.0)
    k = math.sqrt(1e6) / (2 * math.pi * 400)
    assert (f1, f2) == (pytest.approx(13.616 * k), pytest.approx(23.651 * k))


# ── items from the second review of docs/codes (2026-10-11) ──
def test_two_tension_layers_match_5_2_2():
    """several tension rows: As at fsd while ξ ≤ ξb, exactly as 5.2.2 (h0 to the centroid of As)"""
    s = jtg.rect(300, 700, concrete="C30", bars=[(2000.0, 650.0), (2000.0, 580.0)])
    h0 = 615.0
    x = 330 * 4000 / (13.8 * 300)
    assert x / h0 <= 0.53
    r = jtg.flexure(s, 100.0)
    assert r.resistance == pytest.approx(330 * 4000 * (h0 - x / 2) / 1e6, rel=1e-9)


def test_one_side_ratio_uses_edge_rows_only():
    """600 × 600 column, 3φ16 top and bottom, 2φ16 on each side: one side = 3φ16 = 0.168 % < 0.2 % (9.1.12 item 3)"""
    a = math.pi * 16 ** 2 / 4
    s = jtg.rect(600, 600, bars=[(3 * a, 50), (3 * a, 550), (2 * a, 200), (2 * a, 400)])
    r = jtg.axial(s, 1000.0, l0=3000)
    assert any("tension side" in f and "0.2" in f for f in r.warnings)


def test_small_eccentricity_rectangle_5_3_4():
    """5.3.4-1/-2 with σs from 5.1.5-1, solved independently"""
    from scipy.optimize import brentq
    b, h, a, As, Asc = 400.0, 600.0, 40.0, 3 * A20, 4 * A25
    h0, e0 = h - a, 60.0
    e = e0 + h / 2 - a
    sig = lambda x: min(max(0.0033 * 2e5 * (0.8 * h0 / x - 1), -330.0), 330.0)
    x = brentq(lambda x: 13.8 * b * min(x, h) * (h0 - min(x, h) / 2) + 330 * Asc * (h0 - a)
               - e * (13.8 * b * min(x, h) + 330 * Asc - sig(x) * As), 0.53 * h0, 3 * h)
    Nu = 13.8 * b * min(x, h) + 330 * Asc - sig(x) * As
    s = jtg.rect(b, h, bottom=[(3, 20, a)], top=[(4, 25, a)])
    r = jtg.eccentric(s, 1000.0, 1000.0 * e0 / 1e3)
    assert r.details["case"] == "small eccentricity"
    assert r.details["x"] == pytest.approx(x, rel=1e-6)
    assert max(r.resistance, r.details.get("N_reverse", 0)) >= r.resistance  # reverse check never raises Nu
    assert r.resistance == pytest.approx(min(Nu / 1e3, r.details.get("N_reverse", math.inf)), rel=1e-6)


def test_tee_large_eccentricity_5_3_5():
    """5.3.5-1/-2, flange on the compressed side, x > h'f, both bar groups yield; e from the gross centroid"""
    b, h, bf, hf, a = 300.0, 1000.0, 800.0, 150.0, 50.0
    As, Asc = 6 * A25, 4 * A20
    s = jtg.tee(b, h, bf, hf, bottom=[(6, 25, a)], top=[(4, 20, a)])
    zg, h0 = s.zg, h - a
    e0 = 900.0
    es = e0 + (h0 - zg)  # load to As
    # N = fcd [b x + (bf − b) hf] + fsd' As' − fsd As;  N es = fcd [b x (h0 − x/2) + (bf − b) hf (h0 − hf/2)] + fsd' As' (h0 − a)
    F = 13.8 * (bf - b) * hf + 330 * Asc - 330 * As
    M = 13.8 * (bf - b) * hf * (h0 - hf / 2) + 330 * Asc * (h0 - a)
    # (13.8 b x + F) es = 13.8 b x (h0 − x/2) + M  →  6.9 b x² + 13.8 b (es − h0) x + F es − M = 0
    A2, B, C = 6.9 * b, 13.8 * b * (es - h0), F * es - M
    x = (-B + math.sqrt(B * B - 4 * A2 * C)) / (2 * A2)
    assert hf < x <= 0.53 * h0
    r = jtg.eccentric(s, 1000.0, 1000.0 * e0 / 1e3)
    assert r.resistance == pytest.approx((13.8 * b * x + F) / 1e3, rel=1e-6)


def test_combination_favourable_variable_and_per_item_factors():
    G = np.array([100.0])
    hi, lo = jtg.basic_combination(G, (np.array([50.0]), np.array([10.0])), others=[(np.array([-10.0]), np.array([-30.0]))])
    assert hi[0] == pytest.approx(1.2 * 100 + 1.4 * 50)  # the favourable −10 is left out (4.1.4)
    assert lo[0] == pytest.approx(1.0 * 100 - 0.75 * 1.4 * 30)  # the favourable +10 of Q is left out
    hi, _ = jtg.basic_combination([np.array([100.0]), (np.array([20.0]), (1.4, 1.0))], (np.array([0.0]), np.array([0.0])))
    assert hi[0] == pytest.approx(1.2 * 100 + 1.4 * 20)


def test_stirrup_steel_grade():
    s = jtg.rect(300, 700, concrete="C30", steel="HRB400", bottom=[(4, 25, 50)])
    r = jtg.shear(s, 100.0, (2, 8, 250), stirrup_steel="HPB300")  # ρsv = 0.134 %: between 0.11 % and 0.14 %
    assert r.details["fsv"] == 250.0
    assert any("0.14" in f for f in r.warnings)
    assert not jtg.shear(s, 100.0, (2, 8, 250)).warnings  # HRB400 stirrups: 0.11 % suffices


def test_singly_reinforced_alternative_keeps_inner_tension_rows():
    """5.2.4 with x < 2as': the singly reinforced alternative keeps every tension row (here two rows 200 mm apart, beyond
    the edge-row tolerance), as 5.2.2 counts all the tension reinforcement"""
    a25 = 5 * A25
    s = jtg.tee(300, 1500, 2000, 200, concrete="C40", bars=[(a25, 1450), (a25, 1250)], top=[(6, 20, 50)])
    r = jtg.flexure(s, 1.0)
    assert r.details["x"] < 100
    x1 = 330 * 2 * a25 / (18.4 * 2000)
    assert r.details["M_singly_reinforced"] == pytest.approx(330 * a25 * ((1450 - x1 / 2) + (1250 - x1 / 2)) / 1e6, rel=1e-9)
