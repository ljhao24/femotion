"""Hinges, bearings and expansion joints (JTG 3362-2018 8.6–8.8, 9.7, 9.8) and the braking force (JTG D60-2015 4.3.5).

Independent sources, besides hand calculations with the code's own formulas:
  - concrete hinges: Hertz line contact (contact width and pressure of two cylinders, ν = 0.2, from the elastic
    solution, not from the code's constants 7.14 and 2.74);
  - figure 5.7.1, bottom row (strip loads): Ab read off the figure for c ≥ b and c < b;
  - laminated bearings: JT/T 4-2019 (another standard): tan α ≤ 0.5 / 0.7 (5.4.5) against te ≥ 2 Δl / 1.43 Δl, the
    shape factor as "loaded area / free side area of one layer" computed on a polygon (5.4.7 formula vs definition);
    Hooke's law of a sheared layer for 8.7.4, 8.7.5 (commentary p223–224: tan α = Δl / te = Fk / (Ag Ge));
  - movements: D60 table 4.3.12-2 values, commentary tables C-1, C-2 (p250–251, 10-year shrinkage and creep of C40),
    the geometry of a joint gap for the installation width (8.8.3), equilibrium of a deck on springs;
  - braking force: the commentary of D60 4.3.5 (p62): the multi-lane factors are the lane count × the transverse
    reduction of table 4.3.1-5;
  - JT/T 4-2019 5.4.1 (σc), 5.4.3 (Eb), 5.4.4 (μ), 5.3.1.2, 5.3.2.1 for the defaults.
Not independent, but a check of the code's own derivation: the hinge's transverse tension against Mörsch's
z = 1.5 (a − b) / (4h) P with the allowable stresses of the commentary (p222), which only tests the rounding of 0.425;
the lifting loops by hand. Boundaries ("not less than", "not more than") are tested on both sides.
"""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import devices as dv


# ── 8.6 hinges ──────────────────────────────────────────────────────────────
def _hertz(P, l, r1, r2, E, nu=0.2):
    """Hertz line contact of a cylinder r1 in a seat r2 (inf: a plane), both of modulus E and Poisson's ratio ν:
    contact width 2a = 2 √(4 P R / (π l E*)), 1/R = 1/r1 − 1/r2, 1/E* = 2 (1 − ν²) / E; mean pressure P / (2 a l).
    N, mm, MPa."""
    R = 1.0 / (1.0 / r1 - (0.0 if math.isinf(r2) else 1.0 / r2))
    Es = E / (2 * (1 - nu * nu))
    a = math.sqrt(4 * P * R / (math.pi * l * Es))
    return 2 * a, P / (2 * a * l)


@pytest.mark.parametrize("r1, r2", [(2000, math.inf), (1500, 4000), (3000, 3600)])
def test_hinge_contact_width_is_hertz(r1, r2):
    """8.6.1-3 is the Hertz contact width at P = γ0 Fhd / 1.3 (commentary 8-13); the code's 2.74 comes from the
    rounded 0.32 of 8-11 (exact π/4 × 0.4072 = 0.3198) and lies 0.1 % below the exact 2.7426"""
    c = jtg.concrete("C40")
    for Fhd, g0 in [(800.0, 1.0), (2500.0, 1.1)]:
        b = jtg.hinge_contact_width(Fhd, Ec=c.Ec, r1=r1, r2=r2, l=6000, gamma0=g0)
        w, _ = _hertz(g0 * Fhd * 1e3 / 1.3, 6000, r1, r2, c.Ec)
        assert b == pytest.approx(w, rel=1e-3) and b < w


def test_hinge_compression_is_hertz_at_the_allowable_mean_pressure():
    """γ0 Fhd / R of 8.6.1-1 equals (Hertz mean pressure at P = γ0 Fhd / 1.3 over 0.75 β ηs fcd)², the allowable-stress
    reading of the commentary (p221–222), since the Hertz pressure grows with √P"""
    for grade, r1, r2, Ab in [("C40", 2000, math.inf, 3.0e6), ("C60", 1500, 5000, 1.2e6), ("C30", 2500, math.inf, 5.0e6)]:
        c = jtg.concrete(grade)
        chk = jtg.hinge_compression(c, 1800.0, l=5000, r1=r1, r2=r2, Ab=Ab, gamma0=1.1)
        b, pm = _hertz(1.1 * 1800e3 / 1.3, 5000, r1, r2, c.Ec)
        beta = math.sqrt(Ab / (b * 5000))
        es = 1.0 if c.grade <= 50 else 1 - 0.24 * (c.grade - 50) / 30  # 5.7.1
        assert chk.ratio == pytest.approx((pm / (0.75 * beta * es * c.fcd)) ** 2, rel=2e-3)
        assert chk.details["eta_s"] == pytest.approx(es)


def test_hinge_compression_hand_and_figure_5_7_1():
    """8.6.1-1, -2 by hand; Ab of figure 5.7.1 (strip across the full length): b + 2b when c = (a − b)/2 ≥ b, else
    b + 2c = a"""
    c = jtg.concrete("C40")
    l, r1, F = 8000.0, 2000.0, 2000.0
    b = 2.74 * math.sqrt(F * 1e3 / (c.Ec / r1 * l))
    wide = jtg.hinge_compression(c, F, l=l, r1=r1, a=600)
    assert wide.details["b"] == pytest.approx(b)
    assert wide.details["Ab"] == pytest.approx(3 * b * l) and wide.details["beta"] == pytest.approx(math.sqrt(3))
    assert wide.resistance == pytest.approx(7.14 * (math.sqrt(3) * 18.4) ** 2 * l / (c.Ec / r1) / 1e3)
    assert wide.ok and wide.clause.endswith("8.6.1")
    narrow = jtg.hinge_compression(c, F, l=l, r1=r1, a=2 * b)  # c = b/2 < b → Ab = a l
    assert narrow.details["Ab"] == pytest.approx(2 * b * l) and narrow.details["beta"] == pytest.approx(math.sqrt(2))
    # a concave seat spreads the contact: wider b than on a plane, and the plane is the limit r2 → ∞
    seat = jtg.hinge_compression(c, F, l=l, r1=r1, r2=3 * r1, a=600)
    assert seat.details["b"] > wide.details["b"]
    far = jtg.hinge_compression(c, F, l=l, r1=r1, r2=1e15, a=600)
    assert far.resistance == pytest.approx(wide.resistance, rel=1e-9)
    with pytest.raises(ValueError):
        jtg.hinge_compression(c, F, l=l, r1=r1, r2=r1, a=600)  # not a convex cylinder in a larger seat
    with pytest.raises(ValueError):
        jtg.hinge_compression(c, F, l=l, r1=r1)  # neither a nor Ab


def test_hinge_detailing_9_8_3_warns_only():
    """9.8.3: convex radius 1.5 … 3.0 m (closed), concrete not below C30; warnings, not the verdict"""
    for r1, warn in [(1499, True), (1500, False), (3000, False), (3001, True)]:
        chk = jtg.hinge_compression("C40", 500, l=4000, r1=r1, a=500)
        assert bool(chk.warnings) is warn and chk.ok
    low = jtg.hinge_compression("C25", 500, l=4000, r1=2000, a=500)
    assert any("C30" in w for w in low.warnings) and low.ok
    assert not jtg.hinge_compression("C30", 500, l=4000, r1=2000, a=500).warnings


def test_hinge_transverse_morsch():
    """8.6.2 against the commentary's own derivation from Mörsch (8-14, 8-15, p222): 0.68 fsd As ≥ 1.5 (a − b)/(4h)
    γ0 Fhd / 1.3; the code's 0.425 rounds 1.5 / (4 × 1.3 × 0.68) = 0.4242 up, so its capacity is 0.2 % below"""
    a, h, b, As = 600.0, 650.0, 12.0, 2400.0
    chk = jtg.hinge_transverse(1500, a=a, h=h, b=b, As=As, steel="HRB400", gamma0=1.1)
    fsd = 330.0
    morsch = 0.68 * fsd * As / (1.5 * (a - b) / (4 * h) / 1.3) / 1e3
    assert chk.resistance == pytest.approx(h / (0.425 * (a - b)) * fsd * As / 1e3)
    assert chk.resistance == pytest.approx(morsch, rel=2.5e-3) and chk.resistance < morsch
    assert chk.demand == pytest.approx(1650) and chk.details["As_side"] == pytest.approx(0.4 * As)
    assert jtg.hinge_transverse(1500, a=a, h=h, b=b, As=As, steel="HRB500").details["fsd"] == 415  # no 330 cap


def test_hinge_transverse_height_and_side_bars():
    """h within 0.80 … 1.25 a (closed): outside fails; side bars 0.4 As: less is a warning"""
    kw = dict(a=500.0, b=10.0, As=2000.0)
    for h, ok in [(399, False), (400, True), (625, True), (626, False)]:
        assert jtg.hinge_transverse(800, h=h, **kw).ok is ok
    assert not jtg.hinge_transverse(800, h=500, As_side=800.0, **kw).warnings
    w = jtg.hinge_transverse(800, h=500, As_side=799.0, **kw)
    assert w.warnings and w.ok


# ── 8.7 laminated bearings ──────────────────────────────────────────────────
def _polygon_shape_factor(pts, t1):
    """S = loaded area / free side area of one layer (the definition), area by the shoelace formula"""
    x, y = np.asarray(pts, float).T
    A = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    P = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])).sum()
    return A / (P * t1)


def test_shape_factor_and_moduli_jtt4():
    """JT/T 4-2019 5.4.7: S = l0a l0b / (2 t1 (l0a + l0b)), d0 / (4 t1), against the definition; Ee = 5.4 G S², G = 1.0;
    plates 5 mm inside the edges, covers 2.5 mm (figures 1, 2); te = n t1 + 2 covers; k = Ge Ag / te"""
    b = dv.Laminated(300, 400, t1=8, n=3, ts=3)
    assert (b.l0a, b.l0b, b.Ae, b.Ag) == (290, 390, 290 * 390, 300 * 400)
    assert b.S == pytest.approx(_polygon_shape_factor([(0, 0), (290, 0), (290, 390), (0, 390)], 8))
    assert b.te == 29 and b.height == 29 + 4 * 3 and b.G == 1.0 and b.Eb == 2000.0
    assert b.Ee == pytest.approx(5.4 * b.S ** 2)
    assert b.k == pytest.approx(1.0 * 120000 / 29 / 1e3)
    c = dv.Laminated(350, t1=10, n=4, ts=3, circle=True)
    th = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    assert c.S == pytest.approx(_polygon_shape_factor(np.c_[170 * np.cos(th), 170 * np.sin(th)], 10), rel=1e-6)
    assert c.Ae == pytest.approx(math.pi * 340 ** 2 / 4) and c.Ag == pytest.approx(math.pi * 350 ** 2 / 4)
    assert dv.Laminated(300, 400, t1=8, n=3, ts=3, rubber=31, E=600).te == 31
    with pytest.raises(ValueError):
        dv.Laminated(400, 300, t1=8, n=3, ts=3)  # la is the short side


def _checks(chks):
    return {c.name.split(" (")[0]: c for c in chks}


def test_laminated_shear_thickness_is_jtt4_shear_angle():
    """te ≥ 2 Δl ⇔ tan α = Δl / te ≤ 0.5 and te ≥ 1.43 Δl ⇔ tan α ≤ 1/1.43 = 0.699 (JT/T 4-2019 5.4.5: 0.5, 0.7);
    the boundary te = 2 Δl passes"""
    b = dv.Laminated(300, 400, t1=8, n=3, ts=3)  # te = 29
    kw = dict(sigma_c=10.0, fy_plate=235, theta=0.0)
    ok = _checks(dv.laminated_bearing(b, 600, dl=14.5, **kw))["rubber thickness, shear without braking"]
    assert ok.ok and ok.details["tan_alpha"] == pytest.approx(0.5)
    assert not _checks(dv.laminated_bearing(b, 600, dl=14.5 + 1e-6, **kw))["rubber thickness, shear without braking"].ok
    br = _checks(dv.laminated_bearing(b, 600, dl=10.0, dl_brake=29 / 1.43 - 10.0, **kw))
    assert br["rubber thickness, shear with braking"].ok
    assert br["rubber thickness, shear with braking"].details["tan_alpha"] == pytest.approx(1 / 1.43)
    nb = _checks(dv.laminated_bearing(b, 600, dl=10.0, dl_brake=29 * 0.7 - 10.0, **kw))
    assert not nb["rubber thickness, shear with braking"].ok  # 1.43 is a little stricter than tan α = 0.7
    # braking from a force: Fbk te / (Ge Ag) (8.8.2 item 4) = Fbk / k (Hooke)
    f = _checks(dv.laminated_bearing(b, 600, dl=5.0, Fbk=30.0, **kw))["rubber thickness, shear with braking"]
    assert f.details["dl_brake"] == pytest.approx(30e3 * 29 / (1.0 * 120000)) == pytest.approx(30 / b.k)


def test_laminated_slopes():
    """8.7.3-4: te ≥ 2 √(Δl² + Δt²), the resultant of two perpendicular shear deformations; the components of the
    reaction along the grades (Rck sin atan i / k); soffit grade > 1 %, cross slope > 2 % outside 8.7.3"""
    b = dv.Laminated(300, 400, t1=8, n=3, ts=3)
    kw = dict(sigma_c=10.0, fy_plate=235)
    R, i, it = 700.0, 0.01, 0.02
    cs = _checks(dv.laminated_bearing(b, R, dl=3.0, slope=i, slope_t=it, **kw))
    sh = cs["rubber thickness, shear without braking"]
    dli = R * 1e3 * (i / math.sqrt(1 + i * i)) * b.te / (b.G * b.Ag)
    dt = R * 1e3 * (it / math.sqrt(1 + it * it)) * b.te / (b.G * b.Ag)
    assert sh.demand == pytest.approx(2 * math.hypot(3.0 + dli, dt)) and sh.clause.endswith("8.7.3-4") and sh.ok
    assert cs["compression of the bearing"].details["theta"] == pytest.approx(math.atan(0.01))
    bad = _checks(dv.laminated_bearing(b, R, dl=3.0, slope=0.011, slope_t=0.021, **kw))
    assert len(bad["rubber thickness, shear without braking"].failed) == 2
    assert not bad["compression of the bearing"].ok
    pyth = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=8, n=3, ts=3, G=1e9), 600, dl=3.0, **kw))
    assert pyth["rubber thickness, shear without braking"].demand == pytest.approx(6.0)  # stiff rubber: no slope terms


def test_laminated_area_stability_compression_plates():
    """8.7.3-1, -6, -8, -9, -10 by hand, with the boundaries closed (la/10 ≤ te ≤ la/5, θ la/2 ≤ δ ≤ 0.07 te)"""
    b = dv.Laminated(300, 400, t1=8, n=3, ts=3)
    R = 850.0
    cs = _checks(dv.laminated_bearing(b, R, sigma_c=10.0, dl=5.0, fy_plate=235, theta=0.002))
    a = cs["bearing area"]
    assert a.demand == pytest.approx(R * 1e3 / (290 * 390)) and a.resistance == 10.0 and a.ok
    assert a.details["Ae_required"] == pytest.approx(85000)
    st = cs["rubber thickness, stability"]
    assert st.resistance == 60 and not st.ok and "la/10" in st.failed[0]  # te = 29 < 30
    for n, ok in [(3, False), (4, True)]:  # te = 29 / 37 vs 30 … 60
        assert _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=8, n=n, ts=3), R, sigma_c=10.0, dl=5.0,
                                            fy_plate=235))["rubber thickness, stability"].ok is ok
    edge = dv.Laminated(300, 400, t1=10, n=3, ts=3, cover=0.0)  # te = 30 = la/10 exactly
    assert _checks(dv.laminated_bearing(edge, R, sigma_c=10.0, dl=5.0, fy_plate=235))["rubber thickness, stability"].ok
    cm = cs["compression of the bearing"]
    delta = R * 1e3 * 29 / (290 * 390) * (1 / b.Ee + 1 / 2000)
    assert cm.demand == pytest.approx(delta) and cm.resistance == pytest.approx(0.07 * 29) and cm.ok
    th = 2 * delta / 300  # θ la / 2 = δ: still in contact
    assert _checks(dv.laminated_bearing(b, R, sigma_c=10.0, dl=5.0, fy_plate=235, theta=th))[
        "compression of the bearing"].ok
    assert not _checks(dv.laminated_bearing(b, R, sigma_c=10.0, dl=5.0, fy_plate=235, theta=th * 1.001))[
        "compression of the bearing"].ok
    pl = cs["steel plates"]
    assert pl.demand == pytest.approx(1.3 * R * 1e3 * 16 / (290 * 390 * 0.65 * 235)) and pl.ok
    one = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=8, n=1, ts=3), R, sigma_c=10.0, dl=1.0,
                                       fy_plate=235))["steel plates"]
    assert one.details["tes_u_plus_l"] == pytest.approx(8 + 2.5)  # one layer: a plate between a layer and a cover
    circ = dv.Laminated(250, t1=8, n=4, ts=3, circle=True)  # te = 37, d/10 … d/5 = 25 … 50
    cc = _checks(dv.laminated_bearing(circ, 300, sigma_c=10.0, dl=5.0, fy_plate=235))
    assert cc["rubber thickness, stability"].clause.endswith("8.7.3-7") and cc["rubber thickness, stability"].ok


def test_laminated_detailing_and_shape_factor_range():
    """plates ≥ 2 mm, side cover ≥ 5 mm, covers ≥ 2.5 mm: warnings only; S outside 5 … 12 (JT/T 4-2019 5.4.7) makes
    Ee = 5.4 G S² inapplicable (failed) unless E is given (then a warning)"""
    kw = dict(sigma_c=10.0, dl=4.0, fy_plate=345)
    ok = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=8, n=4, ts=2.0), 400, **kw))["steel plates"]
    assert not ok.warnings
    thin = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=8, n=4, ts=1.9, edge=4.9, cover=2.4), 400, **kw))
    pl = thin["steel plates"]
    assert len(pl.warnings) == 3 and pl.ok
    fat = dv.Laminated(300, 400, t1=16, n=2, ts=3)  # S = 5.2
    assert 5 < fat.S < 6
    thick = dv.Laminated(300, 400, t1=20, n=2, ts=3)  # S = 4.1
    cm = _checks(dv.laminated_bearing(thick, 400, **kw))["compression of the bearing"]
    assert cm.failed and not cm.ok
    cm = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=20, n=2, ts=3, E=150.0), 400, **kw))[
        "compression of the bearing"]
    assert cm.warnings and not cm.failed


def test_jtt4_defaults_sigma_c_mu_plates():
    """review M1, M3, L2, L4: JT/T 4-2019 5.4.1 σc = 10 MPa, 8 MPa when S < 7 (S = 7 exactly: 10); a larger σc given
    fails; 5.4.4 μ = 0.3 on concrete, 0.2 on steel; 5.3.2.1 plates Q235C → fy 235 MPa; Rck ≥ 5000 kN outside JT/T 4"""
    sq = dv.Laminated(290, 290, t1=10, n=3, ts=3)  # plates 280 × 280: S = 280 / (4 × 10) = 7 exactly
    assert sq.S == pytest.approx(7.0) and dv.sigma_c_limit(sq) == 10.0
    low = dv.Laminated(300, 400, t1=14, n=2, ts=3)  # S = 5.94
    assert 5 < low.S < 7 and dv.sigma_c_limit(low) == 8.0
    R = 8.0 * low.Ae / 1e3  # exactly 8 MPa
    a = _checks(dv.laminated_bearing(low, R, dl=5.0))["bearing area"]
    assert a.resistance == 8.0 and a.ok and a.details["sigma_c_jtt4"] == 8.0
    assert not _checks(dv.laminated_bearing(low, R * 1.001, dl=5.0))["bearing area"].ok
    given = _checks(dv.laminated_bearing(low, R, dl=5.0, sigma_c=10.0))["bearing area"]
    assert given.failed and not given.ok  # 10 MPa for S < 7: above 5.4.1
    assert _checks(dv.laminated_bearing(low, R, dl=5.0, sigma_c=7.0))["bearing area"].ok is False  # 8 > 7: R too high
    b = dv.Laminated(300, 400, t1=8, n=4, ts=3)
    pl = _checks(dv.laminated_bearing(b, 850, dl=5.0))["steel plates"]
    assert pl.details["sigma_s"] == pytest.approx(0.65 * 235)
    big = dv.Laminated(700, 900, t1=12, n=5, ts=4)
    assert _checks(dv.laminated_bearing(big, 5000, dl=5.0))["bearing area"].warnings
    assert not _checks(dv.laminated_bearing(big, 4999, dl=5.0))["bearing area"].warnings
    cs = dv.laminated_sliding(b, RGk=500.0, dl=6.0)
    assert cs[0].resistance == pytest.approx(150.0) and cs[0].details["mu"] == 0.3
    assert dv.laminated_sliding(b, RGk=500.0, dl=6.0, contact="steel")[0].resistance == pytest.approx(100.0)
    assert dv.laminated_sliding(b, RGk=500.0, dl=6.0, mu=0.25)[0].resistance == pytest.approx(125.0)
    with pytest.raises(ValueError):
        dv.laminated_sliding(b, RGk=500.0, dl=6.0, contact="rubber")


def test_slopes_consistent_and_cross_slope_only_shear():
    """review L1, L6: a cross slope over 2 % fails the shear checks only (the commentary: no uneven compression across);
    laminated_sliding adds the soffit-grade shear like laminated_bearing (with the reaction of 8.7.3)"""
    b = dv.Laminated(300, 400, t1=8, n=4, ts=3)
    cs = _checks(dv.laminated_bearing(b, 700, dl=3.0, slope_t=0.025, Fbk=10.0))
    assert cs["rubber thickness, shear without braking"].failed and cs["rubber thickness, shear with braking"].failed
    assert not cs["compression of the bearing"].failed
    i = 0.008
    bear = _checks(dv.laminated_bearing(b, 700, dl=3.0, slope=i))["rubber thickness, shear without braking"]
    sl = dv.laminated_sliding(b, RGk=450.0, RQk=250.0, dl=3.0, slope=i, Fbk=10.0)
    assert sl[0].details["dl"] == pytest.approx(bear.details["dl"])
    assert sl[1].details["dl"] == pytest.approx(bear.details["dl"])
    assert sl[0].demand == pytest.approx(1.4 * b.k * bear.details["dl"])
    over = dv.laminated_sliding(b, RGk=450.0, RQk=250.0, dl=3.0, slope=0.012, Fbk=10.0)
    assert all(c.failed for c in over)


def test_rubber_layers_between_plates_5mm():
    """review L3: JT/T 4-2019 5.3.1.2, rubber between plates ≥ 5 mm (a warning; closed at 5)"""
    kw = dict(dl=2.0)
    w = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=4.9, n=6, ts=2), 300, **kw))["steel plates"]
    assert any("5.3.1.2" in x for x in w.warnings) and w.ok
    ok = _checks(dv.laminated_bearing(dv.Laminated(300, 400, t1=5.0, n=6, ts=2), 300, **kw))["steel plates"]
    assert not any("5.3.1.2" in x for x in ok.warnings)


def test_sliding_8_7_4_hooke():
    """8.7.4: the horizontal force of a sheared bearing Ge Ag Δl / te is k Δl (Hooke, the commentary's
    tan α = Δl / te = Fk / (Ag Ge)); Rck = RGk + 0.5 RQk with braking; boundary μ R = demand passes"""
    b = dv.Laminated(350, 450, t1=10, n=4, ts=3)
    dl = 9.0
    cs = dv.laminated_sliding(b, mu=0.3, RGk=500.0, RQk=300.0, dl=dl, Fbk=25.0)
    assert cs[0].demand == pytest.approx(1.4 * b.k * dl) and cs[0].resistance == pytest.approx(150.0)
    assert cs[1].demand == pytest.approx(1.4 * b.k * dl + 25.0) and cs[1].resistance == pytest.approx(0.3 * 650.0)
    RG = 1.4 * b.k * dl / 0.3
    assert dv.laminated_sliding(b, mu=0.3, RGk=RG, dl=dl)[0].ok
    assert not dv.laminated_sliding(b, mu=0.3, RGk=RG * 0.999, dl=dl)[0].ok
    assert len(dv.laminated_sliding(b, mu=0.3, RGk=500.0, dl=dl)) == 1


def test_ptfe_8_7_5_shear_angle():
    """8.7.5 ⇔ the friction shears the rubber to tan α = μf R / (Ge Ag) ≤ 0.5 (no braking) / 0.7 (braking), JT/T
    4-2019 5.4.5; Rck = RGk + RQk"""
    b = dv.Laminated(300, 400, t1=8, n=3, ts=3)
    rng = np.random.default_rng(3)
    for _ in range(50):
        mu_f, RG, RQ = rng.uniform(0.01, 0.1), rng.uniform(100, 2500), rng.uniform(0, 1000)
        c0, c1 = dv.ptfe_sliding(b, mu_f=mu_f, RGk=RG, RQk=RQ)
        assert c0.ok is (mu_f * RG * 1e3 / (b.G * b.Ag) <= 0.5)
        assert c1.ok is (mu_f * (RG + RQ) * 1e3 / (b.G * b.Ag) <= 0.7)
        assert c1.details["shear_angle"] == pytest.approx(mu_f * (RG + RQ) * 1e3 / (b.G * b.Ag))
    edge = dv.ptfe_sliding(b, mu_f=0.05, RGk=0.5 * 120000 / 0.05 / 1e3)[0]  # exactly tan α = 0.5
    assert edge.ok and edge.resistance == pytest.approx(60.0)


# ── movements and joints (8.7.1 item 5, 8.8.2, 8.8.3) ───────────────────────
def test_temperature_movements_d60():
    """8.8.2-1, -2 with the effective temperatures of D60 table 4.3.12-2 (concrete bridges: cold 34 / −10, severe cold
    34 / −23, mild south 34 / 0 °C) and αc = 1e-5"""
    for zone, south, hi, lo in [("cold", False, 34, -10), ("severe cold", False, 34, -23), ("mild", True, 34, 0)]:
        m = dv.movements(40_000, T_set=(5, 15), zone=zone, south=south, eps_cs=0.0, sigma_pc=0.0)
        assert m.t_plus == pytest.approx(1e-5 * 40_000 * (hi - 5)) and m.t_minus == pytest.approx(1e-5 * 40_000 * (15 - lo))
        assert m.shrinkage == 0 and m.creep == 0
    s = dv.movements(40_000, T_set=10, zone="cold", structure="steel deck", alpha=jtg.EXPANSION["steel"])
    assert s.t_plus == pytest.approx(12e-6 * 40_000 * 36) and s.t_minus == pytest.approx(12e-6 * 40_000 * 31)
    with pytest.raises(ValueError):
        dv.movements(40_000, T_set=(20, 10), T_max=34, T_min=-10, eps_cs=0.0, sigma_pc=0.0)
    with pytest.raises(ValueError):
        dv.movements(40_000, T_set=(5, 40), T_max=34, T_min=-10, eps_cs=0.0, sigma_pc=0.0)


def test_movements_steel_alpha_and_explicit_concrete_terms():
    """review L5, M2: steel bridges of D60 table 4.3.12-2 default to α = 1.2e-5 of table 4.3.12-1 (steel), concrete ones
    to 1e-5; a concrete bridge must state its shrinkage (concrete or eps_cs) and precompression (sigma_pc), else
    ValueError; a steel bridge takes them as 0 with a note"""
    for st in ("steel deck", "concrete deck"):
        m = dv.movements(30_000, T_set=(10, 20), zone="cold", structure=st)
        hi, lo = jtg.effective_temperature("cold", st)
        assert m.t_plus == pytest.approx(12e-6 * 30_000 * (hi - 10)) and m.details["alpha"] == 12e-6
        assert m.shrinkage == 0 and m.creep == 0 and m.notes
    assert dv.movements(30_000, T_set=(10, 20), zone="cold", structure="steel deck", alpha=1e-5).details["alpha"] == 1e-5
    c = dv.movements(30_000, T_set=(10, 20), zone="cold", eps_cs=0.0, sigma_pc=0.0)
    assert c.details["alpha"] == 1e-5
    with pytest.raises(ValueError, match="shrinkage"):
        dv.movements(30_000, T_set=(10, 20), zone="cold", sigma_pc=0.0)
    with pytest.raises(ValueError, match="sigma_pc"):
        dv.movements(30_000, T_set=(10, 20), zone="cold", concrete="C50", RH=70, h=250, t0=90)
    assert dv.movements(30_000, T_set=(10, 20), zone="cold", concrete="C50", RH=70, h=250, t0=90,
                        sigma_pc=0.0).shrinkage > 0


def test_shrinkage_creep_movements_commentary_tables():
    """8.8.2-3, -4 with appendix C against commentary tables C-1, C-2 (C40, RH 55 / 80 %, 10 years): table C-1 is
    reproduced with tu = ts + 3650 and ts = 7 d (to 0.01·10⁻³), table C-2 with tu = t0 + 3650 (to 0.006)"""
    c = jtg.concrete("C40")
    l, spc = 50_000.0, 6.0
    cells = [(55, 200, 28, 0.38, 2.20), (55, 300, 90, 0.30, 1.65), (80, 100, 60, 0.18, 1.55), (80, 600, 14, 0.14, 1.65)]
    for RH, h, t0, eps, phi in cells:
        sh = dv.movements(l, T_set=15, T_max=34, T_min=-10, concrete=c, RH=RH, h=h, t0=t0, tu=7 + 3650, ts=7,
                          sigma_pc=0.0)
        assert sh.shrinkage / l * 1e3 == pytest.approx(eps, abs=0.0101)
        cr = dv.movements(l, T_set=15, T_max=34, T_min=-10, concrete=c, RH=RH, h=h, t0=t0, ts=7, sigma_pc=spc)
        assert cr.details["tu"] == t0 + 3650  # default: 10 years
        assert cr.creep == pytest.approx(spc / c.Ec * cr.details["phi"] * l)
        assert cr.details["phi"] == pytest.approx(phi, abs=0.006)
    m = dv.movements(l, T_set=15, T_max=34, T_min=-10, concrete=c, RH=70, h=250, t0=60, sigma_pc=spc, elastic=True,
                     theta=0.004, y=900)
    assert m.elastic == pytest.approx(spc / c.Ec * l) and m.rotation == pytest.approx(3.6)
    given = dv.movements(l, T_set=15, T_max=34, T_min=-10, concrete=c, eps_cs=2e-4, phi=1.5, sigma_pc=spc)
    assert given.shrinkage == pytest.approx(10.0) and given.creep == pytest.approx(spc / c.Ec * 1.5 * l)


def test_movements_linear_and_superposed():
    """every term is proportional to l; C⁺ and C⁻ are β times the sums of 8.8.2-5, -6; braking on both sides; the bearing
    Δl takes the worse side (program convention)"""
    kw = dict(T_set=(10, 20), zone="cold", concrete="C50", RH=75, h=250, t0=90, sigma_pc=7.0, brake=3.0)
    a, b2 = dv.movements(20_000, **kw), dv.movements(40_000, **kw)
    for f in ("t_plus", "t_minus", "shrinkage", "creep"):
        assert getattr(b2, f) == pytest.approx(2 * getattr(a, f))
    assert a.closing(1.3) == pytest.approx(1.3 * (a.t_plus + 3.0))
    assert a.opening(1.2) == pytest.approx(1.2 * (a.t_minus + a.shrinkage + a.creep + 3.0))
    assert a.bearing_shear() == pytest.approx(max(a.t_plus, a.t_minus + a.shrinkage + a.creep))
    assert a.bearing_shear(braking=True) == pytest.approx(a.bearing_shear() + 3.0)
    # a warmer installation closes less and opens more
    w = dv.movements(20_000, **(kw | {"T_set": (20, 30)}))
    assert w.t_plus < a.t_plus and w.t_minus > a.t_minus


def test_installation_width_is_the_gap_geometry():
    """8.8.3: a joint set at B with Bmin + C⁺ ≤ B ≤ Bmin + (C − C⁻) never closes below Bmin nor opens beyond Bmin + C;
    when C = C⁺ + C⁻ the range is one width"""
    C, Bmin, Cp, Cm = 80.0, 20.0, 18.0, 37.0
    lo, hi = jtg.installation_width(C, Bmin, Cp, Cm)
    for B in np.linspace(lo, hi, 7):
        assert B - Cp >= Bmin - 1e-9 and B + Cm <= Bmin + C + 1e-9
    assert lo - 1e-3 - Cp < Bmin and hi + 1e-3 + Cm > Bmin + C
    assert jtg.installation_width(55.0, Bmin, Cp, Cm) == pytest.approx((38.0, 38.0))
    with pytest.raises(ValueError):
        jtg.installation_width(54.9, Bmin, Cp, Cm)


def test_expansion_joint_selection():
    """8.8.2-7 C ≥ C⁺ + C⁻ (boundary passes), β 1.2 … 1.4 (below fails, above a note), 9.7.7 joint types (warning)"""
    m = dv.movements(30_000, T_set=(10, 20), T_max=34, T_min=-10, brake=2.0, eps_cs=0.0, sigma_pc=0.0)
    need = m.closing(1.3) + m.opening(1.3)
    ok = jtg.expansion_joint(m, need, beta=1.3, Bmin=10, kind="modular")
    assert ok.ok and not ok.warnings and ok.details["installation"][0] == pytest.approx(ok.details["installation"][1])
    assert not jtg.expansion_joint(m, need - 0.01, beta=1.3).ok
    assert not jtg.expansion_joint(m, 200, beta=1.1).ok
    assert jtg.expansion_joint(m, 200, beta=1.2).ok and jtg.expansion_joint(m, 200, beta=1.4).ok
    hi = jtg.expansion_joint(m, 200, beta=1.5)
    assert hi.ok and any("1.2 … 1.4" in n for n in hi.notes)
    other = jtg.expansion_joint(m, 200, kind="rubber plate")
    assert other.warnings and other.ok


def test_stiffness_tools():
    """a deck rigid along its axis on springs: under a uniform strain ε the spring forces ki ε (xi − x0) balance at the
    stationary point; a force F shares as F ki / Σ k with one common displacement; bearings on flexible piers in series"""
    x, k = np.array([0.0, 30.0, 60.0, 90.0]), np.array([2.0, 9.0, 9.0, 3.0])
    x0 = dv.stationary_point(x, k)
    assert abs(np.sum(k * 1e-4 * (x - x0))) < 1e-12
    assert dv.stationary_point(x, [1, 1, 1, 1]) == pytest.approx(45.0)
    assert dv.stationary_point(x, [1e9, 1, 1, 1]) == pytest.approx(0.0, abs=1e-6)
    F = dv.distribute(300.0, k)
    assert F.sum() == pytest.approx(300.0) and np.allclose(F / k, F[0] / k[0])
    assert dv.series(4.0, 4.0) == pytest.approx(2.0) and dv.series(5.0, 1e12) == pytest.approx(5.0)


# ── D60 4.3.5 braking force ─────────────────────────────────────────────────
def test_braking_force_d60():
    """D60 4.3.5: 10 % of the lane load over the loaded length, ≥ 165 / 90 kN; 2, 2.34, 2.68 for 2–4 lanes are the lane
    count × table 4.3.1-5 (commentary p62); longitudinal reduction of table 4.3.1-6"""
    for n in (2, 3, 4):
        assert jtg.braking_force(20, n) == pytest.approx(165 * n * jtg.transverse_reduction(n))
    assert jtg.braking_force(20) == 165.0 and jtg.braking_force(20, grade="II") == 90.0
    L = 150.0
    assert jtg.braking_force(L) == pytest.approx(0.1 * (10.5 * L + 360))  # 193.5
    assert jtg.braking_force(L, grade="II") == pytest.approx(0.75 * 193.5)
    assert jtg.braking_force(200.0) == pytest.approx(0.1 * (10.5 * 200 + 360) * 0.97)
    assert jtg.braking_force(130.0, L0=30.0) == pytest.approx(0.1 * (10.5 * 130 + 2 * (30 + 130)))
    L1 = (1650 - 360) / 10.5  # where 10 % of the lane load reaches 165 kN
    assert jtg.braking_force(L1) == pytest.approx(165.0) and jtg.braking_force(L1 + 1) > 165.0
    with pytest.raises(ValueError):
        jtg.braking_force(50, 5)


# ── 9.7, 9.8 ────────────────────────────────────────────────────────────────
def test_lifting_loops_9_8_2():
    """9.8.2 by hand: two legs per loop, three of four act, σ ≤ 65 MPa under the characteristic self-weight (boundary
    passes); HPB300, not cold-worked, embedment ≥ 35 d, inner diameter ≥ 3 d and ≥ 60 mm"""
    d = 16.0
    A = 2 * math.pi * d * d / 4
    G = 65.0 * 3 * A / 1e3
    chk = dv.lifting_loop(G, d, n=4)
    assert chk.demand == pytest.approx(65.0) and chk.ok and chk.details["n_effective"] == 3
    assert not dv.lifting_loop(G * 1.001, d, n=4).ok
    assert dv.lifting_loop(G, d, n=2).demand == pytest.approx(65.0 * 3 / 2)
    with pytest.raises(ValueError):
        dv.lifting_loop(G, d, n=6)
    assert dv.lifting_loop(G, d, n=6, n_effective=4).demand == pytest.approx(65.0 * 3 / 4)
    good = dv.lifting_loop(G, d, embed=35 * d, inner_diameter=60.0)
    assert not good.warnings
    bad = dv.lifting_loop(G, d, steel="HRB400", cold_worked=True, embed=35 * d - 1, inner_diameter=59.0)
    assert len(bad.warnings) == 4 and bad.ok
    assert dv.lifting_loop(G, 25.0, inner_diameter=74.0).warnings  # 3 d = 75 governs


def test_culverts_and_bearing_detailing():
    """9.8.1: two layers from 1 m; 9.7.2 rubber by site temperature (−25 / −40 … +60 °C, closed); 9.7.3 one bearing
    along the bridge per support point, more than two across: uplift; 9.7.1 bearing types. Reports: ok stays true"""
    assert not dv.culvert_detailing(999, layers=1).warnings
    r = dv.culvert_detailing(1000, layers=1)
    assert r.warnings and r.ok and not dv.culvert_detailing(1500, layers=2).warnings
    assert not dv.bearing_detailing(rubber="CR", site_T=(-25, 60)).warnings
    assert dv.bearing_detailing(rubber="CR", site_T=(-26, 40)).warnings
    assert not dv.bearing_detailing(rubber="NR", site_T=(-40, 40)).warnings
    assert dv.bearing_detailing(rubber="EPDM", site_T=(-41, 40)).warnings
    rep = dv.bearing_detailing(kind="lead rubber", n_longitudinal=2, n_transverse=3)
    assert len(rep.warnings) == 3 and rep.ok
    assert not dv.bearing_detailing(kind="pot", n_longitudinal=1, n_transverse=2).warnings
