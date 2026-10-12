"""Detailing of prestressed superstructures (JTG 3362-2018 9.4.1–9.4.20), the reinforcement zones and dense anchors of
post-tensioned anchorage zones (8.2.1, 8.2.2, 9.4.18, 9.4.20) and the detailing of columns, piers, joints and pile caps
(9.6.1–9.6.10).

Independent of the formula under test: 9.4.8-2 against the exact resultant of the deviation forces of a tendon turning
through an arc (the code's form is its linearisation); 9.3.14-2 against the circle through three points; the radial
force of 8.2.6-5 against 2 Pd sin(α/2) (commentary p213); the blister length of figure 9.4.19 against its geometry; the
clear spacing on a circle against bar coordinates; the bursting force of a group of anchors against hand-built
resultants and physical limits (a load over the whole end face bursts nothing; anchors symmetric about the centroid
cancel their inclinations). Consistency only (the same model, not an independent source): 9.4.8 against the
commentary's own derivation from the AASHTO check (p232–233, the constant not rounded); the hoop rules of 9.6.1 in
jtg.shear_circle and in column_detailing (written to the same convention). From the commentary: the dense-anchor
distance (AASHTO's 1.5 a widened to 2 a, p212), the surface mesh of 9.6.4 (five 8 mm bars a metre, p239). Table-like
rules: the printed values with their pages and the edges of "not less than" / "more than"."""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import bearing
from femotion.codes.jtg import pier_detailing as pd
from femotion.codes.jtg import regions as rg
from femotion.codes.jtg import tendon_detailing as td


def warned(chk, text):
    return any(text in w for w in chk.warnings)


# ── 9.4.8: cover of curved ducts ─────────────────────────────────────────────
@pytest.mark.parametrize("Pd, r, fcu, ds", [(1951.0, 8000.0, 45.0, 90.0), (3500.0, 15000.0, 50.0, 110.0),
                                            (900.0, 4000.0, 40.0, 70.0)])
def test_curved_cover_from_aashto(Pd, r, fcu, ds):
    """consistent with the commentary's derivation (p232–233, not an independent source): in-plane Fin = P/r ≤ Vc,
    out-of-plane Fout = P/(π r) ≤ Vc, Vc = 0.33 φ dc √f'ci (N/mm), φ = 0.9, f'ci = 0.8 f'cu (cylinder / cube),
    dc = cover + ds/2; the code rounds 0.33 · 0.9 · √0.8 = 0.2656 to 0.266, so the two agree to 0.2 %. It catches
    copying and unit errors (N against kN)"""
    P = Pd * 1e3
    k = 0.33 * 0.9 * math.sqrt(0.8 * fcu)
    Cin, Cout = td.curved_cover(Pd, r, fcu, ds)
    assert Cin + ds / 2 == pytest.approx(P / r / k, rel=2e-3)
    assert Cout + ds / 2 == pytest.approx(P / (math.pi * r) / k, rel=2e-3)


def test_curved_cover_limits():
    """physical properties: the cover needed falls as the curve flattens and tends to −ds/2 for a straight duct (no
    radial force: the 9.1.1 cover governs, item 3); it is linear in the force; out of the plane it is the in-plane
    demand / π"""
    ds = 90.0
    rs = [2000, 5000, 1e4, 1e5, 1e12]
    c = [td.curved_cover(2000, r, 45, ds)[0] for r in rs]
    assert all(a > b for a, b in zip(c, c[1:]))
    assert c[-1] == pytest.approx(-ds / 2, abs=1e-3)
    c1, o1 = td.curved_cover(1000, 6000, 45, ds)
    c2, o2 = td.curved_cover(2000, 6000, 45, ds)
    assert c2 + ds / 2 == pytest.approx(2 * (c1 + ds / 2))
    assert o1 + ds / 2 == pytest.approx((c1 + ds / 2) / math.pi)
    # item 3: below the 9.1.1 cover the 9.1.1 cover governs
    r = td.curved_duct_detailing(500, 50000, 45, ds, c_in=45, c_out=45, cover_min=45)
    assert r.ok and not r.warnings and r.details["Cin"] < 0
    r = td.curved_duct_detailing(500, 50000, 45, ds, c_in=44.9, c_out=45, cover_min=45)
    assert r.ok and len(r.warnings) == 1


def test_curved_stirrup_exact_deviation_force():
    """9.4.8-2 (p110) is the linearised equilibrium of a tendon turning through an arc of length sv (commentary p233:
    Fe = (P/r) sv = 2 fsv Asv1). The exact resultant of the tendon force over that arc is 2 P sin(sv / (2r)); the two
    agree to (sv/2r)²/6. Same structure as 9.3.14-1 (p108): with P = fsd As, Asv1 = m As sv/(2r), m = fsd/fsv"""
    P, fsv = 1951.0, 330.0
    for r, sv in ((5000.0, 100.0), (8000.0, 150.0), (30000.0, 200.0)):
        exact = 2 * P * 1e3 * math.sin(sv / (2 * r)) / (2 * fsv)
        assert td.curved_stirrup(P, r, sv, fsv) == pytest.approx(exact, rel=(sv / (2 * r)) ** 2 / 5)
    As, fsd = 6 * 804.2, 330.0
    assert td.curved_stirrup(fsd * As / 1e3, 4000, 120, 250) == pytest.approx(fsd / 250 * As * 120 / (2 * 4000))


def test_curve_radius_is_the_circle_through_three_points():
    """9.3.14-2 (p108): r = l/2 (1/(4β) + β). Independently: the circumradius abc/(4K) of the chord ends and the crown.
    For a parabola of the same chord and rise the vertex radius is l²/(8f), f/2 less"""
    for l, f in ((12000.0, 600.0), (30000.0, 2500.0), (8000.0, 4000.0)):
        A, B, C = np.array([-l / 2, 0.0]), np.array([0.0, f]), np.array([l / 2, 0.0])
        a, b, c = np.linalg.norm(B - C), np.linalg.norm(A - C), np.linalg.norm(A - B)
        u, v = B - A, C - A
        K = abs(u[0] * v[1] - u[1] * v[0]) / 2
        assert td.curve_radius(l, f) == pytest.approx(a * b * c / (4 * K), rel=1e-12)
        assert td.curve_radius(l, f) - l * l / (8 * f) == pytest.approx(f / 2)


def test_curved_duct_detailing_branches():
    """9.4.8 (p110): c_in ≥ Cin, or the 9.1.1 cover with stirrups in the plane (9.4.8-2); c_out ≥ Cout; detailing only
    warns"""
    Pd, r, fcu, ds = 1951.0, 5000.0, 45.0, 90.0
    Cin, Cout = td.curved_cover(Pd, r, fcu, ds)
    assert Cout > 0
    assert td.curved_duct_detailing(Pd, r, fcu, ds, c_in=Cin, c_out=Cout).warnings == []
    w = td.curved_duct_detailing(Pd, r, fcu, ds, c_in=Cin - 0.1, c_out=Cout - 0.1)
    assert w.ok and len(w.warnings) == 2
    A12 = math.pi * 12 ** 2 / 4
    sv = A12 * 2 * r * 330 / (Pd * 1e3)  # the spacing at which a φ12 leg is just enough
    ok = td.curved_duct_detailing(Pd, r, fcu, ds, c_in=45, cover_min=45, stirrup=(12, sv))
    assert not ok.warnings and ok.details["Asv1 required"] == pytest.approx(A12)
    bad = td.curved_duct_detailing(Pd, r, fcu, ds, c_in=45, cover_min=45, stirrup=(12, 1.01 * sv))
    assert warned(bad, "stirrup leg area") and bad.ok
    assert warned(td.curved_duct_detailing(Pd, r, fcu, ds, c_in=40, cover_min=45, stirrup=(12, sv)), "9.1.1")
    hpb = td.curved_duct_detailing(Pd, r, fcu, ds, c_in=45, cover_min=45, stirrup=(12, sv), stirrup_steel="HPB300")
    assert warned(hpb, "stirrup leg area")  # fsv 250 < 330


def test_curved_spacing_9_4_9_item_2():
    """9.4.9 item 2 (p110–111): the in-plane spacing by 9.4.8-1, not below the straight-duct minimum 40 mm (item 1)"""
    Pd, fcu, ds = 1951.0, 45.0, 90.0
    flat = td.curved_spacing(Pd, 1e6, fcu, ds, clear_in=40, clear_out=40)
    assert flat.details["Cin"] < 40 and not [w for w in flat.warnings if "should" not in w]
    assert warned(td.curved_spacing(Pd, 1e6, fcu, ds, clear_in=39.9), "in the plane")
    Cin, Cout = td.curved_cover(Pd, 6000, fcu, ds)
    assert Cin > 40
    assert not td.curved_spacing(Pd, 6000, fcu, ds, clear_in=Cin, clear_out=max(Cout, 54)).warnings
    assert warned(td.curved_spacing(Pd, 6000, fcu, ds, clear_in=Cin - 0.5), "in the plane")
    assert warned(td.curved_spacing(Pd, 6000, fcu, ds, clear_in=Cin, clear_out=50), "duct diameter")  # 50 < 0.6 × 90


# ── 9.4.9–9.4.12: ducts ──────────────────────────────────────────────────────
def test_duct_rules_printed_values():
    """9.4.9 (p110–111): straight ducts ≥ 40 mm apart and ≥ 0.6 ds (should); two stacked in straight runs; inner area
    ≥ 2 Ap. 9.4.10 (p111): 4 m / 6 m by the wire diameter (≤ 5 mm), threaded bars 12 m / 15 m (≤ 25 mm). 9.4.11 (p111):
    0.80–1.50 m with figure 9-1 (p233). 9.4.12 (p111): grout ≥ 50 MPa"""
    assert not td.duct_detailing(clear=40).warnings
    assert warned(td.duct_detailing(clear=39.9), "clear spacing of straight ducts")
    assert not td.duct_detailing(clear=48, ds=80).warnings
    r = td.duct_detailing(clear=47.9, ds=80)
    assert warned(r, "should") and len(r.warnings) == 1
    assert not td.duct_detailing(stacked=2).warnings
    assert warned(td.duct_detailing(stacked=3), "stacked")
    assert warned(td.duct_detailing(stacked=2, curved=True), "stacked")
    Ap = 9 * 139.0
    d2 = math.sqrt(4 * 2 * Ap / math.pi)  # inner area exactly 2 Ap
    assert not td.duct_detailing(d_inner=d2, Ap=Ap).warnings
    assert warned(td.duct_detailing(d_inner=d2 - 0.1, Ap=Ap), "inner duct area")
    # from the section's tendon rows: 9 strands in a 90 mm duct (5.09 Ap) and 22 in an 80 mm one (1.64 Ap)
    sec = jtg.tee(200, 2000, 2200, 180, concrete="C50", bottom=[(10, 20, 60)],
                  tendons=[(9 * 139, 1800, 1100, 90), (22 * 139, 1650, 1100, 80)])
    r = td.duct_detailing(sec)
    assert len(r.warnings) == 1 and "row 1" in r.warnings[0]
    # 9.4.10
    for kind, d, R in (("strand", 5.0, 4000), ("strand", 5.01, 6000), ("wire", 7.0, 6000), ("bar", 25.0, 12000),
                       ("bar", 32.0, 15000)):
        assert td.curve_radius_min(kind, d) == R
        assert not td.duct_detailing(kind=kind, d=d, radius=R).warnings
        assert warned(td.duct_detailing(kind=kind, d=d, radius=R - 1), "radius")
    assert td.curve_radius_min("strand", 5.2) == jtg.deviation_radius_min(5.2)  # the same rule as 9.4.23
    assert td.duct_detailing(sec, d=5.0, radius=4000).details["tendon radius"]["limit"] == 4000  # kind from sec
    # 9.4.11 and figure 9-1 (12 strands: 3.1 MN → 1.0 m; 22 strands: 5.7 MN → 1.5 m)
    assert [td.anchor_straight_min(F) for F in (None, 1000, 3130, 5740, 15000)] == [800, 800, 1000, 1500, 1500]
    assert not td.duct_detailing(straight_length=800).warnings
    assert warned(td.duct_detailing(straight_length=799), "straight length")
    assert warned(td.duct_detailing(straight_length=1200, F_break=5740), "straight length")
    # 9.4.12
    assert not td.duct_detailing(grout=50).warnings
    assert warned(td.duct_detailing(grout=49.9), "grout")
    pre = jtg.rect(400, 600, concrete="C50", tendons=[(560, 520, 1100)], pretensioned=True)
    assert "pretensioned" in td.duct_detailing(pre, clear=10).notes[0]


# ── 9.4.1–9.4.5 ─────────────────────────────────────────────────────────────
def test_girder_detailing_9_4_1_9_4_2():
    """9.4.1 (p109): web stirrups ≥ 10 mm (T, I) / 12 mm (box), ribbed, ≤ 200 mm (should); within one depth of the
    support closed at ≤ 120 mm; bulb stirrups ≥ 8 mm at ≤ 200 mm; vertical prestress 500–1000 mm (should). 9.4.2:
    partially prestressed members need ordinary tension bars"""
    tee = jtg.tee(200, 2000, 2200, 180, concrete="C50", bottom=[(10, 20, 60)], tendons=[(1251, 1800, 1100, 90)])
    ok = td.girder_detailing(tee, stirrup_d=10, ribbed=True, stirrup_spacing=200, end_spacing=120, end_closed=True,
                             end_length=2000, bulb_d=8, bulb_spacing=200, bulb_closed=True, vertical_spacing=500,
                             pc_class="A")
    assert ok.ok and not ok.warnings and not any("skipped" in n for n in ok.notes)
    for kw, text in ((dict(stirrup_d=9.9), "web stirrup diameter"), (dict(ribbed=False), "plain"),
                     (dict(stirrup_spacing=201), "web stirrup spacing"), (dict(end_spacing=121), "within h"),
                     (dict(end_closed=False), "not closed"), (dict(end_length=1999), "dense end zone"),
                     (dict(bulb_d=7.9), "bulb stirrup diameter"), (dict(bulb_spacing=201), "bulb stirrup spacing"),
                     (dict(vertical_spacing=499), "vertical"), (dict(vertical_spacing=1001), "vertical")):
        r = td.girder_detailing(tee, **kw)
        assert r.ok and len(r.warnings) == 1 and text in r.warnings[0], kw
    box = jtg.box(5000, 2500, 250, 220, 900, b_top=11000, concrete="C50", bottom=[(60, 25, 60)],
                  tendons=[(20000, 2300, 1100, 300)])
    assert warned(td.girder_detailing(box, stirrup_d=10), "12 mm")
    assert not td.girder_detailing(box, stirrup_d=12).warnings
    bare = tee.replace(bars=np.zeros((0, 2)), bar_d=np.zeros(0))
    assert warned(td.girder_detailing(bare, pc_class="B"), "mixed reinforcement")
    assert not td.girder_detailing(bare, pc_class=None).warnings  # full prestressing: 9.4.2 does not apply


def test_pretensioned_9_4_3_to_9_4_5():
    """9.4.4 (p109): strands ≥ 1.5 d and ≥ 25 mm apart (φ15.2: 25 mm governs; φ17.8: 26.7 mm), wires ≥ 15 mm; 9.4.5:
    one tendon a 150 mm spiral, several 3–5 meshes within 10 d; 9.4.3: strands or spiral-rib wires"""
    assert not td.pretensioned_detailing("strand", 15.2, clear=25).warnings
    assert warned(td.pretensioned_detailing("strand", 15.2, clear=24.9), "25")
    assert not td.pretensioned_detailing("strand", 17.8, clear=26.7).warnings
    assert warned(td.pretensioned_detailing("strand", 17.8, clear=26.6), "26.7")
    assert not td.pretensioned_detailing(jtg.tendon("wire", 1570), 5, clear=15).warnings
    assert warned(td.pretensioned_detailing("wire", 5, clear=14), "15")
    assert warned(td.pretensioned_detailing("bar", 25), "9.4.3")
    assert not td.pretensioned_detailing("strand", 15.2, n=1, spiral_length=150).warnings
    assert warned(td.pretensioned_detailing("strand", 15.2, n=1, spiral_length=149), "spiral")
    r = td.pretensioned_detailing("strand", 15.2, n=8, meshes=3)
    assert not r.warnings and r.details["mesh zone"] == pytest.approx(152)
    assert warned(td.pretensioned_detailing("strand", 15.2, n=8, meshes=2), "meshes")
    assert any("3–5" in n for n in td.pretensioned_detailing("strand", 15.2, n=8, meshes=6).notes)


# ── 9.4.6, 9.4.13, 9.4.18; 8.2.1 zones ──────────────────────────────────────
def test_end_zone_9_4_6_9_4_18_9_4_13():
    """9.4.6 (p109): trumpet plates, ρv ≥ 0.5 % (the ρv of 5.7.2); 9.4.18 (p112): flat plates ≥ 4 layers or turns,
    trumpets a spiral as long as the trumpet, closed bursting stirrups ≤ 120 mm, large-eccentricity anchorage: end bars
    to the tension edge; 9.4.13 (p111): sealing concrete ≥ 0.8 × the member's class and ≥ C30"""
    s = 0.005 * 250 / 4 * 50  # spiral area for ρv = 0.5 % with dcor 250 at 50 mm (5.7.2-4)
    assert bearing.spiral_ratio(s, 250, 50) == pytest.approx(0.005)
    ok = td.end_zone_detailing(plate="trumpet", spiral_length=250, trumpet_length=250,
                               rho_v=bearing.spiral_ratio(s, 250, 50), stirrup_spacing=120, closed=True, seal="C40",
                               member="C50")
    assert not ok.warnings
    for kw, text in ((dict(plate="flat", layers=4), "trumpet"), (dict(rho_v=0.00499), "ρv"),
                     (dict(plate="trumpet", spiral_length=249, trumpet_length=250), "spiral length"),
                     (dict(stirrup_spacing=121), "bursting stirrup spacing"), (dict(closed=False), "not closed"),
                     (dict(seal="C35", member="C50"), "sealing"), (dict(seal="C25", member="C35"), "sealing")):
        r = td.end_zone_detailing(**kw)
        assert r.ok and len(r.warnings) == 1 and text in r.warnings[0], kw
    assert warned(td.end_zone_detailing(plate="flat", layers=3), "mesh layers")
    assert not td.end_zone_detailing(seal="C30", member="C35").warnings  # max(28, 30)
    # γ = 2e/h: e = h/6 is the kern's edge (8.2.5: no edge tension), beyond it the end bars go to the tension edge
    assert not td.end_zone_detailing(e=2000 / 6, h=2000, end_bars_to_edge=False).warnings
    assert warned(td.end_zone_detailing(e=2000 / 6 + 1, h=2000, end_bars_to_edge=False), "large-eccentricity")
    assert rg.edge_tension(2094, 2000 / 6, 2000) == 0 < rg.edge_tension(2094, 2000 / 6 + 1, 2000)


def test_anchorage_zone_extent_8_2_1():
    """8.2.1 notes (p78) and figure 8-2 (p210): the zone 1.0–1.2 × max(h, b'f) long, the local zone 1.2 × the plate's
    longer side; zone_steel counts the stirrups inside (ends included). The ties in the web's plane (8.2.2 bursting,
    8.2.5 edge tension) count the web's stirrups from the end of the local zone to factor × h: the commentary to 8.2.2
    (p210) treats the web as a rectangular plate of depth h, whose zone by note 1 is (1.0–1.2) h"""
    z = td.anchorage_zone(2000, bf=2200, plate=(230, 270))
    assert z["length"] == 2200 and z["length_range"] == (2200, pytest.approx(2640))
    assert z["local_length"] == pytest.approx(324) and z["web"] == (pytest.approx(324), 2000)
    assert z["flange"] == (0.0, 2200)
    assert td.anchorage_zone(2000, factor=1.2, plate=(230, 230))["web"][1] == pytest.approx(2400)
    with pytest.raises(ValueError):
        td.anchorage_zone(2000, factor=1.3)
    x = np.arange(50, 2500, 100)  # stirrups every 100 mm from 50 mm
    A = 2 * math.pi * 16 ** 2 / 4
    n = ((x >= 324) & (x <= 2000)).sum()
    assert td.zone_steel(x, A, *z["web"]) == pytest.approx(n * A)
    assert td.zone_steel([324.0, 2200.0, 2200.1], 1.0, 324, 2200) == 2.0


def test_bursting_steel_range_of_a_box_girder():
    """review H1: a box girder 2.5 m deep with a 12.5 m deck. The zone of figure 8-2 is 12.5 m long, but the bursting
    force of 8.2.2 acts in the web's plane: its steel is the web's stirrups within h, whatever the deck width (the
    web as a rectangular plate, commentary p210; the tension within about 3/4 h, commentary to 9.4.7 p232)"""
    z = td.anchorage_zone(2500, bf=12500, plate=(300, 300))
    assert z["length"] == 12500 and z["web"] == (pytest.approx(360), 2500)
    x = np.arange(50, 13000, 100)
    A = 2 * math.pi * 16 ** 2 / 4  # φ16, two legs
    As = td.zone_steel(x, A, *z["web"])
    assert As == pytest.approx(21 * A)  # 450 … 2450 mm
    assert As < td.zone_steel(x, A, *z["flange"]) / 5
    assert z["web"] == td.anchorage_zone(2500, bf=0.0, plate=(300, 300))["web"]  # b'f plays no part


# ── 8.2.2: dense anchors and the bursting force of a group ──────────────────
def test_dense_anchors_8_2_2():
    """note 1 (p79): centres closer than 2 a are dense; the commentary (p212): AASHTO's 1.5 a widened to 2 a. A chain
    of close neighbours is one group (figure 8.2.2b)"""
    a = 230.0
    assert rg.dense_anchors([0, 1.9 * a], a) == [[0, 1]]
    assert rg.dense_anchors([0, 2 * a], a) == [[0], [1]]  # "less than": 2 a is not dense
    assert rg.dense_anchors([0, 1.6 * a], a) == [[0, 1]]  # dense here, separate by AASHTO's 1.5 a
    assert rg.dense_anchors([0, 1.5 * a, 3 * a, 7 * a], a) == [[0, 1, 2], [3]]  # chained, ends 3 a apart
    assert rg.dense_anchors([900, 0, 450], a) == [[1, 2, 0]]
    assert rg.dense_anchors([0, 500], [200, 260]) == [[0, 1]]  # the wider plate: 500 < 520


def test_bursting_force_of_anchor_groups():
    """8.2.2 items 1–3 (p79): a single anchor is 8.2.2-1 itself; dense anchors act as their resultant with a = the
    outer edges of the outer plates; separate anchors each on their own, the largest governs. Physical checks: anchors
    placed symmetrically about the centroid with mirror-image inclinations have a horizontal resultant at the centroid
    (|sin α| = 0); a group covering the whole end face (a → h, e = 0) bursts nothing"""
    h, yc, P = 2000.0, 1000.0, 2094.0
    Tb, db, g = rg.bursting_force_anchors(P, [700.0], 230.0, h, yc=yc, angle=6.0)
    assert (Tb, db) == pytest.approx(rg.bursting_force(P, 230.0, h, 300.0, 6.0))  # above the centroid, going down: +
    Tb, db, g = rg.bursting_force_anchors(P, [1300.0], 230.0, h, yc=yc, angle=6.0)
    assert (Tb, db) == pytest.approx(rg.bursting_force(P, 230.0, h, 300.0, -6.0))  # below it, going down: away
    # symmetric pair 400 mm apart (dense), mirrored inclinations towards the centroid
    Tb, db, g = rg.bursting_force_anchors([P, P], [800.0, 1200.0], 230.0, h, yc=yc, angle=[8.0, -8.0])
    H = 2 * P * math.cos(math.radians(8.0))
    assert len(g) == 1 and g[0]["alpha"] == pytest.approx(0.0, abs=1e-9) and g[0]["e"] == pytest.approx(0.0)
    assert g[0]["P"] == pytest.approx(H) and g[0]["a"] == pytest.approx(630.0)
    assert Tb == pytest.approx(0.25 * H * (1 - 630.0 / h)) and db == pytest.approx(h / 2)
    # resultant position by moment equilibrium of the horizontal forces
    Tb, db, g = rg.bursting_force_anchors([1000.0, 3000.0], [600.0, 900.0], 230.0, h, yc=yc)
    assert g[0]["y"] == pytest.approx((1000 * 600 + 3000 * 900) / 4000) and g[0]["e"] == pytest.approx(175.0)
    # the whole face loaded evenly: no bursting
    ys = np.linspace(115, h - 115, 9)
    assert rg.bursting_force_anchors(P, ys, 230.0, h, yc=yc)[0] == pytest.approx(0.0, abs=1e-9)
    # separate anchors: the largest of the single forces (item 3)
    Tb, db, g = rg.bursting_force_anchors([P, 1.5 * P], [400.0, 1600.0], 230.0, h, yc=yc)
    single = [rg.bursting_force(p, 230.0, h, 600.0, 0.0)[0] for p in (P, 1.5 * P)]
    assert len(g) == 2 and Tb == pytest.approx(max(single))


# ── 9.4.19, 9.4.20: blisters ────────────────────────────────────────────────
def test_blister_radial_force_8_2_6_exact():
    """8.2.6-5 (p81): TR,d = Pd α; the commentary (p213) derives it from the self-equilibrated deviation forces,
    2 Pd sin(α/2), its small-angle form (the 15° limit of 9.4.19 keeps the two within 0.3 %)"""
    for a in (5.0, 10.0, 15.0):
        exact = 2 * 2094 * math.sin(math.radians(a) / 2)
        assert rg.blister_forces(2094, 230, 400, 300, a)["radial"] == pytest.approx(exact, rel=3e-3)


def test_blister_zones_and_rules_9_4_19_9_4_20():
    """9.4.19 (p112–113, figure 9.4.19): deviation ≤ 15°, face angle ≥ 90°, L ≥ (S1 + S2)/sin α; 9.4.20 (p113,
    figure 9.4.20, commentary figure 9-3 p235): bursting stirrups ≤ 150 mm over ≥ 1.2 h; tie-back bars ≥ 1.5 m centred
    on the anchor face line, local-bending bars ≥ 1.5 m centred one wall thickness ahead of it, both within 1.5 a either
    side of the tendon (3 a in all); radial reinforcement over the curved length"""
    z = td.blister_zones(hb=500, a=230, t=250, R=6000, deviation=12, width=600, L=2500)
    assert z["bursting"] == pytest.approx(600) and z["tie_back"] == (-750, 750) and z["bending"] == (-500, 1000)
    assert 2 * z["half_width"] == pytest.approx(3 * 230) and z["radial"] == pytest.approx(6000 * math.radians(12))
    assert z["zone"] == (1800, 3000)  # 8.2.1 note 1 (p78): 3 × the blister's width, its length + 2 × the wall
    assert td.blister_zones(500, 230, 250)["zone"] is None
    S, al = (180, 220), 12.0
    Lmin = 400 / math.sin(math.radians(al))
    ok = td.blister_detailing(hb=500, a=230, t=250, deviation=15, face_angle=90, L=Lmin + 1, S=S, alpha=al, R=6000,
                              kind="strand", d=5.0, stirrup_spacing=150, stirrup_extent=600, tie_back=(-750, 750),
                              bending=(-500, 1000), bar_half_width=345, radial_extent=6000 * math.radians(15),
                              plate="trumpet", spiral_length=300, trumpet_length=300)
    assert ok.ok and not ok.warnings and not any("skipped" in n for n in ok.notes)
    # figure 9.4.19: a face square to the tendon (the 90° allowed) and a slope parallel to it give L = (S1 + S2)/sin α
    B, T = np.zeros(2), 400 * np.array([-math.sin(math.radians(al)), math.cos(math.radians(al))])
    assert B[0] - (T[0] - T[1] / math.tan(math.radians(al))) == pytest.approx(Lmin)
    assert not td.blister_detailing(L=Lmin, S=S, alpha=al).warnings
    base = dict(hb=500, a=230, t=250)
    for kw, text in ((dict(deviation=15.1), "deviation"), (dict(face_angle=89), "angle"),
                     (dict(L=Lmin - 0.5, S=S, alpha=al), "blister length"),
                     (dict(R=3999, kind="strand", d=5.0), "radius"), (dict(stirrup_spacing=151), "spacing"),
                     (dict(stirrup_extent=599), "extent"), (dict(tie_back=(-700, 800)), "tie-back"),
                     (dict(tie_back=(-750, 700)), "tie-back"), (dict(bending=(-450, 1000)), "local-bending"),
                     (dict(bar_half_width=346), "either side"),
                     (dict(R=6000, alpha=12, radial_extent=1000), "radial"),
                     (dict(R=6000, deviation=10, radial_extent=td.blister_zones(0, 0, 0, R=6000, deviation=10)["radial"]
                           - 1), "radial"),
                     (dict(plate="flat", layers=3), "layers")):
        r = td.blister_detailing(**base, **kw)
        assert r.ok and len(r.warnings) == 1 and text in r.warnings[0], kw


# ── 9.4.17: segmental construction ──────────────────────────────────────────
def test_segmental_9_4_17():
    """9.4.17 (p111–112): end meshes ≥ 10 mm; wet joints ≥ 60 mm, not below the segments' class; web keys over ≥ 75 %
    of the depth; key height > 2 × the largest aggregate (the text; figure 9.4.17-2 writes ≥) and ≥ 35 mm; height : mean
    width = 1 : 2"""
    ok = td.segmental_detailing(end_mesh_d=10, wet_joint=60, wet_grade="C50", segment_grade="C50", key_extent=1875,
                                h=2500, key_width=375, web_width=500, key_height=45, aggregate=20, key_mean_width=90)
    assert not ok.warnings
    for kw, text in ((dict(end_mesh_d=8), "mesh"), (dict(wet_joint=59), "wet joint width"),
                     (dict(wet_grade="C45", segment_grade="C50"), "class"), (dict(key_extent=1874, h=2500), "depth"),
                     (dict(key_height=34), "35"), (dict(key_height=40, aggregate=20), "aggregate"),
                     (dict(key_height=40, key_mean_width=90), "1 : 2")):
        r = td.segmental_detailing(**kw)
        assert r.ok and len(r.warnings) == 1 and text in r.warnings[0], kw
    assert not td.segmental_detailing(key_height=40.01, aggregate=20).warnings
    assert any("75 %" in n for n in td.segmental_detailing(key_width=250, web_width=500).notes)


# ── 9.6.1–9.6.3: columns ────────────────────────────────────────────────────
def test_column_clear_spacing_on_a_circle():
    """the clear spacing of bars on a circle: from the bars' coordinates (independent of the section's depths)"""
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    th = 2 * math.pi * (np.arange(28) + 0.5) / 28
    rs = 750 - 75
    p = np.c_[rs * np.cos(th), rs * np.sin(th)]
    clear = np.linalg.norm(p[1] - p[0]) - 28
    r = pd.column_detailing(col)
    assert r.details["least clear spacing of longitudinal bars"]["value"] == pytest.approx(clear)
    # 9.6.1 item 1 (p117): 50–350 mm; 70 bars of φ28 on the same circle are too close, 8 too far apart
    assert warned(pd.column_detailing(jtg.circle(1500, 70, 28, 75)), "least clear spacing")
    assert warned(pd.column_detailing(jtg.circle(1500, 8, 28, 75)), "largest clear spacing")


def test_column_rules_9_6_1():
    """9.6.1 (p116–117): bars ≥ 12 mm, clear 50–350 mm, ≤ 5 % (should); ties closed, ≥ 8 mm and ≥ d/4, spacing ≤ 15 d,
    the least dimension (circle 0.8 D) and 400 mm, above 3 % ≤ 10 d and 200 mm; laps by 9.3.12 item 3 (p107); bars
    within max(150 mm, 15 tie diameters) of a corner bar; embedment ≥ la"""
    rect = jtg.rect(600, 800, concrete="C35", top=[(4, 25, 60)], bottom=[(4, 25, 60)])
    base = dict(clear=(80, 300), ties=(10, 300), closed=True, unrestrained=150, embedment=900, la=900)
    assert not pd.column_detailing(rect, **base).warnings
    for kw, text in ((dict(clear=(49, 300)), "least clear"), (dict(clear=(80, 351)), "largest clear"),
                     (dict(ties=(7.9, 300)), "tie diameter"), (dict(ties=(10, 376)), "tie spacing"),
                     (dict(closed=False), "not closed"), (dict(unrestrained=151), "corner"),
                     (dict(embedment=899), "embedment"), (dict(lap=("tension", 101)), "lap"),
                     (dict(lap=("compression", 201)), "lap")):
        r = pd.column_detailing(rect, **{**base, **kw})
        assert r.ok and len(r.warnings) == 1 and text in r.warnings[0], kw
    assert pd.column_detailing(rect, **base, lap=("compression", 200)).warnings == []
    assert pd.column_detailing(rect, ties=(12, 300), unrestrained=180).warnings == []  # 15 × 12 = 180 > 150
    assert warned(pd.column_detailing(rect, d=10), "longitudinal bar diameter")
    # d/4: φ36 bars need ties of 9 mm
    big = jtg.rect(800, 800, concrete="C35", top=[(4, 36, 60)], bottom=[(4, 36, 60)])
    assert warned(pd.column_detailing(big, ties=(8, 200)), "tie diameter")
    assert not pd.column_detailing(big, ties=(9, 200)).warnings
    # the least dimension: a 350 mm wide wall-like column
    thin = jtg.rect(350, 1200, concrete="C35", top=[(3, 16, 50)], bottom=[(3, 16, 50)])
    assert pd.column_detailing(thin, ties=(8, 240)).details["tie spacing"]["limit"] == pytest.approx(240)  # 15 × 16
    # more than 3 % steel: 10 d and 200 mm; exactly 3 % is not "more than"
    A = 600 * 600
    r3 = pd.column_detailing(d=25, rho=0.03, least=600, ties=(10, 375))
    assert not r3.warnings and r3.details["tie spacing"]["limit"] == pytest.approx(375)
    r4 = pd.column_detailing(d=25, rho=0.0301, least=600, ties=(10, 201))
    assert warned(r4, "tie spacing") and r4.details["tie spacing"]["limit"] == pytest.approx(200)
    assert warned(pd.column_detailing(d=25, rho=0.0501), "should")
    assert A > 0


def test_column_hoops_agree_with_shear_circle():
    """consistency of two code paths for the hoops of a round column (9.6.1), written to the same convention:
    jtg.shear_circle and pier_detailing.column_detailing. Mixed bar sizes: the spacing uses the thinnest bar, the hoop
    diameter the thickest (GB 50010-2010 9.3.2)"""
    col = jtg.circle(1200, 24, 25, 70, concrete="C35")
    mixed = col.replace(bar_d=np.r_[np.full(12, 32.0), np.full(12, 25.0)])
    for sec in (col, mixed):
        for hoop in ((10, 300), (10, 380), (8, 200), (7.9, 200)):
            a = jtg.shear_circle(sec, 100.0, (2, *hoop))
            b = pd.column_detailing(sec, ties=hoop)
            assert any("hoop" in w for w in a.warnings) == bool(b.warnings), (hoop, a.warnings, b.warnings)
    assert pd.column_detailing(mixed, ties=(8, 375)).details["tie spacing"]["limit"] == pytest.approx(375)
    assert warned(pd.column_detailing(mixed, ties=(7.9, 200)), "tie diameter")  # max(8, 32/4)


def test_spiral_column_9_6_2():
    """9.6.2 (p117): the rules axial_spiral applies (items 1, 2, 4) and item 3: embedment ≥ the column's diameter and
    ≥ la. Both functions report the same 9.6.2 rules"""
    col = jtg.circle(800, 12, 22, 60, concrete="C30")
    r = pd.column_detailing(col, indirect=(10, 50), embedment=800, la=700)
    assert not r.warnings and r.details["dcor"] == pytest.approx(2 * 340 + 22)
    assert warned(pd.column_detailing(col, indirect=(10, 50), embedment=799, la=700), "diameter")
    assert warned(pd.column_detailing(col, indirect=(10, 50), embedment=900, la=901), "la")
    thin = jtg.circle(800, 8, 16, 60, concrete="C30")  # 0.42 % of the core, pitch 35 mm
    bad = pd.column_detailing(thin, indirect=(8, 35))
    a = jtg.axial_spiral(thin, 3000.0, 3000.0, spiral=(8, 35))
    spiral_w = [w for w in a.warnings if "9.6.2" in w]
    assert len(spiral_w) == 2 and set(spiral_w) <= set(bad.warnings)
    assert warned(pd.column_detailing(col, indirect=(10, 90)), "pitch")
    # review M1: a spiral used as ordinary ties (a 5.3.1 column) is a 9.6.1 column: pass it as ties
    big = jtg.circle(1500, 28, 28, 75, concrete="C35")
    assert not pd.column_detailing(big, ties=(12, 150), closed=True).warnings
    assert warned(pd.column_detailing(big, indirect=(12, 150)), "pitch")  # only when counted by 5.3.2


def test_side_bars_9_6_3():
    """9.6.3 (p117): eccentric members of h ≥ 600 mm get side bars of 10–16 mm"""
    rows = dict(top=[(4, 25, 60)], bottom=[(4, 25, 60)])
    assert warned(pd.column_detailing(jtg.rect(500, 600, **rows), eccentric=True), "side bars")
    assert not pd.column_detailing(jtg.rect(500, 590, **rows), eccentric=True).warnings
    side = jtg.rect(500, 800, **rows, bars=[(2 * 113.1, 400, 12)])
    assert not pd.column_detailing(side, eccentric=True).warnings
    thin = jtg.rect(500, 800, **rows, bars=[(2 * 50.3, 400, 8)])
    assert warned(pd.column_detailing(thin, eccentric=True), "side bar diameter")
    assert pd.column_detailing(jtg.circle(1500, 28, 28, 75), eccentric=True).warnings == []


# ── 9.6.4–9.6.9 ─────────────────────────────────────────────────────────────
def test_wall_mesh_9_6_4_commentary():
    """9.6.4 (p118): ≥ 250 mm²/m each way, ≤ 400 mm; the commentary (p239): AASHTO's 264 mm²/m, 250 mm²/m being five
    8 mm bars a metre"""
    five8 = 5 * math.pi * 8 ** 2 / 4
    assert five8 > 250 and not pd.wall_mesh_detailing(As_h=five8, As_v=264, spacing=200).warnings
    assert warned(pd.wall_mesh_detailing(As_h=4 * math.pi * 8 ** 2 / 4), "horizontal")
    assert not pd.wall_mesh_detailing(As_v=250, spacing=400).warnings
    assert warned(pd.wall_mesh_detailing(spacing=401), "spacing")


def test_cap_and_tie_beams_9_6_5():
    """9.6.5 (p118): cap beams: stirrups ≥ 8 mm, ≤ 200 mm (should), side bars ≥ 12 mm, ≤ 200 mm (should), C25 at least
    for l/h ≤ 5. Tie beams: 0.8–1.0 D deep, 0.6–0.8 D wide (may), corner bars ≥ 16 mm, ties ≥ 8 mm at ≤ the shorter
    side and 400 mm"""
    assert not pd.cap_beam_detailing(6000, 1600, concrete="C30", stirrup=(8, 200), side=(12, 200)).warnings
    for kw, text in ((dict(stirrup=(7.9, 100)), "stirrup diameter"), (dict(stirrup=(10, 201)), "stirrup spacing"),
                     (dict(side=(11.9, 150)), "side bar diameter"), (dict(side=(12, 201)), "side bar spacing")):
        r = pd.cap_beam_detailing(6000, 1600, **kw)
        assert len(r.warnings) == 1 and text in r.warnings[0]
    assert warned(pd.cap_beam_detailing(6000, 1600, concrete=20), "concrete class")
    assert not pd.cap_beam_detailing(9000, 1600, concrete=20).warnings  # l/h > 5
    assert not pd.tie_beam_detailing(1500, h=1200, b=900, corner_d=16, tie=(8, 400)).warnings
    assert warned(pd.tie_beam_detailing(1500, corner_d=14), "corner")
    assert warned(pd.tie_beam_detailing(1000, h=800, b=350, tie=(8, 351)), "tie spacing")
    assert any("0.6–0.8" in n for n in pd.tie_beam_detailing(1500, h=1200, b=1300).notes)


def test_joint_anchorage_9_6_7_9_6_8():
    """9.6.7 (p118): end joints: straight ≥ la and ≥ 5 d past the column centre (should), or a 90° hook ≥ 0.4 la and
    15 d; interior: top bars through; bottom bars ≥ 12 d (ribbed) / 15 d (plain) when their strength is not used, la
    when it is. 9.6.8 (p118–119): column bars to the cap top, ≥ la from its soffit, or a hook ≥ 0.4 la and 12 d"""
    d, la = 28.0, 980.0
    assert not pd.beam_bar_anchorage(d, la=la, straight=980, beyond_centre=140).warnings
    assert warned(pd.beam_bar_anchorage(d, la=la, straight=979), "la")
    assert warned(pd.beam_bar_anchorage(d, la=la, straight=980, beyond_centre=139), "should")
    assert not pd.beam_bar_anchorage(d, la=la, hook=(392, 420)).warnings
    assert warned(pd.beam_bar_anchorage(d, la=la, hook=(391, 420)), "0.4 la")
    assert warned(pd.beam_bar_anchorage(d, la=la, hook=(392, 419)), "15 d")
    assert not pd.beam_bar_anchorage(d, joint="interior", bar="bottom", strength="none", straight=336).warnings
    assert warned(pd.beam_bar_anchorage(d, joint="interior", bar="bottom", strength="none", straight=335), "12 d")
    assert warned(pd.beam_bar_anchorage(d, joint="interior", bar="bottom", strength="none", ribbed=False,
                                        straight=419), "15 d")
    assert warned(pd.beam_bar_anchorage(d, la=la, joint="interior", bar="bottom", straight=979), "la")
    # the end joint's bottom bars without their strength follow the interior rule
    assert not pd.beam_bar_anchorage(d, bar="bottom", strength="compression", la=600, straight=600).warnings
    assert warned(pd.beam_bar_anchorage(d, joint="interior", through=False), "run through")
    assert not pd.column_bar_anchorage(d, la=la, to_top=True, embedment=980, splice_in_joint=False).warnings
    assert warned(pd.column_bar_anchorage(d, la=la, embedment=979), "la")
    assert not pd.column_bar_anchorage(d, la=la, hook=(392, 336)).warnings
    assert warned(pd.column_bar_anchorage(d, la=la, hook=(392, 335)), "12 d")
    assert warned(pd.column_bar_anchorage(d, to_top=False), "top")
    assert warned(pd.column_bar_anchorage(d, splice_in_joint=True), "spliced")
    with pytest.raises(ValueError):
        pd.beam_bar_anchorage(d, joint="corner")


def test_ramp_piers_9_6_9():
    """9.6.9 (p119): multi-bearing piers preferred, single-column single-bearing piers only where forced and never in a
    row; restrainers at transition piers and abutments"""
    r = pd.ramp_piers(["multi", "single", "multi", "single", "fixed"], restrainers=True)
    assert not r.warnings and any("single" in n for n in r.notes)
    assert warned(pd.ramp_piers(["multi", "single", "single", "multi"]), "consecutive")
    assert warned(pd.ramp_piers(["multi", "multi"], restrainers=False), "restrainers")
    with pytest.raises(ValueError):
        pd.ramp_piers(["double"])


def test_pile_cap_9_6_10():
    """9.6.10 (p119): depth ≥ 1.5 D (should) and ≥ 1.5 m; piles more than 3 D apart: main bars within 1.5 D of the pile
    centres, ≥ 0.1 % outside; one-way bars: cross bars ≥ 12 mm at ≤ 250 mm; mesh over the pile heads 1200–1500 mm²/m,
    12–16 mm; vertical ties ≥ 16 mm; piles 3 D or more apart: hangers ≥ 12 mm at ≤ 200 mm. The commentary (p240): the
    usual least spacing is 2.5 D, so hangers are rare"""
    ok = pd.pile_cap_detailing(h=1800, D=1200, spacing=3000, mesh=(1200, 1500, 12), tie_d=16)
    assert not ok.warnings and not any("hanger" in n for n in ok.notes)  # 2.5 D
    assert warned(pd.pile_cap_detailing(h=1499), "1.5 m")
    r = pd.pile_cap_detailing(h=1500, D=1200)
    assert len(r.warnings) == 1 and "should" in r.warnings[0]
    at3 = pd.pile_cap_detailing(D=1000, spacing=3000, outside_ratio=0.0)
    assert not at3.warnings and any("whole width" in n for n in at3.notes) and any("hanger" in n for n in at3.notes)
    above = pd.pile_cap_detailing(D=1000, spacing=3600, outside_ratio=0.00099, hanger=(12, 200))
    assert len(above.warnings) == 1 and "outside" in above.warnings[0]
    assert warned(pd.pile_cap_detailing(D=1000, spacing=3600, hanger=(11, 200)), "hanger diameter")
    assert warned(pd.pile_cap_detailing(D=1000, spacing=3600, hanger=(12, 201)), "hanger spacing")
    assert not pd.pile_cap_detailing(one_way=True, cross=(12, 250)).warnings
    assert warned(pd.pile_cap_detailing(one_way=True, cross=(12, 251)), "cross bar spacing")
    assert warned(pd.pile_cap_detailing(mesh=(1199, 1300, 14)), "direction 1")
    assert any("1200–1500" in n for n in pd.pile_cap_detailing(mesh=(1600, 1300, 14)).notes)
    assert warned(pd.pile_cap_detailing(mesh=(1300, 1300, 18)), "upper end")
    assert warned(pd.pile_cap_detailing(tie_d=14), "vertical tie")
    # where the main bars go
    assert pd.pile_cap_bar_zones([1000, 4000], 1000, 5000) == [(0.0, 5000.0)]  # 3 D: the whole width
    assert pd.pile_cap_bar_zones([1000, 4500], 1000, 5500) == [(0.0, 2500.0), (3000.0, 5500.0)]
    assert pd.pile_cap_bar_zones([1500, 4600, 7700], 1000, 9200) == [(0.0, 3000.0), (3100.0, 6100.0),
                                                                    (6200.0, 9200.0)]


# ── conventions ─────────────────────────────────────────────────────────────
def test_detailing_reports_never_fail_and_skip_missing_inputs():
    """detailing warns, it never decides `ok` (Check.report); a rule without its input is skipped and named"""
    for f in (td.girder_detailing, td.end_zone_detailing, td.blister_detailing, td.segmental_detailing,
              td.duct_detailing, pd.column_detailing, pd.wall_mesh_detailing, pd.cap_beam_detailing,
              pd.pile_cap_detailing):
        r = f()
        assert r.ok and r.detailing and not r.warnings, f.__name__
    assert any("skipped" in n for n in td.girder_detailing().notes)
    assert td.pretensioned_detailing("strand", 15.2).ok
    bad = td.end_zone_detailing(plate="flat", layers=1, stirrup_spacing=500, rho_v=0.0, closed=False)
    assert bad.ok and len(bad.warnings) == 5 and "WARNING" in str(bad)


# ── after the independent review (not published with this library) ─────────────────────────────────
def test_mixed_reinforcement_hogging_9_4_2():
    """review M2: 9.4.2 (p109) asks for ordinary bars at the tension face, the top one over interior supports"""
    tendons = dict(concrete="C50", tendons=[(1251, 300, 1100, 90)])
    top_only = jtg.tee(200, 2000, 2200, 180, top=[(16, 16, 50)], **tendons)
    bottom_only = jtg.tee(200, 2000, 2200, 180, bottom=[(10, 20, 60)], **tendons)
    assert not td.girder_detailing(top_only, pc_class="A", hogging=True).warnings
    assert warned(td.girder_detailing(top_only, pc_class="A"), "bottom")
    assert warned(td.girder_detailing(bottom_only, pc_class="B", hogging=True), "top")


def test_least_dimension_unknown_is_named_9_6_1():
    """review M3: 9.6.1 item 4 (p117) limits the tie spacing to the least dimension; for a section whose least
    dimension the program cannot tell (a box pier) the rule is skipped and named, or checked with least"""
    pier = jtg.box(3000, 2000, 300, 300, 600, concrete="C40", bottom=[(30, 25, 60)], top=[(30, 25, 60)])
    r = pd.column_detailing(pier, ties=(12, 300))
    assert not r.warnings and any("least dimension" in n for n in r.notes)
    assert warned(pd.column_detailing(pier, ties=(12, 301), least=300), "tie spacing")
    assert not any("least dimension" in n for n in pd.column_detailing(pier, ties=(12, 300), least=300).notes)
    ring = jtg.circle(2000, 40, 28, 80, hollow=1200)  # program convention: 0.8 × the outer diameter
    assert pd.column_detailing(ring).details["least_dimension"] == pytest.approx(1600)


def test_beam_bars_past_the_column_centre_9_6_7():
    """review L2: 9.6.7 item 1 1) (p118): the top bars shall run past the column centre line (> 0), should by ≥ 5 d"""
    r = pd.beam_bar_anchorage(28, la=980, straight=980, beyond_centre=0)
    assert len(r.warnings) == 1 and "should" not in r.warnings[0] and "centre line" in r.warnings[0]
    r = pd.beam_bar_anchorage(28, la=980, straight=980, beyond_centre=1)
    assert len(r.warnings) == 1 and "should" in r.warnings[0]


def test_review_low_items():
    """review L1, L4, L5, L6, L9 and the presence rules added to 9.4.13, 9.4.18 item 3, 9.6.8 item 1"""
    sec = jtg.tee(200, 2000, 2200, 180, concrete="C50", bottom=[(10, 20, 60)], tendons=[(1251, 1800, 1100, 97)])
    assert any("outer" in n for n in td.duct_detailing(sec).notes)  # L1: the section's duct is the outer one
    assert "4.1.8" in pd.ramp_piers(["single", "single"]).warnings[0]  # L4
    assert any("both h and b" in n for n in pd.tie_beam_detailing(1500, h=1200, tie=(8, 300)).notes)  # L5
    # L6: 9.6.10 item 6, figure 9.6.10 (p119): hangers between piles ≥ 3 D apart, beyond 1 D from each centre
    assert pd.hanger_zones([0, 3000], 1000) == [(1000.0, 2000.0)]
    assert pd.hanger_zones([0, 2500], 1000) == []
    assert pd.hanger_zones([0, 3600, 6100], 1200) == [(1200.0, 2400.0)]
    # L9: 1 : 2 within 5 % (program convention)
    assert not td.segmental_detailing(key_height=52, key_mean_width=100).warnings
    assert warned(td.segmental_detailing(key_height=53, key_mean_width=100), "1 : 2")
    # 9.4.13 (p111): bars round the anchors; 9.4.18 item 3 (p112): bars against spalling at the end face
    assert warned(td.end_zone_detailing(seal_bars=False), "sealing")
    assert warned(td.end_zone_detailing(spalling_bars=False), "spalling")
    assert not td.end_zone_detailing(seal_bars=True, spalling_bars=True).warnings
    # 9.6.8 item 1 (p118): column bars run through tie-beam joints
    assert warned(pd.column_bar_anchorage(28, through=False), "tie-beam")


def test_dense_anchors_are_per_web():
    """review L10: 8.2.2 works in a web's plane; two webs with anchors at the same heights are called one by one, and
    each web's group is the same as the other's"""
    a, y = 230.0, [700.0, 1300.0]
    assert rg.dense_anchors(y, a) == [[0], [1]]
    both = rg.dense_anchors(y + y, a)  # the two webs thrown together: wrongly paired, which is why it is per web
    assert both == [[0, 2], [1, 3]]
