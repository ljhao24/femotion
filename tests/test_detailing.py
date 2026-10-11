"""Durability (JTG 3362-2018 4.5) and detailing (9.1, 9.2.5–9.2.8, 9.3).

Independent sources: the code's tables written out again here from the scanned original (printed pages noted) and
compared cell by cell; the commentary's own basis of table 9.1.4 (formula 9-1 with the bond stresses of table 9-1,
p226–227) and of the stirrup ratio (0.35 / fsv after ACI 318, p230); the commentary's equilibrium of a curved bar
(9.3.14, p231) redone by summing the radial pressures over an arc, and the soffit radius by the circumcircle of three
points; the figures 9.1.9-2 (bundle lap) and 9.1.8 (zone l); bars placed one by one and their clear distances measured
by brute force; physical properties (laps at least as long as anchorages, the minima ordered by severity of the
environment and by design life, bundles of equal area). Hand calculations with the clauses' own numbers guard the
implementation and the open or closed ends of every limit."""
import itertools
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import detailing as dt

ENVS = ("I", "II", "III", "IV", "V", "VI", "VII")
COLS = [(m, life) for m in ("superstructure", "substructure", "foundation") for life in (100, 50)]


def _warns(chk, text):
    return [w for w in chk.warnings if text in w]


# ── 4.5 ─────────────────────────────────────────────────────────────────────
def test_environment_classes_of_table_4_5_2():
    """table 4.5.2 (p20): seven classes I … VII; the program also reads 1 … 7 and the Roman numerals of the original"""
    assert tuple(jtg.ENVIRONMENTS) == ENVS
    assert dt.environment(3) == dt.environment("Ⅲ") == dt.environment("class iii") == "III"
    with pytest.raises(ValueError, match="table 4.5.2"):
        dt.environment("VIII")
    assert dt.member_category("pier") == "substructure" and dt.member_category("pile cap") == "foundation"
    with pytest.raises(ValueError, match="design life"):
        dt.min_concrete_class("I", "beam", 70)


def test_table_4_5_3_cell_by_cell():
    """table 4.5.3 (p20), written out from the scan: rows I … VII; columns superstructure 100, 50/30, substructure
    100, 50/30, foundation 100, 50/30; and its structure: a longer life and a harsher environment never ask less"""
    table = """
        I   C35 C30 C30 C25 C25 C25
        II  C40 C35 C35 C30 C30 C25
        III C40 C35 C35 C30 C30 C25
        IV  C40 C35 C35 C30 C30 C25
        V   C40 C35 C35 C30 C30 C25
        VI  C40 C35 C35 C30 C30 C25
        VII C40 C35 C35 C30 C30 C25"""
    for line in table.strip().splitlines():
        env, *cells = line.split()
        for (m, life), cell in zip(COLS, cells):
            g = int(cell[1:])
            assert jtg.min_concrete_class(env, m, life) == g
            if life == 50:
                assert jtg.min_concrete_class(env, m, 30) == g  # one column for 50 and 30 years
    for e in ENVS:
        for m in ("superstructure", "substructure", "foundation"):
            assert jtg.min_concrete_class(e, m, 100) >= jtg.min_concrete_class(e, m, 50)
            assert jtg.min_concrete_class(e, m, 100) >= jtg.min_concrete_class("I", m, 100)
        assert (jtg.min_concrete_class(e, "superstructure") >= jtg.min_concrete_class(e, "substructure")
                >= jtg.min_concrete_class(e, "foundation"))


def test_durability_joins_3_1_2_and_4_5_3_and_never_fails():
    """4.5.3 and 3.1.2 together (the larger governs); a detailing report is ok however many rules it misses"""
    r = jtg.durability("C30", "I", member="beam", life=100)
    assert r.ok and _warns(r, "C35") and r.details["min_class_4.5.3"] == 35
    assert not jtg.durability("C35", "I", life=100).warnings
    assert not jtg.durability("C35", "II", life=50).warnings and jtg.durability("C30", "II", life=50).warnings
    # a pile cap in environment I may be C25 by 4.5.3, but not with HRB400 bars (3.1.2)
    assert not jtg.durability("C25", "I", member="foundation").warnings
    assert _warns(jtg.durability("C25", "I", member="foundation", steel="HRB400"), "3.1.2")
    # prestressed: C40 by 3.1.2 although 4.5.3 asks only C35 (50 years)
    pc = jtg.durability("C35", "I", life=50, prestressed=True)
    assert len(pc.warnings) == 1 and "3.1.2" in pc.warnings[0] and any("4.5.4 item 2" in n for n in pc.notes)
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C30", bottom=[(2, 28, 60)])
    r = jtg.durability(sec, "III")
    assert r.ok and _warns(r, "C40") and jtg.flexure(sec, 100.0).ok  # strength checks keep their own verdict


# ── 9.1.1–9.1.3 cover and bundles ───────────────────────────────────────────
def test_table_9_1_1_cell_by_cell_and_notes():
    """table 9.1.1 (p97), written out from the scan, same columns as table 4.5.3; note 2 (factory precast: −5 mm, not
    below 20) and note 3 (pile caps and foundations on bedding or formwork: −20 mm, not below 30)"""
    table = """
        I   20 20 25 20 40 40
        II  30 25 35 30 45 40
        III 35 30 45 40 65 60
        IV  30 25 35 30 45 40
        V   30 25 40 35 45 40
        VI  35 30 40 35 60 55
        VII 35 30 45 40 65 60"""
    for line in table.strip().splitlines():
        env, *cells = line.split()
        for (m, life), cell in zip(COLS, cells):
            c = float(cell)
            assert jtg.cover_min(env, m, life) == c and jtg.cover_min(env, m, 30 if life == 50 else 100) == c
            assert jtg.cover_min(env, m, life, precast=True) == max(c - 5, 20)
            if m == "foundation":
                assert jtg.cover_min(env, m, life, bedding=True) == max(c - 20, 30)
    assert jtg.cover_min("I", "beam", 100, precast=True) == 20  # 20 − 5 floored at 20
    assert jtg.cover_min("II", "beam", 100, precast=True) == 25
    assert jtg.cover_min("III", "foundation", 100, bedding=True) == 45 and jtg.cover_min("I", "footing", bedding=True) == 30
    with pytest.raises(ValueError, match="note 3"):
        jtg.cover_min("I", "beam", bedding=True)
    for e in ENVS:  # a longer life never asks less; marine and abrasion the most for every member
        for m in ("superstructure", "substructure", "foundation"):
            assert jtg.cover_min(e, m, 100) >= jtg.cover_min(e, m, 50)
            assert jtg.cover_min("III", m, 100) >= jtg.cover_min(e, m, 100) <= jtg.cover_min("VII", m, 100)


def test_cover_items_1_and_3_by_geometry():
    """9.1.1: cover measured to the bar surface, ≥ the bar's diameter (item 1) and, for the outermost steel (here
    stirrups wrapped around the main bars), ≥ table 9.1.1 (item 3). Geometry: the lowest bar of diameter d at depth z
    has its surface h − z − d/2 from the bottom face; the stirrup's surface ds less"""
    h, d, ds = 800.0, 25.0, 10.0
    for cover_out in (19.9, 20.0, 35.0):
        z = h - (cover_out + ds + d / 2)
        sec = jtg.rect(400, h, concrete="C35", bottom=[(3, d, h - z)])
        r = jtg.section_detailing(sec, env="I", stirrup_d=ds)
        assert r.details["cover of the outermost steel at the bottom"]["value"] == pytest.approx(cover_out)
        assert bool(_warns(r, "table 9.1.1")) == (cover_out < 20)
    # item 1: a 32 mm bar with 30 mm to its surface (no stirrups outside it, e.g. a slab)
    s = jtg.rect(1000, 300, concrete="C35", bottom=[(5, 32, 30 + 16)])
    r = jtg.section_detailing(s, env="I")
    assert _warns(r, "9.1.1 item 1") and not _warns(r, "table 9.1.1")
    s = jtg.rect(1000, 300, concrete="C35", bottom=[(5, 32, 32 + 16)])
    assert not jtg.section_detailing(s, env="I").warnings
    # precast members: 5 mm less (note 2); outer_d separate for the top face
    sec = jtg.rect(400, 800, concrete="C40", bottom=[(3, 25, 25 + 10 + 12.5)], top=[(3, 16, 25 + 8)])
    assert _warns(jtg.section_detailing(sec, env="II", stirrup_d=10), "table 9.1.1")
    assert not _warns(jtg.section_detailing(sec, env="II", stirrup_d=10, outer_d=(10, 0), precast=True), "table 9.1.1")


def test_cover_over_50_mm_and_its_mesh_9_1_2():
    """9.1.2: over 50 mm of cover on the main bars calls for measures; a mesh in it: d ≥ 6, s ≤ 100, its cover ≥ 25"""
    s50 = jtg.rect(1000, 600, concrete="C35", bottom=[(5, 20, 50 + 10)])
    s51 = jtg.rect(1000, 600, concrete="C35", bottom=[(5, 20, 51 + 10)])
    assert not _warns(jtg.section_detailing(s50), "9.1.2") and _warns(jtg.section_detailing(s51), "9.1.2")
    assert not _warns(jtg.section_detailing(s51, cover_mesh=(6, 100, 25)), "9.1.2")
    assert len(_warns(jtg.section_detailing(s51, cover_mesh=(5, 101, 24)), "9.1.2")) == 3


def test_ducts_and_pretensioned_tendons_item_2():
    """9.1.1 item 2: a duct's cover ≥ half its diameter; a pretensioned tendon's ≥ its diameter"""
    D = 90.0
    for c in (44.9, 45.0):
        z = 1000 - c - D / 2
        sec = jtg.rect(400, 1000, concrete="C50", bottom=[(2, 20, 50)], tendons=[(1000, z, 1200, D)])
        assert bool(_warns(jtg.section_detailing(sec), "item 2")) == (c < 45)
    pre = jtg.rect(400, 600, concrete="C50", bottom=[(2, 12, 40)], tendons=[(140, 570, 1300)], pretensioned=True)
    assert any("tendon_d" in n for n in jtg.section_detailing(pre).notes)  # no diameter: noted, not checked
    for near in (22.0, 23.0):  # cover near − 7.6 against 15.2
        p = jtg.rect(400, 600, concrete="C50", bottom=[(2, 12, 40)], tendons=[(140, 600 - near, 1300)],
                     pretensioned=True)
        assert bool(_warns(jtg.section_detailing(p, tendon_d=15.2), "item 2")) == (near - 7.6 < 15.2)


def test_bundles_9_1_3():
    """9.1.3: de = √n d has the area of the n bars; single bars ≤ 36 mm; up to 3 bars when d ≤ 28 mm, 2 above; bars or
    bundles above 36 mm in the tension zone need a surface mesh (d ≥ 10 along at ≤ 100, d ≥ 6 across at ≤ 100, 5 de
    beyond the bundles)"""
    for n, d in ((2, 20), (3, 28), (2, 32)):
        de = dt.bundle_diameter(n, d)
        assert math.pi * de ** 2 / 4 == pytest.approx(n * math.pi * d * d / 4)

    def sec(n, d, rows=2):
        return jtg.rect(1200, 1200, concrete="C40", bottom=[(n * rows, d, 120)])

    assert not _warns(jtg.section_detailing(sec(3, 28), bundle=3, surface_mesh=(10, 100, 6, 100, 250)), "9.1.3")
    assert _warns(jtg.section_detailing(sec(4, 28), bundle=4), "at most 3")
    assert _warns(jtg.section_detailing(sec(3, 32), bundle=3), "at most 2")
    assert not _warns(jtg.section_detailing(sec(2, 32), bundle=2, surface_mesh=(10, 100, 6, 100, 230)), "9.1.3")
    assert _warns(jtg.section_detailing(sec(2, 32), bundle=2, surface_mesh=(10, 100, 6, 100, 226)), "overhang")


def test_bundle_mesh_boundaries():
    """9.1.3: the surface mesh starts above 36 mm (single bars or bundles); 5 de overhang"""
    s = jtg.rect(1200, 1200, concrete="C40", bottom=[(4, 25, 120)])
    assert not _warns(jtg.section_detailing(s, bundle=2), "surface mesh")  # de = 35.36
    s36 = jtg.rect(1200, 1200, concrete="C40", bottom=[(4, 36, 120)])
    assert not _warns(jtg.section_detailing(s36), "surface mesh")  # 36 mm single bars: not above 36
    s40 = jtg.rect(1200, 1200, concrete="C40", bottom=[(4, 40, 120)])
    r = jtg.section_detailing(s40)
    assert _warns(r, "surface mesh") and not _warns(r, "a bundle")  # single bars: no bundle rules
    m = jtg.section_detailing(s40, surface_mesh=(10, 100, 6, 100, 199))
    assert len(_warns(m, "surface mesh")) == 1 and "overhang" in _warns(m, "surface mesh")[0]
    assert not _warns(jtg.section_detailing(s40, surface_mesh=(10, 100, 6, 100, 200)), "surface mesh")


# ── 9.3.3 clear spacing (brute force) ───────────────────────────────────────
def _min_clear(xs, d):
    return min(abs(a - b) for a, b in itertools.combinations(xs, 2)) - d


@pytest.mark.parametrize("n,d,b", [(2, 28, 200), (3, 25, 300), (5, 32, 500), (4, 20, 250)])
def test_clear_spacing_against_bars_placed_one_by_one(n, d, b):
    """9.3.3 with the program's layout: n bars evenly between the stirrups' inner faces; the clear distance measured
    on bars placed one by one equals the program's value"""
    c, ds = 30.0, 10.0
    sec = jtg.rect(b, 900, concrete="C40", bottom=[(n, d, c + ds + d / 2)])
    r = jtg.section_detailing(sec, stirrup_d=ds, side_cover=c)
    key = next(k for k in r.details if k.startswith("clear spacing of the bottom"))
    x0, x1 = c + ds + d / 2, b - c - ds - d / 2
    xs = np.linspace(x0, x1, n)
    assert r.details[key]["value"] == pytest.approx(_min_clear(xs, d))
    assert bool(_warns(r, "9.3.3")) == (_min_clear(xs, d) < max(30, d) - 1e-9)


def test_clear_spacing_limits_and_layers():
    """9.3.3: ≥ 30 mm and ≥ d up to 3 layers; ≥ 40 mm and ≥ 1.25 d above; between layers alike; bundles by de"""
    assert dt.clear_spacing_min(25, 3) == 30 and dt.clear_spacing_min(32, 3) == 32
    assert dt.clear_spacing_min(25, 4) == 40 and dt.clear_spacing_min(36, 4) == 45
    # two layers 30 mm apart (clear) pass, 29 fail
    for gap in (29.0, 30.0):
        z1 = 25 + 10 + 12.5
        sec = jtg.rect(400, 900, concrete="C40", bottom=[(2, 25, z1), (2, 25, z1 + 25 + gap)])
        r = jtg.section_detailing(sec, stirrup_d=10)
        assert bool(_warns(r, "layers")) == (gap < 30)
    # four layers: 40 mm needed between them
    rows = [(2, 25, 47.5 + k * 60) for k in range(4)]  # 35 mm clear; they span 180 mm, beyond the 150 mm edge band
    r = jtg.section_detailing(jtg.rect(400, 1500, concrete="C40", bottom=rows), stirrup_d=10)
    assert r.details["layers (bottom)"] == 3 and not _warns(r, "layers")  # three main layers + one side row
    r = jtg.section_detailing(jtg.rect(400, 1500, concrete="C40", bottom=rows), stirrup_d=10, main_rows=4)
    assert r.details["layers (bottom)"] == 4 and len(_warns(r, "layers")) == 3
    # welded frames: layers in contact, ≤ 6 layers and d ≤ 32 (9.3.11)
    rows = [(2, 32, 50 + k * 32) for k in range(7)]
    r = jtg.section_detailing(jtg.rect(400, 1800, concrete="C40", bottom=rows), stirrup_d=10, welded_frame=True,
                              main_rows=7)
    assert _warns(r, "layers of the welded frame") and not _warns(r, "clear distance of the bottom layers")


# ── 9.3.7 side bars ─────────────────────────────────────────────────────────
def test_side_bars_of_webs_9_3_7():
    """9.3.7: side bars ≥ 0.001 b h per web (0.002 for welded frames in thin webs, commentary p229), 6–8 mm, spacing
    ≤ web width and 200 mm in the tension zone, ≤ 300 mm in the compression zone, 100–150 mm near supports"""
    b, h = 180, 1200

    def tee(rows):
        return jtg.tee(b, h, 1600, 160, concrete="C40", bottom=[(2, 28, 60)], top=[(8, 12, 40)],
                       bars=[(2 * math.pi * 16, z, 8) for z in rows])

    full = tee([400, 650, 820, 990])  # gaps from the web top (160) to the main bars (1140): 240, 250, 170, 170, 150
    r = jtg.section_detailing(full, stirrup_d=10)
    rho = 4 * 2 * math.pi * 16 / (b * h)
    assert r.details["side bars of the webs, As/(b h)"]["value"] == pytest.approx(rho) and rho > 0.001
    na = r.details["9.3.7 neutral axis"]  # the cracked axis, ≈ 190: every gap lies in the tension zone (≤ 180)
    assert na == pytest.approx(dt.cracked_axis(full)) and na < 200 and _warns(r, "tension zone")  # 250 > 180
    good = tee([340, 520, 700, 880, 1010])  # 1010: outside the bottom edge band (1140 − 120), a side bar
    r = jtg.section_detailing(good, stirrup_d=10)
    assert not _warns(r, "9.3.7"), r.warnings
    assert _warns(jtg.section_detailing(good, stirrup_d=10, near_support=True), "tension zone")  # 180 > 150
    assert _warns(jtg.section_detailing(full, stirrup_d=10, welded_frame=True), "As/(b h)")  # 0.186 % < 0.2 %
    assert not _warns(jtg.section_detailing(good, stirrup_d=10, welded_frame=True), "As/(b h)")  # 0.233 %
    thin = jtg.tee(b, h, 1600, 160, concrete="C40", bottom=[(2, 28, 60)], top=[(8, 12, 40)],
                   bars=[(2 * math.pi * 6.25, z, 5) for z in (340, 520, 700, 880, 1060)])
    assert _warns(jtg.section_detailing(thin, stirrup_d=10), "5 mm < 6 mm")
    big = jtg.tee(b, h, 1600, 160, concrete="C40", bottom=[(2, 28, 60)], top=[(8, 12, 40)],
                  bars=[(2 * math.pi * 36, z, 12) for z in (340, 520, 700, 880, 1060)])
    assert any("6–8 mm" in n for n in jtg.section_detailing(big, stirrup_d=10).notes)
    # hogging: the cracked axis lies near the bottom, the whole web is in tension (180 mm)
    assert _warns(jtg.section_detailing(tee([340, 700, 1060]), stirrup_d=10, hogging=True), "tension zone")
    # a given neutral axis at 1000 mm: gaps wholly above it are in the compression zone (300 mm)
    r = jtg.section_detailing(tee([340, 700, 1060]), stirrup_d=10, neutral_axis=1000)
    assert _warns(r, "compression zone 360 mm > 300") and _warns(r, "tension zone 360 mm > 180")
    r = jtg.section_detailing(tee([340, 640, 940, 1080]), stirrup_d=10, neutral_axis=950)
    assert not _warns(r, "compression zone") and not _warns(r, "tension zone")
    assert not jtg.section_detailing(jtg.rect(400, 800, concrete="C30", bottom=[(3, 20, 50)], top=[(3, 20, 50)]),
                                     stirrup_d=8).details.get("side bars of the webs, As/(b h)")  # rectangles: none


# ── 9.1.4 anchorage ─────────────────────────────────────────────────────────
TABLE_914 = {  # p98: la/d; C25, C30, C35, ≥C40; None for "—" or a missing column
    "HPB300": ((45, 40, 38, 35), (None,) * 4, (40, 35, 33, 30)),
    "HRB400": ((None, 30, 28, 25), (None, 35, 33, 30), (None, 30, 28, 25)),
    "HRB500": ((None, 35, 33, 30), (None, 45, 43, 40), (None, 35, 33, 30)),
}
GRADES = ("C25", "C30", "C35", "C40")


def test_table_9_1_4_cell_by_cell():
    """table 9.1.4 (p98) written out from the scan; ≥C40 covers C40 … C80; HRBF400 and RRB400 as HRB400"""
    for steel, rows in TABLE_914.items():
        for kind, row in zip(("compression", "straight", "hooked"), rows):
            for g, k in zip(GRADES, row):
                kw = dict(compression=kind == "compression", hook=kind == "hooked")
                if k is None:
                    with pytest.raises(ValueError, match="table 9.1.4"):
                        jtg.anchorage_length(20, steel, g, **kw)
                else:
                    assert jtg.anchorage_length(20, steel, g, **kw) == pytest.approx(20 * k)
                    if g == "C40":
                        assert jtg.anchorage_length(20, steel, "C80", **kw) == pytest.approx(20 * k)
    assert jtg.anchorage_length(25, "HRBF400", "C35") == jtg.anchorage_length(25, "RRB400", "C35") == 33 * 25


def test_anchorage_against_the_commentary_formula_9_1():
    """commentary 9.1.4 (p226–227): la = fsk d / (4 τ) with the bond stresses of table 9-1 (BS 5400); the straight-end
    values of table 9.1.4 lie within 0 … +7 % of it (rounded up by the drafters, 2 % below at most); hooked tension
    bars between 0.7 and 1.0 of the straight formula (a hook takes part of the force)"""
    tau = {"plain_t": (1.4, 1.6, 1.8, 1.9), "plain_c": (1.7, 1.9, 2.1, 2.1),
           "rib_t": (2.5, 2.8, 3.1, 3.3), "rib_c": (3.1, 3.5, 3.8, 4.1)}
    fsk = {"HPB300": 300, "HRB400": 400, "HRB500": 500}
    for steel in fsk:
        rib = "plain" if steel == "HPB300" else "rib"
        for i, g in enumerate(GRADES):
            for kind, kw in (("c", dict(compression=True)), ("t", {}), ("h", dict(hook=True))):
                try:
                    la = jtg.anchorage_length(1.0, steel, g, **kw)
                except ValueError:
                    continue
                f = fsk[steel] / (4 * tau[f"{rib}_{'c' if kind == 'c' else 't'}"][i])
                if kind == "h":
                    assert 0.7 * f <= la <= f
                else:
                    assert 0.97 * f <= la <= 1.07 * f, (steel, g, kind, la, f)


def test_anchorage_notes():
    """table 9.1.4 notes 2–5: epoxy tension ×1.25, disturbed ×1.25 (both: multiplied), bundles"""
    la = jtg.anchorage_length(25, "HRB400", "C30")
    assert jtg.anchorage_length(25, "HRB400", "C30", epoxy=True) == pytest.approx(1.25 * la)
    assert jtg.anchorage_length(25, "HRB400", "C30", disturbed=True) == pytest.approx(1.25 * la)
    assert jtg.anchorage_length(25, "HRB400", "C30", epoxy=True, disturbed=True) == pytest.approx(1.5625 * la)
    lc = jtg.anchorage_length(25, "HRB400", "C30", compression=True)
    assert jtg.anchorage_length(25, "HRB400", "C30", compression=True, epoxy=True) == lc  # note 3: tension bars
    assert jtg.anchorage_length(25, "HRB400", "C30", compression=True, hook=True) == lc  # hooks do not help
    # note 2: bundles in compression or de ≤ 28 anchored as one bar of de; otherwise 1.3, 2.6, 3.9 la, bar by bar
    d = 28 / math.sqrt(2)  # de = 28 exactly
    assert dt.bundle_anchorage(2, d, "HRB400", "C40") == [pytest.approx(30 * 28)] * 2
    assert dt.bundle_anchorage(2, d * 1.001, "HRB400", "C40")[1] == pytest.approx(2.6 * 30 * d * 1.001)
    assert dt.bundle_anchorage(3, 28, "HRB400", "C40") == pytest.approx([1.3 * 840, 2.6 * 840, 3.9 * 840])
    assert dt.bundle_anchorage(3, 28, "HRB400", "C40", compression=True) == pytest.approx(
        [25 * 28 * math.sqrt(3)] * 3)


# ── 9.1.5, 9.1.6 hooks ──────────────────────────────────────────────────────
def test_hooks_table_9_1_5_and_stirrup_hooks_9_1_6():
    """table 9.1.5 (p98–99): HPB300 180° (2.5d, 3d); ribbed 135° (5d, 5d), 90° (5d, 10d); bends ≤ 90° D ≥ 20d; epoxy
    note: D ≥ 5d (d ≤ 20) or 6d (d > 20), straight ≥ 5d. 9.1.6 (p99): stirrups 2.5 ds (HPB300), 5 ds (HRB400), straight
    5 ds, 10 ds seismic"""
    assert dt.hook_min(10, "HPB300", 180) == (25, 30)
    assert dt.hook_min(20, "HRB400", 135) == (100, 100) and dt.hook_min(20, "HRB500", 90) == (100, 200)
    assert dt.hook_min(20, "HPB300", 180, epoxy=True) == (100, 100)
    assert dt.hook_min(22, "HRB400", 135, epoxy=True) == (132, 110) and dt.hook_min(20, "HRB400", 90, epoxy=True) == (100, 200)
    for steel, angle in (("HPB300", 135), ("HRB400", 180)):
        with pytest.raises(ValueError, match="table 9.1.5"):
            dt.hook_min(12, steel, angle)
    assert dt.bend_min(25) == 500
    assert dt.stirrup_hook_min(10, "HPB300") == (25, 50) and dt.stirrup_hook_min(10, "HRB400", seismic=True) == (50, 100)


# ── 9.1.7–9.1.10 splices ────────────────────────────────────────────────────
def test_table_9_1_9_and_notes():
    """table 9.1.9 (p100): HPB300 C25 40d, ≥C30 35d; HRB400, HRBF400, RRB400 ≥C30 45d; HRB500 ≥C30 50d. Note 1:
    ribbed d > 25 +5d, d < 25 −5d (d = 25: the table); note 2 +5d; note 3: ≥ 300 / 200 mm; note 4: epoxy ×1.5; item 1:
    compression 0.7 ×; item 2: ×1.4 (25 % < share ≤ 50 %), ×1.6 (> 50 %), compression ×1.4 (> 50 %)"""
    assert jtg.lap_length(20, "HPB300", "C25") == 800 and jtg.lap_length(20, "HPB300", "C30") == 700
    assert jtg.lap_length(25, "HRB400", "C30") == 45 * 25 and jtg.lap_length(25, "HRB500", "C50") == 50 * 25
    assert jtg.lap_length(26, "HRB400", "C30") == 50 * 26 and jtg.lap_length(24, "HRBF400", "C30") == 40 * 24
    assert jtg.lap_length(28, "HPB300", "C30") == 35 * 28  # plain bars: no note 1 adjustment
    assert jtg.lap_length(25, "HRB400", "C30", disturbed=True) == 50 * 25
    assert jtg.lap_length(25, "HRB400", "C30", epoxy=True) == pytest.approx(1.5 * 45 * 25)
    # epoxy-coated bars below 25 mm: not the permitted −5 d of note 1; the longer of the two readings of notes 1 and 4
    # ((45 − 5) × 1.5 = 60 d, or 1.5 × 45 − 5 = 62.5 d): 1.5 × 45 = 67.5 d; above 25 mm (45 + 5) × 1.5 = 75 d ≥ 72.5 d
    assert jtg.lap_length(20, "HRB400", "C40", epoxy=True) == pytest.approx(67.5 * 20) and 67.5 > max(60, 62.5)
    assert jtg.lap_length(28, "HRB400", "C40", epoxy=True) == pytest.approx(75 * 28) and 75 > 72.5
    assert jtg.lap_length(20, "HRB400", "C40", epoxy=True, compression=True) == pytest.approx(0.7 * 40 * 20)
    assert jtg.lap_length(8, "HPB300", "C30") == 300 and jtg.lap_length(8, "HPB300", "C30", compression=True) == 200
    assert jtg.lap_length(25, "HRB400", "C30", compression=True) == pytest.approx(0.7 * 45 * 25)
    assert jtg.lap_length(25, "HRB400", "C30", compression=True, epoxy=True) == pytest.approx(0.7 * 45 * 25)
    ls = 45 * 25
    for share, f in ((0.0, 1.0), (0.25, 1.0), (0.2501, 1.4), (0.5, 1.4), (0.5001, 1.6), (1.0, 1.6)):
        assert jtg.lap_length(25, "HRB400", "C30", share=share) == pytest.approx(f * ls)
    for share, f in ((0.5, 1.0), (0.5001, 1.4)):
        assert jtg.lap_length(25, "HRB400", "C30", compression=True, share=share) == pytest.approx(0.7 * f * ls)
    with pytest.raises(ValueError, match="table 9.1.9"):
        jtg.lap_length(20, "HRB400", "C25")


def test_laps_are_never_shorter_than_anchorages():
    """physical: a lap splice transfers a bar's full force to another bar, so the tension lap of table 9.1.9 is at
    least the tension anchorage of table 9.1.4 (hooked for HPB300) for every steel, class and diameter, with and without
    the epoxy increase (note 2's +5 d for disturbed concrete does not keep up with the +25 % of table 9.1.4 note 4:
    HPB300 and HRB500 laps then come out shorter than the anchorage, see the documentation)"""
    for steel in ("HPB300", "HRB400", "HRB500"):
        for g in ("C25", "C30", "C35", "C40", "C50"):
            for d in (8, 12, 16, 20, 24, 25, 26, 28, 32):
                for kw in ({}, dict(epoxy=True)):
                    try:
                        la = jtg.anchorage_length(d, steel, g, hook=steel == "HPB300", **kw)
                        ls = jtg.lap_length(d, steel, g, **kw)
                    except ValueError:
                        continue
                    assert ls >= la - 1e-9, (steel, g, d, kw, ls, la)


def test_disturbed_concrete_laps_fall_short_of_anchorages_exactly_where_documented():
    """the inconsistency between table 9.1.9 note 2 (+5 d) and table 9.1.4 note 4 (+25 %) recorded in the
    documentation: with disturbed concrete the lap is shorter than the anchorage for HPB300 in C25–C35 and for HRB500 in
    C30 with d ≤ 25 mm and in C35 with d < 25 mm, nowhere else (d ≥ 8 mm, so the 300 mm floor does not hide it at
    the diameters listed)"""
    short = set()
    for steel in ("HPB300", "HRB400", "HRB500"):
        for g in ("C25", "C30", "C35", "C40", "C50"):
            for d in (12, 16, 20, 24, 25, 26, 28, 32):
                try:
                    la = jtg.anchorage_length(d, steel, g, hook=steel == "HPB300", disturbed=True)
                    ls = jtg.lap_length(d, steel, g, disturbed=True)
                except ValueError:
                    continue
                if ls < la - 1e-9:
                    short.add((steel, g, d))
    expect = {("HPB300", g, d) for g in ("C25", "C30", "C35") for d in (12, 16, 20, 24, 25, 26, 28, 32)}
    expect |= {("HRB500", "C30", d) for d in (12, 16, 20, 24, 25)} | {("HRB500", "C35", d) for d in (12, 16, 20, 24)}
    assert short == expect


def test_bundle_lap_and_weld_zone_figures():
    """figure 9.1.9-2 (p101): three bars lapped one after another at 1.3 ls, plus a through bar over four such steps,
    1.3 (n + 1) ls; figure 9.1.8 (p100): the zone l = 35 d, at least 500 mm (the two meet at d = 500/35)"""
    ls = 1000.0
    stagger, through = dt.bundle_lap(3, ls)
    assert stagger == pytest.approx(1.3 * ls) and through == pytest.approx(4 * stagger)
    assert dt.weld_zone(14) == 500 and dt.weld_zone(500 / 35) == pytest.approx(500) and dt.weld_zone(16) == 560


def test_splice_detailing():
    """9.1.7 (lap bars ≤ 28 mm, ≤ 32 mm for compression bars of compression members, none in tension members), 9.1.8
    (weld 5d / 10d, splice bars ≥ the bar, joints of one bar more than l apart, ≤ 50 % welded in tension, clear ≥ d and
    25), 9.1.9 (length, joints more than 1.3 ls apart, clear ≥ d and 25, HPB300 hooks), 9.1.10 (ribbed only, cover ≥
    20 mm and table 9.1.1, clear ≥ 25)"""
    lap = dt.splice_detailing
    assert not lap("lap", 28, length=1400, clear=28).warnings  # (45 + 5) × 28
    assert _warns(lap("lap", 32), "9.1.7") and not _warns(lap("lap", 32, compression=True, compression_member=True), "9.1.7")
    assert _warns(lap("lap", 36, compression=True, compression_member=True), "9.1.7")
    assert _warns(lap("lap", 20, member="tension"), "small-eccentricity")
    assert _warns(lap("lap", 28, length=1399), "lap length") and _warns(lap("lap", 20, clear=24), "clear distance")
    r = lap("lap", 25, share=0.5, joint_spacing=1.3 * 1.4 * 1125)
    assert r.details["ls_min"] == pytest.approx(1.4 * 1125) and _warns(r, "two laps") and r.notes
    assert not _warns(lap("lap", 25, share=0.5, joint_spacing=1.3 * 1.4 * 1125 + 1), "two laps")
    assert _warns(lap("lap", 12, steel="HPB300", hooks=False), "note 5")
    r = lap("lap", 20, bundle=3)
    assert r.details["bundle_through_bar"] == pytest.approx(1.3 * 4 * r.details["ls_min"])
    w = dt.splice_detailing("weld", 20, length=99, splice_area=300, joint_spacing=700, share=0.51, clear=24)
    assert len(w.warnings) == 5 and w.ok
    w = dt.splice_detailing("weld", 20, length=200, single_sided=True, splice_area=314.2, joint_spacing=700.1,
                            share=0.5, clear=25)
    assert not w.warnings
    assert not dt.splice_detailing("weld", 20, share=0.9, compression=True).warnings
    m = dt.splice_detailing("mechanical", 20, steel="HPB300", cover=19, clear=24, cmin=30)
    assert len(m.warnings) == 4 and _warns(m, "19 mm < 20 mm") and _warns(m, "30 mm (9.1.10, should)")
    m = dt.splice_detailing("mechanical", 20, cover=25, cmin=30)
    assert len(m.warnings) == 1 and "should" in m.warnings[0]
    assert not dt.splice_detailing("mechanical", 28, cover=20, clear=25).warnings


# ── 9.3.12 stirrups ─────────────────────────────────────────────────────────
def test_stirrups_9_3_12():
    """9.3.12 (p107): ds ≥ 8 mm and ≥ d/4; ρsv ≥ 0.14 % (HPB300), 0.11 % (HRB400) — the commentary's 0.35 / fsv
    (ACI 318, p230) gives 0.14 % and 0.106 %; spacing ≤ h/2 and 400, ≤ 15 d of compression bars held; laps: ≤ 5 d and
    100 (tension), ≤ 10 d and 200 (compression); ≤ 100 within h of a support; first stirrup one cover from the end;
    ≤ 50 mm from a junction; compression bars ≤ max(150, 15 ds) from a stirrup corner"""
    assert jtg.rebar("HPB300").rho_sv_min == pytest.approx(0.35 / 250)
    assert abs(jtg.rebar("HRB400").rho_sv_min - 0.35 / 330) < 5e-5  # 0.106 % printed as 0.11 %
    sec = jtg.tee(200, 1000, 1600, 160, concrete="C40", bottom=[(2, 32, 60), (2, 32, 120)], top=[(6, 16, 40)])
    assert not jtg.stirrup_detailing(sec, (2, 8, 200)).warnings  # 8 = 32/4; ρ = 0.251 %
    sec36 = sec.replace(bar_d=np.where(sec.bar_d == 32, 36.0, sec.bar_d))
    assert _warns(jtg.stirrup_detailing(sec36, (2, 8, 200)), "main bar / 4")
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 401)), "spacing") and not jtg.stirrup_detailing(sec, (2, 10, 400)).warnings
    low = jtg.rect(300, 600, concrete="C30", bottom=[(3, 20, 45)])
    assert _warns(jtg.stirrup_detailing(low, (2, 8, 301)), "stirrup spacing 301")  # h/2 = 300
    # ρsv at the limit, HRB400: 2 legs of 8 at s = 2·50.27/(0.0011·200) = 457 → spacing governs; use a wide web
    wide = jtg.rect(600, 1200, concrete="C30", bottom=[(4, 25, 60)])
    s_lim = 2 * math.pi * 16 / (0.0011 * 600)
    assert not _warns(jtg.stirrup_detailing(wide, (2, 8, s_lim * 0.999)), "ρsv")
    assert _warns(jtg.stirrup_detailing(wide, (2, 8, s_lim * 1.001)), "ρsv")
    assert _warns(jtg.stirrup_detailing(wide, (2, 8, s_lim * 0.9), stirrup_steel="HPB300"), "ρsv")
    r = jtg.stirrup_detailing(sec, (2, 10, 200), compression_d=12)
    assert _warns(r, "compression bars held") and any("closed" in n for n in r.notes)
    assert not _warns(jtg.stirrup_detailing(sec, (2, 10, 180), compression_d=12), "compression")
    assert not jtg.stirrup_detailing(sec, (2, 12, 150), corner_distance=180).warnings  # 15 × 12 = 180 > 150
    assert _warns(jtg.stirrup_detailing(sec, (2, 8, 150), corner_distance=151), "corner")
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), lap=("tension", 25, 101)), "lap")
    assert not jtg.stirrup_detailing(sec, (2, 10, 200), lap=("tension", 16, 80)).warnings  # 5 × 16 = 80
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), lap=("tension", 16, 81)), "lap")
    assert not jtg.stirrup_detailing(sec, (2, 10, 200), lap=("compression", 25, 200)).warnings
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), support_spacing=101), "support")
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), end_distance=40, end_cover=30), "end face")
    assert not jtg.stirrup_detailing(sec, (2, 10, 200), end_distance=30, end_cover=30).warnings
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), junction_distance=51), "junction")
    # 9.1.6 hooks: D ≥ 5 ds (HRB400) and larger than the main bar
    assert not jtg.stirrup_detailing(sec, (2, 10, 200), hook=(50, 50)).warnings  # 50 > 32
    assert _warns(jtg.stirrup_detailing(sec, (2, 8, 200), hook=(32, 40)), "not larger")
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), hook=(50, 50), seismic=True), "straight part")


# ── 9.3.14 curved soffit ────────────────────────────────────────────────────
def test_soffit_radius_is_the_circumcircle():
    """9.3.14-2 against the circumradius R = abc / (4 K) of the chord ends and the crown"""
    for l, f in ((6000, 600), (10000, 200), (4000, 2000)):
        p = np.array([(-l / 2, 0.0), (0.0, f), (l / 2, 0.0)])
        a, b, c = (np.linalg.norm(p[i] - p[j]) for i, j in ((0, 1), (1, 2), (2, 0)))
        u, v = p[1] - p[0], p[2] - p[0]
        K = abs(u[0] * v[1] - u[1] * v[0]) / 2
        assert dt.soffit_radius(l, f) == pytest.approx(a * b * c / (4 * K))


def test_curved_soffit_stirrups_by_equilibrium():
    """commentary 9.3.14 (p231) redone numerically: a bar under T = fsd As following an arc of radius r presses on the
    concrete with T/r per unit length; summing the radial pressure over one stirrup spacing (as vectors) gives
    2 T sin(sv / 2r) → T sv / r; two legs at fsv carry it. The clause's 9.3.14-1 is that balance."""
    As, fsd, fsv, r, sv = 2 * 615.8, 330.0, 250.0, 7800.0, 150.0
    T = As * fsd
    th0 = sv / (2 * r)
    th = np.linspace(-th0, th0, 20001)
    p = T / r * r  # radial pressure T/r times the arc element r dθ
    F = np.hypot(np.trapezoid(p * np.sin(th), th), np.trapezoid(p * np.cos(th), th))
    assert F == pytest.approx(2 * T * math.sin(th0), rel=1e-8)
    assert F == pytest.approx(T * sv / r, rel=th0 ** 2)  # the clause's linearised T sv / r
    asv1 = F / (2 * fsv)
    chk = dt.curved_soffit(As, 28, (12, sv), r=r, steel="HRB400", stirrup_steel="HPB300")
    assert chk.details["stirrup leg area Asv1"]["limit"] == pytest.approx(asv1, rel=th0 ** 2)
    assert chk.details["m"] == pytest.approx(330 / 250)
    # a sharp soffit, r = 1 m: 1.32 × 1231.6 × 150 / 2000 = 121.9 mm² per leg: 12 mm (113.1) short, 14 mm (153.9) enough
    assert _warns(dt.curved_soffit(As, 28, (12, sv), r=1000, steel="HRB400", stirrup_steel="HPB300"), "Asv1")
    assert not dt.curved_soffit(As, 28, (14, sv), r=1000, steel="HRB400", stirrup_steel="HPB300").warnings
    assert _warns(dt.curved_soffit(As, 14, (12, 141), r=r), "spacing") and not _warns(
        dt.curved_soffit(As, 14, (12, 140), r=r), "spacing")
    assert dt.curved_soffit(As, 28, (12, sv), chord=6000, rise=600).details["r"] == pytest.approx(7800)


# ── 9.3.8–9.3.11 beam bars ──────────────────────────────────────────────────
def test_cutoff_support_and_bent_bars():
    """9.3.8: la + h0 and 20 d (25 d epoxy); compression 15 d (20 d); 9.3.9: ≥ 2 and ≥ 1/5 through the support,
    10 d (12.5 d epoxy) past it; 9.3.10: h0/2, 20 d tension / 10 d compression, +25 % epoxy, 45°, no floating bars"""
    la = jtg.anchorage_length(25, "HRB400", "C40")
    assert dt.cutoff_lengths(25, 900, "HRB400", "C40") == (la + 900, 500)
    assert dt.cutoff_lengths(25, 900, "HRB400", "C40", epoxy=True) == (1.25 * la + 900, 625)
    assert dt.cutoff_lengths(25, 900, compression=True) == (None, 375)
    assert dt.cutoff_lengths(20, 500, "HPB300", "C30")[0] == 35 * 20 + 500  # hooked
    r = dt.cutoff_detailing(25, 900, concrete="C40", beyond_full=la + 899, beyond_needed=499, in_tension_zone=True)
    assert len(r.warnings) == 3 and r.ok
    assert dt.support_detailing(10, 2).ok and not dt.support_detailing(10, 2).warnings
    assert _warns(dt.support_detailing(12, 2), "through the support") and not dt.support_detailing(12, 3).warnings
    assert _warns(dt.support_detailing(4, 1), "through")
    assert _warns(dt.support_detailing(10, 2, d=25, extension=249), "extension")
    assert not dt.support_detailing(10, 2, d=25, extension=312.5, epoxy=True).warnings
    b = dt.bent_bar_detailing(25, 900, angle=45, start=450, anchorage=500)
    assert not b.warnings
    assert len(dt.bent_bar_detailing(25, 900, angle=60, start=449, anchorage=499, floating=True).warnings) == 4
    assert not dt.bent_bar_detailing(25, 900, anchorage=250, zone="compression").warnings
    assert _warns(dt.bent_bar_detailing(25, 900, anchorage=500, epoxy=True), "anchorage")  # 625 needed


def test_welded_frames_9_3_11():
    """9.3.11: ≤ 6 layers, d ≤ 32 mm; welds 5 d (inclined bars) and 2.5 d (short welds), doubled single-sided"""
    assert not dt.welded_frame_detailing(6, 32, weld=160, short_weld=80).warnings
    assert len(dt.welded_frame_detailing(7, 36, weld=159, short_weld=79).warnings) == 4
    assert _warns(dt.welded_frame_detailing(4, 20, weld=150, single_sided=True), "weld")  # 200 needed


# ── 9.2.5–9.2.8, 9.3.4–9.3.6, 9.3.1, 9.3.15–9.3.17 ─────────────────────────
def test_two_way_slab_9_2_5():
    """9.2.5: three strips each way, the edge ones a quarter of the short side wide (the strips fill the slab), half the
    middle strip's bars, spacing ≤ 250 mm and ≤ 2 t"""
    e, mx, my = dt.two_way_strips(6000, 4000)
    assert e == 1000 and 2 * e + mx == 6000 and 2 * e + my == 4000
    assert not dt.two_way_slab_detailing(6000, 4000, 200, middle=(12, 125), edge=(12, 250)).warnings
    assert _warns(dt.two_way_slab_detailing(6000, 4000, 200, middle=(12, 125), edge=(12, 251)), "/ middle")
    assert _warns(dt.two_way_slab_detailing(6000, 4000, 120, middle=(12, 125), edge=(12, 241)), "edge strip spacing")
    assert not _warns(dt.two_way_slab_detailing(6000, 4000, 120, middle=(12, 120), edge=(12, 240)), "spacing")


def test_skew_composite_and_hinged_slabs():
    """9.2.6 (15°, band of 3, obtuse corner ≥ 12 mm at 100–150 over ≥ 1.0 m), 9.2.7 (≥ 6 mm rough, connectors 10 d,
    ≤ 500 mm), 9.2.8 (groove 2/3 h, topping ≥ 80 mm)"""
    assert not dt.skew_slab_detailing(15, main="axis").warnings
    assert _warns(dt.skew_slab_detailing(15.1, main="axis"), "15°")
    assert not dt.skew_slab_detailing(30, monolithic=False, main="axis").warnings
    assert _warns(dt.skew_slab_detailing(30, main="support", edge_band=2), "free-edge")
    assert not dt.skew_slab_detailing(30, main="support", edge_band=3, obtuse=(12, 150, 1000)).warnings
    assert len(dt.skew_slab_detailing(30, obtuse=(10, 151, 999)).warnings) == 3
    assert dt.skew_slab_detailing(30, obtuse=(12, 90, 1200)).notes
    assert not dt.composite_slab_detailing(roughness=6, connectors=(12, 120, 120, 500)).warnings
    assert len(dt.composite_slab_detailing(roughness=5, connectors=(12, 119, 119, 501)).warnings) == 4
    assert not dt.hinged_slab_detailing(600, groove=400, topping=80).warnings
    assert len(dt.hinged_slab_detailing(600, groove=399, topping=79).warnings) == 2


def test_girder_slabs_9_3_4_to_9_3_6():
    """9.3.4 (9.2.2, 9.2.4 for the top slab; box: ≥ 3 straight bars per metre, ≥ 60 % of the root bars when lc > 2.5 m),
    9.3.5 (both layers per direction ≥ 0.4 %, 0.3 % prestressed; ≥ 10 mm, ≤ 300 mm), 9.3.6 (0.4 % beyond the width)"""
    r = dt.flange_slab_detailing(200, main=(12, 100), distribution=(10, 200), box=True, straight=(3, 1131))
    assert not r.warnings
    assert _warns(dt.flange_slab_detailing(200, main=(8, 210), box=True, straight=(2, 1000)), "9.2.2")
    r = dt.flange_slab_detailing(250, main=(16, 100), box=True, straight=(5, 1005), lc=2500, root_area=2011)
    assert not r.warnings
    r = dt.flange_slab_detailing(250, main=(16, 100), box=True, straight=(5, 1005), lc=2501, root_area=2011)
    assert _warns(r, "root") and r.details["straight top bars / root hogging bars"]["value"] < 0.6
    assert not dt.flange_slab_detailing(250, main=(16, 100), box=True, straight=(5, 1207), lc=3000,
                                        root_area=2011).warnings
    assert _warns(dt.flange_slab_detailing(200, main=(12, 100), outside=(200 * 300, 239)), "beyond")
    assert not dt.flange_slab_detailing(200, main=(12, 100), outside=(200 * 300, 240)).warnings
    # 9.3.5: t = 250, 12 @ 200 in two layers: 2 × 565.5 / 250000 = 0.452 %
    a12 = 2 * math.pi * 36 * 1000 / 200 / 250000
    r = dt.bottom_slab_detailing(250, longitudinal=[(12, 200), (12, 200)], transverse=[(12, 200), (12, 200)])
    assert a12 > 0.004 and not r.warnings
    r = dt.bottom_slab_detailing(300, longitudinal=[(12, 200), (12, 200)], transverse=[(12, 200)], prestressed=True)
    assert len(r.warnings) == 2  # 0.377 % ≥ 0.3 % longitudinally; one transverse layer: 0.19 % and a missing layer
    assert _warns(dt.bottom_slab_detailing(250, longitudinal=[(8, 301), (12, 100)], transverse=[(12, 100)] * 2),
                  "diameter")


def test_diaphragms_and_composite_beams():
    """9.3.1 (10 m with rigid connection; curved boxes below 240 m; cantilevers of 50 m), 9.3.15 (≤ 3 months),
    9.3.16 (≥ 150 mm, ≥ 6 mm), 9.3.17 (≥ 10 d)"""
    assert not dt.diaphragm_detailing("T", spacing=10000, rigid=True).warnings
    assert _warns(dt.diaphragm_detailing("T", spacing=10001, rigid=True), "spacing")
    assert not dt.diaphragm_detailing("T", spacing=12000).warnings  # hinged connection: no spacing rule
    assert _warns(dt.diaphragm_detailing("T", intermediate=False), "intermediate")
    assert not dt.diaphragm_detailing("box", radius=240000, intermediate=False).warnings
    assert _warns(dt.diaphragm_detailing("box", radius=239000, intermediate=False), "240 m")
    assert _warns(dt.diaphragm_detailing("box", radius=200000, spacing=10500), "spacing")
    assert not dt.diaphragm_detailing("box", radius=200000, spacing=10500, prestressed=True).warnings
    assert _warns(dt.diaphragm_detailing("box", cantilever=50000, mid_cantilever=False), "50 m")
    assert not dt.diaphragm_detailing("box", cantilever=49999, mid_cantilever=False).warnings
    assert not dt.composite_beam_detailing(topping=150, roughness=6, stirrups=(10, 100), age_difference=3).warnings
    assert len(dt.composite_beam_detailing(topping=149, roughness=5.9, stirrups=(10, 99),
                                           age_difference=3.1).warnings) == 4


def test_reports_are_ok_and_example_section():
    """every detailing report has ok = True whatever it warns about; the T girder of examples/jtg_rc_girder.py"""
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(5, 28, 60), (5, 28, 120)], top=[(8, 20, 50)])
    r = jtg.section_detailing(sec, env="III", stirrup_d=10)
    assert r.ok and _warns(r, "9.3.3")  # 5 bars of 28 mm side by side do not fit in a 200 mm web
    assert not _warns(r, "4.5.3") and r.details["cmin"] == 35
    assert str(r).startswith("section detailing") and "WARNING" in str(r)


def test_girder_dimensions_9_3_2():
    """9.3.2 (p104–105) for T and I girders: flange at the web ≥ h/10 with the haunch counted at most at 1:3 (as
    slab.haunch_depth for 4.2.6); precast flange tips ≥ 100, 140 with wet joints; web ≥ 160 and ≤ 15 (20) widths
    between the haunches; web transitions ≥ 12 Δb; longitudinal haunches ≤ 1/6"""
    assert not dt.beam_dimensions(1600, flange=160).warnings
    assert _warns(dt.beam_dimensions(1600, flange=159), "flange at the web")
    # haunch 300 long, 150 deep (slope 1:2): counted 100 deep (1:3)
    r = dt.beam_dimensions(2000, flange=100, haunch=(300, 150))
    assert r.details["flange at the web"]["value"] == pytest.approx(200) and not r.warnings
    assert _warns(dt.beam_dimensions(2010, flange=100, haunch=(300, 150)), "flange at the web")
    assert not dt.beam_dimensions(2000, tip=100).warnings and _warns(dt.beam_dimensions(2000, tip=139, wet_joint=True), "tip")
    assert _warns(dt.beam_dimensions(2000, web=159), "web width")
    assert not dt.beam_dimensions(2000, web=160, web_height=2400).warnings
    assert _warns(dt.beam_dimensions(2000, web=160, web_height=2401), "between the haunches")
    assert not dt.beam_dimensions(2000, web=160, web_height=3200, vertical_prestress=True).warnings
    assert not dt.beam_dimensions(2000, web_change=(200, 2400)).warnings
    assert _warns(dt.beam_dimensions(2000, web_change=(200, 2399)), "transition")
    assert not dt.beam_dimensions(2000, longitudinal_haunch=(6000, 1000)).warnings
    assert _warns(dt.beam_dimensions(2000, longitudinal_haunch=(6000, 1001)), "slope")



# ── review fixes ────────────────────────────────────────────────────────────
def test_web_zone_from_the_pieces_not_a_grid():
    """H1: the web is found from the pieces' boundaries, so a flange face on any depth (here every 50 mm of h with
    common flange thicknesses, where a grid of h/2000 used to land on the face and give a zero width) still gives the
    web from hf to h; the side bars are counted and their spacing checked. Boxes alike, from t_top to h − t_bottom."""
    for h in range(500, 3001, 50):
        for hf in (100, 120, 150, 160, 180, 200, 220, 250, 260, 300):
            if hf >= h / 2:
                continue
            band = min(0.1 * h, 150)  # keep the side rows clear of the edge bands of the top and bottom bars
            rows = list(np.arange(max(hf + 50, 40 + band + 20), h - 80 - band, 250))
            sec = jtg.tee(200, h, 1600, hf, concrete="C40", bottom=[(2, 25, 60)], top=[(8, 12, 40)],
                          bars=[(2 * math.pi * 16, z, 8) for z in rows])
            assert dt._web_zone(sec) == (hf, h) and dt._narrowest(sec) == 200
            r = jtg.section_detailing(sec, stirrup_d=10)
            assert r.details["side bars of the webs, As/(b h)"]["value"] == pytest.approx(
                len(rows) * 2 * math.pi * 16 / (200 * h))
            if not rows:
                continue
            pts = [hf] + rows + [h - 60]  # web top, side rows, bottom bars
            na = r.details["9.3.7 neutral axis"]
            gaps = [c - a for a, c in zip(pts[:-1], pts[1:]) if c > na]
            assert r.details["spacing of the side bars, tension zone"]["value"] == pytest.approx(max(gaps))
    for h in (1400, 2000, 2500):
        for tt, tb in ((200, 220), (250, 200), (280, 250)):
            b = jtg.box(5000, h, tt, tb, 900, b_top=11000, concrete="C50", bottom=[(60, 16, 60)], top=[(80, 16, 50)])
            assert dt._web_zone(b) == (tt, h - tb) and dt._narrowest(b) == 900
    # the reviewer's case: h = 1000, hf = 160, side bars at 500 and 840: 340 mm gaps
    sec = jtg.tee(200, 1000, 1600, 160, concrete="C40", bottom=[(2, 25, 60)], top=[(8, 12, 40)],
                  bars=[(2 * math.pi * 16, z, 8) for z in (500, 840)])
    assert _warns(jtg.section_detailing(sec, stirrup_d=10), "tension zone 340 mm > 200")
    # a flange face at 260 mm of 1000 (where RCSection.web on its own grid can read 0): stirrups still find b = 200
    s260 = jtg.tee(200, 1000, 1600, 260, concrete="C40", bottom=[(2, 25, 60)])
    r = jtg.stirrup_detailing(s260, (2, 10, 200))
    assert r.details["stirrup ratio ρsv"]["value"] == pytest.approx(2 * math.pi * 25 / (200 * 200))
    # a bar row exactly on the flange face takes the narrower width there, not 0
    face = jtg.tee(200, 1000, 1600, 160, concrete="C40", bottom=[(2, 25, 60)], top=[(2, 16, 160)])
    r = jtg.section_detailing(face, stirrup_d=10, side_cover=30)
    assert r.details["clear spacing of the top bars at depth 160"]["value"] == pytest.approx(200 - 2 * 40 - 32)


def test_tension_zone_of_reinforced_concrete_webs_is_the_cracked_one():
    """H2: 9.3.7 is about cracking of the web's tension zone (commentary p229), so for reinforced concrete the zone runs
    from the cracked section's neutral axis, far above the gross centroid. cracked_axis against a closed form (singly
    reinforced rectangle: x0 = h0 (−αρ + √((αρ)² + 2αρ))) and against jtg.cracked_section (another implementation,
    7.2.4); the reviewer's T girder: axis 199 mm, gross centroid 393 mm, the 230 mm gap from the web top is in tension."""
    b_, h_, d_ = 300, 700, 25
    rect = jtg.rect(b_, h_, concrete="C30", bottom=[(4, d_, 50)])
    h0, aE = h_ - 50, rect.steel.Es / rect.concrete.Ec
    ar = aE * 4 * math.pi * d_ * d_ / 4 / (b_ * h0)
    assert dt.cracked_axis(rect) == pytest.approx(h0 * (-ar + math.sqrt(ar * ar + 2 * ar)), rel=1e-9)
    t = jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 60)], top=[(8, 12, 40)],
                bars=[(2 * math.pi * 16, z, 8) for z in (380, 560, 740, 920, 1100)])
    assert dt.cracked_axis(t) == pytest.approx(jtg.cracked_section(t)["x0"], rel=1e-9)
    assert dt.cracked_axis(t, hogging=True) == pytest.approx(t.h - jtg.cracked_section(t, hogging=True)["x0"], rel=1e-9)
    t0 = jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 60)], top=[(8, 12, 40)])  # no side bars
    assert dt.cracked_axis(t0) == pytest.approx(199.1, abs=0.1) and t0.zg == pytest.approx(393.1, abs=0.1)
    r = jtg.section_detailing(t, stirrup_d=10)
    assert _warns(r, "tension zone 230 mm > 200")
    assert not _warns(jtg.section_detailing(t, stirrup_d=10, neutral_axis=t.zg), "9.3.7")  # the old reading let it pass
    # prestressed sections keep the gross centroid (an approximation, noted)
    pc = jtg.tee(200, 1300, 1600, 150, concrete="C50", bottom=[(4, 16, 50)], tendons=[(2000, 1150, 1100, 90)],
                 bars=[(2 * math.pi * 16, z, 8) for z in (380, 560, 740, 920, 1100)])
    r = jtg.section_detailing(pc, stirrup_d=10)
    assert r.details["9.3.7 neutral axis"] == pytest.approx(pc.zg) and any("approximation" in n for n in r.notes)


def test_side_faces_and_ducts_m1():
    """M1: the side cover against table 9.1.1 and item 1 when given; the ducts' side cover against D/2 when given,
    else a note that only the top and bottom faces were checked"""
    sec = jtg.tee(300, 1000, 1600, 160, concrete="C40", bottom=[(2, 32, 70)], top=[(8, 12, 40)])
    r = jtg.section_detailing(sec, env="III", stirrup_d=10, outer_d=(10, 0), side_cover=10)
    assert _warns(r, "cover of the outermost steel at the sides 10 mm < 35 mm") and _warns(r, "side cover of the bottom")
    assert not _warns(jtg.section_detailing(sec, env="III", stirrup_d=10, side_cover=35), "sides")
    assert any("side faces not checked" in n for n in jtg.section_detailing(sec, env="III", stirrup_d=10).notes)
    s = jtg.rect(1000, 300, concrete="C35", bottom=[(5, 32, 48)])  # slab, no stirrups: side cover ≥ d (item 1)
    assert _warns(jtg.section_detailing(s, side_cover=31), "side cover of the bottom bars 31 mm < 32")
    assert not _warns(jtg.section_detailing(s, side_cover=32), "side cover")
    pt = jtg.rect(400, 1200, concrete="C50", bottom=[(2, 20, 50)], tendons=[(2000, 1000, 1200, 90)])
    assert any("top and bottom faces only" in n for n in jtg.section_detailing(pt).notes)
    assert _warns(jtg.section_detailing(pt, tendon_side_cover=44.9), "side cover of the ducts")
    assert not _warns(jtg.section_detailing(pt, tendon_side_cover=45), "side cover of the ducts")
    pre = jtg.rect(400, 600, concrete="C50", bottom=[(2, 12, 40)], tendons=[(140, 560, 1300)], pretensioned=True)
    assert _warns(jtg.section_detailing(pre, tendon_d=15.2, tendon_side_cover=15), "side cover of the pretensioned")


def test_circle_and_box_sections():
    """circular piers: cover only (items 1, 3), the rest by 9.6.1; boxes: two webs by default for 9.3.7"""
    c = jtg.circle(1500, 28, 28, 60, concrete="C35")  # bar surface 46 mm from the face
    r = jtg.section_detailing(c, env="I", member="pier", stirrup_d=10)
    assert r.details["cover of the longitudinal bars"]["value"] == pytest.approx(46)
    assert not r.warnings and any("9.6.1" in n for n in r.notes)
    assert _warns(jtg.section_detailing(c, env="III", member="pier", stirrup_d=10), "36 mm < 45 mm")
    b = jtg.box(5000, 2000, 250, 220, 900, b_top=11000, concrete="C50", bottom=[(60, 16, 60)], top=[(80, 16, 50)],
                bars=[(4 * math.pi * 16, z, 8) for z in np.arange(400, 1800, 200)])
    r = jtg.section_detailing(b, stirrup_d=12)
    assert any("2 webs assumed" in n for n in r.notes) and not _warns(r, "spacing of the side bars")
    assert r.details["side bars of the webs, As/(b h)"]["value"] == pytest.approx(7 * 4 * math.pi * 16 / (900 * 2000))


def test_review_low_items():
    """L2 (ls < la noted), L3 (first stirrup's cover), L4 (a layer beyond the edge band noted), L5 (bundles on the
    tension face), L6 (thin webs), L8 (upper bounds noted), L9 ("Ⅱ类"), L10 (prestressed stirrups → 9.4.1), welded
    splice bars of another grade, "should" marked"""
    r = dt.splice_detailing("lap", 24, steel="HRB500", concrete="C30", disturbed=True)
    assert any("shorter than the anchorage" in n and "1350" in n for n in r.notes)  # ls 1200 < la 45·24·1.25
    assert not any("shorter" in n for n in dt.splice_detailing("lap", 24, concrete="C30").notes)
    sec = jtg.tee(200, 1000, 1600, 160, concrete="C40", bottom=[(2, 25, 60)])
    assert _warns(jtg.stirrup_detailing(sec, (2, 10, 200), end_distance=5, end_cover=30, cmin=20), "first stirrup")
    assert not jtg.stirrup_detailing(sec, (2, 10, 200), end_distance=30, end_cover=30, cmin=20).warnings
    low = jtg.rect(300, 500, concrete="C30", bottom=[(3, 25, 47.5), (3, 25, 47.5 + 52.5)])  # band 50 mm, 27.5 clear
    r = jtg.section_detailing(low, stirrup_d=10)
    assert not _warns(r, "layers") and any("main_rows" in n and "27.5" in n for n in r.notes)
    assert _warns(jtg.section_detailing(low, stirrup_d=10, main_rows=2), "27.5 mm < 30")
    bun = jtg.tee(300, 1200, 1600, 160, concrete="C40", bottom=[(4, 25, 60)], top=[(8, 16, 30)])
    r = jtg.section_detailing(bun, bundle=2)  # bottom pairs (de 35.4); the top bars single: cover 22 ≥ 16
    assert r.details["bundle"] == {"bottom": 2, "top": 1} and not _warns(r, "cover of the top bars")
    assert _warns(jtg.section_detailing(bun, bundle=(2, 2)), "cover of the top bars 22 mm < 22.6")
    t = jtg.tee(200, 1200, 1600, 160, concrete="C40", bottom=[(2, 28, 60)], top=[(8, 12, 40)],
                bars=[(2 * math.pi * 16, z, 8) for z in (340, 520, 700, 880)])  # 0.168 % of b h
    assert not _warns(jtg.section_detailing(t, stirrup_d=10), "As/(b h)")
    assert _warns(jtg.section_detailing(t, stirrup_d=10, thin_web=True), "As/(b h) 0.00168 < 0.002 (9.3.7, should)")
    assert any("1.0–1.5 m" in n for n in dt.skew_slab_detailing(30, obtuse=(12, 120, 1600)).notes)
    assert any("deeper" in n for n in dt.hinged_slab_detailing(600, groove=450).notes)
    assert not dt.hinged_slab_detailing(600, groove=440).notes[:-1]
    assert dt.environment("Ⅱ类") == dt.environment("II类") == dt.environment("Ⅱ") == "II"
    pc = jtg.tee(200, 1300, 1600, 150, concrete="C50", bottom=[(4, 16, 50)], tendons=[(2000, 1150, 1100, 90)])
    assert any("9.4.1" in n for n in jtg.stirrup_detailing(pc, (2, 12, 150)).notes)
    assert _warns(dt.splice_detailing("weld", 20, splice_steel="HPB300"), "same strength grade")
    assert not dt.splice_detailing("weld", 20, splice_steel="HRBF400").warnings
    r = dt.bottom_slab_detailing(250, longitudinal=[(8, 200)] * 2, transverse=[(12, 100)] * 2)
    assert "should" in _warns(r, "diameter")[0] and "should" not in _warns(r, "slab section")[0]
