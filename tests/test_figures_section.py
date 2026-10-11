"""femotion.codes.jtg.figures: section drawings. The SVG is parsed back and checked against geometric facts that do not
go through the drawing code: the outline's area (shoelace) against RCSection.area, the bars' circles against As, the
covers against the input depths and the clear spacing section_detailing computes, the compression zone against
RCSection.part(x), the stress diagram's ends against the check's details, every label inside the page."""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg.figures import bar_layout, section_figure
from femotion.viz import TEXT_SIZE
from femotion.viz import text_box as _text_box

NS = "{http://www.w3.org/2000/svg}"
STRAND = jtg.tendon("strand", 1860)


# ── parsing ─────────────────────────────────────────────────────────────────
def _root(fig):
    s = fig.svg()
    assert not re.search(r"\bnan\b|\binf\b", s.lower())  # every coordinate a finite number
    return ET.fromstring(s), fig._transform()


def _rings(d):
    """subpaths of 'M x,y L x,y … Z' as lists of points"""
    out = []
    for sub in d.split("M")[1:]:
        pts = re.findall(r"(-?[\d.]+),(-?[\d.]+)", sub)
        out.append([(float(x), float(y)) for x, y in pts])
    return out


def _shoelace(r):
    a = np.asarray(r)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _path_area(el):
    """area of an even-odd path whose holes run the other way round (signed sum)"""
    return abs(sum(_shoelace(r) for r in _rings(el.get("d"))))


def _role(root, role, tag=None):
    return [e for e in root.iter() if e.get("data-role") == role and (tag is None or e.tag == NS + tag)]


def _world(T, x, y):
    """page → world (inverse of the figure's map)"""
    x0, _, _, y1 = T.box
    ox = T(x0, y1)[0]
    oy = T(x0, y1)[1]
    return x0 + (x - ox) / T.sx, y1 - (y - oy) / T.sy


def _concrete_area(root, T):
    a = sum(_path_area(e) for e in _role(root, "concrete", "path"))
    a += sum(math.pi * float(e.get("r")) ** 2 for e in _role(root, "concrete", "circle"))
    a -= sum(math.pi * float(e.get("r")) ** 2 for e in _role(root, "hole", "circle"))
    return a / T.sx ** 2


def _bars(root, T):
    """[(x, z, d)] in world mm of the drawn bars"""
    out = []
    for e in _role(root, "bar", "circle"):
        x, y = _world(T, float(e.get("cx")), float(e.get("cy")))
        out.append((x, -y, 2 * float(e.get("r")) / T.sx, int(e.get("data-row"))))
    return out


# ── sections ────────────────────────────────────────────────────────────────
def _tee():
    return jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(2, 28, 60), (2, 28, 120)], top=[(8, 16, 40)])


def _sections():
    pre = jtg.tee(200, 1600, 1600, 160, bf_bottom=700, hf_bottom=300, concrete="C50", bottom=[(8, 16, 50)],
                  top=[(10, 12, 40)], tendons=[(1251, 1450, 1150, 90)], tendon=STRAND, sigma_pe=1050)
    return {
        "rect": jtg.rect(400, 600, bottom=[(4, 25, 45)], top=[(2, 16, 40)]),
        "tee": _tee(),
        "I": jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                     top=[(16, 12, 45)]),
        "box": jtg.box(3200, 2000, 250, 220, 900, b_top=6000, concrete="C50", bottom=[(30, 16, 60)],
                       top=[(50, 16, 50)], bars=[(4 * 113.1, 1000, 12)]),
        "box3": jtg.box(5000, 2000, 250, 220, 1200, b_top=8000, concrete="C50", bottom=[(40, 16, 60)], top=[(60, 16, 50)]),
        "hollow": jtg.RCSection(jtg.concrete("C40"), jtg.rebar("HRB400"),
                                [jtg.rc.Rect(2000, 0, 3000), jtg.rc.Rect(1200, 400, 2600, -1)],
                                [(10 * 615.8, 60), (10 * 615.8, 2940), (2 * 615.8, 1500)], 3000,
                                bar_d=[28, 28, 28]),
        "circle": jtg.circle(1500, 28, 28, 75, concrete="C35"),
        "annulus": jtg.circle(2000, 36, 25, 70, concrete="C40", hollow=1200),
        "composite": jtg.composite(pre, b=2200, t=180, concrete="C40", top=[(14, 12, 50)]),
    }


@pytest.mark.parametrize("name", list(_sections()))
def test_outline_area_and_steel(name):
    """the drawn concrete (shoelace of the even-odd paths, circles) has the section's area; the bars' circles have
    the section's As, each row the count and diameter of its input"""
    s = _sections()[name]
    sec = s.ultimate if isinstance(s, jtg.Composite) else s
    root, T = _root(section_figure(s, width=1400))  # wide enough that no bar is drawn at the 0.6 px least radius
    assert _concrete_area(root, T) == pytest.approx(sec.area, rel=2e-5)  # page coordinates to 1e-4 px
    bars = _bars(root, T)
    assert sum(math.pi * d * d / 4 for *_, d, _ in bars) == pytest.approx(sec.bars[:, 0].sum(), rel=2e-4)
    for (a, z), dia in zip(sec.bars, sec.bar_d):
        if sec.ring is None:
            here = [b for b in bars if abs(b[1] - z) < 0.02]
            assert len(here) == round(a / (math.pi * dia * dia / 4))
            assert all(b[2] == pytest.approx(dia, rel=1e-4) for b in here)
    if isinstance(s, jtg.Composite):  # the topping in its own colour, b × t
        top = [e for e in _role(root, "concrete", "path") if e.get("data-part") == "topping"]
        assert len(top) == 1 and _path_area(top[0]) / T.sx ** 2 == pytest.approx(s.b * s.t, rel=2e-5)


@pytest.mark.parametrize("name", ["rect", "tee", "I", "box", "box3", "hollow", "composite"])
def test_bars_inside_the_concrete(name):
    """every bar inside the concrete (not in a cell) with at least 20 mm of concrete round it"""
    s = _sections()[name]
    root, T = _root(section_figure(s))
    rings = [[_world(T, *p) for p in r] for e in _role(root, "concrete", "path") for r in _rings(e.get("d"))]
    edges = [(r[i - 1], r[i]) for r in rings for i in range(len(r))]

    def inside(x, y):  # even-odd over all rings
        c = False
        for (x1, y1), (x2, y2) in edges:
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                c = not c
        return c

    def dist(x, y):
        out = math.inf
        for (x1, y1), (x2, y2) in edges:
            dx, dy = x2 - x1, y2 - y1
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
            out = min(out, math.hypot(x - x1 - t * dx, y - y1 - t * dy))
        return out

    for x, z, d, _ in _bars(root, T):
        assert inside(x, -z)
        assert dist(x, -z) >= d / 2 + 20 - 0.01


def test_layout_is_the_one_section_detailing_assumes():
    """with stirrups: the bottom row's outermost bars lie at the cover of the outermost steel + ds + d/2 from the web
    faces (section_detailing's side cover), and the drawn clear spacing equals the one it reports (9.3.3)"""
    sec = jtg.tee(300, 1300, 1600, 150, concrete="C40", bottom=[(4, 25, 60)], top=[(8, 16, 40)])
    det = jtg.section_detailing(sec, env="I", stirrup_d=10)
    root, T = _root(section_figure(sec, stirrup_d=10))
    row = sorted((x, d) for x, z, d, _ in _bars(root, T) if abs(z - (1300 - 60)) < 0.02)
    assert len(row) == 4
    side = det.details["cover of the outermost steel at the bottom"]["value"]
    assert row[0][0] - (-150) == pytest.approx(side + 10 + 12.5, abs=0.01)
    assert 150 - row[-1][0] == pytest.approx(side + 10 + 12.5, abs=0.01)
    gaps = [b[0] - a[0] - 25 for a, b in zip(row[:-1], row[1:])]
    assert gaps == pytest.approx([det.details["clear spacing of the bottom bars at depth 1240"]["value"]] * 3, abs=0.02)
    st = _role(root, "stirrup", "rect")
    assert len(st) == 1  # one loop round the web
    x0, _ = _world(T, float(st[0].get("x")), 0)
    assert x0 - (-150) == pytest.approx(side + 5, abs=0.01)  # stirrup centre line at side + ds/2


def test_side_bars_pair_up_and_box_rows_share_the_webs():
    """a side row of 2 sits on both faces of a T web; in a box's web zone 4 bars go 2 to each web"""
    t = jtg.tee(200, 1300, 1600, 150, bottom=[(2, 25, 60)], top=[(6, 16, 40)], bars=[(2 * 113.1, 700, 12)])
    rows = {round(r["z"]): r for r in bar_layout(t)}
    xs = sorted(x for x, *_ in rows[700]["bars"])
    assert rows[700]["face"] == "side" and xs[0] == pytest.approx(-xs[1]) and xs[1] > 0
    b = _sections()["box"]
    rows = {round(r["z"]): r for r in bar_layout(b)}
    xs = sorted(x for x, *_ in rows[1000]["bars"])
    assert len(xs) == 4 and xs[0] < -1600 + 450 and xs[1] < -1600 + 450 and xs[2] > 1600 - 450 and xs[3] > 1600 - 450


def test_given_positions_and_lumped_rows():
    sec = jtg.rect(400, 600, bottom=[(3, 25, 45)])
    rows = bar_layout(sec, bar_x={555: [-120, 0, 120]})
    assert [x for x, *_ in rows[0]["bars"]] == [-120, 0, 120]
    with pytest.raises(ValueError):
        bar_layout(sec, bar_x={555: [0, 1]})
    # ten 28 mm bars at one point in a 200 mm web: drawn in layers about the same centroid
    t = jtg.tee(200, 1100, 1600, 160, bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    r = [r for r in bar_layout(t) if r["face"] == "bottom"][0]
    zs = [z for _, z, _ in r["bars"]]
    assert r["layers"] > 1 and len(zs) == 10 and np.mean(zs) == pytest.approx(1100 - 110)
    assert max(zs) + 14 < 1100  # the stack kept inside the beam
    assert "图中按" in section_figure(t).svg()


def test_circle_bars_on_the_bar_circle():
    c = _sections()["circle"]
    root, T = _root(section_figure(c, stirrup_d=10))
    bars = _bars(root, T)
    assert len(bars) == 28
    for x, z, d, _ in bars:
        assert math.hypot(x, z - 750) == pytest.approx(c.ring, abs=0.01)
    hoop = _role(root, "stirrup", "circle")[0]
    assert float(hoop.get("r")) / T.sx == pytest.approx(c.ring + 14 + 5, abs=0.01)


def test_tendons_ducts_and_external():
    """ducts with their area, the tendon discs with Ap; external tendons in the cell with their area"""
    sec = jtg.box(3200, 2000, 250, 220, 900, b_top=6000, concrete="C50", bottom=[(30, 16, 60)], top=[(50, 16, 50)],
                  tendons=[(12 * 139, 1890, 1200, 90)] * 4, sigma_pe=1100, external=[(19 * 139 * 4, 1700, 1000)])
    root, T = _root(section_figure(sec))
    ducts, tend = _role(root, "duct", "circle"), _role(root, "tendon", "circle")
    assert len(ducts) == len(tend) == 4
    area = lambda els: sum(math.pi * float(e.get("r")) ** 2 for e in els) / T.sx ** 2  # noqa: E731
    assert area(ducts) == pytest.approx(sec.tendons[:, 3].sum(), rel=2e-4)
    assert area(tend) == pytest.approx(sec.tendons[:, 0].sum(), rel=2e-4)
    ext = _role(root, "external", "circle")
    assert len(ext) == 2
    assert area(ext) == pytest.approx(sec.external[:, 0].sum(), rel=2e-4)
    for e in ext:  # inside the cell: between the webs, above the bottom slab
        x, y = _world(T, float(e.get("cx")), float(e.get("cy")))
        r = float(e.get("r")) / T.sx
        assert -1600 + 450 + r < x < 1600 - 450 - r and 250 + r < -y < 2000 - 220 - r
    pre = jtg.rect(1000, 250, pretensioned=True, tendons=[(10 * 140, 200, 1200)], tendon=STRAND, bottom=[(8, 12, 30)])
    root, T = _root(section_figure(pre))
    assert len(_role(root, "strand", "circle")) == 10  # ten strands of 140 mm²


def _zone(root, T):
    z = _role(root, "compression", "path")
    assert len(z) == 1
    pts = [_world(T, *p) for r in _rings(z[0].get("d")) for p in r]
    ys = [y for _, y in pts]
    return _path_area(z[0]) / T.sx ** 2, -max(ys), -min(ys)


@pytest.mark.parametrize("Md", [1200.0, 6000.0, -900.0])
def test_compression_zone(Md):
    """the zone is the concrete within x of the compressed face: its area is RCSection.part(x) (of the section turned
    over for hogging), its depth x; the neutral axis at x/β"""
    sec = jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(4, 32, 70)], top=[(8, 16, 40)])
    if Md > 4000:  # the zone reaches the web
        sec = jtg.tee(200, 1300, 1000, 100, concrete="C40", bottom=[(4, 32, 70), (4, 32, 140)], top=[(4, 16, 40)])
    chk = jtg.flexure(sec, Md)
    x = chk.details["x"]
    root, T = _root(section_figure(sec, check=chk))
    area, z0, z1 = _zone(root, T)
    if Md > 0:
        assert z0 == pytest.approx(0, abs=0.01) and z1 == pytest.approx(x, abs=0.01)
        assert area == pytest.approx(sec.part(x)[0], rel=1e-4)
    else:
        assert z1 == pytest.approx(sec.h, abs=0.01) and z0 == pytest.approx(sec.h - x, abs=0.01)
        assert area == pytest.approx(sec.flipped().part(x)[0], rel=1e-4)
    if Md > 4000:
        assert x > 100  # the T-shaped zone
    na = _role(root, "neutral-axis", "line")[0]
    xc = x / sec.concrete.beta
    assert float(na.get("data-z")) == pytest.approx(xc if Md > 0 else sec.h - xc, abs=1e-3)
    assert _world(T, 0, float(na.get("y1")))[1] == pytest.approx(-(xc if Md > 0 else sec.h - xc), abs=0.01)


def test_compression_zone_of_circles_and_boxes():
    c = _sections()["circle"]
    chk = jtg.eccentric(c, 9800, 3100, l0=18000)
    root, T = _root(section_figure(c, check=chk))
    depth = 750 * (1 - math.cos(math.pi * chk.details["alpha"]))  # 5.3.8: the segment of central angle 2πα
    area, z0, z1 = _zone(root, T)
    assert z1 == pytest.approx(depth, rel=1e-4)
    assert area == pytest.approx(c.part(depth)[0], rel=1e-4)  # 720-gon
    assert area == pytest.approx(chk.details["alpha"] * c.area * (1 - math.sin(2 * math.pi * chk.details["alpha"]) /
                                                                 (2 * math.pi * chk.details["alpha"])), rel=1e-4)
    b = jtg.box(3200, 2000, 250, 220, 900, concrete="C30", bottom=[(30, 32, 60), (30, 32, 130)], top=[(30, 16, 50)])
    chk = jtg.flexure(b, 40000.0)
    assert chk.details["x"] > 250  # into the webs
    root, T = _root(section_figure(b, check=chk))
    area, z0, z1 = _zone(root, T)
    assert area == pytest.approx(b.part(chk.details["x"])[0], rel=1e-4)


def test_eccentric_strain_method_and_load_point():
    r = jtg.rect(500, 800, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    chk = jtg.eccentric(r, 1200, 300, l0=4000)
    root, T = _root(section_figure(r, check=chk))
    area, z0, z1 = _zone(root, T)
    assert z1 == pytest.approx(chk.details["x"], abs=0.01) and area == pytest.approx(500 * chk.details["x"], rel=1e-4)
    load = _role(root, "load", "circle")[0]
    assert -_world(T, 0, float(load.get("cy")))[1] == pytest.approx(r.zg - chk.details["e"], abs=0.01)
    root, T = _root(section_figure(r, check=jtg.eccentric(r, 1200, -300, l0=4000), hogging=True))
    area, z0, z1 = _zone(root, T)
    assert z1 == pytest.approx(800, abs=0.01)


def _stress_ends(root):
    p = _role(root, "stress", "polyline")[0]
    ax, k = float(p.get("data-axis")), float(p.get("data-scale"))
    pts = [tuple(map(float, q.split(","))) for q in p.get("points").split()]
    return [(ax - x) / k for x, _ in pts[1:-1]], [y for _, y in pts[1:-1]]


def _pc():
    return jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                   top=[(16, 12, 45)], tendons=[(12 * 139, 1650, 1200, 90)] * 2 + [(9 * 139, 1520, 1200, 80)],
                   tendon=STRAND, sigma_pe=1080, sigma_l6=80)


def test_stress_diagram_matches_the_checks():
    """the stress diagram's ends (page geometry ÷ its scale) are the edge stresses of the check, at the section's top
    and bottom fibres"""
    pc = _pc()
    chk = jtg.service_stresses(pc, 6000.0, pc_class="A")
    root, T = _root(section_figure(pc, check=chk, width=600))
    (s0, s1), (y0, y1) = _stress_ends(root)
    assert s0 == pytest.approx(chk.details["edge_stress"]["top"], abs=2e-3)
    assert s1 == pytest.approx(chk.details["edge_stress"]["bottom"], abs=2e-3)
    assert y0 == pytest.approx(T(0, 0)[1], abs=0.01) and y1 == pytest.approx(T(0, -pc.h)[1], abs=0.01)
    root, T = _root(section_figure(pc, check=chk, width=360))  # narrow: the diagram under the section
    (s0, s1), (y0, y1) = _stress_ends(root)
    assert s0 == pytest.approx(chk.details["edge_stress"]["top"], abs=2e-3) and y0 > T(0, -pc.h)[1]
    chk = jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="A")
    root, T = _root(section_figure(pc, check=chk, lang="en"))
    (s0, s1), _ = _stress_ends(root)
    d = chk.details
    assert s0 == pytest.approx(d["top"]["sigma_pc"] - d["top"]["sigma_st"], abs=2e-3)
    assert s1 == pytest.approx(d["bottom"]["sigma_pc"] - d["bottom"]["sigma_st"], abs=2e-3)
    p = _role(root, "stress", "polyline")[0]  # the exact values ride along
    exact = [d[e]["sigma_pc"] - d[e]["sigma_st"] for e in ("top", "bottom")]
    assert [float(v) for v in p.get("data-sigma").split()] == pytest.approx(exact)
    # class B: the cracked section's compression triangle from the edge stress to zero at x
    weak = pc.replace(tendons=pc.tendons[:1], sigma_pe=np.array([1080.0]))
    chk = jtg.service_stresses(weak, 7000.0, pc_class="B")
    root, T = _root(section_figure(weak, check=chk))
    (s0, s1), (y0, y1) = _stress_ends(root)
    assert s0 == pytest.approx(chk.demand, abs=2e-3) and s1 == pytest.approx(0.0, abs=2e-3)
    assert _world(T, 0, y1)[1] == pytest.approx(-chk.details["x"], rel=1e-4)


def test_cracked_section_dict():
    sec = _tee()
    r = jtg.cracked_section(sec)
    root, T = _root(section_figure(sec, check=r))
    area, z0, z1 = _zone(root, T)
    assert z1 == pytest.approx(r["x0"], abs=0.01)


def test_detailing_rings_what_is_not_met():
    """a rectangle whose bottom cover is short with stirrups: the bottom row ringed, a numbered callout, the clause"""
    r = jtg.rect(400, 600, concrete="C30", bottom=[(5, 25, 40)], top=[(2, 16, 40)])
    det = jtg.section_detailing(r, env="I", stirrup_d=10)
    assert any(w.startswith("cover of the outermost steel at the bottom") for w in det.warnings)
    fig = section_figure(r, check=det, stirrup_d=10)
    root, T = _root(fig)
    rings = _role(root, "warn", "rect")
    assert len(rings) == 1
    x0, y0 = float(rings[0].get("x")), float(rings[0].get("y"))
    x1, y1 = x0 + float(rings[0].get("width")), y0 + float(rings[0].get("height"))
    row = [e for e in _role(root, "bar", "circle") if abs(-_world(T, 0, float(e.get("cy")))[1] - 560) < 1e-3]
    assert len(row) == 5 and all(x0 < float(e.get("cx")) < x1 and y0 < float(e.get("cy")) < y1 for e in row)
    assert len(_role(root, "callout", "circle")) == 1
    s = fig.svg()
    assert "表 9.1.1" in s and "表 4.5.3" in s
    assert "table 9.1.1" in section_figure(r, check=det, lang="en").svg()


def _all_figures():
    """{lang: [figures]}: every section alone and the checks, at a phone's width and a page's"""
    s = _sections()
    pc = _pc()
    t = _tee()
    col = s["circle"]
    det_sec = jtg.tee(200, 1300, 1600, 150, concrete="C30", bottom=[(6, 28, 50)], top=[(8, 16, 40)])
    box = jtg.box(3200, 2000, 250, 220, 900, b_top=6000, concrete="C50", bottom=[(30, 16, 60)], top=[(50, 16, 50)],
                  tendons=[(12 * 139, 1890, 1200, 90)] * 4, sigma_pe=1100, external=[(19 * 139 * 4, 1700, 1000)])
    out = {}
    for lang in ("zh", "en"):
        out[lang] = []
        for w in (360, 560):
            out[lang] += [section_figure(v, lang=lang, width=w) for v in s.values()]
            out[lang] += [
                section_figure(t, check=jtg.flexure(t, 1500.0), lang=lang, width=w, stirrup_d=10, title="T"),
                section_figure(t, check=jtg.flexure(t, -600.0), lang=lang, width=w),
                section_figure(pc, check=jtg.service_stresses(pc, 6000.0, pc_class="A"), lang=lang, width=w),
                section_figure(pc, check=[jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="A"),
                                          jtg.flexure(pc, 9000.0, pc_class="A")], lang=lang, width=w),
                section_figure(box, check=jtg.flexure(box, 30000.0, pc_class="A"), lang=lang, width=w),
                section_figure(col, check=jtg.eccentric(col, 9800, 3100, l0=18000), lang=lang, width=w),
                section_figure(det_sec, check=jtg.section_detailing(det_sec, env="II", stirrup_d=10), stirrup_d=10,
                               lang=lang, width=w)]
    return out


@pytest.fixture(scope="module")
def figures():
    return _all_figures()


def test_every_label_on_the_page_and_english_has_no_chinese(figures):
    """every <text>'s estimated box (viz's measure) lies on the page; the English figures carry no CJK characters"""
    for lang, figs in figures.items():
        for fig in figs:
            root, T = _root(fig)
            W, H = float(root.get("width")), float(root.get("height"))
            txt = ""
            for e in root.iter(NS + "text"):
                cls = e.get("class").split()[0].removeprefix("fm-")
                m = re.search(r"rotate\(([-\d.]+)", e.get("transform") or "")
                b = _text_box(float(e.get("x")), float(e.get("y")), e.text, TEXT_SIZE.get(cls, 12.0),
                              e.get("text-anchor"), e.get("dominant-baseline"), float(m.group(1)) if m else 0.0)
                assert b[0] >= -0.5 and b[1] >= -0.5 and b[2] <= W + 0.5 and b[3] <= H + 0.5, (e.text, b, W, H)
                txt += e.text or ""
            if lang == "en":
                assert not re.search(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", txt), txt


def test_labels_do_not_overlap(figures):
    """the labels in the column right of the section keep apart"""
    for fig in figures["zh"] + figures["en"]:
        root, T = _root(fig)
        boxes = []
        for e in root.iter(NS + "text"):
            if e.get("data-role") == "label":
                cls = e.get("class").split()[0].removeprefix("fm-")
                boxes.append(_text_box(float(e.get("x")), float(e.get("y")), e.text, TEXT_SIZE[cls], "start", "middle"))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert a[3] <= b[1] + 0.5 or b[3] <= a[1] + 0.5 or a[2] <= b[0] or b[2] <= a[0], (a, b)


def test_repr_svg_and_errors():
    s = _tee()._repr_svg_()
    assert ET.fromstring(s).tag == NS + "svg" and "8Φ16" in s
    with pytest.raises(ValueError):
        section_figure(_tee(), lang="fr")
    with pytest.raises(TypeError):
        section_figure(_tee(), check=3.0)
