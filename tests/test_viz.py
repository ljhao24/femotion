"""femotion.viz: plain SVG drawings. Checks of the page mapping, the self-growing margins and the output."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from femotion.viz import Figure, html, nice_ticks, text_width

NS = "{http://www.w3.org/2000/svg}"


def _parse(fig):
    return ET.fromstring(fig.svg())


def test_nice_ticks():
    assert nice_ticks(0, 40, 6) == [0, 5, 10, 15, 20, 25, 30, 35, 40] or nice_ticks(0, 40, 6) == [0, 10, 20, 30, 40]
    assert nice_ticks(-3500, 3500, 5) == [-3000, -2000, -1000, 0, 1000, 2000, 3000]
    t = nice_ticks(0.013, 0.087, 4)
    assert t[0] >= 0.013 - 1e-12 and t[-1] <= 0.087 + 1e-12 and len(t) >= 3
    assert 0.0 in nice_ticks(-1, 1) and all(isinstance(v, float) or isinstance(v, int) for v in nice_ticks(-1, 1))


def test_equal_scale_maps_world_to_page():
    """equal=True: one scale; a 1600 × 1300 rectangle keeps its aspect, y up becomes y down"""
    f = Figure(width=400, margin=(10, 10, 10, 10))
    f.rect(0, 0, 1600, -1300)
    T = f._transform()
    assert T.sx == pytest.approx(T.sy) == pytest.approx(380 / 1600)
    (x0, y0), (x1, y1) = T(0, 0), T(1600, -1300)
    assert (x0, y0) == pytest.approx((10, 10)) and x1 == pytest.approx(390) and y1 > y0
    assert T.H == pytest.approx(10 + 1300 * 380 / 1600 + 10)


def test_margins_grow_to_fit_labels():
    """a label far right of the drawing and a dimension below it: the page keeps them inside"""
    f = Figure(width=300)
    f.rect(0, 0, 100, 100)
    f.text((100, 50), "a rather long label beside the box", anchor="start", dx=6)
    f.dim((0, 0), (100, 0), offset=-20, label="100")
    T = f._transform()
    boxes = [b for it in f._items if it.extent for b in it.extent(T)]
    assert min(b[0] for b in boxes) >= 0 and max(b[2] for b in boxes) <= 300 + 1e-6
    assert max(b[3] for b in boxes) <= T.H + 1e-6
    assert text_width("受压区", 12) == pytest.approx(36) and text_width("ab", 10) == pytest.approx(11.6)


def test_chart_clips_data_and_draws_axes_underneath():
    c = Figure(width=500, height=240, equal=False).set_box(0, -1, 10, 1).axes(xlabel="x", ylabel="y")
    x = np.linspace(0, 10, 50)
    c.polyline(zip(x, 2 * np.sin(x)), "demand")
    root = _parse(c)
    clip = root.find(f"{NS}defs/{NS}clipPath")
    assert clip is not None
    groups = [g for g in root.iter(f"{NS}g") if g.get("clip-path")]
    assert len(groups) == 1 and groups[0].find(f"{NS}polyline") is not None
    kids = [k.tag for k in root]
    assert kids.index(f"{NS}line") < kids.index(f"{NS}g")  # grid before the data
    with pytest.raises(ValueError):
        Figure(equal=False)


def test_force_dimension_and_html():
    f = Figure(width=320, title="试样")
    f.rect(0, 0, 10, -5).force((5, 0), label="P").dim((0, -5), (10, -5), offset=-15, label="10 m")
    s = f.svg()
    root = ET.fromstring(s)
    assert root.get("class") == "fm-root" and "fm-loadhead" in s and "10 m" in s and "试样" in s
    page = html([f, "<p>text</p>", f])
    assert page.count("<svg") == 2 and "<p>text</p>" in page and page.startswith("<!doctype html>")
    ids = [ET.fromstring(Figure().rect(0, 0, 1, 1).svg()).get("id") for _ in range(2)]
    assert ids[0] != ids[1]  # several figures in one page: distinct clip-path ids


def test_dimension_label_reads_from_below_or_right():
    """vertical dimensions read from the right (text rotated −90°), horizontal ones upright"""
    f = Figure(width=200)
    f.rect(0, 0, 10, 10).dim((10, 10), (10, 0), offset=-12, label="10")
    T = f._transform()
    s = f.svg()
    assert "rotate(-90" in s
    g = Figure(width=200)
    g.rect(0, 0, 10, 10).dim((0, 0), (10, 0), offset=-12, label="10")
    assert "rotate(0 " in g.svg() or "rotate(-0 " in g.svg()
    assert math.isfinite(T.H)


# ── the public building blocks ──────────────────────────────────────────────
from femotion import viz  # noqa: E402
from femotion.viz import (EDGE, Note, Placer, beyond, callout, dim_chain, fmt, leader_column, moment_arc,  # noqa: E402
                          number, overlap, plain, shifted, split, spread, support_symbol, svg_arrow, svg_text,
                          text_box, text_item, wrap)


def _shoelace(pts):
    a = np.asarray(pts, float)
    return 0.5 * abs(float(np.dot(a[:, 0], np.roll(a[:, 1], -1)) - np.dot(np.roll(a[:, 0], -1), a[:, 1])))


def test_numbers():
    assert number(1234.4) == "1234" and number(-12.345) == "−12.3" and number(3.14159) == "3.14"
    assert number(0.012345) == "0.0123" and number(2.50, 2) == "2.5" and number(-0.0) == "0"
    assert number(math.inf) == "∞" and number(-math.inf) == "−∞" and number(7.0, 3) == "7"
    assert fmt(1.234567) == "1.23457" and fmt(-1e-9) == "0" and fmt(2.5, 2) == "2.5" and fmt(3.0) == "3"


def test_rich_text_subscripts_and_halo():
    assert plain("q_{k} = 10.5 kN/m") == "qk = 10.5 kN/m"
    assert text_width("q_{k}", 10) == pytest.approx(5.8 + 0.58 * 7.8)
    s = svg_text(10, 20, "M_{max} = −570", "label", anchor="start", halo=True)
    el = ET.fromstring(s)
    sub = el.findall("tspan")
    assert el.get("class") == "fm-label fm-halo" and el.text == "M"
    assert sub[0].text == "max" and sub[0].get("font-size") == "9.36px" and sub[1].text == " = −570"
    assert float(sub[0].get("dy")) == -float(sub[1].get("dy"))  # back on the baseline
    w, h = text_width("abc", 12), 1.25 * 12
    assert text_box(0, 0, "abc", 12, "end") == pytest.approx((-w, -h / 2, 0, h / 2))
    b = text_box(0, 0, "abc", 12, rotate=-90)
    assert b[2] - b[0] == pytest.approx(h) and b[3] - b[1] == pytest.approx(w)
    t, box = text_item(5, 5, "x_{c}", "small", anchor="start")
    assert "<tspan" in t and box == text_box(5, 5, "x_{c}", 10.5, "start")


def test_wrap():
    s = "控制截面 x_{c} = 25.50 m（正弯矩）：γ0Md = 1577.6 kN·m，Mu = 1941.5 kN·m，利用率 0.813 → 满足"
    for width in (160, 240, 320):
        lines = wrap(s, width, 12)
        assert "".join(lines).replace(" ", " ").replace(" ", "") == s.replace(" ", "")
        for ln in lines:
            assert text_width(ln, 12) <= width + 1e-9
            assert not ln.startswith(("m", "kN", "，", "）", "→")) and not ln.rstrip().endswith("（")
            assert "x_{c" not in ln or "x_{c}" in ln  # a subscript is never split
    f = "= 0.7748 × ΣP_{i}η_{i} = 0.7748 × (30×(−1.51) + 120×(−1.43) + 120×(−1.2) + 140×(−1.24))"
    lines = wrap(f, 260, 10.5, at=r"(?<= )(?=[+=−] )", indent=12)
    assert len(lines) > 1 and all(ln.startswith(("=", "+")) for ln in lines)
    assert all(text_width(ln, 10.5) <= 260 - (12 if k else 0) + 1e-9 for k, ln in enumerate(lines))
    assert wrap("short", 100) == ["short"]


def test_placer_priorities_marks_bounds_and_room():
    P = Placer(bounds=(0, 0, 200, 100))
    P.mark(50, 50, 6)
    P.add("low", [(50, 50, "middle"), (50, 80, "middle")], priority=0)
    P.add("high", [(50, 80, "middle"), (150, 20, "middle")], priority=5)
    P.add("gone", [(50, 50, "middle")])  # its only place is on the marker
    P.add("forced", [(50, 50, "middle")], required=True)
    P.add("outside", [(195, 50, "start")], inside=True)  # would leave the bounds
    P.add("narrow", [(100, 30, "middle")], room=(90, 110))  # wider than its room
    P.layout()
    texts = [ET.fromstring(s).text for s, _ in P.placed]
    assert texts == ["high", "forced"] and P.label_boxes[0][1] == pytest.approx(80 - 0.625 * 10.5)
    assert all("fm-halo" in s for s, _ in P.placed)
    b = P.place("low", [(50, 80, "middle"), (150, 50, "middle")])  # "high" took (50, 80)
    assert b[0] == pytest.approx(150 - text_width("low", 10.5) / 2)
    assert overlap((0, 0, 10, 10), (11, 0, 20, 10)) and not overlap((0, 0, 10, 10), (12, 0, 20, 10))
    assert shifted([(0, 0, "start")], -5, 2) == [(0, 0, "start"), (0, -5, "start"), (0, -10, "start")]


def test_spread_and_leader_column():
    assert spread([0, 5, 30], 10) == [0, 10, 30]
    ys = spread([90, 95, 100], 15, top=0, bottom=100)  # pulled up so the last stays at its anchor
    assert ys == pytest.approx([70, 85, 100])
    svg, boxes = leader_column([(10, 50, 3, "4Φ28", "label", None), (10, 52, 3, 2, "warntext", "#c0392b"),
                                (10, 20, 0, "8Φ16", "label", None)], 60)
    root = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{svg}</svg>')
    roles = [e.get("data-role") for e in root.iter() if e.get("data-role")]
    assert roles == ["label", "label", "callout"]  # in the anchors' order: 8Φ16 at 20, 4Φ28 at 50, ② below
    assert all(b2[1] >= b1[3] - 1e-9 for b1, b2 in zip(boxes, boxes[1:]))
    s, b = callout(30, 40, 3)
    assert ">3</text>" in s and b == (23, 33, 37, 47)


def test_dim_chain():
    svg, boxes = dim_chain([0, 100, 120, 300], 50, ["100", "20 000", "180"], ext=10, short="drop")
    root = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{svg}</svg>')
    texts = [e.text for e in root.iter(f"{NS}text")]
    assert texts == ["100", "180"]  # 20 000 does not fit its 20 px
    ticks = [e for e in root.iter(f"{NS}line") if e.get("class") == "fm-dimtick"]
    ext = [e for e in root.iter(f"{NS}line") if e.get("class") == "fm-dim" and e.get("x1") == e.get("x2")]
    assert len(ticks) == 4 and len(ext) == 4 and float(ext[0].get("y1")) == 13 and float(ext[0].get("y2")) == 53
    assert boxes[0][3] <= 50 - 7 + 0.625 * 10.5 + 1e-9  # above the line (side −1)
    svg, boxes = dim_chain([0, 10], 50, ["h0 = 1215"])  # a single dimension too short: beyond its far end
    assert boxes[0][0] > 10
    svg, boxes = dim_chain([200, 100, 20], 40, ["100", "8"], vertical=True, ext=[60, 60, 60])
    root = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{svg}</svg>')
    t = list(root.iter(f"{NS}text"))
    assert "rotate(-90" in t[0].get("transform") and float(t[0].get("x")) == 33  # left of the line, read from the right


def test_primitives():
    s = svg_arrow(0, 0, 0, 40, head=6)
    root = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{s}</svg>')
    tip = root.find(f"{NS}polygon").get("points").split()[0]
    assert tip == "0,40" and float(root.find(f"{NS}line").get("y2")) == pytest.approx(34)
    assert svg_arrow(1, 1, 1, 1) == ""
    assert moment_arc(0, 0, True).count("<polygon") == 1
    for kind in ("fixed", "pin", "roller", "guide"):
        svg, box = support_symbol(kind, 100, 100)
        assert svg and box[0] < 100 < box[2]
    with pytest.raises(ValueError):
        support_symbol("hinge", 0, 0)


def test_split_and_beyond():
    x = np.arange(6.0)
    assert split(x, [0, 1, np.nan, 3, np.inf, 5]) == [[(0, 0), (1, 1)], [(3, 3)], [(5, 5)]]
    parts = beyond(x, [0, 2, 0, -2, 0, 2], 1.0)
    assert parts[0] == [(0.5, 1.0), (1.0, 2.0), (1.5, 1.0)] and parts[1][0] == (4.5, 1.0)
    assert beyond(x, [0, 2, 0, -2, 0, 2], -1.0, above=False) == [[(2.5, -1.0), (3.0, -2.0), (3.5, -1.0)]]


def test_drawing_page_is_solved():
    """free height, equal scale: the largest scale at which a wide label fits; no band left under the drawing"""
    f = Figure(width=300)
    f.rect(0, 0, 1000, -400)
    f.text((1000, -200), "a label sticking out on the right", anchor="start", dx=6)
    T = f._transform()
    boxes = f._boxes(T)
    assert max(b[2] for b in boxes) == pytest.approx(300 - EDGE, abs=1e-3)  # the label touches the edge: largest
    assert T.sx < (300 - 24) / 1000
    assert T.H == pytest.approx(T(0, -400)[1] + 12)  # the bottom margin, exactly
    g = Figure(width=300)
    g.rect(0, 0, 1000, -400)
    assert g._transform().sx == pytest.approx((300 - 24) / 1000)


def test_chart_plot_height_legend_and_notes():
    c = Figure(width=420, plot_height=150, equal=False, title="T").set_box(0, -2000, 30, 2000)
    c.axes(xlabel="x (m)", ylabel="M (kN·m)")
    c.polyline([(0, 0), (30, 1500)], "demand")
    c.legend([("demand", "γ0Md"), ("resist", "Mu"), ("area-demand", "band"), ("mk-d", "point")], at="above")
    c.notes(["plain line", ("met → OK", "ok"), ("not met → NOT OK", "fail"), ("a remark", "small warn"),
             ("numbered", "small", 2), Note("= a + b + c", "small", breaks=r"(?<= )(?=[+=] )", indent=12)])
    T = c._transform()
    (bx0, by0), (_, by1) = T(0, -2000), T(30, 2000)
    assert by0 - by1 == pytest.approx(150)  # the plot box keeps its height
    root = _parse(c)
    texts = {e.text: e for e in root.iter(f"{NS}text") if e.text}
    assert float(texts["γ0Md"].get("y")) < by1 - 4 and float(texts["plain line"].get("y")) > by0 + 30
    assert texts["met → OK"].get("class") == "fm-note fm-noteok"
    assert texts["not met → NOT OK"].get("class") == "fm-note fm-notefail"
    assert texts["a remark"].get("class") == "fm-notesmall fm-notewarn"
    assert float(texts["numbered"].get("x")) == pytest.approx(bx0 + 20)  # after its callout
    swatch = {e.get("class"): e.tag.split("}")[1] for e in root.iter()
              if e.tag.split("}")[1] in ("line", "rect", "circle") and e.get("class") in
              ("fm-demand", "fm-resist", "fm-area-demand", "fm-mk-d")}  # the data is a polyline: these are the legend
    assert swatch == {"fm-demand": "line", "fm-resist": "line", "fm-area-demand": "rect", "fm-mk-d": "circle"}
    H = float(root.get("height"))
    assert H == pytest.approx(T.H) and all(float(e.get("y")) < H for e in root.iter(f"{NS}text"))
    # a narrow page wraps the legend into more rows: the plot box moves down, keeps its height
    n = Figure(width=200, plot_height=150, equal=False).set_box(0, 0, 1, 1)
    n.legend([("demand", "a long legend entry"), ("resist", "another long entry")], at="above")
    Tn = n._transform()
    assert Tn(0, 0)[1] - Tn(0, 1)[1] == pytest.approx(150) and Tn(0, 1)[1] > 16 * 2


def test_labels_keep_clear_and_inside():
    c = Figure(width=400, plot_height=120, equal=False).set_box(0, 0, 10, 10)
    c.dot((5, 5), 3, "mk-s").mark((5, 5), 4)
    c.label((5, 5), "first", alt=[((5, 5), 0, -14, "middle")])  # its first place is on the marker
    c.label((5, 5), "second", alt=[((5, 5), 0, -14, "middle"), ((5, 5), 0, 14, "middle")])
    c.label((10, 10), "outside the box", anchor="start", dx=4)  # no place inside: left out
    c.label((10, 10), "kept", anchor="start", dx=4, required=True, inside=False)
    root = _parse(c)
    t = [e.text for e in root.iter(f"{NS}text") if e.get("class") == "fm-small fm-halo"]
    assert t == ["first", "second", "kept"]
    assert list(root)[-1].tag == f"{NS}text"  # labels over everything
    c.vlines([2.5, 7.5])
    assert len([g for g in _parse(c).iter(f"{NS}g") if g.get("clip-path")]) == 2


def test_coordinates_read_back_to_1e_6():
    f = Figure(width=360)
    pts = [(0, 0), (1600, 0), (1600, -150), (900, -150), (900, -1300), (700, -1300), (700, -150), (0, -150)]
    f.polygon(pts, "concrete")
    T = f._transform()
    page = [tuple(map(float, p.split(","))) for p in _parse(f).find(f"{NS}polygon").get("points").split()]
    assert _shoelace(page) / T.sx ** 2 == pytest.approx(_shoelace(pts), rel=1e-6)


def test_stack_aligns_and_titles():
    a = Figure(width=400, plot_height=80, equal=False).set_box(0, 0, 10, 1).axes(ylabel="a")
    b = Figure(width=400, plot_height=80, equal=False).set_box(0, -1000, 10, 1000).axes(ylabel="long y ticks")
    s = viz.stack([a, b], title="stack")
    root = ET.fromstring(s.svg())
    subs = [e for e in root if e.tag == f"{NS}svg"]
    frames = [next(r for r in sv.iter(f"{NS}rect") if r.get("class") == "fm-axis") for sv in subs]
    assert frames[0].get("x") == frames[1].get("x") and frames[0].get("width") == frames[1].get("width")
    assert root.find(f"{NS}text").text == "stack" and float(subs[0].get("y")) == 24
    assert a.margin == (12, 12, 12, 12)  # restored
    with pytest.raises(ValueError):
        viz.stack([a, Figure(width=400, plot_height=80, equal=False).set_box(0, 0, 5, 1)]).svg()
