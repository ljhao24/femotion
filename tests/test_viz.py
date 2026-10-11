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
