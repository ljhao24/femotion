"""femotion.frame.figures: what is drawn where. The SVG is parsed and every arrow, band, axle and influence ordinate is
compared with the load case or placement it comes from; labels must not overlap."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.frame import Model, Section, Solver
from femotion.frame.figures import elevation_figure, influence_figure, load_figure
from femotion.frame.influence import InfluenceLines, chain_sections
from femotion.frame.loads import LoadCase
from femotion.frame.prestress import Tendon
from femotion.viz import TEXT_SIZE, Figure
from femotion.viz import text_box as _box

NS = "{http://www.w3.org/2000/svg}"


def girder(L=16.0, n=16):
    m = Model()
    xs = np.linspace(0, 2 * L, 2 * n + 1)
    ns = [m.node((x, 0, 0)) for x in xs]
    sec = Section(E=3.25e7, A=0.5, Iy=0.1, Iz=1.0, J=0.01, mass=1.8)
    bs = [m.beam(a, b, sec) for a, b in zip(ns[:-1], ns[1:])]
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    m.fix(ns[n], (0, 1, 1, 1, 0, 0))
    m.fix(ns[-1], (0, 1, 1, 1, 0, 0))
    return m, ns, bs, xs


def pts(el):
    return [tuple(map(float, p.split(","))) for p in el.get("points").split()]


def groups(root, attr, value=None):
    return [g for g in root.iter(f"{NS}g") if g.get(attr) is not None and (value is None or g.get(attr) == value)]


def tip(g):
    """the tip of the (first) arrowhead in a group: the first point of its first polygon"""
    return pts(g.find(f"{NS}polygon"))[0]


def text_boxes(root):
    """page boxes of every label (titles excluded), rebuilt from the SVG with the drawing's own text metrics"""
    out = []
    for t in root.iter(f"{NS}text"):
        cls = (t.get("class") or "").replace("fm-", "").split()
        if not cls or cls[0] == "title":
            continue
        s = (t.text or "") + "".join(("_{" + (sp.text or "") + "}" if sp.get("font-size") else (sp.text or ""))
                                     for sp in t.findall(f"{NS}tspan"))
        if not s.strip():
            continue
        size = TEXT_SIZE.get(cls[0], 12.0)
        x, y = float(t.get("x")), float(t.get("y"))
        rot = t.get("transform")
        if rot and "rotate(0" not in rot:
            continue
        out.append((s, _box(x, y, s, size, t.get("text-anchor") or "start", t.get("dominant-baseline") or "auto")))
    return out


def assert_no_overlap(fig):
    boxes = text_boxes(ET.fromstring(fig.svg()))
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            (sa, A), (sb, B) = boxes[a], boxes[b]
            overlap = A[0] < B[2] - 0.5 and B[0] < A[2] - 0.5 and A[1] < B[3] - 0.5 and B[1] < A[3] - 0.5
            assert not overlap, f"labels overlap: {sa!r} and {sb!r}"


def test_load_figure_positions():
    m, ns, bs, xs = girder()
    case = (LoadCase("组合").self_weight().point(bs, 6.3, 120.0).line(bs, 8.0, start=18.0, end=28.0, q_end=20.0)
            .nodal(ns[16], M=(0, 80, 0)).nodal(ns[-1], F=(-50, 0, 0)).settlement(ns[16], -0.01))
    fig = load_figure(m, case, width=720)
    assert isinstance(fig, Figure) and fig.title == "组合" and fig._repr_svg_().startswith("<svg")
    T = fig._transform()
    root = ET.fromstring(fig.svg())
    # supports from the restrained dofs: pin, then rollers
    kinds = {int(g.get("data-node")): g.get("data-support") for g in groups(root, "data-support")}
    assert kinds == {ns[0]: "pin", ns[16]: "roller", ns[-1]: "roller"}
    # the point load: head on the member at x = 6.3 m, pointing down
    (g,) = [g for g in groups(root, "data-load", "point") if g.get("data-item") == "1"]
    x, y = tip(g)
    assert (x, y) == pytest.approx(T(6.3, 0.0), abs=0.02)
    line = g.find(f"{NS}line")
    assert float(line.get("y1")) < float(line.get("y2")) <= y  # from above, down to the head
    # the trapezoid band covers 18 … 28 m, the self-weight band the whole bridge; arrow heads within
    bands = {g.get("data-item"): g for g in groups(root, "data-load", "band")}
    for item, (u0, u1) in (("2", (18.0, 28.0)), ("0", (0.0, 32.0))):
        b = bands[item]
        assert float(b.get("data-x0")) == pytest.approx(T(u0, 0)[0], abs=0.02)
        assert float(b.get("data-x1")) == pytest.approx(T(u1, 0)[0], abs=0.02)
        heads = [pts(p)[0] for p in b.findall(f"{NS}polygon")[2:] if "loadhead" in p.get("class")]
        assert heads and all(T(u0, 0)[0] - 0.02 <= hx <= T(u1, 0)[0] + 0.02 for hx, _ in heads)
    # the trapezoid's height grows from 8 to 20 kN/m: its top edge
    top, *bottom = [pts(pl) for pl in bands["2"].findall(f"{NS}polyline")]
    bottom = bottom[0] if bottom else [(x, T(0, 0)[1]) for x, _ in top]  # stacked bands have their own base line
    h0, h1 = bottom[0][1] - top[0][1], bottom[-1][1] - top[-1][1]
    assert h1 / h0 == pytest.approx(20.0 / 8.0, rel=3e-3)
    # the moment and the horizontal force, the settlement
    assert groups(root, "data-load", "moment") and groups(root, "data-load", "settlement")
    (gh,) = [g for g in groups(root, "data-load", "point") if g.get("data-item") == "4"]
    hx, hy = tip(gh)
    assert hx == pytest.approx(T(32.0, 0)[0], abs=0.02)
    assert float(gh.find(f"{NS}line").get("x1")) > hx  # −50 kN along x: from the right, pointing left
    # span dimensions
    s = fig.svg()
    assert s.count(">16 m<") == 2
    assert_no_overlap(fig)


def test_tendon_temperature_and_lang():
    m, ns, bs, xs = girder()
    e = 0.4 * np.cos(np.pi * xs / 16.0)
    case = (LoadCase("pre").tendon(Tendon(bs, ecc=e, force=3000.0))
            .imposed(bs, curvature=-2e-4, label="温度梯度"))
    fig = load_figure(m, case, width=600)
    T = fig._transform()
    root = ET.fromstring(fig.svg())
    (g,) = groups(root, "data-load", "tendon")
    k = float(g.get("data-scale"))
    line = pts(g.find(f"{NS}polyline"))
    assert len(line) == 2 * len(bs)
    xs_line = np.array([p[0] for p in line])
    ys_line = np.array([p[1] for p in line])
    for xx, ee in zip(xs, e):
        idx = np.argmin(np.abs(xs_line - T(xx, 0)[0]))
        assert ys_line[idx] == pytest.approx(T(xx, 0)[1] + ee * k, abs=0.02)
    assert max(abs(e)) * k == pytest.approx(16.0)
    assert groups(root, "data-load", "imposed") and "温度梯度" in fig.svg()
    en = load_figure(m, LoadCase("dead").self_weight(), lang="en")
    assert "self-weight" in en.svg()
    with pytest.raises(ValueError):
        load_figure(m, LoadCase("x").self_weight(), lang="fr")
    assert "data-support" in elevation_figure(m).svg()


@pytest.mark.parametrize("width", [720, 380])
def test_influence_figure_lane(width):
    m, ns, bs, xs = girder()
    sv = Solver(m)
    secs = chain_sections(bs)
    il = InfluenceLines(sv, ns, secs)
    live = jtg.LiveLoad(L0=16.0, lanes=1, transverse=1.0, distribution=0.55, mu=0.3)
    for comp, k, which in (("My", 7, 0), ("My", 31, 1), ("Vz", 32, 1), ("Vz", 0, 1)):
        lay = live.layouts(il, comp, k)[which]
        fig = influence_figure(il, lay, width=width)
        T = fig._transform()
        root = ET.fromstring(fig.svg())
        # the line: x by position, y = y0 − η × scale
        (g,) = groups(root, "data-il", "line")
        y0, sc = float(g.get("data-y0")), float(g.get("data-scale"))
        p = np.array(pts(g.find(f"{NS}polyline")))
        assert np.allclose(p[:, 0], [T(s, 0)[0] for s in lay.s], atol=0.02)
        assert np.allclose(p[:, 1], y0 - lay.eta * sc, atol=0.02)
        assert np.abs(lay.eta).max() * sc == pytest.approx(46.0)
        if comp == "Vz":  # the jump at the section: two points at one x
            same = np.where(np.abs(np.diff(p[:, 0])) < 1e-6)[0]
            assert len(same) == 1
            jump = abs(p[same[0] + 1, 1] - p[same[0], 1]) / sc
            assert jump == pytest.approx(1.0, rel=1e-9)
        # the uniform load only over the loaded parts, P where the layout put it, its ordinate written
        bands = groups(root, "data-load", "band")
        assert [(float(b.get("data-s0")), float(b.get("data-s1"))) for b in bands] == \
            pytest.approx([tuple(sg) for sg in lay.segments], abs=1e-3)
        for b, (s0, s1) in zip(bands, lay.segments):
            assert float(b.get("data-x0")) == pytest.approx(T(s0, 0)[0], abs=0.02)
            assert float(b.get("data-x1")) == pytest.approx(T(s1, 0)[0], abs=0.02)
        (gp,) = groups(root, "data-load", "point")
        assert tip(gp)[0] == pytest.approx(T(lay.P_at, 0)[0], abs=0.02)
        (go,) = groups(root, "data-il-ordinate")
        assert float(go.get("data-il-ordinate")) == pytest.approx(lay.eta_P, rel=1e-4)
        # loaded areas darker
        loaded = [a for a in groups(root, "data-il-area") if a.get("data-loaded") == "1"]
        assert loaded and all(int(a.get("data-il-area")) == (1 if lay.sense == "max" else -1) for a in loaded)
        s = fig.svg()
        assert f"{abs(lay.value):.0f}" in s.replace("−", "-")  # the effect, written out
        assert_no_overlap(fig)


@pytest.mark.parametrize("width", [720, 380])
def test_influence_figure_vehicle(width):
    m, ns, bs, xs = girder()
    sv = Solver(m)
    il = InfluenceLines(sv, ns, chain_sections(bs))
    live = jtg.LiveLoad(L0=16.0, lanes=1, transverse=1.0, distribution=0.55, mu=0.3)
    for comp, k, which in (("My", 15, 0), ("Vz", 0, 1), ("My", 31, 1)):
        lay = live.vehicle_layouts(il, comp, k)[which]
        fig = influence_figure(il, lay, width=width, lang="en" if k == 0 else "zh")
        T = fig._transform()
        root = ET.fromstring(fig.svg())
        axles = groups(root, "data-load", "axle")
        assert [float(a.get("data-s")) for a in axles] == pytest.approx(list(lay.positions), abs=1e-3)
        for a, s_, on in zip(axles, lay.positions, lay.on_path()):
            assert bool(int(a.get("data-on"))) == bool(on)
            if on:
                assert tip(a)[0] == pytest.approx(T(s_, 0)[0], abs=0.02)
        (veh,) = groups(root, "data-vehicle")
        assert int(veh.get("data-heading")) == lay.heading
        ords = groups(root, "data-il-ordinate")
        assert [float(o.get("data-il-ordinate")) for o in ords] == \
            pytest.approx([v for v, o in zip(lay.ordinates, lay.on_path()) if o], rel=1e-3, abs=1e-6)
        assert_no_overlap(fig)


def test_sign_regions_close_at_zero():
    from femotion.frame.figures import _sign_regions
    x = np.array([0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    eta = np.array([0.0, 1.0, 0.5, 0.25, -0.5, -0.25, 0.0])  # a jump through zero at x = 3
    r = _sign_regions(x, eta)
    assert [g[0] for g in r] == [1, -1]
    (_, xa, ea), (_, xb, eb) = r
    assert xa[0] == 0.0 and xa[-1] == 3.0 and ea[0] == 0 and ea[-1] == 0
    assert xb[0] == 3.0 and xb[-1] == 5.0
    # areas by the shoelace formula equal the integrals of the line
    area = lambda xs, ys: abs(np.trapezoid(ys, xs))  # noqa: E731
    assert area(xa, ea) == pytest.approx(0.5 + 0.75 + 0.375)
    assert area(xb, eb) == pytest.approx(0.375 + 0.125)
