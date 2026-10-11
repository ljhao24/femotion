"""femotion.codes.jtg.charts: charts along a girder. The SVG is parsed and every drawn point mapped back to world
coordinates through the plot frame drawn in the SVG; the values are checked against the inputs and against the
code's resistances computed here directly (jtg.flexure, jtg.shear, jtg.shear_search), not through the chart module."""
import math
import re
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from femotion import viz
from femotion.codes import jtg
from femotion.codes.jtg import charts
from femotion.viz import wrap

NS = "{http://www.w3.org/2000/svg}"
SPAN = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
PIER = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(6, 28, 95)], top=[(14, 25, 60)])
ZONES = [(0.0, 11.2, SPAN), (11.2, 20.8, PIER), (20.8, 32.0, SPAN)]


def zone_section(u):
    return PIER if 11.2 < u < 20.8 else SPAN  # no data point sits on a zone boundary in these tests


def two_span():
    x = np.linspace(0, 32, 129)
    u = np.minimum(x, 32 - x)
    return x, 25 * u * (14 - u), 10 * u * (8 - u)  # Mmax 1225 at 7 m, −800 at the pier; Mmin −1280 at the pier


# ── reading the SVG ─────────────────────────────────────────────────────────
def root_of(fig):
    s = fig.svg()
    return ET.fromstring(s), s


def has(el, cls):
    return cls in (el.get("class") or "").split()


def frame(root):
    """page box of the plot (the axis rectangle)"""
    r = next(e for e in root.iter(f"{NS}rect") if has(e, "fm-axis"))
    x, y, w, h = (float(r.get(k)) for k in ("x", "y", "width", "height"))
    return x, y, x + w, y + h


def to_world(root, box):
    """page → world through the drawn plot frame and the world box shown"""
    px0, py0, px1, py1 = frame(root)
    x0, y0, x1, y1 = box
    return lambda u, v: (x0 + (u - px0) / (px1 - px0) * (x1 - x0), y1 - (v - py0) / (py1 - py0) * (y1 - y0))


def pts(el):
    return [tuple(map(float, p.split(","))) for p in el.get("points").split()]


def lines(root, cls, tag="polyline"):
    return [pts(e) for e in root.iter(f"{NS}{tag}") if has(e, cls)]


def tol(root, box):
    """world tolerance of 0.006 px (coordinates are written to 0.01 px)"""
    px0, py0, px1, py1 = frame(root)
    return 0.006 * (box[2] - box[0]) / (px1 - px0), 0.006 * (box[3] - box[1]) / (py1 - py0)


def world_lines(fig, cls):
    root, _ = root_of(fig)
    w = to_world(root, fig._box)
    return [np.array([w(*p) for p in ln]) for ln in lines(root, cls)], tol(root, fig._box)


def texts(root):
    return ["".join(e.itertext()) for e in root.iter(f"{NS}text")]


# ── independent facts ───────────────────────────────────────────────────────
def test_moment_envelope_is_drawn_on_the_tension_side_with_gamma0():
    x, Mmax, Mmin = two_span()
    fig = charts.moment_chart(x, Mmax, Mmin, gamma0=1.1, resistance=ZONES, supports=[0, 16, 32])
    (top,), (tx, ty) = world_lines(fig, "fm-ch-md-max")
    (bot,), _ = world_lines(fig, "fm-ch-md-min")
    assert np.allclose(top[:, 0], x, atol=tx) and np.allclose(top[:, 1], -1.1 * Mmax, atol=ty)
    assert np.allclose(bot[:, 1], -1.1 * Mmin, atol=ty)
    # sagging below the zero axis on the page, hogging above it
    root, _ = root_of(fig)
    (px0, py0, px1, py1) = frame(root)
    zero = [ln for ln in root.iter(f"{NS}line") if has(ln, "fm-axis")][0]
    y_axis = float(zero.get("y1"))
    k = int(np.argmax(Mmax))
    page = lines(root, "fm-ch-md-max")[0]
    assert page[k][1] > y_axis + 50 and lines(root, "fm-ch-md-min")[0][64][1] < y_axis - 50
    # the tick labels keep the moment's sign: positive ones below the axis
    ticks = [(float(e.get("y")), float("".join(e.itertext()).replace("−", "-"))) for e in root.iter(f"{NS}text")
             if has(e, "fm-tick") and e.get("text-anchor") == "end"]
    assert all((v > 0) == (yy > y_axis) for yy, v in ticks if v != 0)
    up = charts.moment_chart(x, Mmax, Mmin, resistance=ZONES, tension_side=False)
    (top2,), (_, ty2) = world_lines(up, "fm-ch-md-max")
    assert np.allclose(top2[:, 1], Mmax, atol=ty2)


def test_resisting_moment_steps_equal_flexure():
    x, Mmax, Mmin = two_span()
    fig = charts.moment_chart(x, Mmax, Mmin, resistance=ZONES)
    (pos,), (tx, ty) = world_lines(fig, "fm-ch-mu-pos")
    (neg,), _ = world_lines(fig, "fm-ch-mu-neg")
    for a, b, sec in ZONES:
        Mu_s, Mu_h = jtg.flexure(sec, 1.0).resistance, jtg.flexure(sec, -1.0).resistance
        mid = (a + b) / 2
        # drawn on the tension side: sagging resistance below (−Mu), hogging above (+Mu); the level over the zone
        level_s = [p[1] for p, q in zip(pos[:-1], pos[1:]) if abs(p[1] - q[1]) <= ty and p[0] <= mid <= q[0]]
        level_h = [p[1] for p, q in zip(neg[:-1], neg[1:]) if abs(p[1] - q[1]) <= ty and p[0] <= mid <= q[0]]
        assert level_s == pytest.approx([-Mu_s], abs=ty) and level_h == pytest.approx([Mu_h], abs=ty)
    # the steps change exactly at the zone boundaries
    xs = sorted({round(p[0], 3) for p in pos})
    assert xs == pytest.approx([0.0, 11.2, 20.8, 32.0], abs=1e-3)
    # the profile itself: the weaker side at a boundary, NaN outside
    res = charts.flexural_resistance(ZONES)
    s_, p_ = jtg.flexure(SPAN, 1.0).resistance, jtg.flexure(PIER, 1.0).resistance
    assert res.sagging(11.2)[0] == pytest.approx(min(s_, p_)) and math.isnan(res.sagging(40.0)[0])
    assert res.hogging(16.0)[0] == pytest.approx(-jtg.flexure(PIER, -1.0).resistance)


@pytest.mark.parametrize("gamma0", [1.0, 1.1])
def test_governing_section_is_the_largest_ratio(gamma0):
    x, Mmax, Mmin = two_span()
    fig = charts.moment_chart(x, Mmax, Mmin, gamma0=gamma0, resistance=ZONES)
    # independent: γ0 Md / Mu at every point, sagging and hogging, the section of the zone
    r = []
    for u, mx, mn in zip(x, Mmax, Mmin):
        s = zone_section(u)
        r.append(max(gamma0 * mx / jtg.flexure(s, 1.0).resistance if mx > 0 else 0.0,
                     gamma0 * -mn / jtg.flexure(s, -1.0).resistance if mn < 0 else 0.0))
    r = np.array(r)
    k = int(np.argmax(r))
    c = fig.critical
    assert c["ratio"] == pytest.approx(r[k], rel=1e-12) and c["x"] == pytest.approx(x[k])
    assert np.allclose(fig.utilization, r, rtol=1e-12)
    assert np.allclose(charts.flexure_utilization(x, Mmax, Mmin, ZONES, gamma0), r, rtol=1e-12)
    root, svg = root_of(fig)
    assert f"{r[k]:.3f}" in svg and f"{r[k]:.2f}" in texts(root)
    # the marker: a dot on γ0Md and an open circle on Mu at that section (tension side)
    w = to_world(root, fig._box)
    tx, ty = tol(root, fig._box)
    d = next(e for e in root.iter(f"{NS}circle") if has(e, "fm-ch-crit-d"))
    o = next(e for e in root.iter(f"{NS}circle") if has(e, "fm-ch-crit-r"))
    sag = Mmax[k] * gamma0 / jtg.flexure(zone_section(x[k]), 1.0).resistance >= r[k] - 1e-12
    Md = gamma0 * (Mmax[k] if sag else Mmin[k])
    Mu = jtg.flexure(zone_section(x[k]), 1.0 if sag else -1.0).resistance * (1 if sag else -1)
    assert w(float(d.get("cx")), float(d.get("cy"))) == pytest.approx((x[k], -Md), abs=max(tx, ty))
    assert w(float(o.get("cx")), float(o.get("cy"))) == pytest.approx((x[k], -Mu), abs=max(tx, ty))
    # least margin instead
    m = charts.moment_chart(x, Mmax, Mmin, gamma0=gamma0, resistance=ZONES, critical="margin").critical
    marg = [min(jtg.flexure(zone_section(u), 1.0).resistance - gamma0 * mx if mx > 0 else np.inf,
                jtg.flexure(zone_section(u), -1.0).resistance + gamma0 * mn if mn < 0 else np.inf)
            for u, mx, mn in zip(x, Mmax, Mmin)]
    assert m["margin"] == pytest.approx(min(marg)) and m["x"] == pytest.approx(x[int(np.argmin(marg))])


def test_peak_shaving_drawn_both_ways():
    x, Mmax, Mmin = two_span()
    R, a = 900.0, 2.0
    Mr = jtg.peak_shaving(x, Mmin, 16.0, R, a)
    fig = charts.moment_chart(x, Mmax, Mmin, resistance=ZONES, shaved=Mr, supports=[0, 16, 32])
    (mn,), (_, ty) = world_lines(fig, "fm-ch-md-min")
    assert np.allclose(mn[:, 1], -Mr, atol=ty)
    orig, _ = world_lines(fig, "fm-ch-md-orig")  # the original only near the support, where it differs
    xo = np.concatenate([o[:, 0] for o in orig])
    assert xo.min() >= 16 - a / 2 - 0.26 and xo.max() <= 16 + a / 2 + 0.26
    for o in orig:
        assert np.allclose(o[:, 1], -np.interp(o[:, 0], x, Mmin), atol=ty)
    k = int(np.argmin(np.abs(x - 16)))
    assert fig.utilization[k] == pytest.approx(-Mr[k] / jtg.flexure(PIER, -1.0).resistance)


def simple_beam(L=12.0, q=120.0, h=1100):
    x = np.linspace(0, L, 97)
    V = q * (L / 2 - x)
    Vmax, Vmin = V + 40 * (1 - x / L), V - 40 * x / L
    M = q * x * (L - x) / 2
    return x, Vmax, Vmin, M


def test_inclined_sections_on_the_elevation():
    x, Vmax, Vmin, M = simple_beam()
    st = [(0.0, 2.0, (2, 12, 100)), (2.0, 10.0, (2, 10, 200)), (10.0, 12.0, (2, 12, 100))]
    sup = [(0.0, "end"), (12.0, "end")]
    Vd = np.maximum(np.abs(Vmax), np.abs(Vmin))
    res = jtg.shear_search(x, Vd, M, SPAN, st, supports=sup, gamma0=1.1)
    el = charts.beam_elevation(0.0, 12.0, SPAN, supports=sup, search=res, stirrups=st)
    root, _ = root_of(el)
    beam = next(e for e in root.iter(f"{NS}polygon") if has(e, "fm-ch-beam"))
    bp = np.array(pts(beam))
    (px0, py0), (px1, py1) = bp.min(axis=0), bp.max(axis=0)
    h = SPAN.h / 1e3
    w = lambda u, v: ((u - px0) / (px1 - px0) * 12.0, -(v - py0) / (py1 - py0) * h)  # noqa: E731
    drawn = np.array([np.r_[w(*p[0]), w(*p[1])] for p in lines(root, "fm-ch-incl")])
    want = np.array([[s.x_t, -h, s.x_c, 0.0] for s in res.sections if s is not res.governing])
    assert len(drawn) == len(want) and len(want) > 10  # in the order of the search
    # tension end on the soffit (sagging), compression end on the top; 0.01 px of the page is ~2e-4 m
    assert np.allclose(drawn, want, atol=5e-4)
    (g,) = lines(root, "fm-ch-incl-gov")
    gv = res.governing
    assert np.r_[w(*g[0]), w(*g[1])] == pytest.approx([gv.x_t, -h, gv.x_c, 0.0], abs=2e-3)
    # the stirrup zones under the girder
    t = texts(root)
    assert "Φ12@100 双肢" in t and "Φ10@200 双肢" in t
    # the shear chart: γ0Vd, the governing marker and the 5.2.11 limit at the end supports
    fv = charts.shear_chart(x, Vmax, Vmin, gamma0=1.1, search=res, supports=[0, 12],
                            resistance=charts.shear_resistance(SPAN, st, supports=sup, x=x, Md=M))
    (vm,), (tx, ty) = world_lines(fv, "fm-ch-vd-max")
    assert np.allclose(vm[:, 1], 1.1 * Vmax, atol=ty)
    rootv, svg = root_of(fv)
    wv = to_world(rootv, fv._box)
    d = next(e for e in rootv.iter(f"{NS}circle") if has(e, "fm-ch-crit-d"))
    sgn = 1.0 if abs(np.interp(gv.x_c, x, Vmax)) >= abs(np.interp(gv.x_c, x, Vmin)) else -1.0
    assert wv(float(d.get("cx")), float(d.get("cy"))) == pytest.approx((gv.x_c, sgn * 1.1 * gv.Vd), abs=max(tx, ty))
    assert f"{gv.ratio:.3f}" in svg
    lims, _ = world_lines(fv, "fm-ch-vlim")
    assert len(lims) == len(res.limits) == 2
    for c, ln in zip(res.limits, sorted(lims, key=lambda a: a[0, 0])):
        u = c.details["x"]
        sg = 1.0 if abs(np.interp(u, x, Vmax)) >= abs(np.interp(u, x, Vmin)) else -1.0
        assert ln[0, 0] == pytest.approx(u, abs=tx) and np.allclose(ln[:, 1], sg * c.resistance, atol=ty)
        assert c.resistance == pytest.approx(0.51e-3 * math.sqrt(40) * c.details["b"] * c.details["h0"])


def test_inclined_sections_over_an_interior_support_start_on_top():
    """hogging: the crack opens on the top face (x_t) and the compression end is on the soffit (x_c)"""
    x, Mmax, Mmin = two_span()
    Md = np.where(np.abs(Mmin) > np.abs(Mmax), Mmin, Mmax)
    Vd = 300 + 25 * np.abs(16 - np.minimum(x, 32 - x) - 8)
    sup = [(0.0, "end"), (16.0, "interior", 0.4), (32.0, "end")]
    res = jtg.shear_search(x, Vd, Md, ZONES, (2, 10, 100), supports=sup)
    el = charts.beam_elevation(0.0, 32.0, ZONES, supports=sup, search=res, grid=False)
    root, _ = root_of(el)
    bp = np.array(pts(next(e for e in root.iter(f"{NS}polygon") if has(e, "fm-ch-beam"))))
    (px0, py0), (px1, py1) = bp.min(axis=0), bp.max(axis=0)
    h = SPAN.h / 1e3
    drawn = [((p[0][0] - px0) / (px1 - px0) * 32, -(p[0][1] - py0) / (py1 - py0) * h,
              (p[1][0] - px0) / (px1 - px0) * 32, -(p[1][1] - py0) / (py1 - py0) * h)
             for p in lines(root, "fm-ch-incl") + lines(root, "fm-ch-incl-gov")]
    hog = [s for s in res.sections if s.source != "grid" and s.Md < 0]
    assert len(hog) == 2  # the two diaphragm edges
    for s in hog:
        assert any(np.allclose(d, [s.x_t, 0.0, s.x_c, -h], atol=1e-3) for d in drawn)
    assert len(drawn) == sum(s.source != "grid" for s in res.sections) + (res.governing.source == "grid")


def test_shear_resistance_steps_equal_shear():
    """Vu of the stirrup zones: jtg.shear of the zone, α1 0.9 in the half spans next to the interior support, the
    top as tension face where the moment is hogging"""
    x, Mmax, Mmin = two_span()
    Md = np.where(np.abs(Mmin) > np.abs(Mmax), Mmin, Mmax)
    st = [(0.0, 3.0, (2, 12, 100)), (3.0, 11.2, (2, 10, 200)), (11.2, 20.8, (4, 10, 100)), (20.8, 29.0, (2, 10, 200)),
          (29.0, 32.0, (2, 12, 100))]
    prof = charts.shear_resistance(ZONES, st, supports=[0, 16, 32], x=x, Md=Md)
    zero = [x[k] + (x[k + 1] - x[k]) * Md[k] / (Md[k] - Md[k + 1]) for k in range(len(x) - 1)
            if (Md[k] < 0) != (Md[k + 1] < 0)]
    cuts = sorted({0.0, 32.0, 3.0, 11.2, 20.8, 29.0, 8.0, 24.0, *zero})
    for a, b in zip(cuts[:-1], cuts[1:]):
        u = (a + b) / 2
        stir = next(s for z0, z1, s in st if z0 <= u <= z1)
        a1 = 0.9 if 8.0 < u < 24.0 else 1.0
        hog = np.interp(u, x, Md) < 0
        want = jtg.shear(zone_section(u), 0.0, stir, alpha1=a1, hogging=hog).resistance
        assert prof(u)[0] == pytest.approx(want, rel=1e-12)
    fig = charts.shear_chart(x, 600 - 40 * x, -40 * x, resistance=prof)
    (pos,), (_, ty) = world_lines(fig, "fm-ch-vu-pos")
    (neg,), _ = world_lines(fig, "fm-ch-vu-neg")
    assert np.allclose(pos[:, 1], -neg[:, 1], atol=2 * ty)
    for a, b, v in prof.rows:  # each step at its height over its zone
        level = [p[1] for p, q in zip(pos[:-1], pos[1:]) if abs(p[1] - q[1]) <= ty and p[0] <= (a + b) / 2 <= q[0]]
        assert level == pytest.approx([v], abs=2 * ty)


def test_utilization_beyond_the_limit_in_red():
    x = np.linspace(0, 20, 81)
    r1 = 0.6 + 0.6 * np.sin(np.pi * x / 20)  # 1.2 at midspan, beyond 1 between the crossings
    r2 = np.where(x < 10, 0.4, np.nan)  # a series checked on half the girder
    fig = charts.utilization_chart([("A", x, r1), ("B", x, r2)])
    (a,), (tx, ty) = world_lines(fig, "fm-ch-u0")
    assert np.allclose(a[:, 1], r1, atol=ty)
    (b,), _ = world_lines(fig, "fm-ch-u1")
    assert len(b) == int((x < 10).sum())
    (over,), _ = world_lines(fig, "fm-ch-over0")
    cross = 20 / np.pi * np.arcsin(0.4 / 0.6)
    assert over[0] == pytest.approx((cross, 1.0), abs=0.05) and over[-1] == pytest.approx((20 - cross, 1.0), abs=0.05)
    assert np.all(over[1:-1, 1] > 1.0)
    assert fig.maxima[0] == {"label": "A", "x": pytest.approx(10.0), "ratio": pytest.approx(1.2)}
    root, svg = root_of(fig)
    assert "A（最大 1.20）" in texts(root) and "fm-notewarn" in svg


def test_stresses_deflection_crack_width_against_limits():
    x = np.linspace(0, 30, 61)
    top = (6 + 3 * np.sin(np.pi * x / 30), 2 + np.sin(np.pi * x / 30))
    bot = (8 - 9 * np.sin(np.pi * x / 30), 7 - 9 * np.sin(np.pi * x / 30))  # −2 at midspan: beyond −1.5
    fig = charts.stress_chart(x, top=top, bottom=bot, limits=[(16.2, "0.5fck"), (-1.5, "−0.7ftk")])
    (b0,), (_, ty) = world_lines(fig, "fm-ch-edge1-1")
    assert np.allclose(b0[:, 1], bot[1], atol=ty)
    (lim_c, lim_t), _ = world_lines(fig, "fm-ch-limit")
    assert np.allclose(lim_c[:, 1], 16.2, atol=ty) and np.allclose(lim_t[:, 1], -1.5, atol=ty)
    # only the lower bottom-edge curve goes below −1.5, most at midspan
    assert [e[:3] for e in fig.exceeded] == [("−0.7ftk", pytest.approx(15.0), pytest.approx(-2.0))]
    over, _ = world_lines(fig, "fm-ch-over")
    assert over and all(np.all(o[:, 1] <= -1.5 + ty) for o in over)
    assert fig.extremes["下缘"][0] == pytest.approx(-2.0)

    f = 12 * np.sin(np.pi * x / 30)
    fd = charts.deflection_chart(x, f, limit=[(0, 30, 30 * 1e3 / 600, "L/600")])
    (c,), (_, ty) = world_lines(fd, "fm-ch-q0-0")
    assert np.allclose(c[:, 1], -f, atol=ty)  # downward drawn down
    (lim,), _ = world_lines(fd, "fm-ch-limit")
    assert np.allclose(lim[:, 1], -50.0, atol=ty) and fd._box[3] == 0.0 and fd.exceeded == []
    w = np.where(x > 5, 0.25 * np.sin(np.pi * x / 30), np.nan)
    fw = charts.crack_chart(x, w, limit=0.2)
    assert fw._box[1] == 0.0 and fw.extremes["裂缝宽度"] == (pytest.approx(0.25), pytest.approx(15.0))
    assert fw.exceeded and fw.exceeded[0][2] == pytest.approx(0.25)


def test_stack_aligns_the_plot_boxes():
    x, Mmax, Mmin = two_span()
    st = [(0.0, 32.0, (2, 10, 150))]
    figs = [charts.beam_elevation(0, 32, SPAN, supports=[0, 16, 32], stirrups=st),
            charts.moment_chart(x, Mmax, Mmin, resistance=ZONES, xlabel=False),
            charts.utilization_chart([("一个很长的标签名称", x, np.full_like(x, 0.5))], width=500)]
    s = viz.stack(figs)
    root = ET.fromstring(s.svg())
    subs = [e for e in root if e.tag == f"{NS}svg"]
    assert len(subs) == 3
    beam = next(e for e in subs[0].iter(f"{NS}polygon") if has(e, "fm-ch-beam"))
    edges = [(min(p[0] for p in pts(beam)), max(p[0] for p in pts(beam)))] + [frame(e)[::2] for e in subs[1:]]
    assert all(e == pytest.approx(edges[0], abs=0.02) for e in edges)
    ys = [float(e.get("y")) for e in subs]
    hs = [float(e.get("height")) for e in subs]
    assert ys[0] == 0 and ys[1] == pytest.approx(hs[0] + 4) and float(root.get("height")) == pytest.approx(sum(hs) + 8)
    assert all(float(e.get("width")) == float(root.get("width")) for e in subs)
    assert figs[2].width == 500  # restored
    with pytest.raises(ValueError):
        viz.stack([figs[1], charts.utilization_chart([("a", x[:40], np.ones(40))])]).svg()


# ── every chart: valid SVG, finite numbers, text on the page ───────────────
def _em(ch):
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.58


SIZES = {"fm-title": 15.0, "fm-label": 12.0}


def text_boxes(root):
    for e in root.iter(f"{NS}text"):
        s = "".join(e.itertext())
        cls = (e.get("class") or "").split()
        size = next((SIZES[c] for c in cls if c in SIZES), 10.5)
        w, h = sum(_em(c) for c in s) * size, 1.2 * size
        x, y = float(e.get("x")), float(e.get("y"))
        a = e.get("text-anchor") or "start"
        x0 = x - (w if a == "end" else w / 2 if a == "middle" else 0)
        y0 = y - h / 2
        if "rotate(-90" in (e.get("transform") or ""):
            yield s, (x - h / 2, y - w / 2, x + h / 2, y + w / 2)
        else:
            yield s, (x0, y0, x0 + w, y0 + h)


NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def all_charts(lang, width):
    x, Mmax, Mmin = two_span()
    xb, Vmax, Vmin, M = simple_beam()
    st = [(0.0, 2.0, (2, 12, 100)), (2.0, 10.0, (2, 10, 200)), (10.0, 12.0, (2, 12, 100))]
    sup = [(0.0, "end"), (12.0, "end")]
    res = jtg.shear_search(xb, np.maximum(np.abs(Vmax), np.abs(Vmin)), M, SPAN, st, supports=sup)
    kw = dict(lang=lang, width=width)
    Mr = jtg.peak_shaving(x, Mmin, 16.0, 900.0, 2.0)
    return [
        charts.moment_chart(x, Mmax, Mmin, gamma0=1.1, resistance=ZONES, shaved=Mr, supports=[0, 16, 32],
                            others=[("std", 0.7 * Mmax, 0.7 * Mmin)], **kw),
        charts.shear_chart(xb, Vmax, Vmin, search=res, resistance=charts.shear_resistance(SPAN, st, supports=sup),
                           supports=[0, 12], **kw),
        charts.beam_elevation(0, 12, SPAN, supports=sup, search=res, stirrups=st, **kw),
        charts.utilization_chart([("A", x, 0.9 + 0.3 * np.sin(x)), ("B", *charts.shear_utilization(res))], **kw),
        charts.stress_chart(x, top=(Mmax / 200, Mmin / 300), bottom=Mmax / 150, limits=[(6.0, "a"), (-1.0, "b")],
                            **kw),
        charts.deflection_chart(x, Mmax / 100, limit=8.0, supports=[0, 16, 32], **kw),
        charts.crack_chart(x, {"s": np.where(Mmax > 0, Mmax / 6000, np.nan), "h": np.where(Mmin < 0, -Mmin / 6000,
                                                                                         np.nan)}, limit=0.2, **kw),
    ]


@pytest.mark.parametrize("lang,width", [("zh", 720), ("zh", 380), ("en", 640)])
def test_every_chart_is_valid_and_its_text_on_the_page(lang, width):
    for fig in all_charts(lang, width):
        root, s = root_of(fig)
        W, H = float(root.get("width")), float(root.get("height"))
        assert W == width and math.isfinite(H) and H > 0
        assert not re.search(r"\b(nan|inf)\b", s, re.I)
        for e in root.iter():
            for k in ("x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "width", "height"):
                if e.get(k) is not None and e.tag != f"{NS}svg" and not e.get(k).endswith("%"):
                    assert math.isfinite(float(e.get(k)))
            if e.get("points"):
                assert all(math.isfinite(float(v)) for v in NUM.findall(e.get("points")))
        for txt, (x0, y0, x1, y1) in text_boxes(root):
            assert x0 >= -0.5 and y0 >= -0.5 and x1 <= W + 0.5 and y1 <= H + 0.5, (txt, (x0, y0, x1, y1), W, H)


def test_wrap_and_profiles():
    note = "最不利斜截面 x_c = 9.418 m，x_t = 11.200 m，C = 1.782 m：γ0Vd = 353.4 kN，Vu = 472.4 kN，比值 0.748"
    for wdt in (160, 240, 330):
        out = wrap(note, wdt)
        assert "".join(out).replace(" ", " ").replace(" ", "") == note.replace(" ", "")
        for ln in out:
            assert sum(_em(c) for c in ln) * 10.5 <= wdt + 1e-9
        assert not any(ln.startswith(("m", "kN")) for ln in out)  # a number keeps its unit
    p = charts.Profile.steps([(0, 2, 5.0), (2, 4, -3.0), (6, 8, 1.0)])
    assert p(2.0)[0] == -3.0 and math.isnan(p(5.0)[0]) and len(p.lines()) == 2
    c = charts.Profile.curve([0, 1, 2], [0, 10, 0])
    assert c(0.5)[0] == 5.0 and math.isnan(c(3.0)[0])
    with pytest.raises(ValueError):
        charts.Profile.steps([(2, 1, 0.0)])
