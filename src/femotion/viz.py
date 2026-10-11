"""Engineering drawings as plain SVG, without plotting libraries (nothing beyond the standard library and NumPy).

A Figure collects shapes in world coordinates (any unit, y up) and maps them to the page when it is written: with one
scale for both axes (sections, elevations: equal=True) or with separate x and y scales (charts: equal=False, see axes()).
Sizes that belong to the page rather than to the structure — text, arrow heads, dots, dimension offsets — are in
pixels, and the margins grow by themselves until every label, dimension and arrow fits on the page. A figure shows
itself in Jupyter (_repr_svg_), saves as .svg or .html, and goes into an HTML page with html().

    from femotion.viz import Figure
    fig = Figure(width=480, title="T girder")
    fig.polygon([(0, 0), (1600, 0), (1600, -150), ...], cls="concrete")
    fig.circle((300, -1210), 16, cls="bar")
    fig.dim((0, 0), (1600, 0), offset=18, label="1600")
    fig.save("girder.svg")

    chart = Figure(width=560, height=260, equal=False).set_box(0, -3500, 40, 3500).axes(xlabel="x (m)")
    chart.polyline(zip(x, M), "demand")          # clipped to the box

Styles are CSS classes (STYLE below); every class name is prefixed with "fm-" in the output so that figures can sit in
any page. Text uses a font list that covers Chinese.
"""
from __future__ import annotations

import html as _html
import itertools
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

FONT = "'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', 'Source Han Sans SC', 'Helvetica Neue', Arial, sans-serif"

# one palette for every drawing: print-friendly, distinguishable in grey scale and for the common colour blindness
COLORS = {
    "ink": "#262626", "muted": "#6b6b6b", "faint": "#b8b8b8", "paper": "#ffffff",
    "concrete": "#e7e4dc", "concrete_edge": "#4a4a4a", "topping": "#d9e4ea",
    "steel": "#1f1f1f", "tendon": "#1d5fa6", "duct": "#1d5fa6", "external": "#7b3fa8",
    "compression": "#e08a2c", "tension": "#c0392b", "neutral": "#e08a2c",
    "demand": "#c0392b", "resistance": "#1d5fa6", "load": "#b4541b", "load_fill": "#f6dcc5",
    "warn": "#c0392b", "ok": "#2e7d32", "grid": "#ececec",
}
C = COLORS
STYLE = f"""
.fm-root text {{ font-family: {FONT}; fill: {C['ink']}; }}
.fm-title {{ font-size: 15px; font-weight: 600; }}
.fm-label {{ font-size: 12px; }}
.fm-small {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-concrete {{ fill: {C['concrete']}; stroke: {C['concrete_edge']}; stroke-width: 1.2; stroke-linejoin: round; }}
.fm-topping {{ fill: {C['topping']}; stroke: {C['concrete_edge']}; stroke-width: 1.2; stroke-linejoin: round; }}
.fm-void {{ fill: {C['paper']}; stroke: {C['concrete_edge']}; stroke-width: 1.2; stroke-linejoin: round; }}
.fm-outline {{ fill: none; stroke: {C['concrete_edge']}; stroke-width: 1.2; }}
.fm-bar {{ fill: {C['steel']}; stroke: none; }}
.fm-stirrup {{ fill: none; stroke: {C['steel']}; stroke-width: 1.1; }}
.fm-duct {{ fill: {C['paper']}; stroke: {C['duct']}; stroke-width: 1.1; }}
.fm-tendon {{ fill: {C['tendon']}; stroke: none; }}
.fm-external {{ fill: {C['external']}; stroke: none; }}
.fm-comp {{ fill: {C['compression']}; fill-opacity: 0.28; stroke: {C['compression']}; stroke-width: 1; }}
.fm-tens {{ fill: {C['tension']}; fill-opacity: 0.22; stroke: {C['tension']}; stroke-width: 1; }}
.fm-na {{ fill: none; stroke: {C['neutral']}; stroke-width: 1.2; stroke-dasharray: 6 3; }}
.fm-dim {{ fill: none; stroke: {C['muted']}; stroke-width: 0.8; }}
.fm-dimtext {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-leader {{ fill: none; stroke: {C['muted']}; stroke-width: 0.7; }}
.fm-warn {{ fill: none; stroke: {C['warn']}; stroke-width: 1.6; }}
.fm-warntext {{ font-size: 11px; fill: {C['warn']}; }}
.fm-axis {{ fill: none; stroke: {C['ink']}; stroke-width: 0.9; }}
.fm-grid {{ fill: none; stroke: {C['grid']}; stroke-width: 0.8; }}
.fm-tick {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-demand {{ fill: none; stroke: {C['demand']}; stroke-width: 1.8; stroke-linejoin: round; }}
.fm-demand2 {{ fill: none; stroke: {C['demand']}; stroke-width: 1.4; stroke-dasharray: 5 3; stroke-linejoin: round; }}
.fm-resist {{ fill: none; stroke: {C['resistance']}; stroke-width: 1.8; stroke-linejoin: round; }}
.fm-resist2 {{ fill: none; stroke: {C['resistance']}; stroke-width: 1.4; stroke-dasharray: 5 3; stroke-linejoin: round; }}
.fm-area-demand {{ fill: {C['demand']}; fill-opacity: 0.12; stroke: none; }}
.fm-area-resist {{ fill: {C['resistance']}; fill-opacity: 0.10; stroke: none; }}
.fm-member {{ fill: none; stroke: {C['ink']}; stroke-width: 2.2; stroke-linecap: round; }}
.fm-support {{ fill: {C['paper']}; stroke: {C['ink']}; stroke-width: 1.1; }}
.fm-load {{ fill: none; stroke: {C['load']}; stroke-width: 1.3; }}
.fm-loadhead {{ fill: {C['load']}; stroke: none; }}
.fm-loadfill {{ fill: {C['load_fill']}; fill-opacity: 0.8; stroke: {C['load']}; stroke-width: 1; }}
.fm-loadtext {{ font-size: 11px; fill: {C['load']}; }}
.fm-curve {{ fill: none; stroke: {C['ink']}; stroke-width: 1.4; stroke-linejoin: round; }}
.fm-pos {{ fill: {C['demand']}; fill-opacity: 0.16; stroke: none; }}
.fm-neg {{ fill: {C['resistance']}; fill-opacity: 0.16; stroke: none; }}
"""
# font size (px) of the text classes, for the space a label takes
TEXT_SIZE = {"title": 15.0, "label": 12.0, "small": 10.5, "dimtext": 10.5, "tick": 10.5, "loadtext": 11.0,
             "warntext": 11.0}
AREA_CLASSES = ("pos", "neg", "comp", "tens", "loadfill", "concrete", "topping", "void", "area-demand", "area-resist")

_ids = itertools.count(1)


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1e7 else f"{v:.0f}"


def _cls(c: str | None) -> str:
    if not c:
        return ""
    return ' class="' + " ".join(k if k.startswith("fm-") else "fm-" + k for k in c.split()) + '"'


def text_width(s: str, size: float) -> float:
    """approximate width (px) of a label: CJK and full-width characters one em, others about 0.58 em"""
    return sum(size if unicodedata.east_asian_width(ch) in ("W", "F") else 0.58 * size for ch in str(s))


def _text_box(x, y, s, size, anchor, baseline, rotate=0.0):
    w, h = text_width(s, size), size * 1.2
    x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0.0)
    y0 = y - (h / 2 if baseline in ("middle", "central") else h * 0.85 if baseline in ("auto", "alphabetic") else 0.0)
    if not rotate:
        return (x0, y0, x0 + w, y0 + h)
    a = math.radians(rotate)
    cs = [(cx - x, cy - y) for cx, cy in ((x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h))]
    rx = [x + u * math.cos(a) - v * math.sin(a) for u, v in cs]
    ry = [y + u * math.sin(a) + v * math.cos(a) for u, v in cs]
    return (min(rx), min(ry), max(rx), max(ry))


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """round tick values covering [lo, hi], about n of them (1, 2, 2.5, 5 × 10^k steps)"""
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-12 * max(abs(lo), abs(hi), 1.0):
        return [lo]
    e = math.floor(math.log10((hi - lo) / max(n, 1)))
    steps = [s * 10.0 ** j for j in (e - 1, e, e + 1, e + 2) for s in (1, 2, 2.5, 5)]
    count = lambda st: math.floor(hi / st + 1e-9) - math.ceil(lo / st - 1e-9) + 1  # noqa: E731
    step = next(st for st in steps if count(st) <= n + 2)  # the finest round step with at most n + 2 ticks
    out, t = [], math.ceil(lo / step - 1e-9) * step
    while t <= hi + 1e-9 * step:
        out.append(0.0 if abs(t) < 1e-9 * step else round(t, 12))
        t += step
    return out


@dataclass
class _Item:
    svg: Callable  # T -> str
    pts: list | None = None  # world points that set the content box
    extent: Callable | None = None  # T -> [page boxes (x0, y0, x1, y1)] the item needs on the page
    clip: bool = False  # clipped to the plot box (charts)
    under: bool = False  # drawn before everything else (axes, grid)


class Figure:
    """a drawing in world coordinates (y up). width (px); height (px) is derived from the drawing when equal=True,
    else required. margin (px): the least space kept around the drawing (top, right, bottom, left); it grows to fit
    labels, dimensions and arrows. pad: extra world-unit space around the content."""

    def __init__(self, width: float = 560, height: float | None = None, *, equal: bool = True,
                 margin=(12, 12, 12, 12), title: str | None = None, pad: float = 0.0):
        if not equal and height is None:
            raise ValueError("charts (equal=False) need a height")
        self.width, self.height, self.equal = float(width), (None if height is None else float(height)), equal
        self.margin = tuple(float(m) for m in margin)
        self.title = title
        self.pad = pad
        self._items: list[_Item] = []
        self._box: list | None = None
        self.id = f"fm{next(_ids)}"

    # ── page geometry ──────────────────────────────────────────────────────
    def set_box(self, x0: float, y0: float, x1: float, y1: float) -> "Figure":
        """fix the world box shown (charts: the axis ranges) instead of fitting the content"""
        self._box = [float(x0), float(y0), float(x1), float(y1)]
        return self

    def content_box(self):
        pts = [p for it in self._items if it.pts for p in it.pts]
        if not pts:
            return [0.0, 0.0, 1.0, 1.0]
        a = np.asarray(pts, float)
        (x0, y0), (x1, y1) = a.min(axis=0), a.max(axis=0)
        return [x0 - self.pad, y0 - self.pad, x1 + self.pad, y1 + self.pad]

    def _make(self, m):
        x0, y0, x1, y1 = self._box or self.content_box()
        t, r, b, l = m
        w = max(self.width - l - r, 1.0)
        dx, dy = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
        if self.equal:
            s = w / dx
            if self.height is not None:
                s = min(s, max(self.height - t - b, 1.0) / dy)
            sx = sy = s
            H = self.height if self.height is not None else t + dy * s + b
            ox, oy = l + (w - dx * s) / 2, t
        else:
            H = self.height
            sx, sy = w / dx, max(H - t - b, 1.0) / dy
            ox, oy = l, t

        def T(x, y):
            return ox + (x - x0) * sx, oy + (y1 - y) * sy
        T.sx, T.sy, T.H, T.W, T.box = sx, sy, H, self.width, (x0, y0, x1, y1)
        return T

    def _transform(self):
        """the page map, with margins grown until every item's page extent fits (a few passes)"""
        t, r, b, l = self.margin
        t += 24 if self.title else 0
        m = [t, r, b, l]
        T = self._make(m)
        for _ in range(6):
            boxes = [bx for it in self._items if it.extent for bx in it.extent(T)]
            if not boxes:
                break
            a = np.asarray(boxes, float)
            gx0, gy0, gx1, gy1 = a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()
            top = self.margin[0] + 26 if self.title else 4  # the title's row is its own
            need = [max(0.0, top - gy0), max(0.0, gx1 - (T.W - 4)), max(0.0, gy1 - (T.H - 4)), max(0.0, 4 - gx0)]
            if max(need) < 0.5:
                break
            m = [mi + ni for mi, ni in zip(m, need)]
            T = self._make(m)
        return T

    def to_page(self, x: float, y: float):
        return self._transform()(x, y)

    # ── shapes in world coordinates ────────────────────────────────────────
    def _add(self, svg, pts=None, extent=None, *, clip=False, under=False):
        pts = None if pts is None else [tuple(map(float, p)) for p in pts]
        it = _Item(svg, pts, extent, clip and not self.equal and self._box is not None, under)
        self._items.append(it)
        return self

    def polyline(self, pts, cls="curve", *, fit: bool = True):
        pts = [tuple(p) for p in pts]
        return self._add(lambda T: f'<polyline points="{_pts(T, pts)}"{_cls(cls)}/>', pts if fit else None, clip=True)

    def line(self, p0, p1, cls="curve", *, fit: bool = True):
        return self.polyline([p0, p1], cls, fit=fit)

    def polygon(self, pts, cls="concrete", *, fit: bool = True):
        pts = [tuple(p) for p in pts]
        return self._add(lambda T: f'<polygon points="{_pts(T, pts)}"{_cls(cls)}/>', pts if fit else None, clip=True)

    def path(self, rings, cls="concrete", *, fit: bool = True):
        """a filled shape with holes: rings = [outer, hole, hole, …] (even-odd fill)"""
        rings = [[tuple(p) for p in r] for r in rings]

        def f(T):
            d = " ".join("M" + " L".join("{},{}".format(*map(_fmt, T(*p))) for p in r) + " Z" for r in rings)
            return f'<path d="{d}" fill-rule="evenodd"{_cls(cls)}/>'
        return self._add(f, [p for r in rings for p in r] if fit else None, clip=True)

    def rect(self, x0, y0, x1, y1, cls="concrete", *, fit: bool = True):
        return self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], cls, fit=fit)

    def circle(self, c, r: float, cls="bar", *, fit: bool = True, title: str | None = None):
        """a circle of world radius r (at least 0.6 px on the page); title: a hover tip"""
        c = tuple(c)

        def f(T):
            x, y = T(*c)
            tt = f"<title>{_html.escape(title)}</title>" if title else ""
            return f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(max(r * T.sx, 0.6))}"{_cls(cls)}>{tt}</circle>'
        return self._add(f, [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)] if fit else None)

    def dot(self, c, r_px: float = 3.0, cls="bar", *, fit: bool = True):
        """a dot of page radius r_px at a world point"""
        c = tuple(c)

        def f(T):
            x, y = T(*c)
            return f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(r_px)}"{_cls(cls)}/>'

        def e(T):
            x, y = T(*c)
            return [(x - r_px, y - r_px, x + r_px, y + r_px)]
        return self._add(f, [c] if fit else None, e)

    def text(self, p, s: str, cls="label", *, anchor: str = "middle", dx: float = 0.0, dy: float = 0.0,
             rotate: float = 0.0, baseline: str = "middle", fit: bool = False):
        """text at a world point, shifted by (dx, dy) px (dy down); anchor start / middle / end"""
        p = tuple(p)
        size = TEXT_SIZE.get((cls or "label").split()[0].removeprefix("fm-"), 12.0)

        def at(T):
            x, y = T(*p)
            return x + dx, y + dy

        def f(T):
            x, y = at(T)
            rot = f' transform="rotate({_fmt(rotate)} {_fmt(x)} {_fmt(y)})"' if rotate else ""
            return (f'<text x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}" dominant-baseline="{baseline}"{rot}'
                    f'{_cls(cls)}>{_html.escape(str(s))}</text>')
        return self._add(f, [p] if fit else None, lambda T: [_text_box(*at(T), s, size, anchor, baseline, rotate)])

    def arrow(self, p0, p1, cls="load", *, head: float = 7.0, head_cls: str = "loadhead", fit: bool = True):
        """a line from p0 to p1 (world) with a head of `head` px at p1"""
        p0, p1 = tuple(p0), tuple(p1)
        return self._add(lambda T: _arrow_svg(*T(*p0), *T(*p1), head, cls, head_cls), [p0, p1] if fit else None)

    def force(self, p, direction=(0.0, -1.0), length: float = 36.0, *, cls="load", label: str | None = None,
              head: float = 7.0, label_cls: str = "loadtext"):
        """a force arrow `length` px long whose head touches the world point p, pointing in `direction` (world)"""
        p = tuple(p)
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) or 1.0)
        size = TEXT_SIZE.get(label_cls, 11.0)

        def tail(T):
            x1, y1 = T(*p)
            return x1 - d[0] * length, y1 + d[1] * length

        def lab(T):
            x0, y0 = tail(T)
            return x0 - d[0] * 8, y0 + d[1] * 8  # beyond the tail

        def f(T):
            x1, y1 = T(*p)
            out = _arrow_svg(*tail(T), x1, y1, head, cls, "loadhead")
            if label:
                lx, ly = lab(T)
                out += (f'<text x="{_fmt(lx)}" y="{_fmt(ly)}" text-anchor="middle" dominant-baseline="middle"'
                        f'{_cls(label_cls)}>{_html.escape(label)}</text>')
            return out

        def e(T):
            x0, y0 = tail(T)
            x1, y1 = T(*p)
            out = [(min(x0, x1) - 4, min(y0, y1) - 4, max(x0, x1) + 4, max(y0, y1) + 4)]
            if label:
                out.append(_text_box(*lab(T), label, size, "middle", "middle"))
            return out
        return self._add(f, [p], e)

    def dim(self, p0, p1, *, offset: float = 18.0, label: str | None = None, cls="dim"):
        """a dimension line between two world points, drawn `offset` px to one side of p0→p1 on the page (the sign
        picks the side); the label sits outside the line and reads from below or from the right"""
        p0, p1 = tuple(p0), tuple(p1)

        def geo(T):
            (x0, y0), (x1, y1) = T(*p0), T(*p1)
            L = math.hypot(x1 - x0, y1 - y0) or 1.0
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            nx, ny = uy, -ux
            sg = 1.0 if offset >= 0 else -1.0
            a0 = (x0 + nx * offset, y0 + ny * offset)
            a1 = (x1 + nx * offset, y1 + ny * offset)
            mid = ((a0[0] + a1[0]) / 2 + nx * 8 * sg, (a0[1] + a1[1]) / 2 + ny * 8 * sg)
            ang = math.degrees(math.atan2(uy, ux))
            if ang > 89.999:
                ang -= 180
            elif ang <= -90.001:
                ang += 180
            return (x0, y0), (x1, y1), (ux, uy), (nx, ny), sg, a0, a1, mid, ang

        def f(T):
            q0, q1, (ux, uy), (nx, ny), sg, a0, a1, (mx, my), ang = geo(T)
            seg = lambda p, q: f'<line x1="{_fmt(p[0])}" y1="{_fmt(p[1])}" x2="{_fmt(q[0])}" y2="{_fmt(q[1])}"{_cls(cls)}/>'  # noqa: E731
            out = [seg(a0, a1)]
            for (x, y), a in ((q0, a0), (q1, a1)):
                out.append(seg((x + nx * 3 * sg, y + ny * 3 * sg), (a[0] + nx * 3 * sg, a[1] + ny * 3 * sg)))
                tx, ty = (ux + nx) * 3, (uy + ny) * 3
                out.append(seg((a[0] - tx, a[1] - ty), (a[0] + tx, a[1] + ty)))
            if label is not None:
                out.append(f'<text x="{_fmt(mx)}" y="{_fmt(my)}" text-anchor="middle" dominant-baseline="middle" '
                           f'transform="rotate({_fmt(ang)} {_fmt(mx)} {_fmt(my)})"{_cls("dimtext")}>'
                           f'{_html.escape(label)}</text>')
            return "".join(out)

        def e(T):
            _, _, _, _, _, a0, a1, (mx, my), ang = geo(T)
            out = [(min(a0[0], a1[0]) - 4, min(a0[1], a1[1]) - 4, max(a0[0], a1[0]) + 4, max(a0[1], a1[1]) + 4)]
            if label is not None:
                out.append(_text_box(mx, my, label, TEXT_SIZE["dimtext"], "middle", "middle", ang))
            return out
        return self._add(f, [p0, p1], e)

    def raw(self, svg, pts=None, extent=None, *, clip: bool = False):
        """any SVG: svg(T) -> str, where T maps world (x, y) to the page and has sx, sy, box; extent(T) -> page boxes"""
        return self._add(svg, pts, extent, clip=clip)

    # ── charts ─────────────────────────────────────────────────────────────
    def axes(self, *, xlabel: str = "", ylabel: str = "", xticks=None, yticks=None, grid: bool = True,
             zero: bool = True, xfmt=None, yfmt=None):
        """frame, ticks, grid and labels of a chart (equal=False), drawn under everything else; set_box first (or the
        content box is used)"""
        xf, yf = xfmt or _fmt, yfmt or _fmt

        def ticks(T):
            x0, y0, x1, y1 = T.box
            return (nice_ticks(x0, x1, 6) if xticks is None else list(xticks),
                    nice_ticks(y0, y1, 5) if yticks is None else list(yticks))

        def f(T):
            x0, y0, x1, y1 = T.box
            xt, yt = ticks(T)
            seg = lambda a, b, c: f'<line x1="{_fmt(a[0])}" y1="{_fmt(a[1])}" x2="{_fmt(b[0])}" y2="{_fmt(b[1])}"{_cls(c)}/>'  # noqa: E731
            out = []
            if grid:
                out += [seg(T(x, y0), T(x, y1), "grid") for x in xt]
                out += [seg(T(x0, y), T(x1, y), "grid") for y in yt]
            (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
            out.append(f'<rect x="{_fmt(bx0)}" y="{_fmt(by1)}" width="{_fmt(bx1 - bx0)}" height="{_fmt(by0 - by1)}"'
                       f'{_cls("axis")}/>')
            if zero and y0 < 0 < y1:
                out.append(seg(T(x0, 0.0), T(x1, 0.0), "axis"))
            out += [f'<text x="{_fmt(T(x, y0)[0])}" y="{_fmt(by0 + 13)}" text-anchor="middle" dominant-baseline="middle"'
                    f'{_cls("tick")}>{xf(x)}</text>' for x in xt]
            out += [f'<text x="{_fmt(bx0 - 6)}" y="{_fmt(T(x0, y)[1])}" text-anchor="end" dominant-baseline="middle"'
                    f'{_cls("tick")}>{yf(y)}</text>' for y in yt]
            if xlabel:
                out.append(f'<text x="{_fmt((bx0 + bx1) / 2)}" y="{_fmt(by0 + 31)}" text-anchor="middle" '
                           f'dominant-baseline="middle"{_cls("label")}>{_html.escape(xlabel)}</text>')
            if ylabel:
                yy, xx = (by0 + by1) / 2, bx0 - 12 - max((text_width(yf(y), 10.5) for y in yt), default=0) - 8
                out.append(f'<text x="{_fmt(xx)}" y="{_fmt(yy)}" text-anchor="middle" dominant-baseline="middle" '
                           f'transform="rotate(-90 {_fmt(xx)} {_fmt(yy)})"{_cls("label")}>{_html.escape(ylabel)}</text>')
            return "".join(out)

        def e(T):
            x0, y0, x1, y1 = T.box
            xt, yt = ticks(T)
            (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
            out = [_text_box(T(x, y0)[0], by0 + 13, xf(x), 10.5, "middle", "middle") for x in xt]
            out += [_text_box(bx0 - 6, T(x0, y)[1], yf(y), 10.5, "end", "middle") for y in yt]
            if xlabel:
                out.append(_text_box((bx0 + bx1) / 2, by0 + 31, xlabel, 12, "middle", "middle"))
            if ylabel:
                yy, xx = (by0 + by1) / 2, bx0 - 12 - max((text_width(yf(y), 10.5) for y in yt), default=0) - 8
                out.append(_text_box(xx, yy, ylabel, 12, "middle", "middle", -90))
            return out
        return self._add(f, None, e, under=True)

    def legend(self, entries, *, at: str = "top-right"):
        """entries: [(class of the line or patch, label)]; at: top-right, top-left, bottom-right, bottom-left (inside
        the plot box) or "right" (outside, beside it)"""
        entries = list(entries)

        def geo(T):
            x0, y0, x1, y1 = T.box
            w = max(text_width(s, 10.5) for _, s in entries) + 40
            h = 16 * len(entries) + 8
            if at == "right":
                return T(x1, y1)[0] + 10, T(x1, y1)[1], w, h
            px = T(x0, y0)[0] + 8 if "left" in at else T(x1, y0)[0] - 8 - w
            py = T(x0, y1)[1] + 8 if "top" in at else T(x0, y0)[1] - 8 - h
            return px, py, w, h

        def f(T):
            px, py, w, h = geo(T)
            out = [f'<rect x="{_fmt(px)}" y="{_fmt(py)}" width="{_fmt(w)}" height="{_fmt(h)}" fill="#fff" '
                   f'fill-opacity="0.88" stroke="{C["grid"]}"/>']
            for k, (c, s) in enumerate(entries):
                yy = py + 12 + 16 * k
                if c.split()[0] in AREA_CLASSES:
                    out.append(f'<rect x="{_fmt(px + 6)}" y="{_fmt(yy - 5)}" width="18" height="10"{_cls(c)}/>')
                elif c.split()[0] in ("bar", "tendon", "external"):
                    out.append(f'<circle cx="{_fmt(px + 15)}" cy="{_fmt(yy)}" r="4"{_cls(c)}/>')
                else:
                    out.append(f'<line x1="{_fmt(px + 6)}" y1="{_fmt(yy)}" x2="{_fmt(px + 24)}" y2="{_fmt(yy)}"{_cls(c)}/>')
                out.append(f'<text x="{_fmt(px + 30)}" y="{_fmt(yy)}" dominant-baseline="middle"{_cls("small")}>'
                           f'{_html.escape(s)}</text>')
            return "".join(out)

        def e(T):
            px, py, w, h = geo(T)
            return [(px, py, px + w, py + h)]
        return self._add(f, None, e)

    # ── output ─────────────────────────────────────────────────────────────
    def svg(self) -> str:
        T = self._transform()
        x0, y0, x1, y1 = T.box
        (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
        clip_id = f"{self.id}-clip"
        defs = (f'<defs><clipPath id="{clip_id}"><rect x="{_fmt(bx0)}" y="{_fmt(by1)}" width="{_fmt(bx1 - bx0)}" '
                f'height="{_fmt(by0 - by1)}"/></clipPath></defs>')
        body = []
        for it in sorted(self._items, key=lambda i: not i.under):
            s = it.svg(T)
            body.append(f'<g clip-path="url(#{clip_id})">{s}</g>' if it.clip else s)
        title = (f'<text x="{_fmt(self.margin[3])}" y="{_fmt(self.margin[0] + 9)}" dominant-baseline="middle"'
                 f'{_cls("title")}>{_html.escape(self.title)}</text>') if self.title else ""
        W, H = self.width, T.H
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(W)}" height="{_fmt(H)}" '
                f'viewBox="0 0 {_fmt(W)} {_fmt(H)}" class="fm-root" id="{self.id}"><style>{STYLE}</style>{defs}'
                f'<rect width="100%" height="100%" fill="{C["paper"]}"/>{title}{"".join(body)}</svg>')

    def _repr_svg_(self) -> str:
        return self.svg()

    def save(self, path: str | Path) -> Path:
        """.svg, or .html (a page with just this figure)"""
        p = Path(path)
        p.write_text(html([self], title=self.title or "femotion") if p.suffix.lower() in (".html", ".htm")
                     else self.svg(), encoding="utf-8")
        return p


def _pts(T, pts) -> str:
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in (T(*p) for p in pts))


def _arrow_svg(x0, y0, x1, y1, head, cls, head_cls) -> str:
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return ""
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    bx, by = x1 - ux * min(head, L), y1 - uy * min(head, L)
    px, py = -uy * head * 0.42, ux * head * 0.42
    return (f'<line x1="{_fmt(x0)}" y1="{_fmt(y0)}" x2="{_fmt(bx)}" y2="{_fmt(by)}"{_cls(cls)}/>'
            f'<polygon points="{_fmt(x1)},{_fmt(y1)} {_fmt(bx + px)},{_fmt(by + py)} {_fmt(bx - px)},{_fmt(by - py)}"'
            f'{_cls(head_cls)}/>')


def html(parts, *, title: str = "femotion", css: str = "") -> str:
    """a self-contained HTML page: parts are Figures (or anything with svg()) and HTML strings, in order"""
    body = "\n".join(p.svg() if hasattr(p, "svg") else str(p) for p in parts)
    return (f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>{_html.escape(title)}</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<style>body{{font-family:{FONT};margin:24px;color:{C["ink"]};background:#fff}} '
            f'svg{{max-width:100%;height:auto;display:block;margin:8px 0}}{css}</style></head><body>{body}</body></html>')
