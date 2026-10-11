"""Engineering drawings as plain SVG, without plotting libraries (nothing beyond the standard library and NumPy).

A Figure collects shapes in world coordinates (any unit, y up) and maps them to the page when it is written: with one
scale for both axes (sections, elevations: equal=True) or with separate x and y scales (charts: equal=False, see
axes()).
Sizes that belong to the page rather than to the structure — text, arrow heads, dots, dimension offsets — are in
pixels. The page is solved when it is written: a drawing of free height gets the largest scale at which every label,
dimension and arrow fits across the width, a chart keeps its plot box `plot_height` px high and its margins grow until
the tick labels and the labels outside the box fit; then the top and bottom margins are set exactly. Text blocks
(legend(at="above" / "below"), notes()) flow above and below the drawing and wrap at the page width. A figure shows
itself in Jupyter (_repr_svg_), saves as .svg or .html, and goes into an HTML page with html(); stack() puts charts on
one x axis one above the other.

    from femotion.viz import Figure
    fig = Figure(width=480, title="T girder")
    fig.polygon([(0, 0), (1600, 0), (1600, -150), ...], cls="concrete")
    fig.circle((300, -1210), 16, cls="bar")
    fig.dim((0, 0), (1600, 0), offset=18, label="1600")
    fig.notes(["x = 48 mm", ("Mu = 2076 kN·m → OK", "ok")])
    fig.save("girder.svg")

    chart = Figure(width=560, plot_height=200, equal=False).set_box(0, -3500, 40, 3500).axes(xlabel="x (m)")
    chart.polyline(zip(x, M), "demand")          # clipped to the box
    chart.label((20, 1800), "1800", dy=-9)        # placed clear of the other labels and markers
    chart.legend([("demand", "M")], at="above")

The page-pixel building blocks are public as well, for drawings that lay out their own annotations (codes.jtg.figures,
frame.figures): svg_text (with x_{sub} subscripts and a white halo), text_box, wrap, svg_line, svg_poly, svg_circle,
svg_arrow, moment_arc, hatch, support_symbol, dim_chain, callout, leader_column, Placer (greedy label placement).

Styles are CSS classes (STYLE below); every class name is prefixed with "fm-" in the output so that figures can sit in
any page. The default text style sits in :where() (no specificity), so a text class (fm-small, fm-tick, …) always wins.
Text uses a font list that covers Chinese. Coordinates of the geometry are written to PRECISION decimals (areas read
back from the SVG hold to better than 1e-6), those of text to two.
"""
from __future__ import annotations

import html as _html
import itertools
import math
import re
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
    "warn": "#c0392b", "ok": "#2e7d32", "grid": "#ececec", "guide": "#a8a8a8", "hairline": "#c9c9c9",
    # series of a chart, in order (s1 … s6)
    "s1": "#262626", "s2": "#1d5fa6", "s3": "#d27a14", "s4": "#7b3fa8", "s5": "#2e7d32", "s6": "#6b6b6b",
}
C = COLORS
STYLE = f"""
:where(.fm-root) text {{ font-family: {FONT}; fill: {C['ink']}; }}
.fm-title {{ font-size: 15px; font-weight: 600; }}
.fm-label {{ font-size: 12px; }}
.fm-small {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-dimtext {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-tick {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-loadtext {{ font-size: 12px; fill: {C['load']}; }}
.fm-warntext {{ font-size: 12px; fill: {C['warn']}; }}
.fm-note {{ font-size: 12px; }}
.fm-notesmall {{ font-size: 10.5px; fill: {C['muted']}; }}
.fm-notehead {{ font-weight: 600; }}
.fm-noteok {{ fill: {C['ok']}; font-weight: 500; }}
.fm-notefail {{ fill: {C['warn']}; font-weight: 500; }}
.fm-notewarn {{ fill: {C['warn']}; }}
.fm-halo {{ paint-order: stroke; stroke: {C['paper']}; stroke-width: 3px; stroke-linejoin: round; }}
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
.fm-dimtick {{ fill: none; stroke: {C['muted']}; stroke-width: 1.1; }}
.fm-leader {{ fill: none; stroke: {C['muted']}; stroke-width: 0.7; }}
.fm-guide {{ fill: none; stroke: {C['guide']}; stroke-width: 0.8; stroke-dasharray: 2 3; }}
.fm-warn {{ fill: none; stroke: {C['warn']}; stroke-width: 1.6; }}
.fm-limit {{ fill: none; stroke: {C['warn']}; stroke-width: 1.1; stroke-dasharray: 6 3; }}
.fm-axis {{ fill: none; stroke: {C['ink']}; stroke-width: 0.9; }}
.fm-grid {{ fill: none; stroke: {C['grid']}; stroke-width: 0.8; }}
.fm-demand {{ fill: none; stroke: {C['demand']}; stroke-width: 1.8; stroke-linejoin: round; }}
.fm-demand2 {{ fill: none; stroke: {C['demand']}; stroke-width: 1.4; stroke-dasharray: 5 3; stroke-linejoin: round; }}
.fm-resist {{ fill: none; stroke: {C['resistance']}; stroke-width: 1.8; stroke-linejoin: round; }}
.fm-resist2 {{ fill: none; stroke: {C['resistance']}; stroke-width: 1.4; stroke-dasharray: 5 3;
  stroke-linejoin: round; }}
.fm-area-demand {{ fill: {C['demand']}; fill-opacity: 0.12; stroke: none; }}
.fm-area-resist {{ fill: {C['resistance']}; fill-opacity: 0.10; stroke: none; }}
.fm-s1 {{ fill: none; stroke: {C['s1']}; stroke-width: 1.6; stroke-linejoin: round; }}
.fm-s2 {{ fill: none; stroke: {C['s2']}; stroke-width: 1.6; stroke-dasharray: 7 3; stroke-linejoin: round; }}
.fm-s3 {{ fill: none; stroke: {C['s3']}; stroke-width: 1.8; stroke-dasharray: 2 2; stroke-linejoin: round; }}
.fm-s4 {{ fill: none; stroke: {C['s4']}; stroke-width: 1.6; stroke-dasharray: 9 3 2 3; stroke-linejoin: round; }}
.fm-s5 {{ fill: none; stroke: {C['s5']}; stroke-width: 1.6; stroke-dasharray: 1 3; stroke-linecap: round; }}
.fm-s6 {{ fill: none; stroke: {C['s6']}; stroke-width: 1.4; stroke-dasharray: 12 4; }}
.fm-mk-d {{ fill: {C['demand']}; stroke: {C['paper']}; stroke-width: 1; }}
.fm-mk-r {{ fill: {C['paper']}; stroke: {C['resistance']}; stroke-width: 1.5; }}
.fm-mk-s {{ fill: {C['ink']}; stroke: {C['paper']}; stroke-width: 1; }}
.fm-conn {{ fill: none; stroke: {C['ink']}; stroke-width: 0.9; stroke-dasharray: 2 2; }}
.fm-member {{ fill: none; stroke: {C['ink']}; stroke-width: 2.2; stroke-linecap: round; }}
.fm-support {{ fill: {C['paper']}; stroke: {C['ink']}; stroke-width: 1.1; }}
.fm-hatch {{ fill: none; stroke: {C['ink']}; stroke-width: 0.8; }}
.fm-load {{ fill: none; stroke: {C['load']}; stroke-width: 1.3; }}
.fm-loadhead {{ fill: {C['load']}; stroke: none; }}
.fm-loadfill {{ fill: {C['load_fill']}; fill-opacity: 0.8; stroke: {C['load']}; stroke-width: 1; }}
.fm-curve {{ fill: none; stroke: {C['ink']}; stroke-width: 1.4; stroke-linejoin: round; }}
.fm-pos {{ fill: {C['demand']}; fill-opacity: 0.16; stroke: none; }}
.fm-neg {{ fill: {C['resistance']}; fill-opacity: 0.16; stroke: none; }}
"""
# font size (px) of the text classes, for the space a label takes (the first class of a text decides)
TEXT_SIZE = {"title": 15.0, "label": 12.0, "small": 10.5, "dimtext": 10.5, "tick": 10.5, "loadtext": 12.0,
             "warntext": 12.0, "note": 12.0, "notesmall": 10.5}
AREA_CLASSES = ("pos", "neg", "comp", "tens", "loadfill", "concrete", "topping", "void", "area-demand", "area-resist")
DOT_CLASSES = ("bar", "tendon", "external", "mk-d", "mk-r", "mk-s")

PRECISION = 5  # decimals of the coordinates written
GUTTER = 12.0  # page px: the left edge of titles and of page-aligned text blocks, the right edge text wraps at
EDGE = 4.0  # least distance (px) of any label from the page's edge
TITLE_ROW = 24.0  # height of the title's row
BLOCK_GAP = 12.0  # between the drawing and a text block under it, and between blocks
ABOVE_GAP = 6.0  # between a text block over the drawing (a chart's legend) and the drawing
SUB = 0.78  # size of a subscript relative to its text

_ids = itertools.count(1)


# ── numbers ─────────────────────────────────────────────────────────────────
def fmt(v: float, nd: int = PRECISION) -> str:
    """a coordinate for the SVG: nd decimals, no trailing zeros"""
    if not abs(v) < 1e9:
        return f"{v:.0f}"
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def number(v: float, digits: int | None = None) -> str:
    """a number for a label: `digits` decimals, or by magnitude (none from 100, one from 10, two from 1, three
    significant figures below 1); trailing zeros dropped, a real minus sign (−), ∞ for infinities"""
    if not math.isfinite(v):
        return "∞" if v > 0 or math.isnan(v) else "−∞"
    if digits is None:
        a = abs(v)
        s = f"{v:.0f}" if a >= 100 or a == 0 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.3g}"
    else:
        s = f"{v:.{digits}f}"
    if "." in s and "e" not in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.replace("-", "−")


# ── text ────────────────────────────────────────────────────────────────────
_RICH = re.compile(r"_\{([^}]*)\}")


def _rich(s: str) -> list[tuple[str, bool]]:
    """[(text, subscript?)] of a string with x_{sub} subscripts"""
    out, pos = [], 0
    for mt in _RICH.finditer(s):
        if mt.start() > pos:
            out.append((s[pos:mt.start()], False))
        out.append((mt.group(1), True))
        pos = mt.end()
    if pos < len(s):
        out.append((s[pos:], False))
    return out


def plain(s: str) -> str:
    """the characters of a label as read (x_{c} → xc)"""
    return _RICH.sub(r"\1", str(s))


def _cls(c: str | None) -> str:
    """class attribute: every name prefixed with fm-"""
    if not c:
        return ""
    return ' class="' + " ".join(k if k.startswith("fm-") else "fm-" + k for k in c.split()) + '"'


def _first(cls: str | None) -> str:
    return (cls or "label").split()[0].removeprefix("fm-")


def text_width(s: str, size: float) -> float:
    """approximate width (px) of a label: CJK and full-width characters one em, others about 0.58 em; x_{sub}
    subscripts at 0.78 of the size"""
    def w(t, sz):
        return sum(sz if unicodedata.east_asian_width(ch) in ("W", "F") else 0.58 * sz for ch in t)
    return sum(w(t, size * SUB if sub else size) for t, sub in _rich(str(s)))


def text_box(x: float, y: float, s: str, size: float, anchor: str = "middle", baseline: str = "middle",
             rotate: float = 0.0) -> tuple[float, float, float, float]:
    """page box (x0, y0, x1, y1) of a label at (x, y): text_width wide, 1.25 em high, placed by the SVG text anchor
    (start, middle, end) and dominant baseline (middle / central, auto / alphabetic, hanging), turned by `rotate`
    degrees about (x, y)"""
    w, h = text_width(s, size), size * 1.25
    x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0.0)
    y0 = y - (h / 2 if baseline in ("middle", "central") else h * 0.85 if baseline in ("auto", "alphabetic") else 0.0)
    if not rotate:
        return (x0, y0, x0 + w, y0 + h)
    a = math.radians(rotate)
    cs = [(cx - x, cy - y) for cx, cy in ((x0, y0), (x0 + w, y0), (x0, y0 + h), (x0 + w, y0 + h))]
    rx = [x + u * math.cos(a) - v * math.sin(a) for u, v in cs]
    ry = [y + u * math.sin(a) + v * math.cos(a) for u, v in cs]
    return (min(rx), min(ry), max(rx), max(ry))


def svg_text(x: float, y: float, s, cls: str = "label", *, anchor: str = "middle", baseline: str = "middle",
             rotate: float | None = None, halo: bool = False, fill: str | None = None, weight: str | None = None,
             attrs: str = "") -> str:
    """an SVG <text> at page (x, y): x_{sub} written as subscripts, turned by `rotate` degrees about (x, y) (None: no
    transform), halo a white edge under the glyphs (labels over lines), fill and weight override the class's colour
    and weight"""
    size = TEXT_SIZE.get(_first(cls), 12.0)
    body, down = [], False
    for t, sub in _rich(str(s)):
        t = _html.escape(t)
        if sub:
            body.append(f'<tspan dy="{fmt(0.3 * size, 2)}" font-size="{fmt(SUB * size, 2)}px">{t}</tspan>')
            down = True
        else:
            body.append(f'<tspan dy="{fmt(-0.3 * size, 2)}">{t}</tspan>' if down else t)
            down = False
    rot = f' transform="rotate({fmt(rotate, 2)} {fmt(x, 2)} {fmt(y, 2)})"' if rotate is not None else ""
    st = ";".join(v for v in (f"fill:{fill}" if fill else "", f"font-weight:{weight}" if weight else "") if v)
    st = f' style="{st}"' if st else ""
    return (f'<text x="{fmt(x, 2)}" y="{fmt(y, 2)}" text-anchor="{anchor}" dominant-baseline="{baseline}"{rot}'
            f'{_cls(cls + (" halo" if halo else ""))}{st}{attrs}>{"".join(body)}</text>')


def text_item(x: float, y: float, s, cls: str = "label", *, anchor: str = "middle", baseline: str = "middle",
              rotate: float | None = None, **kw) -> tuple[str, tuple]:
    """(svg_text, text_box) of one label: the SVG and the page box it takes; keywords as svg_text"""
    svg = svg_text(x, y, s, cls, anchor=anchor, baseline=baseline, rotate=rotate, **kw)
    return svg, text_box(x, y, s, TEXT_SIZE.get(_first(cls), 12.0), anchor, baseline, rotate or 0.0)


_CJK = "⺀-鿿豈-﫿＀-￯　-〿"
_TOKEN = re.compile(f"[^ \\t\\n{_CJK}]*(?:_\\{{[^}}]*\\}}[^ \\t\\n{_CJK}]*)+[ \\t]*|[{_CJK}]|[^ \\t\\n{_CJK}]+[ \\t]*"
                    f"|[ \\t\\n]+")
_UNITS = r"(?:m|mm|mm²|mm⁴|cm|kN|kN·m|kN/m|N|MPa|kPa|°|°C|%|t|s|Hz)"
_NOBREAK = re.compile(rf"(?<=[\d.)}}]) (?={_UNITS}(?![A-Za-z²⁴]))|(?<= =) | (?==)|(?<=→) | (?=→)"
                      r"|(?<=最大) |(?<=max) |(?<=利用率) |(?<=比值) |(?<=utilisation) |(?<=ratio) ")
_CLOSING = "，；、。）):;,."
_OPENING = "（(《“"


def wrap(s: str, width: float, size: float = 10.5, *, at: str | None = None, indent: float = 0.0) -> list[str]:
    """lines of s no wider than `width` px at font `size` (continuation lines `indent` px narrower). By default CJK
    characters break anywhere and other words at spaces, but a number stays with its unit, "a = b" and "→ OK" stay
    whole (no-break spaces), no line starts with closing punctuation and x_{sub} is never split. at: a regular
    expression of the places a line may break instead (e.g. r" (?=[+=−] )" for formulas: before + = −)"""
    s = str(s)
    if text_width(s, size) <= width:
        return [s]
    if at is not None:
        tokens = [t for t in re.split(at, s) if t]
    else:
        tokens = _TOKEN.findall(_NOBREAK.sub(" ", s))
    lines, cur = [], ""
    for tok in tokens:
        lim = width - (indent if lines else 0.0)
        if cur.strip() and text_width((cur + tok).rstrip(), size) > lim and tok.strip() and \
                (at is not None or tok.strip()[0] not in _CLOSING):
            head, carry = cur.rstrip(), ""
            if at is None and head[-1] in _OPENING and head[:-1].strip():  # an opening bracket goes with what follows
                head, carry = head[:-1].rstrip(), head[-1]
            lines.append(head)
            cur = carry + tok.lstrip()
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines or [""]


# ── page primitives (px) ────────────────────────────────────────────────────
def svg_line(x0, y0, x1, y1, cls: str = "dim", *, style: str = "", attrs: str = "") -> str:
    st = f' style="{style}"' if style else ""
    return f'<line x1="{fmt(x0)}" y1="{fmt(y0)}" x2="{fmt(x1)}" y2="{fmt(y1)}"{_cls(cls)}{st}{attrs}/>'


def svg_poly(pts, cls: str = "curve", *, closed: bool = True, style: str = "", attrs: str = "") -> str:
    """a polygon (closed) or polyline through page points"""
    tag = "polygon" if closed else "polyline"
    st = f' style="{style}"' if style else ""
    return f'<{tag} points="{" ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)}"{_cls(cls)}{st}{attrs}/>'


def svg_circle(x, y, r, cls: str = "bar", *, style: str = "", attrs: str = "", title: str | None = None) -> str:
    st = f' style="{style}"' if style else ""
    tt = f"<title>{_html.escape(title)}</title>" if title else ""
    return f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}"{_cls(cls)}{st}{attrs}>{tt}</circle>' if tt else \
        f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}"{_cls(cls)}{st}{attrs}/>'


def svg_path(rings, cls: str = "concrete", *, attrs: str = "") -> str:
    """a filled shape with holes from page rings [outer, hole, …] (even-odd fill)"""
    d = " ".join("M" + " L".join(f"{fmt(x)},{fmt(y)}" for x, y in r) + " Z" for r in rings)
    return f'<path d="{d}" fill-rule="evenodd"{_cls(cls)}{attrs}/>'


def svg_arrow(x0, y0, x1, y1, *, head: float = 7.0, cls: str = "load", head_cls: str = "loadhead", style: str = "",
              attrs: str = "") -> str:
    """an arrow from (x0, y0) to its head at (x1, y1), page px: a line and a filled head `head` px long (style, e.g.
    "opacity:0.35", goes on both; a stroke colour in it colours the head too)"""
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return ""
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    h = min(head, L)
    bx, by = x1 - ux * h, y1 - uy * h
    px, py = -uy * h * 0.42, ux * h * 0.42
    return (svg_line(x0, y0, bx, by, cls, style=style, attrs=attrs)
            + svg_poly([(x1, y1), (bx + px, by + py), (bx - px, by - py)], head_cls,
                       style=style.replace("stroke", "fill")))


def moment_arc(x: float, y: float, ccw: bool, r: float = 12.0, cls: str = "load") -> str:
    """a curved arrow round (x, y), page px: 270° open at the bottom, the head showing the sense"""
    a0, a1 = (math.radians(-45), math.radians(225)) if ccw else (math.radians(225), math.radians(-45))
    n = 24
    pts = [(x + r * math.cos(a0 + (a1 - a0) * k / n), y - r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]
    return svg_poly(pts[:-2], cls, closed=False) + svg_arrow(*pts[-3], *pts[-1], head=6.0, cls=cls)


def hatch(x0: float, x1: float, y: float, *, below: bool = True, step: float = 4.5, depth: float = 4.5) -> str:
    """ground hatching along a line at page y from x0 to x1, the strokes below it (or above)"""
    sg = 1 if below else -1
    out = [svg_line(x0, y, x1, y, "support", style="fill:none")]
    x = x0 + 2
    while x <= x1 + 1e-9:
        out.append(svg_line(x, y, x - depth, y + sg * depth, "hatch"))
        x += step
    return "".join(out)


def support_symbol(kind: str, x: float, y: float, *, outward=(1.0, 0.0), vertical_member: bool = False):
    """the symbol of a support at a node (page px): "fixed" (a wall on the side `outward` points to, or ground under a
    vertical member), "pin", "roller", "guide" (horizontal restraint only). Returns (svg, page box)"""
    if kind == "pin":
        h, b = 11.0, 8.0
        svg = svg_poly([(x, y), (x - b, y + h), (x + b, y + h)], "support") + hatch(x - 12, x + 12, y + h)
        return svg, (x - 13, y, x + 13, y + h + 5)
    if kind == "roller":
        h, b, r = 9.0, 7.0, 2.2
        svg = svg_poly([(x, y), (x - b, y + h), (x + b, y + h)], "support")
        svg += "".join(svg_circle(x + dx, y + h + r, r, "support") for dx in (-4.2, 4.2))
        svg += hatch(x - 12, x + 12, y + h + 2 * r)
        return svg, (x - 13, y, x + 13, y + h + 2 * r + 5)
    if kind == "guide":
        h, b = 10.0, 6.5
        sx = -1 if outward[0] >= 0 else 1
        xg = x + sx * h
        svg = svg_poly([(x, y), (xg, y - b), (xg, y + b)], "support") + svg_line(xg, y - 11, xg, y + 11, "support",
                                                                                   style="fill:none")
        return svg, (min(x, xg) - 6, y - 12, max(x, xg) + 6, y + 12)
    if kind != "fixed":
        raise ValueError("support kind: 'fixed', 'pin', 'roller' or 'guide'")
    if vertical_member:
        return hatch(x - 13, x + 13, y), (x - 14, y, x + 14, y + 6)
    ox = 1 if outward[0] >= 0 else -1
    out = [svg_line(x, y - 12, x, y + 12, "support", style="fill:none;stroke-width:1.6")]
    yy = y - 11
    while yy <= y + 12:
        out.append(svg_line(x, yy, x + ox * 4.5, yy + 4.5, "hatch"))
        yy += 4.5
    return "".join(out), (min(x, x + ox * 6) - 1, y - 13, max(x, x + ox * 6) + 1, y + 17)


def dim_chain(pos, at: float, labels, *, vertical: bool = False, ext=None, side: int = -1, short: str = "beyond",
              cls: str = "dim", text_cls: str = "dimtext"):
    """a chain of dimensions on the page: the points `pos` (page x, or page y when vertical) along a dimension line at
    `at` (page y, or x), a 45° tick at each point and labels[k] between pos[k] and pos[k + 1] (None: no label) on the
    `side` of the line (−1 above / left, +1 below / right; vertical labels read from the right). ext: where the
    extension lines start (one page coordinate, or one per point): they run from 3 px off the object to 3 px past the
    line. A label longer than its segment: short="drop" leaves it out; "beyond" puts it past the end of the chain (a
    single dimension: past its far end, pos[1]), between two others a vertical chain writes it across, a horizontal
    one a row further out. Returns (svg, boxes)."""
    pos = [float(p) for p in pos]
    P = (lambda u, v: (v, u)) if vertical else (lambda u, v: (u, v))  # noqa: E731
    out, boxes = [], []
    lo, hi = min(pos), max(pos)
    out.append(svg_line(*P(lo, at), *P(hi, at), cls))
    for k, u in enumerate(pos):
        if ext is not None:
            e = float(ext[k]) if np.ndim(ext) else float(ext)
            sg = 1.0 if at >= e else -1.0
            out.append(svg_line(*P(u, e + 3 * sg), *P(u, at + 3 * sg), cls))
        x, y = P(u, at)
        out.append(svg_line(x - 3, y + 3, x + 3, y - 3, "dimtick"))
    size = TEXT_SIZE.get(_first(text_cls), 10.5)
    n = len(pos) - 1
    off = side * 7.0
    rot = -90.0 if vertical else None
    for k, lab in enumerate(labels):
        if lab is None or k >= n:
            continue
        u0, u1 = pos[k], pos[k + 1]
        w = text_width(lab, size)
        anchor, r = "middle", rot
        if abs(u1 - u0) >= w + 4:
            u, v = (u0 + u1) / 2, at + off
        elif short == "drop":
            continue
        elif n == 1:  # past the far end, the line drawn on to it
            sg = 1.0 if u1 >= u0 else -1.0
            u, v = u1 + sg * (w / 2 + 6), at + off
            out.append(svg_line(*P(u1, at), *P(u1 + sg * 4, at), cls))
        elif k == 0:
            u, v = lo - w / 2 - 6, at + off
        elif k == n - 1:
            u, v = hi + w / 2 + 6, at + off
        elif vertical:  # across the line
            u, v, r, anchor = (u0 + u1) / 2, at + side * 5.0, None, "end" if side < 0 else "start"
        else:
            u, v = (u0 + u1) / 2, at + off + side * size * 1.3
        t, b = text_item(*P(u, v), lab, text_cls, anchor=anchor, rotate=r, halo=True)
        out.append(t)
        boxes.append(b)
    x0, y0 = P(lo, at)
    x1, y1 = P(hi, at)
    boxes.append((min(x0, x1) - 4, min(y0, y1) - 4, max(x0, x1) + 4, max(y0, y1) + 4))
    return "".join(out), boxes


def callout(x: float, y: float, n, *, color: str | None = None, attrs: str = ""):
    """a numbered callout: the number in a circle of radius 7 px centred at (x, y). Returns (svg, page box)"""
    c = color or C["warn"]
    svg = (f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="7" fill="{C["paper"]}" stroke="{c}" stroke-width="1.2"{attrs}/>'
           + svg_text(x, y + 0.5, str(n), "small", fill=c))
    return svg, (x - 7, y - 7, x + 7, y + 7)


def spread(ys, gap: float, *, top: float | None = None, bottom: float | None = None) -> list[float]:
    """positions for labels in one column (page y, in the order given, which should be their anchors' order): each at
    its anchor where it can and at least `gap` below the one before; a column running past `bottom` is pulled up, not
    above `top`"""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + gap) if out else y)
    if out and bottom is not None:
        over = out[-1] - max(ys[-1], bottom)
        if over > 0:
            out = [y - over for y in out]
            for k in range(len(out)):
                lo = (top if top is not None else -math.inf) if k == 0 else out[k - 1] + gap
                out[k] = max(out[k], lo)
    return out


def leader_column(items, x: float, *, gap: float = 15.0, top: float | None = None, bottom: float | None = None):
    """labels in one column starting at page x, each joined by a leader to its anchor. items: (ax, ay, r, text, cls,
    colour) with r the anchor's radius (px; the leader starts 1.5 px outside it) and colour the text's and the leader's
    (None: the class's); a text that is an int is drawn as a numbered callout. Order and positions as spread(). The
    texts carry data-role="label", the callouts data-role="callout". Returns (svg, boxes)."""
    items = sorted(items, key=lambda v: v[1])
    if not items:
        return "", []
    ys = spread([v[1] for v in items], gap, top=top, bottom=bottom)
    out, boxes = [], []
    for (ax, ay, r, txt, cls, colour), y in zip(items, ys):
        dx, dy = x - 4 - ax, y - ay
        dist = math.hypot(dx, dy) or 1.0
        sx, sy = ax + dx / dist * (r + 1.5), ay + dy / dist * (r + 1.5)
        out.append(svg_poly([(sx, sy), (x - 4, y)], "leader", closed=False,
                            style=f"stroke:{colour}" if colour else ""))
        if isinstance(txt, int):
            s, b = callout(x + 7, y, txt, color=colour, attrs=' data-role="callout"')
        else:
            s, b = text_item(x, y, txt, cls, anchor="start", halo=True, fill=colour, attrs=' data-role="label"')
        out.append(s)
        boxes.append(b)
    return "".join(out), boxes


# ── label placement ─────────────────────────────────────────────────────────
def overlap(a, b, pad: float = 1.5) -> bool:
    """whether two page boxes come closer than pad px"""
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def shifted(cands, step: float, n: int = 3) -> list:
    """candidate positions (x, y, anchor), then the same shifted by `step` px down (up when negative) one to n times"""
    cands = list(cands)
    return cands + [(x, y + step * k, a) for k in range(1, n + 1) for (x, y, a) in cands]


class Placer:
    """greedy placement of labels on the page (px): each label takes the first of its candidate positions whose box
    keeps clear of everything blocked or placed so far. Labels queued with add() are placed by layout() in order of
    priority (higher first, then in the order added); place() places one at once. A label with no free candidate is
    left out, or put at its first candidate when required. inside=True keeps a label within `bounds` (a page box), room
    = (x0, x1) page px shows it only where its text fits between them. Placed labels are drawn with a white halo."""

    def __init__(self, bounds=None, pad: float = 1.5):
        self.bounds, self.pad = bounds, pad
        self.boxes = []  # everything taken: blocked boxes and placed labels
        self.placed = []  # (svg, box)
        self._queue = []

    def block(self, box) -> None:
        self.boxes.append(tuple(box))

    def mark(self, x: float, y: float, r: float = 4.0) -> None:
        """a marker at (x, y) that labels keep clear of"""
        self.block((x - r, y - r, x + r, y + r))

    def add(self, text: str, cands, cls: str = "small", *, priority: float = 0, **kw) -> None:
        """queue a label: cands [(x, y, anchor)] in order of preference; keywords as place()"""
        self._queue.append((-priority, len(self._queue), text, list(cands), cls, kw))

    def layout(self) -> "Placer":
        for _, _, text, cands, cls, kw in sorted(self._queue, key=lambda q: q[:2]):
            self.place(text, cands, cls, **kw)
        self._queue = []
        return self

    def place(self, text: str, cands, cls: str = "small", *, required: bool = False, inside: bool = False,
              room=None, halo: bool = True, fill: str | None = None, attrs: str = ""):
        """place one label now; returns its page box (None when left out)"""
        size = TEXT_SIZE.get(_first(cls), 12.0)
        if room is not None and abs(room[1] - room[0]) < text_width(text, size) + 6:
            return None
        first = None
        for x, y, anchor in cands:
            b = text_box(x, y, text, size, anchor, "middle")
            first = first or (x, y, anchor, b)
            if inside and self.bounds is not None:
                bx0, by0, bx1, by1 = self.bounds
                if not (b[0] >= bx0 + 1 and b[2] <= bx1 - 1 and b[1] >= by0 + 1 and b[3] <= by1 - 1):
                    continue
            if any(overlap(b, o, self.pad) for o in self.boxes):
                continue
            return self._put(x, y, anchor, b, text, cls, halo, fill, attrs)
        if required and first is not None:
            return self._put(*first, text, cls, halo, fill, attrs)
        return None

    def _put(self, x, y, anchor, b, text, cls, halo, fill, attrs):
        self.boxes.append(b)
        self.placed.append((svg_text(x, y, text, cls, anchor=anchor, halo=halo, fill=fill, attrs=attrs), b))
        return b

    def svg(self) -> str:
        return "".join(s for s, _ in self.placed)

    @property
    def label_boxes(self) -> list:
        return [b for _, b in self.placed]


# ── data ────────────────────────────────────────────────────────────────────
def split(x, y) -> list[list[tuple[float, float]]]:
    """polylines of the finite points of y(x): NaN or inf breaks a line"""
    out, cur = [], []
    for a, b in zip(np.asarray(x, float), np.asarray(y, float)):
        if math.isfinite(a) and math.isfinite(b):
            cur.append((float(a), float(b)))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def beyond(x, y, level: float, above: bool = True) -> list[list[tuple[float, float]]]:
    """the parts of the polyline y(x) beyond a level (above it, or below with above=False), closed by the crossing
    points (NaN breaks a part)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = (y - level) if above else (level - y)
    out, cur = [], []
    for i in range(len(x)):
        if not (math.isfinite(s[i]) and math.isfinite(x[i])):
            if cur:
                out.append(cur)
            cur = []
            continue
        prev_ok = i > 0 and math.isfinite(s[i - 1])
        if s[i] > 0:
            if not cur and prev_ok and s[i - 1] <= 0:
                t = s[i - 1] / (s[i - 1] - s[i])
                cur.append((float(x[i - 1] + t * (x[i] - x[i - 1])), level))
            cur.append((float(x[i]), float(y[i])))
        elif cur:
            if prev_ok:
                t = s[i - 1] / (s[i - 1] - s[i])
                cur.append((float(x[i - 1] + t * (x[i] - x[i - 1])), level))
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


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


# ── text blocks ─────────────────────────────────────────────────────────────
@dataclass
class Note:
    """a line of a notes block. tone: space-separated words — small (10.5 px, muted), head (semibold, a little space
    above), ok (green: a verdict met), fail (red: a verdict not met), warn (red: a remark). number: a numbered callout
    before the text. breaks: where the line may wrap (see wrap's `at`); indent: of the continuation lines"""
    text: str
    tone: str = ""
    number: int | None = None
    breaks: str | None = None
    indent: float = 0.0

    @classmethod
    def of(cls, v) -> "Note":
        if isinstance(v, Note):
            return v
        if isinstance(v, str):
            return cls(v)
        return cls(*v)


def _swatch(kind, cls, x, y) -> str:
    """a legend symbol 20 px wide starting at page x, centred on y"""
    if callable(kind):
        return kind(x, y)
    if kind == "area":
        return f'<rect x="{fmt(x + 1)}" y="{fmt(y - 5)}" width="18" height="10"{_cls(cls)}/>'
    if kind == "dot":
        return svg_circle(x + 10, y, 3.6, cls)
    if kind == "ring":
        return f'<rect x="{fmt(x + 3)}" y="{fmt(y - 5)}" width="14" height="10" rx="4"{_cls(cls)}/>'
    return svg_line(x, y, x + 20, y, cls)


def _kind(cls: str, kind) -> object:
    if kind is not None:
        return kind
    c = _first(cls)
    return "area" if c in AREA_CLASSES else "dot" if c in DOT_CLASSES else "line"


def _legend_entries(entries):
    out = []
    for e in entries:
        if not e:
            continue
        cls, label, *rest = e
        out.append((cls, label, _kind(cls, rest[0] if rest else None)))
    return out


def _legend_flow(entries, x, y, width):
    """legend entries in rows from page (x, y), wrapping at width: (svg, boxes, height)"""
    out, boxes, cx, row = [], [], x, 0
    for cls, label, kind in entries:
        w = 26 + text_width(label, TEXT_SIZE["small"])
        if cx > x and cx + w > x + width:
            row, cx = row + 1, x
        yy = y + 8 + 16 * row
        out.append(_swatch(kind, cls, cx, yy) + svg_text(cx + 26, yy, label, "small", anchor="start"))
        boxes.append((cx, yy - 8, cx + w, yy + 8))
        cx += w + 14
    return "".join(out), boxes, 16.0 * (row + 1)


def _notes_flow(lines, x, y, width):
    """note lines from page (x, y) down, wrapped at width: (svg, boxes, height)"""
    out, boxes, y0 = [], [], y
    for k, ln in enumerate(lines):
        tone = ln.tone.split()
        small = "small" in tone
        cls = ("notesmall" if small else "note") + "".join(f" note{t}" for t in tone if t in ("head", "ok", "fail",
                                                                                              "warn"))
        size = TEXT_SIZE["notesmall" if small else "note"]
        lh = size + 5
        if "head" in tone and k:
            y += 4
        indent = 0.0
        if ln.number is not None:
            s, b = callout(x + 7, y + lh / 2, ln.number)
            out.append(s)
            boxes.append(b)
            indent = 20.0
        for i, part in enumerate(wrap(ln.text, width - indent, size, at=ln.breaks, indent=ln.indent)):
            t, b = text_item(x + indent + (ln.indent if i else 0.0), y + lh / 2, part, cls, anchor="start")
            out.append(t)
            boxes.append(b)
            y += lh
    return "".join(out), boxes, y - y0


# ── the figure ──────────────────────────────────────────────────────────────
@dataclass
class _Item:
    svg: Callable  # T -> str
    pts: list | None = None  # world points that set the content box
    extent: Callable | None = None  # T -> [page boxes (x0, y0, x1, y1)] the item needs on the page
    clip: bool = False  # clipped to the plot box (charts)
    layer: int = 0  # −1 under everything (axes, grid), 0, 1 over everything (labels)


class Figure:
    """a drawing in world coordinates (y up), width px wide.

    equal=True (sections, elevations): one scale for both axes. Without a height the page is solved: the largest scale
    at which every label, dimension and arrow fits across the width (the left margin growing for what sticks out on the
    left), then the top and bottom margins exactly. With a height the margins grow to fit within it.
    equal=False (charts): separate scales. plot_height: the plot box is that many px high and the margins grow round it
    for the tick labels and anything outside the box; or a fixed total height.
    margin (px): the least space round the drawing (top, right, bottom, left); pad: extra world-unit space round the
    content; title: a heading in the top row; css: further style rules for this figure."""

    def __init__(self, width: float = 560, height: float | None = None, *, equal: bool = True,
                 plot_height: float | None = None, margin=(12, 12, 12, 12), title: str | None = None, pad: float = 0.0,
                 css: str = ""):
        if not equal and height is None and plot_height is None:
            raise ValueError("charts (equal=False) need a height or a plot_height")
        self.width, self.height, self.equal = float(width), (None if height is None else float(height)), equal
        self.plot_height = None if plot_height is None else float(plot_height)
        self.margin = tuple(float(m) for m in margin)
        self.title = title
        self.pad = pad
        self.css = css
        self._items: list[_Item] = []
        self._above: list = []  # text blocks over the drawing: (kind, payload, align)
        self._below: list = []
        self._box: list | None = None
        self._labels: _Labels | None = None
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
        """the page map for margins m = (top, right, bottom, left)"""
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
            sx = w / dx
            if self.plot_height is not None:
                sy, H = self.plot_height / dy, t + self.plot_height + b
            else:
                H = self.height
                sy = max(H - t - b, 1.0) / dy
            ox, oy = l, t

        def T(x, y):
            return ox + (x - x0) * sx, oy + (y1 - y) * sy
        T.sx, T.sy, T.H, T.W, T.box, T.margin = sx, sy, H, self.width, (x0, y0, x1, y1), tuple(m)
        return T

    def _boxes(self, T):
        return [bx for it in self._items if it.extent for bx in it.extent(T)]

    def _top_limit(self):
        return self.margin[0] + TITLE_ROW + 2 if self.title else EDGE

    def _transform(self):
        """the page map: the margins solved (free height, plot_height) or grown (fixed height) until every item's page
        extent and the text blocks fit"""
        t0, r0, b0, l0 = self.margin
        t0 += TITLE_ROW if self.title else 0.0
        if self.height is not None:
            return self._grow([t0, r0, b0, l0])
        if self.equal:
            l, r = self._solve_scale(t0, r0, b0, l0)
        else:
            l, r = self._solve_sides(t0, r0, b0, l0)
        return self._solve_vertical([t0, r, b0, l])

    def _solve_scale(self, t0, r0, b0, l0):
        """(left, right) margins of an equal-scale drawing of free height: the largest scale at which every page
        extent fits across the width (bisection; the left margin grown until nothing sticks out on the left — items at
        fixed page positions do not move with it)"""
        x0, _, x1, _ = self._box or self.content_box()
        dx = max(x1 - x0, 1e-12)
        W = self.width

        def fit(s):
            left = l0
            bx = []
            for _ in range(4):
                bx = self._boxes(self._make([t0, W - left - dx * s, b0, left]))
                if not bx:
                    return left + dx * s <= W - r0 + 1e-9, left
                need = EDGE - min(b[0] for b in bx)
                if need <= 1e-6:
                    break
                left += need
            ok = max(b[2] for b in bx) <= W - EDGE + 1e-6 and left + dx * s <= W - r0 + 1e-9
            return ok, left

        hi = max(W - l0 - r0, 1.0) / dx
        ok, left = fit(hi)
        if not ok:
            lo = 1e-6 * hi
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if fit(mid)[0]:
                    lo = mid
                else:
                    hi = mid
            hi = lo
            left = fit(hi)[1]
        return left, W - left - dx * hi

    def _solve_sides(self, t0, r0, b0, l0):
        """(left, right) margins of a chart: grown from the least ones until the page extents fit across the width"""
        l, r = l0, r0
        for _ in range(8):
            bx = self._boxes(self._make([t0, r, b0, l]))
            if not bx:
                break
            nl, nr = EDGE - min(b[0] for b in bx), max(b[2] for b in bx) - (self.width - EDGE)
            if nl <= 0.01 and nr <= 0.01:
                break
            l, r = l + max(nl, 0.0), r + max(nr, 0.0)
        return l, r

    def _drawing_span(self, T):
        """(highest, lowest) page y of the drawing: the content box and every item's extent"""
        x0, y0, x1, y1 = T.box
        ys = [T(x0, y1)[1], T(x0, y0)[1]]
        bx = self._boxes(T)
        return min(ys + [b[1] for b in bx]), max(ys + [b[3] for b in bx])

    def _solve_vertical(self, m):
        """the top margin grown until the drawing and the blocks over it clear the title (or the page edge), the bottom
        one set so that the page ends just below the drawing and the blocks under it"""
        T = self._make(m)
        top, _ = self._drawing_span(T)
        above = self._block_height(self._above, T)
        need = self._top_limit() - (top - above)
        if need > 0:
            m = [m[0] + need, m[1], m[2], m[3]]
            T = self._make(m)
        _, low = self._drawing_span(T)
        below = self._block_height(self._below, T)
        x0, y0, x1, y1 = T.box
        H = max(T(x0, y0)[1] + self.margin[2], low + below + 2 * EDGE)
        content = (y1 - y0) * T.sy
        m = [m[0], m[1], H - m[0] - content, m[3]]
        return self._make(m)

    def _grow(self, m):
        """the page map with margins grown until every item's page extent fits (a few passes): fixed heights"""
        T = self._make(m)
        for _ in range(8):
            boxes = self._boxes(T) + self._blocks(T)[1]
            if not boxes:
                break
            a = np.asarray(boxes, float)
            gx0, gy0, gx1, gy1 = a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()
            need = [max(0.0, self._top_limit() - gy0), max(0.0, gx1 - (T.W - EDGE)), max(0.0, gy1 - (T.H - EDGE)),
                    max(0.0, EDGE - gx0)]
            if max(need) < 0.5:
                break
            m = [mi + ni for mi, ni in zip(m, need)]
            T = self._make(m)
        return T

    def to_page(self, x: float, y: float):
        return self._transform()(x, y)

    # ── text blocks ────────────────────────────────────────────────────────
    def _block_x(self, T, align):
        if align == "box" or (align is None and not self.equal):
            return T(T.box[0], 0)[0]
        return GUTTER

    def _flow(self, blk, T, x, y):
        kind, payload, _ = blk
        width = max(self.width - GUTTER - x, 120.0)
        return (_legend_flow if kind == "legend" else _notes_flow)(payload, x, y, width)

    def _block_height(self, blocks, T):
        if not blocks:
            return 0.0
        gap = ABOVE_GAP if blocks is self._above else BLOCK_GAP
        return sum(self._flow(b, T, self._block_x(T, b[2]), 0.0)[2] for b in blocks) + gap * len(blocks)

    def _blocks(self, T):
        """(svg, boxes) of the text blocks: those above end ABOVE_GAP over the drawing, those below start BLOCK_GAP
        under it, in the order given"""
        if not self._above and not self._below:
            return "", []
        top, low = self._drawing_span(T)
        out, boxes = [], []
        y = top - self._block_height(self._above, T)
        for b in self._above:
            s, bx, h = self._flow(b, T, self._block_x(T, b[2]), y)
            out.append(s)
            boxes += bx
            y += h + ABOVE_GAP
        y = low
        for b in self._below:
            s, bx, h = self._flow(b, T, self._block_x(T, b[2]), y + BLOCK_GAP)
            out.append(s)
            boxes += bx
            y += h + BLOCK_GAP
        return "".join(out), boxes

    def notes(self, lines, *, at: str = "below", align: str | None = None) -> "Figure":
        """a block of text lines under the drawing (or over it, at="above"), wrapped at the page width: each line a
        string, a Note or (text, tone[, number]) — tone as Note: small, head, ok, fail, warn. align: "box" (with the
        plot box, the default of charts) or "page" (the page's left gutter, the default of drawings)"""
        lines = [n for n in (Note.of(v) for v in lines if v) if n.text]
        if lines:
            (self._above if at == "above" else self._below).append(("notes", lines, align))
        return self

    def legend(self, entries, *, at: str = "above", align: str | None = None) -> "Figure":
        """entries: (class, label[, kind]); kind "line", "area", "dot", "ring" (a rounded outline) or a function (x, y)
        -> svg of a 20 px symbol from x centred on y; by default from the class (areas, dots, else a line).
        at: "above" or "below" the drawing (rows that wrap at the page width, aligned as notes), or inside the plot box
        at "top-right", "top-left", "bottom-right", "bottom-left", or "right" of it"""
        ents = _legend_entries(entries)
        if not ents:
            return self
        if at in ("above", "below"):
            (self._above if at == "above" else self._below).append(("legend", ents, align))
            return self

        def geo(T):
            x0, y0, x1, y1 = T.box
            w = max(text_width(s, TEXT_SIZE["small"]) for _, s, _ in ents) + 40
            h = 16 * len(ents) + 8
            if at == "right":
                return T(x1, y1)[0] + 10, T(x1, y1)[1], w, h
            px = T(x0, y0)[0] + 8 if "left" in at else T(x1, y0)[0] - 8 - w
            py = T(x0, y1)[1] + 8 if "top" in at else T(x0, y0)[1] - 8 - h
            return px, py, w, h

        def f(T):
            px, py, w, h = geo(T)
            out = [f'<rect x="{fmt(px)}" y="{fmt(py)}" width="{fmt(w)}" height="{fmt(h)}" fill="{C["paper"]}" '
                   f'fill-opacity="0.88" stroke="{C["grid"]}"/>']
            for k, (c, s, kind) in enumerate(ents):
                yy = py + 12 + 16 * k
                out.append(_swatch(kind, c, px + 6, yy) + svg_text(px + 32, yy, s, "small", anchor="start"))
            return "".join(out)

        def e(T):
            px, py, w, h = geo(T)
            return [(px, py, px + w, py + h)]
        return self._add(f, None, e, layer=1)

    # ── shapes in world coordinates ────────────────────────────────────────
    def _add(self, svg, pts=None, extent=None, *, clip=False, layer=0):
        pts = None if pts is None else [tuple(map(float, p)) for p in pts]
        it = _Item(svg, pts, extent, clip and not self.equal and self._box is not None, layer)
        self._items.append(it)
        return self

    def polyline(self, pts, cls="curve", *, fit: bool = True):
        pts = [tuple(p) for p in pts]
        return self._add(lambda T: svg_poly([T(*p) for p in pts], cls, closed=False), pts if fit else None, clip=True)

    def line(self, p0, p1, cls="curve", *, fit: bool = True):
        return self.polyline([p0, p1], cls, fit=fit)

    def polygon(self, pts, cls="concrete", *, fit: bool = True):
        pts = [tuple(p) for p in pts]
        return self._add(lambda T: svg_poly([T(*p) for p in pts], cls), pts if fit else None, clip=True)

    def path(self, rings, cls="concrete", *, fit: bool = True):
        """a filled shape with holes: rings = [outer, hole, hole, …] (even-odd fill)"""
        rings = [[tuple(p) for p in r] for r in rings]
        return self._add(lambda T: svg_path([[T(*p) for p in r] for r in rings], cls),
                         [p for r in rings for p in r] if fit else None, clip=True)

    def rect(self, x0, y0, x1, y1, cls="concrete", *, fit: bool = True):
        return self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], cls, fit=fit)

    def circle(self, c, r: float, cls="bar", *, fit: bool = True, title: str | None = None):
        """a circle of world radius r (at least 0.6 px on the page); title: a hover tip"""
        c = tuple(c)
        return self._add(lambda T: svg_circle(*T(*c), max(r * T.sx, 0.6), cls, title=title),
                         [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)] if fit else None)

    def dot(self, c, r_px: float = 3.0, cls="bar", *, fit: bool = True):
        """a dot of page radius r_px at a world point"""
        c = tuple(c)

        def e(T):
            x, y = T(*c)
            return [(x - r_px, y - r_px, x + r_px, y + r_px)]
        return self._add(lambda T: svg_circle(*T(*c), r_px, cls), [c] if fit else None, e)

    def text(self, p, s: str, cls="label", *, anchor: str = "middle", dx: float = 0.0, dy: float = 0.0,
             rotate: float = 0.0, baseline: str = "middle", halo: bool = False, fit: bool = False):
        """text at a world point, shifted by (dx, dy) px (dy down); anchor start / middle / end; x_{sub} subscripts"""
        p = tuple(p)
        size = TEXT_SIZE.get(_first(cls), 12.0)

        def at(T):
            x, y = T(*p)
            return x + dx, y + dy
        return self._add(lambda T: svg_text(*at(T), s, cls, anchor=anchor, baseline=baseline, rotate=rotate or None,
                                            halo=halo),
                         [p] if fit else None, lambda T: [text_box(*at(T), s, size, anchor, baseline, rotate)])

    def arrow(self, p0, p1, cls="load", *, head: float = 7.0, head_cls: str = "loadhead", fit: bool = True):
        """a line from p0 to p1 (world) with a head of `head` px at p1"""
        p0, p1 = tuple(p0), tuple(p1)
        return self._add(lambda T: svg_arrow(*T(*p0), *T(*p1), head=head, cls=cls, head_cls=head_cls),
                         [p0, p1] if fit else None)

    def force(self, p, direction=(0.0, -1.0), length: float = 36.0, *, cls="load", label: str | None = None,
              head: float = 7.0, label_cls: str = "loadtext"):
        """a force arrow `length` px long whose head touches the world point p, pointing in `direction` (world)"""
        p = tuple(p)
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) or 1.0)
        size = TEXT_SIZE.get(_first(label_cls), 12.0)

        def tail(T):
            x1, y1 = T(*p)
            return x1 - d[0] * length, y1 + d[1] * length

        def lab(T):
            x0, y0 = tail(T)
            return x0 - d[0] * 8, y0 + d[1] * 8  # beyond the tail

        def f(T):
            out = svg_arrow(*tail(T), *T(*p), head=head, cls=cls)
            return out + (svg_text(*lab(T), label, label_cls, halo=True) if label else "")

        def e(T):
            x0, y0 = tail(T)
            x1, y1 = T(*p)
            out = [(min(x0, x1) - 4, min(y0, y1) - 4, max(x0, x1) + 4, max(y0, y1) + 4)]
            if label:
                out.append(text_box(*lab(T), label, size))
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
            return (x0, y0), (x1, y1), (nx, ny), sg, a0, a1, mid, ang

        def f(T):
            q0, q1, (nx, ny), sg, a0, a1, (mx, my), ang = geo(T)
            out = [svg_line(*a0, *a1, cls)]
            for (x, y), a in ((q0, a0), (q1, a1)):
                out.append(svg_line(x + nx * 3 * sg, y + ny * 3 * sg, a[0] + nx * 3 * sg, a[1] + ny * 3 * sg, cls))
                out.append(svg_line(a[0] - 3, a[1] + 3, a[0] + 3, a[1] - 3, "dimtick"))
            if label is not None:
                out.append(svg_text(mx, my, label, "dimtext", rotate=ang, halo=True))
            return "".join(out)

        def e(T):
            *_, a0, a1, (mx, my), ang = geo(T)
            out = [(min(a0[0], a1[0]) - 4, min(a0[1], a1[1]) - 4, max(a0[0], a1[0]) + 4, max(a0[1], a1[1]) + 4)]
            if label is not None:
                out.append(text_box(mx, my, label, TEXT_SIZE["dimtext"], "middle", "middle", ang))
            return out
        return self._add(f, [p0, p1], e)

    def vlines(self, xs, cls: str = "guide"):
        """vertical lines across the plot box at world x (supports on a chart), clipped to it"""
        for v in xs:
            self._add(lambda T, v=v: svg_line(T(v, 0)[0], T(v, T.box[1])[1], T(v, 0)[0], T(v, T.box[3])[1], cls),
                      clip=True)
        return self

    def raw(self, svg, pts=None, extent=None, *, clip: bool = False, layer: int = 0):
        """any SVG: svg(T) -> str, where T maps world (x, y) to the page and has sx, sy, box, W, H; extent(T) -> page
        boxes; layer −1 under everything, 1 over everything"""
        return self._add(svg, pts, extent, clip=clip, layer=layer)

    # ── labels placed clear of each other ───────────────────────────────────
    def label(self, p, text: str, cls: str = "small", *, dx: float = 0.0, dy: float = 0.0, anchor: str = "middle",
              alt=(), priority: float = 0, room=None, inside: bool | None = None, required: bool = False):
        """a label at a world point shifted (dx, dy) px, placed by the figure's Placer when the page is laid out: the
        first of its positions — this one, then alt [(world point, dx, dy, anchor)] — clear of the labels placed
        before it (higher priority first) and of mark()s; left out when none is (or at the first when required).
        inside (default: charts) keeps it in the plot box; room = (x0, x1) world shows it only where it fits between.
        Labels are drawn over everything, with a white halo."""
        if self._labels is None:
            self._labels = _Labels(self)
        self._labels.items.append((priority, len(self._labels.items), text, [(tuple(p), dx, dy, anchor), *alt], cls,
                                   room, (not self.equal) if inside is None else inside, required))
        return self

    def mark(self, p, r_px: float = 4.0):
        """a marker at a world point that labels keep clear of"""
        if self._labels is None:
            self._labels = _Labels(self)
        self._labels.marks.append((tuple(p), r_px))
        return self

    # ── charts ─────────────────────────────────────────────────────────────
    def axes(self, *, xlabel: str = "", ylabel: str = "", xticks=None, yticks=None, grid: bool = True,
             zero: bool = True, xfmt=None, yfmt=None):
        """frame, ticks, grid and labels of a chart (equal=False), drawn under everything else; set_box first (or the
        content box is used). Tick labels by number() unless xfmt / yfmt"""
        xf, yf = xfmt or number, yfmt or number
        tick = TEXT_SIZE["tick"]

        def ticks(T):
            x0, y0, x1, y1 = T.box
            return (nice_ticks(x0, x1, 6) if xticks is None else list(xticks),
                    nice_ticks(y0, y1, 5) if yticks is None else list(yticks))

        def ylab_x(T, yt):
            return T(T.box[0], 0)[0] - 12 - max((text_width(yf(y), tick) for y in yt), default=0) - 8

        def f(T):
            x0, y0, x1, y1 = T.box
            xt, yt = ticks(T)
            out = []
            if grid:
                out += [svg_line(*T(x, y0), *T(x, y1), "grid") for x in xt]
                out += [svg_line(*T(x0, y), *T(x1, y), "grid") for y in yt]
            (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
            out.append(f'<rect x="{fmt(bx0)}" y="{fmt(by1)}" width="{fmt(bx1 - bx0)}" height="{fmt(by0 - by1)}"'
                       f'{_cls("axis")}/>')
            if zero and y0 < 0 < y1:
                out.append(svg_line(*T(x0, 0.0), *T(x1, 0.0), "axis"))
            out += [svg_text(T(x, y0)[0], by0 + 13, xf(x), "tick") for x in xt]
            out += [svg_text(bx0 - 6, T(x0, y)[1], yf(y), "tick", anchor="end") for y in yt]
            if xlabel:
                out.append(svg_text((bx0 + bx1) / 2, by0 + 31, xlabel, "label"))
            if ylabel:
                yy, xx = (by0 + by1) / 2, ylab_x(T, yt)
                out.append(svg_text(xx, yy, ylabel, "label", rotate=-90))
            return "".join(out)

        def e(T):
            x0, y0, x1, y1 = T.box
            xt, yt = ticks(T)
            (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
            out = [text_box(T(x, y0)[0], by0 + 13, xf(x), tick) for x in xt]
            out += [text_box(bx0 - 6, T(x0, y)[1], yf(y), tick, "end") for y in yt]
            if xlabel:
                out.append(text_box((bx0 + bx1) / 2, by0 + 31, xlabel, TEXT_SIZE["label"]))
            if ylabel:
                out.append(text_box(ylab_x(T, yt), (by0 + by1) / 2, ylabel, TEXT_SIZE["label"], "middle", "middle",
                                    -90))
            return out
        return self._add(f, None, e, layer=-1)

    # ── output ─────────────────────────────────────────────────────────────
    def svg(self) -> str:
        T = self._transform()
        x0, y0, x1, y1 = T.box
        (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
        clip_id = f"{self.id}-clip"
        defs = (f'<defs><clipPath id="{clip_id}"><rect x="{fmt(bx0)}" y="{fmt(by1)}" width="{fmt(bx1 - bx0)}" '
                f'height="{fmt(by0 - by1)}"/></clipPath></defs>')
        body = []
        for it in sorted(self._items, key=lambda i: i.layer):
            s = it.svg(T)
            body.append(f'<g clip-path="url(#{clip_id})">{s}</g>' if it.clip else s)
        body.append(self._blocks(T)[0])
        title = svg_text(GUTTER, self.margin[0] + 9, self.title, "title", anchor="start") if self.title else ""
        W, H = self.width, T.H
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(W)}" height="{fmt(H)}" '
                f'viewBox="0 0 {fmt(W)} {fmt(H)}" class="fm-root" id="{self.id}"><style>{STYLE}{self.css}</style>'
                f'{defs}<rect width="100%" height="100%" fill="{C["paper"]}"/>{title}{"".join(body)}</svg>')

    def _repr_svg_(self) -> str:
        return self.svg()

    def save(self, path: str | Path) -> Path:
        """.svg, or .html (a page with just this figure)"""
        return _save(self, path)


class _Labels:
    """the labels of a figure (Figure.label, Figure.mark), laid out with a Placer for every page mapping (a narrower
    page drops what no longer fits) and drawn over everything"""

    def __init__(self, fig: Figure):
        self.fig, self.items, self.marks, self._key, self._out = fig, [], [], None, None
        fig._add(self._svg, None, self._extent, layer=1)

    def _layout(self, T) -> Placer:
        key = (T.sx, T.sy, T.W, T.H, T(T.box[0], T.box[3]))
        if key == self._key:
            return self._out
        x0, y0, x1, y1 = T.box
        (bx0, by0), (bx1, by1) = T(x0, y0), T(x1, y1)
        P = Placer(bounds=(bx0, by1, bx1, by0))
        for p, r in self.marks:
            P.mark(*T(*p), r)
        for pr, k, s, cands, cls, room, inside, required in self.items:
            pc = []
            for p, dx, dy, anchor in cands:
                u, v = T(*p)
                pc.append((u + dx, v + dy, anchor))
            rp = None if room is None else (T(room[0], 0)[0], T(room[1], 0)[0])
            P.add(s, pc, cls, priority=pr, required=required, inside=inside, room=rp)
        P.layout()
        self._key, self._out = key, P
        return P

    def _svg(self, T):
        return self._layout(T).svg()

    def _extent(self, T):
        return self._layout(T).label_boxes


# ── several figures ─────────────────────────────────────────────────────────
class Stack:
    """figures one above the other with their plot boxes aligned (same x range, same left and right edges): a girder
    elevation over its moment and shear charts. svg(), save(.svg or .html), Jupyter display and html() as a Figure."""

    def __init__(self, figures, *, gap: float = 4.0, width: float | None = None, title: str | None = None):
        self.figures = list(figures)
        if not self.figures:
            raise ValueError("stack: no figures")
        self.gap, self.width, self.title = float(gap), width, title

    def _layout(self):
        figs = self.figures
        W = float(self.width or max(f.width for f in figs))
        saved = [(f.width, f.margin) for f in figs]
        try:
            for f in figs:
                f.width = W
            Ts = [f._transform() for f in figs]
            b0 = Ts[0].box
            span = max(1.0, abs(b0[2] - b0[0]))
            for T in Ts[1:]:
                if abs(T.box[0] - b0[0]) > 1e-9 * span or abs(T.box[2] - b0[2]) > 1e-9 * span:
                    raise ValueError("stack: the figures must share the x range (xlim)")
            for _ in range(6):
                lefts = [T(T.box[0], T.box[1])[0] for T in Ts]
                rights = [W - T(T.box[2], T.box[1])[0] for T in Ts]
                Lm, Rm = max(lefts), max(rights)
                if all(abs(v - Lm) < 0.01 for v in lefts) and all(abs(v - Rm) < 0.01 for v in rights):
                    break
                for f in figs:
                    t, _, b, _ = f.margin
                    f.margin = (t, Rm, b, Lm)
                Ts = [f._transform() for f in figs]
            out = [(f.svg(), T.H) for f, T in zip(figs, Ts)]
        finally:
            for f, (w, m) in zip(figs, saved):
                f.width, f.margin = w, m
        return W, out

    def svg(self) -> str:
        W, parts = self._layout()
        y = TITLE_ROW if self.title else 0.0
        body = [svg_text(GUTTER, 12 + 9, self.title, "title", anchor="start")] if self.title else []
        for s, H in parts:
            body.append(s.replace("<svg ", f'<svg x="0" y="{fmt(y)}" ', 1))
            y += H + self.gap
        H = y - self.gap
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(W)}" height="{fmt(H)}" '
                f'viewBox="0 0 {fmt(W)} {fmt(H)}" class="fm-root"><style>{STYLE}</style>'
                f'<rect width="100%" height="100%" fill="{C["paper"]}"/>{"".join(body)}</svg>')

    def _repr_svg_(self) -> str:
        return self.svg()

    def save(self, path: str | Path) -> Path:
        """.svg, or .html (a page with just this stack)"""
        return _save(self, path)


def stack(figures, *, gap: float = 4.0, width: float | None = None, title: str | None = None) -> Stack:
    """figures (same x range) one above the other with their plot boxes aligned: e.g. stack([beam_elevation(…),
    moment_chart(…, xlabel=False), shear_chart(…)]). width: of all (default the widest)."""
    return Stack(figures, gap=gap, width=width, title=title)


# ── output ──────────────────────────────────────────────────────────────────
def _save(obj, path) -> Path:
    p = Path(path)
    p.write_text(html([obj], title=getattr(obj, "title", None) or "femotion") if p.suffix.lower() in (".html", ".htm")
                 else obj.svg(), encoding="utf-8")
    return p


def html(parts, *, title: str = "femotion", css: str = "") -> str:
    """a self-contained HTML page: parts are Figures (or anything with svg()) and HTML strings, in order"""
    body = "\n".join(p.svg() if hasattr(p, "svg") else str(p) for p in parts)
    return (f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>{_html.escape(title)}</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<style>body{{font-family:{FONT};margin:24px;color:{C["ink"]};background:#fff}} '
            f'svg{{max-width:100%;height:auto;display:block;margin:8px 0}}{css}</style></head>'
            f'<body>{body}</body></html>')
