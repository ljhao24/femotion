"""Drawings of the sections of femotion.codes.jtg: the concrete outline of an RCSection with its reinforcement, ducts and
tendons, true to scale, and the results of a check laid over it, as plain SVG (femotion.viz, no plotting library).

    from femotion.codes import jtg
    from femotion.codes.jtg.figures import section_figure
    sec = jtg.tee(300, 1300, 1800, 150, concrete="C40", bottom=[(4, 28, 55), (4, 28, 115)], top=[(8, 16, 45)])
    chk = jtg.flexure(sec, 1700.0)
    fig = section_figure(chk.section, check=chk, stirrup_d=10)   # a Check keeps its section (Check.section, .hogging)
    fig.save("girder.svg")    # or fig.svg(), viz.html([fig, …]); in Jupyter a section shows itself (RCSection._repr_svg_)
                              # and a check with its drawing (Check._repr_html_)

What is drawn (program conventions: the section stores no horizontal positions):
- outline: the rectangles stacked symmetric about the vertical centre line (rectangles, T, I; holes centred, e.g. a
  hollow pier); boxes (kind "box", or webs ≥ 2) with the total web width split into `webs` equal webs, the outer ones
  flush with the bottom slab, the cells between them; circles concentric. The cast-in-place topping of a composite
  member (jtg.composite, or a section of kind "composite") in its own colour.
- bars: every bar point is a row at its depth, n = area / (π d²/4) bars of diameter d (a point without a diameter: one
  disc of its area). A row is spread evenly over the concrete width at its depth, between the stirrups: the outermost
  centres at side + ds + d/2 from the faces, side the cover of the outermost steel on the sides (side_cover; by default
  the bottom row's clear cover less ds) — the layout section_detailing assumes for the clear spacing of 9.3.3 (the
  narrower width where a row lies on a boundary between pieces, as it does). Where the concrete at that depth is split
  (the webs of a box, the walls of a hollow pier), the bars are shared out in proportion to the widths and each part laid
  out the same way, so a side row of two sits on both faces of a web. A row stays in one layer, as the program takes
  it, even where its bars touch; only bars that cannot lie in one row (a point standing for several layers, e.g. ten
  28 mm bars in a 200 mm web) are drawn in layers about the same depth, at the 9.3.3 clear distance or closer, with a
  note. Circular sections: the bars at their angles on the bar circle (jtg.circle). bar_x={depth: [x, …]} places a row.
- tendons: one duct per tendon row (its diameter from the duct area, the tendon inside as a disc of area Ap); the ducts
  at one depth spread evenly over the concrete there (equal gaps), tendon_x={depth: [x, …]} places them. Pretensioned
  rows as strands of `strand` (nominal diameter, area) laid out like bars when Ap is within 3 % of a whole number of
  them. External tendons: `external_n` per row (default 2, symmetric), in the cell at their depth next to the webs, or
  beside the section where there is no cell.
- check: flexure (the compression zone cut out of the section's shape to depth x — at the bottom for hogging —,
  neutral axis xc = x/β, h0, ξb h0, a text block with x, ξ, ξb, Mu, γ0Md); eccentric compression and tension (zone,
  the load at e from the gross centroid); stresses in service (service_stresses full / A / B, crack_resistance,
  cracked_section's dict) as a stress diagram beside the section, or under it on a narrow page; section_detailing (the
  rows, ducts and webs that miss a rule ringed and numbered, the clause in the text); any other Check as text. A list of
  checks overlays them all.
- dimensions on the left (x, h0, e nearest, then the pieces, then h), widths above and below; the labels of the steel
  in one column on the right with leaders; the text block and the legend under the drawing. Every label's page box is
  kept on the page (femotion.viz solves the page: the largest scale at which everything fits across the width).
"""
from __future__ import annotations

import math
import re

import numpy as np

from ...viz import COLORS as C
from ...viz import (TEXT_SIZE, Figure, Note, dim_chain, fmt, leader_column, number, svg_circle, svg_line, svg_path,
                    svg_poly, text_item, text_width)
from .composite import Composite
from .rc import Check, Circle, RCSection, Rect

__all__ = ["section_figure", "bar_layout"]

_EPS = 1e-9

# ── wording ─────────────────────────────────────────────────────────────────
_S = {
    "zh": {"concrete": "混凝土", "topping": "现浇层", "bar": "普通钢筋", "duct": "预应力管道和钢束", "strand": "预应力筋",
           "external": "体外束", "comp": "受压区", "na": "中和轴", "na_s": "中和轴", "warn": "构造不满足",
           "stirrup": "箍筋", "side": "侧面", "ok": "满足", "not": "不满足", "C": "压", "T": "拉",
           "sigma": "混凝土应力 σ（MPa）", "dia": "Φ", "duct_w": "管道", "ext_w": "体外束",
           "layers": "{row}在程序中是一个点，图中按 {n} 层示意（合力点不变）", "face": {"bottom": "底", "top": "顶"},
           "sag": "正弯矩", "hog": "负弯矩", "large": "大偏心", "small": "小偏心", "over": "超筋",
           "warnings": "提示（见 Check.warnings）：", "failed": "另有 {n} 项未满足（见 Check.failed）",
           "cracked": "开裂", "ratio": "利用率", "limit": "限值", "edge": {"top": "上缘", "bottom": "下缘"},
           "comp_pos": "压为正", "sep": "，", "lp": "（", "rp": "）"},
    "en": {"concrete": "concrete", "topping": "topping", "bar": "reinforcing bars", "duct": "ducts and tendons",
           "strand": "tendons", "external": "external tendons", "comp": "compression zone", "na": "neutral axis",
           "na_s": "N.A.", "warn": "detailing not met", "stirrup": "stirrup", "side": "side", "ok": "OK",
           "not": "NOT OK", "C": "C", "T": "T", "sigma": "concrete stress σ (MPa)", "dia": "Ø", "duct_w": "duct",
           "ext_w": "external", "layers": "{row} is one point in the calculation, drawn in {n} layers (same centroid)",
           "face": {"bottom": "bottom", "top": "top"}, "sag": "sagging", "hog": "hogging", "large": "large eccentricity",
           "small": "small eccentricity", "over": "over-reinforced", "warnings": "warnings (see Check.warnings): ",
           "failed": "{n} more condition(s) not met (see Check.failed)", "cracked": "cracked", "ratio": "ratio",
           "limit": "limit", "edge": {"top": "top", "bottom": "bottom"}, "comp_pos": "compression +", "sep": ", ",
           "lp": " (", "rp": ")"},
}

# check names (prefix of Check.name) → (zh, en)
_NAMES = [
    ("flexure", "正截面受弯", "flexure"),
    ("eccentric compression", "偏心受压", "eccentric compression"),
    ("axial compression", "轴心受压", "axial compression"),
    ("tension", "受拉构件", "tension"),
    ("concrete compression in service", "使用阶段正截面压应力", "stresses in service"),
    ("crack resistance (normal section", "正截面抗裂", "crack resistance"),
    ("section detailing", "截面构造", "section detailing"),
    ("shear", "斜截面抗剪", "shear"),
]

# section_detailing rules (the name of a details entry) → (zh, en) short names; {face}, {zone} are filled in
_RULES = [
    (r"cover of the (bottom|top) bars", "{face}排钢筋保护层", "{face} bar cover"),
    (r"cover of the outermost steel at the (bottom|top)", "{face}面最外侧钢筋保护层", "{face} outer cover"),
    (r"side cover of the (bottom|top) bars", "{face}排钢筋侧面保护层", "{face} bars, side cover"),
    (r"clear spacing of the (bottom|top) bars at depth (\S+)", "{face}排钢筋净距", "{face} bars, clear spacing"),
    (r"clear distance of the (bottom|top) layers at depths (\S+)–(\S+)", "{face}排钢筋层间净距",
     "{face} layers, clear distance"),
    (r"layers of the welded frame \((bottom|top)\)", "{face}排焊接骨架层数", "{face} welded frame, layers"),
    (r"bars of the welded frame \((bottom|top)\)", "{face}排焊接骨架钢筋直径", "{face} welded frame, bar size"),
    (r"cover of the outermost steel at the sides", "侧面最外侧钢筋保护层", "side outer cover"),
    (r"cover of the ducts \(top, bottom\)", "管道保护层", "duct cover"),
    (r"side cover of the ducts", "管道侧面保护层", "duct side cover"),
    (r"cover of the pretensioned tendons \(top, bottom\)", "预应力筋保护层", "tendon cover"),
    (r"side cover of the pretensioned tendons", "预应力筋侧面保护层", "tendon side cover"),
    (r"side bars of the webs, As/\(b h\)", "腹板侧面钢筋配筋率", "web side bars, ratio"),
    (r"spacing of the side bars, (tension|compression) zone", "腹板侧面钢筋间距（{zone}区）",
     "web side bars, spacing ({zone} zone)"),
    (r"cover of the longitudinal bars", "纵筋保护层", "bar cover"),
    (r"cover of the outermost steel", "最外侧钢筋保护层", "outer cover"),
]


_MEMBER_ZH = {"superstructure": "上部结构", "substructure": "下部结构", "foundation": "基础"}
_FACE_ZH = {"bottom": "底面", "top": "顶面", None: "", "": ""}
# other warnings of the checks → zh (the text kept as it is when none matches)
_WARN_ZH = [
    (r"C(\d+) below C(\d+) for environment (\S+), (\w+), (\d+) years \(table 4\.5\.3\)",
     lambda m: f"混凝土 C{m[1]} 低于 C{m[2]}（环境类别 {m[3]}、{_MEMBER_ZH.get(m[4], m[4])}、"
               f"设计使用年限 {m[5]} 年，表 4.5.3）"),
    (r"C(\d+) with (\S+): at least C30 for bars of 400 MPa and above \(3\.1\.2\)",
     lambda m: f"C{m[1]} 配 {m[2]}：400 MPa 级及以上钢筋要求不低于 C30（3.1.2）"),
    (r"prestressed concrete in C(\d+): at least C40 \(3\.1\.2\)", lambda m: f"预应力混凝土 C{m[1]}：不低于 C40（3.1.2）"),
    (r"cover of the main bars(?: at the (bottom|top))? (\S+) mm > 50 mm.*\(9\.1\.2, should\)",
     lambda m: f"{_FACE_ZH[m[1]]}主筋保护层 {m[2]} mm > 50 mm，宜在保护层内设钢筋网（9.1.2）"),
    (r"bars or bundles of (\S+) mm > 36 mm in the tension zone.*",
     lambda m: f"受拉区钢筋或束筋 {m[1]} mm > 36 mm，宜设表层钢筋网（9.1.3）"),
    (r"(\d+) bars of (\S+) mm in a bundle: at most (\d+) \(9\.1\.3\)",
     lambda m: f"{m[1]} 根 {m[2]} mm 钢筋成束：至多 {m[3]} 根（9.1.3）"),
    (r"bars of a bundle (\S+) mm > 36 mm \(9\.1\.3\)", lambda m: f"束筋单根直径 {m[1]} mm > 36 mm（9.1.3）"),
    (r"tension reinforcement ratio (\S+) < minimum (\S+) \(9\.1\.12\)",
     lambda m: f"受拉钢筋配筋率 {m[1]} < 最小配筋率 {m[2]}（9.1.12）"),
    (r"Mud / Mcr = (\S+) < 1\.0 \(9\.1\.13\)", lambda m: f"Mud / Mcr = {m[1]} < 1.0（9.1.13）"),
    (r"side bars at depth (\S+): (\S+) mm < 6 mm \(9\.3\.7\)",
     lambda m: f"深 {m[1]} 处侧面钢筋 {m[2]} mm < 6 mm（9.3.7）"),
]


def _warning(w: str, lang: str) -> str:
    """a warning of a check in the language (zh: the common ones translated, the others as they are)"""
    if lang == "zh":
        for pat, f in _WARN_ZH:
            m = re.fullmatch(pat, w)
            if m:
                return f(m)
    return w


def _name(chk, lang):
    """the check's name in the language"""
    for k, zh, en in _NAMES:
        if chk.name.startswith(k):
            return zh if lang == "zh" else en
    return chk.name


def _clause(s: str, lang: str) -> str:
    """'9.1.1 item 1' → '9.1.1 第 1 款', 'table 9.1.1' → '表 9.1.1', ', should' → '，宜'"""
    if lang != "zh":
        return s
    s = re.sub(r"table (\S+)", r"表 \1", s)
    s = re.sub(r"(\S+) item (\d+)", r"\1 第 \2 款", s)
    return s.replace(", should", "，宜").replace(", ", "、")


def _clauses(msgs) -> list[str]:
    """the clauses quoted at the end of messages, once each"""
    out = []
    for m in msgs:
        r = re.search(r"\(([^()]*)\)\s*$", m)
        if r:
            c = r.group(1).replace(", should", "")
            if c not in out:
                out.append(c)
    return out


def _mm(v: float) -> str:
    """a length for the text: whole millimetres, or one decimal"""
    return number(v, 0 if abs(v - round(v)) < 0.05 else 1)


def _dim(v: float) -> str:
    """a dimension: whole millimetres"""
    return number(v, 0)


# ── outline ─────────────────────────────────────────────────────────────────
class _Shape:
    """the section drawn symmetric about x = 0, depth z down from the top (drawn at world y = −z): the outer intervals
    (za, zb, x0, x1, class) and the holes (za, zb, x0, x1, key) of the stacked rectangles; circles (zc, r, sign)"""

    def __init__(self, sec: RCSection, webs: int | None = None, topping: bool = False):
        self.h = float(sec.h)
        rects = [p for p in sec.pieces if isinstance(p, Rect)]
        self.circles = [(float(p.zc), p.D / 2, p.sign) for p in sec.pieces if isinstance(p, Circle)]
        self.outer, self.holes = [], []
        self.topping = None
        body = rects
        if topping and rects and rects[0] is sec.pieces[0] and rects[0].sign > 0 and rects[0].top <= _EPS:
            t = rects[0]
            self.topping = (t.top, t.bottom)
            self.outer.append((float(t.top), float(t.bottom), -t.b / 2, t.b / 2, "topping"))
            body = rects[1:]
        self.webs = int(webs) if webs else (2 if sec.kind == "box" else 1)
        self.cells = False
        if body:
            self._stack(body)
        xs = [v for iv in self.outer for v in iv[2:4]] + [s * r for _, r, sg in self.circles if sg > 0 for s in (-1, 1)]
        self.xmin, self.xmax = (min(xs), max(xs)) if xs else (-0.5, 0.5)

    def _stack(self, rects):
        zs = sorted({0.0, self.h} | {float(p.top) for p in rects} | {float(p.bottom) for p in rects})
        iv = []
        for a, b in zip(zs[:-1], zs[1:]):
            if b - a <= _EPS * max(self.h, 1.0):
                continue
            m = (a + b) / 2
            P = sum(p.b for p in rects if p.sign > 0 and p.top < m < p.bottom)
            N = sum(p.b for p in rects if p.sign < 0 and p.top < m < p.bottom)
            iv.append([a, b, P, N])
        n = self.webs
        cell = [False] * len(iv)
        if n >= 2:
            for i, (a, b, P, N) in enumerate(iv):
                up = [q[2] for q in iv[:i]]
                dn = [q[2] for q in iv[i + 1:]]
                cell[i] = N == 0 and P > 0 and bool(up) and bool(dn) and P < max(up) - _EPS and P < max(dn) - _EPS
        for i, (a, b, P, N) in enumerate(iv):
            if P <= 0:
                continue
            if cell[i]:
                j = i - 1
                while j >= 0 and cell[j]:
                    j -= 1
                k = i + 1
                while k < len(iv) and cell[k]:
                    k += 1
                Wo = min(iv[j][2], iv[k][2])
                hole = (Wo - P) / (n - 1)
                if hole > _EPS:
                    self.cells = True
                    tw = P / n
                    self.outer.append((a, b, -Wo / 2, Wo / 2, "concrete"))
                    for q in range(n - 1):
                        x0 = -Wo / 2 + (q + 1) * tw + q * hole
                        self.holes.append((a, b, x0, x0 + hole, f"cell{q}"))
                    continue
            self.outer.append((a, b, -P / 2, P / 2, "concrete"))
            if N > 0:
                self.holes.append((a, b, -N / 2, N / 2, "hole"))

    # rings: outer counter-clockwise, holes clockwise (world, y up)
    @staticmethod
    def _runs(ivs):
        """contiguous runs of intervals with the same key: [[(za, zb, x0, x1), …], …]"""
        out, by = [], {}
        for za, zb, x0, x1, key in sorted(ivs, key=lambda v: (str(v[4]), v[0])):
            by.setdefault(key, []).append((za, zb, x0, x1))
        for key, lst in by.items():
            run = [lst[0]]
            for v in lst[1:]:
                if abs(v[0] - run[-1][1]) < 1e-9 * max(abs(v[0]), 1.0) and v[2] < run[-1][3] and v[3] > run[-1][2]:
                    run.append(v)
                else:
                    out.append((key, run))
                    run = [v]
            out.append((key, run))
        return out

    @staticmethod
    def _trace(run):
        right, left = [], []
        for za, zb, x0, x1 in run:
            right += [(x1, -za), (x1, -zb)]
            left += [(x0, -za), (x0, -zb)]
        ring = right + left[::-1]  # clockwise (y up)
        out = []
        for p in ring:
            if not out or abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
                out.append(p)
        if len(out) > 1 and out[0] == out[-1]:
            out.pop()
        # drop points on a straight line
        k, res = len(out), []
        for i in range(k):
            a, b, c = out[i - 1], out[i], out[(i + 1) % k]
            if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9 * max(1.0, abs(b[0]), abs(b[1])):
                res.append(b)
        return res

    def regions(self):
        """[(class, [rings])]: each outer ring with the holes inside it (concrete class), the topping alone"""
        out = []
        holes = [self._trace(r) for _, r in self._runs(self.holes)]  # clockwise
        for cls, run in self._runs(self.outer):
            ring = self._trace(run)[::-1]  # counter-clockwise
            inside = []
            if cls != "topping":
                y0, y1 = -run[-1][1], -run[0][0]
                inside = [h for h in holes if min(p[1] for p in h) >= y0 - 1e-9 and max(p[1] for p in h) <= y1 + 1e-9]
            out.append((cls, [ring] + inside))
        return out

    def circle_rings(self, n: int = 720):
        """the circles as polygons (for cutting): [(rings)] with outer counter-clockwise, holes clockwise"""
        th = 2 * math.pi * np.arange(n) / n
        rings = []
        for zc, r, sg in sorted(self.circles, key=lambda c: -c[2]):
            pts = [(r * math.cos(t), -zc + r * math.sin(t)) for t in th]
            rings.append(pts if sg > 0 else pts[::-1])
        return rings

    # widths
    def _segs_at(self, z):
        segs = [(x0, x1) for za, zb, x0, x1, _ in self.outer if za < z < zb]
        cuts = [(x0, x1) for za, zb, x0, x1, _ in self.holes if za < z < zb]
        for zc, r, sg in self.circles:
            u = r * r - (z - zc) ** 2
            if u > 0:
                (segs if sg > 0 else cuts).append((-math.sqrt(u), math.sqrt(u)))
        out = []
        for a, b in segs:
            parts = [(a, b)]
            for c0, c1 in cuts:
                nxt = []
                for p0, p1 in parts:
                    if c1 <= p0 or c0 >= p1:
                        nxt.append((p0, p1))
                        continue
                    if c0 > p0:
                        nxt.append((p0, c0))
                    if c1 < p1:
                        nxt.append((c1, p1))
                parts = nxt
            out += [p for p in parts if p[1] - p[0] > _EPS]
        return sorted(out)

    def segments(self, z: float):
        """the solid x-intervals at depth z; on a boundary between pieces the narrower side (a row on a flange's face
        takes the web, as detailing._width_at)"""
        d = 1e-7 * max(self.h, 1.0)
        z = min(max(z, d), self.h - d)
        a, b = self._segs_at(z - d), self._segs_at(z + d)
        wa, wb = sum(q - p for p, q in a), sum(q - p for p, q in b)
        if wa > 0 and (wb <= 0 or wa <= wb):
            return a
        return b

    def holes_at(self, z: float):
        """the cells (holes) at depth z: [(x0, x1)]"""
        out = [(x0, x1) for za, zb, x0, x1, _ in self.holes if za < z < zb]
        for zc, r, sg in self.circles:
            u = r * r - (z - zc) ** 2
            if sg < 0 and u > 0:
                out.append((-math.sqrt(u), math.sqrt(u)))
        return sorted(out)

    def left(self, z: float) -> float:
        """the leftmost concrete at depth z (a corner on a boundary)"""
        xs = [x0 for za, zb, x0, x1, _ in self.outer if za - 1e-9 <= z <= zb + 1e-9]
        for zc, r, sg in self.circles:
            u = r * r - (z - zc) ** 2
            if sg > 0 and u >= -1e-9 * r * r:
                xs.append(-math.sqrt(max(u, 0.0)))
        return min(xs) if xs else self.xmin

    def right(self, z: float) -> float:
        """the rightmost concrete at depth z"""
        xs = [x1 for za, zb, x0, x1, _ in self.outer if za - 1e-9 <= z <= zb + 1e-9]
        for zc, r, sg in self.circles:
            u = r * r - (z - zc) ** 2
            if sg > 0 and u >= -1e-9 * r * r:
                xs.append(math.sqrt(max(u, 0.0)))
        return max(xs) if xs else self.xmax

    def boundaries(self):
        """depths where the outline or a hole changes width"""
        zs = {0.0, self.h}
        for za, zb, *_ in self.outer + self.holes:
            zs |= {za, zb}
        return sorted(zs)

    def all_rings(self):
        """every ring of the concrete (outer counter-clockwise, holes clockwise), circles as polygons"""
        rings = [r for _, rs in self.regions() for r in rs]
        return rings + (self.circle_rings() if self.circles else [])

    def band(self, z0: float, z1: float):
        """the concrete between depths z0 < z1, as rings (Sutherland–Hodgman against y ≥ −z1 and y ≤ −z0)"""
        out = []
        for ring in self.all_rings():
            r = _clip(ring, -z1, lambda p, c: p[1] >= c)
            r = _clip(r, -z0, lambda p, c: p[1] <= c)
            if len(r) >= 3 and abs(_area(r)) > 1e-9:
                out.append(r)
        return out


def _clip(ring, c, inside):
    """Sutherland–Hodgman against a horizontal line y = c"""
    if not ring:
        return []
    out = []
    for i in range(len(ring)):
        p, q = ring[i - 1], ring[i]
        ip, iq = inside(p, c), inside(q, c)
        if iq:
            if not ip:
                out.append(_cut(p, q, c))
            out.append(q)
        elif ip:
            out.append(_cut(p, q, c))
    return out


def _cut(p, q, c):
    """where p → q crosses y = c"""
    t = (c - p[1]) / (q[1] - p[1])
    return (p[0] + t * (q[0] - p[0]), c)


def _area(ring) -> float:
    """signed area (shoelace), counter-clockwise +"""
    a = np.asarray(ring, float)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# ── reinforcement layout ────────────────────────────────────────────────────
def _alloc(k: int, weights) -> list[int]:
    """k items over weights by the largest remainder"""
    w = np.asarray(weights, float)
    if k <= 0 or not len(w):
        return [0] * len(w)
    if w.sum() <= 0:
        w = np.ones(len(w))
    q = k * w / w.sum()
    n = np.floor(q + 1e-9).astype(int)
    rest = k - n.sum()
    for i in np.argsort(-(q - n), kind="stable")[:rest]:
        n[i] += 1
    return [int(v) for v in n]


def _default_side(sec: RCSection, ds: float) -> float:
    """cover of the outermost steel on the sides as section_detailing takes it without side_cover: the bottom cover of
    the outermost steel (the bottom row's clear cover less ds), else the top one; 30 mm without bar diameters"""
    bot, top = sec.edge_masks()
    z, d = sec.bars[:, 1], sec.bar_d
    for mask, face in ((bot, "bottom"), (top, "top")):
        m = mask & ~np.isnan(d)
        if m.any():
            zz = z[m].max() if face == "bottom" else z[m].min()
            k = m & np.isclose(z, zz, atol=1e-6)
            dd = float(d[k].max())
            c = (sec.h - zz if face == "bottom" else zz) - dd / 2
            return c - ds
    return 30.0


def _even(x0, x1, off, k):
    """k positions evenly from x0 + off to x1 − off (one in the middle)"""
    if k <= 0:
        return []
    if k == 1:
        return [(x0 + x1) / 2]
    return list(np.linspace(x0 + off, x1 - off, k))


def _key(z):
    return round(float(z), 6)


def _lookup(table, z):
    """the entry of {depth: …} at depth z"""
    if not table:
        return None
    for k, v in table.items():
        if abs(float(k) - z) < 1e-6 * max(1.0, abs(z)) + 1e-6:
            return list(v)
    return None


def bar_layout(sec: RCSection, *, side_cover: float | None = None, stirrup_d: float | None = None,
               webs: int | None = None, bar_x=None, topping: bool | None = None, main_rows=None) -> list[dict]:
    """the bars of sec as drawn by section_figure: one dict per row (bars at one depth), with z (mm), face ("bottom",
    "top": the edge rows of RCSection.edge_masks; "side": other rows in the web; "inner": the others; "ring"), idx
    (the bar points of sec.bars), groups [(n, d)], n, d (the largest diameter), area, bars [(x, z, d)] (x from the
    centre line), layers (1, or more where a lumped row is drawn in layers), exact (whole numbers of bars), lumped
    (points without a diameter, drawn as one disc of their area). main_rows: the main rows from each face, as
    section_detailing takes them (several layers of a welded frame). Conventions in the module docstring."""
    shape = _Shape(sec, webs, sec.kind == "composite" if topping is None else topping)
    return _bar_rows(sec, shape, side_cover, stirrup_d, bar_x, main_rows)


def _bar_rows(sec, shape, side_cover, stirrup_d, bar_x, main_rows=None):
    """bar_layout on a _Shape"""
    from .detailing import _main_masks
    ds = float(stirrup_d or 0.0)
    side = _default_side(sec, ds) if side_cover is None else float(side_cover)
    bot, top = _main_masks(sec, main_rows)
    rows = []
    ring = _ring_layout(sec)
    web = _web_zone(shape)
    by = {}
    for i, (a, z) in enumerate(sec.bars):
        by.setdefault(_key(z), []).append(i)
    for zk in sorted(by):
        idx = by[zk]
        z = float(sec.bars[idx[0], 1])
        face = ("bottom" if bot[idx].any() else "top" if top[idx].any() else
                "side" if web[0] - 1e-9 <= z <= web[1] + 1e-9 else "inner")
        groups, dias, exact = [], [], True
        for i in idx:
            a, d = sec.bars[i, 0], sec.bar_d[i]
            if np.isnan(d):
                deq = math.sqrt(4 * a / math.pi)
                groups.append((1, deq, a))
                dias.append(deq)
                exact = False
                continue
            c = a / (math.pi * d * d / 4)
            n = max(int(round(c)), 1)
            exact &= abs(c - n) < 1e-3 * max(c, 1.0)
            groups.append((n, float(d), a))
            dias += [float(d)] * n
        row = {"z": z, "face": face, "idx": idx, "groups": [(n, d) for n, d, _ in groups], "n": len(dias),
               "d": max(dias), "area": float(sec.bars[idx, 0].sum()), "exact": exact, "layers": 1,
               "lumped": any(np.isnan(sec.bar_d[i]) for i in idx)}
        if ring is not None:
            row["face"] = "ring"
            row["bars"] = [(ring[i][0], ring[i][1], float(sec.bar_d[i])) for i in idx]
            rows.append(row)
            continue
        dias.sort(reverse=True)
        dmax = dias[0]
        given = _lookup(bar_x, z)
        segs = shape.segments(z) or [(-0.5 * sec.web(), 0.5 * sec.web())]
        off = side + ds + dmax / 2
        if given is not None:
            if len(given) != len(dias):
                raise ValueError(f"bar_x at depth {_mm(z)}: {len(given)} positions for {len(dias)} bars")
            xs = [(float(x), z) for x in given]
        else:
            xs = _spread(segs, len(dias), off, dmax, z, row, side, ds, room=min(z, sec.h - z))
        # the largest bars to the outermost places (nearest a face of their part of the concrete)
        order = sorted(range(len(xs)), key=lambda k: min(min(xs[k][0] - s0, s1 - xs[k][0]) for s0, s1 in segs))
        bars = [None] * len(xs)
        for k, d in zip(order, dias):
            bars[k] = (xs[k][0], xs[k][1], d)
        row["bars"] = bars
        rows.append(row)
    return rows


def _web_zone(shape):
    """(top, bottom) of the narrowest part of the outline below a topping (the web of a T, I or box), the whole depth
    when the width does not change"""
    bnd = [z for z in shape.boundaries() if not shape.topping or z >= shape.topping[1] - 1e-9]
    w = [(sum(b - a for a, b in shape.segments((za + zb) / 2)), za, zb) for za, zb in zip(bnd[:-1], bnd[1:])]
    if not w:
        return 0.0, shape.h
    wmin = min(v[0] for v in w)
    k = [i for i, v in enumerate(w) if v[0] <= 1.01 * wmin]
    return w[k[0]][1], w[k[-1]][2]


def _spread(segs, k, off, d, z, row, side, ds, room=math.inf):
    """k bars of diameter d over the parts segs of the concrete at depth z, the outermost centres off from the faces:
    [(x, z)]; layers (row["layers"]) when one cannot hold them, about z at the 9.3.3 clear distance, closer (down to
    touching) where room (the distance from z to the nearer face) is short"""
    usable = [max(s1 - s0 - 2 * off, 0.0) for s0, s1 in segs]
    counts = _alloc(k, usable if sum(usable) > 0 else [s1 - s0 for s0, s1 in segs])
    # one layer as the program takes it, even when the bars touch or overlap a little (section_detailing reports
    # the clear spacing); layers only when they cannot lie in one row (a lumped point standing for several layers)
    fits = all(c < 2 or (s1 - s0 - 2 * off >= 0 and (s1 - s0 - 2 * off - c * d) / (c - 1) >= -0.5 * d)
               for c, (s0, s1) in zip(counts, segs))
    if fits:
        return [(x, z) for c, (s0, s1) in zip(counts, segs) for x in _even(s0, s1, off, c)]
    s_min = max(30.0, d)
    out, layers = [], 1
    for c, (s0, s1) in zip(counts, segs):
        if c == 0:
            continue
        o = off if s1 - s0 - 2 * off >= d else min(30.0 + ds + d / 2, (s1 - s0) / 2)
        L = max(s1 - s0 - 2 * o, 0.0)
        per = max(1, int(L // (d + s_min)) + 1)
        nl = math.ceil(c / per)
        layers = max(layers, nl)
        split = _alloc(c, [1] * nl)
        pitch = d + s_min if nl < 2 else max(d, min(d + s_min, 2 * (room - d / 2 - 10) / (nl - 1)))
        for j, cj in enumerate(split):
            zj = z + (j - (nl - 1) / 2) * pitch
            out += [(x, zj) for x in _even(s0, s1, o, cj)]
    row["layers"] = layers
    return out


def _ring_layout(sec: RCSection):
    """bar i → (x, z) on the bar circle of a circular section when the bars are those of jtg.circle (n evenly at
    2π(k + ½)/n from the top); None otherwise"""
    if sec.ring is None or sec.kind not in ("circle", "annulus") or not len(sec.bars):
        return None
    n, r, rs = len(sec.bars), sec.h / 2, sec.ring
    th = 2 * math.pi * (np.arange(n) + 0.5) / n
    zk = r - rs * np.cos(th)
    free = list(range(n))
    out = {}
    for i, z in enumerate(sec.bars[:, 1]):
        hit = [k for k in free if abs(zk[k] - z) < 1e-6 * sec.h]
        if not hit:
            return None
        k = hit[0]
        free.remove(k)
        out[i] = (rs * math.sin(th[k]), float(z))
    return out


def _tendon_rows(sec, shape, side, ds, tendon_x, strand):
    """[{z, idx, ducts [(x, z, D, dp)], strands [(x, z, d)], label parts}] for the bonded tendons"""
    out = []
    by = {}
    for i, row in enumerate(sec.tendons):
        by.setdefault(_key(row[1]), []).append(i)
    s_d, s_a = strand
    for zk in sorted(by):
        idx = by[zk]
        z = float(sec.tendons[idx[0], 1])
        segs = shape.segments(z) or [(-0.5, 0.5)]
        g = {"z": z, "idx": idx, "ducts": [], "strands": []}
        given = _lookup(tendon_x, z)
        if sec.pretensioned:
            Ap = float(sec.tendons[idx, 0].sum())
            c = Ap / s_a
            if abs(c - round(c)) < 0.03 * c and round(c) >= 1:
                k = int(round(c))
                xs = [(x, z) for x in given] if given is not None and len(given) == k else \
                    _spread(segs, k, side + ds + s_d / 2, s_d, z, {}, side, ds)
                g["strands"] = [(x, zz, s_d) for x, zz in xs]
                g["n_strands"] = k
            else:
                g["strands"] = [(0.5 * (segs[0][0] + segs[-1][1]), z, math.sqrt(4 * Ap / math.pi))]
            out.append(g)
            continue
        m = len(idx)
        if given is not None:
            if len(given) != m:
                raise ValueError(f"tendon_x at depth {_mm(z)}: {len(given)} positions for {m} tendon rows")
            xs = [float(x) for x in given]
        else:
            counts = _alloc(m, [s1 - s0 for s0, s1 in segs])
            xs = []
            for c, (s0, s1) in zip(counts, segs):
                xs += [s0 + (j + 1) * (s1 - s0) / (c + 1) for j in range(c)]
        for x, i in zip(xs, idx):
            Ap, duct = sec.tendons[i, 0], sec.tendons[i, 3]
            g["ducts"].append((x, z, math.sqrt(4 * duct / math.pi) if duct > 0 else 0.0, math.sqrt(4 * Ap / math.pi)))
        out.append(g)
    return out


def _external_rows(sec, shape, external_n):
    """[{z, idx, n_each, items [(x, z, D, row)]}]: each external row as n tendons of equal area, in the cells at their
    depth (next to the webs) or beside the section"""
    out = []
    by = {}
    for i, row in enumerate(sec.external):
        by.setdefault(_key(row[1]), []).append(i)
    for zk in sorted(by):
        idx = by[zk]
        z = float(sec.external[idx[0], 1])
        n_each = [int(external_n.get(i, 2) if isinstance(external_n, dict) else (external_n or 2)) for i in idx]
        items = [(sec.external[i, 0] / k, i) for i, k in zip(idx, n_each) for _ in range(k)]
        D = max(math.sqrt(4 * a / math.pi) for a, _ in items)
        cells = shape.holes_at(z)
        pts = []
        if cells:
            counts = _alloc(len(items), [c1 - c0 for c0, c1 in cells])
            for c, (c0, c1) in zip(counts, cells):
                g = max(0.5 * D, 50.0)
                pts += _even(c0, c1, g + D / 2, c) if c > 1 else ([(c0 + c1) / 2] if c else [])
        else:
            L, R = shape.left(z), shape.right(z)
            g = max(0.5 * D, 50.0)
            for j in range(len(items)):
                k = j // 2
                pts.append(R + g + D / 2 + k * (D + g) if j % 2 == 0 else L - g - D / 2 - k * (D + g))
        out.append({"z": z, "idx": idx, "n_each": n_each,
                    "items": [(x, z, math.sqrt(4 * a / math.pi), i) for x, (a, i) in zip(sorted(pts), items)]})
    return out


# ── the figure ──────────────────────────────────────────────────────────────
def section_figure(sec, *, check=None, lang: str = "zh", width: float = 560, title: str | None = None,
                   hogging: bool | None = None, stirrup_d: float | None = None, side_cover: float | None = None,
                   webs: int | None = None, bar_x=None, tendon_x=None, external_n=None,
                   strand: tuple[float, float] = (15.2, 140.0), dims: bool = True, labels: bool = True,
                   legend: bool = True, info: bool = True, main_rows=None) -> Figure:
    """the section with its reinforcement, to scale, as a femotion.viz Figure (width px; the height follows).

    sec: an RCSection (jtg.rect, tee, box, circle, …) or a Composite (jtg.composite: drawn as its ultimate section, the
    topping in its own colour). check: a Check of flexure, eccentric, tension, service_stresses, crack_resistance or
    section_detailing, the dict of cracked_section, any other Check (a line of text), or a list of them. lang "zh" or
    "en". hogging: True puts the compression face of eccentric checks and of cracked_section at the bottom (by default
    from Check.hogging: the check was made on the section turned over; flexure, tension and the stresses always follow
    the check). stirrup_d: draws the stirrups (diameter mm) and takes them into the bar layout as
    section_detailing does; side_cover: cover of the outermost steel on the sides (mm, default as section_detailing);
    webs: number of webs of a box (default 2 for kind "box"); bar_x / tendon_x: {depth: [x, …]} positions (mm from the
    centre line) of a bar row / of the ducts at a depth; external_n: tendons drawn per external row (default 2, or
    {row: n}); strand: (nominal diameter, area) of a strand, for counts of strands in the labels and for pretensioned
    rows. dims, labels, legend, info: switch the dimensions, the labels of the steel, the legend and the text block.
    main_rows: the main rows from each face, as section_detailing takes them (default from a section_detailing
    check, else the edge rows)."""
    if lang not in _S:
        raise ValueError("lang: 'zh' or 'en'")
    comp = sec if isinstance(sec, Composite) else None
    s = comp.ultimate if comp is not None else sec
    if not isinstance(s, RCSection):
        raise TypeError("section_figure draws an RCSection or a Composite")
    checks = [] if check is None else (list(check) if isinstance(check, (list, tuple)) else [check])
    if webs is None and comp is not None and comp.precast.kind == "box":
        webs = 2
    if main_rows is None:
        main_rows = next((c.details["main_rows"] for c in checks if isinstance(c, Check) and "main_rows" in c.details),
                         None)
    d = _Drawing(s, comp is not None or s.kind == "composite", lang, width, stirrup_d, side_cover, webs, bar_x, tendon_x,
                 external_n, strand, hogging, main_rows)
    for c in checks:
        d.overlay(c)
    return d.figure(title, dims=dims, labels=labels, legend=legend, info=info)


class _Drawing:
    """one section drawing: the layout of the steel, the overlays of the checks, then the SVG (_geometry in world
    coordinates, _annotations on the page)"""

    def __init__(self, sec, topping, lang, width, stirrup_d, side_cover, webs, bar_x, tendon_x, external_n, strand,
                 hogging, main_rows=None):
        self.sec, self.L, self.lang, self.width = sec, _S[lang], lang, float(width)
        self.hogging = hogging
        self.shape = _Shape(sec, webs, topping)
        self.ds = float(stirrup_d or 0.0)
        self.stirrup = stirrup_d is not None and stirrup_d > 0
        self.side = _default_side(sec, self.ds) if side_cover is None else float(side_cover)
        self.rows = _bar_rows(sec, self.shape, side_cover, stirrup_d, bar_x, main_rows)
        self.tendons = _tendon_rows(sec, self.shape, self.side, self.ds, tendon_x, strand) if sec.has_internal else []
        self.ext = _external_rows(sec, self.shape, external_n) if sec.has_external else []
        self.strand = strand
        # overlays
        self.zone = None  # (z0, z1): the compression zone between these depths
        self.lines = []  # (z, cls, label, data-role, colour or None): lines across the section
        self.vdims = []  # (z0, z1, label): dimensions of the analysis (x, h0, e)
        self.load = None  # (z, label): the axial force, on the centre line
        self.stress = None  # {"prof": [(z, σ)], "lims": [(σ, label)], "cracked": (edge, x) or None, …}
        self.warn = []  # (kind, payload, number or None, unused, text): what section_detailing rings
        self.info = []  # the text block under the drawing: femotion.viz.Note
        self.notes = []
        for r in self.rows:
            if r["layers"] > 1:
                g = "+".join(f"{n}{self.L['dia']}{_mm(dd)}" for n, dd in r["groups"])
                self.notes.append("* " + self.L["layers"].format(row=g, n=r["layers"]))
                r["star"] = True

    # ── overlays ────────────────────────────────────────────────────────────
    def overlay(self, chk):
        if isinstance(chk, dict) and "x0" in chk and "Icr" in chk:
            return self._cracked_dict(chk)
        if not isinstance(chk, Check):
            raise TypeError("check: a Check, cracked_section's dict, or a list of them")
        name = chk.name
        if chk.detailing and name.startswith("section detailing"):
            return self._detailing(chk)
        if name.startswith("flexure"):
            return self._flexure(chk)
        if name.startswith("eccentric compression"):
            return self._eccentric(chk)
        if name.startswith("tension"):
            return self._tension(chk)
        if name.startswith("concrete compression in service") or name.startswith("crack resistance (normal section"):
            return self._stresses(chk)
        self._generic(chk)

    def _head(self, chk, what: str = ""):
        cl = chk.clause.replace("JTG 3362-2018 ", "")
        L = self.L
        nm = _name(chk, self.lang) + (f"{L['lp']}{what}{L['rp']}" if what else "")
        self.info.append(Note(f"{nm}{L['sep'] if self.lang == 'zh' else ' — '}JTG 3362-2018 {cl}", "head"))

    def _verdict(self, chk, text):
        ok = chk.ok
        self.info.append(Note(f"{text} → {self.L['ok'] if ok else self.L['not']}", "ok" if ok else "fail"))
        self._tails(chk)

    def _tails(self, chk):
        L = self.L
        if chk.failed:
            self.info.append(Note(L["failed"].format(n=len(chk.failed)) if self.lang == "zh" else
                                  "not met: " + "; ".join(chk.failed), "small warn"))
        if chk.warnings:
            cl = _clauses(chk.warnings)
            self.info.append(Note(L["warnings"] + "、".join(_clause(c, self.lang) for c in cl) if self.lang == "zh" else
                                  L["warnings"] + ", ".join(cl), "small"))

    def _face(self, comp_bottom, z):
        """depth on the drawing of a depth measured from the compression face"""
        return self.sec.h - z if comp_bottom else z

    def _zone(self, x, comp_bottom):
        h = self.sec.h
        x = min(max(x, 0.0), h)
        self.zone = (h - x, h) if comp_bottom else (0.0, x)

    def _flexure(self, chk):
        L, d, sec = self.L, chk.details, self.sec
        hog = chk.hogging
        x, h0, xi, xib = d["x"], d["h0"], d["xi"], d["xi_b"]
        beta = sec.concrete.beta
        xc = x / beta
        self._zone(x, hog)
        if 0 < xc < sec.h:
            self.lines.append((self._face(hog, xc), "na", f"{L['na_s']} xc = {_mm(xc)}", "neutral-axis", None))
        over = x > xib * h0 + 1e-9
        if xib * h0 < sec.h:
            self.lines.append((self._face(hog, xib * h0), "xb", f"ξb h0 = {_mm(xib * h0)}", "xi-b",
                               C["warn"] if over else None))
        if x < sec.h:
            self.vdims.append((self._face(hog, 0.0), self._face(hog, x), f"x = {_dim(x)}"))
        self.vdims.append((self._face(hog, 0.0), self._face(hog, h0), f"h0 = {_dim(h0)}"))
        self._head(chk, L["hog"] if hog else L["sag"])
        sep = L["sep"]
        self.info.append(Note(f"x = {_mm(x)} mm{sep}xc = x/β = {_mm(xc)} mm{sep}h0 = {_mm(h0)} mm"))
        rel = ">" if over else "≤"
        self.info.append(Note(f"ξ = x/h0 = {xi:.3f} {rel} ξb = {xib:.3f}" + (f"{L['lp']}{L['over']}{L['rp']}" if over
                                                                            else ""), "warn" if over else ""))
        self._verdict(chk, f"Mu = {chk.resistance:.1f} kN·m{sep}γ0Md = {abs(chk.demand):.1f} kN·m{sep}"
                           f"{L['ratio']} {chk.ratio:.3f}")

    def _eccentric(self, chk):
        L, d, sec = self.L, chk.details, self.sec
        far = any("far face is the more compressed" in n for n in chk.notes)
        bottom = bool(chk.hogging if self.hogging is None else self.hogging) != far
        h = sec.h
        sep = L["sep"]
        if "x" in d:
            self._zone(d["x"], bottom)
            xc = d["x"] / sec.concrete.beta
            if 0 < xc < h:
                self.lines.append((self._face(bottom, xc), "na", f"{L['na_s']} xc = {_mm(xc)}", "neutral-axis", None))
            if d["x"] < h:
                self.vdims.append((self._face(bottom, 0.0), self._face(bottom, d["x"]), f"x = {_dim(d['x'])}"))
        elif "alpha" in d and sec.kind == "circle":
            r = h / 2
            self._zone(r * (1 - math.cos(math.pi * d["alpha"])), bottom)
        e = d["e"]
        zN = sec.zg + (e if bottom else -e)
        if -0.6 * h <= zN <= 1.6 * h:
            self.load = (zN, f"γ0Nd{L['lp']}e = {_mm(e)}{L['rp']}" if self.lang == "zh" else f"γ0Nd (e = {_mm(e)})")
            self.vdims.append((sec.zg, zN, f"e = {_dim(e)}"))
        self._head(chk)
        self.info.append(Note(f"e0 = {_mm(d['e0'])} mm{sep}η = {d['eta']:.3f}{sep}e = ηe0 = {_mm(e)} mm"))
        if "x" in d:
            case = L["small"] if d.get("case", "").startswith("small") else L["large"]
            self.info.append(Note(f"x = {_mm(d['x'])} mm{sep}ξ = {d['xi']:.3f}{sep}ξb = {d['xi_b']:.3f}"
                                  f"{L['lp']}{case}{L['rp']}"))
        elif "alpha" in d:
            self.info.append(Note(f"α = {d['alpha']:.3f}{L['lp']}5.3.8{L['rp']}"))
        self._verdict(chk, f"Nu = {chk.resistance:.1f} kN{sep}γ0Nd = {chk.demand:.1f} kN{sep}{L['ratio']} {chk.ratio:.3f}")

    def _tension(self, chk):
        L, d, sec = self.L, chk.details, self.sec
        hog = chk.hogging
        sep = L["sep"]
        if "x" in d:
            self._zone(d["x"], not hog)  # the far face (from the load) is compressed
            self.vdims.append((self._face(not hog, 0.0), self._face(not hog, d["x"]), f"x = {_dim(d['x'])}"))
        e0 = d.get("e0", 0.0)
        zN = sec.zg + (-e0 if hog else e0)
        if -0.6 * sec.h <= zN <= 1.6 * sec.h:
            self.load = (zN, f"γ0Nd{L['lp']}e0 = {_mm(e0)}{L['rp']}" if self.lang == "zh" else f"γ0Nd (e0 = {_mm(e0)})")
            if e0 > 0:
                self.vdims.append((sec.zg, zN, f"e0 = {_dim(e0)}"))
        self._head(chk)
        case = d.get("case", "")
        if self.lang == "zh":
            case = {"axial tension": "轴心受拉", "small eccentricity": "小偏心受拉", "large eccentricity": "大偏心受拉",
                    "eccentric tension (circle)": "偏心受拉（圆形截面）"}.get(case, case)
        line = f"e0 = {_mm(e0)} mm{sep}{case}"
        if "x" in d:
            line += f"{sep}x = {_mm(d['x'])} mm ≤ ξb h0 = {_mm(d['xi_b_h0'])} mm" if d["x"] <= d["xi_b_h0"] else \
                f"{sep}x = {_mm(d['x'])} mm > ξb h0 = {_mm(d['xi_b_h0'])} mm"
        self.info.append(Note(line))
        self._verdict(chk, f"Nu = {chk.resistance:.1f} kN{sep}γ0Nd = {chk.demand:.1f} kN{sep}{L['ratio']} {chk.ratio:.3f}")

    def _stresses(self, chk):
        L, d, sec = self.L, chk.details, self.sec
        h, sep = sec.h, L["sep"]
        prof, lims, cracked = None, [], None
        if "edge_stress" in d:  # full and class A in service
            top, bot = d["edge_stress"]["top"], d["edge_stress"]["bottom"]
            prof = [(0.0, top), (h, bot)]
            lims.append((chk.resistance, "0.5fck"))
        elif "Icr" in d and "x" in d:  # class B, cracked section
            x, scc, edge = d["x"], chk.demand, d["edge"]
            if math.isfinite(x):
                cracked = (edge, x)
                prof = [(0.0, scc), (min(x, h), 0.0)] if edge == "top" else [(h - min(x, h), 0.0), (h, scc)]
                if self.zone is None:  # the compressed concrete of the cracked section
                    self._zone(x, edge == "bottom")
                if x < h:
                    self.lines.append((x if edge == "top" else h - x, "na", f"{L['na_s']} x = {_mm(x)}", "neutral-axis",
                                       None))
            else:
                prof = self._b_uncracked(d, scc, edge)
            lims.append((chk.resistance, "0.5fck"))
        elif "top" in d and "bottom" in d and isinstance(d["top"], dict):  # crack resistance
            top = d["top"]["sigma_pc"] - d["top"]["sigma_st"]
            bot = d["bottom"]["sigma_pc"] - d["bottom"]["sigma_st"]
            prof = [(0.0, top), (h, bot)]
            if "6.3.1-3" in chk.clause:
                lims.append((-0.7 * sec.concrete.ftk, "0.7ftk"))
        if prof is None:
            return self._generic(chk)
        if self.stress is None:
            self.stress = {"prof": prof, "lims": lims, "cracked": cracked, "role": chk.name}
        self._head(chk)
        if cracked is not None:
            self.info.append(Note(f"开裂截面受压区高度 x = {_mm(cracked[1])} mm（{L['edge'][cracked[0]]}受压）"
                                  if self.lang == "zh" else
                                  f"cracked section: compression depth x = {_mm(cracked[1])} mm from the {cracked[0]}"))
        if cracked is None and len(prof) == 2 and prof[0][0] == 0.0 and prof[1][0] == h:
            et, eb = L["edge"]["top"], L["edge"]["bottom"]
            self.info.append(Note(f"{et} σ = {number(prof[0][1], 2)} MPa{sep}{eb} σ = {number(prof[1][1], 2)} MPa"
                                  f"{L['lp']}{L['comp_pos']}{L['rp']}"))
        sym = chk.symbol
        lim_sym = {"6.3.1-1": "0.85σpc", "6.3.1-2": "0.80σpc", "6.3.1-3": "σpc + 0.7ftk", "item 1-3": "σpc"}
        ls = next((v for k, v in lim_sym.items() if k in chk.clause), "0.5fck" if "7.1" in chk.clause else L["limit"])
        self._verdict(chk, f"{sym} = {number(chk.demand, 2)} MPa {'≤' if chk.ok else '>'} {ls} = "
                           f"{number(chk.resistance, 2)} MPa")
        if "tendon_stress" in d and len(d["tendon_stress"]) and "tendon_limit" in d:
            sp = float(np.max(d["tendon_stress"]))
            self.info.append(Note(f"σp,max = {sp:.0f} MPa {'≤' if sp <= d['tendon_limit'] + 1e-9 else '>'} "
                                  f"{d['tendon_limit']:.0f} MPa{L['lp']}7.1.5{L['rp']}", "small"))
        if "external_stress" in d:
            se = float(np.max(d["external_stress"]))
            self.info.append(Note(f"σpe,ex = {se:.0f} MPa {'≤' if se <= d['external_limit'] + 1e-9 else '>'} 0.60fpk = "
                                  f"{d['external_limit']:.0f} MPa{L['lp']}7.1.5-3{L['rp']}", "small"))

    def _b_uncracked(self, d, scc, edge):
        """class B with the whole section compressed: the edge stresses from the resultant on the cracked (here the
        whole) section, σ = F/A + F e0N (c − z)/I with F = Np0 (no axial force), the governing edge from the check"""
        F, A, I, c, e = d["Np0"] * 1e3, d["Acr"], d["Icr"], d["c"], d["e0N"]
        h = self.sec.h
        s0, s1 = F / A + F * e * c / I, F / A + F * e * (c - h) / I  # on the turned section's top / bottom
        s_top_first = s0 >= s1
        # the oriented section's governing edge is its top when s0 ≥ s1; map to the drawing with d["edge"]
        flipped = (edge == "top") != s_top_first
        top, bot = (s1, s0) if flipped else (s0, s1)
        if edge == "top":
            top = scc
        else:
            bot = scc
        return [(0.0, top), (h, bot)]

    def _cracked_dict(self, r):
        L, sec = self.L, self.sec
        x0, Icr = r["x0"], r["Icr"]
        bottom = bool(self.hogging)
        self._zone(x0, bottom)
        if x0 < sec.h:
            self.lines.append((self._face(bottom, x0), "na", f"{L['na_s']} x0 = {_mm(x0)}", "neutral-axis", None))
        edge = "bottom" if bottom else "top"
        prof = [(0.0, 1.0), (min(x0, sec.h), 0.0)] if not bottom else [(sec.h - min(x0, sec.h), 0.0), (sec.h, 1.0)]
        if self.stress is None:
            self.stress = {"prof": prof, "lims": [], "cracked": (edge, x0), "role": "cracked section", "schematic": True}
        name = "开裂截面（7.2.4、6.5.2）" if self.lang == "zh" else "cracked section (7.2.4, 6.5.2)"
        self.info.append(Note(name, "head"))
        self.info.append(Note(f"x0 = {_mm(x0)} mm{L['sep']}Icr = {Icr:.4g} mm⁴".replace("e+", "e")))

    def _generic(self, chk):
        L = self.L
        self._head(chk)
        if chk.detailing:
            n = len(chk.warnings)
            zh = self.lang == "zh"
            text = (f"{n} 项不满足" if zh else f"{n} rule(s) not met") if n else ("规定均满足" if zh else "rules met")
            self.info.append(Note(text, "warn" if n else "ok"))
            for w in chk.warnings:
                self.info.append(Note("· " + _warning(w, self.lang), "small warn"))
            return
        if chk.limit_check:
            u = f" {chk.unit}" if chk.unit else ""
            self._verdict(chk, f"{chk.symbol} = {number(chk.demand, 3)}{u}{L['sep']}{L['limit']} "
                               f"{number(chk.resistance, 3)}{u}")
        else:
            self._verdict(chk, f"γ0Sd = {chk.demand:.1f} {chk.unit}{L['sep']}R = {chk.resistance:.1f} {chk.unit}"
                               f"{L['sep']}{L['ratio']} {chk.ratio:.3f}")

    def _detailing(self, chk):
        lang = self.lang
        self._head(chk)
        n0 = len(self.warn)
        loose = []
        for w in chk.warnings:
            hit = self._locate(w, chk)
            if hit is None:
                loose.append(w)
        k = len(self.warn) - n0
        tot = k + len(loose)
        self.info.append(Note((f"{tot} 项不满足" if lang == "zh" else f"{tot} rule(s) not met") if tot else
                              ("规定均满足" if lang == "zh" else "rules met"), "warn" if tot else "ok"))
        for kind, payload, num, short, full in self.warn[n0:]:
            if num is not None:
                self.info.append(Note(full, "small warn", num))
        for w in loose:
            self.info.append(Note("· " + self._rule_text(w, chk), "small warn"))

    def _rule_text(self, w, chk):
        """a warning as a short line: the rule's name in the language, value, limit and clause"""
        lang = self.lang
        r = re.match(r"^(?P<name>.+?) (?P<v>[-+\d.e]+)(?P<u> mm|) (?P<op>[<>]) (?P<lim>[-+\d.e]+)(?: mm)? "
                     r"\((?P<cl>[^()]*)\)$", w)
        if not r:
            return _warning(w, lang)
        nm = r.group("name")
        short = nm
        for pat, zh, en in _RULES:
            mm = re.fullmatch(pat, nm)
            if mm:
                face = mm.group(1) if mm.groups() and mm.group(1) in ("bottom", "top") else ""
                zone = mm.group(1) if mm.groups() and mm.group(1) in ("tension", "compression") else ""
                tpl = zh if lang == "zh" else en
                short = tpl.format(face=self.L["face"].get(face, face),
                                   zone={"tension": "受拉", "compression": "受压"}.get(zone, zone) if lang == "zh" else zone)
                if lang == "en":
                    short = short[:1].upper() + short[1:]
                break
        dz = re.search(r" at depth (\S+)$", nm)
        if dz:
            short += f"{self.L['lp']}{'深' if lang == 'zh' else 'depth '}{_dim(float(dz.group(1)))}{self.L['rp']}"
        v, lim = float(r.group("v")), float(r.group("lim"))
        if "As/(b h)" in nm:
            val = f"{v:.2%} {r.group('op')} {lim:.2%}"
        else:
            val = f"{_mm(v)} {r.group('op')} {_mm(lim)}{r.group('u') or ''}"
        gap = "" if short.endswith("）") else " "
        return f"{short}{gap}{val}{self.L['lp']}{_clause(r.group('cl'), lang)}{self.L['rp']}"

    def _locate(self, w, chk):
        """ring what a section_detailing warning is about; None when it has no place on the section"""
        sec = self.sec
        num = 1 + sum(1 for v in self.warn if v[2] is not None)
        full = self._rule_text(w, chk)
        short = ""
        m = re.match(r"^(side cover of the|cover of the outermost steel at the|cover of the) (bottom|top)( bars)? ", w)
        rows = [r for r in self.rows if r["face"] != "ring"]
        if m and (m.group(3) or "outermost" in m.group(1)):
            face = m.group(2)
            cand = [r for r in rows if r["face"] == face] or rows
            if not cand:
                return None
            r = max(cand, key=lambda r: r["z"]) if face == "bottom" else min(cand, key=lambda r: r["z"])
            self.warn.append(("rows", [r], num, short, full))
            return True
        m = re.match(r"^clear spacing of the (bottom|top) bars at depth (\S+) ", w)
        if m:
            z = float(m.group(2))
            r = [q for q in rows if abs(q["z"] - z) < 1e-6 * max(z, 1.0) + 1e-6]
            if r:
                self.warn.append(("rows", r, num, short, full))
                return True
        m = re.match(r"^clear distance of the (bottom|top) layers at depths (\S+)–(\S+) ", w)
        if m:
            z1, z2 = float(m.group(2)), float(m.group(3))
            r = [q for q in rows if min(abs(q["z"] - z1), abs(q["z"] - z2)) < 1e-6 * max(z2, 1.0) + 1e-6]
            if r:
                self.warn.append(("rows", r, num, short, full))
                return True
        if re.match(r"^(cover of the longitudinal bars|cover of the outermost steel) ", w) and sec.ring is not None:
            self.warn.append(("ring", sec.ring, num, short, full))
            return True
        if w.startswith("cover of the ducts") and self.tendons:
            z, duct = sec.tendons[:, 1], sec.tendons[:, 3]
            near = np.minimum(z, sec.h - z)
            D = np.sqrt(4 * duct / math.pi)
            k = int(np.argmin(near - D))
            for g in self.tendons:
                if k in g["idx"] and g["ducts"]:
                    x, zz, Dk, _ = g["ducts"][g["idx"].index(k)]
                    self.warn.append(("circle", (x, zz, Dk / 2), num, short, full))
                    return True
        if w.startswith("side cover of the ducts") and self.tendons:
            for g in self.tendons:
                for x, zz, Dk, _ in g["ducts"]:
                    self.warn.append(("circle", (x, zz, Dk / 2), num, short, full))
                    num = None
            return True
        if (w.startswith("side bars of the webs") or w.startswith("spacing of the side bars")
                or re.match(r"^side bars at depth", w)):
            side = [r for r in rows if r["face"] == "side"]
            if side:
                self.warn.append(("rows", side, num, short, full))
            else:
                self.warn.append(("web", None, num, short, full))
            return True
        return None

    # ── drawing ─────────────────────────────────────────────────────────────
    def _budget(self, dims, labels):
        """whether the stress diagram goes beside the section: the page width less the dimensions, the label column and
        the diagram must leave the section at least 30 % of it; else the diagram goes under the section"""
        lab = [text_width(t, TEXT_SIZE[c]) for a, t, c, _ in self._label_items() if isinstance(t, str)] if labels else []
        need = (24 + max(lab, default=0.0) + (22 if self.warn else 0)) if labels else 14.0
        if dims:
            need += 12 + 18 * (len(self.vdims) + 2)
        stress = 16 + 100.0 + 40
        return self.width - need - stress >= 0.3 * self.width

    def figure(self, title, *, dims, labels, legend, info):
        self.beside = self.stress is None or self._budget(dims, labels)
        fig = Figure(width=self.width, title=title)
        sh, sec = self.shape, self.sec
        pts = [(sh.xmin, 0.0), (sh.xmax, -sec.h)]
        for e in self.ext:
            for x, z, D, _ in e["items"]:
                pts += [(x - D / 2, -z - D / 2), (x + D / 2, -z + D / 2)]
        if self.load is not None:
            pts.append((0.0, -self.load[0]))
        span = max(p[0] for p in pts) - min(p[0] for p in pts)
        if span < 1.5 * sec.h:  # a tall section: room beside it, not drawn taller than about half the page width
            m = 0.9 * sec.h
            pts += [(min(-m, min(p[0] for p in pts)), -sec.h / 2), (max(m, max(p[0] for p in pts)), -sec.h / 2)]
        fig.raw(self._geometry, pts, self._geometry_extent)
        fig.raw(lambda T: self._annotations(T, dims, labels)[0], None,
                lambda T: self._annotations(T, dims, labels)[1])
        if info:
            fig.notes(self.info + [Note(n, "small") for n in self.notes])
        if legend:
            fig.legend(self._legend_entries(), at="below")
        return fig

    def _geometry_extent(self, T):
        out = []
        if self.lines:
            x0, _ = T(self.shape.xmin, 0)
            x1, _ = T(self.shape.xmax, 0)
            for z, *_ in self.lines:
                y = T(0, -z)[1]
                out.append((x0 - 10, y - 2, x1 + 10, y + 2))
        return out

    @staticmethod
    def _circ(T, c, r, cls, attrs=""):
        """a circle of world radius r (at least 0.6 px)"""
        return svg_circle(*T(*c), max(r * T.sx, 0.6), cls, attrs=attrs)

    def _geometry(self, T):
        sh, sec, out = self.shape, self.sec, []
        page = lambda rings: [[T(*p) for p in r] for r in rings]  # noqa: E731
        for cls, rings in sh.regions():
            out.append(svg_path(page(rings), cls, attrs=' data-role="concrete"' +
                                (' data-part="topping"' if cls == "topping" else "")))
        for zc, r, sg in sorted(sh.circles, key=lambda c: -c[2]):
            out.append(self._circ(T, (0.0, -zc), r, "concrete" if sg > 0 else "void",
                                  ' data-role="concrete"' if sg > 0 else ' data-role="hole"'))
        if self.zone is not None:
            rings = sh.band(*self.zone)
            if rings:
                out.append(svg_path(page(rings), "comp", attrs=f' data-role="compression" data-z0="{fmt(self.zone[0])}" '
                                                               f'data-z1="{fmt(self.zone[1])}"'))
        if self.stirrup:
            out += self._stirrups(T)
        for g in self.tendons:
            for x, z, D, dp in g["ducts"]:
                if D > 0:
                    out.append(self._circ(T, (x, -z), D / 2, "duct", ' data-role="duct"'))
                out.append(self._circ(T, (x, -z), min(dp, D) / 2 if D > 0 else dp / 2, "tendon", ' data-role="tendon"'))
            for x, z, dd in g["strands"]:
                out.append(self._circ(T, (x, -z), dd / 2, "tendon", ' data-role="strand"'))
        for k, r in enumerate(self.rows):
            for x, z, dd in r["bars"]:
                out.append(self._circ(T, (x, -z), dd / 2, "bar", f' data-role="bar" data-row="{k}" data-d="{fmt(dd)}"'))
        for e in self.ext:
            for x, z, D, i in e["items"]:
                out.append(self._circ(T, (x, -z), D / 2, "external", f' data-role="external" data-row="{i}"'))
        x0p, _ = T(sh.xmin, 0)
        x1p, _ = T(sh.xmax, 0)
        for z, cls, lab, role, col in self.lines:
            y = T(0, -z)[1]
            data = f' data-role="{role}" data-z="{fmt(z)}"'
            if cls == "na":
                out.append(svg_line(x0p - 8, y, x1p + 8, y, "na", attrs=data))
            else:
                out.append(svg_line(x0p - 8, y, x1p + 8, y, "dim", style="stroke-dasharray:7 2 1.5 2" +
                                    (f";stroke:{col}" if col else ""), attrs=data))
        if self.load is not None:
            x, y = T(0.0, -self.load[0])
            out.append(svg_circle(x, y, 5.5, "", style=f"fill:{C['paper']};stroke:{C['load']};stroke-width:1.4",
                                  attrs=' data-role="load"') + svg_circle(x, y, 2, "", style=f"fill:{C['load']}"))
            xg, yg = T(0.0, -sec.zg)
            out.append(f'<path d="M{fmt(xg - 5)},{fmt(yg)} L{fmt(xg + 5)},{fmt(yg)} M{fmt(xg)},{fmt(yg - 5)} '
                       f'L{fmt(xg)},{fmt(yg + 5)}" class="fm-dim" style="stroke:{C["ink"]}" data-role="centroid"/>')
        out += [s for s, _ in self._warn_marks(T)]
        return "".join(out)

    def _stirrups(self, T):
        """closed stirrups: one loop round the bars of each part of the narrowest depth (the web, each web of a box);
        a hoop on circular sections"""
        sec, sh, ds = self.sec, self.shape, self.ds
        out = []
        if sec.ring is not None and sec.kind in ("circle", "annulus"):
            d = float(np.nanmax(sec.bar_d)) if not np.isnan(sec.bar_d).all() else 0.0
            out.append(self._circ(T, (0.0, -sec.h / 2), sec.ring + d / 2 + ds / 2, "stirrup", ' data-role="stirrup"'))
            return out
        rows = [r for r in self.rows if r["face"] in ("bottom", "top")] or self.rows
        if not rows:
            return out
        zt = min(z - dd / 2 for r in rows for _, z, dd in r["bars"]) - ds / 2
        zb = max(z + dd / 2 for r in rows for _, z, dd in r["bars"]) + ds / 2
        # the narrowest depth below the topping
        z0 = sh.topping[1] if sh.topping else 0.0
        cand = [(sum(b - a for a, b in sh.segments((za + zb2) / 2)), (za + zb2) / 2)
                for za, zb2 in zip(sh.boundaries()[:-1], sh.boundaries()[1:]) if za >= z0 - 1e-9]
        if not cand:
            return out
        zm = min(cand)[1]
        dmax = max(r["d"] for r in rows)
        rad = max(2.0 * ds, dmax / 2 + ds / 2)
        for a, b in sh.segments(zm):
            xa, xb = a + self.side + ds / 2, b - self.side - ds / 2
            if xb - xa <= 2 * rad or zb - zt <= 2 * rad:
                continue
            (pa, qa), (pb, qb) = T(xa, -zt), T(xb, -zb)
            out.append(f'<rect x="{fmt(pa)}" y="{fmt(qa)}" width="{fmt(pb - pa)}" height="{fmt(qb - qa)}" '
                       f'rx="{fmt(rad * T.sx)}" class="fm-stirrup" style="stroke-width:{fmt(max(ds * T.sx, 0.8))}" '
                       f'data-role="stirrup"/>')
        return out

    def _warn_marks(self, T):
        """(svg, anchor (px) for the callout) of every ringed item"""
        out = []
        sh = self.shape
        done = set()
        for kind, payload, num, short, full in self.warn:
            if kind == "rows":
                for r in payload:
                    xs = [T(x, -z) for x, z, _ in r["bars"]]
                    rp = r["d"] / 2 * T.sx + 4
                    if id(r) in done:
                        out.append(("", (max(p[0] for p in xs) + rp, sum(p[1] for p in xs) / len(xs), num)))
                        num = None
                        continue
                    done.add(id(r))
                    x0, x1 = min(p[0] for p in xs) - rp, max(p[0] for p in xs) + rp
                    y0, y1 = min(p[1] for p in xs) - rp, max(p[1] for p in xs) + rp
                    out.append((f'<rect x="{fmt(x0)}" y="{fmt(y0)}" width="{fmt(x1 - x0)}" height="{fmt(y1 - y0)}" '
                                f'rx="{fmt(min(rp, (y1 - y0) / 2))}" class="fm-warn" data-role="warn"/>',
                                (x1, (y0 + y1) / 2, num)))
                    num = None
            elif kind == "circle":
                x, z, r = payload
                cx, cy = T(x, -z)
                rp = r * T.sx + 4
                out.append((svg_circle(cx, cy, rp, "warn", attrs=' data-role="warn"'), (cx + rp, cy, num)))
            elif kind == "ring":
                d = float(np.nanmax(self.sec.bar_d)) if not np.isnan(self.sec.bar_d).all() else 0.0
                cx, cy = T(0.0, -self.sec.h / 2)
                rp = (payload + d / 2) * T.sx + 4
                out.append((svg_circle(cx, cy, rp, "warn", style="stroke-dasharray:5 3", attrs=' data-role="warn"'),
                            (cx + rp * math.cos(math.pi / 4), cy - rp * math.sin(math.pi / 4), num)))
            elif kind == "web":
                bnd = sh.boundaries()
                cand = [(sum(b - a for a, b in sh.segments((za + zb) / 2)), za, zb) for za, zb in zip(bnd[:-1], bnd[1:])]
                _, za, zb = min(cand)
                for a, b in sh.segments((za + zb) / 2):
                    (p0, q0), (p1, q1) = T(a, -za), T(b, -zb)
                    out.append((f'<rect x="{fmt(p0 + 2)}" y="{fmt(q0 + 2)}" width="{fmt(p1 - p0 - 4)}" '
                                f'height="{fmt(q1 - q0 - 4)}" class="fm-warn" style="stroke-dasharray:5 3" '
                                f'data-role="warn"/>', (p1 - 2, (q0 + q1) / 2, num)))
                    num = None
        return out

    # ── page annotations ────────────────────────────────────────────────────
    def _annotations(self, T, dims, labels):
        """the dimensions, the labels of the steel and the stress diagram, on the page: (svg, boxes)"""
        svg, boxes = [], []

        def add(s, b=()):
            svg.append(s)
            boxes.extend(b)

        sec, sh = self.sec, self.shape
        xl, _ = T(sh.xmin, 0)
        xr, _ = T(sh.xmax, 0)
        for e in self.ext:
            for x, z, D, _ in e["items"]:
                xl = min(xl, T(x - D / 2, 0)[0])
                xr = max(xr, T(x + D / 2, 0)[0])
        ytop, ybot = T(0, 0)[1], T(0, -sec.h)[1]
        bottom_px = ybot
        if dims:
            bottom_px = max(bottom_px, self._dims(T, add, xl))
        right = xr + 14
        if labels:
            right, low = self._labels(T, add, xr)
            bottom_px = max(bottom_px, low)
        if self.stress is not None and self.beside:
            self._stress_diagram(T, add, right + 16, ytop, ybot)
        elif self.stress is not None:  # under the section, same depth scale up to 150 px
            y0 = bottom_px + 36
            self._stress_diagram(T, add, max(xl, 12.0), y0, y0 + min(ybot - ytop, 150.0), tags=True)
        return "".join(svg), boxes

    def _dims(self, T, add, xl):
        """vertical dimensions on the left (the analysis ones nearest, then the chain of the pieces, then the overall
        depth), horizontal ones above and below; returns the lowest page y used"""
        sec, sh = self.sec, self.shape
        h = sec.h
        circular = bool(sh.circles) and not sh.outer
        cols = [[(z0, z1, lab)] for z0, z1, lab in self.vdims]
        if not circular:
            bnd = sh.boundaries()
            if len(bnd) > 2:
                cols.append([(a, b, _dim(b - a)) for a, b in zip(bnd[:-1], bnd[1:])])
            cols.append([(0.0, h, _dim(h))])
        x, low = xl - 12, T(0, -h)[1]
        for col in cols:
            zs = [col[0][0]] + [z1 for _, z1, _ in col]  # a chain from z0 to the last z1 (one item: its far end last)
            s, b = dim_chain([T(0, -z)[1] for z in zs], x, [lab for *_, lab in col], vertical=True,
                             ext=[T(sh.left(z), -z)[0] for z in zs])
            add(s, b)
            x, low = min(v[0] for v in b) - 9, max([low] + [v[3] for v in b[:-1]])
        if circular:
            outer = max((r for _, r, sg in sh.circles if sg > 0), default=h / 2)
            zc = next(zc for zc, r, sg in sh.circles if sg > 0 and r == outer)
            low = max(low, self._hdim(T, add, -outer, outer, zc, 0, label=f"D = {_dim(2 * outer)}"))
            for zc, r, sg in sh.circles:
                if sg < 0:  # the inner diameter across the hole
                    (p0, q), (p1, _) = T(-r, -zc), T(r, -zc)
                    add(*dim_chain([p0, p1], q, [f"d = {_dim(2 * r)}"]))
            return low
        top_iv = [iv for iv in sh.outer if iv[0] <= 1e-9]
        bot_iv = [iv for iv in sh.outer if abs(iv[1] - h) < 1e-6 * h]
        span = lambda ivs: (min(iv[2] for iv in ivs), max(iv[3] for iv in ivs)) if ivs else None  # noqa: E731
        if top_iv and span(top_iv) != span(bot_iv):  # the top width when it differs from the bottom one
            self._hdim(T, add, *span(top_iv), 0.0, -1)
        rows = []
        if bot_iv:
            a, b = min(iv[2] for iv in bot_iv), max(iv[3] for iv in bot_iv)
            rows.append((a, b, h))
            if sh.webs == 1 and not sh.holes:  # the web of an I section, narrower than both flanges
                z0, z1 = _web_zone(sh)
                segs = sh.segments((z0 + z1) / 2)
                if segs[-1][1] - segs[0][0] < (b - a) - 1e-6 and z1 < h - 1e-6:
                    rows.insert(0, (segs[0][0], segs[-1][1], z1))
        for k, (a, b, zf) in enumerate(rows):
            low = max(low, self._hdim(T, add, a, b, zf, k))
        return low

    def _hdim(self, T, add, a, b, zf, k, label=None):
        """a horizontal dimension of the width a…b with extension lines from depth zf, above the top (k < 0) or in row
        k below the bottom; returns the lowest page y used"""
        (pa, qf), (pb, _) = T(a, -zf), T(b, -zf)
        q = T(0, 0)[1] - 14 if k < 0 else T(0, -self.sec.h)[1] + 14 + 18 * k
        s, bx = dim_chain([pa, pb], q, [_dim(b - a) if label is None else label], ext=qf, side=-1 if k < 0 else 1)
        add(s, bx)
        return max(q + 3, max(v[3] for v in bx[:-1]))

    def _label_items(self):
        """[(anchor world (x, z, r) or ("line", z), text, cls, colour)] of the steel, the lines and the load"""
        L, items = self.L, []
        dia = L["dia"]
        side = [r for r in self.rows if r["face"] == "side"]
        for r in self.rows:
            if r["face"] in ("side", "ring"):
                continue
            if r.get("lumped"):
                txt = f"As = {_mm(r['area'])} mm²"
            else:
                txt = "+".join(f"{n}{dia}{_mm(d)}" for n, d in r["groups"]) + ("*" if r.get("star") else "")
            x, z, d = max(r["bars"], key=lambda b: b[0])
            items.append(((x, z, d / 2), txt, "label", None))
        if side:
            by = {}
            for r in side:
                for n, d in r["groups"]:
                    by[d] = by.get(d, 0) + n
            txt = "+".join(f"{n}{dia}{_mm(d)}" for d, n in sorted(by.items(), reverse=True)) + \
                f"{L['lp']}{L['side']}{L['rp']}"
            mid = side[len(side) // 2]
            x, z, d = max(mid["bars"], key=lambda b: b[0])
            items.append(((x, z, d / 2), txt, "label", None))
        ring = [r for r in self.rows if r["face"] == "ring"]
        if ring:
            bars = [b for r in ring for b in r["bars"]]
            ds_ = sorted({d for _, _, d in bars})
            txt = f"{len(bars)}{dia}{_mm(ds_[0])}" if len(ds_) == 1 else f"{len(bars)} bars"
            r0 = self.sec.h / 2
            x, z, d = max(bars, key=lambda b: b[0] - abs(b[1] - r0 * 0.45))
            items.append(((x, z, d / 2), txt, "label", None))
        s_d, s_a = self.strand
        for g in self.tendons:
            if g["ducts"]:
                D = max(D for _, _, D, _ in g["ducts"])
                Ap = [self.sec.tendons[i, 0] for i in g["idx"]]
                m = len(g["ducts"])
                c = Ap[0] / s_a
                same = all(abs(a - Ap[0]) < 1e-6 * Ap[0] for a in Ap)
                if same and abs(c - round(c)) < 0.03 * c and round(c) >= 1:
                    tail = f"{int(round(c))}{dia}s{number(s_d, 1)}"
                else:
                    tail = f"Ap = {_mm(sum(Ap) / m)}" if same else f"Ap = {_mm(sum(Ap))}"
                pre = f"{m}×" if m > 1 else ""
                txt = f"{pre}{dia}{_mm(D)}{L['duct_w']} {tail}" if self.lang == "zh" else f"{pre}{dia}{_mm(D)} duct, {tail}"
                x, z, D2, _ = max(g["ducts"], key=lambda q: q[0])
                items.append(((x, z, max(D2, 1.0) / 2), txt, "label", None))
            elif g["strands"]:
                if "n_strands" in g:
                    txt = f"{g['n_strands']}{dia}s{number(s_d, 1)}"
                else:
                    txt = f"Ap = {_mm(float(self.sec.tendons[g['idx'], 0].sum()))}"
                x, z, d = max(g["strands"], key=lambda q: q[0])
                items.append(((x, z, d / 2), txt, "label", None))
        for e in self.ext:
            A = [self.sec.external[i, 0] for i in e["idx"]]
            k = sum(e["n_each"])
            txt = (f"{L['ext_w']} {k}×{_mm(sum(A) / k)} mm²" if len({round(a / n, 6) for a, n in zip(A, e['n_each'])}) == 1
                   else f"{L['ext_w']} {_mm(sum(A))} mm²")
            x, z, D, _ = max(e["items"], key=lambda q: q[0])
            items.append(((x, z, D / 2), txt, "label", None))
        for z, cls, lab, role, col in self.lines:
            items.append((("line", z), lab, "small", col))
        if self.load is not None:
            items.append(((0.0, self.load[0], 0.0), self.load[1], "loadtext", C["load"]))
        return items

    def _labels(self, T, add, xr):
        """the labels in one column right of the section, each with a leader to its item (femotion.viz.leader_column);
        returns (right edge, lowest page y)"""
        items = []
        for anchor, txt, cls, colour in self._label_items():
            if anchor[0] == "line":
                items.append((xr + 8, T(0, -anchor[1])[1], 0.0, txt, cls, colour))
            else:
                x, z, rad = anchor
                ax, ay = T(x, -z)
                items.append((ax, ay, rad * T.sx if rad > 1e-6 else 6.0, txt, cls, colour))
        for _, (ax, ay, num) in self._warn_marks(T):
            if num is not None:
                items.append((ax, ay, 0.0, int(num), "warntext", C["warn"]))
        if not items:
            return xr, T(0, -self.sec.h)[1]
        s, b = leader_column(items, xr + 22, top=T(0, 0)[1] - 6, bottom=T(0, -self.sec.h)[1])
        add(s, b)
        return max(v[2] for v in b), max(v[3] for v in b)

    def _stress_diagram(self, T, add, x0, ytop, ybot, tags=False):
        """σ over the depth beside the section (from ytop at the top fibre to ybot at the bottom one): compression (+)
        to the left of the zero line, tension to the right; tags: name the edges (a diagram not beside the section)"""
        st, sec = self.stress, self.sec
        prof, lims = st["prof"], st["lims"]
        schematic = st.get("schematic", False)
        vals = [abs(v) for _, v in prof] + [abs(v) for v, _ in lims]
        smax = max(vals + [1e-9])
        W = max(min(0.18 * self.width, 0.7 * (ybot - ytop), 100.0), 50.0)
        k = W / smax  # px per MPa
        cmax = max([v for _, v in prof] + [v for v, _ in lims] + [0.0])
        lab_w = max([text_width(number(v, 2), TEXT_SIZE["dimtext"]) for _, v in prof] + [0.0]) / 2
        ax = x0 + max(k * cmax, lab_w) + 4
        y = lambda z: ytop + (ybot - ytop) * z / sec.h  # noqa: E731
        # split at zero crossings
        seq = [prof[0]]
        for (za, va), (zb, vb) in zip(prof[:-1], prof[1:]):
            if va * vb < 0:
                seq.append((za + (zb - za) * va / (va - vb), 0.0))
            seq.append((zb, vb))
        for sign, cls in ((1, "comp"), (-1, "tens")):
            poly, cur = [], []
            for z, v in seq:
                if sign * v >= -1e-12:
                    cur.append((ax - k * v, y(z)))
                elif cur:
                    poly.append(cur)
                    cur = []
            if cur:
                poly.append(cur)
            for p in poly:
                if len(p) >= 2 and any(abs(px - ax) > 1e-6 for px, _ in p):
                    add(svg_poly([(ax, p[0][1])] + p + [(ax, p[-1][1])], cls))
        line = [(ax, y(prof[0][0]))] + [(ax - k * v, y(z)) for z, v in prof] + [(ax, y(prof[-1][0]))]
        attrs = (f' data-role="stress" data-axis="{fmt(ax)}" data-scale="{k:.9g}" data-z="'
                 + " ".join(fmt(z) for z, _ in prof) + '" data-sigma="' + " ".join(f"{v:.9g}" for _, v in prof) + '"')
        add(svg_poly(line, "curve", closed=False, attrs=attrs))
        add(svg_line(ax, ytop - 4, ax, ybot + 4, "axis"))
        low = ybot + 4
        if st.get("cracked"):
            edge, x = st["cracked"]
            if x < sec.h:
                za, zb = (x, sec.h) if edge == "top" else (0.0, sec.h - x)
                s, b = text_item(ax + 5, y((za + zb) / 2), self.L["cracked"], "small", anchor="start", halo=True)
                add(s, [b])
        if not schematic:  # the values at the ends
            for (z, v), dy in zip((prof[0], prof[-1]), (-8, 11)):
                if abs(z) < 1e-9 or abs(z - sec.h) < 1e-9:
                    s, b = text_item(ax - k * v, y(z) + dy, number(v, 2), "dimtext", halo=True,
                                     fill=C["compression"] if v > 0 else C["tension"] if v < 0 else None)
                    add(s, [b])
                    low = max(low, b[3])
        for v, lab in lims:  # the label above, running towards the zero line
            px = ax - k * v
            add(svg_line(px, ytop, px, ybot, "dim", style="stroke-dasharray:4 3", attrs=' data-role="limit"'))
            s, b = text_item(px + (-3 if v < 0 else 3), ytop - 21, f"{lab} = {number(abs(v), 2)}", "small",
                             anchor="end" if v < 0 else "start", halo=True)
            add(s, [b])
        if tags:
            for yy, e in ((ytop, "top"), (ybot, "bottom")):
                s, b = text_item(ax + 6, yy, self.L["edge"][e], "small", anchor="start", halo=True)
                add(s, [b])
        cap = self.L["sigma"] if not schematic else \
            ("混凝土压应力（示意）" if self.lang == "zh" else "concrete stress (schematic)")
        s, b = text_item(ax, low + 10, cap, "small")
        add(s, [b])
        low = b[3]
        s, b = text_item(ax - 6, low + 8, f"← {self.L['C']}", "small", anchor="end")
        s2, b2 = text_item(ax + 6, low + 8, f"{self.L['T']} →", "small", anchor="start")
        add(s + s2, [b, b2])

    def _legend_entries(self):
        """(class, label, kind) of what the drawing shows"""
        L, sec = self.L, self.sec
        ent = [("concrete", L["concrete"])]
        if self.shape.topping:
            ent.append(("topping", L["topping"]))
        if len(sec.bars):
            ent.append(("bar", L["bar"], "dot"))
        if self.stirrup:
            ent.append(("stirrup", L["stirrup"], "ring"))
        if self.tendons:
            ent.append(("duct", L["duct"] if any(g["ducts"] for g in self.tendons) else L["strand"],
                        lambda x, y: svg_circle(x + 10, y, 5, "duct") + svg_circle(x + 10, y, 2.6, "tendon")))
        if self.ext:
            ent.append(("external", L["external"], "dot"))
        if self.zone is not None:
            ent.append(("comp", L["comp"]))
        if any(c == "na" for _, c, *_ in self.lines):
            ent.append(("na", L["na"]))
        if self.warn:
            ent.append(("warn", L["warn"], "ring"))
        return ent
