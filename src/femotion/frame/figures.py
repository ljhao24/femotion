"""Drawings of frame models as femotion.viz Figures (plain SVG): load arrangements and influence lines with their
worst placements, the way textbooks and design drawings show them.

    from femotion.frame.figures import load_figure, influence_figure, elevation_figure
    load_figure(m, case)                         # the bridge in elevation with the loads of a LoadCase
    influence_figure(il, layout)                 # above: the worst placement; below: the influence line, same x axis
    elevation_figure(m)                          # members, supports and spans only

The elevation is the vertical plane through the model's main horizontal direction (the principal axis of the nodes in
plan; z is up). Supports are drawn from their restrained degrees of freedom in that plane: vertical and horizontal
translation and rotation restrained → fixed, vertical and horizontal → pin, vertical only → roller. Lengths are true to
scale along the bridge; arrows, bands, text and the influence ordinates are sized in pixels, the labels placed clear of
each other (femotion.viz.Placer), and the page solved so that they all fit (femotion.viz.Figure). Loads out of the plane
are not drawn. lang="zh" (default) or "en". Program conventions and the shared style: docs/figures.md."""
from __future__ import annotations

import math

import numpy as np

from ..viz import COLORS as C
from ..viz import (Figure, Note, Placer, dim_chain, fmt, moment_arc, number, shifted, support_symbol, svg_arrow,
                   svg_line, svg_poly, svg_text)
from .loads import (BarStrain, ElementLoad, Imposed, LineLoad, LoadCase, NodalLoad, PointLoad, SelfWeight, Settlement,
                    TendonLoad, _geometry, _line_pieces, _point_target)
from .model import Model

# ── wording ────────────────────────────────────────────────────────────────────────────────────────────────────────
WORDS = {
    "zh": {"self_weight": "自重", "imposed": "强加变形", "strain": "ε", "curvature": "κ", "tendon": "钢束",
           "exaggerated": "偏心竖向放大", "settlement": "沉降", "bar": "初应变", "section": "截面", "left": "左",
           "right": "右", "lane": "车道荷载", "vehicle": "车辆荷载", "crowd": "人群荷载", "load": "移动荷载",
           "il": "影响线", "worst": "最不利布置", "factor": "系数", "M": "弯矩", "V": "剪力", "N": "轴力", "T": "扭矩",
           "max": "max", "min": "min", "none": "该符号无影响线竖标，效应为 0", "off": "桥外", "heading": "行车方向"},
    "en": {"self_weight": "self-weight", "imposed": "imposed strain", "strain": "ε", "curvature": "κ",
           "tendon": "tendon", "exaggerated": "eccentricity exaggerated", "settlement": "settlement", "bar": "initial strain",
           "section": "section", "left": "left", "right": "right", "lane": "lane load", "vehicle": "vehicle load",
           "crowd": "crowd load", "load": "moving load", "il": "influence line", "worst": "worst placement",
           "factor": "factor", "M": "moment", "V": "shear", "N": "axial force", "T": "torque", "max": "max", "min": "min",
           "none": "no ordinate of this sign: the effect is 0", "off": "off the bridge", "heading": "direction"},
}
_SYM = {"My": "M", "Mz": "M", "Vz": "V", "Vy": "V", "N": "N", "T": "T"}
_UNIT = {"M": "kN·m", "V": "kN", "N": "kN", "T": "kN·m"}

# pixel sizes
BAND_MAX, BAND_MIN, BAND_GAP, LABEL_ROOM = 26.0, 9.0, 4.0, 18.0
ARROW_STEP, HEAD = 13.0, 6.5
POINT_LEN = 34.0
IL_HEIGHT = 46.0  # largest influence ordinate (px)


def _w(lang):
    if lang not in WORDS:
        raise ValueError("lang: 'zh' or 'en'")
    return WORDS[lang]


def _d(v: float) -> str:
    """a number for a data- attribute (full precision, for tests and scripts reading the drawing)"""
    return f"{float(v):.10g}"


def _len(v: float) -> str:
    return number(round(v, 2)) + " m"


def _sci(v: float) -> str:
    """1.2×10⁻⁴ style"""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    if -2 <= e <= 3:
        return number(v)
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return number(v / 10 ** e, 2) + "×10" + sup


# ── the elevation ──────────────────────────────────────────────────────────────────────────────────────────────────
class Elevation:
    """the model seen in the vertical plane through its main horizontal direction d (unit, z = 0): node (u, z) with u the
    distance along d from the first node drawn"""

    def __init__(self, m: Model, members=None):
        self.m = m
        X = np.asarray(m.X, float)
        beams = range(len(m.beams)) if members is None else members
        used = sorted({int(n) for e in beams for n in m.beams[e][:2]} | {int(n) for t in m.trusses for n in t[:2]}
                      | {int(n) for n in m.fixed})
        P = X[used, :2] if used else X[:, :2]
        Pc = P - P.mean(axis=0)
        if len(P) < 2 or np.abs(Pc).max() < 1e-9:
            d = np.array([1.0, 0.0])
        else:
            _, V = np.linalg.eigh(Pc.T @ Pc)
            d = V[:, -1]
        if abs(d[0]) >= abs(d[1]):
            d = d * np.sign(d[0])
        else:
            d = d * np.sign(d[1])
        self.d = np.array([d[0], d[1], 0.0])
        self.n = np.array([-d[1], d[0], 0.0])  # horizontal, normal to the plane
        self.toward_viewer = np.cross(self.d, [0.0, 0.0, 1.0])  # the drawing shows u right, z up
        self.u0 = float((X[used] @ self.d).min()) if used else 0.0
        self.members = list(beams)

    def uz(self, p):
        p = np.asarray(p, float)
        return float(p @ self.d) - self.u0, float(p[2])

    def node(self, n):
        return self.uz(self.m.X[n])

    def support_kind(self, n):
        f = self.m.fixed.get(n)
        if f is None:
            return None
        vert = bool(f[2])
        hor = all(f[a] for a in (0, 1) if abs(self.d[a]) > 0.1)
        rot = all(f[3 + a] for a in (0, 1) if abs(self.n[a]) > 0.1)
        if vert and hor and rot:
            return "fixed"
        if vert and hor:
            return "pin"
        if vert:
            return "roller"
        if hor:
            return "guide"
        return None

    def outward(self, n):
        """page direction pointing away from the members at node n (for the fixed-end hatch)"""
        u, z = self.node(n)
        dirs = []
        for (i, j, *_r) in self.m.beams:
            if n in (i, j):
                o = j if i == n else i
                uo, zo = self.node(o)
                dirs.append((uo - u, zo - z))
        if not dirs:
            return (1.0, 0.0), False
        du, dz = np.mean(dirs, axis=0)
        vertical = abs(dz) > 2 * abs(du)
        return ((-1.0 if du > 0 else 1.0), 0.0), vertical


def _members_svg(fig: Figure, el: Elevation, *, trusses=True):
    m = el.m
    for e in el.members:
        i, j = m.beams[e][:2]
        fig.polyline([el.node(i), el.node(j)], "member")
    if trusses:
        for (i, j, *_r) in m.trusses:
            fig.polyline([el.node(i), el.node(j)], "curve")


def _supports(el: Elevation, T, dy=0.0):
    """support symbols at the restrained nodes, dy px below them (under a girder drawn with depth)"""
    out, boxes, bottoms = [], [], {}
    for n in sorted(el.m.fixed):
        kind = el.support_kind(n)
        if kind is None:
            continue
        x, y = T(*el.node(n))
        ow, vert = el.outward(n)
        svg, b = support_symbol(kind, x, y + (0.0 if vert else dy), outward=ow, vertical_member=vert)
        out.append(f'<g data-support="{kind}" data-node="{n}">{svg}</g>')
        boxes.append(b)
        bottoms[n] = b[3]
    return "".join(out), boxes, bottoms


def _span_dims(el: Elevation, T, bottoms):
    """span dimensions between the vertical supports, below them; returns (svg, boxes, y of the line)"""
    sup = [n for n in el.m.fixed if el.support_kind(n) in ("fixed", "pin", "roller")]
    if len(sup) < 2:
        return "", [], None
    us = sorted({round(el.node(n)[0], 9) for n in sup})
    if len(us) < 2:
        return "", [], None
    xs = [T(u, 0.0)[0] for u in us]
    yb = max(bottoms.values()) if bottoms else T(us[0], 0.0)[1]
    y = yb + 22
    svg, boxes = dim_chain(xs, y, [_len(b - a) for a, b in zip(us[:-1], us[1:])], ext=yb, short="drop")
    return svg, boxes, y


# ── loads in the elevation ─────────────────────────────────────────────────────────────────────────────────────────
class _Band:
    """a distributed load drawn as a band: pts = [(u, z, w)] along the members (w: intensity, + along the band's sense),
    down: True when it pushes down (drawn above the member)"""

    def __init__(self, pts, down, texts, data=""):
        self.pts, self.down, self.texts, self.data = pts, down, texts, data
        self.row = 0

    @property
    def span(self):
        us = [p[0] for p in self.pts]
        return min(us), max(us)

    @property
    def wmax(self):
        return max(abs(p[2]) for p in self.pts)


def _vert(el, g, e, p_local=None, p_global=None):
    """vertical (global z) component of a load given in local or global axes"""
    v = p_global if p_global is not None else g._beam_cache[e][1].T @ p_local
    return float(v[2]), float(np.dot(v, el.d))


def _bands_of_pieces(el, g, pieces, texts, data):
    """element pieces → bands (consecutive pieces joined), split by the sense of the load. Pieces on steep members
    (piers, towers: less than half their length horizontal) do not make bands: they come back as (u, z, w) at their
    middle, for a note beside the member"""
    m = g.m
    segs, steep = [], []
    for pc in pieces:
        i, j = m.beams[pc.beam][:2]
        Xi, Xj = m.X[i], m.X[j]
        L = g._beam_cache[pc.beam][0]
        A = el.uz(Xi + (Xj - Xi) * pc.a / L)
        B = el.uz(Xi + (Xj - Xi) * pc.b / L)
        wa, _ = _vert(el, g, pc.beam, pc.p0)
        wb, _ = _vert(el, g, pc.beam, pc.p1)
        if abs(B[0] - A[0]) < 0.5 * math.hypot(B[0] - A[0], B[1] - A[1]):
            if abs(wa) + abs(wb) > 0:
                steep.append(((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (wa + wb) / 2, pc.beam))
            continue
        if A[0] > B[0]:
            A, B, wa, wb = B, A, wb, wa
        segs.append(((A[0], A[1], wa), (B[0], B[1], wb)))
    segs.sort(key=lambda s: s[0][0])
    bands, cur = [], []
    for sgm in segs:
        if abs(sgm[0][2]) < 1e-12 and abs(sgm[1][2]) < 1e-12:
            continue
        tol = 1e-9 * max(1.0, abs(sgm[0][0]))
        if cur and abs(cur[-1][0] - sgm[0][0]) < tol and abs(cur[-1][1] - sgm[0][1]) < tol and \
                np.sign(cur[-1][2] + sgm[0][2]) == np.sign(sgm[0][2] + sgm[1][2]):
            cur += [sgm[0], sgm[1]]
        else:
            if cur:
                bands.append(cur)
            cur = [sgm[0], sgm[1]]
    if cur:
        bands.append(cur)
    out = []
    for pts in bands:
        down = sum(p[2] for p in pts) < 0
        out.append(_Band([(u, z, abs(w)) for u, z, w in pts], down, texts, data))
    return out, steep


def _steep_notes(steep, text_of, data):
    """one note per member run carrying a load along its axis (piers, towers)"""
    notes, seen = [], set()
    for u, z, w, e in steep:
        key = (round(u, 6), round(w, 6))
        if key in seen:
            continue
        seen.add(key)
        notes.append((u, z, text_of(abs(w)), data))
    return notes


class _Scene:
    """everything drawn in page pixels around the members: bands, arrows, moments, notes, supports, dimensions"""

    def __init__(self, el: Elevation, lang="zh"):
        self.el, self.lang, self.W = el, lang, _w(lang)
        self.bands: list[_Band] = []
        self.points = []  # (u, z, Pv, Ph, label, data, style)
        self.moments = []  # (u, z, M ccw, label)
        self.notes = []  # (u0, u1, z, label, data): imposed strains, …
        self.tendons = []  # (pts [(u, z, ecc)], label)
        self.settlements = []  # (node, dz, label)
        self.texts = []  # (u, z, text): free notes at members
        self.member_notes = []  # (u, z, text, data): loads along steep members, written beside them
        self.section = None  # (u, z, label): a section mark
        self.extra = None  # T -> (svg, boxes) drawn after the loads (influence panel)
        self.dims = True

    # ── from a load case ──
    def add_case(self, case: LoadCase):
        el, W = self.el, self.W
        g = _geometry(el.m)
        m = el.m
        for idx, it in enumerate(case.items):
            data = f' data-item="{idx}"'
            if isinstance(it, LineLoad):
                pieces = _line_pieces(g, it)
                bands, steep = _bands_of_pieces(el, g, pieces, None, data)
                self.member_notes += _steep_notes(steep, lambda w: it.label or f"q = {number(w)} kN/m", data)
                for b in bands:
                    ws = [p[2] for p in b.pts]
                    uniform = max(ws) - min(ws) <= 1e-9 * max(max(ws), 1e-12)
                    b.texts = [it.label] if it.label else ([f"q = {number(ws[0])} kN/m"] if uniform else None)
                    b.kind = "line"
                self.bands += bands
            elif isinstance(it, SelfWeight):
                beams = range(len(m.beams)) if it.beams is None else it.beams
                pcs = []
                for e in beams:
                    L, R, T = g._beam_cache[e]
                    p = R @ np.array([0.0, 0.0, -m.beams[e][2].mass * it.g * it.factor])
                    pcs.append(ElementLoad("uniform", int(e), 0.0, L, p, p))
                bands, steep = _bands_of_pieces(el, g, pcs, None, data)
                self.member_notes += _steep_notes(steep, lambda w: f"{W['self_weight']} {number(w)} kN/m", data)
                for b in bands:
                    ws = [p[2] for p in b.pts]
                    same = max(ws) - min(ws) <= 1e-9 * max(ws)
                    b.texts = [it.label] if it.label else [f"{W['self_weight']} {number(ws[0])} kN/m" if same
                                                           else W["self_weight"]]
                    b.kind = "self_weight"
                    self.bands.append(b)
            elif isinstance(it, PointLoad):
                tgt = _point_target(g, it)
                if tgt[0] == "node":
                    p3, uz = tgt[2], el.node(tgt[1])
                else:
                    _, e, a, p = tgt
                    i, j = m.beams[e][:2]
                    L = g._beam_cache[e][0]
                    uz = el.uz(m.X[i] + (m.X[j] - m.X[i]) * a / L)
                    p3 = g._beam_cache[e][1].T @ p
                self._point(uz, p3, it.label, data)
            elif isinstance(it, NodalLoad):
                uz = el.node(it.node)
                if np.abs(it.F).max() > 0:
                    self._point(uz, it.F, it.label, data)
                Mv = float(np.dot(it.M, el.toward_viewer))
                if abs(Mv) > 0:
                    self.moments.append((*uz, Mv, it.label if np.abs(it.F).max() == 0 and it.label else
                                         f"M = {number(abs(Mv))} kN·m", data))
            elif isinstance(it, Imposed):
                us = []
                for e in it.beams:
                    us += [el.node(n) for n in m.beams[e][:2]]
                u0, u1 = min(p[0] for p in us), max(p[0] for p in us)
                z = max(p[1] for p in us)
                if it.label:
                    lab = it.label
                else:
                    parts = []
                    if np.any(it.strain):
                        v = it.strain
                        parts.append(f"ε = {_sci(v[0])}" if np.ptp(v) == 0 else "ε")
                    if np.any(it.curvature):
                        v = it.curvature
                        parts.append(f"κ = {_sci(v[0])} 1/m" if np.ptp(v) == 0 else "κ")
                    lab = W["imposed"] + " " + ", ".join(parts)
                self.notes.append((u0, u1, z, lab, data))
            elif isinstance(it, TendonLoad):
                td = it.tendon
                ends = td.ends()
                pts = []
                for e, (ei, ej) in zip(td.beams, ends):
                    i, j = m.beams[e][:2]
                    pts += [(*el.node(i), ei), (*el.node(j), ej)]
                pts.sort(key=lambda p: p[0])
                f = td.forces()
                lab = it.label or f"{W['tendon']} N = {number(float(np.max(np.abs(f))))} kN"
                self.tendons.append((pts, lab, data))
            elif isinstance(it, Settlement):
                self.settlements.append((it.node, float(it.d[2]), it.label, data))
            elif isinstance(it, BarStrain):
                for e, ep in zip(it.bars, it.eps0):
                    i, j = m.trusses[e][:2]
                    a, b = el.node(i), el.node(j)
                    self.texts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2,
                                       it.label or f"{W['bar']} {_sci(ep)}", data))
        return self

    def _point(self, uz, p3, label, data, style=""):
        Pv = float(p3[2])
        Ph = float(np.dot(p3, self.el.d))
        self.points.append((uz[0], uz[1], Pv, Ph, label, data, style))

    # ── layout in pixels ──
    def _rows(self, T):
        """stack overlapping bands of the same side in rows; returns per band (base offset, height function)"""
        wref = max([b.wmax for b in self.bands], default=1.0) or 1.0
        for side in (True, False):
            bs = [b for b in self.bands if b.down == side]
            rows = []  # list of [(x0, x1)]
            for b in sorted(bs, key=lambda b: b.span[0]):
                x0, x1 = T(b.span[0], 0)[0], T(b.span[1], 0)[0]
                for r, occ in enumerate(rows):
                    if all(x1 <= o0 + 0.5 or x0 >= o1 - 0.5 for o0, o1 in occ):
                        occ.append((x0, x1))
                        b.row = r
                        break
                else:
                    rows.append([(x0, x1)])
                    b.row = len(rows) - 1
            heights = {}
            for b in bs:
                b.H = BAND_MIN + (BAND_MAX - BAND_MIN) * b.wmax / wref
                heights[b.row] = max(heights.get(b.row, 0.0), b.H)
            base, acc = {}, getattr(self, "_lift", 0.0)
            for r in sorted(heights):
                base[r] = acc
                acc += heights[r] + LABEL_ROOM + BAND_GAP
            for b in bs:
                b.base = base[b.row]

    def _band_top(self, T, u, down=True):
        """highest band offset (px) above (or below) the member at u (the girder band's half depth at least)"""
        top = getattr(self, "_lift", 0.0)
        for b in self.bands:
            if b.down != down:
                continue
            u0, u1 = b.span
            if u0 - 1e-9 <= u <= u1 + 1e-9:
                top = max(top, b.base + b.H)
        return top

    def _tendon_scale(self, T):
        """px per m of tendon eccentricity (largest eccentricity → 16 px) and the half depth of the girder band drawn
        around the tendons (0 without tendons): supports go under it, loads on top of it"""
        emax = max([abs(p[2]) for pts, _, _ in self.tendons for p in pts], default=0.0)
        if not self.tendons:
            return 0.0, 0.0
        emax = emax or 1.0
        kpx = min(16.0 / emax, T.sy * 4)
        return kpx, emax * kpx + 5

    def render(self, T):
        out, P = [], Placer()
        el = self.el
        kpx, lift = self._tendon_scale(T)
        self._lift = lift
        # supports and dimensions
        sup_svg, sup_boxes, bottoms = _supports(el, T, lift)
        out.append(sup_svg)
        for b in sup_boxes:
            P.block(b)
        # the members themselves are obstacles for labels
        for e in el.members:
            i, j = el.m.beams[e][:2]
            (x0, y0), (x1, y1) = T(*el.node(i)), T(*el.node(j))
            P.block((min(x0, x1), min(y0, y1) - 1.5, max(x0, x1), max(y0, y1) + 1.5))
        self._rows(T)
        # imposed-strain brackets go above every band over their range, with room for the bands' labels
        self._note_off = {}
        for (u0, u1, z, lab, data) in self.notes:
            self._note_off[(u0, u1)] = max([self._band_top(T, u0, True)] + [
                b.base + b.H for b in self.bands if b.down and b.span[0] < u1 and b.span[1] > u0]) + 24
        labels = []  # (priority, text, candidates, cls, droppable)
        # bands
        for b in self.bands:
            sg = -1.0 if b.down else 1.0  # page y direction of the band (up for loads pushing down)
            k = b.H / max(b.wmax, 1e-12)
            bottom, top = [], []
            for u, z, w in b.pts:
                x, y = T(u, z)
                bottom.append((x, y + sg * b.base))
                top.append((x, y + sg * (b.base + w * k)))
            body = [svg_poly(bottom + top[::-1], "loadfill", style="stroke:none;fill-opacity:0.55")]
            body.append(svg_poly(top, "load", closed=False))
            if b.base > 0:
                body.append(svg_poly(bottom, "load", closed=False, style="stroke-width:0.9"))
            # arrows, evenly spaced from end to end
            xs = [p[0] for p in bottom]
            x0, x1 = min(xs), max(xs)
            nseg = max(1, int(round((x1 - x0) / ARROW_STEP)))
            for q in range(nseg + 1):
                xa = x0 + (x1 - x0) * q / nseg
                yb = float(np.interp(xa, xs, [p[1] for p in bottom]))
                yt = float(np.interp(xa, xs, [p[1] for p in top]))
                if abs(yt - yb) >= HEAD + 1.5:
                    body.append(svg_arrow(xa, yt, xa, yb, head=5.5, style="stroke-width:1"))
            out.append(f'<g data-load="band"{b.data} data-x0="{_d(x0)}" data-x1="{_d(x1)}">{"".join(body)}</g>')
            ys = [p[1] for p in top] + [p[1] for p in bottom]
            P.block((x0, min(ys), x1, max(ys)))
            ytop = min(p[1] for p in top) if b.down else max(p[1] for p in top)
            ylab = ytop - 10 if b.down else ytop + 10
            step = -14 if b.down else 14
            if b.texts:
                xm = (x0 + x1) / 2
                labels.append((1, b.texts[0], shifted([(xm, ylab, "middle"), (x0, ylab, "start"), (x1, ylab, "end")],
                                                     step), "loadtext", False))
            else:  # a trapezoid: the end values
                (ua, za, wa), (ub, zb, wb) = b.pts[0], b.pts[-1]
                for (uu, zz, ww, anc) in ((ua, za, wa, "start"), (ub, zb, wb, "end")):
                    if ww > 1e-9 * b.wmax:
                        yy = (T(uu, zz)[1] + sg * (b.base + ww * k)) + (-10 if b.down else 10)
                        labels.append((1, f"{number(ww)} kN/m", shifted([(T(uu, zz)[0], yy, anc)], step), "loadtext",
                                       False))
        # point loads
        for (u, z, Pv, Ph, lab, data, style) in self.points:
            x, y = T(u, z)
            body = []
            if abs(Pv) > 0:
                down = Pv < 0
                over = [o for (u0, u1), o in self._note_off.items() if u0 - 1e-9 <= u <= u1 + 1e-9] if down else []
                length = max([POINT_LEN + lift, self._band_top(T, u, down) + 16] + [o + 14 for o in over])
                y0 = y - length if down else y + length
                yh = y - lift if down else y + lift
                body.append(svg_arrow(x, y0, x, yh, style=style))
                P.block((x - 1.5, min(y0, yh), x + 1.5, max(y0, yh)))
                text = lab or f"P = {number(abs(Pv))} kN"
                yl = y0 - 8 if down else y0 + 9
                labels.append((0, text, shifted([(x, yl, "middle"), (x + 5, yl + (8 if down else -8), "start"),
                                                (x - 5, yl + (8 if down else -8), "end")], -14 if down else 14),
                               "loadtext", False))
            if abs(Ph) > 0:
                yh = y - 9 - (self._band_top(T, u, True) + 4 if self._band_top(T, u, True) > 0 else 0)
                x0 = x - 30 if Ph > 0 else x + 30
                body.append(svg_arrow(x0, yh, x, yh, style=style))
                P.block((min(x0, x) - 1, yh - 3, max(x0, x) + 1, yh + 3))
                text = f"H = {number(abs(Ph))} kN" if not (lab and abs(Pv) == 0) else lab
                labels.append((0, text, [(x0 - 3, yh, "end") if Ph > 0 else (x0 + 3, yh, "start"),
                                         ((x0 + x) / 2, yh - 10, "middle")], "loadtext", False))
            out.append(f'<g data-load="point"{data} data-x="{_d(x)}" data-y="{_d(y)}">{"".join(body)}</g>')
        # moments
        for (u, z, Mv, lab, data) in self.moments:
            x, y = T(u, z)
            yc = y - 16 - (self._band_top(T, u, True) + 2 if self._band_top(T, u, True) > 0 else 0)
            out.append(f'<g data-load="moment"{data}>{moment_arc(x, yc, Mv > 0)}</g>')
            P.block((x - 14, yc - 14, x + 14, yc + 8))
            labels.append((0, lab, shifted([(x + 17, yc - 12, "start"), (x - 17, yc - 12, "end"), (x, yc - 23, "middle")],
                                          -14), "loadtext", False))
        # imposed strains: a dash-dot bracket over the beams
        for (u0, u1, z, lab, data) in self.notes:
            (x0, y), (x1, _) = T(u0, z), T(u1, z)
            off = self._note_off[(u0, u1)]
            yy = y - off
            body = [svg_line(x0, yy, x1, yy, "load", style="stroke-dasharray:7 2 1.5 2"),
                    svg_line(x0, yy - 4, x0, yy + 4, "load"), svg_line(x1, yy - 4, x1, yy + 4, "load")]
            out.append(f'<g data-load="imposed"{data}>{"".join(body)}</g>')
            P.block((x0, yy - 4, x1, yy + 4))
            labels.append((1, lab, shifted([((x0 + x1) / 2, yy - 10, "middle"), (x0, yy - 10, "start")], -14),
                           "loadtext", False))
        # tendons: the profile in a faint girder band, eccentricity scaled to at most 16 px
        for (pts, lab, data) in self.tendons:
            ratio = kpx / T.sy
            line = [(T(u, z)[0], T(u, z)[1] + e * kpx) for u, z, e in pts]
            half = lift
            axis = [T(u, z) for u, z, e in pts]
            band = [(x, y - half) for x, y in axis] + [(x, y + half) for x, y in axis[::-1]]
            out.append(f'<g data-load="tendon"{data} data-scale="{_d(kpx)}">'
                       + svg_poly(band, "concrete", style="fill-opacity:0.45;stroke-width:0.8")
                       + svg_poly(line, "curve", closed=False, style=f"stroke:{C['tendon']};stroke-width:1.6")
                       + "".join(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="2.4" fill="{C["tendon"]}"/>'
                                 for x, y in (line[0], line[-1])) + "</g>")
            xs = [p[0] for p in line]
            ytop = min(y for _, y in band)
            P.block((min(xs), ytop, max(xs), max(y for _, y in band)))
            note = lab
            if ratio > 1.05:
                note += (f"（{self.W['exaggerated']} ×{number(ratio)}）" if self.lang == "zh"
                         else f" ({self.W['exaggerated']} ×{number(ratio)})")
            labels.append((1, note, shifted([((min(xs) + max(xs)) / 2, ytop - 9, "middle"), (min(xs), ytop - 9, "start")],
                                           -14), "small", False))
        # settlements: an arrow under the support
        for (n, dz, lab, data) in self.settlements:
            x, y = T(*el.node(n))
            yb = bottoms.get(n, y + 14) + 3
            down = dz < 0
            body = svg_arrow(x, yb, x, yb + 14, head=5) if down else svg_arrow(x, yb + 14, x, yb, head=5)
            out.append(f'<g data-load="settlement"{data}>{body}</g>')
            P.block((x - 2, yb, x + 2, yb + 15))
            bottoms[n] = yb + 15
            text = lab or f"{self.W['settlement']} {number(abs(dz) * 1e3)} mm"
            labels.append((0, text, [(x + 6, yb + 8, "start"), (x - 6, yb + 8, "end")], "loadtext", False))
        for (u, z, text, data) in self.member_notes:
            x, y = T(u, z)
            labels.append((1, text + " ↓", [(x + 6, y, "start"), (x - 6, y, "end")], "loadtext", False))
        for (u, z, text, data) in self.texts:
            x, y = T(u, z)
            labels.append((2, text, [(x, y - 9, "middle"), (x, y + 10, "middle")], "small", True))
        # dimensions
        dim_y = None
        if self.dims:
            dsvg, dboxes, dim_y = _span_dims(el, T, bottoms)
            out.append(dsvg)
            for b in dboxes:
                P.block(b)
        # the section mark: a short tick across the member
        if self.section is not None:
            u, z, lab = self.section
            x, y = T(u, z)
            out.append(f'<g data-section="1" data-x="{_d(x)}">'
                       + svg_line(x, y - 9, x, y + 9, "member", style="stroke-width:1.8;stroke-linecap:butt") + "</g>")
            P.block((x - 2, y - 10, x + 2, y + 10))
        if self.extra is not None:
            esvg, eboxes, elabels = self.extra(T, P, dim_y, bottoms)
            out.append(esvg)
            for b in eboxes:
                P.block(b)
            labels += elabels
        for pr, text, cands, cls, drop in labels:
            P.add(text, cands, cls, priority=-pr, required=not drop)
        return "".join(out) + P.layout().svg(), P.boxes


def _figure(el, width, title, nodes, members_trusses=True):
    fig = Figure(width=width, title=title)
    _members_svg(fig, el, trusses=members_trusses)
    if nodes:
        used = sorted({int(n) for e in el.members for n in el.m.beams[e][:2]})
        for n in used:
            fig.dot(el.node(n), 2.0, "support")
    return fig


def _attach(fig, scene):
    cache = {}

    def run(T):
        key = (T.sx, T.H, T(0, 0))
        if key not in cache:
            cache.clear()
            cache[key] = scene.render(T)
        return cache[key]
    fig.raw(lambda T: run(T)[0], None, lambda T: run(T)[1])
    return fig


# ── public drawings ────────────────────────────────────────────────────────────────────────────────────────────────
def elevation_figure(m: Model, *, lang: str = "zh", width: float = 720, title: str | None = None, members=None,
                     nodes: bool = False, dims: bool = True) -> Figure:
    """the structure in elevation: members (beams thick, bars thin), supports and span dimensions"""
    el = Elevation(m, members)
    fig = _figure(el, width, title, nodes)
    sc = _Scene(el, lang)
    sc.dims = dims
    return _attach(fig, sc)


def load_figure(m: Model, case: LoadCase, *, lang: str = "zh", width: float = 720, title: str | None = None,
                members=None, nodes: bool = False, dims: bool = True) -> Figure:
    """the loads of a LoadCase on the structure in elevation: point loads as arrows touching the member, distributed
    loads as bands of arrows (one per load, stacked where they overlap) with their intensities, nodal moments as curved
    arrows, imposed strains and curvatures as a bracket over the beams they act on, tendons as their profile (the
    eccentricity drawn to at most 16 px, its exaggeration noted), settlements as arrows under the support. title
    defaults to the case name."""
    el = Elevation(m, members)
    fig = _figure(el, width, case.name if title is None else title, nodes)
    sc = _Scene(el, lang).add_case(case)
    sc.dims = dims
    return _attach(fig, sc)


def _section_info(il, k):
    """(s of the section's node along the path, side of its element +1 after / −1 before / 0 not on the path, node)"""
    e, end = il.sections[k]
    i, j = il.sv.m.beams[e][:2]
    here, other = (i, j) if end == "i" else (j, i)
    pos = il._pos
    if here not in pos:
        return None, 0, here
    s = float(il.s[pos[here]])
    side = 0 if other not in pos else (1 if pos[other] > pos[here] else -1)
    return s, side, here


SENSE_WORDS = {
    "zh": {("M", "max"): "最大正弯矩", ("M", "min"): "最大负弯矩", ("V", "max"): "最大正剪力", ("V", "min"): "最大负剪力",
           ("N", "max"): "最大轴拉力", ("N", "min"): "最大轴压力", ("T", "max"): "最大正扭矩", ("T", "min"): "最大负扭矩"},
    "en": {("M", "max"): "largest sagging moment", ("M", "min"): "largest hogging moment",
           ("V", "max"): "largest positive shear", ("V", "min"): "largest negative shear",
           ("N", "max"): "largest tension", ("N", "min"): "largest compression",
           ("T", "max"): "largest positive torque", ("T", "min"): "largest negative torque"},
}


def influence_figure(il, layout, *, lang: str = "zh", width: float = 720, title: str | None = None,
                     nodes: bool = False, dims: bool = True, ordinates: bool = True) -> Figure:
    """the worst placement of a moving load for one section, above, and the section's influence line, below, on the
    same x axis. layout: femotion.frame.influence.LaneLayout (the uniform load over the loaded parts only, the point
    load at its position) or AxleLayout (the axles as arrows under a simplified vehicle, cab and arrow showing the
    direction of travel). The line: positive ordinates above its axis, the parts of the loaded sign shaded darker, the
    jump of a force line at the section drawn vertical; the section is ticked on the member and a dashed line runs
    through both panels; the ordinate under the point load (or each axle) and the effect are written out.
    Loads are labelled with their characteristic values; the effect line shows the factor (impact, lanes,
    distribution) applied to them."""
    from .influence import LaneLayout
    W = _w(lang)
    m = il.sv.m
    el = Elevation(m)
    path_uz = np.array([el.node(n) for n in il.path])

    def at(s):
        return float(np.interp(s, il.s, path_uz[:, 0])), float(np.interp(s, il.s, path_uz[:, 1]))

    sym = _SYM.get(layout.comp, layout.comp)
    unit = _UNIT.get(sym, "")
    lane = isinstance(layout, LaneLayout)
    kind = layout.kind or ("load" if lane else "vehicle")
    s_sec, side, node = _section_info(il, layout.section)
    side_w = "" if (sym in ("M", "T") or side == 0) else (W["right"] if side > 0 else W["left"])
    sec_label = "" if s_sec is None else (f"x = {number(s_sec)} m" + (f"（{side_w}）" if side_w and lang == "zh" else
                                                                  f" ({side_w})" if side_w else ""))
    what = SENSE_WORDS[lang].get((sym, layout.sense), f"{sym} {layout.sense}")
    if title is None:
        if lang == "zh":
            title = f"{W.get(kind, kind)}：{sec_label} {what}"
        else:
            title = f"{W.get(kind, kind).capitalize()}: {what} at {sec_label}"
    fig = _figure(el, width, title, nodes)
    sc = _Scene(el, lang)
    sc.dims = dims
    direction = layout.direction if layout.direction is not None else np.array([0.0, 0.0, -1.0])
    dz = float(direction[2])
    # the loads of the placement
    if lane:
        sym_q, sym_P = ("q_{k}", "P_{k}") if kind == "lane" else ("q", "P")
        for s0, s1 in layout.segments:
            ss = [s for s in il.s if s0 < s < s1]
            pts = [(*at(s), abs(layout.q * dz)) for s in [s0] + ss + [s1]]
            b = _Band(pts, layout.q * dz < 0, [f"{sym_q} = {number(layout.q)} kN/m"],
                      f' data-s0="{_d(s0)}" data-s1="{_d(s1)}"')
            sc.bands.append(b)
        if layout.P_at is not None:
            u, z = at(layout.P_at)
            sc.points.append((u, z, layout.P * dz, 0.0, f"{sym_P} = {number(layout.P, 1)} kN",
                              f' data-s="{_d(layout.P_at)}"', ""))
    sc.section = None if s_sec is None else (*at(s_sec), "")

    def extra(T, Pl, dim_y, bottoms):
        """the vehicle (axle layouts), the section label and the influence-line panel below"""
        out, boxes, labels = [], [], []
        y_member = T(*at(0.0))[1]
        if s_sec is not None:
            xs_, ys_ = T(*at(s_sec))
            labels.append((0, sec_label, shifted([(xs_ + 5, ys_ + 13, "start"), (xs_ - 5, ys_ + 13, "end"),
                                                 (xs_ + 17, ys_ + 13, "start"), (xs_ - 17, ys_ + 13, "end")], 13, 1)
                           + [(xs_ + 5, ys_ - 16, "start"), (xs_ - 5, ys_ - 16, "end")], "small", False))
        # the vehicle
        if not lane and layout.positions is not None:
            on = layout.on_path()
            clamp = [min(max(p, il.s[0]), il.s[-1]) for p in layout.positions]
            xs = [T(*at(c))[0] + T.sx * (p - c) for p, c in zip(layout.positions, clamp)]
            ys = [T(*at(c))[1] for c in clamp]
            top = min(ys) - 24.0
            for k_, (x, y, o) in enumerate(zip(xs, ys, on)):
                st = "" if o else "opacity:0.35"
                out.append(f'<g data-load="axle" data-x="{_d(x)}" data-s="{_d(layout.positions[k_])}"'
                           f' data-on="{int(o)}">' + svg_arrow(x, top, x, y, style=st) + "</g>")
                boxes.append((x - 1.5, top, x + 1.5, y))
            # a simplified truck over the axles: body, cab at the leading end with a window
            dirx = 1 if xs[0] >= xs[-1] else -1
            if len(xs) == 1:
                dirx = layout.heading or 1
            pad = max(0.8 * T.sx, 4.0)
            cab_w = max(1.9 * T.sx, 12.0)
            x_front, x_rear = xs[0] + dirx * pad, xs[-1] - dirx * pad
            y1 = top - 1.5
            bh, ch = 10.0, 16.0
            bx0, bx1 = sorted((x_rear, x_front - dirx * cab_w))
            cx0, cx1 = sorted((x_front - dirx * cab_w, x_front))
            body = (f'<rect x="{fmt(bx0)}" y="{fmt(y1 - bh)}" width="{fmt(bx1 - bx0)}" height="{fmt(bh)}" '
                    f'rx="1.5" fill="#ffffff" stroke="{C["load"]}" stroke-width="1"/>')
            cab = (f'<path d="M{fmt(cx0)},{fmt(y1)} V{fmt(y1 - ch + (4 if dirx < 0 else 0))} '
                   f'L{fmt(cx0 + (4 if dirx < 0 else 0))},{fmt(y1 - ch)} H{fmt(cx1 - (4 if dirx > 0 else 0))} '
                   f'L{fmt(cx1)},{fmt(y1 - ch + (4 if dirx > 0 else 0))} V{fmt(y1)} Z" '
                   f'fill="{C["load_fill"]}" stroke="{C["load"]}" stroke-width="1"/>')
            wx = cx1 - 7 if dirx > 0 else cx0 + 2
            win = (f'<rect x="{fmt(wx)}" y="{fmt(y1 - ch + 4)}" width="5" height="5" fill="#ffffff" '
                   f'stroke="{C["load"]}" stroke-width="0.8"/>')
            out.append(f'<g data-vehicle="1" data-heading="{layout.heading}">{body}{cab}{win}</g>')
            vb = (min(bx0, cx0), y1 - ch, max(bx1, cx1), y1)
            boxes.append(vb)
            total = float(np.sum(layout.loads))
            base = f"{W['vehicle']} {number(total)} kN" if kind == "vehicle" else f"ΣP = {number(total)} kN"
            name = base + " →" if dirx > 0 else "← " + base
            labels.append((0, name, shifted([((vb[0] + vb[2]) / 2, vb[1] - 10, "middle"),
                                            (vb[0], vb[1] - 10, "start"), (vb[2], vb[1] - 10, "end")], -13),
                           "loadtext", False))
            for x, Pk, o in zip(xs, layout.loads, on):
                labels.append((2, number(Pk), [(x - 3, top + 7, "end"), (x + 3, top + 7, "start")], "small", True))
        # the influence panel
        s, eta = np.asarray(layout.s, float), np.asarray(layout.eta, float)
        emax = float(np.max(np.abs(eta))) or 1.0
        k = IL_HEIGHT / emax
        y_ref = max([b[3] for b in Pl.boxes] + [y_member + 10])
        hi = max(float(eta.max()), 0.0) * k
        y0 = y_ref + 30 + hi
        xp = np.array([T(*at(v))[0] for v in s])
        yp = y0 - eta * k
        loaded_sign = (1 if layout.sense == "max" else -1) if (lane and layout.q != 0) else 0
        for sg_r, rx, re_ in _sign_regions(xp, eta):
            ry = y0 - np.asarray(re_) * k
            loaded = sg_r == loaded_sign
            pts = [(rx[0], y0)] + list(zip(rx, ry)) + [(rx[-1], y0)]
            out.append(f'<g data-il-area="{sg_r}" data-loaded="{int(loaded)}">'
                       + svg_poly(pts, "pos" if sg_r > 0 else "neg", style=f"fill-opacity:{0.4 if loaded else 0.13}")
                       + "</g>")
            c = int(np.argmax(np.abs(np.asarray(re_))))
            hpx = abs(re_[c]) * k
            if hpx > 15 and rx[-1] - rx[0] > 30:  # + / − in the larger regions, at the centroid of the area
                wts = np.abs(np.asarray(re_))
                xm = float(np.sum(np.asarray(rx) * wts) / np.sum(wts))
                hm = float(np.interp(xm, rx, np.abs(re_))) * k
                if hm > 15:
                    out.append(svg_text(xm, y0 - sg_r * min(hm * 0.4, 14), "+" if sg_r > 0 else "−", "label",
                                        fill=C["muted"]))
        out.append(svg_line(xp[0], y0, xp[-1], y0, "axis"))
        out.append(f'<g data-il="line" data-y0="{_d(y0)}" data-scale="{_d(k)}">'
                   + svg_poly(list(zip(xp, yp)), "curve", closed=False) + "</g>")
        ybot = y0 + max(0.0, -float(eta.min())) * k
        boxes.append((xp[0], y0 - hi - 2, xp[-1], ybot + 2))
        for (xa, ya), (xb, yb) in zip(zip(xp[:-1], yp[:-1]), zip(xp[1:], yp[1:])):
            Pl.block((min(xa, xb), min(ya, yb) - 1, max(xa, xb), max(ya, yb) + 1))
        Pl.block((xp[0], y0 - 1, xp[-1], y0 + 1))
        il_name = f"{sym} {W['il']}" + (" (m)" if sym in ("M", "T") else "")
        ytop_p = y0 - hi
        labels.append((1, il_name, shifted([(xp[0], ytop_p - 4, "start"), (xp[-1], ytop_p - 4, "end"),
                                           (xp[0], y0 - 9, "start"), (xp[-1], y0 - 9, "end"),
                                           (xp[0], y0 + 10, "start"), (xp[-1], y0 + 10, "end")], -12, 2),
                       "small", False))
        # the section: a dashed guide through both panels
        if s_sec is not None:
            xs_ = T(*at(s_sec))[0]
            out.append(svg_line(xs_, y_member + 10, xs_, ybot + 4, "dim", style="stroke-dasharray:3 3"))
            Pl.block((xs_ - 1, y_member + 10, xs_ + 1, ybot + 4))
        # ordinates under the loads
        if ordinates and lane and layout.P_at is not None:
            xP = T(*at(layout.P_at))[0]
            yP = y0 - layout.eta_P * k
            out.append(f'<g data-il-ordinate="{_d(layout.eta_P)}" data-x="{_d(xP)}">'
                       + svg_line(xP, y0, xP, yP, "curve", style="stroke-width:1")
                       + f'<circle cx="{fmt(xP)}" cy="{fmt(yP)}" r="2.4" fill="{C["ink"]}"/></g>')
            up = layout.eta_P > 0
            dy = -8 if up else 9
            labels.append((0, number(layout.eta_P),
                           shifted([(xP + 5, yP + dy, "start"), (xP - 5, yP + dy, "end")], -12 if up else 12, 1)
                           + [(xP + 6, yP - dy, "start"), (xP - 6, yP - dy, "end")], "label", False))
        if ordinates and not lane and layout.positions is not None:
            onp = layout.on_path()
            for (p, o, v) in zip(layout.positions, onp, layout.ordinates):
                if o:
                    xa, ya = T(*at(p))[0], y0 - v * k
                    out.append(f'<g data-il-ordinate="{_d(v)}" data-x="{_d(xa)}">'
                               + svg_line(xa, y0, xa, ya, "dim", style="stroke-dasharray:2 2")
                               + f'<circle cx="{fmt(xa)}" cy="{fmt(ya)}" r="2" fill="{C["ink"]}"/></g>')
            best = int(np.argmax(np.abs(layout.ordinates) * onp))
            for idx, (p, o, v) in enumerate(zip(layout.positions, onp, layout.ordinates)):
                if o and abs(v) > 1e-12:
                    xa, ya = T(*at(p))[0], y0 - v * k
                    dy = -8 if v > 0 else 9
                    labels.append((0 if idx == best else 2, number(v),
                                   [(xa, ya + dy, "middle"), (xa + 4, ya + dy, "start"), (xa - 4, ya + dy, "end")],
                                   "small", idx != best))
        return "".join(out), boxes, labels

    sc.extra = extra
    _attach(fig, sc)
    # the effect written out under the drawing; a long formula breaks before + = −, continued lines indented
    return fig.notes([Note(text, tone, breaks=r"(?<= )(?=[+=−] )", indent=12.0)
                      for text, tone in _effect_lines(layout, sym, unit, W, lang)])


def _sign_regions(xp, eta):
    """the parts of a piecewise-linear line of one sign: [(sign, xs, ηs)], each closed at zero (crossings inserted,
    a jump through zero split at its x)"""
    out, cur = [], None
    for i in range(len(xp)):
        e = float(eta[i])
        sg = 1 if e > 0 else -1 if e < 0 else 0
        if cur is None:
            if sg:
                x0 = xp[i - 1] if i > 0 and float(eta[i - 1]) == 0 else xp[i]
                cur = (sg, [x0, xp[i]], [0.0, e])
            continue
        if sg == cur[0]:
            cur[1].append(xp[i]); cur[2].append(e)
        elif sg == 0:
            cur[1].append(xp[i]); cur[2].append(0.0)
            out.append(cur)
            cur = None
        else:  # a change of sign between i − 1 and i
            ep = float(eta[i - 1])
            xc = xp[i - 1] + ep / (ep - e) * (xp[i] - xp[i - 1])
            cur[1].append(xc); cur[2].append(0.0)
            out.append(cur)
            cur = (sg, [xc, xp[i]], [0.0, e])
    if cur is not None:
        cur[1].append(xp[-1]); cur[2].append(0.0)
        out.append(cur)
    return [r for r in out if len(r[1]) > 2]


def _paren(v: float) -> str:
    return f"({number(v)})" if v < 0 else number(v)


def _effect_lines(layout, sym, unit, W, lang):
    """the effect written out, [(text, tone)]: S = factor × (P η + q Ω) or factor × Σ Pi ηi"""
    from .influence import AxleLayout
    head = f"{sym}_{{{layout.sense}}} = {number(layout.value)} {unit}"
    f = layout.factor
    fac = "" if abs(f - 1) < 1e-12 else f"{f:.4g} × "
    if isinstance(layout, AxleLayout):
        if layout.positions is None:
            return [(head, ""), (W["none"], "small")]
        terms = [f"{number(P)}×{_paren(v)}" for P, v, o in zip(layout.loads, layout.ordinates, layout.on_path())
                 if o and abs(v) > 1e-12]
        body = " + ".join(terms)
        return [(head, ""), ((f"= {fac}ΣP_{{i}}η_{{i}} = {fac}(" + body + ")") if fac else
                             "= ΣP_{i}η_{i} = " + body, "small")]
    parts, syms = [], []
    kq = "_{k}" if layout.kind == "lane" else ""
    if layout.P_at is not None:
        parts.append(f"{number(layout.P, 1)}×{_paren(layout.eta_P)}")
        syms.append(f"P{kq}η")
    if layout.q != 0 and layout.segments:
        parts.append(f"{number(layout.q)}×{_paren(layout.area)}")
        syms.append(f"q{kq}Ω")
    if not parts:
        return [(head, ""), (W["none"], "small")]
    body, sy = " + ".join(parts), " + ".join(syms)
    return [(head, ""), (f"= {fac}({sy}) = {fac}({body})" if fac else f"= {sy} = {body}", "small")]
