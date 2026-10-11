"""Transverse analysis of box girders for the checks of table 6.1.3 of JTG 3362-2018: the transverse normal stresses at
the upper and lower faces of the top and bottom slabs, the in-plane principal stresses of the slabs and webs (full and
class A prestress: 6.3), or the crack widths (class B and reinforced concrete: 6.4).

The code gives no model for the transverse direction; the commentary to 6.1.3 (table 6-1) only ties each crack to its
stress. The program takes the usual one (program convention, docs/codes/JTG_3362-2018.md): a one-metre slice of the
box as a closed plane frame along the mid-planes of its plates, carried at the foot of each web (one pin, the rest
rollers: the webs are held vertically by the girder's longitudinal shear; distortion and warping of eccentric loads
are not included), solved with femotion.frame.linear. Loads per metre of girder: self-weight, surfacing and barriers,
vehicle wheels spread by 4.2.3 / 4.2.5 (femotion.codes.jtg.slab) and placed for the largest effect at each control
section, transverse tendons in the top slab (femotion.frame.prestress: primary and secondary effects), the vertical
temperature gradient (femotion.frame.thermal and appendix D).

Units: m, kN, kN/m (per metre of girder), kN·m/m; stresses MPa, tension + unless noted. Each plate has two faces:
slabs "top" and "bottom", exterior webs "outer" and "inner", interior webs "left" and "right". M is positive when it
puts the second face in tension (sagging for the slabs, the inner face of exterior webs) and is taken about the centroid
of the check section (4.2.6 depth in haunches). Build the per-metre section with rc.rect(1000, h_check, …): its top is
the first face.

    from femotion.codes.jtg import transverse as tr
    box = tr.BoxGeometry(B=12.5, H=2.5, webs=(-2.9, 2.9), t_top=0.25, t_bottom=0.22, t_web=0.5, t_root=0.55,
                         t_tip=0.2, haunch_top=(0.9, 0.3), haunch_bottom=(0.5, 0.2), carriageway=(-5.75, 5.75))
    fr = tr.TransverseFrame(box, concrete="C50")
    G = fr.self_weight() + fr.surface(2.4, -5.75, 5.75) + fr.line(10.0, -5.95) + fr.line(10.0, 5.95)
    Q = fr.wheels(h=0.10)                                  # (1 + μ) and the lane factor included; .max, .min
    P = fr.tendon([(-6.2, 0.07), (-2.9, 0.07), (0, 0.17), (2.9, 0.07), (6.2, 0.07)], force=900.0)
    print(fr.table(G), fr.table(Q.max), fr.table(P.secondary))
    tau = tr.shear_flow(box, V=3000.0, T=2500.0)            # single cell, MPa
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...frame.linear import Solver, beam_k
from ...frame.model import Model, Section
from ...frame import prestress as _pre, thermal as _thermal
from .actions import design_lanes, transverse_reduction
from .materials import concrete as _concrete
from .rc import Check, RCSection, rect
from . import slab as _slab
from .sls import crack_width as _crack_width


# ── geometry ────────────────────────────────────────────────────────────────
@dataclass
class BoxGeometry:
    """a box girder's cross-section, symmetric or not, single or multi-cell (m; x across the bridge, z up from the
    soffit). webs: web centre lines at the top slab's mid-plane, left to right; webs_bottom: at the bottom slab's mid-plane
    (default the same: vertical webs). t_web: thickness square to the web (one or one per web). t_root, t_tip: the
    cantilevers at the outer web face and at the tip. haunch_top, haunch_bottom: (length across, depth) of the haunches
    on the cell side of every web. carriageway: the curbs (x_left, x_right), default 0.5 m inside the tips."""
    B: float
    H: float
    webs: tuple
    t_top: float
    t_bottom: float
    t_web: float | tuple
    t_root: float
    t_tip: float
    webs_bottom: tuple | None = None
    haunch_top: tuple = (0.0, 0.0)
    haunch_bottom: tuple = (0.0, 0.0)
    carriageway: tuple | None = None

    def __post_init__(self):
        self.webs = tuple(float(v) for v in self.webs)
        self.webs_bottom = self.webs if self.webs_bottom is None else tuple(float(v) for v in self.webs_bottom)
        if len(self.webs) < 2 or len(self.webs_bottom) != len(self.webs):
            raise ValueError("at least two webs, given at the top and at the bottom")
        self.tw = tuple(np.broadcast_to(np.asarray(self.t_web, float), (len(self.webs),)))
        if self.carriageway is None:
            self.carriageway = (-self.B / 2 + 0.5, self.B / 2 - 0.5)
        if self.faces(0)[0] < -self.B / 2 - 1e-9 or self.faces(len(self.webs) - 1)[1] > self.B / 2 + 1e-9:
            raise ValueError("the top slab must reach the outer faces of the outer webs")

    def cantilevers(self) -> tuple[float, float]:
        """lengths of the left and right cantilevers from the outer web faces (m)"""
        return self.faces(0)[0] + self.B / 2, self.B / 2 - self.faces(len(self.webs) - 1)[1]

    @property
    def z_top(self) -> float:
        return self.H - self.t_top / 2

    @property
    def z_bottom(self) -> float:
        return self.t_bottom / 2

    def cos(self, k: int) -> float:
        """cosine of web k's angle to the vertical"""
        dx = self.webs[k] - self.webs_bottom[k]
        return (self.z_top - self.z_bottom) / math.hypot(dx, self.z_top - self.z_bottom)

    def faces(self, k: int, level: str = "top") -> tuple[float, float]:
        """(left, right) faces of web k at the top or bottom slab's mid-plane (thickness measured square to the web)"""
        xc = self.webs[k] if level == "top" else self.webs_bottom[k]
        half = self.tw[k] / (2 * self.cos(k))
        return xc - half, xc + half

    def layout(self, h: float) -> _slab.DeckLayout:
        """the top slab as a deck slab for the wheel distribution widths (4.2.3, 4.2.5); h: surfacing thickness (m)"""
        return _slab.DeckLayout([self.faces(k) for k in range(len(self.webs))], self.t_top, h, centres=list(self.webs),
                                tips=(-self.B / 2, self.B / 2))


@dataclass
class ControlSection:
    name: str
    element: int
    end: str  # "i" or "j"
    plate: str  # "top", "bottom", "web k"
    h: float  # check depth (m): 4.2.6 in haunches
    delta: float  # frame axis above the check section's centroid, along the first face's normal (m)
    faces: tuple  # (first, second)


@dataclass
class Forces:
    """section forces at the control sections of a TransverseFrame, per metre of girder: N (kN/m, tension +), M (kN·m/m,
    + puts the second face in tension) about the check section's centroid, V (kN/m, the shear square to the plate,
    femotion's sign: + upwards on the face looking back along the element for a slab); extra: stresses not carried by
    N, M (MPa, tension +, columns first and second face), e.g. the self-equilibrating temperature stresses of appendix D"""
    sections: list
    N: np.ndarray
    M: np.ndarray
    V: np.ndarray | None = None
    extra: np.ndarray | None = None
    reactions: np.ndarray | None = None  # vertical support reactions at the foot of each web (kN/m, upward +)

    def _x(self):
        return np.zeros((len(self.N), 2)) if self.extra is None else self.extra

    def _v(self):
        return np.zeros(len(self.N)) if self.V is None else self.V

    def __add__(self, o: "Forces") -> "Forces":
        r = None if self.reactions is None or o.reactions is None else self.reactions + o.reactions
        return Forces(self.sections, self.N + o.N, self.M + o.M, self._v() + o._v(), self._x() + o._x(), r)

    def __radd__(self, o):
        return self if o == 0 else self.__add__(o)

    def __mul__(self, k: float) -> "Forces":
        return Forces(self.sections, k * self.N, k * self.M, k * self._v(), k * self._x(),
                      None if self.reactions is None else k * self.reactions)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-o)

    def index(self, name: str) -> int:
        for i, s in enumerate(self.sections):
            if s.name == name:
                return i
        raise KeyError(f"no control section {name!r}; one of: " + ", ".join(s.name for s in self.sections))

    def at(self, name: str) -> tuple[float, float]:
        """(N, M) at a control section"""
        i = self.index(name)
        return float(self.N[i]), float(self.M[i])

    def shear(self, name: str) -> float:
        """V at a control section"""
        return float(self._v()[self.index(name)])

    def stress(self) -> np.ndarray:
        """(n, 2) normal stresses (MPa, tension +) at the first and second faces on the gross per-metre section"""
        h = np.array([s.h for s in self.sections])
        n = self.N / h
        m = 6 * self.M / h ** 2
        return (np.c_[n - m, n + m]) / 1e3 + self._x()


@dataclass
class WheelEffects:
    """envelopes of the vehicle (D60 4.3.1 vehicle load) at every control section: `max` and `min` of the target with
    the concurrent N and M; positions: left-wheel x of each vehicle, per section (max, min)"""
    target: str
    max: Forces
    min: Forces
    positions: list
    details: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ── consistent loads on a beam element (local x along it, w along local z) ──
_GX, _GW = np.polynomial.legendre.leggauss(4)


def _hermite(xi, L):
    """shape functions for (u_i, w_i, θy_i, u_j, w_j, θy_j); θy = −dw/dx as in frame.linear.beam_k"""
    H1 = 1 - 3 * xi ** 2 + 2 * xi ** 3
    H2 = L * (xi - 2 * xi ** 2 + xi ** 3)
    H3 = 3 * xi ** 2 - 2 * xi ** 3
    H4 = L * (-xi ** 2 + xi ** 3)
    return 1 - xi, H1, -H2, xi, H3, -H4


def _feq(L, p_local, a=0.0, b=None):
    """consistent nodal loads (local, 12) of a uniform load p_local (per length, local x, y, z) on [a, b] of the element:
    exact fixed-end forces of a prismatic Euler–Bernoulli beam"""
    b = L if b is None else b
    f = np.zeros(12)
    if b <= a:
        return f
    x = (b - a) / 2 * _GX + (a + b) / 2
    w = (b - a) / 2 * _GW
    Nu_i, Hw_i, Ht_i, Nu_j, Hw_j, Ht_j = _hermite(x / L, L)
    px, pz = p_local[0], p_local[2]
    f[0], f[6] = px * w @ Nu_i, px * w @ Nu_j
    f[2], f[4], f[8], f[10] = pz * w @ Hw_i, pz * w @ Ht_i, pz * w @ Hw_j, pz * w @ Ht_j
    return f


def _feq_point(L, P_local, a):
    f = np.zeros(12)
    Nu_i, Hw_i, Ht_i, Nu_j, Hw_j, Ht_j = _hermite(np.array([a / L]), L)
    f[0], f[6] = P_local[0] * Nu_i[0], P_local[0] * Nu_j[0]
    f[2], f[4], f[8], f[10] = (P_local[2] * v[0] for v in (Hw_i, Ht_i, Hw_j, Ht_j))
    return f


# ── the frame ───────────────────────────────────────────────────────────────
class TransverseFrame:
    """one metre of a box girder as a closed plane frame along the mid-planes of its plates (program convention, see the
    module doc): the top slab and cantilevers on the top slab's mid-plane, the bottom slab on its own, the webs between
    them; plate thickness per element (cantilever taper and haunches followed, the joints inside the web width given the
    adjoining slab's thickness at the face, no rigid zones); E = Ec, EI × 0.8 with allow_cracking (4.3.1); pinned at
    the foot of the first web, vertical rollers at the others. mesh: largest element (m). Control sections at the
    cantilever roots, the web faces and haunch ends of both slabs, the cell middles, and the top, middle and bottom of
    each web (at the haunch ends)."""

    def __init__(self, box: BoxGeometry, *, concrete="C50", mesh: float = 0.1, allow_cracking: bool = False,
                 gamma_c: float = _slab.GAMMA_RC):
        self.box, self.gamma_c, self.mesh = box, gamma_c, mesh
        self.concrete = concrete if not isinstance(concrete, str) else _concrete(concrete)
        self.E = self.concrete.Ec * 1e3  # kPa
        self.fI = 0.8 if allow_cracking else 1.0
        self.m = Model()
        self.notes = []
        self.el = []  # (plate, t, first face, second face, centre (x, z))
        self._build()
        self.sv = Solver(self.m)
        self._k = [beam_k(self.m.beams[e][2], self.sv._beam_cache[e][0]) for e in range(len(self.m.beams))]
        self.sections = self._control_sections()

    # geometry helpers
    def _t_top(self, x):
        b = self.box
        n = len(b.webs)
        hl, hd = b.haunch_top
        f0, fn = b.faces(0)[0], b.faces(n - 1)[1]
        if x < f0:
            return b.t_tip + (b.t_root - b.t_tip) * (x + b.B / 2) / (f0 + b.B / 2)
        if x > fn:
            return b.t_tip + (b.t_root - b.t_tip) * (b.B / 2 - x) / (b.B / 2 - fn)
        for k in range(n):
            fl, fr = b.faces(k)
            if fl <= x <= fr:  # joint: the adjoining slab at the face
                left = x < b.webs[k]
                if (k == 0 and left) or (k == n - 1 and not left):
                    return b.t_root
                return b.t_top + hd
        d = min(min(abs(x - v) for v in b.faces(k)) for k in range(n))
        return b.t_top + (hd * max(0.0, 1 - d / hl) if hl > 0 else 0.0)

    def _t_bottom(self, x):
        b = self.box
        hl, hd = b.haunch_bottom
        for k in range(len(b.webs)):
            fl, fr = b.faces(k, "bottom")
            if fl <= x <= fr:
                return b.t_bottom + hd
        d = min(min(abs(x - v) for v in b.faces(k, "bottom")) for k in range(len(b.webs)))
        return b.t_bottom + (hd * max(0.0, 1 - d / hl) if hl > 0 else 0.0)

    def _sec(self, t):
        return Section(E=self.E, A=t, Iy=self.fI * t ** 3 / 12, Iz=t ** 3 / 12, J=t ** 3 / 3)

    def _points(self, a, b, brk):
        pts = sorted({a, b} | {v for v in brk if a < v < b})
        out = [pts[0]]
        for p, q in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil((q - p) / self.mesh - 1e-9)))
            out += list(p + (q - p) * np.arange(1, n + 1) / n)
        return np.array(out)

    def _build(self):
        b, m = self.box, self.m
        n = len(b.webs)
        zt, zb = b.z_top, b.z_bottom
        hlt, hdt = b.haunch_top
        hlb, hdb = b.haunch_bottom
        # top slab
        brk = set(b.webs)
        for k in range(n):
            fl, fr = b.faces(k)
            brk |= {fl, fr}
            if k > 0:
                brk.add(fl - hlt)
            if k < n - 1:
                brk.add(fr + hlt)
        for k in range(n - 1):
            brk.add((b.webs[k] + b.webs[k + 1]) / 2)
        xs = self._points(-b.B / 2, b.B / 2, brk)
        self.top_nodes = [m.node((x, 0, zt)) for x in xs]
        self.top_x = xs
        self.top_el = []
        for p, q, xa, xb in zip(self.top_nodes[:-1], self.top_nodes[1:], xs[:-1], xs[1:]):
            t = self._t_top((xa + xb) / 2)
            self.top_el.append(m.beam(p, q, self._sec(t), ref=(0, 0, 1), tag="top"))
            self.el.append(("top", t, "top", "bottom", ((xa + xb) / 2, zt)))
        # bottom slab
        brk = set(b.webs_bottom)
        for k in range(n):
            fl, fr = b.faces(k, "bottom")
            brk |= {fl, fr}
            if k > 0:
                brk.add(fl - hlb)
            if k < n - 1:
                brk.add(fr + hlb)
        for k in range(n - 1):
            brk.add((b.webs_bottom[k] + b.webs_bottom[k + 1]) / 2)
        xb_ = self._points(b.webs_bottom[0], b.webs_bottom[-1], brk)
        self.bot_nodes = [m.node((x, 0, zb)) for x in xb_]
        self.bot_x = xb_
        self.bot_el = []
        for p, q, xa, xc in zip(self.bot_nodes[:-1], self.bot_nodes[1:], xb_[:-1], xb_[1:]):
            t = self._t_bottom((xa + xc) / 2)
            self.bot_el.append(m.beam(p, q, self._sec(t), ref=(0, 0, 1), tag="bottom"))
            self.el.append(("bottom", t, "top", "bottom", ((xa + xc) / 2, zb)))
        # webs, bottom to top
        self.web_nodes, self.web_el, self.web_z = [], [], []
        top_of = {round(x, 9): nd for x, nd in zip(xs, self.top_nodes)}
        bot_of = {round(x, 9): nd for x, nd in zip(xb_, self.bot_nodes)}
        for k in range(n):
            x0, x1 = b.webs_bottom[k], b.webs[k]
            zs = self._points(zb, zt, {zb + b.t_bottom / 2, zb + b.t_bottom / 2 + hdb, zt - b.t_top / 2,
                                       zt - b.t_top / 2 - hdt,
                                       (zb + b.t_bottom / 2 + hdb + zt - b.t_top / 2 - hdt) / 2})
            nodes = [bot_of[round(x0, 9)]]
            for z in zs[1:-1]:
                nodes.append(m.node((x0 + (x1 - x0) * (z - zb) / (zt - zb), 0, z)))
            nodes.append(top_of[round(x1, 9)])
            if k == 0:
                ref, faces = (-1, 0, 0), ("outer", "inner")
            elif k == n - 1:
                ref, faces = (1, 0, 0), ("outer", "inner")
            else:
                ref, faces = (-1, 0, 0), ("left", "right")
            els = []
            for p, q, za, zc in zip(nodes[:-1], nodes[1:], zs[:-1], zs[1:]):
                els.append(m.beam(p, q, self._sec(b.tw[k]), ref=ref, tag=f"web {k}"))
                xm = x0 + (x1 - x0) * ((za + zc) / 2 - zb) / (zt - zb)
                self.el.append((f"web {k}", b.tw[k], faces[0], faces[1], (xm, (za + zc) / 2)))
            self.web_nodes.append(nodes)
            self.web_el.append(els)
            self.web_z.append(zs)
        # supports and the plane
        for nd in range(len(m.X)):
            m.fix(nd, (0, 1, 0, 1, 0, 1))
        self._feet = [bot_of[round(b.webs_bottom[k], 9)] for k in range(n)]
        m.fix(self._feet[0], (1, 1, 1, 1, 0, 1))
        for k in range(1, n):
            m.fix(self._feet[k], (0, 1, 1, 1, 0, 1))

    def _control_sections(self):
        b = self.box
        n = len(b.webs)
        hlt, hdt = b.haunch_top
        hlb, hdb = b.haunch_bottom
        he_t = b.t_top + (min(hdt, hlt / 3) if hlt > 0 else 0.0)  # 4.2.6 at the face: s = haunch length
        he_b = b.t_bottom + (min(hdb, hlb / 3) if hlb > 0 else 0.0)
        out = []

        def slab_at(xs, els, x, side):
            i = int(np.argmin(np.abs(xs - x)))
            return (els[i - 1], "j") if side < 0 else (els[i], "i")

        def add(name, e_end, plate, h, delta, faces):
            out.append(ControlSection(name, int(e_end[0]), e_end[1], plate, h, delta, faces))

        top = lambda x, side: slab_at(self.top_x, self.top_el, x, side)
        bot = lambda x, side: slab_at(self.bot_x, self.bot_el, x, side)
        dt = lambda h: (h - b.t_top) / 2  # top slab axis above a check section hanging from the deck
        db = lambda h: (b.t_bottom - h) / 2
        cl, cr = b.cantilevers()
        h_root = {}
        for side, L in (("left", cl), ("right", cr)):  # 4.2.6 on the cantilever's taper: at most 1:3 counts
            if L > 1e-9:
                h_root[side] = min(b.t_root, b.t_tip + L / 3)
                if h_root[side] < b.t_root - 1e-12:
                    self.notes.append(f"{side} cantilever tapers steeper than 1:3: root checked with {h_root[side]:.3f} m "
                                      f"(4.2.6), not t_root = {b.t_root:.3f} m")
        if cl > 1e-9:
            add("left cantilever root", top(b.faces(0)[0], -1), "top", h_root["left"], dt(h_root["left"]),
                ("top", "bottom"))
        for c in range(n - 1):
            fl, fr = b.faces(c)[1], b.faces(c + 1)[0]
            mid = (b.webs[c] + b.webs[c + 1]) / 2
            add(f"top slab, web {c} face, cell {c}", top(fl, +1), "top", he_t, dt(he_t), ("top", "bottom"))
            if hlt > 0:
                add(f"top slab, haunch end, web {c}, cell {c}", top(fl + hlt, +1), "top", b.t_top, 0.0, ("top", "bottom"))
            add(f"top slab, cell {c} middle", top(mid, +1), "top", b.t_top, 0.0, ("top", "bottom"))
            if hlt > 0:
                add(f"top slab, haunch end, web {c + 1}, cell {c}", top(fr - hlt, -1), "top", b.t_top, 0.0,
                    ("top", "bottom"))
            add(f"top slab, web {c + 1} face, cell {c}", top(fr, -1), "top", he_t, dt(he_t), ("top", "bottom"))
        if cr > 1e-9:
            add("right cantilever root", top(b.faces(n - 1)[1], +1), "top", h_root["right"], dt(h_root["right"]),
                ("top", "bottom"))
        for c in range(n - 1):
            fl, fr = b.faces(c, "bottom")[1], b.faces(c + 1, "bottom")[0]
            mid = (b.webs_bottom[c] + b.webs_bottom[c + 1]) / 2
            add(f"bottom slab, web {c} face, cell {c}", bot(fl, +1), "bottom", he_b, db(he_b), ("top", "bottom"))
            if hlb > 0:
                add(f"bottom slab, haunch end, web {c}, cell {c}", bot(fl + hlb, +1), "bottom", b.t_bottom, 0.0,
                    ("top", "bottom"))
            add(f"bottom slab, cell {c} middle", bot(mid, +1), "bottom", b.t_bottom, 0.0, ("top", "bottom"))
            if hlb > 0:
                add(f"bottom slab, haunch end, web {c + 1}, cell {c}", bot(fr - hlb, -1), "bottom", b.t_bottom, 0.0,
                    ("top", "bottom"))
            add(f"bottom slab, web {c + 1} face, cell {c}", bot(fr, -1), "bottom", he_b, db(he_b), ("top", "bottom"))
        zb_h = b.z_bottom + b.t_bottom / 2 + hdb
        zt_h = b.z_top - b.t_top / 2 - hdt
        for k in range(n):
            zs, els = self.web_z[k], self.web_el[k]
            faces = self.el[els[0]][2:4]
            for nm, z, side in (("bottom", zb_h, +1), ("middle", (zb_h + zt_h) / 2, -1), ("top", zt_h, -1)):
                i = int(np.argmin(np.abs(zs - z)))
                e = (els[i - 1], "j") if side < 0 else (els[i], "i")
                add(f"web {k} {nm}", e, f"web {k}", b.tw[k], 0.0, faces)
        return out

    # solving
    def _reactions(self, U, F):
        R = self.sv.K @ U - F
        return np.array([R[6 * self._feet[k] + 2] for k in range(len(self.box.webs))])

    def _forces(self, U, feq=None, end_forces=None) -> Forces:
        N = np.zeros(len(self.sections))
        M = np.zeros(len(self.sections))
        V = np.zeros(len(self.sections))
        for s_, cs in enumerate(self.sections):
            e = cs.element
            if end_forces is not None:
                f = end_forces[e]
            else:
                i, j = self.m.beams[e][0], self.m.beams[e][1]
                T = self.sv._beam_cache[e][2]
                ue = np.r_[U[6 * i:6 * i + 6], U[6 * j:6 * j + 6]]
                f = self._k[e] @ (T @ ue) - (feq.get(e, 0.0) if feq else 0.0)
            n_, m_, v_ = (-f[0], f[4], -f[2]) if cs.end == "i" else (f[6], -f[10], f[8])
            N[s_], V[s_] = n_, v_
            M[s_] = m_ - n_ * cs.delta  # moved to the check section's centroid
        return Forces(self.sections, N, M, V)

    def _apply(self, loads: dict) -> Forces:
        """loads: element → consistent nodal loads (local, 12)"""
        F = np.zeros(self.m.ndof)
        for e, f in loads.items():
            i, j = self.m.beams[e][0], self.m.beams[e][1]
            T = self.sv._beam_cache[e][2]
            F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] += T.T @ f
        U = self.sv.solve(F)
        out = self._forces(U, loads)
        out.reactions = self._reactions(U, F)
        return out

    def _vertical_patch(self, els, xs, w, x1, x2) -> dict:
        """element loads of a downward load w (kN/m per horizontal metre) over x1…x2 on a slab chain"""
        loads = {}
        for e, xa, xb in zip(els, xs[:-1], xs[1:]):
            a, c = max(x1, xa), min(x2, xb)
            if c <= a:
                continue
            R = self.sv._beam_cache[e][1]
            L = xb - xa
            loads[e] = loads.get(e, 0.0) + _feq(L, R @ np.array([0.0, 0.0, -w]), a - xa, c - xa)
        return loads

    def self_weight(self) -> Forces:
        """the concrete's weight, γc × thickness of every element (kN/m per metre of member). The joints are counted
        once: the slab elements inside a web's width weigh nothing, and each web carries its full height (the parts
        above the top slab's and below the bottom slab's mid-planes as loads at its end nodes), so that the total is
        γc × the cross-section's area (straight-line haunches and tapers)."""
        b = self.box
        loads = {}
        joints = {"top": [b.faces(k) for k in range(len(b.webs))],
                  "bottom": [b.faces(k, "bottom") for k in range(len(b.webs))]}
        for e, (plate, t, _, _, (xm, _)) in enumerate(self.el):
            if plate in joints and any(fl < xm < fr for fl, fr in joints[plate]):
                continue
            L, R, _ = self.sv._beam_cache[e]
            loads[e] = _feq(L, R @ np.array([0.0, 0.0, -self.gamma_c * t]))
        for k, els in enumerate(self.web_el):
            ext = {els[0]: ("i", b.t_bottom / 2), els[-1]: ("j", b.t_top / 2)}
            for e, (end, d) in ext.items():
                L, R, _ = self.sv._beam_cache[e]
                P = self.gamma_c * b.tw[k] * d / b.cos(k)
                loads[e] = loads.get(e, 0.0) + _feq_point(L, R @ np.array([0.0, 0.0, -P]), 0.0 if end == "i" else L)
        return self._apply(loads)

    def surface(self, q: float, x1: float | None = None, x2: float | None = None, on: str = "top") -> Forces:
        """a downward load q (kN/m², e.g. surfacing γ × thickness) on the top (or bottom) slab between x1 and x2"""
        els, xs = (self.top_el, self.top_x) if on == "top" else (self.bot_el, self.bot_x)
        x1 = xs[0] if x1 is None else x1
        x2 = xs[-1] if x2 is None else x2
        return self._apply(self._vertical_patch(els, xs, q, x1, x2))

    def line(self, P: float, x: float, on: str = "top") -> Forces:
        """a downward line load P (kN/m, e.g. a barrier or railing) at x on the top (or bottom) slab"""
        els, xs = (self.top_el, self.top_x) if on == "top" else (self.bot_el, self.bot_x)
        k = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(els) - 1))
        e = els[k]
        R = self.sv._beam_cache[e][1]
        return self._apply({e: _feq_point(xs[k + 1] - xs[k], R @ np.array([0.0, 0.0, -P]), x - xs[k])})

    def tendon(self, profile, force: float):
        """transverse tendons in the top slab and cantilevers: profile = (x, depth below the deck surface) points (m),
        piecewise linear between them; force: effective force per metre of girder (kN/m, after losses). Returns an
        object with .total, .primary and .secondary Forces (femotion.frame.prestress: secondary = total − primary;
        on the check sections N_p2, M_p2 for transverse_normal_stresses)."""
        px, pd = np.asarray(profile, float).T
        sel = [k for k, (xa, xb) in enumerate(zip(self.top_x[:-1], self.top_x[1:]))
               if xa >= px[0] - 1e-9 and xb <= px[-1] + 1e-9]
        if not sel:
            raise ValueError("the tendon profile covers no element of the top slab")
        beams = [self.top_el[k] for k in sel]
        xn = np.r_[self.top_x[sel[0]], self.top_x[[k + 1 for k in sel]]]
        ecc = np.interp(xn, px, pd) - self.box.t_top / 2  # below the top slab's axis
        td = _pre.Tendon(beams, ecc=ecc, force=force)
        F = _pre.equivalent_loads(self.sv, td)
        U = self.sv.solve(F)
        tot = self._forces(U)
        tot.reactions = self._reactions(U, F)
        prim = self._forces(None, end_forces=_pre.primary_end_forces(self.sv, td))
        return _TendonEffects(tot, prim, tot - prim)

    def temperature(self, profile, alpha: float = 10e-6) -> Forces:
        """a temperature profile through the depth (femotion.codes.jtg.temperature_gradient of the girder's depth H in
        mm, z from the deck surface; the negative gradient: profile.scaled(−0.5)) acting on every element at its depth:
        slabs through their thickness (free strain and curvature of appendix D on the gross section), webs as an axial
        strain α T at their mid-length depth; a slab element's free strain is taken at the frame's axis (its own centroid
        lies lower in haunches and cantilever roots: ε − κ y by plane sections). Returns the restraint forces of the
        frame (secondary N′, M′) with the self-equilibrating stresses of appendix D at the slab control sections in
        Forces.extra."""
        from .temperature import Profile, thermal_stresses
        b = self.box
        free = {}
        for e, (plate, t, _, _, (_, zm)) in enumerate(self.el):
            if plate in ("top", "bottom"):
                top_depth = (b.H - (zm + b.t_top / 2)) if plate == "top" else (b.H - (zm - b.t_bottom / 2) - t)
                pr = Profile(tuple(z - top_depth * 1e3 for z in profile.z), profile.t)
                th = thermal_stresses(rect(1000.0, t * 1e3, concrete=self.concrete), pr, alpha=alpha, kind="gross")
                eps, kap = th.free_strain()
                # the element's axis lies on the slab's mid-plane, off its own centroid by y (above +): plane sections
                y = (t - b.t_top) / 2 if plate == "top" else -(t - b.t_bottom) / 2
                free[e] = (eps - kap * y, kap)
            else:
                free[e] = (alpha * float(profile((b.H - zm) * 1e3)), 0.0)
        es = list(free)
        eps, kap = [free[e][0] for e in es], [free[e][1] for e in es]
        U, ef = _thermal.solve(self.sv, es, eps, kap)
        out = self._forces(U, end_forces=ef)
        fr = _thermal.restrained_end_forces(self.sv, es, eps, kap)
        F = np.zeros(self.m.ndof)
        for e in es:
            i, j = self.m.beams[e][0], self.m.beams[e][1]
            F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] -= self.sv._beam_cache[e][2].T @ fr[e]
        out.reactions = self._reactions(U, F)
        extra = np.zeros((len(self.sections), 2))
        for s_, cs in enumerate(self.sections):
            if cs.plate in ("top", "bottom"):
                top_depth = 0.0 if cs.plate == "top" else b.H - cs.h
                pr = Profile(tuple(z - top_depth * 1e3 for z in profile.z), profile.t)
                th = thermal_stresses(rect(1000.0, cs.h * 1e3, concrete=self.concrete), pr, alpha=alpha, kind="gross")
                e = 1e-6  # mm inside the faces, where a profile may end exactly
                extra[s_] = (-th.sigma(e), -th.sigma(cs.h * 1e3 - e))  # self-equilibrating part (tension +)
        out.extra = extra
        return out

    # vehicles
    def wheel_influence(self, h: float, dx: float = 0.05):
        """per-metre responses (N, M, V at every control section) to one wheel's unit load spread over b1 + 2h at every
        wheel centre on the carriageway: (xs, {contact (a1, b1): (IL_N, IL_M, IL_V)}), arrays (sections, positions)"""
        b = self.box
        cl, cr = b.carriageway
        xs = np.arange(cl + _slab.CURB_CLEARANCE, cr - _slab.CURB_CLEARANCE + 1e-9, dx)
        out = {}
        for a1, b1 in sorted({(g.a1, g.b1) for g in _slab.axle_groups()}):
            bb = _slab.spread(b1, h)
            F = np.zeros((self.m.ndof, len(xs)))
            feqs = []
            for p, x in enumerate(xs):
                loads = self._vertical_patch(self.top_el, self.top_x, 1.0 / bb, x - bb / 2, x + bb / 2)
                for e, f in loads.items():
                    i, j = self.m.beams[e][0], self.m.beams[e][1]
                    T = self.sv._beam_cache[e][2]
                    F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6], p] += T.T @ f
                feqs.append(loads)
            U = np.zeros_like(F)
            U[self.sv.free] = self.sv._lu.solve(F[self.sv.free])
            ILN = np.zeros((len(self.sections), len(xs)))
            ILM = np.zeros_like(ILN)
            ILV = np.zeros_like(ILN)
            for p in range(len(xs)):
                fo = self._forces(U[:, p], feqs[p])
                ILN[:, p], ILM[:, p], ILV[:, p] = fo.N, fo.M, fo.V
            out[(a1, b1)] = (ILN, ILM, ILV)
        return xs, out

    def wheels(self, h: float, *, target: str = "M", mu: float = _slab.LOCAL_IMPACT, lanes: int | None = None,
               two_way: bool | None = None, lane_factors: bool = True, dx: float = 0.05) -> WheelEffects:
        """the vehicle load (D60 4.3.1) on the carriageway, per metre of girder, enveloped at every control section.
        Each wheel's load per metre is the heaviest of the vehicle's overlapping axle groups at its distribution width
        (slab.DeckLayout.width: 4.2.5 on the cantilevers, 4.2.3-4 over a web, 4.2.3-5 near and 4.2.3-2 away from the
        webs inside a cell; program convention: the width at the wheel's position for every effect, as figure 4-6 of the
        commentary draws it for box top slabs), spread over b1 + 2h (h: surfacing thickness, m), × (1 + μ) (μ = 0.3,
        D60 4.3.2 item 6; the serviceability combinations take no impact, 6.1.1: mu=0). On a cantilever beyond lc =
        2.5 m (outside 4.2.5) the Sawko formula of the commentary also counts and the larger load per metre governs
        (slab.wheel_line_loads; noted). Vehicles side by side (1.8 m track, 1.3 m apart, outer wheels ≥ 0.5 m from the
        curbs), at most `lanes`, by default table 4.3.1-4 for the carriageway width with `two_way` (one-way or two-way
        traffic, required then), × table 4.3.1-5 (lane_factors; 1.2 for one vehicle). target: "M", "N", "V",
        "top"/"first" or "bottom"/"second" (face stress on the gross section): the quantity enveloped; the other forces
        are the concurrent ones. The frame on rigid supports leaves out the box's torsion and distortion under eccentric
        wheels (noted)."""
        b = self.box
        cl, cr = b.carriageway
        if lanes is None:
            if two_way is None:
                raise ValueError("pass two_way (one-way or two-way traffic, table 4.3.1-4) or lanes")
            lanes = design_lanes(cr - cl, two_way=two_way)
        xs, ils = self.wheel_influence(h, dx)
        lay = b.layout(h)
        groups = _slab.axle_groups()
        nsec = len(self.sections)
        # heaviest load per metre at every position, with its contact
        wmax = np.zeros(len(xs))
        cont = [None] * len(xs)
        rule = [""] * len(xs)
        lc_max = 0.0
        for p, x in enumerate(xs):
            for w, g, why, clause, lc in lay.loads(float(x), groups):
                if clause == "4.2.5":
                    lc_max = max(lc_max, lc)
                if w > wmax[p]:
                    wmax[p], cont[p], rule[p] = w, (g.a1, g.b1), why
        RN = np.zeros((nsec, len(xs)))
        RM = np.zeros_like(RN)
        RV = np.zeros_like(RN)
        for p in range(len(xs)):
            ILN, ILM, ILV = ils[cont[p]]
            RN[:, p] = (1 + mu) * wmax[p] * ILN[:, p]
            RM[:, p] = (1 + mu) * wmax[p] * ILM[:, p]
            RV[:, p] = (1 + mu) * wmax[p] * ILV[:, p]
        k = int(round(_slab.WHEEL_TRACK / dx))
        nv = len(xs) - k
        if nv <= 0:
            raise ValueError("the carriageway is too narrow for a vehicle")
        hh = np.array([s.h for s in self.sections])[:, None]
        tgt = {"M": RM, "N": RN, "V": RV, "top": RN / hh - 6 * RM / hh ** 2, "first": RN / hh - 6 * RM / hh ** 2,
               "bottom": RN / hh + 6 * RM / hh ** 2, "second": RN / hh + 6 * RM / hh ** 2}
        if target not in tgt:
            raise ValueError("target: 'M', 'N', 'V', 'first' ('top') or 'second' ('bottom')")
        R = tgt[target]
        mx = Forces(self.sections, np.zeros(nsec), np.zeros(nsec), np.zeros(nsec))
        mn = Forces(self.sections, np.zeros(nsec), np.zeros(nsec), np.zeros(nsec))
        pos = []
        vx = xs[:nv]
        for s_ in range(nsec):
            pr = []
            for sense, F in ((1, mx), (-1, mn)):
                _, kk, ps = _slab.best_row(vx, R[s_, :nv] + R[s_, k:k + nv], kmax=lanes, lane_factors=lane_factors,
                                             sense=sense)
                fac = transverse_reduction(kk) if (lane_factors and kk) else 1.0
                idx = [int(round((p_ - vx[0]) / dx)) for p_ in ps]
                F.N[s_] = fac * sum(RN[s_, i] + RN[s_, i + k] for i in idx)
                F.M[s_] = fac * sum(RM[s_, i] + RM[s_, i + k] for i in idx)
                F.V[s_] = fac * sum(RV[s_, i] + RV[s_, i + k] for i in idx)
                pr.append(ps)
            pos.append(tuple(pr))
        notes = ["the frame on rigid supports leaves out the box's torsion and distortion under eccentric wheels"]
        if lc_max > 2.5:
            n_s = sum(r.startswith("Sawko") for r in rule)
            notes.append(f"wheels on a cantilever with lc up to {lc_max:.2f} m > 2.5 m, outside 4.2.5: the Sawko formula "
                         f"(commentary, 4-5) governs at {n_s} of {len(xs)} wheel positions")
        return WheelEffects(target, mx, mn, pos, {"lanes": lanes, "mu": mu, "xs": xs, "w": wmax, "rule": rule,
                                                  "lc_max": lc_max}, notes + self.notes)

    def table(self, f: Forces, title: str = "") -> str:
        """the forces and the face stresses as a text table"""
        st = f.stress()
        rows = [title] if title else []
        rows.append(f"{'section':44s} {'h':>6s} {'N kN/m':>9s} {'M kN·m/m':>10s} {'σ face 1':>9s} {'σ face 2':>9s}")
        for i, s in enumerate(self.sections):
            rows.append(f"{s.name:44s} {s.h:6.3f} {f.N[i]:9.1f} {f.M[i]:10.2f} {st[i, 0]:9.2f} {st[i, 1]:9.2f}"
                        f"  ({s.faces[0]}/{s.faces[1]})")
        return "\n".join(rows)


@dataclass
class _TendonEffects:
    total: Forces
    primary: Forces
    secondary: Forces


def box_detailing(box: BoxGeometry, *, transverse_prestress: bool = False, vertical_prestress: bool = False) -> Check:
    """the box's plate dimensions against 9.3.2 (items 1–3): cantilever tips ≥ 100 mm, 140 mm with transverse
    prestress (the code writes it for precast T and box girders; program convention: checked for every box); top and
    bottom slabs ≥ their clear span between the web faces / 30 and ≥ 200 mm at mid-span; webs ≥ 160 mm and their height
    between the haunches ≤ 15 web widths (20 with vertical prestress); the haunches item 2 asks for (notes). Returns a
    Check (resistance 1, demand 0) whose `failed` lists the unmet rules."""
    b = box
    chk = Check("box detailing (9.3.2)", 0.0, 1.0, "-", "JTG 3362-2018 9.3.2", {})
    if b.haunch_top[0] <= 0 or b.haunch_top[1] <= 0:
        chk.notes.append("no haunch between the top slab and the webs: 9.3.2 item 2 asks for one")
    if b.haunch_bottom[0] <= 0 or b.haunch_bottom[1] <= 0:
        chk.notes.append("no fillet between the bottom slab and the webs: 9.3.2 item 2 asks for a chamfer (a haunch "
                         "where needed)")
    tip = 0.140 if transverse_prestress else 0.100
    if min(b.cantilevers()) > 1e-9 and b.t_tip < tip - 1e-12:
        chk.failed.append(f"cantilever tip {b.t_tip * 1e3:.0f} mm < {tip * 1e3:.0f} mm (9.3.2 item 1)")
    for nm, t, lvl in (("top", b.t_top, "top"), ("bottom", b.t_bottom, "bottom")):
        for k in range(len(b.webs) - 1):
            ln = b.faces(k + 1, lvl)[0] - b.faces(k, lvl)[1]
            need = max(ln / 30, 0.200)
            chk.details[f"{nm} slab, cell {k}: ln/30"] = ln / 30
            if t < need - 1e-12:
                chk.failed.append(f"{nm} slab {t * 1e3:.0f} mm < max(ln/30, 200) = {need * 1e3:.0f} mm in cell {k} "
                                  "(9.3.2 item 2)")
    hw = (b.z_top - b.t_top / 2 - b.haunch_top[1]) - (b.z_bottom + b.t_bottom / 2 + b.haunch_bottom[1])
    lim = 20 if vertical_prestress else 15
    for k, tw in enumerate(b.tw):
        if tw < 0.160:
            chk.failed.append(f"web {k} {tw * 1e3:.0f} mm < 160 mm (9.3.2 item 3)")
        h_k = hw / b.cos(k)
        if h_k > lim * tw:
            chk.failed.append(f"web {k}: {h_k * 1e3:.0f} mm between the haunches > {lim} × {tw * 1e3:.0f} mm "
                              "(9.3.2 item 3)")
    return chk


# ── shear flow of a single-cell box (thin-walled) ───────────────────────────
def shear_flow(box: BoxGeometry, V: float, T: float = 0.0, *, n: int = 400) -> dict:
    """shear stresses (MPa) in a single-cell box under a vertical shear V (kN, through the shear centre) and a torque T
    (kN·m), by thin-walled theory on the plates' mid-planes (program convention: haunches ignored, the cantilevers'
    taper followed): the open-section flow of V from the free edges and from the plane of symmetry, where it vanishes
    (the box must be symmetric), and Bredt's T / (2 A0) around the cell (the cantilevers carry none). The usual
    centre-line idealisation: the slabs meet the webs at the webs' centre lines, a cantilever runs from there to its tip.
    Returns, for the points "cantilever root" (at the web's outer face), "top slab at web" and "bottom slab at web" (at
    its inner face), "top slab middle", "web top" and "web bottom" (at the slabs' faces), "web at centroid", "bottom
    slab middle": {"tau_V", "tau_T", "tau": |τV| + |τT| (the web and the half of each slab where the two add), "t"},
    plus "I" (m⁴), "A0" (m²) and "z_centroid" (m) of the model."""
    b = box
    if len(b.webs) != 2:
        raise NotImplementedError("shear_flow: single-cell boxes only (multi-cell boxes need one closing flow per cell)")
    if not (math.isclose(b.webs[0], -b.webs[1], abs_tol=1e-9) and math.isclose(b.webs_bottom[0], -b.webs_bottom[1],
                                                                                 abs_tol=1e-9) and b.tw[0] == b.tw[1]):
        raise NotImplementedError("shear_flow: the box must be symmetric about x = 0")
    zt, zb = b.z_top, b.z_bottom
    xw, xwb = b.webs[1], b.webs_bottom[1]
    f_out = b.faces(1)[1]
    # half section (x ≥ 0): (segments as point lists with thickness)
    f_in, fb_in = b.faces(1)[0], b.faces(1, "bottom")[0]
    lc = b.B / 2 - f_out

    def grid(a, c, *nodes):  # a … c through the given nodes (the faces), n points between each
        pts = [a] + sorted(v for v in nodes if min(a, c) < v < max(a, c))[::1 if c > a else -1] + [c]
        return np.concatenate([np.linspace(p, q, n)[:-1] for p, q in zip(pts[:-1], pts[1:])] + [[c]])

    s_c = grid(b.B / 2, xw, f_out)  # cantilever from the tip to the web's centre line (none: zero thickness)
    t_c = np.where(s_c >= f_out - 1e-12, b.t_tip + (b.t_root - b.t_tip) * (b.B / 2 - s_c) / lc, b.t_root) \
        if lc > 1e-9 else np.zeros(len(s_c))
    s_t = grid(0.0, xw, f_in)  # top slab from the axis
    zs_w = grid(zt, zb, zt - b.t_top / 2, zb + b.t_bottom / 2)  # web from the top
    s_b = grid(0.0, xwb, fb_in)
    lw = math.hypot(xw - xwb, zt - zb)

    def trap(y, x):
        return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))

    def cum(y, x):
        return np.r_[0.0, np.cumsum((y[1:] + y[:-1]) / 2 * np.abs(np.diff(x)))]

    # area and first moment of the half section (thin-walled, own inertia of the slabs included in I)
    A = trap(t_c, -s_c) + b.t_top * xw + b.tw[1] * lw + b.t_bottom * xwb
    Qz = trap(t_c, -s_c) * zt + b.t_top * xw * zt + b.tw[1] * lw * (zt + zb) / 2 + b.t_bottom * xwb * zb
    zc = Qz / A
    I_half = (trap(t_c, -s_c) * (zt - zc) ** 2 + trap(t_c ** 3 / 12, -s_c) + b.t_top * xw * (zt - zc) ** 2
              + xw * b.t_top ** 3 / 12 + b.t_bottom * xwb * (zb - zc) ** 2 + xwb * b.t_bottom ** 3 / 12
              + b.tw[1] * lw * (((zt + zb) / 2 - zc) ** 2 + (zt - zb) ** 2 / 12))
    I = 2 * I_half
    k = V / I
    q_c = k * cum(t_c * (zt - zc), s_c)  # kN/m from the tip
    q_t = k * cum(np.full(len(s_t), b.t_top) * (zt - zc), s_t)  # from the axis
    q_w = q_c[-1] + q_t[-1] + k * (lw / (zt - zb)) * cum(b.tw[1] * (zs_w - zc), zs_w)
    q_b = k * cum(np.full(len(s_b), b.t_bottom) * (zb - zc), s_b)
    A0 = (xw + xwb) * (zt - zb)  # the cell's mid-plane area (trapezoid, both halves)
    qT = T / (2 * A0)
    at_w = lambda z: float(np.interp(z, zs_w[::-1], q_w[::-1]))
    t_face = float(np.interp(f_out, s_c[::-1], t_c[::-1])) if lc > 1e-9 else b.t_root
    pts = {"cantilever root": (float(np.interp(f_out, s_c[::-1], q_c[::-1])), 0.0, t_face),
           "top slab at web": (float(np.interp(f_in, s_t, q_t)), qT, b.t_top),
           "top slab middle": (q_t[0], qT, b.t_top), "web top": (at_w(zt - b.t_top / 2), qT, b.tw[1]),
           "web at centroid": (at_w(zc), qT, b.tw[1]), "web bottom": (at_w(zb + b.t_bottom / 2), qT, b.tw[1]),
           "bottom slab at web": (float(np.interp(fb_in, s_b, q_b)), qT, b.t_bottom),
           "bottom slab middle": (q_b[0], qT, b.t_bottom)}
    out = {nm: {"tau_V": qv / t / 1e3, "tau_T": qt / t / 1e3, "tau": (abs(qv) + abs(qt)) / t / 1e3, "t": t}
           for nm, (qv, qt, t) in pts.items()}
    out["I"], out["A0"], out["z_centroid"] = I, A0, zc
    out["closure"] = (q_w[-1] + q_b[-1]) / max(abs(q_w).max(), 1e-12)  # flow balance at the bottom corner (≈ 0)
    return out


# ── stresses and checks ─────────────────────────────────────────────────────
def plate_principal(sigma_x: float, sigma_y: float, tau: float) -> dict:
    """principal stresses in a plate's mid-plane by 6.3.3-1 (compression +): σtp (returned as a tension, +), σcp
    (compression +) and the angle (°) of σcp from the x axis"""
    m, r = (sigma_x + sigma_y) / 2, math.hypot((sigma_x - sigma_y) / 2, tau)
    ang = 0.5 * math.degrees(math.atan2(2 * tau, sigma_x - sigma_y))
    return {"sigma_tp": r - m, "sigma_cp": m + r, "angle": ang}


def inplane_principal(points, *, concrete, pc_class: str, precast: bool = False) -> Check:
    """in-plane principal tension of the slabs or webs of a box (table 6.1.3, full and class A; class B too: 6.3.1 item 2)
    against 6.3.1-5 … -8. points: (σx, σy, τ) in MPa, compression + (σx along the girder from the longitudinal
    analysis with prestress, σy across from the transverse frame and transverse prestress, τ from shear_flow and the
    bent tendons), each from the frequent combination. precast=False: cast in place (including precast segmental)."""
    c = concrete if not isinstance(concrete, str) else _concrete(concrete)
    k = str(pc_class).upper()
    if k == "FULL":
        f, cl = (0.6, "6.3.1-5") if precast else (0.4, "6.3.1-6")
    elif k in ("A", "B"):
        f, cl = (0.7, "6.3.1-7") if precast else (0.5, "6.3.1-8")
    else:
        raise ValueError("pc_class: 'full', 'A' or 'B'")
    res = [plate_principal(*p) | {"point": tuple(p)} for p in points]
    w = max(res, key=lambda r: r["sigma_tp"])
    return Check("in-plane principal tension (table 6.1.3)", w["sigma_tp"], f * c.ftk, "MPa", f"JTG 3362-2018 {cl}",
                 {"points": res, "governing": w}, limit_check=True, symbol="σtp")


def inplane_principal_compression(points, *, concrete) -> Check:
    """in-plane principal compression under the characteristic combination (7.1.6-1): σcp ≤ 0.6 fck. points as for
    inplane_principal, from Mk, Vk and the characteristic transverse forces"""
    c = concrete if not isinstance(concrete, str) else _concrete(concrete)
    res = [plate_principal(*p) | {"point": tuple(p)} for p in points]
    w = max(res, key=lambda r: r["sigma_cp"])
    return Check("in-plane principal compression", w["sigma_cp"], 0.6 * c.fck, "MPa", "JTG 3362-2018 7.1.6-1",
                 {"points": res, "governing": w}, limit_check=True, symbol="σcp")


def transverse_normal_stresses(sec: RCSection, Ns: float, Ms: float, Nl: float, Ml: float, *, pc_class: str,
                               segmental: bool = False, N_p2: float = 0.0, M_p2: float = 0.0,
                               extra=(0.0, 0.0)) -> Check:
    """transverse normal stresses at the upper and lower faces of a box slab (table 6.1.3, full and class A prestress:
    6.3.1 item 1), on the per-metre section sec (rc.rect(1000, h_check, …) with the transverse tendons per metre and
    their σpe): σst = Ns/A0 + Ms y0/I0 and σlt likewise from the frequent (Ns, Ms) and quasi-permanent (Nl, Ml) forces
    of the frame (kN/m tension +, kN·m/m sagging +; program convention: 6.3.2 with the axial force the frame carries,
    as the commentary to 6.3.3 asks for actions with axial force); σpc = the section's prestress (6.1.6 with Mp2) less
    N_p2 / An, the secondary axial force (tension +) of the closed frame. extra: further tensile stresses (MPa, top and
    bottom face) added to σst, e.g. 0.8 × the temperature gradient's (Forces.stress()).
    full: σst − 0.85 σpc ≤ 0 (0.80 segmental; 1.0 where σpc is a tension); A: σst − σpc ≤ 0.7 ftk and σlt − σpc ≤ 0."""
    if not sec.prestressed:
        raise ValueError("table 6.1.3 asks for these stresses on full and class A prestressed slabs; reinforced concrete "
                         "and class B: transverse_crack_width")
    k = str(pc_class).upper()
    if k not in ("FULL", "A"):
        raise ValueError("pc_class 'full' or 'A' (class B: transverse_crack_width)")
    A0, z0, I0 = sec.props("transformed")
    An = sec.props("transformed" if sec.pretensioned else "net")[0]
    ftk = sec.concrete.ftk
    edges = {}
    for (name, z), ex in zip((("top", 0.0), ("bottom", sec.h)), extra):
        t = (z - z0) / I0
        e = {"sigma_st": Ns * 1e3 / A0 + Ms * 1e6 * t + ex, "sigma_lt": Nl * 1e3 / A0 + Ml * 1e6 * t,
             "sigma_pc": float(sec.prestress_stress(z, M_p2)) - N_p2 * 1e3 / An}
        if k == "FULL":
            fac = (0.80 if segmental else 0.85) if e["sigma_pc"] > 0 else 1.0
            e["demand"], e["limit"] = e["sigma_st"], fac * e["sigma_pc"]
        else:
            e["demand"], e["limit"] = e["sigma_st"], e["sigma_pc"] + 0.7 * ftk
        edges[name] = e
    gov = max(edges, key=lambda n: edges[n]["demand"] - edges[n]["limit"])
    g = edges[gov]
    cl = ("6.3.1-2" if segmental else "6.3.1-1") if k == "FULL" else "6.3.1-3, -4"
    chk = Check(f"transverse normal stress ({'full prestress' if k == 'FULL' else 'class A'}, {gov} face)", g["demand"],
                g["limit"], "MPa", f"JTG 3362-2018 table 6.1.3, {cl}", {"edge": gov, **edges}, limit_check=True,
                symbol="σst")
    for n, e in edges.items():
        if n != gov and e["demand"] > e["limit"] + 1e-9:
            chk.failed.append(f"{n}: σst = {e['demand']:.2f} > {e['limit']:.2f} MPa")
        if k == "A" and e["sigma_lt"] > e["sigma_pc"] + 1e-9:
            chk.failed.append(f"{n}: σlt = {e['sigma_lt']:.2f} > σpc = {e['sigma_pc']:.2f} MPa (6.3.1-4)")
    return chk


def transverse_crack_width(sec: RCSection, Ns: float, Ms: float, Nl: float, Ml: float, *, env: str = "I",
                           axial: str = "tension", **kw) -> Check:
    """crack width of a reinforced concrete or class B box slab or web across the girder (table 6.1.3, 6.4), per metre
    (sec = rc.rect(1000, h_check, …); class B: pass pc_class="B"), through sls.crack_width, with C3 of slabs (1.15) in
    flexure. axial: "tension" (default, program convention: a tensile Ns is counted, eccentric tension; a compressive
    one, favourable to σss, is left out and the slab taken in flexure), "all" (compression too: eccentric compression)
    or "none". Ns, Nl tension + (kN/m); Ms, Ml sagging + (kN·m/m); further arguments go to sls.crack_width (class B
    with an axial force: its program convention there)."""
    use = {"tension": Ns > 0, "all": Ns != 0, "none": False}[axial]
    if use:
        return _crack_width(sec, Ms, Ml, N=-Ns, Nl=-Nl if Nl else None, env=env, slab=True, **kw)
    return _crack_width(sec, Ms, Ml, env=env, slab=True, **kw)
