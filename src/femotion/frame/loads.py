"""Load cases of frame models: what acts on the structure, recorded as data, solved, combined with factors and drawn.

A LoadCase is a named list of loads that refer to the model by node and beam ids (as femotion.frame.prestress.Tendon
does); it does not live in the Model. The model stays the shared modelling format of both solvers, one model carries any
number of cases, and a case also applies to another model with the same numbering (the same girder with cracked
stiffness, a construction stage). What a case can hold:

    nodal        forces and moments at a node (global axes)
    point        a point load on a chain of beams at a distance s along it (on an element, or on a node)
    line         a distributed load along a chain of beams: uniform, over part of it, trapezoidal (linear in s)
    self_weight  Section.mass × g on the beams (the bars' weight lumped to their ends), as uniform_load_forces
    imposed      free axial strain and curvature of beams: temperature, shrinkage (femotion.frame.thermal)
    tendon       a prestressing tendon: its equivalent loads (femotion.frame.prestress), the total effect
    settlement   imposed displacements of restrained degrees of freedom (support settlement)
    bar_strain   initial strain of bars, pretension + (as Solver.pretension)

    from femotion.frame.loads import LoadCase, combine
    deck = list(range(64))                                         # the beams of the deck, in order
    dead = LoadCase("自重").self_weight()
    sdl = LoadCase("二期恒载").line(deck, 5.4)                      # kN/m, downwards by default
    crane = LoadCase("吊机").point(deck, 12.5, 300).line(deck, 20, start=4, end=9, q_end=35)
    M = (1.2 * dead + 1.2 * sdl).section_forces(sv, chain_sections(deck), "My")
    r = crane.solve(sv)                                            # displacements, end forces, reactions
    r.U, r.section_forces(secs, "Vz"), r.reactions()

Exact section forces at the element ends: a load inside an element is turned into its fixed-end forces (the consistent
loads of the cubic beam element, exact for prismatic Euler-Bernoulli members), applied to the nodes, and the fixed-end
forces are taken off the element forces again: end forces = K u − f (as uniform_load_forces; full-length uniform loads go
through the very same formula). Imposed strains add their restrained forces (as femotion.frame.thermal), tendons act
through their equivalent nodal loads (prestress.effects' total effect), settlements through K u with the prescribed
displacements.

Directions: a scalar load acts along `direction` (default straight down, (0, 0, −1)); a 3-vector is taken as it is,
in global axes, or with axes="local" in each element's local axes (femotion.frame.model.local_axes). Distributed loads
are per unit length of the member (not of its horizontal projection). Positions s run along the chain from the free end
of its first beam; a point load exactly on a node is a nodal load, unless side = −1 / +1 puts it on the element before /
after the node (it then belongs to that element's end force: the two sides of a jump in shear).

Combinations are linear: k * case scales every load, case + case joins them (combine((1.2, dead), (1.4, live))).

Other solvers: nodal_loads(m) and restrained_end_forces(m) need only the model, so the same case runs on
femotion.frame.opensees (linear: static(case.nodal_loads(m)), end forces = beam_end_forces() + restrained_end_forces(m));
element_loads(m) gives the span loads per element in local axes with their positions, the form of OpenSees' element
loads (beamUniform over a part of the element, beamPoint), and settlements are prescribed displacements (sp).
Units and signs as femotion.frame: m, kN, kN·m; section forces as femotion.frame.post.chain_force.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .influence import _fixed_end, section_forces
from .model import Model, local_axes
from .prestress import Tendon, equivalent_loads
from .thermal import restrained_end_forces as _thermal_restrained

DOWN = (0.0, 0.0, -1.0)
_GAUSS = np.polynomial.legendre.leggauss(3)  # exact for the cubic shape functions times a linear load


# ── what a case holds ──────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class NodalLoad:
    """forces F (3) and moments M (3) at a node, global axes"""
    node: int
    F: np.ndarray
    M: np.ndarray
    label: str | None = None

    def scaled(self, k):
        return replace(self, F=self.F * k, M=self.M * k)


@dataclass
class PointLoad:
    """a point load P (3) on a chain of beams at distance s along it. side: at a node, −1 / +1 puts the load on the
    element before / after the node along the chain, 0 on the node itself"""
    beams: tuple
    s: float
    P: np.ndarray
    side: int = 0
    axes: str = "global"
    label: str | None = None

    def scaled(self, k):
        return replace(self, P=self.P * k)


@dataclass
class LineLoad:
    """a distributed load along a chain of beams from distance `start` to `end`, w0 per length at the start and w1 at
    the end (3 each), linear in between"""
    beams: tuple
    w0: np.ndarray
    w1: np.ndarray
    start: float
    end: float
    axes: str = "global"
    label: str | None = None

    def scaled(self, k):
        return replace(self, w0=self.w0 * k, w1=self.w1 * k)


@dataclass
class SelfWeight:
    """Section.mass × g × factor on the beams (all of them when beams is None) along `direction`; with trusses, the bars'
    weight lumped to their ends (as femotion.frame.influence.uniform_load_forces)"""
    factor: float = 1.0
    g: float = 9.81
    beams: tuple | None = None
    trusses: bool = True
    label: str | None = None

    def scaled(self, k):
        return replace(self, factor=self.factor * k)


@dataclass
class Imposed:
    """free axial strain (elongation +) and curvature (sagging +, 1/m) of beams; one value or one per beam"""
    beams: tuple
    strain: np.ndarray
    curvature: np.ndarray
    label: str | None = None

    def scaled(self, k):
        return replace(self, strain=self.strain * k, curvature=self.curvature * k)


@dataclass
class TendonLoad:
    tendon: Tendon
    label: str | None = None

    def scaled(self, k):
        t = self.tendon
        return replace(self, tendon=Tendon(t.beams, t.ecc, np.asarray(t.force, float) * k))


@dataclass
class Settlement:
    """prescribed displacements d (6: three translations, three rotations; m, rad) of a node's restrained dofs"""
    node: int
    d: np.ndarray
    label: str | None = None

    def scaled(self, k):
        return replace(self, d=self.d * k)


@dataclass
class BarStrain:
    """initial strain of bars, pretension + (one value or one per bar)"""
    bars: tuple
    eps0: np.ndarray
    label: str | None = None

    def scaled(self, k):
        return replace(self, eps0=self.eps0 * k)


ITEM_TYPES = (NodalLoad, PointLoad, LineLoad, SelfWeight, Imposed, TendonLoad, Settlement, BarStrain)


@dataclass
class ElementLoad:
    """a span load on one beam in its local axes: kind "uniform" (p0 at distance a from the i end, p1 at b, linear
    between; per length) or "point" (p0 at a; b = a, p1 = p0). OpenSees: beamUniform over a/L…b/L, beamPoint at a/L"""
    kind: str
    beam: int
    a: float
    b: float
    p0: np.ndarray
    p1: np.ndarray


# ── geometry of the model: local axes and chains ───────────────────────────────────────────────────────────────────
class _Geometry:
    """what the load formulas need: the model and per beam (L, R, T); a Solver is used as it is"""

    def __init__(self, m: Model):
        from .linear import _T12
        self.m = m
        self._beam_cache = []
        for (i, j, sec, ref, tag) in m.beams:
            L, R = local_axes(m.X[i], m.X[j], ref)
            self._beam_cache.append((L, R, _T12(R)))


def _geometry(obj):
    if hasattr(obj, "_beam_cache") and hasattr(obj, "m"):
        return obj
    if isinstance(obj, Model):
        return _Geometry(obj)
    raise TypeError("a femotion.frame.Model or a Solver")


def chain(m: Model, beams) -> list[tuple[int, int, int, float, float, bool]]:
    """walk a chain of consecutive beams: per beam (beam, start node, end node, s at its start, s at its end, reversed:
    the chain runs from its j end to its i end). The chain starts at the node of the first beam not shared with the
    second (a single beam runs from i to j)."""
    beams = [int(b) for b in beams]
    if not beams:
        raise ValueError("an empty chain")
    ends = [m.beams[b][:2] for b in beams]
    if len(beams) == 1:
        first = ends[0][0]
    else:
        shared = set(ends[0]) & set(ends[1])
        if not shared:
            raise ValueError(f"beams {beams[0]} and {beams[1]} do not share a node")
        first = ends[0][0] if ends[0][1] in shared else ends[0][1]
    out, node, s = [], first, 0.0
    for b, (i, j) in zip(beams, ends):
        if node not in (i, j):
            raise ValueError(f"beam {b} does not continue the chain at node {node}")
        rev = node == j
        other = i if rev else j
        L = float(np.linalg.norm(m.X[j] - m.X[i]))
        out.append((b, node, other, s, s + L, rev))
        node, s = other, s + L
    return out


def path_beams(m: Model, path) -> list[int]:
    """the beams joining consecutive nodes of a path (e.g. InfluenceLines.path), in order"""
    lookup = {}
    for e, (i, j, *_r) in enumerate(m.beams):
        lookup.setdefault(frozenset((int(i), int(j))), e)
    out = []
    for a, b in zip(path[:-1], path[1:]):
        e = lookup.get(frozenset((int(a), int(b))))
        if e is None:
            raise ValueError(f"no beam joins path nodes {a} and {b}")
        out.append(e)
    return out


def _vec(v, direction):
    v = np.asarray(v, float)
    if v.ndim == 0:
        return float(v) * np.asarray(direction, float)
    if v.shape != (3,):
        raise ValueError("a load is a number (along direction) or a 3-vector")
    return v.copy()


# ── shape functions of the beam element ────────────────────────────────────────────────────────────────────────────
def _point_fef(L: float, a: float, p: np.ndarray) -> np.ndarray:
    """fixed-end (consistent) forces (local, 12) of a point load p (local x, y, z) at distance a from the i end"""
    x = a / L
    h1, h2, h3, h4 = 1 - 3 * x * x + 2 * x ** 3, L * (x - 2 * x * x + x ** 3), 3 * x * x - 2 * x ** 3, L * (x ** 3 - x * x)
    f = np.zeros(12)
    f[0], f[6] = (1 - x) * p[0], x * p[0]
    f[1], f[5], f[7], f[11] = h1 * p[1], h2 * p[1], h3 * p[1], h4 * p[1]  # v–θz plane
    f[2], f[4], f[8], f[10] = h1 * p[2], -h2 * p[2], h3 * p[2], -h4 * p[2]  # w–θy plane: signs opposite (beam_k)
    return f


def _span_fef(L: float, a: float, b: float, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """fixed-end forces of a load p0 at a … p1 at b (local, per length, linear between); a uniform load over the whole
    element goes through femotion.frame.influence._fixed_end, as uniform_load_forces"""
    if a <= 1e-12 * L and b >= L * (1 - 1e-12) and np.array_equal(p0, p1):
        return _fixed_end(p0, L)
    f = np.zeros(12)
    if b - a <= 1e-12 * L:
        return f
    xg, wg = _GAUSS
    for t, w in zip(xg, wg):
        x = a + (t + 1) / 2 * (b - a)
        p = p0 + (p1 - p0) * (x - a) / (b - a)
        f += w * (b - a) / 2 * _point_fef(L, x, p)
    return f


# ── the load case ───────────────────────────────────────────────────────────────────────────────────────────────────
class LoadCase:
    """a named set of loads on a frame model (see the module notes). The recording methods return the case, so they
    chain: LoadCase("吊装").point(deck, 12.5, 300).line(deck, 20, start=4, end=9)."""

    def __init__(self, name: str = "", items: Iterable | None = None):
        self.name = name
        self.items: list = list(items or [])

    def __repr__(self):
        kinds = {}
        for it in self.items:
            kinds[type(it).__name__] = kinds.get(type(it).__name__, 0) + 1
        return f"LoadCase({self.name!r}: " + ", ".join(f"{v} {k}" for k, v in kinds.items()) + ")"

    def of(self, kind) -> list:
        """the items of one type (NodalLoad, PointLoad, LineLoad, …)"""
        return [it for it in self.items if isinstance(it, kind)]

    # ── recording ──
    def nodal(self, node: int, F=(0.0, 0.0, 0.0), M=(0.0, 0.0, 0.0), *, label: str | None = None) -> "LoadCase":
        """forces F and moments M at a node (global axes; kN, kN·m)"""
        self.items.append(NodalLoad(int(node), _vec(F, DOWN), _vec(M, DOWN), label))
        return self

    def point(self, beams, s: float, P, *, direction=DOWN, side: int = 0, axes: str = "global",
              label: str | None = None) -> "LoadCase":
        """a point load at distance s (m) along a chain of beams: P (kN) along direction, or a 3-vector (global, or the
        element's local axes with axes="local"). On a node: side −1 / +1 = on the element before / after it"""
        self.items.append(PointLoad(tuple(int(b) for b in np.atleast_1d(beams)), float(s), _vec(P, direction),
                                    int(np.sign(side)), _axes(axes), label))
        return self

    def line(self, beams, q, *, start: float = 0.0, end: float | None = None, q_end=None, direction=DOWN,
             axes: str = "global", label: str | None = None) -> "LoadCase":
        """a distributed load (kN/m) along a chain of beams from distance start to end (default: the whole chain), q at
        the start and q_end at the end (default q: uniform), linear between. q: a number along direction or a 3-vector
        (global, or local with axes="local")"""
        beams = tuple(int(b) for b in np.atleast_1d(beams))
        w0 = _vec(q, direction)
        w1 = w0.copy() if q_end is None else _vec(q_end, direction)
        self.items.append(LineLoad(beams, w0, w1, float(start), None if end is None else float(end), _axes(axes),
                                   label))
        return self

    def self_weight(self, factor: float = 1.0, *, g: float = 9.81, beams=None, trusses: bool = True,
                    label: str | None = None) -> "LoadCase":
        """weight of the beams (Section.mass × g × factor, downwards) and of the bars (lumped to their ends)"""
        self.items.append(SelfWeight(float(factor), float(g),
                                     None if beams is None else tuple(int(b) for b in beams), bool(trusses), label))
        return self

    def imposed(self, beams, strain=0.0, curvature=0.0, *, label: str | None = None) -> "LoadCase":
        """free strain (elongation +) and curvature (sagging +, 1/m) of beams: temperature (α ΔT, the free curvature
        of a gradient), shrinkage; one value or one per beam"""
        beams = tuple(int(b) for b in np.atleast_1d(beams))
        eps = np.array(np.broadcast_to(np.asarray(strain, float), (len(beams),)))
        kap = np.array(np.broadcast_to(np.asarray(curvature, float), (len(beams),)))
        self.items.append(Imposed(beams, eps, kap, label))
        return self

    def tendon(self, td: Tendon, *, label: str | None = None) -> "LoadCase":
        """a prestressing tendon (femotion.frame.prestress.Tendon): its equivalent loads, section forces = total effect"""
        self.items.append(TendonLoad(td, label))
        return self

    def settlement(self, node: int, d, *, label: str | None = None) -> "LoadCase":
        """prescribed displacement of a support: d a number (vertical, m, up +: a settlement is negative) or six
        values (three translations, three rotations). Only restrained dofs may be prescribed."""
        v = np.asarray(d, float)
        dd = np.zeros(6)
        if v.ndim == 0:
            dd[2] = float(v)
        elif v.shape == (6,):
            dd[:] = v
        else:
            raise ValueError("settlement: a number (vertical) or six values")
        self.items.append(Settlement(int(node), dd, label))
        return self

    def bar_strain(self, bars, eps0, *, label: str | None = None) -> "LoadCase":
        """initial strain of bars (pretension +), as Solver.pretension"""
        bars = tuple(int(b) for b in np.atleast_1d(bars))
        self.items.append(BarStrain(bars, np.array(np.broadcast_to(np.asarray(eps0, float), (len(bars),))), label))
        return self

    # ── combinations ──
    def scaled(self, k: float, name: str | None = None) -> "LoadCase":
        return LoadCase(name if name is not None else (self.name if k == 1 else f"{_num(k)}×{self.name}"),
                        [it.scaled(float(k)) for it in self.items])

    def __mul__(self, k):
        if not np.isscalar(k):
            return NotImplemented
        return self.scaled(k)

    __rmul__ = __mul__

    def __neg__(self):
        return self.scaled(-1.0, f"−{self.name}")

    def __add__(self, other):
        if not isinstance(other, LoadCase):
            return NotImplemented
        return LoadCase(f"{self.name} + {other.name}", self.items + other.items)

    def __sub__(self, other):
        if not isinstance(other, LoadCase):
            return NotImplemented
        return LoadCase(f"{self.name} − {other.name}", self.items + [it.scaled(-1.0) for it in other.items])

    # ── what the loads do to the model ──
    def element_loads(self, model) -> list[ElementLoad]:
        """the span loads (line, point on an element, self-weight of the beams) per element in local axes"""
        g = _geometry(model)
        m = g.m
        out = []
        for it in self.items:
            if isinstance(it, LineLoad):
                out += _line_pieces(g, it)
            elif isinstance(it, PointLoad):
                tgt = _point_target(g, it)
                if tgt[0] == "beam":
                    _, e, a, p = tgt
                    out.append(ElementLoad("point", e, a, a, p, p))
            elif isinstance(it, SelfWeight):
                for e in (range(len(m.beams)) if it.beams is None else it.beams):
                    L, R, T = g._beam_cache[e]
                    w = np.zeros(3)
                    w[2] = -m.beams[e][2].mass * it.g * it.factor
                    p = R @ w
                    out.append(ElementLoad("uniform", int(e), 0.0, L, p, p))
        return out

    def nodal_loads(self, model) -> np.ndarray:
        """equivalent nodal loads (ndof, global) of everything but settlements: nodal loads, fixed-end forces of the
        span loads, restraint of imposed strains, tendons, bar pretension"""
        return self._assemble(_geometry(model))[0]

    def restrained_end_forces(self, model) -> np.ndarray:
        """(n_beams, 12) local end forces of the beams held at their nodes against the span loads and imposed strains:
        the element end forces of the final state are K u plus these"""
        return self._assemble(_geometry(model))[1]

    def prescribed(self, model) -> np.ndarray:
        """(ndof) prescribed displacements of the settlements (zero elsewhere)"""
        m = model.m if hasattr(model, "m") and not isinstance(model, Model) else model
        Uc = np.zeros(m.ndof)
        for it in self.of(Settlement):
            fx = m.fixed.get(it.node, [False] * 6)
            for a in range(6):
                if it.d[a] != 0 and not fx[a]:
                    raise ValueError(f"settlement of node {it.node}, dof {a}: only restrained dofs can be prescribed")
                Uc[6 * it.node + a] += it.d[a]
        return Uc

    def bar_strains(self, model) -> np.ndarray | None:
        m = model.m if hasattr(model, "m") and not isinstance(model, Model) else model
        items = self.of(BarStrain)
        if not items:
            return None
        eps = np.zeros(len(m.trusses))
        for it in items:
            np.add.at(eps, list(it.bars), it.eps0)
        return eps

    def _assemble(self, g):
        m = g.m
        F = np.zeros(m.ndof)
        r = np.zeros((len(m.beams), 12))
        for it in self.items:
            if isinstance(it, NodalLoad):
                F[6 * it.node:6 * it.node + 3] += it.F
                F[6 * it.node + 3:6 * it.node + 6] += it.M
            elif isinstance(it, SelfWeight):
                _self_weight(g, it, F, r)
            elif isinstance(it, (LineLoad, PointLoad)):
                pieces = _line_pieces(g, it) if isinstance(it, LineLoad) else []
                if isinstance(it, PointLoad):
                    tgt = _point_target(g, it)
                    if tgt[0] == "node":
                        F[6 * tgt[1]:6 * tgt[1] + 3] += tgt[2]
                        continue
                    _, e, a, p = tgt
                    pieces = [ElementLoad("point", e, a, a, p, p)]
                for pc in pieces:
                    L, R, T = g._beam_cache[pc.beam]
                    f = _point_fef(L, pc.a, pc.p0) if pc.kind == "point" else _span_fef(L, pc.a, pc.b, pc.p0, pc.p1)
                    i, j = m.beams[pc.beam][:2]
                    F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] += T.T @ f
                    r[pc.beam] -= f
            elif isinstance(it, Imposed):
                fr = _thermal_restrained(g, it.beams, it.strain, it.curvature)
                for e in dict.fromkeys(it.beams):
                    i, j = m.beams[e][:2]
                    T = g._beam_cache[e][2]
                    F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] -= T.T @ fr[e]
                r += fr
            elif isinstance(it, TendonLoad):
                F += equivalent_loads(g, it.tendon)
            elif isinstance(it, BarStrain):
                for e, ep in zip(it.bars, it.eps0):
                    i, j, sec, tag = m.trusses[e]
                    d = m.X[j] - m.X[i]
                    u = d / np.linalg.norm(d)
                    N0 = sec.E * sec.A * ep
                    F[6 * i:6 * i + 3] += N0 * u
                    F[6 * j:6 * j + 3] -= N0 * u
        return F, r

    # ── solving with the linear solver ──
    def solve(self, sv) -> "CaseResult":
        """displacements and element end forces of the case on femotion.frame.linear.Solver"""
        F, r = self._assemble(sv)
        Uc = self.prescribed(sv.m)
        U = sv.solve(F - sv.K @ Uc) + Uc if Uc.any() else sv.solve(F)
        return CaseResult(self, sv, U, sv.beam_end_forces(U) + r, F, self.bar_strains(sv.m))

    def section_forces(self, sv, sections, comp: str) -> np.ndarray:
        """section forces of the case at the given sections ((beam, "i" | "j"), femotion.frame.influence.chain_sections)"""
        return self.solve(sv).section_forces(sections, comp)


@dataclass
class CaseResult:
    """the solution of a load case: U (ndof), end_forces (n_beams, 12; local, element ends, span loads included), F the
    equivalent nodal loads applied, eps0 the bars' initial strains"""
    case: LoadCase
    sv: object
    U: np.ndarray
    end_forces: np.ndarray
    F: np.ndarray
    eps0: np.ndarray | None = None

    def section_forces(self, sections, comp: str) -> np.ndarray:
        return section_forces(self.sv, self.U, sections, comp, end_forces=self.end_forces)

    def reactions(self) -> np.ndarray:
        """support reactions (ndof; non-zero at restrained dofs): K u − F"""
        return self.sv.K @ self.U - self.F

    def truss_forces(self) -> np.ndarray:
        return self.sv.truss_forces(self.U, self.eps0)


def combine(*terms, name: str | None = None) -> LoadCase:
    """a linear combination: terms are (factor, case) pairs or cases (factor 1)"""
    out = LoadCase("")
    names = []
    for t in terms:
        k, c = (1.0, t) if isinstance(t, LoadCase) else t
        out.items += c.scaled(k).items
        names.append(c.name if k == 1 else f"{_num(k)}×{c.name}")
    out.name = name if name is not None else " + ".join(names)
    return out


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _axes(a: str) -> str:
    if a not in ("global", "local"):
        raise ValueError("axes: 'global' or 'local'")
    return a


def _num(k: float) -> str:
    return f"{k:g}"


def _self_weight(g, it: SelfWeight, F, r):
    """as femotion.frame.influence.uniform_load_forces (same arithmetic, same order)"""
    m = g.m
    beams = range(len(m.beams)) if it.beams is None else it.beams
    for e in beams:
        i, j, sec, ref, tag = m.beams[e]
        L, R, T = g._beam_cache[e]
        w = np.zeros(3)
        w[2] = -sec.mass * it.g * it.factor
        feq = _fixed_end(R @ w, L)
        F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] += T.T @ feq
        r[e] -= feq
    if it.trusses:
        for (i, j, sec, tag) in m.trusses:
            half = sec.mass * np.linalg.norm(m.X[j] - m.X[i]) / 2 * it.g * it.factor
            F[6 * i + 2] -= half
            F[6 * j + 2] -= half


def _line_pieces(g, it: LineLoad) -> list[ElementLoad]:
    """the part of a line load on each element of its chain, local axes, positions from the i end"""
    ch = chain(g.m, it.beams)
    total = ch[-1][4]
    s0, s1 = it.start, total if it.end is None else it.end
    w0, w1 = it.w0, it.w1
    if s1 < s0:  # given from the far end: q at `start` stays at start
        s0, s1, w0, w1 = s1, s0, w1, w0
    tol = 1e-9 * max(total, 1.0)
    if s0 < -tol or s1 > total + tol:
        raise ValueError(f"line load from {s0} to {s1} m: the chain is {total:.6g} m long")
    out = []
    ds = s1 - s0
    for (e, n0, n1, sa, sb, rev) in ch:
        lo, hi = max(s0, sa), min(s1, sb)
        if hi - lo <= tol:
            continue
        L, R, T = g._beam_cache[e]
        wl = w0 + (w1 - w0) * ((lo - s0) / ds if ds > 0 else 0.0)
        wh = w0 + (w1 - w0) * ((hi - s0) / ds if ds > 0 else 0.0)
        if it.axes == "global":
            wl, wh = R @ wl, R @ wh
        a, b = lo - sa, hi - sa  # from the chain's start node of this element
        if rev:  # measured from the i end
            a, b, wl, wh = L - b, L - a, wh, wl
        out.append(ElementLoad("uniform", e, max(a, 0.0), min(b, L), wl, wh))
    return out


def _point_target(g, it: PointLoad):
    """("node", node, P global) or ("beam", beam, a from its i end, P local)"""
    ch = chain(g.m, it.beams)
    total = ch[-1][4]
    tol = 1e-9 * max(total, 1.0)
    s = it.s
    if s < -tol or s > total + tol:
        raise ValueError(f"point load at {s} m: the chain is {total:.6g} m long")
    k_node = None  # index of the node in the chain (0 … len)
    for k, (e, n0, n1, sa, sb, rev) in enumerate(ch):
        if abs(s - sa) <= tol:
            k_node = k
            break
        if k == len(ch) - 1 and abs(s - sb) <= tol:
            k_node = k + 1
    if k_node is not None:
        k_el = k_node - 1 if it.side < 0 else k_node if it.side > 0 else None
        if k_el is None or k_el < 0 or k_el >= len(ch):
            node = ch[k_node][1] if k_node < len(ch) else ch[-1][2]
            P = it.P if it.axes == "global" else _local_to_global(g, ch[min(k_node, len(ch) - 1)][0], it.P)
            return ("node", node, P)
        e, n0, n1, sa, sb, rev = ch[k_el]
        L, R, T = g._beam_cache[e]
        a_chain = 0.0 if it.side > 0 else sb - sa
    else:
        k_el = next(k for k, c in enumerate(ch) if c[3] < s < c[4])
        e, n0, n1, sa, sb, rev = ch[k_el]
        L, R, T = g._beam_cache[e]
        a_chain = s - sa
    a = L - a_chain if rev else a_chain
    p = R @ it.P if it.axes == "global" else it.P.copy()
    return ("beam", e, min(max(a, 0.0), L), p)


def _local_to_global(g, e, p):
    return g._beam_cache[e][1].T @ p
