"""Load cases of continuum models: what acts on the solid, assembled into one load vector per case.

A LoadCase belongs to its Elastic model (Elastic.case(name) creates or returns it); every method adds to the case's load
vector and returns the case, so calls chain. Initial strains (temperature, shrinkage) are also remembered, because the
stress of the case is D (ε − ε0), not D ε. Elastic.solve() then solves every case with one factorisation, and cases are
combined afterwards as results (femotion.continuum.results.CaseResult supports + and scalar *).

    g = s.case("自重").body_force((0, 0, -26e3), groups=["girder"])
    w = s.case("车辆").area_load(s.facets(normal=(0, 0, 1)), (0, 0, -70e3), rect(7.9, 8.1, 1.0, 1.6))
    t = s.case("温度").thermal(lambda x: 14.0 * np.exp(-(H - x[2]) / 0.1))
    p = s.case("预应力").tendon(td, anchor=(0.2, 0.2))
    s.solve()
    r = s.result("自重") + s.result("预应力")

Where facets are wanted, pass facet ids (Elastic.facets selects boundary facets by position and outward normal) or a
predicate on facet midpoints (as Elastic.traction always took). Directions and positions are global; loads are forces
per unit area (tractions), per unit volume (body forces), per unit length (line loads) or plain forces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .embed import resample
from .tendon import Tendon, anchor_points

if TYPE_CHECKING:
    from .elastic import Elastic


def rect(x0: float, x1: float, y0: float, y1: float) -> Callable:
    """indicator of a rectangle in plan (x, y) for area_load: a wheel print, a lane, a bearing plate (its `bbox` lets
    area_load skip the facets far from it)"""
    def f(x):
        return (x[0] >= x0) & (x[0] <= x1) & (x[1] >= y0) & (x[1] <= y1)
    f.bbox = ((x0, x1), (y0, y1))
    return f


class LoadCase:
    def __init__(self, solver: "Elastic", name: str):
        self.s = solver
        self.name = name
        self.f = np.zeros(solver.ib.N)
        self.strains: list = []  # (groups mask or None, eps0(x) → Voigt (ncomp, nel, nq) or constant)
        self.bar_P: dict[str, np.ndarray] = {}  # prestress of embedded bars in this case
        self.uD: dict[int, float] = {}  # prescribed displacements of this case (dof → value)

    def _add(self, f: np.ndarray) -> np.ndarray:
        self.f += f
        return f

    # ── surface loads ───────────────────────────────────────────────────
    def traction(self, where, t, intorder: int | None = None) -> "LoadCase":
        """traction (force per area) on boundary facets: a constant vector or t(x) → components"""
        self._add(self.s._traction_vector(where, t, intorder))
        return self

    def pressure(self, where, p, intorder: int | None = None) -> "LoadCase":
        """pressure (force per area) against the outward normal: a constant or p(x)"""
        self._add(self.s._pressure_vector(where, p, intorder))
        return self

    def area_load(self, where, F, region: Callable, intorder: int = 8) -> "LoadCase":
        """a total force F (vector) spread uniformly over the part of the facets `where` inside region(x) (e.g. rect(...)
        in plan: a wheel, a lane, a plate): traction F / A·[x in region], A being the covered area integrated with the same
        quadrature, so the resultant is exactly F even where the region cuts through cells"""
        fac = self.s._facets(where)
        bb = getattr(region, "bbox", None)
        if bb is not None:  # only the facets whose extent overlaps the region's
            P = self.s.sk.p[:, self.s.sk.facets[:, fac]]  # (dim, nv, nf)
            keep = np.ones(len(fac), bool)
            for i, (lo, hi) in enumerate(bb):
                keep &= (P[i].max(axis=0) >= lo) & (P[i].min(axis=0) <= hi)
            fac = fac[keep]
        A = self.s.facet_integral(fac, lambda x: region(x).astype(float), intorder) if len(fac) else 0.0
        if A <= 0:
            raise ValueError("area_load: the region covers none of the facets")
        F = np.asarray(F, float)
        self._add(self.s._traction_vector(fac, lambda x: [F[i] / A * region(x) for i in range(self.s.dim)], intorder))
        return self

    # ── volume loads ────────────────────────────────────────────────────
    def body_force(self, b, groups=None) -> "LoadCase":
        """force per unit volume (self-weight: (0, 0, −γ)), a constant vector or b(x) → components; only on the cells
        of the given cell groups when groups is given"""
        self._add(self.s._body_vector(b, self.s._cells_of(groups)))
        return self

    def initial_strain(self, eps0, groups=None) -> "LoadCase":
        """a free strain (Voigt, engineering shear): a constant vector or eps0(x) → (ncomp, …) (shrinkage, creep, misfit)"""
        cells = self.s._cells_of(groups)
        fn = eps0 if callable(eps0) else (lambda x, e=np.asarray(eps0, float): e.reshape((-1,) + (1,) * (x.ndim - 1)) + 0 * x[0])
        self.strains.append((cells, lambda x, mat: np.asarray(fn(x))))
        self._add(self.s._strain_vector(self.strains[-1]))
        return self

    def thermal(self, dT, groups=None) -> "LoadCase":
        """a temperature change ΔT (constant or ΔT(x)), each cell expanding by its material's α (Orthotropic: per axis)"""
        cells = self.s._cells_of(groups)
        f = dT if callable(dT) else (lambda x, v=float(dT): v + 0 * x[0])
        kind, dim = self.s.kind, self.s.dim
        self.strains.append((cells, lambda x, mat: mat.thermal_strain(dim, kind).reshape((-1,) + (1,) * (x.ndim - 1))
                             * np.asarray(f(x))[None]))
        self._add(self.s._strain_vector(self.strains[-1]))
        return self

    # ── forces at arbitrary points ──────────────────────────────────────
    def point_load(self, p, F) -> "LoadCase":
        """a force at the node nearest to p"""
        n = int(np.argmin(np.linalg.norm(self.s.sk.p.T - np.asarray(p, float), axis=1)))
        for i, c in enumerate(F):
            self.f[self.s.ib.nodal_dofs[i, n]] += c
        return self

    def point_forces(self, points, F, snap: float = 0.0, outside: str = "raise") -> "LoadCase":
        """forces F (n, dim) at arbitrary points inside the solid, through the shape functions of their cells (snap: accept
        points up to that many cells outside, moved onto the surface; Locator.locate). outside="skip" drops points
        outside the mesh instead of raising (loads of the whole structure applied to a submodel)"""
        P = np.asarray(points, float).reshape(-1, self.s.dim)
        F = np.asarray(F, float).reshape(-1, self.s.dim)
        if not len(P):
            return self
        if outside == "skip":
            keep = self.s.locator.inside(P, snap=snap)
            P, F = P[keep], F[keep]
            if not len(P):
                return self
        self._add(self.s.locator.forces(P, F, snap))
        return self

    def displacement(self, where, u, comps=None) -> "LoadCase":
        """prescribed displacements in this case (support settlement, the cut boundary of a submodel): the dofs located
        where `where` holds (or of given vertex ids) become fixed in every case, with u (a constant vector or u(X) →
        (dim, n) at the dof locations X (dim, n)) in this one and zero in the others unless they prescribe their own"""
        s = self.s
        comps = list(range(s.dim) if comps is None else comps)
        per = s.dofs_where(where, comps)
        X = s.ib.doflocs
        for c in comps:
            d = per[c]
            if not len(d):
                raise ValueError("displacement: the selection holds no nodes")
            s._fixed.append(d)
            if callable(u):
                vals = np.asarray(u(X[:, d]))[c]
            else:
                vals = np.full(len(d), float(np.asarray(u, float)[c]))
            self.uD.update(zip(d.tolist(), vals.tolist()))
        return self

    def line_load(self, points, q, ds: float | None = None, gauss: int = 3, outside: str = "raise") -> "LoadCase":
        """a distributed force along a polyline inside the solid (an anchor bolt, a duct, a wall line): q per unit length,
        a constant vector or one vector per vertex (linear in between); integrated with Gauss points on pieces of at
        most ds (default: the bar spacing of femotion.continuum.embed)"""
        from .embed import default_spacing
        P = np.asarray(points, float)
        Q = np.broadcast_to(np.asarray(q, float), P.shape).astype(float)
        s_v = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        R = resample(P, ds or default_spacing(self.s))
        s_r = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))]
        xi, wi = np.polynomial.legendre.leggauss(gauss)
        pts, frc = [], []
        for k in range(len(R) - 1):
            L = s_r[k + 1] - s_r[k]
            for x_, w_ in zip(xi, wi):
                a = 0.5 * (1 + x_)
                p = R[k] + a * (R[k + 1] - R[k])
                s = s_r[k] + a * L
                qv = np.array([np.interp(s, s_v, Q[:, c]) for c in range(P.shape[1])])
                pts.append(p)
                frc.append(qv * w_ * L / 2)
        return self.point_forces(np.array(pts), np.array(frc), outside=outside)

    # ── prestressing ────────────────────────────────────────────────────
    def tendon(self, td: Tendon, anchor=None, up=(0.0, 0.0, 1.0), outside: str = "raise") -> "LoadCase":
        """the equivalent loads of a tendon (femotion.continuum.tendon): the change of its force along the path, applied
        where it happens. anchor = (a, b): the anchor forces spread over a bearing plate a × b perpendicular to the
        tendon (side a along `up`), instead of acting at the anchor point"""
        F = td.vertex_forces()
        self.point_forces(td.points[1:-1], F[1:-1], outside=outside)  # along the path: strictly inside
        for i in (0, -1):
            if anchor is None:
                self.point_forces(td.points[i][None], F[i][None], outside=outside)
            else:  # a bearing plate may poke out of a sloping end face: its points snap onto the surface
                t = td.tangents()[0 if i == 0 else -1]
                ap = anchor_points(td.points[i], t, anchor, up)
                if outside == "skip" and not self.s.locator.inside(td.points[i][None])[0]:
                    continue  # the anchor itself lies outside this (sub)model
                self.point_forces(ap, np.repeat(F[i][None] / len(ap), len(ap), axis=0), snap=1.0)
        return self

    def bar_prestress(self, bar, P) -> "LoadCase":
        """an initial tension P in an embedded bar (Elastic.bar): the load −Gᵀ P on the structure; the bar's force in this
        case is then EA·elongation/L + P"""
        P = np.broadcast_to(np.asarray(P, float), bar.L.shape).copy()
        self._add(bar.prestress_load(P))
        self.bar_P[bar.name] = self.bar_P.get(bar.name, 0) + P
        return self

    # ── bookkeeping ─────────────────────────────────────────────────────
    def scale_load(self, total: float, direction) -> float:
        """scale this case so that its resultant along `direction` equals `total` (initial strains are not scaled)"""
        k = total / float(np.dot(self.resultant(), direction))
        self.f *= k
        return k

    def resultant(self) -> np.ndarray:
        return np.array([self.f[i::self.s.dim].sum() for i in range(self.s.dim)])
