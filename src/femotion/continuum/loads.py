"""Load cases of continuum models: what acts on the solid, assembled into one load vector per case.

A LoadCase belongs to its Elastic model (Elastic.case(name) creates or returns it); every method adds to the case's load
vector and returns the case, so calls chain. Elastic.solve() then solves every case with one factorisation, and cases are
combined afterwards as results (femotion.continuum.results.CaseResult supports + and scalar *).

    g = s.case("自重").body_force((0, 0, -26e3), groups=["girder"])
    w = s.case("车辆").area_load(s.facets(normal=(0, 0, 1)), (0, 0, -70e3), rect(7.9, 8.1, 1.0, 1.6))
    t = s.case("温度").thermal(lambda x: 14.0 * np.exp(-(H - x[2]) / 0.1))
    p = s.case("预应力").tendon(td, anchor=(0.2, 0.2))
    s.solve()
    r = s.result("自重") + s.result("预应力")

The load vector f has two parts. External loads (f_ext: tractions, body forces, point and line forces, the equivalent
loads of tendons, which act on the concrete from outside it) are what a free body of the solid feels; their recipes are
kept, so a section can re-assemble the part that acts on the cells next to it. Self-equilibrated internal loads (f_int:
initial strains ∫Bᵀ D ε0 and the prestress −Gᵀ P of embedded bars) are no external forces; the stress of the case is
D (ε − ε0) and the bar force EA·δ/L + P, so the initial strains and prestresses are remembered too.

Facets: pass facet ids (Elastic.facets selects boundary facets by position and outward normal) or a predicate on facet
midpoints. Directions and positions are global; loads are forces per unit area (tractions), per unit volume (body
forces), per unit length (line loads) or plain forces.
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
        N = solver.ib.N
        self.f_ext = np.zeros(N)  # external loads
        self.f_int = np.zeros(N)  # self-equilibrated: initial strains, bar prestress
        self.parts: list[Callable] = []  # recipes of the external loads: part(cells | None) → load vector
        self.strains: list = []  # (cells, eps0(x, material) → Voigt (ncomp, …))
        self.bar_P: dict[str, np.ndarray] = {}  # prestress of embedded bars in this case
        self.uD: dict[int, float] = {}  # prescribed displacements of this case (dof → value)

    @property
    def f(self) -> np.ndarray:
        """the load vector: external + internal parts"""
        return self.f_ext + self.f_int

    @property
    def empty(self) -> bool:
        return not (self.parts or self.strains or self.bar_P or self.uD or np.any(self.f_ext))

    def _external(self, part: Callable) -> np.ndarray:
        v = part(None)
        self.f_ext += v
        self.parts.append(part)
        return v

    def restricted(self, cells: np.ndarray) -> np.ndarray:
        """the external loads acting through the given cells only (their share at shared nodes included); nodal loads
        of point_load are not attributed to cells"""
        out = np.zeros(self.s.ib.N)
        for p in self.parts:
            out += p(cells)
        return out

    # ── surface loads ───────────────────────────────────────────────────
    def _facet_part(self, facets: np.ndarray, vector: Callable):
        owner = self.s.sk.f2t[0, facets]

        def part(cells):
            fac = facets if cells is None else facets[np.isin(owner, cells)]
            return vector(fac) if len(fac) else np.zeros(self.s.ib.N)
        return part

    def traction(self, where, t, intorder: int | None = None) -> "LoadCase":
        """traction (force per area) on boundary facets: a constant vector or t(x) → components"""
        fac = self.s._facets(where)
        self._external(self._facet_part(fac, lambda f: self.s._traction_vector(f, t, intorder)))
        return self

    def pressure(self, where, p, intorder: int | None = None) -> "LoadCase":
        """pressure (force per area) against the outward normal: a constant or p(x)"""
        fac = self.s._facets(where)
        self._external(self._facet_part(fac, lambda f: self.s._pressure_vector(f, p, intorder)))
        return self

    def area_load(self, where, F, region: Callable, intorder: int = 12, spacing: float | None = None) -> "LoadCase":
        """a total force F (vector) spread uniformly over the part of the facets `where` inside region(x): a wheel print,
        a lane, a plate. With a rect(...) region in 3D the print is sampled on a grid of points (spacing: at most a tenth
        of its sides and a quarter of the median cell edge) placed on the facets under it and applied through the cells'
        shape functions: resultant exact, point of application right however small the print is against the cells. Other
        regions: traction F / A·[x in region] with A the covered area by the same quadrature (resultant exact, the point
        of application as good as the quadrature resolves the region's edges: raise intorder for small regions)"""
        fac = self.s._facets(where)
        F = np.asarray(F, float)
        bb = getattr(region, "bbox", None)
        if bb is not None:  # only the facets whose extent overlaps the region's
            P = self.s.sk.p[:, self.s.sk.facets[:, fac]]  # (dim, nv, nf)
            keep = np.ones(len(fac), bool)
            for i, (lo, hi) in enumerate(bb):
                keep &= (P[i].max(axis=0) >= lo) & (P[i].min(axis=0) <= hi)
            fac = fac[keep]
        if bb is not None and self.s.dim == 3 and len(fac):
            pts = self._print_points(fac, region, bb, spacing, F)
            if len(pts):
                return self.point_forces(pts, np.repeat(F[None] / len(pts), len(pts), axis=0))
        A = self.s.facet_integral(fac, lambda x: region(x).astype(float), intorder) if len(fac) else 0.0
        if A <= 0:
            raise ValueError("area_load: no point of the facets falls in the region (raise intorder, or check the facets "
                             "and the region)")
        t = (lambda x: [F[i] / A * region(x) for i in range(self.s.dim)])
        self._external(self._facet_part(fac, lambda f: self.s._traction_vector(f, t, intorder)))
        return self

    def _print_points(self, fac, region, bb, spacing, F):
        """grid points of a plan region lying on the given (upward or downward facing) facets, just inside the solid.
        Where several facet layers lie over the same point (a deck slab and the floor of a box cell under it), the
        outermost one along the facets' outward normal takes the load — the surface the load reaches first"""
        from .elastic import facet_normals
        (x0, x1), (y0, y1) = bb
        ds = spacing or min((x1 - x0) / 10, (y1 - y0) / 10, self.s.h / 4)
        nx, ny = max(1, int(np.ceil((x1 - x0) / ds))), max(1, int(np.ceil((y1 - y0) / ds)))
        X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx, y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny)
        g = np.c_[X.ravel(), Y.ravel()]
        g = g[np.asarray(region(np.r_[g.T, np.zeros((1, len(g)))]), bool)]
        Pf = self.s.sk.p[:, self.s.sk.facets[:, fac]]  # (3, nv, nf)
        nrm = self.s.boundary_normals(fac)  # outward (3, nf)
        cen = Pf.mean(axis=1)
        out = np.full(len(g), np.nan)
        score = np.full(len(g), -np.inf)
        side = np.zeros((len(g), 3))
        for k in range(len(fac)):
            if abs(nrm[2, k]) < 1e-6:
                continue
            V = Pf[:2, :, k].T  # plan polygon
            c2 = V.mean(axis=0)
            ang = np.argsort(np.arctan2(V[:, 1] - c2[1], V[:, 0] - c2[0]))
            V = V[ang]
            inside = np.ones(len(g), bool)
            for i in range(len(V)):  # convex polygon: left of every edge
                a, b = V[i], V[(i + 1) % len(V)]
                inside &= (b[0] - a[0]) * (g[:, 1] - a[1]) - (b[1] - a[1]) * (g[:, 0] - a[0]) >= -1e-12
            z = cen[2, k] - (nrm[0, k] * (g[:, 0] - cen[0, k]) + nrm[1, k] * (g[:, 1] - cen[1, k])) / nrm[2, k]
            sc = np.sign(nrm[2, k]) * z  # outermost: highest of upward faces, lowest of downward ones
            new = inside & (sc > score)
            out[new], score[new], side[new] = z[new], sc[new], nrm[:, k]
        ok = ~np.isnan(out)
        return np.c_[g[ok], out[ok]] - 1e-6 * self.s.h * side[ok]

    # ── volume loads ────────────────────────────────────────────────────
    def body_force(self, b, groups=None) -> "LoadCase":
        """force per unit volume (self-weight: (0, 0, −γ)), a constant vector or b(x) → components; only on the cells
        of the given cell groups when groups is given"""
        cg = self.s._cells_of(groups)

        def part(cells):
            c = cg if cells is None else np.intersect1d(cg, cells)
            return self.s._body_vector(b, c) if len(c) else np.zeros(self.s.ib.N)
        self._external(part)
        return self

    def initial_strain(self, eps0, groups=None) -> "LoadCase":
        """a free strain (Voigt, engineering shear): a constant vector or eps0(x) → (ncomp, …) (shrinkage, creep, misfit).
        In plane strain, a 3D strain (6 components) is reduced like a temperature (its out-of-plane part included); 3
        components mean no free strain out of the plane"""
        cells = self.s._cells_of(groups)
        fn = eps0 if callable(eps0) else (lambda x, e=np.asarray(eps0, float): e.reshape((-1,) + (1,) * (x.ndim - 1)) + 0 * x[0])
        s = self.s

        def src(x, mat):
            e = np.asarray(fn(x))
            if s.dim == 2 and e.shape[0] == 6:
                return mat.reduce_strain(e, s.dim, s.kind)
            return e
        self.strains.append((cells, src))
        self.f_int += s._strain_vector(self.strains[-1])
        return self

    def thermal(self, dT, groups=None) -> "LoadCase":
        """a temperature change ΔT (constant or ΔT(x)), each cell expanding by its material's α (Orthotropic: per axis)"""
        cells = self.s._cells_of(groups)
        f = dT if callable(dT) else (lambda x, v=float(dT): v + 0 * x[0])
        kind, dim = self.s.kind, self.s.dim
        self.strains.append((cells, lambda x, mat: mat.thermal_strain(dim, kind).reshape((-1,) + (1,) * (x.ndim - 1))
                             * np.asarray(f(x))[None]))
        self.f_int += self.s._strain_vector(self.strains[-1])
        return self

    # ── forces at arbitrary points ──────────────────────────────────────
    def point_load(self, p, F) -> "LoadCase":
        """a force at the node nearest to p (a nodal load: a section through that node counts it to the negative side)"""
        n = int(np.argmin(np.linalg.norm(self.s.sk.p.T - np.asarray(p, float), axis=1)))
        v = np.zeros(self.s.ib.N)
        for i, c in enumerate(F):
            v[self.s.ib.nodal_dofs[i, n]] += c
        self._external(lambda cells: v if cells is None else 0 * v)
        return self

    def point_forces(self, points, F, snap: float = 0.0, outside: str = "raise") -> "LoadCase":
        """forces F (n, dim) at arbitrary points inside the solid, through the shape functions of their cells (snap: accept
        points up to that many cells outside, moved onto the nearest cell; Locator.locate). outside="skip" drops the
        points outside the mesh (loads of the whole structure applied to a submodel; decided without snapping)"""
        P = np.asarray(points, float).reshape(-1, self.s.dim)
        F = np.asarray(F, float).reshape(-1, self.s.dim)
        if not len(P):
            return self
        if outside == "skip":
            keep = self.s.locator.inside(P)
            P, F = P[keep], F[keep]
            if not len(P):
                return self
        loc = self.s.locator.locate(P, snap=snap)
        M = self.s.locator.matrix(P, loc)
        cells_of = loc[0]
        d = self.s.dim

        def part(cells):
            if cells is None:
                return M.T @ F.reshape(-1)
            k = np.isin(cells_of, cells)
            rows = (d * np.where(k)[0][:, None] + np.arange(d)).ravel()
            return M[rows].T @ F[k].reshape(-1)
        self._external(part)
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
                pts.append(R[k] + a * (R[k + 1] - R[k]))
                s = s_r[k] + a * L
                frc.append(np.array([np.interp(s, s_v, Q[:, c]) for c in range(P.shape[1])]) * w_ * L / 2)
        return self.point_forces(np.array(pts), np.array(frc), outside=outside)

    def displacement(self, where, u, comps=None) -> "LoadCase":
        """prescribed displacements in this case (support settlement, the cut boundary of a submodel): the dofs located
        where `where` holds (or of given vertex ids) become fixed in every case, with u (a constant vector or u(X) →
        (dim, n) at the dof locations X (dim, n)) in this one and zero in the others unless they prescribe their own"""
        s = self.s
        comps = list(range(s.dim) if comps is None else comps)
        per = s.dofs_where(where, comps)
        X = s.ib.doflocs
        alld = np.unique(np.concatenate([per[c] for c in comps]))
        U = np.asarray(u(X[:, alld])) if callable(u) else None  # evaluated once for all components
        pos = {int(k): i for i, k in enumerate(alld)}
        for c in comps:
            d = per[c]
            if not len(d):
                raise ValueError("displacement: the selection holds no nodes")
            s._fixed.append(d)
            vals = U[c, [pos[int(k)] for k in d]] if U is not None else np.full(len(d), float(np.asarray(u, float)[c]))
            self.uD.update(zip(d.tolist(), np.asarray(vals, float).tolist()))
        return self

    # ── prestressing ────────────────────────────────────────────────────
    def tendon(self, td: Tendon, anchor=None, up=(0.0, 0.0, 1.0), outside: str = "raise") -> "LoadCase":
        """the equivalent loads of a tendon (femotion.continuum.tendon): the change of its force along the path, applied
        where it happens. anchor = (a, b): the anchor forces spread over a bearing plate a × b perpendicular to the
        tendon (side a along `up`; in 2D a line of length a), instead of acting at the anchor point"""
        F = td.vertex_forces()
        self.point_forces(td.points[1:-1], F[1:-1], outside=outside)  # along the path: strictly inside
        for i in (0, -1):
            if anchor is None:
                self.point_forces(td.points[i][None], F[i][None], outside=outside)
                continue
            if outside == "skip" and not self.s.locator.inside(td.points[i][None])[0]:
                continue  # the anchor itself lies outside this (sub)model
            t = td.tangents()[0 if i == 0 else -1]
            ap = anchor_points(td.points[i], t, anchor, up)  # a bearing plate may poke out of a sloping end face: snap
            self.point_forces(ap, np.repeat(F[i][None] / len(ap), len(ap), axis=0), snap=1.0)
        return self

    def bar_prestress(self, bar, P) -> "LoadCase":
        """an initial tension P in an embedded bar (Elastic.bar): the self-equilibrated load −Gᵀ P on the structure; the
        bar's force in this case is then EA·elongation/L + P"""
        P = np.broadcast_to(np.asarray(P, float), bar.L.shape).copy()
        self.f_int += bar.prestress_load(P)
        self.bar_P[bar.name] = self.bar_P.get(bar.name, 0) + P
        return self

    # ── bookkeeping ─────────────────────────────────────────────────────
    def scale_load(self, total: float, direction) -> float:
        """scale the external loads of this case so that their resultant along `direction` equals `total` (initial
        strains and bar prestress are not scaled)"""
        R = float(np.dot(self.resultant(), direction))
        if R == 0:
            raise ValueError("scale_load: the resultant along this direction is zero")
        k = total / R
        self.f_ext *= k
        self.parts = [(lambda cells, p=p: k * p(cells)) for p in self.parts]  # loads added later are not scaled
        return k

    def resultant(self) -> np.ndarray:
        """resultant of the external loads (the internal parts add up to zero)"""
        return np.array([self.f_ext[i::self.s.dim].sum() for i in range(self.s.dim)])
