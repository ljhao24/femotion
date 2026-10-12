"""Points inside a continuum mesh: locating them, interpolating at them, and bars embedded in the solid.

Reinforcement, tendons, anchor bolts and point or line loads rarely follow the mesh. Everything here works at arbitrary
points: a point is found in its cell (reference coordinates by inverting the cell map), and the element's own shape
functions carry values from the dofs to the point (interpolation, `Locator.matrix`) or forces from the point to the dofs
(the transpose: consistent nodal forces). A bar between two embedded points then has the elongation t·(u_b − u_a), which
gives its stiffness and, with an initial force P, the load −Gᵀ P it puts on the solid (Bar, femotion.continuum.tendon).

    loc = Locator(solver)                       # solver: femotion.continuum.Elastic
    cells, X = loc.locate(points)               # (n,) cells, (dim, n) reference coordinates
    N = loc.matrix(points)                      # (dim·n, ndof): displacements at the points = N u
    bar = Bar(solver, rebar_polyline, EA=2e5 * 314e-6 * 1e6)   # bonded: resampled finer than the mesh
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _violation(X: np.ndarray, simplex: bool) -> np.ndarray:
    """how far reference coordinates (dim, n) lie outside the reference cell (≤ 0 inside)"""
    if simplex:
        return np.maximum(-X.min(axis=0), X.sum(axis=0) - 1)
    return np.maximum(-X.min(axis=0), X.max(axis=0) - 1)


def inverse_map(mapping, x: np.ndarray, tind: np.ndarray, iters: int = 40) -> np.ndarray:
    """reference coordinates (dim, n, 1) of points x (dim, n, 1) in cells tind; unlike scikit-fem's invF it neither clips
    to the reference cell nor raises for points outside it (they get coordinates outside [0, 1])"""
    if type(mapping).__name__ == "MappingAffine":
        return mapping.invF(x, tind=tind)
    X = np.zeros(x.shape) + 0.5
    for _ in range(iters):
        dX = np.einsum("ijkl,jkl->ikl", mapping.invDF(X, tind), x - mapping.F(X, tind))
        X = np.clip(X + dX, -3.0, 4.0)
        if np.abs(dX).max() < 1e-13:
            break
    return X


class Locator:
    """finds points in the cells of a solver's mesh (scikit-fem reference cells: unit simplex or unit square / cube)"""

    def __init__(self, solver, candidates: int = 12):
        from scipy.spatial import cKDTree
        self.s = solver
        sk = solver.sk
        self.sk = sk
        self.mapping = solver.sb.mapping
        self.simplex = type(sk).__name__.startswith(("MeshTri", "MeshTet"))
        self.centroids = sk.p[:, sk.t].mean(axis=1).T
        self.tree = cKDTree(self.centroids)
        self.k = min(candidates, sk.t.shape[1])

    def locate(self, points, tol: float = 1e-6, snap: float = 0.0):
        """(cells (n,), reference coordinates (dim, n)); points on faces go to the least violated cell. Points outside the
        mesh by more than tol (in reference coordinates of the nearest cell) raise ValueError, unless within `snap` (e.g.
        1.0: one cell): those are moved onto the nearest cell (anchor plates poking out of a sloping end face)"""
        tol = max(tol, snap)
        P = np.atleast_2d(np.asarray(points, float))
        n = len(P)
        _, cand = self.tree.query(P, k=self.k)
        cand = cand.reshape(n, -1)
        best_v = np.full(n, np.inf)
        best_c = np.zeros(n, np.int64)
        best_X = np.zeros((self.sk.p.shape[0], n))
        x = P.T[:, :, None]
        for j in range(cand.shape[1]):
            c = cand[:, j]
            todo = best_v > 0
            if not todo.any():
                break
            X = inverse_map(self.mapping, x[:, todo], c[todo])[:, :, 0]
            v = _violation(X, self.simplex)
            sel = np.where(todo)[0]
            better = v < best_v[sel]
            best_v[sel[better]] = v[better]
            best_c[sel[better]] = c[todo][better]
            best_X[:, sel[better]] = X[:, better]
        bad = best_v > tol
        if bad.any():
            # elongated cells can hide the owner behind more centroids: search the rest once by brute force
            for i in np.where(bad)[0]:
                X = inverse_map(self.mapping, np.repeat(P[i][:, None, None], self.sk.t.shape[1], axis=1),
                                np.arange(self.sk.t.shape[1]))[:, :, 0]
                v = _violation(X, self.simplex)
                k = int(np.argmin(v))
                best_v[i], best_c[i], best_X[:, i] = v[k], k, X[:, k]
            bad = best_v > tol
            if bad.any():
                raise ValueError(f"{int(bad.sum())} point(s) lie outside the mesh, e.g. {P[np.where(bad)[0][0]]}")
        lo, hi = 0.0, 1.0
        best_X = np.clip(best_X, lo, hi)
        return best_c, best_X

    def inside(self, points, tol: float = 1e-6, snap: float = 0.0) -> np.ndarray:
        """bool per point: inside the mesh (within tol, or snap cells)"""
        P = np.atleast_2d(np.asarray(points, float))
        lim = max(tol, snap)
        _, cand = self.tree.query(P, k=self.k)
        cand = cand.reshape(len(P), -1)
        ok = np.zeros(len(P), bool)
        x = P.T[:, :, None]
        for j in range(cand.shape[1]):
            todo = ~ok
            if not todo.any():
                break
            X = inverse_map(self.mapping, x[:, todo], cand[todo, j])[:, :, 0]
            ok[np.where(todo)[0][_violation(X, self.simplex) <= lim]] = True
        # points the nearest cells did not contain: exact test only if they are within the mesh's bounding box
        lo, hi = self.sk.p.min(axis=1), self.sk.p.max(axis=1)
        rest = np.where(~ok & np.all((P >= lo) & (P <= hi), axis=1))[0]
        for i in rest:
            try:
                self.locate(P[i][None], tol=tol, snap=snap)
                ok[i] = True
            except ValueError:
                pass
        return ok

    def _phi(self, cells, X, grad: bool = False):
        """scalar shape functions (Nbfun, n) or their gradients (Nbfun, dim, n) at the points"""
        sb = self.s.sb
        out = []
        for k in range(sb.Nbfun):
            f = sb.elem.gbasis(self.mapping, X[:, :, None], k, tind=cells)[0]
            out.append(f.grad[:, :, 0] if grad else f.value[:, 0])
        return np.array(out)

    def scalar_matrix(self, points, located=None) -> sp.csr_matrix:
        """(n, n_scalar_dofs): values of a scalar field at the points"""
        cells, X = located if located is not None else self.locate(points)
        phi = self._phi(cells, X)
        dofs = self.s.sb.element_dofs[:, cells]
        n = len(cells)
        rows = np.tile(np.arange(n), phi.shape[0])
        return sp.csr_matrix((phi.ravel(), (rows, dofs.ravel())), shape=(n, self.s.sb.N))

    def matrix(self, points, located=None) -> sp.csr_matrix:
        """(dim·n, ndof): displacement component c of point i is row dim·i + c (dof dim·k + c of scalar dof k)"""
        S = self.scalar_matrix(points, located).tocoo()
        d = self.s.dim
        rows = np.concatenate([d * S.row + c for c in range(d)])
        cols = np.concatenate([d * S.col + c for c in range(d)])
        return sp.csr_matrix((np.tile(S.data, d), (rows, cols)), shape=(d * S.shape[0], self.s.ib.N))

    def forces(self, points, F, snap: float = 0.0) -> np.ndarray:
        """consistent nodal load vector of point forces F (n, dim) at the points"""
        F = np.atleast_2d(np.asarray(F, float))
        return self.matrix(points, self.locate(points, snap=snap)).T @ F.reshape(-1)

    def gradients(self, u: np.ndarray, located):
        """displacement gradients at located points: (dim, dim, n), [i, j] = ∂u_i / ∂x_j"""
        cells, X = located
        dphi = self._phi(cells, X, grad=True)  # (nb, dim, n)
        d = self.s.dim
        dofs = self.s.sb.element_dofs[:, cells]  # (nb, n)
        U = np.stack([u[d * dofs + c] for c in range(d)])  # (dim, nb, n)
        return np.einsum("ikn,kjn->ijn", U, dphi)


def resample(points, ds: float) -> np.ndarray:
    """a polyline subdivided so that no piece is longer than ds (the given vertices are kept)"""
    P = np.asarray(points, float)
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / ds - 1e-9)))
        out.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return np.concatenate(out)


def segment_operator(loc: Locator, points) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """G (nseg, ndof) with G u = elongation of each segment of the polyline t·(u_b − u_a), segment lengths L and unit
    tangents t (nseg, dim)"""
    P = np.asarray(points, float)
    d = P[1:] - P[:-1]
    L = np.linalg.norm(d, axis=1)
    if np.any(L <= 0):
        raise ValueError("a polyline has repeated points")
    t = d / L[:, None]
    N = loc.matrix(P).tocsr()  # (dim·n, ndof)
    dim = loc.s.dim
    nseg = len(L)
    G = sp.csr_matrix((nseg, N.shape[1]))
    for c in range(dim):
        Nc = N[c::dim]  # (n, ndof) component c
        G = G + sp.diags(t[:, c]) @ (Nc[1:] - Nc[:-1])
    return G.tocsr(), L, t


def default_spacing(solver) -> float:
    """half the median edge length of the mesh: bonded bars sampled finer than the cells they cross"""
    m = solver.mesh
    e = m.edges()
    return 0.5 * float(np.median(np.linalg.norm(m.points[e[:, 1]] - m.points[e[:, 0]], axis=1)))


class Bar:
    """an axial bar in a solid. bonded=True: perfectly bonded along its whole length (rebar, grouted tendon, bonded plate),
    the polyline resampled to pieces of at most ds, each piece with its own force; bonded=False: attached only at the
    given points (an external tendon at its anchors and frictionless deviators, a plate between its anchors), one force
    along the whole length (the elongations of the straight runs add up). EA: axial stiffness, one value (or, bonded, one
    per piece). Adds Gᵀ·diag(EA/L)·G to the stiffness; its force is EA·elongation/L + the prestress P0 of a load case
    (Elastic.case(...).bar_prestress)."""

    def __init__(self, solver, points, EA, bonded: bool = True, ds: float | None = None, name: str | None = None):
        self.name = name
        P = np.asarray(points, float)
        self.points = resample(P, ds or default_spacing(solver)) if bonded else P
        self.bonded = bonded
        G, L, self.t = segment_operator(solver.locator, self.points)
        if not bonded:
            G, L = sp.csr_matrix(G.sum(axis=0)), np.array([L.sum()])
        self.G, self.L = G, L
        self.EA = np.broadcast_to(np.asarray(EA, float), self.L.shape).copy()
        self.K = (self.G.T @ sp.diags(self.EA / self.L) @ self.G).tocsr()

    def forces(self, u: np.ndarray, P0=0.0) -> np.ndarray:
        """axial force of each piece (tension +)"""
        return self.EA / self.L * (self.G @ u) + np.broadcast_to(np.asarray(P0, float), self.L.shape)

    def prestress_load(self, P) -> np.ndarray:
        """the load −Gᵀ P an initial tension P (one value or one per piece) puts on the structure"""
        return -(self.G.T @ np.broadcast_to(np.asarray(P, float), self.L.shape))
