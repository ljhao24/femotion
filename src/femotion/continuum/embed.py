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
    """finds points in the cells of a solver's mesh (scikit-fem reference cells: unit simplex or unit square / cube).

    Candidates come from a uniform grid of bins holding every cell whose bounding box overlaps the bin, so a point is
    tested against every cell that can contain it — exact for elongated or curved cells, and cheap for points in voids
    (no candidates). Points with no containing cell fall back to their nearest cells by centroid (for `snap` and the
    error message)."""

    def __init__(self, solver, candidates: int = 12, max_bins_per_cell: int = 512):
        from scipy.spatial import cKDTree
        self.s = solver
        sk = solver.sk
        self.sk = sk
        self.mapping = solver.sb.mapping
        self.simplex = type(sk).__name__.startswith(("MeshTri", "MeshTet"))
        V = sk.p[:, sk.t]  # (dim, nv, nel)
        self.lo, self.hi = V.min(axis=1).T, V.max(axis=1).T  # cell bounding boxes (nel, dim)
        span = np.ptp(sk.p, axis=1)
        pad = 1e-9 * max(span.max(), 1e-300)
        self.lo, self.hi = self.lo - pad, self.hi + pad
        self.centroids = V.mean(axis=1).T
        self.tree = cKDTree(self.centroids)
        self.k = min(candidates, sk.t.shape[1])
        h = np.maximum(np.median(self.hi - self.lo, axis=0), 1e-12)
        while True:
            i0 = np.floor((self.lo - self.lo.min(axis=0)) / h).astype(np.int64)
            i1 = np.floor((self.hi - self.lo.min(axis=0)) / h).astype(np.int64)
            ext = i1 - i0 + 1
            if np.prod(ext.max(axis=0)) <= max_bins_per_cell:
                break
            h = h * 2
        self.h, self.origin = h, self.lo.min(axis=0)
        self.shape = i1.max(axis=0) + 1
        cells, keys = [], []
        dim = len(h)
        for off in np.ndindex(*ext.max(axis=0)):
            o = np.array(off)
            ok = np.all(i0 + o <= i1, axis=1)
            b = i0[ok] + o
            cells.append(np.where(ok)[0])
            keys.append(np.ravel_multi_index(tuple(b.T), tuple(self.shape)) if dim else b)
        cells, keys = np.concatenate(cells), np.concatenate(keys)
        order = np.argsort(keys, kind="stable")
        self.bin_cells = cells[order]
        self.bin_keys, self.bin_start = np.unique(keys[order], return_index=True)
        self.bin_end = np.r_[self.bin_start[1:], len(order)]

    def _pairs(self, P: np.ndarray):
        """(point index, cell) pairs: every cell whose bounding box holds the point"""
        b = np.floor((P - self.origin) / self.h).astype(np.int64)
        inside = np.all((b >= 0) & (b < self.shape), axis=1)
        idx = np.where(inside)[0]
        key = np.ravel_multi_index(tuple(b[idx].T), tuple(self.shape))
        j = np.searchsorted(self.bin_keys, key)
        j = np.minimum(j, len(self.bin_keys) - 1)
        hit = self.bin_keys[j] == key
        idx, j = idx[hit], j[hit]
        cnt = self.bin_end[j] - self.bin_start[j]
        pi = np.repeat(idx, cnt)
        starts = np.repeat(self.bin_start[j], cnt)
        within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ci = self.bin_cells[starts + within]
        box = np.all((P[pi] >= self.lo[ci]) & (P[pi] <= self.hi[ci]), axis=1)
        return pi[box], ci[box]

    def _best(self, P: np.ndarray, chunk: int = 200_000):
        """least violation over the candidate cells of every point: (violation, cell, reference coordinates)"""
        n = len(P)
        best_v = np.full(n, np.inf)
        best_c = np.zeros(n, np.int64)
        best_X = np.zeros((P.shape[1], n))
        pi, ci = self._pairs(P)
        for a in range(0, len(pi), chunk):
            p, c = pi[a:a + chunk], ci[a:a + chunk]
            X = inverse_map(self.mapping, P[p].T[:, :, None], c)[:, :, 0]
            v = _violation(X, self.simplex)
            o = np.lexsort((v, p))  # per point, the least violation first
            first = np.r_[True, p[o][1:] != p[o][:-1]]
            k = o[first]
            better = v[k] < best_v[p[k]]
            sel = k[better]
            best_v[p[sel]], best_c[p[sel]], best_X[:, p[sel]] = v[sel], c[sel], X[:, sel]
        return best_v, best_c, best_X

    def _nearest(self, P: np.ndarray):
        """least violation over the nearest cells by centroid (points outside every cell)"""
        _, cand = self.tree.query(P, k=self.k)
        cand = cand.reshape(len(P), -1)
        best_v = np.full(len(P), np.inf)
        best_c = np.zeros(len(P), np.int64)
        best_X = np.zeros((P.shape[1], len(P)))
        for j in range(cand.shape[1]):
            X = inverse_map(self.mapping, P.T[:, :, None], cand[:, j])[:, :, 0]
            v = _violation(X, self.simplex)
            b = v < best_v
            best_v[b], best_c[b], best_X[:, b] = v[b], cand[b, j], X[:, b]
        return best_v, best_c, best_X

    def locate(self, points, tol: float = 1e-6, snap: float = 0.0):
        """(cells (n,), reference coordinates (dim, n)); points on faces go to the least violated cell. Points outside the
        mesh by more than tol (in reference coordinates of the nearest cell) raise ValueError, unless within `snap` (e.g.
        1.0: one cell): those are moved onto the nearest cell (anchor plates poking out of a sloping end face)"""
        P = np.atleast_2d(np.asarray(points, float))
        v, c, X = self._best(P)
        out = v > tol
        if out.any():
            v2, c2, X2 = self._nearest(P[out])
            idx = np.where(out)[0]
            v[idx], c[idx], X[:, idx] = v2, c2, X2
            bad = v > max(tol, snap)
            if bad.any():
                raise ValueError(f"{int(bad.sum())} point(s) lie outside the mesh, e.g. {P[np.where(bad)[0][0]]}")
        if self.simplex:  # onto the simplex: clip negatives, then scale the sum back to ≤ 1
            X = np.clip(X, 0.0, None)
            sm = X.sum(axis=0)
            X = X / np.maximum(sm, 1.0)
        else:
            X = np.clip(X, 0.0, 1.0)
        return c, X

    def inside(self, points, tol: float = 1e-6, snap: float = 0.0) -> np.ndarray:
        """bool per point: inside the mesh (within tol, or within `snap` cells of it)"""
        P = np.atleast_2d(np.asarray(points, float))
        v, _, _ = self._best(P)
        ok = v <= tol
        if snap > 0 and (~ok).any():
            idx = np.where(~ok)[0]
            ok[idx] = self._nearest(P[idx])[0] <= snap
        return ok

    def _phi(self, cells, X, grad: bool = False):
        """scalar shape functions (Nbfun, n) or their gradients (Nbfun, dim, n) at the points"""
        sb = self.s.sb
        out = []
        for k in range(sb.Nbfun):
            f = sb.elem.gbasis(self.mapping, X[:, :, None], k, tind=cells)[0]
            out.append(np.asarray(f.grad)[:, :, 0] if grad else np.asarray(f)[:, 0])
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
