"""Continuum linear elasticity on scikit-fem (extra "continuum"): plane stress, plane strain and 3D solids, linear or
quadratic elements, direct or algebraic-multigrid solution.

    from femotion.mesh import gmsh as G
    from femotion.continuum import Elastic
    m = G.plane(...)
    s = Elastic(m, E=206e9, nu=0.3, kind="plane_stress", order=2, thick=0.06)
    s.traction(lambda x: np.isclose(x[0], 10), (1e6, 0))      # surface traction: constant or t(x) → components
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))           # select nodes by coordinates; all components by default
    s.solve()
    res = s.result_mesh()                                      # Mesh including the mid-edge nodes of quadratic cells
    u = s.nodal_displacement(); S = s.nodal_stress()           # both on the nodes of res

Conventions: 2D stress components (xx, yy, xy); 3D (xx, yy, zz, xy, yz, zx). Tractions are per unit area (times the
thickness in 2D). Dof numbering (scikit-fem vector elements): component c of scalar dof k is dim·k + c.
"""
from __future__ import annotations

import time

import numpy as np

from ..mesh.core import Mesh, orient


def lame(E: float, nu: float, kind: str = "solid"):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    if kind == "plane_stress":
        lam = 2 * lam * mu / (lam + 2 * mu)
    return lam, mu


def _elements(kind: str, order: int):
    from skfem import ElementHex1, ElementHex2, ElementQuad1, ElementQuad2, ElementTetP1, ElementTetP2, ElementTriP1, ElementTriP2
    return {("tri3", 1): ElementTriP1, ("tri3", 2): ElementTriP2, ("quad4", 1): ElementQuad1, ("quad4", 2): ElementQuad2,
            ("tet4", 1): ElementTetP1, ("tet4", 2): ElementTetP2, ("hex8", 1): ElementHex1, ("hex8", 2): ElementHex2}[(kind, order)]()


class Elastic:
    def __init__(self, mesh: Mesh, E: float, nu: float, kind: str | None = None, order: int = 1, thick: float = 1.0,
                 E_cell: np.ndarray | None = None):
        """kind: plane_stress (default in 2D), plane_strain, solid (automatic in 3D).
        E_cell: Young's modulus per cell (topology optimisation, material zones); overrides E when given."""
        from skfem import Basis, BilinearForm, ElementVector, asm
        from skfem.helpers import ddot, eye, sym_grad, trace

        t0 = time.time()
        self.mesh = mesh
        self.sk = mesh.to_skfem()
        self.dim = self.sk.p.shape[0]
        self.kind = kind or ("solid" if self.dim == 3 else "plane_stress")
        self.order = order
        self.thick = thick if self.dim == 2 else 1.0
        self.scalar_elem = _elements(mesh.kind if mesh.kind != "tri6" else "tri3", order)
        self.elem = ElementVector(self.scalar_elem)
        self.ib = Basis(self.sk, self.elem)
        self.lam, self.mu = lame(E, nu, self.kind)
        self.E, self.nu = E, nu
        lam, mu = self.lam, self.mu
        self._scale = None if E_cell is None else np.asarray(E_cell, float) / E

        @BilinearForm
        def a(u, v, w):
            e = sym_grad(u)
            return w.s * ddot(2 * mu * e + lam * eye(trace(e), e.shape[0]), sym_grad(v))
        scale = np.ones(self.sk.t.shape[1]) if self._scale is None else self._scale
        self.K = asm(a, self.ib, s=np.repeat(scale[:, None], self.ib.X.shape[-1], axis=1)) * self.thick
        self.f = np.zeros(self.ib.N)
        self._fixed: list[np.ndarray] = []
        self._presc: dict[int, float] = {}
        self.t_assemble = time.time() - t0

    # ── loads ───────────────────────────────────────────────────────────
    def _facets(self, where):
        if isinstance(where, np.ndarray):
            return where
        return self.sk.facets_satisfying(where, boundaries_only=True)

    def boundary_facets_where(self, pred_vertices) -> np.ndarray:
        """select boundary facets by the coordinates of all their vertices: pred(fx) → bool[nf], fx of shape (dim, vertices per facet, nf)"""
        bf = self.sk.boundary_facets()
        return bf[pred_vertices(self.sk.p[:, self.sk.facets[:, bf]])]

    def traction(self, where, t, intorder: int | None = None) -> np.ndarray:
        """where: boundary facets (a predicate pred(x) on facet midpoints, or an array of facet ids); t: a constant vector or
        t(x) → components. Returns this part of the load vector"""
        from skfem import FacetBasis, LinearForm, asm

        fb = FacetBasis(self.sk, self.elem, facets=self._facets(where), intorder=intorder or 2 * self.order + 2)
        d = self.dim

        @LinearForm
        def lf(v, w):
            tv = t(w.x) if callable(t) else [c + 0 * w.x[0] for c in t]
            return sum(tv[i] * v[i] for i in range(d))
        f = asm(lf, fb) * self.thick
        self.f += f
        return f

    def body_force(self, b) -> np.ndarray:
        """body force per unit volume (e.g. ρg downwards): constant vector"""
        from skfem import LinearForm, asm
        d = self.dim

        @LinearForm
        def lf(v, w):
            return sum(b[i] * v[i] for i in range(d))
        f = asm(lf, self.ib) * self.thick
        self.f += f
        return f

    def point_load(self, p, F):
        """point load at the vertex nearest to p"""
        n = int(np.argmin(np.linalg.norm(self.sk.p.T - np.asarray(p, float), axis=1)))
        for i, c in enumerate(F):
            self.f[self.ib.nodal_dofs[i, n]] += c

    def scale_load(self, total: float, direction) -> float:
        """scale the current load so that its resultant along `direction` equals `total` (to match a design value after
        numerical integration); returns the factor"""
        R = self.resultant()
        k = total / float(np.dot(R, direction))
        self.f *= k
        return k

    def resultant(self, f=None) -> np.ndarray:
        f = self.f if f is None else f
        return np.array([f[i::self.dim].sum() for i in range(self.dim)])

    # ── supports ────────────────────────────────────────────────────────
    def fix(self, where, comps=None, value: float = 0.0):
        """where: a predicate pred(x) on node coordinates (also selects mid-edge nodes of quadratic cells) or an array of
        vertex ids; comps: all components by default"""
        comps = range(self.dim) if comps is None else comps
        if isinstance(where, np.ndarray):
            dofs = self.ib.get_dofs(nodes=where)
        else:
            dofs = self.ib.get_dofs(where)
        for c in comps:
            d = dofs.all(f"u^{c + 1}")
            self._fixed.append(d)
            if value:
                for k in d:
                    self._presc[int(k)] = value

    def fix_point(self, p, comps=None):
        n = int(np.argmin(np.linalg.norm(self.sk.p.T - np.asarray(p, float), axis=1)))
        self.fix(np.array([n]), comps)

    # ── solve ───────────────────────────────────────────────────────────
    def solve(self, method: str = "auto", tol: float = 1e-10, verbose: bool = False):
        """method: direct (sparse LU), amg (pyamg smoothed aggregation + CG with rigid-body modes as near-null space),
        auto (amg above 150,000 dofs)"""
        from scipy.sparse.linalg import spsolve
        t0 = time.time()
        D = np.unique(np.concatenate(self._fixed)) if self._fixed else np.zeros(0, int)
        free = np.setdiff1d(np.arange(self.ib.N), D)
        u = np.zeros(self.ib.N)
        for k, v in self._presc.items():
            u[k] = v
        b = self.f[free] - (self.K[free][:, D] @ u[D] if len(self._presc) else 0)
        A = self.K[free][:, free].tocsr()
        if method == "auto":
            method = "amg" if len(free) > 150_000 else "direct"
        self.cg_iterations = 0
        if method == "direct":
            u[free] = spsolve(A.tocsc(), b)
        else:
            import pyamg
            B = self._rigid_modes()[free]
            d = self.dim
            fb = free.reshape(-1, d) if len(free) % d == 0 else None
            if fb is not None and np.all(fb[:, 0] % d == 0) and np.all(fb == fb[:, :1] + np.arange(d)):
                A = A.tobsr(blocksize=(d, d))  # whole nodes free: aggregate node blocks (far fewer CG steps for quadratic cells)
            # prolongation smoothing weighted row by row (Gershgorin bound): pyamg's default estimates the spectral radius from
            # a random vector, and about 1 draw in 15 underestimates it for quadratic solids, so that CG diverges
            ml = pyamg.smoothed_aggregation_solver(A, B=B, symmetry="hermitian", max_coarse=500,
                                                   smooth=("jacobi", {"omega": 4.0 / 3.0, "weighting": "local"}))
            resid: list[float] = []
            u[free] = ml.solve(b, tol=tol, accel="cg", maxiter=2000, residuals=resid)
            self.cg_iterations = len(resid) - 1
        self.rel_residual = float(np.linalg.norm(b - A @ u[free]) / max(np.linalg.norm(b), 1e-300))
        if not self.rel_residual < 1e-6:
            raise RuntimeError(f"{method} solve did not converge (relative residual {self.rel_residual:.1e}); "
                               "try method='direct', check the supports, or improve the mesh")
        self.u, self.free, self.fixed = u, free, D
        self.t_solve = time.time() - t0
        if verbose:
            print(f"  assembly {self.t_assemble:.1f}s, solve {self.t_solve:.1f}s ({method}), {self.ib.N:,} dofs, relative residual {self.rel_residual:.1e}")
        return u

    def _rigid_modes(self):
        X = self.ib.doflocs  # location of every dof (dim, N)
        c = X - X.mean(axis=1, keepdims=True)
        n, d = self.ib.N, self.dim
        comp = np.arange(n) % d
        if d == 2:
            B = np.zeros((n, 3))
            for i in range(2):
                B[comp == i, i] = 1
            B[comp == 0, 2] = -c[1, comp == 0]
            B[comp == 1, 2] = c[0, comp == 1]
            return B
        B = np.zeros((n, 6))
        for i in range(3):
            B[comp == i, i] = 1
        for k, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):  # rotations about x, y, z
            B[comp == a, 3 + k] = -c[b, comp == a]
            B[comp == b, 3 + k] = c[a, comp == b]
        return B

    # ── results ─────────────────────────────────────────────────────────
    def reactions(self) -> np.ndarray:
        """resultant reaction at the supports (per component)"""
        r = self.K @ self.u - self.f
        out = np.zeros(self.dim)
        for i in range(self.dim):
            d = self.fixed[self.fixed % self.dim == i]
            out[i] = r[d].sum()
        return out

    def compliance(self) -> float:
        return float(self.f @ self.u)

    def _node_basis(self):
        """a Basis evaluated at all nodes of every cell (including mid-edge nodes of quadratic cells)"""
        from skfem import Basis
        ref = self.scalar_elem.doflocs.T
        nb = Basis(self.sk, self.elem, quadrature=(ref, np.ones(ref.shape[1])))
        sb = Basis(self.sk, self.scalar_elem)
        return nb, sb

    def result_mesh(self) -> Mesh:
        """the mesh whose nodes match the result values one-to-one: P1 is the input mesh; 2D P2 is tri6 (with mid-edge nodes);
        3D P2 keeps corner tetrahedra only"""
        if self.order == 1 or self.dim == 3:
            return Mesh.from_skfem(self.sk)  # 3D quadratic: corners only
        _, sb = self._node_basis()
        if self.mesh.kind in ("tri3", "tri6"):
            p = sb.doflocs.T.copy()
            return Mesh(p, {"tri6": orient(p, "tri6", sb.element_dofs.T.copy())})
        return Mesh(self.sk.p.T.copy(), {"quad4": self.sk.t.T.copy()})  # Quad2: corners only

    def _n_result_nodes(self):
        return self.result_mesh().n_points

    def nodal_displacement(self) -> np.ndarray:
        """(N, dim) on the nodes of result_mesh()"""
        return self.u.reshape(-1, self.dim)[: self._n_result_nodes()]

    def stress_at_nodes_per_cell(self) -> np.ndarray:
        """stress of every cell at each of its nodes: (ncomp, nel, nloc)"""
        nb, _ = self._node_basis()
        G = nb.interpolate(self.u).grad  # (dim, dim, nel, nloc)
        eps = 0.5 * (G + G.transpose(1, 0, 2, 3))
        tr = sum(eps[i, i] for i in range(self.dim))
        lam, mu = self.lam, self.mu
        s = lambda i, j: (2 * mu * eps[i, j] + (lam * tr if i == j else 0))  # noqa: E731
        comps = [s(0, 0), s(1, 1), s(0, 1)] if self.dim == 2 else [s(0, 0), s(1, 1), s(2, 2), s(0, 1), s(1, 2), s(2, 0)]
        S = np.stack(comps)
        if self._scale is not None:
            S = S * self._scale[None, :, None]
        return S

    def cell_stress(self) -> np.ndarray:
        """cell-average stress (nel, ncomp)"""
        return self.stress_at_nodes_per_cell().mean(axis=2).T

    def nodal_stress(self) -> np.ndarray:
        """cell stresses at the nodes averaged with area / volume weights: (N, ncomp) on the nodes of result_mesh()"""
        S = self.stress_at_nodes_per_cell()
        _, sb = self._node_basis()
        ed = sb.element_dofs  # (nloc, nel)
        w = np.broadcast_to(self.ib.dx.sum(axis=1)[:, None], (ed.shape[1], ed.shape[0])).T  # (nloc, nel)
        acc = np.zeros((S.shape[0], sb.N))
        wsum = np.bincount(ed.ravel(), weights=w.ravel(), minlength=sb.N)
        for c in range(S.shape[0]):
            acc[c] = np.bincount(ed.ravel(), weights=(S[c].T * w).ravel(), minlength=sb.N)
        return (acc / np.maximum(wsum, 1e-300)).T[: self._n_result_nodes()]

    def strain_energy_density(self) -> np.ndarray:
        """strain energy density per cell (cell average)"""
        from skfem import Functional
        from skfem.helpers import ddot, sym_grad, trace, eye
        lam, mu = self.lam, self.mu

        @Functional
        def W(w):
            e = sym_grad(w["u"])
            return 0.5 * ddot(2 * mu * e + lam * eye(trace(e), e.shape[0]), e)
        e = W.elemental(self.ib, u=self.ib.interpolate(self.u))
        return e / self.ib.dx.sum(axis=1)
