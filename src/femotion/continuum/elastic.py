"""Continuum linear elasticity on scikit-fem (extra "continuum"): plane stress, plane strain and 3D solids, linear or
quadratic elements, several materials, elastic supports, embedded bars, any number of load cases solved with one
factorisation, direct or algebraic-multigrid solution.

    from femotion.mesh import gmsh as G
    from femotion.continuum import Elastic
    m = G.plane(...)
    s = Elastic(m, E=206e9, nu=0.3, kind="plane_stress", order=2, thick=0.06)
    s.traction(lambda x: np.isclose(x[0], 10), (1e6, 0))      # surface traction: constant or t(x) → components
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))           # select nodes by coordinates; all components by default
    s.solve()
    res = s.result_mesh()                                      # Mesh including the mid-edge nodes of quadratic cells
    u = s.nodal_displacement(); S = s.nodal_stress()           # both on the nodes of res

A bridge deck, with materials per cell group, bearings as elastic supports and several load cases:

    s = Elastic(deck, materials={"girder": Isotropic(3.0e10, 0.2, 1e-5), "pavement": Isotropic(3.0e10, 0.2, 1e-5)},
                order=2, serendipity=True)
    for i, (x0, y0) in enumerate(bearings):                    # stiffness per unit area of each bearing plate
        s.spring(s.facets(rect(x0 - 0.1, x0 + 0.1, y0 - 0.125, y0 + 0.125), normal=(0, 0, -1)), k=(kh, kh, kv), name=f"B{i}")
    s.case("自重").body_force((0, 0, -26e3), groups=["girder"])
    s.case("预应力").tendon(td)
    s.solve()                                                  # every case, one factorisation / one AMG hierarchy
    r = s.result("自重") + s.result("预应力")                    # results combine linearly
    r.nodal_stress(); r.spring_force("B0"); r.section((8.0, 0, 0), (1, 0, 0))

Single-case use (traction, body_force, point_load, fix, solve, nodal_stress, …) works on the case "default", as before
load cases existed. Conventions: 2D stress components (xx, yy, xy); 3D (xx, yy, zz, xy, yz, zx). Tractions are per unit
area (times the thickness in 2D). Dof numbering (scikit-fem vector elements): component c of scalar dof k is dim·k + c.
Large quadratic models are assembled in chunks of cells (`chunk`), so memory stays bounded by the chunk, not the mesh.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp

from ..mesh.core import Mesh, orient
from .materials import Isotropic, lame  # noqa: F401  (lame re-exported)


def _elements(kind: str, order: int, serendipity: bool = False):
    from skfem import (ElementHex1, ElementHex2, ElementHexS2, ElementQuad1, ElementQuad2, ElementQuadS2, ElementTetP1,
                       ElementTetP2, ElementTriP1, ElementTriP2)
    if serendipity and order == 2:
        if kind == "hex8":
            return ElementHexS2()
        if kind == "quad4":
            return ElementQuadS2()
    return {("tri3", 1): ElementTriP1, ("tri3", 2): ElementTriP2, ("quad4", 1): ElementQuad1, ("quad4", 2): ElementQuad2,
            ("tet4", 1): ElementTetP1, ("tet4", 2): ElementTetP2, ("hex8", 1): ElementHex1, ("hex8", 2): ElementHex2}[(kind, order)]()


def _voigt(e, dim: int):
    """engineering strain components of a symmetric gradient (dim, dim, …)"""
    if dim == 2:
        return [e[0, 0], e[1, 1], 2 * e[0, 1]]
    return [e[0, 0], e[1, 1], e[2, 2], 2 * e[0, 1], 2 * e[1, 2], 2 * e[2, 0]]


def _arr(f):
    """the array of a scikit-fem field passed through asm(..., name=array)"""
    return np.asarray(f)


class Elastic:
    def __init__(self, mesh: Mesh, E: float | None = None, nu: float | None = None, kind: str | None = None, order: int = 1,
                 thick: float = 1.0, E_cell: np.ndarray | None = None, *, material=None, materials: dict | None = None,
                 serendipity: bool = False, chunk: int | None = None):
        """kind: plane_stress (default in 2D), plane_strain, solid (automatic in 3D). Materials: `material` (or E, nu for
        an Isotropic one) for every cell, `materials` = {cell group: material} overriding it on those groups; every cell
        needs exactly one. E_cell: Young's modulus per cell (topology optimisation) scaling the default isotropic
        material. serendipity: 8-node quadrilaterals / 20-node hexahedra for order 2 (Lagrange 9 / 27 nodes otherwise).
        chunk: cells assembled at a time (default: about 0.5 GB of shape-function data)."""
        from skfem import Basis, ElementVector

        t0 = time.time()
        self.mesh = mesh
        self.sk = mesh.to_skfem()
        self.dim = self.sk.p.shape[0]
        self.kind = kind or ("solid" if self.dim == 3 else "plane_stress")
        self.order = order
        self.thick = thick if self.dim == 2 else 1.0
        self.scalar_elem = _elements(mesh.kind if mesh.kind != "tri6" else "tri3", order, serendipity)
        self.elem = ElementVector(self.scalar_elem)
        simplex = mesh.kind in ("tri3", "tri6", "tet4")
        q1 = (np.full((self.dim, 1), 1 / (self.dim + 1) if simplex else 0.5),
              np.array([1 / math.factorial(self.dim) if simplex else 1.0]))
        # ib (vector) and sb (scalar): dof bookkeeping and interpolation only — one quadrature point per cell keeps them
        # cheap on large meshes, so their dx is not an integration rule (cell volumes: Elastic.cell_volumes)
        self.ib = Basis(self.sk, self.elem, quadrature=q1)
        self.sb = Basis(self.sk, self.scalar_elem, quadrature=q1)
        self._setup_materials(E, nu, material, materials, E_cell)
        nb = self.scalar_elem.doflocs.shape[0] * self.dim
        self._nq = Basis(self.sk, self.elem, elements=np.array([0])).X.shape[1]
        self.chunk = chunk or max(64, int(1.2e7 / (nb * self._nq)))
        self._extra: dict[str, sp.spmatrix] = {}
        self._K = None
        self.springs: dict[str, object] = {}
        self.bars: dict[str, object] = {}
        self.cases: dict = {}
        self.results: dict = {}
        self._fixed: list[np.ndarray] = []
        self._locator = None
        self.Kc = self._assemble_K()
        self.t_assemble = time.time() - t0

    # ── materials and assembly ──────────────────────────────────────────
    def _setup_materials(self, E, nu, material, materials, E_cell):
        nel = self.sk.t.shape[1]
        default = material if material is not None else (Isotropic(E, nu) if E is not None else None)
        mats: list = []
        idx = np.full(nel, -1, np.int64)

        def key(m):
            for i, x in enumerate(mats):
                if x is m:
                    return i
            mats.append(m)
            return len(mats) - 1

        if default is not None:
            idx[:] = key(default)
        seen = np.zeros(nel, bool)
        for g, m in (materials or {}).items():
            cells = self._cells_of([g])
            if seen[cells].any():
                raise ValueError(f"cell group {g!r} overlaps another group of `materials`")
            seen[cells] = True
            idx[cells] = key(m)
        if (idx < 0).any():
            raise ValueError(f"{int((idx < 0).sum())} cells have no material: pass E, nu / material, or cover every cell "
                             "with `materials`")
        self.mats = mats
        self.cell_material = idx
        self.material = default
        if isinstance(default, Isotropic):
            self.E, self.nu = default.E, default.nu
            self.lam, self.mu = default.lame(self.kind)
        self._scale = None
        if E_cell is not None:
            if not isinstance(default, Isotropic) or materials:
                raise ValueError("E_cell scales one isotropic material (E, nu)")
            self._scale = np.asarray(E_cell, float) / default.E

    def _cells_of(self, groups) -> np.ndarray:
        """cell ids (scikit-fem order = mesh order of the main block) of the given cell groups; all cells for None"""
        nel = self.sk.t.shape[1]
        if groups is None:
            return np.arange(nel)
        if isinstance(groups, str):
            groups = [groups]
        k = self.mesh.kind
        out = []
        for g in groups:
            if g not in self.mesh.cell_groups:
                raise KeyError(f"no cell group {g!r} (have {sorted(self.mesh.cell_groups)})")
            out.append(np.asarray(self.mesh.cell_groups[g].get(k, []), np.int64))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def _chunks(self, cells: np.ndarray):
        n = max(1, int(np.ceil(len(cells) / self.chunk)))
        return [c for c in np.array_split(cells, n) if len(c)]

    def _material_chunks(self, cells: np.ndarray | None = None):
        """(material, cell chunk) pairs covering `cells` (all by default)"""
        cells = np.arange(self.sk.t.shape[1]) if cells is None else np.asarray(cells)
        for i, mat in enumerate(self.mats):
            sel = cells[self.cell_material[cells] == i]
            for c in self._chunks(sel):
                yield mat, c

    def D(self, mat) -> np.ndarray:
        """constitutive matrix of a material in this model's dimension and kind"""
        return mat.D(self.dim, self.kind)

    def _assemble_K(self):
        """K = Σ ∫ Bᵀ D B dV, assembled cell by cell with batched matrix products (scikit-fem gives the shape-function
        gradients and the measure; a BilinearForm would evaluate the form once per pair of the 60 × 60 basis functions of a
        20-node hexahedron, about a hundred times slower)"""
        from skfem import Basis
        dim = self.dim
        ncomp = 3 if dim == 2 else 6
        N = self.ib.N
        parts = []
        for mat, cells in self._material_chunks():
            sb = Basis(self.sk, self.scalar_elem, elements=cells)
            Dm = self.D(mat)
            dx = np.asarray(sb.dx) * self.thick
            if self._scale is not None:
                dx = dx * self._scale[cells][:, None]
            B = self._B(sb)  # (nel, nq, ncomp, n)
            nel, nq, _, n = B.shape
            DB = np.einsum("ij,eqjA->eqiA", Dm, B) * dx[:, :, None, None]
            Ke = np.matmul(B.reshape(nel, nq * ncomp, n).transpose(0, 2, 1), DB.reshape(nel, nq * ncomp, n))
            gd = self._vector_dofs(sb)  # (nel, n)
            r = np.broadcast_to(gd[:, :, None], Ke.shape)
            c = np.broadcast_to(gd[:, None, :], Ke.shape)
            parts.append(sp.csr_matrix((Ke.ravel(), (r.ravel(), c.ravel())), shape=(N, N)))
        while len(parts) > 1:  # pairwise sums: O(nnz log chunks) instead of O(nnz · chunks)
            parts = [parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
        return parts[0].tocsr() if parts else sp.csr_matrix((N, N))

    def _B(self, sb) -> np.ndarray:
        """strain-displacement matrices (nel, nq, ncomp, dim·nb) from a scalar basis; column dim·k + c is component c of
        scalar function k (the vector dof order)"""
        dim, nb = self.dim, sb.Nbfun
        G = np.stack([np.asarray(sb.basis[k][0].grad) for k in range(nb)])  # (nb, dim, nel, nq)
        nel, nq = G.shape[2], G.shape[3]
        ncomp = 3 if dim == 2 else 6
        B = np.zeros((nel, nq, ncomp, dim * nb))
        g = G.transpose(2, 3, 1, 0)  # (nel, nq, dim, nb)
        if dim == 2:
            B[:, :, 0, 0::2] = g[:, :, 0]
            B[:, :, 1, 1::2] = g[:, :, 1]
            B[:, :, 2, 0::2] = g[:, :, 1]
            B[:, :, 2, 1::2] = g[:, :, 0]
            return B
        B[:, :, 0, 0::3] = g[:, :, 0]
        B[:, :, 1, 1::3] = g[:, :, 1]
        B[:, :, 2, 2::3] = g[:, :, 2]
        B[:, :, 3, 0::3] = g[:, :, 1]  # γxy = ∂ux/∂y + ∂uy/∂x
        B[:, :, 3, 1::3] = g[:, :, 0]
        B[:, :, 4, 1::3] = g[:, :, 2]  # γyz = ∂uy/∂z + ∂uz/∂y
        B[:, :, 4, 2::3] = g[:, :, 1]
        B[:, :, 5, 2::3] = g[:, :, 0]  # γzx = ∂uz/∂x + ∂ux/∂z
        B[:, :, 5, 0::3] = g[:, :, 2]
        return B

    def _vector_dofs(self, sb) -> np.ndarray:
        """global vector dofs (nel, dim·nb) of the cells of a scalar basis, in the column order of _B"""
        sd = np.asarray(sb.element_dofs)  # (nb, nel)
        d = self.dim
        return (d * sd.T[:, :, None] + np.arange(d)[None, None, :]).reshape(sd.shape[1], -1)

    @property
    def K(self):
        """total stiffness: continuum + elastic supports + embedded bars"""
        if self._K is None:
            K = self.Kc
            for M in self._extra.values():
                K = K + M
            self._K = K.tocsr()
        return self._K

    def _add_matrix(self, name: str, M):
        if name in self._extra:
            raise ValueError(f"{name!r} is already used by a spring or bar")
        self._extra[name] = M.tocsr()
        self._K = None

    @property
    def cell_volumes(self) -> np.ndarray:
        """volume (area × thickness in 2D) of every cell, by the assembly quadrature"""
        if getattr(self, "_vol", None) is None:
            from skfem import Basis
            v = np.zeros(self.sk.t.shape[1])
            for c in self._chunks(np.arange(self.sk.t.shape[1])):
                v[c] = np.asarray(Basis(self.sk, self.scalar_elem, elements=c).dx).sum(axis=1)
            self._vol = v * self.thick
        return self._vol

    @property
    def h(self) -> float:
        """median edge length of the mesh (a length scale for tolerances)"""
        if getattr(self, "_h", None) is None:
            e = self.mesh.edges()
            self._h = float(np.median(np.linalg.norm(self.mesh.points[e[:, 1]] - self.mesh.points[e[:, 0]], axis=1)))
        return self._h

    @property
    def locator(self):
        """finds arbitrary points in the cells (femotion.continuum.embed.Locator)"""
        if self._locator is None:
            from .embed import Locator
            self._locator = Locator(self)
        return self._locator

    # ── boundary facets ─────────────────────────────────────────────────
    def boundary_normals(self, facets: np.ndarray) -> np.ndarray:
        """outward unit normals (dim, n) of boundary facets"""
        sk = self.sk
        P = sk.p[:, sk.facets[:, facets]]  # (dim, nv, n)
        c = P.mean(axis=1)
        if self.dim == 2:
            d = P[:, 1] - P[:, 0]
            n = np.array([d[1], -d[0]])
        else:
            n = facet_normals(P)
        owner = sk.f2t[0, facets]
        cc = sk.p[:, sk.t[:, owner]].mean(axis=1)
        flip = np.einsum("ij,ij->j", n, c - cc) < 0
        n[:, flip] *= -1
        return n / np.linalg.norm(n, axis=0)

    def facets(self, where=None, normal=None, angle: float = 30.0) -> np.ndarray:
        """boundary facet ids: where(x) on facet midpoints (dim, n) → bool, and/or an outward normal within `angle` degrees
        of the direction `normal` (e.g. (0, 0, 1): the top surfaces)"""
        sk = self.sk
        bf = sk.boundary_facets()
        keep = np.ones(len(bf), bool)
        if where is not None:
            keep &= np.asarray(where(sk.p[:, sk.facets[:, bf]].mean(axis=1)), bool)
        if normal is not None:
            d = np.asarray(normal, float)
            keep &= (d / np.linalg.norm(d)) @ self.boundary_normals(bf) >= np.cos(np.radians(angle))
        return bf[keep]

    def _facets(self, where):
        if isinstance(where, np.ndarray):
            return where
        return self.sk.facets_satisfying(where, boundaries_only=True)

    def boundary_facets_where(self, pred_vertices) -> np.ndarray:
        """select boundary facets by the coordinates of all their vertices: pred(fx) → bool[nf], fx of shape (dim, vertices per facet, nf)"""
        bf = self.sk.boundary_facets()
        return bf[pred_vertices(self.sk.p[:, self.sk.facets[:, bf]])]

    def _fbasis(self, facets, intorder=None):
        from skfem import FacetBasis
        return FacetBasis(self.sk, self.elem, facets=facets, intorder=intorder or 2 * self.order + 2)

    def facet_integral(self, facets, g, intorder=None) -> float:
        """∫ g(x) dA over boundary facets (times the thickness in 2D)"""
        from skfem import Functional, asm
        fb = self._fbasis(self._facets(facets), intorder)

        @Functional
        def F(w):
            return g(w.x)
        return float(asm(F, fb)) * self.thick

    # ── load vectors (used by femotion.continuum.loads.LoadCase) ─────────
    def _traction_vector(self, where, t, intorder=None) -> np.ndarray:
        from skfem import LinearForm, asm
        fb = self._fbasis(self._facets(where), intorder)
        d = self.dim

        @LinearForm
        def lf(v, w):
            tv = t(w.x) if callable(t) else [c + 0 * w.x[0] for c in t]
            return sum(tv[i] * v[i] for i in range(d))
        return asm(lf, fb) * self.thick

    def _pressure_vector(self, where, p, intorder=None) -> np.ndarray:
        from skfem import LinearForm, asm
        fb = self._fbasis(self._facets(where), intorder)
        d = self.dim

        @LinearForm
        def lf(v, w):
            pv = p(w.x) if callable(p) else p + 0 * w.x[0]
            return -sum(pv * w.n[i] * v[i] for i in range(d))
        return asm(lf, fb) * self.thick

    def _body_vector(self, b, cells) -> np.ndarray:
        from skfem import Basis, LinearForm, asm
        d = self.dim
        f = np.zeros(self.ib.N)

        @LinearForm
        def lf(v, w):
            bv = b(w.x) if callable(b) else [c + 0 * w.x[0] for c in b]
            return sum(bv[i] * v[i] for i in range(d))
        for c in self._chunks(np.asarray(cells)):
            f += asm(lf, Basis(self.sk, self.elem, elements=c)) * self.thick
        return f

    def _strain_vector(self, source) -> np.ndarray:
        """∫ Bᵀ D ε0 dV of an initial-strain source (cells, eps0(x, material))"""
        from skfem import Basis, LinearForm, asm
        from skfem.helpers import sym_grad
        cells, fn = source
        dim = self.dim
        ncomp = 3 if dim == 2 else 6
        f = np.zeros(self.ib.N)

        @LinearForm
        def lf(v, w):
            ev = _voigt(sym_grad(v), dim)
            S0 = _arr(w.sig0)
            return sum(ev[i] * S0[i] for i in range(ncomp))
        for mat, c in self._material_chunks(cells):
            b = Basis(self.sk, self.elem, elements=c)
            x = _arr(b.global_coordinates())
            e0 = np.broadcast_to(fn(x, mat), (ncomp,) + x.shape[1:])
            S0 = np.einsum("ij,j...->i...", self.D(mat), e0)
            if self._scale is not None:
                S0 = S0 * self._scale[c][None, :, None]
            f += asm(lf, b, sig0=S0) * self.thick
        return f

    # ── load cases ──────────────────────────────────────────────────────
    def case(self, name: str = "default"):
        """the load case `name` (created on first use; femotion.continuum.loads.LoadCase)"""
        from .loads import LoadCase
        if name not in self.cases:
            self.cases[name] = LoadCase(self, name)
        return self.cases[name]

    # single-case use: the case "default"
    @property
    def f(self) -> np.ndarray:
        """load vector of the case "default" (zeros if it does not exist; reading it creates nothing)"""
        return self.cases["default"].f if "default" in self.cases else np.zeros(self.ib.N)

    def traction(self, where, t, intorder: int | None = None) -> np.ndarray:
        """where: boundary facets (a predicate pred(x) on facet midpoints, or an array of facet ids); t: a constant vector or
        t(x) → components. Returns this part of the load vector"""
        c = self.case()
        before = c.f_ext.copy()
        c.traction(where, t, intorder)
        return c.f_ext - before

    def body_force(self, b, groups=None) -> np.ndarray:
        """body force per unit volume (e.g. ρg downwards): constant vector or b(x)"""
        c = self.case()
        before = c.f_ext.copy()
        c.body_force(b, groups)
        return c.f_ext - before

    def point_load(self, p, F):
        """point load at the vertex nearest to p"""
        self.case().point_load(p, F)

    def scale_load(self, total: float, direction) -> float:
        """scale the current load so that its resultant along `direction` equals `total` (to match a design value after
        numerical integration); returns the factor"""
        return self.case().scale_load(total, direction)

    def resultant(self, f=None) -> np.ndarray:
        """resultant of a load vector (default: the case "default")"""
        f = self.f if f is None else f
        return np.array([f[i::self.dim].sum() for i in range(self.dim)])

    # ── supports ────────────────────────────────────────────────────────
    def fix(self, where, comps=None, value: float = 0.0):
        """where: a predicate pred(X) on dof locations X (dim, N): every node where it holds — a point, a line, a face —
        including mid-edge nodes of quadratic cells (on curved boundaries those lie on the chord); or an array of vertex
        ids. comps: all components by default. value: an imposed displacement, part of the case "default" (other cases
        hold these dofs at zero; use case(name).displacement for settlements in named cases). Facet loads and springs
        select facets by their midpoints instead (traction, spring, facets)"""
        comps = range(self.dim) if comps is None else comps
        per = self.dofs_where(where, comps)  # every dof located where the predicate holds: points, lines, faces alike
        for c in comps:
            d = per[c]
            if len(d) == 0:
                raise ValueError("fix: the selection holds no nodes")
            self._fixed.append(d)
            if value:
                self.case().uD.update({int(k): float(value) for k in d})

    def fix_point(self, p, comps=None):
        n = int(np.argmin(np.linalg.norm(self.sk.p.T - np.asarray(p, float), axis=1)))
        self.fix(np.array([n]), comps)

    def spring(self, where, k, name: str | None = None):
        """an elastic support distributed over boundary facets (a bearing, a soil bed): k = stiffness per unit area, a
        vector per global axis or a dim × dim matrix (force per area on the solid = −k u). Returns the support"""
        from skfem import BilinearForm, asm
        facets = self._facets(where)
        if len(facets) == 0:
            raise ValueError("spring: no facets selected")
        kk = np.asarray(k, float)
        kk = np.diag(kk) if kk.ndim == 1 else kk
        if not np.allclose(kk, kk.T):
            raise ValueError("spring: k must be symmetric")
        d = self.dim
        nz = [(i, j) for i in range(d) for j in range(d) if kk[i, j] != 0]

        @BilinearForm
        def a(u, v, w):
            return sum(kk[i, j] * u[j] * v[i] for i, j in nz)
        name = name or f"spring{len(self.springs)}"
        s = Support(name, facets, asm(a, self._fbasis(facets)) * self.thick)
        self._add_matrix(name, s.K)
        self.springs[name] = s
        return s

    def spring_point(self, p, k, name: str | None = None):
        """a discrete spring to the ground at the vertex nearest to p (k per global axis, force on the solid = −k u)"""
        n = int(np.argmin(np.linalg.norm(self.sk.p.T - np.asarray(p, float), axis=1)))
        dofs = self.ib.nodal_dofs[:, n]
        K = sp.csr_matrix((np.asarray(k, float), (dofs, dofs)), shape=(self.ib.N, self.ib.N))
        name = name or f"spring{len(self.springs)}"
        s = Support(name, np.array([n]), K)
        self._add_matrix(name, K)
        self.springs[name] = s
        return s

    def bar(self, points, EA, bonded: bool = True, ds: float | None = None, name: str | None = None):
        """an axial bar embedded in the solid (femotion.continuum.embed.Bar): rebar, a grouted tendon, a bonded plate
        (bonded=True), or an external tendon / a plate attached only at its anchors (bonded=False)"""
        from .embed import Bar
        name = name or f"bar{len(self.bars)}"
        b = Bar(self, points, EA, bonded=bonded, ds=ds, name=name)
        self._add_matrix(name, b.K)
        self.bars[name] = b
        return b

    # ── solve ───────────────────────────────────────────────────────────
    def solve(self, method: str = "auto", tol: float = 1e-10, verbose: bool = False, cases=None):
        """solve every load case (or the named ones) with one factorisation. method: direct (SuperLU), cholesky (CHOLMOD
        through scikit-sparse or, failing that, cholespy: far faster and leaner for 3D solids, needs a positive definite
        K, i.e. enough supports), amg (pyamg smoothed aggregation + CG with rigid-body modes as near-null space, one hierarchy for
        all cases; slow to converge on thin-walled quadratic models), auto (direct up to 150,000 dofs, then cholesky up to
        800,000 if installed, else amg). Returns the displacements of the case "default" (or of the first case)"""
        from .results import CaseResult
        t0 = time.time()
        if not self.cases:
            self.case()
        names = list(self.cases) if cases is None else list(cases)
        K = self.K
        D = np.unique(np.concatenate(self._fixed)) if self._fixed else np.zeros(0, int)
        free = np.setdiff1d(np.arange(self.ib.N), D)
        uD0 = np.zeros(self.ib.N)
        A = K[free][:, free].tocsr()
        any_presc = any(len(self.cases[n].uD) for n in names)
        KfD = K[free][:, D] if any_presc else None
        if method == "auto":
            method = "direct" if len(free) <= 150_000 else ("cholesky" if _has_cholespy() and len(free) <= 800_000 else "amg")
        self.cg_iterations = 0
        self.method = method
        rels = []
        if method == "cholesky":  # CHOLMOD: robust for thin-walled solids
            solve1 = _cholmod(A)
        elif method == "direct":
            from scipy.sparse.linalg import splu
            try:
                lu = splu(A.tocsc())
            except RuntimeError as e:  # exactly singular
                raise RuntimeError(f"direct solve did not converge ({e}); check the supports") from e
            solve1 = lu.solve
        else:
            import pyamg
            B = self._rigid_modes()[free]
            d = self.dim
            Ab = A
            fb = free.reshape(-1, d) if len(free) % d == 0 else None
            if fb is not None and np.all(fb[:, 0] % d == 0) and np.all(fb == fb[:, :1] + np.arange(d)):
                Ab = A.tobsr(blocksize=(d, d))  # whole nodes free: aggregate node blocks (far fewer CG steps for quadratic cells)
            # prolongation smoothing weighted row by row (Gershgorin bound): pyamg's default estimates the spectral radius from
            # a random vector, and about 1 draw in 15 underestimates it for quadratic solids, so that CG diverges
            ml = pyamg.smoothed_aggregation_solver(Ab, B=B, symmetry="hermitian", max_coarse=500,
                                                   smooth=("jacobi", {"omega": 4.0 / 3.0, "weighting": "local"}))

            def solve1(b):
                resid: list[float] = []
                x = ml.solve(b, tol=tol, accel="cg", maxiter=2000, residuals=resid)
                self.cg_iterations = max(self.cg_iterations, len(resid) - 1)
                return x
        for name in names:
            f = self.cases[name].f
            uD = uD0.copy()
            for k, v in self.cases[name].uD.items():
                uD[k] = v
            b = f[free] - (KfD @ uD[D] if KfD is not None else 0)
            u = uD.copy()
            nb = np.linalg.norm(b)
            if nb > 0:
                u[free] = solve1(b)
            rel = float(np.linalg.norm(b - A @ u[free]) / nb) if nb > 0 else 0.0
            if not rel < 1e-6 or not np.all(np.isfinite(u)):
                hint = {"amg": "try method='cholesky' or 'direct', check the supports",
                        "cholesky": "check the supports (K must be positive definite)"}.get(method, "check the supports")
                raise RuntimeError(f"{method} solve did not converge for case {name!r} (relative residual {rel:.1e}); {hint}")
            rels.append(rel)
            self.results[name] = CaseResult(self, u, [(1.0, self.cases[name])], name=name)
        self.rel_residual = max(rels)
        self.free, self.fixed = free, D
        self.u = self._main().u if (len(self.results) == 1 or self._has_default()) else None
        self.t_solve = time.time() - t0
        if verbose:
            print(f"  assembly {self.t_assemble:.1f}s, solve {self.t_solve:.1f}s ({method}), {self.ib.N:,} dofs, "
                  f"{len(names)} case(s), relative residual ≤ {self.rel_residual:.1e}")
        return self.u

    def dofs_where(self, where, comps=None) -> dict:
        """{component: dof ids} located where the predicate holds (dof locations, dim × N), or of given vertex ids"""
        comps = range(self.dim) if comps is None else comps
        if isinstance(where, np.ndarray):
            dofs = self.ib.get_dofs(nodes=where)
            return {c: dofs.all(f"u^{c + 1}") for c in comps}
        hit = np.asarray(where(self.ib.doflocs), bool)
        comp = np.arange(self.ib.N) % self.dim
        return {c: np.where(hit & (comp == c))[0] for c in comps}

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
    def result(self, name: str = "default"):
        """the solution of a load case (femotion.continuum.results.CaseResult; + and scalar * combine them)"""
        if name not in self.results:
            raise KeyError(f"case {name!r} has not been solved (have {sorted(self.results)})")
        return self.results[name]

    def combine(self, factors: dict, name: str | None = None):
        """Σ factor · result(case), e.g. combine({"自重": 1.2, "车辆": 1.4})"""
        if not factors:
            raise ValueError("combine: no cases")
        out = None
        for k, v in factors.items():
            r = v * self.result(k)
            out = r if out is None else out + r
        out.name = name
        return out

    def _has_default(self) -> bool:
        return "default" in self.results and not self.cases["default"].empty

    def _main(self):
        """the result behind the single-case API: the case "default", or the only case"""
        if not self.results:
            raise RuntimeError("solve() first")
        if self._has_default():
            return self.results["default"]
        if len(self.results) == 1:
            return next(iter(self.results.values()))
        raise ValueError(f"several load cases ({', '.join(self.results)}): use result(name)")

    def reactions(self) -> np.ndarray:
        """resultant reaction at the fixed dofs (per component)"""
        return self._main().reactions()

    def compliance(self) -> float:
        return self._main().compliance()

    def result_mesh(self) -> Mesh:
        """the mesh whose nodes match the result values one-to-one: P1 is the input mesh; 2D P2 is tri6 (with mid-edge nodes);
        3D P2 keeps corner cells only (cell groups of the input mesh kept)"""
        if self.order == 1 or self.dim == 3:
            m = Mesh.from_skfem(self.sk)  # 3D quadratic: corners only
            if self.mesh.cell_groups and self.mesh.n_points == m.n_points:
                m.cell_groups = self.mesh.cell_groups
            return m
        sb = self.sb
        if self.mesh.kind in ("tri3", "tri6"):
            p = sb.doflocs.T.copy()
            return Mesh(p, {"tri6": orient(p, "tri6", sb.element_dofs.T.copy())})
        return Mesh(self.sk.p.T.copy(), {"quad4": self.sk.t.T.copy()})  # Quad2: corners only

    def node_basis(self, cells=None):
        """a vector Basis evaluated at all nodes of the given cells (including mid-edge nodes)"""
        from skfem import Basis
        ref = self.scalar_elem.doflocs.T
        return Basis(self.sk, self.elem, quadrature=(ref, np.ones(ref.shape[1])), elements=cells)

    def n_result_nodes(self) -> int:
        if self.order == 1 or self.dim == 3:
            return self.sk.p.shape[1]
        return self.result_mesh().n_points

    def nodal_displacement(self) -> np.ndarray:
        """(N, dim) on the nodes of result_mesh()"""
        return self._main().nodal_displacement()

    def stress_at_nodes_per_cell(self) -> np.ndarray:
        """stress of every cell at each of its nodes: (ncomp, nel, nloc)"""
        return self._main().stress_at_nodes_per_cell()

    def cell_stress(self) -> np.ndarray:
        """cell-average stress (nel, ncomp)"""
        return self._main().cell_stress()

    def nodal_stress(self) -> np.ndarray:
        """cell stresses at the nodes averaged with area / volume weights: (N, ncomp) on the nodes of result_mesh()"""
        return self._main().nodal_stress()

    def strain_energy_density(self) -> np.ndarray:
        """strain energy density per cell (cell average)"""
        return self._main().strain_energy_density()


def _has_cholespy() -> bool:
    import importlib.util
    return any(importlib.util.find_spec(m) is not None for m in ("sksparse", "cholespy"))


def _cholmod(A):
    """factorise an SPD matrix with CHOLMOD and return x = solve(b): scikit-sparse (supernodal, multithreaded BLAS; needs
    SuiteSparse installed) if present, else cholespy (pip wheels; simplicial, an order of magnitude slower on 3D solids)"""
    try:
        import sksparse.cholmod as C
    except ImportError:
        C = None
    if C is not None:
        try:
            F = C.cho_factor(sp.csc_array(A.tocsc()), order="default")
        except Exception as e:
            raise RuntimeError(f"Cholesky factorisation failed ({e}); check the supports") from e
        return lambda b: np.asarray(F.solve(b)).reshape(b.shape)
    from cholespy import CholeskySolverD, MatrixType
    C = A.tocoo()
    try:
        ch = CholeskySolverD(A.shape[0], C.row.astype(np.int32), C.col.astype(np.int32), C.data, MatrixType.COO)
    except Exception as e:
        raise RuntimeError(f"Cholesky factorisation failed ({e}); check the supports") from e

    def solve1(b):
        x = np.zeros_like(b)
        ch.solve(np.ascontiguousarray(b), x)
        return x
    return solve1


def facet_normals(P: np.ndarray) -> np.ndarray:
    """unit normals (3, n) of 3D facets given by their vertices (3, nv, n) in any order (best-fit plane; sign arbitrary)"""
    c = P.mean(axis=1, keepdims=True)
    Q = (P - c).transpose(2, 1, 0)  # (n, nv, 3)
    C = np.einsum("nvi,nvj->nij", Q, Q)
    w, V = np.linalg.eigh(C)
    return V[:, :, 0].T


class Support:
    """an elastic support: name, facets (or the node of a point spring), stiffness matrix"""

    def __init__(self, name, where, K):
        self.name, self.where, self.K = name, where, K.tocsr()

    def __repr__(self):
        kind = "point" if self.K.nnz <= 3 else f"{len(self.where)} facet(s)"
        return f"Support({self.name!r}, {kind})"
