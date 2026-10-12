"""Results of continuum load cases: displacements, stresses (with initial strains taken off), support and bar forces,
values at arbitrary points, and the force resultants on a cut through the solid.

A CaseResult is the solution u of a load case plus the cases it came from, so that stresses subtract the right initial
strains (temperature, shrinkage) and bar forces add the right prestress. Results combine linearly:

    r = 1.0 * s.result("自重") + s.result("预应力") + 0.8 * s.result("温度")
    r.nodal_stress()                              # (N, ncomp) on s.result_mesh()
    r.sample(points)                              # {"u": (n, dim), "stress": (n, ncomp)} at arbitrary points
    r.section((8.25, 0, 0), (1, 0, 0))            # {"F", "M", "area", "centroid"} on the plane x = 8.25
    r.spring_force("B0"); r.bar_force("plate3")
    export_vtu("out/deck.vtu", s, {"自重": g, "合计": r})

section() gives the force F the part on the positive side of a plane exerts on the part on the negative side (N = F·n,
tension positive) and its moment M about `ref` (default: the centroid of the cut) as a vector, right-handed. For a member
along +x with z up, a sagging moment (bottom fibres in tension) is M_y < 0 and a hogging one M_y > 0: the opposite sign
of femotion.frame.post.chain_force (sagging +). The plane must coincide with cell faces (a node layer of a structured or
swept mesh); a plane cutting through cells raises.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .elastic import _arr, _voigt
from .embed import inverse_map


class CaseResult:
    def __init__(self, solver, u: np.ndarray, terms, name: str | None = None):
        self.s = solver
        self.u = u
        self.terms = list(terms)  # (factor, LoadCase)
        self.name = name

    # ── linear combination ──────────────────────────────────────────────
    def __add__(self, other: "CaseResult") -> "CaseResult":
        if other is None or (isinstance(other, (int, float)) and other == 0):
            return self
        if other.s is not self.s:
            raise ValueError("results of different models do not combine")
        return CaseResult(self.s, self.u + other.u, self.terms + other.terms)

    __radd__ = __add__

    def __mul__(self, k: float) -> "CaseResult":
        return CaseResult(self.s, k * self.u, [(k * a, c) for a, c in self.terms])

    __rmul__ = __mul__

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-1.0) * other

    def __repr__(self):
        return f"CaseResult({self.name or ' + '.join(f'{a:g}·{c.name}' for a, c in self.terms)})"

    @property
    def f(self) -> np.ndarray:
        """the load vector of this (combined) result"""
        return sum(a * c.f for a, c in self.terms)

    @property
    def f_ext(self) -> np.ndarray:
        """its external loads (no initial-strain or bar-prestress loads)"""
        return sum(a * c.f_ext for a, c in self.terms)

    @property
    def f_int(self) -> np.ndarray:
        return sum(a * c.f_int for a, c in self.terms)

    # ── initial strains ─────────────────────────────────────────────────
    def _eps0(self, x: np.ndarray, cells: np.ndarray, mat) -> np.ndarray | None:
        """Σ factor · initial strain of the cases at points x (dim, len(cells), …) of the given cells (one material)"""
        out = None
        ncomp = 3 if self.s.dim == 2 else 6
        for a, case in self.terms:
            for src_cells, fn in case.strains:
                inside = np.isin(cells, src_cells)
                if not inside.any():
                    continue
                e = np.broadcast_to(fn(x, mat), (ncomp,) + x.shape[1:]) * a
                e = e * inside.reshape((1, -1) + (1,) * (x.ndim - 2))
                out = e if out is None else out + e
        return out

    def _sigma(self, grad: np.ndarray, x: np.ndarray, cells: np.ndarray, mat) -> np.ndarray:
        """stress (ncomp, …) from displacement gradients (dim, dim, …) at points x of cells of one material"""
        e = 0.5 * (grad + np.swapaxes(grad, 0, 1))
        eps = np.stack(_voigt(e, self.s.dim))
        e0 = self._eps0(x, cells, mat)
        if e0 is not None:
            eps = eps - e0
        S = np.einsum("ij,j...->i...", self.s.D(mat), eps)
        if self.s._scale is not None:
            S = S * self.s._scale[cells].reshape((1, -1) + (1,) * (S.ndim - 2))
        return S

    # ── nodal and cell values ───────────────────────────────────────────
    def nodal_displacement(self) -> np.ndarray:
        """(N, dim) on the nodes of Elastic.result_mesh()"""
        return self.u.reshape(-1, self.s.dim)[: self.s.n_result_nodes()]

    def stress_at_nodes_per_cell(self) -> np.ndarray:
        """stress of every cell at each of its nodes: (ncomp, nel, nloc)"""
        s = self.s
        ncomp = 3 if s.dim == 2 else 6
        nloc = s.scalar_elem.doflocs.shape[0]
        S = np.zeros((ncomp, s.sk.t.shape[1], nloc))
        for mat, c in s._material_chunks():
            nb = s.node_basis(c)
            G = _arr(nb.interpolate(self.u).grad)  # (dim, dim, nel, nloc)
            x = _arr(nb.global_coordinates())
            S[:, c, :] = self._sigma(G, x, c, mat)
        return S

    def cell_stress(self) -> np.ndarray:
        """cell-average stress (nel, ncomp)"""
        return self.stress_at_nodes_per_cell().mean(axis=2).T

    def nodal_stress(self) -> np.ndarray:
        """cell stresses at the nodes averaged with area / volume weights: (N, ncomp) on the nodes of result_mesh()"""
        s = self.s
        S = self.stress_at_nodes_per_cell()
        ed = s.sb.element_dofs  # (nloc, nel)
        vol = s.cell_volumes
        w = np.broadcast_to(vol[None, :], ed.shape)
        acc = np.zeros((S.shape[0], s.sb.N))
        wsum = np.bincount(ed.ravel(), weights=w.ravel(), minlength=s.sb.N)
        for c in range(S.shape[0]):
            acc[c] = np.bincount(ed.ravel(), weights=(S[c].T * w).ravel(), minlength=s.sb.N)
        return (acc / np.maximum(wsum, 1e-300)).T[: s.n_result_nodes()]

    def strain_energy_density(self) -> np.ndarray:
        """strain energy density per cell, ½ (ε − ε0)·σ averaged over the cell"""
        from skfem import Basis
        s = self.s
        out = np.zeros(s.sk.t.shape[1])
        for mat, c in s._material_chunks():
            b = Basis(s.sk, s.elem, elements=c)
            G = _arr(b.interpolate(self.u).grad)
            x = _arr(b.global_coordinates())
            S = self._sigma(G, x, c, mat)
            e = 0.5 * (G + np.swapaxes(G, 0, 1))
            eps = np.stack(_voigt(e, s.dim))
            e0 = self._eps0(x, c, mat)
            if e0 is not None:
                eps = eps - e0
            dx = _arr(b.dx)
            out[c] = 0.5 * np.einsum("ink,ink,nk->n", S, eps, dx) / dx.sum(axis=1)
        return out

    def compliance(self) -> float:
        """work of the external loads, f_ext·u (twice the strain energy when there are no initial strains)"""
        return float(self.f_ext @ self.u)

    # ── supports and bars ───────────────────────────────────────────────
    def reactions(self) -> np.ndarray:
        """resultant reaction at the fixed dofs (per component; elastic supports: spring_force)"""
        s = self.s
        r = s.K @ self.u - self.f
        out = np.zeros(s.dim)  # (the internal parts of f vanish against K u inside the solid; at supports they are reactions)
        for i in range(s.dim):
            d = s.fixed[s.fixed % s.dim == i]
            out[i] = r[d].sum()
        return out

    def spring_force(self, name: str) -> np.ndarray:
        """resultant force (per global component) an elastic support exerts on the solid"""
        g = -(self.s.springs[name].K @ self.u)
        return np.array([g[i::self.s.dim].sum() for i in range(self.s.dim)])

    def bar_force(self, name: str) -> np.ndarray:
        """axial force of each piece of an embedded bar (tension +), including the prestress of the cases"""
        P0 = sum(a * c.bar_P[name] for a, c in self.terms if name in c.bar_P)
        return self.s.bars[name].forces(self.u, P0 if not isinstance(P0, int) else 0.0)

    # ── values at arbitrary points ──────────────────────────────────────
    def sample(self, points, located=None) -> dict:
        """displacement and stress at arbitrary points (the field of the cell each point lies in, not nodal averages)"""
        s = self.s
        P = np.atleast_2d(np.asarray(points, float))
        loc = located if located is not None else s.locator.locate(P)
        cells, _ = loc
        Nu = s.locator.matrix(P, loc) @ self.u
        G = s.locator.gradients(self.u, loc)  # (dim, dim, n)
        ncomp = 3 if s.dim == 2 else 6
        S = np.zeros((ncomp, len(P)))
        for i, mat in enumerate(s.mats):
            k = np.where(s.cell_material[cells] == i)[0]
            if len(k):
                S[:, k] = self._sigma(G[:, :, k], P.T[:, k], cells[k], mat)
        return {"u": Nu.reshape(-1, s.dim), "stress": S.T}

    # ── section resultants ──────────────────────────────────────────────
    def section(self, origin, normal, ref=None, method: str = "nodal", region=None, tol: float = 1e-6,
                gauss: int = 3) -> dict:
        """force and moment resultants on the plane through `origin` with unit normal `normal` (see the module doc for
        signs). method="nodal" (default): statics of the positive part — the external loads, reactions and spring forces
        on its nodes, plus the share of the loads of the cells next to the plane that falls on the nodes in it; exact
        equilibrium, embedded bars crossing the plane included (their force adds to the solid's), initial strains and bar
        prestress taken as internal. Loads given as nodal forces (point_load) and supports on the plane itself count to
        the negative side. method="traction": ∫ σ·n dA over the cell faces in the plane (stress averaged from both sides,
        O(h²) accurate) plus the forces of embedded bars crossing it; region(x) → bool restricts the integral to part of
        the cut (the bottom slab, one web). tol: distance from the plane, relative to the median edge length"""
        s = self.s
        o = np.asarray(origin, float)
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        atol = tol * s.h
        pts, wts, cut = self._cut(o, n, atol, gauss)
        area = wts.sum() * s.thick
        cen = (pts * wts).sum(axis=1) / wts.sum()
        r0 = cen if ref is None else np.asarray(ref, float)
        if method == "nodal":
            if region is not None:
                raise ValueError("region needs method='traction'")
            K = s.K
            for sp_ in s.springs.values():
                K = K - sp_.K
            e = (K @ self.u - self.f_int).reshape(-1, s.dim)  # external loads + reactions + spring forces at every node
            X = s.ib.doflocs[:, ::s.dim].T
            d = (X - o) @ n
            pos, on = d > atol, np.abs(d) <= atol
            # cells on the positive side touching the plane: their loads' share on the nodes in the plane
            sk = s.sk
            vd = (sk.p.T - o) @ n
            cd = vd[sk.t]  # (nv, nel)
            adj = np.where((cd.max(axis=0) > atol) & np.any(np.abs(cd) <= atol, axis=0))[0]
            share = sum(a * c.restricted(adj) for a, c in self.terms).reshape(-1, s.dim)
            F = e[pos].sum(axis=0) + share[on].sum(axis=0)
            M = _moment(X[pos] - r0, e[pos], s.dim) + _moment(X[on] - r0, share[on], s.dim)
        elif method == "traction":
            keep = np.ones(len(wts), bool) if region is None else np.asarray(region(pts), bool)
            T = np.zeros((pts.shape[1], s.dim))
            for side_cells in cut:  # stress from each side, averaged
                X = inverse_map(s.sb.mapping, pts[:, :, None], side_cells)[:, :, 0]
                St = self.sample(pts.T, located=(side_cells, np.clip(X, 0.0, 1.0)))["stress"]
                T += _traction(St, n, s.dim) / 2
            w = wts * keep
            F = (T * w[:, None]).sum(axis=0) * s.thick
            M = _moment(pts.T - r0, T * w[:, None], s.dim) * s.thick
            for name, bar in s.bars.items():  # bars crossing the plane: bonded pieces, or the runs of an unbonded bar
                N = self.bar_force(name)
                d = (bar.points - o) @ n
                for k in np.where((d[:-1] < 0) != (d[1:] < 0))[0]:
                    a, b = bar.points[k], bar.points[k + 1]
                    xc = a + (b - a) * d[k] / (d[k] - d[k + 1])
                    if region is not None and not region(xc[:, None])[0]:
                        continue
                    t = (b - a) / np.linalg.norm(b - a)
                    Fb = (N[k] if bar.bonded else N[0]) * t * np.sign(t @ n)
                    F = F + Fb
                    M = M + _moment((xc - r0)[None], Fb[None], s.dim)
        else:
            raise ValueError("method: nodal or traction")
        return {"F": F, "M": M, "N": float(F @ n), "area": float(area), "centroid": cen}

    def _cut(self, o, n, atol, gauss):
        """Gauss points (dim, nq) and weights of the interior cell faces lying on a plane, with the cell on each side of
        each point; raises if the plane cuts through cells"""
        s = self.s
        sk = s.sk
        vd = (sk.p.T - o) @ n
        cd = vd[sk.t]
        if np.any((cd.min(axis=0) < -atol) & (cd.max(axis=0) > atol)):
            raise ValueError("section: the plane cuts through cells; cut at a node layer (a plane of cell faces)")
        fd = vd[sk.facets]
        fids = np.where(np.all(np.abs(fd) <= atol, axis=0) & (sk.f2t[1] >= 0))[0]  # faces with cells on both sides
        if len(fids) == 0:
            raise ValueError("section: no interior cell faces lie on this plane")
        cc = sk.p[:, sk.t].mean(axis=1)
        t2 = sk.f2t[:, fids]
        first_neg = (n @ (cc[:, t2[0]] - o[:, None])) < 0
        neg = np.where(first_neg, t2[0], t2[1])
        posc = np.where(first_neg, t2[1], t2[0])
        pts, wts, own = _facet_quadrature(sk.p[:, sk.facets[:, fids]], n, gauss)
        return pts, wts, (neg[own], posc[own])


def _moment(arm: np.ndarray, F: np.ndarray, dim: int) -> np.ndarray:
    if dim == 3:
        return np.cross(arm, F).sum(axis=0)
    return np.array([(arm[:, 0] * F[:, 1] - arm[:, 1] * F[:, 0]).sum()])


def _traction(S: np.ndarray, n: np.ndarray, dim: int) -> np.ndarray:
    if dim == 2:
        sxx, syy, sxy = S.T
        return np.stack([sxx * n[0] + sxy * n[1], sxy * n[0] + syy * n[1]], axis=1)
    xx, yy, zz, xy, yz, zx = S.T
    return np.stack([xx * n[0] + xy * n[1] + zx * n[2], xy * n[0] + yy * n[1] + yz * n[2], zx * n[0] + yz * n[1] + zz * n[2]], axis=1)


def _facet_quadrature(P: np.ndarray, n: np.ndarray, gauss: int):
    """Gauss points (dim, nq), weights (nq,) and owning facet index of planar facets P (dim, nv, nf) on a plane with normal n"""
    dim, nv, nf = P.shape
    xi, wi = np.polynomial.legendre.leggauss(gauss)
    if dim == 2:
        a, b = P[:, 0], P[:, 1]
        L = np.linalg.norm(b - a, axis=0)
        pts = np.concatenate([a + 0.5 * (1 + x) * (b - a) for x in xi], axis=1)
        w = np.concatenate([0.5 * wk * L for wk in wi])
        own = np.tile(np.arange(nf), len(xi))
        return pts, w, own
    # in-plane axes
    e1 = np.cross(n, (1.0, 0, 0)) if abs(n[0]) < 0.9 else np.cross(n, (0, 1.0, 0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    c = P.mean(axis=1)
    rel = P - c[:, None, :]
    ang = np.arctan2(np.einsum("i,ivf->vf", e2, rel), np.einsum("i,ivf->vf", e1, rel))
    order = np.argsort(ang, axis=0)
    Q = np.take_along_axis(P, order[None], axis=1)  # vertices counter-clockwise about n
    pts, w, own = [], [], []
    if nv == 4:
        for x, wx in zip(xi, wi):
            for y, wy in zip(xi, wi):
                N = 0.25 * np.array([(1 - x) * (1 - y), (1 + x) * (1 - y), (1 + x) * (1 + y), (1 - x) * (1 + y)])
                dNx = 0.25 * np.array([-(1 - y), (1 - y), (1 + y), -(1 + y)])
                dNy = 0.25 * np.array([-(1 - x), -(1 + x), (1 + x), (1 - x)])
                p = np.einsum("v,ivf->if", N, Q)
                J = np.linalg.norm(np.cross(np.einsum("v,ivf->if", dNx, Q).T, np.einsum("v,ivf->if", dNy, Q).T), axis=1)
                pts.append(p)
                w.append(wx * wy * J)
                own.append(np.arange(nf))
    else:  # triangles: 3-point rule
        for (l1, l2, l3), wk in (((2 / 3, 1 / 6, 1 / 6), 1 / 3), ((1 / 6, 2 / 3, 1 / 6), 1 / 3), ((1 / 6, 1 / 6, 2 / 3), 1 / 3)):
            p = l1 * Q[:, 0] + l2 * Q[:, 1] + l3 * Q[:, 2]
            A = 0.5 * np.linalg.norm(np.cross((Q[:, 1] - Q[:, 0]).T, (Q[:, 2] - Q[:, 0]).T), axis=1)
            pts.append(p)
            w.append(wk * A)
            own.append(np.arange(nf))
    return np.concatenate(pts, axis=1), np.concatenate(w), np.concatenate(own)


def export_vtu(path, solver, results: dict, stress: bool = True) -> Path:
    """write the results of several cases into one VTU file (femotion.io, extra "io"): per case the displacement
    "u:<name>", the stress "S:<name>" (Voigt), the principal stresses "S123:<name>" and the direction of the largest
    one "dir1:<name>" on the nodes of solver.result_mesh(); cell groups become flag arrays"""
    from .. import io
    from ..post.stress import principal_2d, principal_3d
    m = solver.result_mesh()
    pd = {}
    for name, r in results.items():
        pd[f"u:{name}"] = r.nodal_displacement()
        if stress:
            S = r.nodal_stress()
            pd[f"S:{name}"] = S
            if solver.dim == 3:
                w, V = principal_3d(S)
                pd[f"S123:{name}"] = w
                pd[f"dir1:{name}"] = V[:, :, 0]
            else:
                s1, s2, th = principal_2d(S)
                pd[f"S12:{name}"] = np.c_[s1, s2]
    return io.write_mesh(path, m, point_data=pd)
