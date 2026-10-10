"""2D topology optimisation (SIMP + density filter + optimality-criteria update), a numpy/scipy port of the 88-line code of
Andreassen et al. (2011), recording every iteration for animation.

Grid: nelx × nely unit-square quadrilaterals, cells numbered by columns (e = ely + elx·nely, ely counted from the top);
nodes likewise by columns (n = (nely+1)·ix + iy, iy from the top). y points up (a load of −1 points down).
passive / void: cells always kept / always empty, bool arrays of shape (nelx, nely) ([:, :2] is the top two rows) or
flattened by cell id. The matching Mesh is femotion.mesh.structured.grid_for_density (same cell order as the density array).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def ke_q4(nu=0.3):
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    return 1 / (1 - nu**2) / 24 * (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]]))


def node(ix, iy, nely):
    return (nely + 1) * ix + iy


def edof_mat(nelx, nely):
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    tl = (nely + 1) * ex.ravel() + ey.ravel()
    bl = tl + 1
    br = bl + nely + 1
    tr = tl + nely + 1
    return np.stack([2 * bl, 2 * bl + 1, 2 * br, 2 * br + 1, 2 * tr, 2 * tr + 1, 2 * tl, 2 * tl + 1], axis=1)


def run(nelx, nely, F, fixed, volfrac=0.3, penal=3.0, rmin=2.5, passive=None, void=None, maxit=200, tol=0.01, move=0.2,
        Emin=1e-9, nu=0.3, record=True, verbose=True, symmetric=False, record_energy=False, x0=None):
    """record_energy: store the strain energy of every cell in every iteration (after the solve, before the density update),
    to colour "where the material works hard and where it idles". x0: initial densities (flattened or (nelx, nely)),
    volfrac everywhere by default."""
    n_el = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = ke_q4(nu)
    edof = edof_mat(nelx, nely)
    iK = np.kron(edof, np.ones((8, 1), int)).ravel()
    jK = np.kron(edof, np.ones((1, 8), int)).ravel()
    free = np.setdiff1d(np.arange(ndof), fixed)
    # density filter
    rows, cols, vals = [], [], []
    r = int(np.ceil(rmin)) - 1
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i2 in range(max(i1 - r, 0), min(i1 + r + 1, nelx)):
                for j2 in range(max(j1 - r, 0), min(j1 + r + 1, nely)):
                    e2 = i2 * nely + j2
                    w = max(0.0, rmin - np.hypot(i1 - i2, j1 - j2))
                    if w > 0:
                        rows.append(e1); cols.append(e2); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(n_el, n_el))
    Hs = np.asarray(H.sum(axis=1)).ravel()
    # cell id e = ely + elx·nely is exactly the row-major (C order) flattening of an (nelx, nely) array.
    # (Fixed 2026-10-09: an earlier F-order flattening froze the leftmost columns instead of the top rows.)
    pas = np.zeros(n_el, bool) if passive is None else np.asarray(passive, bool).ravel()
    vd = np.zeros(n_el, bool) if void is None else np.asarray(void, bool).ravel()
    x = np.full(n_el, volfrac) if x0 is None else np.asarray(x0, float).ravel().copy()
    x[pas] = 1.0
    x[vd] = 0.0
    xPhys = x.copy()
    hist = []
    comp_hist = []
    energy = []
    for it in range(maxit):
        sK = (KE.ravel()[None, :] * (Emin + xPhys[:, None] ** penal * (1 - Emin))).ravel()
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2
        U = np.zeros(ndof)
        U[free] = spla.spsolve(K[free][:, free], F[free])
        ue = U[edof]
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)
        c = float(np.sum((Emin + xPhys**penal * (1 - Emin)) * ce))
        if record_energy:
            energy.append(((Emin + xPhys**penal * (1 - Emin)) * ce).reshape(nelx, nely).astype(np.float32))
        dc = -penal * (1 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones(n_el)
        dc = np.asarray(H @ (dc / Hs))
        dv = np.asarray(H @ (dv / Hs))
        if symmetric:  # left-right symmetry: average sensitivities with their mirror image
            dc = 0.5 * (dc + dc.reshape(nelx, nely)[::-1].ravel())
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(np.maximum(-dc / dv / lmid, 0))))))
            xnew[pas] = 1.0
            xnew[vd] = 0.0
            xPhys_new = np.asarray(H @ xnew) / Hs
            xPhys_new[pas] = 1.0
            xPhys_new[vd] = 0.0
            if xPhys_new.mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = xnew
        xPhys = xPhys_new
        if record:
            hist.append(xPhys.copy())
        comp_hist.append(c)
        if verbose and (it % 10 == 0 or change < tol):
            print(f"  iteration {it:3d}  compliance {c:10.4f}  volume {xPhys.mean():.3f}  change {change:.3f}")
        if change < tol and it > 30:
            break
    return dict(x=xPhys.reshape(nelx, nely), hist=[h.reshape(nelx, nely) for h in hist], comp=comp_hist, U=U, energy=energy)
