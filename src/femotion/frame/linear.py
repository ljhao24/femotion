"""Fast linear frame solver (direct stiffness method): 3D beams, axial-only bars (cables), three-way spring links.

The model format (Model, Section, local axes, mass rule) lives in femotion.frame.model and is shared with the OpenSees
back-end; this module only solves. Units: m, kN, t (mass), s; E in kPa, densities in t/m³, loads in kN.
- beam: 12-dof Euler-Bernoulli element, local axes from femotion.frame.model.local_axes.
- bar: axial stiffness only, optional initial strain eps0 (> 0 means pretension, applied as a pair of nodal forces).
- link: three-way springs between two nodes in global directions (bearings, hinges, dampers).
Design tool: pretension influence matrices (cable-force optimisation).

When to use it: linear problems solved many times (vehicles crossing frame by frame, influence lines, cable-force
optimisation, modes) once the model has passed femotion.frame.check.crosscheck against OpenSees; use OpenSees for
anything nonlinear. The assembled stiffness Solver.K is also what the display layer shows (equation clouds, assembly).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from .model import Model, Section, local_axes, lumped_masses  # noqa: F401 (Model, Section re-exported)

_frame = local_axes  # old name


def beam_k(sec: Section, L: float) -> np.ndarray:
    """12×12 stiffness in local coordinates. Local z is the reference direction (vertical): vertical bending (x-z plane) uses Iy,
    lateral bending (x-y plane) uses Iz."""
    E, A, G, J = sec.E, sec.A, sec.g(), sec.J
    Iy, Iz = sec.Iy, sec.Iz
    k = np.zeros((12, 12))
    a = E * A / L
    k[0, 0] = k[6, 6] = a
    k[0, 6] = k[6, 0] = -a
    t = G * J / L
    k[3, 3] = k[9, 9] = t
    k[3, 9] = k[9, 3] = -t
    # bending in the x-y plane (v, θz), second moment Iz
    c = E * Iz / L**3
    for (p, q, v) in [(1, 1, 12), (1, 5, 6 * L), (1, 7, -12), (1, 11, 6 * L), (5, 5, 4 * L * L), (5, 7, -6 * L), (5, 11, 2 * L * L),
                      (7, 7, 12), (7, 11, -6 * L), (11, 11, 4 * L * L)]:
        k[p, q] = k[q, p] = c * v
    # bending in the x-z plane (w, θy), second moment Iy; signs opposite to v-θz
    c = E * Iy / L**3
    for (p, q, v) in [(2, 2, 12), (2, 4, -6 * L), (2, 8, -12), (2, 10, -6 * L), (4, 4, 4 * L * L), (4, 8, 6 * L), (4, 10, 2 * L * L),
                      (8, 8, 12), (8, 10, 6 * L), (10, 10, 4 * L * L)]:
        k[p, q] = k[q, p] = c * v
    return k


def _T12(R):
    T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return T


class Solver:
    """assemble once, solve many times (different loads, different initial strains). drop_truss: bars removed from the
    stiffness (broken cables)."""

    def __init__(self, m: Model, drop_truss: set[int] | None = None, rot_mass_floor: float = 1e-3):
        self.m = m
        n = m.ndof
        rows, cols, vals = [], [], []
        self._beam_cache = []
        for (i, j, sec, ref, tag) in m.beams:
            L, R = local_axes(m.X[i], m.X[j], ref)
            T = _T12(R)
            kg = T.T @ beam_k(sec, L) @ T
            dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(kg.ravel())
            self._beam_cache.append((L, R, T))
        drop = drop_truss or set()
        self._truss_cache = []
        for e, (i, j, sec, tag) in enumerate(m.trusses):
            d = m.X[j] - m.X[i]
            L = np.linalg.norm(d)
            u = d / L
            self._truss_cache.append((L, u))
            if e in drop:
                continue
            k3 = sec.E * sec.A / L * np.outer(u, u)
            blk = np.block([[k3, -k3], [-k3, k3]])
            dofs = np.r_[6 * i:6 * i + 3, 6 * j:6 * j + 3]
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(blk.ravel())
        for (i, j, k, kr, tag) in m.links:
            for a in range(3):
                for (p, q, s) in [(i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)]:
                    rows.append(np.array([6 * p + a])); cols.append(np.array([6 * q + a])); vals.append(np.array([s * k[a]]))
            if kr is not None:
                for a in range(3):
                    for (p, q, s) in [(i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)]:
                        rows.append(np.array([6 * p + 3 + a])); cols.append(np.array([6 * q + 3 + a])); vals.append(np.array([s * kr[a]]))
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
        fixed = np.zeros(n, bool)
        for nd, f in m.fixed.items():
            fixed[6 * nd:6 * nd + 6] = f
        # nodes connected only to bars have no rotational stiffness: restrain those rotations
        diag = K.diagonal()
        for nd in range(len(m.X)):
            for a in range(3, 6):
                if abs(diag[6 * nd + a]) < 1e-9:
                    fixed[6 * nd + a] = True
        self.free = np.where(~fixed)[0]
        self.K = K
        self.Kff = K[self.free][:, self.free].tocsc()
        self._lu = spla.splu(self.Kff)
        # lumped masses (rule in femotion.frame.model.lumped_masses, shared with the OpenSees back-end)
        self.M = lumped_masses(m, drop, rot_mass_floor)
        self.drop = drop

    # ── loads ───────────────────────────────────────────────────────────
    def gravity(self, g=9.81) -> np.ndarray:
        """self-weight (including added mass) as lumped mass × g, vertically down (kN)"""
        F = np.zeros(self.m.ndof)
        tm = self.M.reshape(-1, 6)
        F[2::6] = -tm[:, 2] * g
        return F

    def pretension(self, eps0: np.ndarray) -> np.ndarray:
        """initial strain of the bars eps0 (> 0 pretension) → equivalent nodal forces"""
        F = np.zeros(self.m.ndof)
        for e, (i, j, sec, tag) in enumerate(self.m.trusses):
            if e in self.drop or eps0[e] == 0:
                continue
            L, u = self._truss_cache[e]
            N0 = sec.E * sec.A * eps0[e]
            F[6 * i:6 * i + 3] += N0 * u
            F[6 * j:6 * j + 3] -= N0 * u
        return F

    def solve(self, F: np.ndarray) -> np.ndarray:
        U = np.zeros(self.m.ndof)
        U[self.free] = self._lu.solve(F[self.free])
        return U

    # ── internal forces ─────────────────────────────────────────────────
    def truss_forces(self, U: np.ndarray, eps0: np.ndarray | None = None) -> np.ndarray:
        N = np.zeros(len(self.m.trusses))
        for e, (i, j, sec, tag) in enumerate(self.m.trusses):
            if e in self.drop:
                continue
            L, u = self._truss_cache[e]
            dl = np.dot(U[6 * j:6 * j + 3] - U[6 * i:6 * i + 3], u)
            N[e] = sec.E * sec.A * (dl / L + (eps0[e] if eps0 is not None else 0.0))
        return N

    def beam_end_forces(self, U: np.ndarray) -> np.ndarray:
        """local end forces of every beam (12 components, element on node; no span loads)"""
        out = np.zeros((len(self.m.beams), 12))
        for e, (i, j, sec, ref, tag) in enumerate(self.m.beams):
            L, R, T = self._beam_cache[e]
            ue = np.r_[U[6 * i:6 * i + 6], U[6 * j:6 * j + 6]]
            out[e] = beam_k(sec, L) @ (T @ ue)
        return out

    # ── modes ───────────────────────────────────────────────────────────
    def modes(self, n=12):
        Mf = self.M[self.free]
        if len(self.free) <= n + 2:
            # tiny model: diagonal mass, reduce to a standard eigenproblem and solve densely
            s = 1 / np.sqrt(Mf)
            A = (self.Kff.toarray() * s[:, None]) * s[None, :]
            w2, V = sla.eigh(A, subset_by_index=[0, n - 1])
            V = V * s[:, None]
        else:
            # shift-invert (σ = 0, reusing the stiffness LU). A dense eigh is ill-conditioned with very stiff rigid arms and tiny
            # rotational masses (low frequencies off by ~3e-4, found by the OpenSees cross-check); this agrees with OpenSees to ~1e-9
            op = spla.LinearOperator(self.Kff.shape, matvec=self._lu.solve, dtype=float)
            w2, V = spla.eigsh(self.Kff, k=n, M=sp.diags(Mf).tocsc(), sigma=0, which="LM", OPinv=op, v0=np.ones(len(Mf)))  # fixed start vector: reproducible
            o = np.argsort(w2)
            w2, V = w2[o], V[:, o]
        Phi = np.zeros((self.m.ndof, n))
        Phi[self.free] = V
        # mass-normalised: Φᵀ M Φ = I
        Phi /= np.sqrt(np.einsum("ik,i,ik->k", Phi, self.M, Phi))[None, :]
        return np.sqrt(np.maximum(w2, 0)) / (2 * np.pi), Phi


def pretension_influence(sv: Solver, groups: list[list[int]], rows=None, unit: float = 1e-3):
    """response to a unit initial strain `unit` in each group of bars: returns (A_u, A_N),
    A_u[:, g] = displacements of the dofs `rows` (all dofs when rows is None), A_N[:, g] = axial forces of all bars.
    Used for cable-force optimisation: target displacements / forces → least-squares initial strains per group."""
    nt = len(sv.m.trusses)
    Au, AN = [], []
    for g in groups:
        eps = np.zeros(nt)
        eps[list(g)] = unit
        Ug = sv.solve(sv.pretension(eps))
        Au.append(Ug if rows is None else Ug[rows])
        AN.append(sv.truss_forces(Ug, eps))
    return np.array(Au).T, np.array(AN).T
