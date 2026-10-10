"""Structural dynamics by modal superposition (works with any mass-normalised modes, e.g. femotion.frame.linear.Solver.modes).

- step_release: transient response to suddenly removing a set of forces (cable loss, blasting, unloading) — the exact
  step response plus a static correction.
- newmark_modal: Newmark average acceleration per mode for a ground acceleration (earthquakes), with sub-steps.
- resample: interpolate time histories to a video frame rate.
- load_accel: read a ground-motion record (two columns t, a or one column a; values in g are converted to m/s²).
- moving_point_loads: a train of point loads (vehicles) distributed linearly onto a chain of nodes, optionally with an
  eccentricity torque.
Conventions: modes Phi satisfy Φᵀ M Φ = I; damping is the same ratio zeta for every mode.
For nonlinear dynamics (yielding, collapse) use femotion.frame.opensees.OpsFrame.transient / release.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def step_release(U0: np.ndarray, dU_static: np.ndarray, Fr: np.ndarray, freqs: np.ndarray, Phi: np.ndarray,
                 t: np.ndarray, zeta: float = 0.02) -> np.ndarray:
    """Fr is applied suddenly at t = 0 (for unloading, Fr is the removed force reversed).
    dU_static: static displacement of the new structure under Fr; U0: the state before. Returns (len(t), ndof).
    U(t) = U0 + ΔU_static − Σ φ Γ/ω² h(t), h being the damped single-dof step-response remainder (1 at t = 0, decaying to 0)."""
    w = 2 * np.pi * freqs
    gam = Phi.T @ Fr
    wd = w * np.sqrt(1 - zeta**2)
    h = np.exp(-zeta * np.outer(t, w)) * (np.cos(np.outer(t, wd)) + zeta / np.sqrt(1 - zeta**2) * np.sin(np.outer(t, wd)))
    Q = -(gam / w**2)[None, :] * h
    return (U0 + dU_static)[None, :] + Q @ Phi.T


def newmark_modal(freqs: np.ndarray, gam: np.ndarray, ag: np.ndarray, dt: float, zeta: float = 0.02, sub: int = 4) -> np.ndarray:
    """q̈ + 2ζω q̇ + ω² q = −Γ a_g(t), average acceleration, each record step split into `sub` sub-steps (load linearly
    interpolated). gam = Φᵀ M r (r: influence vector of the input direction). Returns modal coordinates (len(ag), nmodes),
    relative to the ground."""
    w = 2 * np.pi * freqs
    h = dt / sub
    q = np.zeros(len(w))
    v = np.zeros(len(w))
    acc = -gam * ag[0]
    Q = [q.copy()]
    keff = w**2 + 2 * zeta * w * 2 / h + 4 / h**2
    for s in range(1, len(ag)):
        for kk in range(sub):
            al = (kk + 1) / sub
            p = -gam * (ag[s - 1] * (1 - al) + ag[s] * al)
            peff = p + (4 / h**2) * q + (4 / h) * v + acc + 2 * zeta * w * (2 / h * q + v)
            qn = peff / keff
            vn = 2 / h * (qn - q) - v
            accn = 4 / h**2 * (qn - q) - 4 / h * v - acc
            q, v, acc = qn, vn, accn
        Q.append(q.copy())
    return np.array(Q)


def resample(Q: np.ndarray, dt: float, fps: int, dur: float) -> np.ndarray:
    """(nt, ...) uniformly sampled history → (dur·fps, ...), column-wise linear interpolation"""
    tq = np.arange(len(Q)) * dt
    tf = np.arange(int(dur * fps)) / fps
    if Q.ndim == 1:
        return np.interp(tf, tq, Q)
    return np.array([np.interp(tf, tq, Q[:, i]) for i in range(Q.shape[1])]).T


def load_accel(path: str | Path) -> tuple[np.ndarray, float]:
    """returns (a [m/s²], dt)"""
    rec = np.loadtxt(path, comments=["#", "%"])
    if rec.ndim == 2 and rec.shape[1] >= 2:
        tt, ag = rec[:, 0], rec[:, 1]
        dt = float(np.median(np.diff(tt)))
    else:
        dt = 0.02
        ag = rec.ravel()
    if np.abs(ag).max() < 2:  # given in g
        ag = ag * 9.81
    return ag, dt


def moving_point_loads(ndof: int, chain_nodes, chain_x: np.ndarray, positions, P: float, dof: int = 2,
                       ecc: float = 0.0, ecc_dof: int = 3, dof_per_node: int = 6) -> np.ndarray:
    """place point loads P (towards −dof) along a chain of nodes (chain_x increasing); a load between two nodes is shared
    linearly; ecc ≠ 0 also adds the eccentricity torque P·ecc (about ecc_dof). Loads off the chain are ignored."""
    F = np.zeros(ndof)
    xs = chain_x
    for x in positions:
        if x < xs[0] or x > xs[-1]:
            continue
        k = min(max(np.searchsorted(xs, x) - 1, 0), len(xs) - 2)
        a = (x - xs[k]) / (xs[k + 1] - xs[k])
        for nn, wgt in ((chain_nodes[k], 1 - a), (chain_nodes[k + 1], a)):
            F[dof_per_node * nn + dof] -= P * wgt
            if ecc:
                F[dof_per_node * nn + ecc_dof] += P * wgt * ecc
    return F
