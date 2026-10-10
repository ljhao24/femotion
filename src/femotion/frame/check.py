"""Cross-check the fast linear solver against OpenSees on the same model.

Run it once for every new model before using femotion.frame.linear for production numbers: both solvers compute the
same linear model — displacements, bar forces and beam end forces under a load F (self-weight by default, optionally
with bar initial strains eps0), and the first n_modes frequencies. Any relative difference above `tol` raises: the model
contains something the linear solver does not handle the same way, so use OpenSees for it. Cables are compared as
tension-and-compression bars (tension-only behaviour is nonlinear and belongs to OpenSees).
"""
from __future__ import annotations

import time

import numpy as np

from .linear import Solver
from .model import Model
from .opensees import OpsFrame


def crosscheck(m: Model, F: np.ndarray | None = None, eps0: np.ndarray | None = None, n_modes: int = 6,
               tol: float = 1e-6, verbose: bool = True) -> dict:
    """returns {quantity: max relative difference}; raises AssertionError if any exceeds tol"""
    t = time.time()
    sv = Solver(m)
    if F is None:
        F = sv.gravity()
    U = sv.solve(F + (sv.pretension(eps0) if eps0 is not None else 0))
    f = OpsFrame(m, eps0=eps0)
    Uo = f.static(F)
    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))  # noqa: E731
    d = dict(displacement=rel(Uo, U))
    if m.trusses:
        d["bar_force"] = rel(f.truss_forces(), sv.truss_forces(U, eps0))
    if m.beams:
        d["beam_end_force"] = rel(f.beam_end_forces(), sv.beam_end_forces(U))
    if n_modes:
        fr, _ = sv.modes(n_modes)
        fo, _ = OpsFrame(m).modes(n_modes)
        d["frequency"] = float(np.max(np.abs(fo - fr) / fr))
    bad = {k: v for k, v in d.items() if v > tol}
    if verbose:
        print(f"crosscheck ({len(m.X)} nodes, {time.time() - t:.1f} s): " + ", ".join(f"{k} {v:.1e}" for k, v in d.items())
              + ("  passed" if not bad else f"  FAILED: {', '.join(bad)}"))
    if bad:
        raise AssertionError(f"linear solver and OpenSees disagree ({bad}); use OpenSees for this model")
    return d
