"""Prestressing tendons on frame models: equivalent loads, total, primary and secondary effects.

A tendon follows a chain of beams with a given eccentricity below the beam axis (towards −local z) at every node of the
chain and a tension force in every element; inside an element it runs straight between its two end points, so a
curved tendon is followed by refining the beams. The tendon acts on the concrete through the force at each kink and
anchorage; per element this is P·u at the tendon point of the i end and −P·u at the j end (u: unit vector along the
segment), moved to the nodes with the moment of the offset. Solving the frame under these loads gives the total effect.
The primary effect is the concrete's share of the tendon force at each section, −P·u acting at the tendon point; what
the supports add on a statically indeterminate structure is the secondary effect: secondary = total − primary.

    from femotion.frame.prestress import Tendon, effects, equivalent_loads
    td = Tendon(deck_beams, ecc=e_nodes, force=P_elements)      # m, kN; ecc > 0 below the axis
    tot, prim, sec = effects(sv, td, chain_sections(deck_beams), "My")
    U = sv.solve(equivalent_loads(sv, td))                      # camber

Signs follow femotion.frame.post.chain_force: N tension +, My sagging + (with the default reference vector). With
effective forces after all losses the secondary effect already includes the long-term reduction (Pe / Pi).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .influence import _sign
from .linear import Solver
from .post import _COMP


@dataclass
class Tendon:
    beams: list  # beam ids along the tendon, in order (consecutive beams share a node)
    ecc: np.ndarray  # eccentricity below the beam axis at each node of the chain (n_beams + 1), or (n_beams, 2) per end (m)
    force: np.ndarray | float  # tension force in each element (n_beams), or one value (kN)

    def ends(self):
        e = np.asarray(self.ecc, float)
        n = len(self.beams)
        if e.shape == (n + 1,):
            return np.c_[e[:-1], e[1:]]
        if e.shape == (n, 2):
            return e
        raise ValueError("ecc: one value per node of the chain (n_beams + 1) or two per beam (n_beams, 2)")

    def forces(self):
        return np.broadcast_to(np.asarray(self.force, float), (len(self.beams),))


def _segment(sv: Solver, e: int, ei: float, ej: float):
    """global tendon points at both ends and the unit vector of the segment"""
    m = sv.m
    i, j, sec, ref, tag = m.beams[e]
    L, R, T = sv._beam_cache[e]
    ez = R[2]
    Ai = m.X[i] - ei * ez
    Aj = m.X[j] - ej * ez
    u = (Aj - Ai) / np.linalg.norm(Aj - Ai)
    return i, j, R, Ai - m.X[i], Aj - m.X[j], u


def equivalent_loads(sv: Solver, td: Tendon) -> np.ndarray:
    """nodal loads (ndof) the tendon exerts on the frame"""
    F = np.zeros(sv.m.ndof)
    for e, (ei, ej), P in zip(td.beams, td.ends(), td.forces()):
        i, j, R, ri, rj, u = _segment(sv, e, ei, ej)
        f = P * u
        F[6 * i:6 * i + 3] += f
        F[6 * i + 3:6 * i + 6] += np.cross(ri, f)
        F[6 * j:6 * j + 3] -= f
        F[6 * j + 3:6 * j + 6] -= np.cross(rj, f)
    return F


def primary_end_forces(sv: Solver, td: Tendon) -> np.ndarray:
    """local end forces (n_beams_total, 12) of the primary state: on each tendon element, P·u at the i end and −P·u at
    the j end with the moments of the tendon offsets (zero for beams the tendon does not follow)"""
    out = np.zeros((len(sv.m.beams), 12))
    for e, (ei, ej), P in zip(td.beams, td.ends(), td.forces()):
        i, j, R, ri, rj, u = _segment(sv, e, ei, ej)
        f = P * u
        out[e] = np.r_[R @ f, R @ np.cross(ri, f), -R @ f, -R @ np.cross(rj, f)]
    return out


def effects(sv: Solver, td: Tendon | list, sections, comp: str):
    """(total, primary, secondary) section forces of one tendon or a list of tendons at the given sections
    ((beam, "i" | "j"), as femotion.frame.influence.chain_sections)"""
    tds = td if isinstance(td, (list, tuple)) else [td]
    F = sum(equivalent_loads(sv, t) for t in tds)
    tot_ef = sv.beam_end_forces(sv.solve(F))
    pri_ef = sum(primary_end_forces(sv, t) for t in tds)
    a, b = _COMP[comp]
    pick = lambda ef: np.array([_sign(comp, end) * ef[e][a if end == "i" else b] for e, end in sections])
    tot, pri = pick(tot_ef), pick(pri_ef)
    return tot, pri, tot - pri
