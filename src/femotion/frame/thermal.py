"""Imposed strains on frame models: uniform temperature, temperature gradients, differential shrinkage.

Each beam of a chosen set gets a free (unrestrained) axial strain ε (elongation +) and a free curvature κ in vertical
bending (sagging +: the bottom fibre lengthens relative to the top; κ > 0 for a section warmer at the bottom). A
statically determinate structure deforms freely and carries no section forces; a statically indeterminate one is held
back by its supports, and those restraint forces are the secondary effects (JTG 3362-2018 4.3.8, D.0.2).

The restrained element carries N = −EA ε and M = −EI κ; its end forces, reversed, are the nodal loads, and the section
forces of the final state are the element forces K u plus the restrained ones.

    from femotion.frame.thermal import effects
    M2 = effects(sv, deck_beams, chain_sections(deck_beams), "My", curvature=kappa)   # 1/m, sagging +
    N2 = effects(sv, deck_beams, secs, "N", strain=alpha * dT)                        # uniform temperature

Signs follow femotion.frame.post.chain_force: N tension +, My sagging + (with the default reference vector).
"""
from __future__ import annotations

import numpy as np

from .influence import section_forces
from .linear import Solver


def restrained_end_forces(sv: Solver, beams, strain=0.0, curvature=0.0) -> np.ndarray:
    """local end forces (n_beams_total, 12) of the given beams held against their free strain and curvature (the other
    beams zero). strain, curvature: one value or one per beam of `beams`"""
    beams = [int(b) for b in beams]
    eps = np.broadcast_to(np.asarray(strain, float), (len(beams),))
    kap = np.broadcast_to(np.asarray(curvature, float), (len(beams),))
    out = np.zeros((len(sv.m.beams), 12))
    for e, ep, ka in zip(beams, eps, kap):
        sec = sv.m.beams[e][2]
        N = -sec.E * sec.A * ep  # section force, tension +
        M = -sec.E * sec.Iy * ka  # section moment, sagging +
        out[e, 0], out[e, 6] = -N, N
        out[e, 4], out[e, 10] = M, -M
    return out


def solve(sv: Solver, beams, strain=0.0, curvature=0.0):
    """(U, end forces) of the structure under the imposed strains: displacements and the local end forces of every
    beam (K u plus the restrained forces)"""
    fr = restrained_end_forces(sv, beams, strain, curvature)
    F = np.zeros(sv.m.ndof)
    for e in {int(b) for b in beams}:
        i, j = sv.m.beams[e][0], sv.m.beams[e][1]
        T = sv._beam_cache[e][2]
        F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] -= T.T @ fr[e]
    U = sv.solve(F)
    return U, sv.beam_end_forces(U) + fr


def effects(sv: Solver, beams, sections, comp: str, strain=0.0, curvature=0.0) -> np.ndarray:
    """secondary (restraint) section forces at the given sections ((beam, "i" | "j"), as
    femotion.frame.influence.chain_sections) of the imposed strain and curvature on `beams`; zero for a statically
    determinate structure"""
    U, ef = solve(sv, beams, strain, curvature)
    return section_forces(sv, U, sections, comp, end_forces=ef)
