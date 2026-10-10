"""Frame post-processing on plain arrays, independent of which solver produced them.

Both femotion.frame.linear.Solver and femotion.frame.opensees.OpsFrame give displacements U (ndof, 6 per node) and beam
end forces (n_beams, 12) in the same order and sign convention, so everything here works with either.
"""
from __future__ import annotations

import numpy as np

_COMP = {"N": (0, 6), "Vy": (1, 7), "Vz": (2, 8), "T": (3, 9), "My": (4, 10), "Mz": (5, 11)}
MODE_KINDS = ("vertical", "lateral", "torsion", "longitudinal")


def translations(U: np.ndarray) -> np.ndarray:
    """6-dof displacement vector → nodal translations (N, 3)"""
    return U.reshape(-1, 6)[:, :3]


def rotations(U: np.ndarray) -> np.ndarray:
    return U.reshape(-1, 6)[:, 3:]


def reactions(K, U: np.ndarray, F: np.ndarray) -> np.ndarray:
    """support reactions (including springs to fixed nodes): K U − F, non-zero only at restrained dofs"""
    return K @ U - F


def chain_force(end_forces: np.ndarray, beam_ids, comp: str = "My") -> np.ndarray:
    """internal force along a chain of consecutive beams (beam_ids in order, n beams → n+1 nodes), averaging the two
    element end values at each node. end_forces: (n_beams, 12) local end forces.
    comp: 'N' axial (tension positive), 'My' vertical bending moment (with local z as the reference direction, sagging
    positive), 'Mz' lateral moment, 'T' torque, 'Vz', 'Vy' shear."""
    a, b = _COMP[comp]
    # end forces act from the element on its nodes: flip the i end for N, T and shear to get section forces; moments
    # use "i end positive, j end negative" to bring both ends to one sign convention
    si, sj = (-1, 1) if comp in ("N", "T", "Vy", "Vz") else (1, -1)
    out = np.zeros(len(beam_ids) + 1)
    cnt = np.zeros(len(beam_ids) + 1)
    for k, e in enumerate(beam_ids):
        out[k] += si * end_forces[e][a]
        out[k + 1] += sj * end_forces[e][b]
        cnt[k] += 1
        cnt[k + 1] += 1
    return out / cnt


def classify_mode(phi: np.ndarray, nodes, torsion_arm: float = 1.0) -> tuple[str, bool]:
    """classify a mode shape from the largest component along a chain of nodes (usually the deck): returns (kind, symmetric).
    kind is one of MODE_KINDS; symmetry compares the sign at 1/4 and 3/4 of the chain. torsion_arm converts rotations
    to edge displacements."""
    dk = phi.reshape(-1, 6)[np.asarray(nodes)]
    vert = np.abs(dk[:, 2]).max()
    lat = np.abs(dk[:, 1]).max()
    tor = np.abs(dk[:, 3]).max() * torsion_arm
    lon = np.abs(dk[:, 0]).max()
    k = int(np.argmax([vert, lat, tor, lon]))
    c = k if k < 3 else 2
    sym = bool(np.sign(dk[len(dk) // 4, c]) == np.sign(dk[3 * len(dk) // 4, c]))
    return MODE_KINDS[k], sym
