"""What the global stiffness matrix looks like: which entries are non-zero and which element fills each one first,
for "assembly" animations."""
from __future__ import annotations

import numpy as np

from ..mesh.core import Mesh


def stiffness_pattern(mesh: Mesh, order: np.ndarray | None = None, G: int = 240, dof_per_node: int | None = None,
                      kind: str | None = None) -> np.ndarray:
    """returns a (G, G) integer array: -1 = always zero; otherwise the assembly rank of the first element contributing to
    that block. order: assembly rank of every cell (default: by id); nodes are not renumbered, keeping the mesher's bandwidth."""
    k = kind or mesh.kind
    T = mesh.linear().cells["tri3" if k == "tri6" else k]
    dpn = dof_per_node or mesh.dim
    rank = np.arange(len(T)) if order is None else np.asarray(order)
    ndof = dpn * mesh.n_points
    big = np.iinfo(np.int64).max
    first = np.full((G, G), big, np.int64)
    for e, el in enumerate(T):
        dofs = (dpn * el[:, None] + np.arange(dpn)[None, :]).ravel()
        cells = np.unique((dofs * G) // ndof)
        a, b = np.meshgrid(cells, cells, indexing="ij")
        cur = first[a, b]
        first[a, b] = np.minimum(cur, rank[e])
    first[first == big] = -1
    return first


def order_by_axis(mesh: Mesh, axis: int = 0) -> np.ndarray:
    """cell ranks sorted by centroid coordinate (assembling "piece by piece" from left to right)"""
    c = mesh.centroids()
    rank = np.empty(len(c), np.int64)
    rank[np.argsort(c[:, axis], kind="stable")] = np.arange(len(c))
    return rank
