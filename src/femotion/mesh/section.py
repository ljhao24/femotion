"""Hexahedral meshes of members with a varying cross-section: a structured quadrilateral cross-section swept along x.

Bridge decks (box girders, multi-cell voided slabs, T-beams), piers and pile caps are columns × layers: across the width
(y) the section splits into columns at the faces of webs, haunches, bearing plates and so on; up the height (z) into
layers between interface curves z_k(y) — soffit, top of the bottom slab (with its haunches), soffit of the top slab (an
arch over each cell), deck surface, top of the pavement. Each column × layer block is solid or void and belongs to a cell
group. The quadrilateral grid this gives is the same at every station along the member; only its node coordinates change
with x, so the end thickening of a girder, a variable depth or a haunch is a smooth change of the interface curves, and
the swept mesh is all hexahedra with nodes lined up across every seam.

    from femotion.mesh.section import LayeredSection, sweep
    sec = LayeredSection(y=[-2, -1.8, 1.8, 2], ny=[2, 12, 2], nz=[2, 6, 2],
                         solid=[[1, 1, 1], [1, 0, 1], [1, 1, 1]],          # (columns, layers): a single-cell box
                         names=[["web", "web", "web"], ["bottom", None, "top"], ["web", "web", "web"]])
    z = lambda x, y: [0 * y, 0.2 + 0 * y, 1.6 + 0 * y, 1.8 + 0 * y]      # interfaces z_0..z_K at (x, y)
    m = sweep(sec, xs=np.linspace(0, 30, 61), z=z)                      # Mesh (hex8) with cell groups web/bottom/top

Interfaces must be ordered (z_k ≤ z_k+1) wherever a solid block uses them; in void blocks they are free. Column
breakpoints `y` may also depend on x (y=callable(x) → breakpoints) for flaring webs or tapering widths. ny, nz: divisions
per column and per layer; `graded` from femotion.mesh.structured suits the x stations (refined at bearings, anchors).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .core import Mesh, orient


class LayeredSection:
    def __init__(self, y, ny, nz, solid, names=None):
        """y: column breakpoints (J + 1, increasing) or a function x → breakpoints; ny (J,), nz (K,): divisions;
        solid (J, K): 1 = material; names (J, K): cell-group name of each solid block (None: no group)"""
        self.y = y
        self.ny = np.asarray(ny, int)
        self.nz = np.asarray(nz, int)
        self.solid = np.asarray(solid, bool)
        J, K = len(self.ny), len(self.nz)
        if self.solid.shape != (J, K):
            raise ValueError(f"solid must be (columns, layers) = ({J}, {K}), got {self.solid.shape}")
        self.names = np.full((J, K), None, object) if names is None else np.asarray(names, object)
        # topology of the quadrilateral grid: node (i, j) = column-node i (0..NY), layer-node j (0..NZ)
        self.NY, self.NZ = int(self.ny.sum()), int(self.nz.sum())
        self.col_of = np.repeat(np.arange(J), self.ny)  # column of each cell column
        self.lay_of = np.repeat(np.arange(K), self.nz)
        I, Jz = np.meshgrid(np.arange(self.NY), np.arange(self.NZ), indexing="ij")
        keep = self.solid[self.col_of[I], self.lay_of[Jz]]
        self.cell_i, self.cell_j = I[keep], Jz[keep]
        nid = lambda i, j: i * (self.NZ + 1) + j  # noqa: E731
        q = np.c_[nid(self.cell_i, self.cell_j), nid(self.cell_i + 1, self.cell_j),
                  nid(self.cell_i + 1, self.cell_j + 1), nid(self.cell_i, self.cell_j + 1)]
        used = np.unique(q)
        self.remap = -np.ones((self.NY + 1) * (self.NZ + 1), np.int64)
        self.remap[used] = np.arange(len(used))
        self.used = used
        self.quads = self.remap[q]
        self.groups = {}
        for g in {n for n in self.names[self.solid].ravel() if n is not None}:
            sel = self.names[self.col_of[self.cell_i], self.lay_of[self.cell_j]] == g
            self.groups[g] = np.where(sel)[0]

    @property
    def n_nodes(self) -> int:
        return len(self.used)

    def _breaks(self, x: float) -> np.ndarray:
        b = np.asarray(self.y(x) if callable(self.y) else self.y, float)
        if len(b) != len(self.ny) + 1 or np.any(np.diff(b) <= 0):
            raise ValueError("column breakpoints: one more than columns, strictly increasing")
        return b

    def node_y(self, x: float = 0.0) -> np.ndarray:
        """y of every grid column of nodes (NY + 1)"""
        b = self._breaks(x)
        return np.concatenate([b[:1]] + [np.linspace(b[j], b[j + 1], n + 1)[1:] for j, n in enumerate(self.ny)])

    def coords(self, x: float, z: Callable) -> np.ndarray:
        """(n_nodes, 2) section coordinates (y, z) at station x; z(x, y) → list of K + 1 interface heights at y"""
        Y = self.node_y(x)
        Zi = np.asarray(z(x, Y), float)  # (K + 1, NY + 1)
        if Zi.shape != (len(self.nz) + 1, len(Y)):
            raise ValueError(f"z must give {len(self.nz) + 1} interfaces")
        t = np.concatenate([[0.0]] + [np.linspace(0, 1, n + 1)[1:] + k for k, n in enumerate(self.nz)])  # layer coordinate
        k = np.minimum(t.astype(int), len(self.nz) - 1)
        frac = t - k
        Zn = Zi[k] + frac[:, None] * (Zi[k + 1] - Zi[k])  # (NZ + 1, NY + 1)
        Yg = np.broadcast_to(Y[None, :], Zn.shape)
        P = np.c_[Yg.T.ravel(), Zn.T.ravel()]  # node id = i·(NZ+1) + j
        return P[self.used]

    def mesh(self, x: float, z: Callable) -> Mesh:
        """the cross-section at station x as a quadrilateral Mesh (plane problems on (y, z))"""
        P = self.coords(x, z)
        return Mesh(P, {"quad4": orient(P, "quad4", self.quads)}, cell_groups={g: {"quad4": c} for g, c in self.groups.items()})


def sweep(section: LayeredSection, xs, z: Callable, transform: Callable | None = None, x_groups: dict | None = None) -> Mesh:
    """hexahedra between consecutive stations xs; z(x, y) → interfaces as in LayeredSection.coords. transform(P) → P
    maps the straight (x, y, z) coordinates (e.g. onto a curved alignment); x_groups = {name: (x0, x1)} adds cell groups
    of the cells whose stations lie within [x0, x1]. Cell groups of the section carry over; cell c of the section between
    stations i and i+1 is cell i·n_quads + c"""
    xs = np.asarray(xs, float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError("stations must increase")
    n = section.n_nodes
    nq = len(section.quads)
    P = np.concatenate([np.c_[np.full(n, x), section.coords(x, z)] for x in xs])
    q = section.quads
    # hexahedron (VTK order): the quadrilateral at station i, then the same one at station i + 1 (orient() fixes handedness)
    h = np.concatenate([np.c_[q + i * n, q + (i + 1) * n] for i in range(len(xs) - 1)])
    if transform is not None:
        P = np.asarray(transform(P), float)
    groups = {g: {"hex8": (np.arange(len(xs) - 1)[:, None] * nq + c[None, :]).ravel()} for g, c in section.groups.items()}
    for name, (x0, x1) in (x_groups or {}).items():
        st = np.where((xs[:-1] >= x0 - 1e-12) & (xs[1:] <= x1 + 1e-12))[0]
        groups[name] = {"hex8": (st[:, None] * nq + np.arange(nq)[None, :]).ravel()}
    return Mesh(P, {"hex8": orient(P, "hex8", h)}, cell_groups=groups)
