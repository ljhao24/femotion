"""The unified mesh: node coordinates plus blocks of same-type cells. Every solver consumes it,
every result hangs on it, and the display layer only understands it.

Cell types and node order (consistent with gmsh and scikit-fem)
- line2: two end points (bars, beams, cables)
- tri3: triangle, counter-clockwise
- tri6: quadratic triangle; 0-1-2 corners, 3 = mid-edge 01, 4 = mid-edge 12, 5 = mid-edge 20
- quad4: quadrilateral, counter-clockwise
- tet4: tetrahedron, positive volume
- hex8: hexahedron (VTK order): 0-1-2-3 bottom face counter-clockwise (seen from above),
  4-5-6-7 top face, each directly above 0-1-2-3
Coordinates: points has shape (N, dim), dim = 2 or 3; y (2D) or z (3D) points up.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

NODES_PER_CELL = {"line2": 2, "tri3": 3, "tri6": 6, "quad4": 4, "tet4": 4, "hex8": 8}
TRI6_SUB = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])  # a quadratic triangle split into 4 linear ones
HEX_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])  # outward normals
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
# scikit-fem orders hexahedron corners by reference coordinates 000, 001, 010, 100, 011, 101, 110, 111; convert to/from VTK
HEX_VTK2SK = np.array([0, 4, 3, 1, 7, 5, 2, 6])  # scikit-fem corner k = VTK corner HEX_VTK2SK[k]
HEX_SK2VTK = np.argsort(HEX_VTK2SK)


@contextmanager
def scratch_cwd():
    """run a mesher in a temporary working directory: some write stray files into the current one (fTetWild's
    __tracked_surface.stl, gmsh's quadqs_statistics.json). Resolve relative input paths before entering."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(old)


def _cross(a, b):
    """z component (scalar) in 2D, the cross product vector in 3D"""
    if a.shape[-1] == 2:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.cross(a, b)


@dataclass
class Mesh:
    points: np.ndarray
    cells: dict[str, np.ndarray] = field(default_factory=dict)
    point_groups: dict[str, np.ndarray] = field(default_factory=dict)  # name → node ids
    cell_groups: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # name → {cell type: cell ids}

    def __post_init__(self):
        self.points = np.asarray(self.points, float)
        self.cells = {k: np.asarray(v, np.int64).reshape(-1, NODES_PER_CELL[k]) for k, v in self.cells.items()}

    # ── basic quantities ────────────────────────────────────────────────
    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def kind(self) -> str:
        """main cell type (the block with the highest dimension)"""
        order = ["hex8", "tet4", "quad4", "tri6", "tri3", "line2"]
        return next(k for k in order if k in self.cells)

    def n_cells(self, kind: str | None = None) -> int:
        return len(self.cells[kind or self.kind])

    def bbox(self):
        return self.points.min(axis=0), self.points.max(axis=0)

    # ── selecting nodes ─────────────────────────────────────────────────
    def nodes_where(self, pred) -> np.ndarray:
        """pred(x) → bool[N], x has shape (dim, N) (the scikit-fem convention)"""
        return np.where(pred(self.points.T))[0]

    def nearest(self, p) -> int:
        return int(np.argmin(np.linalg.norm(self.points - np.asarray(p, float), axis=1)))

    # ── topology ────────────────────────────────────────────────────────
    def linear(self) -> "Mesh":
        """split quadratic cells into linear ones (display only; nodes unchanged)"""
        if "tri6" not in self.cells:
            return self
        c = dict(self.cells)
        t6 = c.pop("tri6")
        c["tri3"] = t6[:, TRI6_SUB].reshape(-1, 3)
        return Mesh(self.points, c, self.point_groups)

    def edges(self, kind: str | None = None) -> np.ndarray:
        """all unique edges (E, 2) (quadratic triangles use their linear sub-edges; tetrahedra have six edges)"""
        k = kind or self.kind
        if k == "line2":
            return np.unique(np.sort(self.cells[k], axis=1), axis=0)
        if k == "tri6":
            return self.linear().edges("tri3")
        loc = {"tri3": [(0, 1), (1, 2), (2, 0)], "quad4": [(0, 1), (1, 2), (2, 3), (3, 0)],
               "tet4": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)], "hex8": HEX_EDGES}[k]
        c = self.cells[k]
        e = np.concatenate([c[:, [a, b]] for a, b in loc])
        return np.unique(np.sort(e, axis=1), axis=0)

    def boundary_edges(self) -> np.ndarray:
        """boundary edges of a 2D mesh (E, 2), oriented like the cells (domain on the left)"""
        k = self.kind
        m = self.linear() if k == "tri6" else self
        k = "tri3" if k == "tri6" else k
        loc = {"tri3": [(0, 1), (1, 2), (2, 0)], "quad4": [(0, 1), (1, 2), (2, 3), (3, 0)]}[k]
        c = m.cells[k]
        e = np.concatenate([c[:, [a, b]] for a, b in loc])
        key = np.sort(e, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return e[cnt[inv.ravel()] == 1]

    def boundary_loops(self) -> list[np.ndarray]:
        """boundary edges chained into closed loops (outline, holes), each a node-id sequence, longest first.
        Independent of cell orientation"""
        adj: dict[int, list[int]] = {}
        for a, b in self.boundary_edges():
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        seen: set[int] = set()
        loops = []
        for start in adj:
            if start in seen:
                continue
            loop, prev, cur = [start], None, start
            seen.add(start)
            while True:
                nxt = next((n for n in adj[cur] if n != prev and n not in seen), None)
                if nxt is None:
                    break
                loop.append(nxt)
                seen.add(nxt)
                prev, cur = cur, nxt
            loops.append(np.array(loop))
        return sorted(loops, key=len, reverse=True)

    def surface(self) -> tuple[np.ndarray, np.ndarray]:
        """outer surface of a 3D mesh: returns (faces with outward normals, owning cell of each face).
        Tetrahedral meshes give triangles (F, 3), hexahedral meshes give quadrilaterals (F, 4)"""
        if "hex8" in self.cells:
            h = self.cells["hex8"]
            f = h[:, HEX_FACES].reshape(-1, 4)
            owner = np.repeat(np.arange(len(h)), 6)
            _, inv, cnt = np.unique(np.sort(f, axis=1), axis=0, return_inverse=True, return_counts=True)
            b = cnt[inv.ravel()] == 1
            return f[b], owner[b]
        t = self.cells["tet4"]
        loc = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])  # face k is opposite vertex k, ordered for an outward normal
        f = t[:, loc].reshape(-1, 3)
        owner = np.repeat(np.arange(len(t)), 4)
        key = np.sort(f, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        b = cnt[inv.ravel()] == 1
        f, owner = f[b], owner[b]
        # safety net: fix orientation by "face centre − cell centre"
        p = self.points
        n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        out = np.einsum("ij,ij->i", n, p[f].mean(1) - p[t[owner]].mean(1)) < 0
        f[out] = f[out][:, [0, 2, 1]]
        return f, owner

    def measure(self, kind: str | None = None) -> np.ndarray:
        """length / area / volume of every cell"""
        k = kind or self.kind
        p, c = self.points, self.cells[k]
        if k == "line2":
            return np.linalg.norm(p[c[:, 1]] - p[c[:, 0]], axis=1)
        if k in ("tri3", "tri6"):
            a, b, d = p[c[:, 0]], p[c[:, 1]], p[c[:, 2]]
            cr = _cross(b - a, d - a)
            return 0.5 * (np.abs(cr) if cr.ndim == 1 else np.linalg.norm(cr, axis=1))
        if k == "quad4":
            a, b, d, e = (p[c[:, i]] for i in range(4))
            return 0.5 * np.abs(_cross(d - a, e - b))
        if k == "hex8":  # six tetrahedra around the 0-6 diagonal
            tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
            return sum(np.abs(np.einsum("ij,ij->i", np.cross(p[c[:, b]] - p[c[:, a]], p[c[:, d]] - p[c[:, a]]), p[c[:, e]] - p[c[:, a]])) / 6
                       for a, b, d, e in tets)
        a, b, d, e = (p[c[:, i]] for i in range(4))
        return np.abs(np.einsum("ij,ij->i", np.cross(b - a, d - a), e - a)) / 6

    def centroids(self, kind: str | None = None) -> np.ndarray:
        k = kind or self.kind
        c = self.cells[k][:, :3] if k == "tri6" else self.cells[k]
        return self.points[c].mean(axis=1)

    def compact(self) -> "Mesh":
        """drop nodes that no cell uses and renumber"""
        used = np.unique(np.concatenate([c.ravel() for c in self.cells.values()]))
        remap = -np.ones(self.n_points, np.int64)
        remap[used] = np.arange(len(used))
        pg = {k: remap[v][remap[v] >= 0] for k, v in self.point_groups.items()}
        return Mesh(self.points[used], {k: remap[v] for k, v in self.cells.items()}, pg, self.cell_groups)

    def subset(self, groups) -> tuple["Mesh", np.ndarray, np.ndarray]:
        """the cells of the given cell groups as a mesh of their own (construction stages: the girder before the pavement
        is cast). Returns (mesh, cell ids of the main block in this mesh, node ids in this mesh); groups are carried over
        where their cells survive"""
        if isinstance(groups, str):
            groups = [groups]
        k = self.kind
        cells = np.unique(np.concatenate([np.asarray(self.cell_groups[g].get(k, []), np.int64) for g in groups]))
        conn = self.cells[k][cells]
        nodes = np.unique(conn)
        remap = -np.ones(self.n_points, np.int64)
        remap[nodes] = np.arange(len(nodes))
        cmap = -np.ones(len(self.cells[k]), np.int64)
        cmap[cells] = np.arange(len(cells))
        cg = {}
        for g, d in self.cell_groups.items():
            c = cmap[np.asarray(d.get(k, []), np.int64)]
            c = c[c >= 0]
            if len(c):
                cg[g] = {k: c}
        pg = {g: remap[v][remap[v] >= 0] for g, v in self.point_groups.items()}
        return Mesh(self.points[nodes], {k: remap[conn]}, pg, cg), cells, nodes

    # ── conversion to / from scikit-fem ─────────────────────────────────
    def to_skfem(self):
        from skfem import MeshHex, MeshQuad, MeshTet, MeshTri
        k = self.kind
        if k in ("tri3", "tri6"):
            c = self.cells[k][:, :3]
            used = np.unique(c)
            if len(used) != self.n_points:  # quadratic mesh: corners only
                remap = -np.ones(self.n_points, np.int64)
                remap[used] = np.arange(len(used))
                return MeshTri(self.points[used].T.copy(), remap[c].T.copy())
            return MeshTri(self.points.T.copy(), c.T.copy())
        if k == "quad4":
            return MeshQuad(self.points.T.copy(), self.cells[k].T.copy())
        if k == "tet4":
            return MeshTet(self.points.T.copy(), self.cells[k].T.copy())
        if k == "hex8":
            return MeshHex(self.points.T.copy(), self.cells[k][:, HEX_VTK2SK].T.copy())
        raise ValueError(f"scikit-fem does not support {k}")

    @classmethod
    def from_skfem(cls, m) -> "Mesh":
        name = type(m).__name__
        kind = {"MeshTri1": "tri3", "MeshTri": "tri3", "MeshQuad1": "quad4", "MeshQuad": "quad4", "MeshTet1": "tet4", "MeshTet": "tet4",
                "MeshHex1": "hex8", "MeshHex": "hex8"}[name]
        p = m.p.T.copy()
        t = m.t.T.copy()
        if kind == "hex8":
            t = t[:, HEX_SK2VTK]
        return cls(p, {kind: orient(p, kind, t)})  # scikit-fem sorts cell vertex ids, so re-orient


def orient(points: np.ndarray, kind: str, c: np.ndarray) -> np.ndarray:
    """make cell orientation uniform: triangles / quadrilaterals counter-clockwise, tetrahedra positive volume"""
    c = c.copy()
    p = points
    if kind in ("tri3", "quad4", "tri6"):
        a, b, d = p[c[:, 0]], p[c[:, 1]], p[c[:, 2]]
        neg = ((b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])) < 0
        if kind == "tri3":
            c[neg] = c[neg][:, [0, 2, 1]]
        elif kind == "tri6":
            c[neg] = c[neg][:, [0, 2, 1, 5, 4, 3]]
        else:
            c[neg] = c[neg][:, ::-1]
    elif kind == "tet4":
        a, b, d, e = (p[c[:, i]] for i in range(4))
        neg = np.einsum("ij,ij->i", np.cross(b - a, d - a), e - a) < 0
        c[neg] = c[neg][:, [0, 2, 1, 3]]
    elif kind == "hex8":  # the three edges at corner 0 form a right-handed frame
        a = p[c[:, 0]]
        neg = np.einsum("ij,ij->i", np.cross(p[c[:, 1]] - a, p[c[:, 3]] - a), p[c[:, 4]] - a) < 0
        c[neg] = c[neg][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return c
