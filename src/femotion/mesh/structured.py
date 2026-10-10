"""Structured meshes: the most direct way to show "cutting into small pieces", and the grid used by topology optimisation.

- line(L, n): a bar cut into n segments
- rect(lx, ly, nx, ny, kind="quad4"|"tri3"): a rectangle, cells numbered row by row from bottom-left to top-right
- box(lx, ly, lz, nx, ny, nz): a cuboid cut into hexahedra (hex8, VTK order)
- grid_for_density(nelx, nely): quadrilateral mesh matching femotion.topopt density arrays one-to-one
  (cell e = ely + elx·nely, ely counted from the top)
- graded / hex_block / weld: block-structured hexahedra for members built from boxes, prisms and tapered blocks
"""
from __future__ import annotations

import numpy as np

from .core import Mesh


def line(L: float, n: int, dim: int = 2, origin=(0.0, 0.0)) -> Mesh:
    x = np.linspace(0, L, n + 1)
    p = np.zeros((n + 1, dim))
    p[:, 0] = x + origin[0]
    p[:, 1] = origin[1]
    c = np.c_[np.arange(n), np.arange(1, n + 1)]
    return Mesh(p, {"line2": c}, {"left": np.array([0]), "right": np.array([n])})


def rect(lx: float, ly: float, nx: int, ny: int, kind: str = "quad4", origin=(0.0, 0.0), diag: str = "alt") -> Mesh:
    """node i + (nx+1)·j; quad4 cell i + nx·j (counter-clockwise); tri3 gives two triangles per square,
    diag="alt" alternates the diagonal (more symmetric)"""
    xs = np.linspace(0, lx, nx + 1) + origin[0]
    ys = np.linspace(0, ly, ny + 1) + origin[1]
    X, Y = np.meshgrid(xs, ys)
    p = np.c_[X.ravel(), Y.ravel()]
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    n0 = i + (nx + 1) * j
    q = np.c_[n0, n0 + 1, n0 + nx + 2, n0 + nx + 1]
    groups = {"left": np.where(np.isclose(p[:, 0], xs[0]))[0], "right": np.where(np.isclose(p[:, 0], xs[-1]))[0],
              "bottom": np.where(np.isclose(p[:, 1], ys[0]))[0], "top": np.where(np.isclose(p[:, 1], ys[-1]))[0]}
    if kind == "quad4":
        return Mesh(p, {"quad4": q}, groups)
    flip = ((i + j) % 2 == 1) if diag == "alt" else np.zeros(len(i), bool)
    t1 = np.where(flip[:, None], q[:, [0, 1, 3]], q[:, [0, 1, 2]])
    t2 = np.where(flip[:, None], q[:, [1, 2, 3]], q[:, [0, 2, 3]])
    t = np.stack([t1, t2], axis=1).reshape(-1, 3)
    return Mesh(p, {"tri3": t}, groups)


def grid_for_density(nelx: int, nely: int, h: float = 1.0) -> Mesh:
    """topology-optimisation grid (cell size h); cell order matches the topopt density array x.ravel()
    (shape (nelx, nely)); y points up"""
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ex, ey = ex.ravel(), ey.ravel()
    node = lambda ix, iy: (nely + 1) * ix + iy  # noqa: E731 (iy counted from the top)
    q = np.c_[node(ex, ey + 1), node(ex + 1, ey + 1), node(ex + 1, ey), node(ex, ey)]  # bottom-left, bottom-right, top-right, top-left
    ix, iy = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1), indexing="ij")
    p = np.c_[ix.ravel() * h, (nely - iy.ravel()) * h]
    return Mesh(p, {"quad4": q})


def box(lx: float, ly: float, lz: float, nx: int, ny: int, nz: int, origin=(0.0, 0.0, 0.0)) -> Mesh:
    """a cuboid cut into nx × ny × nz hexahedra; node i + (nx+1)·(j + (ny+1)·k); z points up"""
    xs = np.linspace(0, lx, nx + 1) + origin[0]
    ys = np.linspace(0, ly, ny + 1) + origin[1]
    zs = np.linspace(0, lz, nz + 1) + origin[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    p = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    nid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)  # noqa: E731
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    h = np.c_[nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
              nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
    groups = {"bottom": np.where(np.isclose(p[:, 2], zs[0]))[0], "top": np.where(np.isclose(p[:, 2], zs[-1]))[0]}
    return Mesh(p, {"hex8": h}, groups)


# ── block-structured hexahedra: members built from boxes, prisms and tapered blocks (piers, pile caps, girder segments…) ──
def graded(breaks, h: float) -> np.ndarray:
    """subdivide the gaps between break points (member edges, steps, changes of section) to a target size h and return
    the merged coordinates. All blocks must share the same coordinates so that nodes line up across seams"""
    b = np.unique(np.asarray(breaks, float))
    out = [b[:1]]
    for a, c in zip(b[:-1], b[1:]):
        n = max(1, int(np.ceil((c - a) / h - 1e-9)))
        out.append(np.linspace(a, c, n + 1)[1:])
    return np.concatenate(out)


def hex_block(xs, ys, z0, z1, nz: int) -> Mesh:
    """one hexahedral block: an xs × ys grid in plan, nz equal layers between z0 and z1.
    z0 and z1 may be constants or functions f(x, y) (tapered cantilevers, sloping soffits); each vertical line is divided separately"""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    nx, ny = len(xs) - 1, len(ys) - 1
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    f0 = z0(X, Y) if callable(z0) else np.full_like(X, z0)
    f1 = z1(X, Y) if callable(z1) else np.full_like(X, z1)
    t = np.linspace(0, 1, nz + 1)[:, None, None]
    Z = f0[None] + t * (f1 - f0)[None]
    p = np.c_[np.broadcast_to(X, Z.shape).ravel(), np.broadcast_to(Y, Z.shape).ravel(), Z.ravel()]
    nid = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)  # noqa: E731
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    h = np.c_[nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
              nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
    return Mesh(p, {"hex8": h})


def weld(blocks, tol: float = 1e-6) -> Mesh:
    """weld several hexahedral blocks into one mesh: nodes closer than tol are merged.
    Blocks must use the same coordinates on their seams (cut from the same graded() result), otherwise hanging nodes remain"""
    pts = np.concatenate([b.points for b in blocks])
    off = np.cumsum([0] + [b.n_points for b in blocks[:-1]])
    conn = np.concatenate([b.cells["hex8"] + o for b, o in zip(blocks, off)])
    key = np.round(pts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return Mesh(pts[first], {"hex8": inv.ravel()[conn]})
