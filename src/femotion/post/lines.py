"""Principal stress trajectories ("force flow lines"): polylines integrated along σ1 (tension lines) or σ2 (compression
lines); drawn, they show "how the force travels".

    lines = trajectories(mesh, S, seeds, which=1, step=0.05)
mesh is a 2D Mesh (tri3 / tri6 / quad4), S the nodal stresses (N, 3); seeds can come from seeds_grid / seeds_along.
"""
from __future__ import annotations

import numpy as np

from ..mesh.core import Mesh
from .stress import principal_2d


def inside_polygon(poly: np.ndarray):
    """returns a test f(x, y) -> bool (ray casting, single point)"""
    px, py = poly[:, 0], poly[:, 1]

    def f(x, y):
        c = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            if ((py[i] > y) != (py[j] > y)) and (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-30) + px[i]):
                c = not c
            j = i
        return c
    return f


def field_interpolator(mesh: Mesh, values: np.ndarray, k: int = 12):
    """linear interpolation of nodal values over the mesh triangles, f(x, y) → value (nan outside the mesh or in holes).
    Finds the k nearest triangles with a centroid KD-tree and picks the containing one by barycentric coordinates."""
    from scipy.spatial import cKDTree
    m = mesh.linear()
    if m.kind == "quad4":
        q = m.cells["quad4"]
        tri = np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    else:
        tri = m.cells["tri3"]
    P = m.points[:, :2]
    A, B, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    det = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
    tree = cKDTree((A + B + C) / 3)
    V = np.asarray(values, float)
    V = V[:, None] if V.ndim == 1 else V
    kk = min(k, len(tri))

    def f(x, y):
        _, idx = tree.query((x, y), k=kk)
        for t in np.atleast_1d(idx):
            l1 = ((B[t, 0] - x) * (C[t, 1] - y) - (C[t, 0] - x) * (B[t, 1] - y)) / det[t]
            l2 = ((C[t, 0] - x) * (A[t, 1] - y) - (A[t, 0] - x) * (C[t, 1] - y)) / det[t]
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-9:
                v = l1 * V[tri[t, 0]] + l2 * V[tri[t, 1]] + l3 * V[tri[t, 2]]
                return v if len(v) > 1 else v[0]
        return np.full(V.shape[1], np.nan) if V.shape[1] > 1 else np.nan
    return f


def _integrate(interp, seeds, which, step, inside, max_len, min_mag):
    lines = []
    for (x0, y0) in seeds:
        pts = []
        for sgn in (1, -1):
            x, y = x0, y0
            prev = None
            seg = []
            L = 0.0
            while L < max_len and inside(x, y):
                s1, s2, th = interp(x, y)
                mag = abs(s1 if which == 1 else s2)
                if not np.isfinite(th) or mag < min_mag:
                    break
                ang = th if which == 1 else th + np.pi / 2
                d = np.array([np.cos(ang), np.sin(ang)])
                if prev is None:
                    d = d * sgn
                elif np.dot(d, prev) < 0:
                    d = -d
                xm, ym = x + 0.5 * step * d[0], y + 0.5 * step * d[1]
                if not inside(xm, ym):
                    break
                s1m, s2m, thm = interp(xm, ym)
                angm = thm if which == 1 else thm + np.pi / 2
                dm = np.array([np.cos(angm), np.sin(angm)])
                if np.dot(dm, d) < 0:
                    dm = -dm
                seg.append((x, y, mag))
                x, y = x + step * dm[0], y + step * dm[1]
                prev = dm
                L += step
            pts = seg if sgn == 1 else (seg[::-1][:-1] + pts if seg else pts)
        if len(pts) > 3:
            lines.append(np.array(pts))
    return lines


def trajectories(mesh: Mesh, S: np.ndarray, seeds, which: int = 1, step: float | None = None, inside=None,
                 max_len: float = np.inf, min_mag: float = 0.0) -> list[np.ndarray]:
    """which=1: tension lines (along σ1), 2: compression lines (along σ2). Returns a list of polylines, each (n, 3) = x, y, |σ|.
    inside(x, y) defaults to "the interpolation falls inside the mesh" (correct with holes too)."""
    f = field_interpolator(mesh, S)
    if step is None:
        lo, hi = mesh.bbox()
        step = float(np.max(hi - lo)) / 200

    def interp(x, y):
        v = f(x, y)
        if not np.all(np.isfinite(v)):
            return np.nan, np.nan, np.nan
        return principal_2d(v)

    if inside is None:
        inside = lambda x, y: bool(np.all(np.isfinite(f(x, y))))  # noqa: E731
    if not np.isfinite(max_len):
        lo, hi = mesh.bbox()
        max_len = 4 * float(np.sum(hi - lo))
    return _integrate(interp, seeds, which, step, inside, max_len, min_mag)


def seeds_grid(mesh: Mesh, nx: int, ny: int, margin: float = 0.02):
    lo, hi = mesh.bbox()
    d = (hi - lo) * margin
    xs = np.linspace(lo[0] + d[0], hi[0] - d[0], nx)
    ys = np.linspace(lo[1] + d[1], hi[1] - d[1], ny)
    return [(x, y) for x in xs for y in ys]


def seeds_along(p0, p1, n: int):
    return [tuple(np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * t) for t in np.linspace(0, 1, n)]
