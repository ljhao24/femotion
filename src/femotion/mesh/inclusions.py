"""Hexahedral meshes that resolve round bars: a rectangular section with circular inclusions (O-grids), swept along the
bars. For local models where the bar's own surface matters — rust expanding around a corroding bar, splitting of the cover
around a bar under bond, a bar in a hole — rather than the embedded line bars of femotion.continuum.opensees.

Each bar sits in a square tile centred on it; inside the tile the cells follow the circle (a core square, then rings), and
the tiles fit into a tensor grid of columns (y) and rows (z) that fills the rest of the rectangle with plain blocks. The
bar may carry a thin ring at its surface (the rust layer of a corroding bar). Sections lie in the (y, z) plane, the bars
run along x, like femotion.mesh.section.

    from femotion.mesh.inclusions import bar_block, ring_cells
    m = bar_block(xs=np.linspace(0, 100, 11), width=150, height=80, bars=[(75, 31, 12)], h=5, ring=0.5)
    m.cell_groups["concrete"], m.cell_groups["steel"], m.cell_groups["rust"], …["steel0"], …["rust0"]

Tiles are as large as fits (the smallest distance from a bar to the edges, or half the distance to its nearest
neighbour, the same for all bars), so the cover between a bar and the surface is meshed radially. Bars sharing a row must
share its height, bars sharing a column its position: tiles that overlap partly (staggered layers) raise an error.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Mesh, orient


def _angles(m: int) -> np.ndarray:
    """4m ray angles around a tile, counter-clockwise from −45° (the corner below the right side)"""
    return -math.pi / 4 + np.arange(4 * m) * (math.pi / 2) / m


def _ray_square(a: float, th: np.ndarray) -> np.ndarray:
    """points of the square of half-size a on the rays at angles th (n, 2)"""
    c, s = np.cos(th), np.sin(th)
    k = a / np.maximum(np.abs(c), np.abs(s))
    return np.c_[k * c, k * s]


def _layers(n: int, grow: float) -> np.ndarray:
    """n + 1 fractions from 0 to 1, each step `grow` times the previous"""
    if n <= 0:
        return np.array([0.0])
    d = grow ** np.arange(n)
    return np.r_[0.0, np.cumsum(d)] / d.sum()


def _ogrid(c, r: float, ring: float, m: int, outer, n_steel: int, n_ring: int, n_out: int, grow: float):
    """quadrilaterals of one inclusion around centre c: core square + steel rings to r − ring, ring layers to r, then
    layers out to `outer` (("square", a) or ("circle", R)); returns points (n, 2), quads (q, 4), region per quad
    (0 steel, 1 ring, 2 outside)"""
    th = _angles(m)
    rs = r - ring
    q = 0.45 * rs
    t = np.tan(-math.pi / 4 + np.arange(m + 1) * (math.pi / 2) / m)
    # core: tensor grid q·tan(θ_i) in both directions
    Y, Z = np.meshgrid(q * t, q * t, indexing="ij")
    core = np.c_[Y.ravel(), Z.ravel()]
    nid = lambda i, j: i * (m + 1) + j  # noqa: E731
    I, J = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    I, J = I.ravel(), J.ravel()
    quads = [np.c_[nid(I, J), nid(I + 1, J), nid(I + 1, J + 1), nid(I, J + 1)]]
    region = [np.zeros(m * m, np.int64)]
    pts = [core]
    # rings of 4m points, from the core square outwards
    unit = np.c_[np.cos(th), np.sin(th)]
    rings = [_ray_square(q, th)]
    kinds = []
    for s in _layers(n_steel, 1.0)[1:]:
        rings.append((1 - s) * _ray_square(q, th) + s * rs * unit)
        kinds.append(0)
    for s in _layers(n_ring, 1.0)[1:] if ring > 0 else []:
        rings.append((rs + s * ring) * unit)
        kinds.append(1)
    edge = _ray_square(outer[1], th) if outer[0] == "square" else outer[1] * unit
    for s in _layers(n_out, grow)[1:]:
        rings.append((1 - s) * r * unit + s * edge)
        kinds.append(2)
    # the first ring is the core boundary: map ring-0 points to core nodes
    n0 = (m + 1) ** 2
    b = []
    for k in range(4 * m):  # walk the core boundary counter-clockwise from the corner (q, −q)
        side, i = divmod(k, m)
        b.append({0: nid(m, i), 1: nid(m - i, m), 2: nid(0, m - i), 3: nid(i, 0)}[side])
    prev = np.array(b)
    off = n0
    for ring_pts, kind in zip(rings[1:], kinds):
        pts.append(ring_pts)
        cur = off + np.arange(4 * m)
        nxt = np.roll(np.arange(4 * m), -1)
        quads.append(np.c_[prev, prev[nxt], cur[nxt], cur])
        region.append(np.full(4 * m, kind))
        prev = cur
        off += 4 * m
    P = np.concatenate(pts) + np.asarray(c, float)
    return P, np.concatenate(quads), np.concatenate(region), prev


def _n_out(span: float, first: float, grow: float) -> int:
    if span <= first:
        return 1
    return max(1, int(math.ceil(math.log(1 + span * (grow - 1) / first) / math.log(grow)))) if grow > 1 else \
        max(1, int(math.ceil(span / first)))


def bar_section(width: float, height: float, bars, *, h: float, ring: float = 0.0, m: int | None = None,
                tile: float | None = None,
                first: float | None = None, grow: float = 1.25, n_steel: int = 2, n_ring: int = 1):
    """the quadrilateral section: points (n, 2) as (y, z), quads (q, 4), region per quad (0 steel, 1 ring, 2 concrete)
    and bar index per quad (−1: concrete outside the tiles). bars: [(y, z, d)] (centre, diameter); h: cell size away
    from the bars; ring: thickness of the ring at each bar's surface, inside its diameter (0: none); m: divisions per
    tile side (4m around each bar); tile: half-size of the tiles (default: as large as fits); first: thickness of the
    first concrete layer at the bar (default: the ring thickness, or d/8); grow: growth of the layers outwards"""
    B = np.asarray(bars, float).reshape(-1, 3)
    if np.any(B[:, 2] <= 0) or (ring and np.any(ring >= B[:, 2] / 2)):
        raise ValueError("bars need d > 0 and a ring thinner than the radius")
    lim = np.minimum.reduce([B[:, 0], width - B[:, 0], B[:, 1], height - B[:, 1]])
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                lim[i] = min(lim[i], 0.5 * np.max(np.abs(B[i, :2] - B[j, :2])))
    a = float(tile if tile is not None else lim.min())
    if np.any(a > lim + 1e-9 * max(width, height)) or np.any(a <= B[:, 2] / 2 * 1.05):
        raise ValueError(f"tile half-size {a:g}: must exceed the bar radius and fit the section and the neighbours")
    tol = 1e-9 * max(width, height)
    m = int(m) if m is not None else max(4, int(math.ceil(2 * a / h - 1e-9)))

    def intervals(cs):
        iv = sorted({(round(float(v - a) / tol), round(float(v + a) / tol)) for v in cs})
        for (_, h0), (l1, _) in zip(iv[:-1], iv[1:]):
            if l1 < h0:
                raise ValueError("tiles of bars overlap partly (staggered bars): align them or give a smaller tile")
        return [(l * tol, u * tol) for l, u in iv]

    ty, tz = intervals(B[:, 0]), intervals(B[:, 1])
    t = np.tan(-math.pi / 4 + np.arange(m + 1) * (math.pi / 2) / m)

    def axis_coords(L, tiles):
        """breakpoints and the coordinates of every column (row): tan spacing in tiles, h elsewhere"""
        cols = []
        edges = [0.0] + [v for iv in tiles for v in iv] + [L]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi - lo <= tol:
                continue
            if any(abs(lo - a0) < tol and abs(hi - a1) < tol for a0, a1 in tiles):
                cols.append(((lo, hi), 0.5 * (lo + hi) + a * t, True))
            else:
                n = max(1, int(math.ceil((hi - lo) / h - 1e-9)))
                cols.append(((lo, hi), np.linspace(lo, hi, n + 1), False))
        return cols

    cy, cz = axis_coords(width, ty), axis_coords(height, tz)
    first = first if first is not None else (ring if ring > 0 else B[:, 2].min() / 8)
    pts, quads, region, owner = [], [], [], []
    off = 0
    for (yr, ys, ytile) in cy:
        for (zr, zs, ztile) in cz:
            if ytile and ztile:
                k = [i for i in range(len(B)) if yr[0] - tol < B[i, 0] < yr[1] + tol and zr[0] - tol < B[i, 1] < zr[1] + tol]
                if k:
                    i = k[0]
                    r = B[i, 2] / 2
                    n_out = _n_out(a - r, first, grow)
                    P, Q, R, _ = _ogrid(B[i, :2], r, ring, m, ("square", a), n_steel, n_ring, n_out, grow)
                    pts.append(P)
                    quads.append(Q + off)
                    region.append(R)
                    owner.append(np.full(len(R), i))
                    off += len(P)
                    continue
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            ny, nz = len(ys) - 1, len(zs) - 1
            nid = lambda i, j, nz=nz: i * (nz + 1) + j  # noqa: E731
            I, J = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
            I, J = I.ravel(), J.ravel()
            pts.append(np.c_[Y.ravel(), Z.ravel()])
            quads.append(np.c_[nid(I, J), nid(I + 1, J), nid(I + 1, J + 1), nid(I, J + 1)] + off)
            region.append(np.full(len(I), 2))
            owner.append(np.full(len(I), -1))
            off += (ny + 1) * (nz + 1)
    P = np.concatenate(pts)
    Q = np.concatenate(quads)
    key = np.round(P / tol).astype(np.int64)
    _, first_i, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    P, Q = P[first_i], np.asarray(inv).ravel()[Q]
    Q = orient(P, "quad4", Q)
    return P, Q, np.concatenate(region), np.concatenate(owner)


def _sweep(P2, Q, xs) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, float)
    n2 = len(P2)
    pts = np.concatenate([np.c_[np.full(n2, x), P2] for x in xs])
    hexes = np.concatenate([np.c_[Q + k * n2, Q + (k + 1) * n2] for k in range(len(xs) - 1)])
    return pts, hexes


def _groups(region, owner, nl: int, nbars: int, ring: bool) -> dict:
    reg = np.tile(region, nl)
    own = np.tile(owner, nl)
    g = {"concrete": {"hex8": np.flatnonzero(reg == 2)}, "steel": {"hex8": np.flatnonzero(reg == 0)}}
    if ring:
        g["rust"] = {"hex8": np.flatnonzero(reg == 1)}
    for i in range(nbars):
        g[f"steel{i}"] = {"hex8": np.flatnonzero((reg == 0) & (own == i))}
        if ring:
            g[f"rust{i}"] = {"hex8": np.flatnonzero((reg == 1) & (own == i))}
    return g


def bar_block(xs, width: float, height: float, bars, *, h: float, ring: float = 0.0, m: int | None = None,
              tile: float | None = None, first: float | None = None, grow: float = 1.25, n_steel: int = 2,
              n_ring: int = 1) -> Mesh:
    """hexahedra of a block y ∈ [0, width], z ∈ [0, height], x over the stations xs, with round bars along x
    (bar_section's arguments). Cell groups: concrete, steel, rust (with ring > 0), and steel<i>, rust<i> per bar"""
    P2, Q, region, owner = bar_section(width, height, bars, h=h, ring=ring, m=m, tile=tile, first=first, grow=grow,
                                       n_steel=n_steel, n_ring=n_ring)
    pts, hexes = _sweep(P2, Q, xs)
    hexes = orient(pts, "hex8", hexes)
    return Mesh(pts, {"hex8": hexes}, cell_groups=_groups(region, owner, len(xs) - 1, len(np.asarray(bars).reshape(-1, 3)),
                                                          ring > 0))


def cylinder(xs, d: float, R: float, *, ring: float = 0.0, m: int = 8, first: float | None = None, grow: float = 1.15,
             n_steel: int = 2, n_ring: int = 1) -> Mesh:
    """a concrete cylinder of outer radius R around one bar of diameter d on the x axis (centre y = z = 0), over the
    stations xs: the thick-walled cylinder of cover-cracking models. Cell groups as bar_block (bar index 0)"""
    r = d / 2
    first = first if first is not None else (ring if ring > 0 else d / 8)
    P2, Q, region, _ = _ogrid((0.0, 0.0), r, ring, m, ("circle", R), n_steel, n_ring, _n_out(R - r, first, grow), grow)
    Q = orient(P2, "quad4", Q)
    pts, hexes = _sweep(P2, Q, xs)
    hexes = orient(pts, "hex8", hexes)
    return Mesh(pts, {"hex8": hexes}, cell_groups=_groups(region, np.zeros(len(region), np.int64), len(xs) - 1, 1,
                                                          ring > 0))


def hoop_normals(mesh: Mesh, bars, cells=None) -> np.ndarray:
    """per hexahedron the hoop direction (0, −sin θ, cos θ) about the nearest bar axis (bars: [(y, z, …)], along x): the
    crack normal of radial splitting cracks, for OpsSolid(band=("projected", hoop_normals(...)))"""
    B = np.asarray(bars, float).reshape(-1, 3)[:, :2]
    c = mesh.points[mesh.cells["hex8"]].mean(axis=1) if cells is None else cells
    d = c[:, None, 1:] - B[None]
    k = np.argmin(np.linalg.norm(d, axis=2), axis=1)
    v = d[np.arange(len(c)), k]
    n = np.c_[np.zeros(len(c)), -v[:, 1], v[:, 0]]
    ln = np.linalg.norm(n, axis=1)
    n[ln > 0] /= ln[ln > 0][:, None]
    n[ln == 0] = (0.0, 0.0, 1.0)
    return n


__all__ = ["bar_section", "bar_block", "cylinder", "hoop_normals"]
