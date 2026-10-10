"""Tetrahedra from messy surface models (extra "tetwild"): fTetWild, through pytetwild, fills a triangle surface with
good-quality tetrahedra even when the surface is not clean — holes, gaps, self-intersections, overlapping parts,
non-manifold edges, as in Blender scenes, downloaded models and scans. gmsh needs a clean, closed CAD solid instead.

    from femotion.mesh import tetwild
    m = tetwild.from_file("tower.obj", edge=0.4)      # OBJ, STL, PLY, OFF, … (anything meshio reads; extra "io")
    m = tetwild.from_surface(vertices, faces, edge=0.4)

fTetWild keeps the boundary of the result within `epsilon` × the bounding-box diagonal of the input (its envelope), so
details smaller than that are smoothed away; lower epsilon for thin parts (slower). Overlapping parts become one solid
and small holes are closed: a winding-number filter keeps what lies inside the surface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Mesh, orient, scratch_cwd


def from_surface(vertices, faces, edge: float | None = None, edge_rel: float = 0.05, epsilon: float = 1e-3,
                 optimize: bool = True, coarsen: bool = False, threads: int = 0) -> Mesh:
    """triangle surface (vertices (N, 3), faces (F, 3)) → tet4 Mesh. edge: target edge length in model units; without it,
    edge_rel × the bounding-box diagonal. coarsen=True makes cells as large as quality allows. threads=0: all cores."""
    import pytetwild  # lazy: the rest of femotion works without it

    with scratch_cwd():
        v, t = pytetwild.tetrahedralize(np.asarray(vertices, float), np.asarray(faces, np.int32), edge_length_fac=edge_rel,
                                        edge_length_abs=edge, epsilon=epsilon, optimize=optimize, coarsen=coarsen,
                                        num_threads=threads, quiet=True)
    v = np.asarray(v, float)
    return Mesh(v, {"tet4": orient(v, "tet4", np.asarray(t, np.int64))})


def triangles(m) -> tuple[np.ndarray, np.ndarray]:
    """all polygon faces of a meshio.Mesh as triangles (quads and polygons fan-triangulated; other cells ignored)"""
    faces = []
    for cb in m.cells:
        if cb.type in ("triangle", "quad") or cb.type.startswith("polygon"):
            c = np.asarray(cb.data, np.int64)
            faces += [c[:, [0, k, k + 1]] for k in range(1, c.shape[1] - 1)]
    if not faces:
        raise ValueError("no surface faces found")
    return np.asarray(m.points, float)[:, :3], np.concatenate(faces)


def from_file(path: str | Path, **kw) -> Mesh:
    """read a surface model with meshio (OBJ, STL, PLY, OFF, VTK, …) and fill it with tetrahedra; kw as for from_surface"""
    import meshio

    return from_surface(*triangles(meshio.read(path)), **kw)
