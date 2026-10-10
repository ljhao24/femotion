"""gmsh meshing → Mesh (extra "mesh"). Geometry is built with gmsh's OpenCASCADE kernel (rectangles, disks, booleans,
extrusions, lofts…) or imported from STEP / IGES / BREP; element size is controlled by a callback or by size fields.

    from femotion.mesh import gmsh as G
    m = G.plane(lambda occ: occ.cut([(2, occ.addRectangle(0, 0, 0, 10, 4))], [(2, occ.addDisk(5, 2, 0, 1, 1))]),
                size=lambda x, y: 0.1 + 0.05 * abs(np.hypot(x - 5, y - 2) - 1))
    m = G.plane(build, size=0.2, quads="quasi")      # quadrilaterals aligned with the boundary (nicer on screen)
    m = G.polygon(outline, size=0.2, holes=[hole_poly])
    m = G.solid(build, fields=lambda gm: [G.circle_field(gm, ...)], hmax=0.06)   # tetrahedra
    m = G.extrude(section, (0, 0, 30), layers=60, size=0.2)  # any cross-section swept straight: hexahedra (hollow piers, girders)
    m = G.hexes(build, size=0.3)                     # box-like volumes (tapered, lofted, curved blocks): structured hexahedra
    m = G.step("saddle.step", scale=1e-3, size=0.05) # a CAD solid in mm → tetrahedra in m

Hexahedra for assemblies of axis-aligned blocks (a pier: footing + column + cap) are simpler with
femotion.mesh.structured.hex_block + weld; gmsh's structured meshing needs every volume to be box-like on its own.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .core import Mesh, orient, scratch_cwd


@contextmanager
def session(name="m", threads=1, verbose=False):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add(name)
        yield gmsh
    finally:
        gmsh.finalize()


def _size_only_from_fields(gmsh, hmax=None, hmin=None):
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    if hmax is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
    if hmin is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)


def _apply_size(gmsh, size, fields, hmax, hmin, dim):
    gm = gmsh.model.mesh
    if callable(size):
        if dim == 2:
            gm.setSizeCallback(lambda d, tag, x, y, z, lc: float(size(x, y)))
        else:
            gm.setSizeCallback(lambda d, tag, x, y, z, lc: float(size(x, y, z)))
        _size_only_from_fields(gmsh, hmax, hmin)
    elif size is not None:
        _size_only_from_fields(gmsh, size, hmin)
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(size))
    if fields is not None:
        fl = fields(gm)
        if fl:
            f = gm.field.add("Min")
            gm.field.setNumbers(f, "FieldsList", list(fl))
            gm.field.setAsBackgroundMesh(f)
        _size_only_from_fields(gmsh, hmax, hmin)


_GMSH_TYPE = {"tri3": (2, 3, 2), "quad4": (3, 4, 2), "tet4": (4, 4, 3), "hex8": (5, 8, 3)}  # gmsh type, nodes, dimension


def collect(gmsh, kind: str) -> Mesh:
    """take the cells (tri3 / quad4 / tet4 / hex8; gmsh's linear node order is VTK's) of the current gmsh model, keep only
    used nodes, unify orientation. Raises if gmsh made cells of another type in that dimension (a mesh with holes otherwise)."""
    etype, k, dim = _GMSH_TYPE[kind]
    gm = gmsh.model.mesh
    types, tags_by_type, _ = gm.getElements(dim)
    other = {gm.getElementProperties(t)[0]: len(tg) for t, tg in zip(types, tags_by_type) if t != etype}
    if other:
        raise ValueError(f"gmsh produced cells other than {kind}: {other}")
    tags, xyz, _ = gm.getNodes()
    _, conn = gm.getElementsByType(etype)
    el = conn.reshape(-1, k)
    used, inv = np.unique(el, return_inverse=True)
    lut = {int(t): i for i, t in enumerate(tags)}
    p = xyz.reshape(-1, 3)[[lut[int(u)] for u in used]]
    if dim == 2:
        p = p[:, :2]
    c = inv.reshape(-1, k)
    return Mesh(p.copy(), {kind: orient(p, kind, c)})


def _quads(gmsh, quads):
    """True: frontal quadrilaterals, recombined with full-quad Blossom (every cell a quadrilateral);
    "quasi": gmsh's quasi-structured quadrilaterals (aligned with the boundary, few irregular nodes)"""
    if quads == "quasi":
        gmsh.option.setNumber("Mesh.Algorithm", 11)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)


def plane(build, size=None, fields=None, hmax=None, hmin=None, quads=False, algorithm=6) -> Mesh:
    """a planar region: build(occ) creates surfaces in the z = 0 plane (booleans allowed); size is a constant or size(x, y);
    fields(gm) returns a list of size-field ids (the minimum is used). quads: False = triangles (gmsh algorithm
    `algorithm`), True = quadrilaterals only, "quasi" = quasi-structured quadrilaterals."""
    with session() as gmsh:
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        _apply_size(gmsh, size, fields, hmax, hmin, 2)
        if quads:
            _quads(gmsh, quads)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        with scratch_cwd():
            gmsh.model.mesh.generate(2)
        return collect(gmsh, "quad4" if quads else "tri3")


def polygon(outline, size=None, holes=(), **kw) -> Mesh:
    """triangulate a polygon outline (optionally with polygonal holes)"""
    def loop(occ, poly):
        pts = [occ.addPoint(float(x), float(y), 0) for x, y in poly]
        return occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))])

    def build(occ):
        occ.addPlaneSurface([loop(occ, outline)] + [loop(occ, h) for h in holes])
    return plane(build, size=size, **kw)


def solid(build, size=None, fields=None, hmax=None, hmin=None, algorithm3d=1, optimize=True, threads=1, verbose=False) -> Mesh:
    """a solid: build(occ) creates volumes (booleans allowed; finish with one or several non-overlapping volumes); tetrahedra"""
    import time
    with session(threads=threads) as gmsh:
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        _apply_size(gmsh, size, fields, hmax, hmin, 3)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm3d)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        t0 = time.time()
        with scratch_cwd():
            gmsh.model.mesh.generate(3)
        if verbose:
            print(f"  gmsh volume mesh {time.time() - t0:.1f}s")
        return collect(gmsh, "tet4")


def step(path, scale: float = 1.0, **kw) -> Mesh:
    """tetrahedra of the solids in a CAD file (STEP, IGES, BREP); scale converts its units (1e-3: mm → m). kw as for solid()"""
    path = Path(path).resolve()

    def build(occ):
        import gmsh
        gmsh.option.setNumber("Geometry.OCCScaling", scale)
        occ.importShapes(str(path))
    return solid(build, **kw)


def extrude(build, vector, layers: int, size=None, fields=None, hmax=None, hmin=None, quads=True) -> Mesh:
    """sweep a cross-section straight along `vector` into hexahedra: build(occ) creates the section as surfaces in the
    z = 0 plane (booleans allowed), meshed with quadrilaterals (quads as in plane()), then extruded in `layers` layers.
    Hollow piers, box girders, any prismatic member; rotate the points afterwards for other orientations."""
    with session() as gmsh:
        occ = gmsh.model.occ
        build(occ)
        occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        _apply_size(gmsh, size, fields, hmax, hmin, 2)
        _quads(gmsh, quads)
        occ.extrude(surfaces, *map(float, vector), numElements=[int(layers)], recombine=True)
        occ.synchronize()
        with scratch_cwd():
            gmsh.model.mesh.generate(3)
        return collect(gmsh, "hex8")


def hexes(build, size=None, fields=None, hmax=None, hmin=None, corner_angle: float = 2.35) -> Mesh:
    """structured hexahedra: build(occ) creates volumes that are each topologically a box (six four-sided faces: boxes,
    tapered or lofted blocks, curved blocks; fragment several into a conforming assembly). gmsh picks the node counts
    from the size and makes opposite edges match. Raises if some volume is not box-like (gmsh then fills it with other
    cells); cut it into box-like blocks first. corner_angle: face angles below this (radians) count as corners."""
    with session() as gmsh:
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        _apply_size(gmsh, size, fields, hmax, hmin, 3)
        gmsh.model.mesh.setTransfiniteAutomatic(cornerAngle=corner_angle, recombine=True)
        with scratch_cwd():
            gmsh.model.mesh.generate(3)
        return collect(gmsh, "hex8")


# ── size fields (use inside the fields(gm) callback) ─────────────────────
def math_field(gm, expr: str) -> int:
    f = gm.field.add("MathEval")
    gm.field.setString(f, "F", expr)
    return f


def circle_field(gm, cx, cy, R, h0, grade, axes=("x", "y")) -> int:
    """coarser away from a circle: h = h0 + grade·|distance to the circle|. axes picks the coordinate plane
    of the circle (for example ("x", "z") in 3D)"""
    a, b = axes
    return math_field(gm, f"{h0}+{grade}*Abs(Sqrt(({a}-({cx}))^2+({b}-({cy}))^2)-{R})")


def point_field(gm, p, h0, grade) -> int:
    names = "xyz"[: len(p)]
    d = "+".join(f"({n}-({v}))^2" for n, v in zip(names, p))
    return math_field(gm, f"{h0}+{grade}*Sqrt({d})")


def box_field(gm, h_in, h_out, xmin, xmax, ymin, ymax, zmin=-1e9, zmax=1e9, thickness=0.0) -> int:
    f = gm.field.add("Box")
    for k, v in dict(VIn=h_in, VOut=h_out, XMin=xmin, XMax=xmax, YMin=ymin, YMax=ymax, ZMin=zmin, ZMax=zmax, Thickness=thickness).items():
        gm.field.setNumber(f, k, v)
    return f


def curve_field(gm, curves, h0, grade, sampling=200) -> int:
    """coarser away from a set of curves (gmsh curve tags)"""
    d = gm.field.add("Distance")
    gm.field.setNumbers(d, "CurvesList", list(curves))
    gm.field.setNumber(d, "Sampling", sampling)
    return math_field(gm, f"{h0}+{grade}*F{d}")
