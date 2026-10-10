"""Exchange with other software through meshio (extra "io"): read meshes written by gmsh, Abaqus, Ansys, Nastran, VTK and
~30 other formats, write them back, and export fem-result/1 files for ParaView.

    from femotion import io
    mesh = io.read_mesh("part.msh")                    # → Mesh; gmsh physical groups become cell / point groups
    io.write_mesh("part.inp", mesh)                    # any format meshio writes, chosen by the suffix
    io.export_result("out/pier.json", "out/pier.vtu")  # a static result → one VTU file
    io.export_result("out/quake.json", "out/quake.pvd")  # an animated result → a VTU per frame + a ParaView collection

fem-result/1 stays the format for animation (typed arrays read in place by @femotion/core); VTU is for looking at results in
other tools. Cell types map one to one (femotion uses VTK node order): line2 = line, tri3 = triangle, tri6 = triangle6,
quad4 = quad, tet4 = tetra, hex8 = hexahedron. Reading, quadratic edges (line3, e.g. the boundary of a second-order gmsh
mesh) become line2 through their end nodes; other cell types are skipped with a warning (vertex cells silently).

VTK files (.vtu, .vtk) get 3D points and, in 2D, node and cell fields with two components padded to three so ParaView treats
them as vectors (Warp By Vector). Groups become uint8 flag arrays "group:<name>" there, because VTK has no sets (and come back as groups
when read); formats that have sets (Abaqus, MED, …) get point and cell sets. A .msh suffix means gmsh (meshio's default for
it is ANSYS); gmsh files are read with their physical groups but written without groups. Extra arrays and polyline sets are
not exported.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .mesh.core import Mesh, orient

TO_MESHIO = {"line2": "line", "tri3": "triangle", "tri6": "triangle6", "quad4": "quad", "tet4": "tetra", "hex8": "hexahedron"}
FROM_MESHIO = {v: k for k, v in TO_MESHIO.items()}
_VTK_SUFFIXES = {".vtu", ".vtk"}
_TOPO_DIM = {"vertex": 0, "line": 1, "line3": 1, "triangle": 2, "triangle6": 2, "quad": 2, "quad8": 2, "quad9": 2,
             "tetra": 3, "tetra10": 3, "hexahedron": 3, "hexahedron20": 3, "hexahedron27": 3, "wedge": 3, "pyramid": 3}


def _meshio():
    import meshio  # lazy: the rest of femotion works without meshio

    return meshio


def _pad(v: np.ndarray, dim: int, frames: bool = False) -> np.ndarray:
    """2D vectors (…, 2) → (…, 3) with z = 0"""
    v = np.asarray(v)
    if dim == 2 and v.ndim == (3 if frames else 2) and v.shape[-1] == 2:
        return np.concatenate([v, np.zeros(v.shape[:-1] + (1,), v.dtype)], axis=-1)
    return v


def to_meshio(mesh: Mesh, point_data: dict | None = None, cell_data: dict | None = None, vtk: bool = False):
    """Mesh (+ data) → meshio.Mesh. point_data: {name: (N, …)}; cell_data: {name: (M, …)} with the cell blocks concatenated in
    the order of mesh.cells (as in fem-result/1). vtk=True: 3D points, 2-component vectors padded, groups as flag arrays."""
    meshio = _meshio()
    pts = mesh.points
    pd = {k: np.asarray(v) for k, v in (point_data or {}).items()}
    cd_flat = {k: np.asarray(v) for k, v in (cell_data or {}).items()}
    if vtk:
        pts = _pad(pts, mesh.dim)
        pd = {k: _pad(v, mesh.dim) for k, v in pd.items()}
        cd_flat = {k: _pad(v, mesh.dim) for k, v in cd_flat.items()}
    kinds = list(mesh.cells)
    bounds = np.cumsum([0] + [len(mesh.cells[k]) for k in kinds])
    cd = {k: [v[bounds[b]:bounds[b + 1]] for b in range(len(kinds))] for k, v in cd_flat.items()}
    ps, cs = {}, {}
    if vtk:
        for g, idx in mesh.point_groups.items():
            flag = np.zeros(mesh.n_points, np.uint8)
            flag[idx] = 1
            pd[f"group:{g}"] = flag
        for g, d in mesh.cell_groups.items():
            flags = []
            for k in kinds:
                f = np.zeros(len(mesh.cells[k]), np.uint8)
                if k in d:
                    f[d[k]] = 1
                flags.append(f)
            cd[f"group:{g}"] = flags
    else:
        ps = {g: np.asarray(idx) for g, idx in mesh.point_groups.items()}
        cs = {g: [np.asarray(d[k]) if k in d else np.zeros(0, np.int64) for k in kinds] for g, d in mesh.cell_groups.items()}
    return meshio.Mesh(pts, [(TO_MESHIO[k], mesh.cells[k]) for k in kinds], point_data=pd, cell_data=cd,
                       point_sets=ps, cell_sets=cs)


def from_meshio(m, dim: int | None = None) -> Mesh:
    """meshio.Mesh → Mesh. Blocks of the same type are merged. Point sets and cell sets become groups, and every cell set also
    becomes a point group of all its nodes (unless a point set has that name), so gmsh physical groups can be used directly for
    boundary conditions; meshio's own "gmsh:*" sets are dropped. "group:<name>" flag arrays (from write_mesh to VTK) become
    groups again. dim=None: 2 when the points are 2D, or all z are zero and there are no solid cells; else 3. Cells are
    re-oriented to femotion's convention (counter-clockwise in 2D, positive volume in 3D). Other data arrays are not
    converted; read them from the meshio object."""
    pts = np.asarray(m.points, float)
    blocks: dict[str, list[np.ndarray]] = {}
    where: list[tuple[str, int] | None] = []  # meshio block → (our type, offset within our merged block)
    skipped = set()
    for cb in m.cells:
        k = "line2" if cb.type == "line3" else FROM_MESHIO.get(cb.type)
        if k is None:
            where.append(None)
            if cb.type != "vertex":
                skipped.add(cb.type)
            continue
        where.append((k, sum(len(c) for c in blocks.get(k, []))))
        blocks.setdefault(k, []).append(np.asarray(cb.data, np.int64)[:, :2] if cb.type == "line3" else np.asarray(cb.data, np.int64))
    if skipped:
        warnings.warn(f"skipped cell types femotion does not support: {', '.join(sorted(skipped))}", stacklevel=2)
    cells = {k: np.concatenate(v) for k, v in blocks.items()}
    if dim is None:
        flat = pts.shape[1] == 2 or (not np.any(pts[:, 2]) and not ({"tet4", "hex8"} & set(cells)))
        dim = 2 if flat else 3
    pts = pts[:, :dim] if pts.shape[1] >= dim else np.column_stack([pts, np.zeros((len(pts), dim - pts.shape[1]))])
    for k in cells:
        if k in ("tet4", "hex8") or (dim == 2 and k != "line2"):
            cells[k] = orient(pts, k, cells[k])
    pg = {g: np.asarray(v, np.int64) for g, v in (m.point_sets or {}).items()}
    pg.update({key[6:]: np.flatnonzero(v) for key, v in (m.point_data or {}).items() if key.startswith("group:")})
    ours = {key[6:]: [np.flatnonzero(v) for v in per_block] for key, per_block in (m.cell_data or {}).items() if key.startswith("group:")}
    cg: dict[str, dict[str, np.ndarray]] = {}
    for g, per_block in list(_cell_sets(m).items()) + list(ours.items()):
        acc: dict[str, list[np.ndarray]] = {}
        nodes = []
        for b, ids in enumerate(per_block):
            if ids is None or len(ids) == 0:
                continue
            nodes.append(np.asarray(m.cells[b].data)[ids].ravel())
            if where[b] is not None:
                k, off = where[b]
                acc.setdefault(k, []).append(np.asarray(ids, np.int64) + off)
        if acc:
            cg[g] = {k: np.concatenate(v) for k, v in acc.items()}
        if nodes and g not in pg and g not in ours:
            pg[g] = np.unique(np.concatenate(nodes)).astype(np.int64)
    return Mesh(pts, cells, pg, cg)


def _cell_sets(m) -> dict[str, list]:
    """named cell sets of a meshio.Mesh without meshio's "gmsh:*" bookkeeping; gmsh 2.2 files carry physical names only as
    field_data {name: [tag, dim]} + cell_data "gmsh:physical", which are turned into sets here"""
    sets = {g: s for g, s in (m.cell_sets or {}).items() if not g.startswith("gmsh:")}
    phys = (m.cell_data or {}).get("gmsh:physical")
    if sets or phys is None:
        return sets
    for name, (tag, d) in (m.field_data or {}).items():
        sets[name] = [np.flatnonzero(np.asarray(t) == tag) if _TOPO_DIM.get(cb.type) == d else None for cb, t in zip(m.cells, phys)]
    return sets


def read_mesh(path: str | Path, dim: int | None = None, **kw) -> Mesh:
    """read any mesh file meshio understands (format from the suffix, or file_format="…")"""
    if Path(path).suffix.lower() == ".msh":
        kw.setdefault("file_format", "gmsh")
    return from_meshio(_meshio().read(path, **kw), dim)


def write_mesh(path: str | Path, mesh: Mesh, point_data: dict | None = None, cell_data: dict | None = None, **kw) -> Path:
    """write a mesh (+ data, as in to_meshio) in the format given by the suffix (or file_format="…"). .msh is written as gmsh
    2.2, which takes mixed cell types; groups are not written to gmsh files (use .inp or .vtu to keep them)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".msh":
        kw.setdefault("file_format", "gmsh22")
    fmt = kw.get("file_format", "")
    m = to_meshio(mesh, point_data, cell_data, vtk=path.suffix.lower() in _VTK_SUFFIXES or fmt.startswith("vtk"))
    if fmt.startswith("gmsh"):
        m.cell_sets = {}
        for tag in ("gmsh:physical", "gmsh:geometrical"):
            m.cell_data.setdefault(tag, [np.zeros(len(c.data), np.int32) for c in m.cells])
    _meshio().write(path, m, **kw)
    return path


def _load(src) -> tuple[Mesh, dict, list[dict], dict]:
    """fem-result/1 file or Result → (mesh, meta, field descriptions, {name: values})"""
    from .result import Result, read

    if isinstance(src, Result):
        arrs = {key: a for key, a, _ in src._arrays}
        return src.mesh, src.meta, src.fields, {f["name"]: arrs[f["key"]] for f in src.fields}
    r = read(src)
    return r["mesh"], r["meta"], r["desc"]["fields"], r["fields"]


def export_result(src, path: str | Path, frame: int | None = None, step: int = 1) -> list[Path]:
    """fem-result/1 (a file path or a Result) → files for other tools. Returns the written paths.
    - no per-frame fields, or a single frame picked with `frame`: one file in any meshio format (suffix of `path`).
    - per-frame fields: `path` must end in .pvd; every `step`-th frame goes to <stem>/<stem>_NNNN.vtu next to it, with static
      fields repeated in each, and the .pvd lists them with their times (meta "t", else frame / meta "fps", else the frame
      number). Open the .pvd in ParaView and press play."""
    mesh, meta, fields, values = _load(src)
    path = Path(path)
    animated = any(f.get("frames") for f in fields)
    nf = max((len(values[f["name"]]) for f in fields if f.get("frames")), default=0)

    def data(k: int | None):
        pd, cd = {}, {}
        for f in fields:
            v = values[f["name"]]
            if f.get("frames"):
                v = v[k]
            (pd if f["loc"] == "node" else cd)[f["name"]] = v
        return pd, cd

    if not animated or frame is not None:
        if animated and not 0 <= frame < nf:
            raise IndexError(f"frame {frame} out of range (0 … {nf - 1})")
        return [write_mesh(path, mesh, *data(frame))]
    if path.suffix.lower() != ".pvd":
        raise ValueError(f"{path.name}: this result has {nf} frames; export it to a .pvd collection or pick one with frame=k")
    t = meta.get("t")
    fps = meta.get("fps")
    folder = path.parent / path.stem
    out = []
    rows = []
    for k in range(0, nf, step):
        f = write_mesh(folder / f"{path.stem}_{k:04d}.vtu", mesh, *data(k))
        out.append(f)
        time = float(t[k]) if t is not None and len(t) == nf else (k / fps if fps else float(k))
        rows.append(f'    <DataSet timestep="{time:.9g}" group="" part="0" file={quoteattr(f.relative_to(path.parent).as_posix())}/>')
    path.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                    "  <Collection>\n" + "\n".join(rows) + "\n  </Collection>\n</VTKFile>\n")
    return [path] + out
