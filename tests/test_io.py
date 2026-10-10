"""meshio exchange: lossless round trips, gmsh physical groups (MSH 4.1 and 2.2), result export for ParaView."""
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

pytest.importorskip("meshio")
from femotion import io  # noqa: E402
from femotion.mesh import Mesh, structured  # noqa: E402
from femotion.result import read  # noqa: E402

GOLDEN = Path(__file__).with_name("data") / "golden_v1.json"

MESHES = {
    "mixed2d": Mesh(np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float),
                    {"tri3": [[0, 1, 4], [0, 4, 3]], "quad4": [[1, 2, 5, 4]], "line2": [[0, 3]]},
                    {"left": np.array([0, 3])}, {"right": {"quad4": np.array([0])}, "edge": {"line2": np.array([0])}}),
    "tri6": Mesh(np.array([[0, 0], [2, 0], [0, 2], [1, 0], [1, 1], [0, 1]], float), {"tri6": [[0, 1, 2, 3, 4, 5]]}),
    "tet4": Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float), {"tet4": [[0, 1, 2, 3]]}),
    "hex8": structured.box(2.0, 1.0, 1.0, 2, 1, 1),
}


def same_cells(a: Mesh, b: Mesh) -> bool:
    return (a.dim == b.dim and np.allclose(a.points, b.points) and list(a.cells) == list(b.cells)
            and all(np.array_equal(a.cells[k], b.cells[k]) for k in a.cells))


@pytest.mark.parametrize("name", MESHES)
@pytest.mark.parametrize("suffix", [".vtu", ".inp", ".msh"])
def test_round_trip(name, suffix, tmp_path):
    m = MESHES[name]
    b = io.read_mesh(io.write_mesh(tmp_path / f"m{suffix}", m))
    assert same_cells(m, b)  # node order of every cell type survives (femotion uses VTK order)
    if suffix == ".vtu":  # groups travel as flag arrays and come back unchanged
        assert {g: v.tolist() for g, v in b.point_groups.items()} == {g: v.tolist() for g, v in m.point_groups.items()}
        assert {g: {k: v.tolist() for k, v in d.items()} for g, d in b.cell_groups.items()} == \
               {g: {k: v.tolist() for k, v in d.items()} for g, d in m.cell_groups.items()}


@pytest.mark.parametrize("version", [4.1, 2.2])
def test_gmsh_physical_groups(version, tmp_path):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        s = occ.cut([(2, occ.addRectangle(0, 0, 0, 4, 2))], [(2, occ.addDisk(2, 1, 0, 0.5, 0.5))])[0][0][1]
        occ.synchronize()
        left = [c for _, c in gmsh.model.getBoundary([(2, s)], oriented=False) if abs(occ.getCenterOfMass(1, c)[0]) < 1e-9]
        gmsh.model.addPhysicalGroup(2, [s], name="plate")
        gmsh.model.addPhysicalGroup(1, left, name="left")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(tmp_path / "plate.msh"))
    finally:
        gmsh.finalize()
    m = io.read_mesh(tmp_path / "plate.msh")
    assert m.dim == 2 and set(m.cells) == {"tri6", "line2"}  # quadratic boundary edges (line3) kept as line2
    assert set(m.cell_groups) == {"plate", "left"} and len(m.cell_groups["plate"]["tri6"]) == len(m.cells["tri6"])
    assert np.all(m.points[m.point_groups["left"], 0] == 0) and len(m.point_groups["left"]) == 2 * len(m.cell_groups["left"]["line2"]) + 1
    a, b, c = (m.points[m.cells["tri6"][:, i]] for i in range(3))
    assert np.all((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) > 0)  # counter-clockwise


def test_export_static_frame_and_series(tmp_path):
    g = read(GOLDEN)
    meshio = pytest.importorskip("meshio")
    with pytest.raises(ValueError):
        io.export_result(GOLDEN, tmp_path / "g.vtu")  # animated: needs .pvd or a frame
    (one,) = io.export_result(GOLDEN, tmp_path / "g1.vtu", frame=1)
    v = meshio.read(one)
    assert v.points.shape == (6, 3) and v.point_data["u"].shape == (6, 3)  # 2D vectors padded for ParaView
    assert np.allclose(v.point_data["u_t"][:, :2], g["fields"]["u_t"][1]) and np.all(v.point_data["u_t"][:, 2] == 0)
    assert np.array_equal(np.concatenate(v.cell_data["rank"]), g["fields"]["rank"])
    paths = io.export_result(GOLDEN, tmp_path / "g.pvd")
    assert len(paths) == 4
    rows = ElementTree.parse(paths[0]).getroot().findall("./Collection/DataSet")
    assert [float(r.get("timestep")) for r in rows] == pytest.approx([0, 1 / 30, 2 / 30])  # meta fps = 30
    for k, r in enumerate(rows):
        f = meshio.read(tmp_path / r.get("file"))
        assert np.allclose(f.point_data["u_t"][:, :2], g["fields"]["u_t"][k])
        assert np.allclose(f.point_data["vm"], g["fields"]["vm"])  # static fields repeated in every frame
