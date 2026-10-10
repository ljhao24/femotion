"""Mesh layer: structured grids, gmsh meshes, surfaces, volumes, block welding, scikit-fem round trips."""
import numpy as np
import pytest

from femotion.mesh import Mesh, structured


def test_rect_tri_boundary_and_area():
    m = structured.rect(4, 1, 8, 2, "tri3")
    assert len(m.boundary_edges()) == 20  # 2 (nx + ny)
    assert m.measure().sum() == pytest.approx(4.0, rel=1e-12)


def test_gmsh_plate_with_hole():
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    h = G.plane(lambda occ: occ.cut([(2, occ.addRectangle(0, 0, 0, 4, 2))], [(2, occ.addDisk(2, 1, 0, 0.5, 0.5))]), size=0.15)
    assert len(h.boundary_loops()) == 2  # outline + hole
    assert h.measure().sum() == pytest.approx(8 - np.pi * 0.25, rel=5e-3)


def test_gmsh_tet_surface_points_outwards():
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    s = G.solid(lambda occ: occ.addBox(0, 0, 0, 1, 0.5, 0.25), size=0.1)
    f, _ = s.surface()
    p = s.points
    vol = np.einsum("ij,ij->i", p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]])).sum() / 6  # divergence theorem
    assert vol == pytest.approx(0.125, rel=1e-9)
    assert s.measure().sum() == pytest.approx(0.125, rel=1e-9)


def test_hex_box_volume_surface():
    m = structured.box(2.0, 1.0, 1.0, 4, 2, 2)
    assert m.measure().sum() == pytest.approx(2.0, rel=1e-12)
    f, _ = m.surface()
    p = m.points
    tri = np.r_[f[:, [0, 1, 2]], f[:, [0, 2, 3]]]
    vol = np.einsum("ij,ij->i", p[tri[:, 0]], np.cross(p[tri[:, 1]], p[tri[:, 2]])).sum() / 6
    assert vol == pytest.approx(2.0, rel=1e-12)
    assert len(f) == 2 * (4 * 2 + 2 * 2 + 2 * 4)


def test_skfem_round_trip_keeps_cells():
    pytest.importorskip("skfem")
    m = structured.box(2.0, 1.0, 1.0, 4, 2, 2)
    assert np.array_equal(Mesh.from_skfem(m.to_skfem()).cells["hex8"], m.cells["hex8"])
    q = structured.rect(3, 1, 6, 2, "quad4")
    assert np.array_equal(Mesh.from_skfem(q.to_skfem()).cells["quad4"], q.cells["quad4"])


def test_weld_blocks():
    xs = structured.graded([0, 1, 2], 0.5)
    ys = structured.graded([0, 1], 0.5)
    a = structured.hex_block(xs, ys, 0.0, 1.0, 2)
    b = structured.hex_block(xs[xs >= 1 - 1e-9], ys, 1.0, lambda x, y: 2 + 0.5 * (x - 1), 3)
    m = structured.weld([a, b])
    assert m.n_points == a.n_points + b.n_points - 9  # 9 seam nodes merged
    assert m.measure().sum() == pytest.approx(3.25, rel=1e-12)
    f, _ = m.surface()
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 3]], f[:, [3, 0]]], axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all()  # closed surface: every edge shared by two faces (no hanging nodes)


def _hex_volume_by_surface(m):
    f, _ = m.surface()
    p = m.points
    tri = np.r_[f[:, [0, 1, 2]], f[:, [0, 2, 3]]]
    return np.einsum("ij,ij->i", p[tri[:, 0]], np.cross(p[tri[:, 1]], p[tri[:, 2]])).sum() / 6


@pytest.mark.parametrize("quads", [True, "quasi"])
def test_gmsh_quadrilaterals_only(quads):
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    h = G.plane(lambda occ: occ.cut([(2, occ.addRectangle(0, 0, 0, 4, 2))], [(2, occ.addDisk(2, 1, 0, 0.5, 0.5))]), size=0.15, quads=quads)
    assert list(h.cells) == ["quad4"] and len(h.boundary_loops()) == 2  # every cell a quadrilateral, no gaps
    assert h.measure().sum() == pytest.approx(8 - np.pi * 0.25, rel=5e-3)


def test_gmsh_extrude_hollow_section_into_hexahedra():
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    m = G.extrude(lambda occ: occ.cut([(2, occ.addRectangle(-3, -1.5, 0, 6, 3))], [(2, occ.addRectangle(-2.5, -1, 0, 5, 2))]),
                  (0, 0, 12), layers=24, size=0.25)
    assert list(m.cells) == ["hex8"]
    assert m.measure().sum() == pytest.approx(8 * 12, rel=1e-12) and _hex_volume_by_surface(m) == pytest.approx(96, rel=1e-12)


def test_gmsh_structured_hexahedra_of_a_tapered_block():
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")

    def loft(occ):
        loops = []
        for (a, b), z in (((2.0, 1.0), 0.0), ((1.0, 0.6), 10.0)):
            pts = [occ.addPoint(x, y, z) for x, y in ((-a, -b), (a, -b), (a, b), (-a, b))]
            loops.append(occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]))
        occ.addThruSections(loops, makeSolid=True, makeRuled=True)

    m = G.hexes(loft, size=0.5)
    assert list(m.cells) == ["hex8"]
    assert m.measure().sum() == pytest.approx(10 / 6 * (8 + 4 * 3 * 1.6 + 2 * 1.2), rel=1e-9)  # prismatoid formula
    with pytest.raises(ValueError):  # a fragmented assembly is not box-like volume by volume
        G.hexes(lambda occ: occ.fragment([(3, occ.addBox(-3, -3, 0, 6, 6, 2))], [(3, occ.addBox(-1, -1, 2, 2, 2, 8))]), size=0.5)


def test_gmsh_step_import_with_units(tmp_path):
    G = pytest.importorskip("femotion.mesh.gmsh")
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 1000, 500, 250)  # mm
        gmsh.model.occ.synchronize()
        gmsh.write(str(tmp_path / "block.step"))
    finally:
        gmsh.finalize()
    m = G.step(tmp_path / "block.step", scale=1e-3, size=0.1)
    assert m.measure().sum() == pytest.approx(0.125, rel=1e-9) and m.bbox()[1] == pytest.approx([1.0, 0.5, 0.25])


def _box_surface(x0, y0, z0, x1, y1, z1):
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6],
                  [1, 3, 5], [3, 7, 5]])
    return v, f


def test_tetwild_fills_a_messy_surface():
    """two overlapping boxes simply concatenated (self-intersecting) with a triangle missing: one solid, the union's volume"""
    pytest.importorskip("pytetwild")
    from femotion.mesh import tetwild

    va, fa = _box_surface(0, 0, 0, 2, 1, 1)
    vb, fb = _box_surface(1, 0.2, 0.2, 3, 0.8, 1.4)
    m = tetwild.from_surface(np.r_[va, vb], np.r_[fa[1:], fb + 8], edge=0.1)
    vol = m.measure()
    assert list(m.cells) == ["tet4"] and vol.min() > 0
    assert vol.sum() == pytest.approx(2 + 2 * 0.6 * 1.2 - 0.6 * 0.8, rel=2e-3)
    f, _ = m.surface()
    p = m.points
    assert np.einsum("ij,ij->i", p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]])).sum() / 6 == pytest.approx(vol.sum(), rel=1e-9)


def test_tetwild_from_an_obj_file(tmp_path):
    pytest.importorskip("pytetwild")
    meshio = pytest.importorskip("meshio")
    from femotion.mesh import tetwild

    v, f = _box_surface(0, 0, 0, 1, 1, 1)
    quads = np.array([[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]])  # Blender-style quads
    meshio.write(tmp_path / "cube.obj", meshio.Mesh(v, [("quad", quads)]))
    m = tetwild.from_file(tmp_path / "cube.obj", edge=0.1)
    assert m.measure().sum() == pytest.approx(1.0, rel=1e-4)  # within fTetWild's envelope
