"""Topology optimisation, result-file round trips and assembly patterns."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from femotion import topopt
from femotion.mesh import structured
from femotion.post import assembly
from femotion.result import FORMAT, Result, read


def _mbb(nelx=60, nely=20):
    F = np.zeros(2 * (nelx + 1) * (nely + 1))
    F[2 * topopt.node(nelx // 2, 0, nely) + 1] = -1
    L, R = topopt.node(0, nely, nely), topopt.node(nelx, nely, nely)
    return F, np.array([2 * L, 2 * L + 1, 2 * R, 2 * R + 1])


def test_topopt_compliance_and_volume():
    F, fixed = _mbb()
    r = topopt.run(60, 20, F, fixed, volfrac=0.3, maxit=40, verbose=False)
    assert r["comp"][-1] < 0.5 * r["comp"][0]
    assert r["x"].mean() == pytest.approx(0.3, rel=0.01)
    assert len(r["hist"]) == len(r["comp"])  # every iteration recorded
    assert structured.grid_for_density(60, 20).n_cells() == r["x"].size


def test_topopt_passive_rows_are_top_rows():
    F, fixed = _mbb()
    pas = np.zeros((60, 20), bool)
    pas[:, :2] = True
    r = topopt.run(60, 20, F, fixed, volfrac=0.3, maxit=5, passive=pas, verbose=False)
    assert (r["x"][:, :2] == 1).all() and r["x"][:2, 2:].mean() < 0.99


def test_topopt_compliance_matches_continuum():
    """same grid and densities, SIMP moduli in femotion.continuum: same compliance as topopt itself"""
    pytest.importorskip("skfem")
    from femotion.continuum import Elastic

    nelx, nely = 60, 20
    F, fixed = _mbb(nelx, nely)
    r = topopt.run(nelx, nely, F, fixed, volfrac=0.3, maxit=40, verbose=False)
    s = Elastic(structured.grid_for_density(nelx, nely), E=1.0, nu=0.3, E_cell=1e-9 + r["x"].ravel() ** 3 * (1 - 1e-9))
    s.f[:] = 0
    s.point_load((nelx // 2, nely), (0, -1))
    s.fix_point((0, 0))
    s.fix_point((nelx, 0))
    s.solve()
    assert s.compliance() == pytest.approx(r["comp"][-1], rel=0.05)


def test_result_round_trip():
    pytest.importorskip("skfem")
    from femotion.continuum import Elastic
    from femotion.post import stress

    m = structured.rect(2, 1, 4, 2, "tri3")
    s = Elastic(m, E=10.0, nu=0.3, order=2)
    s.traction(lambda x: np.isclose(x[0], 2), (1.0, 0.0))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    rm = s.result_mesh()
    t6 = rm.cells["tri6"]
    p = rm.points
    assert np.allclose(p[t6[:, 3]], 0.5 * (p[t6[:, 0]] + p[t6[:, 1]])) and np.allclose(p[t6[:, 5]], 0.5 * (p[t6[:, 2]] + p[t6[:, 0]]))
    sa = (p[t6[:, 1], 0] - p[t6[:, 0], 0]) * (p[t6[:, 2], 1] - p[t6[:, 0], 1]) - (p[t6[:, 1], 1] - p[t6[:, 0], 1]) * (p[t6[:, 2], 0] - p[t6[:, 0], 0])
    assert (sa > 0).all()  # counter-clockwise after scikit-fem's vertex sorting
    assert len(rm.boundary_loops()) == 1
    u = s.nodal_displacement()
    frames = np.stack([u * k for k in np.linspace(0, 1, 5)])
    r = Result(rm, meta=dict(note="test")).field("u", u, unit="m").field("vm", stress.von_mises(s.nodal_stress()))
    r.field("s_cell", s.cell_stress()[:, 0], loc="cell").field("u_t", frames, frames=True).extra("pattern", assembly.stiffness_pattern(m, G=16), "i32")
    r.polylines("lines", [np.array([[0, 0, 1], [1, 1, 2]]), np.array([[2, 0, 0], [2, 1, 1], [1, 1, 1]])])
    with tempfile.TemporaryDirectory() as d:
        r.write(d, "t", verbose=False)
        back = read(Path(d) / "t.json")
    assert back["desc"]["format"] == FORMAT
    assert np.allclose(back["mesh"].points, rm.points, atol=1e-6) and np.array_equal(back["mesh"].cells["tri6"], t6)
    assert np.allclose(back["fields"]["u"], u, rtol=1e-6) and back["fields"]["u_t"].shape == frames.shape
    assert list(back["arrays"]["lines/lines/len"]) == [2, 3]
    pat = back["arrays"]["pattern"]
    assert np.array_equal(pat, pat.T) and (np.diag(pat) >= 0).all()


def test_field_size_checked():
    m = structured.rect(1, 1, 2, 2, "quad4")
    with pytest.raises(ValueError):
        Result(m).field("bad", np.zeros(m.n_points + 1))


def test_golden_file_is_reproduced_byte_for_byte():
    """the writer must keep producing exactly tests/data/golden_v1.* (format stability; the JS readers decode the same file)"""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from golden import DATA, golden

    with tempfile.TemporaryDirectory() as d:
        golden().write(d, "golden_v1", verbose=False)
        for ext in (".json", ".bin"):
            assert (Path(d) / f"golden_v1{ext}").read_bytes() == (DATA / f"golden_v1{ext}").read_bytes()
    back = read(DATA / "golden_v1.json")
    assert back["fields"]["u_t"].shape == (3, 6, 2) and back["arrays"]["mask"].dtype == np.uint8
    assert list(back["mesh"].cells) == ["tri3", "quad4"] and list(back["mesh"].cell_groups["right"]["quad4"]) == [0]
