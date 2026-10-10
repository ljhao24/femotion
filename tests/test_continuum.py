"""Continuum elasticity against closed-form solutions."""
import numpy as np
import pytest

pytest.importorskip("skfem")
from femotion.continuum import Elastic  # noqa: E402
from femotion.mesh import structured  # noqa: E402
from femotion.post import lines, stress  # noqa: E402


@pytest.mark.parametrize("kind,order", [("tri3", 1), ("tri3", 2), ("quad4", 1)])
def test_patch_uniaxial(kind, order):
    """uniaxial tension: σxx = 1 everywhere, end displacement L/E, reaction −1"""
    m = structured.rect(4, 1, 8, 3, kind)
    s = Elastic(m, E=100.0, nu=0.3, order=order)
    s.traction(lambda x: np.isclose(x[0], 4), (1.0, 0.0))
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix_point((0, 0), comps=(1,))
    s.solve()
    S, u, rm = s.nodal_stress(), s.nodal_displacement(), s.result_mesh()
    assert np.abs(S[:, 0] - 1).max() < 1e-8
    assert u[np.isclose(rm.points[:, 0], 4), 0].mean() == pytest.approx(0.04, rel=1e-8)
    assert s.reactions()[0] == pytest.approx(-1.0, rel=1e-8)


def test_kirsch_hole():
    """quarter plate with a hole in tension: Kt → 3 at the hole edge; far-field tension lines horizontal"""
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    R, L = 1.0, 10.0
    m = G.plane(lambda occ: occ.cut([(2, occ.addRectangle(0, 0, 0, L, L))], [(2, occ.addDisk(0, 0, 0, R, R))]),
                size=lambda x, y: min(0.6, 0.04 + 0.12 * (np.hypot(x, y) - R)))
    s = Elastic(m, E=1.0, nu=0.3, order=2)
    s.traction(lambda x: np.isclose(x[0], L), (1.0, 0.0))
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s.solve()
    rm, S = s.result_mesh(), s.nodal_stress()
    assert S[rm.nearest((0, R)), 0] == pytest.approx(3.0, rel=0.03)
    vm = stress.von_mises(S)
    assert np.hypot(*rm.points[int(np.argmax(vm))]) < R + 0.05
    tl = lines.trajectories(rm, S, [(L * 0.9, y) for y in (3.0, 6.0)], which=1, step=0.1)
    ang = np.degrees(np.arctan2(*np.diff(tl[0][-5:, :2], axis=0)[-1][::-1]))
    assert min(abs(ang), abs(abs(ang) - 180)) < 3


def test_plane_cantilever_timoshenko():
    """end-loaded plane-stress cantilever: δ = PL³/3EI + PL/(κGA), κ = 5/6"""
    L, H, P, E, nu = 10.0, 1.0, 1.0, 1000.0, 0.3
    s = Elastic(structured.rect(L, H, 60, 6, "tri3"), E=E, nu=nu, order=2)
    s.traction(lambda x: np.isclose(x[0], L), lambda x: (0 * x[0], -P / H + 0 * x[0]))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    tip = s.result_mesh().nearest((L, H / 2))
    I, G_ = H**3 / 12, E / (2 * (1 + nu))
    assert -s.nodal_displacement()[tip, 1] == pytest.approx(P * L**3 / (3 * E * I) + P * L / (5 / 6 * G_ * H), rel=0.02)


def test_solid_bar_direct_vs_amg():
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    pytest.importorskip("pyamg")
    m = G.solid(lambda occ: occ.addBox(0, 0, 0, 2, 0.4, 0.4), size=0.08)
    res = {}
    for method in ("direct", "amg"):
        s = Elastic(m, E=200.0, nu=0.3)
        s.traction(lambda x: np.isclose(x[0], 2), (1.0, 0, 0))
        s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
        s.fix_point((0, 0, 0), comps=(1, 2))
        s.fix_point((0, 0.4, 0), comps=(2,))
        s.solve(method=method)
        res[method] = s.nodal_displacement()
    assert s.nodal_stress()[:, 0].mean() == pytest.approx(1.0, rel=1e-6)
    assert res["amg"][np.isclose(m.points[:, 0], 2), 0].mean() == pytest.approx(2 / 200, rel=1e-6)
    assert np.abs(res["amg"] - res["direct"]).max() < 1e-10


@pytest.mark.parametrize("order", [1, 2])
def test_hex_patch(order):
    m = structured.box(2.0, 1.0, 1.0, 4, 2, 2)
    s = Elastic(m, E=100.0, nu=0.3, order=order)
    s.traction(lambda x: np.isclose(x[0], 2.0), (1.0, 0, 0))
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix_point((0, 0, 0), comps=(1, 2))
    s.fix_point((0, 1, 0), comps=(2,))
    s.solve()
    S, u, rm = s.nodal_stress(), s.nodal_displacement(), s.result_mesh()
    assert np.abs(S[:, 0] - 1).max() < 1e-8
    assert u[np.isclose(rm.points[:, 0], 2.0), 0].mean() == pytest.approx(0.02, rel=1e-8)


def test_hex_q2_cantilever():
    L, b, P, E, nu = 8.0, 1.0, 1.0, 1000.0, 0.3
    s = Elastic(structured.box(L, b, b, 16, 2, 2), E=E, nu=nu, order=2)
    s.traction(lambda x: np.isclose(x[0], L), (0, 0, -P / b**2))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    tip = np.isclose(s.result_mesh().points[:, 0], L)
    I, G_ = b**4 / 12, E / (2 * (1 + nu))
    assert -s.nodal_displacement()[tip, 2].mean() == pytest.approx(P * L**3 / (3 * E * I) + P * L / (5 / 6 * G_ * b * b), rel=0.02)


def test_welded_blocks_patch():
    """tension along y through two welded blocks: σyy = 1 everywhere only if the seam is welded correctly"""
    xs = structured.graded([0, 1, 2], 0.5)
    ys = structured.graded([0, 1], 0.5)
    a = structured.hex_block(xs, ys, 0.0, 1.0, 2)
    b = structured.hex_block(xs[xs >= 1 - 1e-9], ys, 1.0, lambda x, y: 2 + 0.5 * (x - 1), 3)
    s = Elastic(structured.weld([a, b]), E=100.0, nu=0.3, order=1)
    s.traction(lambda x: np.isclose(x[1], 1.0), (0, 1.0, 0))
    s.fix(lambda x: np.isclose(x[1], 0.0), comps=(1,))
    s.fix_point((0, 0, 0), comps=(0, 2))
    s.fix_point((2, 0, 0), comps=(2,))
    s.solve()
    assert np.abs(s.nodal_stress()[:, 1] - 1).max() < 1e-8


def test_quadratic_solid_amg_matches_direct():
    """quadratic tetrahedra, whole nodes fixed: AMG runs on 3 × 3 node blocks and agrees with the direct solver"""
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    pytest.importorskip("pyamg")
    m = G.solid(lambda occ: occ.addBox(0, 0, 0, 2, 0.4, 0.4), size=0.1)
    res = {}
    for method in ("direct", "amg"):
        s = Elastic(m, E=200.0, nu=0.3, order=2)
        s.body_force((0, 0, -1.0))
        s.fix(lambda x: np.isclose(x[0], 0))
        s.solve(method=method)
        res[method] = s.nodal_displacement()
    assert np.abs(res["amg"] - res["direct"]).max() < 1e-8 * np.abs(res["direct"]).max()


@pytest.mark.filterwarnings("ignore")
def test_unsupported_model_raises():
    G = pytest.importorskip("femotion.mesh.gmsh")
    pytest.importorskip("gmsh")
    s = Elastic(G.solid(lambda occ: occ.addBox(0, 0, 0, 1, 1, 1), size=0.5), E=1.0, nu=0.3)
    s.body_force((0, 0, -1.0))
    with pytest.raises(RuntimeError, match="did not converge"):
        s.solve(method="direct")  # no supports: singular
