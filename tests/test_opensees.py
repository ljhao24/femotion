"""OpenSees back-end: cross-validation with the linear solver and nonlinear checks against closed-form solutions."""
import tempfile

import numpy as np
import pytest

pytest.importorskip("openseespy")
from femotion import dynamics  # noqa: E402
from femotion.frame import Model, Section, Solver, lumped_masses  # noqa: E402
from femotion.frame.check import crosscheck  # noqa: E402
from femotion.frame.opensees import Fiber, OpsFrame, to_result  # noqa: E402
from femotion.models import cable_stayed as cs  # noqa: E402
from femotion.result import read  # noqa: E402


@pytest.fixture(scope="module")
def bridge():
    br = cs.build()
    sv = Solver(br.m)
    G = sv.gravity()
    eps, _ = cs.optimize_pretension(br, sv, sv.solve(G), verbose=False)
    return br, sv, G, eps


def test_same_masses(bridge):
    br, sv, _, _ = bridge
    assert np.array_equal(lumped_masses(br.m), sv.M) and np.array_equal(OpsFrame(br.m).M, sv.M)


def test_cable_stayed_matches_linear_solver(bridge):
    br, sv, G, eps = bridge
    U = sv.solve(G + sv.pretension(eps))
    f = OpsFrame(br.m, eps0=eps)
    assert np.abs(f.static(G) - U).max() < 1e-8
    N = sv.truss_forces(U, eps)
    assert np.abs(f.truss_forces() - N).max() / np.abs(N).max() < 1e-8
    B = sv.beam_end_forces(U)
    assert np.abs(f.beam_end_forces() - B).max() / np.abs(B).max() < 1e-8  # including signs
    fr, _ = sv.modes(10)
    fo, _ = OpsFrame(br.m).modes(10)
    assert np.max(np.abs(fo - fr) / fr) < 1e-6


def test_linear_cases(bridge):
    br, sv, _, _ = bridge
    Fs = [dynamics.moving_point_loads(br.m.ndof, br.deck_nodes, np.asarray(br.deck_x), [x], 300.0) for x in np.linspace(-600, 600, 12)]
    Ul = OpsFrame(br.m).linear_cases(Fs)
    assert max(np.abs(Ul[k] - sv.solve(F)).max() for k, F in enumerate(Fs)) < 1e-8


def test_crosscheck(bridge):
    br, _, _, eps = bridge
    d = crosscheck(br.m, eps0=eps, verbose=False)
    assert len(d) == 4 and max(d.values()) < 1e-6


def test_stale_opsframe_raises():
    m = Model()
    a, b = m.node((0, 0, 0)), m.node((1, 0, 0))
    m.beam(a, b, Section(E=2e8, A=1.0, Iy=1.0, Iz=1.0, J=1.0))
    m.fix(a)
    old = OpsFrame(m)
    OpsFrame(m)
    with pytest.raises(RuntimeError, match="stale"):
        old.U()


def test_euler_buckling():
    """pinned column with an L/1000 bow, corotational: P = 0.9 Pcr at δ = 10 δ0 (Southwell);
    P/Pcr ≈ 1 − δ0/δ + π²/8 (δ/L)² at δ = L/10 (imperfect elastica)"""
    L, n, E, I = 10.0, 20, 2e8, 1e-5
    Pcr = np.pi**2 * E * I / L**2
    d0 = L / 1000
    m = Model()
    ns = [m.node((L * k / n, 0, d0 * np.sin(np.pi * k / n))) for k in range(n + 1)]
    for a, b in zip(ns[:-1], ns[1:]):
        m.beam(a, b, Section(E=E, A=0.01, Iy=I, Iz=I, J=1e-5))
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    m.fix(ns[-1], (0, 1, 1, 0, 0, 0))
    F = np.zeros(m.ndof)
    F[6 * ns[-1]] = -1.0
    mid = ns[n // 2]
    lam, H = OpsFrame(m, transf="Corotational").pushover(F, node=mid, dof=2, target=L / 10, steps=200)
    dz = H[:, 6 * mid + 2] + d0
    k10 = int(np.argmin(np.abs(dz - 10 * d0)))
    assert lam[k10] / Pcr == pytest.approx(0.9, rel=0.01)
    assert lam[-1] / Pcr == pytest.approx(1 - d0 / dz[-1] + np.pi**2 / 8 * (dz[-1] / L) ** 2, rel=0.01)


def test_full_plastic_moment():
    """cantilever with an elastic-perfectly-plastic fiber section: base moment reaches Mp = fy·b·h²/4"""
    Lc, b, h, fy = 5.0, 0.1, 0.2, 3.45e5
    m = Model()
    ns = [m.node((Lc * k / 4, 0, 0)) for k in range(5)]
    for a, c in zip(ns[:-1], ns[1:]):
        m.beam(a, c, Section(E=2e8, A=b * h, Iy=b * h**3 / 12, Iz=h * b**3 / 12, J=1e-4), tag="steel")
    m.fix(ns[0])
    F = np.zeros(m.ndof)
    F[6 * ns[-1] + 2] = -1.0
    lam, H = OpsFrame(m, fiber={"steel": Fiber.rect(b, h, Fiber.steel_pp(fy))}).pushover(F, node=ns[-1], dof=2, target=-0.4, steps=80)
    assert lam.max() * Lc / (fy * b * h**2 / 4) == pytest.approx(1.0, rel=0.01)
    assert -H[1, 6 * ns[-1] + 2] == pytest.approx(lam[1] * Lc**3 / (3 * 2e8 * b * h**3 / 12), rel=0.01)


def test_tension_only_cable():
    m = Model()
    a0, a1, g = m.node((0, 0, 0)), m.node((10, 0, 0)), m.node((10, 0, 0))
    m.truss(a0, a1, Section(E=2e8, A=1e-3), tag="cable")
    m.link(a1, g, k=(1e3, 1e9, 1e9))
    m.fix(a0)
    m.fix(g)
    F = np.zeros(m.ndof)
    F[6 * a1] = -1.0
    f = OpsFrame(m, cable_tags=("cable",))
    assert f.static(F)[6 * a1] == pytest.approx(-1e-3, rel=1e-9)  # only the spring resists compression
    assert abs(f.truss_forces()[0]) < 1e-12


def test_sudden_removal_doubles_increment():
    """two bars carry a mass; removing one suddenly (no damping): peak = old static + 2 × static increment"""
    m = Model()
    top, bot, p = m.node((0, 0, 1)), m.node((0, 0, -1)), m.node((0, 0, 0))
    m.truss(top, p, Section(E=2e8, A=1e-3))
    m.truss(p, bot, Section(E=2e8, A=1e-3))
    m.fix(top)
    m.fix(bot)
    m.fix(p, (1, 1, 0, 0, 0, 0))
    M = np.zeros(m.ndof)
    M[6 * p:6 * p + 3] = 100.0
    f = OpsFrame(m, masses=M)
    F = np.zeros(m.ndof)
    F[6 * p + 2] = -1000.0
    f.static(F)
    k = 2e8 * 1e-3
    _, H = f.release(trusses=[1], t_end=0.5, dt=0.002, zeta=0.0, sub=1)
    assert H[:, 6 * p + 2].min() == pytest.approx(-1000 / (2 * k) + 2 * (-1000 / k + 1000 / (2 * k)), rel=0.005)
    with tempfile.TemporaryDirectory() as d:
        r = read(to_result(m, H[::25]).write(d, "rel", verbose=False))
    assert r["fields"]["u"].shape == (len(H[::25]), 3, 3)
