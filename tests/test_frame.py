"""Fast linear frame solver, modal dynamics and frame post-processing."""
import numpy as np
import pytest

from femotion import dynamics
from femotion.frame import Model, Section, Solver
from femotion.frame.post import chain_force, classify_mode, translations


def cantilever(n=40, L=20.0, E=2e8, I=0.5, mu=2.0):
    m = Model()
    ns = [m.node((L * k / n, 0, 0)) for k in range(n + 1)]
    for a, b in zip(ns[:-1], ns[1:]):
        m.beam(a, b, Section(E=E, A=1.0, Iy=I, Iz=I, J=1.0, mass=mu))
    m.fix(ns[0])
    return m, ns


def test_cantilever_tip_and_frequency():
    L, E, I, mu = 20.0, 2e8, 0.5, 2.0
    m, ns = cantilever()
    sv = Solver(m)
    F = np.zeros(m.ndof)
    F[6 * ns[-1] + 2] = -100.0
    U = sv.solve(F)
    assert -translations(U)[-1, 2] == pytest.approx(100 * L**3 / (3 * E * I), rel=1e-9)
    f, Phi = sv.modes(2)
    assert f[0] == pytest.approx(1.8751**2 / (2 * np.pi) * np.sqrt(E * I / (mu * L**4)), rel=0.01)  # lumped mass
    assert np.allclose(Phi.T @ (sv.M[:, None] * Phi), np.eye(2), atol=1e-10)
    me = m.to_mesh()
    assert me.kind == "line2" and len(me.cells["line2"]) == 40


def test_modes_reproducible():
    m, _ = cantilever()
    f1, P1 = Solver(m).modes(6)
    f2, P2 = Solver(m).modes(6)
    assert np.array_equal(f1, f2) and np.array_equal(P1, P2)


def test_chain_force_cantilever_moment():
    """end load P: bending moment P·(L − x) along the beam (hogging)"""
    m, ns = cantilever(n=10, L=10.0)
    sv = Solver(m)
    F = np.zeros(m.ndof)
    F[6 * ns[-1] + 2] = -1.0
    U = sv.solve(F)
    My = chain_force(sv.beam_end_forces(U), list(range(10)), "My")
    assert np.allclose(np.abs(My), np.linspace(10, 0, 11), atol=1e-9)


def test_classify_first_mode():
    m, _ = cantilever()
    f, Phi = Solver(m).modes(1)
    kind, _ = classify_mode(Phi[:, 0], list(range(41)))
    assert kind in ("vertical", "lateral")  # square section: the first pair is bending


def test_newmark_and_step_release():
    f, z, dt = 1.0, 0.05, 0.01
    w = 2 * np.pi * f
    ag = np.full(3000, 0.5)
    Q = dynamics.newmark_modal(np.array([f]), np.array([1.0]), ag, dt, z, sub=2)
    assert Q[-1, 0] == pytest.approx(-0.5 / w**2, rel=2e-3)  # constant ground acceleration → static −a/ω²
    q1 = Q[: int(1.0 / dt) + 1, 0]
    assert q1.min() == pytest.approx(-0.5 / w**2 * (1 + np.exp(-z * np.pi / np.sqrt(1 - z**2))), rel=0.01)
    t = np.linspace(0, 40, 8001)
    U = dynamics.step_release(np.zeros(1), np.array([1.0]), np.array([w**2]), np.array([f]), np.array([[1.0]]), t, z)
    assert abs(U[0, 0]) < 1e-12  # no jump at t = 0
    assert U[-1, 0] == pytest.approx(1.0, rel=1e-3)  # settles at the new static position
