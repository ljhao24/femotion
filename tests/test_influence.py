"""Influence lines, moving loads and exact distributed-load section forces against closed forms."""
import numpy as np
import pytest

from femotion.frame import Model, Section, Solver
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces


def girder(spans, n, mass=2.0):
    """continuous beam on pinned / roller supports, n elements per span"""
    m = Model()
    xs = [0.0]
    for L in spans:
        xs += list(np.linspace(xs[-1], xs[-1] + L, n + 1)[1:])
    ns = [m.node((x, 0, 0)) for x in xs]
    bs = [m.beam(a, b, Section(E=3e7, A=1.0, Iy=0.1, Iz=0.1, J=0.1, mass=mass)) for a, b in zip(ns[:-1], ns[1:])]
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    k = 0
    for _ in spans:
        k += n
        m.fix(ns[k], (0, 1, 1, 1, 0, 0))
    return m, ns, bs


def test_simple_span_lane_load():
    L = 20.0
    m, ns, bs = girder([L], 20)
    sv = Solver(m)
    il = InfluenceLines(sv, ns, chain_sections(bs))
    mx, mn = il.lane("My", 10.5, 300.0)
    assert mx[19] == pytest.approx(10.5 * L * L / 8 + 300 * L / 4, rel=1e-9)  # j end of the element left of midspan
    assert mn[19] == pytest.approx(0.0, abs=1e-6)


def test_shear_jump_both_sides():
    """the point load goes on the side of the section that gives the larger shear"""
    L = 20.0
    m, ns, bs = girder([L], 20)
    il = InfluenceLines(Solver(m), ns, chain_sections(bs))
    mx, mn = il.lane("Vz", 1.0, 1.0)
    # at the left support (just inside the span): |V| = P + qL/2
    assert max(abs(mx[0]), abs(mn[0])) == pytest.approx(1 + L / 2, rel=1e-9)
    # at midspan, both sides: P/2 + q L/8 either way
    for k in (19, 20):
        assert mx[k] == pytest.approx(0.5 + L / 8, rel=1e-9)
        assert mn[k] == pytest.approx(-(0.5 + L / 8), rel=1e-9)


def test_two_span_hogging():
    L = 20.0
    m, ns, bs = girder([L, L], 40)
    il = InfluenceLines(Solver(m), ns, chain_sections(bs))
    _, mn = il.lane("My", 1.0, 0.0)
    assert mn[79] == pytest.approx(-L * L / 8, rel=1e-3)  # uniform load on both spans; linear between nodes
    _, mn = il.lane("My", 0.0, 1.0)
    assert mn[79] == pytest.approx(-L / (6 * np.sqrt(3)), rel=1e-4)  # peak of the line at x = L/√3


def test_uniform_load_exact_on_coarse_mesh():
    L, mass, g = 20.0, 2.0, 9.81
    m, ns, bs = girder([L], 2, mass)  # two elements only
    sv = Solver(m)
    secs = chain_sections(bs)
    w = mass * g
    assert uniform_load_forces(sv, secs, "My")[1] == pytest.approx(w * L * L / 8, rel=1e-9)
    V = uniform_load_forces(sv, secs, "Vz")
    assert abs(V[0]) == pytest.approx(w * L / 2, rel=1e-9) and V[1] == pytest.approx(0.0, abs=1e-6)


def test_axle_train_against_brute_force():
    L = 20.0
    m, ns, bs = girder([L], 40)
    il = InfluenceLines(Solver(m), ns, chain_sections(bs))
    loads, sp = [30, 120, 120, 140, 140], [3.0, 1.4, 7.0, 1.4]
    mx, _ = il.axles("My", loads, sp)
    off = np.r_[0, np.cumsum(sp)]
    eta = lambda x: np.where((x < 0) | (x > L), 0.0, np.where(x < L / 2, x / 2, (L - x) / 2))
    best = 0.0
    for lead in np.arange(-20, 40, 0.001):
        for o in (off, off[-1] - off):
            best = max(best, float((eta(lead + o) * loads).sum()))
    assert mx[39] == pytest.approx(best, rel=1e-6)
