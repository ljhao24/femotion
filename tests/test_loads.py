"""Load cases (femotion.frame.loads) against closed forms of structural mechanics, the existing load paths
(uniform_load_forces, Solver.gravity, frame.thermal, frame.prestress, Solver.pretension), and the worst placements of
moving loads (InfluenceLines.lane_layout / axle_layout) replayed as ordinary load cases and checked by brute force."""
import numpy as np
import pytest
from scipy.integrate import quad

from femotion.codes import jtg
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces
from femotion.frame.loads import LoadCase, NodalLoad, combine, path_beams
from femotion.frame.prestress import Tendon
from femotion.frame.prestress import effects as prestress_effects
from femotion.frame.thermal import effects as thermal_effects

E, I = 3e7, 0.1
EI = E * I


def girder(spans, n, mass=2.0, flip=()):
    """continuous beam on a pin and rollers, n elements per span; beams in `flip` are numbered j → i"""
    m = Model()
    xs = [0.0]
    for L in spans:
        xs += list(np.linspace(xs[-1], xs[-1] + L, n + 1)[1:])
    ns = [m.node((x, 0, 0)) for x in xs]
    sec = Section(E=E, A=1.0, Iy=I, Iz=I, J=0.1, mass=mass)
    bs = [m.beam(*((b, a) if k in flip else (a, b)), sec) for k, (a, b) in enumerate(zip(ns[:-1], ns[1:]))]
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    k = 0
    for _ in spans:
        k += n
        m.fix(ns[k], (0, 1, 1, 1, 0, 0))
    return m, ns, bs, np.array(xs)


def node_moment(r, bs, k):
    """sagging moment at node k of the chain: at the end of the element before it that touches it (the first element
    for k = 0)"""
    from femotion.frame.loads import chain
    ch = chain(r.sv.m, bs)
    e, n0, n1, *_ = ch[k - 1] if k > 0 else ch[0]
    node = n1 if k > 0 else n0
    i = r.sv.m.beams[e][0]
    return r.section_forces([(e, "i" if node == i else "j")], "My")[0]


def vertical_reaction(r, node):
    return r.reactions()[6 * node + 2]


# ── closed forms: simply supported beam, loads off the nodes ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("flip", [(), (1, 3)])
def test_simple_beam_closed_forms(flip):
    L, n = 10.0, 5  # nodes every 2 m; the loads below sit between them
    m, ns, bs, xs = girder([L], n, flip=flip)
    sv = Solver(m)
    q, P = 12.0, 50.0
    # uniform over the whole span
    r = LoadCase().line(bs, q).solve(sv)
    assert vertical_reaction(r, ns[0]) == pytest.approx(q * L / 2, rel=1e-12)
    for k, x in enumerate(xs):
        assert node_moment(r, bs, k) == pytest.approx(q * x * (L - x) / 2, rel=1e-10, abs=1e-9)
    # a point load at a = 3.3 m (inside an element)
    a = 3.3
    r = LoadCase().point(bs, a, P).solve(sv)
    assert vertical_reaction(r, ns[0]) == pytest.approx(P * (L - a) / L, rel=1e-12)
    for k, x in enumerate(xs):
        Mx = P * (L - a) / L * x - P * max(x - a, 0.0)
        assert node_moment(r, bs, k) == pytest.approx(Mx, rel=1e-10, abs=1e-9)
    # part of the span, c = 1.3 … d = 6.9 m
    c, d = 1.3, 6.9
    r = LoadCase().line(bs, q, start=c, end=d).solve(sv)
    RA = q * (d - c) * (L - (c + d) / 2) / L
    assert vertical_reaction(r, ns[0]) == pytest.approx(RA, rel=1e-12)
    for k, x in enumerate(xs):
        Mx = RA * x - q * (min(max(x, c), d) - c) * (x - (c + min(max(x, c), d)) / 2)
        assert node_moment(r, bs, k) == pytest.approx(Mx, rel=1e-10, abs=1e-9)
    # a triangle rising from 0 at A to q at B: RA = qL/6, M(x) = qLx/6 − qx³/(6L)
    r = LoadCase().line(bs, 0.0, q_end=q).solve(sv)
    assert vertical_reaction(r, ns[0]) == pytest.approx(q * L / 6, rel=1e-12)
    assert vertical_reaction(r, ns[-1]) == pytest.approx(q * L / 3, rel=1e-12)
    for k, x in enumerate(xs):
        assert node_moment(r, bs, k) == pytest.approx(q * L * x / 6 - q * x ** 3 / (6 * L), rel=1e-10, abs=1e-9)
    # end shear = reaction (sign of femotion.frame.post.chain_force), shear just right of the point load
    r = LoadCase().point(bs, a, P).solve(sv)
    assert abs(r.section_forces([(bs[0], "i")], "Vz")[0]) == pytest.approx(P * (L - a) / L, rel=1e-12)


def test_trapezoid_is_uniform_plus_triangle_and_local_axes():
    m, ns, bs, xs = girder([12.0], 6)
    sv = Solver(m)
    secs = chain_sections(bs)
    trap = LoadCase().line(bs, 4.0, start=1.5, end=9.5, q_end=10.0).section_forces(sv, secs, "My")
    back = LoadCase().line(bs, 10.0, start=9.5, end=1.5, q_end=4.0).section_forces(sv, secs, "My")  # from the far end
    assert np.allclose(trap, back, rtol=1e-12, atol=1e-9)
    parts = (LoadCase().line(bs, 4.0, start=1.5, end=9.5) + LoadCase().line(bs, 0.0, start=1.5, end=9.5, q_end=6.0))
    assert np.allclose(trap, parts.section_forces(sv, secs, "My"), rtol=1e-12, atol=1e-9)
    # a horizontal beam: local z is global z, so local (0, 0, −q) is a downward load
    loc = LoadCase().line(bs, (0, 0, -7.0), axes="local").section_forces(sv, secs, "My")
    glo = LoadCase().line(bs, 7.0).section_forces(sv, secs, "My")
    assert np.allclose(loc, glo, rtol=1e-12, atol=1e-9)


def test_inclined_member_total_load():
    """a vertical load per member length on an inclined beam: the reactions carry w × the member length"""
    m = Model()
    ns = [m.node((4.0 * k, 0, 3.0 * k)) for k in range(4)]  # 5 m long elements
    bs = [m.beam(a, b, Section(E=E, A=1.0, Iy=I, Iz=I, J=0.1)) for a, b in zip(ns[:-1], ns[1:])]
    m.fix(ns[0], (1, 1, 1, 1, 0, 0))
    m.fix(ns[-1], (0, 1, 1, 1, 0, 0))
    sv = Solver(m)
    r = LoadCase().line(bs, 3.0, start=2.0, end=13.0).solve(sv)
    R = r.reactions()
    assert R[2::6].sum() == pytest.approx(3.0 * 11.0, rel=1e-12)


# ── closed forms: two equal spans ────────────────────────────────────────────────────────────────────────────────────
def test_two_span_closed_forms():
    L, n = 10.0, 4
    m, ns, bs, xs = girder([L, L], n)
    sv = Solver(m)
    q, P = 9.0, 40.0
    kB = n  # the pier node
    r = LoadCase().line(bs, q).solve(sv)
    assert node_moment(r, bs, kB) == pytest.approx(-q * L * L / 8, rel=1e-10)
    assert vertical_reaction(r, ns[0]) == pytest.approx(3 * q * L / 8, rel=1e-10)
    assert vertical_reaction(r, ns[kB]) == pytest.approx(5 * q * L / 4, rel=1e-10)
    r = LoadCase().line(bs, q, end=L).solve(sv)  # first span only
    assert node_moment(r, bs, kB) == pytest.approx(-q * L * L / 16, rel=1e-10)
    assert vertical_reaction(r, ns[0]) == pytest.approx(7 * q * L / 16, rel=1e-10)
    r = LoadCase().point(bs, L / 2, P).solve(sv)  # a node: nodal load
    assert node_moment(r, bs, kB) == pytest.approx(-3 * P * L / 32, rel=1e-10)
    assert vertical_reaction(r, ns[0]) == pytest.approx(13 * P / 32, rel=1e-10)
    r = LoadCase().line(bs, 0.0, end=L, q_end=q).solve(sv)  # triangle rising to the pier: −qL²/30
    assert node_moment(r, bs, kB) == pytest.approx(-q * L * L / 30, rel=1e-10)
    # a part of span 1 off the nodes: three-moment equation, 2 M_B (L1 + L2) = −6 A ā / L1, with A ā the first
    # moment about A of the simply supported moment diagram, integrated numerically here
    c, d = 2.7, 8.1
    RA = q * (d - c) * (L - (c + d) / 2) / L
    M0 = lambda x: RA * x - q * (min(max(x, c), d) - c) * (x - (c + min(max(x, c), d)) / 2)  # noqa: E731
    Aa = quad(lambda x: M0(x) * x, 0, L, points=[c, d], epsabs=1e-12)[0]
    r = LoadCase().line(bs, q, start=c, end=d).solve(sv)
    assert node_moment(r, bs, kB) == pytest.approx(-6 * Aa / L / (2 * 2 * L), rel=1e-9)


def test_settlement_and_temperature_closed_forms():
    L, n = 10.0, 4
    m, ns, bs, xs = girder([L, L], n)
    sv = Solver(m)
    secs = chain_sections(bs)
    # the pier settles Δ: removing it, the beam over 2L must be pulled down Δ: X = 6 EI Δ / L³, M_B = X L / 2
    dz = -0.012
    r = LoadCase("沉降").settlement(ns[n], dz).solve(sv)
    assert node_moment(r, bs, n) == pytest.approx(3 * EI * abs(dz) / L ** 2, rel=1e-10)
    assert r.U[6 * ns[n] + 2] == pytest.approx(dz, abs=1e-15)
    assert r.reactions()[2::6].sum() == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        LoadCase().settlement(ns[1], -0.01).solve(sv)  # not a support
    # a uniform free curvature κ (sagging +): M_B = −1.5 EI κ; and the same numbers as frame.thermal
    kappa = 2.4e-4
    case = LoadCase("温度梯度").imposed(bs, curvature=kappa, strain=1e-4)
    M = case.section_forces(sv, secs, "My")
    assert node_moment(case.solve(sv), bs, n) == pytest.approx(-1.5 * EI * kappa, rel=1e-10)
    assert np.allclose(M, thermal_effects(sv, bs, secs, "My", strain=1e-4, curvature=kappa), rtol=1e-12, atol=1e-8)
    N = case.section_forces(sv, secs, "N")  # the roller ends let it lengthen: no axial force
    assert np.allclose(N, 0.0, atol=1e-6)


# ── consistency with the other load paths ────────────────────────────────────────────────────────────────────────────
def test_self_weight_matches_uniform_load_forces_and_gravity():
    m, ns, bs, xs = girder([16.0, 16.0], 8, mass=2.6)
    t = m.node((16.0, 0, 6.0))  # a stay to the pier top with weight
    m.fix(t, (1, 1, 1, 1, 1, 1))
    m.truss(t, ns[4], Section(E=2e8, A=0.01, mass=0.08))
    sv = Solver(m)
    secs = chain_sections(bs)
    dead = LoadCase("自重").self_weight()
    for comp in ("My", "Vz", "N"):
        assert np.array_equal(dead.section_forces(sv, secs, comp), uniform_load_forces(sv, secs, comp))
    F = dead.nodal_loads(m)
    Fg = sv.gravity()
    heavy = sv.M.reshape(-1, 6)[:, 2] > 0.5 + 1e-9  # no 0.5 t floor of the lumped masses there
    assert np.allclose(F[2::6][heavy], Fg[2::6][heavy], rtol=1e-12)
    w_total = sum(sec.mass * 2.0 for (_, _, sec, _, _) in m.beams) * 9.81 + 0.08 * np.linalg.norm(m.X[t] - m.X[ns[4]]) * 9.81
    assert -F[2::6].sum() == pytest.approx(w_total, rel=1e-12)
    # model or solver: the same loads
    assert np.array_equal(dead.nodal_loads(m), dead.nodal_loads(sv))
    assert np.array_equal(dead.restrained_end_forces(m), dead.restrained_end_forces(sv))
    # factor and beams
    half = LoadCase().self_weight(0.5, beams=bs[:8], trusses=False)
    assert -half.nodal_loads(m)[2::6].sum() == pytest.approx(0.5 * 2.6 * 16.0 * 9.81, rel=1e-12)


def test_combinations_are_linear():
    m, ns, bs, xs = girder([12.0, 12.0], 6)
    sv = Solver(m)
    secs = chain_sections(bs)
    a = LoadCase("a").self_weight().point(bs, 5.0, 100.0)
    b = LoadCase("b").line(bs, 6.0, start=3.0, end=20.0, q_end=2.0).imposed(bs[:6], curvature=1e-4)
    c = LoadCase("c").settlement(ns[6], -0.005).nodal(ns[3], M=(0, 50, 0))
    sa, sb, sc = (x.section_forces(sv, secs, "My") for x in (a, b, c))
    comb = 1.2 * a + 1.4 * b - 0.5 * c
    assert np.allclose(comb.section_forces(sv, secs, "My"), 1.2 * sa + 1.4 * sb - 0.5 * sc, rtol=1e-12, atol=1e-8)
    comb2 = combine((1.2, a), (1.4, b), (-0.5, c), name="ULS")
    assert comb2.name == "ULS" and len(comb2.items) == len(a.items) + len(b.items) + len(c.items)
    assert np.allclose(comb2.section_forces(sv, secs, "My"), comb.section_forces(sv, secs, "My"), rtol=1e-12, atol=1e-8)
    assert comb.name == "1.2×a + 1.4×b − 0.5×c" and (-a).name == "−a"


def test_tendon_and_bar_strain_match_their_modules():
    m, ns, bs, xs = girder([20.0, 20.0], 10)
    sv = Solver(m)
    secs = chain_sections(bs)
    e = 0.4 * np.sin(np.pi * xs / 20.0) * np.where(xs <= 20, 1, -1)
    td = Tendon(bs, ecc=e, force=np.linspace(3000, 2600, len(bs)))
    tot, pri, sec = prestress_effects(sv, td, secs, "My")
    assert np.allclose(LoadCase().tendon(td).section_forces(sv, secs, "My"), tot, rtol=1e-12, atol=1e-8)
    # bars: initial strain as Solver.pretension
    m2 = Model()
    a, b, c = m2.node((0, 0, 0)), m2.node((4, 0, 0)), m2.node((2, 0, 2))
    m2.beam(a, b, Section(E=E, A=1.0, Iy=I, Iz=I, J=0.1))
    m2.truss(a, c, Section(E=2e8, A=1e-3)), m2.truss(c, b, Section(E=2e8, A=1e-3))
    m2.fix(a, (1, 1, 1, 1, 0, 0)), m2.fix(b, (0, 1, 1, 1, 0, 0)), m2.fix(c, (0, 1, 0, 0, 0, 0))
    sv2 = Solver(m2)
    eps = np.array([2e-4, 1e-4])
    r = LoadCase().bar_strain([0, 1], eps).solve(sv2)
    U = sv2.solve(sv2.pretension(eps))
    assert np.allclose(r.U, U, rtol=1e-12, atol=1e-15)
    assert np.allclose(r.truss_forces(), sv2.truss_forces(U, eps), rtol=1e-12)


def test_element_loads_and_point_sides():
    m, ns, bs, xs = girder([10.0], 5, flip=(2,))
    case = LoadCase().line(bs, 5.0, start=3.0, end=7.0, q_end=9.0).point(bs, 4.0, 10.0, side=1).point(bs, 4.5, 1.0)
    pieces = case.element_loads(m)
    total = sum(-(p.p0[2] + p.p1[2]) / 2 * (p.b - p.a) for p in pieces if p.kind == "uniform")
    assert total == pytest.approx((5 + 9) / 2 * 4.0, rel=1e-12)
    pts = [p for p in pieces if p.kind == "point"]
    assert pts[0].beam == bs[2] and pts[0].a == pytest.approx(2.0)  # flipped element: from its i end at x = 6
    assert pts[1].beam == bs[2] and pts[1].a == pytest.approx(1.5)
    # on a node with side 0 it is a nodal load; side ±1 puts it on the element, which changes only that element's shear
    sv = Solver(m)
    r0 = LoadCase().point(bs, 4.0, 10.0).solve(sv)
    r1 = LoadCase().point(bs, 4.0, 10.0, side=1).solve(sv)
    rm = LoadCase().point(bs, 4.0, 10.0, side=-1).solve(sv)
    assert np.allclose(r0.U, r1.U) and np.allclose(r0.U, rm.U)
    v = lambda r, sec: r.section_forces([sec], "Vz")[0]  # noqa: E731
    # node 4 m: the j end of bs[1] (before it) and, bs[2] being flipped, the j end of bs[2] (after it)
    before, after = (bs[1], "j"), (bs[2], "j")
    assert abs(v(r1, after) - v(r0, after)) == pytest.approx(10.0, rel=1e-12)
    assert v(r1, before) == pytest.approx(v(r0, before), abs=1e-9)
    assert abs(v(rm, before) - v(r0, before)) == pytest.approx(10.0, rel=1e-12)
    assert v(rm, after) == pytest.approx(v(r0, after), abs=1e-9)
    # just inside an element, the load at the node is the limit of a load approaching it
    r_eps = LoadCase().point(bs, 4.0 + 1e-7, 10.0).solve(sv)
    assert v(r_eps, after) == pytest.approx(v(r1, after), abs=1e-5)
    with pytest.raises(ValueError):
        LoadCase().point(bs, 11.0, 1.0).solve(sv)


# ── worst placements of moving loads ─────────────────────────────────────────────────────────────────────────────────
def lumped(case, m):
    """the influence-line model of a load case: every span load moved to the two ends of its element in linear
    shares, kept on that element (the side of a jump), nodal loads unchanged"""
    out = LoadCase("lumped", [it for it in case.items if isinstance(it, NodalLoad)])
    from femotion.frame.model import local_axes
    for p in case.element_loads(m):
        i, j, sec, ref, tag = m.beams[p.beam]
        L, R = local_axes(m.X[i], m.X[j], ref)
        if p.kind == "point":
            Fi, Fj = (1 - p.a / L) * p.p0, p.a / L * p.p0
        else:
            xg, wg = np.polynomial.legendre.leggauss(2)
            Fi = Fj = 0.0
            for t, w in zip(xg, wg):
                x = p.a + (t + 1) / 2 * (p.b - p.a)
                load = (p.p0 + (p.p1 - p.p0) * (x - p.a) / (p.b - p.a)) * w * (p.b - p.a) / 2
                Fi, Fj = Fi + (1 - x / L) * load, Fj + x / L * load
        out.point([p.beam], 0.0, R.T @ Fi, side=1).point([p.beam], L, R.T @ Fj, side=-1)
    # point loads on nodes (side 0) are nodal loads
    from femotion.frame.loads import PointLoad, _point_target, _geometry
    for it in case.of(PointLoad):
        tgt = _point_target(_geometry(m), it)
        if tgt[0] == "node":
            out.nodal(tgt[1], tgt[2])
    return out


def _cases():
    m1, n1, b1, _ = girder([20.0], 20)
    m2, n2, b2, _ = girder([20.0, 20.0], 16)
    return [(m1, n1, b1, True), (m2, n2, b2, False)]


@pytest.mark.parametrize("comp", ["My", "Vz"])
def test_layouts_reproduce_effects_and_replay(comp):
    live = jtg.LiveLoad(L0=20.0, lanes=2, mu=0.25, distribution=0.6)
    crowd = jtg.CrowdLoad(L0=20.0, width=1.5, distribution=0.7)
    for m, ns, bs, determinate in _cases():
        sv = Solver(m)
        secs = chain_sections(bs)
        il = InfluenceLines(sv, ns, secs)
        assert path_beams(m, il.path) == bs
        for eff, lays in [(live.effects(il, comp), live.layouts(il, comp)),
                          (live.vehicle_effects(il, comp), live.vehicle_layouts(il, comp)),
                          (crowd.effects(il, comp), crowd.layouts(il, comp))]:
            # the very numbers of effects()
            assert np.array_equal(eff[0], [a.value for a, b in lays]) and np.array_equal(eff[1], [b.value for a, b in lays])
            env = max(abs(lay.value) for pair in lays for lay in pair)
            for k in range(0, len(secs), 3):
                for lay in lays[k]:
                    case = lay.load_case(il)
                    v = case.section_forces(sv, [secs[k]], comp)[0]
                    scale = max(abs(lay.value), 1.0)
                    if determinate:  # the influence line is exactly linear between nodes
                        assert v == pytest.approx(lay.value, abs=1e-8 * scale)
                    else:  # cubic inside the elements: O(h²) away (h = 1.25 m here); the lumped model is exact
                        assert abs(v - lay.value) <= 5e-3 * env
                        vl = lumped(case, m).section_forces(sv, [secs[k]], comp)[0]
                        assert vl == pytest.approx(lay.value, abs=1e-8 * scale)
    assert live.layouts(il, comp, 3)[0].value == live.effects(il, comp)[0][3]  # one section → one pair


def test_replay_converges_on_continuous_beam():
    errs = []
    for n in (8, 16, 32):
        m, ns, bs, _ = girder([20.0, 20.0], n)
        sv = Solver(m)
        secs = chain_sections(bs)
        il = InfluenceLines(sv, ns, secs)
        k = 2 * n - 1  # the pier, left
        lay = il.lane_layout(k, "My", 10.5, 300.0, "min")
        errs.append(abs(lay.load_case(il).section_forces(sv, [secs[k]], "My")[0] - lay.value) / abs(lay.value))
    assert errs[1] < errs[0] / 3 and errs[2] < errs[1] / 3  # O(h²)


def _true_line(sv, bs, sec, comp, xs, side=0):
    """ordinates by actual solves of a unit point load at each x (exact Hermite fixed-end forces)"""
    return np.array([LoadCase().point(bs, x, 1.0, side=side).section_forces(sv, [sec], comp)[0] for x in xs])


@pytest.mark.parametrize("comp,k", [("My", 7), ("Vz", 8), ("Vz", 0)])
def test_brute_force_finds_nothing_worse(comp, k):
    """sweep P and the vehicle over a fine grid with true solves on a simple span (exact lines): the recorded
    placement is the worst one"""
    L = 20.0
    m, ns, bs, _ = girder([L], 20)
    sv = Solver(m)
    secs = chain_sections(bs)
    il = InfluenceLines(sv, ns, secs)
    q, P = 10.5, 300.0
    grid = np.linspace(0, L, 801)
    line = _true_line(sv, bs, secs[k], comp, grid)
    # at the section's own node the jump: both sides
    s_node = il.s[il._pos[m.beams[secs[k][0]][0 if secs[k][1] == "i" else 1]]]
    sides = [_true_line(sv, bs, secs[k], comp, [s_node], side=sd)[0] for sd in (-1, 1)]
    for sense in ("max", "min"):
        lay = il.lane_layout(k, comp, q, P, sense)
        sg = 1 if sense == "max" else -1
        best_P = max(sg * line.max() if sg > 0 else -line.min(), max(sg * v for v in sides))
        assert sg * lay.eta_P == pytest.approx(best_P, abs=1e-9)
        uni = lay.load_case(il)
        base = uni.section_forces(sv, [secs[k]], comp)[0]
        assert base == pytest.approx(lay.value, abs=1e-8 * max(abs(lay.value), 1))
        # moving the ends of every loaded segment by ±0.25 m never helps
        for dx in (-0.25, 0.25):
            alt = LoadCase()
            for s0, s1 in lay.segments:
                a, b = max(s0 - dx, 0.0), min(s1 + dx, L)
                alt.line(bs, q, start=a, end=b)
            if lay.P_at is not None:
                alt.point(bs, lay.P_at, P, side=lay.P_side)
            assert sg * alt.section_forces(sv, [secs[k]], comp)[0] <= sg * base + 1e-9
    # the vehicle: every lead position on a 0.01 m grid, both directions, the fine true line interpolated
    loads, sp = np.array(jtg.VEHICLE_AXLES), np.array(jtg.VEHICLE_SPACINGS)
    off = np.r_[0.0, np.cumsum(sp)]

    def eta(x):
        v = np.interp(x, grid, line)
        return np.where((x < 0) | (x > L), 0.0, v)
    for sense in ("max", "min"):
        lay = il.axle_layout(k, comp, loads, sp, sense)
        sg = 1 if sense == "max" else -1
        best = 0.0
        for o in (off, off[-1] - off):
            lead = np.arange(-off[-1] - 1, L + 1, 0.01)
            v = (eta(lead[:, None] + o[None, :]) * loads).sum(axis=1)
            best = max(best, sg * v.max() if sg > 0 else -v.min())
        # the jump at the section's node is a single point: the grid may miss it, so compare one way only
        assert sg * lay.value >= best - 1e-6
        if lay.positions is not None:
            assert lay.load_case(il).section_forces(sv, [secs[k]], comp)[0] == pytest.approx(lay.value, abs=1e-6)
            assert lay.lead == lay.positions[0]
            assert np.allclose(np.abs(np.diff(lay.positions)), sp)
            assert lay.heading in (-1, 1) and np.sign(lay.positions[0] - lay.positions[-1]) == lay.heading


def test_layout_details():
    """the jump side of P at a support section, empty placements, the vehicle heading"""
    L = 20.0
    m, ns, bs, _ = girder([L], 20)
    sv = Solver(m)
    secs = chain_sections(bs)
    il = InfluenceLines(sv, ns, secs)
    lay = il.lane_layout(0, "Vz", 10.5, 300.0, "min")  # just right of the left support: P inside the element
    assert lay.P_at == 0.0 and lay.P_side == 1 and abs(lay.eta_P) == pytest.approx(1.0)
    assert lay.segments == [(0.0, L)]
    empty = il.lane_layout(0, "Vz", 10.5, 300.0, "max")  # nothing of the other sign
    assert empty.P_at is None and empty.segments == [] and empty.value == 0.0
    assert il.axle_layout(0, "Vz", [100.0], [], "max").positions is None
    k_mid = 19
    lay = il.lane_layout(k_mid, "My", 10.5, 300.0, "max")
    assert lay.P_at == pytest.approx(L / 2) and lay.area == pytest.approx(L * L / 8)
    with pytest.raises(ValueError):
        il.lane_layout(0, "My", 1.0, 1.0, "largest")


def test_the_same_case_on_opensees():
    """the case needs only the model: OpenSees under nodal_loads(m), plus restrained_end_forces(m), gives the linear
    solver's end forces"""
    pytest.importorskip("openseespy")
    from femotion.frame.opensees import OpsFrame
    m, ns, bs, xs = girder([16.0, 16.0], 8)
    case = (LoadCase("mixed").self_weight().line(bs, 6.0, start=3.3, end=21.7, q_end=12.0).point(bs, 9.1, 150.0)
            .imposed(bs, curvature=-1.5e-4).tendon(Tendon(bs, ecc=0.3 * np.sin(np.pi * xs / 16.0), force=2500.0)))
    r = case.solve(Solver(m))
    ops = OpsFrame(m)
    U = ops.static(case.nodal_loads(m))
    ef = ops.beam_end_forces() + case.restrained_end_forces(m)
    assert np.allclose(U, r.U, rtol=1e-8, atol=1e-12)
    assert np.allclose(ef, r.end_forces, rtol=1e-7, atol=1e-6)
