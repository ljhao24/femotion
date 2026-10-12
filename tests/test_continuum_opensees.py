"""OpenSees solid back-end: cross-validation with the scikit-fem solver, closed-form checks of the crack band, embedded
bars and prestress, and a reinforced concrete beam against cracked-section theory and code crack-width formulas."""
import numpy as np
import pytest

pytest.importorskip("openseespy")
pytest.importorskip("skfem")
from femotion.continuum import Elastic  # noqa: E402
from femotion.continuum.opensees import (HEX_TETS, CellLocator, ConcreteDamage, ElasticIso, ElasticOrtho, OpsSolid,  # noqa: E402
                                         band_widths, elastic, interpolate, steel, submodel_boundary)
from femotion.mesh import structured  # noqa: E402
from femotion.mesh.core import Mesh  # noqa: E402


def tets_of(m: Mesh) -> Mesh:
    """the hexahedra of a structured mesh cut into six tetrahedra each"""
    return Mesh(m.points, {"tet4": m.cells["hex8"][:, HEX_TETS].reshape(-1, 4)})


def cantilever(m, mat, P=-5.0):
    s = OpsSolid(m, mat)
    s.fix(lambda x: np.isclose(x[0], 0))
    L, b, h = m.points.max(0)
    s.static(s.traction(lambda x: np.isclose(x[0], L), (0, 0, P / (b * h))), steps=1)
    e = Elastic(m, E=mat.E, nu=mat.nu, order=1)
    e.traction(lambda x: np.isclose(x[0], L), (0, 0, P / (b * h)))
    e.fix(lambda x: np.isclose(x[0], 0))
    e.solve()
    return s, e


# ── linear range: same element, same mesh, another solver ─────────────────
@pytest.mark.parametrize("kind", ["hex8", "tet4"])
@pytest.mark.parametrize("concrete", [False, True])
def test_linear_range_matches_scikit_fem(kind, concrete):
    """below cracking, stdBrick / FourNodeTetrahedron with ElasticIsotropic or ASDConcrete3D give the scikit-fem order-1
    solution on the same mesh (displacements and cell stresses); the reaction balances the load"""
    m = structured.box(2.0, 0.2, 0.4, 10, 2, 4)
    m = tets_of(m) if kind == "tet4" else m
    E, nu = 3.0e7, 0.2  # kN, m
    mat = ConcreteDamage.from_strength(38e3, E=E, nu=nu, units="kN-m") if concrete else ElasticIso(E, nu)
    s, e = cantilever(m, mat)  # bending stress ≈ 1.9 MPa < 0.9 ft
    U, Ue = s.history[-1].U[:m.n_points], e.nodal_displacement()
    assert np.abs(U - Ue).max() < 1e-7 * np.abs(Ue).max()
    S, Se = s.history[-1].cell["stress"], e.cell_stress()
    assert np.abs(S - Se).max() < 1e-6 * np.abs(Se).max()
    assert s.reaction()[2] == pytest.approx(5.0, rel=1e-6)
    assert not s.history[-1].cell["cracked"].any()


def test_orthotropic_uniaxial():
    """ElasticOrthotropic: uniaxial stress along y gives εy = σ/Ey, εx = −νxy … in the compliance convention of OpenSees"""
    m = structured.box(1.0, 1.0, 1.0, 2, 2, 2)
    mat = ElasticOrtho(E=(10.0, 20.0, 30.0), nu=(0.1, 0.2, 0.3), G=(4.0, 5.0, 6.0))
    s = OpsSolid(m, mat)
    s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
    s.static(s.traction(lambda x: np.isclose(x[1], 1.0), (0, 1.0, 0)))
    eps = s.history[-1].cell["strain"]
    assert np.allclose(eps[:, 1], 1 / 20.0, rtol=1e-8)
    assert np.allclose(s.history[-1].cell["eps_in"], 0, atol=1e-12)  # the compliance used for cracks is the same


# ── loads and geometry ────────────────────────────────────────────────────
def test_loads_are_consistent():
    m = structured.box(2.0, 1.0, 1.0, 4, 2, 3)
    s = OpsSolid(m, ElasticIso(1.0, 0.2))
    assert s.body_force((0, 0, -2.0)).sum(0) == pytest.approx([0, 0, -4.0])
    assert s.self_weight(3.0).sum(0)[2] == pytest.approx(-6.0)
    assert s.traction(lambda x: np.isclose(x[2], 1.0), (0.5, 0, 0)).sum(0) == pytest.approx([1.0, 0, 0])
    assert s.pressure(lambda x: np.isclose(x[0], 2.0), 3.0).sum(0) == pytest.approx([-3.0, 0, 0])
    F = s.point_force((0.3, 0.4, 0.55), (0, 0, -1.0))
    assert F.sum(0) == pytest.approx([0, 0, -1.0])
    nz = np.flatnonzero(F[:, 2])  # shape functions reproduce the point: Σ N_i x_i = p
    assert (-F[nz, 2]) @ m.points[nz] == pytest.approx([0.3, 0.4, 0.55])
    # traction t(x) linear in x: moment of the nodal forces = moment of the traction
    Ft = s.traction(lambda x: np.isclose(x[2], 1.0), lambda x: np.vstack([0 * x[0], 0 * x[0], x[0]]))
    assert Ft[:, 2].sum() == pytest.approx(2.0) and (Ft[:, 2] @ m.points[:, 0]) == pytest.approx(8 / 3)


def test_locator_and_interpolation():
    m = structured.box(1.0, 2.0, 1.5, 3, 4, 3)
    m.points[:, 2] += 0.1 * m.points[:, 0] * m.points[:, 1]  # skewed, non-planar hexahedra
    lin = lambda X: 1.0 + 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]  # noqa: E731
    pts = np.array([[0.2, 0.3, 0.4], [0.9, 1.9, 1.5], [0.5, 1.0, 0.7]])
    v = interpolate(m, lin(m.points), pts)
    assert np.allclose(v, lin(pts), atol=1e-9)
    assert np.isnan(interpolate(m, lin(m.points), [[5.0, 0, 0]])).all()
    loc = CellLocator(tets_of(m))
    assert loc.find((0.33, 1.1, 0.8))[1] == "tet4"
    assert np.allclose(band_widths(structured.box(2, 1, 1, 4, 2, 2)), 0.5)
    assert np.allclose(band_widths(structured.box(2, 1, 1, 4, 2, 2), ("projected", (1, 1, 0))), 1.0 / np.sqrt(2))


# ── crack band: peak, fracture energy, mesh objectivity ───────────────────
def tension_prism(nb: int, nu: float, implex: bool = False, steps: int = 150, kind: str = "hex8"):
    """a 400 × 100 × 100 mm prism pulled through a rigid end (tie + displacement control); the middle cell layer is 5 %
    weaker (a notch) so the crack forms there. kind="tet4": every cube cut into six tetrahedra"""
    L, b = 400.0, 100.0
    kw = dict(units="N-mm", prepeak="linear", nu=nu, implex=implex)
    c = ConcreteDamage.from_strength(38.0, **kw)
    cw = ConcreteDamage.from_strength(38.0, ft=0.95 * c.ft, **kw)
    m = structured.box(L, b, b, 4 * nb, nb, nb)
    m = tets_of(m) if kind == "tet4" else m
    hx = L / (4 * nb)
    weak = np.abs(m.centroids()[:, 0] - (L / 2 + hx / 2)) < hx / 2
    m.cell_groups = {"c": {kind: np.flatnonzero(~weak)}, "w": {kind: np.flatnonzero(weak)}}
    s = OpsSolid(m, {"c": c, "w": cw})
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
    end = s.tie(lambda x: np.isclose(x[0], L), comps=(0,))
    st = s.push(s.nodal_force(None, (1.0, 0, 0), nodes=[end]), end, 0, 0.3, steps=steps, record=steps)
    assert st.converged
    u, F = s.curve(end, 0, at=s.nodes(lambda x: np.isclose(x[0], 0)))
    return s, c, u, F, b * b, hx


def peak_within_a_step(F, A, ft, E, du=0.3 / 150, L=400.0):
    """the peak of the discrete curve is within one load step (σ increment E·Δu/L) of the strength"""
    d = E * du / L
    return ft - d <= F.max() / A <= ft + d


def test_crack_band_energy_and_objectivity():
    """peak stress = ft of the weak layer (to one load step); fracture energy dissipated per unit area = Gf with ν = 0
    (exactly the input law); with ν = 0.2 the lateral restraint of the softening band by its elastic neighbours raises the
    effective-stress based equivalent strain and the band dissipates ~6 % less; in both cases the energy does not
    depend on the cell size (crack band regularisation)"""
    W = {}
    for nb in (1, 2):
        for nu in (0.0, 0.2):
            s, c, u, F, A, hx = tension_prism(nb, nu)
            assert peak_within_a_step(F, A, 0.95 * c.ft, c.E)
            assert F[-1] < 1e-2 * F.max()  # fully separated
            W[nb, nu] = np.trapezoid(F, u) / (c.Gf * A)
            cracked = s.history[-1].cell["cracked"]
            assert np.allclose(s.cell_centroids()[cracked, 0], 200 + hx / 2)  # only the weak layer
    assert W[1, 0.0] == pytest.approx(1.0, abs=0.02) and W[2, 0.0] == pytest.approx(1.0, abs=0.02)
    assert 0.90 < W[2, 0.2] < 0.97
    assert W[1, 0.2] == pytest.approx(W[2, 0.2], rel=0.02)


@pytest.mark.parametrize("nb", [1, 2])
def test_crack_band_of_tetrahedra(nb):
    """tetrahedra (cubes cut in six): with the band ∛(6V) a crack through one layer of cubes dissipates Gf (ν = 0) and
    the whole end displacement is its width"""
    s, c, u, F, A, hx = tension_prism(nb, 0.0, kind="tet4")
    assert np.trapezoid(F, u) / (c.Gf * A) == pytest.approx(1.0, abs=0.02)
    assert s.cracks_along((0, 31, 29), (400, 31, 29))["max_width"] == pytest.approx(u[-1] - F[-1] / A / c.E * 400, rel=0.01)


def test_crack_width_of_a_separated_prism():
    """once the prism is fully separated, the whole end displacement less the elastic stretch is one crack: the cell
    widths, the line read-out and the surface read-out agree with it"""
    s, c, u, F, A, hx = tension_prism(2, 0.2)
    w_true = u[-1] - F[-1] / A / c.E * 400.0
    r = s.cracks_along((0, 30, 30), (400, 30, 30))
    assert len(r["cracks"]) == 1 and r["cracks"][0]["width"] == pytest.approx(w_true, rel=0.02)
    assert r["cracks"][0]["s"] == pytest.approx(200 + hx / 2, abs=hx / 2)
    assert s.crack_widths().max() == pytest.approx(w_true, rel=0.02)
    n = s.history[-1].cell["crack_normal"][s.crack_widths() > 0]
    assert np.allclose(np.abs(n[:, 0]), 1.0, atol=1e-3)  # crack plane normal to the pull
    surf = s.surface_cracks(lambda x: np.isclose(x[2], 100.0))
    assert surf["width"].max() == pytest.approx(w_true, rel=0.02)


def test_implex_follows_the_same_softening():
    """IMPL-EX with displacement control (pseudo-time set per step): two iterations per step and the same separation;
    at the peak, where every layer is equally stressed, the extrapolation lets all layers pass the peak by one step,
    which costs a little extra energy"""
    s, c, u, F, A, hx = tension_prism(2, 0.2, implex=True)
    assert peak_within_a_step(F, A, 0.95 * c.ft, c.E)
    assert np.trapezoid(F, u) / (c.Gf * A) == pytest.approx(1.0, abs=0.1)
    assert s.crack_widths().argmax() in np.flatnonzero(np.isclose(s.cell_centroids()[:, 0], 225.0))


def test_snap_back_limit():
    c = ConcreteDamage.from_strength(38.0, units="N-mm")
    hmax = c.h_max()
    assert hmax == pytest.approx(1.25 * c.E * c.Gf / c.ft**2, rel=1e-2)  # MC2010 bilinear: E·w1/(0.8 ft)
    e, sg = c.uniaxial(2 * hmax)
    assert np.all(np.diff(e) > 0)
    e1, s1 = c.uniaxial(50.0)
    w = (e1 - s1 / c.E - c.eps_in_peak) * 50.0
    pk = int(np.argmax(s1))
    assert np.trapezoid(s1[pk:], w[pk:]) == pytest.approx(c.Gf, rel=2e-3)  # area of the branch (with its tiny residual)


# ── embedded bars and prestress ───────────────────────────────────────────
@pytest.mark.parametrize("offset", [0.0, 0.013])
def test_embedded_bar_axial_stiffness(offset):
    """bar along the axis (on mesh nodes, or off them and tied by ASDEmbeddedNodeElement): axial stiffness EcAg + EsAs
    (the bar overlaps the concrete) and a uniform bar force"""
    L, b = 1.0, 0.2
    m = structured.box(L, b, b, 5, 2, 2)
    Ec, Es, As = 3.0e7, 2.0e8, 1e-3
    s = OpsSolid(m, ElasticIso(Ec, 0.2))
    s.bar([(0, 0.1 + offset, 0.1 - offset), (L, 0.1 + offset, 0.1 - offset)], area=As, material=elastic(Es))
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix(nodes=[s.nearest((0, 0, 0))], where=None, comps=(1, 2))
    s.fix(nodes=[s.nearest((0, b, 0))], where=None, comps=(2,))
    right = s.nodes(lambda x: np.isclose(x[0], L))
    U = np.full((len(right), 3), np.nan)
    U[:, 0] = 1e-4
    s.static(imposed=(right, U))
    k = s.reaction(at=right)[0] / 1e-4 * L
    assert k == pytest.approx(Ec * b * b + Es * As, rel=1e-5)
    N = s.history[-1].bar["N"]
    assert np.allclose(N, Es * As * 1e-4 / L, rtol=1e-4)
    if offset:
        assert len(s._tie) > 0  # really embedded


def test_bonded_prestress_closed_form():
    """centroidal bonded tendon with σ0 (InitStressMaterial, tension +): equilibrium and compatibility give a uniform
    concrete stress −σ0 Ap / (Ag + n Ap) away from the anchorages and the tendon force σ0 Ap (1 − n Ap / (Ag + n Ap))"""
    L, b = 2.0, 0.2
    m = structured.box(L, b, b, 20, 4, 4)
    Ec, Ep, Ap, s0 = 3.45e7, 1.95e8, 6e-4, 1.2e6  # kN, m, kPa
    s = OpsSolid(m, ElasticIso(Ec, 0.2))
    s.bar([(0, 0.1, 0.1), (L, 0.1, 0.1)], area=Ap, material=steel(1.86e6, Ep), prestress=s0)
    s.fix(nodes=[s.nearest((0, 0, 0))], where=None)
    s.fix(nodes=[s.nearest((0, b, 0))], where=None, comps=(0, 2))
    s.fix(nodes=[s.nearest((0, 0, b))], where=None, comps=(0, 1))
    s.static(None)
    n = Ep / Ec
    sc = -s0 * Ap / (b * b + n * Ap)
    mid = np.abs(s.cell_centroids()[:, 0] - L / 2) < 0.3
    assert s.history[-1].cell["stress"][mid, 0].mean() == pytest.approx(sc, rel=0.01)
    r = s.bar_results(0)
    assert r["N"][np.abs(r["s"] - L / 2) < 0.3].mean() == pytest.approx(s0 * Ap * (1 - n * Ap / (b * b + n * Ap)), rel=0.01)


def test_unbonded_tendon_equals_its_anchor_forces():
    """an unbonded eccentric tendon anchored at two mesh nodes loads the concrete exactly like two opposite point forces
    of the final tendon force at the anchors (scikit-fem, same mesh): the equivalent-load method"""
    L, b, h = 3.0, 0.4, 0.6
    m = structured.box(L, b, h, 15, 2, 6)
    Ec, Ep, Ap, s0 = 3.45e7, 1.95e8, 1.4e-3, 1.1e6
    c = ElasticIso(Ec, 0.2)
    s = OpsSolid(m, c)
    a0, a1 = (0.0, 0.2, 0.2), (L, 0.2, 0.2)  # e = 0.1 below the centroid, inside the kern (h/6)
    s.bar([a0, a1], area=Ap, material=elastic(Ep), prestress=s0, bond=False)
    fixed = [s.nearest((L / 2, 0, 0)), s.nearest((L / 2, b, 0)), s.nearest((L / 2, 0, h))]
    s.fix(nodes=[fixed[0]], where=None)
    s.fix(nodes=[fixed[1]], where=None, comps=(0, 2))
    s.fix(nodes=[fixed[2]], where=None, comps=(0, 1))
    s.static(None)
    N = s.history[-1].bar["N"][0]
    assert 0.85 * s0 * Ap < N < s0 * Ap  # loss from elastic shortening and from the indentation under the point anchors
    e = Elastic(m, E=Ec, nu=c.nu, order=1)
    e.point_load(a0, (N, 0, 0))
    e.point_load(a1, (-N, 0, 0))
    for i, n in enumerate(fixed):
        e.fix(np.array([n]), comps=[(0, 1, 2), (0, 2), (0, 1)][i])
    e.solve()
    S, Se = s.history[-1].cell["stress"], e.cell_stress()
    assert np.abs(S - Se).max() < 1e-5 * np.abs(Se).max()


# ── dowels (anchors) ──────────────────────────────────────────────────────
@pytest.mark.parametrize("rot", [True, False])
def test_dowel_cantilever_from_a_rigid_block(rot):
    """an anchor set 0.3 deep into a (practically) rigid block, standing 0.5 proud, loaded at its head: with its
    rotation tied to the solid it is a clamped cantilever, δ = Pa³/3EI exactly; tied in translation only, the embedded
    part bends between the tied nodes (spacing s) and the head moves by about s/a more; shear and moment at the surface
    are P and Pa"""
    m = structured.box(1.0, 1.0, 1.0, 4, 4, 4)
    Es, d, P, a, seg = 2e8, 0.02, 1.0, 0.5, 0.05
    s = OpsSolid(m, ElasticIso(2e14, 0.2))
    k = s.dowel([(0.51, 0.52, 0.7), (0.51, 0.52, 1.0 + a)], d=d, E=Es, rot=rot, seg=seg)
    s.fix(lambda x: np.isclose(x[2], 0))
    tip = s.dowels[k]["nodes"][-1]
    s.static(s.nodal_force(None, (P, 0, 0), nodes=[tip]))
    d_cl = P * a**3 / (3 * Es * np.pi * d**4 / 64)
    ratio = s.history[-1].U[tip, 0] / d_cl
    assert ratio == pytest.approx(1.0, rel=1e-6) if rot else 1.0 < ratio < 1.0 + 1.2 * seg / a
    r = s.dowel_results(k)
    above = r["x"][:, 2] > 1.0
    assert np.allclose(r["V"][above], P) and r["M"][above].max() == pytest.approx(P * a)


def test_strip_on_dowel_anchors():
    """a prestressed FRP strip under the soffit between two bolt heads 10 mm below the surface: the strip force is the
    shear in both bolts at their heads, the support reactions vanish (self-equilibrated), the force fades along the
    embedded length, and the bolts bend (dowel action) — the strip loses force as the heads deflect towards each other"""
    m = structured.box(2.0, 0.4, 0.6, 20, 4, 6)
    s = OpsSolid(m, ConcreteDamage.from_strength(38e3, units="kN-m"))
    k1 = s.dowel([(0.3, 0.2, -0.01), (0.3, 0.2, 0.15)], d=0.016, E=2e8)
    k2 = s.dowel([(1.7, 0.2, -0.01), (1.7, 0.2, 0.15)], d=0.016, E=2e8)
    Ap, s0 = 50 * 1.4e-6, 0.4 * 2.8e6
    s.bar([(0.3, 0.2, -0.01), (1.7, 0.2, -0.01)], area=Ap, material=frp_strip(), prestress=s0, bond=False, outside="free")
    s.fix(None, nodes=[s.nearest((1.0, 0, 0.6))])
    s.fix(None, nodes=[s.nearest((1.0, 0.4, 0.6))], comps=(0, 2))
    s.fix(None, nodes=[s.nearest((0, 0.2, 0.6))], comps=(1, 2))
    st = s.static(None, steps=2)
    assert st.converged
    N = s.history[-1].bar["N"][0]
    i, j, _ = s._segs[0]
    eps = (s.history[-1].U[j, 0] - s.history[-1].U[i, 0]) / 1.4
    assert N == pytest.approx(Ap * (s0 + 1.65e8 * eps), rel=1e-6) and eps < 0  # σ = σ0 + E ε: the heads came closer
    for k in (k1, k2):
        r = s.dowel_results(k)
        assert r["V"][0] == pytest.approx(N, rel=1e-6)  # head element: all of the strip force
        assert r["V"][-1] < 0.1 * N
    assert np.abs(s.reaction()).max() < 1e-6 * N


def frp_strip():
    from femotion.continuum.opensees import frp

    return frp(1.65e8, 0.016)


# ── supports, ties, sub-models ────────────────────────────────────────────
def test_winkler_support_and_reactions():
    """a block on a distributed spring support under uniform pressure settles by q / k; the spring forces are reported
    as reactions"""
    m = structured.box(2.0, 1.0, 0.5, 4, 2, 2)
    s = OpsSolid(m, ElasticIso(1e7, 0.0))
    s.face_spring(lambda x: np.isclose(x[2], 0), (1e3, 1e3, 5e4))
    q = 100.0
    s.static(s.pressure(lambda x: np.isclose(x[2], 0.5), q))
    U = s.history[-1].U
    bottom = s.nodes(lambda x: np.isclose(x[2], 0))
    assert U[bottom, 2] == pytest.approx(-q / 5e4, rel=1e-3)
    assert s.reaction()[2] == pytest.approx(q * 2.0, rel=1e-6)


def test_submodel_reproduces_the_global_solution():
    """a local model cut out of a global linear model and driven by the interpolated global displacements on its cut
    faces reproduces the global field inside (same elements, same nodes there)"""
    G = structured.box(4.0, 1.0, 1.0, 16, 4, 4)
    e = Elastic(G, E=3e7, nu=0.2, order=1)
    e.fix(lambda x: np.isclose(x[0], 0))
    e.traction(lambda x: np.isclose(x[0], 4.0), (0, 0, -100.0))
    e.solve()
    Ug = e.nodal_displacement()
    loc = structured.box(1.0, 1.0, 1.0, 4, 4, 4, origin=(1.5, 0, 0))
    s = OpsSolid(loc, ElasticIso(3e7, 0.2))
    nodes, vals = submodel_boundary(G, Ug, s, lambda x: np.isclose(x[0], 1.5) | np.isclose(x[0], 2.5))
    s.static(imposed=(nodes, vals))
    inner = interpolate(G, Ug, loc.points)
    assert np.abs(s.history[-1].U[:loc.n_points] - inner).max() < 1e-5 * np.abs(inner).max()


def test_stale_model_raises():
    m = structured.box(1, 1, 1, 1, 1, 1)
    a = OpsSolid(m, ElasticIso(1.0, 0.2))
    a.fix(lambda x: np.isclose(x[2], 0))
    a.static(None)
    b = OpsSolid(m, ElasticIso(1.0, 0.2))
    b.fix(lambda x: np.isclose(x[2], 0))
    b.static(None)
    with pytest.raises(RuntimeError, match="stale"):
        a.static(None)


def test_bar_outside_the_solid_raises():
    s = OpsSolid(structured.box(1, 1, 1, 1, 1, 1), ElasticIso(1.0, 0.2))
    with pytest.raises(ValueError, match="outside"):
        s.bar([(0.5, 0.5, 0.5), (0.5, 0.5, 2.0)], area=1.0, material=elastic(1.0))


# ── a reinforced concrete beam ────────────────────────────────────────────
def rc_beam(prepeak: str, target: float, steps: int):
    """simply supported beam 200 × 300 mm, span 2400, loads at the third points ±300 from midspan (a = 900), 2 Ø16 at
    40 mm; a quarter by symmetry, 50 mm hexahedra; elastic pads under the bearing and the loading plate"""
    B, D, L2, H = 200.0, 300.0, 1300.0, 50.0
    m = structured.box(L2, B / 2, D, 26, 2, 6)
    cen = m.centroids()
    pad = ((np.abs(cen[:, 0] - 100) < 50) & (cen[:, 2] < H)) | ((np.abs(cen[:, 0] - 1000) < 50) & (cen[:, 2] > D - H))
    m.cell_groups = {"concrete": {"hex8": np.flatnonzero(~pad)}, "pads": {"hex8": np.flatnonzero(pad)}}
    c = ConcreteDamage.from_strength(38.0, units="N-mm", prepeak=prepeak)
    s = OpsSolid(m, {"concrete": c, "pads": ElasticIso(c.E, c.nu)})
    s.bar([(20, 50, 40), (L2, 50, 40)], area=201.06, material=steel(450.0, 2.0e5))
    s.fix(lambda x: np.isclose(x[0], L2), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], B / 2), comps=(1,))
    sup = s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 100), comps=(2,))  # a line bearing (no end clamping)
    F = s.traction(lambda x: np.isclose(x[2], D) & (np.abs(x[0] - 1000) <= 25 + 1e-6), (0, 0, -1.0))
    mid = s.nearest((L2, B / 2, 0))
    st = s.push(F, mid, 2, -target, steps=steps)
    assert st.converged
    u, P = s.curve(mid, 2, at=sup, sign=1.0)
    return s, c, -u, 4 * P  # the whole beam carries four times the quarter's reaction


def section_properties(c, As=402.12, d=260.0, B=200.0, D=300.0):
    n = 2.0e5 / c.E
    A = B * D + n * As  # embedded bars overlap the concrete
    yb = (B * D * D / 2 + n * As * (D - d)) / A
    Ig = B * D**3 / 12 + B * D * (D / 2 - yb) ** 2 + n * As * (yb - (D - d)) ** 2
    x = (-n * As + np.sqrt((n * As) ** 2 + 2 * B * n * As * d)) / B
    Icr = B * x**3 / 3 + n * As * (d - x) ** 2
    return n, yb, Ig, x, Icr


def test_rc_beam_cracking_moment():
    """with a linear pre-peak law the first crack forms when the bottom integration points reach ft: M = ft·Ig / (yb − zgp),
    the transformed-section cracking moment evaluated at the Gauss points of the bottom cell layer (8 % above ft·W here)"""
    s, c, u, P = rc_beam("linear", 1.0, 40)
    n, yb, Ig, x, Icr = section_properties(c)
    zgp = 25.0 * (1 - 1 / np.sqrt(3))
    i = s.first_crack()
    M0, M1 = P[i - 1] / 2 * 900, P[i] / 2 * 900
    assert M0 * 0.99 < c.ft * Ig / (yb - zgp) < M1 * 1.01
    cen = s.cell_centroids()[s.history[i].cell["cracked"]]
    assert np.all(cen[:, 0] > 950) and np.all(cen[:, 2] < 50)  # in the constant-moment zone, at the bottom


@pytest.mark.slow
def test_rc_beam_service_response():
    """default concrete law, to 3.5 mm: the load–deflection curve lies between the uncracked and the fully cracked beam;
    the steel stress in the constant-moment zone approaches the cracked-section value; the largest crack is of the size
    of the short-term maximum crack widths of the codes, on the same basis as the monotonic short-term analysis: JTG
    3362-2018 6.4.3 with C2 = 1 (Ml = 0), Eurocode 2 7.3.4 with kt = 0.6 and hc,ef = min(2.5 (h − d), (h − x)/3, h/2)
    (here 0.95 and 0.75 of them); the cracks are 100–200 mm apart"""
    from femotion.codes import jtg
    from femotion.codes.jtg import sls

    s, c, u, P = rc_beam("mc2010", 3.5, 70)
    n, yb, Ig, x, Icr = section_properties(c)
    a, L = 900.0, 2400.0
    k_b = a * (3 * L**2 - 4 * a**2) / (24 * c.E)  # deflection per unit load P/2, bending only
    late = P > 0.75 * P.max()
    assert np.all(u[late] > P[late] / 2 * k_b / Ig) and np.all(u[late] < 1.15 * P[late] / 2 * k_b / Icr)
    sec = jtg.rect(200, 300, concrete="C30", steel="HRB400", bottom=[(2, 16, 40)])
    i = s.recorded()[-1]
    M = P[i] / 2 * a
    r = s.bar_results(0, i)
    sig_fe = r["stress"][r["x"][:, 0] > 950].max()
    sig_cr = n * M * (260 - x) / Icr
    assert 0.8 < sig_fe / sig_cr < 1.0  # concrete between cracks still carries tension (perfect bond)
    w_jtg = sls.crack_width(sec, Ms=M / 1e6, Ml=0.0, env="I").demand  # short term: C2 = 1
    rho = 402.12 / (200.0 * min(2.5 * 40.0, (300.0 - x) / 3, 150.0))
    wk = (3.4 * 32 + 0.425 * 0.8 * 0.5 * 16 / rho) * max(sig_cr - 0.6 * c.ft / rho * (1 + n * rho), 0.6 * sig_cr) / 2e5
    cr = s.cracks_along((150, 50, 1), (1300, 50, 1), step=i)
    assert 0.8 < cr["max_width"] / w_jtg < 1.2 and 0.6 < cr["max_width"] / wk < 0.95
    xs = sorted(cc["s"] for cc in cr["cracks"] if cc["width"] > 0.05)
    assert len(xs) >= 3 and 80 < np.median(np.diff(xs)) < 250


def test_export_and_save(tmp_path):
    """fem-result/1 with the bars as line2 cells, a ParaView series, and the numpy archive"""
    pytest.importorskip("meshio")
    from femotion.result import read

    m = structured.box(1.0, 0.2, 0.2, 5, 1, 1)
    s = OpsSolid(m, ConcreteDamage.from_strength(38e3, units="kN-m"))
    s.bar([(0.0, 0.1, 0.05), (1.0, 0.1, 0.05)], area=2e-4, material=steel(4.5e5, 2e8))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.static(s.traction(lambda x: np.isclose(x[0], 1.0), (0, 0, -10.0)), steps=3)
    r = s.to_result()
    assert r.mesh.cells["line2"].shape == (len(s._segs), 2) and len(r.meta["t"]) == 4  # initial + 3 steps
    path = r.write(tmp_path, "beam", verbose=False)
    back = read(path)
    assert back["fields"]["u"].shape == (4, s.n_nodes, 3)
    assert back["fields"]["crack_width"].shape == (4, s.nc + len(s._segs))
    files = s.export(tmp_path / "beam.pvd")
    assert files[0].suffix == ".pvd" and len(files) == 5
    npz = np.load(s.save_npz(tmp_path / "beam.npz"))
    assert npz["U"].shape == (4, s.n_nodes, 3) and npz["cell/width"].shape == (4, s.nc)


def test_width_history_and_stop():
    """the stop condition ends a stage at a crack width; width_history gives the largest width per recorded step"""
    s, c, u, F, A, hx = tension_prism(1, 0.0, steps=150)
    idx, w = s.width_history()
    assert w[0] == 0 and w[-1] == pytest.approx(s.crack_widths().max())
    m = structured.box(400.0, 100.0, 100.0, 4, 1, 1)
    s2 = OpsSolid(m, ConcreteDamage.from_strength(38.0, units="N-mm", prepeak="linear", implex=False))
    s2.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s2.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s2.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
    end = s2.tie(lambda x: np.isclose(x[0], 400.0), comps=(0,))
    st = s2.push(s2.nodal_force(None, (1.0, 0, 0), nodes=[end]), end, 0, 0.3, steps=150,
                 stop=lambda h: h.cell["width"].max() > 0.05)
    assert "stop" in st.message and 0.05 < s2.crack_widths().max() < 0.06


def test_free_rigid_body_modes_are_reported():
    """a beam on a line bearing (y, z) and a roller (z), held in x only along one edge at midspan, can still turn about
    the vertical through the corner of the bearing: one free rigid-body motion, which the check finds and warns about;
    holding x along a line across the beam removes it"""
    m = structured.box(3000.0, 200.0, 300.0, 6, 2, 3)
    s = OpsSolid(m, ElasticIso(3e4, 0.2))
    s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 0), comps=(1, 2))
    s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 3000), comps=(2,))
    s.fix(lambda x: np.isclose(x[0], 1500) & np.isclose(x[1], 0), comps=(0,))
    assert s.free_rigid_modes() == 1
    with pytest.warns(UserWarning, match="rigid-body"):
        s.static(None)
    s2 = OpsSolid(m, ElasticIso(3e4, 0.2))
    s2.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 0), comps=(1, 2))
    s2.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 3000), comps=(2,))
    s2.fix(lambda x: np.isclose(x[0], 1500) & np.isclose(x[2], 0), comps=(0,))
    assert s2.free_rigid_modes() == 0
    s2.spring(None, (0, 0, 1.0), nodes=[0])  # springs count as supports too
    assert s2.free_rigid_modes() == 0 and OpsSolid(m, ElasticIso(3e4, 0.2)).free_rigid_modes() == 6


# ── review fixes: embedded bars in bending, holds under IMPL-EX, unloading, elongated cells, sub-models, deviators ──
def test_bar_in_bending_follows_the_hexahedra():
    """a bar embedded off the nodes of a hexahedral beam in pure bending: tied with the cells' own trilinear
    interpolation, so its strain is the strain of the solid at its level everywhere (no saw-tooth between segments);
    its force equals scikit-fem's embedded bar on the same mesh (continuum.embed.Bar, the same interpolation) and the
    transformed-section value n M y / I of beam theory"""
    L, B, D, z_b, y_b = 1000.0, 100.0, 200.0, 40.0, 45.0
    m = structured.box(L, B, D, 40, 4, 8)
    E, Es, As, M = 30000.0, 2.0e5, 201.0, 2.0e6
    t = lambda x: np.stack([-M * (x[2] - D / 2) / (B * D**3 / 12), 0 * x[0], 0 * x[0]])  # noqa: E731
    pts = [(0.0, y_b, z_b), (L, y_b, z_b)]
    s = OpsSolid(m, ElasticIso(E, 0.0))
    s.bar(pts, area=As, material=elastic(Es), seg=12.5)
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix(None, nodes=[s.nearest((0, 0, 0))], comps=(1, 2))
    s.fix(None, nodes=[s.nearest((0, B, 0))], comps=(2,))
    s.static(s.traction(lambda x: np.isclose(x[0], L), t))
    r = s.bar_results(0)
    mid = (r["x"][:, 0] > 300) & (r["x"][:, 0] < 700)
    N = r["N"][mid]
    assert np.ptp(N) < 1e-4 * np.abs(N).max()  # uniform along the bar (sub-tetrahedron ties: ±17 … 33 %)
    e = Elastic(m, E=E, nu=0.0, order=1)
    e.bar(pts, EA=Es * As, bonded=True, ds=12.5)
    e.traction(lambda x: np.isclose(x[0], L), t)
    e.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    e.fix(np.array([s.nearest((0, 0, 0))]), comps=(1, 2))
    e.fix(np.array([s.nearest((0, B, 0))]), comps=(2,))
    e.solve()
    Ne = e.result().bar_force("bar0")
    assert np.abs(r["N"] - Ne).max() < 5e-5 * np.abs(Ne).max()  # penalty ties: 1e-5
    n = Es / E  # the bar overlaps the concrete: n·As, not (n − 1)·As
    A = B * D + n * As
    zc = (B * D * D / 2 + n * As * z_b) / A
    Itr = B * D**3 / 12 + B * D * (D / 2 - zc) ** 2 + n * As * (zc - z_b) ** 2
    assert N.mean() / As == pytest.approx(n * M * (zc - z_b) / Itr, rel=0.03)  # 8 cells over the depth: within 3 %


def anchored_strip(implex: bool, steps: int):
    """a prestressed FRP strip on two bolts under a member (coarse mesh): the prestress cracks the concrete at the bolts"""
    m = structured.box(2.0, 0.4, 0.6, 20, 4, 6)
    s = OpsSolid(m, ConcreteDamage.from_strength(38e3, units="kN-m", implex=implex), band=("projected", (0, 1, 0)))
    s.dowel([(0.3, 0.2, -0.01), (0.3, 0.2, 0.15)], d=0.016, E=2e8)
    s.dowel([(1.7, 0.2, -0.01), (1.7, 0.2, 0.15)], d=0.016, E=2e8)
    s.bar([(0.3, 0.2, -0.01), (1.7, 0.2, -0.01)], area=70e-6, material=frp_strip(), prestress=2.0e6, bond=False,
          outside="free")
    s.fix(None, nodes=[s.nearest((1.0, 0, 0.6))])
    s.fix(None, nodes=[s.nearest((1.0, 0.4, 0.6))], comps=(0, 2))
    s.fix(None, nodes=[s.nearest((0, 0.2, 0.6))], comps=(1, 2))
    st = s.static(None, steps=steps)
    h = s.history[-1]
    return s, st, h.cell["width"].max(), h.bar["N"][0], h.cell["cracked"]


def test_hold_stage_under_implex_matches_implicit():
    """a prestress-only stage (no load increment) under IMPL-EX: the prestress, applied in full at once, is first
    settled without extrapolation (a fixed-point iteration on the damage), so the result is the implicit one whatever
    `steps` is (before: −89 % … +23 % depending on steps)"""
    _, _, w0, N0, c0 = anchored_strip(False, 1)
    assert c0.any() and w0 > 0
    for steps in (1, 5):
        s, st, w, N, c = anchored_strip(True, steps)
        assert [x.name for x in s.stages] == ["prestress", st.name] and "settled" in s.stages[0].message
        assert "settled" in st.message and len(st.steps) <= 2
        assert w == pytest.approx(w0, rel=0.1) and N == pytest.approx(N0, rel=5e-3)
        assert abs(int(c.sum()) - int(c0.sum())) <= 2


@pytest.mark.parametrize("w_rel", [0.5, 2.5, 4.0])
def test_unloading_keeps_the_plastic_share(w_rel):
    """a crack opened to w and unloaded to zero stress keeps plastic_t·w open (0.2 by default), on the first and on the
    second branch of the softening law alike (before: 0.79–0.98 of w beyond the kink, the damage being capped at 0.999)"""
    L, b = 400.0, 100.0
    kw = dict(units="N-mm", prepeak="linear", nu=0.0, implex=False)
    c = ConcreteDamage.from_strength(38.0, **kw)
    cw = ConcreteDamage.from_strength(38.0, ft=0.95 * c.ft, **kw)
    m = structured.box(L, b, b, 4, 1, 1)
    weak = np.abs(m.centroids()[:, 0] - 250) < 50
    m.cell_groups = {"c": {"hex8": np.flatnonzero(~weak)}, "w": {"hex8": np.flatnonzero(weak)}}
    s = OpsSolid(m, {"c": c, "w": cw})
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
    end = s.tie(lambda x: np.isclose(x[0], L), comps=(0,))
    left = s.nodes(lambda x: np.isclose(x[0], 0))
    w1 = cw.t_post[1][0]
    target = w_rel * w1 + cw.ft / cw.E * L
    F = s.nodal_force(None, (1.0, 0, 0), nodes=[end])
    s.push(F, end, 0, target, steps=100, record=False)
    u, P = s.curve(end, 0, at=left)
    w_top = u[-1] - P[-1] / (b * b) / c.E * L
    s.push(F, end, 0, -target, steps=400, record=False, stop=lambda h: -s.reaction(left, -1)[0] < 0)
    u, P = s.curve(end, 0, at=left)
    u0 = np.interp(0.0, [P[-1], P[-2]], [u[-1], u[-2]])  # displacement where the force reaches zero
    assert u0 / w_top == pytest.approx(c.plastic_t, abs=0.01)


def test_elongated_cells_warn_and_projected_band():
    """cells twice as long across the crack as along its normal: band='volume' (∛V) warns, and its crack dissipates
    only about 63 % of Gf; band=('projected', n) along the crack normal gives Gf"""
    L, b = 400.0, 100.0
    kw = dict(units="N-mm", prepeak="linear", nu=0.0, implex=False)
    c = ConcreteDamage.from_strength(38.0, **kw)
    cw = ConcreteDamage.from_strength(38.0, ft=0.95 * c.ft, **kw)
    m = structured.box(L, b, b, 8, 1, 1)  # cells 50 × 100 × 100
    weak = np.abs(m.centroids()[:, 0] - 225) < 25
    m.cell_groups = {"c": {"hex8": np.flatnonzero(~weak)}, "w": {"hex8": np.flatnonzero(weak)}}
    W = {}
    for band in ("volume", ("projected", (1, 0, 0))):
        if band == "volume":
            with pytest.warns(UserWarning, match="elongated"):
                s = OpsSolid(m, {"c": c, "w": cw}, band=band)
        else:
            s = OpsSolid(m, {"c": c, "w": cw}, band=band)
        s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
        s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
        s.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
        end = s.tie(lambda x: np.isclose(x[0], L), comps=(0,))
        s.push(s.nodal_force(None, (1.0, 0, 0), nodes=[end]), end, 0, 0.3, steps=150, record=False)
        u, F = s.curve(end, 0, at=s.nodes(lambda x: np.isclose(x[0], 0)))
        W[band if isinstance(band, str) else "projected"] = np.trapezoid(F, u) / (c.Gf * b * b)
    assert W["projected"] == pytest.approx(1.0, abs=0.02)
    assert W["volume"] == pytest.approx(50.0 / np.cbrt(50.0 * 100 * 100), rel=0.05)


def test_submodel_from_a_quadratic_global_model():
    """driven by a CaseResult, the sub-model takes the global model's own (quadratic) interpolation on its cut faces;
    a corner-node mesh with linear interpolation would be off by a few per cent there"""
    G = structured.box(4.0, 1.0, 1.0, 4, 1, 1)
    e = Elastic(G, E=3e7, nu=0.2, order=2)
    e.fix(lambda x: np.isclose(x[0], 0))
    e.traction(lambda x: np.isclose(x[0], 4.0), (0, 0, -100.0))
    e.solve()
    r = e.result()
    loc = structured.box(1.0, 1.0, 1.0, 8, 8, 8, origin=(1.5, 0, 0))
    s = OpsSolid(loc, ElasticIso(3e7, 0.2))
    cut = lambda x: np.isclose(x[0], 1.5) | np.isclose(x[0], 2.5)  # noqa: E731
    nodes, vals = s.boundary_from(r, cut)
    assert np.allclose(vals, r.sample(loc.points[nodes])["u"], rtol=0, atol=1e-15)
    _, lin = submodel_boundary(e.result_mesh(), e.nodal_displacement(), s, cut)
    assert np.abs(lin - vals).max() > 0.01 * np.abs(vals).max()  # corner-linear: what the CaseResult avoids
    s.static(imposed=(nodes, vals))
    inner = loc.nearest((2.0, 0.5, 0.5))
    assert s.history[-1].U[inner] == pytest.approx(r.sample(loc.points[[inner]])["u"][0], rel=0.02)


def deviated_tendon(slide: bool):
    L, B, D = 6.0, 0.4, 0.6
    m = structured.box(L, B, D, 30, 2, 6)
    pts = [(0, 0.2, 0.3), (3.0, 0.2, 0.1), (L, 0.2, 0.3)]
    s = OpsSolid(m, ElasticIso(3.45e7, 0.2))
    s.bar(pts, area=1.4e-3, material=elastic(1.95e8), prestress=1.1e6, bond=False, slide=slide)
    s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 0), comps=(0, 1, 2))
    s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], L), comps=(1, 2))
    load = lambda x: np.isclose(x[2], D) & (x[0] < 2.0)  # noqa: E731
    s.static(None, name="prestress")
    N0 = s.history[-1].bar["N"].copy()
    s.static(s.traction(load, (0, 0, -200.0)), name="load on the left")
    return s, m, pts, load, s.history[-1].bar["N"] - N0


def test_unbonded_tendon_slides_over_a_deviator():
    """an unbonded tendon over a frictionless deviator keeps one force along its whole length under an unsymmetric load,
    and the member deflects as in scikit-fem with continuum.embed.Bar(bonded=False) (one force from the summed
    elongations of the spans) and the same prestress; slide=False anchors it at the deviator (a force per span)"""
    s, m, pts, load, dN = deviated_tendon(True)
    N = s.history[-1].bar["N"]
    assert np.ptp(N) < 1e-6 * N.mean() and dN.mean() > 0
    e = Elastic(m, E=3.45e7, nu=0.2, order=1)
    bar = e.bar(pts, EA=1.95e8 * 1.4e-3, bonded=False)
    e.case().bar_prestress(bar, 1.1e6 * 1.4e-3)
    e.traction(load, (0, 0, -200.0))
    e.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 0))
    e.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 6.0), comps=(1, 2))
    e.solve()
    Ne = e.result().bar_force(bar.name)[0]
    assert N.mean() == pytest.approx(Ne, rel=1e-5)
    Ue = e.nodal_displacement()
    assert np.abs(s.history[-1].U[:m.n_points] - Ue).max() < 1e-5 * np.abs(Ue).max()
    *_, dN2 = deviated_tendon(False)
    assert abs(dN2[0] - dN2[1]) > 0.3 * dN2.mean()  # the force increments of the two spans differ


def test_dowel_twist_is_held_and_loose_bar_nodes_are_reported():
    """a dowel tied in translation only gets a torsional spring at its first embedded node (else it turns freely about
    its axis); a free node between two bars in line is reported, a run of a bonded bar outside the solid is one bar"""
    m = structured.box(1.0, 1.0, 1.0, 4, 4, 4)
    s = OpsSolid(m, ConcreteDamage.from_strength(38e3, units="kN-m"))
    k = s.dowel([(0.51, 0.52, 0.7), (0.51, 0.52, 1.1)], d=0.02, E=2e8, seg=0.05)
    s.fix(lambda x: np.isclose(x[2], 0))
    tip = s.dowels[k]["nodes"][-1]
    s._build()
    s.ops.timeSeries("Linear", 99)
    s.ops.pattern("Plain", 99, 99)
    s.ops.load(tip + 1, 1.0, 0.0, 0.0, 0.0, 0.0, 1e-3)  # a lateral force and a small torque at the head
    st = s.static(None)
    assert st.converged
    s2 = OpsSolid(m, ElasticIso(3e7, 0.2))
    s2.bar([(0.5, 0.5, 0.5), (0.5, 0.5, 1.4), (0.5, 0.5, 1.8)], area=1e-4, material=elastic(2e8), outside="free")
    z = s2.bars[0]["points"][:, 2]
    assert np.all((z <= 1.0 + 1e-9) | np.isclose(z, 1.4) | np.isclose(z, 1.8))  # not subdivided outside the solid
    assert len(s2.loose_bar_nodes()) == 2  # the point at z = 1.4 between two bars in line, and the free end
    s2.fix(None, nodes=[s2.bars[0]["nodes"][-1]])
    assert len(s2.loose_bar_nodes()) == 1


def test_continuum_materials_are_converted():
    """femotion.continuum.Isotropic / Orthotropic are taken directly; the orthotropic Poisson's ratios are converted to
    OpenSees' νzx = ν13·E3/E1 (each column of the compliance checked by a uniaxial stress along x, y and z)"""
    from femotion.continuum import Isotropic, Orthotropic

    o = Orthotropic(E=(10.0, 20.0, 30.0), nu=(0.1, 0.15, 0.2), G=(4.0, 5.0, 6.0))
    S = np.linalg.inv(o.D6())
    m = structured.box(1.0, 1.0, 1.0, 1, 1, 1)
    for d in range(3):
        s = OpsSolid(m, o)
        for c in range(3):
            s.fix(lambda x, c=c: np.isclose(x[c], 0), comps=(c,))
        t = np.zeros(3)
        t[d] = 1.0
        s.static(s.traction(lambda x, d=d: np.isclose(x[d], 1.0), t))
        assert np.allclose(s.history[-1].cell["strain"][0, :3], S[:3, d], rtol=1e-8)
    assert isinstance(OpsSolid(m, Isotropic(3e4, 0.2)).materials[0], ElasticIso)
    with pytest.raises(ValueError, match="rotated"):
        OpsSolid(m, Orthotropic.transversely_isotropic(10.0, 5.0, 0.2, 0.3, 2.0, axis=(1, 1, 0)))


# ── a notched beam against an independent discrete cohesive-crack model ──
def cohesive_reference(E: float, nu: float, t_post, cmod_max: float = 0.3, n: int = 150, hx: float = 2.5):
    """Hillerborg's fictitious crack, independently of OpenSees: half of a three-point bend beam in plane stress
    (scikit-fem, bilinear quadrilaterals hx), the crack on the symmetry plane above the notch with the cohesive law σ(w)
    (ConcreteDamage.t_post, the input to be reproduced) on the ligament nodes, condensed onto their openings and the
    load and solved by Newton's method under crack-mouth-opening control; unloading of a ligament node along its secant.
    Beam 850 × 100 × 100 mm, span 790, notch 50 mm (zero width), load spread over 30 mm at midspan. Returns
    (CMOD, P) of the whole beam."""
    import scipy.sparse.linalg as spla
    from skfem import Basis, BilinearForm, ElementQuad1, ElementVector, MeshQuad, asm
    from skfem.helpers import ddot, sym_grad, trace

    D, B, HALF, SUP, A0, LW = 100.0, 100.0, 425.0, 395.0, 50.0, 15.0
    mq = MeshQuad.init_tensor(np.linspace(0, HALF, int(HALF / hx) + 1), np.linspace(0, D, int(D / hx) + 1))
    ib = Basis(mq, ElementVector(ElementQuad1()), intorder=2)
    mu, lam = E / (2 * (1 + nu)), E * nu / (1 - nu**2)

    @BilinearForm
    def a(u, v, w):
        eu = sym_grad(u)
        return B * (2 * mu * ddot(eu, sym_grad(v)) + lam * trace(eu) * trace(sym_grad(v)))

    K = asm(a, ib).tolil()
    X, dofs = mq.p, ib.nodal_dofs
    lig = np.flatnonzero(np.isclose(X[0], 0) & (X[1] >= A0 - 1e-9))
    lig = lig[np.argsort(X[1, lig])]
    dz = np.diff(X[1, lig])
    Atr = np.r_[dz / 2, 0] + np.r_[0, dz / 2]
    Atr = Atr * B
    kp = 1e8  # the closed ligament: σ = kp w until the crack opens
    for i, d in enumerate(dofs[0, lig]):
        K[d, d] += 2 * kp * Atr[i]
    top = np.flatnonzero(np.isclose(X[1], D) & (X[0] <= LW + 1e-9))
    top = top[np.argsort(X[0, top])]
    xt = X[0, top]
    wt = np.r_[np.diff(xt) / 2, 0] + np.r_[0, np.diff(xt) / 2]
    f = np.zeros(K.shape[0])
    f[dofs[1, top]] = -0.5 * wt / wt.sum()  # half of a unit load on the half beam
    sup = np.flatnonzero(np.isclose(X[0], SUP) & np.isclose(X[1], 0))[0]
    free = np.setdiff1d(np.arange(K.shape[0]), [dofs[1, sup]])
    lu = spla.splu(K.tocsc()[free][:, free])
    pos = {d: i for i, d in enumerate(free)}
    nl = len(lig)
    rhs = np.zeros((len(free), nl + 1))
    for i, d in enumerate(dofs[0, lig]):
        rhs[pos[d], i] = 1.0
    rhs[:, nl] = f[free]
    sol = lu.solve(rhs)
    rows = [pos[d] for d in dofs[0, lig]]
    G, g = sol[rows, :nl], sol[rows, nl]
    mouth = pos[dofs[0, np.flatnonzero(np.isclose(X[0], 0) & np.isclose(X[1], 0))[0]]]
    cm, cmP = sol[mouth, :nl], sol[mouth, nl]
    wl = np.array([p[0] for p in t_post])
    sl = np.array([p[1] for p in t_post])
    w0 = sl[0] / kp
    whist = np.zeros(nl)

    def law(w):
        wpk = np.maximum(whist, w)
        soft = wpk > w0
        env = np.where(soft, np.interp(wpk - w0, wl, sl), kp * wpk)
        i = np.clip(np.searchsorted(wl, wpk - w0, side="right") - 1, 0, len(wl) - 2)
        denv = np.where(soft, np.where(wpk - w0 >= wl[-1], 0.0, (sl[i + 1] - sl[i]) / (wl[i + 1] - wl[i])), kp)
        unl = w < whist - 1e-15
        sec = np.where(wpk > 0, env / np.where(wpk > 0, wpk, 1), kp)
        s = np.where(w <= 0, kp * w, np.where(unl, sec * w, env))
        ds = np.where(w <= 0, kp, np.where(unl, sec, denv))
        return kp * w - s, kp - ds

    u, P, out = np.zeros(nl), 0.0, [(0.0, 0.0)]
    for cmod in np.linspace(0, cmod_max, n + 1)[1:]:
        for _ in range(100):
            r, dr = law(2 * u)
            R = np.r_[u - G @ (Atr * r) - g * P, 2 * (cm @ (Atr * r) + cmP * P) - cmod]
            J = np.zeros((nl + 1, nl + 1))
            J[:-1, :-1] = np.eye(nl) - G * (Atr * dr * 2)[None, :]
            J[:-1, -1] = -g
            J[-1, :-1] = 4 * cm * Atr * dr
            J[-1, -1] = 2 * cmP
            dx = np.linalg.solve(J, -R)
            u, P = u + dx[:-1], P + dx[-1]
            if np.abs(dx[:-1]).max() < 1e-10 and abs(dx[-1]) < 1e-9 * max(abs(P), 1):
                break
        whist = np.maximum(whist, 2 * u)
        out.append((cmod, P))
    return np.array(out)


def test_notched_beam_against_a_cohesive_crack_model():
    """three-point bend beam with a 50 mm notch (RILEM-type, 100 × 100 mm, span 790): the crack band model (10 mm cells
    above the notch, band along the beam, implicit) reproduces an independent discrete cohesive-crack solution: peak load
    and the softening branch of the load–CMOD curve (the independent review found the same with uniform 10 and 5 mm
    meshes; IMPL-EX overshoots the peak by a few per cent unless the steps near it are small)"""
    c = ConcreteDamage.from_strength(38.0, units="N-mm", prepeak="linear", nu=0.2, implex=False)
    ref = cohesive_reference(c.E, c.nu, c.t_post)
    D, B = 100.0, 100.0
    xs = np.unique(np.r_[np.linspace(0, 30, 2), np.linspace(30, 330, 7), np.arange(330.0, 521.0, 10.0),
                         np.linspace(520, 820, 7), np.linspace(820, 850, 2)])
    blk = structured.hex_block(xs, [0.0, B], 0.0, D, 10)
    cen = blk.centroids()
    keep = ~((np.abs(cen[:, 0] - 425.0) < 5) & (cen[:, 2] < 50.0))  # the notch: one column of cells, 10 mm wide
    m = Mesh(blk.points, {"hex8": blk.cells["hex8"][keep]}).compact()
    cen = m.centroids()
    near = np.abs(cen[:, 0] - 425.0) < 100.0
    m.cell_groups = {"c": {"hex8": np.flatnonzero(near)}, "e": {"hex8": np.flatnonzero(~near)}}
    s = OpsSolid(m, {"c": c, "e": ElasticIso(c.E, c.nu)}, band=("projected", (1, 0, 0)))
    s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 30.0), comps=(0, 2))
    s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 820.0), comps=(2,))
    s.fix(None, nodes=[s.nearest((30.0, 0, 0))], comps=(1,))
    F = s.traction(lambda x: np.isclose(x[2], D) & (np.abs(x[0] - 425.0) <= 15.0 + 1e-6), (0, 0, -1.0 / (30.0 * B)))
    mR, mL = s.nearest((430.0, 0, 0)), s.nearest((420.0, 0, 0))
    st = s.push(F, mR, 0, 0.45, steps=120, record=False)  # the mouth's right edge (the beam also slides on its roller)
    assert st.converged
    U = np.array([h.U for h in s.history])
    cmod = U[:, mR, 0] - U[:, mL, 0]
    sup = s.nodes(lambda x: np.isclose(x[2], 0) & (np.isclose(x[0], 30.0) | np.isclose(x[0], 820.0)))
    P = np.array([s.reaction(sup, i)[2] for i in range(len(s.history))])
    assert cmod[-1] > 0.3 and P.max() == pytest.approx(ref[:, 1].max(), rel=0.02)  # here +0.1 %
    for w in (0.1, 0.15, 0.2, 0.25, 0.3):  # here within 1.3 %
        assert np.interp(w, cmod, P) == pytest.approx(np.interp(w, ref[:, 0], ref[:, 1]), rel=0.04)
