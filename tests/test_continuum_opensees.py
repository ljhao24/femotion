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
    the steel stress in the constant-moment zone approaches the cracked-section value; crack widths are of the order of
    JTG 3362-2018 6.4.3 (Wcr), Eurocode 2 (wk) and fib MC2010; the cracks are 100–200 mm apart"""
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
    w_jtg = sls.crack_width(sec, Ms=M / 1e6, Ml=M / 1e6, env="I").demand
    cr = s.cracks_along((150, 50, 1), (1300, 50, 1), step=i)
    assert 0.4 < cr["max_width"] / w_jtg < 1.2
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
