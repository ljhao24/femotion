"""Solid models of femotion.continuum beyond one material and one load: materials, elastic supports, embedded bars and
tendons, initial strains, load cases, sections, sweeps. Every check has an independent reference: a closed-form result
of elasticity or beam theory, a textbook formula, or an exact equilibrium statement."""
import numpy as np
import pytest

pytest.importorskip("skfem")
from femotion.continuum import Elastic, Isotropic, Orthotropic, Tendon, fillet_path, rect  # noqa: E402
from femotion.mesh import structured  # noqa: E402
from femotion.mesh.core import Mesh  # noqa: E402
from femotion.mesh.section import LayeredSection, sweep  # noqa: E402


def _groups_by(m: Mesh, **preds) -> Mesh:
    c = m.centroids()
    m.cell_groups = {k: {m.kind: np.where(p(c.T))[0]} for k, p in preds.items()}
    return m


def _box_supports(s, L):
    """statically determinate supports of a box [0, L] × [0, b] × [0, h]: no reactions to self-equilibrated loads"""
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.fix_point((0, 0, 0), comps=(1, 2))
    s.fix_point((0, 1, 0), comps=(2,))


# ── materials ────────────────────────────────────────────────────────────
def test_orthotropic_with_equal_constants_is_isotropic():
    E, nu = 3.0e10, 0.2
    G = E / (2 * (1 + nu))
    iso = Isotropic(E, nu)
    ort = Orthotropic(E=(E, E, E), nu=(nu, nu, nu), G=(G, G, G), axes=[[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1]])
    assert np.allclose(ort.D6(), iso.D6(), rtol=1e-12, atol=1e-3)


def test_orthotropic_plane_stress_textbook():
    """plane stress of an orthotropic lamina in its axes: Q11 = E1/(1 − ν12ν21), Q12 = ν12E2/(1 − ν12ν21), Q66 = G12"""
    E1, E2, n12, G12 = 165e9, 10e9, 0.3, 5e9
    m = Orthotropic(E=(E1, E2, E2), nu=(n12, n12, 0.4), G=(G12, G12, E2 / 2.8))
    n21 = n12 * E2 / E1
    Q = m.D(2, "plane_stress")
    assert Q[0, 0] == pytest.approx(E1 / (1 - n12 * n21), rel=1e-12)
    assert Q[1, 1] == pytest.approx(E2 / (1 - n12 * n21), rel=1e-12)
    assert Q[0, 1] == pytest.approx(n12 * E2 / (1 - n12 * n21), rel=1e-12)
    assert Q[2, 2] == pytest.approx(G12, rel=1e-12)


def test_isotropic_plane_strain_thermal_strain():
    """plane strain: the in-plane free strain of a temperature change is (1 + ν) α ΔT"""
    m = Isotropic(3e10, 0.2, alpha=1e-5)
    assert np.allclose(m.thermal_strain(2, "plane_strain"), [1.2e-5, 1.2e-5, 0], rtol=1e-12)
    assert np.allclose(m.thermal_strain(2, "plane_stress"), [1e-5, 1e-5, 0], rtol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 30.0, 90.0])
def test_rotated_orthotropic_block_uniaxial(theta):
    """off-axis tension of an orthotropic block: 1/Ex = c⁴/E1 + (1/G12 − 2ν12/E1) s²c² + s⁴/E2 (classical lamination
    theory), with the shear coupling free to develop"""
    E1, E2, n12, G12 = 140e9, 9e9, 0.3, 4.5e9
    t = np.radians(theta)
    c, s_ = np.cos(t), np.sin(t)
    mat = Orthotropic(E=(E1, E2, E2), nu=(n12, n12, 0.45), G=(G12, G12, E2 / 2.9),
                      axes=[[c, s_, 0], [-s_, c, 0], [0, 0, 1]])
    L = 2.0
    s = Elastic(structured.box(L, 1, 1, 4, 2, 2), material=mat)
    s.traction(lambda x: np.isclose(x[0], L), (1e6, 0, 0))
    _box_supports(s, L)
    s.solve()
    ux = s.nodal_displacement()[np.isclose(s.result_mesh().points[:, 0], L), 0].mean()
    Ex = 1 / (c**4 / E1 + (1 / G12 - 2 * n12 / E1) * s_**2 * c**2 + s_**4 / E2)
    assert ux / L == pytest.approx(1e6 / Ex, rel=1e-8)


def test_two_materials_in_series():
    """a bar of two materials in series: δ = σ (L1/E1 + L2/E2)"""
    m = _groups_by(structured.box(3.0, 1, 1, 6, 2, 2), a=lambda c: c[0] < 1.0, b=lambda c: c[0] > 1.0)
    s = Elastic(m, materials={"a": Isotropic(10e9, 0.0), "b": Isotropic(40e9, 0.0)})
    s.traction(lambda x: np.isclose(x[0], 3), (2e6, 0, 0))
    _box_supports(s, 3)
    s.solve()
    ux = s.nodal_displacement()[np.isclose(s.result_mesh().points[:, 0], 3), 0].mean()
    assert ux == pytest.approx(2e6 * (1 / 10e9 + 2 / 40e9), rel=1e-10)


def test_missing_material_raises():
    m = _groups_by(structured.box(1, 1, 1, 2, 2, 2), a=lambda c: c[0] < 0.5)
    with pytest.raises(ValueError, match="no material"):
        Elastic(m, materials={"a": Isotropic(1.0, 0.2)})


# ── assembly ─────────────────────────────────────────────────────────────
def test_chunked_assembly_equals_whole():
    m = structured.box(2, 1, 1, 6, 3, 3)
    K1 = Elastic(m, E=1.0, nu=0.3, order=2, serendipity=True).K
    K2 = Elastic(m, E=1.0, nu=0.3, order=2, serendipity=True, chunk=7).K
    assert abs(K1 - K2).max() < 1e-12 * abs(K1).max()


def test_serendipity_hex_cantilever():
    """20-node hexahedra, end-loaded cantilever: δ = PL³/3EI + PL/(κGA), κ = 5/6"""
    L, b, P, E, nu = 8.0, 1.0, 1.0, 1000.0, 0.3
    s = Elastic(structured.box(L, b, b, 16, 2, 2), E=E, nu=nu, order=2, serendipity=True)
    s.traction(lambda x: np.isclose(x[0], L), (0, 0, -P / b**2))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    tip = np.isclose(s.result_mesh().points[:, 0], L)
    I, G_ = b**4 / 12, E / (2 * (1 + nu))
    assert -s.nodal_displacement()[tip, 2].mean() == pytest.approx(P * L**3 / (3 * E * I) + P * L / (5 / 6 * G_ * b * b), rel=0.02)


# ── supports ─────────────────────────────────────────────────────────────
def test_block_on_area_springs():
    """a stiff block on an elastic bed under uniform pressure q: settlement q / k, bed force = load"""
    q, kz = 2e5, 5e7
    s = Elastic(structured.box(2, 1, 0.5, 4, 2, 2), E=1e12, nu=0.2)
    s.traction(lambda x: np.isclose(x[2], 0.5), (0, 0, -q))
    s.spring(s.facets(normal=(0, 0, -1)), k=(1e5, 1e5, kz), name="bed")
    s.solve()
    assert s.nodal_displacement()[:, 2].mean() == pytest.approx(-q / kz, rel=1e-4)
    assert s.result().spring_force("bed") == pytest.approx([0, 0, q * 2.0], rel=1e-9, abs=1e-9 * q)


def test_facets_by_normal_and_position():
    s = Elastic(structured.box(2, 1, 1, 4, 2, 2), E=1.0, nu=0.3)
    top = s.facets(normal=(0, 0, 1))
    assert s.facet_integral(top, lambda x: 1 + 0 * x[0]) == pytest.approx(2.0)
    part = s.facets(lambda x: x[0] < 1.0, normal=(0, 0, -1))
    assert s.facet_integral(part, lambda x: 1 + 0 * x[0]) == pytest.approx(1.0)


# ── load cases ───────────────────────────────────────────────────────────
def test_cases_one_factorisation_equal_separate_solves_and_combine():
    m = structured.box(4, 1, 1, 8, 2, 2)

    def model():
        s = Elastic(m, E=3e10, nu=0.2, order=2)
        s.fix(lambda x: np.isclose(x[0], 0))
        return s
    s = model()
    s.case("g").body_force((0, 0, -25e3))
    s.case("p").traction(lambda x: np.isclose(x[0], 4), (0, 1e5, -2e5))
    s.solve()
    for name, add in (("g", lambda c: c.body_force((0, 0, -25e3))),
                      ("p", lambda c: c.traction(lambda x: np.isclose(x[0], 4), (0, 1e5, -2e5)))):
        t = model()
        add(t.case())
        t.solve()
        assert np.allclose(s.result(name).u, t.u, rtol=1e-10, atol=1e-16)
    both = model()
    both.body_force((0, 0, -25e3))
    both.traction(lambda x: np.isclose(x[0], 4), (0, 1e5, -2e5))
    both.solve()
    r = s.combine({"g": 1.0, "p": 1.0})
    assert np.allclose(r.u, both.u, rtol=1e-10, atol=1e-16)
    assert np.allclose(r.nodal_stress(), both.nodal_stress(), rtol=1e-8, atol=1e-3)
    assert np.allclose((1.2 * s.result("g")).u, 1.2 * s.result("g").u)


def test_area_line_and_point_loads_resultants():
    """the resultant and moment of distributed loads are exact, wherever they cut through cells"""
    s = Elastic(structured.box(4, 2, 1, 5, 3, 2), E=1.0, nu=0.3, order=2)
    F = np.array([0.0, 0.0, -1e5])
    s.case("w").area_load(s.facets(normal=(0, 0, 1)), F, rect(1.13, 1.33, 0.27, 0.87))
    assert s.case("w").resultant() == pytest.approx(F, abs=1e-6)
    q = np.array([1e4, 0, 0])
    pts = np.array([[0.3, 0.37, 0.1], [0.3, 0.37, 0.9]])
    s.case("bolt").line_load(pts, q)
    assert s.case("bolt").resultant() == pytest.approx(q * 0.8, rel=1e-12)
    P = np.array([[1.234, 0.77, 0.31]])
    Fp = np.array([[3.0, -2.0, 5.0]])
    c = s.case("pt").point_forces(P, Fp)
    X = s.ib.doflocs.T[0::3]
    f = c.f.reshape(-1, 3)
    assert f.sum(axis=0) == pytest.approx(Fp[0], rel=1e-12)
    assert np.cross(X, f).sum(axis=0) == pytest.approx(np.cross(P[0], Fp[0]), rel=1e-10)


def test_sample_reproduces_a_linear_field():
    s = Elastic(structured.box(2, 1, 1, 3, 2, 2), E=100.0, nu=0.3, order=1)
    s.traction(lambda x: np.isclose(x[0], 2), (1.0, 0, 0))
    _box_supports(s, 2)
    s.solve()
    pts = np.array([[0.37, 0.21, 0.66], [1.9, 0.9, 0.05], [2.0, 1.0, 1.0]])
    r = s.result().sample(pts)
    assert np.allclose(r["stress"][:, 0], 1.0, rtol=1e-9)
    assert np.allclose(r["u"][:, 0], pts[:, 0] / 100.0, rtol=1e-9)


# ── initial strains ─────────────────────────────────────────────────────
def test_free_thermal_expansion_has_no_stress():
    s = Elastic(structured.box(2, 1, 1, 4, 2, 2), material=Isotropic(3e10, 0.2, alpha=1e-5), order=2)
    s.case().thermal(25.0)
    _box_supports(s, 2)
    s.solve()
    assert np.abs(s.nodal_stress()).max() < 1e-9 * 3e10 * 1e-5 * 25  # roundoff of E α ΔT
    ux = s.nodal_displacement()[np.isclose(s.result_mesh().points[:, 0], 2), 0]
    assert ux.mean() == pytest.approx(2 * 25e-5, rel=1e-10)


def test_bimetal_strip_curvature():
    """Timoshenko's bimetal strip: κ = 6 Δα ΔT (1 + m)² / (h [3 (1 + m)² + (1 + m n)(m² + 1/(m n))]), m = h1/h2, n = E1/E2"""
    L, h1, h2, E1, E2, a1, a2, dT = 1.0, 0.01, 0.015, 200e9, 70e9, 12e-6, 23e-6, 50.0
    m = _groups_by(structured.rect(L, h1 + h2, 100, 10, "quad4"), lo=lambda c: c[1] < h1, hi=lambda c: c[1] > h1)
    s = Elastic(m, materials={"lo": Isotropic(E1, 0.0, a1), "hi": Isotropic(E2, 0.0, a2)}, order=2)
    s.case().thermal(dT)
    s.fix_point((0, 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[0], 0), comps=(0,))
    s.solve()
    mm, n, h = h1 / h2, E1 / E2, h1 + h2
    kappa = 6 * (a2 - a1) * dT * (1 + mm) ** 2 / (h * (3 * (1 + mm) ** 2 + (1 + mm * n) * (mm**2 + 1 / (mm * n))))
    rm = s.result_mesh()
    tip = np.isclose(rm.points[:, 0], L) & np.isclose(rm.points[:, 1], 0)
    assert -s.nodal_displacement()[tip, 1][0] == pytest.approx(kappa * L**2 / 2, rel=0.01)  # curves towards the low-α layer


# ── embedded bars and tendons ───────────────────────────────────────────
def test_embedded_bar_adds_axial_stiffness():
    """a prism with a bonded bar along it, stretched by δ: N = (E A + Es As) δ / L (the bar spans whole and part cells)"""
    L, E, EsAs, d = 3.0, 1e9, 4e8, 1e-3
    s = Elastic(structured.box(L, 1, 1, 6, 2, 2), E=E, nu=0.25, order=2)
    s.bar([(0, 0.33, 0.61), (L, 0.33, 0.61)], EA=EsAs, name="rebar")
    s.fix(lambda x: np.isclose(x[0], L), comps=(0,), value=d)
    _box_supports(s, L)
    s.solve()
    for method in ("nodal", "traction"):  # the cut force of solid and bar together
        assert s.result().section((L / 2, 0, 0), (1, 0, 0), method=method)["N"] == pytest.approx((E * 1.0 + EsAs) * d / L, rel=1e-9)
    assert s.result().bar_force("rebar") == pytest.approx(np.full(len(s.bars["rebar"].L), EsAs * d / L), rel=1e-9)


def test_unbonded_bar_one_force_over_its_length():
    """a bar attached at its ends and at a frictionless deviator carries one force; under a uniform stretch ε of the solid
    its force is EA ε (sum of the projected elongations of both runs) / total length × (run lengths) — here exactly EA ε
    cos² weighted: checked against the closed form Σ Li cos²θi ε EA / ΣLi"""
    L, d, EA = 4.0, 2e-3, 1.0  # a bar too soft to disturb the uniform strain of the solid
    s = Elastic(structured.box(L, 1, 1, 8, 2, 2), E=1e9, nu=0.0, order=1)
    pts = [(0.0, 0.5, 0.8), (2.0, 0.5, 0.2), (4.0, 0.5, 0.8)]
    b = s.bar(pts, EA=EA, bonded=False, name="ext")
    s.fix(lambda x: np.isclose(x[0], L), comps=(0,), value=d)
    _box_supports(s, L)
    s.solve()
    runs = np.diff(np.asarray(pts), axis=0)
    Li = np.linalg.norm(runs, axis=1)
    elong = np.sum(Li * (runs[:, 0] / Li) ** 2) * d / L  # ε_xx = d/L, other strains 0 (ν = 0)
    assert len(b.L) == 1 and b.L[0] == pytest.approx(Li.sum())
    assert s.result().bar_force("ext")[0] == pytest.approx(EA * elong / Li.sum(), rel=1e-9)


def test_fillet_path_geometry():
    """a 90° bend with radius R: tangent length R, arc length πR/2, path through the tangent points"""
    R = 2.0
    p = fillet_path([(0, 0, 0), (5, 0, 0), (5, 5, 0)], [R], arc_step=0.01)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    assert seg == pytest.approx((5 - R) * 2 + np.pi * R / 2, rel=1e-4)
    assert np.any(np.all(np.isclose(p, (3, 0, 0)), axis=1)) and np.any(np.all(np.isclose(p, (5, 2, 0)), axis=1))
    with pytest.raises(ValueError):
        fillet_path([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [5.0])


def test_tendon_friction_and_equilibrium():
    """friction from both ends: P(x) = P0 e^{−(μθ + kx)}; the loads on the concrete add up to zero; the radial push of an
    arc adds up to 2 P sin(Δ/2) along its bisector"""
    td = Tendon.from_pis([(0, 0, 1.0), (4, 0, 0.1), (12, 0, 0.1), (16, 0, 1.0)], [10.0, 10.0], arc_step=0.01)
    tf = td.with_friction(1000e3, mu=0.25, k=0.0015, ends="both")
    P = tf.forces()
    L = tf.lengths()
    x = np.cumsum(L) - L / 2
    th = np.arctan(0.9 / 4)
    mid = np.argmin(abs(x - 8))
    assert P[mid] == pytest.approx(1000e3 * np.exp(-(0.25 * th + 0.0015 * x[mid])), rel=2e-3)
    assert P[0] == pytest.approx(1000e3 * np.exp(-0.0015 * x[0]), rel=1e-12)
    assert np.allclose(P, P[::-1], rtol=1e-6)
    F = Tendon(td.points, 1000e3).vertex_forces()
    assert np.abs(F.sum(axis=0)).max() < 1e-6
    turn = np.r_[0, td.turn_angles(), 0] > 1e-9
    first_arc = np.where(turn)[0]
    first_arc = first_arc[first_arc < len(td.points) // 2]
    Fa = F[first_arc].sum(axis=0)
    assert np.linalg.norm(Fa) == pytest.approx(2 * 1000e3 * np.sin(th / 2), rel=1e-3)


def _beam(L, b, h, nx, ny, nz, order=2):
    return Elastic(structured.box(L, b, h, nx, ny, nz), E=3e10, nu=0.2, order=order, serendipity=True)


def test_straight_eccentric_tendon_beam_theory():
    """a straight tendon at eccentricity e below the centroid of a statically determinate beam: σ = −P/A ∓ P e y / I
    (Saint-Venant: away from the anchors), no reactions"""
    L, b, h, P, e = 8.0, 0.4, 0.6, 1.0e6, 0.15
    s = _beam(L, b, h, 16, 2, 4)
    s.case().tendon(Tendon([(0, b / 2, h / 2 - e), (L, b / 2, h / 2 - e)], P), anchor=(0.2, 0.2))
    s.fix(lambda x: np.isclose(x[0], 0) & np.isclose(x[2], 0), comps=(0, 2))
    s.fix(lambda x: np.isclose(x[0], L) & np.isclose(x[2], 0), comps=(2,))
    s.fix_point((0, 0, 0), comps=(1,))
    s.solve()
    A, I = b * h, b * h**3 / 12
    r = s.result().sample([(L / 2, b / 2, 0.0), (L / 2, b / 2, h)])
    assert r["stress"][0, 0] == pytest.approx(-P / A - P * e * (h / 2) / I, rel=5e-3)
    assert r["stress"][1, 0] == pytest.approx(-P / A + P * e * (h / 2) / I, rel=5e-3)
    assert np.abs(s.reactions()).max() < 1e-6 * P


def test_parabolic_tendon_load_balancing_and_section():
    """a parabolic tendon (sag f, anchored at the centroid) on a simply supported beam: upward w = 8 P f / L², camber
    5 w L⁴ / (384 E I) + w L² / (8 κ G A) (shear); midspan section: N = −P, M = −P f (about the centroid)"""
    L, b, h, P, f = 10.0, 0.3, 0.5, 1.0e6, 0.15
    xs = np.linspace(0, L, 201)
    zs = h / 2 - f * 4 * xs / L * (1 - xs / L)
    td = Tendon(np.c_[xs, np.full_like(xs, b / 2), zs], P)
    s = _beam(L, b, h, 20, 2, 4)
    s.case().tendon(td, anchor=(0.2, 0.2))
    s.fix(lambda x: np.isclose(x[0], 0) & np.isclose(x[2], h / 2), comps=(0, 2))
    s.fix(lambda x: np.isclose(x[0], L) & np.isclose(x[2], h / 2), comps=(2,))
    s.fix_point((0, 0, h / 2), comps=(1,))
    s.solve()
    E, nu = 3e10, 0.2
    I, A, G = b * h**3 / 12, b * h, E / (2 * (1 + nu))
    w = 8 * P * f / L**2
    camber = 5 * w * L**4 / (384 * E * I) + w * L**2 / (8 * 5 / 6 * G * A)
    uz = s.result().sample([(L / 2, b / 2, h / 2)])["u"][0, 2]
    assert uz == pytest.approx(camber, rel=0.02)
    sec = s.result().section((L / 2, 0, 0), (1, 0, 0))
    assert sec["N"] == pytest.approx(-P, rel=1e-3)  # the tendon is horizontal at midspan (P varies along it: arc sampling)
    assert sec["M"][1] == pytest.approx(P * f, rel=2e-3)
    tr = s.result().section((L / 2, 0, 0), (1, 0, 0), method="traction")
    assert tr["N"] == pytest.approx(-P, rel=0.01) and tr["M"][1] == pytest.approx(P * f, rel=0.01)  # +y moment = hogging here: the tendon below the centroid lifts the beam
    assert sec["area"] == pytest.approx(A, rel=1e-12)


def test_cantilever_section_forces():
    """a cut through an end-loaded cantilever carries V = P and M = P (L − a)"""
    L, b, P = 6.0, 1.0, 1e4
    s = Elastic(structured.box(L, b, b, 24, 4, 4), E=3e10, nu=0.2, order=2, serendipity=True)
    s.traction(lambda x: np.isclose(x[0], L), (0, 0, -P / b**2))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    for a in (1.5, 3.0):
        sec = s.result().section((a, 0, 0), (1, 0, 0))  # statics: exact
        assert sec["F"][2] == pytest.approx(-P, rel=1e-9)
        assert sec["M"][1] == pytest.approx(P * (L - a), rel=1e-9)
        assert abs(sec["N"]) < 1e-9 * P
        tr = s.result().section((a, 0, 0), (1, 0, 0), method="traction")  # the stress field: O(h²) close (10 % at 12 cells)
        assert tr["F"][2] == pytest.approx(-P, rel=0.03)
        assert tr["M"][1] == pytest.approx(P * (L - a), rel=0.005)
        top = s.result().section((a, 0, 0), (1, 0, 0), method="traction", region=lambda x: x[2] > b / 2)
        assert top["N"] == pytest.approx(P * (L - a) / (b**3 / 12) * (b / 4) * (b * b / 2), rel=0.01)  # σ = M y / I


# ── meshes ──────────────────────────────────────────────────────────────
def _box_section():
    """a single-cell box 2.0 × 1.0 with 0.2 webs and slabs"""
    sec = LayeredSection(y=[0, 0.2, 1.8, 2.0], ny=[2, 8, 2], nz=[2, 6, 2], solid=[[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                         names=[["web", "web", "web"], ["bottom", None, "top"], ["web", "web", "web"]])
    return sec, (lambda x, y: [0 * y, 0.2 + 0 * y, 0.8 + 0 * y, 1.0 + 0 * y])


def test_sweep_box_volume_and_groups():
    sec, z = _box_section()
    m = sweep(sec, np.linspace(0, 5, 11), z)
    assert m.measure().sum() == pytest.approx((2.0 * 1.0 - 1.6 * 0.6) * 5, rel=1e-12)
    assert set(m.cell_groups) == {"web", "bottom", "top"}
    vol = {g: m.measure()[c["hex8"]].sum() for g, c in m.cell_groups.items()}
    assert vol["bottom"] == pytest.approx(1.6 * 0.2 * 5) and vol["web"] == pytest.approx(2 * 0.2 * 1.0 * 5)
    sub, cells, nodes = m.subset(["web", "bottom"])
    assert sub.measure().sum() == pytest.approx(vol["web"] + vol["bottom"])
    assert np.allclose(sub.points, m.points[nodes]) and len(cells) == sub.n_cells()


def test_swept_box_cantilever_beam_theory():
    """a thin-walled box cantilever under an end shear: δ = P L³ / (3 E I) + P L / (G A_web) (shear through the webs)"""
    sec, z = _box_section()
    L, P, E, nu = 12.0, 1e5, 3e10, 0.2
    s = Elastic(sweep(sec, np.linspace(0, L, 25), z), E=E, nu=nu, order=2, serendipity=True)
    s.traction(lambda x: np.isclose(x[0], L), (0, 0, -P / (2.0 - 1.6 * 0.6)))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    I = (2.0 * 1.0**3 - 1.6 * 0.6**3) / 12
    Aw = 2 * 0.2 * 1.0
    d = P * L**3 / (3 * E * I) + P * L / (E / (2 * (1 + nu)) * Aw)
    tip = s.result().sample([(L, 0.1, 0.5)])["u"][0, 2]
    assert -tip == pytest.approx(d, rel=0.03)


def test_section_with_varying_interfaces_is_valid():
    """a thickening near x = 0 (as at a girder end): all cells positive, volume as integrated by hand"""
    sec, _ = _box_section()
    z = lambda x, y: [0 * y, (0.2 + 0.2 * max(0.0, 1 - x)) + 0 * y, 0.8 + 0 * y, 1.0 + 0 * y]  # noqa: E731
    m = sweep(sec, np.linspace(0, 3, 31), z)
    v = m.measure()
    assert v.min() > 0
    # void: 1.6 wide × (0.6 − 0.2·(1 − x)) over x ∈ [0, 1], 1.6 × 0.6 beyond
    void = 1.6 * (0.6 * 3 - 0.2 * 0.5)
    assert v.sum() == pytest.approx(2.0 * 1.0 * 3 - void, rel=1e-9)


def test_export_vtu(tmp_path):
    pytest.importorskip("meshio")
    from femotion.continuum import export_vtu
    s = Elastic(structured.box(2, 1, 1, 2, 1, 1), material=Isotropic(1e9, 0.2, 1e-5))
    s.case("t").thermal(10.0)
    s.case("g").body_force((0, 0, -1e3))
    _box_supports(s, 2)
    s.solve()
    p = export_vtu(tmp_path / "r.vtu", s, {"t": s.result("t"), "sum": s.result("t") + s.result("g")})
    import meshio
    m = meshio.read(p)
    assert {"u:t", "S:sum", "S123:sum", "dir1:t"} <= set(m.point_data)


# ── prescribed displacements, submodels, solvers ────────────────────────
def test_prescribed_displacement_per_case():
    """two cases stretch a bar by different amounts: reactions in proportion, E A δ / L"""
    s = Elastic(structured.box(2, 1, 1, 4, 2, 2), E=1e9, nu=0.0, order=2)
    _box_supports(s, 2)
    s.case("a").displacement(lambda X: np.isclose(X[0], 2), (1e-3, 0, 0), comps=(0,))
    s.case("b").displacement(lambda X: np.isclose(X[0], 2), (3e-3, 0, 0), comps=(0,))
    s.solve()
    for name, d in (("a", 1e-3), ("b", 3e-3)):
        assert s.result(name).section((1.0, 0, 0), (1, 0, 0))["N"] == pytest.approx(1e9 * d / 2, rel=1e-9)


def test_submodel_driven_by_a_coarse_model():
    """the middle of a cantilever, meshed finer and driven by the coarse model's displacements on its cuts (with its own
    self-weight), gives the bending stress of beam theory, and the same as the coarse model inside"""
    from femotion.continuum.submodel import drive
    L, b, h, g = 6.0, 0.5, 0.5, 25e3
    coarse = Elastic(structured.box(L, b, h, 12, 2, 2), E=3e10, nu=0.2, order=2, serendipity=True)
    coarse.fix(lambda x: np.isclose(x[0], 0))
    coarse.case("g").body_force((0, 0, -g))
    coarse.solve()
    part = structured.box(1.0, b, h, 8, 4, 4)
    part.points[:, 0] += 2.5
    sub = Elastic(part, E=3e10, nu=0.2, order=2, serendipity=True)
    drive(sub, coarse.result("g"), lambda X: np.isclose(X[0], 2.5) | np.isclose(X[0], 3.5), "g")
    sub.case("g").body_force((0, 0, -g))
    sub.solve()
    x = 3.1  # inside cells of both meshes (on a face the stress of either neighbour may be picked)
    M = g * b * h * (L - x) ** 2 / 2  # hogging
    top = M * (h / 2) / (b * h**3 / 12)
    p = [(x, 0.2, h)]
    assert sub.result("g").sample(p)["stress"][0, 0] == pytest.approx(top, rel=0.01)
    assert sub.result("g").sample(p)["stress"][0, 0] == pytest.approx(coarse.result("g").sample(p)["stress"][0, 0], rel=0.01)


def test_cholesky_matches_direct():
    pytest.importorskip("cholespy")
    m = structured.box(3, 1, 1, 6, 2, 2)
    out = {}
    for method in ("direct", "cholesky"):
        s = Elastic(m, E=3e10, nu=0.2, order=2, serendipity=True)
        s.spring(s.facets(normal=(0, 0, -1)), k=(1e7, 1e7, 1e9))
        s.case("g").body_force((0, 0, -25e3))
        s.case("p").traction(lambda x: np.isclose(x[0], 3), (1e5, 0, 0))
        s.solve(method=method)
        out[method] = (s.result("g").u, s.result("p").u)
    for a, b in zip(*out.values()):
        assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()


# ── regressions of the independent review (shared/refs/femotion/CONTINUUM_REVIEW.md in the authors' repository) ──
def _prism(L=2.0, nx=4, order=2, **kw):
    return Elastic(structured.box(L, 1, 1, nx, 2, 2), order=order, serendipity=order == 2, **kw)


@pytest.mark.parametrize("method", ["nodal", "traction"])
def test_section_with_initial_strains_and_bar_prestress(method):
    """initial strains and bar prestress are internal: free expansion N = 0; restrained −E α ΔT A; free bending under a
    linear temperature M = 0; a free prism with a tensioned bonded bar: concrete + bar N = 0"""
    E, a, dT = 3e10, 1e-5, 20.0
    s = _prism(material=Isotropic(E, 0.0, a))
    _box_supports(s, 2.0)
    s.case("free").thermal(dT)
    s.case("bend").thermal(lambda x: dT * (x[2] - 0.5))
    s.solve()
    for name in ("free", "bend"):
        sec = s.result(name).section((1.0, 0, 0), (1, 0, 0), method=method)
        assert abs(sec["N"]) < 1e-6 * E * a * dT and np.abs(sec["M"]).max() < 1e-6 * E * a * dT
    r = _prism(material=Isotropic(E, 0.0, a))
    r.fix(lambda X: np.isclose(X[0], 0) | np.isclose(X[0], 2.0), comps=(0,))
    r.fix_point((0, 0, 0), comps=(1, 2))
    r.fix_point((0, 1, 0), comps=(2,))
    r.case("hot").thermal(dT)
    r.solve()
    assert r.result("hot").section((1.0, 0, 0), (1, 0, 0), method=method)["N"] == pytest.approx(-E * a * dT * 1.0, rel=1e-6)
    b = _prism(E=E, nu=0.2)
    bar = b.bar([(0.0, 0.5, 0.5), (2.0, 0.5, 0.5)], EA=2e8, name="t")
    _box_supports(b, 2.0)
    b.case("P").bar_prestress(bar, 1e6)
    b.solve()
    # statics: exact; the stress field near the bar's anchored ends converges more slowly
    assert abs(b.result("P").section((1.0, 0, 0), (1, 0, 0), method=method)["N"]) < (1e-6 if method == "nodal" else 1e-2) * 1e6


def test_bar_prestress_elastic_shortening():
    """a bonded bar tensioned to P in a free prism keeps N = P (1 − EAb / (EAc + EAb)) (Poisson's ratio 0: uniaxial)"""
    E, A, EAb, P = 3e10, 1.0, 3e9, 1e6
    s = _prism(L=8.0, nx=16, E=E, nu=0.0)  # long enough for the anchored ends to fade out (Saint-Venant)
    bar = s.bar([(0.0, 0.5, 0.5), (8.0, 0.5, 0.5)], EA=EAb, name="t")
    _box_supports(s, 8.0)
    s.case("P").bar_prestress(bar, P)
    s.solve()
    N = s.result("P").bar_force("t")
    assert N[len(N) // 2] == pytest.approx(P * (1 - EAb / (E * A + EAb)), rel=1e-5)


def test_embedded_bars_bending_stiffness():
    """two bonded bars at ±e double the bending stiffness when 2 EA e² = E I: the tip deflection halves"""
    L, b, E = 10.0, 1.0, 1e9
    I = b**4 / 12
    e = 0.4
    EA = E * I / (2 * e**2)
    tips = []
    for bars in (False, True):
        s = Elastic(structured.box(L, b, b, 20, 2, 2), E=E, nu=0.0, order=2, serendipity=True)
        if bars:
            s.bar([(0, 0.5, 0.5 - e), (L, 0.5, 0.5 - e)], EA=EA)
            s.bar([(0, 0.5, 0.5 + e), (L, 0.5, 0.5 + e)], EA=EA)
        s.traction(lambda x: np.isclose(x[0], L), (0, 0, -1e3))
        s.fix(lambda x: np.isclose(x[0], 0))
        s.solve()
        tips.append(s.result().sample([(L, 0.5, 0.5)])["u"][0, 2])
    assert tips[0] / tips[1] == pytest.approx(2.0, rel=0.02)  # shear deformation is not doubled: a little below 2


def test_constrained_thermal_stress():
    """a block held on every face: σ = −E α ΔT / (1 − 2ν) in each direction"""
    E, nu, a, dT = 3e10, 0.2, 1e-5, 30.0
    s = Elastic(structured.box(1, 1, 1, 2, 2, 2), material=Isotropic(E, nu, a), order=2)
    s.case().thermal(dT)
    s.fix(lambda X: np.isclose(X[0], 0) | np.isclose(X[0], 1), comps=(0,))
    s.fix(lambda X: np.isclose(X[1], 0) | np.isclose(X[1], 1), comps=(1,))
    s.fix(lambda X: np.isclose(X[2], 0) | np.isclose(X[2], 1), comps=(2,))
    s.solve()
    S = s.nodal_stress()
    assert np.allclose(S[:, :3], -E * a * dT / (1 - 2 * nu), rtol=1e-8)


@pytest.mark.parametrize("serendipity", [False, True])
def test_section_shear_under_self_weight(serendipity):
    """a cut through a cantilever under its own weight carries V = w (L − a) exactly by statics (the cells next to the
    cut give their share of the load on the nodes in it)"""
    L, w = 6.0, 25e3
    s = Elastic(structured.box(L, 1, 1, 24, 2, 2), E=3e10, nu=0.2, order=2 if serendipity else 1, serendipity=serendipity)
    s.body_force((0, 0, -w))
    s.fix(lambda x: np.isclose(x[0], 0))
    s.solve()
    sec = s.result().section((3.0, 0, 0), (1, 0, 0))
    assert sec["F"][2] == pytest.approx(-w * (L - 3.0), rel=1e-9)
    assert sec["M"][1] == pytest.approx(w * (L - 3.0) ** 2 / 2, rel=1e-9)


def test_unbonded_bar_in_traction_section():
    L, d, E, EA = 4.0, 1e-3, 1e9, 1e8
    s = Elastic(structured.box(L, 1, 1, 8, 2, 2), E=E, nu=0.0, order=1)
    s.bar([(0.0, 0.5, 0.5), (L, 0.5, 0.5)], EA=EA, bonded=False, name="ext")
    s.fix(lambda x: np.isclose(x[0], L), comps=(0,), value=d)
    _box_supports(s, L)
    s.solve()
    for method in ("nodal", "traction"):
        assert s.result().section((2.0, 0, 0), (1, 0, 0), method=method)["N"] == pytest.approx((E + EA) * d / L, rel=1e-9)


def test_section_through_cells_raises():
    s = _prism(E=1.0, nu=0.3)
    _box_supports(s, 2.0)
    s.traction(lambda x: np.isclose(x[0], 2.0), (1.0, 0, 0))
    s.solve()
    with pytest.raises(ValueError, match="cuts through cells"):
        s.result().section((0.8, 0, 0), (1, 0, 0))


def test_scale_load_keeps_initial_strains():
    """scaling the external load of a case leaves its temperature alone"""
    E, a = 3e10, 1e-5
    s = _prism(material=Isotropic(E, 0.0, a))
    _box_supports(s, 2.0)
    c = s.case().thermal(10.0).traction(lambda x: np.isclose(x[0], 2.0), (1e5, 0, 0))
    c.scale_load(2e5, (1, 0, 0))
    s.solve()
    assert np.allclose(s.nodal_stress()[:, 0], 2e5, rtol=1e-8)


def test_single_case_api_does_not_create_cases():
    s = _prism(E=1.0, nu=0.3)
    _box_supports(s, 2.0)
    s.case("g").traction(lambda x: np.isclose(x[0], 2.0), (1.0, 0, 0))
    assert not np.any(s.f) and not np.any(s.resultant())
    s.solve()
    assert list(s.results) == ["g"] and np.any(s.u)
    s.case("h").traction(lambda x: np.isclose(x[0], 2.0), (2.0, 0, 0))
    s.solve()
    with pytest.raises(ValueError, match="several load cases"):
        s.reactions()


def test_fix_value_belongs_to_one_case():
    """a settlement given with fix(value=…) is part of the case "default"; combining other cases does not repeat it"""
    s = _prism(E=1e9, nu=0.0)
    _box_supports(s, 2.0)
    s.fix(lambda X: np.isclose(X[0], 2.0), comps=(0,), value=1e-3)
    s.case("a").body_force((0, 0, -1.0))
    s.case("b").body_force((0, 0, -2.0))
    s.solve()
    end = np.isclose(s.ib.doflocs[0], 2.0) & (np.arange(s.ib.N) % 3 == 0)
    assert np.allclose(s.result("default").u[end], 1e-3)
    assert np.allclose(s.combine({"a": 1, "b": 1}).u[end], 0.0)


def test_fillet_overlap_raises():
    with pytest.raises(ValueError, match="need"):
        fillet_path([(0, 0, 0), (2, 0, 0), (2.2, 2, 0), (4.2, 2, 0)], [3.0, 3.0])


def test_plane_strain_initial_strain_like_temperature():
    """in plane strain a 3D free strain α ΔT (1, 1, 1, 0, 0, 0) acts as the temperature does"""
    m = structured.rect(2, 1, 4, 2, "quad4")
    out = []
    for how in ("thermal", "strain"):
        s = Elastic(m, material=Isotropic(3e10, 0.25, 1e-5), kind="plane_strain", order=2)
        c = s.case()
        c.thermal(10.0) if how == "thermal" else c.initial_strain([1e-4, 1e-4, 1e-4, 0, 0, 0])
        s.fix(lambda X: np.isclose(X[0], 0) | np.isclose(X[0], 2), comps=(0,))
        s.fix_point((0, 0), comps=(1,))
        s.solve()
        out.append(s.nodal_stress())
    assert np.allclose(out[0], out[1], rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize("kind,order,ser", [("hex8", 2, True), ("hex8", 1, False), ("tet4", 2, False), ("quad4", 2, True)])
def test_batched_assembly_matches_scikit_fem(kind, order, ser):
    """the B-matrix assembly against scikit-fem's own BilinearForm (fourth-order tensor C_ijkl ε_kl ε_ij)"""
    from skfem import Basis, BilinearForm, asm
    from skfem.helpers import sym_grad
    from femotion.continuum.materials import _tensor
    if kind == "tet4":
        G = pytest.importorskip("femotion.mesh.gmsh")
        pytest.importorskip("gmsh")
        m = G.solid(lambda occ: occ.addBox(0, 0, 0, 2, 1, 1), size=0.5)
    elif kind == "quad4":
        m = structured.rect(2, 1, 4, 2, "quad4")
    else:
        m = structured.box(2, 1, 1, 3, 2, 2)
    mat = Orthotropic(E=(140e9, 9e9, 7e9), nu=(0.3, 0.25, 0.45), G=(4.5e9, 4e9, 3e9),
                      axes=[[0.8, 0.36, 0.48], [-0.6, 0.48, 0.64], [0, -0.8, 0.6]] if kind != "quad4" else np.eye(3))
    s = Elastic(m, material=mat, order=order, serendipity=ser, kind="plane_strain" if kind == "quad4" else None)
    C = _tensor(mat.D6())
    if kind == "quad4":
        C = C[:2, :2, :2, :2]

    @BilinearForm
    def a(u, v, w):
        eu, ev = sym_grad(u), sym_grad(v)
        d = eu.shape[0]
        return sum(C[i, j, k, l] * eu[k, l] * ev[i, j] for i in range(d) for j in range(d) for k in range(d) for l in range(d))
    K0 = asm(a, Basis(s.sk, s.elem))
    assert abs(K0 - s.Kc).max() < 1e-12 * abs(K0).max()


def test_locator_finds_points_in_elongated_cells():
    """a strongly sheared, elongated hexahedral mesh: every point inside is found (checked against a brute-force search)"""
    m = structured.box(10.0, 0.2, 0.2, 2, 6, 6)
    m.points[:, 1] += 0.8 * m.points[:, 0] * m.points[:, 2]
    s = Elastic(m, E=1.0, nu=0.3)
    rng = np.random.default_rng(1)
    c = rng.integers(0, s.sk.t.shape[1], 300)
    X = rng.random((3, 300))
    P = s.sb.mapping.F(X[:, :, None], tind=c)[:, :, 0].T
    cells, Xr = s.locator.locate(P)
    assert np.allclose(s.sb.mapping.F(Xr[:, :, None], tind=cells)[:, :, 0].T, P, atol=1e-9)
    assert s.locator.inside(P).all()
    assert not s.locator.inside([(5.0, -3.0, 0.1), (11.0, 0.1, 0.1)]).any()


def test_area_load_point_of_application():
    """a small print inside large facets: its centre is placed right (to the sampling of the print)"""
    s = Elastic(structured.box(4, 4, 1, 4, 4, 2), E=1.0, nu=0.3, order=2)
    c = s.case("w").area_load(s.facets(normal=(0, 0, 1)), (0, 0, -1.0), rect(1.31, 1.51, 2.23, 2.83))
    X = s.ib.doflocs.T[0::3]
    f = c.f.reshape(-1, 3)
    xc = (X[:, 0] * f[:, 2]).sum() / f[:, 2].sum()
    yc = (X[:, 1] * f[:, 2]).sum() / f[:, 2].sum()
    assert abs(xc - 1.41) < 1e-3 and abs(yc - 2.53) < 1e-3
    assert c.resultant() == pytest.approx([0, 0, -1.0], abs=1e-12)
