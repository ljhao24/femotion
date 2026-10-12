"""Free (initial) strains in the OpenSees solid back-end — shrinkage, temperature, rust — and bond-slip of bars: closed-form
elastic checks, the scikit-fem solver on the same mesh, Timoshenko's bimetal strip, restrained-shrinkage cracking against
the crack band law and the tension chord model, and rust expansion around a bar against Lamé's cylinder and Tepfers'
ring models."""
import math

import numpy as np
import pytest

pytest.importorskip("openseespy")
pytest.importorskip("skfem")
from femotion.continuum import Elastic  # noqa: E402
from femotion.continuum.corrosion import (alonso_1998, depth_for_expansion, depth_from_mass_loss,  # noqa: E402
                                          el_maaddawy_soudki_2007, free_expansion, gbt51355_2019, lame_cylinder,
                                          liu_weyers_1998, mass_loss, rust_strain, tepfers)
from femotion.continuum.opensees import (BondSlip, ConcreteDamage, ElasticIso, Eigenstrain, OpsSolid, elastic,  # noqa: E402
                                         steel)
from femotion.mesh import structured  # noqa: E402
from femotion.mesh.inclusions import bar_block, bar_section, cylinder, hoop_normals  # noqa: E402


def statically_determinate(s: OpsSolid, L: float, b: float):
    """six support components on three corners of the face z = 0: no restraint against free strains"""
    s.fix(lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0) & np.isclose(x[2], 0), (0, 1, 2))
    s.fix(lambda x: np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0), (1, 2))
    s.fix(lambda x: np.isclose(x[0], 0) & np.isclose(x[1], b) & np.isclose(x[2], 0), (2,))


# ── elastic: closed forms and the scikit-fem solver ─────────────────────────
def test_free_expansion_is_stress_free():
    """a uniform temperature change of a statically supported block (with a bar of the same α) moves it by α ΔT x and
    stresses nothing; the material strain is zero and eps0 records the free strain"""
    m = structured.box(1.0, 0.4, 0.3, 6, 3, 2)
    s = OpsSolid(m, ElasticIso(30e3, 0.2, alpha=1e-5))
    s.bar([(0.1, 0.2, 0.15), (0.9, 0.2, 0.15)], 1e-3, elastic(200e3), alpha=1e-5)
    statically_determinate(s, 1.0, 0.4)
    s.static(strain=s.thermal(20.0), steps=2)
    h = s.history[-1]
    sig = 30e3 * 2e-4  # the stress full restraint would give (×(1 − 2ν)): round-off of the bar's penalty ties only
    assert np.abs(h.cell["stress"]).max() < 1e-6 * sig
    assert np.abs(h.cell["strain"]).max() < 1e-6 * 2e-4
    assert np.allclose(h.cell["eps0"], [2e-4, 2e-4, 2e-4, 0, 0, 0])
    assert np.abs(h.bar["N"]).max() < 1e-6 * 200e3 * 1e-3 * 2e-4 and np.allclose(h.bar["eps0"], 2e-4)
    assert h.U[s.nearest((1.0, 0.4, 0.3))] == pytest.approx(2e-4 * np.array([1.0, 0.4, 0.3]), rel=1e-6)


def test_full_restraint():
    """all faces held normal to themselves: σ = −E ε0 / (1 − 2ν) in every direction; an orthotropic free strain gives
    σ = −D ε0"""
    m = structured.box(1.0, 0.4, 0.3, 4, 2, 2)
    E, nu = 30e3, 0.2
    s = OpsSolid(m, ElasticIso(E, nu))
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1.0), (0,))
    s.fix(lambda x: np.isclose(x[1], 0) | np.isclose(x[1], 0.4), (1,))
    s.fix(lambda x: np.isclose(x[2], 0) | np.isclose(x[2], 0.3), (2,))
    s.static(strain=s.initial_strain(1e-4), steps=1)
    S = s.history[-1].cell["stress"]
    assert np.allclose(S[:, :3], -E * 1e-4 / (1 - 2 * nu), rtol=1e-9) and np.abs(S[:, 3:]).max() < 1e-9
    e0 = np.array([1e-4, -2e-4, 5e-5, 0, 0, 0])
    s.static(strain=s.initial_strain(e0) + s.initial_strain(-1e-4), steps=1)  # second stage: the sum of both
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    D = lam * np.ones((3, 3)) + 2 * mu * np.eye(3)
    assert s.history[-1].cell["stress"][0, :3] == pytest.approx(-D @ e0[:3], rel=1e-9)


@pytest.mark.parametrize("concrete", [False, True])
def test_initial_strain_matches_scikit_fem(concrete):
    """a free strain field varying inside the cells (6 components, evaluated at the integration points) and a
    temperature field give the scikit-fem order-1 solution on the same mesh (femotion.continuum.Elastic, which applies
    them as the equivalent loads ∫Bᵀ D ε0): same element, same quadrature — round-off; ASDConcrete3D below cracking to
    its own small departures from linearity (its stresses differ by ~1e-4 even without free strains)"""
    m = structured.box(1.0, 0.4, 0.3, 8, 3, 3)
    E, nu = 30e3, 0.2
    f = lambda x: np.stack([1e-5 * x[0], -2e-5 * x[1] * x[2], 5e-6 + 0 * x[0], 3e-6 * x[0], 0 * x[0], 1e-6 * x[2]])  # noqa: E731
    T = lambda x: 10.0 * x[2] / 0.3 + 2.0 * x[0]  # noqa: E731
    for case in ("strain", "thermal"):
        e = Elastic(m, E=E, nu=nu, order=1)
        e.fix(lambda x: np.isclose(x[0], 0))
        if case == "strain":
            e.case("default").initial_strain(f)
        else:
            e.case("default").initial_strain(lambda x: np.stack([1e-5 * T(x)] * 3 + [0 * x[0]] * 3))
        e.solve()
        mat = ConcreteDamage.from_strength(38.0, E=E, nu=nu, units="N-mm", alpha=1e-5) if concrete else \
            ElasticIso(E, nu, alpha=1e-5)
        s = OpsSolid(m, mat)
        s.fix(lambda x: np.isclose(x[0], 0))
        s.static(strain=s.initial_strain(f) if case == "strain" else s.thermal(T), steps=1)
        U, Ue = s.history[-1].U[:m.n_points], e.nodal_displacement()
        tol_u, tol_s = (1e-5, 2e-4) if concrete else (1e-9, 1e-8)
        assert np.abs(U - Ue).max() < tol_u * np.abs(Ue).max()
        S, Se = s.history[-1].cell["stress"], e.cell_stress()
        if case == "strain":  # cell averages agree where ε0 is at most bilinear (Elastic averages ε0 differently otherwise)
            assert np.abs(S - Se).max() < tol_s * np.abs(Se).max()
        assert not s.history[-1].cell["cracked"].any()


def timoshenko_bimetal(E1, E2, h1, h2, dalpha_dT):
    """curvature of a bimetal strip (Timoshenko 1925, J. Opt. Soc. Am. 11:233)"""
    m, n = h1 / h2, E1 / E2
    return 6 * dalpha_dT * (1 + m) ** 2 / ((h1 + h2) * (3 * (1 + m) ** 2 + (1 + m * n) * (m**2 + 1 / (m * n))))


def test_bimaterial_strip_timoshenko():
    """two bonded layers heated, different α and E, statically supported (ν = 0, beam theory exact): the curvature is
    Timoshenko's; the FE converges to it as the cells get short (shear locking of the 8-node brick in bending)"""
    L, b, h1, h2 = 100.0, 4.0, 4.0, 6.0
    E1, E2, a1, a2, dT = 200e3, 70e3, 1.2e-5, 2.3e-5, 50.0
    k_ref = timoshenko_bimetal(E1, E2, h1, h2, (a2 - a1) * dT)
    ratio = []
    for nx in (50, 100):
        m = structured.box(L, b, h1 + h2, nx, 1, 5)
        lo = m.centroids()[:, 2] < h1
        m.cell_groups = {"a": {"hex8": np.flatnonzero(lo)}, "b": {"hex8": np.flatnonzero(~lo)}}
        s = OpsSolid(m, {"a": ElasticIso(E1, 0.0, alpha=a1), "b": ElasticIso(E2, 0.0, alpha=a2)})
        statically_determinate(s, L, b)
        s.static(strain=s.thermal(dT))
        P, U = m.points, s.history[-1].U
        k = np.isclose(P[:, 1], 0) & np.isclose(P[:, 2], 0) & (P[:, 0] > 0.1 * L) & (P[:, 0] < 0.9 * L)
        ratio.append(-2 * np.polyfit(P[k, 0], U[k, 2], 2)[0] / k_ref)
    assert ratio[0] == pytest.approx(1.0, abs=0.03) and ratio[1] == pytest.approx(1.0, abs=0.008)
    assert abs(ratio[1] - 1) < abs(ratio[0] - 1)


def test_bar_thermal_mismatch():
    """a heated prism with a steel bar of another α (ν = 0): the bar and the concrete share the composite strain
    (Ec Ac αc + Es As αs) ΔT / (Ec Ac + Es As), and their forces balance"""
    L, b = 1.0, 0.2
    Ec, Es, ac, as_, As, dT = 30e9, 200e9, 1.0e-5, 1.2e-5, 4e-4, 30.0
    m = structured.box(L, b, b, 20, 2, 2)
    s = OpsSolid(m, ElasticIso(Ec, 0.0, alpha=ac))
    bi = s.bar([(0, b / 2, b / 2), (L, b / 2, b / 2)], As, elastic(Es), alpha=as_)
    statically_determinate(s, L, b)
    s.static(strain=s.thermal(dT))
    eps = (Ec * b * b * ac + Es * As * as_) * dT / (Ec * b * b + Es * As)
    r = s.bar_results(bi)
    mid = np.abs(r["s"] - L / 2) < 0.15 * L  # away from the ends, where the bar hands its force over
    assert r["N"][mid] == pytest.approx(Es * As * (eps - as_ * dT), rel=1e-3)
    assert s.history[-1].U[s.nearest((L, 0, 0)), 0] == pytest.approx(eps * L, rel=2e-3)


def test_eigenstrain_algebra_and_stopping():
    """Eigenstrain objects add (overlapping cells and segments sum) and scale; a stage stopped early keeps the share of
    its free strain it reached, the next stage continues from there"""
    m = structured.box(1.0, 0.2, 0.2, 4, 1, 1)
    s = OpsSolid(m, ElasticIso(1.0, 0.0, alpha=1.0))
    s.bar([(0, 0.1, 0.1), (1.0, 0.1, 0.1)], 1e-3, elastic(1.0), alpha=2.0)
    a, b = s.shrinkage(1e-3, lambda x: x[0] < 0.5), s.thermal(1e-3)
    c = a + 2 * b
    assert isinstance(c, Eigenstrain) and len(c.cells) == s.nc
    assert c.cell_mean(s.nc)[0, 0] == pytest.approx(-1e-3 + 2e-3) and c.cell_mean(s.nc)[-1, 0] == pytest.approx(2e-3)
    assert np.allclose(c.seg_values, 4e-3) and np.allclose((-c).seg_values, -4e-3)
    statically_determinate(s, 1.0, 0.2)
    st = s.static(strain=b, steps=10, stop=lambda h: h.lam > 0.35)
    assert "stopped" in st.message and s.history[-1].cell["eps0"][0, 0] == pytest.approx(0.4e-3)
    s.static(strain=b, steps=2)
    assert s.history[-1].cell["eps0"][0, 0] == pytest.approx(1.4e-3)


# ── cracking under a free strain ─────────────────────────────────────────
@pytest.mark.parametrize("implex", [False, True])
def test_restrained_shrinkage_cracks_by_the_band_law(implex):
    """a prism held at both ends shrinks (ν = 0): it carries σ = E ε_sh until the weak layer reaches ft, then the
    shortening goes into one crack, w = ε_sh L − σ L / E, and the energy dissipated is Gf — the free strain enters the
    material as σ = f(ε − ε0); equivalent nodal forces ∫Bᵀ D ε0 would leave this prism without any stress at all"""
    L, b = 400.0, 100.0
    kw = dict(units="N-mm", prepeak="linear", nu=0.0, implex=implex)
    c = ConcreteDamage.from_strength(38.0, **kw)
    cw = ConcreteDamage.from_strength(38.0, ft=0.95 * c.ft, **kw)
    m = structured.box(L, b, b, 4, 1, 1)
    weak = np.abs(m.centroids()[:, 0] - 250.0) < 1.0
    m.cell_groups = {"c": {"hex8": np.flatnonzero(~weak)}, "w": {"hex8": np.flatnonzero(weak)}}
    s = OpsSolid(m, {"c": c, "w": cw})
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], L), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
    esh, steps = 0.3 / L, 150
    assert s.static(strain=s.shrinkage(esh), steps=steps).converged
    lam = np.array([h.lam for h in s.history])
    sig = np.array([abs(s.reaction(lambda x: np.isclose(x[0], L), i)[0]) for i in range(len(s.history))]) / b**2
    w = lam * esh * L - sig * L / c.E
    d = c.E * esh / steps
    assert 0.95 * c.ft - d <= sig.max() <= 0.95 * c.ft + 2 * d
    assert np.trapezoid(sig, w) / c.Gf == pytest.approx(1.0, abs=0.04 if implex else 0.01)
    assert s.history[-1].cell["width"].sum() == pytest.approx(w[-1], rel=0.01)


# ── bond-slip ────────────────────────────────────────────────────────────
def test_bond_slip_pullout_capacity():
    """a bar with a rigid-plastic bond law pulled out of a stiff block: the force is τ × perimeter × embedded length
    (the bonded length counts only inside the solid), every spring at τ; the MC2010 law follows table 6.1-1"""
    L, d, tau = 200.0, 12.0, 4.0
    m = structured.box(L, 100, 100, 20, 4, 4)
    s = OpsSolid(m, ElasticIso(30e3, 0.2))
    s.fix(lambda x: np.isclose(x[0], 0))
    bi = s.bar([(0.0, 50, 50), (L, 50, 50), (L + 50, 50, 50)], math.pi * d * d / 4, elastic(2e5),
               bond=BondSlip.rigid_plastic(tau, 0.005), outside="free")
    end = s.bars[bi]["nodes"][-1]
    s.fix(None, comps=(1, 2), nodes=[end])
    F = np.zeros((s.n_nodes, 3))
    F[end, 0] = 1.0
    assert s.push(F, end, 0, 1.0, steps=10).converged
    assert max(h.lam for h in s.history) == pytest.approx(tau * math.pi * d * L, rel=1e-9)
    r = s.bar_results(bi)["bond"]
    assert np.allclose(r["tau"], tau) and np.all(r["slip"] > 0.5)
    mc = BondSlip.mc2010(38.0)
    assert max(mc.tau) == pytest.approx(2.5 * math.sqrt(38.0)) and mc.s[3] == pytest.approx(1.0)
    assert mc.tau[0] == pytest.approx(2.5 * math.sqrt(38.0) * (1 / 16) ** 0.4) and mc.tau[-1] == pytest.approx(0.4 * max(mc.tau))
    sp = BondSlip.mc2010(38.0, failure="splitting")
    assert max(sp.tau) == pytest.approx(7.0 * (38 / 20) ** 0.25) and sp.tau[-1] < 1e-2 * max(sp.tau)


def test_restrained_shrinkage_tie_crack_spacing():
    """an RC tie fixed at both ends shrinks until it is cracked through several times (ν = 0, bar with the
    rigid-plastic bond of the tension chord model τ_b0 = 2 fct, tensile strength varying ±5 % between cross-sections):
    - spacing of the cracks within the tension chord model's range [s_rm0/2, s_rm0], s_rm0 = φ fct (1 − ρ)/(2 τ_b0 ρ)
      (Marti, Alvarez, Kaufmann & Sigrist 1998), and below EN 1992-1-1 (7.11) s_r,max;
    - the crack widths add up to the shortening the restraint stops, L (ε_sh − R / Ec Ac) (kinematics, exact in 1D);
    - the widest crack stays below EN 1992-3 annex M (M.1) for end-restrained members, w_k = s_r,max · 0.5 αe kc k fct
      (1 + 1/(αe ρ)) / Es (characteristic value at the cracking force; here the restraint force has dropped below it)"""
    L, b, d = 600.0, 100.0, 16.0
    nx = 30
    rng = np.random.default_rng(1)
    m = structured.box(L, b, b, nx, 2, 2)
    sl = np.floor(m.centroids()[:, 0] / (L / nx)).astype(int)
    base = ConcreteDamage.from_strength(38.0, units="N-mm", nu=0.0, prepeak="linear")
    fac = 1 + 0.05 * rng.uniform(-1, 1, nx)
    m.cell_groups = {f"s{k}": {"hex8": np.flatnonzero(sl == k)} for k in range(nx)}
    mats = {f"s{k}": ConcreteDamage.from_strength(38.0, ft=base.ft * fac[k], units="N-mm", nu=0.0, prepeak="linear")
            for k in range(nx)}
    s = OpsSolid(m, mats, band=("projected", (1.0, 0.0, 0.0)))
    As, tau = math.pi * d * d / 4, 2 * base.ft
    bi = s.bar([(0, 50, 50), (L, 50, 50)], As, elastic(2e5), bond=BondSlip.rigid_plastic(tau, 0.01))
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], L), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[2], 0), comps=(2,))
    ends = [s.bars[bi]["nodes"][0], s.bars[bi]["nodes"][-1]]
    s.fix(None, comps=(0, 1, 2), nodes=ends)  # anchored in the restraining blocks
    esh = 6e-4
    assert s.static(strain=s.shrinkage(esh), steps=150, record=50).converged
    R = abs(s.reaction(np.r_[s.nodes(lambda x: np.isclose(x[0], L)), [ends[1]]])[0])
    ca = s.cracks_along((0, 25, 25), (L, 25, 25))
    ws = np.array([c["width"] for c in ca["cracks"]])
    xs = np.array([c["s"] for c in ca["cracks"]])
    main = ws > 0.25 * ws.max()
    rho, Ec, ae = As / b**2, base.E, 2e5 / base.E
    srm0 = d * base.ft * (1 - rho) / (2 * tau * rho)
    gaps = np.diff(xs[main])
    assert len(gaps) >= 3 and srm0 / 2 <= gaps.mean() <= srm0
    assert ca["total"] == pytest.approx(L * (esh - R / (Ec * b * b)), rel=0.03)
    srmax = 3.4 * (b / 2 - d / 2) + 0.425 * 0.8 * 1.0 * d / rho
    assert gaps.max() < srmax
    wk = srmax * 0.5 * ae * base.ft * (1 + 1 / (ae * rho)) / 2e5
    assert 0.2 * wk < ws.max() < wk
    assert np.abs(s.bar_results(bi)["bond"]["tau"]).max() <= tau * (1 + 1e-9)  # the bond law caps it


# ── a round bar in the mesh, rust expansion ────────────────────────────────
def quad_area(P, Q):
    a, b, c, d = (P[Q[:, i]] for i in range(4))
    return 0.5 * ((c - a)[:, 0] * (d - b)[:, 1] - (c - a)[:, 1] * (d - b)[:, 0])


def test_inclusion_mesh():
    """O-grid sections: all cells positive, the section area exact, the bar a polygon of 4m sides, the ring its
    annulus; aligned bars share rows, staggered ones raise; the block sweeps it with groups per bar"""
    P, Q, R, O = bar_section(150, 80, [(75, 31, 12)], h=5, ring=0.5)
    A = quad_area(P, Q)
    n = 4 * 13  # tile 2 × 31 / h → 13 per side
    assert A.min() > 0 and A.sum() == pytest.approx(150 * 80, rel=1e-9)
    assert A[R < 2].sum() == pytest.approx(n / 2 * 36 * math.sin(2 * math.pi / n), rel=1e-9)
    assert A[R == 1].sum() == pytest.approx(n / 2 * (36 - 5.5**2) * math.sin(2 * math.pi / n), rel=1e-9)
    P2, Q2, _, O2 = bar_section(300, 80, [(75, 31, 12), (225, 31, 12)], h=5)
    assert quad_area(P2, Q2).sum() == pytest.approx(300 * 80, rel=1e-9) and set(np.unique(O2)) == {-1, 0, 1}
    with pytest.raises(ValueError, match="staggered"):
        bar_section(300, 120, [(75, 31, 12), (150, 80, 12)], h=5)
    m = bar_block(np.linspace(0, 20, 3), 150, 80, [(75, 31, 12)], h=5, ring=0.5)
    g = m.cell_groups
    assert len(g["concrete"]["hex8"]) + len(g["steel"]["hex8"]) + len(g["rust"]["hex8"]) == len(m.cells["hex8"])
    assert np.array_equal(g["rust0"]["hex8"], g["rust"]["hex8"])
    n = hoop_normals(m, [(75, 31, 12)])
    c = m.points[m.cells["hex8"]].mean(axis=1)
    assert np.allclose(np.einsum("ij,ij->i", n[:, 1:], c[:, 1:] - [75, 31]), 0, atol=1e-9)


def test_corrosion_conversions():
    d = 12.0
    assert mass_loss(depth_from_mass_loss(0.05, d), d) == pytest.approx(0.05)
    x = 0.02
    assert free_expansion(x, d, n=3.0) == pytest.approx(2 * x, rel=0.01)  # (n − 1) x for x ≪ d
    assert depth_for_expansion(free_expansion(x, d, 3.5, 0.0125), d, 3.5, 0.0125) == pytest.approx(x, rel=1e-12)
    assert free_expansion(0.005, d, 3.0, porous=0.0125) == 0.0
    # the references: published numbers and their internal consistency
    assert alonso_1998(25, 12) == pytest.approx((7.53 + 9.32 * 25 / 12) * 1e-3)
    assert gbt51355_2019(25, 12, 30, corner=True) == pytest.approx(0.012 * 25 / 12 + 0.00084 * 30 + 0.018)
    assert gbt51355_2019(25, 12, 30) == pytest.approx(0.015 * (25 / 12) ** 1.55 + 0.0014 * 30 + 0.016)
    lw = liu_weyers_1998(25, 12, 2.9, 33550, d0=0.0)  # rust/steel volume (1/α)/(ρr/ρs) = 3.82: depth × (n − 1) = ds
    assert free_expansion(lw["depth"], 12, n=1 / 0.57 / (3600 / 7850)) == pytest.approx(lw["expansion"], rel=1e-3)
    em = el_maaddawy_soudki_2007(25, 12, 2.9, 33550, creep=0.0, delta0=0.0)  # the same cylinder, volume ratio ≈ 3.2
    assert free_expansion(em["depth"], 12, n=2 / 0.622) == pytest.approx(lw["expansion"], rel=0.03)


def rust_cylinder(m: int, mats: dict, d=12.0, R=40.0, t=0.6, band=None):
    msh = cylinder([0.0, 1.0], d, R, ring=t, m=m)
    s = OpsSolid(msh, mats, band=band if band is not None else "volume")
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1.0), comps=(0,))  # plane strain
    s.fix(lambda x: np.isclose(x[1], 0) & (np.abs(x[2]) > 0) & np.isclose(x[0], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[2], 0) & (np.abs(x[1]) > 0) & np.isclose(x[0], 0), comps=(2,))
    return msh, s


def radial(msh, U):
    P = msh.points
    rho = np.hypot(P[:, 1], P[:, 2])
    return rho, (U[:msh.n_points, 1] * P[:, 1] + U[:msh.n_points, 2] * P[:, 2]) / np.maximum(rho, 1e-12)


def lame_three_layers(r_i, r, R, delta, E, nu):
    """exact plane-strain solution of a steel core (r < r_i), a rust ring (r_i…r) with the free strain of
    u* = δ (ρ − r_i)/(r − r_i) and a concrete cylinder (r…R) free outside: u = Aρ + B/ρ per layer (+ u* in the ring)"""
    lm = [(e * nu / ((1 + nu) * (1 - 2 * nu)), e / (2 * (1 + nu))) for e in E]
    (l1, m1), (l2, m2), (l3, m3) = lm
    M = np.array([[r_i, -r_i, -1 / r_i, 0, 0],
                  [2 * (l1 + m1), -2 * (l2 + m2), 2 * m2 / r_i**2, 0, 0],
                  [0, r, 1 / r, -r, -1 / r],
                  [0, 2 * (l2 + m2), -2 * m2 / r**2, -2 * (l3 + m3), 2 * m3 / r**2],
                  [0, 0, 0, 2 * (l3 + m3), -2 * m3 / R**2]])
    _, _, _, A3, B3 = np.linalg.solve(M, [0, 0, -delta, 0, 0])
    return (lambda p: A3 * p + B3 / p), (lambda p: 2 * (l3 + m3) * A3 + 2 * m3 * B3 / p**2)


def test_rust_ring_matches_lame():
    """an elastic rust ring with the free strain of rust_strain inside a concrete cylinder (plane strain) matches the
    exact three-layer Lamé solution: radial displacement of the bar surface and of the outside, hoop stress in the
    concrete next to the bar"""
    d, R, t, delta = 12.0, 40.0, 0.6, 0.01
    r = d / 2
    E = (200e3, 20e3, 33e3)
    u_ex, hoop_ex = lame_three_layers(r - t, r, R, delta, E, 0.2)
    msh, s = rust_cylinder(16, {"steel": ElasticIso(E[0], 0.2), "rust": ElasticIso(E[1], 0.2),
                                "concrete": ElasticIso(E[2], 0.2)})
    s.static(strain=s.initial_strain(rust_strain((0, 0, 0), (1, 0, 0), r - t, t, delta), "rust"))
    rho, ur = radial(msh, s.history[-1].U)
    for p in (r, R):
        assert ur[np.isclose(rho, p, rtol=1e-9)].mean() == pytest.approx(u_ex(p), rel=2e-3)
    S = s.history[-1].cell["stress"]
    C = s.cell_centroids()
    n = hoop_normals(msh, [(0, 0, 0)])
    T = np.zeros((s.nc, 3, 3))
    for a, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]):
        T[:, i, j] = T[:, j, i] = S[:, a]
    hoop = np.einsum("ci,cij,cj->c", n, T, n)
    conc = s.cells_of("concrete")
    first = conc[np.argsort(np.hypot(C[conc, 1], C[conc, 2]))[:64]]
    pc = np.hypot(C[first, 1], C[first, 2]).mean()
    assert hoop[first].mean() == pytest.approx(hoop_ex(pc), rel=3e-3)


def test_rust_cracks_the_cover_between_tepfers_bounds():
    """rust expanding a stiff ring inside a concrete cylinder (cover c = 25 mm, bar 12 mm, plane strain, smeared cracking
    all round): the first crack appears when Lamé's hoop stress at the bar reaches ft (within one step); the peak
    pressure on the concrete lies between Tepfers' partly cracked elastic ring and his plastic ring (a softening ring
    carries more than a brittle one, less than a plastic one); the outer surface cracks after the peak"""
    d, c, t = 12.0, 25.0, 0.5
    r, R = d / 2, d / 2 + c
    conc = ConcreteDamage.from_strength(38.0, units="N-mm", prepeak="linear")
    msh = cylinder([0.0, 1.0], d, R, ring=t, m=6)
    s = OpsSolid(msh, {"concrete": conc, "steel": ElasticIso(2e5, 0.3), "rust": ElasticIso(2e5, 0.3)},
                 band=("projected", hoop_normals(msh, [(0, 0, 0)])))
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1.0), comps=(0,))
    s.fix(lambda x: np.isclose(x[1], 0) & (np.abs(x[2]) > 0) & np.isclose(x[0], 0), comps=(1,))
    s.fix(lambda x: np.isclose(x[2], 0) & (np.abs(x[1]) > 0) & np.isclose(x[0], 0), comps=(2,))
    dmax, steps = 0.009, 90
    assert s.static(strain=s.initial_strain(rust_strain((0, 0, 0), (1, 0, 0), r - t, t, dmax), "rust"),
                    steps=steps).converged
    rho, _ = radial(msh, np.zeros((msh.n_points, 3)))
    inner = np.isclose(rho, r, rtol=1e-9)
    C = s.cell_centroids()
    nr = np.c_[np.zeros(s.nc), C[:, 1], C[:, 2]] / np.hypot(C[:, 1], C[:, 2])[:, None]
    rust, conc_c = s.cells_of("rust"), s.cells_of("concrete")
    hexc = msh.cells["hex8"]
    outer = np.intersect1d(np.flatnonzero(np.isclose(rho[hexc], R, rtol=1e-9).any(axis=1)), conc_c)
    u, p, first, surf = [], [], [], []
    for h in s.history[1:]:
        u.append(radial(msh, h.U)[1][inner].mean())
        S = h.cell["stress"]
        srr = (S[:, 1] * nr[:, 1] ** 2 + S[:, 2] * nr[:, 2] ** 2 + 2 * S[:, 4] * nr[:, 1] * nr[:, 2])
        p.append(-srr[rust].mean())
        first.append(h.cell["cracked"][conc_c].any())
        surf.append(h.cell["cracked"][outer].any())
    u, p = np.array(u), np.array(p)
    lc, tp = lame_cylinder(r, R, conc.E, conc.nu), tepfers(c, d, conc.ft)
    k1 = int(np.argmax(first))
    du = dmax / steps
    assert lc["delta1"](conc.ft) - du <= u[k1] <= lc["delta1"](conc.ft) * 1.05 + du
    assert tp["partly_cracked"] < p.max() < tp["plastic"]
    ks = int(np.argmax(surf))
    assert np.any(surf) and ks >= int(np.argmax(p))
    # the thick-walled cylinder models (Liu & Weyers 1998; El Maaddawy & Soudki 2007 alike) put the plastic pressure on
    # an uncracked elastic cylinder; the cracked ring is softer, so the FE needs more expansion to crack the cover
    lw = liu_weyers_1998(c, d, conc.ft, conc.E, nu=conc.nu, d0=0.0)
    assert lw["pressure"] == pytest.approx(tp["plastic"]) and 1.2 < u[ks] / lw["expansion"] < 4.0
