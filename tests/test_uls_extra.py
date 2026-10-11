"""Ultimate-limit-state odds and ends of JTG 3362-2018: 5.2.7 (over-reinforced sections), 5.2.8–5.2.11 (inclined
sections along a beam), 5.2.13 (stirrup and bent-up bar design), 5.5 (compatibility torsion, thin-walled boxes),
9.3.13 (torsion detailing), 4.3.4 (cantilever arms of box girders), table B.3.4 note (confined nodes), 4.3.1, 4.3.5,
4.3.6 (analysis).

Independent sources are named in each test: closed-form textbook results (the maximum moment of an under-reinforced
section, the horizontal projection of a uniformly loaded beam solved as a quadratic), the frame solver (the moment
diagram with the reaction spread over the bearing), statics and geometry by other methods (segment intersection,
Mohr's circle by eigenvalues, the sand heap for Wt), and properties (back-substitution, monotony, continuity).
"""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import chain_sections, uniform_load_forces

A = lambda d: math.pi * d * d / 4  # noqa: E731


# ── 5.2.7 ───────────────────────────────────────────────────────────────────
def test_527_balanced_moment_closed_form():
    """the resistance 5.2.7 allows for an over-reinforced section is the moment with the tension steel cut to x = ξb h0:
    for a singly reinforced rectangle the textbook maximum Mu,max = fcd b h0² ξb (1 − 0.5 ξb) (equilibrium at the
    balanced point, independent of the program's scaling and root finding); with compression bars + f'sd A's (h0 − a's);
    a T section with x in the web + fcd (b'f − b) h'f (h0 − h'f / 2)"""
    c = jtg.concrete("C30")
    heavy = jtg.rect(250, 500, bottom=[(8, 32, 60)])
    r = jtg.flexure(heavy, 10.0)
    h0, xb = 440.0, 0.53
    Mmax = c.fcd * 250 * h0 ** 2 * xb * (1 - 0.5 * xb) / 1e6
    assert r.details["clause_5_2_7"]["Mu_balanced"] == pytest.approx(Mmax, rel=1e-9)
    assert r.details["clause_5_2_7"]["x_balanced"] == pytest.approx(xb * h0, rel=1e-9)
    # doubly reinforced: x_b ≥ 2a's, the top bars at f'sd
    dbl = jtg.rect(250, 500, bottom=[(8, 32, 60)], top=[(2, 20, 40)])
    Md2 = (c.fcd * 250 * h0 ** 2 * xb * (1 - 0.5 * xb) + 330 * 2 * A(20) * (h0 - 40)) / 1e6
    assert jtg.flexure(dbl, 10.0).details["clause_5_2_7"]["Mu_balanced"] == pytest.approx(Md2, rel=1e-9)
    # T section, x in the web
    t = jtg.tee(200, 700, 600, 100, concrete="C30", bottom=[(10, 32, 70)])
    h0t = 630.0
    xbt = xb * h0t
    Mt = c.fcd * (200 * xbt * (h0t - xbt / 2) + 400 * 100 * (h0t - 50)) / 1e6
    rt = jtg.flexure(t, 10.0)
    assert any("over-reinforced" in f for f in rt.failed)
    assert rt.details["clause_5_2_7"]["Mu_balanced"] == pytest.approx(Mt, rel=1e-9)


def test_527_uls_steel_only():
    """5.2.7 and its commentary (p175): x may be found with only the tension steel the ultimate limit state needs.
    Properties: the steel share k_uls gives Mu = γ0 Md exactly and x ≤ ξb h0; the check passes exactly when
    γ0 Md ≤ Mu,b; the strain-compatible resistance with all the steel is above Mu,b (more steel, more moment)"""
    heavy = jtg.rect(250, 500, bottom=[(8, 32, 60)])
    full = jtg.flexure(heavy, 200.0)
    assert not full.ok and any("uls_steel_only" in n for n in full.notes)  # default: as before, with the hint
    r = jtg.flexure(heavy, 200.0, gamma0=1.1, uls_steel_only=True)
    bal = r.details["clause_5_2_7"]
    assert r.ok and r.resistance == pytest.approx(bal["Mu_balanced"])
    assert full.resistance > bal["Mu_balanced"]
    # the steel the ultimate limit state needs, back-substituted
    bars = heavy.bars.copy()
    bars[:, 0] *= bal["k_uls"]
    uls = jtg.flexure(heavy.replace(bars=bars), 1.0)
    assert uls.resistance == pytest.approx(220.0, rel=1e-7)
    assert uls.details["x"] <= uls.details["xi_b"] * uls.details["h0"] and not uls.failed
    assert bal["x_uls"] == pytest.approx(uls.details["x"], rel=1e-7)
    # just above Mu,b: fails even with 5.2.7
    over = jtg.flexure(heavy, bal["Mu_balanced"] * 1.001, uls_steel_only=True)
    assert not over.ok and any("5.2.7" in f for f in over.failed)
    # an under-reinforced section is not affected
    light = jtg.rect(250, 500, bottom=[(3, 20, 45)])
    assert jtg.flexure(light, 50.0, uls_steel_only=True).resistance == jtg.flexure(light, 50.0).resistance


def test_527_prestressed_and_hogging():
    """prestressed: As and Ap are scaled together (h0 unchanged), x = ξb h0 at the balanced share; hogging works on the
    flipped section (the same numbers as the section turned over by hand)"""
    pc = jtg.rect(300, 700, concrete="C50", bottom=[(6, 32, 60)], tendons=[(2800, 620, 1100, 90)])
    r = jtg.flexure(pc, 100.0, pc_class="full", uls_steel_only=True)
    bal = r.details["clause_5_2_7"]
    assert bal["x_balanced"] == pytest.approx(0.40 * r.details["h0"], rel=1e-9)
    hog = jtg.rect(250, 500, top=[(8, 32, 60)])
    up = jtg.rect(250, 500, bottom=[(8, 32, 60)])
    assert jtg.flexure(hog, -10.0).details["clause_5_2_7"]["Mu_balanced"] == pytest.approx(
        jtg.flexure(up, 10.0).details["clause_5_2_7"]["Mu_balanced"])


# ── 5.2.8–5.2.11: inclined sections ─────────────────────────────────────────
def _simple(L=16.0, q=80.0, n=161):
    x = np.linspace(0, L, n)
    return x, q * (L / 2 - x), q * x * (L - x) / 2


SEC = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(4, 16, 50)])


def test_projection_from_h2_closed_form():
    """uniform load on a simple span: m h0 = M / V = x (L − x) / (L − 2x), so the compression end x of the inclined
    section whose tension end is xt = h/2 solves (x − xt)(L − 2x) = 0.6 x (L − x), i.e. the smaller root of
    1.4 x² − (0.4 L + 2 xt) x + xt L = 0 (a closed form, against the program's numerical search)"""
    L = 16.0
    x, V, M = _simple(L, n=16001)  # fine data: the program interpolates linearly between the points
    r = jtg.shear_search(x, V, M, SEC, (2, 10, 150), supports=[(0, "end"), (L, "end")], grid=False)
    s = r.at("h/2 from end support x = 0")[0]
    xt = SEC.h / 2e3
    bq = 0.4 * L + 2 * xt
    xc = (bq - math.sqrt(bq * bq - 5.6 * xt * L)) / 2.8
    assert s.x_t == pytest.approx(xt) and s.x_c == pytest.approx(xc, rel=1e-6)
    h0 = SEC.h0(external=False) / 1e3
    assert s.m == pytest.approx(xc * (L - xc) / (L - 2 * xc) / h0, rel=1e-6) and s.m < 3
    assert s.C == pytest.approx(xc - xt, rel=1e-6)
    # the other end mirrors it
    s2 = r.at("h/2 from end support x = 16")[0]
    assert s2.x_c == pytest.approx(L - xc, rel=1e-6) and s2.x_t == pytest.approx(L - xt)


def test_search_properties():
    """the governing inclined section is never less unfavourable than any 5.2.8 section; with one section, one set of
    stirrups and no bent-up bars the resistance is the same everywhere (P capped), so the governing section is the one
    with the largest Vd at its compression end: the h/2 section; each check equals jtg.shear at x_c (another code path)"""
    L = 16.0
    x, V, M = _simple(L)
    r = jtg.shear_search(x, V, M, SEC, (2, 10, 150), supports=[(0, "end"), (L, "end")])
    g = r.governing
    assert all(g.ratio >= s.ratio - 1e-12 for s in r.sections)
    assert "h/2" in g.source
    direct = jtg.shear(SEC, float(np.interp(g.x_c, x, V)), (2, 10, 150))
    assert g.check.resistance == pytest.approx(direct.resistance, rel=1e-12)
    # m is capped at 3 near midspan: C = 1.8 h0
    mid = [s for s in r.sections if s.source == "grid" and 6.0 < s.x_c < 7.0][0]
    assert mid.m == 3.0 and mid.C == pytest.approx(1.8 * SEC.h0(external=False) / 1e3)


def test_search_stirrup_zones_and_bent_bars():
    """an inclined section crossing two stirrup zones takes the mean Asv/sv over its projection (the stirrups the crack
    crosses); a bent-up bar counts when it crosses the crack with the opposite inclination, checked against the
    intersection of the two straight segments solved as a 2 × 2 linear system (another method)"""
    L = 16.0
    x, V, M = _simple(L)
    zones = [(0, 1.0, (2, 12, 100)), (1.0, 16, (2, 10, 200))]
    bent = [(0.9, 0.0, 1000.0, 45.0), (1.8, 0.9, 800.0, 45.0), (0.2, 1.1, 500.0, 45.0)]  # the last one wrongly inclined
    r = jtg.shear_search(x, V, M, SEC, zones, supports=[(0, "end"), (L, "end")], bent=bent)
    s = r.at("h/2 from end support x = 0")[0]
    lo, hi = s.x_t, s.x_c
    q = ((1.0 - lo) * 2 * A(12) / 100 + (hi - 1.0) * 2 * A(10) / 200) / (hi - lo)
    assert s.check.details["rho_sv_mean"] == pytest.approx(q / 200)

    def crosses(bar, crack):
        # crack and bar as segments between depth 1 (bottom) and 0 (top): (x_bottom, x_top)
        (xb1, xt1), (xb2, xt2) = bar, crack
        Amat = np.array([[xt1 - xb1, -(xt2 - xb2)], [-1.0, 1.0]])
        if abs(np.linalg.det(Amat)) < 1e-14:
            return False
        t, u = np.linalg.solve(Amat, [xb2 - xb1, 0.0])  # x and depth equal at bar parameter t, crack parameter u
        return 0 <= t <= 1 and 0 <= u <= 1 and np.sign(xt1 - xb1) == -np.sign(xt2 - xb2)

    expect = [(Ab, th) for xb_, xt_, Ab, th in bent if crosses((xb_, xt_), (s.x_t, s.x_c))]
    assert [tuple(b) for b in s.check.details["bent_crossed"]] == expect and len(expect) == 2
    assert s.check.details["Vsb"] == pytest.approx(0.75e-3 * 330 * 1800 * math.sin(math.pi / 4))


def test_search_interior_support_and_limit():
    """near an interior support the compression end sits at the diaphragm edge and the crack runs away from the support
    (figure 5.2.9b); α1 = 0.9 in hogging (auto); 5.2.11 takes the least b, h0 and the largest Vd over the projection"""
    L = 10.0
    x = np.linspace(0, L, 101)  # propped cantilever under uniform load q: hogging at the fixed end x = L
    q = 100.0
    V = q * (3 * L / 8 - x)  # from the propped end
    M = q * x * (3 * L / 8 - x / 2) * 1.0
    pier = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(4, 16, 60)], top=[(10, 28, 110)])
    r = jtg.shear_search(x, V, M, pier, (2, 12, 100), supports=[(0, "end"), (L, "interior", 0.6)])
    s = r.at("diaphragm edge")[0]
    assert s.x_c == pytest.approx(L - 0.3) and s.x_t < s.x_c
    assert s.check.details["alpha1"] == 0.9 and "hogging" in s.check.name
    lo, hi = s.x_t, s.x_c
    Vmax = max(abs(q * (3 * L / 8 - u)) for u in np.linspace(lo, hi, 2001))
    assert s.check.details["Vd_max_range"] == pytest.approx(Vmax, rel=1e-6)
    sf = pier.flipped()
    lim = 0.51e-3 * math.sqrt(40) * sf.web() * sf.h0(external=False)
    assert s.check.details["V_section_max"] == pytest.approx(lim)
    # sagging region near the propped end: α1 = 1.0
    assert r.at("h/2")[0].check.details["alpha1"] == 1.0


# ── 5.2.13: design ──────────────────────────────────────────────────────────
def test_stirrup_design_back_substitution():
    """5.2.13-1 back into 5.2.9-2 (jtg.shear): 0.2e-6 in 5.2.13-1 stands for 0.45² × 1e-6 = 0.2025e-6, so the
    unrounded spacing gives Vcs = √(0.2025 / 0.2) ξ γ0 V'd exactly; the designed diameter covers the need"""
    Vd = 520.0
    d = jtg.stirrup_design(SEC, Vd, legs=2, dia=10, xi=0.6, gamma0=1.1, step=None)
    chk = jtg.shear(SEC, 0.6 * Vd, (2, 10, d.spacing_required), gamma0=1.1)
    assert chk.details["Vcs"] / (0.6 * 1.1 * Vd) == pytest.approx(math.sqrt(0.2025 / 0.2), rel=1e-12)
    assert d.spacing == pytest.approx(min(d.spacing_required, 400.0, 2 * A(10) / (0.0011 * 200)))
    # rounded down to 10 mm and within 9.3.12
    d10 = jtg.stirrup_design(SEC, 600.0, legs=2, dia=10, xi=1.0)
    assert d10.spacing % 10 == 0 and d10.spacing <= d10.spacing_required and d10.check.ok
    # the diameter from a spacing
    dd = jtg.stirrup_design(SEC, 600.0, legs=2, spacing=100, xi=1.0)
    Asv = dd.details["Asv_required"]
    ref = jtg.shear(SEC, 600.0, (2, math.sqrt(4 * Asv / 2 / math.pi), 100))
    assert ref.details["Vcs"] / 600.0 == pytest.approx(math.sqrt(0.2025 / 0.2), rel=1e-9)
    assert dd.dia in jtg.STIRRUP_DIAMETERS and dd.dia >= dd.dia_required and dd.check.ok
    with pytest.raises(ValueError):
        jtg.stirrup_design(SEC, 500.0, legs=2, dia=10, xi=0.5)


def test_stirrup_design_prestressed():
    """commentary to 5.2.13, formula 5-9: sv = 0.2e-6 α1² α2² α3² (2 + 0.6p) √fcu,k Asv fsv b h0² / (γ0Vd − Vpb)²;
    back-substituted Vcs = √(0.2025 / 0.2) (γ0 Vd − Vpb)"""
    pc = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1470, 1150, 90)],
                 tendon="strand1860")
    bt = [(980.0, 8.0)]
    d = jtg.stirrup_design(pc, 1100.0, legs=2, dia=12, pc_class="A", bent_tendons=bt, step=None)
    chk = jtg.shear(pc, 1100.0, (2, 12, d.spacing_required), pc_class="A", bent_tendons=bt)
    Vpb = 0.75e-3 * 1260 * 980 * math.sin(math.radians(8))
    assert chk.details["Vpb"] == pytest.approx(Vpb)
    assert chk.details["Vcs"] / (1100.0 - Vpb) == pytest.approx(math.sqrt(0.2025 / 0.2), rel=1e-12)
    assert chk.details["alpha2"] == 1.25


def test_bent_bar_design_linear_envelope():
    """a linear envelope V = V0 − k x: row 1 takes (1 − ξ) V'd (V'd at h/2), row i ≥ 2 takes V((i − 1) run) − ξ V'd,
    rows up to the one whose top bend passes x_e = (V0 − ξ V'd) / k (closed form); top bends at the support and at the
    previous bottom bend (9.3.10); Asb = γ0 Vsb / (0.75e-3 fsd sin θ) (5.2.13-2)"""
    V0, k = 700.0, 60.0
    x = np.linspace(0, 8, 81)
    V = V0 - k * x
    h, z, xi = 1100.0, 900.0, 0.6
    bb = jtg.bent_bar_design(x, V, support=0.0, h=h, z=z, xi=xi, gamma0=1.1)
    Vdp = V0 - k * 0.55
    line = xi * Vdp
    run = 0.9
    xe = (V0 - line) / k
    n = math.ceil(xe / run)
    assert len(bb.rows) == n
    for i, r in enumerate(bb.rows, 1):
        assert r.x_top == pytest.approx((i - 1) * run) and r.x_bottom == pytest.approx(i * run)
        Vsb = (1 - xi) * Vdp if i == 1 else V0 - k * (i - 1) * run - line
        assert r.Vsb == pytest.approx(Vsb) and r.Asb == pytest.approx(1.1 * Vsb / (0.75e-3 * 330 * math.sin(math.pi / 4)))
    # back-substitution at the design section: designed stirrups (ξ, unrounded) + row 1 carry γ0 V'd with the margin
    # √(0.2025/0.2) on the stirrup share only
    st = jtg.stirrup_design(SEC, Vdp, legs=2, dia=10, xi=xi, gamma0=1.1, step=None)
    chk = jtg.shear(SEC, Vdp, (2, 10, st.spacing_required), gamma0=1.1,
                    bent=[(bb.rows[0].Asb, 45.0)])
    assert chk.ratio == pytest.approx(1 / (xi * math.sqrt(0.2025 / 0.2) + 1 - xi), rel=1e-12)
    # interior support: the first top bend and V'd at the diaphragm edge, mirrored to the left of the support
    xm = np.linspace(-8, 0, 81)
    bi = jtg.bent_bar_design(xm, V0 + k * xm, support=0.0, h=h, z=z, kind="interior", diaphragm=0.6, xi=xi)
    assert bi.x_design == pytest.approx(-0.3) and bi.rows[0].x_top == pytest.approx(-0.3)
    assert bi.rows[0].x_bottom == pytest.approx(-1.2)


# ── 4.3.1, 4.3.5, 4.3.6 ─────────────────────────────────────────────────────
def test_peak_shaving_against_frame():
    """the parabola of peak_shaving is the moment the reaction makes when it is spread over a instead of acting at a
    point: a two-span beam is solved for its support moment and reaction (frame solver); the same span 2L as a simple
    beam with the load and the reaction as an upward uniform load R / a over the bearing gives, by statics, the shaved
    diagram (floor 0: the full M' = R a / 8)"""
    Ls, ne, a = 10.0, 100, 0.8
    xs = np.linspace(0, 2 * Ls, 2 * ne + 1)
    w = 30.0

    def beam(fix_mid):
        m = Model()
        nodes = [m.node((xx, 0, 0)) for xx in xs]
        sec = Section(E=3e7, A=0.5, Iy=0.05, Iz=0.5, J=0.01)
        beams = [m.beam(p, q, sec) for p, q in zip(nodes[:-1], nodes[1:])]
        m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
        m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
        if fix_mid:
            m.fix(nodes[ne], (0, 1, 1, 1, 0, 0))
        return Solver(m), beams

    sv, beams = beam(True)
    secs = chain_sections(beams)
    load = np.zeros((len(beams), 3))
    load[:, 2] = -w
    M = uniform_load_forces(sv, secs, "My", w=load)
    Vz = uniform_load_forces(sv, secs, "Vz", w=load)
    R = abs(Vz[2 * (ne - 1) + 1] - Vz[2 * ne])
    assert R == pytest.approx(1.25 * w * Ls, rel=1e-6)  # two equal spans: 5/4 w L
    xsec = np.array([xs[b + (e == "j")] for b, e in secs])
    shaved = jtg.peak_shaving(xsec, M, Ls, R, a, floor=0.0)
    sv2, beams2 = beam(False)
    load2 = load.copy()
    for e in range(len(beams2)):
        mid = (xs[e] + xs[e + 1]) / 2
        if abs(mid - Ls) < a / 2:
            load2[e, 2] += R / a
    M2 = uniform_load_forces(sv2, secs, "My", w=load2)
    assert np.allclose(shaved, M2, atol=1e-6 * abs(M).max())
    k = int(np.argmin(np.abs(xsec - Ls)))
    assert shaved[k] - M[k] == pytest.approx(R * a / 8)
    # 4.3.5: at most 10 % of the support moment
    Me, d = jtg.support_moment_reduction(M[k], R, 4.0)
    assert d["capped"] and Me == pytest.approx(0.9 * M[k])
    assert jtg.peak_shaving(xsec, M, Ls, R, 4.0)[k] == pytest.approx(0.9 * M[k])


def test_analysis_helpers():
    """4.3.1 0.8 Ec I for members allowed to crack, Ec I otherwise; 4.3.5 a = bearing + 2y, circular 0.8 D; 4.3.6 ratio 2"""
    s = jtg.frame_section(SEC, member="rc", Iz=0.3, J=0.02, mass=1.5)
    assert s.E == pytest.approx(SEC.concrete.Ec * 1e3) and s.Iy == pytest.approx(0.8 * SEC.I * 1e-12)
    assert s.Iz == pytest.approx(0.24) and s.J == 0.02 and s.A == pytest.approx(SEC.area * 1e-6)
    assert jtg.frame_section(SEC, member="A").Iy == pytest.approx(SEC.I * 1e-12)
    assert jtg.stiffness_factor("B") == 0.8 and jtg.stiffness_factor("full") == 1.0
    assert jtg.spread_length(0.5, 0.7) == pytest.approx(1.9)
    assert jtg.spread_length(0.6, 0.7, circular=True) == pytest.approx(0.48 + 1.4)
    assert jtg.variable_inertia_needed(2.1, 1.0) and not jtg.variable_inertia_needed(2.0, 1.0)
    assert jtg.support_moment_reduction(100.0, 500.0, 1.0) == (100.0, {"M_prime": 62.5, "applied": 0.0, "capped": False})


# ── 5.5: compatibility torsion, thin walls; 9.3.13 ──────────────────────────
def _sand_heap(b, h, n=1200):
    x = (np.arange(n) + 0.5) / n * b
    y = (np.arange(n) + 0.5) / n * h
    X, Y = np.meshgrid(x, y)
    d = np.minimum(np.minimum(X, b - X), np.minimum(Y, h - Y))
    return 2 * d.sum() * (b / n) * (h / n)


def test_compatibility_torque():
    """Tcr = 0.7 ftd Wt (commentary to 5.5.1) with Wt from the sand heap (independent of 5.5.2); prestressed: the
    principal tensile stress of the plane stress state (σx = −σpc, τ = Tcr / (0.7 Wt)) equals ftd (eigenvalues, Mohr)"""
    sec = jtg.rect(400, 800, concrete="C40", bottom=[(4, 25, 50)])
    T, d = jtg.compatibility_torque(sec, 500.0)
    Tcr = 0.7 * 1.65 * _sand_heap(400, 800) / 1e6
    assert d["Tcr"] == pytest.approx(Tcr, rel=1e-4) and T == pytest.approx(d["Tcr"]) and d["reduced"]
    assert jtg.compatibility_torque(sec, -20.0)[0] == -20.0  # below Tcr: unchanged
    pc = jtg.rect(500, 1000, concrete="C50", bottom=[(4, 20, 50)], tendons=[(1400, 500, 1100, 80)])
    T, d = jtg.compatibility_torque(pc, 1e4)
    tau = d["Tcr"] * 1e6 / (0.7 * d["Wt"])
    s1 = np.linalg.eigvalsh(np.array([[-d["sigma_pc"], tau], [tau, 0.0]])).max()
    assert s1 == pytest.approx(1.83, rel=1e-9) and d["sigma_pc"] > 0


def test_thin_walled_box():
    """thin_walls=True (program convention): the steel term of 5.5.1-1 alone; the shear part of 5.5.4-1 with βt = 0.5
    equals 5.2.9 (jtg.shear, another path); the wall shear stress by Bredt, T / (2 A0 t), close to T / Wt for a thin
    uniform tube (sand heap, independent); the result never exceeds the code's two-term formula on a box just inside
    the 5.5.1 limits (continuity from the safe side)"""
    thin = jtg.box(6000, 2000, 250, 250, 1000, concrete="C50", bottom=[(20, 25, 60)], top=[(30, 16, 50)])
    kw = dict(stirrups=(4, 14, 100), Ast=30000, core=(5900, 1900))
    assert any("outside 5.5" in f for f in jtg.torsion(thin, Td=500, **kw).failed)
    r = jtg.torsion(thin, Td=500, thin_walls=True, **kw)
    zeta = min(330 * 30000 * 100 / (330 * A(14) * 2 * (5900 + 1900)), 1.7)
    assert r.resistance == pytest.approx(1.2 * math.sqrt(zeta) * 330 * A(14) / 100 * 5900 * 1900 / 1e6)
    assert not any("outside" in f for f in r.failed)
    rv = jtg.torsion(thin, Td=500, Vd=2000, thin_walls=True, **kw)
    ref = jtg.shear(thin, 2000.0, (4, 14, 100), alpha3=rv.details["alpha3"])
    assert rv.details["Vu"] == pytest.approx(ref.details["Vcs"], rel=1e-12)
    # Bredt against the sand heap for a thin square tube (t = 0.04 b): within 5 %
    b, t = 2000.0, 80.0
    tube = jtg.box(b, b, t, t, 2 * t, concrete="C50", bottom=[(20, 25, 40)])
    rt = jtg.torsion(tube, Td=300, thin_walls=True, stirrups=(2, 12, 100), Ast=8000, core=(1900, 1900))
    x = (np.arange(1600) + 0.5) / 1600 * b
    X, Y = np.meshgrid(x, x)
    dd = np.minimum(np.minimum(np.minimum(X, b - X), np.minimum(Y, b - Y)), t)
    heap = 2 * dd.sum() * (b / 1600) ** 2
    assert 2 * rt.details["A0"] * t == pytest.approx(heap, rel=0.05)
    # continuity: slabs of exactly 0.1 × 4000 mm use the code's formula; 1 mm thinner, the thin-wall convention gives
    # less (the concrete term dropped), never more
    ok_box = jtg.box(4000, 2000, 400, 400, 800, concrete="C50", bottom=[(20, 25, 60)])
    thin_box = jtg.box(4000, 2000, 399, 399, 800, concrete="C50", bottom=[(20, 25, 60)])
    kw2 = dict(stirrups=(4, 14, 100), Ast=20000, core=(3900, 1900))
    code = jtg.torsion(ok_box, Td=500, **kw2)
    assert not any("outside" in f for f in code.failed)
    tw = jtg.torsion(thin_box, Td=500, thin_walls=True, **kw2)
    assert any("thin-walled" in n for n in tw.notes)
    assert code.resistance > tw.resistance > 0.8 * code.resistance


def test_torsion_detailing_9313():
    """9.3.13 items 1, 2 from the inputs given; item 4 as the sum of the flexural (9.1.12) and torsion minimums (areas
    summed, program convention): a large flexural As covers a small Ast; hand numbers"""
    sec = jtg.rect(400, 800, concrete="C40", bottom=[(4, 25, 50)], top=[(2, 16, 45)])
    kw = dict(stirrups=(2, 12, 100), core=(320, 720))
    r = jtg.torsion(sec, Td=60.0, Vd=200.0, Ast=6 * 491, closed=False, hook=90, bar_spacing=350, corner_bars=False, **kw)
    w = " ".join(r.warnings)
    assert "item 1" in w and "135°" in w and "300 mm" in w and "corners" in w
    none = jtg.torsion(sec, Td=60.0, Vd=200.0, Ast=6 * 491, **kw)
    assert any("items 1, 2 not checked" in n for n in none.notes)
    bt = none.details["beta_t"]
    rmin = max(0.45 * 1.65 / 330, 0.002)
    need = rmin * 400 * 750 + 0.08 * (2 * bt - 1) * 18.4 / 330 * 400 * 800
    assert none.details["long_required"] == pytest.approx(need)
    assert none.details["long_provided"] == pytest.approx(4 * A(25) + 6 * 491)
    # Ast alone below its own minimum, but the flexural bars exceed theirs: the sum (the code's rule) is met
    small = jtg.torsion(sec, Td=60.0, Vd=200.0, Ast=300.0, **kw)
    assert small.details["rho_st"] < small.details["rho_st_min"]
    assert not any("item 4" in x for x in small.warnings)
    bare = jtg.rect(400, 800, concrete="C40", bottom=[(2, 16, 50)])
    assert any("item 4" in x for x in jtg.torsion(bare, Td=60.0, Vd=200.0, Ast=300.0, **kw).warnings)
    # the longitudinal minimum divides by fsd of the bars (the main text; the commentary p231 prints fsv): HPB300 stirrups
    hp = jtg.torsion(sec, Td=60.0, Vd=200.0, Ast=6 * 491, stirrup_steel="HPB300", **kw)
    assert hp.details["rho_st_min"] == pytest.approx(0.08 * (2 * hp.details["beta_t"] - 1) * 18.4 / 330)
    # flanges: pure torsion minimum alone
    t = jtg.tee(250, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 90)])
    f = jtg.torsion_flange(t, Td=60, which="top", stirrups=(2, 10, 150), Ast=100, core=(800, 90), closed=True,
                           hook=135, bar_spacing=200, corner_bars=True)
    assert any("item 4" in x for x in f.warnings) and not any("item 1" in x or "item 2" in x for x in f.warnings)


# ── 4.3.4 cantilever arms; table B.3.4 note ─────────────────────────────────
def test_box_flange_cantilever():
    """table 4.3.4, cantilever row: ρs bi with li = 1.5 l over the whole arm; into the next span over a = bi ≤ 0.25 l
    from that value to the span's ρf bi; continuity across the support (the same value from both sides)"""
    bi, arm = 3000.0, 12000.0
    ws = jtg.box_flange_coefficient(bi, 1.5 * arm, support=True) * bi
    for xx in (0.0, 4000.0, arm):
        assert jtg.box_flange_width(bi, arm, xx, kind="cantilever") == pytest.approx(ws)
    l = 40000.0
    w0 = jtg.box_flange_width(bi, l, 0.0, kind="simple", cantilever_left=arm)
    wf = jtg.box_flange_coefficient(bi, l) * bi
    assert w0 == pytest.approx(ws)
    assert jtg.box_flange_width(bi, l, 3000.0, kind="simple", cantilever_left=arm) == pytest.approx(wf)
    assert jtg.box_flange_width(bi, l, 1500.0, kind="simple", cantilever_left=arm) == pytest.approx((ws + wf) / 2)
    short = 8000.0  # a = 0.25 × 8000 = 2000 < bi
    w1 = jtg.box_flange_coefficient(bi, 1.5 * short, support=True) * bi
    assert jtg.box_flange_width(bi, l, l - 1000.0, kind="end span", right=30000, cantilever_right=short) == \
        pytest.approx((w1 + jtg.box_flange_coefficient(bi, 0.8 * l) * bi) / 2)


def test_confined_node():
    """table B.3.4 note: βn raised for a confined node by the user's factor; EN 1992-1-1 3.1.9: fck,c / fck is continuous
    at σ2 = 0.05 fck (both branches 1.25)"""
    def model(k):
        m = jtg.StrutTie("C35")
        a = m.node(0, 0, support="xz", bearing=400 * 400, confine=k)
        b = m.node(3000, 0, support="z", bearing=400 * 400)
        c = m.node(1500, 1200, load=(0, -3000))
        m.strut(a, c, area=400 * 500)
        m.strut(b, c, area=400 * 500)
        m.tie(a, b, As=12 * 491)
        return [r for r in m.check() if "bearing" in r.name]

    plain, conf = model(1.0), model(1.1)
    assert conf[0].resistance == pytest.approx(1.1 * plain[0].resistance)
    assert conf[0].details["confine"] == 1.1 and any("B.3.4 note" in n for n in conf[0].notes)
    assert conf[1].resistance == pytest.approx(plain[1].resistance)
    with pytest.raises(ValueError):
        jtg.StrutTie("C35").node(0, 0, confine=0.9)
    c = jtg.concrete("C40")
    s2 = 0.05 * c.fck
    assert jtg.confinement_factor(c, s2) == pytest.approx(1.25)
    assert jtg.confinement_factor(c, s2 * (1 + 1e-9)) == pytest.approx(1.25)
    assert jtg.confinement_factor(c, 4.0) == pytest.approx(1.125 + 2.5 * 4.0 / 26.8)
    assert jtg.EC2_NODE_BONUS == 1.10


def test_search_inputs():
    """zones of sections, a compression-end position given by hand (its x_c is the position), extra jtg.shear arguments
    as a callable of x, and a numeric α1 (identity checks against jtg.shear at the same x_c)"""
    L = 16.0
    x, V, M = _simple(L)
    deep = jtg.tee(200, 1300, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(4, 16, 50)])
    r = jtg.shear_search(x, V, M, [(0, 3, deep), (3, 16, SEC)], [2, 10, 150], supports=[(0, "end"), (L, "end")],
                         positions=[(5.0, "compression", "load point")], alpha1=0.95,
                         shear_kw=lambda u: {"alpha3": 1.0})
    assert r.at("section change x = 3")
    p = r.at("load point")[0]
    assert p.x_c == pytest.approx(5.0) and p.x_t < 5.0
    ref = jtg.shear(SEC, float(np.interp(5.0, x, V)), (2, 10, 150), alpha1=0.95, alpha3=1.0)
    assert p.check.resistance == pytest.approx(ref.resistance, rel=1e-12)
    # 5.2.11 over a range that reaches the deeper section takes the least h0 (the shallower one)
    s3 = r.at("section change x = 3")[0]
    assert s3.check.details["h0_min"] == pytest.approx(SEC.h0(external=False))
