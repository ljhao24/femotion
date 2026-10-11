"""Construction stages and creep (femotion.frame.stages, femotion.codes.jtg.creep), JTG 3362-2018 4.3.8, 4.3.9, 6.5.6,
Appendix C. Every check names its independent source: closed-form creep solutions (Dischinger's rate-of-creep method,
which is what the commentary formulas 4-8…4-20 are), Volterra integral equations of statically indeterminate structures
solved here with their own discretisation and Appendix C written out anew (non-parallel creep), the theorems of linear
creep, the age-adjusted effective modulus method of Trost and Bažant (as a comparison), the commentary's tables of
10-year values and its fly-ash formula, plain statics and femotion.frame.thermal."""
import copy
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import creep as jc
from femotion.codes.jtg import prestress as ps
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import chain_sections, uniform_load_forces
from femotion.frame.prestress import Tendon, effects as prestress_effects
from femotion.frame.stages import Creep, Stage, analyse, parallel_creep
from femotion.frame.thermal import effects as thermal_effects

C50 = jtg.concrete(50)
SEC = Section(E=C50.Ec * 1e3, A=1.0, Iy=0.3, Iz=1.0, J=0.5, mass=2.6)
CR = jc.concrete_creep(C50, 70, 300)


def _phi1(cr, t, t0):
    return float(cr.phi(np.array([float(t)]), np.array([float(t0)]))[0])


def _rcm(t0):
    """rate-of-creep law on the Appendix C curve of concrete first loaded at age t0"""
    return parallel_creep(lambda t: CR.phi(t, np.full(np.shape(t), t0)))


def _girder(spans=(20.0,), n=20, sec=SEC, kr=1e13):
    """precast spans, each on its own bearings (pinned at its left end, roller at its right end; two nodes at every
    pier), and per pier a stiff link that makes the spans continuous once it is built"""
    m = Model()
    deck, piers, x0, prev = [], [], 0.0, None
    for L in spans:
        nodes = [m.node((x, 0, 0)) for x in np.linspace(x0, x0 + L, n + 1)]
        deck += [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
        m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
        m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
        if prev is not None:
            m.link(prev, nodes[0], k=(kr,) * 3, kr=(kr,) * 3)
            piers.append((prev, nodes[0]))
        prev, x0 = nodes[-1], x0 + L
    return m, deck, piers


def _without_links(m):
    m1 = copy.deepcopy(m)
    m1.links = []
    return m1


def _continuous(spans=(20.0, 20.0), n=20, sec=SEC):
    m = Model()
    xs = np.concatenate([np.linspace(sum(spans[:k]), sum(spans[:k + 1]), n + 1)[(k > 0):] for k in range(len(spans))])
    nodes = [m.node((x, 0, 0)) for x in xs]
    deck = [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
    m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
    for k in range(1, len(spans) + 1):
        m.fix(nodes[k * n], (0, 1, 1, 1, 0, 0))
    return m, deck, nodes


# ── Appendix C on arrays ────────────────────────────────────────────────────
def test_concrete_creep_is_appendix_c():
    """concrete_creep evaluates on arrays exactly what prestress.creep_coefficient and shrinkage_strain give"""
    for c, RH, h, meth, fa in [(C50, 70, 300, "table", None), (jtg.concrete(40), 55, 150, "formula", None),
                               (jtg.concrete(60), 85, 800, "table", None), (jtg.concrete(40), 75, 250, "table", 20)]:
        cr = jc.concrete_creep(c, RH, h, method=meth, fly_ash=fa, ts=5)
        t0 = np.array([7.0, 7.0, 14.0, 28.0, 45.0, 90.0, 400.0])
        t = t0 + np.array([0.01, 1.0, 30.0, 365.0, 1000.0, 3650.0, 20000.0])
        if fa:
            t0, t = t0[:-1], t[:-1]
        ref = [ps.creep_coefficient(c, RH, h, a, b, meth, fa) for a, b in zip(t, t0)]
        assert np.allclose(cr.phi(t, t0), ref, rtol=1e-12, atol=0)
        assert np.allclose(cr.shrinkage(t, t0), [ps.shrinkage_strain(c, RH, h, a, b, ts=5, method=meth) for a, b in zip(t, t0)],
                           rtol=1e-12, atol=0)
        assert cr.phi(np.array([7.0]), np.array([7.0]))[0] == 0.0 and cr.shrinkage(np.array([5.0]), np.array([7.0]))[0] == 0.0


def test_concrete_creep_reproduces_commentary_tables_c1_c2():
    """commentary tables C-1 and C-2 (p250–251): 10-year shrinkage εcs(tu, t0) and creep φ(tu, t0) of C40 with RH = 55 %
    / 80 %. Table method: φ(t0 + 3650, t0) rounds to 45 of the 48 cells of C-2, the other three are 0.01 lower
    (2.2051 → 2.20, 1.7953 → 1.79, 1.3652 → 1.36); εcs from ts = 7 d to ts + 3650 d is within 0.01·10⁻³ (one unit of
    the last printed digit) of every cell of C-1 (the table mixes rounding and truncation)"""
    c = jtg.concrete(40)
    C2 = {55: [[3.78, 3.36, 3.14, 2.79], [3.23, 2.88, 2.68, 2.39], [2.83, 2.51, 2.35, 2.09], [2.48, 2.20, 2.06, 1.83],
               [2.14, 1.91, 1.78, 1.58], [1.99, 1.76, 1.65, 1.46]],
          80: [[2.73, 2.52, 2.39, 2.20], [2.32, 2.15, 2.05, 1.88], [2.04, 1.89, 1.79, 1.65], [1.79, 1.65, 1.58, 1.44],
               [1.55, 1.43, 1.36, 1.25], [1.44, 1.32, 1.26, 1.15]]}
    C1 = {55: [[0.50, 0.45, 0.38, 0.25], [0.43, 0.41, 0.36, 0.24], [0.38, 0.38, 0.34, 0.23], [0.31, 0.34, 0.32, 0.22],
               [0.27, 0.32, 0.30, 0.21]],
          80: [[0.30, 0.26, 0.23, 0.15], [0.25, 0.24, 0.21, 0.14], [0.22, 0.22, 0.20, 0.13], [0.18, 0.20, 0.19, 0.12],
               [0.16, 0.19, 0.18, 0.12]]}
    off = 0
    for RH in (55, 80):
        for j, h in enumerate((100, 200, 300, 600)):
            cr = jc.concrete_creep(c, RH, h, ts=7)
            for i, t0 in enumerate((3, 7, 14, 28, 60, 90)):
                v = _phi1(cr, t0 + 3650, t0)
                assert abs(v - C2[RH][i][j]) < 0.0101, (RH, h, t0)
                off += round(v, 2) != C2[RH][i][j]
            for i, t0 in enumerate((7, 14, 28, 60, 90)):
                v = float(cr.shrinkage(np.array([3657.0]), np.array([float(t0)]))[0]) * 1e3
                assert abs(v - C1[RH][i][j]) < 0.01, (RH, h, t0)
    assert off == 3


# ── closed forms (commentary 4-7…4-20) ─────────────────────────────────────
def test_closed_forms_by_hand():
    assert jc.average_effective_factor([900, 1100], [1000, 1200]) == pytest.approx(2000 / 2200)
    assert jc.average_effective_factor([900, 1100], [1000, 1200], weights=[1, 3]) == pytest.approx(4200 / 4600)
    assert np.allclose(jc.long_term_secondary([100.0, -50.0], 850, 1000), [85.0, -42.5])
    phi = 1.7
    assert jc.creep_redistribution(0.0, -1000.0, phi) == pytest.approx(-1000 * (1 - math.exp(-1.7)))  # 4-12
    assert jc.creep_redistribution(200.0, -1000.0, phi) == pytest.approx(200 - 1200 * (1 - math.exp(-1.7)))
    assert jc.prestress_creep_redistribution(-300.0, 450.0, 0.0, phi) == pytest.approx(-300 + 450 * (1 - math.exp(-1.7)))  # 4-18
    assert jc.prestress_creep_redistribution(-300.0, 450.0, 100.0, phi) == pytest.approx(-300 + 350 * (1 - math.exp(-1.7)))  # 4-17
    assert jc.redistribution_phi(CR, 3678, 28) == pytest.approx(ps.creep_coefficient(C50, 70, 300, 3678, 28))
    assert jc.redistribution_phi(CR, 3678, 60, 28) == pytest.approx(ps.creep_coefficient(C50, 70, 300, 3678, 28)
                                                                    - ps.creep_coefficient(C50, 70, 300, 60, 28))
    with pytest.raises(ValueError):
        jc.redistribution_phi(CR, 3678, 20, 28)


def _transfer_run(per_decade, first, tau0=28.0, tau=28.0, tendon=None, t_end=3650.0):
    """two 20 m precast spans loaded (own weight, and a tendon) at age tau0, made continuous at age tau"""
    m, deck, piers = _girder((20.0, 20.0))
    st = [Stage(0.0, "spans", beams=deck, age=tau0, tendons=tendon or ()), Stage(tau - tau0, "joint", links=[0])]
    res = analyse(m, st, creep=_rcm(tau0), t_end=t_end, per_decade=per_decade, first=first)
    return m, deck, piers, res


def test_dead_load_redistribution_converges_to_4_12():
    """simple spans made continuous at the age of loading τ: under the rate-of-creep law the step-by-step pier moment
    tends to M1g + (M2g − M1g)(1 − e^−φ(t, τ)) (4-12, Dischinger) at second order in the step size; the same law holds
    for shears and support reactions. Independent source: Dischinger's closed-form solution."""
    tau, t = 28.0, 3650.0
    errs = []
    for pd in (4, 8, 16):
        m, deck, piers, res = _transfer_run(pd, 1e-6)
        sv1, sv2 = Solver(_without_links(m)), Solver(m)
        phi = jc.redistribution_phi(_rcm(tau), t + tau, tau)
        k = res.index(t)
        secs = [(deck[19], "j"), (deck[10], "i"), (deck[19], "i")]
        tr = jc.system_transformation(sv1, sv2, secs, "My", phi)["dead"]
        errs.append(np.abs(res.section_forces(k, secs, "My") - tr.final).max() / abs(tr.final[0]))
    assert errs[0] / errs[1] > 3 and errs[1] / errs[2] > 3 and errs[2] < 3e-5
    assert tr.first[0] == pytest.approx(0, abs=1e-9) and tr.second[0] < 0  # simply supported → hogging at the pier
    # shears and reactions follow the same law
    for comp, secs in (("Vz", [(deck[19], "j"), (deck[20], "i"), (deck[0], "i")]),
                       ("R", [(piers[0][0], 2), (piers[0][1], 2), (0, 2)])):
        tr = jc.system_transformation(sv1, sv2, secs, comp, phi)["dead"]
        got = (res.section_forces(k, secs, "Vz") if comp == "Vz"
               else np.array([res.reactions(k)[n, d] for n, d in secs]))
        assert np.abs(got - tr.final).max() < 5e-5 * np.abs(tr.final).max()
        assert np.abs(tr.final - tr.first).max() > 0.02 * np.abs(tr.final).max()


def test_redistribution_with_later_transformation_4_19():
    """loaded at τ0 = 7 d, made continuous at τ = 60 d: exponent φ(t, τ0) − φ(τ, τ0) (4-19). Independent source:
    Dischinger's solution started at τ"""
    m, deck, piers, res = _transfer_run(16, 1e-6, tau0=7.0, tau=60.0)
    phi = jc.redistribution_phi(_rcm(7.0), 3650 + 7, 60, 7)
    secs = [(deck[19], "j"), (deck[5], "i")]
    tr = jc.system_transformation(Solver(_without_links(m)), Solver(m), secs, "My", phi)["dead"]
    got = res.section_forces(res.index(3650), secs, "My")
    assert np.abs(got - tr.final).max() < 1e-4 * np.abs(tr.final).max()
    # just before the joint the spans are still simply supported: nothing at the pier
    assert abs(res.section_forces(res.index(53, before=True), secs, "My")[0]) < 1e-6


def test_prestress_redistribution_4_17_4_18():
    """tendons stressed on the simple spans, made continuous at once: Mpt = M0 + M′2(1 − e^−φ) (4-18). Then a statically
    indeterminate earlier structure (a two-span girder) whose left end is clamped afterwards: Mpt = M1 + (M′2 − M′1)(1 −
    e^−φ) (4-17). Rate-of-creep law. Independent source: Dischinger's solution."""
    P, e = 3000.0, 0.25
    m, deck, piers = _girder((20.0, 20.0))
    tds = [Tendon(deck[:20], np.full(21, e), P), Tendon(deck[20:], np.full(21, e), P)]
    _, _, _, res = _transfer_run(16, 1e-6, tendon=tds)
    phi = jc.redistribution_phi(_rcm(28.0), 3678, 28)
    secs = [(deck[19], "j"), (deck[10], "i"), (deck[30], "j")]
    sv1, sv2 = Solver(_without_links(m)), Solver(m)
    tr = jc.system_transformation(sv1, sv2, secs, "My", phi, tendons=tds)
    assert np.allclose(tr["prestress"].first, -P * e)  # statically determinate: primary only, M′1 = 0
    assert tr["prestress"].final[0] == pytest.approx(-P * e + 1.5 * P * e * (1 - math.exp(-phi)), rel=1e-4)  # M′2 = 1.5Pe
    got = res.section_forces(res.index(3650), secs, "My")
    ref = tr["dead"].final + tr["prestress"].final
    assert np.abs(got - ref).max() < 5e-5 * np.abs(ref).max()
    # 4-17: two-span girder with a straight tendon, the left end clamped later
    m, deck, nodes = _continuous()
    td = Tendon(deck, np.full(41, e), P)
    st = [Stage(0.0, "girder", beams=deck, age=28, self_weight=False, tendons=[td]),
          Stage(0.0, "clamp", fix={nodes[0]: (0, 0, 0, 0, 1, 0)})]
    res = analyse(m, st, creep=_rcm(28.0), t_end=3650, per_decade=16, first=1e-6)
    m2 = copy.deepcopy(m)
    m2.fix(nodes[0], (0, 0, 0, 0, 1, 0))
    tr = jc.system_transformation(Solver(m), Solver(m2), secs, "My", phi, w=np.zeros((len(deck), 3)), tendons=[td])["prestress"]
    assert abs(tr.first[0] - (-P * e + 1.5 * P * e)) < 1e-6 * P * e  # M1 = M0 + M′1 at the pier
    assert np.abs(res.section_forces(res.index(3650), secs, "My") - tr.final).max() < 5e-5 * np.abs(tr.final).max()
    assert np.abs(tr.final - tr.first).max() > 0.05 * P * e


def test_statically_determinate_creep_and_shrinkage():
    """simple beam (Appendix C creep and shrinkage): the deflection grows as δ0 [1 + φ(t, t0)], the section forces stay
    those of the load, shrinkage only shortens the beam (εcs L at the roller). Independent source: the definition of φ and
    statics of a determinate beam"""
    m, deck, _ = _girder((20.0,))
    res = analyse(m, [Stage(0.0, beams=deck, age=14)], creep=CR, t_end=3650, per_decade=4)
    k0, mid = res.after(0), 10
    M0 = res.section_forces(k0, [(deck[9], "j")], "My")[0]
    assert M0 == pytest.approx(2.6 * 9.81 * 20 ** 2 / 8, rel=1e-9)
    for t in (10.0, 100.0, 3650.0):
        k = res.index(t)
        assert res.displacements(k)[mid, 2] == pytest.approx(res.displacements(k0)[mid, 2] * (1 + _phi1(CR, t + 14, 14)), rel=1e-10)
        assert res.section_forces(k, [(deck[9], "j")], "My")[0] == pytest.approx(M0, rel=1e-10)
        eps = float(CR.shrinkage(np.array([t + 14]), np.array([14.0]))[0])
        assert res.displacements(k)[20, 0] == pytest.approx(-eps * 20, rel=1e-10)
        assert abs(res.section_forces(k, [(deck[0], "i")], "N")[0]) < 1e-6


def test_homogeneous_creep_first_theorem_and_4_7():
    """continuous girder built in one go (no system transformation), homogeneous creep and shrinkage of Appendix C:
    creep does not change the section forces of loads (first theorem of linear creep, e.g. Bažant & Jirásek 2018,
    "Creep and Hygrothermal Effects in Concrete Structures", ch. 5), so the secondary moment of a tendon whose force
    falls from Pi to Pe is exactly C = Pe/Pi times the elastic one at transfer (commentary 4-7). Where 4-7 is an
    approximation (tendon straight in span 1, parabolic in span 2, so the two spans weigh differently in the
    compatibility at the pier): span 1 cast 53 days before span 2 (creep no longer homogeneous) +0.9 %; losses growing
    along the tendon (C an average) −0.5 %, where the staged result equals the elastic one of the actual forces exactly
    (the program's femotion.frame.prestress way)."""
    P, Pe = 3000.0, 2550.0
    m, deck, nodes = _continuous()
    xs = m.X[nodes, 0]
    xi = np.where(xs <= 20, xs / 20, (xs - 20) / 20)
    td = Tendon(deck, np.where(xs <= 20, 0.25, 1.6 * xi * (1 - xi)), P)
    loss = lambda dt: (P - Pe) * (1 - math.exp(-dt / 200.0))  # noqa: E731
    secs = [(deck[19], "j"), (deck[8], "i")]
    sv = Solver(m)
    M_g = uniform_load_forces(sv, secs, "My")
    tot, pri, sec = prestress_effects(sv, td, secs, "My")
    C = jc.average_effective_factor(P - loss(3650), P)

    def run(stages, creep=CR):
        res = analyse(m, stages, creep=creep, t_end=3650, per_decade=8)
        return res.section_forces(res.index(3650), secs, "My")

    M_t = run([Stage(0.0, beams=deck, age=7, tendons=[(td, loss)])])
    assert np.allclose(M_t, M_g + C * tot, rtol=1e-9)
    assert np.allclose(M_t - M_g - C * pri, jc.long_term_secondary(sec, P - loss(3650), P), rtol=1e-8)
    # span 1 older than span 2 at stressing
    M_a = run([Stage(0.0, beams=deck[:20], age=60, self_weight=False),
               Stage(0.0, beams=deck[20:], age=7, self_weight=False, tendons=[(td, loss)])])
    assert 0.005 < (M_a - C * pri)[0] / (C * sec[0]) - 1 < 0.015
    # losses growing along the tendon
    grow = np.linspace(0.4, 1.6, 40)
    M_l = run([Stage(0.0, beams=deck, age=7, self_weight=False, tendons=[(td, lambda dt: loss(dt) * grow)])])
    tot_l, pri_l, sec_l = prestress_effects(sv, Tendon(deck, td.ecc, P - loss(3650) * grow), secs, "My")
    assert np.allclose(M_l, tot_l, rtol=1e-9)
    C_l = jc.average_effective_factor(P - loss(3650) * grow, np.full(40, P))
    assert -0.008 < (M_l - pri_l)[0] / (C_l * sec[0]) - 1 < -0.003


def test_restrained_shrinkage_and_relaxation_vs_aaem():
    """bar fixed at both ends (Appendix C, C50, RH 70 %, h = 300 mm, from age 28 d), against the age-adjusted effective
    modulus method (Trost 1967, Bažant 1972) with the literature value χ = 0.8:
    - suddenly imposed strain (relaxation): N(t)/N0 = 1 − φ/(1 + χφ); the step-by-step value is 5.9 % above it (the χ
      that fits is 0.84; over other ages, sizes and durations 0.75–0.93, around the 0.8 of the literature);
    - shrinkage: N = EA εcs / (1 + χφ). AAEM with 0.8 is 15 % low: shrinkage of a 300 mm member develops much more
      slowly than its creep (βs reaches 0.5 after about 1050 d, βc after about 101 d), so the restraint force arises late
      and creeps as if loaded older; the χ that fits this history is ≈ 0.58 (χ = 0.8 was set for relaxation-type
      histories). Both lie between the elastic solution EA εcs and the effective-modulus one EA εcs / (1 + φ)."""
    sec = Section(E=C50.Ec * 1e3, A=1.0, Iy=0.1, Iz=0.1, J=0.1)
    EA = sec.E * sec.A
    m = Model()
    a, b = m.node((0, 0, 0)), m.node((10, 0, 0))
    e = m.beam(a, b, sec)
    m.fix(a)
    m.fix(b)
    t0, t = 28.0, 3650.0
    phi = _phi1(CR, t + t0, t0)
    res = analyse(m, [Stage(0.0, beams=[e], age=t0, strain={e: 1e-4})], creep=Creep(CR.phi), t_end=t, first=1e-4, per_decade=16)
    r = res.section_forces(res.index(t), [(e, "i")], "N")[0] / res.section_forces(res.after(0), [(e, "i")], "N")[0]
    assert r == pytest.approx(1 - phi / (1 + 0.8 * phi), rel=0.07)  # +5.9 %
    chi = (phi / (1 - r) - 1) / phi
    assert 0.8 < chi < 0.9  # 0.837
    res = analyse(m, [Stage(0.0, beams=[e], age=t0)], creep=CR, t_end=t, first=1e-4, per_decade=16)
    N = res.section_forces(res.index(t), [(e, "i")], "N")[0]
    eps = float(CR.shrinkage(np.array([t + t0]), np.array([t0]))[0])
    aaem = EA * eps / (1 + 0.8 * phi)
    assert EA * eps / (1 + phi) < aaem < N < EA * eps
    assert -0.20 < aaem / N - 1 < -0.10
    assert (EA * eps / N - 1) / phi == pytest.approx(0.58, abs=0.02)


def test_rate_of_creep_relaxation_and_shrinkage_closed_forms():
    """rate-of-creep law: a suddenly imposed strain relaxes as e^−φ, and shrinkage proportional to the creep curve
    (εcs = k φb) builds the restraint force EA k (1 − e^−φ) (Dischinger). Independent source: the closed-form solutions
    of dσ/dφ + σ = E dε/dφ"""
    sec = Section(E=C50.Ec * 1e3, A=1.0, Iy=0.1, Iz=0.1, J=0.1)
    m = Model()
    a, b = m.node((0, 0, 0)), m.node((10, 0, 0))
    e = m.beam(a, b, sec)
    m.fix(a)
    m.fix(b)
    base = lambda t: CR.phi(t, np.full(np.shape(t), 28.0))  # noqa: E731
    kk = 1e-4
    cr = parallel_creep(base, shrinkage=lambda t, t0: kk * np.maximum(base(t) - base(t0), 0.0))
    res = analyse(m, [Stage(0.0, beams=[e], age=28, strain={e: 1e-4})], creep=cr, t_end=3650, times=(30, 365),
                  first=1e-6, per_decade=16)
    N0 = res.section_forces(res.after(0), [(e, "i")], "N")[0]
    for t in (30.0, 365.0, 3650.0):
        phi = float(base(np.array([t + 28]))[0])
        N = res.section_forces(res.index(t), [(e, "i")], "N")[0]
        assert abs(N - (N0 * math.exp(-phi) + sec.E * kk * (1 - math.exp(-phi)))) < 5e-5 * abs(N0), t


def test_imposed_curvature_matches_thermal_and_relaxes():
    """imposed curvature on a two-span continuous girder: at once the secondary moments of femotion.frame.thermal; under
    homogeneous creep they relax exactly like a fixed bar under imposed strain (second theorem of linear creep)"""
    m, deck, nodes = _continuous()
    kap = -1e-4
    secs = chain_sections(deck)
    res = analyse(m, [Stage(0.0, beams=deck, age=28, self_weight=False, curvature={b: kap for b in deck})],
                  creep=Creep(CR.phi), t_end=3650, per_decade=8)
    M0 = res.section_forces(res.after(0), secs, "My")
    assert np.allclose(M0, thermal_effects(Solver(m), deck, secs, "My", curvature=kap), rtol=1e-9, atol=1e-6)
    assert M0[secs.index((19, "j"))] == pytest.approx(-1.5 * SEC.E * SEC.Iy * kap, rel=1e-9)
    bar = Model()
    a, b = bar.node((0, 0, 0)), bar.node((10, 0, 0))
    e = bar.beam(a, b, SEC)
    bar.fix(a)
    bar.fix(b)
    rb = analyse(bar, [Stage(0.0, beams=[e], age=28, strain={e: 1e-4})], creep=Creep(CR.phi), t_end=3650, per_decade=8)
    for t in (100.0, 3650.0):
        ratio = rb.section_forces(rb.index(t), [(e, "i")], "N")[0] / rb.section_forces(rb.after(0), [(e, "i")], "N")[0]
        assert np.allclose(res.section_forces(res.index(t), secs, "My"), M0 * ratio, rtol=1e-9, atol=1e-6)


def test_supports_links_bars_and_removal():
    """statics of stage changes: a temporary middle support released → simple-beam moments (wL²/8); a temporary prop
    (link to the ground) removed → the same; an inclined pretensioned bar → the forces of Solver.pretension"""
    w = 2.6 * 9.81
    m, deck, nodes = _continuous((20.0,), n=20)
    mid = nodes[10]
    st = [Stage(0.0, beams=deck, fix={mid: (0, 0, 1, 0, 0, 0)}), Stage(5.0, release={mid: (0, 0, 1, 0, 0, 0)})]
    res = analyse(m, st)
    s = [(deck[9], "j")]
    assert res.section_forces(res.after(0), s, "My")[0] == pytest.approx(-w * 10 ** 2 / 8, rel=1e-9)
    assert res.reactions(res.after(0))[mid, 2] == pytest.approx(1.25 * w * 10, rel=1e-9)
    assert res.section_forces(res.after(1), s, "My")[0] == pytest.approx(w * 20 ** 2 / 8, rel=1e-9)
    assert res.reactions(res.after(1))[mid, 2] == 0.0
    assert res.reactions(res.after(1))[nodes[0], 2] == pytest.approx(w * 10, rel=1e-9)
    # a prop as a link to a fixed ground node, removed later
    m, deck, nodes = _continuous((20.0,), n=20)
    g = m.node((10, 0, -5))
    m.fix(g)
    m.link(nodes[10], g, k=(0, 0, 1e12))
    res = analyse(m, [Stage(0.0, beams=deck, links=[0]), Stage(5.0, remove_links=[0])])
    assert res.section_forces(res.after(0), s, "My")[0] == pytest.approx(-w * 10 ** 2 / 8, rel=1e-5)
    assert res.section_forces(res.after(1), s, "My")[0] == pytest.approx(w * 20 ** 2 / 8, rel=1e-9)
    # pretensioned bar from the girder to an anchor, installed with its pretension
    m, deck, nodes = _continuous((20.0,), n=20)
    anc = m.node((0, 0, 8))
    m.fix(anc)
    bar = m.truss(nodes[12], anc, Section(E=1.95e8, A=0.004))
    res = analyse(m, [Stage(0.0, beams=deck, self_weight=False), Stage(1.0, trusses=[bar], eps0={bar: 2e-3}, self_weight=False)])
    sv = Solver(m)
    eps = np.zeros(1)
    eps[0] = 2e-3
    U = sv.solve(sv.pretension(eps))
    k = res.after(1)
    assert res.N[k, 0] == pytest.approx(sv.truss_forces(U, eps)[0], rel=1e-9)
    assert np.allclose(res.U[k], U, atol=1e-12)


def test_cantilever_segments_inherit_the_tangent():
    """a cantilever cast in two segments: the second segment's nodes start from the tangent of the deflected first one
    (total = since built + inherited); their displacement since built is the frame solution of the second segment's
    weight on the whole cantilever. Independent source: cantilever formulas qL⁴/8EI, qL³/6EI and femotion.frame"""
    m = Model()
    xs = np.linspace(0, 10, 11)
    nodes = [m.node((x, 0, 0)) for x in xs]
    deck = [m.beam(a, b, SEC) for a, b in zip(nodes[:-1], nodes[1:])]
    m.fix(nodes[0])
    res = analyse(m, [Stage(0.0, beams=deck[:5]), Stage(10.0, beams=deck[5:])])
    q, EI = 2.6 * 9.81, SEC.E * SEC.Iy
    k1, k2 = res.after(0), res.after(1)
    w1, th1 = -q * 5 ** 4 / (8 * EI), -q * 5 ** 3 / (6 * EI)  # θy = −dw/dx with local z up: a sagging tip turns +θy
    assert res.displacements(k1)[5, 2] == pytest.approx(w1, rel=1e-9)
    assert res.displacements(k1)[5, 4] == pytest.approx(-th1, rel=1e-9)
    w_new = uniform_load_forces(Solver(m), [], None, w=np.r_[np.zeros((5, 3)), np.tile((0, 0, -q), (5, 1))])[1]
    assert np.allclose(res.U[k2] - res.U[k1], w_new, atol=1e-12)
    assert res.displacements(k2, total=True)[10, 2] == pytest.approx(w_new[6 * 10 + 2] + w1 + th1 * 5, rel=1e-9)
    assert res.built[10] == k2 and res.built[3] == 0
    assert np.all(res.displacements(k1, total=True)[6:] == 0)  # not built yet


def test_later_losses_6_2_7():
    """the loss function: relaxation × table C.3.1 plus σl6(t) of 6.2.7 with Appendix C φ and εcs, by hand"""
    Ap, l5 = 2780.0, 40.0
    f = jc.later_losses(Ap, l5, CR, 7.0, Ep=1.95e5, alpha_EP=5.65, sigma_pc=10.0, rho=0.006, rho_ps=2.5)
    phi, eps = _phi1(CR, 107, 7), float(CR.shrinkage(np.array([107.0]), np.array([7.0]))[0])
    l6 = 0.9 * (1.95e5 * eps + 5.65 * 10 * phi) / (1 + 15 * 0.006 * 2.5)
    assert float(f(100.0)) == pytest.approx((l5 * 1.0 + l6) * Ap / 1e3, rel=1e-12)
    assert float(f(10.0)) == pytest.approx((l5 * 0.61 + ps.creep_shrinkage_loss(
        1.95e5, float(CR.shrinkage(np.array([17.0]), np.array([7.0]))[0]), 5.65, 10.0, _phi1(CR, 17, 7), 0.006, 2.5)) * Ap / 1e3)
    assert float(f(0.0)) == 0.0 and float(f(1.0)) == pytest.approx((l5 * 0.25 + ps.creep_shrinkage_loss(
        1.95e5, float(CR.shrinkage(np.array([8.0]), np.array([7.0]))[0]), 5.65, 10.0, _phi1(CR, 8, 7), 0.006, 2.5)) * Ap / 1e3)


def test_unstable_and_misuse():
    m, deck, nodes = _continuous((20.0,), n=4)
    with pytest.raises(RuntimeError):
        analyse(m, [Stage(0.0, beams=deck, release={nodes[-1]: (0, 0, 1, 0, 0, 0)})])
    with pytest.raises(ValueError):
        analyse(m, [Stage(5.0, beams=deck), Stage(1.0)])
    with pytest.raises(ValueError):
        analyse(m, [Stage(0.0, beams=deck[:2], w={3: (0, 0, -1)})])
    res = analyse(m, [Stage(0.0, beams=deck)], t_end=10)
    with pytest.raises(ValueError):
        res.index(3.3333)


# ── independent Volterra solutions (no femotion code): non-parallel creep on statically indeterminate structures ──
# Appendix C written out from C.1.1 and C.2.1 for C40 (fcm = 0.8 · 40 + 8 = 40 MPa), RH 70 %, h = 300 mm, ts = 3 d
_FCM, _RH, _H, _TS = 40.0, 70.0, 300.0, 3.0
_PHI_RH = 1 + (1 - _RH / 100) / (0.46 * (_H / 100) ** (1 / 3))
_B_FCM = 5.3 / math.sqrt(_FCM / 10)
_B_H = min(150 * (1 + (1.2 * _RH / 100) ** 18) * _H / 100 + 250, 1500)
_EPS0 = (160 + 10 * 5.0 * (9 - _FCM / 10)) * 1e-6 * 1.55 * (1 - (_RH / 100) ** 3)


def _phi_c40(t, s):
    t, s = np.broadcast_arrays(np.asarray(t, float), np.asarray(s, float))
    d = np.maximum(t - s, 0.0)
    return _PHI_RH * _B_FCM / (0.1 + s ** 0.2) * (d / (_B_H + d)) ** 0.3


def _eps_c40(t, t0):
    bs = lambda d: np.sqrt(d / (350 * (_H / 100) ** 2 + d))  # noqa: E731
    return _EPS0 * (bs(t - _TS) - bs(t0 - _TS))


_XG, _WG = np.polynomial.legendre.leggauss(12)
_XG, _WG = (_XG + 1) / 2, _WG / 2


def _volterra(K, f, starts, t_end, jumps=(), per_decade=80, first=1e-4):
    """y(t) from Σ ΔYj K(t, tj) + ∫ K(t, s) dy(s) = f(t), with the jumps ΔYj at tj and y continuous and piecewise
    linear in t on a grid geometric after every start; each segment's integral by 12-point Gauss–Legendre, the last one
    with s = t − h u^(10/3) against the (t − s)^0.3 of the kernel; solved node by node (a different discretisation from
    the step-by-step method of femotion.frame.stages, which takes the stress linear in log time and its own quadratures)"""
    r = 10 ** (1 / per_decade)
    g = set()
    marks = sorted(starts) + [t_end]
    for a, b in zip(marks[:-1], marks[1:]):
        d = first * r ** np.arange(int(np.log(max((b - a) / first, 1)) / np.log(r)) + 2)
        g |= set((a + d[d < b - a]).tolist()) | {a, b}
    g = np.array(sorted(g))
    sl = np.zeros(len(g))
    p = 10 / 3
    for k in range(1, len(g)):
        t, a, b = g[k], g[:k], g[1:k + 1]
        h = b - a
        Ik = (K(t, a[:, None] + h[:, None] * _XG[None, :]) * _WG).sum(axis=1) * h
        Ik[-1] = (K(t, t - h[-1] * _XG ** p) * p * _XG ** (p - 1) * _WG).sum() * h[-1]
        rhs = f(t) - sum(dy * float(K(t, tj)) for tj, dy in jumps if tj <= t)
        sl[k] = (rhs - (sl[1:k] * Ik[:-1]).sum()) / Ik[-1]
    yc = np.r_[0.0, np.cumsum(sl[1:] * np.diff(g))]
    return lambda t: float(np.interp(t, g, yc)) + sum(dy for tj, dy in jumps if tj <= t)


_CR40 = jc.concrete_creep(jtg.concrete(40), 70, 300, method="formula", ts=3)


def test_volterra_two_spans_of_different_ages():
    """two-span continuous girder 20 + 30 m, span 1 cast 53 days before span 2 (ages 60 d and 7 d when loaded by the
    weight on the continuous girder), 10 kN/m more on span 2 at day 100. Reference: the pier moment X from compatibility
    at the pier, ∫ Σ δi [1 + φi(t, s)] dX(s) = −Σ Δi [1 + φi(t, tload)], δi = Li/3EI, Δi = w Li³/24EI, solved by
    _volterra. Non-parallel (Appendix C) creep, non-uniform ages, a later load: within 2 × 10⁻⁵"""
    L1, L2, w, q = 20.0, 30.0, 2.6 * 9.81, 10.0
    d1, d2 = L1 / 3, L2 / 3
    K = lambda t, s: d1 * (1 + _phi_c40(t + 60, s + 60)) + d2 * (1 + _phi_c40(t + 7, s + 7))  # noqa: E731
    D1, D2, D2q = w * L1 ** 3 / 24, w * L2 ** 3 / 24, q * L2 ** 3 / 24
    f = lambda t: -(D1 * (1 + _phi_c40(t + 60, 60.0)) + D2 * (1 + _phi_c40(t + 7, 7.0))  # noqa: E731
                    + (D2q * (1 + _phi_c40(t + 7, 107.0)) if t >= 100 else 0.0))
    X = _volterra(K, f, [0.0, 100.0], 3650.0, jumps=[(0.0, -(D1 + D2) / (d1 + d2)), (100.0, -D2q / (d1 + d2))])
    sec = Section(E=jtg.concrete(40).Ec * 1e3, A=1.0, Iy=0.3, Iz=1.0, J=0.5)
    m = Model()
    xs = np.r_[np.linspace(0, L1, 21), np.linspace(L1, L1 + L2, 31)[1:]]
    nodes = [m.node((x, 0, 0)) for x in xs]
    deck = [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
    m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
    m.fix(nodes[20], (0, 1, 1, 1, 0, 0))
    m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
    st = [Stage(0.0, beams=deck[:20], age=60, self_weight=False),
          Stage(0.0, beams=deck[20:], age=7, self_weight=False, w={b: (0, 0, -w) for b in deck}),
          Stage(100.0, w={b: (0, 0, -q) for b in deck[20:]})]
    res = analyse(m, st, creep=Creep(_CR40.phi), t_end=3650, times=(10.0, 1000.0), first=1e-4, per_decade=16)
    for k, t in [(res.after(1), 0.0), (res.index(10), 10.0), (res.index(100, before=True), 99.999999),
                 (res.after(2), 100.0), (res.index(1000), 1000.0), (res.index(3650), 3650.0)]:
        assert res.section_forces(k, [(deck[19], "j")], "My")[0] == pytest.approx(X(t), rel=2e-5), t


def test_volterra_transformation_with_appendix_c_creep():
    """two 20 m precast spans loaded at 28 d, made continuous at 60 d, Appendix C (non-parallel) creep: the pier moment
    is x(t) M2g with ∫ [1 + φ(t, s)] dx(s) = φ(t, 28) − φ(60, 28) from 60 d on (compatibility at the joint; the closed
    form 4-19 would take φ(t, s) parallel). Reference _volterra; step-by-step within 3 × 10⁻⁵ of M2g; 4-19 is 35 %
    high on the redistributed moment here"""
    m, deck, piers = _girder((20.0, 20.0), sec=Section(E=jtg.concrete(40).Ec * 1e3, A=1.0, Iy=0.3, Iz=1.0, J=0.5, mass=2.6))
    M2 = uniform_load_forces(Solver(m), [(deck[19], "j")], "My")[0]
    x = _volterra(lambda t, s: 1 + _phi_c40(t, s), lambda t: float(_phi_c40(t, 28.0) - _phi_c40(60.0, 28.0)), [60.0], 3678.0)
    res = analyse(m, [Stage(0.0, beams=deck, age=28), Stage(32.0, links=[0])], creep=Creep(_CR40.phi), t_end=3650,
                  times=(42.0, 132.0, 1032.0), per_decade=16)
    for t in (42.0, 132.0, 1032.0, 3650.0):
        assert abs(res.section_forces(res.index(t), [(deck[19], "j")], "My")[0] - x(t + 28) * M2) < 3e-5 * abs(M2), t
    dischinger = 1 - math.exp(-float(_phi_c40(3678.0, 28.0) - _phi_c40(60.0, 28.0)))
    assert dischinger / x(3678.0) - 1 == pytest.approx(0.35, abs=0.02)


def test_volterra_restrained_bar():
    """bar fixed at both ends from age 28 d (C40, Appendix C): relaxation of an imposed strain N(t)/N0 and restraint of
    shrinkage N/(EA εcs), against _volterra of ∫ [1 + φ(t, s)] dy(s) = f(t); within 2 × 10⁻⁵ (default steps 7 × 10⁻⁵)"""
    sec = Section(E=jtg.concrete(40).Ec * 1e3, A=1.0, Iy=0.1, Iz=0.1, J=0.1)
    m = Model()
    a, b = m.node((0, 0, 0)), m.node((10, 0, 0))
    e = m.beam(a, b, sec)
    m.fix(a)
    m.fix(b)
    K = lambda t, s: 1 + _phi_c40(t, s)  # noqa: E731
    yr = _volterra(K, lambda t: 1.0, [28.0], 3678.0, jumps=[(28.0, 1.0)])
    ys = _volterra(K, lambda t: float(_eps_c40(t, 28.0)), [28.0], 3678.0)
    for pd, tol in ((8, 7e-5), (32, 2e-5)):
        r1 = analyse(m, [Stage(0.0, beams=[e], age=28, strain={e: 1e-4})], creep=Creep(_CR40.phi), t_end=3650,
                     times=(100.0,), per_decade=pd)
        n0 = r1.section_forces(r1.after(0), [(e, "i")], "N")[0]
        for t in (100.0, 3650.0):
            assert abs(r1.section_forces(r1.index(t), [(e, "i")], "N")[0] / n0 - yr(t + 28)) < tol, (pd, t)
        r2 = analyse(m, [Stage(0.0, beams=[e], age=28)], creep=_CR40, t_end=3650, per_decade=pd)
        eps = float(_eps_c40(3678.0, 28.0))
        assert r2.section_forces(r2.index(3650), [(e, "i")], "N")[0] / (sec.E * eps) == pytest.approx(ys(3678.0) / eps, abs=tol)


def test_fly_ash_outside_table_c24():
    """table C.2.4 inside 7–90 d (as prestress.creep_coefficient); the commentary's (C-8), (C-9) (α as a fraction)
    reproduce the table within 0.005; outside, the larger of the edge value and (C-9) carried on from the edge, at most
    1: before 7 d the formula, after 90 d the 90-day value. A staged analysis with fly ash runs to 10 years"""
    tab = {(7, 20): 0.65, (28, 20): 0.50, (90, 10): 0.58, (60, 30): 0.38, (14, 10): 0.70}
    for (t0, a), v in tab.items():
        assert jc.fly_ash_factor(a, t0) == pytest.approx(v)
        assert float(jc._c9(a, np.array(float(t0)))) == pytest.approx(v, abs=0.005)
    c = jtg.concrete(40)
    assert jc.fly_ash_factor(20, 40) == pytest.approx(ps.creep_coefficient(c, 75, 250, 400, 40, fly_ash=20)
                                                      / ps.creep_coefficient(c, 75, 250, 400, 40), rel=1e-12)
    assert jc.fly_ash_factor(20, 365) == pytest.approx(0.46)  # the 90-day value; (C-9) carried on: 0.432
    c9 = lambda t0: float(jc._c9(20, np.array(t0)))  # noqa: E731
    assert jc.fly_ash_factor(20, 5) == pytest.approx(0.65 * c9(5.0) / c9(7.0))
    assert jc.fly_ash_factor(10, 3) == 1.0  # (C-9) carried on exceeds 1
    with pytest.raises(ValueError):
        jc.fly_ash_factor(5, 28)
    cr = jc.concrete_creep(c, 75, 250, fly_ash=20)
    assert _phi1(cr, 3650, 40) == pytest.approx(ps.creep_coefficient(c, 75, 250, 3650, 40, fly_ash=20), rel=1e-12)
    m, deck, _ = _girder((20.0,))
    res = analyse(m, [Stage(0.0, beams=deck, age=14)], creep=cr, t_end=3650, per_decade=4)
    k0, k = res.after(0), res.index(3650)
    assert res.displacements(k)[10, 2] == pytest.approx(res.displacements(k0)[10, 2] * (1 + _phi1(cr, 3664, 14)), rel=1e-10)


def test_inheritance_on_supports_and_closures():
    """total displacement of new nodes: nothing on a support, linear between two ends, the same whatever the numbering.
    A 9 m cantilever, then a 2 m segment (two elements) to an abutment: the abutment node inherits 0, the middle node
    half of the tangent; a closure of two elements between two equal cantilevers built in either element order: the
    middle node inherits the average of the two tangents, both drooping towards the gap: w + θ·1"""
    q, EI = 2.6 * 9.81, SEC.E * SEC.Iy
    m = Model()
    nodes = [m.node((x, 0, 0)) for x in np.linspace(0, 11, 12)]
    deck = [m.beam(a, b, SEC) for a, b in zip(nodes[:-1], nodes[1:])]
    m.fix(nodes[0])
    res = analyse(m, [Stage(0.0, beams=deck[:9]), Stage(5.0, beams=deck[9:], fix={nodes[11]: (0, 1, 1, 1, 0, 0)})])
    w9, th9 = -q * 9 ** 4 / (8 * EI), -q * 9 ** 3 / (6 * EI)
    U0 = res.U0.reshape(-1, 6)
    assert U0[11, 2] == 0.0 and res.displacements(res.after(1), total=True)[11, 2] == 0.0
    assert U0[10, 2] == pytest.approx((w9 + th9 * 1) / 2, rel=1e-9)
    for order in (1, -1):
        m = Model()
        left = [m.node((x, 0, 0)) for x in np.linspace(0, 10, 11)]
        right = [m.node((x, 0, 0)) for x in np.linspace(22, 12, 11)]
        mid = m.node((11, 0, 0))
        bl = [m.beam(a, b, SEC) for a, b in zip(left[:-1], left[1:])]
        br = [m.beam(a, b, SEC) for a, b in zip(right[:-1], right[1:])]
        m.fix(left[0])
        m.fix(right[0])
        pair = [(left[-1], mid), (right[-1], mid)][::order]
        cl = [m.beam(a, b, SEC) for a, b in pair]
        res = analyse(m, [Stage(0.0, beams=bl + br), Stage(5.0, beams=cl, self_weight=False)])
        w10, th10 = -q * 10 ** 4 / (8 * EI), -q * 10 ** 3 / (6 * EI)
        assert res.U0[6 * mid + 2] == pytest.approx(w10 + th10, rel=1e-9)
        assert res.U0[6 * mid + 4] == pytest.approx(0.0, abs=1e-15)


def test_input_checks():
    m, deck, nodes = _continuous((20.0,), n=4)
    extra = m.node((30, 0, 0))
    F = np.zeros(m.ndof)
    F[6 * extra + 2] = -10.0
    with pytest.raises(ValueError):
        analyse(m, [Stage(0.0, beams=deck, F=F)])
    with pytest.raises(ValueError):
        jc.redistribution_phi(CR, 50, 60, 28)
