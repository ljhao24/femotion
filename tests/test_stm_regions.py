"""Strut-and-tie models (JTG 3362-2018 appendix B) and the disturbed regions of chapter 8 (8.2–8.5). The two-pile cap
models are checked against the worked example in the commentary to 8.5.4 (figures 8-14, 8-15), the stiffness branch
against the three-bar truss of structural mechanics; the rest against hand calculations with the code's formulas."""
import math

import pytest

from femotion.codes import jtg
from femotion.codes.jtg import regions as rg
from femotion.codes.jtg.stm import BETA_N, beta_c, eps1_of, strut_strength

H0, A_, X1, C_ = 1880.0, 0.15 * 1880.0, 1250.0, 3000.0


def test_commentary_two_pile_cap_inclined_top_strut():
    # figure 8-15: N1d = 6000, N2d = 5000 kN; the commentary gives θ1 = 50.82°, C1 = 7740.3, T = 4890.0, θ2 = 45.67°
    # (atan(5000 / 4890) = 45.64°), C2 = 6993.7 kN, e = 314 mm
    th1 = math.atan(H0 / (X1 + A_))
    T = 6000 / math.tan(th1)
    th2 = math.atan(5000 / T)
    e = H0 - (A_ + X1) * math.tan(th2)
    assert math.degrees(th1) == pytest.approx(50.82, abs=0.01) and math.degrees(th2) == pytest.approx(45.67, abs=0.05)
    assert e == pytest.approx(314, abs=1)
    m = jtg.StrutTie("C30")
    A = m.node(0, 0, support="xz")
    B = m.node(2 * X1 + 2 * A_ + C_, 0, support="z")  # x1, a, c, a, x2 across the top
    C = m.node(X1 + A_, H0, load=(0, -(6000 + T * e / C_)))
    G = m.node(X1 + A_ + C_, H0 - e, load=(0, -(5000 - T * e / C_)))
    m.strut(A, C, area=1e6)
    m.strut(B, G, area=1e6)
    m.strut(C, G, area=1e6)
    m.tie(A, B, As=1e4)
    N = m.solve()
    assert N == pytest.approx([-7740.3, -6993.7, N[2], 4890.0], abs=1.0)
    assert m.R[A, 1] == pytest.approx(6000) and m.R[B, 1] == pytest.approx(5000)
    # the four-bar model carries only loads that match its geometry
    m.loads[C] = (0, -6000.0)
    m.loads[G] = (0, -5000.0)
    with pytest.raises(ValueError, match="mechanism"):
        m.solve()


def test_commentary_two_pile_cap_vertical_ties():
    # figure 8-14 with the commentary's numbers: C1 = 7740.3, T1 = 4890.0, C'1 = 6177.0, C2 = 6450, T2 = 4075.0,
    # C'2 = 5147.5, T' = 3774.0 and 3145.0, F1d = 6629.0, F2d = 4371.0 kN. θ' = atan(h0 / (c − 2a)) = 37.66° (printed as
    # 27.66° in the commentary; only 37.66° gives its forces). The bottom ties run from each pile to the far vertical.
    # Here c is the pier width (piles 2 x1 + c apart), which is what θ' and F1d, F2d of the commentary need; the figure
    # marks c between the two forces as in figure 8-15 (the test above), and the T' subscripts of the text are swapped.
    thp = math.atan(H0 / (C_ - 2 * A_))
    assert math.degrees(thp) == pytest.approx(37.66, abs=0.01)
    th1 = math.atan(H0 / (X1 + A_))
    F1 = 6000 + (6000 - 5000) / math.tan(th1) * math.tan(thp)
    m = jtg.StrutTie("C30")
    A = m.node(0, 0, support="xz")
    B = m.node(2 * X1 + C_, 0, support="z")
    C = m.node(X1 + A_, H0, load=(0, -F1))
    G = m.node(X1 + C_ - A_, H0, load=(0, -(11000 - F1)))
    D = m.node(X1 + A_, 0)
    E = m.node(X1 + C_ - A_, 0)
    for i, j in [(A, C), (B, G), (E, C), (D, G)]:
        m.strut(i, j, area=1e6)
    for i, j in [(A, E), (B, D), (D, C), (E, G)]:
        m.tie(i, j, As=1e4)
    N = m.solve()
    assert N == pytest.approx([-7740.3, -6450, -6177.0, -5147.5, 4890.0, 4075.0, 3145.0, 3774.0], abs=1.5)
    assert F1 == pytest.approx(6629, abs=1)


def test_three_bar_truss_by_stiffness():
    # statically indeterminate: a vertical bar and two at ±θ, equal EA, load P: N_v = P / (1 + 2 cos³θ)
    th, L, P = math.radians(40), 3000.0, 900.0
    m = jtg.StrutTie("C40")
    n = m.node(0, 0, load=(0, -P))
    for x in (-L * math.tan(th), 0.0, L * math.tan(th)):
        s = m.node(x, L, support="xz")
        m.tie(s, n, As=5000)
    N = m.solve()
    Nv = P / (1 + 2 * math.cos(th) ** 3)
    assert N[1] == pytest.approx(Nv, rel=1e-9) and N[0] == pytest.approx(Nv * math.cos(th) ** 2, rel=1e-9)


def test_strut_tie_checks():
    c = jtg.concrete("C35")
    assert beta_c(c) == 1.30 and beta_c(jtg.concrete("C55")) == 1.35
    assert BETA_N == {"CCC": 0.85, "CCT": 0.75, "CTT": 0.65}
    assert strut_strength(c, None) == pytest.approx(0.85 * 1.30 * 16.1)
    e1 = eps1_of(0.0015, 40)
    assert e1 == pytest.approx(0.0015 + 0.0035 / math.tan(math.radians(40)) ** 2)
    assert strut_strength(c, e1) == pytest.approx(min(1.3 * 16.1 / (0.8 + 170 * e1), 0.85 * 1.3 * 16.1))
    # a symmetric deep beam: two struts and a tie
    m = jtg.StrutTie("C35")
    A = m.node(0, 0, support="xz", bearing=400 * 500)
    B = m.node(3000, 0, support="z", bearing=400 * 500)
    C = m.node(1500, 1200, load=(0, -5000))
    m.strut(A, C, area=600 * 500, ends=(None, 400 * 500))
    m.strut(B, C, area=600 * 500)
    m.tie(A, B, As=12 * 491)
    res = m.check(gamma0=1.1)
    th = math.atan(1200 / 1500)
    T = 2500 / math.tan(th)
    Cf = 2500 / math.sin(th)
    tie = next(r for r in res if r.name.startswith("tie"))
    assert tie.demand == pytest.approx(1.1 * T) and tie.resistance == pytest.approx(330 * 12 * 491 / 1e3)
    strut = next(r for r in res if r.name.startswith("strut 0"))
    es = T * 1e3 / (12 * 491 * 2e5)
    fce = strut_strength(c, eps1_of(es, math.degrees(th)))
    assert strut.details["fce_d"] == pytest.approx(fce) and strut.resistance == pytest.approx(fce * 3e5 / 1e3)
    assert strut.demand == pytest.approx(1.1 * Cf)
    face = next(r for r in res if r.name.startswith("node 2 face"))
    assert face.details["node_type"] == "CCC" and face.resistance == pytest.approx(0.85 * 1.3 * 16.1 * 2e5 / 1e3)
    bearing = next(r for r in res if r.name == "node 0 bearing face")
    assert bearing.details["node_type"] == "CCT" and bearing.demand == pytest.approx(1.1 * math.hypot(0, 2500))
    # flat strut: angle to the tie below 25°
    m2 = jtg.StrutTie("C35")
    A = m2.node(0, 0, support="xz")
    B = m2.node(6000, 0, support="z")
    C = m2.node(3000, 1000, load=(0, -1000))
    m2.strut(A, C, area=1e5)
    m2.strut(B, C, area=1e5)
    m2.tie(A, B, As=5000)
    assert any("25°" in f for r in m2.check() for f in r.failed)
    # B.3.5 surface mesh
    mesh = m2.check(mesh=(0.0025, 0.004, 350))[-1]
    assert len(mesh.warnings) == 2 and mesh.ok


def test_anchorage_zone():
    Pd = rg.anchor_force(1000.0)
    assert Pd == 1200.0
    Tb, db = rg.bursting_force(Pd, a=300, h=1500)
    assert Tb == pytest.approx(0.25 * Pd * (1 - 300 / 1500)) and db == pytest.approx(750)
    g, a_ = 2 * 200 / 1500, math.radians(10)
    Tb, db = rg.bursting_force(Pd, a=300, h=1500, e=200, alpha=10)
    assert Tb == pytest.approx(0.25 * Pd * (1 + g) ** 2 * ((1 - g) - 0.2) + 0.5 * Pd * math.sin(a_))
    assert db == pytest.approx(0.5 * (1500 - 400) + 200 * math.sin(a_))
    assert rg.spalling_force([800, 1200]) == pytest.approx(24)
    assert rg.spalling_force_wide(1000, 1200, s=1200, h=1500) == pytest.approx(0.45 * 1100 * (2 * 1200 / 1500 - 1))
    assert rg.spalling_force_wide(1000, 1200, s=760, h=1500) == pytest.approx(24)  # floor 0.02 max
    assert rg.edge_tension(Pd, e=200, h=1500) == 0.0  # γ = 0.267 ≤ 1/3
    g = 2 * 500 / 1500
    assert rg.edge_tension(Pd, e=500, h=1500) == pytest.approx((3 * g - 1) ** 2 / (12 * g) * Pd)
    f = rg.blister_forces(Pd, a=250, d=300, e=400, alpha=12)
    assert f["bursting"] == pytest.approx(0.25 * Pd * (1 - 250 / 600)) and f["end face"] == pytest.approx(48)
    assert f["behind anchor"] == pytest.approx(240) and f["radial"] == pytest.approx(Pd * math.radians(12))
    assert f["edge bending"] == pytest.approx((800 - 300) ** 2 / (12 * 400 * 700) * Pd)
    assert rg.tie_check(300, As=1000, gamma0=1.1).ratio == pytest.approx(330 / 330)


def test_diaphragm_and_cap_beams():
    r = rg.diaphragm_tie(3000, Bw=4000, h=2500, s=1500, As=30 * 314)
    assert r.demand == pytest.approx((0.20 + (1.6 - 0.5) * (0.87 - 1500 / 4000)) * 3000)
    assert any("0.5" in f for f in rg.diaphragm_tie(3000, Bw=6000, h=2500, s=1500, As=1e4).failed)
    assert rg.cap_beam_class(9000, 1600) == "general" and rg.cap_beam_class(6000, 1600) == "short"
    assert rg.cap_beam_class(3000, 1600) == "deep"
    sec = jtg.rect(1600, 1600, concrete="C30", bottom=[(20, 28, 70)], top=[(10, 20, 50)])
    r = rg.cap_flexure(sec, 5000, l=6000)
    As, Asc = 20 * math.pi * 28 ** 2 / 4, 10 * math.pi * 20 ** 2 / 4
    x = (330 * As - 330 * Asc) / (13.8 * 1600)
    z = (0.75 + 0.05 * 6000 / 1600) * (1530 - 0.5 * x)
    assert r.resistance == pytest.approx(330 * As * z / 1e6) and r.details["x"] == pytest.approx(x)
    v = rg.cap_shear(sec, 4000, 6000, (4, 12, 100), alpha1=0.9)
    P = 100 * As / (1600 * 1530)
    rho = 4 * math.pi * 36 / (1600 * 100)
    assert v.resistance == pytest.approx(0.5e-4 * 0.9 * (14 - 3.75) * 1600 * 1530 * math.sqrt(
        (2 + 0.6 * P) * math.sqrt(30) * rho * 330))
    assert v.details["V_section_max"] == pytest.approx(0.33e-4 * (3.75 + 10.3) * math.sqrt(30) * 1600 * 1530)
    t = rg.cap_cantilever_tie(3200, x=600, bc=1200, h0=1500, As=20 * 491, h=1600)
    assert t.demand == pytest.approx((600 + 600) / (0.9 * 1500) * 3200)
    assert rg.cap_cantilever_tie(3200, x=1700, bc=1200, h0=1500, As=1e4, h=1600).failed
    p = rg.pier_head_tie(4000, s=2000, b_prime=1800, h=2600, b=2400, As=16 * 491)
    assert p.demand == pytest.approx(0.45 * 4000 * (4000 - 1800) / 2400)  # h > b: h = b
    assert rg.cap_beam_C3(6000, 1600) == pytest.approx((0.4 * 3.75 + 1) / 3)
    sec_c = jtg.rect(1600, 1600, concrete="C30", bottom=[(20, 28, 70)])
    w0 = jtg.crack_width(sec_c, 3000, 2000)
    w1 = jtg.crack_width(sec_c, 3000, 2000, C3=rg.cap_beam_C3(6000, 1600))
    assert w1.demand == pytest.approx(w0.demand * rg.cap_beam_C3(6000, 1600))


def test_pile_caps():
    N = rg.pile_forces(11000, Mx=1000, My=2000, xy=[(0, 0), (3, 0), (0, 3), (3, 3)])
    assert N == pytest.approx([2750 - 1000 * 1.5 / 9 - 2000 * 1.5 / 9, 2750 - 1000 * 1.5 / 9 + 2000 * 1.5 / 9,
                               2750 + 1000 * 1.5 / 9 - 2000 * 1.5 / 9, 2750 + 1000 * 1.5 / 9 + 2000 * 1.5 / 9])
    assert rg.pile_cap_width(8000, a=1000, D=1200, n=3, spacing=3000) == 8000
    assert rg.pile_cap_width(8000, a=1000, D=1000, n=3, spacing=3500) == pytest.approx(2000 + 6000)
    assert rg.pile_cap_moment([5000, 6000], [1.2, 3.2]) == pytest.approx(6000 + 19200)
    s = rg.pile_cap_shear(9000, As=40 * 616, bs=6000, h0=1800, a=600, concrete="C30")
    P = 100 * 40 * 616 / (6000 * 1800)
    assert s.resistance == pytest.approx(0.9e-4 * (2 + 0.6 * P) * math.sqrt(30) / 0.5 * 6000 * 1800)  # m ≥ 0.5
    res = rg.pile_cap_stm(6000, 5000, 1250, 1250, 1880, b=1200, s=100, d=28, As=40 * 616, bs=4000, concrete="C30",
                          gamma0=1.1)
    th = math.atan(1880 / (0.15 * 1880 + 1250))
    C1, T1 = 6000 / math.sin(th), 6000 / math.tan(th)
    t = 1200 * math.sin(th) + (100 + 6 * 28) * math.cos(th)
    es = T1 * 1e3 / (40 * 616 * 2e5)
    e1 = es + (es + 0.002) / math.tan(th) ** 2
    fce = min(1.3 * 13.8 / (0.8 + 170 * e1), 0.85 * 1.3 * 13.8)
    assert res[0].demand == pytest.approx(1.1 * C1) and res[0].resistance == pytest.approx(t * 4000 * fce / 1e3)
    assert res[2].demand == pytest.approx(1.1 * T1) and res[2].resistance == pytest.approx(330 * 40 * 616 / 1e3)
    low = rg.pile_cap_stm(6000, 5000, 1250, 1250, 1880, b=1200, s=100, d=28, As=10 * 616, bs=4000, concrete="C30")
    assert any("0.15 %" in f for f in low[2].warnings)


def test_pile_cap_punching():
    h0, ftd = 1500.0, 1.39
    r = rg.pile_cap_punching_column(20000, bx=3000, by=2000, ax=200, ay=1800, h0=h0, concrete="C30")
    ax, ay = 0.2 * h0, h0  # clamped to 0.2 h0 … h0
    apx, apy = 1.2 / (ax / h0 + 0.2), 1.2 / (ay / h0 + 0.2)
    assert r.resistance == pytest.approx(0.6 * ftd * h0 * (2 * apx * (2000 + ay) + 2 * apy * (3000 + ax)) / 1e3)
    c = rg.pile_cap_punching_corner(5000, bx=900, by=900, ax=600, ay=750, h0=h0, concrete="C30")
    a1, a2 = 0.8 / (0.4 + 0.2), 0.8 / (0.5 + 0.2)
    assert c.resistance == pytest.approx(0.6 * ftd * h0 * (a1 * (900 + 375) + a2 * (900 + 300)) / 1e3)
    e = rg.pile_cap_punching_edge(5000, bp=800, bx=900, ax=600, h0=h0, b=6000, concrete="C30")
    assert e.resistance == pytest.approx(0.6 * ftd * h0 * (a1 * (800 + h0) + 0.667 * (1800 + 600)) / 1e3)
    assert rg.pile_cap_punching_edge(5000, bp=800, bx=900, ax=600, h0=h0, b=3000, concrete="C30").failed


def test_review_fixes_stm():
    # a tie of bars and prestressing steel: εs the larger of the two rules (program convention)
    m = jtg.StrutTie("C40")
    A = m.node(0, 0, support="xz")
    B = m.node(4000, 0, support="z")
    C = m.node(2000, 2000, load=(0, -2000))
    m.strut(A, C, area=2e5)
    m.strut(B, C, area=2e5)
    m.tie(A, B, As=3000, Ap=1000, fpe=1000)
    m.solve()
    assert m._eps_s(2) == pytest.approx(max(1000e3 / (3000 * 2e5), (1260 - 1000) / 1.95e5))
    # a tie running straight through a node counts once: CCT, not CTT; a node type can be given
    m = jtg.StrutTie("C40")
    A = m.node(0, 0, support="xz")
    D = m.node(2000, 0, support="z", bearing=1e5)
    B = m.node(4000, 0, support="z", kind="CCC")
    m.tie(A, D, As=1000)
    m.tie(D, B, As=1000)
    assert m._node_kind(D) == "CCT" and m._node_kind(B) == "CCC"
    # check() solves again after the loads change
    m = jtg.StrutTie("C35")
    A = m.node(0, 0, support="xz")
    B = m.node(3000, 0, support="z")
    C = m.node(1500, 1500, load=(0, -1000))
    m.strut(A, C, area=1e5)
    m.strut(B, C, area=1e5)
    m.tie(A, B, As=3000)
    r1 = m.check()[2].demand
    m.loads[C] = (0, -2000.0)
    assert m.check()[2].demand == pytest.approx(2 * r1)


def test_review_fixes_regions():
    # cap beam with strong top bars: x by 5.2.2-2 below 2a's → moments about A's or A's left out, the larger
    sec = jtg.rect(1600, 1600, concrete="C35", bottom=[(20, 28, 75)], top=[(24, 28, 75)])
    r = rg.cap_flexure(sec, 3000, l=6000)
    As = 20 * math.pi * 28 ** 2 / 4
    k, h0 = 0.75 + 0.05 * 6000 / 1600, 1525.0
    x1 = 330 * As / (16.1 * 1600)
    z = max(k * (h0 - 75), k * (h0 - 0.5 * x1))
    assert r.resistance == pytest.approx(330 * As * z / 1e6) and r.notes
    # unequal piles: both struts with the larger T in ε1 (one tie)
    res = rg.pile_cap_stm(5000, 12000, 1400, 200, 1880, b=960, s=100, d=28, As=8000, bs=3500, concrete="C30")
    a = 0.15 * 1880
    th1, th2 = math.atan(1880 / (a + 1400)), math.atan(1880 / (a + 200))
    T = max(5000 / math.tan(th1), 12000 / math.tan(th2))
    es = T * 1e3 / (8000 * 2e5)
    for r_, th in zip(res[:2], (th1, th2)):
        assert r_.details["eps1"] == pytest.approx(es + (es + 0.002) / math.tan(th) ** 2)
    # bursting force never negative
    assert rg.bursting_force(1000, a=400, h=1500, e=600)[0] == 0.0  # (1 − γ) − a/h < 0: large eccentricity
