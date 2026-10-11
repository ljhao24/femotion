"""Deck slabs (JTG 3362-2018 4.2), punching (5.6) and slab detailing (9.2).

Independent sources: structural mechanics (fixed-end moments, statics of strips and cantilevers), the theory of plates
(Lévy's solution of a simply supported strip under a patch load; the equilibrium of the Sawko distribution), the
code's commentary (4.2.1, 4.2.2, 5.6.2) and a second clause of the code implemented elsewhere (8.5.5 for pile caps), a
brute-force search against the vehicle placement, and a polygon area by the shoelace formula; hand calculations with
the code's formulas only guard the implementation."""
import itertools
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import regions as rg
from femotion.codes.jtg import slab
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import uniform_load_forces


# ── 4.2.1, 4.2.2 ────────────────────────────────────────────────────────────
def test_one_way_two_way_and_commentary_shares():
    """commentary to 4.2.1: equal mid-span deflections give q1 = l2⁴/(l1⁴ + l2⁴) q; at l1/l2 = 2, q1 = q2/16 and
    M1/M2 = (l2/l1)² = 1/4, hence the one-way limit"""
    assert slab.slab_kind(4.0, 2.0) == "one-way" and slab.slab_kind(3.9, 2.0) == "two-way"
    q2, P2 = slab.two_way_shares(4.0, 2.0)
    q1 = 1 - q2
    assert q1 / q2 == pytest.approx(1 / 16)
    assert (q1 * 4.0 ** 2) / (q2 * 2.0 ** 2) == pytest.approx(1 / 4)  # M1/M2 = q1 l1² / (q2 l2²)
    assert P2 == pytest.approx(8 / 9)


def test_spans_and_moment_coefficients_against_a_fixed_ended_strip():
    """4.2.2 and its commentary (4-3), (4-4): a fixed-ended strip under uniform load has −q l²/12 = −2/3 M0 at the
    supports and q l²/24 at mid-span (frame solver); the code takes −0.7 M0, and +0.5 M0 or +0.7 M0 at mid-span"""
    assert slab.slab_span(clear=2.0, t=0.2) == pytest.approx(2.2)
    assert slab.slab_span(clear=2.0, t=0.2, centres=2.1) == pytest.approx(2.1)
    assert slab.slab_span(clear=2.0, shear=True) == pytest.approx(2.0)
    assert slab.slab_span(centres=2.4, monolithic=False) == pytest.approx(2.4)
    l, q, t = 2.2, 10.0, 0.2
    m = Model()
    ns = [m.node((l * k / 20, 0, 0)) for k in range(21)]
    bs = [m.beam(a, b, Section(E=3.45e7, A=t, Iy=t ** 3 / 12, Iz=t ** 3 / 12, J=t ** 3 / 3), ref=(0, 0, 1))
          for a, b in zip(ns[:-1], ns[1:])]
    m.fix(ns[0])
    m.fix(ns[-1])
    for n in ns[1:-1]:
        m.fix(n, (0, 1, 0, 1, 0, 1))
    sv = Solver(m)
    w = np.zeros((len(bs), 3))
    w[:, 2] = -q
    M = uniform_load_forces(sv, [(bs[0], "i"), (bs[10], "i")], "My", w=w)
    M0 = q * l * l / 8
    assert M[0] / M0 == pytest.approx(-2 / 3, rel=1e-6) and M[1] / M0 == pytest.approx(1 / 3, rel=1e-6)
    sup, mid = slab.continuous_moments(M0, t=0.2, h_rib=1.0)
    assert sup / M0 == pytest.approx(-0.7) and abs(sup) > abs(M[0]) and mid == pytest.approx(0.5 * M0)
    assert slab.continuous_moments(M0, t=0.25, h_rib=1.0)[1] == pytest.approx(0.7 * M0)  # t/h ≥ 1/4


def test_skew_slab_and_haunch():
    """4.2.4 (and table 4-2 of its commentary), 4.2.6 (tan α ≤ 1/3)"""
    assert slab.skew_slab(10, 9, 10)[0] == pytest.approx(10 * math.cos(math.radians(10)))  # l/b ≤ 1.3
    assert slab.skew_slab(14, 9, 10)[0] == pytest.approx(14)  # l/b > 1.3
    assert slab.skew_slab(10, 12, 30)[0] == pytest.approx((10 + 10 * math.cos(math.radians(30))) / 2)
    assert slab.skew_slab(10, 4, 45, precast=True)[0] == pytest.approx(10)
    with pytest.raises(ValueError):
        slab.skew_slab(10, 9, 45)
    assert slab.haunch_depth(0.25, 0.6, 0.2) == pytest.approx(0.37)
    assert slab.haunch_depth(0.25, 0.6, 0.5) == pytest.approx(0.25 + 0.6 / 3)  # steeper than 1:3


# ── 4.2.3, 4.2.5 widths ─────────────────────────────────────────────────────
def test_widths_follow_figure_4_6():
    """4.2.3-1 … -5 and items 5, 6: the trapezoid of figure 4-6 (commentary) across a box cell"""
    a1, b1, h, t = 0.2, 0.6, 0.1, 0.25
    assert slab.spread(b1, h) == pytest.approx(0.8)
    l = 5.0
    am = slab.width_mid(a1, h, l)
    assert am == pytest.approx(max(0.4 + l / 3, 2 * l / 3))
    assert slab.width_mid(a1, h, 2.0) == pytest.approx(4 / 3)  # the 2l/3 floor
    assert slab.width_mid(a1, h, l, d=1.4) == pytest.approx(am + 1.4)  # 4.2.3-3
    lay = slab.DeckLayout([(-3.15, -2.65), (2.65, 3.15)], t, h)  # webs at ±2.9, 0.5 thick; span min(5.3 + 0.25, 5.8)
    assert lay.span(0) == pytest.approx(5.55)
    a_face, cl, _ = lay.width(-2.65 + 0.4, a1, b1)  # the spread touches the web face
    assert cl == "4.2.3-5" and a_face == pytest.approx(0.4 + t)
    a_x, _, x = lay.width(-2.65 + 0.4 + 0.3, a1, b1)
    assert x == pytest.approx(0.3) and a_x == pytest.approx(0.4 + t + 0.6)
    a_mid, cl, _ = lay.width(0.0, a1, b1)
    assert cl == "4.2.3-2" and a_mid == pytest.approx(slab.width_mid(a1, h, 5.55))
    a_web, cl, _ = lay.width(-2.9, a1, b1)
    assert cl == "4.2.3-4" and a_web == pytest.approx(0.4 + t)
    a_c, cl, lc = lay.width(-4.0, a1, b1)  # cantilever: lc to the outer edge of the spread
    assert cl == "4.2.5" and lc == pytest.approx(4.0 - 3.15 + 0.4) and a_c == pytest.approx(0.4 + 2 * lc)
    # items 5, 6: never wider than the slab or a precast slab
    assert slab.DeckLayout([(-3.15, -2.65), (2.65, 3.15)], t, h, precast_width=1.0).width(0.0, a1, b1)[0] == 1.0


def _strip_mx(l, u, v, y, nu=0.2, terms=4000):
    """Lévy series for a strip simply supported at x = 0, l, infinite in y, under P = 1 spread over u (along x, centred
    at l/2) × v (along y, centred at 0): m_x at x = l/2 and the given y (exact integrals of the line-load solution
    D W = (1 + α|r|) e^{−α|r|} / (4 α³) over the patch)"""
    m = np.arange(1, 2 * terms, 2)[:, None]
    al = m * np.pi / l
    qm = 2 / (u * v) / l * 2 / al * np.sin(al * l / 2) * np.sin(al * u / 2)
    G = lambda r: np.sign(r) * (2 - (2 + al * np.abs(r)) * np.exp(-al * np.abs(r))) / (4 * al ** 4)  # ∫0^r DW
    dG = lambda r: -r * np.exp(-al * np.abs(r)) / (4 * al)  # (DW)'
    y = np.atleast_1d(y)[None, :]
    W = qm * (G(y + v / 2) - G(y - v / 2))
    W2 = qm * (dG(y + v / 2) - dG(y - v / 2))
    return np.sum(np.sin(al * l / 2) * (al ** 2 * W - nu * W2), axis=0)


def test_mid_span_width_is_on_the_safe_side_of_plate_theory():
    """commentary to 4.2.3: the 2l/3 floor comes from elastic theory. The isotropic plate (Lévy) under the D60 rear
    wheel spread through 100 mm of surfacing (0.8 × 0.4 m) has m_x whose integral across the strip is the beam moment
    P (2l − u)/8 (statics, checks the series), and an effective width P (2l − u) / (8 m_x) at least the code's a"""
    u, v = slab.spread(0.6, 0.1), slab.spread(0.2, 0.1)
    for l in (1.5, 2.0, 3.0, 4.0, 5.0):
        ys = np.linspace(-15 * l, 15 * l, 6001)
        mx = _strip_mx(l, u, v, ys)
        total = np.sum((mx[1:] + mx[:-1]) / 2 * np.diff(ys))
        assert total == pytest.approx((2 * l - u) / 8, rel=2e-3)
        a_plate = (2 * l - u) / 8 / _strip_mx(l, u, v, 0.0)[0]
        a_code = slab.width_mid(0.2, 0.1, l)
        assert a_code <= a_plate and a_code / a_plate > 0.55


def test_sawko_formula_statics_and_tip_load():
    """commentary to 4.2.5 (4-5): ∫ m_x dy = −P lc for any A' (sech integrates to π), so its effective width is
    π lc / A'; at the tip (A' = 1.53) that is 2.05 lc, against 2 lc of 4.2.5 for a point load"""
    P, a0 = 100.0, 3.0
    for lc in (0.75, 1.5, 2.25, 3.0):
        y = np.linspace(-60 * lc, 60 * lc, 200001)
        m = np.array([slab.sawko_moment(P, lc, a0, yy) for yy in y[::50]])
        yy = y[::50]
        assert np.sum((m[1:] + m[:-1]) / 2 * np.diff(yy)) == pytest.approx(-P * lc, rel=1e-4)
    assert slab.sawko_A(3.0, 3.0) == pytest.approx(1.53) and slab.sawko_A(1.875, 3.0) == pytest.approx(1.235)
    a_sawko = P * a0 / -slab.sawko_moment(P, a0, a0)
    assert a_sawko == pytest.approx(math.pi * a0 / 1.53)
    assert a_sawko / slab.width_cantilever(0.0, 0.0, a0) == pytest.approx(1.0, abs=0.03)
    with pytest.raises(ValueError):
        slab.sawko_A(0.5, 3.0)


# ── vehicles ────────────────────────────────────────────────────────────────
def test_best_row_against_brute_force():
    """the dynamic programme against all placements of one and two vehicles (1.8 m + 1.3 m apart), with table
    4.3.1-5 (1.2 for one vehicle, 1.0 for two)"""
    rng = np.random.default_rng(3)
    dx = 0.1
    xs = np.arange(0, 6.0, dx)
    for _ in range(20):
        v = rng.normal(size=len(xs))
        gap = int(round(3.1 / dx))
        best = max([1.2 * a for a in v] + [v[i] + v[j] for i, j in itertools.combinations(range(len(xs)), 2)
                                           if j - i >= gap] + [0.0])
        val, k, pos = slab.best_row(xs, v, kmax=2)
        assert val == pytest.approx(best)
        worst = min([1.2 * a for a in v] + [v[i] + v[j] for i, j in itertools.combinations(range(len(xs)), 2)
                                            if j - i >= gap] + [0.0])
        assert slab.best_row(xs, v, kmax=2, sense=-1)[0] == pytest.approx(worst)
    assert slab.best_row(xs, -np.ones(len(xs)), kmax=2) == (0.0, 0, [])  # no unfavourable vehicle


def test_axle_groups_overlap():
    """4.2.3-3: the two rear axles (1.4 m apart) act together once one wheel's width exceeds 1.4 m"""
    g = slab.axle_groups()
    assert len(g) == 15 and max(x.P for x in g) == pytest.approx(275.0)
    w = {(x.P, x.d): w for w, _, x in slab.group_widths(1.2, g)}
    assert (70.0, 0.0) in w and (140.0, 1.4) not in w
    w = {(x.P, x.d): w for w, _, x in slab.group_widths(2.0, g)}
    assert w[(140.0, 1.4)] == pytest.approx(140 / 3.4) and w[(70.0, 0.0)] == pytest.approx(35.0)


def test_deck_slab_by_hand():
    """4.2.2 with 4.2.3: one rear wheel (70 kN) at mid-span spread over b × a_mid, simply supported M0 = (P/a)(l/4 −
    b/8) × 1.3 × 1.2 (one vehicle); the support shear from the wheel at the face with a = a1 + 2h + t"""
    s = slab.deck_slab(clear=1.6, t=0.18, h=0.1, h_rib=1.3, surfacing=2.4)
    l = 1.78
    a = max(0.4 + l / 3, 2 * l / 3)
    assert s.spans["moment"] == pytest.approx(l) and s.details["a_mid"] == pytest.approx(a)
    assert s.M0[1] == pytest.approx(70 / a * (l / 4 - 0.8 / 8) * 1.3 * 1.2, rel=1e-9)
    assert s.M0[0] == pytest.approx((26 * 0.18 + 2.4) * l * l / 8)
    assert s.M_support[1] == pytest.approx(-0.7 * s.M0[1]) and s.M_mid[1] == pytest.approx(0.5 * s.M0[1])
    r = (0.8 - 0.8 ** 2 / (2 * 1.6)) / 0.8  # the spread from the face over 0.8 m on a 1.6 m clear span
    assert s.V[1] == pytest.approx(70 / (0.4 + 0.18) * r * 1.3 * 1.2, rel=1e-9)
    assert s.V[0] == pytest.approx((26 * 0.18 + 2.4) * 0.8)


def test_cantilever_slab_statics():
    """4.2.5 on a cantilever: the dead load's root moment γ ∫ t(s) s ds against numerical integration; the wheel 0.5 m
    from the curb with the overlapping rear axles: a = (a1 + 2h + 2lc) + 1.4"""
    L, tr_, tt = 2.5, 0.5, 0.2
    c = slab.cantilever_slab(L, tr_, tt, h=0.1, curb=2.0, line_loads=[(10.0, 2.25)], lanes=2)
    s = np.linspace(0, L, 20001)
    t = tr_ + (tt - tr_) * s / L
    MG = -26 * np.sum(((t * s)[1:] + (t * s)[:-1]) / 2 * np.diff(s)) - 10 * 2.25
    assert c.M_support[0] == pytest.approx(MG, rel=1e-8)
    lc = 1.5 + 0.4
    a = 0.4 + 2 * lc + 1.4
    outer = -140 / a * 1.5  # the spread 1.1 … 1.9 m, centroid at 1.5 m
    inner = -(70 / (0.4 + 2 * (-0.3 + 0.4))) / 0.8 * 0.1 ** 2 / 2  # the other wheel at −0.3 m reaches 0.1 m out
    assert c.M_support[1] == pytest.approx((outer + inner) * 1.3 * 1.2, rel=1e-9)


# ── 5.6 punching ────────────────────────────────────────────────────────────
def test_punching_equals_pile_cap_clause_at_45_degrees():
    """commentary to 8.5.5: for caps βh = 0.85 and at a span-to-depth ratio of 1 (45°) αp = 1; 8.5.5-1 is then
    0.6 ftd h0 Um, and 5.6.1 with βh = 0.85 is 0.7 × 0.85 = 0.595 ftd Um h0 (regions.py implements 8.5.5 separately)"""
    h0, bx, by = 1500.0, 1200.0, 2000.0
    cap = rg.pile_cap_punching_column(1.0, bx=bx, by=by, ax=h0, ay=h0, h0=h0, concrete="C30")
    pun = slab.punching(1.0, Um=slab.punching_perimeter(h0, bx=bx, by=by), h0=h0, h=1600, concrete="C30")
    assert pun.details["beta_h"] == pytest.approx(0.85)
    assert pun.resistance / cap.resistance == pytest.approx(0.595 / 0.6, rel=1e-12)


def test_punching_with_steel_commentary_ratios():
    """commentary to 5.6.2: with punching steel the resistance is capped at 1.5 times that of the slab without it
    (1.05 = 1.5 × 0.7), and the concrete keeps about half (0.35 = 0.7 / 2)"""
    kw = dict(Um=slab.punching_perimeter(250, d=600), h0=250, h=300, concrete="C40")
    plain = slab.punching(500, **kw)
    st = slab.punching(500, stirrups=8 * 2 * 78.5, Um_outer=6000, **kw)
    assert st.details["V_section_max"] / plain.resistance == pytest.approx(1.5)
    assert st.details["V_concrete"] / plain.resistance == pytest.approx(0.5)
    assert st.details["V_steel"] == pytest.approx(0.75 * 330 * 8 * 2 * 78.5 / 1e3)
    bent = slab.punching(500, bent=(6 * 201.1, 45), Um_outer=6000, **kw)
    assert bent.details["V_steel"] == pytest.approx(0.75 * 330 * 6 * 201.1 * math.sin(math.pi / 4) / 1e3)
    with pytest.raises(ValueError):
        slab.punching(500, stirrups=1000, bent=(1000, 45), **kw)


def test_punching_by_hand():
    """5.6.1: βh linear between 300 and 800 mm, a round column as a 0.8 d square, σpc,m term; 5.6.2 checks"""
    assert slab.beta_h(250) == 1.0 and slab.beta_h(550) == pytest.approx(0.925) and slab.beta_h(900) == 0.85
    Um = slab.punching_perimeter(420, d=1000)
    assert Um == pytest.approx(4 * (800 + 420))
    c = jtg.concrete("C40")
    chk = slab.punching(1500, Um=Um, h0=420, h=480, concrete="C40", sigma_pc=2.0, gamma0=1.1)
    bh = 1.0 - 0.15 * 180 / 500
    assert chk.resistance == pytest.approx((0.7 * bh * c.ftd + 0.15 * 2.0) * Um * 420 / 1e3)
    assert chk.demand == pytest.approx(1650)
    assert any("1.0 … 3.5" in n for n in slab.punching(1, Um=Um, h0=420, h=480, concrete="C40", sigma_pc=4.0).notes)
    thin = slab.punching(5000, Um=Um, h0=100, h=140, concrete="C40", stirrups=2000, Um_outer=100,
                         detailing={"stirrup_d": 6, "stirrup_s": 60})
    assert len(thin.failed) == 5  # 1.05 limit, outside the steel, h < 150, d < 8, s > h0/3
    steep = slab.punching(100, Um=Um, h0=420, h=480, concrete="C40", bent=(1000, 60), Um_outer=1e5,
                          detailing={"bent_d": 10, "bent_n": 4})
    assert len(steep.failed) == 3


def test_footing_area_against_the_polygon():
    """figure 5.6.3: ABCDEF by the shoelace formula on its vertices, for a strip deeper and shallower than the corner
    cuts; and 5.6.3-1 by hand"""
    def shoelace(p):
        x, y = np.array(p).T
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    for B, L, bt, bc, h0 in ((4000, 4000, 800, 800, 700), (3000, 5000, 600, 600, 500), (6000, 3200, 600, 800, 650)):
        # plan: x across the checked face (width B), y towards the edge; cone base line at y0 = bc/2 + h0
        y0, y1 = bc / 2 + h0, L / 2
        xc = bt / 2 + h0  # cone base corner
        w, c = y1 - y0, B / 2 - xc
        if c <= w:
            poly = [(-B / 2, y1), (B / 2, y1), (B / 2, y0 + c), (xc, y0), (-xc, y0), (-B / 2, y0 + c)]
        else:
            poly = [(-xc - w, y1), (xc + w, y1), (xc, y0), (-xc, y0)]
        assert slab.footing_punching_area(B, L, bt, bc, h0) == pytest.approx(shoelace(poly))
    c30 = jtg.concrete("C30")
    chk = slab.footing_punching(ps=0.25, A=2.0e6, bt=800, h0=700, h=750, concrete="C30")
    assert chk.demand == pytest.approx(500.0)
    assert chk.resistance == pytest.approx(0.7 * (1 - 0.15 * 450 / 500) * c30.ftd * (800 + 700) * 700 / 1e3)


def test_slab_detailing():
    """9.2.1, 9.2.2, 9.2.3, 9.2.4 thresholds"""
    assert slab.slab_detailing(200, main=(12, 150), distribution=(10, 200)).ok
    bad = slab.slab_detailing(200, main=(8, 220), distribution=(6, 250), bent_at_support=(2, 0.2))
    assert len(bad.failed) == 5
    assert slab.slab_detailing(70, main=(8, 150), sidewalk=True).failed
    assert slab.slab_detailing(70, main=(8, 150), sidewalk=True, precast=True).ok
    assert slab.slab_detailing(300, main=(12, 150), distribution=(8, 200)).notes  # 0.084 % < 0.1 % (advisory)
