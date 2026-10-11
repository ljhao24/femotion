"""Class B prestressed members under axial force (crack width, 6.4.4 item 3 generalised), and the temperature gradient
on cracked sections: class B stresses in service (7.1.4) and crack widths (6.4), JTG 3362-2018 with appendix D and J.

Independent sources: the reinforced-concrete formulas of 6.4.4 (6.4.4-3, -4 with ηs, a separate branch of the code)
as the limit of a vanishing prestress, the cubic of appendix J for the neutral axis, and a fibre model of the cracked
section (strips of concrete, steel at its own modulus, plane strain solved for equilibrium; it implements the same
convention — the axis fixed by the loads — but not the same formulas) in place of appendix D's closed form and of the
appendix J solver (sections chosen so that the notes of J.0.1 do not act, or only the self-stress compared). Checks of
continuity and monotony (N → 0±, F = Np0 + N through zero, a small secondary axial force) and limits (zero temperature,
a secondary moment alone) against the results without these terms. Taking moments about the top fibre is only a
consistency check (algebraically the same equilibrium)."""
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from femotion.codes import jtg

BUNDLES = [(1, 8, 15.2)]
A25 = math.pi * 25 ** 2 / 4


def _pc_rect(**kw):
    """400 × 1000 C50, bottom bars 4φ20, one post-tensioned tendon 8φs15.2 at 900, σl6 = 0 (Np0 = σp0 Ap)"""
    return jtg.rect(400, 1000, concrete="C50", bottom=[(4, 20, 40)], top=[(4, 16, 40)],
                    tendons=[(1112.0, 900, 1150, 80)], sigma_pe=1050.0, **kw)


# ── fibre model of a cracked section (independent of appendix D's formulas and of _cracked) ──
def _fibre_model(sec, *, hogging=False, dz=0.01):
    s = sec.flipped() if hogging else sec
    n = int(round(s.h / dz))
    ze = np.linspace(0.0, s.h, n + 1)
    width = np.array([s.width(v) for v in (ze[:-1] + ze[1:]) / 2])
    c = s.concrete
    st = [(a, d, s.steel.Es) for a, d in s.bars] + [(a, d, s.tendon.Ep) for a, d, _, _ in s.tendons]
    return s, ze, width, c.Ec, st


def _fibres(model, x):
    """(area, depth, modulus) of the concrete strips above x (all for x ≥ h; midpoints) and of every bar and tendon"""
    s, ze, width, Ec, st = model
    z1, z2 = ze[:-1], np.minimum(ze[1:], x)
    keep = z2 > z1
    A = width[keep] * (z2[keep] - z1[keep])
    z = (z1[keep] + z2[keep]) / 2
    As = np.array([a for a, _, _ in st])
    ds = np.array([d for _, d, _ in st])
    Es = np.array([e for _, _, e in st])
    return np.r_[A, As], np.r_[z, ds], np.r_[np.full(len(A), Ec), Es]


def _fibre_axis(model, F=0.0, z_R=None):
    """neutral axis: pure bending (F = 0) the first moment of the transformed area about the axis vanishes; else the
    linear stress field that is zero at x carries F at depth z_R"""
    def q(x):
        A, z, E = _fibres(model, x)
        return (A * E * (x - z)).sum()
    s = model[0]
    x0 = brentq(q, 1e-6, s.h)
    if F == 0:
        return x0

    def g(x):
        A, z, E = _fibres(model, x)
        w = A * E * (x - z)
        return (w * z).sum() / w.sum() - z_R
    if g(1e7 * s.h) <= 0:  # the line of action at or below the centroid of the whole section: no tension face
        return math.inf
    return brentq(g, x0 * (1 + 1e-9), 1e7 * s.h, xtol=1e-9)


def _fibre_solve(model, x, F, z_R, t, alpha, psi):
    """plane strain u = a + b z (elongation +) on the fibres above x with the free strain ψ α t(z) of the concrete and of
    the steel above x: Σ σ A = F, Σ σ A z = F z_R; returns σ(z) as concrete stress (compression +) and a, b"""
    A, z, E = _fibres(model, x)
    tt = np.where(z <= x, psi * alpha * t(z), 0.0)
    K = np.array([[(A * E).sum(), (A * E * z).sum()], [(A * E * z).sum(), (A * E * z * z).sum()]])
    rhs = np.array([(A * E * tt).sum() - F, (A * E * tt * z).sum() - F * (z_R if F else 0.0)])
    a, b = np.linalg.solve(K, rhs)
    Ec = model[3]
    return (lambda zz, live=True: Ec * ((psi * alpha * t(zz) if live else 0.0) - a - b * zz)), a, b


# ── 1. class B with an axial force (6.4.4 item 3, program convention) ──
def test_class_b_axial_force_limit_of_vanishing_prestress():
    """Np0 → 0 (one pretensioned wire of 1e-9 mm²): the generalised 6.4.4-11 must reduce to the reinforced-concrete
    formulas of 6.4.4 item 1 — eccentric compression 6.4.4-4 with es = ηs e0 + ys (6.4.4-6, -8) and eccentric tension
    6.4.4-3 about A's — computed by the separate reinforced-concrete branch (itself checked against hand calculations
    in test_sls.py). The compression bars at 150 mm make h0 − a's (810) shorter than 0.87 h0 (835), so that the net
    tension side z = min(h0 − a', 0.87 h0) is the z of 6.4.4-3"""
    kw = dict(concrete="C50", bottom=[(4, 20, 40)], top=[(4, 16, 150)])
    tiny = jtg.rect(400, 1000, tendons=[(1e-9, 960, 1000)], pretensioned=True, sigma_pe=1000.0, **kw)
    rc = jtg.rect(400, 1000, **kw)
    for N, l0 in [(500.0, None), (500.0, 20000.0), (-500.0, None), (-80.0, None)]:
        b = jtg.crack_width(tiny, 900.0, 700.0, N=N, l0=l0, pc_class="B", tendon_bundles=BUNDLES)
        r = jtg.crack_width(rc, 900.0, 700.0, N=N, l0=l0)
        assert b.details["sigma_ss"] == pytest.approx(r.details["sigma_ss"], rel=1e-9)
        assert b.details["C3"] == r.details["C3"] == (0.9 if N > 0 else 1.1)
        if N > 0:
            assert b.details["eta_s"] == pytest.approx(r.details["eta_s"], rel=1e-12)
            assert b.details["z"] == pytest.approx(r.details["z"], rel=1e-9)
        else:
            assert b.details["a'"] == pytest.approx(150.0) and b.details["z"] == pytest.approx(960.0 - 150.0)


def test_class_b_axial_force_flexure_and_member_type():
    """N = 0 is 6.4.4-11 exactly (hand calculation as test_sls.test_crack_width_class_b). The nature of the forces is
    the member's (6.4.3: C3 by the kind of member): declared flexural, σss and Wcr are continuous as N → 0±; by
    default the sign of the load's N decides (C3 0.9 / 1.1)"""
    sec = _pc_rect()
    As, Ap = 4 * math.pi * 100, 1112.0
    Np0 = 1150 * Ap
    h0 = (As * 960 + Ap * 900) / (As + Ap)
    ep = h0 - 900
    e = ep + 900e6 / Np0
    z = (0.87 - 0.12 * (h0 / e) ** 2) * h0
    sss = (900e6 - Np0 * (z - ep)) / ((As + Ap) * z)
    r0 = jtg.crack_width(sec, 900.0, 700.0, pc_class="B", tendon_bundles=BUNDLES)
    assert r0.details["sigma_ss"] == pytest.approx(sss, rel=1e-12) and r0.details["C3"] == 1.0
    for N in (1e-7, -1e-7):
        r = jtg.crack_width(sec, 900.0, 700.0, N=N, pc_class="B", tendon_bundles=BUNDLES, member="flexural")
        assert r.details["sigma_ss"] == pytest.approx(sss, rel=1e-9) and r.demand == pytest.approx(r0.demand, rel=1e-9)
        d = jtg.crack_width(sec, 900.0, 700.0, N=N, pc_class="B", tendon_bundles=BUNDLES)
        assert d.details["C3"] == (0.9 if N > 0 else 1.1)
    with pytest.raises(ValueError):
        jtg.crack_width(sec, 900.0, 700.0, pc_class="B", tendon_bundles=BUNDLES, member="bending")


def test_class_b_axial_force_consistency_about_the_top():
    """consistency check about another moment centre (not an independent source): the steel force of the generalised
    formula balances the external actions taken about the top fibre, (Np0 + N)(h0 − z) − T z = Np0 hps + N zg −
    (ηs Ms + Mp2), with the compression resultant at h0 − z; Np0 at hps from σp0 Ap (σl6 = 0), N at the gross
    centroid"""
    sec = _pc_rect()
    As, Ap = 4 * math.pi * 100, 1112.0
    At = As + Ap
    Np0, hps = 1150 * Ap, 900.0
    for N, Mp2, l0 in [(800.0, 0.0, None), (800.0, 120.0, 22000.0), (-300.0, -60.0, None), (-1800.0, 0.0, None)]:
        r = jtg.crack_width(sec, 1100.0, 800.0, N=N, M_p2=Mp2, l0=l0, pc_class="B", tendon_bundles=BUNDLES)
        dd = r.details
        z, h0, eta = dd["z"], dd["h0"], dd.get("eta_s", 1.0)
        F = Np0 + N * 1e3
        T = (F * (h0 - z) - Np0 * hps - N * 1e3 * 500.0 + (eta * 1100.0 + Mp2) * 1e6) / z
        assert dd["sigma_ss"] == pytest.approx(T / At, rel=1e-10)
        assert dd["h0"] == pytest.approx((As * 960 + Ap * 900) / At)
    # ηs acts on Ms only, not on Mp2 (D60 4.2.2 item 1: no eccentricity amplification of the prestress)
    r = jtg.crack_width(sec, 1100.0, 800.0, N=800.0, M_p2=120.0, l0=22000.0, pc_class="B", tendon_bundles=BUNDLES)
    assert r.details["eta_s"] == pytest.approx(1 + (22000 / 1000) ** 2 / (4000 * (1100 / 800 * 1e3) / r.details["h0"]))


def test_class_b_net_tension_continuous_at_F_zero():
    """F = Np0 + N through zero: z → 0.87 h0 from the compression side (6.4.4-5 as e → ∞) and z = min(h0 − a', 0.87 h0)
    = 0.87 h0 on the tension side (a' = 40 mm): Wcr continuous, and growing with the tension"""
    sec = _pc_rect()
    Np0 = 1150 * 1112.0 / 1e3
    kw = dict(pc_class="B", tendon_bundles=BUNDLES, member="tension")
    w = {d: jtg.crack_width(sec, 1100.0, 800.0, N=-Np0 + d, **kw) for d in (1e-6, -1e-6)}
    assert w[1e-6].demand == pytest.approx(w[-1e-6].demand, rel=1e-6)
    assert w[-1e-6].details["z"] == pytest.approx(0.87 * w[-1e-6].details["h0"])
    W = [jtg.crack_width(sec, 1100.0, 800.0, N=-n, **kw).demand for n in np.linspace(300.0, 2500.0, 23)]
    assert all(b > a for a, b in zip(W[:-1], W[1:]))


def test_class_b_tension_zone_below_the_cracked_axis():
    """6.4.4-11 takes the steel of the tension zone: rows above the neutral axis of the cracked section are in
    compression and left out. The axis is found here on a fibre model (independent of appendix J's solver): web rows
    of 2φ20 at 560, 680 and 800 mm; with N = 1500 kN the axis drops below 560 and 680"""
    sec = _web_bars()
    model = _fibre_model(sec)
    Np0, hps = 1150 * 1112.0, 900.0
    for N in (0.0, 1500.0):
        r = jtg.crack_width(sec, 1300.0, 900.0, N=N, pc_class="B", tendon_bundles=BUNDLES, d=20.0)
        F = Np0 + N * 1e3
        x = _fibre_axis(model, F, (Np0 * hps + N * 1e3 * 500.0 - 1300e6) / F)
        assert r.details["x"] == pytest.approx(x, rel=1e-6)
        rows = [(a, z) for a, z in sec.bars] + [(1112.0, 900.0)]
        sel = [(a, z) for a, z in rows if z > max(500.0, x)]
        assert r.details["h0"] == pytest.approx(sum(a * z for a, z in sel) / sum(a for a, _ in sel), rel=1e-12)
        if N:
            assert 680 < x < 800
    # decompressed at the face, but the axis of the cracked section below every row (Ms 940 … 1000): no steel in the
    # tension zone, the steel below the centroid instead (the rule before the axis was used: tendon 1100 + bars 1150,
    # not A'p at 120), with a note; Wcr varies smoothly there. Between the tendon and the bars (Ms ≈ 1030) only the bars
    # are in the tension zone: the documented upward step; from Ms ≈ 1060 the tendon is back in
    s3 = jtg.rect(500, 1200, concrete="C50", bottom=[(4, 20, 50)],
                  tendons=[(1390, 1100, 1100, 90), (278, 120, 1150, 70)], sigma_pe=[1000.0, 1050.0])
    As = 4 * math.pi * 100
    w = {}
    for M in (950.0, 955.0, 1000.0, 1030.0, 1060.0):
        r = jtg.crack_width(s3, M, 700.0, pc_class="B", tendon_bundles=BUNDLES)
        w[M] = r.demand
        if M <= 1000:
            assert r.details["x"] > 1150 and "below all the bonded steel" in r.notes[0]
            assert r.details["h0"] == pytest.approx((1390 * 1100 + As * 1150) / (1390 + As), rel=1e-12)
        else:
            assert not r.notes and r.details["x"] < 1150
    assert 0 < w[950.0] and w[955.0] == pytest.approx(w[950.0], rel=1e-2)
    assert jtg.crack_width(s3, 1030.0, 700.0, pc_class="B", tendon_bundles=BUNDLES).details["h0"] == pytest.approx(1150)
    assert w[1030.0] > 1.5 * max(w[1000.0], w[1060.0])
    s_g = _pc_rect().replace(bars=np.zeros((0, 2)), bar_d=np.zeros(0), tendons=np.array([[1112.0, 300.0, 1150.0, 5000.0]]))
    with pytest.raises(ValueError, match="below the centroid"):  # no bonded steel below the centroid at all
        jtg.crack_width(s_g, 300.0, 200.0, pc_class="B", d=15.2, c=40.0)


def _web_bars():
    """_pc_rect with three web rows of 2φ20 at 560, 680 and 800 mm"""
    s = _pc_rect()
    web = np.array([[2 * math.pi * 100, z] for z in (560.0, 680.0, 800.0)])
    return s.replace(bars=np.r_[s.bars, web], bar_d=np.r_[s.bar_d, [20.0, 20.0, 20.0]])


def test_class_b_axial_force_decompression():
    """the tension face under Ms and N (at the gross centroid) on the transformed section against σpc; a large
    compression keeps it compressed: no crack"""
    sec = _pc_rect()
    A0, z0, I0 = sec.props("transformed")
    r = jtg.crack_width(sec, 1100.0, 800.0, N=600.0, pc_class="B", tendon_bundles=BUNDLES)
    sst = 1100e6 * (1000 - z0) / I0 - 600e3 / A0 - 600e3 * (500 - z0) * (1000 - z0) / I0
    assert r.details["sigma_st"] == pytest.approx(sst, rel=1e-12)
    big = jtg.crack_width(sec, 1100.0, 800.0, N=6000.0, pc_class="B", tendon_bundles=BUNDLES)
    assert big.demand == 0.0 and "not decompressed" in big.notes[0]
    # an axial tension decompresses earlier: larger σss than flexure, compression smaller
    w = [jtg.crack_width(sec, 1100.0, 800.0, N=N, pc_class="B", tendon_bundles=BUNDLES).details["sigma_ss"]
         for N in (-400.0, 0.0, 400.0)]
    assert w[0] > w[1] > w[2]


def _cubic_J(b, bf, hf, eN, aEP, aES, tension, comp_s):
    """appendix J (J.0.1-1 … -5): roots of A x³ + B x² + C x + D with the tension rows (α·A, h) and compression bars"""
    b0 = bf - b
    C = 3 * b0 * hf * (2 * eN + hf)
    D = -b0 * hf ** 2 * (3 * eN + 2 * hf)
    for aA, hh in tension:
        C += 6 * aA * (hh + eN)
        D -= 6 * aA * hh * (hh + eN)
    for A_, a_ in comp_s:
        C += 6 * aES * A_ * (a_ + eN)
        D -= 6 * aES * A_ * a_ * (a_ + eN)
    return [v.real for v in np.roots([b, 3 * b * eN, C, D]) if abs(v.imag) < 1e-6 and v.real > 0]


def test_appendix_J_axis_with_axial_force():
    """the cracked section under Np0 + N on their common line of action (z_R = (Np0 hps + N zg − M)/F): the cubic of
    appendix J with eN = −z_R (the cubic depends on the line of action only), T section, axis in the web"""
    sec = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], top=[(6, 12, 40)],
                  tendons=[(2000, 1470, 1000, 90)], sigma_pe=950.0)
    c = sec.concrete
    aEP, aES = 1.95e5 / c.Ec, 2e5 / c.Ec
    Np0, hps = 1000 * 2000.0, 1470.0
    As, Asc = 6 * math.pi * 64, 6 * math.pi * 36
    g = jtg.temperature_gradient(1600)
    for M, N in [(5500.0, 0.0), (5500.0, 900.0), (6000.0, -500.0)]:
        F = Np0 + N * 1e3
        zR = (Np0 * hps + N * 1e3 * sec.zg - M * 1e6) / F
        x = max(_cubic_J(200, 1800, 180, -zR, aEP, aES, [(aEP * 2000, 1470), (aES * As, 1550)], [(Asc, 40)]))
        th = jtg.thermal_stresses(sec, g, cracked=True, M=M, N=N)
        assert 180 < x < 1600 and th.z_max == pytest.approx(x, rel=1e-8)


# ── 2. class B stresses in service with the temperature gradient (7.1.4, appendix D on the cracked section) ──
def _tee_b():
    return jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], top=[(6, 12, 40)],
                   tendons=[(2000, 1470, 1000, 90)], sigma_pe=950.0)


def test_service_class_b_temperature_limits():
    """no temperature (a zero profile) changes nothing, bit for bit; a gradient with only a secondary moment acts as
    that moment in Mk (7.1.4 commentary: Mk holds the moments of all actions)"""
    sec = _tee_b()
    zero = jtg.thermal_stresses(sec, jtg.Profile((0.0, 1600.0), (0.0, 0.0)))
    for Mk in (5500.0, 1500.0):
        base = jtg.service_stresses(sec, Mk, pc_class="B")
        r = jtg.service_stresses(sec, Mk, pc_class="B", temperature=zero)
        assert r.demand == base.demand and np.array_equal(r.details["tendon_stress"], base.details["tendon_stress"])
        sec_only = jtg.thermal_stresses(sec, jtg.Profile((0.0, 1600.0), (0.0, 0.0)), M_sec=400.0)
        r = jtg.service_stresses(sec, Mk, pc_class="B", temperature=sec_only)
        cases = r.details["cases"]
        assert cases["positive"] == pytest.approx(jtg.service_stresses(sec, Mk + 400.0, pc_class="B").demand, rel=1e-12)
        assert cases["negative"] == pytest.approx(jtg.service_stresses(sec, Mk - 200.0, pc_class="B").demand, rel=1e-12)
        assert r.demand == max(cases.values())


@pytest.mark.parametrize("Mk, Msec, Nsec", [(5500.0, 300.0, 0.0), (6500.0, -250.0, 400.0), (1500.0, 200.0, 0.0)])
def test_service_class_b_temperature_fibres(Mk, Msec, Nsec):
    """σcc and the tendon stress with the gradient, against the fibre model: the cracked section fixed by Np0 (with the
    secondary axial force) and Mk + Mt′ (axis found on the fibres), then plane strain on those fibres with the free
    strain of the gradient above the axis; Mk = 1500 leaves the whole section compressed (x = ∞, the whole depth)"""
    sec = _tee_b()
    g = jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=60)
    th = jtg.thermal_stresses(sec, g, M_sec=Msec, N_sec=Nsec)
    r = jtg.service_stresses(sec, Mk, pc_class="B", temperature=th)
    model = _fibre_model(sec)
    Np0, hps = 1000 * 2000.0, 1470.0
    worst_c, worst_p = -1e9, -1e9
    for k, prof in ((1.0, g), (-0.5, g.scaled(-0.5))):
        M = Mk + k * Msec
        F = Np0 - k * Nsec * 1e3
        zR = (Np0 * hps + (-k * Nsec * 1e3) * sec.zg - M * 1e6) / F
        x = _fibre_axis(model, F, zR)
        sig, a, b = _fibre_solve(model, x, F, zR, prof, 1e-5, 1.0)
        top = sig(0.0)
        if x > sec.h:
            top = max(top, sig(sec.h))
        tendon = 1000 - 1.95e5 / sec.concrete.Ec * sig(1470.0, live=1470.0 <= x)
        name = "positive" if k > 0 else "negative"
        assert r.details["cases"][name] == pytest.approx(top, rel=1e-6, abs=1e-6)
        worst_c, worst_p = max(worst_c, top), max(worst_p, tendon)
    base = jtg.service_stresses(sec, Mk, pc_class="B")
    assert r.demand == pytest.approx(max(worst_c, base.demand), rel=1e-6)
    assert r.details["tendon_stress"][0] == pytest.approx(max(worst_p, base.details["tendon_stress"][0]), rel=1e-7)
    assert (r.details["x"] == math.inf) == (Mk == 1500.0)


def test_service_class_b_temperature_hogging_fibres():
    """hogging over a support: tendons near the top, the cracked top above the axis; the gradient (at the top) lies in
    the cracked zone and is ignored there (appendix D commentary, program reading for hogging)"""
    sec = jtg.rect(500, 1200, concrete="C50", bottom=[(5, 20, 50)], top=[(6, 20, 50)],
                   tendons=[(1668.0, 140, 1050, 90)], sigma_pe=950.0)
    g = jtg.temperature_gradient(1200)
    th = jtg.thermal_stresses(sec, g, M_sec=-150.0)
    Mk = -2600.0
    r = jtg.service_stresses(sec, Mk, pc_class="B", temperature=th)
    model = _fibre_model(sec, hogging=True)
    Np0, hps = 1050 * 1668.0, 1200 - 140.0  # in the turned section
    for k, prof in ((1.0, g), (-0.5, g.scaled(-0.5))):
        M = -(Mk + k * -150.0)
        zR = hps - M * 1e6 / Np0
        x = _fibre_axis(model, Np0, zR)
        t_s = lambda z, p=prof: p(sec.h - np.asarray(z))
        sig, _, _ = _fibre_solve(model, x, Np0, zR, t_s, 1e-5, 1.0)
        assert r.details["cases"]["positive" if k > 0 else "negative"] == pytest.approx(sig(0.0), rel=1e-6)
    assert r.details["edge"] == "bottom"


# ── 3. temperature gradient in the crack width (6.4 with appendix D on the cracked section) ──
def test_crack_width_temperature_limits():
    """zero profile: the result without temperature, bit for bit; secondary moment only: Ms + ψ Mt′ (ψ = 0.8) in σss
    and in C2's Ms, Ml unchanged (6.3.2 via 6.4.3)"""
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    zero = jtg.Profile((0.0, 1100.0), (0.0, 0.0))
    base = jtg.crack_width(t, 820.0, 640.0)
    r = jtg.crack_width(t, 820.0, 640.0, temperature=jtg.thermal_stresses(t, zero))
    assert r.demand == base.demand and r.details["sigma_ss"] == base.details["sigma_ss"]
    r = jtg.crack_width(t, 820.0, 640.0, temperature=jtg.thermal_stresses(t, zero, M_sec=150.0))
    ref = jtg.crack_width(t, 820.0 + 0.8 * 150.0, 640.0)
    assert r.details["temperature"] == "positive" and r.demand == pytest.approx(ref.demand, rel=1e-12)
    assert r.details["C2"] == pytest.approx(1 + 0.5 * 640 / (820 + 120))
    pc = _pc_rect()
    zb = jtg.thermal_stresses(pc, jtg.Profile((0.0, 1000.0), (0.0, 0.0)))
    b0 = jtg.crack_width(pc, 900.0, 700.0, pc_class="B", tendon_bundles=BUNDLES)
    b1 = jtg.crack_width(pc, 900.0, 700.0, pc_class="B", tendon_bundles=BUNDLES, temperature=zb)
    assert b1.demand == b0.demand


@pytest.mark.parametrize("Ms, hog", [(820.0, False), (-700.0, True)])
def test_crack_width_temperature_self_stress_rc_fibres(Ms, hog):
    """reinforced concrete in bending: the bar stress added to σss is ψ αES × the self-stress at h0 on the cracked
    section (axis of 7.2.4), against the fibre model with zero resultant, for the positive and the negative profile;
    the case with the larger Wcr is reported (σss = load part + that). A profile over the whole depth, so that the
    sagging and the hogging (turned) sections both see it"""
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(10, 28, 60)])
    prof = jtg.Profile((0.0, 300.0, 1100.0), (12.0, 4.0, 9.0))
    model = _fibre_model(t, hogging=hog)
    x = _fibre_axis(model)
    h0 = model[0].h0()
    th = jtg.thermal_stresses(t, prof)
    for psi in (0.8, 1.0):
        sst = {}
        for name, p in (("positive", prof), ("negative", prof.scaled(-0.5))):
            ts = (lambda z, p=p: p(t.h - np.asarray(z))) if hog else p
            sig, _, _ = _fibre_solve(model, x, 0.0, None, ts, 1e-5, psi)
            sst[name] = -2e5 / t.concrete.Ec * sig(h0, live=False)  # tension +
        assert sst["negative"] == pytest.approx(-0.5 * sst["positive"], rel=1e-9)  # linear on a fixed axis
        r = jtg.crack_width(t, Ms, 0.8 * Ms, temperature=th, psi_t=psi)
        d = r.details
        gov = max(sst, key=sst.get)
        assert d["temperature"] == gov and sst[gov] > 0
        assert d["sigma_s_t"] == pytest.approx(sst[gov], rel=1e-6)
        assert d["x_t"] == pytest.approx(x, rel=1e-8)
        assert d["sigma_ss"] == pytest.approx(d["sigma_ss_load"] + d["sigma_s_t"], rel=1e-12)
        assert d["sigma_ss_load"] == pytest.approx(jtg.crack_width(t, Ms, 0.8 * Ms).details["sigma_ss"], rel=1e-12)
    # the gradient of D60 is at the top: hogging cracks the top, the gradient is ignored there and nothing reaches the
    # bars; sagging: the negative gradient pulls the bottom bars
    g = jtg.thermal_stresses(t, jtg.temperature_gradient(1100, surfacing="asphalt", asphalt=100))
    r = jtg.crack_width(t, Ms, 0.8 * Ms, temperature=g)
    if hog:
        assert r.demand == jtg.crack_width(t, Ms, 0.8 * Ms).demand
    else:
        assert r.details["temperature"] == "negative" and r.details["sigma_s_t"] > 0


def test_crack_width_temperature_rc_eccentric_compression_fibres():
    """eccentric compression: the cracked section under N at the eccentricity ηs e0 (fibre axis), the self-stress at
    h0 added to 6.4.4-4"""
    sec = jtg.tee(300, 1000, 1200, 150, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 20, 40)])
    g = jtg.temperature_gradient(1000)
    th = jtg.thermal_stresses(sec, g, M_sec=60.0)
    r = jtg.crack_width(sec, 900.0, 700.0, N=800.0, temperature=th)
    name = r.details["temperature"]
    k = {"positive": 1.0, "negative": -0.5}[name]
    Ms, prof = 900.0 + 0.8 * k * 60.0, g.scaled(k)
    model = _fibre_model(sec)
    zR = sec.zg - Ms / 800.0 * 1e3
    x = _fibre_axis(model, 800e3, zR)
    sig, _, _ = _fibre_solve(model, x, 0.0, None, prof, 1e-5, 0.8)
    assert r.details["sigma_s_t"] == pytest.approx(-2e5 / sec.concrete.Ec * sig(950.0, live=False), rel=1e-6)
    load = jtg.crack_width(sec, Ms, 700.0, N=800.0)
    assert r.details["sigma_ss_load"] == pytest.approx(load.details["sigma_ss"], rel=1e-12)


def test_crack_width_temperature_class_b_fibres():
    """class B: the cracked section of appendix J under Np0 with Ms + ψ Mt′ ± Mp2, the self-stress at the resultant of
    Ap + As × αEP (Ep is the modulus of 6.4.3 for these members); the decompression test takes ψ × the uncracked
    temperature stress at the tension face"""
    sec = _tee_b()
    g = jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=100)
    th = jtg.thermal_stresses(sec, g, M_sec=-200.0)
    Ms, Ml, Mp2 = 5200.0, 4200.0, 150.0
    r = jtg.crack_width(sec, Ms, Ml, M_p2=Mp2, pc_class="B", tendon_bundles=[(2, 7, 15.2)], temperature=th)
    d = r.details
    k = {"positive": 1.0, "negative": -0.5}[d["temperature"]]
    model = _fibre_model(sec)
    Np0, hps = 1000 * 2000.0, 1470.0
    M = Ms + 0.8 * k * -200.0 + Mp2
    zR = hps - M * 1e6 / Np0
    x = _fibre_axis(model, Np0, zR)
    sig, _, _ = _fibre_solve(model, x, 0.0, None, g.scaled(k), 1e-5, 0.8)
    As, Ap = 6 * math.pi * 64, 2000.0
    h0 = (As * 1550 + Ap * 1470) / (As + Ap)
    assert d["h0"] == pytest.approx(h0)
    assert d["sigma_s_t"] == pytest.approx(-1.95e5 / sec.concrete.Ec * sig(h0, live=False), rel=1e-6)
    assert d["x_t"] == pytest.approx(x, rel=1e-7)
    A0, z0, I0 = sec.props("transformed")
    thk = th if k > 0 else th.negative()
    assert d["sigma_st"] == pytest.approx((Ms * 1e6 * (1600 - z0) / I0) - 0.8 * thk.sigma(1600.0), rel=1e-12)


def test_crack_width_temperature_special_cases():
    sec = jtg.rect(400, 600, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    th = jtg.thermal_stresses(sec, jtg.temperature_gradient(600))
    r = jtg.crack_width(sec, 0.0, 0.0, N=-500.0, Nl=-400.0, temperature=th)  # axial tension: no self-stress
    assert r.demand == jtg.crack_width(sec, 0.0, 0.0, N=-500.0, Nl=-400.0).demand
    col = jtg.circle(D=1500, n=28, d=28, cover=60, concrete="C35")
    th_col = jtg.thermal_stresses(col, jtg.temperature_gradient(1500))
    with pytest.raises(NotImplementedError):
        jtg.crack_width(col, 2500.0, 2000.0, N=3000.0, temperature=th_col)
    th_cr = jtg.thermal_stresses(sec, jtg.temperature_gradient(600), cracked=True)
    with pytest.raises(ValueError):  # the cracked section is built inside
        jtg.crack_width(sec, 150.0, 100.0, temperature=th_cr)


# ── 4. review cases: member type with the temperature's axial force, A'p, external tendons, hogging, axial forces ──
def test_temperature_axial_force_keeps_the_member_type():
    """the gradient's secondary axial force is an internal force, not a change of the member's nature: C3 stays that of
    the member and Wcr is continuous as N't → 0 (a flexural member's N goes through moments about the tension steel,
    which tends to 6.4.4-2 / 6.4.4-11); reinforced concrete without compression-face bars works with N't in tension"""
    zero = jtg.Profile((0.0, 1600.0), (0.0, 0.0))
    pc = _pc_rect()
    base = jtg.crack_width(pc, 1100.0, 800.0, pc_class="B", tendon_bundles=BUNDLES,
                           temperature=jtg.thermal_stresses(pc, zero, M_sec=250.0))
    for nt in (-5.0, 5.0, 1e-7, -1e-7):
        r = jtg.crack_width(pc, 1100.0, 800.0, pc_class="B", tendon_bundles=BUNDLES,
                            temperature=jtg.thermal_stresses(pc, zero, M_sec=250.0, N_sec=nt))
        assert r.details["C3"] == 1.0
        assert r.demand == pytest.approx(base.demand, rel=1e-8 if abs(nt) < 1 else 2e-3)
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)])
    g = jtg.temperature_gradient(1100, surfacing="asphalt", asphalt=100)
    ref = jtg.crack_width(t, 820.0, 640.0, temperature=jtg.thermal_stresses(t, g, M_sec=150.0))
    for nt in (-5.0, 5.0):
        r = jtg.crack_width(t, 820.0, 640.0, temperature=jtg.thermal_stresses(t, g, M_sec=150.0, N_sec=nt))
        assert r.details["C3"] == 1.0 and r.details["temperature"] == ref.details["temperature"] == "positive"
        assert r.demand == pytest.approx(ref.demand, rel=2e-3)
    small = jtg.crack_width(t, 820.0, 640.0, temperature=jtg.thermal_stresses(t, g, M_sec=150.0, N_sec=1e-7))
    assert small.demand == pytest.approx(ref.demand, rel=1e-9)


def test_reinforced_concrete_declared_flexural_with_axial_force():
    """member="flexural" with a small load N: C3 1.0, σss → Ms / (0.87 As h0) as N → 0± (6.4.4-2); the tension side
    uses z = 0.87 h0 here (h0 − a's = 920 > 0.87 h0), the compression side 6.4.4-5"""
    sec = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)], top=[(2, 16, 40)])
    base = jtg.crack_width(sec, 150.0, 100.0)
    for N in (1e-6, -1e-6):
        r = jtg.crack_width(sec, 150.0, 100.0, N=N, member="flexural")
        assert r.details["C3"] == 1.0 and r.demand == pytest.approx(base.demand, rel=1e-8)
    As, h0 = 4 * A25, 550.0
    r = jtg.crack_width(sec, 150.0, 100.0, N=-40.0, member="flexural")
    z = 0.87 * h0
    assert r.details["sigma_ss"] == pytest.approx((150e6 - 40e3 * (h0 - 300) + 40e3 * z) / (As * z))
    with pytest.raises(NotImplementedError):  # circular sections: compression members only
        jtg.crack_width(jtg.circle(D=1500, n=28, d=28, cover=60), 2500.0, 2000.0, N=3000.0, member="flexural")


def test_service_class_b_compression_tendon_with_temperature():
    """a compression-zone tendon still in tension (note 2 of J.0.1) keeps its sign in the cubic but counts with its
    positive area in the temperature self-stress (program convention): the self-stress at the top edge against the
    fibre model on the same axis, every tendon at +αEP Ap"""
    sec = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], top=[(6, 12, 40)],
                  tendons=[(2000, 1470, 1000, 90), (1400, 120, 1050, 80)], sigma_pe=[950.0, 1000.0])
    g = jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=60)
    th = jtg.thermal_stresses(sec, g)
    r = jtg.service_stresses(sec, 9000.0, pc_class="B", temperature=th)
    d = r.details
    assert d["temperature"] == "positive" and d["x"] > 120
    sig, _, _ = _fibre_solve(_fibre_model(sec), d["x"], 0.0, None, g, 1e-5, 1.0)
    assert d["sigma_t_edge"] == pytest.approx(sig(0.0), rel=1e-6)


def test_crack_width_temperature_external_tendons_fibres():
    """external tendons only (E = Es): they are part of Np0 at σpe,ex, not of the cracked section; the self-stress at
    the bars' resultant against the fibre model (bars only)"""
    sec = jtg.tee(600, 2500, 6000, 250, bf_bottom=3000, hf_bottom=220, concrete="C50", bottom=[(20, 20, 60)],
                  top=[(20, 16, 50)], external=[(14 * 1946.0, 2150, 1000)])
    g = jtg.temperature_gradient(2500)
    th = jtg.thermal_stresses(sec, g, M_sec=2000.0)
    r = jtg.crack_width(sec, 70000.0, 56000.0, pc_class="B", temperature=th)
    d = r.details
    k = {"positive": 1.0, "negative": -0.5}[d["temperature"]]
    assert d["E"] == 2e5
    Np0, hps = 14 * 1946.0 * 1000, 2150.0
    x = _fibre_axis(_fibre_model(sec), Np0, hps - (70000.0 + 0.8 * k * 2000.0) * 1e6 / Np0)
    sig, _, _ = _fibre_solve(_fibre_model(sec), x, 0.0, None, g.scaled(k), 1e-5, 0.8)
    assert d["x_t"] == pytest.approx(x, rel=1e-6)
    assert d["sigma_s_t"] == pytest.approx(-2e5 / sec.concrete.Ec * sig(2440.0, live=False), rel=1e-6)


def test_crack_width_temperature_class_b_hogging_fibres():
    """hogging with an axial force: the turned section, appendix J under Np0 + N, a profile over the whole depth so the
    compressed bottom sees it; the self-stress at Ap + As against the fibre model of the turned section"""
    sec = jtg.rect(500, 1200, concrete="C50", bottom=[(5, 20, 50)], top=[(6, 20, 50)],
                   tendons=[(1668.0, 140, 1050, 90)], sigma_pe=950.0)
    prof = jtg.Profile((0.0, 300.0, 1200.0), (6.0, 2.0, 8.0))
    th = jtg.thermal_stresses(sec, prof, M_sec=-150.0)
    r = jtg.crack_width(sec, -2600.0, -2000.0, N=400.0, pc_class="B", tendon_bundles=[(1, 12, 15.2)], temperature=th)
    d = r.details
    k = {"positive": 1.0, "negative": -0.5}[d["temperature"]]
    model = _fibre_model(sec, hogging=True)
    Np0, hps = 1050 * 1668.0, 1060.0
    M = -(-2600.0 + 0.8 * k * -150.0)
    F = Np0 + 400e3
    x = _fibre_axis(model, F, (Np0 * hps + 400e3 * 600.0 - M * 1e6) / F)
    sig, _, _ = _fibre_solve(model, x, 0.0, None, lambda z: k * prof(1200.0 - np.asarray(z)), 1e-5, 0.8)
    As, Ap = 6 * math.pi * 100, 1668.0
    h0 = (As * 1150 + Ap * 1060) / (As + Ap)
    assert d["h0"] == pytest.approx(h0) and d["x_t"] == pytest.approx(x, rel=1e-6)
    assert d["sigma_s_t"] == pytest.approx(-1.95e5 / sec.concrete.Ec * sig(h0, live=False), rel=1e-6)


def test_secondary_axial_force_at_the_gross_centroid():
    """N't acts at the gross centroid on the uncracked section too (as the load's N): σ = −N/A − N (zg − zc)(z − zc)/I"""
    sec = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2000, 1470, 1000, 90)],
                  sigma_pe=950.0)
    g = jtg.temperature_gradient(1600)
    a = jtg.thermal_stresses(sec, g)
    b = jtg.thermal_stresses(sec, g, N_sec=300.0)
    A, zc, I = sec.props("transformed")
    for z in (0.0, 700.0, 1600.0):
        assert b.sigma(z) - a.sigma(z) == pytest.approx(-300e3 / A - 300e3 * (sec.zg - zc) * (z - zc) / I, rel=1e-12)
    assert b.negative().sigma(0.0) == pytest.approx(-0.5 * b.sigma(0.0), rel=1e-12)


def test_service_stresses_axial_force():
    """an axial force in 7.1 (program extension): full and class A, N/A0 + N e y0/I0 at the edges and at the tendons;
    class B, the same resultant as a temperature's secondary axial force of −N (ψ = 1), and the fibre model"""
    a = jtg.rect(400, 1000, concrete="C50", bottom=[(4, 20, 40)], tendons=[(1112.0, 850, 1180, 80)], sigma_pe=1080.0,
                 sigma_l6=60.0)
    A0, z0, I0 = a.props("transformed")
    s0 = jtg.service_stresses(a, 1500.0, pc_class="A")
    s1 = jtg.service_stresses(a, 1500.0, pc_class="A", N=800.0)
    for name, z in (("top", 0.0), ("bottom", 1000.0)):
        dz = 800e3 / A0 + 800e3 * (500 - z0) * (z - z0) / I0
        assert s1.details["edge_stress"][name] - s0.details["edge_stress"][name] == pytest.approx(dz, rel=1e-12)
    dp = 800e3 / A0 + 800e3 * (500 - z0) * (850 - z0) / I0
    assert s1.details["tendon_stress"][0] - s0.details["tendon_stress"][0] == pytest.approx(-1.95e5 / a.concrete.Ec * dp)
    sec = _tee_b()
    zero = jtg.thermal_stresses(sec, jtg.Profile((0.0, 1600.0), (0.0, 0.0)), N_sec=-700.0)
    r = jtg.service_stresses(sec, 6000.0, pc_class="B", N=700.0)
    assert r.demand == pytest.approx(jtg.service_stresses(sec, 6000.0, pc_class="B", temperature=zero).details["cases"]
                                     ["positive"], rel=1e-12)
    F, zg = 2000e3 + 700e3, sec.zg
    x = _fibre_axis(_fibre_model(sec), F, (2000e3 * 1470 + 700e3 * zg - 6000e6) / F)
    sig, _, _ = _fibre_solve(_fibre_model(sec), x, F, (2000e3 * 1470 + 700e3 * zg - 6000e6) / F, lambda z: 0 * z,
                             1e-5, 1.0)
    assert r.demand == pytest.approx(sig(0.0), rel=1e-6)
