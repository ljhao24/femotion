"""Serviceability and stress checks (femotion.codes.jtg.sls) against hand calculations with the formulas of JTG
3362-2018 chapters 6, 7 and appendix J, and the combinations of JTG D60-2015 4.1.6."""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg.sls import _cracked

A25 = math.pi * 25 ** 2 / 4


# ── combinations (D60 4.1.6, 4.1.8) ──
def test_combinations():
    G = [np.array([100.0, -50.0]), (np.array([10.0, 10.0]), (1.2, 1.0))]  # factors of a basic-combination item ignored
    Q = (np.array([80.0, 10.0]), np.array([-20.0, -60.0]))
    crowd = (np.array([10.0, 0.0]), np.array([0.0, -5.0]))
    hi, lo = jtg.frequent_combination(G, Q, [crowd], kinds=["crowd"])
    assert hi == pytest.approx([110 + 0.7 * 80 + 0.4 * 10, -40 + 0.7 * 10])
    assert lo == pytest.approx([110 - 0.7 * 20, -40 - 0.7 * 60 - 0.4 * 5])
    hi, lo = jtg.quasi_permanent_combination(G, Q, [crowd], kinds=["crowd"])
    assert hi == pytest.approx([110 + 0.4 * 80 + 0.4 * 10, -40 + 0.4 * 10])
    assert lo == pytest.approx([110 - 0.4 * 20, -40 - 0.4 * 60 - 0.4 * 5])
    hi, lo = jtg.characteristic_combination(G, Q, [crowd])
    assert hi == pytest.approx([200, -30])
    assert lo == pytest.approx([90, -105])
    # wind as another action: ψq = 0.75; a number works too
    hi, _ = jtg.frequent_combination(np.zeros(2), Q, [crowd], kinds=[0.5])
    assert hi == pytest.approx([0.7 * 80 + 0.5 * 10, 7.0])
    assert jtg.PSI["wind"] == (0.75, 0.75) and jtg.PSI["temperature gradient"] == (0.8, 0.8)


def test_strength_at():
    assert jtg.strength_at(30) == pytest.approx((20.1, 2.01))
    assert jtg.strength_at(32.5) == pytest.approx(((20.1 + 23.4) / 2, (2.01 + 2.20) / 2))
    with pytest.raises(ValueError):
        jtg.strength_at(20)


# ── 6.4 crack width ──
def _wcr(C1, C2, C3, sss, E, c, d, rte):
    return C1 * C2 * C3 * sss / E * (c + d) / (0.36 + 1.7 * min(max(rte, 0.01), 0.1))


def test_crack_width_rc_flexure():
    sec = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    r = jtg.crack_width(sec, 150.0, 100.0)
    As, h0 = 4 * A25, 550.0
    sss = 150e6 / (0.87 * As * h0)
    assert r.details["sigma_ss"] == pytest.approx(sss)
    assert r.demand == pytest.approx(_wcr(1.0, 1 + 0.5 * 100 / 150, 1.0, sss, 2e5, 37.5, 25, As / (2 * 50 * 300)))
    assert r.resistance == 0.20 and r.ok
    # slab C3, plain bars C1, epoxy C1, environment III
    assert jtg.crack_width(sec, 150, 100, slab=True).demand == pytest.approx(1.15 * r.demand)
    assert jtg.crack_width(sec, 150, 100, epoxy=True).demand == pytest.approx(1.15 * r.demand)
    plain = jtg.rect(300, 600, concrete="C30", steel="HPB300", bottom=[(4, 25, 50)])
    assert jtg.crack_width(plain, 150, 100).details["C1"] == 1.4
    assert jtg.crack_width(plain, 150, 100).details["E"] == 2.1e5
    assert jtg.crack_width(sec, 150, 100, env="III").resistance == 0.15
    # hogging: the top bars; cover capped at 50; ρte at most 0.1; mixed diameters → de
    s2 = jtg.rect(300, 600, concrete="C30", bottom=[(2, 12, 40)], top=[(2, 28, 90), (2, 20, 90)])
    r2 = jtg.crack_width(s2, -120.0, -90.0)
    de = (2 * 28 ** 2 + 2 * 20 ** 2) / (2 * 28 + 2 * 20)
    As2 = 2 * math.pi * (28 ** 2 + 20 ** 2) / 4
    sss2 = 120e6 / (0.87 * As2 * 510)
    assert r2.details["d"] == pytest.approx(de)
    assert r2.details["c"] == 50.0  # 90 − 14 = 76 → 50
    assert r2.demand == pytest.approx(_wcr(1.0, 1 + 0.5 * 90 / 120, 1.0, sss2, 2e5, 50, de, As2 / (2 * 90 * 300)))
    # welded cage: d × 1.3
    r3 = jtg.crack_width(sec, 150, 100, welded_cage=True)
    assert r3.details["d"] == pytest.approx(32.5)


def test_crack_width_tee_effective_area():
    """T beam: tension face is the web (b = 200); with a bottom flange thicker than 2as, Ate = 2 as b of that flange"""
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 28, 70)], bf_bottom=500, hf_bottom=200)
    r = jtg.crack_width(t, 900.0, 700.0)
    assert r.details["Ate"] == pytest.approx(2 * 70 * 500)


def test_crack_width_eccentric_compression_rect():
    sec = jtg.rect(400, 600, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    As, h0 = 4 * A25, 555.0
    for l0, eta in [(6000, 1.0), (12000, 1 + 20 ** 2 / (4000 * 600 / 555))]:
        r = jtg.crack_width(sec, 300.0, 240.0, N=500.0, l0=l0)
        es = eta * 600 + (555 - 300)
        z = (0.87 - 0.12 * (h0 / es) ** 2) * h0
        sss = 500e3 * (es - z) / (As * z)
        assert r.details["eta_s"] == pytest.approx(eta)
        assert r.details["sigma_ss"] == pytest.approx(sss)
        assert r.demand == pytest.approx(_wcr(1.0, 1.4, 0.9, sss, 2e5, 32.5, 25, As / (2 * 45 * 400)))
    # e0/h ≤ 0.55: not needed
    r = jtg.crack_width(sec, 150.0, 100.0, N=500.0)
    assert r.demand == 0 and r.ok and "0.55" in r.notes[0]


def test_crack_width_eccentric_compression_tee_flange():
    t = jtg.tee(300, 1000, 1200, 150, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 20, 40)])
    r = jtg.crack_width(t, 900.0, 700.0, N=800.0)
    h0 = 950.0
    gf = (1200 - 300) * 150 / (300 * h0)
    assert r.details["gamma_f"] == pytest.approx(gf)
    es = 900 / 800 * 1e3 + h0 - t.zg
    z = (0.87 - 0.12 * (1 - gf) * (h0 / es) ** 2) * h0
    assert r.details["sigma_ss"] == pytest.approx(800e3 * (es - z) / (5 * A25 * z))
    # h'f ≤ 0.2 h0
    t2 = jtg.tee(300, 1000, 1200, 250, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 20, 40)])
    assert jtg.crack_width(t2, 900.0, 700.0, N=800.0).details["gamma_f"] == pytest.approx(900 * 0.2 * h0 / (300 * h0))


def test_crack_width_circle():
    col = jtg.circle(D=1500, n=28, d=28, cover=60, concrete="C35")
    r_, rs, a_s = 750.0, 690.0, 60.0
    Ns, Ms, l0 = 3000.0, 2500.0, 25000.0
    e0 = Ms / Ns * 1e3
    eta = 1 + (l0 / 1500) ** 2 / (4000 * e0 / (1500 - a_s))
    t = eta * e0 / r_
    As = 28 * math.pi * 28 ** 2 / 4
    sss = 0.6 * (t - 0.1) ** 3 / ((0.45 + 0.26 * rs / r_) * (t + 0.2) ** 2) * Ns * 1e3 / As
    rho = As / (math.pi * r_ ** 2)
    beta = (0.4 + 2.5 * rho) * (1 + 0.353 * t ** -2)
    rte = beta * As / (math.pi * (r_ ** 2 - (r_ - 2 * a_s) ** 2))
    r = jtg.crack_width(col, Ms, 2000.0, N=Ns, l0=l0)
    assert r.details["sigma_ss"] == pytest.approx(sss)
    assert r.demand == pytest.approx(_wcr(1.0, 1.4, 0.75, sss, 2e5, 46.0, 28, rte))
    assert jtg.crack_width(col, Ms, 2000.0, N=Ns, l0=20000).details["eta_s"] == 1.0  # l0/2r ≤ 14
    assert jtg.crack_width(col, 1000.0, 800.0, N=3000.0).demand == 0.0  # e0/r ≤ 0.55
    with pytest.raises(NotImplementedError):
        jtg.crack_width(jtg.circle(D=1500, n=28, d=28, cover=60, hollow=900), Ms, 2000.0, N=Ns)


def test_crack_width_tension():
    sec = jtg.rect(400, 600, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    As = 8 * A25
    r = jtg.crack_width(sec, 0.0, 0.0, N=-800.0, Nl=-600.0)
    sss = 800e3 / As
    assert r.demand == pytest.approx(_wcr(1.0, 1 + 0.5 * 600 / 800, 1.2, sss, 2e5, 32.5, 25, As / (400 * 600)))
    # eccentric tension: moments about A's
    r = jtg.crack_width(sec, 100.0, 80.0, N=-500.0)
    e0 = 200.0
    es_ = 300 + e0 - 45
    sss = 500e3 * es_ / (4 * A25 * (555 - 45))
    assert r.details["sigma_ss"] == pytest.approx(sss)
    assert r.demand == pytest.approx(_wcr(1.0, 1.4, 1.1, sss, 2e5, 32.5, 25, 4 * A25 / (2 * 45 * 400)))


def test_crack_width_class_b():
    sec = jtg.rect(400, 1000, concrete="C50", bottom=[(4, 20, 40)], top=[(4, 16, 40)],
                   tendons=[(1112.0, 900, 1150, 80)], sigma_pe=1050.0, sigma_l6=(60.0, 40.0))
    Ms, Ml, Mp2 = 900.0, 700.0, 50.0
    As, Ap = 4 * math.pi * 100, 1112.0
    # Np0 = σp0 Ap − σl6 As (σ'l6 = 0: no tendon above the centroid, 6.1.7 note)
    Np0 = 1150 * Ap - 60 * As
    zN = (1150 * Ap * 900 - 60 * As * 960) / Np0
    h0 = (As * 960 + Ap * 900) / (As + Ap)
    ep = h0 - zN
    e = ep + (Ms + Mp2) * 1e6 / Np0
    z = (0.87 - 0.12 * (h0 / e) ** 2) * h0
    sss = ((Ms + Mp2) * 1e6 - Np0 * (z - ep)) / ((As + Ap) * z)
    de = (4 * 20 ** 2 + 1 * (math.sqrt(8) * 15.2) ** 2) / (4 * 20 + math.sqrt(8) * 15.2)
    r = jtg.crack_width(sec, Ms, Ml, pc_class="B", M_p2=Mp2, tendon_bundles=[(1, 8, 15.2)])
    assert r.details["sigma_ss"] == pytest.approx(sss)
    assert r.details["d"] == pytest.approx(de)
    rte = (As + Ap) / (2 * (1000 - h0) * 400)
    assert r.demand == pytest.approx(_wcr(1.0, 1 + 0.5 * Ml / Ms, 1.0, sss, 1.95e5, 30.0, de, rte))
    assert r.resistance == 0.10
    assert not jtg.crack_width(sec, Ms, Ml, pc_class="B", M_p2=Mp2, tendon_bundles=[(1, 8, 15.2)], env="V").ok
    # not decompressed → no crack
    assert jtg.crack_width(sec, 100.0, 80.0, pc_class="B", tendon_bundles=[(1, 8, 15.2)]).demand == 0.0
    with pytest.raises(ValueError):
        jtg.crack_width(sec, Ms, Ml, pc_class="A")
    with pytest.raises(ValueError):
        jtg.crack_width(sec, Ms, Ml, pc_class="B")  # no tendon_bundles or d
    # threaded bars: the reinforced-concrete column of table 6.4.2
    sb = sec.replace(tendon=jtg.tendon("bar", 930))
    assert jtg.crack_width_limit("III", sb) == 0.15 and jtg.crack_width_limit("III", sec) == 0.10


# ── cracked sections (7.2.4, appendix J) ──
def test_cracked_rect_and_tee():
    sec = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    aE = 2e5 / 3e4
    As, h0 = 4 * A25, 550.0
    x = (-aE * As + math.sqrt((aE * As) ** 2 + 2 * 300 * aE * As * h0)) / 300
    r = jtg.cracked_section(sec)
    assert r["x0"] == pytest.approx(x)
    assert r["Icr"] == pytest.approx(300 * x ** 3 / 3 + aE * As * (h0 - x) ** 2)
    # with compression bars (αES A's, 7.2.4 commentary)
    s2 = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)], top=[(2, 20, 40)])
    Asc = 2 * math.pi * 100
    a_, b_, c_ = 150, aE * (As + Asc), -aE * (As * h0 + Asc * 40)
    x2 = (-b_ + math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    assert jtg.cracked_section(s2)["x0"] == pytest.approx(x2)
    # T with the axis in the web
    t = jtg.tee(200, 1000, 1000, 100, concrete="C40", bottom=[(8, 32, 80)])
    aE = 2e5 / 3.25e4
    As, h0 = 8 * math.pi * 256, 920.0
    # 0.5 b x² + (bf − b) hf (x − hf/2) = αE As (h0 − x)
    a_, b_, c_ = 100, 800 * 100 + aE * As, -(800 * 100 * 50 + aE * As * h0)
    x = (-b_ + math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    assert x > 100
    r = jtg.cracked_section(t)
    assert r["x0"] == pytest.approx(x)
    I = 200 * x ** 3 / 3 + 800 * 100 ** 3 / 12 + 800 * 100 * (x - 50) ** 2 + aE * As * (h0 - x) ** 2
    assert r["Icr"] == pytest.approx(I)


def _cubic_J(b, bf, hf, eN, aEP, aES, tension, comp_p, comp_s, sign_p=1.0):
    """appendix J: A x³ + B x² + C x + D = 0; tension rows (Ap or As as (α·A, h)), comp_p (A'p, a'p), comp_s (A's, a's)"""
    b0 = bf - b
    C = 3 * b0 * hf * (2 * eN + hf)
    D = -b0 * hf ** 2 * (3 * eN + 2 * hf)
    for aA, hh in tension:
        g = hh + eN
        C += 6 * aA * g
        D -= 6 * aA * hh * g
    for A_, a_ in comp_p:
        g = a_ + eN
        C += sign_p * 6 * aEP * A_ * g
        D -= sign_p * 6 * aEP * A_ * a_ * g
    for A_, a_ in comp_s:
        g = a_ + eN
        C += 6 * aES * A_ * g
        D -= 6 * aES * A_ * a_ * g
    roots = np.roots([b, 3 * b * eN, C, D])
    return [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]


def test_class_b_cracked_matches_appendix_J():
    sec = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], top=[(6, 12, 40)],
                  tendons=[(2780, 1470, 1150, 90)], sigma_pe=1080.0, sigma_l6=(70.0, 50.0))
    c = sec.concrete
    aEP, aES = 1.95e5 / c.Ec, 2e5 / c.Ec
    Mk = 6500.0
    r = jtg.service_stresses(sec, Mk, pc_class="B")
    Np0, hps = sec.prestress_resultant("p0")
    eN = Mk * 1e6 / Np0 - hps
    As, Asc = 6 * math.pi * 64, 6 * math.pi * 36
    xs = _cubic_J(200, 1800, 180, eN, aEP, aES, [(aEP * 2780, 1470), (aES * As, 1550)], [], [(Asc, 40)])
    assert r.details["x"] == pytest.approx(max(xs), rel=1e-7)
    # 7.1.4-1 and -5 from Acr, Icr, c, e0N
    d = r.details
    scc = Np0 / d["Acr"] + Np0 * d["e0N"] * d["c"] / d["Icr"]
    assert r.demand == pytest.approx(scc)
    sp = aEP * (Np0 / d["Acr"] - Np0 * d["e0N"] * (1470 - d["c"]) / d["Icr"])
    assert d["sigma_p"][0] == pytest.approx(sp)
    assert d["tendon_stress"][0] == pytest.approx(1150 - sp)


def test_class_b_compression_tendon_sign_note2():
    """J.0.1 note 2: a compression-zone tendon still in tension after the concrete increment counts with a minus sign"""
    sec = jtg.rect(500, 1200, concrete="C50", bottom=[(4, 20, 50)],
                   tendons=[(1390, 1100, 1100, 90), (278, 120, 1150, 70)], sigma_pe=[1000.0, 1050.0])
    c = sec.concrete
    aEP, aES = 1.95e5 / c.Ec, 2e5 / c.Ec
    Mk = 2600.0
    r = jtg.service_stresses(sec, Mk, pc_class="B")
    Np0, hps = sec.prestress_resultant("p0")
    eN = Mk * 1e6 / Np0 - hps
    As = 4 * math.pi * 100
    xs = _cubic_J(500, 500, 0, eN, aEP, aES, [(aEP * 1390, 1100), (aES * As, 1150)], [(278, 120)], [], sign_p=-1.0)
    assert r.details["x"] == pytest.approx(max(xs), rel=1e-7)
    assert r.details["x"] > 120
    plus = _cubic_J(500, 500, 0, eN, aEP, aES, [(aEP * 1390, 1100), (aES * As, 1150)], [(278, 120)], [])
    assert abs(max(plus) - r.details["x"]) > 1.0  # the sign matters here


def test_compression_bar_cap():
    """7.2.4 commentary / J.0.1 note 1: compression bars at most f'sd"""
    sec = jtg.rect(200, 400, concrete="C25", bottom=[(4, 28, 40)], top=[(2, 12, 40)])
    M = 450e6  # far beyond service, only to reach f'sd
    r = _cracked(sec, M=M)
    aS = 2e5 / 2.8e4
    sc = r["sigma"](40.0)
    w = r["w_bars"][1]
    assert w < 1
    assert w * aS * sc == pytest.approx(330.0, rel=1e-5)


# ── 6.3 crack resistance ──
def _pc_rect():
    return jtg.rect(400, 1000, concrete="C50", bottom=[(4, 20, 40)], tendons=[(1112.0, 850, 1180, 80)],
                    sigma_pe=1080.0, sigma_l6=60.0)


def test_crack_resistance():
    sec = _pc_rect()
    A0, z0, I0 = sec.props("transformed")
    An, zn, In = sec.props("net")
    W0 = I0 / (1000 - z0)
    As = 4 * math.pi * 100
    Np = 1080 * 1112 - 60 * As
    epn = (1080 * 1112 * 850 - 60 * As * 960) / Np - zn
    Mp2 = 80.0
    spc = Np / An + Np * epn * (1000 - zn) / In - Mp2 * 1e6 * (1000 - zn) / In
    Ms, Ml = 500.0, 380.0
    full = jtg.crack_resistance(sec, Ms, Ml, pc_class="full", M_p2=Mp2)
    assert full.details["sigma_pc"] == pytest.approx(spc)
    assert full.demand == pytest.approx(Ms * 1e6 / W0)
    assert full.resistance == pytest.approx(0.85 * spc)
    assert jtg.crack_resistance(sec, Ms, Ml, pc_class="full", M_p2=Mp2, segmental=True).resistance == pytest.approx(0.80 * spc)
    a = jtg.crack_resistance(sec, Ms, Ml, pc_class="A", M_p2=Mp2)
    assert a.resistance == pytest.approx(spc + 0.7 * 2.65)
    assert a.ok == (Ms * 1e6 / W0 <= spc + 0.7 * 2.65 and Ml * 1e6 / W0 <= spc)
    bad = jtg.crack_resistance(sec, Ms, spc * W0 / 1e6 + 10, pc_class="A", M_p2=Mp2)
    assert any("6.3.1-4" in f for f in bad.failed)
    b = jtg.crack_resistance(sec, 900.0, 600.0, pc_class="B", M_p2=Mp2, M_G=300.0)
    assert b.demand == pytest.approx(300e6 / W0) and b.resistance == pytest.approx(spc)
    with pytest.raises(ValueError):
        jtg.crack_resistance(sec, 900.0, 600.0, pc_class="B")
    with pytest.raises(ValueError):
        jtg.crack_resistance(sec.replace(sigma_pe=None), Ms, Ml, pc_class="A")


def test_crack_resistance_pretensioned_and_hogging():
    pre = jtg.rect(400, 800, concrete="C50", tendons=[(556.0, 700, 1200)], pretensioned=True, sigma_pe=1100.0)
    A0, z0, I0 = pre.props("transformed")
    spc = 1200 * 556 / A0 + 1200 * 556 * (700 - z0) * (800 - z0) / I0  # Np0 on the transformed section (6.1.6-1)
    assert jtg.crack_resistance(pre, 200.0, 150.0, pc_class="A").details["sigma_pc"] == pytest.approx(spc)
    # hogging: the top edge
    hog = jtg.rect(400, 800, concrete="C50", tendons=[(556.0, 100, 1200)], pretensioned=True, sigma_pe=1100.0)
    r = jtg.crack_resistance(hog, -200.0, -150.0, pc_class="A")
    A0h, z0h, I0h = hog.props("transformed")
    spc_top = 1200 * 556 / A0h + 1200 * 556 * (z0h - 100) * z0h / I0h
    assert r.details["edge"] == "top"
    assert r.details["sigma_pc"] == pytest.approx(spc_top)
    assert r.demand == pytest.approx(200e6 * z0h / I0h)


def test_principal_stresses():
    sec = jtg.rect(300, 900, concrete="C50")  # no steel: τ = 1.5 V / (b h) at the centroid
    p = jtg.principal_stresses(sec, 0.0, 400.0)
    assert len(p) == 1 and p[0]["z"] == pytest.approx(450)
    assert p[0]["tau"] == pytest.approx(1.5 * 400e3 / (300 * 900))
    assert p[0]["sigma_tp"] == pytest.approx(p[0]["tau"])
    # with M and vertical prestress at another depth
    p = jtg.principal_stresses(sec, 300.0, 400.0, depths=[300.0], vertical=(2 * 804.2, 500, 600.0), sigma_cy=0.3)
    I = 300 * 900 ** 3 / 12
    scx = 300e6 * 150 / I
    scy = 0.6 * 2 * 804.2 * 600 / (300 * 500) + 0.3
    tau = 400e3 * (300 * 300 * 300) / (300 * I)
    m, r = (scx + scy) / 2, math.hypot((scx - scy) / 2, tau)
    assert p[0]["sigma_tp"] == pytest.approx(r - m) and p[0]["sigma_cp"] == pytest.approx(m + r)
    # T section: points at the centroid and the web–flange junction (web width)
    t = jtg.tee(200, 1200, 1600, 150, concrete="C50")
    pts = jtg.principal_stresses(t, 0.0, 500.0)
    assert [q["z"] for q in pts][0] == pytest.approx(150) and all(q["b"] == 200 for q in pts)


def test_principal_stresses_prestressed_bent_tendon():
    sec = _pc_rect()
    A0, z0, I0 = sec.props("transformed")
    An, zn, In = sec.props("net")
    V, th = 600.0, 8.0
    p = jtg.principal_stresses(sec, 400.0, V, depths=[z0], bent_tendons=[(1112.0, th)])
    tau = V * 1e3 * sec.first_moment(z0) / (400 * I0) - 1080 * 1112 * math.sin(math.radians(th)) * sec.first_moment(z0, "net") / (400 * In)
    assert p[0]["tau"] == pytest.approx(tau)
    assert p[0]["sigma_cx"] == pytest.approx(float(sec.prestress_stress(z0)))
    r = jtg.inclined_crack_resistance(sec, 400.0, V, pc_class="A", depths=[z0], bent_tendons=[(1112.0, th)])
    assert r.resistance == pytest.approx(0.7 * 2.65)
    assert jtg.inclined_crack_resistance(sec, 400.0, V, pc_class="full", precast=False).resistance == pytest.approx(0.4 * 2.65)


def test_principal_compression_and_stirrups():
    sec = _pc_rect()
    r = jtg.principal_compression(sec, 700.0, 900.0, stirrups=(2, 10, 300))
    assert r.resistance == pytest.approx(0.6 * 32.4)
    st = r.details["sigma_tp"]
    assert st > 0.5 * 2.65
    assert r.details["sv_max"] == pytest.approx(400 * 2 * math.pi * 25 / (st * 400))
    assert (300 > r.details["sv_max"]) == bool(r.failed)
    low = jtg.principal_compression(sec, 700.0, 100.0)
    assert "detailing" in low.notes[0]


# ── 6.5 stiffness and deflection ──
def test_stiffness_rc():
    sec = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    c = sec.concrete
    A0, z0, I0 = sec.props("transformed")
    B0 = 0.95 * c.Ec * I0
    S0 = sec.first_moment(z0)
    Mcr = 2 * S0 * c.ftk  # γ ftk W0 with γ = 2 S0 / W0
    Icr = jtg.cracked_section(sec)["Icr"]
    Bcr = c.Ec * Icr
    Ms = 150e6
    B, d = jtg.stiffness(sec, 150.0)
    assert d["Mcr"] == pytest.approx(Mcr / 1e6)
    assert B == pytest.approx(B0 / ((Mcr / Ms) ** 2 + (1 - (Mcr / Ms) ** 2) * B0 / Bcr))
    assert jtg.stiffness(sec, 0.5 * Mcr / 1e6)[0] == pytest.approx(B0)
    # hogging uses the flipped section
    assert jtg.stiffness(sec, -10.0)[1]["I0"] == pytest.approx(I0)


def test_stiffness_prestressed():
    sec = _pc_rect()
    c = sec.concrete
    A0, z0, I0 = sec.props("transformed")
    assert jtg.stiffness(sec, 900.0, pc_class="A")[0] == pytest.approx(0.95 * c.Ec * I0)
    Mcr = jtg.cracking_moment(sec)[0] * 1e6
    Bcr = c.Ec * jtg.cracked_section(sec)["Icr"]
    M = (1500.0 + 50.0) * 1e6
    B, d = jtg.stiffness(sec, 1500.0, pc_class="B", M_p2=50.0)
    assert M > Mcr
    assert B == pytest.approx(M / (Mcr / (0.95 * c.Ec * I0) + (M - Mcr) / Bcr))


def test_long_term_and_limits():
    assert jtg.long_term_factor(jtg.concrete(30)) == 1.60
    assert jtg.long_term_factor(jtg.concrete(40)) == pytest.approx(1.45)
    assert jtg.long_term_factor(jtg.concrete(60)) == pytest.approx(1.40)
    assert jtg.long_term_factor(jtg.concrete(80)) == pytest.approx(1.35)
    assert jtg.deflection_limit(30000) == 50 and jtg.deflection_limit(3000, cantilever=True) == 10
    assert jtg.deflection_check(45.0, 30000).ok and not jtg.deflection_check(51.0, 30000).ok
    assert jtg.precamber_rc(5.0, 4.0, 16000) == 0.0  # 9 ≤ 10
    assert jtg.precamber_rc(8.0, 6.0, 16000) == pytest.approx(11.0)
    assert jtg.precamber_pc(20.0, 25.0) == 0.0 and jtg.precamber_pc(30.0, 25.0) == pytest.approx(5.0)


# ── 7.1 stresses in service ──
def test_service_stresses_class_a():
    sec = _pc_rect()
    A0, z0, I0 = sec.props("transformed")
    Mk, Mp2 = 800.0, 60.0
    top = Mk * 1e6 * z0 / I0 + float(sec.prestress_stress(0.0, Mp2))
    r = jtg.service_stresses(sec, Mk, pc_class="A", M_p2=Mp2)
    assert r.demand == pytest.approx(top) and r.resistance == pytest.approx(0.5 * 32.4)
    skt = Mk * 1e6 * (850 - z0) / I0
    assert r.details["tendon_stress"][0] == pytest.approx(1080 + 1.95e5 / 3.45e4 * skt)
    assert r.details["tendon_limit"] == pytest.approx(0.65 * 1860)
    # threaded bars: 0.75 fpk
    rb = jtg.service_stresses(sec.replace(tendon=jtg.tendon("bar", 930)), Mk, pc_class="A")
    assert rb.details["tendon_limit"] == pytest.approx(0.75 * 930)
    # small moment: the bottom edge (precompressed) governs
    r0 = jtg.service_stresses(sec, 0.0, pc_class="A")
    assert "bottom" in r0.name


# ── 7.2 transient situations ──
def test_transient_rc():
    sec = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    aE = 2e5 / 3e4
    As, h0 = 4 * A25, 550.0
    x = (-aE * As + math.sqrt((aE * As) ** 2 + 2 * 300 * aE * As * h0)) / 300
    I = 300 * x ** 3 / 3 + aE * As * (h0 - x) ** 2
    M, V = 120.0, 150.0
    r = jtg.transient_rc(sec, M, V, fcu=25)
    assert r.demand == pytest.approx(M * 1e6 * x / I)
    assert r.resistance == pytest.approx(0.8 * 16.7)
    assert r.details["sigma_s"][0] == pytest.approx(aE * M * 1e6 * (h0 - x) / I)
    assert r.details["sigma_tp"] == pytest.approx(V * 1e3 / (300 * (h0 - x / 3)))
    assert r.details["z0"] == pytest.approx(h0 - x / 3)
    big = jtg.transient_rc(sec, M, 900.0, fcu=25)
    assert any("7.2.5" in f for f in big.failed)
    with pytest.raises(ValueError):
        jtg.transient_rc(_pc_rect(), M, fcu=40)


def test_transient_pc():
    sec = jtg.rect(400, 1000, concrete="C50", top=[(4, 12, 40)], tendons=[(1390.0, 850, 1300, 90)],
                   sigma_pe=1250.0)
    An, zn, In = sec.props("net")
    M = 150.0
    Np = 1250 * 1390.0
    sig = lambda z: Np / An + Np * (850 - zn) * (z - zn) / In + M * 1e6 * (zn - z) / In
    r = jtg.transient_pc(sec, M, fcu=40)
    assert r.details["top"] == pytest.approx(sig(0)) and r.details["bottom"] == pytest.approx(sig(1000))
    assert r.demand == pytest.approx(max(sig(0), sig(1000)))
    assert r.resistance == pytest.approx(0.7 * 26.8)
    sct = -sig(0)
    assert sct > 0
    ftk = 2.40
    need = 0.002 if sct <= 0.7 * ftk else min(0.002 + 0.002 * (sct - 0.7 * ftk) / (0.45 * ftk), 0.004)
    assert r.details["rho_needed"] == pytest.approx(need)
    assert r.details["rho_pretension_zone"] == pytest.approx(4 * math.pi * 36 / 400000)
    # after grouting the loads act on the transformed section
    A0, z0, I0 = sec.props("transformed")
    rg = jtg.transient_pc(sec, M, fcu=40, grouted=True)
    assert rg.details["top"] == pytest.approx(Np / An + Np * (850 - zn) * (0 - zn) / In + M * 1e6 * z0 / I0)


# ── cases from the independent review ──
def _rect_props(b, h, pts):
    """area, centroid depth and second moment of a b × h rectangle plus (area, depth) points"""
    A = b * h + sum(a for a, _ in pts)
    z = (b * h * h / 2 + sum(a * d for a, d in pts)) / A
    I = b * h ** 3 / 12 + b * h * (h / 2 - z) ** 2 + sum(a * (d - z) ** 2 for a, d in pts)
    return A, z, I


def test_class_b_uncracked_small_moment():
    """small Mk (and Mk = 0): the whole section stays compressed; the larger edge compression is checked (7.1.4 with
    the full section) — no error"""
    sec = jtg.rect(400, 1000, concrete="C50", bottom=[(4, 20, 40)], tendons=[(1112.0, 850, 1180, 80)], sigma_pe=1080.0)
    c = sec.concrete
    aEP, aES = 1.95e5 / c.Ec, 2e5 / c.Ec
    As = 4 * math.pi * 100
    A, z, I = _rect_props(400, 1000, [(aEP * 1112, 850), (aES * As, 960)])  # gross concrete + αE·A (appendix J)
    Np0 = 1180 * 1112.0  # σl6 = 0
    hps = 850.0
    for Mk in (0.0, 200.0):
        zR = hps - Mk * 1e6 / Np0
        e = z - zR
        bottom = Np0 / A + Np0 * e * (z - 1000) / I
        r = jtg.service_stresses(sec, Mk, pc_class="B")
        assert r.details["x"] == math.inf and r.details["edge"] == "bottom"
        assert r.demand == pytest.approx(bottom)
    # hogging moment on the same section: the bottom is the compressed face
    r = jtg.service_stresses(sec, -300.0, pc_class="B")
    assert r.details["edge"] == "bottom" and r.demand > 0


def _equilibrium(sec, r, Np0, zR):
    """resultant force and its depth of the stress field r["sigma"] over the cracked section (numerical integration of
    the concrete above x, plus every bar and tendon at αE times the concrete stress at its depth)"""
    c = sec.concrete
    x = min(r["x"], sec.h)
    cuts = sorted({0.0, x} | {v for p in sec.pieces for v in (p.top, p.bottom) if 0 < v < x})
    F = Mz = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):  # constant width between cuts, linear stress: Simpson is exact
        w = sec.width((a + b) / 2)
        m = (a + b) / 2
        F += w * (b - a) / 6 * (r["sigma"](a) + 4 * r["sigma"](m) + r["sigma"](b))
        Mz += w * (b - a) / 6 * (r["sigma"](a) * a + 4 * r["sigma"](m) * m + r["sigma"](b) * b)
    for (a, d), wb in zip(sec.bars, r["w_bars"]):
        F += 2e5 / c.Ec * wb * a * r["sigma"](d)
        Mz += 2e5 / c.Ec * wb * a * r["sigma"](d) * d
    for (a, d, _, _), wp in zip(sec.tendons, r["w_tendons"]):
        F += sec.tendon.Ep / c.Ec * wp * a * r["sigma"](d)
        Mz += sec.tendon.Ep / c.Ec * wp * a * r["sigma"](d) * d
    return F, Mz / F


def test_class_b_box_multi_row_equilibrium():
    """a box girder with three rows of web tendons: the cracked stress field carries Np0 along its shifted line"""
    box = jtg.box(5000, 2500, 300, 250, 1000, b_top=12000, concrete="C50", bottom=[(40, 16, 60)], top=[(80, 16, 50)],
                  tendons=[(12000.0, 2350, 1150, 120), (6000.0, 1900, 1150, 120), (6000.0, 1400, 1150, 120)],
                  sigma_pe=1080.0)
    for Mk in (30000.0, 50000.0, 70000.0):
        res = jtg.service_stresses(box, Mk, pc_class="B")
        Np0, hps = box.prestress_resultant("p0")
        zR = hps - Mk * 1e6 / Np0
        r = _cracked(box, N=Np0, z_R=zR)
        assert all(r["w_tendons"] == 1)  # the web tendons are below the centroid: no sign change (note 2 is for A'p)
        F, zF = _equilibrium(box, r, Np0, zR)
        assert F == pytest.approx(Np0, rel=1e-9) and zF == pytest.approx(zR, rel=1e-9)
        assert res.demand == pytest.approx(max(r["sigma"](0.0), r["sigma"](box.h) if r["x"] == math.inf else -1e9))


def test_class_b_cracked_section_properties_by_hand():
    """Acr, its centroid and Icr of a T section from x (7.1.4-1), computed here from the shapes"""
    sec = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)],
                  tendons=[(2780, 1470, 1150, 90)], sigma_pe=1080.0)
    c = sec.concrete
    aEP, aES = 1.95e5 / c.Ec, 2e5 / c.Ec
    r = jtg.service_stresses(sec, 7000.0, pc_class="B")
    x = r.details["x"]
    assert 180 < x < 1600
    parts = [(1800 * 180, 90.0, 1800 * 180 ** 3 / 12), (200 * (x - 180), (180 + x) / 2, 200 * (x - 180) ** 3 / 12),
             (aEP * 2780, 1470.0, 0.0), (aES * 6 * math.pi * 64, 1550.0, 0.0)]
    A = sum(p[0] for p in parts)
    zc = sum(p[0] * p[1] for p in parts) / A
    I = sum(p[2] + p[0] * (p[1] - zc) ** 2 for p in parts)
    assert r.details["Acr"] == pytest.approx(A) and r.details["c"] == pytest.approx(zc) and r.details["Icr"] == pytest.approx(I)


def test_crack_resistance_other_edge():
    """the prestress puts the top in tension: 6.3.1 is checked there too"""
    pre = jtg.rect(1000, 600, concrete="C50", tendons=[(16 * 139.0, 520, 1250)], pretensioned=True, sigma_pe=1150.0)
    aEP = 1.95e5 / 3.45e4
    Ap = 16 * 139.0
    A0, z0, I0 = _rect_props(1000, 600, [((aEP - 1) * Ap, 520)])  # pretensioned: the strand occupies its own area
    Np0 = 1250 * Ap
    top_pc = Np0 / A0 + Np0 * (520 - z0) * (0 - z0) / I0
    Ms, Ml = 60.0, 48.0
    sst_top, slt_top = -Ms * 1e6 * z0 / I0, -Ml * 1e6 * z0 / I0
    assert top_pc < 0 and sst_top - top_pc > 0.7 * 2.65
    r = jtg.crack_resistance(pre, Ms, Ml, pc_class="A")
    assert r.details["edge"] == "top" and not r.ok
    assert r.details["top"]["sigma_pc"] == pytest.approx(top_pc)
    assert r.demand == pytest.approx(sst_top) and r.resistance == pytest.approx(top_pc + 0.7 * 2.65)
    assert any("top" in f and "6.3.1-4" in f for f in r.failed)
    # full prestress: a tension σpc is not reduced by 0.85
    f = jtg.crack_resistance(pre, Ms, Ml, pc_class="full")
    assert f.details["edge"] == "top" and f.resistance == pytest.approx(top_pc)
    assert "margin" in str(f)


def test_sigma_l6_independent_of_orientation():
    sec = jtg.rect(400, 1000, concrete="C50", bottom=[(4, 20, 40)], top=[(4, 16, 40)],
                   tendons=[(1112.0, 850, 1180, 80)], sigma_pe=1080.0, sigma_l6=(60.0, 40.0))
    N, z = sec.prestress_resultant("p0")
    Nf, zf = sec.flipped().prestress_resultant("p0")
    assert N == pytest.approx(1180 * 1112 - 60 * 4 * math.pi * 100)  # top bars: no tendon above, σ'l6 = 0
    assert Nf == pytest.approx(N) and zf == pytest.approx(1000 - z)


def test_crack_width_thin_tension_flange():
    """Ate = 2 as b with b the bottom-flange width, also when the flange is thinner than 2 as (6.4.5-1)"""
    s = jtg.tee(200, 1200, 1600, 160, concrete="C40", bf_bottom=500, hf_bottom=150, bottom=[(6, 25, 50), (6, 25, 130)])
    r = jtg.crack_width(s, 1500.0, 1100.0)
    assert r.details["Ate"] == pytest.approx(2 * 90 * 500)


def test_crack_width_C2_and_circle_bending():
    sec = jtg.rect(400, 600, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    r = jtg.crack_width(sec, 300.0, 200.0, N=500.0, Nl=450.0)
    assert r.details["C2"] == pytest.approx(1 + 0.5 * 0.9)  # the larger of 200/300 and 450/500
    assert jtg.crack_width(sec, 150.0, -50.0).details["C2"] == 1.0  # never below 1
    col = jtg.circle(D=1500, n=28, d=28, cover=60, concrete="C35")
    with pytest.raises(NotImplementedError):
        jtg.crack_width(col, 1500.0, 1200.0)


def test_principal_stresses_axial_box_pretensioned():
    sec = jtg.rect(300, 900, concrete="C50")
    p0 = jtg.principal_stresses(sec, 0.0, 400.0)[0]
    p1 = jtg.principal_stresses(sec, 0.0, 400.0, N=-1000.0, sigma_cx=0.2)[0]
    assert p1["sigma_cx"] == pytest.approx(p0["sigma_cx"] - 1000e3 / (300 * 900) + 0.2)
    # box: points at the centroid and the four web–slab junctions on the web side
    box = jtg.box(5000, 2500, 300, 250, 1000, b_top=12000, concrete="C50")
    pts = jtg.principal_stresses(box, 0.0, 3000.0)
    assert sorted(round(q["z"]) for q in pts)[0] == 300 and all(q["b"] == pytest.approx(1000) for q in pts)
    # pretensioned bent tendon: the transformed section for the tendon term (6.1.5)
    pre = jtg.rect(400, 900, concrete="C50", tendons=[(556.0, 700, 1200)], pretensioned=True, sigma_pe=1100.0)
    A0, z0, I0 = pre.props("transformed")
    p = jtg.principal_stresses(pre, 0.0, 500.0, depths=[z0], bent_tendons=[(556.0, 5.0)])[0]
    tau = (500e3 - 1100 * 556 * math.sin(math.radians(5))) * pre.first_moment(z0) / (400 * I0)
    assert p["tau"] == pytest.approx(tau)


def test_combination_variants():
    G = (np.array([10.0, -5.0]), (1.2, 1.0))  # one (array, factors) pair
    assert jtg.frequent_combination(G)[0] == pytest.approx([10.0, -5.0])
    hi, lo = jtg.basic_combination(G, (np.zeros(2), np.zeros(2)))
    assert hi == pytest.approx([12.0, -5.0]) and lo == pytest.approx([10.0, -6.0])
    crowd = (np.array([8.0, 0.0]), np.array([0.0, -3.0]))
    hi, lo = jtg.frequent_combination(np.zeros(2), crowd, leading="crowd")
    assert hi == pytest.approx([8.0, 0.0]) and lo == pytest.approx([0.0, -3.0])
