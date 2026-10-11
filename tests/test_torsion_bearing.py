"""Torsion (JTG 3362-2018 5.5, 9.3.13) and local compression (5.7), against hand calculations with the code's formulas
and, for the plastic torsion moduli, against an independent sand-heap integration."""
import math

import numpy as np
import pytest

from femotion.codes import jtg

A12, A25 = math.pi * 12 ** 2 / 4, math.pi * 25 ** 2 / 4


def _sand_heap(b, h, t=None, n=1600):
    """plastic torsion modulus as twice the volume of the sand heap (roof of slope 1) over the section; a box with
    uniform walls t: the roof is cut flat at height t over the hole (independent of the code's formulas)"""
    x = (np.arange(n) + 0.5) / n * b
    y = (np.arange(n) + 0.5) / n * h
    X, Y = np.meshgrid(x, y)
    d = np.minimum(np.minimum(X, b - X), np.minimum(Y, h - Y))
    if t is not None:
        d = np.minimum(d, t)
    return 2 * d.sum() * (b / n) * (h / n)


def test_torsion_moduli_independent():
    for b, h in [(400, 800), (300, 300), (800, 400)]:
        w = jtg.torsion_moduli(jtg.rect(b, h, concrete="C40", bottom=[(2, 20, 50)]))
        assert w["body"] == pytest.approx(_sand_heap(b, h), rel=1e-5)
    # single-cell box with uniform walls: 5.5.2-2 equals the truncated sand heap
    bx = jtg.box(2400, 1600, 300, 300, 600, concrete="C50", bottom=[(10, 25, 60)])
    w = jtg.torsion_moduli(bx)
    assert w["dims"] == (1600, 2400, 300, 300)
    assert w["body"] == pytest.approx(_sand_heap(2400, 1600, 300), rel=1e-5)
    b, h, t1, t2 = 1600, 2400, 300, 300
    assert w["body"] == pytest.approx(b * b / 6 * (3 * h - b) - (b - 2 * t1) ** 2 / 6 * (3 * (h - 2 * t2) - (b - 2 * t1)))
    assert w["beta_a"] == pytest.approx(min(4 * t2 / b, 4 * t1 / h))  # 0.5, 0.75 → 0.5


def test_pure_torsion_rect_hand():
    sec = jtg.rect(400, 800, concrete="C40", bottom=[(4, 25, 50)], top=[(2, 16, 45)])
    r = jtg.torsion(sec, Td=95.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))
    Wt = 400 ** 2 / 6 * (3 * 800 - 400)
    Acor, Ucor = 320 * 720, 2 * (320 + 720)
    zeta = 330 * 6 * 491 * 100 / (330 * A12 * Ucor)
    Tu = 0.35 * 1.65 * Wt + 1.2 * math.sqrt(zeta) * 330 * A12 * Acor / 100
    assert r.details["zeta"] == pytest.approx(zeta)
    assert r.resistance == pytest.approx(Tu / 1e6) and r.demand == 95.0 and r.ok
    assert r.clause.endswith("5.5.1")
    # ζ limits: > 1.7 taken as 1.7, < 0.6 fails
    big = jtg.torsion(sec, Td=95.0, stirrups=(2, 12, 100), Ast=20 * 491, core=(320, 720))
    assert big.details["zeta_used"] == 1.7
    small = jtg.torsion(sec, Td=95.0, stirrups=(2, 12, 100), Ast=2 * 491, core=(320, 720))
    assert any("ζ" in f for f in small.failed)


def test_shear_torsion_rect_hand():
    sec = jtg.rect(400, 800, concrete="C40", bottom=[(4, 25, 50)], top=[(2, 16, 45)])
    Vd, Td = 300.0, 40.0
    r = jtg.torsion(sec, Td=Td, Vd=Vd, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720), gamma0=1.1)
    Wt, b, h0 = 400 ** 2 / 6 * 2000, 400.0, 750.0
    bt = 1.5 / (1 + 0.5 * Vd * 1e3 * Wt / (Td * 1e6 * b * h0))
    assert bt == pytest.approx(0.9) and r.details["beta_t"] == pytest.approx(bt)
    zeta = 330 * 6 * 491 * 100 / (330 * A12 * 2080)
    P = 100 * 4 * A25 / (b * h0)
    rho_sv = 2 * A12 / (b * 100)
    coef = 0.5e-4 * (10 - 2 * bt) * b * h0
    K = (2 + 0.6 * P) * math.sqrt(40) * 330
    Vu = coef * math.sqrt(K * rho_sv)
    assert r.details["Vu"] == pytest.approx(Vu)
    # 5.5.6 item 2: the stirrups cover shear and torsion together: the shear share (5.5.4-1 solved for ρsv, over both
    # legs) comes off the leg, the rest enters 5.5.4-2
    q_shear = (1.1 * Vd / coef) ** 2 / K * b / 2
    Tu = bt * 0.35 * 1.65 * Wt + 1.2 * math.sqrt(zeta) * 330 * (A12 / 100 - q_shear) * 320 * 720
    assert r.details["q_shear"] == pytest.approx(q_shear)
    assert r.resistance == pytest.approx(Tu / 1e6) and r.demand == pytest.approx(1.1 * Td)
    # 5.5.3: section limit and detailing threshold
    tau = 1.1 * Vd * 1e3 / (b * h0) + 1.1 * Td * 1e6 / Wt
    assert r.details["tau_section"] == pytest.approx(tau)
    over = jtg.torsion(sec, Td=400.0, Vd=1500.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))
    assert any("5.5.3-1" in f for f in over.failed)
    tiny = jtg.torsion(sec, Td=5.0, Vd=30.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))
    assert any("5.5.3-2" in n for n in tiny.notes)
    # βt within 0.5 … 1.0; Td = 0 with shear: 0.5 (the limit of 5.5.4-3), Vu then equals 5.2.9
    assert jtg.torsion(sec, Td=10.0, Vd=900.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720)).details["beta_t"] == 0.5
    t0 = jtg.torsion(sec, Td=0.0, Vd=300.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))
    assert t0.details["beta_t"] == 0.5
    assert t0.details["Vu"] == pytest.approx(jtg.shear(sec, 300.0, (2, 12, 100), alpha3=1.0).details["Vcs"])
    # 9.3.13: minimum stirrups (2βt − 1)(0.055 fcd/fsv − c) + c and longitudinal 0.08 (2βt − 1) fcd/fsd
    assert r.details["rho_sv_min"] == pytest.approx((2 * bt - 1) * (0.055 * 18.4 / 330 - 0.0011) + 0.0011)
    assert r.details["rho_st_min"] == pytest.approx(0.08 * (2 * bt - 1) * 18.4 / 330)
    assert r.details["rho_st"] == pytest.approx(6 * 491 / (400 * 800))


def test_shear_and_torsion_share_the_stirrups():
    # each formula on its own passes with the same stirrups, together they need 1.59 mm²/mm per leg against 1.13
    sec = jtg.rect(500, 1000, concrete="C50", bottom=[(5, 25, 50)])
    r = jtg.torsion(sec, Td=175.0, Vd=900.0, stirrups=(2, 12, 100), Ast=10 * 491, core=(420, 920))
    assert r.details["shear_ratio"] < 1.0 and not r.ok
    assert r.details["q_shear"] + 0.0 > 0.9  # most of the leg goes to shear


def test_torsion_tee_split():
    t = jtg.tee(250, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 90)])
    w = jtg.torsion_moduli(t)
    Wtw = 250 ** 2 / 6 * (3 * 1300 - 250)
    Wtf = 150 ** 2 / 2 * (6 * 150)  # overhang 1350 capped at 6 h'f (b'f ≤ b + 6 h'f)
    assert w["body"] == pytest.approx(Wtw) and w["top"] == pytest.approx(Wtf) and w["capped"]
    r = jtg.torsion(t, Td=60, Vd=500, stirrups=(2, 12, 100), Ast=8 * 314, core=(190, 1240))
    assert r.details["T_share"] == pytest.approx(60 * Wtw / (Wtw + Wtf))
    f = jtg.torsion_flange(t, Td=60, which="top", stirrups=(2, 10, 150), Ast=8 * 113, core=(800, 90))
    assert f.demand == pytest.approx(60 * Wtf / (Wtw + Wtf))
    zeta = 330 * 8 * 113 * 150 / (330 * math.pi * 100 / 4 * 2 * (800 + 90))
    Tu = 0.35 * 1.65 * Wtf + 1.2 * math.sqrt(min(zeta, 1.7)) * 330 * math.pi * 100 / 4 * 800 * 90 / 150
    assert f.resistance == pytest.approx(Tu / 1e6)
    assert f.details["rho_sv_min"] == pytest.approx(0.055 * 18.4 / 330)  # pure torsion (9.3.13 item 3)
    with pytest.raises(ValueError):
        jtg.torsion_flange(t, Td=60, which="bottom", stirrups=(2, 10, 150), Ast=800, core=(800, 90))
    # web b/hw ≥ 0.15 (5.5.5 note)
    thin = jtg.tee(150, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 90)])
    assert any("0.15" in s for s in jtg.torsion(thin, Td=20, stirrups=(2, 12, 100), Ast=2000, core=(90, 1240)).failed)


def test_torsion_box():
    bx = jtg.box(2400, 1600, 300, 300, 600, concrete="C50", bottom=[(10, 25, 60)], top=[(10, 16, 50)])
    r = jtg.torsion(bx, Td=900, Vd=2000, stirrups=(4, 14, 100), Ast=40 * 491, core=(2300, 1500))
    w = jtg.torsion_moduli(bx)
    assert r.details["beta_a"] == pytest.approx(0.5)
    b, h0 = 600.0, bx.h0()
    bt = 1.5 / (1 + 0.5 * 2000e3 * 0.5 * w["body"] / (900e6 * b * h0))  # βa Wt in βt for boxes
    assert r.details["beta_t"] == pytest.approx(min(max(bt, 0.5), 1.0))
    # walls thinner than 0.1 of the sides: outside 5.5.1
    thin = jtg.box(6000, 2000, 250, 250, 1000, concrete="C50", bottom=[(20, 25, 60)])
    assert any("outside 5.5" in s for s in jtg.torsion(thin, Td=500, stirrups=(4, 14, 100), Ast=8000,
                                                         core=(5900, 1900)).failed)


def test_torsion_prestress_term():
    # centric tendons (ep0 ≤ h/6), ζ ≥ 1.7: + 0.05 Np0/A0 Wt, ζ = 1.7; Np0 at most 0.3 fcd A0
    sec = jtg.rect(500, 1000, concrete="C50", bottom=[(4, 20, 50)], top=[(4, 20, 50)],
                   tendons=[(1400, 500, 1100, 80)], sigma_pe=1050)
    rc_ = jtg.torsion(sec, Td=200, stirrups=(2, 12, 100), Ast=20 * 491, core=(420, 920))
    A0, z0, _ = sec.props("transformed")
    N, zN = sec.prestress_resultant("p0")
    Np0 = min(N, 0.3 * 22.4 * A0)
    Wt = 500 ** 2 / 6 * (3000 - 500)
    Tu = (0.35 * 1.83 + 0.05 * Np0 / A0) * Wt + 1.2 * math.sqrt(1.7) * 330 * A12 * 420 * 920 / 100
    assert rc_.details["prestress_counted"] and rc_.resistance == pytest.approx(Tu / 1e6)
    # eccentric tendons: not counted
    ecc = sec.replace(tendons=np.array([[1400, 900, 1100, math.pi * 40 ** 2]]))
    assert not jtg.torsion(ecc, Td=200, stirrups=(2, 12, 100), Ast=20 * 491, core=(420, 920)).details["prestress_counted"]


def test_bearing_base_area():
    assert jtg.bearing_base_area(320, 320, cx=400, cy=400) == pytest.approx(960 * 960)
    assert jtg.bearing_base_area(320, 320, cx=100, cy=400) == pytest.approx(520 * 960)
    assert jtg.bearing_base_area(300, 200) == pytest.approx(700 * 600)  # extended by the shorter side
    assert jtg.bearing_base_area(300, 200, cx=0, cy=0) == pytest.approx(300 * 200)  # corner: Ab = Al
    assert jtg.bearing_base_area(200, 0, circle=True) == pytest.approx(math.pi * 600 ** 2 / 4)


def test_local_bearing_hand():
    Al = 320.0 * 320
    Aln = Al - math.pi * 45 ** 2
    Ab = 960.0 * 960
    r = jtg.local_bearing("C50", 2000, Al=Al, Aln=Aln, Ab=Ab, mesh=(5, 113, 280, 5, 113, 280, 50), fcu=45, layers=5)
    fcd = 20.5  # table 3.1.4 at f'cu = 45
    beta = math.sqrt(Ab / Al)
    rho_v = (5 * 113 * 280 * 2) / (280 * 280 * 50)
    bcor = math.sqrt(280 * 280 / Al)
    R = 0.9 * (beta * fcd + 2.0 * rho_v * bcor * 330) * Aln / 1e3
    size = 1.3 * beta * fcd * Aln / 1e3
    assert r.details["F_resistance"] == pytest.approx(R) and r.details["F_section_max"] == pytest.approx(size)
    assert r.resistance == pytest.approx(min(R, size))  # the smaller of 5.7.2-1 and 5.7.1-1 governs
    assert jtg.fcd_at(47.5) == pytest.approx((20.5 + 22.4) / 2)
    # C60: ηs = 1 − 0.24/3, k = 2 − 0.3/3
    r60 = jtg.local_bearing("C60", 2000, Al=Al, Ab=Ab, spiral=(113, 280, 50))
    es, k = 1 - 0.24 / 3, 2 - 0.1
    rho_v = 4 * 113 / (280 * 50)
    R = 0.9 * (es * 3 * 26.5 + k * rho_v * math.sqrt(math.pi * 280 ** 2 / 4 / Al) * 330) * Al / 1e3
    assert r60.details["F_resistance"] == pytest.approx(R)
    # Acor > Ab: Acor = Ab; fewer than 4 layers; unequal mesh
    big = jtg.local_bearing("C50", 2000, Al=Al, Ab=Al * 1.2, mesh=(5, 113, 600, 5, 113, 600, 50), layers=3)
    assert big.details["Acor"] == pytest.approx(Al * 1.2) and any("4" in f for f in big.failed)
    bad = jtg.local_bearing("C50", 2000, Al=Al, Ab=Ab, mesh=(5, 201, 280, 5, 113, 280, 50))
    assert any("50 %" in f for f in bad.failed)
    # no indirect reinforcement: outside 5.7, never "OK"
    only = jtg.local_bearing("C50", 200, Al=Al, Ab=Ab)
    assert not only.ok and only.resistance == pytest.approx(0.9 * 3 * 22.4 * Al / 1e3)
