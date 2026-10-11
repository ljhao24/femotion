"""Serviceability and construction-stage details of JTG 3362-2018: the transfer zone of pretensioned members (6.1.8),
construction-stage loads, the 80 % rules and the shear stress diagram (7.2.1–7.2.8), tendons jacked at both ends
(appendix G.0.3), negative-moment regions and class B composite members (8.1.8, 8.1.9). Each feature is checked against
a source independent of the program's own formula where one exists: the commentary's formula behind a table, closed-form
geometry, equilibrium of the computed stresses, a monolithic section built another way, a numerical solution in place
of a closed form, and limiting cases that must reproduce the existing checks exactly."""
import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from femotion.codes import jtg
from femotion.codes.jtg import prestress as ps


# ── 6.1.8 transfer length of pretensioned tendons ──
def _pre(bars=(), sigma_l6=0.0):
    """a pretensioned 400 × 1000 C50 beam with two strand layers (σp0 1250 / 1200 MPa, σpe 1150 / 1100 MPa)"""
    return jtg.rect(400, 1000, concrete="C50", bottom=list(bars), tendons=[(834.0, 900, 1250, 0), (556.0, 820, 1200, 0)],
                    pretensioned=True, sigma_pe=[1150.0, 1100.0], sigma_l6=sigma_l6)


def test_transfer_table_reproduces_commentary_formula():
    """table 6.1.8 against the commentary's formula 6-1 (p194–195) it was computed from: ltr = β σpe / ftk · d,
    β = 0.16 for 1×7 strand and 0.14 for helical-rib wire, ftk of table 3.1.3; the table rounds to whole d"""
    for kind, fpk, beta in (("strand", 1860, 0.16), ("wire", 1570, 0.14)):
        t = jtg.tendon(kind, fpk)
        for g in (40, 45, 50, 55):
            lt = ps.transfer_length(t, g, 15.2, 1000.0)
            assert lt == pytest.approx(beta * 1000 / jtg.concrete(g).ftk * 15.2, abs=0.5 * 15.2)


def test_transfer_factors():
    """figure 6.1.8: 0 at the start of transfer, σpe at its end, linear; sudden release from 0.25 ltr (note 3); a
    debonded length moves the start (commentary to 6.1.8 with figure 5-1)"""
    tr = jtg.Transfer(x=300.0, ltr=600.0)
    assert tr.factors(2) == pytest.approx([0.5, 0.5])
    assert jtg.Transfer(x=0.0, ltr=600.0).factors(1)[0] == 0.0
    assert jtg.Transfer(x=900.0, ltr=600.0).factors(1)[0] == 1.0
    ab = jtg.Transfer(x=450.0, ltr=600.0, abrupt=True)
    assert ab.factors(1)[0] == pytest.approx((450 - 150) / 600)
    assert jtg.Transfer(x=150.0, ltr=600.0, abrupt=True).factors(1)[0] == 0.0
    db = jtg.Transfer(x=1300.0, ltr=(600.0, 500.0), debond=(0.0, 1000.0))
    assert db.factors(2) == pytest.approx([1.0, 300 / 500])


def test_transfer_reduces_precompression_linearly():
    """without bars (σl6 term absent) σpc at both edges is k × the full value (Np0 = k Σ σp0 Ap on the transformed
    section, 6.1.6-1), zero at the member end; the loads' stresses are untouched"""
    sec = _pre()
    full = jtg.crack_resistance(sec, 100.0, 80.0, pc_class="A")
    lt = 700.0
    for x in (0.0, 175.0, 350.0, 700.0):
        r = jtg.crack_resistance(sec, 100.0, 80.0, pc_class="A", transfer=jtg.Transfer(x, lt))
        k = min(x / lt, 1.0)
        for edge in ("top", "bottom"):
            assert r.details[edge]["sigma_pc"] == pytest.approx(k * full.details[edge]["sigma_pc"], abs=1e-12)
            assert r.details[edge]["sigma_st"] == full.details[edge]["sigma_st"]
        assert "6.1.8" in r.clause and r.details["transfer_factor"] == pytest.approx([k, k])
    # at the end only the loads act: class A needs σst ≤ 0.7 ftk
    end = jtg.crack_resistance(sec, 100.0, 80.0, pc_class="A", transfer=jtg.Transfer(0.0, lt))
    assert end.details["bottom"]["sigma_pc"] == 0.0


def test_transfer_beyond_ltr_is_the_plain_check():
    """beyond the transfer length the checks are the existing ones, bit for bit (normal and inclined sections)"""
    sec = _pre(bars=[(4, 16, 40)], sigma_l6=80.0)
    tr = jtg.Transfer(x=2000.0, ltr=(700.0, 650.0), abrupt=True)
    a = jtg.crack_resistance(sec, 300.0, 200.0, pc_class="full", transfer=tr)
    b = jtg.crack_resistance(sec, 300.0, 200.0, pc_class="full")
    assert (a.demand, a.resistance, a.details["top"], a.details["bottom"]) == \
        (b.demand, b.resistance, b.details["top"], b.details["bottom"])
    pa = jtg.principal_stresses(sec, 200.0, 300.0, transfer=tr)
    pb = jtg.principal_stresses(sec, 200.0, 300.0)
    assert pa == pb
    ia = jtg.inclined_crack_resistance(sec, 200.0, 300.0, pc_class="A", transfer=tr)
    ib = jtg.inclined_crack_resistance(sec, 200.0, 300.0, pc_class="A")
    assert ia.demand == ib.demand and "6.1.8" in ia.clause and "6.1.8" not in ib.clause


def test_transfer_inclined_section_grows_toward_the_end():
    """less precompression near the end: σtp at the centroid grows monotonically toward the member end (τ unchanged)"""
    sec = _pre()
    z0 = sec.props("transformed")[1]
    s = [jtg.principal_stresses(sec, 0.0, 400.0, depths=[z0], transfer=jtg.Transfer(x, 700.0))[0] for x in
         (700.0, 525.0, 350.0, 175.0, 0.0)]
    assert all(p["tau"] == s[0]["tau"] for p in s)
    tp = [p["sigma_tp"] for p in s]
    assert all(b > a for a, b in zip(tp, tp[1:]))
    assert tp[-1] == pytest.approx(s[-1]["tau"])  # no normal stress at the end: σtp = τ


def test_transfer_needs_pretensioning():
    post = jtg.rect(400, 1000, concrete="C50", tendons=[(1390.0, 850, 1300, 90)], sigma_pe=1200.0)
    with pytest.raises(ValueError):
        jtg.crack_resistance(post, 100.0, 80.0, pc_class="A", transfer=jtg.Transfer(100.0, 700.0))


# ── 7.2 construction stages ──
def _pc_beam():
    return jtg.rect(400, 1000, concrete="C50", top=[(4, 12, 40)], bottom=[(4, 16, 40)],
                    tendons=[(1390.0, 850, 1300, 90)], sigma_pe=1250.0)


def test_transient_unit_factors_reproduce_the_plain_check():
    """dynamic factor 1.0 and crane factor 1.0: the same numbers as the construction loads given directly"""
    sec = _pc_beam()
    a = jtg.transient_pc(sec, 120.0, fcu=45)
    b = jtg.transient_pc(sec, 20.0, fcu=45, M_G=60.0, M_crane=40.0, dynamic=1.0, crane_factor=1.0)
    assert (a.demand, a.details["top"], a.details["bottom"]) == (b.demand, b.details["top"], b.details["bottom"])
    rc = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    a = jtg.transient_rc(rc, 120.0, 150.0, fcu=25)
    b = jtg.transient_rc(rc, 20.0, 50.0, fcu=25, M_G=60.0, V_G=60.0, M_crane=40.0, V_crane=40.0, dynamic=1.0,
                         crane_factor=1.0)
    assert (a.demand, a.details["sigma_tp"]) == (b.demand, b.details["sigma_tp"])
    assert np.array_equal(a.details["sigma_s"], b.details["sigma_s"])


def test_transient_dynamic_factor_takes_the_worse():
    """7.2.2 with D60 4.1.10: lifting tries 1.2 and 0.85 on the member's weight. On a prestressed beam the sagging weight
    relieves the hogging prestress, so 0.85 governs both the precompressed edge and the pretensioned edge; the crane of
    7.2.1 is × 1.15"""
    sec = _pc_beam()
    r = jtg.transient_pc(sec, fcu=45, M_G=150.0, lifting=True)
    lo, hi = jtg.transient_pc(sec, 0.85 * 150.0, fcu=45), jtg.transient_pc(sec, 1.2 * 150.0, fcu=45)
    assert r.details["dynamic_factor"] == 0.85 and r.demand == lo.demand > hi.demand
    assert r.details["sigma_tct"] == lo.details["sigma_tct"] > hi.details["sigma_tct"]
    assert "7.2.2" in r.clause and len(r.details["cases"]) == 2
    # a hogging weight (cantilevered lifting points) adds to the prestress: 1.2 governs
    h = jtg.transient_pc(sec, fcu=45, M_G=-150.0, lifting=True)
    assert h.details["dynamic_factor"] == 1.2 and h.demand == jtg.transient_pc(sec, -180.0, fcu=45).demand
    # override and crane factor
    assert jtg.transient_pc(sec, fcu=45, M_G=150.0, lifting=True, dynamic=1.3).demand == \
        jtg.transient_pc(sec, 195.0, fcu=45).demand
    c = jtg.transient_pc(sec, 100.0, fcu=45, M_crane=200.0)
    assert c.demand == jtg.transient_pc(sec, 100.0 + 1.15 * 200.0, fcu=45).demand and "7.2.1" in c.clause
    # reinforced concrete: the larger factor when the weight is the only load
    rc = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    g = jtg.transient_rc(rc, fcu=25, M_G=100.0, V_G=120.0, lifting=True)
    assert g.demand == jtg.transient_rc(rc, 120.0, 144.0, fcu=25).demand
    assert g.details["sigma_tp"] == pytest.approx(jtg.transient_rc(rc, 120.0, 144.0, fcu=25).details["sigma_tp"])


def test_transient_rc_stresses_satisfy_equilibrium():
    """7.2.4 on a singly reinforced rectangle: the triangular compression block and the bar force balance (C = T) and
    their couple is the moment (T z0 = M) — statics, independent of the transformed-section formulas"""
    rc = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)])
    M = 150.0
    r = jtg.transient_rc(rc, M, 100.0, fcu=30)
    x0, scc = r.details["x0"], r.demand
    C = 0.5 * scc * x0 * 300
    T = float(r.details["sigma_s"].sum() * 4 * math.pi * 25 ** 2 / 4)
    assert C == pytest.approx(T, rel=1e-9)
    assert T * r.details["z0"] == pytest.approx(M * 1e6, rel=1e-9)


def test_transient_modulus_and_strength_at_prestressing():
    """7.2.3: f'cu ≥ 0.8 of the class and E'c ≥ 0.8 Ec(28 d) when prestressing, as warnings; the stage modulus enters
    αEP, αES as a section with that modulus would"""
    sec = _pc_beam()
    ok = jtg.transient_pc(sec, 100.0, fcu=40)
    assert not any("7.2.3" in w for w in ok.warnings)
    low = jtg.transient_pc(sec, 100.0, fcu=35, Ec=0.75 * sec.concrete.Ec)
    assert sum("7.2.3" in w for w in low.warnings) == 2
    alt = jtg.transient_pc(sec.replace(concrete=dataclasses.replace(sec.concrete, Ec=0.75 * sec.concrete.Ec)), 100.0,
                           fcu=35)
    assert (low.details["top"], low.details["bottom"]) == (alt.details["top"], alt.details["bottom"])
    assert low.details["top"] != jtg.transient_pc(sec, 100.0, fcu=35).details["top"]


def test_transient_pretensioned_zone_detailing_warns():
    """7.2.8: the minimum reinforcement of the pretensioned zone, ribbed bars of at most 14 mm: warnings, not failures"""
    thin = jtg.rect(400, 1000, concrete="C50", top=[(2, 16, 40)], tendons=[(1390.0, 850, 1300, 90)], sigma_pe=1250.0)
    r = jtg.transient_pc(thin, 250.0, fcu=45)
    assert r.details["sigma_tct"] > 0 and r.details["rho_pretension_zone"] < r.details["rho_needed"]
    assert r.ok and not r.failed
    assert any("pretensioned zone (7.2.8)" in w for w in r.warnings) and any("φ16" in w for w in r.warnings)
    fine = jtg.rect(400, 1000, concrete="C50", top=[(14, 12, 40)], tendons=[(1390.0, 850, 1300, 90)], sigma_pe=1250.0)
    assert not jtg.transient_pc(fine, 250.0, fcu=45).warnings
    plain = jtg.rect(400, 1000, concrete="C50", steel="HPB300", top=[(14, 12, 40)], tendons=[(1390.0, 850, 1300, 90)],
                     sigma_pe=1250.0)
    assert any("ribbed" in w for w in jtg.transient_pc(plain, 250.0, fcu=45).warnings)
    unknown = jtg.rect(400, 1000, concrete="C50", bars=[(1600.0, 40)], tendons=[(1390.0, 850, 1300, 90)],
                       sigma_pe=1250.0)
    assert any("14 mm" in n for n in jtg.transient_pc(unknown, 250.0, fcu=45).notes)
    # the 1.15 f'tk limit stays a failure
    big = jtg.transient_pc(thin, -250.0, fcu=45)
    assert big.details["sigma_tct"] > 1.15 * big.details["ftk_t"] and not big.ok


# ── 7.2.6 shear stress diagram ──
def test_transient_shear_triangle_closed_forms():
    """figure 7.2.6 for a triangular diagram (τmax at the support falling linearly to 0 over L): a = L (1 − c0/τmax),
    c0 = 0.25 f'tk; Ω = (τmax − τv)² L / (2 τmax) when τv ≥ c0, else the trapezoid a ((τmax + c0)/2 − τv) — geometry,
    not the program's piecewise integration"""
    fcu, b, L = 30.0, 200.0, 6000.0
    ftk = jtg.strength_at(fcu)[1]
    c0, tmax = 0.25 * ftk, 1.6
    x = np.linspace(0, L, 7)
    tau = tmax * (1 - x / L)
    a_exact = L * (1 - c0 / tmax)
    fsk = jtg.rebar("HRB400").fsk
    tvs = []
    for st in [(2, 8, 200), (2, 6, 250)]:
        tv = st[0] * math.pi * st[1] ** 2 / 4 * 0.75 * fsk / (b * st[2])
        tvs.append(tv)
        r = jtg.transient_shear(x, tau, fcu=fcu, b=b, stirrups=st)
        Om = (tmax - tv) ** 2 * L / (2 * tmax) if tv >= c0 else a_exact * ((tmax + c0) / 2 - tv)
        assert r.details["tau_v"] == pytest.approx(tv)
        assert r.details["a"] == pytest.approx(a_exact)
        assert r.details["Omega"] == pytest.approx(Om)
        assert r.details["Asb_needed"] == pytest.approx(b * Om / (math.sqrt(2) * 0.75 * fsk))
    assert tmax > tvs[0] > c0 > tvs[1]  # both branches
    assert jtg.transient_shear(x, tau, fcu=fcu, b=b, stirrups=(2, 10, 100)).details["Omega"] == 0.0  # τv > τmax


def test_transient_shear_against_dense_quadrature():
    """an irregular diagram: Ω against dense sampling (10⁶ points, midpoint rule) of 1[τ > c0] · max(τ − τv, 0); extra
    stations on the same lines change nothing (the integral is exact for the piecewise-linear diagram)"""
    x = np.array([0.0, 800.0, 1500.0, 2600.0, 4000.0, 5200.0])
    tau = np.array([1.9, 1.5, 1.6, 0.9, 0.35, 0.0])
    fcu, b = 30.0, 250.0
    r = jtg.transient_shear(x, tau, fcu=fcu, b=b, stirrups=(2, 8, 150))
    c0, tv = 0.25 * jtg.strength_at(fcu)[1], r.details["tau_v"]
    xs = np.linspace(0, x[-1], 1_000_001)
    xm = (xs[1:] + xs[:-1]) / 2
    tm = np.interp(xm, x, tau)
    dx = xs[1] - xs[0]
    assert r.details["Omega"] == pytest.approx(float((np.where(tm > c0, np.maximum(tm - tv, 0), 0) * dx).sum()), rel=1e-6)
    assert r.details["a"] == pytest.approx(float((tm > c0).sum() * dx), rel=1e-5)
    xf = np.linspace(0, x[-1], 53)
    rf = jtg.transient_shear(xf, np.interp(xf, x, tau), fcu=fcu, b=b, stirrups=(2, 8, 150))
    assert rf.details["Omega"] == pytest.approx(r.details["Omega"], rel=1e-12)


def test_transient_shear_verdicts():
    x = np.array([0.0, 3000.0])
    small = jtg.transient_shear(x, [0.4, 0.0], fcu=30, b=200)
    assert small.details["a"] == 0.0 and small.ok and "detailing" in small.notes[0]
    r = jtg.transient_shear(x, [1.6, 0.0], fcu=30, b=200, stirrups=(2, 8, 200), Asb=200.0)
    assert r.details["Asb_needed"] > 200 and any("7.2.6-3" in f for f in r.failed)
    assert jtg.transient_shear(x, [1.6, 0.0], fcu=30, b=200, stirrups=(2, 8, 200), Asb=2000.0).ok
    over = jtg.transient_shear(x, [2.5, 0.0], fcu=30, b=200)  # f'tk(C30) = 2.01: 7.2.5
    assert not over.ok


# ── G.0.3 tendons jacked at both ends ──
SIG, EP = 1395.0, 1.95e5


def _single_numeric(dsd, l, dl, Ep, x):
    """σl1 + σl2 from one jacking end by figure G.0.2 solved numerically: before anchoring σcon − Δσd x; after it a line
    of slope +Δσd (reverse friction) that meets the first at lf ≤ l, or lies below it over the whole length; the area
    between the two lines (by quadrature) equals ΣΔl·Ep, the unknown found by root finding"""
    xs = np.linspace(0, l, 20001)

    def area(after):
        g = np.maximum((SIG - dsd * xs) - after(xs), 0.0)
        return float(((g[1:] + g[:-1]) / 2 * np.diff(xs)).sum())

    if area(lambda s: SIG - dsd * l - dsd * (l - s)) >= dl * Ep:  # lf ≤ l: the after-line meets at lf
        lf = brentq(lambda f: area(lambda s: SIG - dsd * f - dsd * (f - s)) - dl * Ep, 0.0, l)
        after = lambda s: SIG - dsd * lf - dsd * (lf - s)
    else:  # the after-line below the before-line along the whole length: σ(0) = σcon − Δσ'
        d = brentq(lambda d_: area(lambda s: SIG - d_ + dsd * s) - dl * Ep, 0.0, 10 * SIG)
        after = lambda s: SIG - d + dsd * s
    return SIG - np.minimum(SIG - dsd * x, after(x))


def test_two_ends_symmetric_and_without_overlap():
    """equal anchorage sets: the loss is symmetric about the middle; with lf < l/2 the reverse-friction lengths do not
    overlap and each half behaves as a tendon jacked at one end and fixed at the middle (G.0.2 on half the length,
    another formulation), beyond lf the friction loss of the nearer end"""
    l, s_end = 60000.0, 1255.0
    dsd = (SIG - s_end) / l
    x = np.linspace(0, l, 601)
    loss, (lfa, lfb) = ps.anchorage_loss_two_ends(SIG, s_end, l, 6.0, EP, x)
    assert lfa == lfb < l / 2
    assert loss == pytest.approx(loss[::-1], abs=1e-9)
    half, lf_h = ps.anchorage_loss_reverse(SIG, SIG - dsd * l / 2, l / 2, 6.0, EP, np.minimum(x, l - x))
    assert lf_h == pytest.approx(lfa)
    assert loss == pytest.approx(dsd * np.minimum(x, l - x) + half, abs=1e-9)


def test_two_ends_overlap_against_numerical_single_ends():
    """a straight 30 m tendon with little friction: the reverse-friction lengths overlap. Inside one length only the loss
    is that end's (G.0.2), in the overlap the smaller of the two (G.0.3); the single-end losses here are solved
    numerically (_single_numeric), not by the closed forms"""
    l = 30000.0
    s_end = SIG * math.exp(-0.0015 * 30)
    dsd = (SIG - s_end) / l
    x = np.linspace(0, l, 61)
    loss, (lfa, lfb) = ps.anchorage_loss_two_ends(SIG, s_end, l, 6.0, EP, x)
    assert l / 2 < lfa < l and lfa == lfb
    la, lb = _single_numeric(dsd, l, 6.0, EP, x), _single_numeric(dsd, l, 6.0, EP, l - x)
    in_a, in_b = x < lfa, l - x < lfb
    want = np.where(in_a & ~in_b, la, np.where(in_b & ~in_a, lb, np.minimum(la, lb)))
    assert loss == pytest.approx(want, rel=2e-4)
    # in the overlap the stress is no lower than either single-end stress would allow, i.e. the loss no larger
    ov = in_a & in_b
    assert ov.any() and (loss[ov] <= la[ov] * (1 + 1e-6)).all() and (loss[ov] <= lb[ov] * (1 + 1e-6)).all()


def test_two_ends_whole_length_and_unequal_sets():
    """lf > l from both ends (G.0.2-5 from each): the whole tendon is the overlap; unequal sets: the zones differ and a
    zero set at end B leaves the friction from B beyond end A's length"""
    l = 12000.0
    s_end = SIG * math.exp(-0.0015 * 12)
    x = np.linspace(0, l, 121)
    loss, (lfa, lfb) = ps.anchorage_loss_two_ends(SIG, s_end, l, 6.0, EP, x)
    assert lfa > l and lfb > l
    la, _ = ps.anchorage_loss_reverse(SIG, s_end, l, 6.0, EP, x)
    lb, _ = ps.anchorage_loss_reverse(SIG, s_end, l, 6.0, EP, l - x)
    dsd = (SIG - s_end) / l
    assert loss == pytest.approx(np.minimum(dsd * x + la, dsd * (l - x) + lb))
    l = 60000.0
    s_end = 1255.0
    dsd = (SIG - s_end) / l
    x = np.linspace(0, l, 601)
    loss, (lfa, lfb) = ps.anchorage_loss_two_ends(SIG, s_end, l, 6.0, EP, x, dl_b=0.0)
    la, _ = ps.anchorage_loss_reverse(SIG, s_end, l, 6.0, EP, x)
    assert loss[x < lfa] == pytest.approx((dsd * x + la)[x < lfa])
    assert loss[x >= lfa] == pytest.approx(np.minimum(dsd * x, dsd * (l - x))[x >= lfa], abs=1e-6)


def test_two_ends_with_exact_friction():
    """friction= takes σl1 from each end (6.2.2, exponential); with the straight lines passed explicitly the default is
    reproduced"""
    l, s_end = 40000.0, 1290.0
    dsd = (SIG - s_end) / l
    x = np.linspace(0, l, 81)
    d, _ = ps.anchorage_loss_two_ends(SIG, s_end, l, 6.0, EP, x)
    e, _ = ps.anchorage_loss_two_ends(SIG, s_end, l, 6.0, EP, x, friction=(dsd * x, dsd * (l - x)))
    assert d == pytest.approx(e, abs=1e-12)
    th = np.linspace(0, 0.3, 81)  # angle grows evenly along the tendon
    l1a = np.array([ps.friction_loss(SIG, 0.25, t, 0.0015, xx / 1000) for t, xx in zip(th, x)])
    l1b = np.array([ps.friction_loss(SIG, 0.25, 0.3 - t, 0.0015, (l - xx) / 1000) for t, xx in zip(th, x)])
    f, _ = ps.anchorage_loss_two_ends(SIG, SIG - l1a[-1], l, 6.0, EP, x, friction=(l1a, l1b))
    assert f == pytest.approx(f[::-1], abs=1e-9)  # symmetric tendon
    assert (f >= np.minimum(l1a, l1b) - 1e-9).all()


# ── 8.1.8, 8.1.9 negative-moment regions and class B composite members ──
def _comp(precast_tendons=True, concrete="C40"):
    td = [(1390.0, 900, 1250, 90)] if precast_tendons else []
    pre = jtg.rect(400, 1000, concrete="C50", bottom=[(4, 16, 50)], tendons=td, sigma_pe=1150.0 if td else None)
    return pre, jtg.composite(pre, b=1200, t=200, concrete=concrete)


def test_composite_stage2_matches_a_monolithic_section():
    """same concrete in the topping (n = 1) and no stage-1 moment: the stage-2 stresses at the topping's top and at the
    precast bottom are those of the monolithic T section built directly with jtg.tee (another construction path)"""
    pre, comp = _comp(concrete="C50")
    mono = jtg.tee(400, 1200, 1200, 200, concrete="C50", bottom=[(4, 16, 50)], tendons=[(1390.0, 1100, 1250, 90)],
                   sigma_pe=1150.0)
    A0, z0, I0 = mono.props("transformed")
    M2s, M2l = -900.0, -700.0  # hogging
    r = jtg.composite_crack_resistance(comp, M1k=0.0, M1Gk=0.0, M2s=M2s, M2l=M2l, pc_class="A")[1]
    top = r.details["edges"]["topping top"]
    assert top["sigma_st"] == pytest.approx(M2s * 1e6 * (0 - z0) / I0, rel=1e-12)
    assert top["sigma_lt"] == pytest.approx(M2l * 1e6 * (0 - z0) / I0, rel=1e-12)
    assert top["sigma_pc"] == 0.0 and top["ftk"] == jtg.concrete(50).ftk
    bot = r.details["edges"]["precast bottom"]
    assert bot["sigma_st"] == pytest.approx(M2s * 1e6 * (1200 - z0) / I0, rel=1e-12)
    # hogging: the topping governs and, without stage-2 prestress, cracks under the class A limits
    assert top["demand"] > top["limit"] and not r.ok


def test_composite_topping_edge_uses_modulus_ratio():
    """a C40 topping on a C50 girder: the actual topping stress is n × the transformed-section stress"""
    pre, comp = _comp()
    _, z0, I0 = comp.elastic.props("transformed")
    r = jtg.composite_crack_resistance(comp, M1k=0.0, M1Gk=0.0, M2s=-900.0, M2l=-700.0, pc_class="full")[1]
    assert r.details["edges"]["topping top"]["sigma_st"] == pytest.approx(comp.n * 900e6 * z0 / I0)
    assert r.details["edges"]["topping top"]["ftk"] == jtg.concrete(40).ftk


def test_composite_prestress_stress_equilibrium():
    """stage-2 tendons in the topping (duct a hole in the topping, scaled by n in the transformed section): the stress
    field integrated over the actual concrete (topping at its actual width and stress, girder, less the duct, the steel
    at αE) balances the tendon force and its moment about the tendon — statics by quadrature"""
    pre, comp = _comp(precast_tendons=False)
    Ap, d, spe, dd = 1112.0, 100.0, 1000.0, 80.0
    duct = math.pi * dd ** 2 / 4
    sig = jtg.composite_prestress_stress(comp, [(Ap, d, spe, dd)])
    n = comp.n
    zs = np.linspace(0, 1200, 1_200_001)
    zm = (zs[1:] + zs[:-1]) / 2
    w = np.where(zm < 200, 1200.0, 400.0)
    s = sig(zm)
    dz = zs[1] - zs[0]
    F = float((s * w).sum() * dz) - sig(d) * duct
    Mo = float((s * w * (zm - d)).sum() * dz)
    aS = 2e5 / pre.concrete.Ec
    bars = comp.elastic.bars
    F += sum((aS - 1) * a * sig(z) for a, z in bars)  # bars at αES − 1 over the concrete they displace (the girder's)
    Mo += sum((aS - 1) * a * sig(z) * (z - d) for a, z in bars)
    assert F == pytest.approx(Ap * spe, rel=1e-6)
    assert Mo == pytest.approx(0.0, abs=1e-6 * Ap * spe * 1200)
    # a secondary moment Mp2 (hogging here) adds a self-equilibrated linear field on the same section: no net force, a
    # couple equal to Mp2, the top in tension
    Mp2 = -300.0
    dsig = lambda z: jtg.composite_prestress_stress(comp, [(Ap, d, spe, dd)], M_p2=Mp2)(z) - sig(z)
    ds = dsig(zm)
    F2 = float((ds * w).sum() * dz) - dsig(d) * duct + sum((aS - 1) * a * dsig(z) for a, z in bars)
    M2 = float((ds * w * (600.0 - zm)).sum() * dz) - dsig(d) * duct * (600.0 - d) + \
        sum((aS - 1) * a * dsig(z) * (600.0 - z) for a, z in bars)
    assert F2 == pytest.approx(0.0, abs=1e-6 * Ap * spe)
    assert M2 == pytest.approx(Mp2 * 1e6, rel=1e-6)
    assert dsig(0.0) < 0 < dsig(1200.0)


def test_composite_negative_moment_with_deck_tendons():
    """the deck tendons' precompression enters σpc at every composite edge (sigma_pc2) and the topping can meet class A"""
    pre, comp = _comp()
    M2s, M2l = -900.0, -700.0
    bare = jtg.composite_crack_resistance(comp, M1k=0.0, M1Gk=0.0, M2s=M2s, M2l=M2l, pc_class="A")[1]
    sig = jtg.composite_prestress_stress(comp, [(2 * 1112.0, 100.0, 1050.0, 95.0)])
    r = jtg.composite_crack_resistance(comp, M1k=0.0, M1Gk=0.0, M2s=M2s, M2l=M2l, pc_class="A", sigma_pc2=sig)[1]
    for name, z in (("topping top", 0.0), ("precast top", 200.0), ("precast bottom", 1200.0)):
        e0, e1 = bare.details["edges"][name], r.details["edges"][name]
        assert e1["sigma_pc"] == pytest.approx(e0["sigma_pc"] + sig(z))
        assert e1["sigma_st"] == e0["sigma_st"]
    t1 = r.details["edges"]["topping top"]
    assert t1["sigma_pc"] > 0 and t1["demand"] <= t1["limit"] and t1["sigma_lt"] <= t1["sigma_pc"]
    t0 = bare.details["edges"]["topping top"]
    assert t0["demand"] > t0["limit"]
    s0 = jtg.composite_service_stresses(comp, M1k=0.0, M2k=-1000.0, pc_class="A")
    s1 = jtg.composite_service_stresses(comp, M1k=0.0, M2k=-1000.0, pc_class="A", sigma_pc2=sig)
    assert s1.details["edges"]["precast bottom"][0] == pytest.approx(s0.details["edges"]["precast bottom"][0] + sig(1200.0))


def test_composite_class_b_self_weight():
    """class B (6.3.1 item 1-3 in the two-stage form of 8.1.8): under the self-weight alone the edge in tension under the
    frequent combination stays compressed: M1Gk / W01 + M2Gk / W0 ≤ σpc (hand calculation); M2Gk required"""
    pre, comp = _comp()
    _, z01, I01 = pre.props("transformed")
    _, z0, I0 = comp.elastic.props("transformed")
    spc = float(pre.prestress_stress(1000.0))
    kw = dict(M1k=650.0, M1Gk=600.0, M2s=900.0, M2l=700.0, pc_class="B")
    with pytest.raises(ValueError):
        jtg.composite_crack_resistance(comp, **kw)
    r1, r2 = jtg.composite_crack_resistance(comp, M2Gk=150.0, **kw)
    assert r2.details["edge"] == "precast bottom"
    assert r2.demand == pytest.approx(600e6 * (1000 - z01) / I01 + 150e6 * (1200 - z0) / I0)
    assert r2.resistance == pytest.approx(spc) and r2.ok and "1-3" in r2.clause
    assert r1.details["edge"] == "precast top" and r1.demand == pytest.approx(600e6 * (0 - z01) / I01)
    # 8.1.9: class B takes the class A limits (6.3.1-7, -8 give both)
    a = jtg.composite_inclined_crack_resistance(comp, 600.0, 200.0, 900.0, 300.0, pc_class="A")
    b = jtg.composite_inclined_crack_resistance(comp, 600.0, 200.0, 900.0, 300.0, pc_class="B")
    assert (a.demand, a.resistance) == (b.demand, b.resistance)
