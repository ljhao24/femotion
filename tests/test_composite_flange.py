"""Effective flange widths (JTG 3362-2018 4.3.3, 4.3.4) and composite flexural members (8.1), against hand
calculations: the composite section's properties by the parallel-axis theorem, two-stage stresses superposed by hand."""
import math

import pytest

from femotion.codes import jtg
from femotion.codes.jtg.flange import rho_f, rho_s


def test_tee_flange_width():
    # L/3, spacing, b + 2bh + 12h'f each governing
    assert jtg.tee_flange_width(b=200, hf=160, spacing=2200, L=6000) == pytest.approx(2000)
    assert jtg.tee_flange_width(b=200, hf=160, spacing=1800, L=30000) == pytest.approx(1800)
    assert jtg.tee_flange_width(b=200, hf=120, spacing=2400, L=30000, bh=300, hh=150) == pytest.approx(200 + 600 + 1440)
    # haunch flatter than 1:3: bh replaced by 3 hh
    assert jtg.tee_flange_width(b=200, hf=120, spacing=4000, L=30000, bh=600, hh=150) == pytest.approx(200 + 900 + 1440)
    assert jtg.tee_flange_width(b=200, hf=500, spacing=9000, L=30000, region="interior span") == pytest.approx(6000)
    assert jtg.tee_flange_width(b=200, hf=700, spacing=9000, L=30000, region="end span") == pytest.approx(8100)
    assert jtg.tee_flange_width(b=200, hf=400, spacing=9000, region="support", spans=(30000, 40000)) == pytest.approx(4900)
    assert jtg.tee_flange_width_outer(2000, b=200, overhang=900, t_overhang=160) == pytest.approx(1000 + 100 + 900)
    assert jtg.tee_flange_width_outer(2000, b=200, overhang=1200, t_overhang=160) == pytest.approx(1000 + 100 + 960)


def test_box_flange_width():
    r = 0.3
    assert rho_f(r) == pytest.approx(-6.44 * 0.0081 + 10.10 * 0.027 - 3.56 * 0.09 - 1.44 * 0.3 + 1.08)
    assert rho_s(r) == pytest.approx(21.86 * 0.0081 - 38.01 * 0.027 + 24.57 * 0.09 - 7.67 * 0.3 + 1.27)
    assert jtg.box_flange_coefficient(3000, 10000) == pytest.approx(rho_f(0.3))
    assert jtg.box_flange_coefficient(3000, 10000, support=True) == pytest.approx(rho_s(0.3))
    assert jtg.box_flange_coefficient(100, 10000) == 1.0  # the fitted ρf exceeds 1 for tiny bi/li
    assert jtg.box_flange_coefficient(3000, 10000, h=10001) == 1.0  # h ≥ bi / 0.3
    # from bi/li = 0.7 on: the constants of the commentary's figure 4-9 (0.173 li, 0.104 li), close to the curves at 0.7
    assert jtg.box_flange_coefficient(8000, 10000) * 8000 == pytest.approx(1730)
    assert jtg.box_flange_coefficient(8000, 10000, support=True) * 8000 == pytest.approx(1040)
    assert rho_f(0.7) * 0.7 == pytest.approx(0.173, abs=0.002) and rho_s(0.7) * 0.7 == pytest.approx(0.104, abs=0.003)
    assert jtg.box_flange_coefficient(6650, 10000) == pytest.approx(rho_f(0.665))  # 0.64 … 0.7: the fitted curves
    # along a simply supported span: ρs at the supports, ρf beyond a = bi, straight line between
    bi, l = 3000.0, 40000.0
    ws, wf = rho_s(bi / l) * bi, rho_f(bi / l) * bi
    assert jtg.box_flange_width(bi, l, 0) == pytest.approx(ws)
    assert jtg.box_flange_width(bi, l, 1500) == pytest.approx((ws + wf) / 2)
    assert jtg.box_flange_width(bi, l, 20000) == pytest.approx(wf)
    # end span: li = 0.8 l at the end support and the span, 0.2 (l + l') at the interior support, c = 0.1 l there
    w0 = jtg.box_flange_width(bi, l, 0, kind="end span", right=50000)
    wm = jtg.box_flange_width(bi, l, 20000, kind="end span", right=50000)
    w1 = jtg.box_flange_width(bi, l, l, kind="end span", right=50000)
    wc = jtg.box_flange_width(bi, l, l - 2000, kind="end span", right=50000)
    assert w0 == pytest.approx(rho_s(bi / 32000) * bi) and wm == pytest.approx(rho_f(bi / 32000) * bi)
    assert w1 == pytest.approx(rho_s(bi / 18000) * bi) and wc == pytest.approx((wm + w1) / 2)
    assert jtg.box_theoretical_span(40000, "interior span", left=30000, right=50000) == (14000, 24000, 18000)
    assert jtg.box_theoretical_span(40000, "end span", right=50000) == (32000, 32000, 18000)
    assert jtg.box_theoretical_span(10000, "cantilever") == (15000, 15000, 15000)
    # interior span: c = 0.1 l zones at both ends, each support with its own neighbour
    l = 50000.0
    wl = jtg.box_flange_width(4000, l, 0, kind="interior span", left=40000, right=30000)
    wm = jtg.box_flange_width(4000, l, 25000, kind="interior span", left=40000, right=30000)
    wr = jtg.box_flange_width(4000, l, l, kind="interior span", left=40000, right=30000)
    assert wl == pytest.approx(rho_s(4000 / 18000) * 4000) and wr == pytest.approx(rho_s(4000 / 16000) * 4000)
    assert wm == pytest.approx(rho_f(4000 / 30000) * 4000)
    assert jtg.box_flange_width(4000, l, 2500, kind="interior span", left=40000, right=30000) == pytest.approx((wl + wm) / 2)


def _pc_composite():
    pre = jtg.tee(200, 1600, 1600, 160, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1450, 1150, 90)],
                  sigma_pe=1050)
    return pre, jtg.composite(pre, b=2200, t=180, concrete="C40", top=[(14, 12, 50)])


def test_composite_sections():
    pre, comp = _pc_composite()
    n = 3.25e4 / 3.45e4
    assert comp.n == pytest.approx(n) and comp.h == 1780 and comp.h1 == 1600
    e, u = comp.elastic, comp.ultimate
    assert e.pieces[0].b == pytest.approx(2200 * n) and u.pieces[0].b == 2200
    assert u.concrete.grade == 40 and e.concrete.grade == 50
    # transformed properties by the parallel-axis theorem: precast transformed + topping + its bars
    A1, z1, I1 = pre.props("transformed")
    At, zt, It = 2200 * n * 180, 90.0, 2200 * n * 180 ** 3 / 12
    Ab = 14 * math.pi * 36 * (2e5 / 3.45e4 - 1)
    A = A1 + At + Ab
    z = ((A1 * (z1 + 180)) + At * zt + Ab * 50) / A
    I = I1 + A1 * (z1 + 180 - z) ** 2 + It + At * (zt - z) ** 2 + Ab * (50 - z) ** 2
    A0, z0, I0 = e.props("transformed")
    assert A0 == pytest.approx(A) and z0 == pytest.approx(z) and I0 == pytest.approx(I)
    # bars and tendons moved down by the topping
    assert e.tendons[0, 1] == pytest.approx(1630) and e.bars[-1, 1] == pytest.approx(1780 - 50)


def test_composite_design_effects():
    assert jtg.composite_design_moments(1800, 150, 600, 2600) == pytest.approx((1.2 * 1800 + 1.4 * 150,
                                                                                  1.2 * 1800 + 1.2 * 600 + 2600))


def test_composite_crack_resistance_hand():
    pre, comp = _pc_composite()
    r1, r2 = jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="A")
    _, z01, I01 = pre.props("transformed")
    _, z0, I0 = comp.elastic.props("transformed")
    W01, W0 = I01 / (1600 - z01), I0 / (1780 - z0)
    spc = float(pre.prestress_stress(1600))
    b1, b2 = r1.details["edges"]["precast bottom"], r2.details["edges"]["precast bottom"]
    assert b1["demand"] == pytest.approx(1950e6 / W01) and b1["limit"] == pytest.approx(spc + 0.7 * 2.65)
    assert b2["demand"] == pytest.approx(1800e6 / W01 + 1700e6 / W0)
    assert b2["sigma_lt"] == pytest.approx(1800e6 / W01 + 1100e6 / W0)
    # both precast edges and the topping's top are checked; the least margin governs (the precast top here)
    assert r1.details["edge"] == "precast top" and r1.ok and r2.ok
    full = jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="full")[1]
    assert full.details["edge"] == "precast bottom" and full.resistance == pytest.approx(0.85 * spc)
    with pytest.raises(ValueError):
        jtg.composite_crack_resistance(comp, M1k=1, M1Gk=1, M2s=1, M2l=1, pc_class="B")


def test_composite_principal_and_service_hand():
    pre, comp = _pc_composite()
    _, z01, I01 = pre.props("transformed")
    _, z0, I0 = comp.elastic.props("transformed")
    zc = 180 + z01
    p = jtg.composite_principal_stresses(comp, 1800, 300, 1700, 400, depths=[zc])[0]
    sx = float(pre.prestress_stress(z01)) + 0 + 1700e6 * (z0 - zc) / I0
    tau = 300e3 * pre.first_moment(z01) / (200 * I01) + 400e3 * comp.elastic.first_moment(zc) / (200 * I0)
    assert p["sigma_cx"] == pytest.approx(sx) and p["tau"] == pytest.approx(tau)
    assert p["sigma_tp"] == pytest.approx(math.hypot(sx / 2, tau) - sx / 2)
    r = jtg.composite_inclined_crack_resistance(comp, 1800, 300, 1700, 400, pc_class="A")
    assert r.resistance == pytest.approx(0.7 * 2.65)
    with pytest.raises(ValueError):
        jtg.composite_principal_stresses(comp, 1, 1, 1, 1, depths=[100])  # in the topping
    s = jtg.composite_service_stresses(comp, M1k=1950, M2k=2300, pc_class="A")
    top_pre = float(pre.prestress_stress(0)) + 1950e6 * z01 / I01 + 2300e6 * (z0 - 180) / I0
    assert s.details["edges"]["precast top"][0] == pytest.approx(top_pre)
    assert s.details["edges"]["topping top"] == pytest.approx((comp.n * 2300e6 * z0 / I0, 0.5 * 26.8))
    dp = 1450.0
    sp = 1050 + 1.95e5 / 3.45e4 * (1950e6 * (dp - z01) / I01 + 2300e6 * (dp + 180 - z0) / I0)
    assert s.details["tendon_stress"][0] == pytest.approx(sp)


def test_composite_shear_and_joint():
    pre, comp = _pc_composite()
    pre_chk, cmp_chk = jtg.composite_shear(comp, 500, 900, (2, 12, 150), gamma0=1.1, pc_class="A")
    low = comp.ultimate.replace(concrete=jtg.concrete("C40"))
    raw = jtg.shear(low, 900, (2, 12, 150), gamma0=1.1, alpha2=1.0)
    floor = jtg.shear(pre, 0, (2, 12, 150), pc_class="A").resistance
    assert cmp_chk.resistance == pytest.approx(max(raw.resistance, floor))
    assert cmp_chk.details["alpha2"] == 1.0 and cmp_chk.details["concrete_used"] == "C40"
    j = jtg.interface_shear(comp, 900, (2, 12, 150), gamma0=1.1)
    h0 = comp.ultimate.h0()
    # b: the joint width, the precast girder's top flange
    assert j.resistance == pytest.approx((0.12 * 18.4 * 1600 * h0 + 0.85 * 330 * 2 * math.pi * 36 / 150 * h0) / 1e3)
    s = jtg.interface_shear_slab(300, b=1000, h0=400, gamma0=1.0, s=200, Asv=150)
    assert s.demand == pytest.approx(0.75) and s.details["Asv_min"] == pytest.approx(0.3 * 1000 * 200 / 330)
    assert any("8.1.7-3" in f for f in s.failed)
    assert any("9.2.7" in n for n in jtg.interface_shear_slab(150, b=1000, h0=400).notes)


def test_composite_rc_crack_width_and_stiffness():
    pre = jtg.tee(200, 1100, 1000, 120, concrete="C40", bottom=[(8, 25, 60)])
    comp = jtg.composite(pre, b=1600, t=150, concrete="C40", top=[(10, 12, 40)])
    r = jtg.composite_crack_width(comp, M1Gk=500, M2s=600, M2l=350)
    As = 8 * math.pi * 25 ** 2 / 4
    h01, h0 = 1040.0, 1190.0
    M1u = jtg.flexure(pre, 1.0).resistance
    assert 500 >= 0.35 * M1u  # so h1 stays the precast depth here
    s1 = 500e6 / (0.87 * As * h01)
    s2 = 0.5 * (1 + 1100 / 1250) * 600e6 / (0.87 * As * h0)
    C2 = 1 + 0.5 * (500 + 350) / (500 + 600)
    rho = As / (2 * 60 * 200)
    W = C2 * (s1 + s2) / 2e5 * (min(60 - 12.5, 50) + 25) / (0.36 + 1.7 * min(max(rho, 0.01), 0.1))
    assert r.details["sigma_ss"] == pytest.approx(s1 + s2) and r.demand == pytest.approx(W)
    small = jtg.composite_crack_width(comp, M1Gk=0.2 * M1u, M2s=600, M2l=350)
    assert small.details["h1_used"] == 1250  # M1Gk < 0.35 M1u: h1 = h
    B, d = jtg.composite_stiffness(comp, 1100)
    assert B == pytest.approx(0.9 * jtg.stiffness(comp.elastic, 1100)[0])
    _, pc = _pc_composite()
    B, d = jtg.composite_stiffness(pc, 3000, pc_class="A")
    assert B == pytest.approx(0.80 * 3.45e4 * pc.elastic.props("transformed")[2])
    assert jtg.composite_long_term_factor("C30") == 1.80
    assert jtg.composite_long_term_factor("C40") == pytest.approx(1.65)
    assert jtg.composite_long_term_factor("C60") == pytest.approx(1.60)
    assert jtg.COMPOSITE_CAMBER_LONG_TERM == 1.75


def test_shrinkage_difference():
    pre, comp = _pc_composite()
    prof = jtg.shrinkage_difference_profile(comp, -5.0)
    th = jtg.thermal_stresses(comp.elastic, prof)
    A0, z0, I0 = comp.elastic.props("transformed")
    At = 2200 * comp.n * 180
    Nt = 1e-5 * 3.45e4 * (At * -5.0 + sum(a * -5.0 for a, d in comp.elastic._steel_points("transformed") if d < 180))
    assert th.Nt == pytest.approx(Nt, rel=1e-9)
    assert th.sigma(1780) < 0  # the bottom of the precast girder is pulled into tension


def test_composite_review_fixes():
    pre, comp = _pc_composite()
    # composite sections are not taken for T sections by the torsion functions
    assert comp.elastic.kind == "composite"
    with pytest.raises(NotImplementedError):
        jtg.torsion_moduli(comp.ultimate)
    # the flange of the composite still gives α3 = 1.1 (width-based)
    assert jtg.shear(comp.ultimate, 500, (2, 12, 150), alpha2=1.0).details["alpha3"] == 1.1
    # 3.1.2 on the precast concrete: a C30 deck on a prestressed C50 girder is not flagged
    c30 = jtg.composite(pre, b=2200, t=180, concrete="C30")
    assert not any("3.1.2" in f for f in jtg.flexure(c30.ultimate, 3000, pc_class="A").failed)
    # 7.1.6 on the composite member: σcp ≤ 0.6 fck of the precast concrete
    r = jtg.composite_principal_compression(comp, 1800, 300, 2300, 500, stirrups=(2, 12, 150))
    w = max(jtg.composite_principal_stresses(comp, 1800, 300, 2300, 500), key=lambda q: q["sigma_cp"])
    assert r.demand == pytest.approx(w["sigma_cp"]) and r.resistance == pytest.approx(0.6 * 32.4)
    # extra stresses: σst of the composite member and σcx of the principal stresses
    base = jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="A")[1]
    more = jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="A",
                                          sigma_extra=0.5)[1]
    # a number goes to the precast bottom only (as before the other edges were checked); a function to every edge
    eb, em = base.details["edges"], more.details["edges"]
    assert em["precast bottom"]["sigma_st"] == pytest.approx(eb["precast bottom"]["sigma_st"] + 0.5)
    assert em["precast top"]["sigma_st"] == eb["precast top"]["sigma_st"]
    assert em["topping top"]["sigma_st"] == eb["topping top"]["sigma_st"] and "precast bottom only" in more.notes[0]
    fn = jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="A",
                                        sigma_extra=lambda z: 0.5)[1]
    assert all(fn.details["edges"][k]["sigma_st"] == pytest.approx(eb[k]["sigma_st"] + 0.5) for k in eb)
    p0 = jtg.composite_principal_stresses(comp, 1800, 300, 1700, 400, depths=[900])[0]
    p1 = jtg.composite_principal_stresses(comp, 1800, 300, 1700, 400, depths=[900], sigma_cx=lambda z: -0.3)[0]
    assert p1["sigma_cx"] == pytest.approx(p0["sigma_cx"] - 0.3)
    # stage-1 γ0 can be applied
    pre1 = jtg.composite_shear(comp, 500, 900, (2, 12, 150), gamma0=1.1, gamma0_precast=1.1, pc_class="A")[0]
    assert pre1.demand == pytest.approx(550)
