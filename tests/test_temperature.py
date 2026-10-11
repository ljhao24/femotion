"""Crowd load (D60 4.3.6), temperature actions (D60 4.3.12), temperature stresses (JTG 3362-2018 appendix D) and the
secondary effects of imposed strains on frames (femotion.frame.thermal), against hand calculations."""
import numpy as np
import pytest

from femotion.codes import jtg
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import InfluenceLines, chain_sections
from femotion.frame.thermal import effects


def test_crowd_load():
    assert jtg.crowd_load(20) == 3.0 and jtg.crowd_load(50) == 3.0
    assert jtg.crowd_load(100) == pytest.approx(3.25 - 0.5)
    assert jtg.crowd_load(150) == 2.5 and jtg.crowd_load(400) == 2.5
    assert jtg.crowd_load(30, dense=True) == pytest.approx(3.45)
    assert jtg.crowd_load(300, footbridge=True) == 3.5
    assert jtg.SIDEWALK_SLAB_LOAD == 4.0 and jtg.RAILING_LOADS == (0.75, 1.0)


def _beam(spans, L=20.0, n=20, sec=None):
    sec = sec or Section(E=3.45e7, A=0.6, Iy=0.25, Iz=1.0, J=0.1)
    m = Model()
    xs = np.linspace(0, spans * L, spans * n + 1)
    nodes = [m.node((x, 0, 0)) for x in xs]
    beams = [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
    m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
    for k in range(1, spans + 1):
        m.fix(nodes[k * n], (0, 1, 1, 1, 0, 0))
    return Solver(m), nodes, beams, sec


def test_crowd_effects_simple_beam():
    sv, nodes, beams, _ = _beam(1)
    il = InfluenceLines(sv, nodes, chain_sections(beams))
    hi, lo = jtg.CrowdLoad(L0=20, width=2.0).effects(il, "My")
    assert hi.max() == pytest.approx(3.0 * 2.0 * 20 ** 2 / 8, rel=1e-9)  # uniform over the whole span, no point load
    assert lo.min() == pytest.approx(0.0, abs=1e-9)


def test_thermal_frame_secondary():
    # simply supported: free; two equal spans with free curvature κ: support reaction 3EIκ/L, moment −1.5 EIκ
    sv, nodes, beams, sec = _beam(1)
    assert np.abs(effects(sv, beams, chain_sections(beams), "My", curvature=-1e-4)).max() < 1e-6
    sv, nodes, beams, sec = _beam(2)
    M = effects(sv, beams, chain_sections(beams), "My", curvature=-1e-4)
    assert M.max() == pytest.approx(-1.5 * sec.E * sec.Iy * -1e-4, rel=1e-9)
    # linear along the span (no load inside the spans)
    secs = chain_sections(beams)
    assert M[secs.index((9, "j"))] == pytest.approx(M.max() * 10 / 20, rel=1e-9)
    # axial strain of a beam fixed axially at both ends: N = −EA ε
    m = Model()
    a, b = m.node((0, 0, 0)), m.node((10, 0, 0))
    e = m.beam(a, b, sec)
    m.fix(a)
    m.fix(b)
    N = effects(Solver(m), [e], [(e, "i")], "N", strain=2e-4)
    assert N[0] == pytest.approx(-sec.E * sec.A * 2e-4)


def test_thermal_frame_three_spans():
    # three equal spans, uniform free curvature: both interior supports −1.2 EIκ (three-moment equation)
    sv, nodes, beams, sec = _beam(3)
    M = effects(sv, beams, chain_sections(beams), "My", curvature=-1e-4)
    secs = chain_sections(beams)
    for b in (19, 39):
        assert M[secs.index((b, "j"))] == pytest.approx(-1.2 * sec.E * sec.Iy * -1e-4, rel=1e-9)


def test_effective_temperature():
    assert jtg.effective_temperature("severe cold", "concrete") == (34, -23)
    assert jtg.effective_temperature("cold", "steel deck") == (46, -21)
    assert jtg.effective_temperature("mild", "concrete deck", south=True) == (39, -1)
    with pytest.raises(ValueError):
        jtg.effective_temperature("cold", "concrete", south=True)
    assert jtg.EXPANSION["concrete"] == 1e-5 and jtg.EXPANSION["steel"] == 1.2e-5


def test_gradient():
    g = jtg.temperature_gradient(1600)
    assert g.z == (0, 100, 400) and g.t == (25, 6.7, 0)
    assert g(50) == pytest.approx((25 + 6.7) / 2) and g(250) == pytest.approx(6.7 / 2) and g(800) == 0
    g = jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=75)
    assert g.t[:2] == pytest.approx((17.0, 6.1))
    g = jtg.temperature_gradient(300, surfacing="asphalt", asphalt=100)
    assert g.z == (0, 100, 300) and g.t == (14, 5.5, 0)  # A = H − 100 below 400 mm
    # steel girder with a 250 mm concrete deck: A = 300, the steel below the deck keeps T(t) down to the bottom
    g = jtg.temperature_gradient(1200, steel=True, t=250)
    assert g.z == (0, 100, 250, 1200) and g.t == pytest.approx((25, 6.7, 6.7 / 2, 6.7 / 2))
    assert g(800) == pytest.approx(3.35) and jtg.temperature_gradient(300, steel=True, t=150).z == (0, 100, 150, 300)
    with pytest.raises(ValueError):
        jtg.temperature_gradient(1200, steel=True)
    with pytest.raises(ValueError):
        jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=120)
    assert jtg.NEGATIVE_GRADIENT == -0.5


def test_thermal_stresses_rect_hand():
    # rectangle 500 × 1000, gross section, gradient 25 / 6.7 / 0 at 0 / 100 / 400 mm; Ec of C50
    sec = jtg.rect(500, 1000, concrete="C50", bottom=[(4, 25, 50)])
    g = jtg.temperature_gradient(1000)
    th = jtg.thermal_stresses(sec, g, kind="gross")
    E, a, b, h = 3.45e4, 1e-5, 500.0, 1000.0
    # ∫ t dz and ∫ t z dz of the two linear pieces
    p1 = (25 + 6.7) / 2 * 100
    m1 = 100 ** 2 * (25 / 6 + 6.7 / 3)  # ∫0^100 t z dz, t linear 25 → 6.7
    p2 = 6.7 / 2 * 300
    m2 = 6.7 * 300 / 2 * (100 + 300 / 3)  # triangle 6.7 → 0 over 100 … 400: centroid at 100 + 100
    Nt = a * E * b * (p1 + p2)
    zc, A, I = h / 2, b * h, b * h ** 3 / 12
    Mt0 = -a * E * b * ((p1 + p2) * zc - (m1 + m2))
    assert th.Nt == pytest.approx(Nt, rel=1e-12) and th.Mt0 == pytest.approx(Mt0, rel=1e-12)
    for z in (0, 100, 300, 1000):
        s = -Nt / A + Mt0 * (zc - z) / I + g(z) * a * E
        assert th.sigma(z) == pytest.approx(s, rel=1e-12)
        assert th.negative().sigma(z) == pytest.approx(-0.5 * s, rel=1e-12)
    # self-equilibrating: ∫ σ b dz = 0, ∫ σ b (zc − z) dz = 0
    z = np.linspace(0, h, 200001)
    zm = (z[:-1] + z[1:]) / 2
    s = th.sigma(zm)
    assert abs((s * b * np.diff(z)).sum()) < 1e-6 * Nt
    assert abs((s * b * (zc - zm) * np.diff(z)).sum()) < 1e-6 * abs(Mt0)
    # free strain and curvature for the frame
    eps, kap = th.free_strain()
    assert eps == pytest.approx(Nt / (E * A)) and kap == pytest.approx(Mt0 / (E * I) * 1e3)
    # secondary moment adds Mt' y / I (D.0.2)
    th2 = jtg.thermal_stresses(sec, g, kind="gross", M_sec=100.0)
    assert th2.sigma(0) - th.sigma(0) == pytest.approx(100e6 * zc / I)


def test_thermal_stresses_balance_on_transformed_tee():
    sec = jtg.tee(200, 1600, 2000, 200, concrete="C50", bottom=[(6, 25, 60)], top=[(10, 12, 40)])
    th = jtg.thermal_stresses(sec, jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=100))
    A, zc, I = sec.props("transformed")
    assert (th.A, th.zc, th.I) == pytest.approx((A, zc, I))
    # a uniform temperature over the transformed section (bars included) gives no stress
    u = jtg.thermal_stresses(sec, jtg.Profile((0.0, 1600.0), (10.0, 10.0)))
    assert max(abs(u.sigma(z)) for z in (0, 300, 800, 1600)) < 1e-9
    # balance over the transformed section: concrete strips plus bars
    z = np.linspace(0, 1600, 160001)
    zm = (z[:-1] + z[1:]) / 2
    bw = np.array([sec.width(v) for v in zm])
    s = th.sigma(zm)
    # bar points carry (αEs − 1) As of transformed area at the stress of their fibre
    pts = sec._steel_points("transformed")
    F = (s * bw * np.diff(z)).sum() + sum(a_ * th.sigma(d) for a_, d in pts)
    M = (s * bw * (zc - zm) * np.diff(z)).sum() + sum(a_ * th.sigma(d) * (zc - d) for a_, d in pts)
    assert abs(F) < 1e-6 * th.Nt and abs(M) < 1e-6 * abs(th.Mt0)


def test_thermal_cracked():
    sec = jtg.rect(400, 800, concrete="C30", bottom=[(4, 25, 50)])
    g = jtg.temperature_gradient(800)
    th = jtg.thermal_stresses(sec, g, cracked=True)
    x0 = jtg.cracked_section(sec)["x0"]
    assert th.z_max == pytest.approx(x0)
    assert th.t(x0 + 1) == 0 and th.t(10) == pytest.approx(g(10))
    hog = jtg.thermal_stresses(sec.replace(bars=np.array([[4 * 491, 50.0]]), bar_d=np.array([25.0])), g, cracked=True,
                               hogging=True)
    assert hog.t(10) == 0  # the top is cracked: the gradient there is ignored


def test_crack_resistance_with_temperature():
    pc = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1470, 1150, 90)],
                 sigma_pe=1080.0)
    th = jtg.thermal_stresses(pc, jtg.temperature_gradient(1600))
    base = jtg.crack_resistance(pc, 3000, 2500, pc_class="A")
    r = jtg.crack_resistance(pc, 3000, 2500, pc_class="A", temperature=th)
    for edge in ("top", "bottom"):
        t_ = -0.8 * min(0.0, th.sigma(0.0 if edge == "top" else 1600.0), th.negative().sigma(0.0 if edge == "top" else 1600.0))
        assert r.details[edge]["sigma_st"] == pytest.approx(base.details[edge]["sigma_st"] + t_)
        assert r.details[edge]["sigma_lt"] == pytest.approx(base.details[edge]["sigma_lt"])  # 6.3.2: not in Ml
    # the negative gradient puts the top in tension
    assert r.details["top"]["sigma_t"] == pytest.approx(-0.8 * th.negative().sigma(0.0))
    # principal stresses: σcx takes ψ × the more tensile gradient
    p0 = jtg.principal_stresses(pc, 3000, 800, depths=[400])[0]
    p1 = jtg.principal_stresses(pc, 3000, 800, depths=[400], temperature=th, psi_t=0.8)[0]
    assert p1["sigma_cx"] - p0["sigma_cx"] == pytest.approx(0.8 * min(0.0, th.sigma(400), th.negative().sigma(400)))
    i0 = jtg.inclined_crack_resistance(pc, 3000, 800, pc_class="A", depths=[400], temperature=th)
    assert i0.details["at"]["sigma_cx"] == pytest.approx(p1["sigma_cx"])
    # service stresses: the more compressive gradient at the edges
    s0 = jtg.service_stresses(pc, 4000, pc_class="A")
    s1 = jtg.service_stresses(pc, 4000, pc_class="A", temperature=th)
    assert s1.details["edge_stress"]["top"] - s0.details["edge_stress"]["top"] == pytest.approx(
        max(0.0, th.sigma(0.0), th.negative().sigma(0.0)))
    with pytest.raises(NotImplementedError):
        jtg.service_stresses(pc, 4000, pc_class="B", temperature=th)


def test_thermal_literal_and_cracked_prestressed():
    sec = jtg.tee(200, 1600, 2000, 200, concrete="C50", bottom=[(6, 25, 60)], top=[(10, 12, 40)])
    g = jtg.temperature_gradient(1600, surfacing="asphalt", asphalt=100)
    lit = jtg.thermal_stresses(sec, g, kind="literal")
    gross = jtg.thermal_stresses(sec, g, kind="gross")
    A0, z0, I0 = sec.props("transformed")
    # appendix D as printed: Ay the concrete, A0 and I0 transformed
    zc = z0
    Nt = gross.Nt  # the concrete-only integrals of the gross case, with the first moment moved to the transformed centroid
    q0 = gross.Nt / (1e-5 * 3.45e4)
    q1 = -gross.Mt0 / (1e-5 * 3.45e4) + q0 * (zc - sec.zg)  # shift the first moment to the transformed centroid
    assert lit.Nt == pytest.approx(Nt) and lit.Mt0 == pytest.approx(-1e-5 * 3.45e4 * q1)
    assert lit.sigma(0) == pytest.approx(-Nt / A0 + lit.Mt0 * zc / I0 + 14 * 0.345)
    # cracked prestressed (class B): neutral axis by appendix J under Np0 and M
    pc = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2000, 1470, 1000, 90)],
                 sigma_pe=950.0)
    with pytest.raises(ValueError):
        jtg.thermal_stresses(pc, g, cracked=True)
    th = jtg.thermal_stresses(pc, g, cracked=True, M=5500.0)
    r = jtg.service_stresses(pc, 5500.0, pc_class="B")
    assert th.z_max == pytest.approx(r.details["x"]) and th.A == pytest.approx(r.details["Acr"])
    with pytest.raises(ValueError):
        th.free_strain()  # an imposed curvature comes from an uncracked section


def test_principal_compression_with_temperature():
    pc = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1470, 1150, 90)],
                 sigma_pe=1080.0)
    th = jtg.thermal_stresses(pc, jtg.temperature_gradient(1600))
    r = jtg.principal_compression(pc, 4000, 900, depths=[180], temperature=th)
    p0 = jtg.principal_stresses(pc, 4000, 900, depths=[180])[0]
    up = max(0.0, th.sigma(180), th.negative().sigma(180))
    dn = min(0.0, th.sigma(180), th.negative().sigma(180))
    assert r.details["at"]["sigma_cx"] == pytest.approx(p0["sigma_cx"] + up)  # σcp: the more compressive gradient
    t = jtg.principal_stresses(pc, 4000, 900, depths=[180], temperature=th, worst="tension")[0]
    assert t["sigma_cx"] == pytest.approx(p0["sigma_cx"] + dn) and r.details["sigma_tp"] == pytest.approx(t["sigma_tp"])
