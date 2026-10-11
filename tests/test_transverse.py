"""Transverse analysis of box girders (table 6.1.3 of JTG 3362-2018).

Independent sources: the slope-deflection solution of a closed rectangular frame, statics (reactions, the balance of a
cut through the cell under self-equilibrated actions), the compatibility theorem (a linear temperature field causes no
stress in a body free to deform), the cantilever by the strip formulas of femotion.codes.jtg.slab (a second
implementation), the shear stresses of sectionproperties' finite-element warping analysis (vertical shear and St
Venant torsion), the invariants of plane stress, and sls.crack_resistance / sls.crack_width for the same checks
without axial force."""
import math

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import slab
from femotion.codes.jtg import transverse as tr
from femotion.codes.jtg.temperature import Profile
from femotion.frame.linear import Solver, beam_k

BOX = dict(B=12.5, H=2.5, webs=(-2.9, 2.9), t_top=0.25, t_bottom=0.22, t_web=0.5, t_root=0.55, t_tip=0.2,
           haunch_top=(0.9, 0.3), haunch_bottom=(0.5, 0.2), carriageway=(-5.75, 5.75))


@pytest.fixture(scope="module")
def frame():
    return tr.TransverseFrame(tr.BoxGeometry(**BOX), concrete="C50")


def _inextensible(fr):
    """the same frame with rigid axial members (the slope-deflection assumption)"""
    for (_, _, sec, _, _) in fr.m.beams:
        sec.A = 1e4
    fr.sv = Solver(fr.m)
    fr._k = [beam_k(fr.m.beams[e][2], fr.sv._beam_cache[e][0]) for e in range(len(fr.m.beams))]
    return fr


def test_closed_frame_against_slope_deflection():
    """a rectangular cell (top t1, webs t2, bottom t3, no cantilevers) under q on the top slab: by symmetry two joint
    rotations, θA (1 2k1 + 4k2 − 2k2²/(2k2 + k3)) = q b²/12, θD = −k2 θA/(2k2 + k3); end moments 2k(2θ + θ') …"""
    t1, t2, t3, w = 0.25, 0.4, 0.2, 2.5
    box = tr.BoxGeometry(B=2 * w + t2, H=2.0, webs=(-w, w), t_top=t1, t_bottom=t3, t_web=t2, t_root=t1, t_tip=t1,
                         carriageway=(-w, w))
    assert box.cantilevers() == pytest.approx((0.0, 0.0))
    fr = _inextensible(tr.TransverseFrame(box, concrete="C50", mesh=0.05))
    assert not any("cantilever" in s.name for s in fr.sections)
    q, b, h = 10.0, 2 * w, box.z_top - box.z_bottom
    k1, k2, k3 = t1 ** 3 / b, t2 ** 3 / h, t3 ** 3 / b  # common factor E/12 cancels
    thA = (q * b * b / 12) / (2 * k1 + 4 * k2 - 2 * k2 ** 2 / (2 * k2 + k3))
    thD = -k2 * thA / (2 * k2 + k3)
    MA = 2 * k1 * thA - q * b * b / 12  # top corner, sagging +
    MD = 2 * k3 * thD  # bottom corner
    F = fr.surface(q, -w, w)
    assert F.at("top slab, cell 0 middle")[1] == pytest.approx(q * b * b / 8 + MA, rel=1e-6)
    assert F.at("bottom slab, cell 0 middle")[1] == pytest.approx(MD, rel=1e-5)
    d = t2 / 2
    assert F.at("top slab, web 0 face, cell 0")[1] == pytest.approx(MA + q * b / 2 * d - q * d * d / 2, rel=1e-6)
    Mw = lambda z: -MD + (MA + MD) * (z - box.z_bottom) / h  # inner face in tension +
    zt, zb = box.z_top - t1 / 2, box.z_bottom + t3 / 2
    for nm, z in (("web 0 top", zt), ("web 0 middle", (zt + zb) / 2), ("web 0 bottom", zb)):
        N, M = F.at(nm)
        assert M == pytest.approx(Mw(z), rel=1e-5) and N == pytest.approx(-q * b / 2, rel=1e-6)
    Nt, Nb = F.at("top slab, cell 0 middle")[0], F.at("bottom slab, cell 0 middle")[0]
    assert Nt == pytest.approx(-Nb, rel=1e-9)
    # the cut through the cell's middle: the left half carries the reaction q b/2 at x = −b/2 and the load q b/2 at
    # x = −b/4, so M_t + M_b − N_t z_t − N_b z_b = q b²/4 − q b²/8 = q b²/8
    lhs = F.at("top slab, cell 0 middle")[1] + F.at("bottom slab, cell 0 middle")[1] - Nt * box.z_top - Nb * box.z_bottom
    assert lhs == pytest.approx(q * b * b / 8, rel=1e-6)


def test_statics_and_symmetry(frame):
    """reactions balance the loads (forces and moments about x = 0); a symmetric load gives mirrored forces"""
    fr = frame
    s = fr.surface(2.4, -5.75, 5.75) + fr.line(10.0, -5.95) + fr.line(10.0, 5.95)
    assert s.reactions.sum() == pytest.approx(2.4 * 11.5 + 20.0, rel=1e-10)
    assert s.reactions[0] == pytest.approx(s.reactions[1], rel=1e-9)
    for i, sec in enumerate(fr.sections):
        mirror = (sec.name.replace("left", "@").replace("right", "left").replace("@", "right")
                  .replace("web 0", "@").replace("web 1", "web 0").replace("@", "web 1"))
        j = s.index(mirror)
        assert s.M[i] == pytest.approx(s.M[j], rel=1e-7, abs=1e-9) and s.N[i] == pytest.approx(s.N[j], rel=1e-7, abs=1e-9)
    e = fr.line(50.0, -4.0) + fr.surface(3.0, 1.0, 3.0)
    assert e.reactions.sum() == pytest.approx(56.0)
    assert e.reactions @ np.array(fr.box.webs_bottom) == pytest.approx(-50 * 4.0 + 3.0 * 2.0 * 2.0)


def test_cantilever_root_against_the_strip_formulas(frame):
    """the cantilever is statically determinate: the frame's root forces equal those of slab.cantilever_slab (dead
    load, barrier, surfacing, and the vehicle by 4.2.5 with the same placement rules)"""
    fr = frame
    L = fr.box.cantilevers()[0]
    G = fr.self_weight() + fr.surface(2.4, -5.75, 5.75) + fr.line(10.0, -5.95)
    c = slab.cantilever_slab(L, 0.55, 0.2, h=0.10, curb=L - 0.5, surfacing=(2.4, 0.0, L - 0.5),
                             line_loads=[(10.0, L - 0.3)], lanes=2)
    # the frame takes each element's thickness at its middle: the taper's moment is off by γ t' L Δ²/12 (Δ = 0.1 m)
    assert G.at("left cantilever root")[1] == pytest.approx(c.M_support[0], rel=2e-4)
    assert G.shear("left cantilever root") == pytest.approx(c.V[0], rel=2e-4)
    Q = fr.wheels(h=0.10, two_way=True)
    assert Q.min.at("left cantilever root")[1] == pytest.approx(c.M_support[1], rel=1e-9)
    assert fr.wheels(h=0.10, target="V", two_way=True).max.shear("left cantilever root") == pytest.approx(c.V[1], rel=1e-9)
    assert Q.max.at("left cantilever root")[1] == pytest.approx(0.0, abs=1e-6)


def test_wheel_envelope_against_direct_loading(frame):
    """the influence-line envelope against applying the governing vehicles directly as surface loads (P/a over b on the
    slab, × 1.3 × the lane factor)"""
    fr = frame
    Q = fr.wheels(h=0.10, two_way=True)
    lay = fr.box.layout(0.10)
    for nm in ("top slab, cell 0 middle", "top slab, web 0 face, cell 0", "web 1 top", "bottom slab, web 1 face, cell 0"):
        i = Q.max.index(nm)
        for F, ps in ((Q.max, Q.positions[i][0]), (Q.min, Q.positions[i][1])):
            if not ps:
                continue
            tot = 0
            for x0 in ps:
                for x in (x0, x0 + 1.8):
                    a, _, _ = lay.width(x, 0.2, 0.6)
                    w = max(v for v, _, _ in slab.group_widths(a, [g for g in slab.axle_groups() if g.b1 == 0.6]))
                    tot = tot + fr.surface(w / 0.8, x - 0.4, x + 0.4)
            k = len(ps)
            fac = 1.3 * jtg.transverse_reduction(k)
            assert F.M[i] == pytest.approx(fac * tot.M[i], rel=1e-9) and F.N[i] == pytest.approx(fac * tot.N[i], rel=1e-9)
    assert Q.details["lanes"] == jtg.design_lanes(11.5)
    # the envelope is at least any single vehicle's effect (here: one vehicle centred on the cell, × 1.2)
    i = Q.max.index("top slab, cell 0 middle")
    one = sum(fr.surface(70 / lay.width(x, 0.2, 0.6)[0] / 0.8, x - 0.4, x + 0.4) for x in (-0.9, 0.9))
    assert Q.max.M[i] >= 1.3 * 1.2 * one.M[i] - 1e-9


def test_transverse_prestress_is_self_equilibrated(frame):
    """a tendon in the top slab: no reactions; across the cell the concrete's horizontal forces balance the tendon's
    (N_top + N_bottom + P cos = 0) and the secondary forces balance themselves (N and moment about the cut); the
    cantilever is determinate (no secondary effect) and its primary moment is P times the tendon's lever about the
    check section's centroid"""
    fr = frame
    P = 900.0
    prof = [(-6.2, 0.07), (-2.9, 0.07), (0.0, 0.17), (2.9, 0.07), (6.2, 0.07)]
    p = fr.tendon(prof, force=P)
    assert np.allclose(p.total.reactions, 0.0, atol=1e-6)
    zt, zb = fr.box.z_top, fr.box.z_bottom
    top, bot = "top slab, cell 0 middle", "bottom slab, cell 0 middle"
    slope = 0.10 / 2.9  # the tendon on the element just right of x = 0
    assert p.total.at(top)[0] + p.total.at(bot)[0] == pytest.approx(-P / math.hypot(1, slope), rel=1e-9)
    Nt, Mt = p.secondary.at(top)
    Nb, Mb = p.secondary.at(bot)
    assert Nt + Nb == pytest.approx(0.0, abs=1e-6)
    assert Mt + Mb - Nt * zt - Nb * zb == pytest.approx(0.0, abs=1e-6)
    assert p.secondary.at("left cantilever root") == pytest.approx((0.0, 0.0), abs=1e-6)
    assert p.primary.at("left cantilever root")[1] == pytest.approx(P * (0.55 / 2 - 0.07), rel=1e-9)
    assert p.total.at("left cantilever root")[0] == pytest.approx(-P, rel=1e-9)


def test_linear_temperature_field_causes_no_stress(frame):
    """compatibility: a temperature varying linearly with height (here over the whole depth) is free to develop in the
    frame and gives no restraint forces and no self-equilibrating stresses; the D60 gradient is self-equilibrated
    (no reactions, a cut through the cell carries no resultant)"""
    fr = frame
    H = fr.box.H * 1e3
    lin = fr.temperature(Profile((0.0, H), (20.0, 5.0)))
    scale = abs(fr.temperature(jtg.temperature_gradient(H, surfacing="asphalt", asphalt=100)).stress()).max()
    assert abs(lin.stress()).max() < 2e-3 * scale and abs(lin.reactions).max() < 1e-6
    T = fr.temperature(jtg.temperature_gradient(H, surfacing="asphalt", asphalt=100))
    assert np.allclose(T.reactions, 0.0, atol=1e-6)
    Nt, Mt = T.at("top slab, cell 0 middle")
    Nb, Mb = T.at("bottom slab, cell 0 middle")
    assert Nt + Nb == pytest.approx(0.0, abs=1e-6)
    assert Mt + Mb - Nt * fr.box.z_top - Nb * fr.box.z_bottom == pytest.approx(0.0, abs=1e-6)
    assert T.at("left cantilever root") == pytest.approx((0.0, 0.0), abs=1e-6)
    assert abs(T.extra[T.index("left cantilever root")]).max() > 0.5  # self-equilibrating stresses only (MPa)


def test_shear_flow_against_sectionproperties():
    """thin-walled shear flow of a rectangular single cell (6.0 × 2.5 m, 0.3 m walls) against sectionproperties'
    finite-element shear and torsion stresses at mid-height of a web (vertical shear and Bredt within 1 %),
    and Bredt's J = 4 A0² / ∮ ds/t within 5 % of the warping analysis"""
    pytest.importorskip("sectionproperties")
    from sectionproperties.analysis import Section as SpSection
    from sectionproperties.pre.library import rectangular_hollow_section
    t, B, H = 0.3, 6.0, 2.5
    g = rectangular_hollow_section(d=H, b=B, t=t, r_out=0, n_r=1)
    g.create_mesh(mesh_sizes=0.006)
    s = SpSection(g)
    s.calculate_geometric_properties()
    s.calculate_warping_properties()
    V, T = 3000.0, 2500.0
    r = s.calculate_stress(vy=V, mzz=T).get_stress()[0]
    xy = np.asarray(s.mesh["vertices"])
    cy = s.get_c()[1]
    band = (np.abs(xy[:, 1] - cy) < 0.05) & (xy[:, 0] < t)  # across the left web at the centroid
    tau_V = np.mean(np.asarray(r["sig_zy_vy"])[band]) / 1e3
    tau_T = np.mean(np.abs(np.asarray(r["sig_zy_mzz"])[band])) / 1e3
    box = tr.BoxGeometry(B=B, H=H, webs=(-(B - t) / 2, (B - t) / 2), t_top=t, t_bottom=t, t_web=t, t_root=t, t_tip=t)
    f = tr.shear_flow(box, V, T)
    assert f["web at centroid"]["tau_V"] == pytest.approx(tau_V, rel=0.01)
    assert f["web at centroid"]["tau_T"] == pytest.approx(tau_T, rel=0.01)
    A0 = (B - t) * (H - t)
    assert f["A0"] == pytest.approx(A0)
    J_bredt = 4 * A0 ** 2 / (2 * (B - t + H - t) / t)
    assert J_bredt == pytest.approx(s.get_j(), rel=0.05)
    assert f["I"] == pytest.approx(s.get_ic()[0], rel=0.01)  # the centre-line model lacks the four outer corners
    assert abs(f["closure"]) < 1e-9


def test_shear_flow_with_cantilevers_and_inclined_webs():
    """statics of the open flow: the webs' vertical components carry V (the flow integrated over a web, here from
    the web's ends: τ t averaged × height ≈ V/2 by Simpson's rule on the three points), and a symmetric section has no
    flow on its axis; multi-cell boxes are refused"""
    box = tr.BoxGeometry(B=12.5, H=2.5, webs=(-3.0, 3.0), webs_bottom=(-2.6, 2.6), t_top=0.25, t_bottom=0.22,
                         t_web=0.45, t_root=0.5, t_tip=0.2)
    f = tr.shear_flow(box, V=2000.0)
    assert f["top slab middle"]["tau_V"] == 0.0 and f["bottom slab middle"]["tau_V"] == 0.0
    # at the web's outer face the cantilever's flow is (V/I)(zt − z̄) ∫ t ds over the tapered cantilever (2.775 m)
    Lc = 6.25 - box.faces(1)[1]
    q = 2000.0 / f["I"] * (box.z_top - f["z_centroid"]) * Lc * (0.2 + 0.5) / 2
    assert f["cantilever root"]["tau_V"] == pytest.approx(q / 0.5 / 1e3, rel=1e-6)
    assert abs(f["closure"]) < 1e-9
    with pytest.raises(NotImplementedError):
        tr.shear_flow(tr.BoxGeometry(B=16, H=2.5, webs=(-5, 0, 5), t_top=0.25, t_bottom=0.22, t_web=0.45,
                                     t_root=0.5, t_tip=0.2), V=1000.0)


def test_plate_principal_invariants():
    """σ1 + σ2 = σx + σy, σ1 σ2 = σx σy − τ², the same values for the state rotated by any angle; pure shear ±τ"""
    rng = np.random.default_rng(7)
    for _ in range(20):
        sx, sy, t = rng.normal(size=3) * 5
        p = tr.plate_principal(sx, sy, t)
        s1, s2 = p["sigma_cp"], -p["sigma_tp"]
        assert s1 + s2 == pytest.approx(sx + sy) and s1 * s2 == pytest.approx(sx * sy - t * t)
        th = rng.uniform(0, math.pi)
        c, s = math.cos(th), math.sin(th)
        sxr = sx * c * c + sy * s * s + 2 * t * s * c
        syr = sx * s * s + sy * c * c - 2 * t * s * c
        tr_ = (sy - sx) * s * c + t * (c * c - s * s)
        q = tr.plate_principal(sxr, syr, tr_)
        assert q["sigma_tp"] == pytest.approx(p["sigma_tp"]) and q["sigma_cp"] == pytest.approx(p["sigma_cp"])
        a = math.radians(p["angle"])  # σcp acts along the returned angle
        assert sx * math.cos(a) ** 2 + sy * math.sin(a) ** 2 + 2 * t * math.sin(a) * math.cos(a) == pytest.approx(s1)
    p = tr.plate_principal(0.0, 0.0, 1.5)
    assert p["sigma_tp"] == pytest.approx(1.5) and p["sigma_cp"] == pytest.approx(1.5)


def test_inplane_principal_limits():
    """6.3.1-5 … -8 and 7.1.6-1 on the in-plane states"""
    c = jtg.concrete("C50")
    pts = [(6.0, 1.0, 1.2), (3.0, 0.5, 2.0)]
    chk = tr.inplane_principal(pts, concrete="C50", pc_class="A", precast=False)
    worst = max(tr.plate_principal(*p)["sigma_tp"] for p in pts)
    assert chk.demand == pytest.approx(worst) and chk.resistance == pytest.approx(0.5 * c.ftk)
    assert tr.inplane_principal(pts, concrete="C50", pc_class="full").resistance == pytest.approx(0.4 * c.ftk)
    assert tr.inplane_principal(pts, concrete="C50", pc_class="full", precast=True).resistance == pytest.approx(0.6 * c.ftk)
    assert tr.inplane_principal(pts, concrete="C50", pc_class="B", precast=True).resistance == pytest.approx(0.7 * c.ftk)
    cc = tr.inplane_principal_compression(pts, concrete="C50")
    assert cc.resistance == pytest.approx(0.6 * c.fck) and cc.demand == pytest.approx(
        max(tr.plate_principal(*p)["sigma_cp"] for p in pts))


def _slab_section():
    # one metre of a 250 mm top slab: four 15.2 strands per tendon at 500 mm, i.e. 2 tendons per metre (lumped)
    Ap = 2 * 4 * 140.0
    return jtg.rect(1000, 250, concrete="C50", top=[(7, 16, 45)], bottom=[(7, 16, 45)],
                    tendons=[(Ap, 70, 1150.0, 70 * math.sqrt(2))], sigma_pe=1100.0)


def test_transverse_normal_stresses_match_crack_resistance_without_axial_force():
    """with N = 0 the check is 6.3.1 item 1 exactly as sls.crack_resistance (a separate implementation); an axial
    tension adds N/A0 at both faces, the secondary axial force N_p2 takes N_p2/An off σpc"""
    sec = _slab_section()
    for cls in ("full", "A"):
        a = tr.transverse_normal_stresses(sec, 0.0, -40.0, 0.0, -25.0, pc_class=cls)
        b = jtg.crack_resistance(sec, -40.0, -25.0, pc_class=cls)
        assert a.demand == pytest.approx(b.demand) and a.resistance == pytest.approx(b.resistance)
        assert a.details["edge"] == b.details["edge"] and a.ok == b.ok
    A0 = sec.props("transformed")[0]
    An = sec.props("net")[0]
    c = tr.transverse_normal_stresses(sec, 30.0, -40.0, 0.0, -25.0, pc_class="A", N_p2=20.0)
    d = tr.transverse_normal_stresses(sec, 0.0, -40.0, 0.0, -25.0, pc_class="A")
    for e in ("top", "bottom"):
        assert c.details[e]["sigma_st"] - d.details[e]["sigma_st"] == pytest.approx(30e3 / A0)
        assert d.details[e]["sigma_pc"] - c.details[e]["sigma_pc"] == pytest.approx(20e3 / An)
    with pytest.raises(ValueError):
        tr.transverse_normal_stresses(jtg.rect(1000, 250, bottom=[(7, 16, 45)]), 0, 1, 0, 1, pc_class="A")


def test_transverse_crack_width_routes():
    """reinforced concrete slab: a tensile N is eccentric tension (sls.crack_width with N < 0); a compressive one is
    left out, the slab in flexure with C3 = 1.15 (slab=True), unless axial="all" """
    sec = jtg.rect(1000, 250, concrete="C40", top=[(7, 16, 40)], bottom=[(7, 16, 40)])
    t = tr.transverse_crack_width(sec, 40.0, 50.0, 30.0, 35.0)
    assert t.demand == pytest.approx(jtg.crack_width(sec, 50.0, 35.0, N=-40.0, Nl=-30.0).demand)
    c = tr.transverse_crack_width(sec, -40.0, 50.0, -30.0, 35.0)
    assert c.demand == pytest.approx(jtg.crack_width(sec, 50.0, 35.0, slab=True).demand)
    assert tr.transverse_crack_width(sec, -40.0, 50.0, -30.0, 35.0, axial="all").demand == pytest.approx(
        jtg.crack_width(sec, 50.0, 35.0, N=40.0, Nl=30.0).demand)
    # class B: the tensile force goes in too (sls.crack_width with N, its program convention), and widens the crack
    b = jtg.rect(1000, 250, concrete="C50", top=[(7, 16, 45)], bottom=[(7, 16, 45)],
                 tendons=[(560.0, 170, 1080.0, 55)], sigma_pe=1050.0)
    kw = dict(pc_class="B", tendon_bundles=[(4 / 3, 3, 15.2)])  # BM15-3 every 0.75 m
    wb = tr.transverse_crack_width(b, 80.0, 95.0, 50.0, 60.0, **kw)
    assert wb.demand == pytest.approx(jtg.crack_width(b, 95.0, 60.0, N=-80.0, Nl=-50.0, slab=True, **kw).demand)
    assert wb.demand > tr.transverse_crack_width(b, 80.0, 95.0, 50.0, 60.0, axial="none", **kw).demand


def test_multi_cell_frame_and_geometry_checks():
    """a two-cell box (three webs, one pin and two rollers): statics, symmetry, control sections per cell"""
    box = tr.BoxGeometry(B=16.0, H=2.5, webs=(-5.0, 0.0, 5.0), t_top=0.25, t_bottom=0.22, t_web=0.45, t_root=0.5,
                         t_tip=0.2, haunch_top=(0.8, 0.25))
    fr = tr.TransverseFrame(box, concrete="C50")
    G = fr.self_weight()
    s = fr.surface(2.4)
    assert s.reactions.sum() == pytest.approx(2.4 * 16.0)
    assert s.reactions[0] == pytest.approx(s.reactions[2], rel=1e-9)
    assert G.at("top slab, cell 0 middle")[1] == pytest.approx(G.at("top slab, cell 1 middle")[1], rel=1e-8)
    assert G.at("web 1 middle")[1] == pytest.approx(0.0, abs=1e-8)  # the middle web is not bent under symmetry
    with pytest.raises(ValueError):
        tr.BoxGeometry(B=10.0, H=2.5, webs=(-5.0, 5.0), t_top=0.25, t_bottom=0.22, t_web=0.45, t_root=0.5, t_tip=0.2)


def test_box_detailing():
    """9.3.2 items 1–3: cantilever tip, slab thickness ≥ max(ln/30, 200 mm), web ≥ 160 mm and ≤ 15 (20) widths tall"""
    assert not tr.box_detailing(tr.BoxGeometry(**BOX), transverse_prestress=True).warnings
    bad = tr.BoxGeometry(B=16.0, H=3.4, webs=(-4.5, 4.5), t_top=0.25, t_bottom=0.18, t_web=0.15, t_root=0.5,
                         t_tip=0.12)
    chk = tr.box_detailing(bad, transverse_prestress=True)
    # tip 120 < 140; top 250 < 8.85/30 = 295; bottom 180 < 295; each web 150 < 160 and 2.97 m > 15 × 0.15
    assert len(chk.warnings) == 7 and chk.ok
    assert len(tr.box_detailing(bad, vertical_prestress=True).warnings) == 4  # tip 120 ≥ 100, 2.97 m ≤ 20 × 0.15


def _shoelace(p):
    x, y = np.array(p).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_self_weight_equals_the_section_area(frame):
    """the joints are counted once: the reactions of the self-weight are γc × the cross-section's area, found here by
    the shoelace formula on the outline less the cell (haunches as chamfers)"""
    H, tt, tb = 2.5, 0.25, 0.22
    outer = [(-6.25, H), (6.25, H), (6.25, H - 0.2), (3.15, H - 0.55), (3.15, 0.0), (-3.15, 0.0), (-3.15, H - 0.55),
             (-6.25, H - 0.2)]
    zt, zb = H - tt, tb
    hole = [(-2.65, zt - 0.3), (-1.75, zt), (1.75, zt), (2.65, zt - 0.3), (2.65, zb + 0.2), (2.15, zb), (-2.15, zb),
            (-2.65, zb + 0.2)]
    A = _shoelace(outer) - _shoelace(hole)
    G = frame.self_weight()
    assert G.reactions.sum() == pytest.approx(26.0 * A, rel=1e-9)


def _hand_width(x):
    """distribution width of one middle / rear wheel (a1 = 0.2, b1 = 0.6, h = 0.1) on the BOX fixture, written out from
    4.2.3 and 4.2.5 with the program's conventions (webs ±2.9 m, faces ±2.65 / ±3.15, t = 0.25, l = 5.55 m)"""
    lc = (-3.15 - x + 0.4) if x < -2.9 else (x - 3.15 + 0.4 if x > 2.9 else 0.0)
    if lc > 0:
        return 0.4 + 2 * lc
    if abs(x) >= 2.65:
        return 0.4 + 0.25
    xn = max(min(x - 0.4 + 2.65, 2.65 - x - 0.4), 0.0)
    return min(0.4 + 0.25 + 2 * xn, max(0.4 + 5.55 / 3, 2 * 5.55 / 3))


def test_wheel_envelope_by_brute_force(frame):
    """the envelope against an independent route: widths written out from the clauses (not slab.DeckLayout), the
    governing load per metre max(70/a, 140/(a + 1.4) when a > 1.4), the frame loaded directly at every wheel position,
    and every arrangement of one to three vehicles searched with prefix and suffix maxima (not the dynamic programme);
    one-way traffic, three lanes, table 4.3.1-5"""
    fr = frame
    Q = fr.wheels(h=0.10, two_way=False, dx=0.05)
    assert Q.details["lanes"] == 3
    xs = np.round(np.arange(-5.25, 5.25 + 1e-9, 0.05), 10)
    resp = []
    for x in xs:
        a = _hand_width(x)
        w = max(70 / a, 140 / (a + 1.4) if a > 1.4 else 0.0)
        resp.append(fr.surface(1.3 * w / 0.8, x - 0.4, x + 0.4).M)
    resp = np.array(resp).T  # sections × positions
    k, g = 36, 62  # 1.8 m track, 3.1 m between vehicles' left wheels
    n = len(xs) - k
    for i, sec in enumerate(fr.sections):
        v = resp[i, :n] + resp[i, k:k + n]
        for sense, F in ((1, Q.max), (-1, Q.min)):
            u = sense * v
            best = max(0.0, 1.2 * u.max())
            pre = np.maximum.accumulate(u)
            suf = np.maximum.accumulate(u[::-1])[::-1]
            pairs = [u[j] + pre[j - g] for j in range(g, n)]
            best = max(best, 1.0 * max(pairs))
            trip = [u[j] + pre[j - g] + suf[j + g] for j in range(g, n - g)]
            best = max(best, 0.78 * max(trip))
            assert sense * F.M[i] == pytest.approx(best, rel=1e-9, abs=1e-9), sec.name


def test_long_cantilever_takes_the_sawko_formula():
    """lc > 2.5 m is outside 4.2.5: the commentary's Sawko formula (4-5) is taken as well, the larger governing. A 4 m
    cantilever, the outer wheel 0.5 m from the curb at s = 3.0 m (lc = 3.4 m): by hand, A' = 1.30 (lc/a0 = 0.75), the
    wheels of all five axles (15, 60, 60, 70, 70 kN at 0, 3, 4.4, 11.4, 12.8 m) superposed along the root, against
    4.2.5 with the overlap (a = 0.4 + 2 × 3.4 + 1.4); the inner wheel at s = 1.2 m by 4.2.5 (lc = 1.6 m). The strip
    formulas and the frame agree."""
    A = 1.30
    y = np.linspace(0, 12.8, 12801)
    m_s = max(sum(P / math.pi * A / np.cosh(A * (y - yk) / 3.0) for P, yk in
                  zip((15, 60, 60, 70, 70), (0, 3, 4.4, 11.4, 12.8))))
    m_425 = 140 / (0.4 + 2 * 3.4 + 1.4) * 3.0
    assert m_s > m_425
    inner = 140 / (0.4 + 2 * 1.6 + 1.4) * 1.2
    c = slab.cantilever_slab(4.0, 0.6, 0.2, h=0.10, curb=3.5, lanes=1)
    assert c.M_support[1] == pytest.approx(-1.3 * 1.2 * (m_s + inner), rel=1e-6)
    assert any("Sawko" in n for n in c.notes)
    box = tr.BoxGeometry(B=2 * (2.9 + 0.25 + 4.0), H=2.5, webs=(-2.9, 2.9), t_top=0.25, t_bottom=0.22, t_web=0.5,
                         t_root=0.6, t_tip=0.2, carriageway=(-6.65, 6.65))
    fr = tr.TransverseFrame(box, concrete="C50")
    Q = fr.wheels(h=0.10, lanes=1)
    assert Q.min.at("left cantilever root")[1] == pytest.approx(c.M_support[1], rel=1e-6)
    assert any("Sawko" in n for n in Q.notes) and any("distortion" in n for n in Q.notes)
    with pytest.raises(ValueError, match="two_way"):
        fr.wheels(h=0.10)
