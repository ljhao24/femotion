"""Checks for the calculation-report tests (tests/test_report.py): one case per branch of every template of
femotion.codes.jtg.report, including the cases of the independent review (shared/refs/codes/REPORT_REVIEW.md: F, S, C,
T, Q, W, K, P, V, D, X, L, U, B, J), and a catalogue of the other kinds of check of the library."""
import numpy as np

from femotion.codes import jtg
from femotion.codes.jtg import detailing as dt
from femotion.codes.jtg import devices as dv
from femotion.codes.jtg import pier_detailing as pd
from femotion.codes.jtg import regions as rg
from femotion.codes.jtg import slab as sl
from femotion.codes.jtg import tendon_detailing as td
from femotion.codes.jtg import transverse as tr

S = jtg.tendon("strand", 1860)


def _pc(sigma_pe=1080.0):
    return jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                   top=[(16, 12, 45)], tendons=[(12 * 139, 1650, 1200, 90)] * 2 + [(9 * 139, 1520, 1200, 80)],
                   tendon=S, sigma_pe=sigma_pe, sigma_l6=80)


def flexure():
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 28, 95)], top=[(14, 25, 60)])
    box = jtg.box(3000, 2000, 250, 220, 800, b_top=6000, concrete="C50", bottom=[(20, 16, 50)], top=[(30, 16, 50)],
                  tendons=[(4 * 19 * 139, 1850, 1200, 100)], tendon=S, sigma_pe=1100,
                  external=[(2 * 19 * 139, 1700, 1000)])
    return [
        ("F1 rect single", jtg.flexure(jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 45)]), 250.0)),
        ("F1b rect x≥2a's", jtg.flexure(jtg.rect(300, 700, concrete="C35", bottom=[(6, 28, 60)], top=[(3, 20, 40)]),
                                        600.0)),
        ("F2 T web", jtg.flexure(jtg.tee(300, 1200, 900, 100, concrete="C30", bottom=[(8, 28, 100)],
                                          top=[(4, 16, 40)]), 1200.0)),
        ("F3 T hogging", jtg.flexure(t, -1300.0)),
        ("F4 As<A's", jtg.flexure(t, 300.0)),
        ("F4b T x<2a'", jtg.flexure(jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)],
                                             top=[(8, 20, 50)]), 1577.6)),
        ("F5 PC I", jtg.flexure(_pc(), 9000.0, pc_class="A")),
        ("F6 A'p in tension", jtg.flexure(jtg.rect(600, 1200, concrete="C50", bottom=[(6, 20, 50)], top=[(6, 16, 45)],
                                                    tendons=[(12 * 139, 1100, 1150, 90), (4 * 139, 120, 1150, 70)],
                                                    tendon=S, sigma_pe=1050), 3500.0)),
        ("F7 box external", jtg.flexure(box, 30000.0, pc_class="A")),
        ("F11 A'p compressed x<2a'", jtg.flexure(jtg.rect(600, 1200, concrete="C50", bottom=[(4, 16, 50)],
                                                           top=[(6, 16, 45)], tendons=[(6 * 139, 1100, 1150, 90),
                                                                                       (4 * 139, 100, 200, 70)],
                                                           tendon=S, sigma_pe=1050), 1000.0)),
        ("F12 A'p compressed x≥2a'", jtg.flexure(jtg.rect(400, 1200, concrete="C50", bottom=[(4, 20, 50)],
                                                           top=[(4, 16, 45)], tendons=[(12 * 139, 1100, 1150, 90),
                                                                                       (2 * 139, 100, 200, 70)],
                                                           tendon=S, sigma_pe=1050), 2000.0)),
        ("F14 over-reinforced", jtg.flexure(jtg.rect(250, 500, concrete="C30", bottom=[(6, 32, 60)]), 300.0)),
        ("F14b 5.2.7", jtg.flexure(jtg.rect(250, 500, concrete="C30", bottom=[(6, 32, 60)]), 200.0,
                                   uls_steel_only=True)),
        ("F15 γ0 1.1", jtg.flexure(jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 45)]), 250.0, gamma0=1.1)),
    ]


def _limit(t):
    """the 5.2.11 limit at an end support from jtg.shear_search"""
    x = np.linspace(0, 10, 41)
    s = jtg.shear_search(x, 600 * (1 - x / 5), 600 * x * (1 - x / 10), [(0, 10, t)], [(0, 10, (2, 10, 150))],
                         supports=[(0.0, "end"), (10.0, "end")])
    return s.limits[0]


def shear():
    t = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(4, 22, 60)], top=[(8, 20, 50)])
    return [
        ("S0 P capped", jtg.shear(jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)],
                                           top=[(8, 20, 50)]), 514.4, (2, 12, 100))),
        ("S1 bent bars", jtg.shear(t, 500.0, (2, 10, 150), bent=[(2 * 490.9, 45), (2 * 380.1, 45)])),
        ("S2 PC, 5.2.11 not met", jtg.shear(_pc(), 1500.0, (2, 12, 150), pc_class="A", bent_tendons=[(9 * 139, 8.0)],
                                             vertical=(2 * 804, 500, "bar930"))),
        ("S3 slab", jtg.shear(jtg.rect(1000, 250, bottom=[(7, 12, 30)]), 100.0, None, slab=True)),
        ("S4 two zones", jtg.shear(t, 400.0, (2, 10, 200), rho_sv=0.0030)),
        ("S5 hogging", jtg.shear(t, 400.0, (4, 10, 150), hogging=True, alpha1=0.9)),
        ("S7 HRB500 stirrups", jtg.shear(t, 400.0, (2, 10, 150), stirrup_steel="HRB500")),
        ("S8 section limit", _limit(t)),
    ]


def compression():
    r = jtg.rect(500, 800, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 25, 50)])
    t = jtg.tee(400, 1200, 1200, 200, concrete="C40", bottom=[(6, 25, 50)], top=[(6, 25, 50)])
    col = jtg.circle(1500, 28, 28, 60, concrete="C35")
    s14 = jtg.rect(400, 600, concrete="C30", bottom=[(4, 28, 45)], top=[(2, 16, 50)])
    return [
        ("C1 large", jtg.eccentric(r, 800.0, 600.0, l0=6000)),
        ("C2 small", jtg.eccentric(r, 1500.0, 450.0, l0=6000)),
        ("C3 x>h", jtg.eccentric(r, 6500.0, 100.0, l0=6000)),
        ("C4 circle", jtg.eccentric(col, 9800.0, 3100.0, l0=18000)),
        ("C5 circle α>0.625", jtg.eccentric(col, 30000.0, 600.0, l0=8000)),
        ("C6 T large x>hf", jtg.eccentric(t, 5000.0, 2200.0, l0=8000)),
        ("C7 T small", jtg.eccentric(t, 12000.0, 1500.0, l0=8000)),
        ("C10 5.3.6", jtg.eccentric(r, 300.0, 600.0, l0=4000)),
        ("C11 short", jtg.eccentric(r, 800.0, 10.0)),
        ("C3b x>h", jtg.eccentric(jtg.rect(400, 800, concrete="C25", bottom=[(8, 25, 50)], top=[(8, 25, 50)]),
                                  10000.0, 1.0)),
        ("C12 ζ capped", jtg.eccentric(jtg.rect(400, 400, concrete="C30", bottom=[(3, 22, 45)], top=[(3, 22, 45)]),
                                       900.0, 200.0, l0=8000)),
        ("C14 reverse", jtg.eccentric(s14, 4000, 200)),
        ("C8 axial ρ>3%", jtg.axial(jtg.rect(400, 400, concrete="C30", bottom=[(4, 32, 50)], top=[(4, 32, 50)]), 3000,
                                    5000)),
        ("C9 axial", jtg.axial(jtg.rect(500, 500, concrete="C30", bottom=[(3, 25, 50)], top=[(3, 25, 50)]), 3000,
                               5000)),
    ]


def tension_torsion():
    r = jtg.rect(400, 600, bottom=[(4, 22, 45)], top=[(4, 22, 45)])
    tor = jtg.rect(400, 800, concrete="C30", bottom=[(4, 22, 50)], top=[(2, 22, 50)])
    Wt = 400 ** 2 / 6 * (3 * 800 - 400)
    Vd = 2 * 95.0 * 1e6 * 400 * 750 / Wt / 1e3  # βt = 0.75
    return [
        ("T1 axial HRB500", jtg.tension(jtg.rect(400, 600, steel="HRB500", bottom=[(4, 22, 45)], top=[(4, 22, 45)]),
                                        800, 0)),
        ("T2 small ecc", jtg.tension(r, 800, 80)),
        ("T3 large ecc x<2a's", jtg.tension(r, 300, 200)),
        ("T4 large ecc", jtg.tension(jtg.rect(400, 600, bottom=[(6, 25, 45)], top=[(2, 12, 40)]), 200, 300)),
        ("Q1 pure torsion", jtg.torsion(tor, Td=95.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))),
        ("Q2 torsion with shear", jtg.torsion(tor, Td=95.0, Vd=300.0, stirrups=(2, 12, 100), Ast=6 * 491,
                                              core=(320, 720))),
        ("Q3 ζ capped", jtg.torsion(tor, Td=60.0, stirrups=(2, 12, 100), Ast=10 * 491, core=(320, 720))),
        ("Q4 βt = 0.75", jtg.torsion(tor, Td=95.0, Vd=Vd, stirrups=(4, 14, 100), Ast=10 * 491, core=(320, 720))),
    ]


def serviceability():
    r = jtg.rect(500, 800, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 25, 50)])
    pcB = jtg.tee(200, 1300, 1600, 150, concrete="C50", bottom=[(6, 20, 50)], top=[(8, 12, 40)],
                  tendons=[(9 * 139, 1180, 1250, 80)], tendon=S, sigma_pe=1050)
    pc = _pc()
    pch = jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                  top=[(16, 12, 45)], tendons=[(12 * 139, 200, 1200, 90)] * 2, tendon=S, sigma_pe=1080)
    th = jtg.thermal_stresses(pc, jtg.temperature_gradient(1800.0), M_sec=300.0)
    span = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    pier = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 28, 95)], top=[(14, 25, 60)])
    bar = jtg.rect(300, 700, concrete="C50", bottom=[(4, 16, 50)], tendons=[(4 * 804, 600, 650, 50)], tendon="bar930",
                   sigma_pe=600)
    return [
        ("W0 flexural", jtg.crack_width(span, 769.3, 567.7, env="I")),
        ("W0b hogging", jtg.crack_width(pier, -837.2, -707.7, env="I")),
        ("W1 compression", jtg.crack_width(r, 400.0, 300.0, N=600.0, l0=6000, env="I", member="compression")),
        ("W2 axial tension", jtg.crack_width(jtg.rect(400, 400, bottom=[(3, 20, 45)], top=[(3, 20, 45)]), 0.0, 0.0,
                                             N=-300.0, Nl=-200.0, env="I")),
        ("W3 slab", jtg.crack_width(jtg.rect(1000, 250, bottom=[(7, 12, 30)]), 40.0, 30.0, slab=True, env="II")),
        ("W4 class B", jtg.crack_width(pcB, 2600.0, 1800.0, pc_class="B", env="I", tendon_bundles=[(1, 9, 15.2)])),
        ("W5 HPB300 c>50", jtg.crack_width(jtg.rect(300, 600, steel="HPB300", bottom=[(4, 20, 70)]), 120.0, 90.0,
                                           env="I")),
        ("K1 class A", jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="A")),
        ("K1b 6.3.1-4 not met", jtg.crack_resistance(pc, 500.0, 300.0, pc_class="A")),
        ("K2 full", jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="full")),
        ("K3 class B", jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="B", M_G=2000.0)),
        ("K4 hogging", jtg.crack_resistance(pch, -3000.0, -2000.0, pc_class="A")),
        ("K5 temperature", jtg.crack_resistance(pc, 6000.0, 4500.0, pc_class="A", temperature=th)),
        ("P1 inclined", jtg.inclined_crack_resistance(pc, 4000.0, 600.0, pc_class="A")),
        ("P2 principal compression", jtg.principal_compression(pc, 6000.0, 800.0)),
        ("V1 service A", jtg.service_stresses(pc, 6000.0, pc_class="A")),
        ("V2 service A, temperature", jtg.service_stresses(pc, 6000.0, pc_class="A", temperature=th)),
        ("V3 service B", jtg.service_stresses(pc.replace(tendons=pc.tendons[:1], sigma_pe=np.array([1080.0])), 7000.0,
                                              pc_class="B")),
        ("V4 threaded bar", jtg.service_stresses(bar, 300.0, pc_class="A")),
        ("V5 tendon stress not met", jtg.service_stresses(_pc(1250.0), 3000.0, pc_class="A")),
        ("D1 deflection", jtg.deflection_check(12.486, 16000)),
        ("D2 cantilever", jtg.deflection_check(8.0, 3000.0, cantilever=True)),
        ("X1 transient PC", jtg.transient_pc(pc, 300.0, fcu=45)),
        ("X2 transient RC", jtg.transient_rc(jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)],
                                                      top=[(2, 16, 40)]), 120.0, 150.0, fcu=25)),
    ]


def local_devices():
    b = jtg.Laminated(la=300, lb=400, t1=8, n=3, ts=3)
    thin = jtg.Laminated(la=400, lb=500, t1=5, n=2, ts=3)
    out = [
        ("L1 mesh", jtg.local_bearing("C50", 2000, Al=2e5, Aln=1.8e5, Ab=1.8e6, mesh=(5, 113, 280, 5, 113, 280, 50),
                                      fcu=45, layers=5)),
        ("L2 none", jtg.local_bearing("C50", 2000, Al=2e5, Aln=1.8e5, Ab=1.8e6)),
        ("L3 spiral Acor>Ab", jtg.local_bearing("C40", 1500, Al=1.0e5, Ab=1.5e5, spiral=(113, 500, 50), layers=6)),
        ("L4 size governs", jtg.local_bearing("C40", 2500, Al=2.0e5, Ab=4.0e5, mesh=(8, 201, 400, 8, 201, 400, 40),
                                              layers=6)),
        ("U1 punching", sl.punching(400, Um=3220, h0=205, h=250, concrete="C40")),
        ("U2 punching stirrups", sl.punching(900, Um=3220, h0=205, h=250, concrete="C40", stirrups=1200,
                                             Um_outer=4500)),
        ("U3 punching bent", sl.punching(600, Um=3220, h0=205, h=250, concrete="C40", bent=(1500, 45), Um_outer=4500,
                                         sigma_pc=2.0)),
        ("U4 footing", sl.footing_punching(ps=0.25, A=2.0e6, bt=800, h0=700, h=750, concrete="C30")),
    ]
    out += [("B " + c.name, c) for c in jtg.laminated_bearing(b, 800.0, dl=12.0, Fbk=20.0, theta=0.003)]
    out += [("B thin " + c.name, c) for c in jtg.laminated_bearing(thin, 600.0, dl=5.0) if "stability" in c.name]
    out += [("B " + c.name, c) for c in jtg.laminated_sliding(b, RGk=500.0, RQk=300.0, dl=12.0, Fbk=20.0)]
    out += [("B " + c.name, c) for c in jtg.ptfe_sliding(b, mu_f=0.03, RGk=500.0, RQk=300.0)]
    mv = jtg.movements(30000.0, T_set=(10, 20), T_max=40, T_min=-10, brake=5.0, eps_cs=0.0002, phi=1.5,
                       sigma_pc=5.0, concrete="C50")
    out.append(("J joint", jtg.expansion_joint(mv, 80.0)))
    return out


def templated():
    """every case of the templates"""
    return flexure() + shear() + compression() + tension_torsion() + serviceability() + local_devices()


def catalogue():
    """checks of the kinds not met in the examples nor in templated()"""
    t = jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(4, 32, 70)], top=[(8, 16, 40)])
    pc = jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                 top=[(16, 12, 45)], tendons=[(12 * 139, 1650, 1200, 90)] * 2 + [(9 * 139, 1520, 1200, 80)],
                 tendon=S, sigma_pe=1080, sigma_l6=80)
    r = jtg.rect(500, 800, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    sx, sy = jtg.rect_biaxial(600, 900, layout=(4, 5, 25, 60), concrete="C40", steel="HRB500")
    ft = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 25, 60)], top=[(8, 12, 40)])
    pre = jtg.rect(1000, 250, pretensioned=True, tendons=[(10 * 140, 200, 1200)], tendon=S, bottom=[(8, 12, 30)])
    x = np.linspace(0, 3000, 7)
    out = [
        jtg.flexure(t, 9000.0), jtg.shear(col, 500.0, (2, 12, 150)), jtg.eccentric(r, 1200, -300, l0=4000),
        jtg.axial_spiral(jtg.circle(600, 8, 22, 50, concrete="C30"), 3000, 3000, spiral=(10, 50)),
        jtg.biaxial_tension(sx, sy, 800, 120, 90), jtg.biaxial(sx, sy, 3000, 300, 200, l0x=6000, l0y=6000),
        jtg.biaxial(jtg.circle(1500, 28, 28, 60, concrete="C35"), None, 9800.0, 2500.0, 1800.0, l0x=18000),
        jtg.torsion_flange(ft, Td=60, which="top", stirrups=(2, 10, 150), Ast=8 * 113, core=(800, 90)),
        jtg.crack_resistance(pre, 20.0, 15.0, pc_class="A", transfer=jtg.Transfer(x=300.0, ltr=600.0)),
        jtg.transient_shear(x, 1.8 * (1 - x / 3000), fcu=30, b=250, steel="HRB400", stirrups=(2, 8, 200)),
        jtg.interface_shear_slab(300, b=1000, h0=400, gamma0=1.0, s=200, Asv=150),
        rg.pile_cap_punching_corner(5000, bx=900, by=900, ax=600, ay=750, h0=1500, concrete="C30"),
        rg.pile_cap_punching_edge(5000, bp=800, bx=900, ax=600, h0=1500, b=6000, concrete="C30"),
        rg.pile_cap_shear(9000, As=40 * 616, bs=6000, h0=1800, a=600, concrete="C30"),
        tr.inplane_principal([(6.0, 1.0, 1.2), (3.0, 0.5, 2.0)], concrete="C50", pc_class="A"),
        tr.inplane_principal_compression([(6.0, 1.0, 1.2), (3.0, 0.5, 2.0)], concrete="C50"),
        jtg.composite_crack_width(jtg.composite(jtg.tee(200, 1200, 1600, 160, concrete="C40", bottom=[(8, 25, 60)]),
                                                b=2000, t=180, concrete="C40", top=[(12, 12, 50)]),
                                  M1Gk=500, M2s=600, M2l=350),
        dt.splice_detailing("weld", 20, length=99, splice_area=300, joint_spacing=700, share=0.51, clear=24),
        dt.splice_detailing("mechanical", 20, cover=25, cmin=30),
        dt.curved_soffit(6000, 28, (12, 100), r=20000, steel="HRB400", stirrup_steel="HPB300"),
        dt.cutoff_detailing(28, 1000), dt.support_detailing(12, 2),
        dt.bent_bar_detailing(25, 900, angle=45, start=450, anchorage=500),
        dt.bottom_slab_detailing(250, longitudinal=[(12, 200), (12, 200)], transverse=[(12, 200), (12, 200)]),
        dt.beam_dimensions(1600, flange=159), dt.diaphragm_detailing("T", spacing=10001, rigid=True),
        dt.composite_beam_detailing(topping=149, roughness=5.9, stirrups=(10, 99)),
        dt.two_way_slab_detailing(6000, 4000, 200, middle=(12, 125), edge=(12, 251)),
        dt.skew_slab_detailing(15.1, main="axis"),
        dt.composite_slab_detailing(roughness=5, connectors=(12, 119, 119, 501)),
        dt.hinged_slab_detailing(600, groove=399, topping=79),
        dt.welded_frame_detailing(5, 28, weld=140, short_weld=70),
        dv.bearing_detailing(rubber="CR", site_T=(-26, 40)), dv.lifting_loop(20.0, 20, n=4),
        dv.culvert_detailing(1000, layers=1), pd.column_detailing(col), td.girder_detailing(pc, stirrup_d=10),
        td.duct_detailing(pc), jtg.Check.report("a rule", "JTG 3362-2018 9.1.1"),
        tr.box_detailing(tr.BoxGeometry(B=12.5, H=2.5, webs=(-2.9, 2.9), t_top=0.25, t_bottom=0.22, t_web=0.5,
                                        t_root=0.55, t_tip=0.2, haunch_top=(0.9, 0.3), haunch_bottom=(0.5, 0.2),
                                        carriageway=(-5.75, 5.75)), transverse_prestress=True),
    ]
    m_ = jtg.StrutTie("C35")
    A = m_.node(0, 0, support="xz", bearing=400 * 500)
    B = m_.node(3000, 0, support="z", bearing=400 * 500)
    C = m_.node(1500, 1200, load=(0, -5000))
    m_.strut(A, C, area=600 * 500, ends=(None, 400 * 500))
    m_.strut(B, C, area=600 * 500)
    m_.tie(A, B, As=8000)
    m_.solve()
    out += m_.check(mesh=(0.002, 0.004, 350))
    return out


def all_checks():
    return [c for _, c in templated()] + catalogue()

