"""Check.section and Check.hogging: every check made on a section keeps that section (the one passed in, never the copy
turned over for a hogging moment) and whether it turned it over, so that the check can be drawn
(codes.jtg.figures.section_figure) and shows itself in Jupyter (Check._repr_html_)."""
import math
import re

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import detailing as dt
from femotion.codes.jtg import pier_detailing as pd
from femotion.codes.jtg import regions as rg
from femotion.codes.jtg import tendon_detailing as td
from femotion.codes.jtg import transverse as tr
from femotion.codes.jtg.figures import section_figure

STRAND = jtg.tendon("strand", 1860)


def _tee():
    return jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(4, 32, 70)], top=[(8, 16, 40)])


def _pc():
    return jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                   top=[(16, 12, 45)], tendons=[(12 * 139, 1650, 1200, 90)] * 2 + [(9 * 139, 1520, 1200, 80)],
                   tendon=STRAND, sigma_pe=1080, sigma_l6=80)


def _composite():
    pre = jtg.tee(200, 1600, 1600, 160, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1450, 1150, 90)],
                  sigma_pe=1050)
    return jtg.composite(pre, b=2200, t=180, concrete="C40", top=[(14, 12, 50)])


def _has(chk, sec, hogging=False):
    assert chk.section is sec, chk.name
    assert chk.hogging is hogging, chk.name


def test_ultimate_limit_state_checks_keep_the_section():
    t, r = _tee(), jtg.rect(500, 800, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    _has(jtg.flexure(t, 1500.0), t)
    _has(jtg.flexure(t, -600.0), t, True)  # the section as given, not the flipped copy the check used
    _has(jtg.shear(t, 400.0, (2, 10, 150)), t)
    _has(jtg.shear(t, 400.0, (2, 10, 150), hogging=True), t, True)
    slab = jtg.rect(1000, 250, bottom=[(7, 12, 30)])
    _has(jtg.shear(slab, 100.0, None, slab=True), slab)
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    _has(jtg.shear(col, 500.0, (2, 12, 150), hogging=True), col, True)  # through shear_circle
    _has(jtg.shear_circle(col, 500.0, (2, 12, 150)), col)
    _has(jtg.axial(r, 3000.0, 4000.0), r)
    _has(jtg.eccentric(r, 1200, 300, l0=4000), r)
    _has(jtg.eccentric(r, 1200, -300, l0=4000), r, True)
    _has(jtg.eccentric(col, 9800, 3100, l0=18000), col)
    sp = jtg.circle(600, 8, 22, 50, concrete="C30")
    _has(jtg.axial_spiral(sp, 3000, 3000, spiral=(10, 50)), sp)
    s = jtg.rect(400, 600, bottom=[(4, 22, 45)], top=[(4, 22, 45)])
    _has(jtg.tension(s, 800, 80), s)
    _has(jtg.tension(s, 800, -80), s, True)
    sx, sy = jtg.rect_biaxial(600, 900, layout=(4, 5, 25, 60), concrete="C40", steel="HRB500")
    _has(jtg.biaxial_tension(sx, sy, 800, 120, 90), sx)
    _has(jtg.biaxial(sx, sy, 3000, 300, 200, l0x=6000, l0y=6000), sx)
    _has(jtg.biaxial(col, None, 9800, 2000, 1500, l0x=18000), col)
    tor = jtg.rect(400, 800, concrete="C30", bottom=[(4, 22, 50)], top=[(2, 22, 50)])
    _has(jtg.torsion(tor, Td=95.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720)), tor)
    _has(jtg.torsion(tor, Td=95.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720), hogging=True), tor, True)
    ft = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 25, 60)], top=[(8, 12, 40)])
    _has(jtg.torsion_flange(ft, Td=60, which="top", stirrups=(2, 10, 150), Ast=8 * 113, core=(800, 90)), ft)
    cap = jtg.rect(1600, 1600, concrete="C30", bottom=[(20, 28, 70)], top=[(10, 20, 50)])
    _has(rg.cap_flexure(cap, 5000, l=6000), cap)
    _has(rg.cap_flexure(cap, -2000, l=6000), cap, True)
    _has(rg.cap_shear(cap, 4000, 6000, (4, 12, 100), alpha1=0.9, hogging=True), cap, True)


def test_shear_search_limits_and_sections():
    x = np.linspace(0, 12, 49)
    V = 120 * (6 - x)
    M = 60 * x * (12 - x)
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    res = jtg.shear_search(x, np.abs(V), M, sec, (2, 10, 150), supports=[(0.0, "end"), (12.0, "end")])
    assert res.limits and all(c.section is sec for c in res.limits)
    assert all(s.check.section is sec for s in res.sections)


def test_serviceability_checks_keep_the_section():
    pc = _pc()
    _has(jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="A"), pc)
    _has(jtg.crack_resistance(pc, -400.0, -300.0, pc_class="A"), pc, True)
    pre = jtg.rect(1000, 250, pretensioned=True, tendons=[(10 * 140, 200, 1200)], tendon=STRAND, bottom=[(8, 12, 30)])
    tf = jtg.crack_resistance(pre, 20.0, 15.0, pc_class="A", transfer=jtg.Transfer(x=300.0, ltr=600.0))
    _has(tf, pre)  # the section given, not the one with the reduced tendon stresses
    _has(jtg.inclined_crack_resistance(pc, 4000.0, 600.0, pc_class="A"), pc)
    _has(jtg.principal_compression(pc, 6000.0, 800.0), pc)
    _has(jtg.service_stresses(pc, 6000.0, pc_class="A"), pc)
    weak = pc.replace(tendons=pc.tendons[:1], sigma_pe=np.array([1080.0]))
    _has(jtg.service_stresses(weak, 7000.0, pc_class="B"), weak)
    t = _tee()
    _has(jtg.crack_width(t, 1100.0, 900.0, env="I"), t)
    _has(jtg.crack_width(t, -500.0, -400.0, env="I"), t, True)
    r = jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)], top=[(2, 16, 40)])
    _has(jtg.transient_rc(r, 120.0, 150.0, fcu=25), r)
    _has(jtg.transient_pc(pc, 300.0, fcu=45), pc)
    slab = jtg.rect(1000, 250, concrete="C50", top=[(7, 16, 45)], bottom=[(7, 16, 45)],
                    tendons=[(1120.0, 70, 1150.0, 70 * math.sqrt(2))], sigma_pe=1100.0)
    _has(tr.transverse_normal_stresses(slab, 0.0, -40.0, 0.0, -25.0, pc_class="A"), slab, True)
    rs = jtg.rect(1000, 250, concrete="C40", top=[(7, 16, 45)], bottom=[(7, 16, 45)])
    _has(tr.transverse_crack_width(rs, 40.0, 50.0, 30.0, 35.0), rs)


def test_composite_checks_keep_the_composite():
    comp = _composite()
    pre_chk, cmp_chk = jtg.composite_shear(comp, 500, 900, (2, 12, 150), gamma0=1.1, pc_class="A")
    assert pre_chk.section is comp.precast and cmp_chk.section is comp
    _has(jtg.interface_shear(comp, 900, (2, 12, 150), gamma0=1.1), comp)
    r1, r2 = jtg.composite_crack_resistance(comp, M1k=1950, M1Gk=1800, M2s=1700, M2l=1100, pc_class="A")
    assert r1.section is comp.precast and r2.section is comp and not r1.hogging and not r2.hogging
    _has(jtg.composite_inclined_crack_resistance(comp, 1800, 300, 1700, 400, pc_class="A"), comp)
    _has(jtg.composite_principal_compression(comp, 1800, 300, 2300, 500, stirrups=(2, 12, 150)), comp)
    _has(jtg.composite_service_stresses(comp, M1k=1950, M2k=2300, pc_class="A"), comp)
    _has(jtg.composite_service_stresses(comp, M1k=0.0, M2k=-1000.0, pc_class="A"), comp, True)
    rc_pre = jtg.tee(200, 1200, 1600, 160, concrete="C40", bottom=[(8, 25, 60)])
    rc = jtg.composite(rc_pre, b=2000, t=180, concrete="C40", top=[(12, 12, 50)])
    _has(jtg.composite_crack_width(rc, M1Gk=500, M2s=600, M2l=350), rc)


def test_detailing_reports_keep_the_section_and_rules_without_one_have_none():
    t = _tee()
    _has(jtg.section_detailing(t, env="I", stirrup_d=10), t)
    _has(jtg.section_detailing(t, env="I", stirrup_d=10, hogging=True), t, True)
    _has(jtg.stirrup_detailing(t, (2, 10, 200)), t)
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    _has(pd.column_detailing(col), col)
    pc = _pc()
    _has(td.girder_detailing(pc, stirrup_d=10), pc)
    _has(td.duct_detailing(pc), pc)
    for chk in (jtg.Check.report("a rule", "JTG 3362-2018 9.1.1"), dt.cutoff_detailing(28, 1000),
                jtg.local_bearing("C50", 2000, Al=2e5, Aln=1.8e5, Ab=1.8e6, mesh=(5, 113, 280, 5, 113, 280, 50),
                                  fcu=45, layers=5)):
        assert chk.section is None and chk.hogging is False


def test_section_and_hogging_stay_out_of_repr_and_equality():
    t = _tee()
    a, b = jtg.flexure(t, 1500.0), jtg.flexure(t, 1500.0)
    assert "section=" not in repr(a) and a == b
    b.section = jtg.rect(300, 600)
    assert a == b  # the section does not take part in the comparison
    assert str(a) == str(b)  # __str__ as before


def _svgs(html):
    return re.findall(r"<svg\b", html)


def test_repr_html():
    t = _tee()
    ok = jtg.flexure(t, 1500.0)
    h = ok._repr_html_()
    assert "flexure" in h and "JTG 3362-2018 5.2.2" in h and "OK" in h and len(_svgs(h)) == 1
    assert "γ0Sd" in h and f"{ok.ratio:.3f}" in h and "#2e7d32" in h  # the verdict green
    over = jtg.flexure(t, 9000.0)
    h = over._repr_html_()
    assert not over.ok and "NOT OK" in h and "#c0392b" in h and len(_svgs(h)) == 1
    hog = jtg.flexure(t, -600.0)
    no_id = lambda v: re.sub(r"fm\d+", "fm", v)  # noqa: E731  (every figure has its own id)
    assert no_id(section_figure(t, check=hog).svg()) in no_id(hog._repr_html_())  # the drawing of this check
    # the stress checks draw their diagram; detailing reports their rings; the composite its topping
    pc = _pc()
    assert 'data-role="stress"' in jtg.service_stresses(pc, 6000.0, pc_class="A")._repr_html_()
    det = jtg.section_detailing(jtg.rect(400, 600, concrete="C30", bottom=[(5, 25, 40)], top=[(2, 16, 40)]), env="I",
                                stirrup_d=10)
    assert 'data-role="warn"' in det._repr_html_() and "rule(s) not met" in det._repr_html_()
    comp = _composite()
    assert 'data-part="topping"' in jtg.interface_shear(comp, 900, (2, 12, 150))._repr_html_()
    # no section: text only, no error
    for chk in (jtg.Check.report("a rule", "JTG 3362-2018 9.1.1"),
                jtg.local_bearing("C50", 2000, Al=2e5, Aln=1.8e5, Ab=1.8e6, mesh=(5, 113, 280, 5, 113, 280, 50),
                                  fcu=45, layers=5),
                jtg.deflection_check(30.0, 30000.0)):
        h = chk._repr_html_()
        assert chk.name in h and not _svgs(h)
    lim = jtg.deflection_check(80.0, 30000.0)._repr_html_()
    assert "NOT OK" in lim and "limit" in lim


@pytest.mark.parametrize("Md", [1500.0, -600.0])
def test_figure_follows_the_check_without_hogging(Md):
    """eccentric compression: the compression face from Check.hogging; hogging= still overrides it"""
    r = jtg.rect(500, 800, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    chk = jtg.eccentric(r, 1200, 300 if Md > 0 else -300, l0=4000)
    fig = section_figure(r, check=chk)
    s = fig.svg()
    z0 = float(re.search(r'data-role="compression" data-z0="([-\d.]+)"', s).group(1))
    assert (z0 > 0) == (Md < 0)  # the zone at the bottom for a negative moment
    forced = section_figure(r, check=chk, hogging=Md > 0).svg()
    z0f = float(re.search(r'data-role="compression" data-z0="([-\d.]+)"', forced).group(1))
    assert (z0f > 0) == (Md > 0)
