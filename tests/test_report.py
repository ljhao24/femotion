"""Calculation reports: femotion.report (the document) and femotion.codes.jtg.report (checks written into it).

Independent checks: the numbers a formula shows are read back from the HTML (the values tagged data-sym, as printed,
rounded) and put into the code's formula written out here once more; the result must reproduce the check's resistance
or demand to the rounding of the printed values. Every kind of check the library makes (every literal name of a
Check in its source) is written into a report without error; the summary counts what was written; the HTML parses,
loads nothing from outside, and every number, cross-reference and link resolves."""
import contextlib
import glob
import html.parser
import io
import math
import os
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from femotion.codes import jtg
from femotion.codes.jtg import detailing as dt
from femotion.codes.jtg import devices as dv
from femotion.codes.jtg import pier_detailing as pd
from femotion.codes.jtg import regions as rg
from femotion.codes.jtg import report as jr
from femotion.codes.jtg import slab as sl
from femotion.codes.jtg import tendon_detailing as td
from femotion.codes.jtg import transverse as tr
from femotion.codes.jtg.rc import Check
from femotion.report import Report, _pdf_dest_pages, math as mmath, rich

ROOT = Path(__file__).resolve().parents[1]
STRAND = jtg.tendon("strand", 1860)


# ── reading the HTML back ───────────────────────────────────────────────────
class _Doc(html.parser.HTMLParser):
    """ids, hrefs, external resources and the text of every element with an id"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.hrefs, self.srcs, self.links = [], [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if "href" in a:
            self.hrefs.append(a["href"])
        if "src" in a:
            self.srcs.append(a["src"])
        if tag == "link":
            self.links.append(a)


def _parse(h: str) -> _Doc:
    d = _Doc()
    d.feed(h)
    d.close()
    return d


def _block(h: str, anchor: str) -> str:
    """the HTML of the section whose heading has this id, up to the next heading"""
    i = h.index(f'id="{anchor}"')
    j = re.compile(r"<h[1-4] id=").search(h, i + 10)
    return h[i:j.start() if j else None]


def _vals(seg: str) -> dict:
    """the substituted values of a section as printed: data-sym → float (first occurrence), real minus signs read"""
    out = {}
    for k, v in re.findall(r'<span class="v" data-sym="([^"]+)">([^<]*)</span>', seg):
        if k not in out:
            out[k] = float(v.replace("−", "-"))
    return out


def _written(chk, **kw):
    """a one-check report: (html, values of the check's section)"""
    rep = Report("t")
    ch = rep.chapter("c")
    jr.write_check(ch, chk, id="k", **kw)
    h = rep.html()
    return h, _vals(_block(h, rep.labels["k"][1]))


# ── the document layer ──────────────────────────────────────────────────────
def test_rich_text_and_math():
    assert str(rich("a < b & c_{d}")) == "a &lt; b &amp; c<sub>d</sub>"
    assert str(rich(3.14159)) == "3.14"
    s = str(mmath("f_{cd}bx(h_{0} - \\frac{x}{2})"))
    assert "<i>f</i><sub>cd</sub><i>b</i><i>x</i>" in s and "−" in s and 'class="fr"' in s  # bx: a product
    assert "max" in str(mmath("\\op{max}(a, b)")) and "<i>m</i>" not in str(mmath("max(a, b)"))  # functions upright
    assert '<span class="v" data-sym="As">6158</span>' in str(mmath("\\v{As}{6158}"))
    assert '<span class="mu">kN·m</span>' in str(mmath("\\u{kN·m}"))
    assert 'class="m"' in str(rich("text $x_{1} <= 2$ more"))
    assert "≤" in str(rich("$x <= 2$"))
    with pytest.raises(ValueError):
        mmath("\\nosuch{x}")
    with pytest.raises(ValueError):
        mmath("\\frac{a}{b")


def _small_report(lang="zh"):
    rep = Report("标题 <&>", project="P", member="M", author="A", checker="C", date="2026-10-12", lang=lang)
    c1 = rep.chapter("一")
    c1.paragraph("见", rep.ref("fig:b"), "、", rep.ref("tab:a"), "和", rep.ref("sec:s"), "，", rep.ref("eq:e"))
    c1.table([[1, 2.5], ["x", -3.25]], header=["a", "b"], units=["", "kN"], caption="表一", id="tab:a")
    c1.figure("<svg width='10' height='10'></svg>", caption="图一", id="fig:a")
    s = c1.section("节", id="sec:s")
    s.figure("<svg width='10' height='10'></svg>", caption="图二", id="fig:b")
    s.formula([("x", "1 + 1"), ("", "2")], tag="auto", id="eq:e")
    s.table([[1]], caption="表二")
    c2 = rep.chapter("二")
    c2.figure("<svg width='10' height='10'></svg>", caption="图三")
    c2.table([[1]], caption="表三", id="tab:c")
    ap = rep.appendix("附")
    ap.figure("<svg width='10' height='10'></svg>", caption="图四", id="fig:app")
    return rep


def test_numbering_and_references():
    rep = _small_report()
    h = rep.html()
    caps = re.findall(r'<span class="cap-n">([^<]*)</span>', h)
    assert [c.strip() for c in caps] == ["表 1-1", "图 1-1", "图 1-2", "表 1-2", "图 2-1", "表 2-1", "图 A-1"]
    assert rep.labels["fig:b"][0] == "图 1-2" and rep.labels["tab:a"][0] == "表 1-1"
    assert rep.labels["sec:s"][0] == "1.1 节" and rep.labels["eq:e"][0] == "式 (1-1)"
    assert ">图 1-2</a>" in h and ">表 1-1</a>" in h and ">1.1 节</a>" in h and ">式 (1-1)</a>" in h
    doc = _parse(h)
    assert len(doc.ids) == len(set(doc.ids))  # ids unique
    for href in doc.hrefs:  # every link has its target
        assert href.startswith("#") and href[1:] in doc.ids, href
    bad = Report("t")
    bad.chapter("c").paragraph(bad.ref("nowhere"))
    with pytest.raises(KeyError):
        bad.html()
    dup = Report("t")
    c = dup.chapter("c")
    c.table([[1]], caption="a", id="x")
    c.table([[1]], caption="b", id="x")
    with pytest.raises(ValueError):
        dup.html()
    en = _small_report("en").html()
    assert "Figure 1-2" in en and "Table 1-1" in en and "Contents" in en and "Page" in en
    bare = Report("t")
    cc = bare.chapter("c")
    cc.section("s", id="s1").paragraph("x")
    cc.paragraph(bare.ref("s1", bare=True))
    assert '>1.1</a>' in bare.html()


def test_self_contained_and_print_styles():
    h = _small_report().html()
    doc = _parse(h)
    assert not doc.links and all(not re.match(r"(https?:)?//", s) for s in doc.srcs)
    assert "@import" not in h and not re.search(r"url\(\s*['\"]?(https?:)?//", h)
    assert all(not re.match(r"(https?:)?//", x) for x in doc.hrefs)
    assert "@page" in h and "counter(page)" in h and "counter(pages)" in h and "size: A4" in h
    assert "break-before: page" in h and "break-inside: avoid" in h and "table-header-group" in h
    assert "&lt;&amp;&gt;" in h  # the title escaped


def test_table_cells():
    rep = Report("t")
    rep.chapter("c").table([[1, 2.0, -0.5, "a_{b}", None]], header=["x", "y", "z", "w", "v"], digits=[None, 2, 2, None,
                                                                                                        None])
    h = rep.html()
    assert '<td class="num">2.00</td>' in h and "−0.50" in h and "a<sub>b</sub>" in h


def test_pdf_destinations_parser():
    """the page of each named destination, read from a PDF of the structure Chrome writes (Kids nested)"""
    pdf = b"""%PDF-1.4
1 0 obj
<</Type /Catalog
/Pages 2 0 R
/Dests 9 0 R>>
endobj
2 0 obj
<</Type /Pages
/Count 3
/Kids [3 0 R 8 0 R]>>
endobj
3 0 obj
<</Type /Page
/Parent 2 0 R>>
endobj
8 0 obj
<</Type /Pages
/Kids [4 0 R 5 0 R]>>
endobj
4 0 obj
<</Type /Page>>
endobj
5 0 obj
<</Type /Page>>
endobj
9 0 obj
<</s-1 [4 0 R /XYZ 6 800 0]
/s-1-2 [5 0 R /XYZ 6 400 0]
/fig-1-1 [3 0 R /XYZ 6 400 0]>>
endobj
"""
    assert _pdf_dest_pages(pdf) == {"s-1": 2, "s-1-2": 3, "fig-1-1": 1}
    assert _pdf_dest_pages(b"not a pdf") == {}


def _chrome():
    from femotion.report import _find_chrome
    try:
        return _find_chrome()
    except RuntimeError:
        return None


@pytest.mark.skipif(_chrome() is None, reason="no Chrome to print with")
def test_pdf_with_chrome(tmp_path):
    rep = _small_report()
    rep.chapter("三").paragraph("x" * 50)
    out = rep.pdf(tmp_path / "r.pdf", html_path=tmp_path / "r.html")
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and b"/Outlines" in data
    # the cover on page 1, the contents on page 2, every chapter on a new page
    assert rep.pages["s-1"] == 3 and rep.pages["s-2"] == 4 and rep.pages["s-3"] >= 5
    assert '<span class="tp">3</span>' in (tmp_path / "r.html").read_text()


# ── checks written: the formulas reproduce the results ──────────────────────
def _close(a, b, rel=2e-3):
    return abs(a - b) <= rel * max(abs(a), abs(b))


@pytest.mark.parametrize("sec, Md", [
    (jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 45)]), 250.0),  # singly reinforced rectangle
    (jtg.rect(300, 700, concrete="C35", bottom=[(6, 28, 60)], top=[(3, 20, 40)]), 600.0),  # x ≥ 2a's
    (jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)]), 1577.6),  # x < 2a'
    (jtg.tee(300, 1200, 900, 100, concrete="C30", bottom=[(8, 28, 100)], top=[(4, 16, 40)]), 1200.0),  # in the web
    (jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 28, 95)], top=[(14, 25, 60)]), -1300.0),  # hogging
    (jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 28, 95)], top=[(14, 25, 60)]), 300.0),  # As < A's
])
def test_flexure_formula_reproduces_the_resistance(sec, Md):
    chk = jtg.flexure(sec, Md)
    h, v = _written(chk)
    assert "Mu" in v, "the template was not used"
    assert _close(v["Mu"], chk.resistance, 1e-3)
    if "Mu1" in v:  # 5.2.4: moments about the compression steel, and the singly reinforced section
        Mu1 = v["fsd"] * v["As"] * (v["h"] - v["as"] - v["aref"]) / 1e6
        bf = v.get("bf", v.get("b"))
        Mu2 = v["fcd"] * bf * v["x1"] * (v["h0"] - v["x1"] / 2) / 1e6
        assert _close(Mu1, v["Mu1"]) and _close(Mu2, v["Mu2"])
        assert _close(max(Mu1, Mu2), chk.resistance)
        assert _close(v["x1"], v["fsd"] * v["As"] / (v["fcd"] * bf))
        return
    Asc = v.get("Asc", 0.0)
    fsdc = v.get("fsdc", 0.0)
    T, Cs = v["fsd"] * v["As"], fsdc * Asc
    if "hf" in v and "b" in v and "bf" in v and v["x"] > v["hf"]:  # the zone in the web (5.2.3-2, -3)
        b, bf, hf = v["b"], v["bf"], v["hf"]
        x = (T - Cs - v["fcd"] * (bf - b) * hf) / (v["fcd"] * b)
        Mu = v["fcd"] * (b * x * (v["h0"] - x / 2) + (bf - b) * hf * (v["h0"] - hf / 2))
    else:
        b = v.get("bf", v.get("b"))
        x = (T - Cs) / (v["fcd"] * b)
        Mu = v["fcd"] * b * x * (v["h0"] - x / 2)
    if Asc:
        Mu += fsdc * Asc * (v["h0"] - v["asc"])
    assert _close(x, v["x"]) and _close(Mu / 1e6, chk.resistance)
    assert _close(abs(chk.demand), abs(Md))


@pytest.mark.parametrize("Vd, stirrups, kw", [
    (514.4, (2, 12, 100), {}),  # P capped at 2.5
    (300.0, (2, 10, 200), {"alpha1": 0.9}),
])
def test_shear_formula_reproduces_the_resistance(Vd, stirrups, kw):
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(4, 22, 60)], top=[(8, 20, 50)]) if kw else \
        jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    chk = jtg.shear(sec, Vd, stirrups, **kw)
    h, v = _written(chk)
    P = min(100 * v["As"] / (v["b"] * v["h0"]), 2.5)
    assert _close(P, v["P"])
    rho = v["Asv"] / (v["sv"] * v["b"])
    assert _close(rho, v["rho_sv"])
    Vcs = 0.45e-3 * v["a1"] * v["a2"] * v["a3"] * v["b"] * v["h0"] * math.sqrt(
        (2 + 0.6 * v["P"]) * math.sqrt(v["fcuk"]) * v["rho_sv"] * v["fsv"])
    assert _close(Vcs, chk.details["Vcs"]) and _close(Vcs, chk.resistance) and _close(v["Vu"], chk.resistance, 1e-3)
    assert _close(0.51e-3 * math.sqrt(v["fcuk"]) * v["b"] * v["h0"], v["Vmax"])
    assert _close(0.50e-3 * v["a2"] * v["ftd"] * v["b"] * v["h0"], v["Vmin"])


@pytest.mark.parametrize("sec, Nd, Md, l0", [
    (jtg.rect(500, 800, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 25, 50)]), 800.0, 600.0, 6000),  # large
    (jtg.rect(500, 800, concrete="C30", bottom=[(5, 25, 50)], top=[(5, 25, 50)]), 1500.0, 450.0, 6000),  # small
    (jtg.circle(1500, 28, 28, 60, concrete="C35"), 9800.0, 3100.0, 18000),  # 5.3.8
])
def test_eccentric_formula_reproduces_the_resistance(sec, Nd, Md, l0):
    chk = jtg.eccentric(sec, Nd, Md, l0=l0)
    h, v = _written(chk)
    assert "Nu" in v
    if "alpha" in v:
        a, at = v["alpha"], max(1.25 - 2 * v["alpha"], 0.0)
        assert _close(at, v["alphat"]) if v["alphat"] else at == 0
        N = a * v["fcd"] * v["A"] * (1 - math.sin(2 * math.pi * a) / (2 * math.pi * a)) + (a - at) * v["fsd"] * v["As"]
    else:
        if "ecu" in v:  # small eccentricity: σs by 5.1.5-1 from the values shown
            ss = v["ecu"] * v["Es"] * (v["beta"] * v["h0"] / v["x"] - 1)
            assert _close(ss, v["ss"], 3e-3)
        else:
            ss = v["ss"]
            assert ss == sec.steel.fsd
        N = v["fcd"] * v["b"] * v["x"] + v["fsdc"] * v["Asc"] - ss * v["As"]
    assert _close(N / 1e3, chk.resistance)
    # the eccentricity and η shown
    assert _close(v["e0"], max(abs(Md) / Nd * 1e3, 20.0, sec.h / 30), 1e-3)
    if v.get("eta", 1.0) != 1.0 and "z1" in v:
        eta = 1 + (v["l0"] / v["h"]) ** 2 * v["z1"] * v["z2"] / (1300 * v["e0"] / v["h0"])
        assert _close(eta, v["eta"])


@pytest.mark.parametrize("Ms, Ml", [(769.3, 567.7), (-837.2, -707.7)])
def test_crack_width_formula_reproduces_the_demand(Ms, Ml):
    span = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    pier = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 28, 95)], top=[(14, 25, 60)])
    chk = jtg.crack_width(span if Ms > 0 else pier, Ms, Ml, env="I")
    h, v = _written(chk)
    sss = v["Ms"] * 1e6 / (0.87 * v["As"] * v["h0"])
    assert _close(sss, v["sss"])
    assert _close(1 + 0.5 * v["Ml"] / v["Ms"], v["C2"])
    rho = min(max(v["As"] / v["Ate"], 0.01), 0.1)
    assert _close(rho, v["rho_te"])
    W = v["C1"] * v["C2"] * v["C3"] * v["sss"] / v["Es"] * (v["c"] + v["d"]) / (0.36 + 1.7 * v["rho_te"])
    assert _close(W, chk.demand, 3e-3) and _close(v["Wcr"], chk.demand, 5e-3)


def test_other_templates_reproduce_their_results():
    r = jtg.rect(500, 500, concrete="C30", bottom=[(3, 25, 50)], top=[(3, 25, 50)])
    a = jtg.axial(r, 3000, 5000)
    _, v = _written(a)
    assert _close(0.9 * v["phi"] * (v["fcd"] * v["A"] + v["fsdc"] * v["As"]) / 1e3, a.resistance)
    lb = jtg.local_bearing("C50", 2000, Al=2e5, Aln=1.8e5, Ab=1.8e6, mesh=(5, 113, 280, 5, 113, 280, 50), fcu=45,
                           layers=5)
    _, v = _written(lb)
    F2 = 0.9 * (v["eta_s"] * v["beta"] * v["fcd"] + v["k"] * v["rho_v"] * v["beta_cor"] * v["fsd"]) * v["Aln"] / 1e3
    F1 = 1.3 * v["eta_s"] * v["beta"] * v["fcd"] * v["Aln"] / 1e3
    assert _close(min(F1, F2), lb.resistance)
    pu = sl.punching(400, Um=3220, h0=205, h=250, concrete="C40")
    _, v = _written(pu)
    assert _close(0.7 * v["beta_h"] * v["ftd"] * v["Um"] * v["h0"] / 1e3, pu.resistance)
    tor = jtg.rect(400, 800, concrete="C30", bottom=[(4, 22, 50)], top=[(2, 22, 50)])
    t = jtg.torsion(tor, Td=95.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))
    _, v = _written(t)
    assert "Tu" in v and _close(v["Tu"], t.resistance, 1e-3)
    d = jtg.deflection_check(12.486, 16000)
    _, v = _written(d)
    assert _close(v["L"] / 600, d.resistance)


def test_fallback_to_the_general_form():
    """a section with web bars (not only the edge rows of the code's formula) is written in the general form, its
    verdict as the check's"""
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)],
                  bars=[(2 * 113.1, 500, 12)])
    chk = jtg.flexure(sec, 1500.0)
    h, v = _written(chk)
    assert "Mu" not in v and "统一算法" in h and "<th>参数</th>" in h
    assert f"{chk.resistance:.1f}" in h.replace("−", "-")


# ── every kind of check ─────────────────────────────────────────────────────
def _catalog():
    """checks of every kind not met in the examples"""
    t = jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(4, 32, 70)], top=[(8, 16, 40)])
    pc = jtg.tee(200, 1800, 2200, 180, bf_bottom=700, hf_bottom=280, concrete="C50", bottom=[(10, 16, 50)],
                 top=[(16, 12, 45)], tendons=[(12 * 139, 1650, 1200, 90)] * 2 + [(9 * 139, 1520, 1200, 80)],
                 tendon=STRAND, sigma_pe=1080, sigma_l6=80)
    r = jtg.rect(500, 800, concrete="C30", bottom=[(4, 25, 45)], top=[(4, 25, 45)])
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    sx, sy = jtg.rect_biaxial(600, 900, layout=(4, 5, 25, 60), concrete="C40", steel="HRB500")
    ft = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(6, 25, 60)], top=[(8, 12, 40)])
    pre = jtg.rect(1000, 250, pretensioned=True, tendons=[(10 * 140, 200, 1200)], tendon=STRAND, bottom=[(8, 12, 30)])
    x = np.linspace(0, 3000, 7)
    out = [
        jtg.flexure(t, 1500.0), jtg.flexure(t, -600.0), jtg.flexure(t, 9000.0),
        jtg.shear(jtg.rect(1000, 250, bottom=[(7, 12, 30)]), 100.0, None, slab=True),
        jtg.shear(col, 500.0, (2, 12, 150)), jtg.axial(r, 3000.0, 4000.0), jtg.eccentric(r, 1200, -300, l0=4000),
        jtg.axial_spiral(jtg.circle(600, 8, 22, 50, concrete="C30"), 3000, 3000, spiral=(10, 50)),
        jtg.tension(jtg.rect(400, 600, bottom=[(4, 22, 45)], top=[(4, 22, 45)]), 800, 80),
        jtg.tension(jtg.rect(400, 600, bottom=[(4, 22, 45)], top=[(4, 22, 45)]), 800, 0),
        jtg.biaxial_tension(sx, sy, 800, 120, 90), jtg.biaxial(sx, sy, 3000, 300, 200, l0x=6000, l0y=6000),
        jtg.torsion_flange(ft, Td=60, which="top", stirrups=(2, 10, 150), Ast=8 * 113, core=(800, 90)),
        jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="A"), jtg.crack_resistance(pc, 4200.0, 3000.0,
                                                                                     pc_class="full"),
        jtg.crack_resistance(pc, 4200.0, 3000.0, pc_class="B", M_G=2000.0),
        jtg.crack_resistance(pre, 20.0, 15.0, pc_class="A", transfer=jtg.Transfer(x=300.0, ltr=600.0)),
        jtg.inclined_crack_resistance(pc, 4000.0, 600.0, pc_class="A"), jtg.principal_compression(pc, 6000.0, 800.0),
        jtg.service_stresses(pc, 6000.0, pc_class="A"),
        jtg.service_stresses(pc.replace(tendons=pc.tendons[:1], sigma_pe=np.array([1080.0])), 7000.0, pc_class="B"),
        jtg.transient_rc(jtg.rect(300, 600, concrete="C30", bottom=[(4, 25, 50)], top=[(2, 16, 40)]), 120.0, 150.0,
                         fcu=25),
        jtg.transient_shear(x, 1.8 * (1 - x / 3000), fcu=30, b=250, steel="HRB400", stirrups=(2, 8, 200)),
        jtg.transient_pc(pc, 300.0, fcu=45),
        jtg.local_bearing("C50", 2000, Al=2e5, Aln=1.8e5, Ab=1.8e6),
        jtg.interface_shear_slab(300, b=1000, h0=400, gamma0=1.0, s=200, Asv=150),
        sl.punching(400, Um=3220, h0=205, h=250, concrete="C40", stirrups=1200, Um_outer=4500),
        sl.footing_punching(ps=0.25, A=2.0e6, bt=800, h0=700, h=750, concrete="C30"),
        rg.pile_cap_punching_corner(5000, bx=900, by=900, ax=600, ay=750, h0=1500, concrete="C30"),
        rg.pile_cap_punching_edge(5000, bp=800, bx=900, ax=600, h0=1500, b=6000, concrete="C30"),
        tr.inplane_principal([(6.0, 1.0, 1.2), (3.0, 0.5, 2.0)], concrete="C50", pc_class="A"),
        tr.inplane_principal_compression([(6.0, 1.0, 1.2), (3.0, 0.5, 2.0)], concrete="C50"),
        rg.pile_cap_shear(9000, As=40 * 616, bs=6000, h0=1800, a=600, concrete="C30"),
        jtg.torsion(jtg.rect(400, 800, concrete="C30", bottom=[(4, 22, 50)], top=[(2, 22, 50)]), Td=95.0,
                    stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720)),
        jtg.composite_crack_width(jtg.composite(jtg.tee(200, 1200, 1600, 160, concrete="C40", bottom=[(8, 25, 60)]),
                                                b=2000, t=180, concrete="C40", top=[(12, 12, 50)]),
                                  M1Gk=500, M2s=600, M2l=350),
        jtg.deflection_check(30.0, 30000.0), jtg.deflection_check(80.0, 30000.0, cantilever=True),
        dt.splice_detailing("weld", 20, length=99, splice_area=300, joint_spacing=700, share=0.51, clear=24),
        dt.splice_detailing("mechanical", 20, cover=25, cmin=30),
        dt.curved_soffit(6000, 28, (12, 100), r=20000, steel="HRB400", stirrup_steel="HPB300"),
        dt.cutoff_detailing(28, 1000), dt.support_detailing(12, 2), dt.bent_bar_detailing(25, 900, angle=45, start=450,
                                                                                         anchorage=500),
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


@pytest.fixture(scope="module")
def harvest(tmp_path_factory):
    """every Check made by the examples (codes.jtg), and those of _catalog"""
    made = []
    orig = Check.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        made.append(self)
    Check.__init__ = init
    cwd, argv = os.getcwd(), sys.argv
    os.chdir(tmp_path_factory.mktemp("examples"))
    try:
        for f in sorted(glob.glob(str(ROOT / "examples" / "jtg_*.py"))):
            if f.endswith("jtg_report_rc_girder.py"):
                continue
            sys.argv = [f]
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(f, run_name="__main__")
    finally:
        os.chdir(cwd)
        sys.argv = argv
        Check.__init__ = orig
    return made + _catalog()


def _literal_names():
    """a regular expression for every literal name of a Check in the library's source"""
    out = []
    for f in sorted((ROOT / "src" / "femotion" / "codes" / "jtg").glob("*.py")):
        for isf, s in re.findall(r'Check(?:\.report)?\(\s*(f?)"((?:[^"\\]|\\.)*)"', f.read_text()):
            parts = re.split(r"(\{[^{}]*\})", s) if isf else [s]
            out.append((f.name, s, "".join(".*" if (isf and p.startswith("{")) else re.escape(p) for p in parts)))
    return out


def test_every_kind_of_check_is_written(harvest):
    kinds = {}
    for c in harvest:
        kinds.setdefault(re.sub(r"[-\d.]+", "#", c.name), c)
    names = _literal_names()
    assert len(names) > 80
    missing = [(f, s) for f, s, rx in names if not any(re.fullmatch(rx, c.name) for c in harvest)]
    assert not missing, missing
    for lang in ("zh", "en"):
        rep = Report("all", lang=lang)
        ch = rep.chapter("checks")
        for k, c in enumerate(kinds.values()):
            jr.write_check(ch, c, where=f"#{k}", figure=k % 3 == 0)  # drawings for a third of them, for time
        jr.summary(rep.chapter("summary"))
        h = rep.html()
        doc = _parse(h)
        assert len(doc.ids) == len(set(doc.ids))
        assert all(x[1:] in doc.ids for x in doc.hrefs)
        assert len(rep.store["jtg.checks"]) == len(kinds)


def test_templates_used_where_they_apply(harvest):
    """the checks the templates cover are written with formulas (values tagged), not in the general form"""
    want = ("flexure", "shear", "crack width", "deflection", "local compression", "eccentric compression",
            "axial compression", "concrete compression in service (top", "crack resistance (normal section, class A")
    done = set()
    for c in harvest:
        k = next((w for w in want if c.name.startswith(w)), None)
        if k is None or k in done or c.failed:
            continue
        h, v = _written(c, figure=False)
        if v:
            done.add(k)
    assert done == set(want), set(want) - done


def test_summary_counts():
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
    checks = [jtg.flexure(sec, 1500.0), jtg.flexure(sec, 9000.0), jtg.shear(sec, 400, (2, 10, 400)),
              jtg.crack_width(sec, 769.3, 567.7, env="I"), jtg.section_detailing(sec, env="I", stirrup_d=10),
              jtg.deflection_check(40.0, 16000)]
    rep = Report("s")
    ch = rep.chapter("c")
    for k, c in enumerate(checks):
        jr.write_check(ch, c, where=f"p{k}", figure=False)
    jr.summary(rep.chapter("sum"))
    h = rep.html()
    seg = h[h.index('id="tab-2-1"'):]
    tab = seg[:seg.index("</table>")]
    body = tab[tab.index("<tbody>"):]
    assert body.count("<tr") == len(checks)
    bad = [c for c in checks if not c.detailing and not c.ok]
    assert body.count('<tr class="fail">') == len(bad) >= 2
    warns = sum(len(c.warnings) for c in checks)
    seg2 = h[h.index('id="tab-2-2"'):]
    tab2 = seg2[:seg2.index("</table>")]
    assert tab2[tab2.index("<tbody>"):].count("<tr") == warns > 0
    n = sum(1 for c in checks if not c.detailing)
    assert f"共 {n} 项验算，其中 {len(bad)} 项不满足要求" in h
    ok = Report("s")
    jr.write_check(ok.chapter("c"), checks[0], figure=False)
    jr.summary(ok.chapter("s"))
    assert "共 1 项验算，全部满足要求" in ok.html()


def test_design_data_tables():
    sec = jtg.tee(200, 1600, 1800, 180, concrete="C50", bottom=[(6, 16, 50)], tendons=[(2780, 1470, 1150, 90)],
                  tendon="strand1860", sigma_pe=1050)
    from femotion.frame import Model, Section, Solver  # noqa: F401
    from femotion.frame.loads import LoadCase
    m_ = Model()
    n = [m_.node((x, 0, 0)) for x in (0, 5, 10)]
    bs = [m_.beam(a, b, Section(E=3e7, A=1, Iy=1, Iz=1, J=1, mass=2.0)) for a, b in zip(n[:-1], n[1:])]
    m_.fix(n[0])
    case = LoadCase("G").self_weight().line(bs, 5.0).point(bs, 2.0, 100).nodal(n[2], F=(0, 0, -10)) \
        .settlement(n[0], -0.005).imposed(bs, curvature=1e-4)
    for lang in ("zh", "en"):
        rep = Report("d", lang=lang)
        ch = rep.chapter("c")
        jr.design_basis(ch)
        jr.materials(ch, {"s": sec})
        jr.section_properties(ch, {"s": sec})
        jr.load_cases(ch, [case], model=m_)
        jr.live_load(ch, jtg.LiveLoad(L0=20, lanes=2, mu=0.2))
        jr.combinations(ch, crowd=True, characteristic=True)
        h = rep.html()
        assert "1860" in h and "19.62" in h and "JTG 3362" in h
    assert "<i>f</i><sub>pd</sub>" in h and "1260" in h  # fpd of 1860 strand


def test_example_report(tmp_path):
    out = tmp_path / "r.html"
    argv = sys.argv
    sys.argv = ["x", "--out", str(out)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / "examples" / "jtg_report_rc_girder.py"), run_name="__main__")
    finally:
        sys.argv = argv
    h = out.read_text()
    doc = _parse(h)
    assert len(doc.ids) == len(set(doc.ids)) and all(x[1:] in doc.ids for x in doc.hrefs)
    assert not doc.links and not doc.srcs
    caps = re.findall(r'<span class="cap-n">([^<]*)</span>', h)
    for kind in ("图", "表"):
        nums = [tuple(map(int, c.split()[1].split("-"))) for c in caps if c.startswith(kind)]
        assert len(nums) == len(set(nums))
        for ch_ in {a for a, _ in nums}:  # consecutive within each chapter
            ks = sorted(k for a, k in nums if a == ch_)
            assert ks == list(range(1, len(ks) + 1))
    assert "全部满足要求" in h and "1941.5" in h and "0.146" in h
