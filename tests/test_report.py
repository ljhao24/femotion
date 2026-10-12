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
import os
import re
import runpy
import sys
from pathlib import Path

import pytest

import report_cases as rc
import test_report_templates as TT
from femotion.codes import jtg
from femotion.codes.jtg import report as jr
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


# ── checks written (the formulas of every template: tests/test_report_templates.py) ─────
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
    return made + rc.all_checks()


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


def _kinds(harvest):
    kinds = {}
    for c in harvest:
        kinds.setdefault(re.sub(r"[-\d.]+", "#", c.name), c)
    return list(kinds.values())


def test_chinese_names_and_messages(harvest):
    """every kind of check made by the library and the examples: the Chinese name has no English word (the first
    match of the translation table names it, H5 of the review), nor the program's messages"""
    names = sorted({(c.name, jr.check_name(c)) for c in harvest if TT._english(jr.check_name(c))})
    assert not names, names
    bad = sorted({(x, jr.message(x)) for c in harvest for x in list(c.failed) + list(c.warnings) + list(c.notes)
                  if TT._english(jr.message(x))})
    assert not bad, bad[:20]


def test_chinese_tables_of_values(harvest):
    """the general form and the detailing reports of every kind of check: names of the values and rules in Chinese"""
    bad = set()
    for c in _kinds(harvest):
        rep = Report("t")
        jr.write_check(rep.chapter("c"), c, figure=False, formulas=False)
        for cell in re.findall(r"<tr><td[^>]*>(.*?)</td>", rep.html(), re.S):
            t = re.sub(r"<[^>]+>", "", re.sub(r'<span class="m">.*?</span>', "", cell, flags=re.S))
            if TT._english(t):
                bad.add((c.name, t))
    assert not bad, sorted(bad)[:20]


def test_english_report_has_no_chinese(harvest):
    rep = Report("all", lang="en")
    ch = rep.chapter("checks")
    for k, c in enumerate(_kinds(harvest)):
        jr.write_check(ch, c, where=f"#{k}", figure=False)
    jr.summary(rep.chapter("summary"))
    h = rep.html()
    body = h[h.index("<main"):]
    cjk = re.findall(r".{0,40}[\u3000-\u303f\uff00-\uffef\u4e00-\u9fff].{0,20}", body)
    assert not cjk, cjk[:5]


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


def test_example_formulas(tmp_path):
    """every check of the example report is written with its formulas (guards ran), its formulas recompute its
    results from the values printed, and every inequality of the report holds for the numbers it compares"""
    out = tmp_path / "r.html"
    argv = sys.argv
    sys.argv = ["x", "--out", str(out)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = runpy.run_path(str(ROOT / "examples" / "jtg_report_rc_girder.py"), run_name="__main__")
    finally:
        sys.argv = argv
    entries = g["rep"].store["jtg.checks"]
    assert len(entries) >= 14
    for e in entries:
        if e.check.detailing:
            continue
        assert e.guards >= 1, e.title
        seg, vals, _ = TT._write(e.check)
        for row in TT._recompute(e.check, vals):
            label, calc, key = row[:3]
            if key is not None:
                assert abs(calc - TT.V(vals, key)) <= max(3e-3 * abs(calc), 0.06), (e.title, label)
            if len(row) > 3:
                assert TT._close(calc, row[3], 3e-3, 2e-3), (e.title, label, calc, row[3])
    h = out.read_text()
    found = TT._inequalities(h)
    assert len(found) > 30
    bad = [x for x in found if not TT._holds(*x[:3])]
    assert not bad, bad[:5]
    assert "side_cover" not in h and "lnf" not in h
