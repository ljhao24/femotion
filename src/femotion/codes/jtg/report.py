"""Calculation reports (计算书) of JTG checks, on the document layer femotion.report.

    from femotion.report import Report
    from femotion.codes import jtg
    from femotion.codes.jtg import report as jr

    rep = Report("两跨连续 T 梁计算书", project="某大桥", member="主梁 T 梁")
    ch = rep.chapter("持久状况承载能力极限状态")
    jr.write_check(ch, jtg.flexure(sec, Md, gamma0), where="x = 25.50 m（跨中）")    # one check: one section
    jr.write_check(ch, jtg.shear(sec, Vd, (2, 12, 100), gamma0=gamma0), where="x = 0.55 m（端支点 h/2 处）")
    jr.summary(rep.chapter("验算汇总"))                    # every check written so far, the warnings, the conclusion
    rep.save("girder.html")

write_check writes a Check as a numbered section: the location (given by the caller), the clause, the calculation with
each formula first in symbols, then with the values substituted, then the result (in N, mm, MPa; moments in kN·m), the
verdict (γ0Sd ≤ R, or the value against its limit, with the ratio; a condition not met first), what failed, the warnings
(detailing, minimum reinforcement, materials: they do not change the verdict) apart, the notes, and the section drawn
with the check over it.

Every template follows two rules:
- what it writes reproduces the program: it computes each result it prints from the formula it prints and the values it
  substitutes, and compares it with the Check (W.guard, relative 1e-6); a template that does not reproduce the check is
  dropped and the check is written in the general form (the values of Check.details as a table, names translated). That
  is the case for sections outside the formula's premises (bars between the edge rows, composite sections, the load
  between the gross and plastic centroids …), which the program solves by its unified algorithm
  (docs/codes/JTG_3362-2018.md §10);
- every inequality is set by the two numbers it compares (W.rel), never by the verdict; a condition not met stands on a
  row of its own.

Templates: flexure (rectangles, T, I and box sections, flange or web, x < 2a′ by 5.2.4 / 5.2.6,
                    over-reinforced sections,
internal and external tendons, 9.1.12, 9.1.13 with Mcr), shear (5.2.9, 5.2.11, 5.2.12, slabs, the section limit at end
supports), axial compression (5.3.1), eccentric compression (5.3.4, 5.3.5 large and small eccentricity, x ≥ 2a′s,
reverse failure, 5.3.6; circles by 5.3.8 with both equations), tension (5.4.1, 5.4.2 small and large eccentricity),
torsion (5.5.1, 5.5.4 with the stirrups shared by 5.5.6, 5.5.3), crack width (6.4.3 with σss of 6.4.4 for flexural,
eccentric compression, tension and class B members), crack resistance (6.3.1, 6.3.2), principal stresses (6.3.3,
6.3.1-5…-8, 7.1.6), stresses in service (7.1.3, 7.1.5; 7.1.4 cracked class B), deflection (6.5.3), transient stresses
(7.2.4, 7.2.8), local compression (5.7, with and without indirect reinforcement), punching (5.6.1, 5.6.2, 5.6.3),
bearings (8.7.3, 8.7.4, 8.7.5) and expansion joints (8.8.2), detailing reports (a table of the rules).

Design data: design_basis (the codes), materials (from the sections), section_properties, load_cases (LoadCase),
live_load, combinations. Charts and drawings come from femotion.codes.jtg.charts, .figures and femotion.frame.figures.
Language: lang="zh" (default: the report's) or "en". In Chinese the check names, the program's messages,
                     the rules of the
detailing reports and the names of the values are translated (femotion.codes.jtg._report_text).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from ...report import Cell, Markup, Node, fixed, m, num
from . import _report_text as TX
from .rc import Check, Rect

__all__ = ["write_check", "write_checks", "summary", "design_basis", "materials", "section_properties", "load_cases",
           "live_load", "combinations", "shear_search_table", "stirrup_design_block", "bent_bar_table", "check_name",
           "clause_text", "message", "Entry"]

# ── words ───────────────────────────────────────────────────────────────────
_L = {
    "zh": {"where": "验算位置", "clause": "依据", "units": "以下各式中力以 N、长度以 mm、应力以 MPa 计，弯矩换算为 kN·m、"
           "力换算为 kN。", "ok": "满足要求", "not": "不满足要求", "ratio": "利用率", "margin": "余量",
           "failed": "不满足的条件", "warnings": "构造及最小配筋提示（不影响承载力验算结论）", "notes": "说明",
           "param": "参数", "value": "数值", "unit": "单位", "verdict": "结论", "no": "序号", "item": "验算项目",
           "loc": "位置", "demand": "作用效应", "resist": "抗力或限值", "summary_cap": "验算结果汇总",
           "warn_cap": "构造检查提示汇总", "content": "内容", "all_ok": "共 {n} 项验算，全部满足要求。",
           "some_fail": "共 {n} 项验算，其中 {k} 项不满足要求（表中红色行）。", "warn_n": "另有构造检查提示 {k} 条（见",
           "warn_n2": "），不影响承载力和正常使用验算的结论，应在构造设计中处理。", "no_warn": "构造检查无提示。",
           "rules": "构造规则", "actual": "实际值", "limit": "限值", "rule_ok": "满足", "rule_fail": "不满足",
           "det_ok": "各项构造满足", "det_warn": "{k} 项提示", "general": "程序按截面平衡的统一算法（平截面假定，钢筋应力按 "
           "5.1.5 并受 5.1.3 限制）求解，主要中间结果如下。", "top": "上缘", "bottom": "下缘", "edge": "边缘",
           "section": "章节", "sep": "，", "end": "。", "semi": "；", "colon": "：", "but": "但", "lp": "（", "rp": "）",
           "also": "另：", "design": "（设计回代，不计入验算项数）"},
    "en": {"where": "Location", "clause": "Clause", "units": "Forces in N, lengths in mm, stresses in MPa; moments "
           "converted to kN·m, forces to kN.", "ok": "OK", "not": "NOT OK", "ratio": "ratio", "margin": "margin",
           "failed": "Conditions not met", "warnings": "Detailing and minimum-reinforcement warnings (they do not"
                                                       " change "
           "the verdict)", "notes": "Notes", "param": "Quantity", "value": "Value", "unit": "Unit",
           "verdict": "Verdict",
           "no": "No.", "item": "Check", "loc": "Location", "demand": "Effect", "resist": "Resistance or limit",
           "summary_cap": "Summary of the checks", "warn_cap": "Detailing warnings", "content": "Warning",
           "all_ok": "{n} checks, all met.", "some_fail": "{n} checks, {k} not met (red rows).",
           "warn_n": "{k} detailing warning(s) (see ", "warn_n2": "); they do not change the verdicts and are to be "
           "resolved in the detailing.", "no_warn": "No detailing warnings.", "rules": "Rule", "actual": "Value",
           "limit": "Limit", "rule_ok": "met", "rule_fail": "not met", "det_ok": "rules met",
           "det_warn": "{k} warning(s)", "general": "The program solves the section by the unified algorithm (plane "
           "sections, steel stresses by 5.1.5 within 5.1.3); the main values:", "top": "top", "bottom": "bottom",
           "edge": "edge", "section": "Section", "sep": ", ", "end": ".", "semi": "; ", "colon": ": ", "but": "but",
           "lp": " (", "rp": ")", "also": "Also: ", "design": " (design back-substitution, not counted)"},
}


def _lang(node: Node | None, lang: str | None) -> str:
    if lang is not None:
        return lang
    return node.report.lang if node is not None else "zh"


def check_name(chk: Check, lang: str = "zh") -> str:
    """the check's name in the language (zh: femotion.codes.jtg._report_text; en: the program's name)"""
    if lang == "zh":
        zh = TX.name_zh(chk.name)
        if zh is not None:
            return zh
    return chk.name[:1].upper() + chk.name[1:]


def clause_text(clause: str, lang: str = "zh") -> str:
    """'JTG 3362-2018 5.2.2 / 5.2.3' → 'JTG 3362—2018 5.2.2、5.2.3' (zh)"""
    s = str(clause)
    if lang != "zh":
        return s
    mt = re.match(r"(JTG \S+?)-(\d{4})\s*(.*)", s)
    if not mt:
        return TX.clause(s)
    code, year, rest = mt.groups()
    rest = TX.clause(rest.replace(" / ", ", ").replace("–", "～"))
    return f"{code}—{year} {rest}".strip()


def message(s: str, lang: str = "zh") -> str:
    """a warning, note or failed condition of a check in the language"""
    s = str(s)
    if lang == "zh":
        zh = TX.message_zh(s)
        s = s if zh is None else zh
    return re.sub(r"(?<![\w.)])-(?=\d)", "−", s)  # a minus sign, not a hyphen


# ── numbers ─────────────────────────────────────────────────────────────────
def _nv(v: float, sig: int = 4) -> str:
    """a value substituted into a formula: whole numbers from 10000, one decimal from 1000, else `sig` significant
    figures; rounded half up, real minus sign"""
    v = float(v)
    if not math.isfinite(v):
        return "∞"
    a = abs(v)
    if a >= 10000 or a == 0:
        return fixed(v, 0)
    if a >= 1000:
        s = fixed(v, 1)
    else:
        s = fixed(v, max(sig - 1 - int(math.floor(math.log10(a))), 0))
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _sci_math(v: float, sig: int = 4) -> str:
    """a large number in the math language: a×10^{n}"""
    v = float(v)
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    return f"{fixed(v / 10 ** e, sig - 1)}×10^{{{e}}}"


_TRACE = None  # a list to collect why templates fell back (debugging and tests)


# formula numbers of JTG 3362-2018 without a dash (the other numbers without one are clauses)
_FORMULA_NUMBERS = {"5.2.11", "5.2.12", "5.3.1", "5.4.1", "5.6.1", "6.4.3", "7.2.5", "7.2.8", "9.1.13"}


class _Mismatch(Exception):
    """a template's formula does not reproduce the program's result: the check is written in the general form"""


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


class _W:
    """what a template writes into: the node of the check, the language, helpers for formulas"""

    def __init__(self, node: Node, chk: Check, lang: str):
        self.node, self.chk, self.lang = node, chk, lang
        self.sec = chk.section
        self.s = None
        if self.sec is not None and hasattr(self.sec, "flipped"):
            self.s = self.sec.flipped() if chk.hogging else self.sec
        self.units_said = False
        self.hide_notes: list = []  # notes the template has written in its own words
        self.plain_figure = False  # draw the section without the check (its x differs from the one written)
        self.clause = None  # the clauses the written calculation follows, when they differ from Check.clause
        self.guards = 0

    def t(self, zh: str, en: str) -> str:
        return zh if self.lang == "zh" else en

    @property
    def zh(self) -> bool:
        return self.lang == "zh"

    def units(self):
        """the units of the formulas, once per chapter"""
        if self.units_said:
            return
        self.units_said = True
        ch = self.node
        while ch.parent is not None:
            ch = ch.parent
        said = self.node.report.store.setdefault("jtg.units_said", set())
        if id(ch) in said:
            return
        said.add(id(ch))
        self.node.paragraph(_L[self.lang]["units"], cls="muted")

    def p(self, *parts):
        self.node.paragraph(*parts)

    def tag(self, t):
        """a formula number as it is; a clause (no formula of that number) as "(5.3.9 条)" / "(cl. 5.3.9)";
        "§x" forces a clause"""
        if t is None or t.startswith("("):
            return t
        clause = t.startswith("§")
        t = t.lstrip("§")
        if not clause:
            clause = "-" not in t and t not in _FORMULA_NUMBERS and re.fullmatch(r"[\d.]+", t) is not None
        if clause:
            return f"({t} 条)" if self.zh else f"(cl. {t})"
        return t

    def eq(self, rows, tag=None, where=None):
        rows = [r[:3] + (self.tag(r[3]),) if isinstance(r, tuple) and len(r) > 3 else r for r in rows]
        self.node.formula(rows, tag=self.tag(tag), where=where)

    def head(self, text: str):
        self.node.paragraph(text, cls="lead")

    @staticmethod
    def v(key: str, value: float, nd: int | None = None, paren: bool = False) -> str:
        """a substituted value \\v{key}{text}; paren: a negative value in parentheses (after an operator)"""
        txt = _nv(value) if nd is None else fixed(value, nd)
        out = f"\\v{{{key}}}{{{txt}}}"
        return f"({out})" if (paren and txt.startswith("−")) else out

    @staticmethod
    def sv(key: str, value: float, sig: int = 4) -> str:
        """a large substituted value as a×10^n, tagged"""
        return f"\\v{{{key}}}{{{_sci_math(value, sig)}}}"

    def vp(self, key: str, value: float, nd: int | None = None) -> str:
        return self.v(key, value, nd, paren=True)

    @staticmethod
    def n(value: float, nd: int | None = None, paren: bool = False) -> str:
        """a number written without a tag (a result or an intermediate value)"""
        txt = _nv(value) if nd is None else fixed(value, nd)
        return f"({txt})" if (paren and txt.startswith("−")) else txt

    def guard(self, shown: float, program: float, rel: float = 1e-6):
        """the value the template computed from its formula against the program's"""
        self.guards += 1
        if not _close(float(shown), float(program), rel):
            raise _Mismatch(f"{shown} ≠ {program}")

    @staticmethod
    def rel(a: float, b: float, ge: bool = False) -> str:
        """the relation of a to b as written: "<=" or ">" (ge: ">=" or "<")"""
        tol = 1e-9 * max(abs(a), abs(b), 1.0)
        if ge:
            return ">=" if a >= b - tol else "<"
        return "<=" if a <= b + tol else ">"

    def cmp(self, a: float, b: float, right: str, ge: bool = False):
        """the row "rel right" under a row ending in the value a, right ending in b"""
        return "", self.rel(a, b, ge), right

    def take(self, text_zh: str, text_en: str) -> str:
        return f"\\t{{{text_zh}}}" if self.zh else f"\\t{{{text_en}}}"


# ── section data for the templates ──────────────────────────────────────────
@dataclass
class _Steel:
    As: float
    ds: float
    Asc: float
    asc: float
    Ap: float
    dp: float
    Apc: float
    apc: float
    sp0c: float
    Fex: float
    Aex: float
    sex: float
    dex: float
    edge_only: bool
    tension_rows: int


def _steel(s) -> _Steel:
    bot, top = s.edge_masks()
    pb, pt = s.tendon_masks()

    def mean(a, d):
        return float(np.average(d, weights=a)) if a.sum() > 0 else 0.0
    bars, td, ex = s.bars, s.tendons, s.external
    As, Asc = bars[bot, 0].sum(), bars[top, 0].sum()
    Ap, Apc = (td[pb, 0].sum(), td[pt, 0].sum()) if len(td) else (0.0, 0.0)
    ex_ok = bool((ex[:, 1] > s.zg).all()) if len(ex) else True
    Fex = float((ex[:, 2] * ex[:, 0]).sum()) if len(ex) else 0.0
    rows = len(np.unique(np.round(bars[bot, 1], 6)))
    return _Steel(float(As), mean(bars[bot, 0], bars[bot, 1]), float(Asc), mean(bars[top, 0], bars[top, 1]),
                  float(Ap), mean(td[pb, 0], td[pb, 1]) if Ap else 0.0, float(Apc),
                  mean(td[pt, 0], td[pt, 1]) if Apc else 0.0, mean(td[pt, 0], td[pt, 2]) if Apc else 0.0,
                  Fex, float(ex[:, 0].sum()) if len(ex) else 0.0, Fex / ex[:, 0].sum() if len(ex) else 0.0,
                  mean(ex[:, 0] * ex[:, 2], ex[:, 1]) if len(ex) else 0.0,
                  bool((bot | top).all()) and (bool((pb | pt).all()) if len(td) else True) and ex_ok, rows)


def _flange(s):
    """(b'f, h'f, b) of the compressed top of a section of stacked solid rectangles: the top rectangle and the one under
    it (a rectangle: b'f = b, h'f = h); None for other shapes; b None when the top is narrower than what is under it"""
    if not all(isinstance(p, Rect) and p.sign > 0 for p in s.pieces):
        return None
    top = [p for p in s.pieces if abs(p.top) < 1e-9]
    if len(top) != 1:
        return None
    p0 = top[0]
    if p0.bottom >= s.h - 1e-9:
        return p0.b, s.h, p0.b
    nxt = [p for p in s.pieces if abs(p.top - p0.bottom) < 1e-9]
    if len(nxt) != 1:
        return None
    if p0.b < nxt[0].b - 1e-9:
        return p0.b, p0.bottom, None  # narrower on top (a turned T): a rectangle while x ≤ h'f
    return p0.b, p0.bottom, nxt[0].b


def _join(terms, op=" + "):
    return op.join(terms).replace("+ -", "- ")


# ── flexure ─────────────────────────────────────────────────────────────────
def _h0_rows(w: _W, g: _Steel, s, fsd, fpd, h0):
    """h0 = h − a, a the resultant of As (fsd), Ap (fpd) and the external tendons (σpe,ex) (5.2.2, figure 5.2.5)"""
    v, h = w.v, s.h
    parts = [("f_{sd}A_{s}", "a_{s}", fsd, g.As, h - g.ds, "fsd", "As", "as")]
    if g.Ap:
        parts.append(("f_{pd}A_{p}", "a_{p}", fpd, g.Ap, h - g.dp, "fpd", "Ap", "ap"))
    if g.Fex:
        parts.append(("σ_{pe,ex}A_{ex}", "a_{ex}", g.sex, g.Aex, h - g.dex, "sex", "Aex", "aex"))
    F = sum(f * A for _, _, f, A, _, _, _, _ in parts)
    a = sum(f * A * aa for _, _, f, A, aa, _, _, _ in parts) / F
    w.guard(h - a, h0)
    if len(parts) == 1:
        w.eq([("h_{0}", f"h - a_{{s}} = {v('h', h)} - {v('as', a)} = {v('h0', h0)}\\,\\u{{mm}}")])
        return
    num_s = " + ".join(f"{fs}{asym}" for fs, asym, *_ in parts)
    den_s = " + ".join(fs for fs, *_ in parts)
    num_v = " + ".join(f"{v(kf, f)} × {v(kA, A)} × {v(ka, aa)}" for _, _, f, A, aa, kf, kA, ka in parts)
    den_v = " + ".join(f"{v(kf, f)} × {v(kA, A)}" for _, _, f, A, aa, kf, kA, ka in parts)
    w.eq([("a", f"\\frac{{{num_s}}}{{{den_s}}} = \\frac{{{num_v}}}{{{den_v}}} = {v('a', a)}\\,\\u{{mm}}"),
          ("h_{0}", f"h - a = {v('h', h)} - {v('a', a)} = {v('h0', h0)}\\,\\u{{mm}}")])


def _tpl_flexure(w: _W) -> bool:
    chk, s = w.chk, w.s
    d = chk.details
    if s is None or s.kind not in ("rect", "tee", "box"):
        return False
    if any(f.startswith("over-reinforced") for f in chk.failed) or "clause_5_2_7" in d:
        return _flexure_over(w)
    if chk.failed:
        return False
    g = _steel(s)
    fl = _flange(s)
    if not g.edge_only or fl is None:
        return False
    bf, hf, b = fl
    c, st, t = s.concrete, s.steel, s.tendon
    fcd, fsd, fsdc = c.fcd, st.fsd, st.fsd_c
    fpd = t.fpd if (t is not None and g.Ap) else 0.0
    fpdc = t.fpd_c if (t is not None and g.Apc) else 0.0
    h0 = d["h0"]
    T = fsd * g.As + fpd * g.Ap + g.Fex
    Cs = fsdc * g.Asc + (fpdc - g.sp0c) * g.Apc
    flange = b is not None and bf > b + 1e-9
    if b is None:
        b = bf
    in_fl = T - Cs <= fcd * bf * hf + 1e-9 or not flange
    x = (T - Cs) / (fcd * bf) if in_fl else (T - Cs - fcd * (bf - b) * hf) / (fcd * b)
    aref, comp_p = s.comp_ref()
    has_c = (g.Asc > 0 or (g.Apc > 0 and comp_p)) and aref is not None
    if (x <= 0 and not has_c) or (not flange and x > hf + 1e-9):
        return False
    no_x = x <= 0  # the compression steel alone balances the tension steel: x < 2a' whatever the concrete
    case_b = has_c and x < 2 * aref
    if case_b != bool(chk.notes and any("2a'" in n for n in chk.notes)):
        return False
    asym = "a'" if comp_p else "a'_{s}"  # 5.2.2-4 with A'p in compression, else 5.2.2-5

    def conc_M(xx, inside):
        if inside:
            return fcd * bf * xx * (h0 - xx / 2)
        return fcd * (b * xx * (h0 - xx / 2) + (bf - b) * hf * (h0 - hf / 2))
    x1 = None
    Msingle = 0.0
    if not case_b:
        Mu = conc_M(x, in_fl) + fsdc * g.Asc * (h0 - g.asc) + (fpdc - g.sp0c) * g.Apc * (h0 - g.apc)
    else:
        Mab = s.design_tension_moment(aref, comp_p)
        if not g.Apc:
            in1 = T <= fcd * bf * hf + 1e-9 or not flange
            x1 = T / (fcd * bf) if in1 else (T - fcd * (bf - b) * hf) / (fcd * b)
            if x1 <= d["xi_b"] * h0:
                Msingle = conc_M(x1, in1)
        Mu = max(Mab, Msingle)
    w.guard(Mu / 1e6, chk.resistance)
    if case_b:
        w.hide_notes = [n_ for n_ in chk.notes if "2a'" in n_]
        w.plain_figure = True  # the drawing's zone is the strain solution's x, not the x written here
        w.clause = chk.clause.replace("5.2.2 / 5.2.3", "5.2.2 / 5.2.3 / 5.2.4")  # 5.2.6 is already in 5.2.5 / 5.2.6
    v, tt, n = w.v, w.t, w.n
    w.units()
    ten_s = [("f_{sd}A_{s}", f"{v('fsd', fsd)} × {v('As', g.As)}")]
    if g.Ap:
        ten_s.append(("f_{pd}A_{p}", f"{v('fpd', fpd)} × {v('Ap', g.Ap)}"))
    if g.Fex:
        ten_s.append(("σ_{pe,ex}A_{ex}", f"{v('sex', g.sex)} × {v('Aex', g.Aex)}"))
    com_s = []
    if g.Asc:
        com_s.append(("f'_{sd}A'_{s}", f"{v('fsdc', fsdc)} × {v('Asc', g.Asc)}"))
    if g.Apc:
        com_s.append(("(f'_{pd} - σ'_{p0})A'_{p}", f"({v('fpdc', fpdc)} - {v('sp0c', g.sp0c)}) × {v('Apc', g.Apc)}"))
    sym_T, val_T = " + ".join(a for a, _ in ten_s), " + ".join(b_ for _, b_ in ten_s)
    sym_C, val_C = " + ".join(a for a, _ in com_s), " + ".join(b_ for _, b_ in com_s)
    k = 1
    w.head(tt(f"（{k}）截面有效高度", f"({k}) Effective depth"))
    k += 1
    _h0_rows(w, g, s, fsd, fpd, h0)
    if has_c and comp_p and g.Asc:
        Fs, Fp = fsdc * g.Asc, (fpdc - g.sp0c) * g.Apc
        a_ = (Fs * g.asc + Fp * g.apc) / (Fs + Fp)
        w.guard(a_, aref)
        w.eq([("a'", f"\\frac{{f'_{{sd}}A'_{{s}}a'_{{s}} + (f'_{{pd}} -"
                     f" σ'_{{p0}})A'_{{p}}a'_{{p}}}}{{f'_{{sd}}A'_{{s}} + "
                     f"(f'_{{pd}} - σ'_{{p0}})A'_{{p}}}} = {v('aref', aref)}\\,\\u{{mm}}")])
    if flange:
        tag = "5.2.5" if g.Fex else "5.2.3"
        w.head(tt(f"（{k}）判别 T 形截面类型（{tag}）", f"({k}) Flange or web ({tag})"))
        k += 1
        Cf = fcd * bf * hf + Cs
        rhs_sym = "f_{cd}b'_{f}h'_{f}" + (f" + {sym_C}" if com_s else "")
        rhs_val = f"{v('fcd', fcd)} × {v('bf', bf)} × {v('hf', hf)}" + (f" + {val_C}" if com_s else "")
        w.eq([(sym_T, f"{val_T} = {n(T / 1e3, 1)}\\,\\u{{kN}}"),
              w.cmp(T, Cf, f"{rhs_sym} = {rhs_val} = {n(Cf / 1e3, 1)}\\,\\u{{kN}}")])
        if in_fl:
            w.p(tt(f"受压区在翼缘内，按宽度 $b'_{{f}}$ = {_nv(bf)} mm 的矩形截面计算（{tag}-1）。" if not g.Fex else
                   f"受压区在翼缘内，按宽度 $b'_{{f}}$ = {_nv(bf)} mm 的矩形截面计算（5.2.5 第 2 款 1）。",
                   f"The compression zone lies in the flange: a rectangle $b'_{{f}}$ = {_nv(bf)} mm wide."))
        else:
            w.p(tt("受压区进入腹板，按 T 形截面计算。", "The compression zone reaches into the web."))
    w.head(tt(f"（{k}）受压区高度", f"({k}) Depth of the compression zone"))
    k += 1
    if no_x:
        w.eq([(sym_T, f"{val_T} = {n(T / 1e3, 1)}\\,\\u{{kN}}"),
              w.cmp(T, Cs, f"{sym_C} = {val_C} = {n(Cs / 1e3, 1)}\\,\\u{{kN}}")], tag="5.2.2-2")
        w.p(tt(f"受拉钢筋的拉力不大于受压钢筋按抗压强度设计值的压力，受压钢筋达不到抗压强度设计值，$x < 2{asym}$，"
               f"按 {'5.2.6' if g.Fex else '5.2.4'} 条对受压钢筋合力点取矩计算。",
               "The tension steel's force does not exceed the compression steel's at its design strength: x < 2a′; "
               "moments about the compression steel (5.2.4)."))
    elif in_fl:
        num_sym = sym_T + (f" - ({sym_C})" if len(com_s) > 1 else (f" - {sym_C}" if com_s else ""))
        num_val = val_T + (f" - ({val_C})" if len(com_s) > 1 else (f" - {val_C}" if com_s else ""))
        bsym = "f_{cd}b'_{f}" if flange else "f_{cd}b"
        bval = f"{v('fcd', fcd)} × {v('bf' if flange else 'b', bf)}"
        tag = ("5.2.5-4" if flange else "5.2.5-2") if g.Fex else "5.2.2-2"
        w.eq([("x", f"\\frac{{{num_sym}}}{{{bsym}}}"), ("", f"\\frac{{{num_val}}}{{{bval}}}"),
              ("", f"{v('x', x, 1)}\\,\\u{{mm}}")], tag=tag)
    else:
        num_sym = sym_T + (f" - {sym_C}" if com_s else "") + " - f_{cd}(b'_{f} - b)h'_{f}"
        num_val = val_T + (f" - {val_C}" if com_s else "") + \
            f" - {v('fcd', fcd)} × ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)}"
        w.eq([("x", f"\\frac{{{num_sym}}}{{f_{{cd}}b}}"), ("", f"\\frac{{{num_val}}}{{{v('fcd', fcd)} × {v('b', b)}}}"),
              ("", f"{v('x', x, 1)}\\,\\u{{mm}}")], tag="5.2.5-6" if g.Fex else "5.2.3-3")
    xb = d["xi_b"]
    if not no_x:
        w.eq([("ξ", f"\\frac{{x}}{{h_{{0}}}} = \\frac{{{v('x', x, 1)}}}{{{v('h0', h0)}}} = {n(x / h0, 3)}"),
              w.cmp(x / h0, xb, f"ξ_{{b}} = {v('xib', xb)}")], tag="5.2.2-3")
        if not has_c:
            w.p(tt("无受压钢筋" + ("（$A'_{p}$ 受拉）" if g.Apc else "") + "，不需验算 $x >= 2a'$。",
                   "No compression steel" + (" (A′p in tension)" if g.Apc else "") + ": x ≥ 2a′ does not apply."))
        else:
            w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"),
                  w.cmp(x, 2 * aref, f"2{asym} = 2 × {v('aref', aref)} = {n(2 * aref)}\\,\\u{{mm}}", ge=True)],
                 tag="5.2.2-4" if comp_p else "5.2.2-5")
            if case_b:
                w.p(tt(f"不满足 $x >= 2{asym}$，按 {'5.2.6' if g.Fex else '5.2.4'} 条对受压钢筋合力点取矩计算。",
                       "x < 2a′: moments about the compression steel (5.2.4)."))
    w.head(tt(f"（{k}）正截面抗弯承载力", f"({k}) Flexural resistance"))
    if not case_b:
        if in_fl:
            csym = "f_{cd}" + ("b'_{f}" if flange else "b") + "x(h_{0} - \\frac{x}{2})"
            cval = f"{v('fcd', fcd)} × {v('bf' if flange else 'b', bf)} × {v('x', x, 1)} × ({v('h0', h0)} - " \
                   f"\\frac{{{v('x', x, 1)}}}{{2}})"
        else:
            csym = "f_{cd}[bx(h_{0} - \\frac{x}{2}) + (b'_{f} - b)h'_{f}(h_{0} - \\frac{h'_{f}}{2})]"
            cval = (f"{v('fcd', fcd)} × [{v('b', b)} × {v('x', x, 1)} × ({v('h0', h0)} - \\frac{{{v('x', x, 1)}}}{{2}})"
                    f" + ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)} × ({v('h0', h0)} -"
                    f" \\frac{{{v('hf', hf)}}}{{2}})]")
        msym, mval = [csym], [cval]
        if g.Asc:
            msym.append("f'_{sd}A'_{s}(h_{0} - a'_{s})")
            mval.append(f"{v('fsdc', fsdc)} × {v('Asc', g.Asc)} × ({v('h0', h0)} - {v('asc', g.asc)})")
        if g.Apc:
            msym.append("(f'_{pd} - σ'_{p0})A'_{p}(h_{0} - a'_{p})")
            mval.append(f"({v('fpdc', fpdc)} - {v('sp0c', g.sp0c)}) × {v('Apc', g.Apc)} × ({v('h0', h0)} - "
                        f"{v('apc', g.apc)})")
        tag = ("5.2.5-3" if (flange and in_fl) else "5.2.5-5" if flange else "5.2.5-1") if g.Fex else \
            ("5.2.3-2" if (flange and not in_fl) else "5.2.2-1")
        w.eq([("M_{u}", " + ".join(msym)), ("", _join(mval)), ("", f"{v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")], tag=tag)
    else:
        h = s.h
        terms_s, terms_v = [], []
        if g.Ap:
            terms_s.append(f"f_{{pd}}A_{{p}}(h - a_{{p}} - {asym})")
            terms_v.append(f"{v('fpd', fpd)} × {v('Ap', g.Ap)} × ({v('h', h)} - {v('ap', h - g.dp)} - "
                           f"{v('aref', aref)})")
        terms_s.append(f"f_{{sd}}A_{{s}}(h - a_{{s}} - {asym})")
        terms_v.append(f"{v('fsd', fsd)} × {v('As', g.As)} × ({v('h', h)} - {v('as', h - g.ds)} - "
                       f"{v('aref', aref)})")
        if g.Fex:
            terms_s.append(f"σ_{{pe,ex}}A_{{ex}}(h - a_{{p,ex}} - {asym})")
            terms_v.append(f"{v('sex', g.sex)} × {v('Aex', g.Aex)} × ({v('h', h)} - {v('aex', h - g.dex)} - "
                           f"{v('aref', aref)})")
        if g.Apc and not comp_p:
            terms_s.append("- (f'_{pd} - σ'_{p0})A'_{p}(a'_{p} - a'_{s})")
            terms_v.append(f"- ({v('fpdc', fpdc)} - {v('sp0c', g.sp0c)}) × {v('Apc', g.Apc)} × "
                           f"({v('apc', g.apc)} - {v('asc', g.asc)})")
        tag = ("5.2.6-1" if comp_p else "5.2.6-2") if g.Fex else ("5.2.4-1" if comp_p else "5.2.4-2")
        w.eq([("M_{u1}", _join(terms_s)), ("", _join(terms_v)), ("", f"{v('Mu1', Mab / 1e6, 1)}\\,\\u{{kN·m}}")],
             tag=tag)
        if x1 is not None:
            in1 = T <= fcd * bf * hf + 1e-9 or not flange
            w.p(tt("受压区没有预应力钢筋，另按不计受压普通钢筋的单筋截面计算（程序约定，二者取大值）：",
                   "Without compression tendons the section is also taken as singly reinforced (program convention: "
                   "the larger governs):"))
            bs = "b'_{f}" if flange else "b"
            if in1:
                w.eq([("x_{1}", f"\\frac{{{sym_T}}}{{f_{{cd}}{bs}}} = \\frac{{{val_T}}}{{{v('fcd', fcd)} × "
                                f"{v('bf' if flange else 'b', bf)}}} = {v('x1', x1, 1)}\\,\\u{{mm}}"),
                      w.cmp(x1, xb * h0, f"ξ_{{b}}h_{{0}} = {n(xb * h0, 1)}\\,\\u{{mm}}")])
            else:
                w.eq([("x_{1}", f"\\frac{{{sym_T} - f_{{cd}}(b'_{{f}} - b)h'_{{f}}}}{{f_{{cd}}b}} = {v('x1', x1, 1)}"
                                "\\,\\u{mm}"), w.cmp(x1, xb * h0, f"ξ_{{b}}h_{{0}} = {n(xb * h0, 1)}\\,\\u{{mm}}")])
            if Msingle:
                if in1:
                    w.eq([("M_{u2}", f"f_{{cd}}{bs}x_{{1}}(h_{{0}} - \\frac{{x_{{1}}}}{{2}})"),
                          ("", f"{v('fcd', fcd)} × {v('bf' if flange else 'b', bf)} × {v('x1', x1, 1)} × "
                               f"({v('h0', h0)} - \\frac{{{v('x1', x1, 1)}}}{{2}})"),
                          ("", f"{v('Mu2', Msingle / 1e6, 1)}\\,\\u{{kN·m}}")])
                else:
                    w.eq([("M_{u2}", f"f_{{cd}}[bx_{{1}}(h_{{0}} - \\frac{{x_{{1}}}}{{2}}) + (b'_{{f}} - b)h'_{{f}}"
                                     f"(h_{{0}} - \\frac{{h'_{{f}}}}{{2}})] ="
                                     f" {v('Mu2', Msingle / 1e6, 1)}\\,\\u{{kN·m}}")])
            w.eq([("M_{u}", f"\\op{{max}}(M_{{u1}}, M_{{u2}}) = {v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")])
        else:
            w.eq([("M_{u}", f"M_{{u1}} = {v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")])
    _flexure_minimums(w, g, h0)
    return True


def _flexure_minimums(w: _W, g: _Steel, h0: float):
    """9.1.12 (reinforced concrete) and 9.1.13 with Mcr (prestressed)"""
    chk, d, v, n = w.chk, w.chk.details, w.v, w.n
    if "rho" in d:
        rho = g.As / (d["b"] * h0)
        w.guard(rho, d["rho"])
        w.eq([("ρ", f"\\frac{{A_{{s}}}}{{bh_{{0}}}} = \\frac{{{v('As', g.As)}}}{{{v('bweb', d['b'])} × {v('h0', h0)}}}"
                    f" = {n(100 * rho, 3)}\\%"),
              w.cmp(rho, d["rho_min"], f"ρ_{{min}} = \\op{{max}}(45f_{{td}}/f_{{sd}}, 0.20)\\% = "
                                       f"{n(100 * d['rho_min'], 3)}\\%", ge=True)], tag="9.1.12")
    if "Mcr" in d and isinstance(d.get("cracking"), dict):
        cr = d["cracking"]
        ftk = w.s.concrete.ftk
        gam = 2 * cr["S0"] / cr["W0"]
        Mcr = (cr["sigma_pc"] + gam * ftk) * cr["W0"] / 1e6
        w.guard(gam, cr["gamma"])
        w.guard(Mcr, d["Mcr"])
        w.eq([("γ", "=", f"\\frac{{2S_{{0}}}}{{W_{{0}}}} = \\frac{{2 ×"
                         f" {w.sv('S0', cr['S0'])}}}{{{w.sv('W0', cr['W0'])}}} = "
                         f"{v('gamma', gam)}", "6.5.2-8"),
              ("M_{cr}", "=", f"(σ_{{pc}} + γf_{{tk}})W_{{0}} = ({v('spc', cr['sigma_pc'])} + {v('gamma', gam)} × "
                              f"{v('ftk', ftk)}) × {w.sv('W0', cr['W0'])} = {v('Mcr', Mcr, 1)}\\,\\u{{kN·m}}",
                              "6.5.2-7")])
        w.eq([("\\frac{M_{ud}}{M_{cr}}", f"\\frac{{{n(chk.resistance, 1)}}}{{{n(Mcr, 1)}}} ="
                                         f" {n(chk.resistance / Mcr, 3)}"),
              w.cmp(chk.resistance / Mcr, 1.0, "1.0", ge=True)], tag="9.1.13")


def _flexure_over(w: _W) -> bool:
    """an over-reinforced section (5.2.2-3 not met): x by ΣN = 0 with the tension steel by 5.1.5-1; with
    uls_steel_only (5.2.7) the resistance with the tension steel cut to x = ξb h0"""
    chk, s, d = w.chk, w.s, w.chk.details
    if chk.resistance != chk.resistance:
        return False
    g = _steel(s)
    fl = _flange(s)
    if not g.edge_only or fl is None or g.Ap or g.Apc or g.Fex or g.tension_rows != 1:
        return False
    bf, hf, b = fl
    b = bf if b is None else b
    c, st = s.concrete, s.steel
    x, h0, xb = d["x"], d["h0"], d["xi_b"]
    v, n, tt = w.v, w.n, w.t
    bal = d.get("clause_5_2_7")
    if bal is not None and _close(chk.resistance, bal["Mu_balanced"]):  # uls_steel_only: Mu,b at x = ξb h0
        xbh = xb * h0
        inside = xbh <= hf + 1e-9 or bf == b
        aref, _ = s.comp_ref()
        sc = st.fsd_c if (g.Asc and xbh >= 2 * aref) else 0.0
        if g.Asc and not sc:
            return False
        Mc = c.fcd * (bf * xbh * (h0 - xbh / 2) if inside else b * xbh * (h0 - xbh / 2) + (bf - b) * hf * (h0 - hf / 2))
        Mb = Mc + sc * g.Asc * (h0 - g.asc)
        w.guard(Mb / 1e6, chk.resistance)
        w.units()
        w.p(tt("全部受拉钢筋时受压区高度 ", "With all the tension steel "), m(f"x = {v('x', x, 1)}\\,\\u{{mm}}"),
            tt("，", ", "), m(f"ξ = {n(x / h0, 3)} {w.rel(x / h0, xb)} ξ_{{b}} = {v('xib', xb)}"),
            tt("，超筋。按 5.2.7 条只计承载能力极限状态所需的受拉钢筋，取界限受压区高度计算承载力：",
               ": over-reinforced. By 5.2.7 only the tension steel the ultimate limit state needs counts; the"
               " resistance "
               "at the balanced depth:"))
        bsym = "b'_{f}" if (inside and bf != b) else "b"
        w.eq([("x_{b}", f"ξ_{{b}}h_{{0}} = {v('xib', xb)} × {v('h0', h0)} = {v('xb', xbh, 1)}\\,\\u{{mm}}")])
        sym = f"f_{{cd}}{bsym}x_{{b}}(h_{{0}} - \\frac{{x_{{b}}}}{{2}})" if inside else \
            "f_{cd}[bx_{b}(h_{0} - \\frac{x_{b}}{2}) + (b'_{f} - b)h'_{f}(h_{0} - \\frac{h'_{f}}{2})]"
        val = (f"{v('fcd', c.fcd)} × {v('bf' if bsym != 'b' else 'b', bf)} × {v('xb', xbh, 1)} × ({v('h0', h0)} - "
               f"\\frac{{{v('xb', xbh, 1)}}}{{2}})") if inside else \
            (f"{v('fcd', c.fcd)} × [{v('b', b)} × {v('xb', xbh, 1)} × ({v('h0', h0)} -"
             f" \\frac{{{v('xb', xbh, 1)}}}{{2}}) + "
             f"({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)} × ({v('h0', h0)} - \\frac{{{v('hf', hf)}}}{{2}})]")
        if sc:
            sym += " + f'_{sd}A'_{s}(h_{0} - a'_{s})"
            val += f" + {v('fsdc', sc)} × {v('Asc', g.Asc)} × ({v('h0', h0)} - {v('asc', g.asc)})"
        w.eq([("M_{u,b}", sym), ("", val), ("", f"{v('Mu', Mb / 1e6, 1)}\\,\\u{{kN·m}}")], tag="5.2.7")
        w.plain_figure = True
        return True
    inside = x <= hf + 1e-9 or bf == b
    ss = float(np.clip(c.eps_cu * st.Es * (c.beta * h0 / x - 1), -st.fsd_c, st.fsd))
    if g.Asc:
        aref, _ = s.comp_ref()
        if x >= 2 * aref:
            sc = st.fsd_c
        else:
            sc = float(np.clip(c.eps_cu * st.Es * (1 - c.beta * g.asc / x), -st.fsd, st.fsd_c))
    else:
        sc = 0.0
    C = c.fcd * (bf * x if inside else b * x + (bf - b) * hf)
    if abs(C + sc * g.Asc - ss * g.As) > 1e-6 * C:
        return False
    Mc = c.fcd * (bf * x * (h0 - x / 2) if inside else b * x * (h0 - x / 2) + (bf - b) * hf * (h0 - hf / 2))
    Mu = Mc + sc * g.Asc * (h0 - g.asc)
    w.guard(Mu / 1e6, chk.resistance)
    w.units()
    w.head(tt("（1）受压区高度（受拉钢筋应力按 5.1.5-1）", "(1) Depth of the compression zone (tension steel by 5.1.5-1)"))
    w.eq([("σ_{s}", f"ε_{{cu}}E_{{s}}(\\frac{{βh_{{0}}}}{{x}} - 1) = {v('ecu', c.eps_cu)} × {v('Es', st.Es)} × "
                    f"(\\frac{{{v('beta', c.beta)} × {v('h0', h0)}}}{{{v('x', x, 1)}}} - 1) ="
                    f" {v('ss', ss)}\\,\\u{{MPa}}")],
         tag="5.1.5-1")
    bsym = "b'_{f}" if (inside and bf != b) else "b"
    csym = f"f_{{cd}}{bsym}x" if inside else "f_{cd}[bx + (b'_{f} - b)h'_{f}]"
    cval = f"{v('fcd', c.fcd)} × {v('bf' if bsym != 'b' else 'b', bf)} × {v('x', x, 1)}" if inside else \
        f"{v('fcd', c.fcd)} × [{v('b', b)} × {v('x', x, 1)} + ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)}]"
    w.p(tt("由截面轴力平衡 ΣN = 0 解得受压区高度（受拉钢筋未屈服）：", "From ΣN = 0 (the tension steel below yield):"))
    lhs = csym + (" + σ'_{s}A'_{s}" if g.Asc else "")
    lval = cval + (f" + {v('sc', sc)} × {v('Asc', g.Asc)}" if g.Asc else "")
    w.eq([(lhs, f"{lval} = {n(C / 1e3 + sc * g.Asc / 1e3, 1)}\\,\\u{{kN}}"),
          ("σ_{s}A_{s}", f"{v('ss', ss)} × {v('As', g.As)} = {n(ss * g.As / 1e3, 1)}\\,\\u{{kN}}")])
    w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"),
          ("ξ", f"\\frac{{x}}{{h_{{0}}}} = \\frac{{{v('x', x, 1)}}}{{{v('h0', h0)}}} = {n(x / h0, 3)}"),
          w.cmp(x / h0, xb, f"ξ_{{b}} = {v('xib', xb)}")], tag="5.2.2-3")
    w.p(tt("不满足式（5.2.2-3），为超筋截面。下面的承载力按受拉钢筋未屈服时的应力求得，仅供参考。",
           "5.2.2-3 is not met: over-reinforced. The resistance below takes the tension steel below yield, for "
           "information."))
    w.head(tt("（2）正截面抗弯承载力", "(2) Flexural resistance"))
    sym = f"f_{{cd}}{bsym}x(h_{{0}} - \\frac{{x}}{{2}})" if inside else \
        "f_{cd}[bx(h_{0} - \\frac{x}{2}) + (b'_{f} - b)h'_{f}(h_{0} - \\frac{h'_{f}}{2})]"
    val = (f"{v('fcd', c.fcd)} × {v('bf' if bsym != 'b' else 'b', bf)} × {v('x', x, 1)} × ({v('h0', h0)} - "
           f"\\frac{{{v('x', x, 1)}}}{{2}})") if inside else \
        (f"{v('fcd', c.fcd)} × [{v('b', b)} × {v('x', x, 1)} × ({v('h0', h0)} - \\frac{{{v('x', x, 1)}}}{{2}}) + "
         f"({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)} × ({v('h0', h0)} - \\frac{{{v('hf', hf)}}}{{2}})]")
    if g.Asc:
        sym += " + σ'_{s}A'_{s}(h_{0} - a'_{s})"
        val += f" + {v('sc', sc, paren=True)} × {v('Asc', g.Asc)} × ({v('h0', h0)} - {v('asc', g.asc)})"
    w.eq([("M_{u}", sym), ("", val), ("", f"{v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")])
    return True


# ── shear ───────────────────────────────────────────────────────────────────
def _tpl_shear(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None or "Vcs" not in d or "Asv" not in d:
        return False
    c = s.concrete
    b, h0, P = d["b"], d["h0"], d["P"]
    P_raw = 100 * (d["As"] + d["Ap"]) / (b * h0)
    w.guard(min(P_raw, 2.5), P)
    rho_pv = d["rho_pv"]
    K = (2 + 0.6 * P) * math.sqrt(c.fcuk) * (d["rho_sv"] * d["fsv"] + 0.6 * rho_pv * d["fpv"])
    Vcs = 0.45e-3 * d["alpha1"] * d["alpha2"] * d["alpha3"] * b * h0 * math.sqrt(K)
    w.guard(Vcs, d["Vcs"])
    w.guard(Vcs + d["Vsb"] + d["Vpb"] + d["Vpb_ex"], chk.resistance)
    v, tt, n = w.v, w.t, w.n
    w.units()
    w.head(tt("（1）计算参数", "(1) Parameters"))
    Ap_t = f" + {v('Ap', d['Ap'])}" if d["Ap"] else ""
    Ap_s = " + A_{p}" if d["Ap"] else ""
    head = (f"100\\frac{{A_{{s}}{Ap_s}}}{{bh_{{0}}}} = "
            f"100 × \\frac{{{v('As', d['As'])}{Ap_t}}}{{{v('b', b)} × {v('h0', h0)}}}")
    if P_raw > 2.5:
        w.eq([("P", f"{head} = {n(P_raw, 3)}"),
              ("", ">", "2.5" + w.take("，取 ", ", P = ") + ("P = " if w.zh else "") + "\\v{P}{2.5}")])
    else:
        w.eq([("P", f"{head} = {v('P', P)}")])
    if _close(d["rho_sv"], d["rho_sv_provided"]):
        w.eq([("ρ_{sv}", f"\\frac{{A_{{sv}}}}{{s_{{v}}b}} = \\frac{{{d['legs']} × π × {_nv(d['dia'])}^{{2}}/4}}"
                         f"{{{v('sv', d['sv'])} × {v('b', b)}}} = \\frac{{{v('Asv', d['Asv'])}}}{{{v('sv', d['sv'])} × "
                         f"{v('b', b)}}} = {v('rho_sv', d['rho_sv'])}")])
    else:
        w.p(tt("斜截面跨越两个箍筋区段，配箍率取其范围内的平均值：", "The inclined section crosses two stirrup zones; "
               "the mean stirrup ratio over it:"), m(f"ρ_{{sv}} = {v('rho_sv', d['rho_sv'])}"), tt("。", "."))
    if d.get("stirrup_fsd", d["fsv"]) > d["fsv"] + 1e-9:
        w.eq([("f_{sv}", f"f_{{sd}} = {v('fsdv', d['stirrup_fsd'])}\\,\\u{{MPa}}"),
              ("", ">", "330" + w.take("，取 ", ", ") + f"f_{{sv}} = {v('fsv', d['fsv'])}\\,\\u{{MPa}}")],
             tag="5.2.9")
    a3_txt = tt("受压翼缘", "compression flange") if d["alpha3"] > 1.0 else tt("矩形或受拉翼缘", "no compression flange")
    sep, end = _L[w.lang]["sep"], _L[w.lang]["end"]
    w.p(tt("系数：", "Factors: "), m(f"α_{{1}} = {v('a1', d['alpha1'])}"), sep, m(f"α_{{2}} = {v('a2', d['alpha2'])}"),
        sep,
        m(f"α_{{3}} = {v('a3', d['alpha3'])}"), f"（{a3_txt}）" if w.zh else f" ({a3_txt})",
        _L[w.lang]["semi"], m(f"f_{{cu,k}} = {v('fcuk', c.fcuk)}\\,\\u{{MPa}}"), sep,
        m(f"f_{{sv}} = {v('fsv', d['fsv'])}\\,\\u{{MPa}}"), end)
    w.head(tt("（2）斜截面抗剪承载力（5.2.9）", "(2) Shear resistance (5.2.9)"))
    pv_s = " + 0.6ρ_{pv}f_{pv}" if rho_pv else ""
    pv_v = f" + 0.6 × {v('rho_pv', rho_pv)} × {v('fpv', d['fpv'])}" if rho_pv else ""
    w.eq([("V_{cs}", f"0.45×10^{{-3}}α_{{1}}α_{{2}}α_{{3}}bh_{{0}}\\sqrt{{(2 + 0.6P)\\sqrt{{f_{{cu,k}}}}"
                     f"(ρ_{{sv}}f_{{sv}}{pv_s})}}")], tag="5.2.9-2")
    w.eq([("(2 + 0.6P)\\sqrt{f_{cu,k}}(ρ_{sv}f_{sv}" + pv_s + ")",
           f"(2 + 0.6 × {v('P', P)}) × \\sqrt{{{v('fcuk', c.fcuk)}}} × ({v('rho_sv', d['rho_sv'])} × "
           f"{v('fsv', d['fsv'])}{pv_v}) = {v('K', K)}"),
          ("V_{cs}", f"0.45×10^{{-3}} × {v('a1', d['alpha1'])} × {v('a2', d['alpha2'])} × {v('a3', d['alpha3'])} × "
                     f"{v('b', b)} × {v('h0', h0)} × \\sqrt{{{v('K', K)}}} = {v('Vcs', Vcs, 1)}\\,\\u{{kN}}")])
    parts = [("V_{cs}", Vcs)]
    st = s.steel
    if d["Vsb"]:
        rows = " + ".join(f"{_nv(A)} × \\op{{sin}}{_nv(th)}°" for A, th in d["bent"])
        Vsb = sum(0.75e-3 * st.fsd * A * math.sin(math.radians(th)) for A, th in d["bent"])
        w.guard(Vsb, d["Vsb"])
        w.eq([("V_{sb}", "0.75×10^{-3}f_{sd}ΣA_{sb}\\op{sin}θ_{s}"),
              ("", f"0.75×10^{{-3}} × {v('fsd', st.fsd)} × ({rows}) = {v('Vsb', d['Vsb'], 1)}\\,\\u{{kN}}")],
             tag="5.2.9-3")
        parts.append(("V_{sb}", d["Vsb"]))
    if d["Vpb"]:
        rows = " + ".join(f"{_nv(A)} × \\op{{sin}}{_nv(th)}°" for A, th in d["bent_tendons"])
        w.eq([("V_{pb}", "0.75×10^{-3}f_{pd}ΣA_{pb}\\op{sin}θ_{p}"),
              ("", f"0.75×10^{{-3}} × {v('fpd', s.tendon.fpd)} × ({rows}) = {v('Vpb', d['Vpb'], 1)}\\,\\u{{kN}}")],
             tag="5.2.9-4")
        parts.append(("V_{pb}", d["Vpb"]))
    if d["Vpb_ex"]:
        w.eq([("V_{pb,ex}", f"0.75×10^{{-3}}Σσ_{{pe,ex}}A_{{ex}}\\op{{sin}}θ_{{ex}} = {v('Vpbex', d['Vpb_ex'], 1)}"
                            "\\,\\u{kN}")], tag="5.2.9-5")
        parts.append(("V_{pb,ex}", d["Vpb_ex"]))
    if len(parts) > 1:
        w.eq([("V_{u}", " + ".join(p for p, _ in parts) + " = " + " + ".join(n(val, 1) for _, val in parts)
               + f" = {v('Vu', chk.resistance, 1)}\\,\\u{{kN}}")], tag="5.2.9-1")
    else:
        w.eq([("V_{u}", f"V_{{cs}} = {v('Vu', chk.resistance, 1)}\\,\\u{{kN}}")], tag="5.2.9-1")
    share = re.match(r"shear share ξ = ([\d.]+)", chk.name)
    if share:  # a design back-substitution: the stirrups and concrete for ξγ0V'd; the limits below for γ0V'd
        w.p(tt(f"上式抗力与 ξ$γ_{{0}}V'_{{d}}$ 比较（ξ = {share.group(1)}，混凝土和箍筋分担的部分）；截面尺寸和构造配箍条件按全部剪力 "
               "$γ_{0}V'_{d}$ 判断。", f"The resistance above is compared with ξ$γ_{{0}}V'_{{d}}$ (ξ = {share.group(1)},"
                                    f" the "
               "share of the concrete and stirrups); the section limit and the threshold take the whole"
               " $γ_{0}V'_{d}$."))
    w.head(tt("（3）截面尺寸和构造配箍条件", "(3) Section limit and the detailing threshold"))
    dem = abs(chk.demand) / float(share.group(1)) if share else abs(chk.demand)
    dsym = "γ_{0}V'_{d}" if share else "γ_{0}V_{d}"
    vmax, vmin = d["V_section_max"], d["V_detailing"]
    w.guard(0.51e-3 * math.sqrt(c.fcuk) * b * h0, vmax)
    w.guard(0.50e-3 * d["alpha2"] * c.ftd * b * h0, vmin)
    w.eq([(dsym, f"{n(dem, 1)}\\,\\u{{kN}}"),
          w.cmp(dem, vmax, f"0.51×10^{{-3}}\\sqrt{{f_{{cu,k}}}}bh_{{0}} = 0.51×10^{{-3}} ×"
                           f" \\sqrt{{{v('fcuk', c.fcuk)}}} × "
                           f"{v('b', b)} × {v('h0', h0)} = {v('Vmax', vmax, 1)}\\,\\u{{kN}}")], tag="5.2.11")
    w.eq([(dsym, f"{n(dem, 1)}\\,\\u{{kN}}"),
          w.cmp(dem, vmin, f"0.50×10^{{-3}}α_{{2}}f_{{td}}bh_{{0}} = 0.50×10^{{-3}} × {v('a2', d['alpha2'])} × "
                           f"{v('ftd', c.ftd)} × {v('b', b)} × {v('h0', h0)} = {v('Vmin', vmin, 1)}\\,\\u{{kN}}")],
         tag="5.2.12")
    w.p(tt("截面尺寸满足 5.2.11 条。" if dem <= vmax else "截面尺寸不满足 5.2.11 条。",
           "The section limit of 5.2.11 is met." if dem <= vmax else "The section limit of 5.2.11 is not met."),
        tt(f"${dsym}$ 不大于 5.2.12 条的限值，可只按构造配置箍筋。" if dem <= vmin else
           f"${dsym}$ 大于 5.2.12 条的限值，箍筋按计算配置（上式）。",
           f" ${dsym}$ is within 5.2.12: detailing stirrups suffice." if dem <= vmin else
           f" ${dsym}$ exceeds 5.2.12: the stirrups are designed by calculation (above)."))
    w.hide_notes = [x for x in chk.notes if x.startswith("γ0Vd ≤")]
    return True


def _tpl_section_limit(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if w.sec is None or "V0d" not in d:
        return False
    c = w.sec.concrete
    R = 0.51e-3 * math.sqrt(c.fcuk) * d["b"] * d["h0"]
    w.guard(R, chk.resistance)
    v, n = w.v, w.n
    w.units()
    w.eq([("γ_{0}V_{0d}", f"{n(abs(chk.demand), 1)}\\,\\u{{kN}}"),
          w.cmp(abs(chk.demand), R, f"0.51×10^{{-3}}\\sqrt{{f_{{cu,k}}}}bh_{{0}} = 0.51×10^{{-3}} × "
                                    f"\\sqrt{{{v('fcuk', c.fcuk)}}} × {v('b', d['b'])} × {v('h0', d['h0'])} = "
                                    f"{v('R', R, 1)}\\,\\u{{kN}}")], tag="5.2.11")
    return True


def _tpl_slab_shear(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None:
        return False
    c = s.concrete
    R = 1.25 * 0.50e-3 * d["alpha2"] * c.ftd * d["b"] * d["h0"]
    w.guard(R, chk.resistance)
    v = w.v
    w.units()
    w.eq([("V_{u}", "1.25 × 0.50×10^{-3}α_{2}f_{td}bh_{0}"),
          ("", f"1.25 × 0.50×10^{{-3}} × {v('a2', d['alpha2'])} × {v('ftd', c.ftd)} × {v('b', d['b'])} × "
               f"{v('h0', d['h0'])} = {v('Vu', R, 1)}\\,\\u{{kN}}")], tag="5.2.12")
    w.p(w.t("程序约定：不配箍筋的板以式（5.2.12）右边乘 1.25 作为抗剪承载力（5.2.12 条注明板可不做斜截面抗剪验算的条件）。",
            "Program convention: a slab without stirrups takes 1.25 × the right-hand side of 5.2.12 as its shear "
            "resistance (5.2.12 gives it as the condition under which slabs need no shear check)."), )
    return True


# ── compression ─────────────────────────────────────────────────────────────
def _tpl_axial(w: _W) -> bool:
    chk, s, d = w.chk, w.sec, w.chk.details
    if s is None or "A_used" not in d:
        return False
    c, st = s.concrete, s.steel
    Nu = 0.9 * d["phi"] * (c.fcd * d["A_used"] + st.fsd_c * d["As"]) / 1e3
    w.guard(Nu, chk.resistance)
    v, tt, n = w.v, w.t, w.n
    w.units()
    if d.get("l0"):
        sym = {"b": "b", "d": "d", "i": "i"}[d["slenderness_by"]]
        lam = d["l0"] / d["slenderness_dim"]
        w.eq([(f"l_{{0}}/{sym}", f"{v('l0', d['l0'])}/{v('dim', d['slenderness_dim'])} = {n(lam, 2)}"),
              ("φ", f"{v('phi', d['phi'])}" + w.take("（表 5.3.1，线性内插）", " (table 5.3.1, interpolated)"))])
    else:
        w.p(tt("未给出计算长度，取 φ = 1.0。", "No effective length given: φ = 1.0."))
    if d["rho"] > 0.03:
        w.p(tt(f"纵向钢筋配筋率 {100 * d['rho']:.2f}% > 3%，A 取净截面面积 $A_{{n}} = A - A′_{{s}}$。",
               f"Longitudinal steel {100 * d['rho']:.2f} % > 3 %: A is the net area An = A − A's."))
    w.eq([("N_{u}", "0.90φ(f_{cd}A + f'_{sd}A'_{s})"),
          ("", f"0.90 × {v('phi', d['phi'])} × ({v('fcd', c.fcd)} × {v('A', d['A_used'])} + {v('fsdc', st.fsd_c)} × "
               f"{v('As', d['As'])})"), ("", f"{v('Nu', Nu, 1)}\\,\\u{{kN}}")], tag="5.3.1")
    return True


def _ecc_head(w: _W, d, s, l0, h0, circle: bool):
    v, tt, n = w.v, w.t, w.n
    w.hide_notes += [x for x in w.chk.notes if x.startswith("e0 = ") and "raised to the minimum" in x]
    w.head(tt("（1）初始偏心距", "(1) Initial eccentricity"))
    Md, Nd = abs(d["Md"]), d["Nd"]
    e0r = Md / Nd * 1e3
    w.guard(e0r, d["e0_from_forces"])
    lo = max(20.0, s.h / 30)
    if d["e0"] > e0r + 1e-9:
        w.eq([("e_{0}", f"\\frac{{M_{{d}}}}{{N_{{d}}}} = \\frac{{{v('Md', Md, 1)}}}{{{v('Nd', Nd, 1)}}} × 10^{{3}} = "
                        f"{n(e0r, 1)}\\,\\u{{mm}}"),
              w.cmp(e0r, lo, f"\\op{{max}}(20, h/30) = {n(lo, 1)}\\,\\u{{mm}}" + w.take("，取 ", ", ")
                    + f"e_{{0}} = {v('e0', d['e0'], 1)}\\,\\u{{mm}}", ge=True)], tag="5.3.9")
    else:
        w.eq([("e_{0}", f"\\frac{{M_{{d}}}}{{N_{{d}}}} = \\frac{{{v('Md', Md, 1)}}}{{{v('Nd', Nd, 1)}}} × 10^{{3}} = "
                        f"{v('e0', d['e0'], 1)}\\,\\u{{mm}}"),
              w.cmp(e0r, lo, f"\\op{{max}}(20, h/30) = {n(lo, 1)}\\,\\u{{mm}}", ge=True)], tag="5.3.9")
    w.head(tt("（2）偏心距增大系数", "(2) Moment magnifier"))
    eta = d["eta"]
    e0 = d["e0"]
    if eta == 1.0:
        if not l0:
            w.p(tt("未给出计算长度，取 ", "No effective length: "), m(f"η = {v('eta', 1.0)}"), tt("。", "."))
        else:
            i = s.i
            w.eq([("\\frac{l_{0}}{i}", f"\\frac{{{v('l0', l0)}}}{{{v('i', i)}}} = {n(l0 / i, 2)}"),
                  w.cmp(l0 / i, 17.5, "17.5" + w.take("，取 ", ", ") + f"η = {v('eta', 1.0)}")], tag="5.3.9")
    else:
        h = s.h
        z1r, z2r = 0.2 + 2.7 * e0 / h0, 1.15 - 0.01 * l0 / h
        z1, z2 = min(z1r, 1.0), min(z2r, 1.0)

        def cap(r, z, k):
            return (f"{n(r, 3)} > 1.0" + w.take("，取 ", ", ") + f"ζ_{{{k}}} = {v('z' + k, z)}") if r > 1.0 \
                else v('z' + k, z)
        eta_c = 1 + (l0 / h) ** 2 * z1 * z2 / (1300 * e0 / h0)
        w.guard(eta_c, eta)
        w.eq([("ζ_{1}", "=", f"0.2 + 2.7\\frac{{e_{{0}}}}{{h_{{0}}}} = 0.2 + 2.7 ×"
                             f" \\frac{{{v('e0', e0, 1)}}}{{{v('h0', h0)}}} = "
                             f"{cap(z1r, z1, '1')}", "5.3.9-2"),
              ("ζ_{2}", "=", f"1.15 - 0.01\\frac{{l_{{0}}}}{{h}} = 1.15 - 0.01 ×"
                             f" \\frac{{{v('l0', l0)}}}{{{v('h', h)}}} = "
                             f"{cap(z2r, z2, '2')}", "5.3.9-3"),
              ("η", "=", f"1 + \\frac{{1}}{{1300e_{{0}}/h_{{0}}}}(\\frac{{l_{{0}}}}{{h}})^{{2}}ζ_{{1}}ζ_{{2}} = 1 + "
                         f"\\frac{{1}}{{1300 × {v('e0', e0, 1)}/{v('h0', h0)}}} ×"
                         f" (\\frac{{{v('l0', l0)}}}{{{v('h', h)}}})^{{2}}"
                         f" × {v('z1', z1)} × {v('z2', z2)} = {v('eta', eta)}", "5.3.9-1")])


def _tpl_eccentric(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None or "e0" not in d or d.get("Nd") is None:
        return False
    v, tt, n = w.v, w.t, w.n
    c, st = s.concrete, s.steel
    l0 = d.get("l0")
    eta, e0, e, h0 = d["eta"], d["e0"], d["e"], d["h0"]
    if "alpha" in d:  # 5.3.8, solid circle
        r, rs = s.h / 2, s.ring
        A = math.pi * r * r
        As = s.bars[:, 0].sum()
        a = d["alpha"]
        at_raw = 1.25 - 2 * a
        at = max(at_raw, 0.0)
        N = a * c.fcd * A * (1 - math.sin(2 * math.pi * a) / (2 * math.pi * a)) + (a - at) * st.fsd * As
        M = 2 / 3 * c.fcd * A * r * math.sin(math.pi * a) ** 3 / math.pi + \
            st.fsd * As * rs * (math.sin(math.pi * a) + math.sin(math.pi * at)) / math.pi
        w.guard(N / 1e3, chk.resistance)
        w.guard(M, N * e, 1e-6)
        w.units()
        _ecc_head(w, d, s, l0, h0, circle=True)
        w.eq([("ηe_{0}", f"{v('eta', eta)} × {v('e0', e0, 1)} = {v('e', e, 1)}\\,\\u{{mm}}")])
        w.head(tt("（3）承载力（5.3.8，沿周边均匀配筋的圆形截面）", "(3) Resistance (5.3.8, circle with bars evenly round)"))
        w.p(tt("由式（5.3.8-1）、（5.3.8-2）联立求得受压区混凝土面积所对圆心角与 2π 之比：",
               "From 5.3.8-1 and -2 together, the angle ratio of the compression zone:"), m(f"α = {v('alpha', a)}"),
            tt("。", "."))
        if at_raw < 0:
            w.eq([("α_{t}", f"1.25 - 2α = 1.25 - 2 × {v('alpha', a)} = {n(at_raw, 4)}"),
                  ("", "<", "0" + w.take("（α > 0.625），取 ", " (α > 0.625): ") + f"α_{{t}} = {v('alphat', 0.0)}")],
                 tag="5.3.8-3")
        else:
            w.eq([("α_{t}", f"1.25 - 2α = 1.25 - 2 × {v('alpha', a)} = {v('alphat', at)}")], tag="5.3.8-3")
        w.eq([("N_{ud}", "αf_{cd}A(1 - \\frac{\\op{sin}2πα}{2πα}) + (α - α_{t})f_{sd}A_{s}"),
              ("", f"{v('alpha', a)} × {v('fcd', c.fcd)} × {v('A', A)} × (1 - \\frac{{\\op{{sin}}(2π ×"
                   f" {v('alpha', a)})}}"
                   f"{{2π × {v('alpha', a)}}}) + ({v('alpha', a)} - {v('alphat', at)}) × {v('fsd', st.fsd)} × "
                   f"{v('As', As)}"), ("", f"{v('Nu', N / 1e3, 1)}\\,\\u{{kN}}")], tag="5.3.8-1")
        w.eq([("M_{ud}", "\\frac{2}{3}f_{cd}Ar\\frac{\\op{sin}^{3}πα}{π} + f_{sd}A_{s}r_{s}"
                         "\\frac{\\op{sin}πα + \\op{sin}πα_{t}}{π}"),
              ("", f"\\frac{{2}}{{3}} × {v('fcd', c.fcd)} × {v('A', A)} × {v('r', r)} × "
                   f"\\frac{{\\op{{sin}}^{{3}}(π × {v('alpha', a)})}}{{π}} + {v('fsd', st.fsd)} × {v('As', As)} × "
                   f"{v('rs', rs)} × \\frac{{\\op{{sin}}(π × {v('alpha', a)}) + \\op{{sin}}(π ×"
                   f" {v('alphat', at)})}}{{π}}"),
              ("", f"{v('Mud', M / 1e6, 1)}\\,\\u{{kN·m}}"),
              ("N_{ud}ηe_{0}", f"{n(N / 1e3, 1)} × {v('e', e, 1)} × 10^{{-3}} = {n(N * e / 1e6, 1)}\\,\\u{{kN·m}}")],
             tag="5.3.8-2")
        w.p(tt("两式同时成立（α 由此求得），偏心受压承载力 ", "Both equations hold (they fix α): "),
            m(f"N_{{u}} = N_{{ud}} = {v('Nu', N / 1e3, 1)}\\,\\u{{kN}}"), tt("。", "."))
        return True
    if s.kind not in ("rect", "tee", "box") or s.prestressed:
        return False
    if any("between the gross" in x for x in chk.notes):
        return False
    g = _steel(s)
    fl = _flange(s)
    if not g.edge_only or fl is None or "x" not in d:
        return False
    bf, hf, b = fl
    flange = b is not None and bf > b + 1e-9
    b = bf if b is None else b
    x = d["x"]
    xc = min(x, s.h)
    inside = xc <= hf + 1e-9 or not flange
    if not flange and xc > hf + 1e-9 and _flange(s)[2] is not None:
        return False
    aref, _ = s.comp_ref()
    large = d["case"].startswith("large")
    es = e + (h0 - s.zg)  # from the load to As (5.3.4-3)
    note36 = next((x_ for x_ in chk.notes if "5.3.6" in x_), None)
    if note36 is not None:  # 5.3.6: x < 2a's, moments about A's (5.2.4-2 with Md → Nd e's)
        if not large or g.Ap or g.Apc:
            return False
        e_s = e - s.zg + aref
        Nu = st.fsd * g.As * (g.ds - aref) / e_s
        w.guard(Nu / 1e3, chk.resistance)
        w.units()
        _ecc_head(w, d, s, l0, h0, circle=False)
        w.head(tt("（3）大小偏心判别", "(3) Large or small eccentricity"))
        w.p(tt("由式（5.3.4-1）、（5.3.4-2）联立解得受压区高度 ", "From 5.3.4-1 and -2: "),
            m(f"x = {v('x', x, 1)}\\,\\u{{mm}}"), tt("。", "."))
        w.eq([("ξ", f"\\frac{{x}}{{h_{{0}}}} = \\frac{{{v('x', x, 1)}}}{{{v('h0', h0)}}} = {n(x / h0, 3)}"),
              w.cmp(x / h0, d["xi_b"], f"ξ_{{b}} = {v('xib', d['xi_b'])}")], tag="5.3.3")
        w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"),
              w.cmp(x, 2 * aref, f"2a'_{{s}} = 2 × {v('asc', aref)} = {n(2 * aref)}\\,\\u{{mm}}", ge=True)],
              tag="5.2.2-5")
        w.head(tt("（4）按 5.3.6 条对受压钢筋合力点取矩", "(4) Moments about the compression steel (5.3.6)"))
        yc_s, yc_v = ("\\frac{h}{2}", f"\\frac{{{v('h', s.h)}}}{{2}}") if s.kind == "rect" else ("y_{c}", v('yc', s.zg))
        w.eq([("e'_{s}", f"ηe_{{0}} - {yc_s} + a'_{{s}} = {v('eta', eta)} × {v('e0', e0, 1)} - {yc_v} + "
                         f"{v('asc', aref)} = {v('es1', e_s, 1)}\\,\\u{{mm}}"),
              ("N_{u}", f"\\frac{{f_{{sd}}A_{{s}}(h_{{0}} - a'_{{s}})}}{{e'_{{s}}}} = \\frac{{{v('fsd', st.fsd)} × "
                        f"{v('As', g.As)} × ({v('h0', g.ds)} - {v('asc', aref)})}}"
                        f"{{{v('es1', e_s, 1)}}} = {v('Nu', Nu / 1e3, 1)}\\,\\u{{kN}}")], tag="5.3.6")
        w.hide_notes += [note36]
        return True
    if large:
        ss = st.fsd
    else:
        ss = float(np.clip(c.eps_cu * st.Es * (c.beta * h0 / x - 1), -st.fsd_c, st.fsd))
        if g.tension_rows > 1:
            return False
    sc = st.fsd_c if (g.Asc and x >= 2 * aref - 1e-9) else None
    if g.Asc and sc is None:
        return False
    if inside:
        Cc, Mc = c.fcd * bf * xc, c.fcd * bf * xc * (h0 - xc / 2)
    else:
        Cc = c.fcd * (b * xc + (bf - b) * hf)
        Mc = c.fcd * (b * xc * (h0 - xc / 2) + (bf - b) * hf * (h0 - hf / 2))
    Nu = Cc + (sc or 0.0) * g.Asc - ss * g.As
    rev = d.get("N_reverse")
    Nu_out = min(Nu, rev * 1e3) if rev is not None else Nu
    w.guard(Nu_out / 1e3, chk.resistance)
    Mr = Mc + (sc or 0.0) * g.Asc * (h0 - g.asc)
    w.guard(Nu * es, Mr, 1e-4)
    w.units()
    _ecc_head(w, d, s, l0, h0, circle=False)
    rect = s.kind == "rect"
    if rect:
        w.eq([("e", f"ηe_{{0}} + \\frac{{h}}{{2}} - a = {v('eta', eta)} × {v('e0', e0, 1)} +"
                    f" \\frac{{{v('h', s.h)}}}{{2}} - "
                    f"{v('a', s.h - h0)} = {v('es', es, 1)}\\,\\u{{mm}}")], tag="5.3.4-3")
    else:
        w.eq([("e", f"ηe_{{0}} + h_{{0}} - y_{{c}} = {v('eta', eta)} × {v('e0', e0, 1)} + {v('h0', h0)} - "
                    f"{v('yc', s.zg)} = {v('es', es, 1)}\\,\\u{{mm}}")], tag="5.3.4-3")
        w.p(tt("$y_{c}$：截面重心至受压较大边的距离（T 形截面按实际重心，程序约定）。",
               "$y_{c}$: the centroid's depth from the more compressed face (T sections: the actual centroid, program "
               "convention)."), )
    w.head(tt("（3）大小偏心判别", "(3) Large or small eccentricity"))
    w.p(tt("由式（5.3.4-1）、（5.3.4-2）联立（钢筋应力按 5.1.5 条）解得受压区高度：",
           "From 5.3.4-1 and -2 together (steel stresses by 5.1.5), the depth of the compression zone:"),
        m(f"x = {v('x', x, 1)}\\,\\u{{mm}}"), tt("。", "."))
    xb = d["xi_b"]
    w.eq([("ξ", f"\\frac{{x}}{{h_{{0}}}} = \\frac{{{v('x', x, 1)}}}{{{v('h0', h0)}}} = {n(x / h0, 3)}"),
          w.cmp(x / h0, xb, f"ξ_{{b}} = {v('xib', xb)}")], tag="5.3.3")
    if large:
        w.p(tt("为大偏心受压，受拉钢筋应力取 ", "Large eccentricity: the tension steel at "),
            m(f"σ_{{s}} = f_{{sd}} = {v('ss', ss)}\\,\\u{{MPa}}"), tt("。", "."))
    else:
        w.p(tt("为小偏心受压，受拉一侧（受压较小边）钢筋应力按式（5.1.5-1）：",
               "Small eccentricity: the far steel by 5.1.5-1:"))
        raw = c.eps_cu * st.Es * (c.beta * h0 / x - 1)
        clip = "" if _close(raw, ss) else (f" = {n(raw)}" + w.take("，限于 −f′sd～fsd，取 ", ", within −f′sd…fsd: ")
                                           + "")
        w.eq([("σ_{s}", f"ε_{{cu}}E_{{s}}(\\frac{{βh_{{0}}}}{{x}} - 1) = {v('ecu', c.eps_cu)} × {v('Es', st.Es)} × "
                        f"(\\frac{{{v('beta', c.beta)} × {v('h0', h0)}}}{{{v('x', x, 1)}}} - 1){clip} = "
                        f"{v('ss', ss)}\\,\\u{{MPa}}")], tag="5.1.5-1")
    if g.Asc:
        w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"),
              w.cmp(x, 2 * aref, f"2a'_{{s}} = 2 × {v('asc', g.asc)} = {n(2 * aref)}\\,\\u{{mm}}"
                    + w.take("，受压钢筋应力取 ", ": ") + f"f'_{{sd}} = {v('fsdc', st.fsd_c)}\\,\\u{{MPa}}", ge=True)],
             tag="5.2.2-5")
    if x > s.h:
        w.p(tt(f"x = {_nv(x)} mm > h = {_nv(s.h)} mm：混凝土部分按 x = h 计（5.3.4 注），钢筋应力仍按算出的 x 计。",
               f"x = {_nv(x)} mm > h = {_nv(s.h)} mm: the concrete taken with x = h (5.3.4 note), the steel stresses"
               f" with "
               "the x found."))
    w.head(tt("（4）偏心受压承载力", "(4) Resistance"))
    xs_ = "h" if x > s.h else "x"
    csym = (f"f_{{cd}}b'_{{f}}{xs_}" if flange else f"f_{{cd}}b{xs_}") if inside else \
        f"f_{{cd}}[b{xs_} + (b'_{{f}} - b)h'_{{f}}]"
    xv = v('x', x, 1) if x <= s.h else v('h', s.h)
    cval = f"{v('fcd', c.fcd)} × {v('bf' if flange else 'b', bf)} × {xv}" if inside else \
        f"{v('fcd', c.fcd)} × [{v('b', b)} × {xv} + ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)}]"
    sym = [csym] + (["f'_{sd}A'_{s}"] if g.Asc else []) + ["- σ_{s}A_{s}"]
    val = [cval] + ([f"{v('fsdc', sc)} × {v('Asc', g.Asc)}"] if g.Asc else []) + \
        [f"- {v('ss', ss, paren=True)} × {v('As', g.As)}"]
    w.eq([("N_{u}", _join(sym)), ("", _join(val)), ("", f"{v('Nu1', Nu / 1e3, 1)}\\,\\u{{kN}}")],
         tag="5.3.5-1" if (flange and not inside) else "5.3.4-1")
    hx = f"(h_{{0}} - \\frac{{{xs_}}}{{2}})"
    msym = (f"f_{{cd}}b'_{{f}}{xs_}{hx}" if flange else f"f_{{cd}}b{xs_}{hx}") if inside else \
        f"f_{{cd}}[b{xs_}{hx} + (b'_{{f}} - b)h'_{{f}}(h_{{0}} - \\frac{{h'_{{f}}}}{{2}})]"
    w.p(tt("校核弯矩平衡（对受拉钢筋合力点取矩）：", "Moment equilibrium about the tension steel:"))
    w.eq([("N_{u}e", f"{n(Nu / 1e3, 1)} × {v('es', es, 1)}×10^{{-3}} = {n(Nu * es / 1e6, 1)}\\,\\u{{kN·m}}"),
          ("", msym + (" + f'_{sd}A'_{s}(h_{0} - a'_{s})" if g.Asc else "") + f" = {n(Mr / 1e6, 1)}\\,\\u{{kN·m}}")],
         tag="5.3.5-2" if (flange and not inside) else "5.3.4-2")
    if rev is not None:
        _reverse_rows(w, s, g, d, e0, rev, Nu)
    else:
        w.eq([("N_{u}", f"{v('Nu', Nu / 1e3, 1)}\\,\\u{{kN}}")])
    w.hide_notes += [x_ for x_ in chk.notes if "reverse failure" in x_]
    return True


def _reverse_rows(w: _W, s, g: _Steel, d, e0, rev, Nu):
    """5.3.4-4, -5 (5.3.5-3, -4): the load between the two steel groups, failure from the far face"""
    v, tt, n = w.v, w.t, w.n
    c, st = s.concrete, s.steel
    aref, _ = s.comp_ref()
    e0r = d["e0_from_forces"]
    e_r = s.zg - e0r - aref
    A = s.area
    Mr = c.fcd * A * (s.zg - aref) + st.fsd_c * g.As * (g.ds - aref)
    Nr = Mr / e_r
    w.guard(Nr / 1e3, rev)
    w.head(tt("（5）反向破坏（轴力作用在两侧钢筋合力点之间）", "(5) Reverse failure (load between the two steel groups)"))
    if s.kind == "rect":
        w.eq([("e'", f"\\frac{{h}}{{2}} - e_{{0}} - a' = \\frac{{{v('h', s.h)}}}{{2}} - {v('e0r', e0r, 1)} - "
                     f"{v('asc', aref)} = {v('er', e_r, 1)}\\,\\u{{mm}}")], tag="5.3.4-5")
        w.eq([("N_{r}", "\\frac{f_{cd}bh(h'_{0} - \\frac{h}{2}) + f'_{sd}A_{s}(h'_{0} - a_{s})}{e'}"),
              ("", f"\\frac{{{v('fcd', c.fcd)} × {v('b', s.pieces[0].b)} × {v('h', s.h)} × ({v('h01', s.h - aref)} - "
                   f"\\frac{{{v('h', s.h)}}}{{2}}) + {v('fsdc', st.fsd_c)} × {v('As', g.As)} × ({v('h01', s.h - aref)}"
                   f" - "
                   f"{v('as', s.h - g.ds)})}}{{{v('er', e_r, 1)}}}"),
              ("", f"{v('Nr', Nr / 1e3, 1)}\\,\\u{{kN}}")], tag="5.3.4-4")
    else:
        w.eq([("e'", f"y_{{c}} - e_{{0}} - a' = {v('yc', s.zg)} - {v('e0r', e0r, 1)} - {v('asc', aref)} = "
                     f"{v('er', e_r, 1)}\\,\\u{{mm}}")], tag="5.3.5")
        w.eq([("N_{r}", f"\\frac{{f_{{cd}}A(y_{{c}} - a') + f'_{{sd}}A_{{s}}(h'_{{0}} - a_{{s}})}}{{e'}} = "
                        f"\\frac{{{v('fcd', c.fcd)} × {v('A', A)} × ({v('yc', s.zg)} - {v('asc', aref)}) + "
                        f"{v('fsdc', st.fsd_c)} × {v('As', g.As)} × {v('h0r', g.ds - aref)}}}{{{v('er', e_r, 1)}}} = "
                        f"{v('Nr', Nr / 1e3, 1)}\\,\\u{{kN}}")], tag="5.3.5-3")
    w.p(tt("$e_{0}$ 取 $M_{d}/N_{d}$，不计 5.3.9 的下限，也不乘 η（偏安全，程序约定）。",
           "$e_{0}$ is $M_{d}/N_{d}$ without the minimum of 5.3.9 and without η (safe side, program convention)."))
    w.eq([("N_{u}", f"\\op{{min}}({n(Nu / 1e3, 1)}, {n(Nr / 1e3, 1)}) = {v('Nu', min(Nu, Nr) / 1e3, 1)}\\,\\u{{kN}}")])


# ── tension ─────────────────────────────────────────────────────────────────
def _fsd_cap_rows(w: _W, fsd: float, ft: float):
    """table 3.2.3-1 note 1: bars of axial and small-eccentricity tension members at most 330 MPa"""
    if ft < fsd - 1e-9:
        w.eq([("f_{sd}", f"{w.v('fsd0', fsd)}\\,\\u{{MPa}}"),
              ("", ">", "330" + w.take("，取 ", ", ") + f"f_{{sd}} = {w.v('fsd', ft)}\\,\\u{{MPa}}")],
             tag="表 3.2.3-1 注 1" if w.zh else "table 3.2.3-1 note 1")


def _tpl_tension(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None or "case" not in d or s.prestressed:
        return False
    v, tt, n = w.v, w.t, w.n
    st = s.steel
    ft = d["fsd_tension"]
    if d["case"] == "axial tension":
        As = s.bars[:, 0].sum()
        w.guard(ft * As / 1e3, chk.resistance)
        w.units()
        _fsd_cap_rows(w, st.fsd, ft)
        w.eq([("N_{u}",
               f"f_{{sd}}ΣA_{{s}} = {v('fsd', ft)} × {v('As', As)} = {v('Nu', ft * As / 1e3, 1)}\\,\\u{{kN}}")],
             tag="5.4.1")
        w.hide_notes = [x for x in chk.notes if "table 3.2.3-1 note 1" in x]
        return True
    g = _steel(s)
    if not g.edge_only:
        return False
    if d["case"] == "small eccentricity":
        e, e_ = d["e"], d["e_prime"]
        dt, dc = g.ds, g.asc
        N1 = ft * g.Asc * (dt - dc) / e
        N2 = ft * g.As * (dt - dc) / e_
        w.guard(N1 / 1e3, d["N_5_4_2_1"])
        w.guard(N2 / 1e3, d["N_5_4_2_2"])
        w.guard(min(N1, N2) / 1e3, chk.resistance)
        w.units()
        _fsd_cap_rows(w, st.fsd, ft)
        w.p(tt("轴向拉力作用在两侧钢筋合力点之间，为小偏心受拉：", "The load lies between the two steel groups: small "
               "eccentricity:"))
        if s.kind == "rect":
            w.eq([("e", f"\\frac{{h}}{{2}} - e_{{0}} - a_{{s}} = \\frac{{{v('h', s.h)}}}{{2}} - {v('e0', d['e0'], 1)}"
                        f" - "
                        f"{v('as', s.h - dt)} = {v('e', e, 1)}\\,\\u{{mm}}"),
                  ("e'", f"\\frac{{h}}{{2}} + e_{{0}} - a'_{{s}} = \\frac{{{v('h', s.h)}}}{{2}} +"
                         f" {v('e0', d['e0'], 1)} - "
                         f"{v('asc', dc)} = {v('ep', e_, 1)}\\,\\u{{mm}}")])
        else:
            w.eq([("e", f"h_{{0}} - y_{{c}} - e_{{0}} = {v('h0', dt)} - {v('yc', s.zg)} - {v('e0', d['e0'], 1)} = "
                        f"{v('e', e, 1)}\\,\\u{{mm}}"),
                  ("e'", f"y_{{c}} + e_{{0}} - a'_{{s}} = {v('yc', s.zg)} + {v('e0', d['e0'], 1)} - {v('asc', dc)} = "
                         f"{v('ep', e_, 1)}\\,\\u{{mm}}")])
        w.eq([("N_{u1}", f"\\frac{{f_{{sd}}A'_{{s}}(h_{{0}} - a'_{{s}})}}{{e}} = \\frac{{{v('fsd', ft)} ×"
                         f" {v('Asc', g.Asc)} × "
                         f"({v('h0', dt)} - {v('asc', dc)})}}{{{v('e', e, 1)}}} = {v('N1', N1 / 1e3, 1)}\\,\\u{{kN}}")],
             tag="5.4.2-1")
        w.eq([("N_{u2}", f"\\frac{{f_{{sd}}A_{{s}}(h'_{{0}} - a_{{s}})}}{{e'}} = \\frac{{{v('fsd', ft)} ×"
                         f" {v('As', g.As)} × "
                         f"({v('h01', s.h - dc)} - {v('as', s.h - dt)})}}{{{v('ep', e_, 1)}}} = "
                         f"{v('N2', N2 / 1e3, 1)}\\,\\u{{kN}}")], tag="5.4.2-2")
        w.eq([("N_{u}", f"\\op{{min}}(N_{{u1}}, N_{{u2}}) = {v('Nu', min(N1, N2) / 1e3, 1)}\\,\\u{{kN}}")])
        w.hide_notes = [x for x in chk.notes if "table 3.2.3-1 note 1" in x]
        return True
    if d["case"] == "large eccentricity":
        if g.Ap or g.Apc or chk.failed:
            return False
        fl = _flange(s)
        if fl is None:
            return False
        bf, hf, b = fl
        b = bf if b is None else b
        x = d["x"]
        fcd, fsd = s.concrete.fcd, st.fsd
        h0 = g.ds
        es = d["e0"] - (h0 - s.zg)  # load to As (beyond it)
        aref, _ = s.comp_ref()
        case_b = g.Asc > 0 and aref is not None and x < 2 * aref
        if not case_b:
            if x > hf + 1e-9 and bf != b:
                return False
            sc = st.fsd_c if g.Asc else 0.0
            Nu = fsd * g.As - sc * g.Asc - fcd * bf * x
            Mr = fcd * bf * x * (h0 - x / 2) + sc * g.Asc * (h0 - g.asc)
            w.guard(Nu / 1e3, chk.resistance)
            w.guard(Nu * es, Mr, 1e-4)
        else:
            e_c = s.zg + d["e0"] - aref
            N_about = fsd * g.As * (h0 - aref) / e_c
            w.guard(N_about / 1e3, d["N_about_compression_steel"])
            N_single = d["N_singly_reinforced"] * 1e3
            x1 = None
            if N_single:
                k = h0 + es
                x1 = k - math.sqrt(k * k - 2 * fsd * g.As * es / (fcd * bf))
                if x1 > hf + 1e-9 and bf != b:
                    return False
                w.guard((fsd * g.As - fcd * bf * x1), N_single)
            w.guard(max(N_about, N_single) / 1e3, chk.resistance)
        w.units()
        w.p(tt("轴向拉力作用在两侧钢筋合力点以外，为大偏心受拉。", "The load lies outside the two steel groups: large "
               "eccentricity."))
        if s.kind == "rect":
            w.eq([("e", f"e_{{0}} - \\frac{{h}}{{2}} + a_{{s}} = {v('e0', d['e0'], 1)} - \\frac{{{v('h', s.h)}}}{{2}}"
                        f" + "
                        f"{v('as', s.h - h0)} = {v('es', es, 1)}\\,\\u{{mm}}")])
        else:
            w.eq([("e", f"e_{{0}} - (h_{{0}} - y_{{c}}) = {v('e0', d['e0'], 1)} - ({v('h0', h0)} - {v('yc', s.zg)}) = "
                        f"{v('es', es, 1)}\\,\\u{{mm}}")])
        w.p(tt("由式（5.4.2-3）、（5.4.2-4）联立解得 ", "From 5.4.2-3 and -4 together: "),
            m(f"x = {v('x', x, 1)}\\,\\u{{mm}}"), tt("。", "."))
        xbh = d["xi_b_h0"]
        w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"), w.cmp(x, xbh, f"ξ_{{b}}h_{{0}} = {n(xbh, 1)}\\,\\u{{mm}}")],
             tag="5.2.2-3")
        if g.Asc:
            w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"),
                  w.cmp(x, 2 * aref, f"2a'_{{s}} = 2 × {v('asc', aref)} = {n(2 * aref)}\\,\\u{{mm}}", ge=True)],
                 tag="5.2.2-5")
        if not case_b:
            w.eq([("N_{u}", "f_{sd}A_{s} - " + ("f'_{sd}A'_{s} - " if g.Asc else "") + "f_{cd}bx"),
                  ("", f"{v('fsd', fsd)} × {v('As', g.As)} - " +
                       (f"{v('fsdc', st.fsd_c)} × {v('Asc', g.Asc)} - " if g.Asc else "") +
                       f"{v('fcd', fcd)} × {v('b', bf)} × {v('x', x, 1)} = {v('Nu', Nu / 1e3, 1)}\\,\\u{{kN}}")],
                 tag="5.4.2-3")
            w.eq([("N_{u}e", f"{n(Nu / 1e3, 1)} × {v('es', es, 1)}×10^{{-3}} = {n(Nu * es / 1e6, 1)}\\,\\u{{kN·m}}"),
                  ("", "f_{cd}bx(h_{0} - \\frac{x}{2})" + (" + f'_{sd}A'_{s}(h_{0} - a'_{s})" if g.Asc else "") +
                   f" = {n(Mr / 1e6, 1)}\\,\\u{{kN·m}}")], tag="5.4.2-4")
            return True
        w.p(tt("$x < 2a'_{s}$，按 5.2.4 条对受压钢筋合力点取矩（式中 $M_{d}$ 换成 $N_{d}e'_{s}$，5.4.2 条）：",
               "x < 2a′s: moments about the compression steel by 5.2.4 (Md → Nd e′s, 5.4.2):"))
        yc_s, yc_v = ("\\frac{h}{2}", f"\\frac{{{v('h', s.h)}}}{{2}}") if s.kind == "rect" else ("y_{c}", v('yc', s.zg))
        w.eq([("e'_{s}", f"e_{{0}} + {yc_s} - a'_{{s}} = {v('e0', d['e0'], 1)} + {yc_v} - {v('asc', aref)} = "
                         f"{v('esc', e_c, 1)}\\,\\u{{mm}}"),
              ("N_{u1}", f"\\frac{{f_{{sd}}A_{{s}}(h_{{0}} - a'_{{s}})}}{{e'_{{s}}}} = \\frac{{{v('fsd', fsd)} ×"
                         f" {v('As', g.As)} × "
                         f"({v('h0', h0)} - {v('asc', aref)})}}{{{v('esc', e_c, 1)}}} = {v('N1', N_about / 1e3, 1)}"
                         "\\,\\u{kN}")], tag="5.2.4-2")
        if x1 is not None:
            w.p(tt("另按不计受压钢筋的截面由式（5.4.2-3）、（5.4.2-4）求解（程序约定，二者取大值）：",
                   "Also the section without its compression bars by 5.4.2-3, -4 (program convention: the larger "
                   "governs):"))
            w.eq([("x_{1}", f"(h_{{0}} + e) - \\sqrt{{(h_{{0}} + e)^{{2}} - \\frac{{2f_{{sd}}A_{{s}}e}}{{f_{{cd}}b}}}}"
                            f" = "
                            f"{v('x1', x1, 1)}\\,\\u{{mm}}"),
                  w.cmp(x1, xbh, f"ξ_{{b}}h_{{0}} = {n(xbh, 1)}\\,\\u{{mm}}")])
            w.eq([("N_{u2}", f"f_{{sd}}A_{{s}} - f_{{cd}}bx_{{1}} = {v('fsd', fsd)} × {v('As', g.As)} -"
                             f" {v('fcd', fcd)} × "
                             f"{v('b', bf)} × {v('x1', x1, 1)} = {v('N2', N_single / 1e3, 1)}\\,\\u{{kN}}")],
                             tag="5.4.2-3")
            w.eq([("N_{u}", f"\\op{{max}}(N_{{u1}}, N_{{u2}}) = {v('Nu', chk.resistance, 1)}\\,\\u{{kN}}")])
        else:
            w.eq([("N_{u}", f"N_{{u1}} = {v('Nu', chk.resistance, 1)}\\,\\u{{kN}}")])
        w.hide_notes = [x_ for x_ in chk.notes if "2a'" in x_]
        return True
    return False


# ── torsion ─────────────────────────────────────────────────────────────────
def _tpl_torsion(w: _W) -> bool:
    chk, s, d = w.chk, w.sec, w.chk.details
    if s is None or "Wt" not in d or "zeta_used" not in d or "Ast" not in d or d.get("Acor") is None or \
            "tau_bredt" in d:
        return False
    c = s.concrete
    bt = d.get("beta_t")
    q = d["q_torsion"] if bt is not None else d["q_provided"]
    Np0, A0 = d.get("Np0", 0.0), None
    p_term = 0.0
    if d.get("prestress_counted"):
        A0 = s.props("transformed")[0]
        p_term = 0.05 * Np0 * 1e3 / A0
    conc = (0.35 * d["beta_a"] * c.ftd + p_term) * (bt if bt is not None else 1.0)
    stl = 1.2 * math.sqrt(d["zeta_used"]) * d["fsv"] * q * d["Acor"]
    Tu = (conc * d["Wt"] + stl) / 1e6
    w.guard(Tu, chk.resistance)
    zeta = d["fsd"] * d["Ast"] * d["sv"] / (d["fsv"] * d["Asv1"] * d["Ucor"])
    w.guard(zeta, d["zeta"])
    v, tt, n = w.v, w.t, w.n
    g0 = d.get("gamma0", 1.0)
    b, h0 = d["b"], d["h0"]
    w.units()
    k = 1
    if "T_share" in d and not _close(d["T_share"], d.get("Td", d["T_share"])):
        w.p(tt("扭矩按受扭塑性抵抗矩分配（5.5.5）：", "The torque shared by the plastic moduli (5.5.5): "),
            m(f"T = {v('Tsh', d['T_share'], 1)}\\,\\u{{kN·m}}"), tt("。", "."))
    # 5.5.3: section limit and detailing threshold
    tau, fcv = d["tau_section"], d["f_cv"]
    Vd_abs, Td_abs = abs(d["Vd"]), abs(d["Td"])
    tau_c = g0 * Vd_abs * 1e3 / (b * h0) + g0 * Td_abs * 1e6 / d["Wt_total"]
    w.guard(tau_c, tau, 1e-6)
    w.guard(0.51 * math.sqrt(c.fcuk), fcv)
    w.head(tt(f"（{k}）截面限制条件（5.5.3）", f"({k}) Section limits (5.5.3)"))
    k += 1
    if Vd_abs:
        lhs = "\\frac{γ_{0}V_{d}}{bh_{0}} + \\frac{γ_{0}T_{d}}{W_{t}}"
        val = (f"\\frac{{{v('gVd', g0 * Vd_abs, 1)}×10^{{3}}}}{{{v('b', b)} × {v('h0', h0)}}} + "
               f"\\frac{{{v('gTd', g0 * Td_abs, 1)}×10^{{6}}}}{{{w.sv('Wtt', d['Wt_total'])}}}")
    else:
        lhs = "\\frac{γ_{0}T_{d}}{W_{t}}"
        val = f"\\frac{{{v('gTd', g0 * Td_abs, 1)}×10^{{6}}}}{{{w.sv('Wtt', d['Wt_total'])}}}"
    a2 = d.get("alpha2", 1.0)
    lo = 0.50 * a2 * c.ftd
    w.eq([(lhs, f"{val} = {v('tau', tau, 3)}\\,\\u{{MPa}}"),
          w.cmp(tau, fcv, f"0.51\\sqrt{{f_{{cu,k}}}} = 0.51 × \\sqrt{{{v('fcuk', c.fcuk)}}} ="
                          f" {v('fcv', fcv, 3)}\\,\\u{{MPa}}")],
         tag="5.5.3-1")
    w.eq([(lhs, f"{v('tau', tau, 3)}\\,\\u{{MPa}}"),
          w.cmp(tau, lo, f"0.50α_{{2}}f_{{td}} = 0.50 × {v('a2', a2)} × {v('ftd', c.ftd)} = {n(lo, 3)}\\,\\u{{MPa}}")],
         tag="5.5.3-2")
    w.p(tt("满足式（5.5.3-2）时可只按构造配置抗扭钢筋（9.3.13）；不满足时按计算配置（下式）。" if tau <= lo else
           "不满足式（5.5.3-2），抗扭钢筋按计算配置（下式）。",
           "Within 5.5.3-2: detailing torsion reinforcement suffices (9.3.13)." if tau <= lo else
           "Beyond 5.5.3-2: the torsion reinforcement is designed by calculation (below)."))
    w.hide_notes = [x for x in chk.notes if x.startswith("ζ = ") or "(5.5.3-2, 9.3.13)" in x]
    # ζ
    w.head(tt(f"（{k}）纵筋与箍筋的配筋强度比", f"({k}) Strength ratio of longitudinal bars to stirrups"))
    k += 1
    rows = [("ζ", f"\\frac{{f_{{sd}}A_{{st}}s_{{v}}}}{{f_{{sv}}A_{{sv1}}U_{{cor}}}} = \\frac{{{v('fsd', d['fsd'])} × "
                  f"{v('Ast', d['Ast'])} × {v('sv', d['sv'])}}}{{{v('fsv', d['fsv'])} × {v('Asv1', d['Asv1'])} × "
                  f"{v('Ucor', d['Ucor'])}}} = {v('zeta', d['zeta'])}")]
    if d["zeta"] > 1.7:
        rows.append(("", ">", "1.7" + w.take("，取 ", ", ") + f"ζ = {v('zeta_u', d['zeta_used'])}"))
    elif d["zeta"] < 0.6:
        rows.append(("", "<", "0.6" + w.take("（不满足 0.6 ≤ ζ）", " (below 0.6)")))
    w.eq(rows, tag="5.5.1-2")
    p_s = " + 0.05\\frac{N_{p0}}{A_{0}}" if p_term else ""
    p_v = f" + 0.05 × \\frac{{{v('Np0', Np0 * 1e3)}}}{{{v('A0', A0 or 1)}}}" if p_term else ""
    if bt is None:
        w.head(tt(f"（{k}）受扭承载力", f"({k}) Torsional resistance"))
        w.eq([("T_{u}", f"(0.35β_{{a}}f_{{td}}{p_s})W_{{t}} +"
                        f" 1.2\\sqrt{{ζ}}f_{{sv}}\\frac{{A_{{sv1}}}}{{s_{{v}}}}A_{{cor}}"),
              ("", f"(0.35 × {v('beta_a', d['beta_a'])} × {v('ftd', c.ftd)}{p_v}) × {w.sv('Wt', d['Wt'])} + 1.2 × "
                   f"\\sqrt{{{v('zeta_u', d['zeta_used'])}}} × {v('fsv', d['fsv'])} × {v('q', q)} ×"
                   f" {v('Acor', d['Acor'])}"),
              ("", f"{v('Tu', Tu, 1)}\\,\\u{{kN·m}}")], tag="5.5.1-1")
        return True
    # shear with torsion: βt (5.5.4-3), the shear part (5.5.4-1), the stirrups shared (5.5.6), 5.5.4-2
    if "Vu" not in d or "legs" not in d:
        raise _Mismatch("no shear part")
    Td = abs(d["T_share"])
    Wt_bt = d["beta_a"] * d["Wt"] if d["moduli"].get("box") else d["Wt"]
    bt_raw = 1.5 / (1 + 0.5 * Vd_abs * 1e3 * Wt_bt / (Td * 1e6 * b * h0)) if Td else 0.5
    w.guard(min(max(bt_raw, 0.5), 1.0), bt)
    w.head(tt(f"（{k}）剪扭构件混凝土受扭承载力降低系数", f"({k}) Reduction factor of the concrete in shear with torsion"))
    k += 1
    rows = [("β_{t}", f"\\frac{{1.5}}{{1 + 0.5\\frac{{V_{{d}}W_{{t}}}}{{T_{{d}}bh_{{0}}}}}} = \\frac{{1.5}}{{1 + 0.5 × "
                      f"\\frac{{{v('Vd', Vd_abs, 1)}×10^{{3}} × {w.sv('Wtb', Wt_bt)}}}{{{v('Td', Td, 1)}×10^{{6}} ×"
                      f" {v('b', b)} × "
                      f"{v('h0', h0)}}}}} = " + (n(bt_raw, 3) if not _close(bt_raw, bt) else v('beta_t', bt)))]
    if bt_raw > 1.0 + 1e-12:
        rows.append(("", ">", "1.0" + w.take("，取 ", ", ") + f"β_{{t}} = {v('beta_t', bt)}"))
    elif bt_raw < 0.5 - 1e-12:
        rows.append(("", "<", "0.5" + w.take("，取 ", ", ") + f"β_{{t}} = {v('beta_t', bt)}"))
    w.eq(rows, tag="5.5.4-3")
    btv = d["beta_t_shear"]
    P, a1, a3 = d["P"], d["alpha1"], d["alpha3"]
    K = (2 + 0.6 * P) * math.sqrt(c.fcuk) * d["fsv"]
    coef = 0.5e-4 * a1 * a2 * a3 * (10 - 2 * btv) * b * h0
    Vu = coef * math.sqrt(K * d["rho_sv"])
    w.guard(Vu, d["Vu"])
    rho_need = (g0 * Vd_abs / coef) ** 2 / K
    w.guard(rho_need, d["rho_sv_shear"])
    legs = d["legs"]
    w.guard(rho_need * b / legs, d["q_shear"])
    w.head(tt(f"（{k}）抗剪承载力（5.5.4-1）与箍筋分配（5.5.6）", f"({k}) Shear resistance (5.5.4-1) and the stirrups shared "
              "(5.5.6)"))
    k += 1
    w.eq([("ρ_{sv}", f"\\frac{{nA_{{sv1}}}}{{s_{{v}}b}} = \\frac{{{legs} ×"
                     f" {v('Asv1', d['Asv1'])}}}{{{v('sv', d['sv'])} × "
                     f"{v('b', b)}}} = {v('rho_sv', d['rho_sv'])}")])
    w.eq([("V_{u}", "0.5×10^{-4}α_{1}α_{2}α_{3}(10 - 2β_{t})bh_{0}\\sqrt{(2 + 0.6P)\\sqrt{f_{cu,k}}ρ_{sv}f_{sv}}"),
          ("", f"0.5×10^{{-4}} × {v('a1', a1)} × {v('a2', a2)} × {v('a3', a3)} × (10 - 2 × {v('beta_t', btv)}) ×"
               f" {v('b', b)}"
               f" × {v('h0', h0)} × \\sqrt{{(2 + 0.6 × {v('P', P)}) × \\sqrt{{{v('fcuk', c.fcuk)}}} ×"
               f" {v('rho_sv', d['rho_sv'])} × "
               f"{v('fsv', d['fsv'])}}}"),
          ("", f"{v('Vu', Vu, 1)}\\,\\u{{kN}}"),
          w.cmp(Vu, g0 * Vd_abs, f"γ_{{0}}V_{{d}} = {v('gVd', g0 * Vd_abs, 1)}\\,\\u{{kN}}", ge=True)], tag="5.5.4-1")
    w.p(tt("剪扭构件的箍筋为抗剪和抗扭所需之和（5.5.6）。由式（5.5.4-1）反算抗剪所需的配箍率，按全部肢数分摊后从外肢中扣除，余下的用于抗扭：",
           "The stirrups carry shear and torsion together (5.5.6): the ratio shear needs, from 5.5.4-1, spread over"
           " all "
           "the legs and taken off the outer leg; the rest carries torsion:"))
    q_raw = d["q_provided"] - d["q_shear"]
    w.guard(max(q_raw, 0.0), q)
    qrows = [("ρ_{sv,V}", f"\\frac{{(γ_{{0}}V_{{d}})^{{2}}}}{{[0.5×10^{{-4}}α_{{1}}α_{{2}}α_{{3}}(10 -"
                          f" 2β_{{t}})bh_{{0}}]^{{2}}"
                          f"(2 + 0.6P)\\sqrt{{f_{{cu,k}}}}f_{{sv}}}} = {v('rhoV', rho_need)}"),
             ("\\frac{A_{sv1,V}}{s_{v}}", f"\\frac{{ρ_{{sv,V}}b}}{{n}} = \\frac{{{v('rhoV', rho_need)} ×"
                                          f" {v('b', b)}}}{{{legs}}} = "
                                          f"{v('qs', d['q_shear'])}\\,\\u{{mm²/mm}}"),
             ("\\frac{A_{sv1,T}}{s_{v}}", f"\\frac{{A_{{sv1}}}}{{s_{{v}}}} - \\frac{{A_{{sv1,V}}}}{{s_{{v}}}} = "
                                          f"\\frac{{{v('Asv1', d['Asv1'])}}}{{{v('sv', d['sv'])}}} -"
                                          f" {v('qs', d['q_shear'])} = "
                                          + (v('q', q) if q_raw >= 0 else n(q_raw, 4)) + "\\,\\u{mm²/mm}")]
    if q_raw < 0:
        qrows.append(("", "<", "0" + w.take("，箍筋不足以分担扭矩，取 ", ": no stirrup share left for torsion, ")
                      + f"\\frac{{A_{{sv1,T}}}}{{s_{{v}}}} = {v('q', 0.0)}"))
    w.eq(qrows)
    w.head(tt(f"（{k}）剪扭构件受扭承载力", f"({k}) Torsional resistance in shear with torsion"))
    w.eq([("T_{u}", f"β_{{t}}(0.35β_{{a}}f_{{td}}{p_s})W_{{t}} +"
                    f" 1.2\\sqrt{{ζ}}f_{{sv}}\\frac{{A_{{sv1,T}}}}{{s_{{v}}}}A_{{cor}}"),
          ("", f"{v('beta_t', bt)} × (0.35 × {v('beta_a', d['beta_a'])} × {v('ftd', c.ftd)}{p_v}) ×"
               f" {w.sv('Wt', d['Wt'])}"
               f" + 1.2 × \\sqrt{{{v('zeta_u', d['zeta_used'])}}} × {v('fsv', d['fsv'])} × {v('q', q)} × "
               f"{v('Acor', d['Acor'])}"),
          ("", f"{v('Tu', Tu, 1)}\\,\\u{{kN·m}}")], tag="5.5.4-2")
    return True


# ── crack width ─────────────────────────────────────────────────────────────
def _tpl_crack_width(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    need = ("sigma_ss", "C1", "C2", "C3", "c", "d", "rho_te_used", "E")
    if not all(k in d for k in need) or s is None:
        return False
    W = d["C1"] * d["C2"] * d["C3"] * d["sigma_ss"] / d["E"] * (d["c"] + d["d"]) / (0.36 + 1.7 * d["rho_te_used"])
    w.guard(W, chk.demand)
    v, tt, n = w.v, w.t, w.n
    w.units()
    w.head(tt("（1）钢筋应力", "(1) Steel stress"))
    if not _sigma_ss_rows(w, s, d):
        return False
    w.head(tt("（2）系数", "(2) Factors"))
    rows = []
    Ms, Ml, N, Nl = d.get("Ms"), d.get("Ml"), d.get("N", 0.0), d.get("Nl")
    if d.get("member") == "tension" and not Ms and Nl is not None and N:
        C2 = 1 + 0.5 * max(Nl / N, 0.0)
        w.guard(C2, d["C2"])
        rows.append(("C_{2}", f"1 + 0.5\\frac{{N_{{l}}}}{{N_{{s}}}} = 1 + 0.5 × \\frac{{{v('Nl', abs(Nl), 1)}}}"
                              f"{{{v('Ns', abs(N), 1)}}} = {v('C2', d['C2'])}"))
    elif Ms:
        r = Ml / Ms
        if N and Nl is not None:
            r = max(r, Nl / N)
        C2 = 1 + 0.5 * max(r, 0.0)
        w.guard(C2, d["C2"])
        rows.append(("C_{2}", f"1 + 0.5\\frac{{M_{{l}}}}{{M_{{s}}}} = 1 + 0.5 × \\frac{{{v('Ml', abs(Ml), 1)}}}"
                              f"{{{v('Ms', abs(Ms), 1)}}} = {v('C2', d['C2'])}"))
    else:
        rows.append(("C_{2}", f"{v('C2', d['C2'])}"))
    classB = d.get("At") is not None
    As_sym = "A_{s} + A_{p}" if classB else "A_{s}"
    if "Ate" in d and d.get("As_te"):
        Ate_raw = 2 * d["a_s"] * d["b_te"]
        w.guard(min(Ate_raw, s.area), d["Ate"])
        w.guard(d["As_te"] / d["Ate"], d["rho_te"])
        ate = v('Ate', d['Ate']) if Ate_raw <= s.area + 1e-6 else n(Ate_raw)
        rows.append(("A_{te}", f"2a_{{s}}b = 2 × {v('a_s', d['a_s'])} × {v('bte', d['b_te'])} = {ate}\\,\\u{{mm²}}"))
        if Ate_raw > s.area + 1e-6:
            rows.append(("", ">", f"A = {n(s.area)}\\,\\u{{mm²}}" + w.take("，取 ", ", ") +
                         f"A_{{te}} = {v('Ate', d['Ate'])}\\,\\u{{mm²}}"))
        rho_row = ("ρ_{te}", f"\\frac{{{As_sym}}}{{A_{{te}}}} ="
                             f" \\frac{{{v('As', d['As_te'])}}}{{{v('Ate', d['Ate'])}}} = ")
    elif d.get("member") == "tension" and not d.get("Ms") and "As_te" in d and d["As_te"] is None:
        As_all = s.bars[:, 0].sum()
        w.guard(As_all / s.area, d["rho_te"])
        rho_row = ("ρ_{te}", f"\\frac{{A_{{s}}}}{{A}} = \\frac{{{v('As', As_all)}}}{{{v('A', s.area)}}} = ")
    else:
        raise _Mismatch("ρte not written out")
    if _close(d["rho_te"], d["rho_te_used"]):
        rows.append((rho_row[0], rho_row[1] + v('rho_te', d['rho_te_used'])))
    else:
        rows.append((rho_row[0], rho_row[1] + n(d['rho_te'], 4)))
        hi = d["rho_te"] > 0.1
        rows.append(("", ">" if hi else "<", ("0.1" if hi else "0.01") + w.take("，取 ", ", ")
                     + f"ρ_{{te}} = {v('rho_te', d['rho_te_used'])}"))
    rows = [(r[0], "=", r[1]) if len(r) == 2 else r for r in rows]
    rows = [r + ("6.4.5-1",) if r[0] == "ρ_{te}" else r for r in rows]
    w.eq(rows)
    c_raw = d.get("c_raw", d["c"])
    sep, end = _L[w.lang]["sep"], _L[w.lang]["end"]
    cpart = m(f"c = {v('c', d['c'])}\\,\\u{{mm}}") if _close(c_raw, d["c"]) else \
        m(f"c = {n(c_raw)} > 50" + w.take("，取 ", ", ") + f"c = {v('c', d['c'])}\\,\\u{{mm}}")
    Esym = "E_{p}" if d.get("At") is not None and s.has_internal else "E_{s}"
    w.p(m(f"C_{{1}} = {v('C1', d['C1'])}"), sep, m(f"C_{{3}} = {v('C3', d['C3'])}"), sep, cpart, sep,
        m(f"d = {v('d', d['d'])}\\,\\u{{mm}}"), sep, m(f"{Esym} = {v('Es', d['E'])}\\,\\u{{MPa}}"), end)
    if d.get("At") is not None:
        w.p(tt("B 类构件：受拉区钢筋取 $A_{p} + A_{s}$，弹性模量取 $E_{p}$（6.4.3 条文说明）；d 为等效直径（表 6.4.3）。",
               "Class B: the tension steel is $A_{p} + A_{s}$, the modulus $E_{p}$ (commentary to 6.4.3); d the "
               "equivalent diameter (table 6.4.3)."))
    w.head(tt("（3）最大裂缝宽度", "(3) Largest crack width"))
    w.eq([("W_{cr}", f"C_{{1}}C_{{2}}C_{{3}}\\frac{{σ_{{ss}}}}{{{Esym}}}(\\frac{{c + d}}{{0.36 + 1.7ρ_{{te}}}})"),
          ("", f"{v('C1', d['C1'])} × {v('C2', d['C2'])} × {v('C3', d['C3'])} × \\frac{{{v('sss', d['sigma_ss'])}}}"
               f"{{{v('Es', d['E'])}}} × \\frac{{{v('c', d['c'])} + {v('d', d['d'])}}}{{0.36 + 1.7 × "
               f"{v('rho_te', d['rho_te_used'])}}}"),
          ("", f"{v('Wcr', W, 3)}\\,\\u{{mm}}")], tag="6.4.3")
    return True


def _gamma_f_rows(w: _W, s, h0: float, gf: float):
    """γ'f = (b'f − b) h'f / (b h0), h'f ≤ 0.2 h0 (6.4.4-7)"""
    if not gf:
        return
    fl = _flange(s)
    if fl is None or fl[2] is None:
        raise _Mismatch("γ'f")
    bf, hf, b = fl
    w.guard((bf - b) * min(hf, 0.2 * h0) / (b * h0), gf)
    v = w.v
    hcap = "" if hf <= 0.2 * h0 else (w.take("（h′f 不大于 0.2h0）", " (h′f at most 0.2 h0)"))
    w.eq([("γ'_{f}", f"\\frac{{(b'_{{f}} - b)h'_{{f}}}}{{bh_{{0}}}} = \\frac{{({v('bf', bf)} - {v('b', b)}) × "
                     f"{v('hf', min(hf, 0.2 * h0))}}}{{{v('b', b)} × {v('h0', h0)}}} = {v('gf', gf)}{hcap}")],
                     tag="6.4.4-7")


def _sigma_ss_rows(w: _W, s, d) -> bool:
    """σss of 6.4.4 for the member's nature; False when the case is not written out"""
    v, tt, n = w.v, w.t, w.n
    sss = d["sigma_ss"]
    load = d.get("sigma_ss_load", sss)
    Ms, N = d.get("Ms", 0.0) or 0.0, d.get("N", 0.0) or 0.0
    mt = d.get("member")
    if d.get("At") is not None:  # class B (6.4.4-11 …)
        Np0, ep, z, h0, At = d["Np0"] * 1e3, d["ep"], d["z"], d["h0"], d["At"]
        Mp2 = d.get("M_p2", 0.0) or 0.0
        if N or "e" not in d:
            return False
        M = abs(Ms + Mp2) * 1e6
        val = (M - Np0 * (z - ep)) / (At * z)
        w.guard(val, load)
        e = ep + M / Np0
        w.guard(e, d["e"])
        z_c = min((0.87 - 0.12 * (1 - d["gamma_f"]) * (h0 / e) ** 2) * h0, 0.87 * h0)
        w.guard(z_c, z)
        _gamma_f_rows(w, s, h0, d["gamma_f"])
        w.eq([("e", f"e_{{p}} + \\frac{{M_{{s}} ± M_{{p2}}}}{{N_{{p0}}}} = {v('ep', ep)} +"
                    f" \\frac{{{v('Ms', M / 1e6, 1)}×10^{{6}}}}"
                    f"{{{v('Np0', Np0)}}} = {v('e', e, 1)}\\,\\u{{mm}}"),
              ("z", f"[0.87 - 0.12(1 - γ'_{{f}})(\\frac{{h_{{0}}}}{{e}})^{{2}}]h_{{0}} = [0.87 - 0.12 × (1 - "
                    f"{v('gf', d['gamma_f'])}) × (\\frac{{{v('h0', h0)}}}{{{v('e', e, 1)}}})^{{2}}] × {v('h0', h0)} = "
                    f"{v('z', z, 1)}\\,\\u{{mm}}")], tag="6.4.4-12")
        w.eq([("σ_{ss}", "\\frac{M_{s} ± M_{p2} - N_{p0}(z - e_{p})}{(A_{p} + A_{s})z}"),
              ("", f"\\frac{{{v('Ms', M / 1e6, 1)}×10^{{6}} - {v('Np0', Np0)} × ({v('z', z, 1)} - {v('ep', ep)})}}"
                   f"{{{v('At', At)} × {v('z', z, 1)}}} = {v('sssl', load)}\\,\\u{{MPa}}")], tag="6.4.4-11")
    elif mt == "flexural" and not N and "z" not in d:
        As = d["As_te"]
        val = abs(Ms) * 1e6 / (0.87 * As * d["h0"])
        w.guard(val, load)
        w.eq([("σ_{ss}", "\\frac{M_{s}}{0.87A_{s}h_{0}}"),
              ("", f"\\frac{{{v('Ms', abs(Ms), 1)}×10^{{6}}}}{{0.87 × {v('As', As)} × {v('h0', d['h0'])}}}"),
              ("", f"{v('sssl', load)}\\,\\u{{MPa}}")], tag="6.4.4-2")
        if d.get("C3", 1.0) == 1.15:
            w.p(tt("板式受弯构件，$C_{3}$ = 1.15。", "A slab: C3 = 1.15."))
    elif mt == "compression" and N > 0 and "es" in d and "e0" in d:
        As, h0, z, es, e0, gf = d["As_te"], d["h0"], d["z"], d["es"], d["e0"], d["gamma_f"]
        w.guard(abs(Ms) / N * 1e3, e0)
        ys = h0 - s.zg
        w.guard(d["eta_s"] * e0 + ys, es)
        z_c = min((0.87 - 0.12 * (1 - gf) * (h0 / es) ** 2) * h0, 0.87 * h0)
        w.guard(z_c, z)
        val = N * 1e3 * (es - z) / (As * z)
        w.guard(val, load)
        rows = [("e_{0}", f"\\frac{{M_{{s}}}}{{N_{{s}}}} = \\frac{{{v('Ms', abs(Ms), 1)}}}{{{v('Ns', N, 1)}}} ×"
                          f" 10^{{3}} = "
                          f"{v('e0', e0, 1)}\\,\\u{{mm}}")]
        l0, h = d.get("l0"), s.h
        if d["eta_s"] != 1.0:
            w.guard(1 + (l0 / h) ** 2 / (4000 * e0 / h0), d["eta_s"])
            rows.append(("η_{s}", f"1 + \\frac{{1}}{{4000e_{{0}}/h_{{0}}}}(\\frac{{l_{{0}}}}{{h}})^{{2}} = 1 + "
                                  f"\\frac{{1}}{{4000 × {v('e0', e0, 1)}/{v('h0', h0)}}} ×"
                                  f" (\\frac{{{v('l0', l0)}}}{{{v('h', h)}}})^{{2}}"
                                  f" = {v('eta_s', d['eta_s'])}"))
            tag_eta = "6.4.4-8"
        elif l0:
            rows.append(("\\frac{l_{0}}{h}", f"\\frac{{{v('l0', l0)}}}{{{v('h', h)}}} = {n(l0 / h, 2)}"))
            rows.append(w.cmp(l0 / h, 14, "14" + w.take("，取 ", ", ") + f"η_{{s}} = {v('eta_s', 1.0)}"))
            tag_eta = "6.4.4-8"
        else:
            tag_eta = None
        if tag_eta:
            w.eq(rows, tag=tag_eta)
            rows = []
        rows.append(("e_{s}", f"η_{{s}}e_{{0}} + y_{{s}} = {v('eta_s', d['eta_s'])} × {v('e0', e0, 1)} + {v('ys', ys)}"
                              f" = "
                              f"{v('es', es, 1)}\\,\\u{{mm}}"))
        w.eq(rows, tag="6.4.4-6")
        _gamma_f_rows(w, s, h0, gf)
        z_raw = (0.87 - 0.12 * (1 - gf) * (h0 / es) ** 2) * h0
        zr = [("z", f"[0.87 - 0.12(1 - γ'_{{f}})(\\frac{{h_{{0}}}}{{e_{{s}}}})^{{2}}]h_{{0}} = [0.87 - 0.12 × (1 -"
                    f" {v('gf', gf)}) × "
                    f"(\\frac{{{v('h0', h0)}}}{{{v('es', es, 1)}}})^{{2}}] × {v('h0', h0)} = " +
               (v('z', z, 1) if _close(z_raw, z) else n(z_raw, 1)) + "\\,\\u{mm}")]
        if not _close(z_raw, z):
            zr.append(("", ">", "0.87h_{0}" + w.take("，取 ", ", ") + f"z = {v('z', z, 1)}\\,\\u{{mm}}"))
        w.eq(zr, tag="6.4.4-5")
        w.eq([("σ_{ss}", f"\\frac{{N_{{s}}(e_{{s}} - z)}}{{A_{{s}}z}} = \\frac{{{v('Ns', N, 1)}×10^{{3}} ×"
                         f" ({v('es', es, 1)} - "
                         f"{v('z', z, 1)})}}{{{v('As', As)} × {v('z', z, 1)}}} = {v('sssl', load)}\\,\\u{{MPa}}")],
             tag="6.4.4-4")
    elif mt == "tension" and N < 0 and not Ms:
        As = s.bars[:, 0].sum()
        val = -N * 1e3 / As
        w.guard(val, load)
        w.eq([("σ_{ss}", f"\\frac{{N_{{s}}}}{{A_{{s}}}} = \\frac{{{v('Ns', -N, 1)}×10^{{3}}}}{{{v('As', As)}}} = "
                         f"{v('sssl', load)}\\,\\u{{MPa}}")], tag="6.4.4-1")
        w.p(tt("轴心受拉构件：$A_{s}$ 取全部纵筋，$ρ_{te} = A_{s}/A$。", "Axial tension: all the bars; ρte = As / A."))
    elif mt == "tension" and N < 0 and "e's" in d:
        As, h0, esp = d["As_te"], d["h0"], d["e's"]
        _, top = s.edge_masks()
        a_c = float(np.average(s.bars[top, 1], weights=s.bars[top, 0]))
        val = -N * 1e3 * esp / (As * (h0 - a_c))
        w.guard(val, load)
        w.eq([("σ_{ss}", f"\\frac{{N_{{s}}e'_{{s}}}}{{A_{{s}}(h_{{0}} - a'_{{s}})}} ="
                         f" \\frac{{{v('Ns', -N, 1)}×10^{{3}} × "
                         f"{v('esp', esp, 1)}}}{{{v('As', As)} × ({v('h0', h0)} - {v('asc', a_c)})}} = "
                         f"{v('sssl', load)}\\,\\u{{MPa}}")], tag="6.4.4-3")
    else:
        return False
    if "sigma_s_t" in d:
        w.eq([("σ_{ss}", f"{v('sssl', load)} + {v('sst', d['sigma_s_t'], paren=True)} = {v('sss', sss)}\\,\\u{{MPa}}"
                         + w.take("（计入温差自应力）", " (with the temperature self-stress)"))])
    elif not _close(load, sss):
        return False
    return True


# ── crack resistance, principal stresses, stresses in service ──────────────
def _edge_sign(s, edge: str) -> int:
    """σ = sign × M / W0 at an edge (tension +, sagging M +): −1 at the top, +1 at the bottom"""
    return -1 if edge == "top" else 1


def _tpl_crack_resistance(w: _W) -> bool:
    chk, d, s = w.chk, w.chk.details, w.chk.section
    if "edge" not in d or "top" not in d or "bottom" not in d or "Ms" not in d or s is None or \
            not hasattr(s, "props"):
        return False
    v, tt, n, E = w.v, w.t, w.n, _L[w.lang]
    gov = d["edge"]
    kind = "full" if "full prestress" in chk.name else "B" if "class B" in chk.name else "A"
    Ms, Ml, MG = d["Ms"], d["Ml"], d.get("M_G")
    for e in ("top", "bottom"):  # every value of the table from the formulas
        r = d[e]
        sg = _edge_sign(s, e)
        w.guard(sg * Ms * 1e6 / r["W0"] + r.get("sigma_t", 0.0), r["sigma_st"], 1e-6)
        w.guard(sg * Ml * 1e6 / r["W0"], r["sigma_lt"], 1e-6)
    w.units()
    rows = []
    for e in ("top", "bottom"):
        r = d[e]
        rows.append([E[e], Cell(r["W0"] / 1e6, digits=1), r["sigma_st"], r["sigma_lt"],
                     r["sigma_pc"]])
    w.node.table(rows, header=[E["edge"], "W_{0}", "σ_{st}", "σ_{lt}", "σ_{pc}"],
                 units=["", "×10^{6} mm³", "MPa", "MPa", "MPa"], digits=[None, 1, 2, 2, 2], compact=True)
    w.p(tt("$σ_{st}$、$σ_{lt}$：频遇组合、准永久组合弯矩在边缘产生的应力（拉为正，换算截面）；$σ_{pc}$：扣除全部损失后的预加力在边缘产生的应力（压为正，含次弯矩）。",
           "σst, σlt: edge stresses of the frequent and quasi-permanent moments (tension +, transformed section); σpc: "
           "the prestress after all losses (compression +, with its secondary moment)."))
    g = d[gov]
    sg = _edge_sign(s, gov)
    pre = "-" if sg < 0 else ""
    st_rows = [("σ_{st}", f"{pre}\\frac{{M_{{s}}}}{{W_{{0}}}}" + (" + ψσ_{t}" if "sigma_t" in g else "") + " = " +
                f"{pre}\\frac{{{v('Ms', Ms, 1, paren=bool(pre))}×10^{{6}}}}{{{w.sv('W0', g['W0'])}}}" +
                (f" + {v('st_t', g['sigma_t'], 2, paren=True)}" if "sigma_t" in g else "") +
                f" = {v('sst', g['sigma_st'], 2)}\\,\\u{{MPa}}")]
    w.eq(st_rows, tag="6.3.2-1")
    ftk = s.concrete.ftk if hasattr(s, "concrete") else None
    if kind == "A":
        lhs = g["sigma_st"] - g["sigma_pc"]
        w.eq([("σ_{st} - σ_{pc}", f"{v('sst', g['sigma_st'], 2)} - {v('spc', g['sigma_pc'], 2, paren=True)} = "
                                  f"{n(lhs, 2)}\\,\\u{{MPa}}"),
              w.cmp(lhs, 0.7 * ftk, f"0.7f_{{tk}} = 0.7 × {v('ftk', ftk)} = {n(0.7 * ftk, 2)}\\,\\u{{MPa}}")],
             tag="6.3.1-3")
        for e in ("top", "bottom"):
            r = d[e]
            lt = r["sigma_lt"] - r["sigma_pc"]
            w.eq([(f"σ_{{lt}} - σ_{{pc}}\\t{{（{E[e]}）}}" if w.zh else f"σ_{{lt}} - σ_{{pc}}\\t{{ ({E[e]})}}",
                   f"{n(r['sigma_lt'], 2)} - {n(r['sigma_pc'], 2, paren=True)} = {n(lt, 2)}\\,\\u{{MPa}}"),
                  w.cmp(lt, 0.0, "0")], tag="6.3.1-4")
        w.hide_notes = []
    elif kind == "full":
        f = g["limit"] / g["sigma_pc"] if g["sigma_pc"] else 1.0
        lhs = g["sigma_st"] - g["limit"]
        w.eq([(f"σ_{{st}} - {n(f, 2)}σ_{{pc}}", f"{v('sst', g['sigma_st'], 2)} - {n(f, 2)} × "
                                                f"{v('spc', g['sigma_pc'], 2, paren=True)} = {n(lhs, 2)}\\,\\u{{MPa}}"),
              w.cmp(lhs, 0.0, "0")], tag=chk.clause.split()[-1])
    else:
        sG = g["sigma_G"]
        w.guard(sg * MG * 1e6 / g["W0"], sG)
        lhs = sG - g["sigma_pc"]
        w.eq([("σ_{G}", f"{pre}\\frac{{M_{{G}}}}{{W_{{0}}}} = {pre}\\frac{{{v('MG', MG, 1, paren=bool(pre))}×10^{{6}}}}"
                        f"{{{w.sv('W0', g['W0'])}}} = {v('sG', sG, 2)}\\,\\u{{MPa}}"),
              ("σ_{G} - σ_{pc}", f"{v('sG', sG, 2)} - {v('spc', g['sigma_pc'], 2, paren=True)} ="
                                 f" {n(lhs, 2)}\\,\\u{{MPa}}"),
              w.cmp(lhs, 0.0, "0")], tag="6.3.1")
    return True


def _tpl_principal(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "at" not in d or "points" not in d or not isinstance(d["at"], dict) or "z" not in d["at"]:
        return False
    a = d["at"]
    v, tt, n = w.v, w.t, w.n
    sx, sy, tau = a["sigma_cx"], a["sigma_cy"], a["tau"]
    tp = math.hypot((sx - sy) / 2, tau) - (sx + sy) / 2
    cp = math.hypot((sx - sy) / 2, tau) + (sx + sy) / 2
    w.guard(tp, a["sigma_tp"])
    w.guard(cp, a["sigma_cp"])
    w.units()
    rows = [[p["z"], p["b"], p["sigma_cx"], p["sigma_cy"], p["tau"], p["sigma_tp"], p["sigma_cp"]] for p in d["points"]]
    k = [i for i, p in enumerate(d["points"]) if p is a or p["z"] == a["z"]]
    w.node.table(rows, header=[tt("计算点深度", "depth"), "b", "σ_{cx}", "σ_{cy}", "τ", "σ_{tp}", "σ_{cp}"],
                 units=["mm", "mm", "MPa", "MPa", "MPa", "MPa", "MPa"], digits=[0, 0, 2, 2, 2, 2, 2], compact=True,
                 row_cls=["total" if i in k else "" for i in range(len(rows))])
    w.p(tt(f"计算点取换算截面重心和腹板与翼缘交接处；控制点距顶面 {_nv(a['z'])} mm（$σ_{{cx}}$、$σ_{{cy}}$ 压为正，$σ_{{tp}}$ 拉为正）：",
           f"Points at the transformed centroid and the web–flange junctions; governing point {_nv(a['z'])} mm below"
           f" the "
           "top (σcx, σcy compression +, σtp tension +):"))
    if "sigma_pc" in a and not a.get("extra") and not a.get("sigma_t"):
        sm = a["M"] * 1e6 * a["y0"] / a["I0"]
        w.guard(a["sigma_pc"] + sm, sx)
        ms, vs = ("M_{s}", "V_{s}") if "crack resistance" in chk.name else ("M_{k}", "V_{k}")
        w.p(tt("$y_{0}$：计算点至换算截面重心轴的距离（重心轴以上为正）。", "$y_{0}$: the point's distance from the transformed "
               "centroid (positive above it)."))
        w.eq([("σ_{cx}", f"σ_{{pc}} + \\frac{{{ms}y_{{0}}}}{{I_{{0}}}} = {v('spc', a['sigma_pc'], 2)} + "
                         f"\\frac{{{v('M', a['M'], 1, paren=True)}×10^{{6}} × {v('y0', a['y0'], 1, paren=True)}}}"
                         f"{{{w.sv('I0', a['I0'])}}} = {v('scx', sx, 2)}\\,\\u{{MPa}}")], tag="6.3.3-2")
        if not a.get("bent"):
            tau_c = a["V"] * 1e3 * a["S0"] / (a["b"] * a["I0"])
            w.guard(tau_c, tau)
            w.eq([("τ", f"\\frac{{{vs}S_{{0}}}}{{bI_{{0}}}} = \\frac{{{v('V', a['V'], 1)}×10^{{3}} ×"
                        f" {w.sv('S0', a['S0'])}}}"
                        f"{{{v('b', a['b'])} × {w.sv('I0', a['I0'])}}} = {v('tau', tau, 2)}\\,\\u{{MPa}}")],
                        tag="6.3.3-5")
    c = w.sec.concrete if w.sec is not None and hasattr(w.sec, "concrete") else None
    if "crack resistance" in chk.name:
        w.eq([("σ_{tp}", "\\sqrt{(\\frac{σ_{cx} - σ_{cy}}{2})^{2} + τ^{2}} - \\frac{σ_{cx} + σ_{cy}}{2}"),
              ("", f"\\sqrt{{(\\frac{{{v('scx', sx, 2)} - {v('scy', sy, 2, paren=True)}}}{{2}})^{{2}} + "
                   f"{v('tau', tau, 2, paren=True)}^{{2}}}} - \\frac{{{v('scx', sx, 2)} +"
                   f" {v('scy', sy, 2, paren=True)}}}"
                   f"{{2}} = {v('stp', tp, 2)}\\,\\u{{MPa}}"),
              w.cmp(tp, chk.resistance, f"{_lim_text(chk)} = {n(chk.resistance, 2)}\\,\\u{{MPa}}")], tag="6.3.3-1")
        if c is not None:
            cl = re.search(r"6\.3\.1-\d", chk.clause)
            cl = cl.group(0) if cl else "6.3.1"
            w.p(tt(f"限值 ${_lim_text(chk)}$ 按式（{cl}），$f_{{tk}}$ = {_nv(c.ftk)} MPa。",
                   f"Limit ${_lim_text(chk)}$ by {cl}, $f_{{tk}}$ = {_nv(c.ftk)} MPa."))
    else:
        w.eq([("σ_{cp}", "\\frac{σ_{cx} + σ_{cy}}{2} + \\sqrt{(\\frac{σ_{cx} - σ_{cy}}{2})^{2} + τ^{2}}"),
              ("", f"\\frac{{{v('scx', sx, 2)} + {v('scy', sy, 2, paren=True)}}}{{2}} +"
                   f" \\sqrt{{(\\frac{{{v('scx', sx, 2)} - "
                   f"{v('scy', sy, 2, paren=True)}}}{{2}})^{{2}} + {v('tau', tau, 2, paren=True)}^{{2}}}} = "
                   f"{v('scp', cp, 2)}\\,\\u{{MPa}}"),
              w.cmp(cp, chk.resistance, f"0.6f_{{ck}} = {n(chk.resistance, 2)}\\,\\u{{MPa}}")], tag="7.1.6-1")
        w.hide_notes = [x for x in chk.notes if x.startswith("σtp = ") and "detailing stirrups" in x]
        if w.hide_notes:
            ftk = c.ftk
            w.eq([("σ_{tp,max}", f"{v('stpm', d['sigma_tp'], 2)}\\,\\u{{MPa}}"),
                  w.cmp(d["sigma_tp"], 0.5 * ftk, f"0.5f_{{tk}} = {n(0.5 * ftk, 2)}\\,\\u{{MPa}}" +
                        w.take("，箍筋可按构造设置", ": detailing stirrups"))], tag="7.1.6")
    return True


def _lim_text(chk) -> str:
    """the limit of 6.3.1-5 … -8 in symbols: k f_tk"""
    c = chk.section.concrete if chk.section is not None and hasattr(chk.section, "concrete") else None
    if c is None:
        return "[σ_{tp}]"
    k = chk.resistance / c.ftk
    return f"{fixed(k, 1)}f_{{tk}}"


def _tpl_service(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None:
        return False
    if "cracked section" in chk.name:
        return _service_cracked(w)
    if "sigma_kc" not in d or "edge_stress" not in d:
        return False
    v, tt, n, E = w.v, w.t, w.n, _L[w.lang]
    sec = w.sec
    I0, z0 = d["I0"], d["z0"]
    for e, z in (("top", 0.0), ("bottom", sec.h)):
        w.guard(d["Mk"] * 1e6 * (z0 - z) / I0, d["sigma_kc"][e])
    gov = "top" if "top edge" in chk.name else "bottom"
    w.guard(d["edge_stress"][gov], chk.demand)
    w.units()
    rows = []
    for e in ("top", "bottom"):
        rest = d["edge_stress"][e] - d["sigma_kc"][e] - d["sigma_pt"][e]
        rows.append([E[e], d["sigma_kc"][e], d["sigma_pt"][e], rest, d["edge_stress"][e]])
    w.node.table(rows, header=[E["edge"], "σ_{kc}", "σ_{pt}", tt("轴力、温差", "N, temperature"), "σ_{kc} + σ_{pt}"],
                 units=["", "MPa", "MPa", "MPa", "MPa"], digits=[None, 2, 2, 2, 2], compact=True)
    y0 = z0 - (0.0 if gov == "top" else sec.h)
    w.eq([("σ_{kc}", f"\\frac{{M_{{k}}y_{{0}}}}{{I_{{0}}}} = \\frac{{{v('Mk', d['Mk'], 1, paren=True)}×10^{{6}} × "
                     f"{v('y0', y0, 1, paren=True)}}}{{{w.sv('I0', I0)}}} ="
                     f" {v('skc', d['sigma_kc'][gov], 2)}\\,\\u{{MPa}}")],
         tag="7.1.3-1")
    other = d["edge_stress"][gov] - d["sigma_kc"][gov] - d["sigma_pt"][gov]
    tail = f" + {n(other, 2, paren=True)}" if abs(other) > 5e-3 else ""
    w.eq([("σ_{kc} + σ_{pt}", f"{v('skc', d['sigma_kc'][gov], 2)} +"
                              f" {v('spt', d['sigma_pt'][gov], 2, paren=True)}{tail} = "
                              f"{v('sigma', d['edge_stress'][gov], 2)}\\,\\u{{MPa}}"),
          w.cmp(d["edge_stress"][gov], chk.resistance, f"0.5f_{{ck}} = 0.5 × {v('fck', sec.concrete.fck)} = "
                                                       f"{n(chk.resistance, 2)}\\,\\u{{MPa}}")], tag="7.1.5-1")
    sp = np.asarray(d["tendon_stress"])
    if len(sp):
        i = int(np.argmax(sp))
        lim = d.get("tendon_limit")
        t = sec.tendon
        k = 0.75 if t.kind == "bar" else 0.65
        skt = d["sigma_kt"][i]
        val = d["sigma_pe"][i] + d["alpha_EP"] * skt
        w.guard(val, sp[i])
        w.eq([("σ_{p}", f"α_{{EP}}σ_{{kt}} = {v('aEP', d['alpha_EP'])} × {v('skt', skt, 2, paren=True)} = "
                        f"{n(d['alpha_EP'] * skt, 2)}\\,\\u{{MPa}}")], tag="7.1.3-2")
        w.eq([("σ_{pe} + σ_{p}", f"{v('spe', d['sigma_pe'][i], 1)} + {n(d['alpha_EP'] * skt, 2, paren=True)} = "
                                 f"{v('sp', sp[i], 1)}\\,\\u{{MPa}}"),
              w.cmp(sp[i], lim, f"{k}f_{{pk}} = {k} × {v('fpk', t.fpk)} = {n(lim, 1)}\\,\\u{{MPa}}")],
             tag="7.1.5-4" if t.kind == "bar" else "7.1.5-2")
        w.p(tt(f"$σ_{{kt}}$ 取第 {i + 1} 排预应力钢筋（距顶面 {_nv(d['tendon_depth'][i])} mm）处 $M_{{k}}$"
               f" 产生的混凝土应力（拉为正），逐排计算后取最大的钢筋应力。",
               f"σkt at tendon row {i + 1} ({_nv(d['tendon_depth'][i])} mm deep), tension +; every row computed, the "
               "largest tendon stress shown."))
        w.hide_notes = [x for x in chk.failed if x.startswith("tendon stress")]
    return True


def _service_cracked(w: _W) -> bool:
    """7.1.4: class B, cracked section under Np0 and Mk ± Mp2"""
    chk, d = w.chk, w.chk.details
    if d.get("temperature") is not None or d.get("Nt") or "Acr" not in d or d["x"] == math.inf:
        return False
    v, tt, n = w.v, w.t, w.n
    Np0 = d["Np0"] * 1e3
    eN = d["Mt"] * 1e6 / Np0 - d["hps"] if d["Mt"] >= 0 else None
    if eN is None:
        return False
    w.guard(eN, d["eN"], 1e-6)
    scc = Np0 / d["Acr"] + Np0 * d["e0N"] * d["c"] / d["Icr"]
    w.guard(scc, chk.demand)
    w.units()
    w.eq([("e_{N}", f"\\frac{{M_{{k}} ± M_{{p2}}}}{{N_{{p0}}}} - h_{{ps}} = \\frac{{{v('Mt', d['Mt'], 1)}×10^{{6}}}}"
                    f"{{{v('Np0', Np0)}}} - {v('hps', d['hps'])} = {v('eN', eN, 1)}\\,\\u{{mm}}"),
          ("e_{0N}", "=", f"e_{{N}} + c = {v('eN', eN, 1)} + {v('c', d['c'], 1)} = {v('e0N', d['e0N'], 1)}\\,\\u{{mm}}",
           "7.1.4-2")], tag="7.1.4-3")
    sep = _L[w.lang]["sep"]
    w.p(tt("开裂截面按附录 J 求得：", "Cracked section by appendix J: "), m(f"x = {v('x', d['x'], 1)}\\,\\u{{mm}}"), sep,
        m(f"A_{{cr}} = {v('Acr', d['Acr'])}\\,\\u{{mm²}}"), sep,
        m(f"I_{{cr}} = {w.sv('Icr', d['Icr'])}\\,\\u{{mm⁴}}"), sep, m(f"c = {v('c', d['c'], 1)}\\,\\u{{mm}}"),
        tt("。", "."))
    w.eq([("σ_{cc}", "\\frac{N_{p0}}{A_{cr}} + \\frac{N_{p0}e_{0N}c}{I_{cr}}"),
          ("", f"\\frac{{{v('Np0', Np0)}}}{{{v('Acr', d['Acr'])}}} + \\frac{{{v('Np0', Np0)} × {v('e0N', d['e0N'], 1)}"
               f" × "
               f"{v('c', d['c'], 1)}}}{{{w.sv('Icr', d['Icr'])}}} = {v('scc', scc, 2)}\\,\\u{{MPa}}"),
          w.cmp(scc, chk.resistance, f"0.5f_{{ck}} = {n(chk.resistance, 2)}\\,\\u{{MPa}}")], tag="7.1.4-1")
    sp = np.asarray(d["tendon_stress"])
    if len(sp):
        i = int(np.argmax(sp))
        hp = d["hp"][i]
        sig = d["alpha_EP"] * (Np0 / d["Acr"] - Np0 * d["e0N"] * (hp - d["c"]) / d["Icr"])
        w.guard(sig, d["sigma_p"][i], 1e-6)
        lim = d.get("tendon_limit")
        w.eq([("σ_{p}", f"α_{{EP}}[\\frac{{N_{{p0}}}}{{A_{{cr}}}} - \\frac{{N_{{p0}}e_{{0N}}(h_{{p}} -"
                        f" c)}}{{I_{{cr}}}}] = "
                        f"{v('aEP', d['alpha_EP'])} × [\\frac{{{v('Np0', Np0)}}}{{{v('Acr', d['Acr'])}}} - "
                        f"\\frac{{{v('Np0', Np0)} × {v('e0N', d['e0N'], 1)} × ({v('hp', hp)} - {v('c', d['c'], 1)})}}"
                        f"{{{w.sv('Icr', d['Icr'])}}}] = {v('sp_', sig, 2)}\\,\\u{{MPa}}"),
              ("σ_{p0} - σ_{p}", f"{v('sp0', d['sigma_p0'][i], 1)} - {v('sp_', sig, 2, paren=True)} ="
                                 f" {v('sp', sp[i], 1)}"
                                 "\\,\\u{MPa}"),
              w.cmp(sp[i], lim, _tendon_lim(w.sec, lim))], tag="7.1.4-5")
    return True


def _tendon_lim(sec, lim: float) -> str:
    """the tendon stress limit of 7.1.5 in symbols and value"""
    t = sec.tendon
    k = 0.75 if t.kind == "bar" else 0.65
    if _close(k * t.fpk, lim):
        return f"{k}f_{{pk}} = {k} × {_nv(t.fpk)} = {fixed(lim, 1)}\\,\\u{{MPa}}"
    return f"{fixed(lim, 1)}\\,\\u{{MPa}}"


def _tpl_deflection(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "L" not in d:
        return False
    v, tt, n = w.v, w.t, w.n
    k = 300 if d.get("cantilever") else 600
    w.guard(d["L"] / k, chk.resistance)
    w.eq([("f", f"{v('f', chk.demand, 3)}\\,\\u{{mm}}"),
          w.cmp(chk.demand, chk.resistance, f"\\frac{{L}}{{{k}}} = \\frac{{{v('L', d['L'])}}}{{{k}}} = "
                                            f"{n(chk.resistance, 3)}\\,\\u{{mm}}")], tag="6.5.3")
    w.p(tt("f 为汽车荷载（不计冲击）频遇值产生的挠度乘以长期增长系数 ηθ，已扣除结构自重产生的长期挠度。",
           "f: the deflection under the frequent live load without impact × ηθ, the self-weight part taken off."))
    return True


def _tpl_transient(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "top" not in d or "fck_t" not in d:
        return False
    v, tt, n = w.v, w.t, w.n
    gov = "top" if "top edge" in chk.name else "bottom"
    w.guard(d[gov], chk.demand)
    w.guard(0.70 * d["fck_t"], chk.resistance)
    sep = _L[w.lang]["sep"]
    w.p(tt("预加力和施工荷载（构件自重计动力系数）作用下截面边缘的混凝土法向应力（压为正）：",
           "Edge stresses (compression +) under the prestress and the construction loads:"),
        m(f"σ_{{top}} = {v('stop', d['top'], 2)}\\,\\u{{MPa}}"), sep,
        m(f"σ_{{bottom}} = {v('sbot', d['bottom'], 2)}\\,\\u{{MPa}}"), tt("。", "."))
    w.eq([("σ^{t}_{cc}", f"{v('scc', d[gov], 2)}\\,\\u{{MPa}}"),
          w.cmp(d[gov], chk.resistance, f"0.70f'_{{ck}} = 0.70 × {v('fckt', d['fck_t'])} = {n(chk.resistance, 2)}"
                                        "\\,\\u{MPa}")], tag="7.2.8")
    if "sigma_tct" in d:
        st_, ft_ = d["sigma_tct"], d["ftk_t"]
        w.eq([("σ^{t}_{ct}", f"{v('sct', st_, 2)}\\,\\u{{MPa}}"),
              w.cmp(st_, 1.15 * ft_, f"1.15f'_{{tk}} = 1.15 × {v('ftkt', ft_)} = {n(1.15 * ft_, 2)}\\,\\u{{MPa}}")],
             tag="§7.2.8")
        if "rho_needed" in d:
            w.p(tt("预拉区纵向钢筋的配筋率 $(A'_{s} + A'_{p})/A$ 按 7.2.8 条第 2 款在 0.2%（$σ^{t}_{ct} <= 0.70f'_{tk}$）与"
                   " 0.4%（$σ^{t}_{ct} = 1.15f'_{tk}$）之间"
                   f"直线内插，需 {100 * d['rho_needed']:.3f}%，实配 {100 * d['rho_pretension_zone']:.3f}%。",
                   "The pretensioned-zone ratio (A′s + A′p)/A by 7.2.8 item 2, interpolated between 0.2 % (σtct ≤ "
                   f"0.70 f′tk) and 0.4 % (σtct = 1.15 f′tk): {100 * d['rho_needed']:.3f} % needed, "
                   f"{100 * d['rho_pretension_zone']:.3f} % provided."))
    return True


def _tpl_transient_rc(w: _W) -> bool:
    """7.2.4: reinforced concrete in a construction stage, cracked section"""
    chk, d = w.chk, w.chk.details
    if "x0" not in d or "alpha_ES" not in d or "M_steel" not in d:
        return False
    v, tt, n = w.v, w.t, w.n
    M, x0, Icr = d["M"], d["x0"], d["Icr"]
    scc = abs(M) * 1e6 * x0 / Icr
    w.guard(scc, chk.demand)
    w.guard(0.80 * d["fck_t"], chk.resistance)
    w.units()
    sep = _L[w.lang]["sep"]
    dyn = d.get("dynamic_factor", 1.0)
    dyn_t = w.take(f"（动力系数 {_nv(dyn)}）", f" (dynamic factor {_nv(dyn)})") if dyn != 1.0 else ""
    w.p(tt("开裂截面（7.2.4）：", "Cracked section (7.2.4): "), m(f"x_{{0}} = {v('x0', x0, 1)}\\,\\u{{mm}}"), sep,
        m(f"I_{{cr}} = {w.sv('Icr', Icr)}\\,\\u{{mm⁴}}"), sep, m(f"M = {v('M', abs(M), 1)}\\,\\u{{kN·m}}" + dyn_t),
        tt("。", "."))
    w.eq([("σ^{t}_{cc}", f"\\frac{{M_{{k}}^{{t}}x_{{0}}}}{{I_{{cr}}}} = \\frac{{{v('M', abs(M), 1)}×10^{{6}} ×"
                         f" {v('x0', x0, 1)}}}"
                         f"{{{w.sv('Icr', Icr)}}} = {v('scc', scc, 2)}\\,\\u{{MPa}}"),
          w.cmp(scc, chk.resistance, f"0.80f'_{{ck}} = 0.80 × {v('fckt', d['fck_t'])} = {n(chk.resistance, 2)}"
                                     "\\,\\u{MPa}")], tag="7.2.4-1")
    ss = np.asarray(d["sigma_s"])
    if len(ss):
        i = int(np.argmax(ss))
        h0i = d["h0i"][i]
        Ms_, x0s, Ics = abs(d["M_steel"]), d["x0_steel"], d["Icr_steel"]
        val = d["alpha_ES"] * Ms_ * 1e6 * (h0i - x0s) / Ics
        w.guard(val, ss[i])
        lim = 0.75 * d["fsk"]
        w.eq([("σ^{t}_{s}", f"α_{{ES}}\\frac{{M_{{k}}^{{t}}(h_{{0i}} - x_{{0}})}}{{I_{{cr}}}} ="
                            f" {v('aES', d['alpha_ES'])} × "
                            f"\\frac{{{v('Ms', Ms_, 1)}×10^{{6}} × ({v('h0i', h0i)} -"
                            f" {v('x0s', x0s, 1)})}}{{{w.sv('Icrs', Ics)}}}"
                            f" = {v('ss', ss[i], 1)}\\,\\u{{MPa}}"),
              w.cmp(ss[i], lim, f"0.75f_{{sk}} = {n(lim, 1)}\\,\\u{{MPa}}")], tag="7.2.4-2")
    if "sigma_tp" in d:
        w.eq([("σ^{t}_{tp}", f"\\frac{{V}}{{bz_{{0}}}} = {v('stp', d['sigma_tp'], 2)}\\,\\u{{MPa}}"),
              w.cmp(d["sigma_tp"], d["ftk_t"], f"f'_{{tk}} = {v('ftkt', d['ftk_t'])}\\,\\u{{MPa}}")], tag="7.2.5")
    return True


# ── local compression, punching ─────────────────────────────────────────────
def _tpl_local(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "beta" not in d or "Aln" not in d:
        return False
    v, tt = w.v, w.t
    beta = math.sqrt(d["Ab"] / d["Al"])
    w.guard(beta, d["beta"])
    w.units()
    w.eq([("β", f"\\sqrt{{\\frac{{A_{{b}}}}{{A_{{l}}}}}} ="
                f" \\sqrt{{\\frac{{{v('Ab', d['Ab'])}}}{{{v('Al', d['Al'])}}}}} = "
                f"{v('beta', d['beta'])}")], tag="5.7.1-2")
    if "rho_v" not in d:  # no indirect reinforcement: the program's 5.7.2-1 with ρv = 0
        R0 = 0.9 * d["eta_s"] * d["beta"] * d["fcd"] * d["Aln"] / 1e3
        w.guard(R0, chk.resistance)
        w.p(tt("未配间接钢筋，不在 5.7 条适用范围内（该条针对配置间接钢筋的构件）。下面按式（5.7.2-1）取 $ρ_{v}$ = 0 求得的抗力仅供参考（程序约定），"
               "验算结论为不满足。", "No indirect reinforcement: outside 5.7 (written for members with indirect reinforcement). "
               "The resistance below, 5.7.2-1 with ρv = 0, is for information (program convention); the check is not"
               " met."))
        w.eq([("F_{u}", "0.9η_{s}βf_{cd}A_{ln}"),
              ("", f"0.9 × {v('eta_s', d['eta_s'])} × {v('beta', d['beta'])} × {v('fcd', d['fcd'])} × "
                   f"{v('Aln', d['Aln'])} = {v('Fu', R0, 1)}\\,\\u{{kN}}")], tag="5.7.2-1")
        w.hide_notes = []
        return True
    F1 = 1.3 * d["eta_s"] * d["beta"] * d["fcd"] * d["Aln"] / 1e3
    w.guard(F1, d["F_section_max"])
    w.eq([("F_{u1}", "1.3η_{s}βf_{cd}A_{ln}"),
          ("", f"1.3 × {v('eta_s', d['eta_s'])} × {v('beta', d['beta'])} × {v('fcd', d['fcd'])} × {v('Aln', d['Aln'])}"
               f" = "
               f"{v('F1', F1, 1)}\\,\\u{{kN}}")], tag="5.7.1-1")
    mesh, spiral = d.get("mesh"), d.get("spiral")
    Acor_i = d.get("Acor_indirect", d["Acor"])
    if mesh is not None:
        n1, As1, l1, n2, As2, l2, sp = mesh
        rho = (n1 * As1 * l1 + n2 * As2 * l2) / (Acor_i * sp)
        w.guard(rho, d["rho_v"])
        w.eq([("ρ_{v}", f"\\frac{{n_{{1}}A_{{s1}}l_{{1}} + n_{{2}}A_{{s2}}l_{{2}}}}{{A_{{cor}}s}} = \\frac{{{_nv(n1)}"
                        f" × "
                        f"{_nv(As1)} × {_nv(l1)} + {_nv(n2)} × {_nv(As2)} × {_nv(l2)}}}{{{v('Acor0', Acor_i)} ×"
                        f" {_nv(sp)}}} = "
                        f"{v('rho_v', d['rho_v'])}")], tag="5.7.2-3")
    elif spiral is not None:
        Ass1, dcor, sp = spiral
        rho = 4 * Ass1 / (dcor * sp)
        w.guard(rho, d["rho_v"])
        w.eq([("ρ_{v}", f"\\frac{{4A_{{ss1}}}}{{d_{{cor}}s}} = \\frac{{4 × {_nv(Ass1)}}}{{{_nv(dcor)} × {_nv(sp)}}} = "
                        f"{v('rho_v', d['rho_v'])}")], tag="5.7.2-4")
    if Acor_i > d["Acor"] + 1e-6:
        w.p(tt(f"$A_{{cor}}$ = {_nv(Acor_i)} mm² > $A_{{b}}$，取 $A_{{cor}} = A_{{b}}$ = {_nv(d['Acor'])} mm²（5.7.2）。",
               f"Acor = {_nv(Acor_i)} mm² > Ab: Acor = Ab = {_nv(d['Acor'])} mm² (5.7.2)."))
        w.hide_notes = [x for x in w.chk.notes if x.startswith("Acor = ")]
    bc = math.sqrt(d["Acor"] / d["Al"])
    w.guard(bc, d["beta_cor"])
    w.eq([("β_{cor}", f"\\sqrt{{\\frac{{A_{{cor}}}}{{A_{{l}}}}}} = \\sqrt{{\\frac{{{v('Acor', d['Acor'])}}}"
                      f"{{{v('Al', d['Al'])}}}}} = {v('beta_cor', d['beta_cor'])}")], tag="5.7.2-2")
    F2 = 0.9 * (d["eta_s"] * d["beta"] * d["fcd"] + d["k"] * d["rho_v"] * d["beta_cor"] * d["fsd"]) * d["Aln"] / 1e3
    w.guard(F2, d["F_resistance"])
    w.guard(min(F1, F2), chk.resistance)
    w.eq([("F_{u2}", "0.9(η_{s}βf_{cd} + kρ_{v}β_{cor}f_{sd})A_{ln}"),
          ("", f"0.9 × ({v('eta_s', d['eta_s'])} × {v('beta', d['beta'])} × {v('fcd', d['fcd'])} + {v('k', d['k'])} × "
               f"{v('rho_v', d['rho_v'])} × {v('beta_cor', d['beta_cor'])} × {v('fsd', d['fsd'])}) ×"
               f" {v('Aln', d['Aln'])}"),
          ("", f"{v('F2', F2, 1)}\\,\\u{{kN}}")], tag="5.7.2-1")
    w.eq([("F_{u}", f"\\op{{min}}(F_{{u1}}, F_{{u2}}) = {v('Fu', chk.resistance, 1)}\\,\\u{{kN}}")])
    return True


def _tpl_punching(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    v, tt, n = w.v, w.t, w.n
    if "Um" in d and "beta_h" in d:
        w.units()
        sp = d["sigma_pc_m"]
        sp_s = " + 0.15σ_{pc,m}" if sp else ""
        sp_v = f" + 0.15 × {v('spc', sp)}" if sp else ""
        bh = float(np.interp(d["h"], (300.0, 800.0), (1.0, 0.85))) if "h" in d else d["beta_h"]
        w.guard(bh, d["beta_h"])
        if "V_concrete" not in d:
            R = (0.7 * d["beta_h"] * d["ftd"] + 0.15 * sp) * d["Um"] * d["h0"] / 1e3
            w.guard(R, chk.resistance)
            w.eq([("F_{u}", f"(0.7β_{{h}}f_{{td}}{sp_s})U_{{m}}h_{{0}}"),
                  ("", f"(0.7 × {v('beta_h', d['beta_h'])} × {v('ftd', d['ftd'])}{sp_v}) × {v('Um', d['Um'])} × "
                       f"{v('h0', d['h0'])} = {v('Fu', R, 1)}\\,\\u{{kN}}")], tag="5.6.1")
            return True
        conc = (0.35 * d["beta_h"] * d["ftd"] + 0.15 * sp) * d["Um"] * d["h0"] / 1e3
        w.guard(conc, d["V_concrete"])
        if "Asvu" in d:
            Vs = 0.75 * d["fsv"] * d["Asvu"] / 1e3
            ssym, sval, tag = "0.75f_{sv}A_{svu}", f"0.75 × {v('fsv', d['fsv'])} × {v('Asvu', d['Asvu'])}", "5.6.2-2"
        else:
            Vs = 0.75 * d["fsd"] * d["Asbu"] * math.sin(math.radians(d["theta"])) / 1e3
            ssym, tag = "0.75f_{sd}A_{sbu}\\op{sin}θ", "5.6.2-3"
            sval = f"0.75 × {v('fsd', d['fsd'])} × {v('Asbu', d['Asbu'])} × \\op{{sin}}{_nv(d['theta'])}°"
        w.guard(Vs, d["V_steel"])
        w.guard(conc + Vs, chk.resistance)
        w.eq([("F_{u}", f"(0.35β_{{h}}f_{{td}}{sp_s})U_{{m}}h_{{0}} + {ssym}"),
              ("", f"(0.35 × {v('beta_h', d['beta_h'])} × {v('ftd', d['ftd'])}{sp_v}) × {v('Um', d['Um'])} × "
                   f"{v('h0', d['h0'])} + {sval}"),
              ("", f"{v('Vc', conc, 1)} + {v('Vs', Vs, 1)} = {v('Fu', chk.resistance, 1)}\\,\\u{{kN}}")], tag=tag)
        dem = abs(chk.demand)
        vmax = d["V_section_max"]
        w.guard(1.05 * d["beta_h"] * d["ftd"] * d["Um"] * d["h0"] / 1e3, vmax)
        w.eq([("γ_{0}F_{ld}", f"{n(dem, 1)}\\,\\u{{kN}}"),
              w.cmp(dem, vmax, f"1.05β_{{h}}f_{{td}}U_{{m}}h_{{0}} = 1.05 × {v('beta_h', d['beta_h'])} ×"
                               f" {v('ftd', d['ftd'])} × "
                               f"{v('Um', d['Um'])} × {v('h0', d['h0'])} = {v('Vmax', vmax, 1)}\\,\\u{{kN}}")],
             tag="5.6.2-1")
        if "V_outer" in d:
            out = (0.7 * d["beta_h"] * d["ftd"] + 0.15 * sp) * d["Um_outer"] * d["h0"] / 1e3
            w.guard(out, d["V_outer"])
            w.p(tt("配筋区以外 $0.5h_{0}$ 处的周长上按 5.6.1 验算：", "Outside the punching steel, 0.5 h0 beyond it, by 5.6.1:"))
            w.eq([("γ_{0}F_{ld}", f"{n(dem, 1)}\\,\\u{{kN}}"),
                  w.cmp(dem, out, f"(0.7β_{{h}}f_{{td}}{sp_s})U_{{m}}h_{{0}} = (0.7 × {v('beta_h', d['beta_h'])} × "
                                  f"{v('ftd', d['ftd'])}{sp_v}) × {v('Umo', d['Um_outer'])} × {v('h0', d['h0'])} = "
                                  f"{v('Vout', out, 1)}\\,\\u{{kN}}")], tag="5.6.1")
        return True
    if "bm" in d and "ps" in d:
        w.units()
        Fld = d["ps"] * d["A"] / 1e3
        bm = (d["bt"] + d["bb"]) / 2
        R = 0.7 * d["beta_h"] * d["ftd"] * bm * d["h0"] / 1e3
        w.guard(Fld, d["Fld"])
        w.guard(bm, d["bm"])
        w.guard(R, chk.resistance)
        w.eq([("F_{ld}", f"p_{{s}}A = {v('ps', d['ps'])} × {v('A', d['A'])} = {v('Fld', Fld, 1)}\\,\\u{{kN}}")],
             tag="5.6.3-2")
        w.eq([("b_{m}", f"\\frac{{b_{{t}} + b_{{b}}}}{{2}} = \\frac{{{v('bt', d['bt'])} + {v('bb', d['bb'])}}}{{2}} = "
                        f"{v('bm', bm)}\\,\\u{{mm}}")], tag="5.6.3-3")
        w.eq([("F_{u}", f"0.7β_{{h}}f_{{td}}b_{{m}}h_{{0}} = 0.7 × {v('beta_h', d['beta_h'])} × {v('ftd', d['ftd'])} × "
                        f"{v('bm', bm)} × {v('h0', d['h0'])} = {v('Fu', R, 1)}\\,\\u{{kN}}")], tag="5.6.3-1")
        return True
    return False


# ── bearings and expansion joints ───────────────────────────────────────────
def _tpl_devices(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    v, tt, n = w.v, w.t, w.n
    nm = chk.name
    if nm.startswith("bearing area"):
        sig = d["Rck"] * 1e3 / d["Ae"]
        w.guard(sig, chk.demand)
        w.eq([("\\frac{R_{ck}}{A_{e}}", f"\\frac{{{v('Rck', d['Rck'], 1)}×10^{{3}}}}{{{v('Ae', d['Ae'])}}} = "
                                        f"{v('sig', sig, 2)}\\,\\u{{MPa}}"),
              w.cmp(sig, chk.resistance, f"σ_{{c}} = {n(chk.resistance, 1)}\\,\\u{{MPa}}")], tag="8.7.3-1")
        return True
    if nm.startswith("rubber thickness, shear"):
        k = 2.0 if "without" in nm else 1.43
        dl, dt = d["dl"], d.get("dt", 0.0)
        val = k * math.hypot(dl, dt)
        w.guard(val, chk.demand)
        ks = "2" if k == 2.0 else "1.43"
        lhs = f"{ks}\\sqrt{{Δ_{{l}}^{{2}} + Δ_{{t}}^{{2}}}}" if dt else f"{ks}Δ_{{l}}"
        sub = f"{ks} × \\sqrt{{{v('dl', dl, 2)}^{{2}} + {v('dt', dt, 2)}^{{2}}}}" if dt else f"{ks} × {v('dl', dl, 2)}"
        w.eq([(lhs, f"{sub} = {v('req', val, 2)}\\,\\u{{mm}}"),
              w.cmp(val, chk.resistance, f"t_{{e}} = {n(chk.resistance, 1)}\\,\\u{{mm}}")], tag=chk.clause.split()[-1])
        return True
    if nm.startswith("rubber thickness, stability"):
        te, la = d["te"], d["la"]
        w.guard(la / 5, chk.resistance)
        w.eq([("t_{e}", f"{v('te', te, 1)}\\,\\u{{mm}}"),
              w.cmp(te, la / 10, f"\\frac{{l_{{a}}}}{{10}} = \\frac{{{v('la', la)}}}{{10}} ="
                                 f" {n(la / 10, 1)}\\,\\u{{mm}}",
                    ge=True)], tag=chk.clause.split()[-1])
        w.eq([("t_{e}", f"{v('te', te, 1)}\\,\\u{{mm}}"),
              w.cmp(te, la / 5, f"\\frac{{l_{{a}}}}{{5}} = \\frac{{{v('la', la)}}}{{5}} = {n(la / 5, 1)}\\,\\u{{mm}}")],
             tag=chk.clause.split()[-1])
        w.hide_notes = []
        return True
    if nm.startswith("compression of the bearing") and "Rck" in d:
        delta = d["Rck"] * 1e3 * d["te"] / (d["Ae"] * d["Ee"]) + d["Rck"] * 1e3 * d["te"] / (d["Ae"] * d["Eb"])
        w.guard(delta, d["delta_cm"])
        w.eq([("δ_{c,m}", "\\frac{R_{ck}t_{e}}{A_{e}E_{e}} + \\frac{R_{ck}t_{e}}{A_{e}E_{b}}"),
              ("", f"\\frac{{{v('Rck', d['Rck'], 1)}×10^{{3}} × {v('te', d['te'])}}}{{{v('Ae', d['Ae'])} ×"
                   f" {v('Ee', d['Ee'])}}}"
                   f" + \\frac{{{v('Rck', d['Rck'], 1)}×10^{{3}} × {v('te', d['te'])}}}{{{v('Ae', d['Ae'])} × "
                   f"{v('Eb', d['Eb'])}}} = {v('dcm', delta, 3)}\\,\\u{{mm}}")], tag="8.7.3-8")
        w.eq([("δ_{c,m}", f"{v('dcm', delta, 3)}\\,\\u{{mm}}"),
              w.cmp(delta, chk.resistance, f"0.07t_{{e}} = {n(chk.resistance, 3)}\\,\\u{{mm}}")], tag="8.7.3-9")
        w.eq([("δ_{c,m}", f"{v('dcm', delta, 3)}\\,\\u{{mm}}"),
              w.cmp(delta, d["uplift_limit"], f"θ\\frac{{l_{{a}}}}{{2}} = {n(d['uplift_limit'], 3)}\\,\\u{{mm}}",
                    ge=True)],
             tag="8.7.3-9")
        return True
    if nm.startswith("steel plates") and "Rck" in d:
        ts = d["Kp"] * d["Rck"] * 1e3 * d["tes_u_plus_l"] / (d["Ae"] * d["sigma_s"])
        w.guard(ts, chk.demand)
        w.eq([("t_{s,req}", "\\frac{K_{p}R_{ck}(t_{es,u} + t_{es,l})}{A_{e}σ_{s}}"),
              ("", f"\\frac{{{v('Kp', d['Kp'])} × {v('Rck', d['Rck'], 1)}×10^{{3}} × {v('tes', d['tes_u_plus_l'])}}}"
                   f"{{{v('Ae', d['Ae'])} × {v('sigma_s', d['sigma_s'])}}} = {v('ts_req', ts, 2)}\\,\\u{{mm}}"),
              w.cmp(ts, chk.resistance, f"t_{{s}} = {n(chk.resistance, 1)}\\,\\u{{mm}}")], tag="8.7.3-10")
        return True
    if nm.startswith("sliding"):
        R = d["R"]
        H = 1.4 * d["Ge"] * d["Ag"] * d["dl"] / d["te"] / 1e3 + d.get("Fbk", 0.0)
        w.guard(H, chk.demand)
        w.guard(d["mu"] * R, chk.resistance)
        fb = " + F_{bk}" if "Fbk" in d else ""
        fbv = f" + {v('Fbk', d['Fbk'], 1)}" if "Fbk" in d else ""
        Rsym = "R_{ck}" if "Fbk" in d else "R_{Gk}"
        w.eq([("1.4G_{e}A_{g}\\frac{Δ_{l}}{t_{e}}" + fb,
               f"1.4 × {v('Ge', d['Ge'])} × {v('Ag', d['Ag'])} × \\frac{{{v('dl', d['dl'], 2)}}}{{{v('te', d['te'])}}}"
               f" × "
               f"10^{{-3}}{fbv} = {v('H', H, 1)}\\,\\u{{kN}}"),
              w.cmp(H, chk.resistance, f"μ{Rsym} = {v('mu', d['mu'])} × {v('R', R, 1)} ="
                                       f" {n(chk.resistance, 1)}\\,\\u{{kN}}")],
             tag=chk.clause.split()[-1])
        if "Fbk" in d:
            w.p(tt("$R_{ck} = R_{Gk} + 0.5R_{Qk}$（8.7.4）。", "$R_{ck} = R_{Gk} + 0.5R_{Qk}$ (8.7.4)."))
        return True
    if nm.startswith("PTFE friction"):
        F = d["mu_f"] * d["R"]
        R = d["Ge"] * d["Ag"] * d["tan_alpha"] / 1e3
        w.guard(F, chk.demand)
        w.guard(R, chk.resistance)
        Rs = "R_{Gk}" if "without" in nm else "(R_{Gk} + R_{Qk})"
        w.eq([(f"μ_{{f}}{Rs}", f"{v('muf', d['mu_f'])} × {v('R', d['R'], 1)} = {v('muR', F, 1)}\\,\\u{{kN}}"),
              w.cmp(F, R, f"G_{{e}}A_{{g}}\\op{{tan}}α = {v('Ge', d['Ge'])} × {v('Ag', d['Ag'])} ×"
                          f" {v('tana', d['tan_alpha'])} × "
                          f"10^{{-3}} = {n(R, 1)}\\,\\u{{kN}}")], tag=chk.clause.split()[-1])
        return True
    if nm.startswith("expansion joint"):
        b = d["beta"]
        Cp = b * (d["t_plus"] + d["brake"])
        Cm = b * (d["t_minus"] + d["shrinkage"] + d["creep"] + d["brake"])
        w.guard(Cp, d["C_plus"])
        w.guard(Cm, d["C_minus"])
        w.guard(Cp + Cm, chk.demand)
        w.eq([("C^{+}", f"β(Δl_{{t}}^{{+}} + Δl_{{b}}^{{+}}) = {v('beta', b)} × ({v('tp', d['t_plus'], 1)} + "
                        f"{v('br', d['brake'], 1)}) = {v('Cp', Cp, 1)}\\,\\u{{mm}}")], tag="8.8.2-5")
        w.eq([("C^{-}", f"β(Δl_{{t}}^{{-}} + Δl_{{s}}^{{-}} + Δl_{{c}}^{{-}} + Δl_{{b}}^{{-}}) = {v('beta', b)} × "
                        f"({v('tm', d['t_minus'], 1)} + {v('sh', d['shrinkage'], 1)} + {v('cr', d['creep'], 1)} + "
                        f"{v('br', d['brake'], 1)}) = {v('Cm', Cm, 1)}\\,\\u{{mm}}")], tag="8.8.2-6")
        w.eq([("C^{+} + C^{-}", f"{n(Cp + Cm, 1)}\\,\\u{{mm}}"),
              w.cmp(Cp + Cm, chk.resistance, f"C = {n(chk.resistance, 1)}\\,\\u{{mm}}")], tag="8.8.2-7")
        return True
    return False


# ── detailing reports ───────────────────────────────────────────────────────
def _tpl_detailing(w: _W) -> bool:
    chk = w.chk
    if not chk.detailing:
        return False
    L = _L[w.lang]
    rows = []
    for k, val in chk.details.items():
        if isinstance(val, dict) and "value" in val and "limit" in val:
            bad = any(x.startswith(k + " ") or x.startswith(k + ":") for x in chk.warnings)
            name = (TX.rule_zh(k) or k) if w.zh else k
            rows.append([name, _fmt_val(val["value"], w.lang), _fmt_val(val["limit"], w.lang),
                         Cell(L["rule_fail"] if bad else L["rule_ok"], "fail" if bad else "ok")])
    if rows:
        w.node.table(rows, header=[L["rules"], L["actual"], L["limit"], L["verdict"]], compact=True,
                     row_cls=["fail" if r[3].cls == "fail" else "" for r in rows], align="lrrc")
    other = {k: val for k, val in chk.details.items() if not (isinstance(val, dict) and "value" in val)}
    if other:
        _details_table(w, other)
    if not rows and not other and not chk.warnings:
        w.p(w.t("各项构造要求满足。", "All rules met."))
    return True


# ── the general form ────────────────────────────────────────────────────────
_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "eta": "η", "zeta": "ζ", "sigma": "σ", "tau": "τ", "rho": "ρ",
          "phi": "φ", "lambda": "λ", "mu": "μ", "nu": "ν", "psi": "ψ", "theta": "θ", "xi": "ξ", "delta": "δ",
          "eps": "ε", "Omega": "Ω", "kappa": "κ"}


def _key_symbol(k: str) -> str | None:
    """a key of Check.details as a symbol: alpha_px → α_{px}, Nu0 → N_{u0}; None when it is not one"""
    parts = k.split("_")
    base = parts[0]
    if base in _GREEK:
        sym, rest = _GREEK[base], parts[1:]
    else:
        mt = re.fullmatch(r"([A-Za-z])([a-z0-9]{0,3})", base)
        if not mt or any(len(p_) > 4 for p_ in parts[1:]):
            return None
        sym, rest = mt.group(1), ([mt.group(2)] if mt.group(2) else []) + parts[1:]
    if any(not re.fullmatch(r"[A-Za-z0-9]+", p_) for p_ in rest):
        return None
    return sym + (f"_{{{','.join(rest)}}}" if rest else "")


def _key_name(k: str, lang: str, check: str = "") -> tuple[str, str]:
    """(name, unit) of a value of Check.details for the table of the general form"""
    special = next((v for (c, kk), v in TX.KEYS_IN.items() if kk == k and check.startswith(c)), None)
    if special is not None or k in TX.KEYS:
        name, unit = special or TX.KEYS[k]
        if lang != "zh":
            sym = re.search(r"\$([^$]+)\$", name)
            return (f"${sym.group(1)}$ ({k})" if sym else k), {"年": "years"}.get(unit, unit)
        return name, unit
    if lang == "zh":
        rz = TX.rule_zh(k)
        if rz is not None:
            return rz, ""
    sym = _key_symbol(k)
    return (f"${sym}$" if sym else k), ""


def _fmt_val(val, lang: str = "zh"):
    if isinstance(val, (bool, np.bool_)):
        return ("是" if val else "否") if lang == "zh" else ("yes" if val else "no")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, np.ndarray):
        a = val.ravel()
        if a.size <= 6 and np.issubdtype(a.dtype, np.number):
            return "、".join(num(float(x)) for x in a) if lang == "zh" else ", ".join(num(float(x)) for x in a)
        if np.issubdtype(a.dtype, np.number) and a.size:
            return (f"{a.size} 项，{num(float(np.nanmin(a)))} ～ {num(float(np.nanmax(a)))}" if lang == "zh" else
                    f"{a.size} values, {num(float(np.nanmin(a)))} … {num(float(np.nanmax(a)))}")
        return f"{a.size} 项" if lang == "zh" else f"{a.size} values"
    if isinstance(val, (list, tuple)):
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in val) and len(val) <= 6:
            return ("、" if lang == "zh" else ", ").join(num(float(x)) for x in val)
        s = ", ".join(str(x) if not isinstance(x, float) else num(x) for x in val)
        if lang == "zh":
            s = f"{len(val)} 项" if re.search(r"[A-Za-z]{2,}", s) else s
        return s if len(s) < 120 else s[:117] + "…"
    if val is None:
        return "—"
    s = str(val)
    if lang == "zh":
        s = TX.VALUES.get(s, TX.word(s))
        if re.search(r"[A-Za-z]{3,}", s):
            return "—"
    return s if len(s) < 120 else s[:117] + "…"


def _flatten(d: dict):
    """(parent key or None, key, value) of Check.details, one level of nested dicts opened, lists of dicts counted"""
    for k, val in d.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                    v2 = len(v2)
                yield k, k2, v2
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            yield None, k, len(val)
        else:
            yield None, k, val


def _details_table(w: _W, details: dict):
    rows = []
    for head, k, val in _flatten(details):
        name, unit = _key_name(str(k), w.lang, w.chk.name)
        if head is not None:
            hname, _ = _key_name(str(head), w.lang, w.chk.name)
            name = f"{hname}：{name}" if w.zh else f"{hname}: {name}"
        rows.append([name, Cell(_fmt_val(val, w.lang), digits=None), unit])
    if rows:
        L = _L[w.lang]
        w.node.table(rows, header=[L["param"], L["value"], L["unit"]], compact=True, align="lrl", keep=len(rows) <= 30)


def _tpl_general(w: _W) -> bool:
    if w.chk.section is not None and w.chk.name.startswith(("flexure", "eccentric", "tension")):
        w.p(_L[w.lang]["general"])
    _details_table(w, w.chk.details)
    return True


_TEMPLATES = [
    (lambda c: c.name.startswith("flexure"), _tpl_flexure),
    (lambda c: c.name.startswith("shear (slab without"), _tpl_slab_shear),
    (lambda c: c.name.startswith("section limit"), _tpl_section_limit),
    (lambda c: c.name.startswith("shear") and "Vcs" in c.details, _tpl_shear),
    (lambda c: c.name == "axial compression", _tpl_axial),
    (lambda c: c.name == "eccentric compression" or c.name == "biaxial eccentric compression (circle)", _tpl_eccentric),
    (lambda c: c.name.startswith("tension"), _tpl_tension),
    (lambda c: c.name.startswith("torsion"), _tpl_torsion),
    (lambda c: c.name in ("crack width",), _tpl_crack_width),
    (lambda c: c.name.startswith("crack resistance (normal"), _tpl_crack_resistance),
    (lambda c: c.name in ("crack resistance (inclined section)", "principal compressive stress (service)"),
     _tpl_principal),
    (lambda c: c.name.startswith("concrete compression in service (") and "composite" not in c.name, _tpl_service),
    (lambda c: c.name.startswith("deflection"), _tpl_deflection),
    (lambda c: c.name.startswith("transient stresses (prestressed"), _tpl_transient),
    (lambda c: c.name == "transient stresses (reinforced concrete)", _tpl_transient_rc),
    (lambda c: c.name.startswith("local compression"), _tpl_local),
    (lambda c: c.name.startswith("punching"), _tpl_punching),
    (lambda c: c.clause.startswith("JTG 3362-2018 8.7") or c.clause.startswith("JTG 3362-2018 8.8"), _tpl_devices),
]


# ── the verdict ─────────────────────────────────────────────────────────────
_SYM = [("flexure", "M_{u}", "γ_{0}M_{d}"), ("cap beam flexure", "M_{u}", "γ_{0}M_{d}"),
        ("shear share", "V_{u}", "ξγ_{0}V'_{d}"),
        ("shear", "V_{u}", "γ_{0}V_{d}"), ("section limit", "0.51×10^{-3}\\sqrt{f_{cu,k}}bh_{0}", "γ_{0}V_{0d}"),
        ("axial", "N_{u}", "γ_{0}N_{d}"), ("eccentric", "N_{u}", "γ_{0}N_{d}"), ("biaxial", "N_{u}", "γ_{0}N_{d}"),
        ("tension", "N_{u}", "γ_{0}N_{d}"), ("torsion", "T_{u}", "γ_{0}T_{d}"),
        ("local compression", "F_{u}", "γ_{0}F_{ld}"), ("punching", "F_{u}", "γ_{0}F_{ld}"),
        ("pile cap punching", "F_{u}", "γ_{0}F_{ld}")]
_LIMIT_SYM = [
    ("crack width", ("W_{cr}", "[W_{cr}]")), ("deflection", ("f", "L/600")),
    ("crack resistance (normal section, class A", ("σ_{st}", "σ_{pc} + 0.7f_{tk}")),
    ("crack resistance (normal section, full prestress", ("σ_{st}", "kσ_{pc}")),
    ("crack resistance (normal section, class B", ("σ_{G}", "σ_{pc}")),
    ("crack resistance (inclined", ("σ_{tp}", "[σ_{tp}]")), ("principal compressive", ("σ_{cp}", "0.6f_{ck}")),
    ("concrete compression in service (cracked", ("σ_{cc}", "0.5f_{ck}")),
    ("concrete compression in service", ("σ_{kc} + σ_{pt}", "0.5f_{ck}")),
    ("transient stresses (reinforced concrete)", ("σ^{t}_{cc}", "0.80f'_{ck}")),
    ("transient shear", ("σ^{t}_{tp}", "f'_{tk}")),
    ("transient stresses (prestressed", ("σ^{t}_{cc}", "0.70f'_{ck}")),
    ("bearing area", ("R_{ck}/A_{e}", "σ_{c}")), ("rubber thickness, shear without", ("2Δ_{l}", "t_{e}")),
    ("rubber thickness, shear with", ("1.43Δ_{l}", "t_{e}")), ("rubber thickness, stability", ("t_{e}", "l_{a}/5")),
    ("compression of the bearing", ("δ_{c,m}", "0.07t_{e}")), ("steel plates", ("t_{s,req}", "t_{s}")),
    ("sliding without", ("1.4G_{e}A_{g}Δ_{l}/t_{e}", "μR_{Gk}")),
    ("sliding with", ("1.4G_{e}A_{g}Δ_{l}/t_{e} + F_{bk}", "μR_{ck}")),
    ("PTFE friction", ("μ_{f}R", "G_{e}A_{g}\\op{tan}α")), ("expansion joint", ("C^{+} + C^{-}", "C")),
    ("lifting loops", ("σ", "[σ]")), ("in-plane principal tension", ("σ_{tp}", "[σ_{tp}]")),
    ("in-plane principal compression", ("σ_{cp}", "0.6f_{ck}")),
]


def _syms(chk: Check):
    if chk.limit_check:
        for k, pair in _LIMIT_SYM:
            if chk.name.startswith(k):
                if k == "deflection" and chk.details.get("cantilever"):
                    return "f", "L/300"
                return pair
        s = chk.symbol.replace("γ0", "γ_{0}")
        return s, f"[{s}]"
    r_sym, d_sym = "R", "γ_{0}S_{d}"
    for k, rs, ds in _SYM:
        if chk.name.startswith(k):
            r_sym, d_sym = rs, ds
            break
    return d_sym, r_sym


def _verdict_parts(chk: Check, lang: str) -> list:
    """the verdict line: the comparison with its relation by the numbers, and the ratio or the margin"""
    L = _L[lang]
    a, b = _syms(chk)
    sep = L["sep"]
    if chk.limit_check:
        p = 3 if chk.unit == "mm" else 2
        u = f"\\,\\u{{{chk.unit}}}" if chk.unit else ""
        rel = _W.rel(chk.demand, chk.resistance)
        parts = [m(f"{a} = {fixed(chk.demand, p)}{u} {rel} {b} = {fixed(chk.resistance, p)}{u}")]
        if chk.resistance > 0:
            parts.append(f"{sep}{L['ratio']} {chk.ratio:.3f}")
        else:
            parts.append(f"{sep}{L['margin']} {fixed(chk.resistance - chk.demand, p)}"
                         + (f" {chk.unit}" if chk.unit else ""))
        return parts
    u = f"\\,\\u{{{chk.unit}}}" if chk.unit else ""
    rel = _W.rel(abs(chk.demand), chk.resistance)
    return [m(f"{a} = {fixed(abs(chk.demand), 1)}{u} {rel} {b} = {fixed(chk.resistance, 1)}{u}"),
            f"{sep}{L['ratio']} {chk.ratio:.3f}"]


def _failed_clause(chk: Check) -> str:
    """the clause of the first condition not met (for the summary)"""
    for f in chk.failed:
        r = re.search(r"\(([^()]*)\)\s*$", f)
        if r:
            return r.group(1)
    return ""


# ── public ──────────────────────────────────────────────────────────────────
@dataclass
class Entry:
    """a check written into a report: where, the check, its title, the id of its section, whether it counts as a check
    of its own (False for a design back-substitution such as the stirrup design), the number of results its formulas
    reproduced (0: written in the general form) and the clauses written (those of the calculation shown)"""
    where: str
    check: Check
    title: str
    id: str
    counted: bool = True
    guards: int = 0
    clause: str = ""


_FIGURE_CHECKS = ("flexure", "eccentric compression", "axial compression", "tension", "concrete compression in service",
                  "crack resistance (normal section")


def write_check(node: Node, chk: Check, *, where: str | None = None, title: str | None = None, id: str | None = None,
                lang: str | None = None, figure: bool | str = "auto", figure_kw: dict | None = None,
                formulas: bool = True, details: bool = False, counted: bool = True) -> Node:
    """write a check as a numbered section under `node` (see the module notes). where: the location ("x = 25.50 m（跨中）");
    title: default the location and the check's name; id: for cross-references (default "chk:<n>"); figure: the section
    drawn with the check (codes.jtg.figures.section_figure, figure_kw passed on): True, False, or "auto" (default) for
    the checks the drawing shows something of (flexure, compression, tension, stresses, normal-section crack resistance,
    section detailing); formulas: False writes the general form only; details: also the table of Check.details;
    counted: False for a design step that is not a check of its own (summary shows it apart and does not count it).
    Returns the new section."""
    lang = _lang(node, lang)
    L = _L[lang]
    store = node.report.store.setdefault("jtg.checks", [])
    uid = id or f"chk:{len(store) + 1}"
    name = check_name(chk, lang)
    head = title or (f"{where} {name}" if (where and lang == "zh") else (f"{name}, {where}" if where else name))
    sec = node.section(head, id=uid)
    w = _W(sec, chk, lang)
    sec.paragraph("", cls="muted")  # the location and clause, filled in below (the template may refine the clause)
    info_block = sec.blocks[-1]
    done = False
    if formulas and not chk.detailing:
        for pred, tpl in _TEMPLATES:
            if not pred(chk):
                continue
            mark = len(sec.blocks)
            units_said = w.units_said
            try:
                if tpl(w):
                    done = True
                    break
            except (_Mismatch, KeyError, TypeError, ValueError, ZeroDivisionError, IndexError, AttributeError) as exc:
                if _TRACE is not None:
                    import traceback
                    _TRACE.append((chk.name, exc, traceback.format_exc()))
            del sec.blocks[mark:]
            w.units_said, w.hide_notes, w.plain_figure, w.clause, w.guards = units_said, [], False, None, 0
            break
    elif chk.detailing:
        done = _tpl_detailing(w)
    if not done:
        _tpl_general(w)
    elif details and not chk.detailing:
        _details_table(w, chk.details)
    info = []
    if where:
        info.append(f"{L['where']}{L['colon']}{where}")
    info.append(f"{L['clause']}{L['colon']}{clause_text(w.clause or chk.clause, lang)}")
    info_block.parts = (L["semi"].join(info) + L["end"],)
    # the verdict
    if chk.detailing:
        if chk.warnings:
            sec.box("warn", title=L["warnings"], items=[message(x, lang) for x in chk.warnings])
        else:
            sec.box("ok", title=L["det_ok"] if lang == "en" else "构造满足要求")
    else:
        ok = chk.ok
        lines = []
        if chk.failed:
            lines.append(Markup(f"<div>{_html_text(message(chk.failed[0], lang))}</div>"))
            lines.append(Markup("<div>" + "".join(str(_md(x)) for x in _verdict_parts(chk, lang)) + "</div>"))
        else:
            lines += _verdict_parts(chk, lang)
        sec.box("ok" if ok else "fail", *lines, title=(L["ok"] if ok else L["not"]))
        if len(chk.failed) > 1:
            sec.box("fail", title=L["failed"], items=[message(x, lang) for x in chk.failed[1:]])
        if chk.warnings:
            sec.box("warn", title=L["warnings"], items=[message(x, lang) for x in chk.warnings])
    notes = [x for x in chk.notes if x not in w.hide_notes]
    if notes:
        sec.box("note", title=L["notes"], items=[message(x, lang) for x in notes])
    if figure == "auto":
        figure = chk.name.startswith(_FIGURE_CHECKS) or chk.name.startswith("section detailing")
    if figure and chk.section is not None:
        from .figures import section_figure
        kw = {"lang": lang, "title": "", "width": 560}
        kw.update(figure_kw or {})
        try:
            fig = section_figure(chk.section, check=None if w.plain_figure else chk, **kw)
        except (TypeError, ValueError, NotImplementedError):
            fig = None
        if fig is not None:
            cap = (f"{where} {name}" if where else name) if lang == "zh" else (f"{name}, {where}" if where else name)
            sec.figure(fig, caption=cap, width="560px")
    store.append(Entry(where or "", chk, name, uid, counted, w.guards if done else 0, w.clause or chk.clause))
    return sec


def _html_text(s: str) -> Markup:
    from ...report import rich
    return rich(s)


def _md(x):
    from ...report import rich
    return rich(x)


def write_checks(node: Node, items, **kw) -> list[Node]:
    """write_check for each (where, Check) pair"""
    return [write_check(node, c, where=w, **kw) for w, c in items]


def summary(node: Node, entries=None, *, lang: str | None = None, id: str = "tab:summary",
            warn_id: str = "tab:warnings", caption: str | None = None) -> Node:
    """the summary of the checks (default: every check written with write_check so far): one table of all checks
    (section, location, check, clause, γ0Sd, R or limit, ratio or margin, verdict — a check not met in red with the
    clause of the condition not met; each linked to its section), one of the warnings, and the conclusion. entries:
    Entry records or (where, Check) pairs. Design steps written with counted=False are listed apart from the count."""
    lang = _lang(node, lang)
    L = _L[lang]
    rep = node.report
    if entries is None:
        entries = list(rep.store.get("jtg.checks", []))
    ents = [e if isinstance(e, Entry) else Entry(e[0], e[1], check_name(e[1], lang), "") for e in entries]
    rows, cls = [], []
    short = {"zh": ("满足", "不满足"), "en": ("OK", "NOT OK")}[lang]

    def cl(e):
        return clause_text(e.clause or e.check.clause, lang).replace("JTG 3362—2018 ", "").replace("JTG 3362-2018 ", "")
    for k, e in enumerate(ents, 1):
        c = e.check
        sec_ref = rep.ref(e.id, bare=True) if e.id else k  # the section the check is written in, linked
        title = e.title + ("" if e.counted else L["design"])
        if c.detailing:
            verdict = L["det_warn"].format(k=len(c.warnings)) if c.warnings else short[0]
            rows.append([sec_ref, e.where, title, cl(e), "", "", "", "", Cell(verdict, "warn" if c.warnings else "ok")])
            cls.append("warn" if c.warnings else "")
            continue
        p = 3 if c.unit == "mm" else (2 if c.limit_check else 1)
        ok = c.ok
        if c.limit_check and c.resistance <= 0:
            ratio = Cell(f"{L['margin']} {fixed(c.resistance - c.demand, p)}")
        else:
            ratio = Cell(c.ratio if math.isfinite(c.ratio) else "∞", digits=3)
        fc = _failed_clause(c)
        verdict = short[0] if ok else (short[1] + (f"（{TX.clause(fc)}）" if (fc and lang == "zh") else
                                                    (f" ({fc})" if fc else "")))
        rows.append([sec_ref, e.where, title, cl(e), Cell(abs(c.demand) if not c.limit_check else c.demand, digits=p),
                     Cell(c.resistance, digits=p), Cell(c.unit, "unit"), ratio, Cell(verdict, "ok" if ok else "fail")])
        cls.append(("" if ok else "fail") if e.counted else "sub")
    head = [L["section"], L["loc"], L["item"], L["clause"], L["demand"], L["resist"], L["unit"], L["ratio"],
            L["verdict"]]
    node.table(rows, header=head, caption=caption or L["summary_cap"], id=id, row_cls=cls,
               align="llllrrlrc", keep=False, cls="summary fixed",
               widths=["4.6em", "17%", None, "11%", "5.2em", "5.2em", "3.6em", "4.4em", "5em"])
    counted = [e for e in ents if e.counted and not e.check.detailing]
    n = len(counted)
    bad = sum(1 for e in counted if not e.check.ok)
    warns = [(e, w) for e in ents for w in e.check.warnings]
    if warns:
        wrows = [[k, rep.ref(e.id, bare=True) if e.id else "", e.where, e.title, message(w, lang)]
                 for k, (e, w) in enumerate(warns, 1)]
        node.table(wrows, header=[L["no"], L["section"], L["loc"], L["item"], L["content"]],
                   caption=L["warn_cap"], id=warn_id, align="cllll", keep=False, cls="summary fixed",
                   widths=["2.6em", "4.8em", "15%", "22%", None])
    concl = L["all_ok"].format(n=n) if not bad else L["some_fail"].format(n=n, k=bad)
    title = "结论" if lang == "zh" else "Conclusion"
    if warns:
        node.box("ok" if not bad else "fail", concl, L["warn_n"].format(k=len(warns)), rep.ref(warn_id), L["warn_n2"],
                 title=title)
    else:
        node.box("ok" if not bad else "fail", concl, L["no_warn"], title=title)
    return node


# ── design data ─────────────────────────────────────────────────────────────
CODES = {
    "zh": [("JTG D60—2015", "《公路桥涵设计通用规范》"), ("JTG 3362—2018", "《公路钢筋混凝土及预应力混凝土桥涵设计规范》")],
    "en": [("JTG D60-2015", "General Specifications for Design of Highway Bridges and Culverts"),
           ("JTG 3362-2018", "Specifications for Design of Highway Reinforced Concrete and Prestressed Concrete"
                             " Bridges "
                             "and Culverts")],
}


def design_basis(node: Node, extra=(), *, lang: str | None = None) -> Node:
    """the codes the report follows (JTG D60-2015, JTG 3362-2018 and `extra` rows of (number, title))"""
    lang = _lang(node, lang)
    items = [f"{a}{b}" if lang == "zh" else f"{a}: {b}" for a, b in list(CODES[lang]) + list(extra)]
    node.list(items, ordered=True)
    return node


def _bars_text(sec, lang: str) -> list:
    """the rows of bars of a section: 'n Φd' and the depth"""
    out = []
    for (a, z), dd in zip(sec.bars, sec.bar_d):
        if dd and not math.isnan(dd):
            k = a / (math.pi * dd * dd / 4)
            txt = f"{k:.0f}Φ{dd:g}" if abs(k - round(k)) < 0.02 else f"A = {a:.0f} mm²（Φ{dd:g}）"
        else:
            txt = f"A = {a:.0f} mm²"
        out.append((txt, z))
    return out


_TENDON_ZH = {"strand": "钢绞线", "wire": "消除应力钢丝", "bar": "预应力螺纹钢筋"}
_TENDON_EN = {"strand": "strand 1×7", "wire": "stress-relieved wire", "bar": "prestressing threaded bar"}


def materials(node: Node, sections, *, lang: str | None = None, caption: str | None = None,
              id: str | None = "tab:materials") -> Node:
    """the material values of the sections' concrete, bars and tendons (tables 3.1.3–3.1.5, 3.2.2–3.2.4)"""
    lang = _lang(node, lang)
    secs = list(sections.values()) if isinstance(sections, dict) else list(sections)
    concretes, bars, tendons = {}, {}, {}
    for s in secs:
        s = getattr(s, "precast", s)
        concretes[s.concrete.grade] = s.concrete
        bars[s.steel.name] = s.steel
        if getattr(s, "tendon", None) is not None and s.has_internal:
            tendons[(s.tendon.kind, s.tendon.fpk)] = s.tendon
        if getattr(s, "ext_tendon", None) is not None and s.has_external:
            tendons[(s.ext_tendon.kind, s.ext_tendon.fpk)] = s.ext_tendon
    zh = lang == "zh"
    rows = []
    for g, c in sorted(concretes.items()):
        nm = f"{'混凝土' if zh else 'Concrete'} C{g}"
        for label, sym, val in [("轴心抗压强度标准值" if zh else "characteristic compressive strength", "f_{ck}", c.fck),
                                ("轴心抗拉强度标准值" if zh else "characteristic tensile strength", "f_{tk}", c.ftk),
                                ("轴心抗压强度设计值" if zh else "design compressive strength", "f_{cd}", c.fcd),
                                ("轴心抗拉强度设计值" if zh else "design tensile strength", "f_{td}", c.ftd),
                                ("弹性模量" if zh else "modulus of elasticity", "E_{c}", c.Ec)]:
            rows.append([nm, label, f"${sym}$", Cell(val, digits=None if val < 1e4 else 0), "MPa"])
            nm = ""
    for n_, st in bars.items():
        nm = f"{'普通钢筋' if zh else 'Reinforcing bars'} {n_}"
        for label, sym, val in [("抗拉强度标准值" if zh else "characteristic strength", "f_{sk}", st.fsk),
                                ("抗拉强度设计值" if zh else "design tensile strength", "f_{sd}", st.fsd),
                                ("抗压强度设计值" if zh else "design compressive strength", "f'_{sd}", st.fsd_c),
                                ("弹性模量" if zh else "modulus of elasticity", "E_{s}", st.Es)]:
            rows.append([nm, label, f"${sym}$", Cell(val, digits=0), "MPa"])
            nm = ""
    for (kind, fpk), t in tendons.items():
        nm = (f"预应力钢筋（{_TENDON_ZH.get(kind, kind)}，$f_{{pk}}$ = {fpk:g} MPa）" if zh else
              f"Prestressing steel ({_TENDON_EN.get(kind, kind)}, fpk {fpk:g} MPa)")
        for label, sym, val in [("抗拉强度标准值" if zh else "characteristic strength", "f_{pk}", t.fpk),
                                ("抗拉强度设计值" if zh else "design tensile strength", "f_{pd}", t.fpd),
                                ("抗压强度设计值" if zh else "design compressive strength", "f'_{pd}", t.fpd_c),
                                ("弹性模量" if zh else "modulus of elasticity", "E_{p}", t.Ep),
                                ("张拉控制应力限值" if zh else "largest jacking stress", "σ_{con}", t.sigma_con_max())]:
            rows.append([nm, label, f"${sym}$", Cell(val, digits=0 if abs(val - round(val)) < 1e-9 else 1), "MPa"])
            nm = ""
    node.table(rows, header=["材料" if zh else "Material", "项目" if zh else "Property", "符号" if zh else "Symbol",
                             "数值" if zh else "Value", "单位" if zh else "Unit"],
               caption=caption or ("材料参数" if zh else "Materials"), id=id, align="lllrl")
    return node


def section_properties(node: Node, sections: dict, *, lang: str | None = None, caption: str | None = None,
                       id: str | None = "tab:sections", transformed: bool = True, hogging=()) -> Node:
    """the gross (and transformed) section properties and the reinforcement of each section: {name: RCSection}.
    hogging: the names of the sections checked under a negative moment: As, A's and h0 are those of the section turned
    over (the top face in tension)"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    rows = []
    for name, s in sections.items():
        A0, z0, I0 = s.props("transformed")
        so = s.flipped() if name in hogging else s
        bot, top = so.edge_masks()
        r = [name + (("（负弯矩）" if zh else " (hogging)") if name in hogging else ""), s.h, s.web(),
             Cell(s.area, digits=0), s.zg, Markup(m(_sci_math(s.I)))]
        if transformed:
            r += [Cell(A0, digits=0), z0, Markup(m(_sci_math(I0)))]
        r += [Cell(so.bars[bot, 0].sum(), digits=1), Cell(so.bars[top, 0].sum(), digits=1), so.h0()]
        rows.append(r)
    head = ["截面" if zh else "Section", "h", "b", "A", "y_{c}", "I"]
    units = ["", "mm", "mm", "mm²", "mm", "mm⁴"]
    if transformed:
        head += ["A_{0}", "y_{0}", "I_{0}"]
        units += ["mm²", "mm", "mm⁴"]
    head += ["A_{s}", "A'_{s}", "h_{0}"]
    units += ["mm²", "mm²", "mm"]
    note = ("y_{c}、y_{0}：毛截面、换算截面重心至截面顶面的距离；A_{s}、A'_{s}：受拉、受压边排纵筋面积；h_{0}：截面有效高度。"
            "按正弯矩取值，标注“负弯矩”的截面按上缘受拉取值。" if zh else
            "y_{c}, y_{0}: depth of the gross and transformed centroid; A_{s}, A'_{s}: the tension and compression"
            " edge "
            "rows; h_{0}: effective depth; under a sagging moment, or the top face in tension where marked hogging.")
    node.table(rows, header=head, units=units, caption=caption or ("截面几何特性及配筋" if zh else "Section properties"),
               id=id, digits=0, align="l" + "r" * (len(head) - 1), notes=[note])
    for name, s in sections.items():
        bt = _bars_text(s, lang)
        if bt:
            node.paragraph(f"{name}：" if zh else f"{name}: ",
                           "；".join(f"{t}（距顶面 {z:.0f} mm）" if zh else f"{t} at {z:.0f} mm" for t, z in bt),
                           "。" if zh else ".")
    return node


def load_cases(node: Node, cases, *, model=None, lang: str | None = None, caption: str | None = None,
               id: str | None = "tab:loads") -> Node:
    """a table of the load cases (femotion.frame.loads.LoadCase): case, kind of load, value, where. model: for the
    self-weight per length (Section.mass × g) and node positions"""
    from ...frame.loads import (BarStrain, Imposed, LineLoad, NodalLoad, PointLoad, SelfWeight, Settlement,
                                TendonLoad)
    lang = _lang(node, lang)
    zh = lang == "zh"
    rows = []
    for case in cases:
        first = True
        for it in case.items:
            nm = case.name if first else ""
            first = False
            if isinstance(it, SelfWeight):
                q = None
                if model is not None:
                    ms = {round(sec.mass, 9) for (_, _, sec, _, _) in model.beams}
                    if len(ms) == 1:
                        q = ms.pop() * it.g * it.factor
                kind = "结构自重" if zh else "self-weight"
                val = f"$q = m g$ = {q:.2f} kN/m" if q is not None else "$m g$"
                where = ("全部构件" if zh else "all members") if it.beams is None else \
                    (f"{len(it.beams)} 个单元" if zh else f"{len(it.beams)} elements")
            elif isinstance(it, LineLoad):
                w0, w1 = float(np.linalg.norm(it.w0)), float(np.linalg.norm(it.w1))
                kind = "分布荷载" if zh else "distributed load"
                val = f"$q$ = {w0:.2f} kN/m" if math.isclose(w0, w1) else f"$q$ = {w0:.2f} → {w1:.2f} kN/m"
                if it.start == 0 and it.end is None:
                    where = "全长" if zh else "whole length"
                else:
                    end = ("末端" if zh else "end") if it.end is None else f"{it.end:g} m"
                    where = f"s = {it.start:g} m ～ {end}"
            elif isinstance(it, PointLoad):
                kind = "集中荷载" if zh else "point load"
                val = f"$P$ = {float(np.linalg.norm(it.P)):.1f} kN"
                where = f"s = {it.s:g} m"
            elif isinstance(it, NodalLoad):
                kind = "节点荷载" if zh else "nodal load"
                val = f"F = {num(float(np.linalg.norm(it.F)))} kN, M = {num(float(np.linalg.norm(it.M)))} kN·m"
                where = f"{'节点' if zh else 'node'} {it.node}"
            elif isinstance(it, Imposed):
                kind = "强迫变形（温度、收缩）" if zh else "imposed strain"
                k = np.atleast_1d(it.curvature)
                e = np.atleast_1d(it.strain)
                val = f"κ = {num(float(k.max()))} 1/m, ε = {num(float(e.max()))}"
                where = f"{len(it.beams)} {'个单元' if zh else 'elements'}"
            elif isinstance(it, TendonLoad):
                kind = "预加力" if zh else "prestress"
                f_ = np.atleast_1d(it.tendon.force)
                val = f"N = {num(float(f_.max()))} kN"
                where = f"{len(it.tendon.beams)} {'个单元' if zh else 'elements'}"
            elif isinstance(it, Settlement):
                kind = "支座沉降" if zh else "settlement"
                val = f"Δ = {num(float(it.d[2]) * 1e3)} mm"
                where = f"{'节点' if zh else 'node'} {it.node}"
            elif isinstance(it, BarStrain):
                kind = "初应变" if zh else "initial strain"
                val = f"ε0 = {num(float(np.atleast_1d(it.eps0).max()))}"
                where = f"{len(it.bars)} {'根杆' if zh else 'bars'}"
            else:
                kind, val, where = type(it).__name__, "", ""
            if it.label and not re.search(r"\d", it.label):  # a label repeating the value is left out
                val = f"{val}（{it.label}）" if zh else f"{val} ({it.label})"
            rows.append([nm, kind, val, where])
    node.table(rows, header=["工况" if zh else "Case", "类型" if zh else "Kind", "荷载" if zh else "Load",
                             "作用位置" if zh else "Where"], caption=caption or ("荷载工况" if zh else "Load cases"),
               id=id, align="llll")
    return node


def live_load(node: Node, live, *, mu=None, lang: str | None = None, caption: str | None = None,
              id: str | None = "tab:live", notes=None) -> Node:
    """the lane load of D60 4.3.1 behind a jtg.LiveLoad: grade, qk, Pk (×1.2 for shear), lanes and reduction factors,
    distribution and impact factors. mu: (label, μ) rows when the impact factor differs by effect. One girder of a
    deck (lanes=1, transverse=1 and a lateral distribution factor) is said so: the lanes and the transverse reduction
    are in the distribution factor"""
    from .actions import lane_load, transverse_reduction
    lang = _lang(node, lang)
    zh = lang == "zh"
    q, P = lane_load(live.L0, live.grade)
    grade = {"I": "Ⅰ", "II": "Ⅱ"}.get(str(live.grade), str(live.grade))
    in_m = live.lanes == 1 and live.transverse == 1.0 and live.distribution != 1.0
    tr = live.transverse if live.transverse is not None else transverse_reduction(live.lanes)
    rows = [["公路等级" if zh else "class", "", f"公路—{grade}级" if zh else f"Class {live.grade}", ""],
            ["计算跨径" if zh else "computed span", "$L_{0}$", live.L0, "m"],
            ["均布荷载标准值" if zh else "uniform load", "$q_{k}$", q, "kN/m"],
            ["集中荷载标准值" if zh else "concentrated load", "$P_{k}$", Cell(P, digits=1), "kN"],
            ["集中荷载（计算剪力）" if zh else "concentrated load for shear", "$1.2P_{k}$", Cell(1.2 * P, digits=1), "kN"]]
    if in_m:
        rows.append(["设计车道数、横向车道布载系数" if zh else "design lanes, transverse reduction", "",
                     "计入 m" if zh else "in m", ""])
    else:
        rows += [["设计车道数" if zh else "design lanes", "", live.lanes, ""],
                 ["横向车道布载系数" if zh else "transverse reduction", "", tr, ""]]
    rows.append(["荷载横向分布系数" if zh else "lateral distribution factor", "$m$", live.distribution, ""])
    for label, mu_v in (mu or [("冲击系数" if zh else "impact factor", live.mu)]):
        rows.append([label, "$μ$", Cell(mu_v, digits=3), ""])
    nts = list(notes or [])
    if in_m:
        nts.insert(0, "按单片梁计算：荷载横向分布系数 m 已计入车道数和横向车道布载系数（表 4.3.1-5），故车道荷载不再乘这两项。"
                   if zh else "One girder: the lateral distribution factor m includes the lanes and the transverse "
                   "reduction (table 4.3.1-5); the lane load is not multiplied by them again.")
    node.table(rows, header=["项目" if zh else "Item", "符号" if zh else "Symbol", "数值" if zh else "Value",
                             "单位" if zh else "Unit"], caption=caption or ("汽车荷载" if zh else "Traffic load"), id=id,
               align="llrl", notes=nts)
    return node


def combinations(node: Node, *, gamma0: float = 1.0, crowd: bool = False, characteristic: bool = False,
                 lang: str | None = None, caption: str | None = None, id: str | None = "tab:combinations") -> Node:
    """the combinations used and their factors (D60 4.1.5, 4.1.6, 4.1.8; values of femotion.codes.jtg.actions)"""
    from .actions import PSI
    lang = _lang(node, lang)
    zh = lang == "zh"
    pf, pq = PSI["vehicle"]
    cf, cq = PSI["crowd"]
    g0 = f"$γ_{{0}}$ = {gamma0:.1f}"
    rows = [["基本组合" if zh else "basic", "4.1.5-1",
             m("S_{ud} = γ_{0}S(Σγ_{Gi}G_{ik}, γ_{Q1}Q_{1k}, ψ_{c}Σγ_{Qj}Q_{jk})"),
             f"{g0}；$γ_{{G}}$ = 1.2（有利时 1.0）；$γ_{{Q1}}$ = 1.4（车道荷载，含冲击）"
             + ("；人群 $γ_{Qj}$ = 1.4，$ψ_{c}$ = 0.75" if crowd else "")
             if zh else f"{g0}; $γ_{{G}}$ = 1.2 (1.0 favourable); $γ_{{Q1}}$ = 1.4 (lane load with impact)"
             + ("; crowd $γ_{Qj}$ = 1.4, $ψ_{c}$ = 0.75" if crowd else "")],
            ["频遇组合" if zh else "frequent", "4.1.6-1", m("S_{fd} = S(ΣG_{ik}, ψ_{f1}Q_{1k}, Σψ_{qj}Q_{jk})"),
             f"汽车荷载 $ψ_{{f1}}$ = {pf:g}（不计冲击）" + (f"；人群 $ψ_{{q}}$ = {cq:g}" if crowd else "") if zh else
             f"traffic $ψ_{{f1}}$ = {pf:g} (no impact)" + (f"; crowd $ψ_{{q}}$ = {cq:g}" if crowd else "")],
            ["准永久组合" if zh else "quasi-permanent", "4.1.6-2", m("S_{qd} = S(ΣG_{ik}, Σψ_{qj}Q_{jk})"),
             f"汽车荷载 $ψ_{{q}}$ = {pq:g}（不计冲击）" + (f"；人群 $ψ_{{q}}$ = {cq:g}" if crowd else "") if zh else
             f"traffic $ψ_{{q}}$ = {pq:g} (no impact)" + (f"; crowd $ψ_{{q}}$ = {cq:g}" if crowd else "")]]
    if characteristic:
        rows.append(["标准值组合" if zh else "characteristic", "4.1.8", m("S = S(ΣG_{ik}, Q_{1k}, ΣQ_{jk})"),
                     "汽车荷载计入冲击" if zh else "traffic with impact"])
    note = ("作用效应按线性叠加。本计算书各表、各图中的基本组合效应 S(…) 均未乘 $γ_{0}$，验算时以 $γ_{0}S_{d}$ 与抗力比较。" if zh else
            "Effects are added linearly. The basic-combination effects in the tables and charts are without $γ_{0}$; "
            "the checks compare $γ_{0}S_{d}$ with the resistance.")
    node.table(rows, header=["组合" if zh else "Combination", "条文" if zh else "Clause",
                             "表达式" if zh else "Expression", "系数" if zh else "Factors"],
               caption=caption or ("作用组合（JTG D60—2015）" if zh else "Combinations (JTG D60-2015)"), id=id,
               align="llll", widths=["6em", "4.6em", "44%", None], notes=[note])
    return node


# ── shear along the girder ──────────────────────────────────────────────────
def shear_search_table(node: Node, search, *, lang: str | None = None, caption: str | None = None,
                       id: str | None = "tab:inclined", all_sections: bool = False) -> Node:
    """the inclined sections of jtg.shear_search: the 5.2.8 positions (all with all_sections) and the governing one,
    with x_c, x_t, C, m, γ0Vd, Vu and the ratio; the section limits at the end supports"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    gov = search.governing
    secs = [s for s in search.sections if all_sections or s.source != "grid" or s is gov]
    rows, cls = [], []
    ok_s = ("满足", "不满足") if zh else ("OK", "NOT OK")
    for k, s in enumerate(secs, 1):
        src = _source(s.source, lang) + (("（控制）" if zh else " (governing)") if s is gov else "")
        rows.append([k, src, s.x_c, s.x_t, s.C, Cell(s.m, digits=2), Cell(abs(s.check.demand), digits=1),
                     Cell(s.check.resistance, digits=1), Cell(s.ratio, digits=3),
                     Cell(ok_s[0] if s.ok else ok_s[1], "ok" if s.ok else "fail")])
        cls.append("total" if s is gov else ("fail" if not s.ok else ""))
    for c in search.limits:
        rows.append(["", check_name(c, lang), "", "", "", "", Cell(abs(c.demand), digits=1),
                     Cell(c.resistance, digits=1), Cell(c.ratio, digits=3),
                     Cell(ok_s[0] if c.ok else ok_s[1], "ok" if c.ok else "fail")])
        cls.append("sub" if c.ok else "fail")
    node.table(rows, header=["序号" if zh else "No.", "验算位置（5.2.8）" if zh else "Position (5.2.8)", "x_{c}", "x_{t}",
                             "C", "m", "γ_{0}V_{d}", "V_{u}", "利用率" if zh else "ratio", "结论" if zh else "verdict"],
               units=["", "", "m", "m", "m", "", "kN", "kN", "", ""], caption=caption or (
                   "斜截面抗剪验算（5.2.8～5.2.11）" if zh else "Inclined sections (5.2.8–5.2.11)"), id=id,
               digits=[None, None, 3, 3, 3, 2, 1, 1, 3, None], row_cls=cls, align="clrrrrrrrc")
    return node


def _source(src: str, lang: str) -> str:
    if lang != "zh":
        return src
    s = src
    s = re.sub(r"h/2 from end support x = (\S+)", r"端支点（x = \1 m）h/2 处", s)
    s = re.sub(r"diaphragm edge, support x = (\S+)", r"中支点（x = \1 m）横隔梁边缘", s)
    s = re.sub(r"stirrup change x = (\S+)", r"箍筋变化处 x = \1 m", s)
    s = re.sub(r"web change x = (\S+)", r"腹板变化处 x = \1 m", s)
    s = re.sub(r"bend x = (\S+)", r"弯起点 x = \1 m", s)
    return s.replace("grid", "搜索网格")


def stirrup_design_block(node: Node, design, Vd: float | None = None, *, where: str = "", lang: str | None = None,
                         gamma0: float | None = None) -> Node:
    """the stirrup design of jtg.stirrup_design (5.2.13-1) with the values substituted, and the check back-substituted
    (listed in the summary apart, not counted as a check). Vd, gamma0: default those of the design"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    d = design.details
    Vd = d.get("Vd", Vd) if Vd is None else Vd
    g0 = d.get("gamma0", 1.0) if gamma0 is None else gamma0
    xi = d.get("xi")
    sec = node.section("箍筋设计（5.2.13）" if zh else "Stirrup design (5.2.13)", id="sec:stirrups")
    if xi is not None:
        sec.paragraph((f"设计剪力取{where}的 " if zh else f"Design shear at {where}: "),
                      m(f"V'_{{d}} = {fixed(Vd, 1)}\\,\\u{{kN}}"), "，" if zh else ", ",
                      f"混凝土和箍筋共同承担 ξ = {_nv(xi)}（不少于 0.6），其余由弯起钢筋承担（5.2.13 第 1 款）。" if zh else
                      f"the concrete and stirrups take ξ = {_nv(xi)} (at least 0.6), the bent-up bars the rest.")
    Asv = design.legs * math.pi * design.dia ** 2 / 4
    if design.spacing_required is not None and xi is not None:
        fcuk = design.check.section.concrete.fcuk
        K = 0.2e-6 * d["alpha1"] ** 2 * d["alpha3"] ** 2 * (2 + 0.6 * d["P"]) * math.sqrt(fcuk) * Asv * d["fsv"] * \
            d["b"] * d["h0"] ** 2
        dem = xi * g0 * Vd
        if not (_close(dem, d["demand"]) and _close(K / dem ** 2, design.spacing_required)):
            raise ValueError("stirrup design: the formula does not reproduce the design")
        sec.formula([("s_{v}", "\\frac{0.2×10^{-6}α_{1}^{2}α_{3}^{2}(2 + 0.6P)\\sqrt{f_{cu,k}}A_{sv}f_{sv}bh_{0}^{2}}"
                               "{(ξγ_{0}V'_{d})^{2}}"),
                     ("", f"\\frac{{0.2×10^{{-6}} × {_nv(d['alpha1'])}^{{2}} × {_nv(d['alpha3'])}^{{2}} × (2 + 0.6 × "
                          f"{_nv(d['P'])}) × \\sqrt{{{_nv(fcuk)}}} × {_nv(Asv)} × {_nv(d['fsv'])} × {_nv(d['b'])} × "
                          f"{_nv(d['h0'])}^{{2}}}}{{({_nv(xi)} × {_nv(g0)} × {_nv(Vd)})^{{2}}}}"),
                     ("", f"{fixed(design.spacing_required, 1)}\\,\\u{{mm}}")], tag="5.2.13-1")
    adopted = m(f"{design.legs}\\t{{ 肢 }}Φ{_nv(design.dia)}@{_nv(design.spacing)}" if zh else
                f"{design.legs}\\t{{ legs }}Φ{_nv(design.dia)} @ {_nv(design.spacing)}")
    sec.paragraph(("取 " if zh else "Adopted: "), adopted,
                  ("（按 9.3.12 第 2 款取间距不大于 h/2 和 400 mm、配箍率不小于最小值；支座中心一倍梁高范围内的加密要求另见下面的提示）。"
                   if zh else " (spacing within h/2 and 400 mm and the least ratio of 9.3.12 item 2; the closer"
                              " spacing "
                   "near the support is in the warnings below)."))
    write_check(sec, design.check, where=where, figure=False, lang=lang, counted=False)
    return sec


def bent_bar_table(node: Node, design, *, lang: str | None = None, caption: str | None = None,
                   id: str | None = "tab:bent") -> Node:
    """the rows of bent-up bars of jtg.bent_bar_design (5.2.13-2)"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    node.paragraph(("按式（5.2.13-2）" if zh else "By 5.2.13-2 "),
                   m("A_{sb} = \\frac{γ_{0}V_{sb}}{0.75×10^{-3}f_{sd}\\op{sin}θ_{s}}"),
                   (f"，$θ_{{s}}$ = {design.theta:g}°，分配线 $ξV'_{{d}}$ = {design.V_line:.1f} kN。" if zh else
                    f", $θ_{{s}}$ = {design.theta:g}°, the line $ξV'_{{d}}$ = {design.V_line:.1f} kN."))
    rows = [[k, r.x_top, r.x_bottom, r.x_read, Cell(r.Vsb, digits=1), Cell(r.Asb, digits=0)]
            for k, r in enumerate(design.rows, 1)]
    node.table(rows, header=["排" if zh else "Row", "上弯点" if zh else "top bend", "下弯点" if zh else "bottom bend",
                             "取值位置" if zh else "read at", "V_{sb}", "A_{sb}"],
               units=["", "m", "m", "m", "kN", "mm²"], digits=[None, 3, 3, 3, 1, 0],
               caption=caption or ("弯起钢筋设计（5.2.13）" if zh else "Bent-up bars (5.2.13)"), id=id, align="crrrrr")
    return node
