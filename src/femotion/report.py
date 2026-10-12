"""Calculation reports (计算书) as one self-contained HTML file: open it in a browser to read it, print it to PDF ("Save as
PDF", A4) to hand it in. Nothing here knows about design codes; femotion.codes.jtg.report writes checks into it.

    from femotion.report import Report
    rep = Report("两跨连续 T 梁计算书", project="某大桥", member="主梁", author="张三", checker="李四", date="2026-10-12")
    ch = rep.chapter("设计资料")
    s = ch.section("材料")
    s.paragraph("混凝土 C40，$f_{cd} = 18.4 \\u{MPa}$；钢筋 HRB400（见", rep.ref("tab:mat"), "）。")
    s.table([["C40", 26.8, 18.4]], header=["等级", "f_{ck}", "f_{cd}"], units=["", "MPa", "MPa"],
            caption="混凝土", id="tab:mat")
    s.figure(fig, caption="截面", id="fig:sec")                  # a femotion.viz Figure / Stack, or SVG text
    s.formula([("x", r"\\frac{f_{sd}A_{s}}{f_{cd}b'_{f}}"), ("", r"\\v{x}{54.2}\\,\\u{mm}")], tag="5.2.2-2")
    s.box("ok", "满足要求。")
    rep.save("report.html")          # or rep.html(); rep.pdf("report.pdf") prints it with Chrome, TOC page numbers in

The document is a tree: Report → chapters (and appendices) → sections → subsections (any depth, numbered 1, 1.1,
1.1.1 …). Each node holds blocks in order: paragraphs, tables, figures, formulas, boxes, lists, raw HTML. Figures,
tables and auto-numbered formulas are numbered per chapter (图 3-2, 表 4-1, (5-3)); appendices are lettered (附录 A,
图 A-1). Cross-references (Report.ref) are resolved when the HTML is written and may point forwards; an unknown target
raises KeyError, so a report never carries a dangling reference.

Text: every string goes through rich(): HTML-escaped, x_{sub} and x^{sup} as sub- and superscripts, $…$ as inline math
(math() below); Markup passes as it is (raw HTML). Math (math()) is a small TeX-like language: single Latin letters and
lowercase Greek letters are variables (italic), longer letter runs are upright (sin, max, kN), subscripts upright;
\\frac{a}{b}, \\sqrt{a}, \\t{text} (upright text, e.g. Chinese), \\u{unit} (a unit, upright), \\v{key}{value} (a value
substituted into a formula, tagged data-sym="key" so scripts and tests can read the numbers shown), \\, (thin space),
' (prime), - (minus sign), <= and >= (≤, ≥).

Output: one HTML file, styles inline, no external resources, fonts from femotion.viz.FONT. On screen: a contents
sidebar (a collapsible list on narrow screens), the width follows the window. In print (A4): the cover on its own page,
the contents with page numbers when known (Report.pdf), every chapter on a new page, figures, formulas, boxes and short
tables never split across pages (long tables break between rows and repeat their header), a running head (project,
member) and the page number "第 n 页 共 N 页" in the page margins (CSS @page margin boxes: Chrome 131 and later; other
browsers print without them). Report.pdf prints twice with headless Chrome: the first pass finds the page of every
heading from the PDF's link destinations, the second prints the contents with those page numbers; the PDF also gets an
outline (bookmarks) from the headings.
"""
from __future__ import annotations

import html as _html
import math as _pymath
import os
import re
import shutil
import subprocess
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from .viz import COLORS, FONT

__all__ = ["Report", "Node", "Markup", "Cell", "escape", "rich", "math", "m", "fixed", "num", "sci"]

# ── markup ──────────────────────────────────────────────────────────────────


class Markup(str):
    """HTML that is written as it is (not escaped)"""

    def __add__(self, other):
        return Markup(str(self) + str(rich(other)))

    def __radd__(self, other):
        return Markup(str(rich(other)) + str(self))


def escape(s) -> Markup:
    """HTML-escaped text"""
    return s if isinstance(s, Markup) else Markup(_html.escape(str(s), quote=True))


_TOKEN = re.compile(r"⟦(ref|num|page):([^⟧]+)⟧")
_SUBSUP = re.compile(r"([_^])\{([^{}]*)\}")


def rich(s) -> Markup:
    """a text for the document: Markup as it is; numbers by femotion.viz.number; strings escaped with x_{sub}, x^{sup}
    and $…$ inline math (\\$ for a dollar sign)"""
    if isinstance(s, Markup):
        return s
    if s is None:
        return Markup("")
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return Markup(num(float(s)))
    s = str(s)
    out, pos = [], 0
    for mt in re.finditer(r"(?<!\\)\$(.+?)(?<!\\)\$", s):
        out.append(_plain_rich(s[pos:mt.start()]))
        out.append(m(mt.group(1)))
        pos = mt.end()
    out.append(_plain_rich(s[pos:]))
    return Markup("".join(out))


def _plain_rich(s: str) -> str:
    s = s.replace("\\$", "$")
    parts, pos = [], 0
    for mt in _SUBSUP.finditer(s):
        parts.append(_html.escape(s[pos:mt.start()], quote=False))
        tag = "sub" if mt.group(1) == "_" else "sup"
        parts.append(f"<{tag}>{_html.escape(mt.group(2), quote=False)}</{tag}>")
        pos = mt.end()
    parts.append(_html.escape(s[pos:], quote=False))
    return "".join(parts)


def _join(parts) -> Markup:
    """parts (strings, numbers, Markup) as one rich text"""
    return Markup("".join(str(rich(p)) for p in parts))


# ── math ────────────────────────────────────────────────────────────────────
_ARGS = {"frac": 2, "sqrt": 1, "t": 1, "u": 1, "v": 2, "b": 1, "op": 1}
_REL = {"<=": "≤", ">=": "≥"}
# letter runs written upright in math (functions); any other run is a product of single-letter variables. Units go in
# \u{…}, words in \t{…} or \op{…}
_WORDS = {"sin", "cos", "tan", "cot", "arcsin", "arccos", "arctan", "max", "min", "ln", "log", "exp", "lim", "sgn"}


def _group(s: str, i: int) -> tuple[str, int]:
    """the text of the {…} group starting at s[i] == "{", and the index after it"""
    if i >= len(s) or s[i] != "{":
        raise ValueError(f"math: '{{' expected at {i} in {s!r}")
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\" and j + 1 < len(s):
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"math: unbalanced braces in {s!r}")


def _is_greek_lower(c: str) -> bool:
    return "\u03b1" <= c <= "\u03c9" or c in "\u03d1\u03d5\u03f5"


def _math(s: str, sub: bool = False) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            j = i + 1
            if j < n and not s[j].isalpha():
                ch = s[j]
                out.append({",": "\u2009", ";": " ", " ": " "}.get(ch, _html.escape(ch)))
                i = j + 1
                continue
            while j < n and s[j].isalpha():
                j += 1
            cmd = s[i + 1:j]
            if cmd not in _ARGS:
                raise ValueError(f"math: unknown command \\{cmd} in {s!r}")
            args = []
            for _ in range(_ARGS[cmd]):
                a, j = _group(s, j)
                args.append(a)
            i = j
            if cmd == "frac":
                out.append(f'<span class="fr"><span class="nu">{_math(args[0], sub)}</span>'
                           f'<span class="de">{_math(args[1], sub)}</span></span>')
            elif cmd == "sqrt":
                out.append(f'<span class="sq"><span class="rad">√</span><span class="sqc">{_math(args[0], sub)}'
                           f'</span></span>')
            elif cmd == "t":
                out.append(f'<span class="mt">{_plain_rich(args[0])}</span>')
            elif cmd == "u":
                out.append(f'<span class="mu">{_html.escape(args[0])}</span>')
            elif cmd == "v":
                out.append(f'<span class="v" data-sym="{_html.escape(args[0])}">{_math(args[1], True)}</span>')
            elif cmd == "b":
                out.append(f"<b>{_math(args[0], sub)}</b>")
            elif cmd == "op":
                out.append(f'<span class="mt">{_html.escape(args[0])}</span>')
            continue
        if c in "_^" and i + 1 < n and s[i + 1] == "{":
            a, i = _group(s, i + 1)
            tag = "sub" if c == "_" else "sup"
            out.append(f"<{tag}>{_math(a, sub=(c == '_') or sub)}</{tag}>")
            continue
        if s.startswith(("<=", ">="), i):
            out.append(_REL[s[i:i + 2]])
            i += 2
            continue
        if c.isascii() and c.isalpha():
            j = i
            while j < n and s[j].isascii() and s[j].isalpha():
                j += 1
            run = s[i:j]
            if sub or run in _WORDS:
                out.append(run)
            else:  # a product of variables: bh, bx
                out.append("".join(f"<i>{ch}</i>" for ch in run))
            i = j
            continue
        if _is_greek_lower(c):
            out.append(c if sub else f"<i>{c}</i>")
            i += 1
            continue
        if c == "'":
            out.append("′")
        elif c == "-":
            out.append("−")
        else:
            out.append(_html.escape(c, quote=False))
        i += 1
    return "".join(out)


def math(s: str) -> Markup:
    """the HTML of a formula in the math language of the module notes (no wrapper)"""
    return Markup(_math(str(s)))


def m(s: str) -> Markup:
    """inline math: a formula in a line of text"""
    return Markup(f'<span class="m">{_math(str(s))}</span>')


def fixed(v: float, nd: int = 1) -> str:
    """a number with nd decimals rounded half up (790.5 → 791, as by hand: the decimal digits shown, not the binary
    value), a real minus sign, −0 as 0 (the form of values substituted into formulas)"""
    v = float(v)
    if not _pymath.isfinite(v):
        return "∞" if v > 0 else "−∞"
    q = Decimal(repr(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = f"{q:f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s[1:]
    return s.replace("-", "−")


def num(v: float) -> str:
    """a number by its magnitude as femotion.viz.number (none from 100, one decimal from 10, two from 1, three
    significant figures below 1; trailing zeros dropped), rounded half up"""
    v = float(v)
    if not _pymath.isfinite(v):
        return "∞" if v > 0 else "−∞"
    a = abs(v)
    if a >= 100 or a == 0:
        s = fixed(v, 0)
    elif a >= 10:
        s = fixed(v, 1)
    elif a >= 1:
        s = fixed(v, 2)
    else:
        s = fixed(v, max(2 - int(_pymath.floor(_pymath.log10(a))), 0))
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("−0", "") else s


def sci(v: float, sig: int = 4) -> str:
    """a number in the math language: plain below 1e6, else a × 10^{n} with sig significant figures"""
    v = float(v)
    if v == 0 or 1e-3 <= abs(v) < 1e6:
        return num(v) if abs(v) < 1e6 else fixed(v, 0)
    e = int(f"{v:e}".split("e")[1])
    a = v / 10 ** e
    return f"{fixed(a, sig - 1)}×10^{{{fixed(e, 0)}}}"


# ── words ───────────────────────────────────────────────────────────────────
_W = {
    "zh": {"fig": "图", "tab": "表", "eq": "式", "sec": "节", "ch": "第 {n} 章", "app": "附录", "toc": "目录",
           "page": "第 {n} 页", "pages": "第 \" counter(page) \" 页 共 \" counter(pages) \" 页",
           "where": "式中：", "note": "注：", "project": "项目名称", "member": "构件", "number": "计算书编号",
           "version": "版本", "date": "日期", "role": "职责", "name": "姓名", "sign": "签字", "author": "编制",
           "checker": "校核", "approver": "审核", "org": "", "unit": "单位", "contents_short": "目录",
           "calc": "计算书", "sep": "  "},
    "en": {"fig": "Figure", "tab": "Table", "eq": "Eq.", "sec": "Section", "ch": "Chapter {n}", "app": "Appendix",
           "toc": "Contents", "page": "page {n}", "pages": "Page \" counter(page) \" of \" counter(pages) \"",
           "where": "where", "note": "Note: ", "project": "Project", "member": "Member", "number": "Report no.",
           "version": "Version", "date": "Date", "role": "Role", "name": "Name", "sign": "Signature",
           "author": "Prepared by", "checker": "Checked by", "approver": "Approved by", "org": "", "unit": "unit",
           "contents_short": "Contents", "calc": "Calculation report", "sep": "  "},
}


# ── blocks ──────────────────────────────────────────────────────────────────
class Cell:
    """a table cell with options: value (as any cell), cls (CSS classes, e.g. "fail", "ok", "warn", "num", "c"),
    colspan, rowspan, digits (decimals of a number)"""

    def __init__(self, value, cls: str = "", *, colspan: int = 1, rowspan: int = 1, digits: int | None = None):
        self.value, self.cls, self.colspan, self.rowspan, self.digits = value, cls, colspan, rowspan, digits


def _is_num(v) -> bool:
    if isinstance(v, Cell):
        v = v.value
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        import numpy as np
        if isinstance(v, np.number):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(v, str) and bool(re.fullmatch(r"\s*[−+-]?\d[\d,]*(\.\d+)?(e[−+-]?\d+)?\s*%?\s*", v))


def _cell_text(v, digits) -> Markup:
    if isinstance(v, Cell):
        return _cell_text(v.value, v.digits if v.digits is not None else digits)
    if v is None:
        return Markup("")
    if isinstance(v, bool):
        return escape(str(v))
    try:
        import numpy as np
        if isinstance(v, np.number):
            v = v.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(v, (int, float)):
        if isinstance(v, int):
            return Markup(f"{v}".replace("-", "−"))
        return Markup(fixed(v, digits) if digits is not None else num(v))
    return rich(v)


class _Block:
    kind = "block"
    id = None


class _Para(_Block):
    kind = "p"

    def __init__(self, parts, cls):
        self.parts, self.cls = parts, cls

    def html(self, ctx):
        c = f' class="{self.cls}"' if self.cls else ""
        return f"<p{c}>{_join(self.parts)}</p>"


class _Raw(_Block):
    kind = "raw"

    def __init__(self, markup):
        self.markup = markup

    def html(self, ctx):
        return str(self.markup)


class _List(_Block):
    kind = "list"

    def __init__(self, items, ordered, cls):
        self.items, self.ordered, self.cls = items, ordered, cls

    def html(self, ctx):
        t = "ol" if self.ordered else "ul"
        c = f' class="{self.cls}"' if self.cls else ""
        items = "".join(f"<li>{_join(it if isinstance(it, (list, tuple)) else [it])}</li>" for it in self.items)
        return f"<{t}{c}>{items}</{t}>"


class _Box(_Block):
    kind = "box"
    KINDS = ("ok", "fail", "warn", "info", "note")

    def __init__(self, kind, parts, title, items):
        if kind not in self.KINDS:
            raise ValueError(f"box kind {kind!r}: one of {', '.join(self.KINDS)}")
        self.box_kind, self.parts, self.title, self.items = kind, parts, title, items

    def html(self, ctx):
        out = [f'<div class="box box-{self.box_kind}">']
        if self.title:
            out.append(f'<div class="box-t">{rich(self.title)}</div>')
        if self.parts:
            out.append(f"<div>{_join(self.parts)}</div>")
        if self.items:
            out.append("<ul>" + "".join(f"<li>{_join(it if isinstance(it, (list, tuple)) else [it])}</li>"
                                        for it in self.items) + "</ul>")
        out.append("</div>")
        return "".join(out)


class _Table(_Block):
    kind = "tab"

    def __init__(self, rows, header, units, caption, id, align, digits, widths, notes, row_cls, keep, compact, cls):
        self.rows = [list(r) for r in rows]
        self.header = header
        self.units = units
        self.caption, self.id = caption, id
        ncol = max([len(r) for r in self.rows] + [len(header or ()), len(units or ())] + [0])
        self.ncol = ncol
        if isinstance(digits, int) or digits is None:
            digits = [digits] * ncol
        self.digits = list(digits) + [None] * (ncol - len(digits))
        if align is None:
            align = ""
            for j in range(ncol):
                vals = [r[j] for r in self.rows if j < len(r) and r[j] not in (None, "", "—")]
                align += "r" if vals and all(_is_num(v) for v in vals) else "l"
        self.align = align.ljust(ncol, "l")
        self.widths, self.notes, self.row_cls = widths, list(notes or []), row_cls
        self.keep = keep if keep is not None else len(self.rows) <= 24
        self.compact, self.cls = compact, cls

    def html(self, ctx):
        num = ctx.number("tab", self.id) if self.caption is not None else None
        cls = ["tbl"] + (["keep"] if self.keep else []) + (["compact"] if self.compact else []) + \
            ([self.cls] if self.cls else [])
        out = [f'<div class="{" ".join(cls)}"' + (f' id="{num[1]}"' if num else "") + ">"]
        if num:
            out.append(f'<div class="cap cap-t"><span class="cap-n">{num[0]}</span>{rich(self.caption)}</div>')
        out.append('<div class="tblw"><table>')
        if self.widths:
            out.append("<colgroup>" + "".join(f'<col style="width:{w}">' if w else "<col>" for w in self.widths)
                       + "</colgroup>")
        al = {"l": "", "r": ' class="num"', "c": ' class="c"'}
        if self.header or self.units:
            out.append("<thead>")
            if self.header:
                out.append("<tr>" + "".join(self._head(h, j) for j, h in enumerate(self.header)) + "</tr>")
            if self.units:
                out.append('<tr class="units">' + "".join(
                    f"<th{al[self.align[j]] if j < self.ncol else ''}>{rich(u) if u else ''}</th>"
                    for j, u in enumerate(self.units)) + "</tr>")
            out.append("</thead>")
        out.append("<tbody>")
        for i, r in enumerate(self.rows):
            rc = self.row_cls(i, r) if callable(self.row_cls) else (self.row_cls[i] if self.row_cls else "")
            out.append(f'<tr class="{rc}">' if rc else "<tr>")
            for j, v in enumerate(r):
                c = [] if self.align[j] == "l" else (["num"] if self.align[j] == "r" else ["c"])
                attrs = ""
                if isinstance(v, Cell):
                    c += v.cls.split()
                    if v.colspan > 1:
                        attrs += f' colspan="{v.colspan}"'
                    if v.rowspan > 1:
                        attrs += f' rowspan="{v.rowspan}"'
                cc = f' class="{" ".join(c)}"' if c else ""
                out.append(f"<td{cc}{attrs}>{_cell_text(v, self.digits[j])}</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")
        if self.notes:
            out.append('<div class="tnote">' + "".join(f"<div>{rich(n)}</div>" for n in self.notes) + "</div>")
        out.append("</div>")
        return "".join(out)

    def _head(self, h, j):
        if isinstance(h, Cell):
            attrs = (f' colspan="{h.colspan}"' if h.colspan > 1 else "") + \
                (f' rowspan="{h.rowspan}"' if h.rowspan > 1 else "")
            return f"<th{attrs}>{rich(h.value)}</th>"
        al = {"l": "", "r": ' class="num"', "c": ' class="c"'}[self.align[j]] if j < self.ncol else ""
        return f"<th{al}>{rich(h)}</th>"


class _Figure(_Block):
    kind = "fig"

    def __init__(self, figs, caption, id, subcaptions, width):
        self.figs = figs if isinstance(figs, (list, tuple)) else [figs]
        self.caption, self.id, self.subcaptions, self.width = caption, id, subcaptions, width

    @staticmethod
    def _svg(f):
        if hasattr(f, "svg"):
            return f.svg()
        return str(f)

    def html(self, ctx):
        num = ctx.number("fig", self.id) if self.caption is not None else None
        out = ['<figure class="fig"' + (f' id="{num[1]}"' if num else "") + ">"]
        style = f' style="max-width:{self.width}"' if self.width else ""
        if len(self.figs) == 1:
            out.append(f'<div class="fig-b"{style}>{self._svg(self.figs[0])}</div>')
        else:
            out.append('<div class="fig-row">')
            subs = self.subcaptions or [None] * len(self.figs)
            for k, (f, sc) in enumerate(zip(self.figs, subs)):
                lab = f"({chr(97 + k)}) " if self.subcaptions else ""
                out.append(f'<div class="fig-b">{self._svg(f)}'
                           + (f'<div class="subcap">{lab}{rich(sc)}</div>' if sc else "") + "</div>")
            out.append("</div>")
        if num:
            out.append(f'<figcaption class="cap"><span class="cap-n">{num[0]}</span>{rich(self.caption)}</figcaption>')
        out.append("</figure>")
        return "".join(out)


class _Formula(_Block):
    kind = "eq"

    def __init__(self, rows, tag, id, where, cls):
        self.rows, self.tag, self.id, self.where, self.cls = rows, tag, id, where, cls

    def html(self, ctx):
        anchor = None
        tags = []
        for k, r in enumerate(self.rows):
            t = r[3] if isinstance(r, tuple) and len(r) > 3 else (self.tag if k == 0 else None)
            if t == "auto":
                lab, anchor = ctx.number("eq", self.id if k == 0 else None)
                t = lab
            elif t is not None and k == 0 and self.id:
                anchor = ctx.label(self.id, "eq", f"({t})", num=f"({t})")
            tags.append(t)
        cls = "eq" + (f" {self.cls}" if self.cls else "")
        out = [f'<div class="{cls}"' + (f' id="{anchor}"' if anchor else "") + ">"]
        for r, t in zip(self.rows, tags):
            tg = f'<span class="tag">{_html.escape(t if t.startswith("(") else f"({t})")}</span>' if t else \
                '<span class="tag"></span>'
            if isinstance(r, str):
                out.append(f'<span class="row1">{math(r)}</span>{tg}')
                continue
            if len(r) == 2:
                lhs, rel, rhs = r[0], "=", r[1]
            else:
                lhs, rel, rhs = r[0], r[1], r[2]
            out.append(f'<span class="lhs">{math(lhs) if lhs else ""}</span>'
                       f'<span class="rel">{math(rel) if rhs is not None else ""}</span>'
                       f'<span class="rhs">{math(rhs) if rhs is not None else ""}</span>{tg}')
        out.append("</div>")
        if self.where:
            w = ctx.word("where")
            out.append('<div class="where"><span class="where-h">' + w + "</span>"
                       + "".join(f'<div>{m(sym)}{"——" if ctx.lang == "zh" else " — "}{rich(txt)}</div>'
                                 for sym, txt in self.where) + "</div>")
        return "".join(out)


# ── the tree ────────────────────────────────────────────────────────────────
class Node:
    """a chapter, section or subsection: a heading and blocks in order. The adding methods return what they add (a
    child Node for section(), the node itself for the blocks), so blocks chain: node.paragraph(…).table(…)"""

    def __init__(self, report, parent, title, level, id=None, appendix=False, page_break=None):
        self.report, self.parent, self.title, self.level = report, parent, title, level
        self.id, self.appendix = id, appendix
        self.page_break = (level == 1) if page_break is None else page_break
        self.blocks: list = []  # blocks and child nodes in order
        self.number = ""  # set when written
        self.anchor = ""

    # structure
    def section(self, title, *, id: str | None = None) -> "Node":
        """a numbered child section (1.1 under chapter 1, 1.1.1 under 1.1, …)"""
        n = Node(self.report, self, title, self.level + 1, id=id, appendix=self.appendix)
        self.blocks.append(n)
        return n

    subsection = section

    @property
    def children(self) -> list["Node"]:
        return [b for b in self.blocks if isinstance(b, Node)]

    # blocks
    def paragraph(self, *parts, cls: str | None = None) -> "Node":
        """a paragraph of rich text parts (strings, numbers, Markup, Report.ref tokens)"""
        self.blocks.append(_Para(parts, cls))
        return self

    p = paragraph

    def table(self, rows, *, header=None, units=None, caption=None, id: str | None = None, align: str | None = None,
              digits=None, widths=None, notes=None, row_cls=None, keep: bool | None = None, compact: bool = False,
              cls: str | None = None) -> "Node":
        """a table: rows of cells (strings as rich text, numbers formatted by femotion.viz.number or with `digits`
        decimals — one value or one per column —, Markup, Cell); header; units: a row under the header (e.g. "kN·m");
        caption: numbered (表 n-k) above the table, None for an unnumbered one; align: one letter per column, l / r / c
        (default: right for columns of numbers); widths: CSS widths; notes: lines under the table; row_cls: a list or
        f(i, row) of row classes ("fail" red, "ok", "warn", "sub" muted, "total" bold); keep: never split across pages
        (default for 24 rows or fewer; longer tables break between rows and repeat the header); compact: smaller
        padding (parameter lists)"""
        self.blocks.append(_Table(rows, header, units, caption, id, align, digits, widths, notes, row_cls, keep,
                                  compact, cls))
        return self

    def kv(self, rows, *, caption=None, id=None, cols: int = 1) -> "Node":
        """a parameter list: rows of (name, value[, unit]) laid out in `cols` column groups, unnumbered unless
        caption"""
        rows = [tuple(r) + ("",) * (3 - len(r)) for r in rows]
        k = -(-len(rows) // cols) if rows else 0
        out = []
        for i in range(k):
            line = []
            for c in range(cols):
                j = c * k + i
                line += list(rows[j]) if j < len(rows) else ["", "", ""]
            out.append(line)
        return self.table(out, caption=caption, id=id, align="lrl" * cols, compact=True, cls="kv")

    def figure(self, fig, *, caption=None, id: str | None = None, subcaptions=None, width: str | None = None) -> "Node":
        """a figure: a femotion.viz Figure or Stack (anything with svg()), SVG text, or a list of them side by side
        (subcaptions (a), (b) …); caption numbered (图 n-k) under it, None for none; width: a CSS max-width"""
        self.blocks.append(_Figure(fig, caption, id, subcaptions, width))
        return self

    def formula(self, rows, *, tag: str | None = None, id: str | None = None, where=None,
                cls: str | None = None) -> "Node":
        """a displayed formula: rows aligned at their relation sign, each a string (one cell), (lhs, rhs) with "=",
        (lhs, rel, rhs) or (lhs, rel, rhs, tag); a continuation row has an empty lhs. tag: of the first row, e.g.
        "5.2.2-1" (shown in parentheses), "auto" for the chapter's numbering ((3-1)); where: rows of (symbol, meaning)
        under the formula ("式中：")"""
        if isinstance(rows, (str, tuple)):
            rows = [rows]
        self.blocks.append(_Formula(list(rows), tag, id, where, cls))
        return self

    def box(self, kind: str, *parts, title=None, items=None) -> "Node":
        """a box: kind "ok" (met, green), "fail" (not met, red), "warn" (warnings, amber), "info", "note"; a title, rich
        text parts and a list of items"""
        self.blocks.append(_Box(kind, parts, title, items))
        return self

    def list(self, items, *, ordered: bool = False, cls: str | None = None) -> "Node":
        """a bulleted (or numbered) list; an item is rich text or a list of parts"""
        self.blocks.append(_List(items, ordered, cls))
        return self

    def html(self, markup) -> "Node":
        """raw HTML (Markup or str taken as HTML)"""
        self.blocks.append(_Raw(Markup(str(markup))))
        return self

    def add(self, block) -> "Node":
        """any object with html(ctx) -> str"""
        self.blocks.append(block)
        return self

    def walk(self):
        """this node and its descendants, depth first"""
        yield self
        for c in self.children:
            yield from c.walk()


class _Ctx:
    """numbering while the document is written"""

    def __init__(self, report):
        self.report, self.lang = report, report.lang
        self.labels: dict[str, tuple[str, str]] = {}  # user id → (label text, anchor)
        self.chapter = ""
        self.count = {"fig": 0, "tab": 0, "eq": 0}
        self.free = 0
        self.nums: dict[str, str] = {}  # user id → the bare number ("3-2", "5.1.2")

    def word(self, k):
        return _W[self.lang][k]

    def start_chapter(self, num):
        self.chapter = num
        self.count = {"fig": 0, "tab": 0, "eq": 0}

    def number(self, kind, uid):
        self.count[kind] += 1
        k = self.count[kind]
        n = f"{self.chapter}-{k}" if self.chapter else f"{k}"
        anchor = f"{kind}-{n}".replace(".", "-")
        if kind == "eq":
            text = f"({n})"
        else:
            text = f"{self.word(kind)} {n}"
        if uid:
            self.label(uid, kind, text, anchor, num=n if kind != "eq" else f"({n})")
        return (text if kind == "eq" else text + self.word("sep")), anchor

    def label(self, uid, kind, text, anchor=None, num=None):
        if anchor is None:
            self.free += 1
            anchor = f"{kind}-x{self.free}"
        if uid in self.labels:
            raise ValueError(f"duplicate id {uid!r}")
        if num is None:
            num = text
        if kind == "eq" and not text.startswith(self.word("eq")):
            text = f"{self.word('eq')} {text}"
        self.labels[uid] = (text, anchor)
        self.nums[uid] = num
        return anchor


class Report:
    """a calculation report (see the module notes). title: the document title (cover, browser tab); project, member:
    the running head in print; number: report number; author, checker, approver: 编制, 校核, 审核 (names, or "" for a
    blank signature line); date, version; organisation: under the title on the cover; subtitle; lang "zh" or "en";
    toc_depth: heading levels in the printed contents (2: chapters and sections); store: a dict for extensions (e.g.
    the checks written by femotion.codes.jtg.report)"""

    def __init__(self, title: str, *, project: str = "", member: str = "", number: str = "", author: str = "",
                 checker: str = "", approver: str = "", date: str = "", version: str = "", organisation: str = "",
                 subtitle: str = "", lang: str = "zh", toc_depth: int = 2, cover: bool = True):
        if lang not in _W:
            raise ValueError("lang: 'zh' or 'en'")
        self.title, self.project, self.member, self.number = title, project, member, number
        self.author, self.checker, self.approver = author, checker, approver
        self.date, self.version, self.organisation, self.subtitle = date, version, organisation, subtitle
        self.lang, self.toc_depth, self.cover = lang, toc_depth, cover
        self.chapters: list[Node] = []
        self.store: dict = {}
        self.pages: dict[str, int] = {}  # anchor → page (Report.pdf)
        self._ctx: _Ctx | None = None

    # structure
    def chapter(self, title, *, id: str | None = None, page_break: bool = True) -> Node:
        """a numbered chapter, on a new page in print (page_break=False: straight after the one before)"""
        n = Node(self, None, title, 1, id=id, page_break=page_break)
        self.chapters.append(n)
        return n

    def appendix(self, title, *, id: str | None = None) -> Node:
        """a lettered appendix (附录 A), after the chapters"""
        n = Node(self, None, title, 1, id=id, appendix=True)
        self.chapters.append(n)
        return n

    @staticmethod
    def ref(id: str, *, bare: bool = False) -> Markup:
        """a cross-reference to the figure, table, formula or section with this id ("见图 3-2" is "见" + ref), a link;
        bare=True: the number alone ("3-2", "5.1.2"). Resolved when the HTML is written."""
        return Markup(f"⟦{'num' if bare else 'ref'}:{id}⟧")

    def word(self, k: str) -> str:
        return _W[self.lang][k]

    # output
    def _number_tree(self):
        k = a = 0
        for ch in self.chapters:
            if ch.appendix:
                ch.number = chr(65 + a)
                a += 1
            else:
                k += 1
                ch.number = str(k)
            ch.anchor = f"s-{ch.number}"
            for node in ch.walk():
                if node is ch:
                    continue
                idx = node.parent.children.index(node) + 1
                node.number = f"{node.parent.number}.{idx}"
                node.anchor = "s-" + node.number.replace(".", "-")

    def _heading_text(self, node) -> str:
        if node.level == 1 and node.appendix:
            return f"{self.word('app')} {node.number}"
        return node.number

    def _render_node(self, node: Node, ctx: _Ctx) -> str:
        lvl = min(node.level, 4)
        h = f"h{lvl}"
        num = self._heading_text(node)
        head = (f'<{h} id="{node.anchor}"><span class="hn">{num}</span>{rich(node.title)}</{h}>')
        if node.id:
            sec_label = (f"{num} {self.word('sec')}" if self.lang == "zh" and node.level > 1 else
                         (f"{self.word('sec')} {num}" if node.level > 1 else
                          (self.word("ch").format(n=num) if not node.appendix else num)))
            if node.level == 1 and node.appendix:
                sec_label = num
            ctx.label(node.id, "sec", sec_label, node.anchor, num=num)
        body = self._blocks_html(node.blocks, ctx, head)
        cls = f"sec lv{node.level}" + (" chapter" if node.level == 1 else "") + (" pb" if node.page_break else "")
        return f'<section class="{cls}">' + "".join(body) + "</section>"

    @staticmethod
    def _block(b, ctx) -> str:
        return b.html(ctx)

    def _blocks_html(self, blocks, ctx, head: str) -> list[str]:
        """the blocks of a node, kept with what follows them in print: the heading, and every lead-in paragraph (class
        "lead" or "muted": "(1) …", "验算位置：…"), stay on one page with the next block (unless that is a long table)"""
        out, pending = [], [head]

        def flush():
            if pending:
                out.append(f'<div class="keepnext">{"".join(pending)}</div>' if len(pending) > 1 else pending[0])
                pending.clear()
        for b in blocks:
            if isinstance(b, Node):
                flush()
                out.append(self._render_node(b, ctx))
                continue
            lead = b.kind == "p" and bool(set((b.cls or "").split()) & {"lead", "muted"})
            if b.kind == "tab" and not b.keep:
                flush()
            h = self._block(b, ctx)
            if lead:
                pending.append(h)
            elif pending:
                pending.append(h)
                flush()
            else:
                out.append(h)
        flush()
        return out

    def _toc(self, levels: int, *, side: bool) -> str:
        def items(nodes, depth):
            out = []
            for n in nodes:
                kids = [c for c in n.children] if depth < levels else []
                num = self._heading_text(n)
                page = self.pages.get(n.anchor)
                pg = f'<span class="tp">{page}</span>' if (page and not side) else ('<span class="tp"></span>'
                                                                                     if not side else "")
                link = (f'<a href="#{n.anchor}"><span class="tn">{num}</span><span class="tt">{rich(n.title)}</span>'
                        f'{"" if side else "<span class=lead></span>"}{pg}</a>')
                sub = f'<ul>{"".join(items(kids, depth + 1))}</ul>' if kids else ""
                if side and kids and depth == 1:
                    out.append(f'<li class="d{depth}"><details><summary>{link}</summary>{sub}</details></li>')
                else:
                    out.append(f'<li class="d{depth}">{link}{sub}</li>')
            return out
        return "<ul>" + "".join(items(self.chapters, 1)) + "</ul>"

    def _cover(self) -> str:
        w = self.word
        meta = [(w("project"), self.project), (w("member"), self.member), (w("number"), self.number),
                (w("version"), self.version), (w("date"), self.date)]
        meta_rows = "".join(f"<tr><th>{escape(k)}</th><td>{rich(v)}</td></tr>" for k, v in meta if v)
        sign = [(w("author"), self.author), (w("checker"), self.checker), (w("approver"), self.approver)]
        sign_rows = "".join(f"<tr><th>{escape(k)}</th><td>{rich(v)}</td><td></td>"
                            f"<td>{rich(self.date) if k == w('author') else ''}</td></tr>"
                            for k, v in sign if v is not None)
        org = f'<div class="cv-org">{rich(self.organisation)}</div>' if self.organisation else ""
        sub = f'<div class="cv-sub">{rich(self.subtitle)}</div>' if self.subtitle else ""
        return (f'<section class="cover"><div class="cv-top">{org}<h1 class="cv-title">{rich(self.title)}</h1>{sub}'
                f'<table class="cv-meta">{meta_rows}</table></div>'
                f'<table class="cv-sign"><thead><tr><th>{w("role")}</th><th>{w("name")}</th><th>{w("sign")}</th>'
                f'<th>{w("date")}</th></tr></thead><tbody>{sign_rows}</tbody></table></section>')

    def html(self) -> str:
        """the whole report as one self-contained HTML document"""
        self._number_tree()
        ctx = _Ctx(self)
        self._ctx = ctx
        chapters = []
        for ch in self.chapters:
            ctx.start_chapter(ch.number)
            chapters.append(self._render_node(ch, ctx))
        body = "".join(chapters)
        body = self._resolve(body, ctx)
        side = self._toc(3, side=True)
        toc = self._toc(self.toc_depth, side=False)
        cover = self._cover() if self.cover else ""
        head_l = self.project or self.title
        head_r = " · ".join(v for v in (self.member, self.number) if v) or (self.title if self.project else "")
        css = _CSS + _print_css(head_l, head_r, self.word("pages"), bool(self.cover))
        lang = "zh-CN" if self.lang == "zh" else "en"
        return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<meta name="generator" content="femotion.report"><title>{escape(self.title)}</title>'
                f"<style>{css}</style></head><body>"
                f'<nav class="side" aria-label="{self.word("toc")}"><details class="side-d" open>'
                f'<summary>{self.word("toc")}</summary>{side}</details></nav>'
                f'<main class="doc">{cover}<section class="toc"><h1 class="toc-h">{self.word("toc")}</h1>{toc}'
                f"</section>{body}</main>{_SCRIPT}</body></html>")

    def _resolve(self, body: str, ctx: _Ctx) -> str:
        def rep(mt):
            kind, uid = mt.group(1), mt.group(2)
            if uid not in ctx.labels:
                raise KeyError(f"reference to an unknown id {uid!r}")
            text, anchor = ctx.labels[uid]
            if kind == "page":
                return str(self.pages.get(anchor, ""))
            if kind == "num":
                text = ctx.nums.get(uid, text)
            return f'<a class="xref" href="#{anchor}">{_html.escape(text)}</a>'
        return _TOKEN.sub(rep, body)

    @property
    def labels(self) -> dict[str, tuple[str, str]]:
        """id → (label, anchor) of everything numbered with an id, after html()"""
        return dict(self._ctx.labels) if self._ctx else {}

    def save(self, path) -> Path:
        """write the HTML file"""
        p = Path(path)
        p.write_text(self.html(), encoding="utf-8")
        return p

    def pdf(self, path, *, chrome: str | None = None, html_path=None, passes: int = 3, timeout: float = 300) -> Path:
        """print to PDF (A4) with headless Chrome, twice or more: the first pass finds the page of every heading listed
        in the contents (from the PDF's link destinations), the next prints the contents with those page numbers, until
        they stay the same (at most `passes` prints). The pages are kept in Report.pages, so a later save() writes them
        into the HTML as well. chrome: the browser's executable (default: $CHROME, then the usual install locations);
        html_path: also write the final HTML there. The PDF carries an outline (bookmarks) of the headings."""
        exe = chrome or _find_chrome()
        out = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            h = Path(tmp) / "report.html"
            pdf = Path(tmp) / "report.pdf"
            for _ in range(max(passes, 1)):
                h.write_text(self.html(), encoding="utf-8")
                _print(exe, h, pdf, timeout)
                pages = _pdf_dest_pages(pdf.read_bytes())
                if not pages or pages == self.pages:
                    break
                self.pages = pages
            shutil.copyfile(pdf, out)
        if html_path is not None:
            self.save(html_path)
        return out


# ── printing ────────────────────────────────────────────────────────────────
def _find_chrome() -> str:
    cands = [os.environ.get("CHROME", ""),
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    for c in cands:
        if c and Path(c).exists():
            return c
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("Report.pdf needs Chrome or Chromium: pass chrome=… or set $CHROME")


def _print(exe: str, html_file: Path, pdf_file: Path, timeout: float):
    if pdf_file.exists():
        pdf_file.unlink()
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline",
           f"--print-to-pdf={pdf_file}", html_file.resolve().as_uri()]
    subprocess.run(cmd, check=True, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not pdf_file.exists():
        raise RuntimeError("Chrome wrote no PDF")


def _pdf_objects(data: bytes) -> dict[int, bytes]:
    return {int(mt.group(1)): mt.group(2) for mt in re.finditer(rb"(\d+) 0 obj\s*(.*?)\s*endobj", data, re.S)}


def _pdf_dest_pages(data: bytes) -> dict[str, int]:
    """anchor → page (1-based) of the named destinations of a PDF printed by Chrome (its link targets); {} when the
    file has another structure"""
    objs = _pdf_objects(data)
    root = None
    for k, v in objs.items():
        if re.search(rb"/Type\s*/Catalog\b", v):
            root = v
            break
    if root is None:
        return {}
    mt = re.search(rb"/Pages\s+(\d+)\s+0\s+R", root)
    if not mt:
        return {}
    order: list[int] = []

    def kids(n, depth=0):
        o = objs.get(n, b"")
        km = re.search(rb"/Kids\s*\[([^\]]*)\]", o)
        if km and depth < 32 and re.search(rb"/Type\s*/Pages\b", o):
            for r in re.findall(rb"(\d+)\s+0\s+R", km.group(1)):
                kids(int(r), depth + 1)
        else:
            order.append(n)
    kids(int(mt.group(1)))
    page_of = {n: i + 1 for i, n in enumerate(order)}
    dm = re.search(rb"/Dests\s+(\d+)\s+0\s+R", root)
    dests = objs.get(int(dm.group(1)), b"") if dm else b""
    if not dests:
        nm = re.search(rb"/Names\s*<<.*?/Dests\s+(\d+)\s+0\s+R", root, re.S)
        dests = objs.get(int(nm.group(1)), b"") if nm else b""
    out = {}
    for name, ref in re.findall(rb"/([A-Za-z0-9_.#-]+)\s*\[\s*(\d+)\s+0\s+R\s*/XYZ", dests):
        n = re.sub(rb"#([0-9A-Fa-f]{2})", lambda x: bytes([int(x.group(1), 16)]), name).decode("utf-8", "replace")
        if int(ref) in page_of:
            out[n] = page_of[int(ref)]
    for name, ref in re.findall(rb"\((.*?)\)\s*\[\s*(\d+)\s+0\s+R\s*/XYZ", dests):  # name trees
        if int(ref) in page_of:
            out[name.decode("utf-8", "replace")] = page_of[int(ref)]
    return out


# ── style ───────────────────────────────────────────────────────────────────
# the contents fold away on narrow screens (they open with a tap); nothing else needs a script
_SCRIPT = ("<script>(function(){var d=document.querySelector('.side-d');"
           "if(d&&window.matchMedia&&window.matchMedia('(max-width: 1239px)').matches){d.open=false;}})();</script>")
C = COLORS
_MATH_FONT = "'Times New Roman', Times, 'STIX Two Text', 'Nimbus Roman', serif"
_CSS = f"""
:root {{ --ink: {C['ink']}; --muted: {C['muted']}; --rule: #c8c8c8; --soft: #f4f3ef; --ok: {C['ok']};
  --fail: {C['warn']}; --warn: #9a5b00; --link: {C['resistance']}; }}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: #ecebe7; color: var(--ink); font-family: {FONT}; font-size: 15px; line-height: 1.7; }}
.doc {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 40px 56px 64px; min-height: 100vh;
  box-shadow: 0 0 0 1px #dedcd6; }}
.side {{ position: fixed; top: 0; left: 0; bottom: 0; width: 270px; overflow-y: auto; background: #f7f6f2;
  border-right: 1px solid #dedcd6; padding: 16px 12px 40px 16px; font-size: 13px; line-height: 1.45; z-index: 2; }}
@media (min-width: 1240px) {{ body {{ padding-left: 270px; }} }}
.side summary {{ cursor: pointer; }}
.side-d > summary {{ font-weight: 600; font-size: 14px; margin-bottom: 8px; list-style: none; }}
.side ul {{ list-style: none; margin: 0; padding: 0; }}
.side li {{ margin: 2px 0; }}
.side li ul {{ padding-left: 14px; }}
.side a {{ color: var(--ink); text-decoration: none; display: inline; }}
.side a:hover {{ color: var(--link); }}
.side details > summary {{ list-style-position: outside; }}
.side li.d1 > a, .side li.d1 > details > summary > a {{ font-weight: 600; }}
.side .tn {{ color: var(--muted); margin-right: .45em; font-variant-numeric: tabular-nums; }}
@media (max-width: 1239px) {{
  .side {{ position: static; width: auto; max-width: 900px; margin: 0 auto; border-right: 0;
    border-bottom: 1px solid #dedcd6; padding: 10px 16px; }}
  .side-d:not([data-touched]) {{ }}
}}
@media (max-width: 700px) {{
  body {{ font-size: 14.5px; }}
  .doc {{ padding: 20px 16px 40px; }}
  .eq {{ margin-left: 0 !important; }}
}}
h1, h2, h3, h4 {{ line-height: 1.35; margin: 1.4em 0 .5em; font-weight: 700; }}
h1 {{ font-size: 1.45em; padding-bottom: .3em; border-bottom: 2px solid var(--ink); margin-top: 2em; }}
h2 {{ font-size: 1.2em; }}
h3 {{ font-size: 1.06em; }}
h4 {{ font-size: 1em; }}
.hn {{ margin-right: .7em; font-variant-numeric: tabular-nums; }}
p {{ margin: .45em 0; text-align: justify; }}
ul, ol {{ margin: .4em 0; padding-left: 1.6em; }}
a.xref {{ color: var(--link); text-decoration: none; }}
sub, sup {{ font-size: .72em; line-height: 0; }}
.m, .eq {{ font-family: {_MATH_FONT}, {FONT}; }}
.m {{ white-space: nowrap; }}
.m i, .eq i {{ font-style: italic; }}
.mt {{ font-family: {FONT}; font-style: normal; }}
.mu {{ font-style: normal; }}
.v {{ font-style: normal; }}
.fr {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; margin: 0 .12em;
  line-height: 1.25; }}
.fr > .nu {{ padding: 0 .2em .05em; }}
.fr > .de {{ padding: .05em .2em 0; border-top: 1px solid currentColor; }}
.sq {{ white-space: nowrap; }}
.sq > .rad {{ font-family: {_MATH_FONT}; padding-right: .05em; }}
.sq > .sqc {{ border-top: 1px solid currentColor; padding: 0 .12em; display: inline-block; line-height: 1.3; }}
.eq {{ display: grid; grid-template-columns: auto auto minmax(0, 1fr) auto; column-gap: .45em; row-gap: .3em;
  align-items: center; margin: .5em 0 .6em 2em; font-size: 1.04em; }}
.eq .lhs {{ text-align: right; white-space: nowrap; }}
.eq .rel {{ text-align: center; }}
.eq .rhs {{ min-width: 0; }}
.eq .row1 {{ grid-column: 1 / 4; }}
.eq .tag {{ text-align: right; white-space: nowrap; color: var(--muted); font-family: {FONT}; font-size: .9em;
  padding-left: 1.5em; }}
.where {{ margin: -.2em 0 .7em 2em; font-size: .93em; }}
.where > div {{ padding-left: 3em; }}
.where-h {{ display: block; }}
.tbl {{ margin: .9em 0 1.1em; }}
.tblw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9em; line-height: 1.45; }}
.tbl table {{ border-top: 1.5px solid var(--ink); border-bottom: 1.5px solid var(--ink); }}
.tbl th {{ font-weight: 600; text-align: left; padding: .35em .6em; border-bottom: 1px solid var(--ink);
  background: var(--soft); vertical-align: bottom; }}
.tbl thead tr.units th {{ font-weight: 400; color: var(--muted); font-size: .92em; padding-top: 0; }}
.tbl thead tr:first-child th {{ border-bottom-color: var(--rule); }}
.tbl thead tr:last-child th {{ border-bottom-color: var(--ink); }}
.tbl td {{ padding: .3em .6em; border-bottom: 1px solid #e3e1db; vertical-align: top; }}
.tbl tr:last-child td {{ border-bottom: 0; }}
.tbl .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.tbl .c {{ text-align: center; }}
.tbl.compact td, .tbl.compact th {{ padding: .18em .6em; }}
.tbl.kv table {{ border-top: 0; border-bottom: 0; }}
.tbl.kv td {{ border-bottom: 1px dotted #d6d3cb; }}
.tbl.kv td:nth-child(3n+3) {{ color: var(--muted); }}
.tbl tr.fail td {{ color: var(--fail); }}
.tbl tr.warn td {{ color: var(--warn); }}
.tbl tr.ok td.verdict, .tbl td.ok {{ color: var(--ok); }}
.tbl td.fail {{ color: var(--fail); font-weight: 600; }}
.tbl tr.sub td {{ color: var(--muted); }}
.tbl tr.total td {{ font-weight: 600; border-top: 1px solid var(--ink); }}
.tbl.fixed table {{ table-layout: fixed; }}
.tbl.fixed td, .tbl.fixed th {{ overflow-wrap: anywhere; }}
.tbl td.unit {{ white-space: nowrap; overflow-wrap: normal; }}
.tbl.summary table {{ font-size: .84em; }}
.tnote {{ font-size: .85em; color: var(--muted); margin-top: .3em; }}
.cap {{ font-size: .92em; font-weight: 600; text-align: center; margin: .3em 0; }}
.cap-n {{ font-weight: 700; }}
.fig {{ margin: 1em 0 1.2em; text-align: center; }}
.fig svg {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.fig-b {{ margin: 0 auto; }}
.fig-row {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; align-items: flex-end; }}
.fig-row > .fig-b {{ flex: 1 1 260px; margin: 0; }}
.subcap {{ font-size: .88em; color: var(--muted); }}
.box {{ border-left: 4px solid; padding: .45em .9em; margin: .7em 0 .9em; border-radius: 0 3px 3px 0; }}
.box ul {{ margin: .2em 0; }}
.box-t {{ font-weight: 700; }}
.box-ok {{ border-color: var(--ok); background: #eef6ee; }}
.box-ok .box-t {{ color: var(--ok); }}
.box-fail {{ border-color: var(--fail); background: #fbecea; }}
.box-fail .box-t {{ color: var(--fail); }}
.box-warn {{ border-color: #d08a14; background: #fdf4e3; }}
.box-warn .box-t {{ color: var(--warn); }}
.box-info {{ border-color: var(--link); background: #edf2f8; }}
.box-note {{ border-color: #b8b6ae; background: #f6f5f1; font-size: .93em; }}
.cover {{ padding: 24px 0 40px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; }}
.cv-org {{ font-size: 1.05em; color: var(--muted); letter-spacing: .1em; }}
.cv-title {{ font-size: 2em; border: 0; margin: .3em 0 .2em; letter-spacing: .04em; }}
.cv-sub {{ font-size: 1.15em; color: var(--muted); }}
.cv-meta {{ width: auto; margin-top: 1.4em; font-size: 1em; }}
.cv-meta th {{ text-align: left; font-weight: 400; color: var(--muted); padding: .25em 2em .25em 0; }}
.cv-meta td {{ padding: .25em 0; }}
.cv-sign {{ margin-top: 2em; max-width: 520px; border-top: 1.5px solid var(--ink);
  border-bottom: 1.5px solid var(--ink); }}
.cv-sign th, .cv-sign td {{ padding: .45em .8em; border-bottom: 1px solid var(--rule); text-align: left; }}
.cv-sign thead th {{ background: var(--soft); font-weight: 600; }}
.cv-sign td:nth-child(3) {{ min-width: 7em; }}
.toc {{ display: none; }}
.toc ul {{ list-style: none; padding: 0; margin: 0; }}
.toc li ul {{ padding-left: 2.2em; }}
.toc a {{ display: flex; align-items: baseline; color: var(--ink); text-decoration: none; padding: .12em 0; }}
.toc li.d1 > a {{ font-weight: 600; margin-top: .45em; }}
.toc .tn {{ min-width: 2.6em; }}
.toc li ul .tn {{ min-width: 3.2em; }}
.toc .lead {{ flex: 1 1 auto; border-bottom: 1px dotted #9a9a9a; margin: 0 .4em; transform: translateY(-.3em); }}
.toc .tp {{ min-width: 2.2em; text-align: right; font-variant-numeric: tabular-nums; }}
.keepnext {{ }}
p.muted {{ color: var(--muted); font-size: .93em; }}
p.lead {{ font-weight: 600; margin-top: .8em; }}
"""


def _css_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _print_css(left: str, right: str, pages: str, cover: bool) -> str:
    first = ("@page :first { @top-left { content: none; border: 0 } @top-right { content: none; border: 0 } "
             "@bottom-center { content: none } }") if cover else ""
    return f"""
@page {{ size: A4; margin: 24mm 18mm 20mm 22mm;
  @top-left {{ content: {_css_string(left)}; font: 8.5pt {FONT}; color: #555; vertical-align: bottom;
    padding-bottom: 2.5mm; border-bottom: .6pt solid #777; }}
  @top-right {{ content: {_css_string(right)}; font: 8.5pt {FONT}; color: #555; vertical-align: bottom;
    padding-bottom: 2.5mm; border-bottom: .6pt solid #777; }}
  @bottom-center {{ content: "{pages}"; font: 8.5pt {FONT}; color: #555; }}
}}
{first}
@media print {{
  html, body {{ background: #fff; }}
  body {{ padding: 0 !important; font-size: 10.5pt; line-height: 1.6; }}
  .side {{ display: none !important; }}
  .doc {{ max-width: none; margin: 0; padding: 0; box-shadow: none; min-height: 0; }}
  .cover {{ height: 247mm; display: flex; flex-direction: column; justify-content: space-between; border: 0;
    padding: 30mm 6mm 10mm; margin: 0; break-after: page; }}
  .cv-title {{ font-size: 24pt; }}
  .toc {{ display: block; break-after: page; }}
  .toc-h {{ border: 0; text-align: center; letter-spacing: .5em; margin-top: 0; }}
  .sec.pb {{ break-before: page; }}
  h1 {{ margin-top: 0; font-size: 15pt; }}
  .sec.lv1:not(.pb) h1 {{ margin-top: 1.6em; }}
  h2 {{ font-size: 12.5pt; }}
  h3 {{ font-size: 11pt; }}
  h1, h2, h3, h4 {{ break-after: avoid; }}
  .keepnext, .fig, .eq, .where, .box, .tbl.keep, .cap {{ break-inside: avoid; }}
  thead {{ display: table-header-group; }}
  tr {{ break-inside: avoid; }}
  .tblw {{ overflow: visible; }}
  a {{ color: inherit; }}
  .box, .tbl th, .tbl tr.fail td, .cv-sign thead th {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  p {{ orphans: 3; widows: 3; }}
}}
"""
