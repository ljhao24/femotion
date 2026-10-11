# Calculation reports (计算书)

femotion writes a design calculation report as **one self-contained HTML file**: open it in a browser to read it (a
contents sidebar, the width follows the window, readable on a phone), print it to A4 ("Save as PDF") to hand it in. Two
layers:

| Module | Does |
|---|---|
| `femotion.report` | the document: cover, numbered chapters and sections, contents, paragraphs, tables, figures, formulas, boxes, cross-references; HTML out, print styles; `Report.pdf` prints it with Chrome, the contents with page numbers. Knows nothing about design codes |
| `femotion.codes.jtg.report` | writes the `Check`s of `femotion.codes.jtg` into it: each formula in symbols, then with the values substituted, the result and the verdict; the summary of every check; tables of the design data |

Example: `examples/jtg_report_rc_girder.py` writes the complete report of the two-span reinforced-concrete T girder of
`examples/jtg_rc_girder.py` (basis, design data, model, actions and effects, ultimate and serviceability limit states,
detailing, summary and conclusion).

```python
from femotion.report import Report
from femotion.codes import jtg
from femotion.codes.jtg import report as jr

rep = Report("两跨连续 T 梁计算书", project="某大桥", member="主梁", number="JS-001", author="张三", checker="李四",
             date="2026-10-12", version="A")
ch = rep.chapter("设计资料")
jr.materials(ch.section("材料"), [sec])
ch = rep.chapter("持久状况承载能力极限状态")
jr.write_check(ch, jtg.flexure(sec, Md, gamma0), where="x = 25.50 m（跨中）")
jr.write_check(ch, jtg.shear(sec, Vd, (2, 12, 100), gamma0=gamma0), where="x = 0.55 m（端支点 h/2 处）")
jr.summary(rep.chapter("验算汇总和结论"))
rep.save("girder.html")
rep.pdf("girder.pdf")          # optional: needs Chrome or Chromium
```

## The document: `femotion.report`

### Report

`Report(title, *, project="", member="", number="", author="", checker="", approver="", date="", version="",
organisation="", subtitle="", lang="zh", toc_depth=2, cover=True)`

- `chapter(title, *, id=None, page_break=True)` → a `Node`; `appendix(title, *, id=None)` → lettered (附录 A).
- `ref(id, *, bare=False)`: a cross-reference, written into any text ("见" + `rep.ref("fig:sec")` → "见图 3-2"), resolved
  when the HTML is written: forward references work, an unknown id raises `KeyError`, a duplicate id `ValueError`.
  `bare=True` gives the number alone ("3-2", "5.1.2").
- `html()` → str; `save(path)`; `pdf(path, *, chrome=None, html_path=None, passes=3)`.
- `labels`: id → (label, anchor) after `html()`; `pages`: anchor → page after `pdf()`; `store`: a dict for extensions
  (the jtg layer keeps the checks it wrote there).

### Node (chapter, section, subsection …)

Every adding method returns the node, so they chain; `section` returns the new child.

| Method | Adds |
|---|---|
| `section(title, *, id=None)` | a numbered child (1.1, 1.1.1, …; any depth) |
| `paragraph(*parts, cls=None)` / `p` | rich text parts (strings, numbers, `Markup`, `ref` tokens); `cls="lead"` a lead-in kept with what follows it, `"muted"` |
| `table(rows, *, header, units, caption, id, align, digits, widths, notes, row_cls, keep, compact)` | a table: the caption numbered above it (表 n-k), a units row under the header, columns of numbers right-aligned, `digits` (one value or one per column), rows classed `"fail"` (red), `"warn"`, `"ok"`, `"sub"`, `"total"`; `Cell(value, cls, colspan=, rowspan=, digits=)` for single cells |
| `kv(rows, *, cols=1)` | a parameter list of (name, value, unit) in `cols` column groups |
| `figure(fig, *, caption, id, subcaptions, width)` | a `femotion.viz.Figure` or `Stack` (anything with `svg()`), SVG text, or a list of them side by side with (a), (b) subcaptions; the caption numbered under it (图 n-k) |
| `formula(rows, *, tag, id, where)` | a displayed formula, rows aligned at their relation sign: `(lhs, rhs)` for "=", `(lhs, rel, rhs)`, `(lhs, rel, rhs, tag)` or a string; an empty lhs continues the row above; `tag="5.2.2-1"`, or `"auto"` for the chapter's numbering (n-k); `where`: rows of (symbol, meaning) ("式中：") |
| `box(kind, *parts, title, items)` | `"ok"` (green), `"fail"` (red), `"warn"` (amber), `"info"`, `"note"` |
| `list(items, *, ordered=False)`, `html(markup)`, `add(block)` | a list, raw HTML, any object with `html(ctx)` |

### Text and formulas

Strings are HTML-escaped; `x_{sub}` and `x^{sup}` are sub- and superscripts, `$…$` is inline math; `Markup` is written as
it is. The math language (`math()`, `m()` inline, the rows of `formula`):

| Write | Get |
|---|---|
| single Latin letters, lowercase Greek | variables in italic; a run of letters is a product (`bh_{0}` is *b h*₀) |
| `sin cos tan max min ln log exp`, `\op{…}` | upright functions |
| `_{…}`, `^{…}` | subscripts (upright: `f_{cd}`, `σ_{pe,ex}`), superscripts |
| `\frac{a}{b}`, `\sqrt{a}` | a built-up fraction, a radical with its bar |
| `\u{kN·m}`, `\t{取}` | a unit, upright text (Chinese) |
| `\v{key}{value}` | a substituted value, tagged `data-sym="key"` (scripts and tests read the numbers shown) |
| `'`, `-`, `<=`, `>=`, `\,` | prime ′, minus sign −, ≤, ≥, thin space |

No MathJax or other library: fractions and radicals are CSS (inline-flex columns, a border for the bar).

### Output

- **Self-contained**: one file, styles inline, nothing loaded from outside (a three-line inline script folds the
  contents on narrow screens); the fonts of `femotion.viz.FONT`, math letters in Times New Roman falling back to them.
- **Screen**: a fixed contents sidebar on wide screens (chapters fold open), a folding contents list at the top of the
  page below 1240 px; the page at most 900 px wide; on a phone wide tables scroll sideways instead of widening the page.
- **Print (A4, CSS paged media)**: the cover on its own page (title block, the report's data, a signature table of
  编制 / 校核 / 审核), the contents on the next, every chapter on a new page (`page_break=False` runs on), a running head
  (the project on the left, member and number on the right, over a rule) and "第 n 页 共 N 页" (`Page n of N`) in the
  page margins: CSS `@page` margin boxes with `counter(page)` and `counter(pages)`, which Chrome supports from version
  131 (other browsers print without them); no head on the cover. Figures, formulas, boxes and tables of up to 24 rows
  are never split; longer tables break between rows and repeat their header; a heading stays with its first block, a
  lead-in paragraph ("（1）…", "验算位置：…") with the block after it.
- **Contents with page numbers**: CSS cannot do it in Chrome (`target-counter()` is not supported), so `Report.pdf`
  prints twice with headless Chrome (`--print-to-pdf`, `--generate-pdf-document-outline`): it reads the page of every
  heading from the link destinations of the first PDF (the contents link to each heading), writes them into the
  contents and prints again, until they no longer change. They stay in `Report.pages`, so a later `save()` writes them
  into the HTML too (right for Chrome's A4 layout). The PDF also carries an outline (bookmarks) of the headings.
  Printed from a browser without this step, the contents are links without page numbers.

## Checks in the report: `femotion.codes.jtg.report`

| Function | Writes |
|---|---|
| `write_check(node, chk, *, where, title, id, lang, figure="auto", figure_kw, formulas=True, details=False)` | one check as a numbered section (below); returns it |
| `summary(node, entries=None)` | every check written so far (or `entries`): one table (section, location, check, clause, γ0Sd, R or limit, unit, ratio, verdict; those not met in red, each linked to its section), the warnings table and the conclusion |
| `design_basis(node, extra=())` | the codes |
| `materials(node, sections)` | concrete, reinforcing and prestressing steel of the sections: fck, ftk, fcd, ftd, Ec; fsk, fsd, f′sd, Es; fpk, fpd, f′pd, Ep, σcon |
| `section_properties(node, {name: section})` | h, b, A, yc, I, A0, y0, I0, As, A′s, h0, and the bars of each row |
| `load_cases(node, cases, *, model=None)` | the items of `LoadCase`s: self-weight (q = m g), distributed, point and nodal loads, imposed strains, tendons, settlements |
| `live_load(node, live, *, mu=None)`, `combinations(node, *, gamma0, crowd, characteristic)` | the lane load and its factors; the combinations of D60 4.1.5, 4.1.6, 4.1.8 with their factors |
| `shear_search_table(node, search)`, `stirrup_design_block(node, design, Vd)`, `bent_bar_table(node, design)` | the inclined sections of `jtg.shear_search`; the stirrups of 5.2.13-1 with the check back-substituted; the bent-up bars |
| `check_name`, `clause_text`, `message` | a check's name, its clause and its messages in the language |

### One check

`write_check` writes the location (as the caller gives it) and the clause; the calculation, each formula first in
symbols, then with the values substituted, then the result, tagged with the code's equation number; the verdict in a
green or red box (γ0Sd ≤ R, or the value against its limit, with the ratio); what failed; the warnings apart in an
amber box (detailing, minimum reinforcement, materials: they do not change the verdict); the notes; and, for flexure,
compression, tension, stresses and section detailing, the section drawn with the check over it
(`codes.jtg.figures.section_figure`). Values are substituted in N, mm and MPa (said once per chapter); moments come out
in kN·m.

Templates:

| Check | Written |
|---|---|
| `flexure` | rectangles, T, I and box sections: the flange or web test (5.2.3-1); x (5.2.2-2, 5.2.3-3; 5.2.5-2, -4, -6 with external tendons); ξ ≤ ξb; x ≥ 2a′; Mu (5.2.2-1, 5.2.3-2, 5.2.5-…); x < 2a′ by 5.2.4 / 5.2.6 with the singly reinforced section (program convention), also when the compression steel alone balances the tension steel; internal tendons on both faces; ρ ≥ ρmin (9.1.12), Mud / Mcr (9.1.13) |
| `shear` | P (at most 2.5), ρsv from the stirrups, α1 α2 α3, Vcs (5.2.9-2), Vsb, Vpb, Vpb,ex, Vu; 5.2.11 and 5.2.12; slabs without stirrups (1.25 × 5.2.12); the section limit at the end supports of `shear_search` |
| `axial` | l0/b (i, d), φ of table 5.3.1, Nu (5.3.1) |
| `eccentric` | e0 with its minimum, ζ1, ζ2, η (5.3.9); x, ξ against ξb (large or small eccentricity), σs by 5.1.5-1, Nu (5.3.4-1, 5.3.5-1) and the moment equilibrium (5.3.4-2) as a check; circles by 5.3.8 (α, αt, Nu) |
| `tension` | axial tension (5.4.1); small eccentricity (5.4.2-1, -2) |
| `torsion` | βt (5.5.4-3), ζ (5.5.1-2), Tu (5.5.1-1; 5.5.4-2 with the stirrups left for torsion, 5.5.6) |
| `crack_width` | σss (6.4.4-2 for flexural members; the other forms as the value), C2, ρte with its limits, Wcr (6.4.3) |
| `crack_resistance`, `inclined_crack_resistance`, `principal_compression` | both edges (W0, σst, σlt, σpc) and the governing inequality (6.3.1-1 … -4); the principal-stress points and σtp, σcp (6.3.3-1) |
| `service_stresses`, `transient_pc` | σkc and σpt at both edges against 0.5 fck (7.1.5-1), the largest tendon stress (7.1.5-2); the edge stresses at transfer (7.2.8) |
| `deflection_check` | f ≤ L/600 (L/300 for cantilevers), 6.5.3 |
| `local_bearing`, `punching`, `footing_punching` | β, Fu1 (5.7.1-1), βcor, ρv, Fu2 (5.7.2-1); 5.6.1, 5.6.2, 5.6.3 |
| bearings and joints (`devices`) | 8.7.3-1 … -10 (area, rubber thickness for shear and stability, compression, plates), 8.7.4 sliding, 8.7.5 PTFE friction, 8.8.2-7 joint movement |
| detailing reports (`Check.report`) | the rules (value, limit, met or not) and the other values |

Each template recomputes the result from the formula it writes and compares it with the check (relative 1e-6). When it
does not reproduce it — a section outside the formula's premises (bars between the edge rows, over-reinforced, a
composite section, circles by the strain method, class B cracked sections …), which the program solves by its unified
algorithm (docs/codes/JTG_3362-2018.md §10) — the check is written in the general form: a table of `Check.details` with
symbols and units. Every kind of check in the library can be written.

### Language

`lang="zh"` (default: the report's) or `"en"`: the words of the report, the check names and the common messages; the
program's other messages stay in English.

## Tests

`tests/test_report.py`: the numbers a formula shows are read back from the HTML (`data-sym`, as printed, rounded) and put
into the code's formula written out once more in the test — flexure (rectangle singly and doubly reinforced, T with the
zone in the flange or the web, x < 2a′, hogging, A′s > As), shear (5.2.9 with P capped or not), eccentric compression
(large, small, 5.3.8), crack width, axial compression, local compression, punching, torsion, deflection — and must give
the check's resistance or demand to the rounding of the printed values. Every literal check name in the source of
`femotion.codes.jtg` is written (the checks of all the examples and a catalogue), in Chinese and in English. The summary
counts the checks, those not met and the warnings. The HTML parses, its ids are unique, every link and reference
resolves, nothing is loaded from outside; figures and tables are numbered consecutively in each chapter. The parser of
PDF destinations is tested on a fixed PDF; with Chrome present, the two-pass print puts the right page numbers into the
contents.
