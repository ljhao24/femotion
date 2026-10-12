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
| `\v{key}{value}` | a substituted value, tagged `data-sym="key"` (scripts and tests read the numbers shown); the value may be `a×10^{n}` |
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
| `write_check(node, chk, *, where, title, id, lang, figure="auto", figure_kw, formulas=True, details=False, counted=True)` | one check as a numbered section (below); returns it. It keeps an `Entry` in `rep.store["jtg.checks"]`: where, check, title, id, `counted`, `guards` (results the formulas reproduced, 0 for the general form), `clause` (the clauses written) |
| `summary(node, entries=None)` | every check written so far (or `entries`): one table (section, location, check, clause, γ0Sd, R or limit, unit, ratio — the margin when the limit is not above zero —, verdict; those not met in red with the clause of the condition not met, each linked to its section; design steps written with `counted=False` listed but not counted), the warnings table and the conclusion |
| `design_basis(node, extra=())` | the codes |
| `materials(node, sections)` | concrete, reinforcing and prestressing steel of the sections: fck, ftk, fcd, ftd, Ec; fsk, fsd, f′sd, Es; fpk, fpd, f′pd, Ep, σcon |
| `section_properties(node, {name: section}, *, hogging=())` | h, b, A, yc, I, A0, y0, I0, As, A′s, h0, and the bars of each row; the sections named in `hogging` with As, A′s and h0 of the top face in tension |
| `load_cases(node, cases, *, model=None)` | the items of `LoadCase`s: self-weight (q = m g), distributed, point and nodal loads, imposed strains, tendons, settlements |
| `live_load(node, live, *, mu=None)`, `combinations(node, *, gamma0, crowd, characteristic)` | the lane load and its factors (1.2Pk for shear; one girder with `lanes=1, transverse=1` and a distribution factor: a note that m includes the lanes and table 4.3.1-5); the combinations of D60 4.1.5, 4.1.6, 4.1.8 with their factors, S_ud = γ0 S(…), and a note that the effects tabled are without γ0 |
| `shear_search_table(node, search)`, `stirrup_design_block(node, design, Vd=None)`, `bent_bar_table(node, design)` | the inclined sections of `jtg.shear_search`; the stirrups of 5.2.13-1 with the ξ, γ0 and V′d of the design written out in the denominator (ξγ0V′d)², and the check back-substituted (a design step, not counted); the bent-up bars |
| `check_name`, `clause_text`, `message` | a check's name, its clause and its messages in the language |

### One check

`write_check` writes the location (as the caller gives it) and the clauses the calculation follows (5.2.4 when x < 2a′
governs, for example); the calculation, each formula first in symbols, then with the values substituted, then the
result, tagged with the code's formula number (a clause without a formula of its own as "(5.3.9 条)" / "(cl. 5.3.9)");
the verdict in a green or red box (a condition not met first, then γ0Sd against R or the value against its limit, with
the ratio, or the margin when the limit is not above zero); the other conditions not met; the warnings apart in an amber
box (detailing, minimum reinforcement, materials: they do not change the verdict); the notes the formulas have not
already said; and, for flexure, compression, tension, stresses and section detailing, the section drawn with the check
over it (`codes.jtg.figures.section_figure`; the main rows of a detailing check with `main_rows` as the check takes
them). Values are substituted in N, mm and MPa (said once per chapter); moments come out in kN·m; a negative value after
an operator in parentheses.

Two rules hold for every template:

- **The page reproduces the program.** Each template computes every result it prints from the formula it prints and the
  values it substitutes, and compares it with the check (`guard`, relative 1e-6). When one does not match — a section
  outside the formula's premises (bars between the edge rows, a composite section, circles by the strain method, the
  load between the gross and the plastic centroids …), which the program solves by its unified algorithm
  (docs/codes/JTG_3362-2018.md §10) — the template is dropped and the check is written in the general form: a table of
  `Check.details`, every name translated, with symbols and units. Every kind of check in the library can be written.
- **Every inequality is set by the two numbers it compares**, never by the verdict; a condition not met stands on a row
  of its own.

Templates:

| Check | Written |
|---|---|
| `flexure` | rectangles, T, I and box sections: h0 = h − a with the resultant a of As, Ap and external tendons; a′ of A′s and A′p; the flange or web test (5.2.3; 5.2.5); x (5.2.2-2, 5.2.3-3; 5.2.5-2, -4, -6); ξ ≤ ξb; x ≥ 2a′ (5.2.2-4) or 2a′s (5.2.2-5); Mu (5.2.2-1, 5.2.3-2, 5.2.5-…); x < 2a′ by 5.2.4-1, -2 / 5.2.6 with the singly reinforced section (program convention), also when the compression steel alone balances the tension steel; over-reinforced sections (σs by 5.1.5-1, ΣN = 0, Mu for information) and 5.2.7 (Mu,b at x = ξb h0); ρ ≥ ρmin (9.1.12), Mcr = (σpc + γftk)W0 with γ = 2S0/W0 (6.5.2-7, -8) and Mud / Mcr (9.1.13) |
| `shear` | P (at most 2.5), ρsv, fsv at most 330 MPa (5.2.9), α1 α2 α3, Vcs (5.2.9-2), Vsb, Vpb, Vpb,ex, Vu; 5.2.11 and 5.2.12; slabs without stirrups (1.25 × 5.2.12, program convention, said); the section limit at the end supports of `shear_search`; the back-substitution of a stirrup design (ξγ0V′d; the limits with γ0V′d) |
| `axial` | l0/b (i, d), φ of table 5.3.1, A or An, Nu (5.3.1) |
| `eccentric` | e0 = Md/Nd with its minimum, l0/i ≤ 17.5 or ζ1, ζ2 (with their caps), η (5.3.9-1 … -3); e = ηe0 + h/2 − a (5.3.4-3); x, ξ against ξb, σs by 5.1.5-1, x ≥ 2a′s, x > h (concrete with x = h, 5.3.4 note); Nu (5.3.4-1, 5.3.5-1 when x > h′f) and the moment equilibrium (5.3.4-2, 5.3.5-2); the reverse failure (e′ = h/2 − e0 − a′, 5.3.4-4, -5; 5.3.5-3); 5.3.6 (e′s = ηe0 − h/2 + a′s); circles by 5.3.8 (α, αt with "< 0, taken 0", Nud by 5.3.8-1 and Mud = Nud ηe0 by 5.3.8-2) |
| `tension` | axial tension (5.4.1) and small eccentricity (5.4.2-1, -2, e = h/2 − e0 − as, e′ = h/2 + e0 − a′s) with fsd at most 330 MPa (table 3.2.3-1 note 1); large eccentricity (5.4.2-3, -4), x < 2a′s by 5.2.4-2 with Md → Nd e′s and the section without its compression bars |
| `torsion` | 5.5.3-1, -2; ζ (5.5.1-2, its cap); Tu (5.5.1-1); with shear: βt (5.5.4-3, its limits), Vu with all the stirrups (5.5.4-1), the stirrup ratio shear needs spread over the legs and taken off the outer leg (5.5.6), Tu = βt(0.35βa ftd + 0.05Np0/A0)Wt + 1.2√ζ fsv Asv1/sv Acor (5.5.4-2) |
| `crack_width` | σss by the member's nature (6.4.4-1 axial tension, -2 flexural, -3 eccentric tension, -4 … -8 eccentric compression with ηs, es, γ′f, z; class B 6.4.4-11, -12 with Ap + As and Ep), the temperature self-stress, C2 (Ml/Ms or Nl/Ns), Ate = 2asb (at most A), ρte (6.4.5-1) with its limits, c with its 50 mm cap, Wcr (6.4.3) |
| `crack_resistance` | both edges (W0, σst, σlt, σpc); σst = ∓Ms/W0 + ψσt signed by the edge (6.3.2-1); class A 6.3.1-3 and both edges of 6.3.1-4; full prestress 6.3.1-1, -2; class B σG = ∓MG/W0 against σpc |
| `inclined_crack_resistance`, `principal_compression` | the points; at the governing one σcx = σpc + My0/I0 (6.3.3-2), τ = VS0/(bI0) (6.3.3-5), σtp or σcp (6.3.3-1) against kftk (6.3.1-5 … -8) or 0.6fck (7.1.6-1); σtp ≤ 0.5ftk for detailing stirrups (7.1.6) |
| `service_stresses` | σkc = Mk y0/I0 (7.1.3-1) at both edges, σkc + σpt against 0.5fck (7.1.5-1); the tendon stress σpe + αEPσkt (7.1.3-2) against 0.65fpk (7.1.5-2) or 0.75fpk for threaded bars (7.1.5-4); class B cracked sections by 7.1.4 (eN, e0N, σcc, σp0 − σp) |
| `transient_pc`, `transient_rc` | the edge stresses at transfer against 0.70f′ck (7.2.8) and σtct against 1.15f′tk with the pretensioned-zone ratio; the cracked section of 7.2.4: σtcc = Mk x0/Icr ≤ 0.80f′ck, σts = αES Mk(h0i − x0)/Icr ≤ 0.75fsk, σttp ≤ f′tk (7.2.5) |
| `deflection_check` | f ≤ L/600 (L/300 for cantilevers), 6.5.3 |
| `local_bearing` | β (5.7.1-2), Fu1 (5.7.1-1), ρv (5.7.2-3, -4), Acor at most Ab, βcor (5.7.2-2), Fu2 (5.7.2-1); without indirect reinforcement 0.9ηsβfcdAln for information (program convention, not met) |
| `punching`, `footing_punching` | 5.6.1; with punching steel 5.6.2-2, -3 (concrete and steel parts), 5.6.2-1 and 5.6.1 on the perimeter outside the steel; 5.6.3-1 … -3 |
| bearings and joints (`devices`) | Rck/Ae ≤ σc (8.7.3-1); 2Δl, 1.43√(Δl² + Δt²) ≤ te; la/10 ≤ te ≤ la/5 with both sides; δc,m (8.7.3-8) against 0.07te and θla/2 (8.7.3-9); ts (8.7.3-10); 1.4GeAgΔl/te (+ Fbk) ≤ μRGk or μRck (8.7.4); μfR ≤ GeAg tanα (8.7.5); C⁺, C⁻, C⁺ + C⁻ ≤ C (8.8.2-5 … -7) |
| detailing reports (`Check.report`) | the rules (value, limit, met or not) and the other values |

### Language

`lang="zh"` (default: the report's) or `"en"`: the words of the report. In Chinese the check names, the program's messages
(failed conditions, warnings, notes, "skipped" lists), the rules of the detailing reports and strut-and-tie models, the
names of `Check.details` and their text values are translated (`femotion.codes.jtg._report_text`: tables of patterns,
the first match wins); the tests check that no English word is left.

## Tests

`tests/test_report_templates.py`, on the cases of `tests/report_cases.py` (one per branch of every template, the cases of
the independent review included): every check is written and read back from the HTML (`data-sym`, as printed, rounded).
The template must have written it (its guards ran); a symbol printed twice must read the same; the code's formula,
written out once more in the test, must recompute from the printed values the result printed and the program's result
to the rounding of the printed values; every inequality on the page (a comparison row against the row above, two numbers
on one row, the verdict line) must hold for the numbers it compares; the verdict box must state the condition not met.
Independent references: results computed in the test from the code's formulas with the values of its tables typed in
(one or more per kind of template). No English word in the Chinese names, messages and tables of values.

`tests/test_report.py`: the document layer; every literal check name in the source of `femotion.codes.jtg` is written
(the checks of all the examples and the catalogue), in Chinese and in English, and the English report has no Chinese
character; the summary counts the checks, those not met and the warnings; every check of the example report is written
with its formulas, they recompute its results, and its inequalities hold. The HTML parses, its ids are unique, every
link and reference resolves, nothing is loaded from outside; figures and tables are numbered consecutively in each
chapter. The parser of PDF destinations is tested on a fixed PDF; with Chrome present, the two-pass print puts the
right page numbers into the contents.
