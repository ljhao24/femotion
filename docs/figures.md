# Figures: drawings and charts as plain SVG

femotion draws what a design calculation report (计算书) needs as plain SVG, with nothing beyond the standard library
and NumPy:

| Module | Draws |
|---|---|
| `femotion.codes.jtg.figures` | a reinforced or prestressed concrete section to scale with its bars, stirrups, ducts and tendons, and the result of a check laid over it (`section_figure`) |
| `femotion.codes.jtg.charts` | charts along a girder: moment envelope against the resisting moment diagram, shear envelope against the stirrup zones with the inclined sections on the girder's elevation, utilisation, edge stresses, deflection and crack width against their limits |
| `femotion.frame.figures` | a frame model in elevation: its loads by load case, and the worst placement of a moving load over the section's influence line |
| `femotion.viz` | the drawing layer all three are built on, public for drawings of your own |

Every drawing is a `femotion.viz.Figure` (or a `Stack` of them), drawn in one consistent style: the same title, legend,
text block, font sizes, line weights and colours.

```python
from femotion import viz
from femotion.codes import jtg
from femotion.codes.jtg import charts
from femotion.codes.jtg.figures import section_figure

sec = jtg.tee(300, 1300, 1800, 150, concrete="C40", bottom=[(4, 28, 55), (4, 28, 115)], top=[(8, 16, 45)])
chk = jtg.flexure(sec, 1700.0)
fig = section_figure(chk.section, check=chk, stirrup_d=10)   # chk.section is sec
fig.save("girder.svg")                                       # or .html; fig.svg() gives the text
page = viz.html(["<h2>Flexure</h2>", fig, viz.stack([elevation, shear_chart])], title="report")
```

## Viewing

- **Jupyter**: a `Figure` and a `Stack` show themselves (`_repr_svg_`); so does an `RCSection` (its drawing without a
  check) and every `Check` (`_repr_html_`: name, clause, verdict in green or red, γ0Sd, R and ratio or the value
  against its limit, what failed, the warnings and notes, and the section drawn with the check laid over it).
- **SVG file**: `fig.save("x.svg")`, or `fig.svg()` for the text. The file is self-contained (its own `<style>`), so it
  opens in a browser, Inkscape or a word processor.
- **HTML page**: `fig.save("x.html")` for one figure; `viz.html(parts, title=…, css=…)` for a page of figures, stacks
  and HTML strings in order. The SVGs scale down with the page (`max-width: 100%`).

## Program conventions

- **Coordinates**: shapes are given in world units (mm for sections, m along a girder), y up; text, arrow heads, dots,
  dimension offsets and the like are in page pixels. Geometry is written to five decimals (areas read back from the
  SVG hold to better than 1e-6 relative), text positions to two.
- **The page is solved, not guessed.** A drawing of free height (`equal=True`, no `height`) gets the largest scale at
  which every label, dimension and arrow fits across the page width; the left margin grows for what sticks out on the
  left; then the top and bottom margins are set exactly, so no empty band is left under the drawing. A chart
  (`equal=False`) keeps its plot box `plot_height` px high and its left and right margins grow until the tick labels,
  the axis title and every label outside the box fit. With a fixed `height` the margins grow to fit inside it.
- **Text blocks flow round the drawing.** `legend(at="above")` (charts) and `legend(at="below")` (drawings) and
  `notes()` are laid out over and under the drawing's extent and wrap at the page width; a chart's blocks align with
  the plot box, a drawing's with the page's left gutter (12 px).
- **Labels never overlap.** Labels on a drawing are placed by a `Placer`: each label tries its candidate positions in
  order, higher priority first, and takes the first one clear of the labels already placed and of the marked points;
  a label with no free place is left out (or put at its first place when required). They are laid out again for every
  page width, so a narrow page drops what no longer fits. Labels over lines carry a white halo.
- **Text**: CJK, full-width and Latin text are measured (CJK one em, others about 0.58 em) to keep everything on the
  page. `x_{sub}` in any string is written as a subscript (`q_{k}`, `M_{max}`, `x_{c}`). Numbers in labels use a real
  minus sign (−) and drop trailing zeros (`viz.number`). Wrapping breaks CJK text anywhere and other words at spaces,
  but keeps a number with its unit, `a = b` and `→ OK` whole, never starts a line with closing punctuation, never ends
  one with an opening bracket and never splits a subscript.
- **Style**: one palette (`viz.COLORS`) and one set of CSS classes (`viz.STYLE`), every class prefixed `fm-` so a
  figure can sit in any page. Font sizes: title 15 px semibold; labels and the text block 12 px; small text, ticks and
  dimensions 10.5 px (`viz.TEXT_SIZE`). Colour meanings: demand red, resistance blue, compression orange, tension and
  anything not met red, met green, loads brown.
- **Text block tones** (`viz.Note`): plain; `head` semibold (a check's name and clause); `ok` green (a verdict met);
  `fail` red (a verdict not met); `warn` red (a remark); `small` 10.5 px muted; a number puts a numbered callout
  before the line (the same callout as on the drawing).
- **Machine-readable**: drawn elements carry `data-role` and other `data-*` attributes with the exact values (the bars,
  the compression zone, the stress diagram's values and scale, the loads and influence ordinates), and the charts tag
  their elements with `ch-…` classes, for tests and scripts that read the SVG back.
- **Language**: `lang="zh"` (default) or `"en"` for every drawing function.

## `femotion.viz`

### Figure

`Figure(width=560, height=None, *, equal=True, plot_height=None, margin=(12, 12, 12, 12), title=None, pad=0.0, css="")`

- `equal=True`: one scale for both axes; `equal=False` (charts) needs `plot_height` (the plot box in px) or `height`.
- `set_box(x0, y0, x1, y1)`: the world box shown (a chart's axis ranges) instead of the content's extent.
- Shapes in world coordinates: `polyline`, `line`, `polygon`, `path` (rings with holes, even-odd), `rect`, `circle`
  (world radius), `dot` (px radius), `text` (px offsets, rotation, halo), `arrow`, `force` (an arrow of fixed length
  whose head touches a point, with its label), `dim` (an aligned dimension), `vlines` (verticals across a chart's box),
  `raw(svg, pts, extent, clip=, layer=)` for anything else: `svg(T)` gets the page map `T(x, y)` with `T.sx`, `T.sy`,
  `T.box`, `T.W`, `T.H`; `extent(T)` returns the page boxes the item needs, which the page solver keeps on the page.
- `label(p, text, cls="small", *, dx, dy, anchor, alt=[(p, dx, dy, anchor), …], priority, room, inside, required)` and
  `mark(p, r_px)`: labels placed clear of each other and of marked points, drawn over everything.
- `legend(entries, at="above")`: entries `(class, label[, kind])`, kind `"line"`, `"area"`, `"dot"`, `"ring"` or a
  function `(x, y) -> svg` for a 20 px symbol; by default from the class. `at`: `"above"`, `"below"`, or inside the
  plot box (`"top-right"`, `"top-left"`, `"bottom-right"`, `"bottom-left"`) or `"right"` of it.
- `notes(lines, at="below", align=None)`: lines as strings, `Note`s or `(text, tone[, number])`.
- `axes(xlabel, ylabel, xticks, yticks, grid, zero, xfmt, yfmt)`: frame, round ticks (`nice_ticks`), grid and axis
  titles of a chart, drawn under everything.
- `svg()`, `save(path)` (.svg or .html), `_repr_svg_()`, `to_page(x, y)`.

`Stack(figures, gap=4, width=None, title=None)` / `stack(…)`: charts on one x range one above the other, their plot
boxes aligned (same left and right edges): e.g. a girder's elevation over its shear chart. `html(parts, title, css)`:
a page.

### Building blocks (page px)

| Function | Does |
|---|---|
| `svg_text(x, y, s, cls, anchor, baseline, rotate, halo, fill, weight, attrs)` | a `<text>`, `x_{sub}` as subscripts |
| `text_width(s, size)`, `text_box(x, y, s, size, anchor, baseline, rotate)`, `text_item(…)` | the space a label takes; `text_item` returns (svg, box) |
| `wrap(s, width, size=10.5, at=None, indent=0)` | lines of a text (see the conventions); `at` a regular expression of the places a formula may break |
| `svg_line`, `svg_poly`, `svg_circle`, `svg_path` | elements with `fm-` classes, optional style and attributes |
| `svg_arrow(x0, y0, x1, y1, head=7)` | an arrow with a filled head at (x1, y1) |
| `moment_arc(x, y, ccw, r=12)` | a curved arrow round a point (a moment) |
| `hatch(x0, x1, y, below=True)`, `support_symbol(kind, x, y, outward, vertical_member)` | ground hatching; fixed, pin, roller and guide supports, returning (svg, box) |
| `dim_chain(pos, at, labels, vertical=False, ext=None, side=-1, short="beyond")` | a chain of dimensions with 45° ticks and extension lines; labels too long for their segment dropped (`short="drop"`) or put past the end |
| `callout(x, y, n)` | a numbered callout (number in a circle) |
| `spread(ys, gap, top, bottom)`, `leader_column(items, x, gap, top, bottom)` | labels in one column with leaders to their anchors, kept in order and apart, pulled up when they run past the bottom |
| `Placer(bounds)` with `block`, `mark`, `add(text, cands, cls, priority, required, inside, room)`, `layout`, `place`, `svg` | greedy label placement in page px (what `Figure.label` uses) |
| `overlap(a, b, pad)`, `shifted(cands, step, n)` | box test; candidate positions repeated `step` px further out |
| `split(x, y)`, `beyond(x, y, level, above)`, `nice_ticks(lo, hi, n)` | polylines broken at NaN or inf; the parts of a curve beyond a limit with the crossing points; round ticks |
| `number(v, digits)`, `fmt(v, nd)`, `plain(s)` | a label's number; a coordinate; the characters of a rich text |

Constants: `COLORS`, `STYLE`, `TEXT_SIZE`, `FONT`, `PRECISION`, `GUTTER`, `EDGE`, `TITLE_ROW`, `BLOCK_GAP`.

## Section drawings: `femotion.codes.jtg.figures`

`section_figure(sec, *, check=None, lang="zh", width=560, title=None, hogging=None, stirrup_d=None, side_cover=None,
webs=None, bar_x=None, tendon_x=None, external_n=None, strand=(15.2, 140.0), dims=True, labels=True, legend=True,
info=True)`

- `sec`: an `RCSection` or a `Composite` (drawn as its ultimate section, the cast-in-place topping in its own colour).
- `check`: a `Check` or a list of them, or the dict of `cracked_section`. Flexure: the compression zone cut out of the
  section's shape to depth x (at the bottom for hogging), the neutral axis at x/β, h0 and ξb h0. Eccentric compression
  and tension: the zone and the load at e. Stresses in service, crack resistance and cracked sections: the stress
  diagram beside the section (under it on a narrow page). `section_detailing`: what a rule is about ringed and
  numbered, the clause in the text. Any other check: its result in the text block.
- The compression face follows `Check.hogging`; `hogging=` overrides it for eccentric compression and gives it for
  `cracked_section`'s dict.
- `stirrup_d` draws the stirrups and takes them into the bar layout as `section_detailing` does; `bar_x`, `tendon_x`
  place rows and ducts by hand; `bar_layout(sec, …)` returns the layout drawn.

What is drawn where (the section stores no horizontal positions) is written in the module's docstring: bars spread
over the concrete at their depth between the stirrups, shared over the webs of a box or the walls of a hollow pier,
ducts spread evenly, external tendons in the cell, bars on the bar circle of a circular section.

### Checks carry their section

Every check made on a section keeps it: `Check.section` is the `RCSection` (or `Composite`) passed in — never the copy
turned over for a hogging moment — and `Check.hogging` is True when the check turned it over (a hogging moment, the
tension face on top, the bottom face in compression). Neither takes part in comparisons or the repr. Checks without a
section (single detailing rules, bearings, devices, strut-and-tie, punching) have `section=None`. So a check can be
drawn without repeating its inputs, `section_figure(chk.section, check=chk)`, and shows itself in Jupyter with its
drawing. `Check.report(…, section=, hogging=)` builds a detailing report with them.

## Charts along a girder: `femotion.codes.jtg.charts`

| Function | Chart |
|---|---|
| `moment_chart(x, Mmax, Mmin, *, gamma0, resistance, shaved, others, supports, tension_side=True, critical="ratio", …)` | γ0 Md envelope (on the tension side, sagging below the axis), resisting moment steps, the envelope before peak shaving, the governing section marked; `fig.critical`, `fig.utilization` |
| `shear_chart(x, Vmax, Vmin, *, gamma0, resistance, search, supports, …)` | γ0 Vd envelope, Vu of the stirrup zones, the governing inclined section and the 5.2.11 limits at the end supports; `fig.governing`, `fig.limits` |
| `beam_elevation(x0, x1, section, *, supports, search, stirrups, bent, depth=34, grid=True, …)` | the girder's elevation with the inclined sections of `jtg.shear_search`, bent-up bars and the stirrup zones; stack it over the shear chart |
| `utilization_chart(series, *, limit=1.0, …)` | utilisation of several checks along the girder, beyond the limit red; `fig.maxima` |
| `stress_chart(x, *, top, bottom, limits, compression_positive=True, …)` | edge stresses (envelopes as bands) against their limits; `fig.extremes`, `fig.exceeded` |
| `deflection_chart(x, f, *, limit, …)`, `crack_chart(x, w, *, limit, …)` | deflection (drawn down) and crack width against their limits |

Common arguments: `supports` (positions, or rows `(x, kind[, diaphragm width])` as `jtg.shear_search`), `xlim`,
`lang`, `width` (px), `height` (the plot box in px), `title`, `xlabel`, `legend`, `notes`. Envelopes of the basic
combination are passed without γ0 (the chart applies it and says so). Resistances along the girder are `Profile`s
(steps or sampled curves): `flexural_resistance(zones)`, `FlexuralResistance.from_arrays`, `shear_resistance(…)`;
`flexure_utilization` and `shear_utilization` give the ratios for `utilization_chart`.

## Load drawings: `femotion.frame.figures`

| Function | Draws |
|---|---|
| `elevation_figure(m, *, members=None, nodes=False, dims=True, …)` | members, supports (from the restrained degrees of freedom: fixed, pin, roller, guide) and span dimensions |
| `load_figure(m, case, …)` | the loads of a `LoadCase`: point loads as arrows touching the member, distributed loads as bands (stacked where they overlap), nodal moments as curved arrows, imposed strains as a bracket, tendons as their profile (eccentricity exaggerated and noted), settlements under the support |
| `influence_figure(il, layout, *, ordinates=True, …)` | the worst placement of a lane load (`LaneLayout`) or a vehicle (`AxleLayout`) above, the section's influence line below on the same x axis, the loaded parts shaded, the ordinates under the loads, and the effect written out (S = factor × (P η + q Ω) or factor × Σ Pi ηi) |

The elevation is the vertical plane through the model's main horizontal direction; lengths are to scale along the
bridge, arrows and bands in pixels.

## Examples

- `examples/jtg_figures_section.py`: seven sections with their checks (T girder sagging and hogging, post-tensioned I
  girder with stresses in service, box girder with external tendons, composite girder, circular pier in eccentric
  compression, detailing not met) on one HTML page.
- `examples/jtg_charts.py`: the charts of a two-span reinforced concrete girder and the edge stresses of a
  post-tensioned one (`--lang en`, `--width 380`).
- `examples/jtg_loads_figures.py`: load cases and influence-line placements of the girder of `jtg_rc_girder.py`.
