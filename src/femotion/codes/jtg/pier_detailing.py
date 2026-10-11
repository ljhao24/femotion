"""Detailing of columns, piers, cap beams, tie beams, joints and pile caps, JTG 3362-2018 9.6.1–9.6.10.

Every function returns a detailing report (Check.report, `ok` always true) whose `warnings` list the rules not met
("should" marks the code's 宜); a rule whose input is not given is skipped and named in the notes. Inputs that the
section knows (bar diameters, the steel ratio, the least dimension, the spacing of bars on a circle) are read from it.
The anchorage length la of table 9.1.4 is an input (9.1.4 is not repeated here). Units: mm, mm², MPa.
Clauses: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    from femotion.codes.jtg import pier_detailing as pd
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    pd.column_detailing(col, ties=(12, 200), closed=True, embedment=1200, la=980)
    pd.cap_beam_detailing(6000, 1600, stirrup=(16, 100), side=(16, 150))
    pd.pile_cap_detailing(h=2000, D=1200, spacing=3600, tie_d=16, hanger=(16, 150))
"""
from __future__ import annotations

import math

import numpy as np

from .members import spiral_rules
from .rc import Check, RCSection
from .tendon_detailing import _grade, _Rules

COLUMN_BAR_D_MIN = 12.0  # longitudinal bars (9.6.1 item 1)
COLUMN_CLEAR = (50.0, 350.0)  # clear spacing of longitudinal bars (9.6.1 item 1)
COLUMN_RHO_MAX = 0.05  # all longitudinal bars / section (should)
TIE_D_MIN = 8.0  # ties: ≥ 8 mm and ≥ d/4 (9.6.1 item 3)
TIE_SPACING = (15, 400.0)  # ≤ 15 d, the least dimension (circle 0.8 D), 400 mm (9.6.1 item 4)
TIE_DENSE = (0.03, 10, 200.0)  # more than 3 % longitudinal steel: ≤ 10 d and 200 mm
TIE_REACH = (150.0, 15)  # bars within max(150 mm, 15 tie diameters) of a corner bar, else cross-ties (9.6.1 item 5)
LAP_TIE_SPACING = {"tension": (5, 100.0), "compression": (10, 200.0)}  # within lap splices, by 9.3.12 item 3
SIDE_BAR_DEPTH = 600.0  # eccentric members of h ≥ 600 mm: side bars 10–16 mm (9.6.3)
SIDE_BAR_D_MIN = 10.0
WALL_MESH_MIN = 250.0  # thin-walled piers, ribbed abutments: ≥ 250 mm²/m each way (9.6.4)
WALL_MESH_SPACING_MAX = 400.0


def _d_range(sec: RCSection | None, d):
    """(smallest, largest) longitudinal bar diameter: from d (one value or a pair) or the section"""
    if d is not None:
        return (float(d), float(d)) if np.isscalar(d) else (float(min(d)), float(max(d)))
    if sec is not None and np.isfinite(sec.bar_d).any():
        return float(np.nanmin(sec.bar_d)), float(np.nanmax(sec.bar_d))
    return None, None


def _least_dimension(sec: RCSection | None):
    """the least dimension for the tie spacing: rectangle min(b, h), circle 0.8 D (9.6.1 item 4); other shapes None"""
    if sec is None:
        return None
    if sec.kind in ("circle", "annulus"):
        return 0.8 * sec.h
    if sec.kind == "rect":
        return min(sec.pieces[0].b, sec.h)
    return None


def _circle_clear(sec: RCSection, d_max: float):
    """clear distance between neighbouring bars on a circle: the chord 2 rs sin(π/n) less one bar diameter"""
    n = len(sec.bars)
    return 2 * sec.ring * math.sin(math.pi / n) - d_max if n > 1 else None


def column_detailing(sec: RCSection | None = None, *, d=None, clear=None, rho: float | None = None,
                     least: float | None = None, ties=None, closed: bool | None = None, lap=None,
                     unrestrained: float | None = None, embedment: float | None = None, la: float | None = None,
                     spiral=None, dcor: float | None = None, eccentric: bool = False, side_d: float | None = None,
                     horizontal_cast: bool = False) -> Check:
    """columns and piers (9.6.1, 9.6.2, 9.6.3); not for driven or bored piles (9.6.1 excludes them: JTG 3363). From sec:
    the bar diameters (d: one value or (smallest, largest) otherwise), the total steel ratio (rho), the least dimension
    (rectangle min(b, h), circle 0.8 D; least otherwise) and, on a circle, the clear spacing (the chord less one bar).
    9.6.1: bars ≥ 12 mm; clear spacing 50–350 mm (clear: one value or (smallest, largest); horizontally cast precast
    members: the minimum by 9.3.3, not checked here); all bars ≤ 5 % (should); ties = (diameter, spacing), closed,
    diameter ≥ 8 mm and ≥ 1/4 of the largest bar; spacing ≤ 15 d, the least dimension and 400 mm, with more than 3 %
    steel ≤ 10 d and 200 mm (d: the smallest bar, as GB 50010-2010 9.3.2 reads it, program convention); within lap
    splices (lap = ("tension" or "compression", tie spacing)) ≤ 5 d and 100 mm, or 10 d and 200 mm (9.3.12 item 3);
    unrestrained: the largest distance of a bar from a corner bar (or a bar held by a cross-tie), ≤ max(150 mm,
    15 tie diameters), else cross-ties (item 5; not for circles, whose hoops hold every bar); embedment of the bars
    into the foundation and the cap beam ≥ la (item 2).
    spiral = (diameter, pitch) and dcor (default for jtg.circle: the bar circle plus one bar): 9.6.2 items 1, 2, 4
    (members.spiral_rules, as jtg.axial_spiral) and item 3, embedment ≥ the column's diameter and ≥ la; the tie rules of
    9.6.1 are then not applied. eccentric=True (9.6.3): h ≥ 600 mm needs side bars of 10–16 mm (side bars: those of sec
    outside the edge rows, or side_d). The minimum reinforcement of 9.1.12: jtg.axial, jtg.eccentric."""
    R = _Rules("column detailing (9.6.1–9.6.3)", "JTG 3362-2018 9.6.1–9.6.3")
    d_min, d_max = _d_range(sec, d)
    if rho is None and sec is not None:
        rho = float(sec.bars[:, 0].sum() / sec.area)
    least = least if least is not None else _least_dimension(sec)
    R.chk.details.update(d_min=d_min, d_max=d_max, rho=rho, least_dimension=least)
    R.min("longitudinal bar diameter", d_min, COLUMN_BAR_D_MIN, "9.6.1 item 1")
    if clear is None and sec is not None and sec.kind in ("circle", "annulus") and sec.ring and d_max is not None:
        clear = _circle_clear(sec, d_max)
    c_lo, c_hi = (None, None) if clear is None else ((clear, clear) if np.isscalar(clear) else (min(clear), max(clear)))
    if horizontal_cast:
        R.note("horizontally cast precast member: the least clear spacing by 9.3.3 (not checked here)")
    else:
        R.min("least clear spacing of longitudinal bars", c_lo, COLUMN_CLEAR[0], "9.6.1 item 1")
    R.max("largest clear spacing of longitudinal bars", c_hi, COLUMN_CLEAR[1], "9.6.1 item 1")
    R.max("longitudinal steel ratio", rho, COLUMN_RHO_MAX, "9.6.1 item 1", pct=True, should=True)
    if spiral is not None:  # 9.6.2 in place of the tie rules and the embedment of 9.6.1
        ds, s = spiral
        if dcor is None and sec is not None and sec.ring:
            dcor = 2 * sec.ring + (d_max or 0.0)
        if sec is None or dcor is None:
            R.skip("spiral column rules (give sec and dcor)", "9.6.2 items 1, 2, 4")
        else:
            Acor = math.pi * dcor ** 2 / 4
            R.chk.details.update(dcor=dcor, Acor=Acor)
            R.chk.warnings += spiral_rules(float(sec.bars[:, 0].sum()), Acor, sec.area, ds, s, dcor, d_max or 0.0)
        D = sec.h if sec is not None and sec.kind == "circle" else None
        R.min("embedment of the bars (≥ the column's diameter)", embedment, D, "9.6.2 item 3")
        R.min("embedment of the bars (≥ la)", embedment, la, "9.6.2 item 3")
    else:
        R.min("embedment into the foundation / cap beam (≥ la)", embedment, la, "9.6.1 item 2")
        R.flag("closed ties", closed, "ties not closed", "9.6.1 item 3")
        if ties is None:
            R.skip("ties (give ties = (diameter, spacing))", "9.6.1 items 3, 4")
        else:
            dt, st = ties
            R.min("tie diameter", dt, max(TIE_D_MIN, (d_max or 0.0) / 4), "9.6.1 item 3")
            lim = [TIE_SPACING[1]] + ([TIE_SPACING[0] * d_min] if d_min else []) + ([least] if least else [])
            if rho is not None and rho > TIE_DENSE[0]:
                lim += [TIE_DENSE[2]] + ([TIE_DENSE[1] * d_min] if d_min else [])
            R.max("tie spacing", st, min(lim), "9.6.1 item 4")
            if sec is None or sec.kind not in ("circle", "annulus"):  # a circular hoop holds every bar
                R.max("distance of a bar from a restrained corner bar", unrestrained,
                      max(TIE_REACH[0], TIE_REACH[1] * dt), "9.6.1 item 5")
        if lap is not None:
            kind, sl = lap
            k, cap = LAP_TIE_SPACING[kind]
            R.max(f"tie spacing within {kind} lap splices", sl, min(cap, k * d_min) if d_min else cap,
                  "9.6.1 item 4, 9.3.12 item 3")
    if eccentric:
        if sec is None:
            R.skip("side bars (give sec)", "9.6.3")
        elif sec.kind in ("circle", "annulus"):
            R.note("9.6.3 side bars: the bars go all round a circle")
        elif sec.h >= SIDE_BAR_DEPTH:
            if side_d is None:
                bot, top = sec.edge_masks()
                side = ~(bot | top)
                side_d = 0.0 if not side.any() else (float(np.nanmin(sec.bar_d[side]))
                                                      if np.isfinite(sec.bar_d[side]).any() else None)
            R.flag("side bars", None if side_d is None else side_d > 0,
                   f"eccentric member of h = {sec.h:.0f} mm ≥ 600 mm without side bars", "9.6.3")
            if side_d:
                R.min("side bar diameter", side_d, SIDE_BAR_D_MIN, "9.6.3")
            R.note("9.6.3: compound ties where needed (not checked)")
    return R.report()


def wall_mesh_detailing(*, As_h: float | None = None, As_v: float | None = None,
                        spacing: float | None = None) -> Check:
    """surface mesh of thin-walled piers and of the back walls and ribs of ribbed abutments (9.6.4): ≥ 250 mm²/m
    horizontally and vertically (the main bars included), spacing ≤ 400 mm (commentary: AASHTO's 264 mm²/m; 250 mm²/m
    is five 8 mm bars a metre)"""
    R = _Rules("wall surface mesh (9.6.4)", "JTG 3362-2018 9.6.4")
    R.min("horizontal mesh", As_h, WALL_MESH_MIN, "9.6.4", unit="mm²/m")
    R.min("vertical mesh", As_v, WALL_MESH_MIN, "9.6.4", unit="mm²/m")
    R.max("mesh spacing", spacing, WALL_MESH_SPACING_MAX, "9.6.4")
    return R.report()


CAP_CONCRETE_MIN = 25  # cap beams of l/h ≤ 5 (9.6.5)
CAP_STIRRUP = (8.0, 200.0)  # diameter ≥ 8 mm, spacing ≤ 200 mm (should)
CAP_SIDE_BARS = (12.0, 200.0)  # horizontal bars on both faces: ≥ 12 mm, ≤ 200 mm (both should)
TIE_BEAM_SIZE = ((0.8, 1.0), (0.6, 0.8))  # depth, width / column diameter or longer side ("may")
TIE_BEAM_CORNER_D = 16.0
TIE_BEAM_TIE = (8.0, 400.0)  # ties ≥ 8 mm at ≤ the shorter side and 400 mm


def cap_beam_detailing(l: float | None = None, h: float | None = None, *, concrete=None, stirrup=None,
                       side=None) -> Check:
    """cap beams (9.6.5, first paragraph): with l/h ≤ 5 a higher concrete class, not below C25; stirrups = (diameter,
    spacing): ≥ 8 mm, ≤ 200 mm (should); side = (diameter, spacing) of the horizontal bars on both side faces: ≥ 12 mm,
    ≤ 200 mm (should)"""
    R = _Rules("cap beam detailing (9.6.5)", "JTG 3362-2018 9.6.5")
    if l is not None and h is not None:
        R.chk.details["l/h"] = l / h
        if l / h <= 5.0:
            R.min("concrete class", None if concrete is None else _grade(concrete), CAP_CONCRETE_MIN, "9.6.5",
                  unit="MPa")
            R.note("l/h ≤ 5: a higher concrete class is advised (9.6.5)")
    if stirrup is None:
        R.skip("stirrups", "9.6.5")
    else:
        R.min("stirrup diameter", stirrup[0], CAP_STIRRUP[0], "9.6.5")
        R.max("stirrup spacing", stirrup[1], CAP_STIRRUP[1], "9.6.5", should=True)
    if side is None:
        R.skip("side bars", "9.6.5")
    else:
        R.min("side bar diameter", side[0], CAP_SIDE_BARS[0], "9.6.5", should=True)
        R.max("side bar spacing", side[1], CAP_SIDE_BARS[1], "9.6.5", should=True)
    return R.report()


def tie_beam_detailing(D: float, *, h: float | None = None, b: float | None = None, corner_d: float | None = None,
                       tie=None) -> Check:
    """tie beams between the columns of a bent (9.6.5, second paragraph). D: column diameter or longer side; depth
    0.8–1.0 D and width 0.6–0.8 D ("may": notes); four corner bars ≥ 16 mm; ties = (diameter, spacing): ≥ 8 mm, spacing
    ≤ the beam's shorter side and 400 mm"""
    R = _Rules("tie beam detailing (9.6.5)", "JTG 3362-2018 9.6.5")
    for lab, v, (lo, hi) in (("depth", h, TIE_BEAM_SIZE[0]), ("width", b, TIE_BEAM_SIZE[1])):
        if v is not None:
            R.chk.details[f"{lab} / D"] = v / D
            if not lo - 1e-9 <= v / D <= hi + 1e-9:
                R.note(f"tie beam {lab} {v / D:.2f} D: 9.6.5 suggests {lo}–{hi} D")
    R.min("corner bar diameter", corner_d, TIE_BEAM_CORNER_D, "9.6.5")
    if tie is None:
        R.skip("ties", "9.6.5")
    else:
        R.min("tie diameter", tie[0], TIE_BEAM_TIE[0], "9.6.5")
        lim = min([TIE_BEAM_TIE[1]] + [v for v in (h, b) if v is not None])
        R.max("tie spacing", tie[1], lim, "9.6.5")
    return R.report()


def beam_bar_anchorage(d: float, *, la: float | None = None, joint: str = "end", bar: str = "top",
                       strength: str = "tension", ribbed: bool = True, through: bool | None = None,
                       straight: float | None = None, beyond_centre: float | None = None, hook=None) -> Check:
    """anchorage of the longitudinal bars of a cap beam or tie beam in a joint with a column (9.6.7). d: bar diameter;
    la: anchorage length of table 9.1.4 (tension or compression, as the bar works); joint: "end" (the column at the end
    of the beam) or "interior"; bar: "top" or "bottom"; strength: "tension", "compression" (the bar's strength used) or
    "none". straight: straight embedment; beyond_centre: how far it runs past the column centre line;
    hook = (horizontal, vertical) projections of a 90° hook, the bend included.
    End joints, top bars, and bottom bars whose tensile strength is used (item 1): straight ≥ la and past the centre
    line by ≥ 5 d (should); or a 90° hook bent down inside the column's outer bars, horizontal ≥ 0.4 la,
    vertical ≥ 15 d.
    Interior joints (item 2): top bars run through (through=False: a warning); bottom bars preferably through, when
    anchored: strength not used ≥ 12 d (ribbed) or 15 d (plain); used ≥ la; a short column a 90° hook as item 1
    (should). End-joint bottom bars whose strength is not used, or only in compression, follow the interior rule."""
    R = _Rules("beam bars in a joint (9.6.7)", "JTG 3362-2018 9.6.7")
    if joint not in ("end", "interior") or bar not in ("top", "bottom") or \
            strength not in ("tension", "compression", "none"):
        raise ValueError("joint: 'end' or 'interior'; bar: 'top' or 'bottom'; strength: 'tension', 'compression', "
                         "'none'")
    item1 = joint == "end" and (bar == "top" or strength == "tension")
    if joint == "interior":
        if bar == "top":
            R.flag("top bars through the joint", through, "top beam bars anchored in an interior joint: 9.6.7 item 2 "
                   "asks for them to run through", "9.6.7 item 2")
            return R.report()
        if through:
            R.note("bottom bars run through the joint (9.6.7 item 2)")
            return R.report()
    if item1:
        cl = "9.6.7 item 1"
        if hook is not None:
            R.min("hook: horizontal projection (≥ 0.4 la)", hook[0], None if la is None else 0.4 * la, cl)
            R.min("hook: vertical projection (≥ 15 d)", hook[1], 15 * d, cl)
            R.note("a 90° hook runs to the inside of the column's outer bars and bends into the joint (9.6.7 item 1)")
        else:
            R.min("straight anchorage (≥ la)", straight, la, cl)
            R.min("past the column centre line (≥ 5 d)", beyond_centre, 5 * d, cl, should=True)
    else:
        cl = "9.6.7 item 2"
        if hook is not None:
            R.min("hook: horizontal projection (≥ 0.4 la)", hook[0], None if la is None else 0.4 * la, cl, should=True)
            R.min("hook: vertical projection (≥ 15 d)", hook[1], 15 * d, cl, should=True)
        elif strength == "none":
            R.min(f"straight anchorage (≥ {12 if ribbed else 15} d)", straight, (12 if ribbed else 15) * d, cl)
        else:
            R.min("straight anchorage (≥ la)", straight, la, cl)
    return R.report()


def column_bar_anchorage(d: float, *, la: float | None = None, to_top: bool | None = None,
                         embedment: float | None = None, hook=None, splice_in_joint: bool | None = None) -> Check:
    """the column's longitudinal bars in a joint with a cap beam or tie beam (9.6.8): they run through tie-beam joints,
    splices outside the joint zone (item 1); into a cap beam they reach its top (to_top) with an anchorage length from
    the cap's soffit ≥ la (embedment, item 2), or, the cap being too shallow, a 90° hook running to the inside of the
    cap's top bars: vertical projection ≥ 0.4 la, horizontal ≥ 12 d (hook = (vertical, horizontal), the bend
    included). d: bar diameter; la: table 9.1.4."""
    R = _Rules("column bars in a joint (9.6.8)", "JTG 3362-2018 9.6.8")
    R.flag("splices outside the joint", None if splice_in_joint is None else not splice_in_joint,
           "column bars spliced inside the joint zone", "9.6.8 item 1")
    R.flag("bars up to the cap top", to_top, "column bars stop short of the cap beam's top", "9.6.8 item 2")
    if hook is not None:
        R.min("hook: vertical projection (≥ 0.4 la)", hook[0], None if la is None else 0.4 * la, "9.6.8 item 2")
        R.min("hook: horizontal projection (≥ 12 d)", hook[1], 12 * d, "9.6.8 item 2")
    else:
        R.min("anchorage from the cap soffit (≥ la)", embedment, la, "9.6.8 item 2")
    return R.report()


def ramp_piers(supports, *, restrainers: bool | None = None) -> Check:
    """piers of box-girder ramps (9.6.9): supports, the piers of one continuous unit in order: "multi" (several bearings
    across: multi-column, or a column with two bearings), "single" (one column, one bearing) or "fixed" (monolithic with
    the girder). Multi-bearing piers with the bearings far apart are preferred (item 1); a single-bearing pier only
    where the site forces it (a note per pier), never two in a row (item 2); restrainers and devices against unseating
    at transition piers and abutments (item 3, should). The overturning check itself: 4.1.8 (commentary)."""
    R = _Rules("box-girder ramp piers (9.6.9)", "JTG 3362-2018 9.6.9")
    s = [str(x).lower() for x in supports]
    bad = [x for x in s if x not in ("multi", "single", "fixed")]
    if bad:
        raise ValueError(f"supports: 'multi', 'single' or 'fixed', not {bad}")
    singles = [i for i, x in enumerate(s) if x == "single"]
    R.chk.details["single-bearing piers"] = singles
    if singles:
        R.note(f"single-column single-bearing piers {singles}: only where the site demands it (9.6.9 item 2); check "
               "the overturning of the girder (4.1.8)")
    runs = [i for i in singles if i + 1 in singles]
    R.flag("no consecutive single-bearing piers", not runs,
           f"consecutive single-column single-bearing piers at {runs} and the next", "9.6.9 item 2")
    R.flag("restrainers at transition piers and abutments", restrainers,
           "no restrainers or unseating devices at the transition piers and abutments", "9.6.9 item 3", should=True)
    return R.report()


PILE_CAP_DEPTH = (1.5, 1500.0)  # ≥ 1.5 D (should) and ≥ 1.5 m (9.6.10 item 1)
PILE_CAP_SPACING = 3.0  # pile spacing / D: main bars over the whole width up to it, else strips of ±1.5 D (item 2)
PILE_CAP_OUTSIDE_RATIO = 0.001  # bars outside the strips
PILE_CAP_CROSS = (12.0, 250.0)  # one-way main bars: bars across them ≥ 12 mm at ≤ 250 mm (item 3)
PILE_CAP_MESH = ((1200.0, 1500.0), (12.0, 16.0))  # mesh over the pile heads: mm²/m each way, bar diameters (item 4)
PILE_CAP_TIE_D = 16.0  # vertical ties (item 5)
PILE_CAP_HANGER = (12.0, 200.0)  # hangers between piles ≥ 3 D apart, within 1 D of each pile's centre (item 6)


def pile_cap_bar_zones(centres, D: float, B: float) -> list[tuple[float, float]]:
    """where the main bars of a pile cap go (9.6.10 item 2), across the bars' direction: piles at most 3 D apart (every
    neighbour), the whole width (0, B); farther apart, strips 1.5 D either side of each pile centre (overlapping strips
    merged, clipped to the cap). centres: pile centres measured from one cap edge (mm). Program convention for mixed
    spacings: strips. The bars outside the strips: ≥ 0.1 % (pile_cap_detailing)."""
    x = np.sort(np.asarray(centres, float))
    if len(x) < 2 or np.diff(x).max() <= PILE_CAP_SPACING * D + 1e-9:
        return [(0.0, float(B))]
    out = []
    for c in x:
        lo, hi = max(c - 1.5 * D, 0.0), min(c + 1.5 * D, B)
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((float(lo), float(hi)))
    return out


def pile_cap_detailing(*, h: float | None = None, D: float | None = None, spacing: float | None = None,
                       outside_ratio: float | None = None, one_way: bool | None = None, cross=None, mesh=None,
                       tie_d: float | None = None, hanger=None) -> Check:
    """pile caps (9.6.10, besides JTG D63). h: cap depth, ≥ 1.5 D (should) and ≥ 1.5 m (item 1); spacing: pile centre
    spacing; above 3 D the main bars lie within 1.5 D of the pile centres and the rest of the width gets ≥ 0.1 %
    (outside_ratio; program convention: on the width outside the strips × h0, as the 0.15 % of 8.5.4) (item 2;
    pile_cap_bar_zones); bars' clear spacing and layers by 9.3.3, cover by 9.1.1 (not here). one_way: main bars in one
    direction only, then cross = (diameter, spacing) across them ≥ 12 mm, ≤ 250 mm (item 3). mesh = (mm²/m one way,
    mm²/m the other, bar diameter): one layer over the pile heads, 1200–1500 mm²/m each way (should; more: a note),
    bars 12–16 mm (item 4). tie_d: vertical ties ≥ 16 mm (item 5). hanger = (diameter, spacing): with the piles ≥ 3 D
    apart, hangers between them (should), within 1 D of each pile's centre-to-centre middle part, ≥ 12 mm at ≤ 200 mm
    (item 6). Note the code's two thresholds: item 2 "more than 3 D", item 6 "3 D or more"."""
    R = _Rules("pile cap detailing (9.6.10)", "JTG 3362-2018 9.6.10")
    R.min("cap depth (≥ 1.5 m)", h, PILE_CAP_DEPTH[1], "9.6.10 item 1")
    R.min("cap depth (≥ 1.5 D)", h, None if D is None else PILE_CAP_DEPTH[0] * D, "9.6.10 item 1", should=True)
    if spacing is not None and D is not None:
        r = spacing / D
        R.chk.details["spacing / D"] = r
        if r > PILE_CAP_SPACING + 1e-9:
            R.min("bars outside the pile strips", outside_ratio, PILE_CAP_OUTSIDE_RATIO, "9.6.10 item 2", pct=True)
            R.note("piles more than 3 D apart: main bars within 1.5 D of the pile centres (9.6.10 item 2)")
        else:
            R.note("piles at most 3 D apart: main bars evenly over the whole width (9.6.10 item 2)")
        if r >= PILE_CAP_SPACING - 1e-9:
            if hanger is None:
                R.note("piles 3 D or more apart: hangers between the piles advised (9.6.10 item 6)")
            else:
                R.min("hanger diameter", hanger[0], PILE_CAP_HANGER[0], "9.6.10 item 6")
                R.max("hanger spacing", hanger[1], PILE_CAP_HANGER[1], "9.6.10 item 6")
    else:
        R.skip("bar layout by the pile spacing (give spacing and D)", "9.6.10 items 2, 6")
    if one_way:
        if cross is None:
            R.skip("bars across one-way main bars (give cross)", "9.6.10 item 3")
        else:
            R.min("cross bar diameter", cross[0], PILE_CAP_CROSS[0], "9.6.10 item 3")
            R.max("cross bar spacing", cross[1], PILE_CAP_CROSS[1], "9.6.10 item 3")
    if mesh is None:
        R.skip("mesh over the pile heads", "9.6.10 item 4")
    else:
        (lo, hi), (dlo, dhi) = PILE_CAP_MESH
        for k, a in enumerate(mesh[:2]):
            R.min(f"mesh over the pile heads, direction {k + 1}", a, lo, "9.6.10 item 4", unit="mm²/m", should=True)
            if a > hi:
                R.note(f"mesh direction {k + 1}: {a:.0f} mm²/m, 9.6.10 item 4 suggests 1200–1500 mm²/m")
        R.min("mesh bar diameter (lower end)", mesh[2], dlo, "9.6.10 item 4", should=True)
        R.max("mesh bar diameter (upper end)", mesh[2], dhi, "9.6.10 item 4", should=True)
    R.min("vertical tie diameter", tie_d, PILE_CAP_TIE_D, "9.6.10 item 5")
    return R.report()
