"""Detailing of prestressed concrete superstructures, JTG 3362-2018 9.4.1–9.4.20, and the reinforcement zones of
post-tensioned anchorage zones (8.2.1 notes, 9.4.18, 9.4.20). External tendons (9.4.21–9.4.27): jtg.external_detailing;
dense anchors and the bursting force of a set of anchors (8.2.2): femotion.codes.jtg.regions.

Detailing does not decide a verdict: every *_detailing function returns a detailing report (Check.report, `ok` always
true) whose `warnings` list the rules not met ("should" marks the code's 宜); a rule whose input is not given is skipped
and named in the notes. 9.4.8 (cover of curved ducts), 9.4.9 item 2 (spacing of curved ducts), 9.3.14-2 (radius of a
curve) and the zones of 8.2.1 and 9.4.20 are formulas and come as plain functions as well.

Units: mm, mm², MPa; forces kN (9.4.8 is printed in N: the functions convert); angles in degrees.
Clauses: docs/codes/JTG_3362-2018.md.

    from femotion.codes.jtg import tendon_detailing as td
    Pd = 1.2 * 1300 * 9 * 139 / 1e3                    # kN: 1.2 × the force after the ring, set and friction losses
    r = td.curve_radius(12000, 600)                     # 9.3.14-2: chord 12 m, rise 0.6 m
    td.curved_cover(Pd, r, fcu=45, ds=90)               # (Cin, Cout) of 9.4.8-1, -3
    td.curved_duct_detailing(Pd, r, fcu=45, ds=90, c_in=60, c_out=55, cover_min=45, stirrup=(12, 100))
    td.duct_detailing(sec, clear=60, ds=90, radius=r, straight_length=1000, grout=55)
    td.end_zone_detailing(plate="trumpet", spiral_length=300, trumpet_length=250, rho_v=0.012, stirrup_spacing=100)
"""
from __future__ import annotations

import math

import numpy as np

from .external import deviation_radius_min
from .materials import Rebar, Tendon, rebar as _rebar
from .rc import Check, RCSection


# ── a detailing report ───────────────────────────────────────────────────────
def _g(v: float) -> str:
    return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"


def _grade(c) -> float:
    """strength class as a number: 40, "C40" or a jtg.concrete"""
    return float(getattr(c, "grade", None) or str(c).upper().lstrip("C"))


class _Rules:
    """collects the rules of one detailing report: min / max / flag record the value against its limit in
    `details` and add the unmet ones to `warnings`; a rule without its input is skipped and listed in the notes"""

    def __init__(self, name: str, clause: str, details: dict | None = None):
        self.chk = Check.report(name, clause, details)
        self.skipped: list[str] = []

    def _put(self, label, value, limit, rule, ok, cl, unit, should, pct):
        self.chk.details[label] = {"value": value, "limit": limit, "rule": rule}
        if not ok:
            f = (lambda v: f"{100 * v:.3g} %") if pct else (lambda v: _g(v) + (f" {unit}" if unit else ""))
            neg = {"≥": "<", ">": "≤", "≤": ">", "<": "≥"}[rule]
            self.chk.warnings.append(f"{label} {f(value)} {neg} {f(limit)} ({cl}{'; should' if should else ''})")
        return ok

    def min(self, label, value, limit, cl, *, unit="mm", should=False, strict=False, pct=False):
        """value ≥ limit (strict: value > limit)"""
        if value is None or limit is None:
            return self.skip(label, cl)
        tol = 1e-9 * max(1.0, abs(limit))
        ok = value > limit + tol if strict else value >= limit - tol
        return self._put(label, value, limit, ">" if strict else "≥", ok, cl, unit, should, pct)

    def max(self, label, value, limit, cl, *, unit="mm", should=False, strict=False, pct=False):
        """value ≤ limit (strict: value < limit)"""
        if value is None or limit is None:
            return self.skip(label, cl)
        tol = 1e-9 * max(1.0, abs(limit))
        ok = value < limit - tol if strict else value <= limit + tol
        return self._put(label, value, limit, "<" if strict else "≤", ok, cl, unit, should, pct)

    def flag(self, label, ok, message, cl, *, should=False):
        """a yes / no rule: ok None skips it"""
        if ok is None:
            return self.skip(label, cl)
        self.chk.details[label] = bool(ok)
        if not ok:
            self.chk.warnings.append(f"{message} ({cl}{'; should' if should else ''})")
        return bool(ok)

    def skip(self, label, cl):
        self.skipped.append(f"{label} ({cl})")
        return None

    def note(self, text):
        self.chk.notes.append(text)

    def report(self) -> Check:
        if self.skipped:
            self.chk.notes.append("skipped (no input): " + "; ".join(self.skipped))
        return self.chk


# ── 9.4.1, 9.4.2: girders ────────────────────────────────────────────────────
VERTICAL_TENDON_SPACING = (500.0, 1000.0)  # longitudinal spacing of vertical prestressing bars (9.4.1, should)
WEB_STIRRUP_D_MIN = {"tee": 10.0, "box": 12.0}  # web stirrups of T and I girders, of box girders (9.4.1)
WEB_STIRRUP_SPACING_MAX = 200.0  # (should)
END_STIRRUP_SPACING_MAX = 120.0  # closed stirrups within one girder depth from the support centre
BULB_STIRRUP_D_MIN = 8.0  # closed stirrups in the bottom bulb of T and I girders
BULB_STIRRUP_SPACING_MAX = 200.0


def girder_detailing(sec: RCSection | None = None, *, kind: str | None = None, h: float | None = None,
                     stirrup_d: float | None = None, ribbed: bool | None = None, stirrup_spacing: float | None = None,
                     end_spacing: float | None = None, end_closed: bool | None = None, end_length: float | None = None,
                     bulb_d: float | None = None, bulb_spacing: float | None = None, bulb_closed: bool | None = None,
                     vertical_spacing: float | None = None, pc_class: str | None = None,
                     hogging: bool = False) -> Check:
    """stirrups and vertical prestress of prestressed concrete girders (9.4.1) and the mixed reinforcement of partially
    prestressed ones (9.4.2). kind: "tee" (T and I girders) or "box" (default from sec); h: girder depth (default
    sec.h). Web stirrups: ribbed bars, diameter ≥ 10 mm (T, I) or 12 mm (box), spacing ≤ 200 mm (should); within one
    girder depth from the support centre (end_length ≥ h) closed stirrups at ≤ 120 mm; the bottom bulb of T and I
    girders: closed stirrups ≥ 8 mm at ≤ 200 mm; vertical prestressing bars at 500–1000 mm along the girder (should).
    pc_class "A" or "B" (partially prestressed): with sec, the tension face needs ordinary bars (9.4.2): the bottom
    rows (edge_masks), or with hogging=True (over interior supports) the top rows."""
    R = _Rules("prestressed girder detailing (9.4.1, 9.4.2)", "JTG 3362-2018 9.4.1, 9.4.2")
    if kind is None and sec is not None and sec.kind in ("tee", "box"):
        kind = sec.kind
    h = sec.h if h is None and sec is not None else h
    if kind not in (None, "tee", "box"):
        raise ValueError("kind: 'tee' (T and I girders) or 'box'")
    if stirrup_d is not None and kind is None:
        R.note("9.4.1 sets the web stirrup diameter of T, I and box girders only: give kind")
    R.min("web stirrup diameter", stirrup_d, WEB_STIRRUP_D_MIN.get(kind), "9.4.1")
    R.flag("ribbed web stirrups", ribbed, "web stirrups of plain bars: 9.4.1 asks for ribbed bars", "9.4.1")
    R.max("web stirrup spacing", stirrup_spacing, WEB_STIRRUP_SPACING_MAX, "9.4.1", should=True)
    R.max("stirrup spacing within h of the support", end_spacing, END_STIRRUP_SPACING_MAX, "9.4.1")
    R.flag("closed stirrups near the support", end_closed, "stirrups within h of the support centre not closed",
           "9.4.1")
    R.min("length of the dense end zone from the support centre", end_length, h, "9.4.1")
    if kind == "box" and (bulb_d is not None or bulb_spacing is not None):
        R.note("the bottom bulb rules of 9.4.1 are for T and I girders")
    else:
        R.min("bulb stirrup diameter", bulb_d, BULB_STIRRUP_D_MIN, "9.4.1")
        R.max("bulb stirrup spacing", bulb_spacing, BULB_STIRRUP_SPACING_MAX, "9.4.1")
        R.flag("closed bulb stirrups", bulb_closed, "stirrups in the bottom bulb not closed", "9.4.1")
    if vertical_spacing is not None:
        lo, hi = VERTICAL_TENDON_SPACING
        R.min("vertical prestressing bar spacing (lower end)", vertical_spacing, lo, "9.4.1", should=True)
        R.max("vertical prestressing bar spacing (upper end)", vertical_spacing, hi, "9.4.1", should=True)
    if pc_class in ("A", "B") and sec is not None:
        bot, top = sec.edge_masks()
        face = top if hogging else bot
        side = "top" if hogging else "bottom"
        R.flag("mixed reinforcement", bool(sec.bars[face, 0].sum() > 0),
               f"partially prestressed member without ordinary bars at the tension face ({side}): 9.4.2 asks for "
               "mixed reinforcement", "9.4.2")
        R.note("9.4.2: the tension-face bars preferably ribbed, of small diameter, closely spaced (not checked)")
    return R.report()


# ── 9.4.3–9.4.5: pretensioned members ────────────────────────────────────────
PRETENSIONED_STRAND_CLEAR = (1.5, 25.0)  # 1×7 strand: ≥ 1.5 d and ≥ 25 mm (9.4.4)
PRETENSIONED_WIRE_CLEAR = 15.0  # wires (9.4.4)
END_SPIRAL_MIN = 150.0  # single tendon: spiral at the end (9.4.5)
END_MESHES = (3, 5)  # several tendons: meshes within 10 d of the end (9.4.5)


def _kind(t) -> str:
    return t.kind if isinstance(t, Tendon) else str(t).lower()


def pretensioned_detailing(tendon, d: float, *, clear: float | None = None, n: int | None = None,
                           spiral_length: float | None = None, meshes: int | None = None,
                           plain: bool = False) -> Check:
    """pretensioned members (9.4.3–9.4.5). tendon: jtg.tendon or its kind ("strand", "wire", "bar"); d: nominal
    diameter of the strand or wire. 9.4.3: strands or spiral-rib wires (should; plain wires need measures for the
    anchorage, a note). 9.4.4: clear spacing of strands ≥ 1.5 d and ≥ 25 mm (1×7), of wires ≥ 15 mm. 9.4.5: a single
    tendon gets a spiral ≥ 150 mm long at the member end (n = 1, spiral_length); several tendons 3–5 meshes within
    10 d of the end (meshes: the number placed there)."""
    k = _kind(tendon)
    R = _Rules("pretensioned member detailing (9.4.3–9.4.5)", "JTG 3362-2018 9.4.3–9.4.5", {"d": d})
    R.flag("tendon type", k != "bar", "threaded bars in a pretensioned member: 9.4.3 asks for strands or spiral-rib "
           "wires", "9.4.3", should=True)
    if plain and k == "wire":
        R.note("plain wires: take measures for a reliable anchorage in the concrete (9.4.3; commentary: plain wires "
               "slipped before reaching their strength)")
    if k == "strand":
        R.min("clear spacing of strands", clear, max(PRETENSIONED_STRAND_CLEAR[0] * d, PRETENSIONED_STRAND_CLEAR[1]),
              "9.4.4")
    elif k == "wire":
        R.min("clear spacing of wires", clear, PRETENSIONED_WIRE_CLEAR, "9.4.4")
    elif clear is not None:
        R.note("9.4.4 gives no clear spacing for threaded bars")
    if n is not None and n <= 1:
        R.min("end spiral length (single tendon)", spiral_length, END_SPIRAL_MIN, "9.4.5")
    elif n is not None:
        R.chk.details["mesh zone"] = 10 * d
        if R.min(f"meshes within 10 d = {10 * d:.0f} mm of the end", meshes, END_MESHES[0], "9.4.5", unit="") \
                and meshes is not None and meshes > END_MESHES[1]:
            R.note(f"{meshes} meshes: 9.4.5 asks for 3–5")
    else:
        R.skip("end spiral or meshes (give n)", "9.4.5")
    return R.report()


# ── 9.3.14-2, 9.4.8, 9.4.9 item 2, 9.4.10, 9.4.11: curved tendons ──────────
def curve_radius(l: float, f: float) -> float:
    """radius of a curve from its chord l and rise f (9.3.14-2, used by 9.4.8): r = l/2 (1/(4β) + β), β = f/l. This is
    exactly the radius of the circular arc through the chord ends and the crown; for a parabola it is the radius at the
    vertex plus f/2."""
    beta = f / l
    return l / 2 * (1 / (4 * beta) + beta)


def curved_cover(Pd: float, r: float, fcu: float, ds: float) -> tuple[float, float]:
    """least concrete cover of a curved duct (9.4.8): in the plane of the curve, towards its centre,
    Cin ≥ Pd / (0.266 r √f'cu) − ds/2 (9.4.8-1); out of the plane, Cout ≥ Pd / (0.266 π r √f'cu) − ds/2 (9.4.8-3).
    Pd: design tendon force (kN here, N in the code): 1.2 × the force after the anchorage-ring friction, the anchorage
    set and the duct friction up to the section; r: duct radius (curve_radius for a curve other than a circle); f'cu:
    cube strength at tensioning; ds: outer duct diameter. The raw values (they can be negative); the 9.1.1 minimum
    governs where it is larger (item 3), see curved_duct_detailing."""
    k = Pd * 1e3 / (0.266 * r * math.sqrt(fcu))
    return k - ds / 2, k / math.pi - ds / 2


def curved_stirrup(Pd: float, r: float, sv: float, fsv: float) -> float:
    """area of one stirrup leg in the plane of a curved duct whose cover is the 9.1.1 minimum instead of Cin (9.4.8-2):
    Asv1 ≥ Pd sv / (2 r fsv) (mm²). Pd in kN; sv: stirrup spacing; fsv: stirrup design strength (table 3.2.3-1)."""
    return Pd * 1e3 * sv / (2 * r * fsv)


def curved_duct_detailing(Pd: float, r: float, fcu: float, ds: float, *, c_in: float | None = None,
                          c_out: float | None = None, cover_min: float | None = None, stirrup=None,
                          stirrup_steel="HRB400") -> Check:
    """cover of a curved duct (9.4.8). Pd (kN), r, fcu, ds as in curved_cover; c_in, c_out: the covers provided in and
    out of the plane of the curve; cover_min: the 9.1.1 minimum for the member (it governs where larger, item 3).
    In the plane: c_in ≥ max(Cin, cover_min); or, where Cin is large, the 9.1.1 cover with stirrups in the plane of the
    curve, one leg ≥ Pd sv / (2 r fsv) (9.4.8-2; stirrup = (diameter, spacing)). Out of the plane: c_out ≥ max(Cout,
    cover_min). details: Cin, Cout and, with a stirrup, Asv1 required."""
    Cin, Cout = curved_cover(Pd, r, fcu, ds)
    R = _Rules("curved duct cover (9.4.8)", "JTG 3362-2018 9.4.8", {"Cin": Cin, "Cout": Cout, "r": r})
    cm = cover_min if cover_min is not None else 0.0
    if cover_min is None:
        R.note("cover_min (9.1.1) not given: only the formulas checked")
    need = max(Cin, cm)
    if c_in is None:
        R.skip("cover in the plane of the curve", "9.4.8-1")
    elif stirrup is None or c_in >= need - 1e-9 * max(1.0, abs(need)):
        R.min("cover in the plane of the curve", c_in, need, "9.4.8-1")
    else:  # the 9.1.1 cover with stirrups in the plane of the curve
        st = stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel)
        d, sv = stirrup
        Asv1 = curved_stirrup(Pd, r, sv, st.fsv)
        R.chk.details["Asv1 required"] = Asv1
        R.note(f"c_in = {c_in:g} mm < Cin = {Cin:.0f} mm: the 9.1.1 cover with stirrups in the plane of the curve "
               "(9.4.8)")
        if cover_min is None:
            R.skip("cover in the plane against 9.1.1 (give cover_min)", "9.4.8, 9.1.1")
        else:
            R.min("cover in the plane of the curve", c_in, cover_min, "9.4.8, 9.1.1")
        R.min("stirrup leg area in the plane of the curve", math.pi * d * d / 4, Asv1, "9.4.8-2", unit="mm²")
    R.min("cover out of the plane of the curve", c_out, max(Cout, cm), "9.4.8-3")
    return R.report()


DUCT_CLEAR_MIN = 40.0  # clear spacing of straight ducts (9.4.9 item 1)
DUCT_CLEAR_RATIO = 0.6  # and at least 0.6 duct diameters (should)
DUCT_STACK_MAX = 2  # corrugated metal or plastic ducts, iron-sheet ducts: two stacked vertically in straight runs
DUCT_AREA_RATIO = 2.0  # inner duct area ≥ 2 × tendon area (9.4.9 item 3)


def curved_spacing(Pd: float, r: float, fcu: float, ds: float, *, clear_in: float | None = None,
                   clear_out: float | None = None) -> Check:
    """clear spacing of two adjacent curved ducts (9.4.9 item 2). In the plane of the curve by 9.4.8-1 with Pd and r of
    the duct of the larger radius (the outer duct presses on the concrete between them), at least the straight-duct
    minimum 40 mm (9.4.9 item 1); out of the plane by 9.4.8-3 (the code does not say which duct: pass the one of the
    larger Pd/r), program convention: also at least 40 mm. The 0.6 ds of item 1 is checked as a "should" rule."""
    Cin, Cout = curved_cover(Pd, r, fcu, ds)
    R = _Rules("curved duct spacing (9.4.9 item 2)", "JTG 3362-2018 9.4.9", {"Cin": Cin, "Cout": Cout})
    R.min("clear spacing in the plane of the curve", clear_in, max(Cin, DUCT_CLEAR_MIN), "9.4.9 item 2, 9.4.8-1")
    R.min("clear spacing out of the plane", clear_out, max(Cout, DUCT_CLEAR_MIN), "9.4.9 item 2, 9.4.8-3")
    for lab, c in (("in the plane", clear_in), ("out of the plane", clear_out)):
        if c is not None:
            R.min(f"clear spacing {lab} / duct diameter", c / ds, DUCT_CLEAR_RATIO, "9.4.9 item 1", unit="",
                  should=True)
    return R.report()


def curve_radius_min(kind, d: float) -> float:
    """least radius of a curved post-tensioned tendon (9.4.10, should): wires and strands 4 m when their wires are at
    most 5 mm in diameter, 6 m above (item 1, = jtg.deviation_radius_min, d the wire diameter); threaded bars 12 m up to
    25 mm, 15 m above (item 2, d the bar diameter). mm."""
    k = _kind(kind)
    if k in ("strand", "wire"):
        return deviation_radius_min(d)
    if k == "bar":
        return 12000.0 if d <= 25.0 else 15000.0
    raise ValueError(f"tendon kind {kind!r}: strand, wire or bar")


# figure 9-1 (commentary to 9.4.11, p233): least straight length behind the anchor against the least breaking load
# (VSL): 0.8 m, 1.0 m from about 1.3 MN, 1.5 m from about 5.3 MN (the steps read off the figure)
STRAIGHT_LENGTH = (800.0, 1500.0)  # 9.4.11: 0.80–1.50 m (should)
STRAIGHT_LENGTH_STEPS = ((1300.0, 800.0), (5300.0, 1000.0), (math.inf, 1500.0))  # (breaking load up to, kN; length)


def anchor_straight_min(F_break: float | None = None) -> float:
    """least straight length of a curved wire or strand tendon behind its anchor (9.4.11, 0.80–1.50 m, should; mm).
    With the tendon's least breaking load F_break (kN) the length within that range from figure 9-1 of the commentary
    (steps at about 1.3 MN and 5.3 MN, read off the figure); without it 0.80 m."""
    if F_break is None:
        return STRAIGHT_LENGTH[0]
    return next(L for F, L in STRAIGHT_LENGTH_STEPS if F_break <= F)


GROUT_STRENGTH_MIN = 50.0  # 28-day strength of 40 × 40 × 160 mm prisms (9.4.12)


def duct_detailing(sec: RCSection | None = None, *, ds: float | None = None, d_inner: float | None = None,
                   Ap: float | None = None, clear: float | None = None, stacked: int | None = None,
                   curved: bool = False, kind=None, d: float | None = None, radius: float | None = None,
                   straight_length: float | None = None, F_break: float | None = None,
                   grout: float | None = None) -> Check:
    """ducts and curved tendons of post-tensioned members (9.4.9 items 1, 3, 4; 9.4.10; 9.4.11; 9.4.12).
    Straight ducts: clear spacing ≥ 40 mm and ≥ 0.6 ds (should); corrugated metal or plastic ducts and iron-sheet ducts
    may be stacked two high in straight runs (stacked: the number touching one above the other; curved=True for a
    curved run). Inner duct area ≥ 2 × tendon area: with d_inner and Ap, or for every tendon row of sec (its duct
    diameter, normally the outer one, taken as the inner one unless d_inner is given; a note says so).
    Radius ≥ curve_radius_min(kind, d) (kind default from sec, d: the strand's wire diameter or the bar diameter).
    Straight length behind the anchor ≥ anchor_straight_min(F_break) (should). Grout strength ≥ 50 MPa. Curved ducts
    (9.4.9 item 2): curved_spacing; their cover: 9.4.8, curved_duct_detailing."""
    R = _Rules("duct detailing (9.4.9–9.4.12)", "JTG 3362-2018 9.4.9–9.4.12")
    if sec is not None and sec.pretensioned:
        R.note("pretensioned member: no ducts (9.4.9–9.4.12 are for post-tensioned members)")
        return R.report()
    R.min("clear spacing of straight ducts", clear, DUCT_CLEAR_MIN, "9.4.9 item 1")
    if clear is not None and ds is not None:
        R.min("clear spacing / duct diameter", clear / ds, DUCT_CLEAR_RATIO, "9.4.9 item 1", unit="", should=True)
    if stacked is not None:
        R.max("ducts stacked vertically", stacked, 1 if curved else DUCT_STACK_MAX,
              "9.4.9 item 1" + (": curved runs" if curved else ""), unit="")
        if stacked > 1:
            R.note("stacking is for corrugated metal or plastic ducts and iron-sheet ducts in straight runs (9.4.9)")
    if d_inner is not None and Ap is not None:
        R.min("inner duct area / tendon area", math.pi * d_inner ** 2 / 4 / Ap, DUCT_AREA_RATIO, "9.4.9 item 3",
              unit="")
    elif sec is not None and sec.has_internal:
        dd = None if d_inner is None else math.pi * d_inner ** 2 / 4
        for i, (a, z, _, duct) in enumerate(sec.tendons):
            R.min(f"tendon row {i} (depth {z:.0f} mm): inner duct area / tendon area", (dd or duct) / a,
                  DUCT_AREA_RATIO, "9.4.9 item 3", unit="")
        if d_inner is None:
            R.note("9.4.9 item 3: the section's duct diameter taken as the inner diameter; it is normally the outer "
                   "one (deducted for the net section), which overstates the inner area: pass d_inner")
    else:
        R.skip("inner duct area (give d_inner and Ap, or sec)", "9.4.9 item 3")
    k = kind if kind is not None else (sec.tendon.kind if sec is not None and sec.has_internal else None)
    if radius is not None and k is not None and d is not None:
        R.min("tendon radius", radius, curve_radius_min(k, d), "9.4.10", should=True)
        R.note("9.4.10 commentary: special ducts (e.g. semicircular hoop tendons of cable-stayed pylons, radius about "
               "1.5 m) may go below with special measures")
    else:
        R.skip("tendon radius (give radius, d and the tendon kind)", "9.4.10")
    if k == "bar" and straight_length is not None:
        R.note("9.4.11 is for wire and strand tendons")
    else:
        R.min("straight length behind the anchor", straight_length, anchor_straight_min(F_break), "9.4.11",
              should=True)
    R.min("grout strength", grout, GROUT_STRENGTH_MIN, "9.4.12", unit="MPa")
    R.note("9.4.9 item 4: ducts take the pre-camber where the member has one (not checked)")
    return R.report()


# ── 9.4.6, 9.4.13, 9.4.18: anchorage zones ───────────────────────────────────
ANCHOR_RHO_V_MIN = 0.005  # indirect reinforcement under the plate, ρv of 5.7.2 (9.4.6)
LOCAL_LAYERS_MIN = 4  # flat plates: ≥ 4 mesh layers or spiral turns (9.4.18 item 1)
END_ZONE_STIRRUP_SPACING_MAX = 120.0  # closed stirrups against bursting in the general zone (9.4.18 item 2)
SEAL_RATIO, SEAL_MIN = 0.8, 30  # sealing concrete over the anchors: ≥ 0.8 × the member's class and ≥ C30 (9.4.13)


def _local_zone(R: _Rules, plate, layers, spiral_length, trumpet_length, cl: str):
    """9.4.18 item 1: flat plates ≥ 4 mesh layers or spiral turns; plates with a trumpet a spiral whose length (turns ×
    pitch) is at least the trumpet's"""
    if plate == "flat":
        R.min("mesh layers or spiral turns under a flat plate", layers, LOCAL_LAYERS_MIN, cl, unit="")
    elif plate == "trumpet":
        R.min("spiral length under the plate (≥ trumpet length)", spiral_length, trumpet_length, cl)
    elif plate is None:
        R.skip("local-zone reinforcement (give plate)", cl)
    else:
        raise ValueError("plate: 'trumpet' or 'flat'")


def end_zone_detailing(*, plate: str | None = None, layers: int | None = None, spiral_length: float | None = None,
                       trumpet_length: float | None = None, rho_v: float | None = None,
                       stirrup_spacing: float | None = None, closed: bool | None = None, e: float | None = None,
                       h: float | None = None, end_bars_to_edge: bool | None = None,
                       spalling_bars: bool | None = None, seal=None, member=None,
                       seal_bars: bool | None = None) -> Check:
    """the end anchorage zone of a post-tensioned member (9.4.6, 9.4.13, 9.4.18). plate: "trumpet" (anchor plate with a
    trumpet, required at end anchorages by 9.4.6) or "flat". Local zone (9.4.18 item 1): flat plates ≥ 4 mesh layers or
    spiral turns (layers); trumpet plates a spiral at least as long as the trumpet (spiral_length, trumpet_length);
    ρv of 5.7.2 ≥ 0.5 % (9.4.6; bearing.spiral_ratio, mesh_ratio). General zone (item 2): closed stirrups against the
    bursting force at ≤ 120 mm. End face (item 3): bars against spalling (spalling_bars); with a large-eccentricity
    anchorage (e, h: γ = 2e/h > 1/3, the force outside the kern, 8.2.5) the end-face bars bent and run on to the
    longitudinal tension edge (should; end_bars_to_edge). Anchors left in the member (9.4.13): bars round them tied to
    the member before sealing (seal_bars); seal, member: strength classes of the sealing concrete and of the member,
    seal ≥ 0.8 member and ≥ C30. The zones (8.2.1 notes): anchorage_zone."""
    R = _Rules("end anchorage zone detailing (9.4.6, 9.4.13, 9.4.18)", "JTG 3362-2018 9.4.6, 9.4.13, 9.4.18")
    R.flag("trumpet anchor plate", None if plate is None else plate == "trumpet",
           "flat anchor plate at an end anchorage: 9.4.6 asks for plates with a trumpet", "9.4.6")
    R.min("ρv under the plate", rho_v, ANCHOR_RHO_V_MIN, "9.4.6", pct=True)
    _local_zone(R, plate, layers, spiral_length, trumpet_length, "9.4.18 item 1")
    R.max("bursting stirrup spacing", stirrup_spacing, END_ZONE_STIRRUP_SPACING_MAX, "9.4.18 item 2")
    R.flag("closed bursting stirrups", closed, "bursting stirrups not closed", "9.4.18 item 2")
    if e is not None and h is not None:
        g = 2 * e / h
        R.chk.details["gamma"] = g
        if g > 1 / 3:
            R.flag("end-face bars to the tension edge", end_bars_to_edge, "large-eccentricity anchorage (γ = "
                   f"{g:.2f} > 1/3): end-face bars should be bent and run on to the longitudinal tension edge",
                   "9.4.18 item 3", should=True)
    else:
        R.skip("large-eccentricity anchorage (give e and h)", "9.4.18 item 3")
    R.flag("bars against spalling at the end face", spalling_bars, "no bars against the spalling force at the end "
           "face", "9.4.18 item 3")
    R.flag("bars round the sealed anchors", seal_bars, "no bars round the anchors tying the sealing concrete to the "
           "member", "9.4.13")
    if seal is not None and member is not None:
        R.min("sealing concrete class", _grade(seal), max(SEAL_RATIO * _grade(member), SEAL_MIN), "9.4.13", unit="MPa")
    else:
        R.skip("sealing concrete (give seal and member)", "9.4.13")
    return R.report()


def anchorage_zone(h: float, *, bf: float = 0.0, plate=None, factor: float = 1.0) -> dict:
    """extent of the anchorage zone at a member end (8.2.1 notes, figure 8-2) and the ranges in which the ties of 8.2.1
    count their bars (mm from the end face). The zone: across the whole end section, along the member factor ×
    max(h, b'f), factor 1.0–1.2 (h: depth, b'f: flange width); the local zone 1.2 × the longer side of the plate
    (plate = (a1, a2)) deep, over the bearing area of figure 5.7.1; the general zone the rest.
    Program convention for counting As (8.2.1 sends As to "the bars within the ranges of 9.4.18 and 9.4.20", and 9.4.18
    gives none): "web" = (end of the local zone, factor × h) for the ties in the plane of the web, the bursting force of
    8.2.2 and the edge tension of 8.2.5. The commentary to 8.2.2 treats the web of a T or box girder as a rectangular
    plate of depth h, whose own zone by note 1 is (1.0–1.2) h; the bursting tension lies within about h of the end (the
    transverse stress under the plate is compression, commentary figure 8-3b; the commentary to 9.4.7 puts the tension
    within about 3/4 h), so the web's stirrups farther than h from the end carry none of it. "flange" = (0, the zone's
    length) for the spread of the force into the flanges (the transverse spalling of the slabs, commentary figure
    8-7b). Returns {"length", "length_range", "local_length", "web", "flange"} ("web" None without the plate)."""
    if not 1.0 <= factor <= 1.2:
        raise ValueError("8.2.1 note 1: the zone length is 1.0–1.2 × max(h, b'f)")
    L = factor * max(h, bf)
    out = {"length": L, "length_range": (max(h, bf), 1.2 * max(h, bf)), "local_length": None, "web": None,
           "flange": (0.0, L)}
    if plate is not None:
        loc = 1.2 * max(plate)
        out.update(local_length=loc, web=(loc, factor * h))
    return out


def zone_steel(positions, area, lo: float, hi: float) -> float:
    """steel area of the bars (or stirrups) whose positions lie within [lo, hi] (mm²): As of the ties of 8.2.1 counted
    within the zones of 9.4.18 / 9.4.20 (anchorage_zone, blister_zones). area: one value or one per position (e.g. the
    legs × the bar area of each stirrup)."""
    x = np.asarray(positions, float)
    a = np.broadcast_to(np.asarray(area, float), x.shape)
    m = (x >= lo - 1e-9) & (x <= hi + 1e-9)
    return float(a[m].sum())


# ── 9.4.19, 9.4.20: anchor blisters ──────────────────────────────────────────
BLISTER_DEVIATION_MAX = 15.0  # tendon deviation inside the blister (9.4.19, should)
BLISTER_FACE_ANGLE_MIN = 90.0  # between the anchor face and the blister's slope (9.4.19, should)
BLISTER_STIRRUP_SPACING_MAX = 150.0  # stirrups against bursting under the anchor (9.4.20 item 1, should)
BLISTER_STIRRUP_EXTENT = 1.2  # × the blister height, along the tendon (9.4.20 item 1, should)
BLISTER_BAR_LENGTH = 1500.0  # longitudinal bars against tie-back and local bending (9.4.20 items 3, 4, should)
BLISTER_BAR_HALF_WIDTH = 1.5  # × the plate width, either side of the tendon axis (9.4.20 items 3, 4, should)


def blister_zones(hb: float, a: float, t: float, *, R: float | None = None, deviation: float | None = None,
                  width: float | None = None, L: float | None = None) -> dict:
    """zones of the reinforcement of a triangular anchor blister (9.4.20, figure 9.4.20, commentary figure 9-3).
    Along the wall plate: x from the line where the anchor face meets the wall, positive ahead of the anchor face (under
    the blister, the way the anchorage force acts); across it: y from the tendon axis. hb: blister height at the anchor
    face (figure 9.4.20 h); a: plate width; t: wall thickness; R, deviation: the tendon's radius and the angle it turns
    through in its curve (degrees); width, L: the blister's width and length.
    "bursting": the least extent of the stirrups along the tendon from the anchor face, 1.2 hb (item 1);
    "tie_back": inner-edge bars, x ∈ [−750, 750] at least, |y| ≤ 1.5 a (item 3); "bending": outer-edge bars,
    x ∈ [t − 750, t + 750] at least, |y| ≤ 1.5 a (item 4); "radial": the curved length R × deviation (item 5);
    "zone": the blister's anchorage zone of 8.2.1 note 1, (across 3 × width, along L + 2t). None where not given."""
    half = BLISTER_BAR_LENGTH / 2
    return {"bursting": BLISTER_STIRRUP_EXTENT * hb, "tie_back": (-half, half), "bending": (t - half, t + half),
            "half_width": BLISTER_BAR_HALF_WIDTH * a,
            "radial": None if R is None or deviation is None else R * math.radians(deviation),
            "zone": None if width is None or L is None else (3 * width, L + 2 * t)}


def blister_detailing(*, hb: float | None = None, a: float | None = None, t: float | None = None,
                      deviation: float | None = None, face_angle: float | None = None, L: float | None = None,
                      S=None, alpha: float | None = None, R: float | None = None, kind=None, d: float | None = None,
                      stirrup_spacing: float | None = None, stirrup_extent: float | None = None, tie_back=None,
                      bending=None, bar_half_width: float | None = None, radial_extent: float | None = None,
                      plate: str | None = None, layers: int | None = None, spiral_length: float | None = None,
                      trumpet_length: float | None = None) -> Check:
    """geometry (9.4.19) and reinforcement (9.4.20) of a post-tensioned anchor blister. Geometry (all should): tendon
    radius R ≥ curve_radius_min(kind, d) (9.4.10, 9.4.16); deviation inside the blister ≤ 15°; angle between the anchor
    face and the blister's slope ≥ 90°; length L ≥ (S1 + S2) / sin α (figure 9.4.19 prints ">"; it is exactly the
    length of a blister whose face is square to the tendon and whose slope runs parallel to it, i.e. the 90° the
    clause allows, so the program takes "≥"; S = (S1, S2), α the tendon's angle to the wall at the anchor).
    Reinforcement (x along the wall as in blister_zones): bursting stirrups at ≤ 150 mm over ≥ 1.2 hb along the tendon
    (item 1, should); local zone as 9.4.18 item 1 (plate, layers, spiral_length, trumpet_length);
    tie_back = (x0, x1), the extent of the inner-edge longitudinal bars, covering [−750, 750] (item 3, should: ≥ 1.5 m
    centred on the anchor face line); bending = (x0, x1), the outer-edge bars, covering [t − 750, t + 750] (item 4);
    bar_half_width: how far either side of the tendon axis those bars are counted, ≤ 1.5 a; radial
    stirrups and U-bars over the whole curved length R α (item 5, radial_extent, should). The tie forces: regions.
    blister_forces; As within the zones: zone_steel."""
    Rr = _Rules("anchor blister detailing (9.4.19, 9.4.20)", "JTG 3362-2018 9.4.19, 9.4.20")
    if R is not None and kind is not None and d is not None:
        Rr.min("tendon radius in the blister", R, curve_radius_min(kind, d), "9.4.19, 9.4.10", should=True)
    else:
        Rr.skip("tendon radius (give R, kind, d)", "9.4.19, 9.4.10")
    Rr.max("deviation inside the blister", deviation, BLISTER_DEVIATION_MAX, "9.4.19", unit="°", should=True)
    Rr.min("anchor face / blister slope angle", face_angle, BLISTER_FACE_ANGLE_MIN, "9.4.19", unit="°", should=True)
    if L is not None and S is not None and alpha is not None:
        Rr.min("blister length", L, (S[0] + S[1]) / math.sin(math.radians(alpha)), "figure 9.4.19", should=True)
    else:
        Rr.skip("blister length (give L, S, alpha)", "figure 9.4.19")
    Rr.max("bursting stirrup spacing", stirrup_spacing, BLISTER_STIRRUP_SPACING_MAX, "9.4.20 item 1", should=True)
    Rr.min("bursting stirrup extent along the tendon", stirrup_extent,
           None if hb is None else BLISTER_STIRRUP_EXTENT * hb, "9.4.20 item 1", should=True)
    _local_zone(Rr, plate, layers, spiral_length, trumpet_length, "9.4.20 item 1, 9.4.18 item 1")
    half = BLISTER_BAR_LENGTH / 2
    for lab, ext, c, cl in (("tie-back bars", tie_back, 0.0, "9.4.20 item 3"),
                            ("local-bending bars", bending, t, "9.4.20 item 4")):
        if ext is None or c is None:
            Rr.skip(f"{lab} (give their extent" + (" and t)" if c is None else ")"), cl)
            continue
        Rr.max(f"{lab}: start", ext[0], c - half, cl, should=True)
        Rr.min(f"{lab}: end", ext[1], c + half, cl, should=True)
    Rr.max("longitudinal bars counted either side of the tendon axis", bar_half_width,
           None if a is None else BLISTER_BAR_HALF_WIDTH * a, "9.4.20 items 3, 4", should=True)
    turn = deviation if deviation is not None else alpha  # the curve's turn: the deviation, else the anchor angle
    Rr.min("radial reinforcement extent", radial_extent,
           None if R is None or turn is None else blister_zones(0.0, 0.0, 0.0, R=R, deviation=turn)["radial"],
           "9.4.20 item 5", should=True)
    Rr.note("9.4.20 item 2: end-face stirrups of the blister run on to the outer face of the wall (not checked)")
    return Rr.report()


# ── 9.4.17: precast segmental construction ───────────────────────────────────
SEGMENT_END_MESH_D_MIN = 10.0
WET_JOINT_WIDTH_MIN = 60.0
SHEAR_KEY_EXTENT = 0.75  # web keys over ≥ 0.75 of the girder depth (should)
SHEAR_KEY_WIDTH = 0.75  # key width across the web ≈ 0.75 of the web width (should, a target)
SHEAR_KEY_HEIGHT_MIN = 35.0  # and > 2 × the largest aggregate
SHEAR_KEY_RATIO = 0.5  # key height / mean width = 1 : 2
SHEAR_KEY_RATIO_TOL = 0.05  # program convention: within 5 % of 1 : 2


def segmental_detailing(*, end_mesh_d: float | None = None, wet_joint: float | None = None, wet_grade=None,
                        segment_grade=None, key_extent: float | None = None, h: float | None = None,
                        key_width: float | None = None, web_width: float | None = None,
                        key_height: float | None = None, aggregate: float | None = None,
                        key_mean_width: float | None = None) -> Check:
    """precast segmental structures (9.4.17): segment ends meshes of bars ≥ 10 mm (item 1); cast-in-place wet joints
    ≥ 60 mm wide, concrete not below the segments' class (item 2); web shear keys over ≥ 75 % of the girder depth
    (should) and ≈ 75 % of the web width across (a target: a note); key height > 2 × the largest aggregate and
    ≥ 35 mm; key height : mean width = 1 : 2 (item 3; program convention: within 5 %)."""
    R = _Rules("precast segmental detailing (9.4.17)", "JTG 3362-2018 9.4.17")
    R.min("segment end mesh bar diameter", end_mesh_d, SEGMENT_END_MESH_D_MIN, "9.4.17 item 1")
    R.min("wet joint width", wet_joint, WET_JOINT_WIDTH_MIN, "9.4.17 item 2")
    if wet_grade is not None and segment_grade is not None:
        R.min("wet joint concrete class", _grade(wet_grade), _grade(segment_grade), "9.4.17 item 2", unit="MPa")
    R.min("web shear keys / girder depth", None if key_extent is None or h is None else key_extent / h,
          SHEAR_KEY_EXTENT, "9.4.17 item 3", unit="", should=True)
    if key_width is not None and web_width is not None:
        R.chk.details["key width / web width"] = key_width / web_width
        if abs(key_width / web_width - SHEAR_KEY_WIDTH) > 0.05:
            R.note(f"key width {key_width / web_width:.0%} of the web: 9.4.17 suggests about 75 %")
    R.min("shear key height", key_height, SHEAR_KEY_HEIGHT_MIN, "9.4.17 item 3")
    R.min("shear key height / largest aggregate", None if key_height is None or aggregate is None
          else key_height / aggregate, 2.0, "9.4.17 item 3", unit="", strict=True)
    if key_height is not None and key_mean_width is not None:
        r = key_height / key_mean_width
        R.chk.details["key height / mean width"] = r
        if abs(r / SHEAR_KEY_RATIO - 1) > SHEAR_KEY_RATIO_TOL:
            R.chk.warnings.append(f"shear key height / mean width {r:.3f}: 9.4.17 takes 1 : 2 (9.4.17 item 3)")
    R.note("9.4.17 item 2: epoxy joints evenly coated and pressed until cured (not checked)")
    return R.report()
