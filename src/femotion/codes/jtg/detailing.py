"""Durability (4.5) and the general, slab and beam detailing of JTG 3362-2018 (9.1, 9.2.5–9.2.8, 9.3).

Units: mm throughout (bar diameters, covers, spacings, lengths; also the diaphragm spacing and the radius of 9.3.1),
mm² for areas, MPa for strengths; ages in months. Lookups return the tabulated minimum (cover_min, anchorage_length,
lap_length, hook_min, …); the checks return a detailing report (Check.report): `ok` is always true, the unmet rules are
in `warnings` (the code's "shall" and "should" alike), `details` hold each value against its limit and `notes` what was
skipped for want of an input or cannot be checked by numbers. Every input of a check is optional unless the rule
cannot be stated without it. Clauses, tables and program conventions: docs/codes/JTG_3362-2018.md (the durability and
detailing chapter).

    from femotion.codes import jtg
    sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(5, 28, 60), (5, 28, 120)], top=[(8, 16, 40)])
    jtg.durability(sec, "II", life=100)                               # 4.5.3 and 3.1.2
    jtg.cover_min("II", "superstructure", 100)                        # table 9.1.1: 30 mm
    jtg.section_detailing(sec, env="II", stirrup_d=10)                # covers, bundles, clear spacing, web bars
    jtg.stirrup_detailing(sec, (2, 10, 200), lap=("tension", 28, 100))
    jtg.anchorage_length(28, "HRB400", "C40"), jtg.lap_length(28, "HRB400", "C40", share=0.5)

Durability is a separate check (not part of every strength check): the environment belongs to the member, not to the
cross-section, and one report per member is enough; the strength checks keep reporting 3.1.2 through
RCSection.material_rules.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .materials import Concrete, Rebar, concrete as _concrete, rebar as _rebar
from .rc import Check, Circle, RCSection, Rect
from .slab import haunch_depth, slab_detailing

_TOL = 1e-9

# ── 4.5 durability ──────────────────────────────────────────────────────────
# table 4.5.2: environment classes (p20)
ENVIRONMENTS = {
    "I": "general: carbonation of the concrete only",
    "II": "freeze-thaw: repeated freezing and thawing",
    "III": "near-shore or marine chloride: chlorides of the marine environment",
    "IV": "de-icing salt and other chlorides",
    "V": "salt crystallisation: sulphate crystals expanding in the pores of the concrete",
    "VI": "chemical attack: strongly acid or alkaline chemicals",
    "VII": "abrasion: friction, cutting and impact of wind, water flow or material carried by the water",
}

# member categories of tables 4.5.3 and 9.1.1
MEMBERS = {
    "superstructure": "beams, slabs, towers, arch rings, culvert superstructures",
    "substructure": "pier and abutment shafts, culvert substructures",
    "foundation": "pile caps, foundations",
}
_MEMBER_ALIASES = {
    "beam": "superstructure", "girder": "superstructure", "slab": "superstructure", "tower": "superstructure",
    "pylon": "superstructure", "arch": "superstructure", "deck": "superstructure",
    "pier": "substructure", "abutment": "substructure", "column": "substructure",
    "pile cap": "foundation", "pilecap": "foundation", "footing": "foundation", "pile": "foundation",
}

# table 4.5.3: lowest strength class (p20); columns: superstructure 100, 50/30 years, substructure 100, 50/30,
# foundation 100, 50/30
MIN_CONCRETE_CLASS = {
    "I": (35, 30, 30, 25, 25, 25),
    "II": (40, 35, 35, 30, 30, 25),
    "III": (40, 35, 35, 30, 30, 25),
    "IV": (40, 35, 35, 30, 30, 25),
    "V": (40, 35, 35, 30, 30, 25),
    "VI": (40, 35, 35, 30, 30, 25),
    "VII": (40, 35, 35, 30, 30, 25),
}

# table 9.1.1: minimum cover cmin (mm) of the outermost bars (p97), same columns
COVER_MIN = {
    "I": (20, 20, 25, 20, 40, 40),
    "II": (30, 25, 35, 30, 45, 40),
    "III": (35, 30, 45, 40, 65, 60),
    "IV": (30, 25, 35, 30, 45, 40),
    "V": (30, 25, 40, 35, 45, 40),
    "VI": (35, 30, 40, 35, 60, 55),
    "VII": (35, 30, 45, 40, 65, 60),
}

_ROMAN = {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V", "Ⅵ": "VI", "Ⅶ": "VII"}
_NUM = {str(k + 1): r for k, r in enumerate(ENVIRONMENTS)}


def environment(env) -> str:
    """the environment class of table 4.5.2 as "I" … "VII" (also accepts 1 … 7, "Ⅲ", "Ⅲ类", "class III")"""
    e = str(env).upper().replace("CLASS", "").replace("类", "").strip()
    e = _NUM.get(_ROMAN.get(e, e), _ROMAN.get(e, e))
    if e not in ENVIRONMENTS:
        raise ValueError(f"environment class {env!r}: one of {', '.join(ENVIRONMENTS)} (table 4.5.2)")
    return e


def member_category(member: str) -> str:
    """"superstructure", "substructure" or "foundation" (tables 4.5.3, 9.1.1); also "beam", "pier", "pile cap", …"""
    m = str(member).strip().lower().replace("_", " ")
    m = _MEMBER_ALIASES.get(m, m)
    if m not in MEMBERS:
        raise ValueError(f"member {member!r}: one of {', '.join(MEMBERS)} (tables 4.5.3, 9.1.1)")
    return m


def _column(member: str, life: int) -> int:
    if life not in (100, 50, 30):
        raise ValueError(f"design life {life!r}: tables 4.5.3 and 9.1.1 have 100, 50 and 30 years")
    return 2 * list(MEMBERS).index(member_category(member)) + (0 if life == 100 else 1)


def min_concrete_class(env, member: str = "superstructure", life: int = 100) -> int:
    """lowest concrete strength class (cube strength, MPa) of table 4.5.3 for an environment class, member category and
    design life (100, 50 or 30 years, 4.5.1)"""
    return MIN_CONCRETE_CLASS[environment(env)][_column(member, life)]


def _grade_of(c) -> int:
    """the class number of a Concrete, a number or a string such as "C40" (not limited to table 3.1.3)"""
    if isinstance(c, Concrete):
        return c.grade
    return int(c) if isinstance(c, (int, float)) else int(str(c).upper().lstrip("C"))


def durability(concrete, env, *, member: str = "superstructure", life: int = 100, steel=None,
               prestressed: bool | None = None) -> Check:
    """minimum concrete class for durability (4.5.3) together with the strength minima of 3.1.2. concrete: an
    RCSection (its concrete, bars and tendons decide 3.1.2), a Concrete or a class ("C40", 40); steel and prestressed
    complete 3.1.2 when no section is given. env: environment class (table 4.5.2); member: "superstructure",
    "substructure" or "foundation"; life: design life in years. The other measures of 4.5.4 (cover: section_detailing;
    multiple protection of the prestressing system, impermeability and frost-resistance grades, drainage and
    ventilation) are listed in notes. Returns a detailing report."""
    e = environment(env)
    need = min_concrete_class(e, member, life)
    if isinstance(concrete, RCSection):
        c = concrete.rules_concrete or concrete.concrete
        g, rules = c.grade, concrete.material_rules()
        prestressed = concrete.prestressed
    else:
        g = _grade_of(concrete)
        rules = []
        s = None if steel is None else (steel if isinstance(steel, Rebar) else _rebar(steel))
        if s is not None and s.fsk >= 400 and g < 30:
            rules.append(f"C{g} with {s.name}: at least C30 for bars of 400 MPa and above (3.1.2)")
        if prestressed and g < 40:
            rules.append(f"prestressed concrete in C{g}: at least C40 (3.1.2)")
        if g < 25:
            rules.append(f"C{g}: reinforced concrete at least C25 (3.1.2)")
    chk = Check.report(f"durability, environment {e} (4.5)", "JTG 3362-2018 4.5.2, 4.5.3, 3.1.2",
                       {"environment": e, "member": member_category(member), "life": life, "class": g,
                        "min_class_4.5.3": need})
    if g < need:
        chk.warnings.append(f"C{g} below C{need} for environment {e}, {member_category(member)}, {life} years "
                            "(table 4.5.3)")
    chk.warnings += rules
    chk.notes.append("4.5.4: cover per 9.1.1 (section_detailing); impermeability and frost-resistance grades where "
                     "required; forms and details that drain and ventilate (not checked by numbers)")
    if prestressed:
        chk.notes.append("4.5.4 item 2: multiple protection of the prestressing system (tendons, couplers, anchorages)")
    return chk


# ── 9.1.1–9.1.3 cover and bundles ───────────────────────────────────────────
def cover_min(env, member: str = "superstructure", life: int = 100, *, precast: bool = False,
              bedding: bool = False) -> float:
    """minimum cover cmin (mm) of the outermost bars (longitudinal, stirrups or distribution bars), table 9.1.1:
    precast=True for members cast in a factory (note 2: 5 mm less, not below 20 mm); bedding=True for pile caps and
    foundations cast on a bedding layer or against formwork (note 3: 20 mm less, not below 30 mm). The cover is
    measured from the outer surface of the bar. Items 1 and 2 (cover ≥ the bar diameter, ≥ half the duct diameter)
    are checked by section_detailing."""
    cat = member_category(member)
    c = float(COVER_MIN[environment(env)][_column(cat, life)])
    if precast:
        c = max(c - 5.0, 20.0)
    if bedding:
        if cat != "foundation":
            raise ValueError("table 9.1.1 note 3 (bedding, formwork) is for pile caps and foundations")
        c = max(c - 20.0, 30.0)
    return c


def bundle_diameter(n: int, d: float) -> float:
    """equivalent diameter de = √n d of a bundle of n bars of diameter d (9.1.3)"""
    return math.sqrt(n) * d


def _bundle_rules(n: int, d: float, chk: Check, clause="9.1.3"):
    if n <= 1:
        return
    if d > 36:
        chk.warnings.append(f"bars of a bundle {d:g} mm > 36 mm ({clause})")
    nmax = 3 if d <= 28 else 2
    if n > nmax:
        chk.warnings.append(f"{n} bars of {d:g} mm in a bundle: at most {nmax} ({clause})")


def _rows(sec: RCSection, mask):
    """bar rows of the bars in mask: [(depth, count, max diameter, area)], bars at the same depth merged"""
    out = []
    a, z, dia = sec.bars[mask, 0], sec.bars[mask, 1], sec.bar_d[mask]
    for zz in sorted(set(np.round(z, 6))):
        k = np.isclose(z, zz, atol=1e-6)
        dk = dia[k]
        n = float(np.sum(a[k] / (math.pi * dk * dk / 4))) if not np.isnan(dk).any() else math.nan
        out.append((float(zz), n, float(np.max(dk)) if not np.isnan(dk).any() else math.nan, float(a[k].sum())))
    return out


def clear_spacing_min(d: float, layers: int = 1) -> float:
    """smallest clear spacing (mm) between the main bars of a flexural member and between their layers (9.3.3):
    ≥ 30 mm and ≥ d with up to 3 layers, ≥ 40 mm and ≥ 1.25 d with more; d of a bundle is its equivalent diameter"""
    return max(30.0, d) if layers <= 3 else max(40.0, 1.25 * d)


def _intervals(sec: RCSection):
    """[(top, bottom, width)] between consecutive boundaries of the section's pieces, the width taken inside each
    interval (constant there for rectangles), never at a boundary, where the pieces' open intervals give 0"""
    zs = {0.0, float(sec.h)}
    for p in sec.pieces:
        if isinstance(p, Rect):
            zs |= {float(p.top), float(p.bottom)}
        elif isinstance(p, Circle):
            zs |= {p.zc - p.D / 2, p.zc + p.D / 2}
    zs = sorted(z for z in zs if 0.0 <= z <= sec.h)
    return [(a, b, sec.width((a + b) / 2)) for a, b in zip(zs[:-1], zs[1:]) if b - a > 1e-9]


def _narrowest(sec: RCSection) -> float:
    """web width b: the narrowest width of the section (sum of the webs of a box), from the pieces' boundaries"""
    return min(w for _, _, w in _intervals(sec) if w > 0)


def _width_at(sec: RCSection, z: float) -> float:
    """width at depth z; on a boundary between pieces the narrower side (a bar row exactly at a flange's face)"""
    ws = [w for a, b, w in _intervals(sec) if a - 1e-9 <= z <= b + 1e-9 and w > 0]
    return min(ws) if ws else 0.0


def _web_zone(sec: RCSection):
    """(top, bottom) depths of the web: the intervals whose width is within 1 % of the narrowest, from the first to
    the last of them"""
    iv = _intervals(sec)
    b = _narrowest(sec)
    k = [i for i, (_, _, w) in enumerate(iv) if 0 < w <= 1.01 * b]
    return iv[k[0]][0], iv[k[-1]][1]


def cracked_axis(sec: RCSection, hogging: bool = False) -> float:
    """depth (mm, from the top face) of the neutral axis of a reinforced-concrete section cracked in pure bending:
    concrete in compression on one side, every bar αES As (the transformed section of 6.4, 6.5, 7.2.4); for reinforced
    concrete it does not depend on the moment. hogging: the top face in tension. Used for the tension zone of 9.3.7."""
    s = sec.flipped() if hogging else sec
    aE = s.steel.Es / s.concrete.Ec
    a, d = s.bars[:, 0], s.bars[:, 1]

    def f(x):
        A, Q = s.part(x)
        return x * A - Q + aE * float((a * (x - d)).sum())

    x = brentq(f, 1e-9 * s.h, s.h)
    return s.h - x if hogging else x


def _need(chk: Check, name: str, value: float, limit: float, kind: str, clause: str, unit: str = "mm",
          should: bool = False):
    """value against limit ("min": value ≥ limit, "max": value ≤ limit); the record goes to details, a miss to
    warnings; should=True marks a "should" (宜) rule in the warning"""
    chk.details[name] = {"value": value, "limit": limit}
    tol = _TOL * max(abs(limit), 1.0)
    ok = value >= limit - tol if kind == "min" else value <= limit + tol
    if not ok:
        u = f" {unit}" if unit else ""
        sh = ", should" if should else ""
        chk.warnings.append(f"{name} {_f(value)}{u} {'<' if kind == 'min' else '>'} {_f(limit)}{u} ({clause}{sh})")
    return ok


def _f(v: float) -> str:
    if isinstance(v, float) and not v.is_integer():
        return f"{v:.3g}" if abs(v) < 1 else f"{v:.1f}"
    return f"{v:g}"


def section_detailing(sec: RCSection, *, env=None, member: str = "superstructure", life: int = 100,
                      precast: bool = False, bedding: bool = False, stirrup_d: float | None = None,
                      outer_d=None, side_cover: float | None = None, bundle=1, hogging: bool = False,
                      welded_frame: bool = False, webs: int | None = None, near_support: bool = False,
                      thin_web: bool = False, neutral_axis: float | None = None, tendon_d: float | None = None,
                      tendon_side_cover: float | None = None, cover_mesh=None, surface_mesh=None,
                      main_rows=None) -> Check:
    """detailing of a section's longitudinal steel: cover (9.1.1 items 1–3, 9.1.2), bundles (9.1.3), clear spacing of
    the main bars and their layers (9.3.3), welded frames (9.3.11 item 4), the web's side bars (9.3.7), and with env the
    concrete class of 4.5.3 and 3.1.2.

    The rows are the section's bar points, each at its own depth (a row given as one lumped point counts as one layer);
    the main bars are the edge rows of RCSection.edge_masks on the bottom and top faces (within min(0.1 h, 150 mm) of the
    outermost row), the others side (web) bars; main_rows = (bottom, top) or bottom alone: the number of rows from each
    face that are main bars, when they spread further (e.g. four layers; a note points out a row just beyond the band
    that looks like one).
    stirrup_d: stirrups around the main bars (they set the horizontal room); outer_d: diameter of the steel outside the
    main bars at the (bottom, top) faces, default stirrup_d on both (0 without stirrups: the main bars are outermost);
    side_cover: cover of the outermost steel on the sides, checked against table 9.1.1 and item 1 when given (the
    horizontal room takes the bottom one otherwise). bundle: bars per bundle of the tension face's main bars, or
    (bottom, top). hogging: the top face is in tension (9.1.3 mesh, tension zone of 9.3.7). welded_frame: the main bars
    are welded frames (9.3.11: ≤ 6 layers, d ≤ 32 mm; layers in contact, so no clear spacing between them). webs: number
    of webs for the side bars of 9.3.7 (default 1 for T and I sections, 2 for boxes, none for other shapes);
    near_support: a high-shear zone or a prestressed anchorage zone (side bars at 100–150 mm); thin_web: a thin web
    (the commentary's upper value 0.002 b h, also taken with welded frames); neutral_axis: depth from the top face of
    the boundary of the tension zone for 9.3.7, default the cracked section's axis for reinforced concrete
    (cracked_axis) and the gross centroid for prestressed sections. tendon_d: nominal diameter of pretensioned tendons;
    tendon_side_cover: cover of the ducts (post-tensioned) or tendons (pretensioned) on the sides (item 2; else top
    and bottom only). cover_mesh: (d, spacing, cover) of a mesh in a cover over 50 mm (9.1.2); surface_mesh: (d along,
    spacing along, d across, spacing across, overhang beyond the bundles) of the surface mesh over bars or bundles above
    36 mm in the tension zone (9.1.3)."""
    chk = Check.report("section detailing (9.1.1–9.1.3, 9.3.3, 9.3.7, 9.3.11)", "JTG 3362-2018 9.1, 9.3")
    if env is not None:
        d = durability(sec, env, member=member, life=life)
        chk.warnings += d.warnings
        chk.details["min_class_4.5.3"] = d.details["min_class_4.5.3"]
        cmin = cover_min(env, member, life, precast=precast, bedding=bedding)
        chk.details["cmin"] = cmin
    else:
        cmin = None
        chk.warnings += sec.material_rules()
        chk.notes.append("no environment class (env=…): table 9.1.1 and 4.5.3 not checked")
    if np.isnan(sec.bar_d).any():
        chk.notes.append("bars without a diameter (points given without one): skipped in the cover and spacing rules")
    ds = 0.0 if stirrup_d is None else float(stirrup_d)
    ob, ot = (ds, ds) if outer_d is None else ((outer_d, outer_d) if np.isscalar(outer_d) else tuple(outer_d))
    nbun = ((1, bundle) if hogging else (bundle, 1)) if np.isscalar(bundle) else tuple(bundle)
    if max(nbun) > 1:
        chk.details["bundle"] = {"bottom": nbun[0], "top": nbun[1]}

    # circular sections: cover only (column spacing: 9.6.1)
    if sec.kind in ("circle", "annulus"):
        r, d0 = sec.h / 2, float(np.nanmax(sec.bar_d)) if not np.isnan(sec.bar_d).all() else math.nan
        if not math.isnan(d0):
            c = r - sec.ring - d0 / 2
            _need(chk, "cover of the longitudinal bars", c, d0, "min", "9.1.1 item 1")
            if cmin is not None:
                _need(chk, "cover of the outermost steel", c - ob, cmin, "min", "table 9.1.1")
            _cover_mesh(chk, c, cover_mesh)
        chk.notes.append("circular section: clear spacing and hoops of columns are 9.6.1")
        return chk

    bot, top = _main_masks(sec, main_rows)
    nb_rows = (main_rows, None) if np.isscalar(main_rows) or main_rows is None else tuple(main_rows)
    tension = "top" if hogging else "bottom"
    c_out_bottom = None
    spaced = False
    for k, (face, mask) in enumerate((("bottom", bot), ("top", top))):
        n_b = nbun[k]
        rows = [r for r in _rows(sec, mask) if not math.isnan(r[2])]
        if not rows:
            continue
        d_max = max(r[2] for r in rows)
        dd = bundle_diameter(n_b, d_max)
        out = rows[-1] if face == "bottom" else rows[0]  # the row nearest the face
        c = (sec.h - out[0] if face == "bottom" else out[0]) - out[2] / 2
        _need(chk, f"cover of the {face} bars", c, bundle_diameter(n_b, out[2]), "min", "9.1.1 item 1")
        oc = c - (ob if face == "bottom" else ot)
        if face == "bottom":
            c_out_bottom = oc
        if cmin is not None:
            _need(chk, f"cover of the outermost steel at the {face}", oc, cmin, "min", "table 9.1.1")
        if side_cover is not None:  # the main bars sit inside the stirrups on the sides
            _need(chk, f"side cover of the {face} bars", side_cover + ds, dd, "min", "9.1.1 item 1")
        _cover_mesh(chk, c, cover_mesh, face)
        # bundles and big bars (9.1.3)
        _bundle_rules(n_b, d_max, chk)
        if face == tension and dd > 36:
            _surface_mesh(chk, dd, surface_mesh)
        # layers and clear spacing (9.3.3, 9.3.11)
        layers = len(rows)
        if welded_frame:
            _need(chk, f"layers of the welded frame ({face})", layers, 6, "max", "9.3.11 item 4", unit="", should=True)
            _need(chk, f"bars of the welded frame ({face})", d_max, 32, "max", "9.3.11 item 4", should=True)
        side = side_cover if side_cover is not None else (c_out_bottom if c_out_bottom is not None else oc)
        for z, n, dmax, _ in rows:
            nu = n / n_b
            if nu < 1.5:
                continue
            du = bundle_diameter(n_b, dmax)
            s = (_width_at(sec, z) - 2 * (side + ds) - nu * du) / (nu - 1)
            _need(chk, f"clear spacing of the {face} bars at depth {_f(z)}", s, clear_spacing_min(du, layers), "min",
                  "9.3.3")
            spaced = True
        if not welded_frame:
            for (z1, _, d1, _), (z2, _, d2, _) in zip(rows[:-1], rows[1:]):
                du = bundle_diameter(n_b, max(d1, d2))
                gap = z2 - z1 - (du if n_b > 1 else (d1 + d2) / 2)
                _need(chk, f"clear distance of the {face} layers at depths {_f(z1)}–{_f(z2)}", gap,
                      clear_spacing_min(du, layers), "min", "9.3.3")
        chk.details[f"layers ({face})"] = layers
        chk.details[f"spacing limit ({face})"] = clear_spacing_min(dd, layers)
        if nb_rows[k] is None:
            _next_layer_note(chk, sec, mask, rows, face, n_b)
    if side_cover is None:
        chk.notes.append("side faces not checked (side_cover=…); the horizontal room takes the bottom cover of the "
                         "outermost steel")
    elif cmin is not None:
        _need(chk, "cover of the outermost steel at the sides", side_cover, cmin, "min", "table 9.1.1")
    if spaced:
        chk.notes.append("9.3.3 horizontal clear spacing: bars spread evenly between the stirrups across the width at "
                         "their depth, the largest the layout allows (program convention)")

    # ducts and pretensioned tendons (9.1.1 item 2)
    if sec.has_internal:
        z, duct = sec.tendons[:, 1], sec.tendons[:, 3]
        near = np.minimum(z, sec.h - z)
        if sec.pretensioned:
            if tendon_d is None:
                chk.notes.append("pretensioned tendons: give tendon_d for the cover ≥ the tendon diameter (9.1.1 item 2)")
            else:
                _need(chk, "cover of the pretensioned tendons (top, bottom)", float((near - tendon_d / 2).min()),
                      tendon_d, "min", "9.1.1 item 2")
                if tendon_side_cover is not None:
                    _need(chk, "side cover of the pretensioned tendons", tendon_side_cover, tendon_d, "min",
                          "9.1.1 item 2")
        elif (duct > 0).any():
            D = np.sqrt(4 * duct / math.pi)
            k = int(np.argmin(near - D))
            _need(chk, "cover of the ducts (top, bottom)", float(near[k] - D[k] / 2), float(D[k] / 2), "min",
                  "9.1.1 item 2")
            if tendon_side_cover is not None:
                _need(chk, "side cover of the ducts", tendon_side_cover, float(D.max() / 2), "min", "9.1.1 item 2")
        if tendon_side_cover is None:
            chk.notes.append("tendons: covers to the top and bottom faces only (tendon_side_cover=… for the sides)")

    # side bars of webs (9.3.7)
    nw = webs if webs is not None else {"tee": 1, "box": 2}.get(sec.kind)
    if nw:
        if webs is None and sec.kind == "box":
            chk.notes.append("box: 2 webs assumed for the side bars of 9.3.7 (webs=…)")
        _web_bars(chk, sec, bot, top, nw, hogging, welded_frame or thin_web, near_support, neutral_axis)
    chk.warnings = list(dict.fromkeys(chk.warnings))
    return chk


def _main_masks(sec: RCSection, main_rows):
    """(bottom, top) masks of the main bars: RCSection.edge_masks, or the main_rows rows nearest each face"""
    bot, top = sec.edge_masks()
    if main_rows is None:
        return bot, top
    nb, nt = (main_rows, None) if np.isscalar(main_rows) else main_rows
    z, zg = sec.bars[:, 1], sec.zg
    if nb is not None:
        zs = sorted({round(float(v), 6) for v in z[z > zg]}, reverse=True)[:nb]
        bot = (z > zg) & np.isin(np.round(z, 6), zs)
    if nt is not None:
        zs = sorted({round(float(v), 6) for v in z[z <= zg]})[:nt]
        top = (z <= zg) & np.isin(np.round(z, 6), zs)
    return bot, top


def _next_layer_note(chk: Check, sec: RCSection, mask, rows, face: str, n_b: int):
    """a note when the next row beyond the main bars, on the same side of the centroid, looks like another layer of
    them (at least half their diameter, clear distance under twice the 9.3.3 limit): main_rows=… would count it"""
    z, zg, dia = sec.bars[:, 1], sec.zg, sec.bar_d
    side = (z > zg) if face == "bottom" else (z <= zg)
    cand = side & ~mask & ~np.isnan(dia)
    if not cand.any():
        return
    inner = rows[0] if face == "bottom" else rows[-1]
    k = np.nonzero(cand)[0]
    k = k[np.argmax(z[k])] if face == "bottom" else k[np.argmin(z[k])]
    d_main = max(r[2] for r in rows)
    gap = abs(inner[0] - z[k]) - (inner[2] + dia[k]) / 2
    if dia[k] >= 0.5 * d_main and gap < 2 * clear_spacing_min(bundle_diameter(n_b, d_main), len(rows)):
        chk.notes.append(f"the row at depth {_f(float(z[k]))} ({dia[k]:g} mm, {_f(float(gap))} mm clear of the {face} "
                         f"main bars) lies beyond the edge band and is taken as a side bar; if it is another layer of "
                         "main bars, main_rows=… counts it (9.3.3)")


def _cover_mesh(chk: Check, c: float, mesh, face: str = ""):
    """9.1.2: cover of the longitudinal main bars over 50 mm calls for measures, e.g. a mesh (d ≥ 6, s ≤ 100, its own
    cover ≥ 25 mm)"""
    if c <= 50 + _TOL:
        return
    where = f" at the {face}" if face else ""
    if mesh is None:
        chk.warnings.append(f"cover of the main bars{where} {_f(c)} mm > 50 mm: measures against cracking and spalling "
                            "of the cover, e.g. a mesh in it (9.1.2, should)")
        return
    d, s, cm = mesh
    _need(chk, "bars of the cover mesh", d, 6, "min", "9.1.2")
    _need(chk, "spacing of the cover mesh", s, 100, "max", "9.1.2")
    _need(chk, "cover of the cover mesh", cm, 25, "min", "9.1.2", should=True)


def _surface_mesh(chk: Check, d: float, mesh):
    """9.1.3: a surface mesh over bars or bundles above 36 mm in the tension zone"""
    if mesh is None:
        chk.warnings.append(f"bars or bundles of {_f(d)} mm > 36 mm in the tension zone: a surface mesh "
                            "(9.1.3, should; surface_mesh=…)")
        return
    da, sa, dc, sc, over = mesh
    _need(chk, "surface mesh: bars along the bundles", da, 10, "min", "9.1.3")
    _need(chk, "surface mesh: spacing along", sa, 100, "max", "9.1.3")
    _need(chk, "surface mesh: bars across", dc, 6, "min", "9.1.3")
    _need(chk, "surface mesh: spacing across", sc, 100, "max", "9.1.3")
    _need(chk, "surface mesh: overhang beyond the bundles", over, 5 * d, "min", "9.1.3")


def _web_bars(chk: Check, sec: RCSection, bot, top, webs: int, hogging: bool, thin: bool, near_support: bool,
              neutral_axis: float | None):
    """9.3.7: side bars of 6–8 mm on both faces of each web, (0.001–0.002) b h per web (the upper value for thin webs,
    commentary), spacing ≤ the web width and 200 mm in the tension zone, ≤ 300 mm in the compression zone, 100–150 mm
    near supports and in anchorage zones"""
    side = ~(bot | top)
    b, h = _narrowest(sec), sec.h
    bw = b / webs
    z1, z2 = _web_zone(sec)
    tz = sec.bars[top, 1]
    bz = sec.bars[bot, 1]
    lo = max([z1] + [z for z in tz if z <= z2])  # upper boundary of the side-bar field
    hi = min([z2] + [z for z in bz if z >= z1])  # lower boundary
    rows = [r for r in _rows(sec, side) if lo < r[0] < hi]
    area = sum(r[3] for r in rows)
    rho = area / (b * h)
    _need(chk, "side bars of the webs, As/(b h)", rho, 0.002 if thin else 0.001, "min", "9.3.7", unit="", should=True)
    for z, n, d, _ in rows:
        if not math.isnan(d):
            if d < 6:
                chk.warnings.append(f"side bars at depth {_f(z)}: {d:g} mm < 6 mm (9.3.7)")
            elif d > 8:
                chk.notes.append(f"side bars at depth {_f(z)}: {d:g} mm, above the 6–8 mm 9.3.7 names")
    if neutral_axis is not None:
        na, how = float(neutral_axis), "given"
    elif sec.prestressed:
        na, how = sec.zg, "the gross centroid (prestressed section, an approximation)"
    else:
        na, how = cracked_axis(sec, hogging), "the cracked section's axis (reinforced concrete)"
    chk.details["9.3.7 neutral axis"] = na
    pts = [lo] + [r[0] for r in rows] + [hi]
    worst = {}
    for a, c in zip(pts[:-1], pts[1:]):
        in_tension = (a < na) if hogging else (c > na)  # any part of the gap in the tension zone
        lim = min(bw, 200.0) if in_tension else 300.0
        if near_support:
            lim = min(lim, 150.0)
        key = ("tension" if in_tension else "compression") + " zone"
        if c - a - lim > worst.get(key, (-math.inf,))[0]:
            worst[key] = (c - a - lim, c - a, lim)
    for key, (_, s, lim) in worst.items():
        _need(chk, f"spacing of the side bars, {key}", s, lim, "max", "9.3.7", should=near_support and lim == 150.0)
    chk.notes.append(f"9.3.7 tension zone {'above' if hogging else 'below'} depth {_f(na)}, {how}; a gap reaching into "
                     "it takes the tension limit")


# ── 9.1.4 anchorage ─────────────────────────────────────────────────────────
# table 9.1.4: la / d (p98); columns C25, C30, C35, ≥C40 (None: not tabulated)
#                compression (straight end)   tension, straight end      tension, hooked end
ANCHORAGE = {
    "HPB300": ((45, 40, 38, 35), (None, None, None, None), (40, 35, 33, 30)),
    "HRB400": ((None, 30, 28, 25), (None, 35, 33, 30), (None, 30, 28, 25)),
    "HRB500": ((None, 35, 33, 30), (None, 45, 43, 40), (None, 35, 33, 30)),
}
_GROUP = {"HPB300": "HPB300", "HRB400": "HRB400", "HRBF400": "HRB400", "RRB400": "HRB400", "HRB500": "HRB500"}


def _steel(steel) -> str:
    n = steel.name if isinstance(steel, Rebar) else str(steel).upper()
    if n not in _GROUP:
        raise ValueError(f"reinforcing steel {steel!r}: one of {', '.join(_GROUP)}")
    return _GROUP[n]


def _grade(concrete) -> int:
    return concrete.grade if isinstance(concrete, Concrete) else _concrete(concrete).grade


def anchorage_length(d: float, steel="HRB400", concrete="C30", *, compression: bool = False, hook: bool = False,
                     epoxy: bool = False, disturbed: bool = False) -> float:
    """minimum anchorage length la (mm) of a bar whose strength is fully used, table 9.1.4: d nominal diameter (mm);
    compression=True: bars in compression (a hook does not count, as the commentary explains); hook=True: tension bars
    with an end hook (la is then the projected length including the hook, note 5; plain HPB300 bars in tension must
    have hooks). epoxy: epoxy-coated tension bars ×1.25 (note 3); disturbed: concrete liable to be disturbed while it
    hardens ×1.25 (note 4); both together multiply (program convention). Bundles: bundle_anchorage."""
    s, g = _steel(steel), _grade(concrete)
    col = 0 if g <= 25 else 1 if g <= 30 else 2 if g <= 35 else 3
    comp, straight, hooked = ANCHORAGE[s]
    k = (comp if compression else hooked if hook else straight)[col]
    if k is None:
        if s == "HPB300":
            raise ValueError("table 9.1.4: plain HPB300 bars in tension are anchored with hooks (hook=True)")
        raise ValueError(f"table 9.1.4 has no C{g} column for {s} (3.1.2: at least C30)")
    la = k * d
    if epoxy and not compression:
        la *= 1.25
    if disturbed:
        la *= 1.25
    return la


def bundle_anchorage(n: int, d: float, steel="HRB400", concrete="C30", *, compression: bool = False,
                     **kw) -> list[float]:
    """anchorage lengths (mm) of the n bars of a bundle, table 9.1.4 note 2: bundles in compression and bundles in
    tension with de = √n d ≤ 28 mm are anchored as one bar of diameter de (all bars cut at one point); tension bundles
    with de > 28 mm are cut bar by bar at 1.3, 2.6, 3.9 times the single bar's la from the start of the anchorage"""
    de = bundle_diameter(n, d)
    if compression or de <= 28 + _TOL:
        return [anchorage_length(de, steel, concrete, compression=compression, **kw)] * n
    la = anchorage_length(d, steel, concrete, **kw)
    return [1.3 * (k + 1) * la for k in range(n)]


# ── 9.1.5, 9.1.6 hooks and bends ────────────────────────────────────────────
def hook_min(d: float, steel="HRB400", angle: int = 135, *, epoxy: bool = False) -> tuple[float, float]:
    """(smallest inner bend diameter D, smallest straight extension) in mm of an end hook of a tension bar, table
    9.1.5: HPB300 180° (2.5 d, 3 d); HRB400, HRB500, HRBF400, RRB400 135° (5 d, 5 d) and 90° (5 d, 10 d). Epoxy-coated
    bars (table note): D ≥ 5 d up to d = 20 mm, ≥ 6 d above, and a straight part ≥ 5 d (program convention: for both
    diameters, as the note's last clause reads)."""
    s = _steel(steel)
    table = {"HPB300": {180: (2.5, 3.0)}, "HRB400": {135: (5.0, 5.0), 90: (5.0, 10.0)}}
    rows = table["HPB300" if s == "HPB300" else "HRB400"]
    if angle not in rows:
        raise ValueError(f"table 9.1.5: end hooks of {s} are {', '.join(f'{a}°' for a in rows)}")
    kD, kl = rows[angle]
    if epoxy:
        kD = max(kD, 5.0 if d <= 20 else 6.0)
        kl = max(kl, 5.0)
    return kD * d, kl * d


def bend_min(d: float) -> float:
    """smallest inner diameter (mm) of an intermediate bend of ≤ 90° of any bar: 20 d (table 9.1.5)"""
    return 20.0 * d


def stirrup_hook_min(ds: float, steel="HRB400", *, seismic: bool = False) -> tuple[float, float]:
    """(smallest inner bend diameter, smallest straight extension) in mm of a stirrup hook (9.1.6): 2.5 ds for HPB300,
    5 ds for HRB400 (and, program convention, the other ribbed bars), and larger than the diameter of the main bar it
    holds (stirrup_detailing checks that); straight part 5 ds, 10 ds in seismic structures. The hook may be bent 135°."""
    return (2.5 if _steel(steel) == "HPB300" else 5.0) * ds, (10.0 if seismic else 5.0) * ds


# ── 9.1.7–9.1.10 splices ────────────────────────────────────────────────────
# table 9.1.9: ls / d of tension lap splices (p100); columns C25, ≥C30
LAP = {"HPB300": (40, 35), "HRB400": (None, 45), "HRB500": (None, 50)}


def lap_length(d: float, steel="HRB400", concrete="C30", *, compression: bool = False, epoxy: bool = False,
               disturbed: bool = False, share: float = 0.0) -> float:
    """minimum lap length ls (mm) of a tied lap splice, 9.1.9 and table 9.1.9. Tension: the table value; ribbed bars
    +5 d above 25 mm and −5 d below 25 mm (note 1, the reduction is permitted and taken, except for epoxy-coated bars);
    +5 d where the concrete is liable to be disturbed while it hardens (note 2); ×1.5 for epoxy-coated bars (note 4,
    after notes 1 and 2; program convention: the longer of the two readings of notes 1 and 4); ×1.4 with more than
    25 % and up to 50 % of the bars lapped within 1.3 ls, ×1.6 above 50 % (item 2); at least 300 mm (note 3).
    Compression: 0.7 times the tension lap from the table and notes 1, 2 (program convention: note 4 concerns tension
    bars), ×1.4 with more than 50 % lapped, at least 200 mm. share: the lapped fraction of the bars' area within the
    zone (0–1). The lap can come out shorter than the anchorage length with disturbed concrete (notes 2 of table 9.1.9
    and 4 of table 9.1.4 differ); splice_detailing points that out."""
    s, g = _steel(steel), _grade(concrete)
    if not 0 <= share <= 1:
        raise ValueError("share: the lapped fraction of the bars' area, 0 … 1 (9.1.9 item 2)")
    k = LAP[s][0 if g <= 25 else 1]
    if k is None:
        raise ValueError(f"table 9.1.9 has no C{g} column for {s} (3.1.2: at least C30)")
    base = k * d
    if s != "HPB300":
        base += 5 * d if d > 25 else (-5 * d if d < 25 and not (epoxy and not compression) else 0.0)
    if disturbed:
        base += 5 * d
    if compression:
        return max(0.7 * base * (1.4 if share > 0.5 + _TOL else 1.0), 200.0)
    if epoxy:
        base *= 1.5
    f = 1.0 if share <= 0.25 + _TOL else 1.4 if share <= 0.5 + _TOL else 1.6
    return max(base * f, 300.0)


def bundle_lap(n: int, ls: float) -> tuple[float, float]:
    """lap splice of a bundle (9.1.9 item 4, figure 9.1.9-2): (stagger of the single-bar laps 1.3 ls, length of the
    additional through bar 1.3 (n + 1) ls), ls the lap length of one bar"""
    return 1.3 * ls, 1.3 * (n + 1) * ls


def weld_zone(d: float) -> float:
    """length l (mm) of the zone around a welded joint in which one bar may not have two joints and the share of
    welded bars is counted: 35 d and at least 500 mm (9.1.8 item 2)"""
    return max(35.0 * d, 500.0)


def splice_detailing(kind: str, d: float, *, steel="HRB400", concrete="C30", compression: bool = False,
                     member: str = "flexural", compression_member: bool = False, length: float | None = None,
                     share: float | None = None, joint_spacing: float | None = None, clear: float | None = None,
                     hooks: bool | None = None, epoxy: bool = False, disturbed: bool = False, bundle: int = 1,
                     single_sided: bool = False, splice_area: float | None = None, splice_steel=None,
                     precast: bool = False, cover: float | None = None, cmin: float | None = None) -> Check:
    """splices of reinforcing bars (9.1.7–9.1.10): kind "lap" (tied lap), "weld" (welded) or "mechanical" (couplers);
    d bar diameter (mm). Each rule is checked when its input is given.
    lap: not for the main bars of members in axial or small-eccentricity tension (member="tension"); bar diameter
    ≤ 28 mm, ≤ 32 mm for compression bars of compression members (compression_member=True) (9.1.7); length ≥ lap_length
    (share: lapped fraction within 1.3 ls; tension ≤ 25 %, compression ≤ 50 % preferably, beyond that the longer lap);
    joint_spacing (centre to centre of two laps of one bar) > 1.3 ls; clear distance at the lap ≥ d and 25 mm; HPB300
    laps in tension with hooks (9.1.9). bundle: bars per bundle (the stagger and the through bar of item 4 in details).
    A note gives the anchorage length when the lap comes out shorter (table 9.1.9 note 2 against table 9.1.4 note 4).
    weld: weld length ≥ 5 d double-sided, 10 d single-sided (single_sided=True, only where double-sided welding is
    impossible); splice bars (splice_steel) of the same grade with splice_area ≥ the bar's area; joint_spacing > l =
    max(35 d, 500); share of welded bars in the zone ≤ 50 % in tension (none in compression or between precast members,
    precast=True); clear distance at strap or lap welds ≥ d and 25 mm (9.1.8).
    mechanical: ribbed HRB400, HRB500, HRBF400, RRB400 only; cover of the couplers ≥ 20 mm and ≥ cmin (table 9.1.1,
    should); clear distance between couplers, or coupler and bar, ≥ 25 mm (9.1.10). Warnings of "should" rules say so."""
    k = kind.lower()
    s = _steel(steel)
    a = math.pi * d * d / 4
    if k == "lap":
        chk = Check.report(f"lap splice of {d:g} mm {s} (9.1.7, 9.1.9)", "JTG 3362-2018 9.1.7, 9.1.9")
        if member == "tension":
            chk.warnings.append("lap splices are not used for the main bars of members in axial or small-eccentricity "
                                "tension (9.1.7)")
        dmax = 32.0 if (compression and compression_member) else 28.0
        _need(chk, "bar diameter of a lap splice", d, dmax, "max", "9.1.7", should=True)
        sh = 0.0 if share is None else share
        ls = lap_length(d, s, concrete, compression=compression, epoxy=epoxy, disturbed=disturbed, share=sh)
        if share is not None:
            lim = 0.5 if compression else 0.25
            if share > lim + _TOL:
                chk.notes.append(f"{share:.0%} of the bars lapped within 1.3 ls, over the {lim:.0%} of 9.1.9 item 2: "
                                 "the longer lap is required")
        chk.details["ls_min"] = ls
        la = anchorage_length(d, s, concrete, compression=compression, hook=s == "HPB300" and not compression,
                              epoxy=epoxy, disturbed=disturbed)
        if ls < la - _TOL:
            chk.notes.append(f"the lap {_f(ls)} mm is shorter than the anchorage length la = {_f(la)} mm (table 9.1.9 "
                             "note 2 against table 9.1.4 note 4): taking la is the designer's call")
        if length is not None:
            _need(chk, "lap length", length, ls, "min", "9.1.9")
        ls_ref = ls if length is None else max(length, ls)
        if joint_spacing is not None:
            lz = 1.3 * ls_ref
            chk.details["distance between laps of one bar"] = {"value": joint_spacing, "limit": lz}
            if joint_spacing <= lz + _TOL:
                chk.warnings.append(f"two laps of one bar {_f(joint_spacing)} mm apart, within l = 1.3 ls = {_f(lz)} mm "
                                    "(9.1.9 item 2)")
        if clear is not None:
            _need(chk, "clear distance at the lap", clear, max(d, 25.0), "min", "9.1.9 item 3")
        if s == "HPB300" and not compression:
            if hooks is None:
                chk.notes.append("HPB300 laps in tension need hooks at the bar ends (table 9.1.9 note 5)")
            elif not hooks:
                chk.warnings.append("HPB300 laps in tension without hooks (table 9.1.9 note 5)")
        if bundle > 1:
            _bundle_rules(bundle, d, chk)
            stagger, through = bundle_lap(bundle, ls_ref)
            chk.details.update(bundle_stagger=stagger, bundle_through_bar=through)
        return chk
    if k == "weld":
        chk = Check.report(f"welded splice of {d:g} mm {s} (9.1.8)", "JTG 3362-2018 9.1.8")
        if single_sided:
            chk.notes.append("single-sided welds only where double-sided welding is impossible (9.1.8 item 1)")
        if length is not None:
            _need(chk, "weld length", length, (10 if single_sided else 5) * d, "min", "9.1.8 item 1")
        if splice_area is not None:
            _need(chk, "area of the splice bars", splice_area, a, "min", "9.1.8 item 1", unit="mm²")
        if splice_steel is not None and _steel(splice_steel) != s:
            chk.warnings.append(f"splice bars of {_steel(splice_steel)} on {s}: the same strength grade (9.1.8 item 1)")
        lz = weld_zone(d)
        chk.details["zone_l"] = lz
        if joint_spacing is not None:
            chk.details["distance between joints of one bar"] = {"value": joint_spacing, "limit": lz}
            if joint_spacing <= lz + _TOL:
                chk.warnings.append(f"two joints of one bar {_f(joint_spacing)} mm apart, within l = {_f(lz)} mm "
                                    "(9.1.8 item 2)")
        if share is not None and not compression and not precast:
            _need(chk, "share of welded bars in the zone", share, 0.5, "max", "9.1.8 item 2", unit="", should=True)
        if clear is not None:
            _need(chk, "clear distance at the welded joint", clear, max(d, 25.0), "min", "9.1.8 item 3")
        return chk
    if k == "mechanical":
        chk = Check.report(f"mechanical splice of {d:g} mm {s} (9.1.10)", "JTG 3362-2018 9.1.10")
        if s == "HPB300":
            chk.warnings.append("mechanical splices are for ribbed bars (HRB400, HRB500, HRBF400, RRB400), not HPB300 "
                                "(9.1.10)")
        if cover is not None:
            _need(chk, "cover of the couplers", cover, 20.0, "min", "9.1.10")
            if cmin is not None:
                _need(chk, "cover of the couplers against table 9.1.1", cover, cmin, "min", "9.1.10", should=True)
        if clear is not None:
            _need(chk, "clear distance at the couplers", clear, 25.0, "min", "9.1.10")
        chk.notes.append("couplers per JGJ 107-2016, JG/T 163-2013 and SL 677-2014 (9.1.10, 9.1.11)")
        return chk
    raise ValueError("kind: 'lap', 'weld' or 'mechanical'")


# ── 9.3.12, 9.1.6 stirrups ──────────────────────────────────────────────────
def stirrup_detailing(sec: RCSection, stirrups, *, stirrup_steel=None, main_d: float | None = None,
                      compression_d: float | None = None, corner_distance: float | None = None, lap=None,
                      support_spacing: float | None = None, end_distance: float | None = None,
                      end_cover: float | None = None, cmin: float | None = None,
                      junction_distance: float | None = None, hook=None, seismic: bool = False) -> Check:
    """stirrups of a beam (9.3.12) and their hooks (9.1.6). stirrups: (legs, diameter, spacing) as for jtg.shear;
    stirrup_steel: when it differs from the longitudinal bars. main_d: diameter of the main bars (default the largest of
    the section's edge rows). Item 1: diameter ≥ 8 mm and ≥ main_d / 4, ρsv ≥ 0.14 % (HPB300) or 0.11 % (HRB400; HRB500
    taken alike). Item 2: compression_d (the stirrups hold compression bars needed by calculation, or the hogging zone
    of continuous and cantilever beams): closed stirrups; corner_distance: the largest distance of such a bar in the
    row from a stirrup corner ≤ max(150 mm, 15 ds), else additional ties. Item 3: spacing ≤ h/2 and 400 mm, ≤ 15
    compression_d; lap = (kind, lapped bar diameter, stirrup spacing along the lap) with kind "tension" (≤ 5 d and
    100 mm) or "compression" (≤ 10 d and 200 mm); support_spacing: spacing within one beam depth from the support
    centre (≤ 100 mm, should). Item 4: end_distance of the first stirrup from the end face, one cover (end_cover) at
    most, and with cmin (jtg.cover_min) at least table 9.1.1, as it is that stirrup's cover; junction_distance: the
    stirrup nearest a beam-beam or beam-column junction ≤ 50 mm from it (should). hook = (inner bend diameter, straight
    extension[, angle]): 9.1.6 (stirrup_hook_min; larger than main_d). Item 1 is written for reinforced-concrete beams:
    on a prestressed section a note points to 9.4.1 (web stirrups of prestressed girders)."""
    legs, ds, s = stirrups
    st = sec.steel if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar)
                                                  else _rebar(stirrup_steel))
    chk = Check.report("stirrups (9.3.12, 9.1.6)", "JTG 3362-2018 9.3.12, 9.1.6")
    if sec.prestressed:
        chk.notes.append("prestressed girder: the web stirrups also by 9.4.1 (≥ 10 mm in T and I webs, ≥ 12 mm in box "
                         "webs, ≤ 200 mm, closed within one depth of the support at ≤ 120 mm)")
    if main_d is None:
        bot, top = sec.edge_masks()
        dd = sec.bar_d[bot | top]
        main_d = float(np.nanmax(dd)) if len(dd) and not np.isnan(dd).all() else None
    b, h = _narrowest(sec), sec.h
    _need(chk, "stirrup diameter", ds, 8.0, "min", "9.3.12 item 1")
    if main_d is None:
        chk.notes.append("main bar diameter unknown (main_d=…): ds ≥ d/4 not checked")
    else:
        _need(chk, "stirrup diameter (≥ main bar / 4)", ds, main_d / 4, "min", "9.3.12 item 1")
    rho = legs * math.pi * ds * ds / 4 / (s * b)
    _need(chk, "stirrup ratio ρsv", rho, st.rho_sv_min, "min", "9.3.12 item 1", unit="")
    _need(chk, "stirrup spacing", s, min(h / 2, 400.0), "max", "9.3.12 item 3")
    if compression_d is not None:
        chk.notes.append("stirrups holding compression bars: closed (9.3.12 item 2)")
        _need(chk, "stirrup spacing (compression bars held)", s, min(15 * compression_d, 400.0), "max",
              "9.3.12 item 3")
    if corner_distance is not None:
        lim = max(150.0, 15 * ds)
        chk.details["corner distance"] = {"value": corner_distance, "limit": lim}
        if corner_distance > lim + _TOL:
            chk.warnings.append(f"compression bar {_f(corner_distance)} mm from a stirrup corner > {_f(lim)} mm: "
                                "additional ties (9.3.12 item 2)")
    if lap is not None:
        kind, dl, sl = lap
        if kind not in ("tension", "compression"):
            raise ValueError("lap: ('tension' or 'compression', bar diameter, stirrup spacing)")
        lim = min(5 * dl, 100.0) if kind == "tension" else min(10 * dl, 200.0)
        _need(chk, f"stirrup spacing along a {kind} lap", sl, lim, "max", "9.3.12 item 3")
    if support_spacing is not None:
        _need(chk, "stirrup spacing within h of the support", support_spacing, 100.0, "max", "9.3.12 item 3",
              should=True)
    if end_distance is not None:
        if end_cover is None:
            chk.notes.append("end_cover not given: the first stirrup's position not checked (9.3.12 item 4)")
        else:
            _need(chk, "first stirrup from the end face", end_distance, end_cover, "max", "9.3.12 item 4")
        if cmin is not None:
            _need(chk, "cover of the first stirrup at the end face", end_distance, cmin, "min", "9.3.12 item 4, "
                  "table 9.1.1")
    if junction_distance is not None:
        _need(chk, "stirrup from the junction face", junction_distance, 50.0, "max", "9.3.12 item 4", should=True)
    if hook is not None:
        D, straight = hook[:2]
        Dmin, lmin = stirrup_hook_min(ds, st, seismic=seismic)
        _need(chk, "stirrup hook bend diameter", D, Dmin, "min", "9.1.6")
        _need(chk, "stirrup hook straight part", straight, lmin, "min", "9.1.6")
        if main_d is not None and D <= main_d + _TOL:
            chk.warnings.append(f"stirrup hook bend diameter {_f(D)} mm not larger than the main bar {main_d:g} mm "
                                "(9.1.6)")
        if len(hook) > 2 and hook[2] != 135:
            chk.notes.append(f"stirrup hook bent {hook[2]}°; 9.1.6 names 135°")
    return chk


# ── 9.3.14 curved soffits ───────────────────────────────────────────────────
def soffit_radius(chord: float, rise: float) -> float:
    """radius r = l/2 (1/(4β) + β), β = f/l, of a circular soffit with chord l and rise f (9.3.14-2; approximate for
    other curves)"""
    beta = rise / chord
    return chord / 2 * (1 / (4 * beta) + beta)


def curved_soffit(As: float, main_d: float, stirrups, *, r: float | None = None, chord: float | None = None,
                  rise: float | None = None, steel="HRB400", stirrup_steel=None) -> Check:
    """stirrups holding the tension bars near a concave curved soffit (9.3.14): spacing ≤ 10 main_d, diameter ≥ 8 mm,
    one leg Asv1 ≥ m As sv / (2 r) (9.3.14-1), m = fsd / fsv (the main bars' to the stirrups' design strength), As the
    main-bar area held by one two-legged stirrup, r the soffit radius or soffit_radius(chord, rise). stirrups: (diameter,
    spacing) of the two-legged stirrups."""
    ds, sv = stirrups
    if r is None:
        if chord is None or rise is None:
            raise ValueError("give r, or chord and rise (9.3.14-2)")
        r = soffit_radius(chord, rise)
    ms = steel if isinstance(steel, Rebar) else _rebar(steel)
    ss = ms if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))
    m = ms.fsd / ss.fsv
    need = m * As * sv / (2 * r)
    chk = Check.report("stirrups at a curved soffit (9.3.14)", "JTG 3362-2018 9.3.14", {"r": r, "m": m})
    _need(chk, "stirrup leg area Asv1", math.pi * ds * ds / 4, need, "min", "9.3.14-1", unit="mm²")
    _need(chk, "stirrup spacing", sv, 10 * main_d, "max", "9.3.14")
    _need(chk, "stirrup diameter", ds, 8.0, "min", "9.3.14")
    chk.notes.append("crossing tension bars at a re-entrant corner: each anchored la beyond the crossing, into the "
                     "compression zone (9.3.14, not checked by numbers)")
    return chk


# ── 9.3.8–9.3.11 longitudinal bars of beams ─────────────────────────────────
def cutoff_lengths(d: float, h0: float, steel="HRB400", concrete="C30", *, compression: bool = False,
                   epoxy: bool = False, disturbed: bool = False) -> tuple[float | None, float]:
    """extensions (mm) of a bar cut off in a reinforced-concrete beam (9.3.8): tension bars (la + h0 beyond the section
    where the bar's strength is fully used, la by table 9.1.4 for tension (hooked for HPB300); 20 d, epoxy 25 d, beyond
    the section where it is no longer needed); compression bars cut in the span (None, 15 d, epoxy 20 d)"""
    if compression:
        return None, (20.0 if epoxy else 15.0) * d
    la = anchorage_length(d, steel, concrete, hook=_steel(steel) == "HPB300", epoxy=epoxy, disturbed=disturbed)
    return la + h0, (25.0 if epoxy else 20.0) * d


def cutoff_detailing(d: float, h0: float, *, steel="HRB400", concrete="C30", beyond_full: float | None = None,
                     beyond_needed: float | None = None, compression: bool = False, in_tension_zone: bool = False,
                     epoxy: bool = False, disturbed: bool = False) -> Check:
    """the extensions of a cut-off bar against cutoff_lengths (9.3.8); in_tension_zone: tension bars should not be cut
    in the tension zone"""
    full, needed = cutoff_lengths(d, h0, steel, concrete, compression=compression, epoxy=epoxy, disturbed=disturbed)
    chk = Check.report(f"cut-off of a {d:g} mm bar (9.3.8)", "JTG 3362-2018 9.3.8",
                       {"beyond_full_min": full, "beyond_needed_min": needed})
    if in_tension_zone and not compression:
        chk.warnings.append("tension bars cut off in the tension zone (9.3.8, should not)")
    if beyond_full is not None and full is not None:
        _need(chk, "extension beyond the fully used section", beyond_full, full, "min", "9.3.8")
    if beyond_needed is not None:
        _need(chk, "extension beyond the section where not needed", beyond_needed, needed, "min", "9.3.8")
    return chk


def support_detailing(n_bottom: int, n_through: int, *, d: float | None = None, extension: float | None = None,
                      steel="HRB400", epoxy: bool = False, hooks: bool | None = None) -> Check:
    """bottom tension bars at an end support of a reinforced-concrete beam (9.3.9): at least two, and at least 1/5 of
    the n_bottom bars, pass the support; the two outer ones run past it, turn up at a right angle to the top and join
    the top bars (not checked by numbers); the other straight bars between them extend ≥ 10 d past the support centre
    section (12.5 d epoxy), HPB300 with semicircular hooks"""
    chk = Check.report("bars over an end support (9.3.9)", "JTG 3362-2018 9.3.9")
    _need(chk, "bottom bars through the support", n_through, max(2.0, n_bottom / 5), "min", "9.3.9", unit="")
    if extension is not None:
        if d is None:
            raise ValueError("extension needs the bar diameter d (9.3.9)")
        _need(chk, "extension past the support", extension, (12.5 if epoxy else 10.0) * d, "min", "9.3.9")
    if _steel(steel) == "HPB300" and hooks is not None and not hooks:
        chk.warnings.append("HPB300 bars over the support without semicircular hooks (9.3.9)")
    chk.notes.append("the two outer bars run past the support, turn up at a right angle and join the top bars (9.3.9)")
    return chk


def bent_bar_detailing(d: float, h0: float, *, angle: float | None = None, start: float | None = None,
                       anchorage: float | None = None, zone: str = "tension", steel="HRB400", epoxy: bool = False,
                       hooks: bool | None = None, floating: bool = False) -> Check:
    """bent-up bars of a reinforced-concrete beam (9.3.10): angle preferably 45°; the bend in the tension zone starts
    ≥ h0/2 beyond the section where the bar is fully used (start); the end straight part (anchorage) ≥ 20 d in the
    tension zone, ≥ 10 d in the compression zone, +25 % for epoxy-coated bars; HPB300 with semicircular hooks; no
    floating bars (floating=True: bent bars not anchored as bars). The first bent bar's top bend at the support centre
    (simple, end of continuous) or the diaphragm face (cantilever, interior support), each next one's within the
    previous one's bottom bend: not checked by numbers."""
    chk = Check.report(f"bent-up bar of {d:g} mm (9.3.10)", "JTG 3362-2018 9.3.10")
    if angle is not None and abs(angle - 45) > 1e-6:
        chk.warnings.append(f"bent-up angle {_f(angle)}° rather than 45° (9.3.10, should)")
    if start is not None:
        _need(chk, "bend start beyond the fully used section", start, h0 / 2, "min", "9.3.10")
    if anchorage is not None:
        if zone not in ("tension", "compression"):
            raise ValueError("zone: 'tension' or 'compression'")
        k = (20.0 if zone == "tension" else 10.0) * (1.25 if epoxy else 1.0)
        _need(chk, f"end anchorage in the {zone} zone", anchorage, k * d, "min", "9.3.10")
    if _steel(steel) == "HPB300" and hooks is not None and not hooks:
        chk.warnings.append("HPB300 bent-up bars without semicircular hooks (9.3.10)")
    if floating:
        chk.warnings.append("floating bars are not allowed as bent-up bars (9.3.10)")
    return chk


def welded_frame_detailing(layers: int, d: float, *, weld: float | None = None, short_weld: float | None = None,
                           single_sided: bool = False) -> Check:
    """welded frames of reinforced-concrete beams (9.3.11, all "should"): at most 6 layers, bars ≤ 32 mm (item 4); the
    welds of the inclined to the longitudinal bars 5 d long (double-sided), the short welds between longitudinal bars
    2.5 d; single-sided welds twice as long (item 3)"""
    chk = Check.report("welded frame (9.3.11)", "JTG 3362-2018 9.3.11")
    _need(chk, "layers of the frame", layers, 6, "max", "9.3.11 item 4", unit="", should=True)
    _need(chk, "bar diameter of the frame", d, 32, "max", "9.3.11 item 4", should=True)
    k = 2.0 if single_sided else 1.0
    if single_sided:
        chk.notes.append("single-sided welds only where needed (9.3.11 item 3): lengths doubled")
    if weld is not None:
        _need(chk, "weld of the inclined bars", weld, 5 * k * d, "min", "9.3.11 item 3", should=True)
    if short_weld is not None:
        _need(chk, "short weld between longitudinal bars", short_weld, 2.5 * k * d, "min", "9.3.11 item 3",
              should=True)
    return chk


# ── 9.3.4–9.3.6 slabs of T and box girders ──────────────────────────────────
def flange_slab_detailing(t: float, *, main, distribution=None, box: bool = False, straight=None,
                          lc: float | None = None, root_area: float | None = None, outside=None,
                          prestressed: bool = False) -> Check:
    """the top slab of T and box girders (9.3.4) and the flange beyond the effective width (9.3.6), per metre. main,
    distribution: (diameter, spacing) of the tension bars for local loads and the bars across them, as 9.2.2 and 9.2.4
    (slab.slab_detailing). box=True: straight = (bars per metre, area per metre mm²) of the bars not bent down at the
    webs, ≥ 3 per metre and run to the cantilever tip; with lc > 2500 mm (cantilever span of 4.2.5) their area ≥ 60 %
    of the root's hogging bars root_area (mm² per metre). outside = (concrete area beyond the effective width, area of
    the bars there) of reinforced-concrete T and box girders: detailing bars ≥ 0.4 % of it (where provided; the main
    bars belong inside the effective width of 4.3.3, 4.3.4)."""
    chk = Check.report("girder top slab (9.3.4, 9.3.6)", "JTG 3362-2018 9.3.4, 9.3.6, 9.2.2, 9.2.4")
    s = slab_detailing(t, main=main, distribution=distribution)
    chk.warnings += s.warnings
    chk.notes += s.notes
    chk.details.update(s.details)
    if box:
        if straight is None:
            chk.notes.append("box: give straight=(bars per metre, area per metre) of the bars not bent down (9.3.4)")
        else:
            n, a = straight
            _need(chk, "straight top bars per metre", n, 3, "min", "9.3.4", unit="")
            if lc is not None and lc > 2500:
                if root_area is None:
                    chk.notes.append("lc > 2.5 m: give root_area for the 60 % rule (9.3.4)")
                else:
                    _need(chk, "straight top bars / root hogging bars", a / root_area, 0.6, "min", "9.3.4", unit="")
    if outside is not None:
        if prestressed:
            chk.notes.append("prestressed T and box girders: most of the tendons within the effective width (9.3.6)")
        else:
            ac, asb = outside
            _need(chk, "bars beyond the effective width / concrete there", asb / ac, 0.004, "min", "9.3.6", unit="")
    return chk


def _per_metre(layers) -> float:
    return sum(math.pi * d * d / 4 * 1000.0 / s for d, s in layers)


def bottom_slab_detailing(t: float, *, longitudinal, transverse, prestressed: bool = False) -> Check:
    """detailing bars of a box girder's bottom slab (9.3.5), per metre: longitudinal and transverse are lists of
    (diameter, spacing) layers (top and bottom). In each direction both layers together ≥ 0.4 % of the slab's section
    (0.3 % for prestressed bridges; program convention: the two layers summed per direction, since the commentary takes
    the ratio from AASHTO 8.17.2.3, whose longitudinal minimum of the bottom slab may sit in a single layer; reading it
    per layer would double the steel); both layers present; bars ≥ 10 mm, spacing ≤ 300 mm (should)."""
    chk = Check.report("box bottom slab (9.3.5)", "JTG 3362-2018 9.3.5")
    lim = 0.003 if prestressed else 0.004
    for name, layers in (("longitudinal", longitudinal), ("transverse", transverse)):
        layers = list(layers)
        if len(layers) < 2:
            chk.warnings.append(f"{name} bars: top and bottom layers both needed (9.3.5)")
        _need(chk, f"{name} bars / slab section", _per_metre(layers) / (t * 1000.0), lim, "min", "9.3.5", unit="")
        for k, (d, s) in enumerate(layers):
            _need(chk, f"{name} layer {k + 1} diameter", d, 10.0, "min", "9.3.5", should=True)
            _need(chk, f"{name} layer {k + 1} spacing", s, 300.0, "max", "9.3.5", should=True)
    return chk


def beam_dimensions(h: float, *, flange: float | None = None, haunch=None, tip: float | None = None,
                    wet_joint: bool = False, web: float | None = None, web_height: float | None = None,
                    vertical_prestress: bool = False, web_change=None, longitudinal_haunch=None) -> Check:
    """the dimensions of T and I girders by 9.3.2 (box girders: transverse.box_detailing): flange thickness at the web
    ≥ h/10, flange = its thickness without the haunch, haunch = (length, depth) of the haunch counted at most at a 1:3
    slope (item 1); tip of a precast T flange ≥ 100 mm, ≥ 140 mm with cast-in-place transverse joints (wet_joint, item
    1); web ≥ 160 mm, its height between the haunches web_height ≤ 15 web widths, 20 with vertical prestress (item 3);
    web_change = (difference of the web widths, length of the transition) ≥ 12 times (item 3); longitudinal_haunch =
    (length, depth) of a haunch of a continuous beam along the span: depth / length ≤ 1/6 (item 4)"""
    chk = Check.report("girder dimensions (9.3.2)", "JTG 3362-2018 9.3.2")
    if flange is not None:
        hf = flange if haunch is None else haunch_depth(flange, haunch[0], haunch[1] / haunch[0])
        _need(chk, "flange at the web", hf, h / 10, "min", "9.3.2 item 1")
    if tip is not None:
        _need(chk, "flange tip", tip, 140.0 if wet_joint else 100.0, "min", "9.3.2 item 1")
    if web is not None:
        _need(chk, "web width", web, 160.0, "min", "9.3.2 item 3")
        if web_height is not None:
            _need(chk, "web height between the haunches", web_height, (20 if vertical_prestress else 15) * web, "max",
                  "9.3.2 item 3")
    if web_change is not None:
        db, lt = web_change
        _need(chk, "web transition length", lt, 12 * db, "min", "9.3.2 item 3", should=True)
    if longitudinal_haunch is not None:
        ln, dh = longitudinal_haunch
        _need(chk, "longitudinal haunch slope", dh / ln, 1 / 6, "max", "9.3.2 item 4", unit="", should=True)
    chk.notes.append("webs under torsion: their mean width also by the note to 5.5.5 (9.3.2 item 3)")
    return chk


# ── 9.3.1, 9.3.15–9.3.17 girders ────────────────────────────────────────────
def diaphragm_detailing(kind: str, *, end: bool = True, intermediate: bool | None = None,
                        spacing: float | None = None, rigid: bool = False, radius: float | None = None,
                        prestressed: bool = False, cantilever: float | None = None,
                        mid_cantilever: bool | None = None) -> Check:
    """diaphragms of concrete superstructures (9.3.1), lengths in mm. kind "T": precast T-beam bridges (end and
    intermediate diaphragms; with rigid transverse connection (rigid=True) spacing ≤ 10 m); "composite box": precast
    composite box girders (end diaphragms; intermediate ones as the structure needs); "box": box girders (end
    diaphragms in the box; curved boxes with an inner radius below 240 m need intermediate ones, ≤ 10 m apart in
    reinforced concrete, by analysis in prestressed concrete; boxes cantilevered 50 m or more (cantilever) a diaphragm
    in the middle of the cantilever (mid_cantilever)). Inspection openings: not checked."""
    k = kind.lower()
    chk = Check.report(f"diaphragms of a {kind} superstructure (9.3.1)", "JTG 3362-2018 9.3.1")
    if not end:
        chk.warnings.append("no end diaphragms (9.3.1)")
    if k == "t":
        if intermediate is not None and not intermediate:
            chk.warnings.append("precast T-beam bridge without intermediate diaphragms (9.3.1 item 1)")
        if rigid and spacing is not None:
            _need(chk, "diaphragm spacing", spacing, 10000.0, "max", "9.3.1 item 1")
    elif k == "composite box":
        chk.notes.append("intermediate diaphragms as the structure needs (9.3.1 item 2)")
    elif k == "box":
        if radius is not None and radius < 240000.0:
            if intermediate is not None and not intermediate:
                chk.warnings.append(f"curved box, inner radius {_f(radius / 1000)} m < 240 m, without intermediate "
                                    "diaphragms (9.3.1 item 3)")
            if prestressed:
                chk.notes.append("prestressed curved box: diaphragm spacing by structural analysis (9.3.1 item 3)")
            elif spacing is not None:
                _need(chk, "diaphragm spacing", spacing, 10000.0, "max", "9.3.1 item 3")
        if cantilever is not None and cantilever >= 50000.0 and mid_cantilever is not None and not mid_cantilever:
            chk.warnings.append(f"box cantilever {_f(cantilever / 1000)} m ≥ 50 m without a diaphragm in the middle of "
                                "the cantilever (9.3.1 item 3)")
        chk.notes.append("diaphragms of box girders with inspection openings where conditions allow (9.3.1 item 3)")
    else:
        raise ValueError("kind: 'T', 'composite box' or 'box'")
    return chk


def composite_beam_detailing(*, topping: float | None = None, roughness: float | None = None, stirrups=None,
                             age_difference: float | None = None) -> Check:
    """composite beams of a precast girder and a cast-in-place deck (9.3.16, 9.3.17, which 8.1.6 requires) and the
    cast-in-place joints of precast girders (9.3.15): topping thickness at the joint ≥ 150 mm (preferably); roughness
    of the precast top ≥ 6 mm; stirrups = (diameter, embedment) of the girder's stirrups into the deck, ≥ 10 diameters;
    age_difference (months) between the precast concrete and the joining cast-in-place concrete ≤ 3"""
    chk = Check.report("composite beam (9.3.15–9.3.17)", "JTG 3362-2018 9.3.15, 9.3.16, 9.3.17")
    if topping is not None:
        _need(chk, "cast-in-place layer at the joint", topping, 150.0, "min", "9.3.16", should=True)
    if roughness is not None:
        _need(chk, "roughness of the precast top", roughness, 6.0, "min", "9.3.16")
    if stirrups is not None:
        d, embed = stirrups
        _need(chk, "stirrups into the deck", embed, 10 * d, "min", "9.3.17")
    if age_difference is not None:
        _need(chk, "age difference of precast and joining concrete", age_difference, 3.0, "max", "9.3.15",
              unit="months")
    return chk


# ── 9.2.5–9.2.8 slabs ───────────────────────────────────────────────────────
def two_way_strips(lx: float, ly: float) -> tuple[float, float, float]:
    """the strips of a two-way slab supported on four sides (9.2.5): (edge strip width = the short side / 4, middle
    strip width along x, along y)"""
    e = min(lx, ly) / 4
    return e, lx - 2 * e, ly - 2 * e


def two_way_slab_detailing(lx: float, ly: float, t: float, *, middle, edge) -> Check:
    """bars of a two-way slab supported on four sides (9.2.5), one direction: middle, edge = (diameter, spacing) in the
    middle strip (by calculation) and the edge strips (half of the middle strip's bars), spacing ≤ 250 mm and ≤ 2 t
    (program convention: both strips)"""
    e, mx, my = two_way_strips(lx, ly)
    chk = Check.report("two-way slab (9.2.5)", "JTG 3362-2018 9.2.5", {"edge_strip": e, "middle_x": mx, "middle_y": my})
    am, ae = _per_metre([middle]), _per_metre([edge])
    _need(chk, "edge strip bars / middle strip bars", ae / am, 0.5, "min", "9.2.5", unit="")
    lim = min(250.0, 2 * t)
    for name, (d, s) in (("middle", middle), ("edge", edge)):
        _need(chk, f"{name} strip spacing", s, lim, "max", "9.2.5")
    return chk


def skew_slab_detailing(phi: float, *, monolithic: bool = True, main: str | None = None,
                        edge_band: int | None = None, obtuse=None) -> Check:
    """bars of skew slabs (9.2.6, figure 9.2.6): phi skew angle (degrees, between the normal to the support axis and the
    bridge axis). Monolithic slabs: main bars along the bridge axis ("axis") up to 15°, beyond it preferably normal to
    the support axis ("support"), with a band of ≥ 3 main bars along each free edge, top and bottom, held by stirrups
    (edge_band: bars in the band); precast slabs may keep their main bars along the axis. obtuse = (diameter, spacing,
    extent) of the obtuse-corner bars (top: normal to the bisector, bottom: along it): ≥ 12 mm preferably, at
    100–150 mm, over 1.0–1.5 m from the corner along both sides. Distribution bars: 9.2.4 (slab_detailing)."""
    chk = Check.report(f"skew slab, {phi:g}° (9.2.6)", "JTG 3362-2018 9.2.6")
    if monolithic and phi > 15:
        if main == "axis":
            chk.warnings.append(f"monolithic skew slab of {_f(phi)}° > 15° with the main bars along the bridge axis: "
                                "normal to the support axis (9.2.6 item 1, should)")
        if edge_band is not None:
            _need(chk, "main bars in the free-edge band", edge_band, 3, "min", "9.2.6 item 1", unit="")
        elif main is not None:
            chk.notes.append("free edges need a band of ≥ 3 main bars, top and bottom (9.2.6 item 1)")
    if obtuse is not None:
        d, s, ext = obtuse
        _need(chk, "obtuse-corner bars", d, 12.0, "min", "9.2.6 item 1", should=True)
        _need(chk, "obtuse-corner bar spacing", s, 150.0, "max", "9.2.6 item 1")
        if s < 100 - _TOL:
            chk.notes.append(f"obtuse-corner bars at {_f(s)} mm, closer than the 100–150 mm of 9.2.6")
        _need(chk, "obtuse-corner reinforcement extent", ext, 1000.0, "min", "9.2.6 item 1")
        if ext > 1500 + _TOL:
            chk.notes.append(f"obtuse-corner reinforcement over {_f(ext)} mm, beyond the 1.0–1.5 m of 9.2.6")
    return chk


def composite_slab_detailing(*, roughness: float | None = None, connectors=None) -> Check:
    """composite slabs of a precast slab and cast-in-place concrete (9.2.7): roughness of the precast top ≥ 6 mm;
    connectors = (diameter, embedment in the precast slab, embedment in the topping, spacing along the span) of vertical
    bars across the interface: embedments ≥ 10 diameters, spacing ≤ 500 mm"""
    chk = Check.report("composite slab (9.2.7)", "JTG 3362-2018 9.2.7")
    if roughness is not None:
        _need(chk, "roughness of the precast top", roughness, 6.0, "min", "9.2.7")
    if connectors is not None:
        d, e1, e2, s = connectors
        _need(chk, "connector embedment in the precast slab", e1, 10 * d, "min", "9.2.7")
        _need(chk, "connector embedment in the topping", e2, 10 * d, "min", "9.2.7")
        _need(chk, "connector spacing", s, 500.0, "max", "9.2.7")
    return chk


def hinged_slab_detailing(h: float, *, groove: float | None = None, topping: float | None = None) -> Check:
    """precast slabs joined by hinges (9.2.8): groove depth 2/3 of the precast slab height h (should; shallower warns,
    over 10 % deeper is noted); a cast-in-place reinforced topping over the hinged slabs ≥ 80 mm (should); the hinge's
    top opening wide enough for a poker vibrator and bars from the slabs into the hinge (not checked by numbers)"""
    chk = Check.report("hinged slabs (9.2.8)", "JTG 3362-2018 9.2.8")
    if groove is not None:
        _need(chk, "hinge groove depth", groove, 2 * h / 3, "min", "9.2.8", should=True)
        if groove > 1.1 * 2 * h / 3:
            chk.notes.append(f"hinge groove {_f(groove)} mm, over 10 % deeper than the 2/3 h = {_f(2 * h / 3)} mm "
                             "9.2.8 names")
    if topping is not None:
        _need(chk, "topping over hinged slabs", topping, 80.0, "min", "9.2.8", should=True)
    chk.notes.append("hinge opening for a poker vibrator; bars from the precast slabs into the hinge (9.2.8)")
    return chk
