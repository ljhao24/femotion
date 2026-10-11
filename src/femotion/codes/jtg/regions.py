"""Disturbed regions of JTG 3362-2018 chapter 8: post-tensioned anchorage zones (8.2), diaphragms over bearings (8.3),
cap beams (8.4) and pile caps (8.5). The code gives the tie forces of these regions in closed form (appendix B
commentary: in place of building and solving a strut-and-tie model); general models: femotion.codes.jtg.stm.

Units: mm, mm², MPa; forces kN, moments kN·m; angles in degrees unless noted. Clauses: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    from femotion.codes.jtg import regions as rg
    Pd = rg.anchor_force(9 * 139 * 1395 / 1e3)                         # 1.2 × the jacking force (8.2.2)
    Tb, db = rg.bursting_force(Pd, a=270, h=2000, e=300, alpha=8)      # 8.2.2
    Tb, db, groups = rg.bursting_force_anchors([Pd] * 3, y=[500, 900, 1500], a=270, h=2000, yc=900)  # dense or not
    rg.tie_check(Tb, As=10 * 113, gamma0=1.1, name="bursting")         # 8.2.1 (As within the zone: tendon_detailing)
    rg.cap_cantilever_tie(Fd=3200, x=600, bc=1200, h0=1500, As=20 * 491, h=1600)   # 8.4.6
    rg.pile_cap_stm(N1d=6000, N2d=5000, x1=1250, x2=1250, h0=1880, b=1200, s=100, d=28, As=40 * 616, bs=4000,
                    concrete="C30")                                     # 8.5.4
"""
from __future__ import annotations

import math

import numpy as np

from .materials import Concrete, Rebar, Tendon, concrete as _concrete, rebar as _rebar, tendon as _tendon, xi_b
from .rc import Check, RCSection
from .stm import beta_c, eps1_of, strut_strength


def _c(x):
    return x if isinstance(x, Concrete) else _concrete(x)


def _s(x):
    return x if isinstance(x, Rebar) else _rebar(x)


def _t(x):
    return x if isinstance(x, Tendon) else _tendon(x)


def tie_check(T: float, As: float, Ap: float = 0.0, *, gamma0: float = 1.0, steel="HRB400", tendon="strand1860",
              name: str = "tie", clause: str = "JTG 3362-2018 8.2.1") -> Check:
    """γ0 T ≤ fsd As (8.2.1, 8.4.7-1, 8.5.4-6) or γ0 T ≤ fsd As + fpd Ap where the clause counts prestressing steel
    (8.3.2-1, 8.4.6-1, B.3.2)"""
    R = (_s(steel).fsd * As + (_t(tendon).fpd * Ap if Ap else 0.0)) / 1e3
    chk = Check(name, gamma0 * T, R, "kN", clause, {"As": As, "Ap": Ap})
    if Ap and clause.endswith(("8.2.1", "8.4.7", "8.5.4-6")):
        chk.notes.append(f"{clause} counts the reinforcing bars only (fsd As); Ap included here")
    return chk


# ── 8.2 anchorage zones ──────────────────────────────────────────────────────
def anchor_force(jacking: float) -> float:
    """design anchorage force Pd = 1.2 × the jacking force (8.2.2; also Fld of 5.7.1 under an anchorage)"""
    return 1.2 * jacking


def bursting_force(Pd: float, a: float, h: float, e: float = 0.0, alpha: float = 0.0) -> tuple[float, float]:
    """(Tb,d, db) of an end anchorage zone (8.2.2): Tb,d = 0.25 Pd (1 + γ)² [(1 − γ) − a/h] + 0.5 Pd |sin α|
    (8.2.2-1), db = 0.5 (h − 2e) + e sin α (8.2.2-2) from the end face, γ = 2e/h. a: plate width (a group of dense anchors,
    centres closer than twice the plate width: the resultant force and a = the distance between the outer edges of the
    outermost plates; separate anchors: each on its own, the largest governs); h: depth of the end section; e: distance of
    the force from the centroid (T girders and boxes: the web as a rectangle, e from the girder's centroid, commentary);
    alpha: tendon inclination (−5° … +20°; positive when the line of the force runs from the anchor towards the
    centroid, figure 8-5). Program: a negative result of 8.2.2-1 (wide plates or a large eccentricity, where the edge
    tension of 8.2.5 governs) is returned as 0."""
    g = 2 * e / h
    sa = math.sin(math.radians(alpha))
    Tb = 0.25 * Pd * (1 + g) ** 2 * ((1 - g) - a / h) + 0.5 * Pd * abs(sa)
    return max(Tb, 0.0), 0.5 * (h - 2 * e) + e * sa


def dense_anchors(y, a) -> list[list[int]]:
    """groups of dense anchors on an end face (8.2.2 note 1): neighbouring plates whose centres are closer than twice the
    plate width are dense; a chain of such neighbours is one group (figure 8.2.2b). y: anchor centres along the depth
    (mm); a: plate widths, one value or one per anchor (two different neighbours: the wider plate, program convention).
    Returns lists of anchor indices in order of y; a separate anchor is a group of its own. (The commentary: AASHTO LRFD
    groups anchors closer than 1.5 a; the code widens it to 2 a.) y is one-dimensional: the anchors of one plane, i.e.
    one web (8.2.2 works in the web's plane, commentary p210); call it once per web, anchors in different webs at the
    same height are not neighbours."""
    y = np.asarray(y, float).reshape(-1)
    a = np.broadcast_to(np.asarray(a, float), y.shape)
    order = [int(i) for i in np.argsort(y, kind="stable")]
    groups = [[order[0]]] if order else []
    for i, j in zip(order[:-1], order[1:]):
        if y[j] - y[i] < 2 * max(a[i], a[j]):
            groups[-1].append(j)
        else:
            groups.append([j])
    return groups


def bursting_force_anchors(Pd, y, a, h: float, *, yc: float, angle=0.0) -> tuple[float, float, list[dict]]:
    """bursting force of the anchors on one end face (8.2.2): dense anchors (dense_anchors) as one force, their resultant,
    with a = the distance between the outer edges of the two outermost plates (item 2, note 2); separate anchors each on
    their own; the largest governs (item 3). Pd: design anchorage forces (kN, one per anchor, anchor_force); y: anchor
    centres from the top of the end section (mm), the anchors of one web (dense_anchors); a: plate widths; h: depth of the end section; yc: depth of the
    centroid of the member's section (T girders and boxes: of the whole girder, commentary to 8.2.2); angle: each
    tendon's inclination at the anchor, degrees, positive where the force line (from the anchor into the member) runs
    down. A group's resultant: the vector sum of its forces, acting at the force-weighted depth of their horizontal
    components; its inclination turned into the sign of 8.2.2 (positive towards the centroid; a resultant at the
    centroid leaves it: negative). Returns (Tb,d, db of that group, groups: dicts of anchors, P, y, a, e, alpha, Tb,
    db)."""
    y = np.asarray(y, float).reshape(-1)
    P = np.broadcast_to(np.asarray(Pd, float), y.shape)
    aa = np.broadcast_to(np.asarray(a, float), y.shape)
    th = np.radians(np.broadcast_to(np.asarray(angle, float), y.shape))
    out = []
    for g in dense_anchors(y, aa):
        H, V = float((P[g] * np.cos(th[g])).sum()), float((P[g] * np.sin(th[g])).sum())
        yg = float((P[g] * np.cos(th[g]) * y[g]).sum() / H)
        ag = float((y[g] + aa[g] / 2).max() - (y[g] - aa[g] / 2).min()) if len(g) > 1 else float(aa[g[0]])
        incl = math.degrees(math.atan2(V, H))  # downward +
        alpha = incl if yg < yc - 1e-9 else -incl if yg > yc + 1e-9 else -abs(incl)
        e = abs(yg - yc)
        Tb, db = bursting_force(math.hypot(H, V), ag, h, e, alpha)
        out.append({"anchors": g, "P": math.hypot(H, V), "y": yg, "a": ag, "e": e, "alpha": alpha, "Tb": Tb, "db": db})
    k = int(np.argmax([o["Tb"] for o in out]))
    return out[k]["Tb"], out[k]["db"], out


def spalling_force(Pd) -> float:
    """surface spalling force round the plates Ts,d = 0.02 max Pdi (8.2.3)"""
    return 0.02 * float(np.max(Pd))


def spalling_force_wide(Pd1: float, Pd2: float, s: float, h: float) -> float:
    """spalling force between two anchors farther apart than h/2 (8.2.4): 0.45 P̄d (2s/h − 1), P̄d = (Pd1 + Pd2)/2, at least
    0.02 × the larger"""
    if s <= h / 2:
        raise ValueError("8.2.4 applies when the anchors are more than h/2 apart")
    return max(0.45 * (Pd1 + Pd2) / 2 * (2 * s / h - 1), 0.02 * max(Pd1, Pd2))


def edge_tension(Pd: float, e: float, h: float) -> float:
    """longitudinal edge tension of an eccentric anchorage (8.2.5): 0 for γ = 2e/h ≤ 1/3, else (3γ − 1)² / (12 γ) Pd"""
    g = 2 * e / h
    return 0.0 if g <= 1 / 3 else (3 * g - 1) ** 2 / (12 * g) * Pd


def blister_forces(Pd: float, a: float, d: float, e: float, alpha: float) -> dict:
    """the five tie forces of a blister (triangular anchor block, 8.2.6): "bursting" 0.25 Pd (1 − a/(2d)) (8.2.6-1),
    "end face" 0.04 Pd (-2), "behind anchor" 0.2 Pd (-3), "edge bending" (2e − d)² / (12 e (e + d)) Pd (-4), "radial"
    Pd α (-5). a: plate width; d: anchor centre to the top edge of the blister; e: anchor centre to the centre of the wall
    plate; alpha: the angle the tendon turns through (degrees here, radians in 8.2.6-5)"""
    return {"bursting": 0.25 * Pd * (1 - a / (2 * d)), "end face": 0.04 * Pd, "behind anchor": 0.2 * Pd,
            "edge bending": (2 * e - d) ** 2 / (12 * e * (e + d)) * Pd, "radial": Pd * math.radians(alpha)}


# ── 8.3 diaphragms over bearings ─────────────────────────────────────────────
def diaphragm_tie(Vd: float, Bw: float, h: float, s: float, As: float, Ap: float = 0.0, *, gamma0: float = 1.0,
                  steel="HRB400", tendon="strand1860") -> Check:
    """top transverse tie of a single-cell box diaphragm over its bearings (8.3.2), 0.5 ≤ Bw/h ≤ 2:
    Tt,d = [0.20 + (Bw/h − 0.5)(0.87 − s/Bw)] Vd (8.3.2-2), γ0 Tt,d ≤ fsd As + fpd Ap (8.3.2-1). Vd: the shear one web
    passes on (two bearings: one bearing reaction Rd; one bearing: Rd/2); s: two bearings their centre distance, one
    bearing half its plate width; Bw: distance between the web centre lines (inclined webs: at their mid-depth); h: the
    girder depth at the bearing. As, Ap: steel in the diaphragm thickness and one diaphragm thickness either side, within
    the top slab. Bw/h > 2: an ordinary flexural member (chapters 5–7, 8.3.1)."""
    r = Bw / h
    T = (0.20 + (r - 0.5) * (0.87 - s / Bw)) * Vd
    chk = tie_check(T, As, Ap, gamma0=gamma0, steel=steel, tendon=tendon, name="diaphragm top tie",
                    clause="JTG 3362-2018 8.3.2")
    chk.details.update(Bw_h=r, T=T)
    if not 0.5 <= r <= 2.0:
        chk.failed.append(f"Bw/h = {r:.2f} outside 0.5 … 2 (8.3.2; above 2: chapters 5–7)")
        if r < 0.5:
            chk.notes.append(f"Bw/h < 0.5: the commentary to 8.3.2 gives a top tension tending to 0.2 Vd = {0.2 * Vd:.0f} kN")
    return chk


# ── 8.4 cap beams ────────────────────────────────────────────────────────────
def cap_beam_class(l: float, h: float) -> str:
    """8.4.2: "general" (l/h > 5.0, chapters 5–7), "short" (2.5 < l/h ≤ 5.0, 8.4.3–8.4.5), "deep" (l/h ≤ 2.5: not in
    8.4.2; program convention: a disturbed region, strut-and-tie model by 4.1.9 and appendix B). l: computed span (between
    bearing centres, 8.4.1), h: depth"""
    r = l / h
    return "general" if r > 5.0 else "short" if r > 2.5 else "deep"


def _short(l, h, chk):
    if cap_beam_class(l, h) != "short":
        chk.failed.append(f"l/h = {l / h:.2f}: 8.4.3–8.4.5 cover 2.5 < l/h ≤ 5.0 (8.4.2)")


def cap_flexure(sec: RCSection, Md: float, l: float, *, gamma0: float = 1.0) -> Check:
    """flexural resistance of a reinforced concrete cap beam with 2.5 < l/h ≤ 5.0 (8.4.3): γ0 Md ≤ fsd As z,
    z = k (h0 − 0.5 x), k = 0.75 + 0.05 l/h (8.4.3-2), x by 5.2.2-2 (rectangle: (fsd As − f'sd A's)/(fcd b)); As, A's the
    edge rows (femotion.codes.jtg.rc). With A's counted x must reach 2a's (5.2.2-5); below it, as 5.2.4 for flexural
    members (program convention, the general rules kept): moments about A's, z = k (h0 − a's), or the section without
    A's (its own x by 5.2.2-2), the larger. Also x ≤ ξb h0 (5.2.2-3) and 9.1.12. Hogging moments on the section turned
    over."""
    if sec.prestressed or sec.kind != "rect":
        raise NotImplementedError("8.4.3 is for reinforced concrete cap beams (rectangular sections here)")
    s = sec if Md >= 0 else sec.flipped()
    c, st = s.concrete, s.steel
    t, comp = s.groups()
    As, Asc = t[:, 0].sum(), comp[:, 0].sum()
    b, h0 = s.pieces[0].b, s.h0()
    k = 0.75 + 0.05 * l / s.h
    x = (st.fsd * As - st.fsd_c * Asc) / (c.fcd * b)
    notes = []
    if Asc > 0:
        a_s = float(np.average(comp[:, 1], weights=comp[:, 0]))
        if x < 2 * a_s:
            x1 = st.fsd * As / (c.fcd * b)  # A's left out
            z_about, z_single = k * (h0 - a_s), k * (h0 - 0.5 * x1)
            z = max(z_about, z_single)
            notes.append(f"x = {x:.0f} < 2a's = {2 * a_s:.0f} mm: " + ("moments about A's, z = k (h0 − a's)"
                         if z_about >= z_single else f"A's left out (x = {x1:.0f} mm)") + " (5.2.2-5, 5.2.4)")
            x = 2 * a_s if z_about >= z_single else x1
        else:
            z = k * (h0 - 0.5 * x)
    else:
        z = k * (h0 - 0.5 * x)
    chk = Check("cap beam flexure" + (" (hogging)" if Md < 0 else ""), gamma0 * Md, st.fsd * As * z / 1e6, "kN·m",
                "JTG 3362-2018 8.4.3", {"x": x, "z": z, "h0": h0, "l_h": l / s.h, "k": k}, notes, section=sec,
                hogging=Md < 0)
    _short(l, s.h, chk)
    xb = xi_b(c, st)
    if x > xb * h0:
        chk.failed.append(f"over-reinforced: x = {x:.0f} > ξb h0 = {xb * h0:.0f} mm (5.2.2-3)")
    rho, rmin = As / (b * h0), max(0.45 * c.ftd / st.fsd, 0.002)
    if rho < rmin:
        chk.warnings.append(f"tension reinforcement ratio {rho:.4%} < minimum {rmin:.4%} (9.1.12)")
    chk.warnings += s.material_rules()
    return chk


def cap_shear(sec: RCSection, Vd: float, l: float, stirrups, *, gamma0: float = 1.0, alpha1: float = 1.0,
              hogging: bool = False, stirrup_steel=None) -> Check:
    """shear of a reinforced concrete cap beam with 2.5 < l/h ≤ 5.0: section γ0 Vd ≤ 0.33·10⁻⁴ (l/h + 10.3) √fcu,k b h0
    (8.4.4); resistance γ0 Vd ≤ 0.5·10⁻⁴ α1 (14 − l/h) b h0 √((2 + 0.6P) √fcu,k ρsv fsv) (8.4.5), P = 100 As/(b h0) ≤ 2.5,
    ρsv per 9.3.12. α1: 1.0 near end supports, 0.9 near interior supports and the joints of a rigid frame.
    stirrups = (legs, diameter, spacing)."""
    if sec.prestressed or sec.kind != "rect":
        raise NotImplementedError("8.4.4, 8.4.5 are for reinforced concrete cap beams (rectangular sections here)")
    s = sec.flipped() if hogging else sec
    c = s.concrete
    sst = s.steel if stirrup_steel is None else _s(stirrup_steel)
    b, h0 = s.pieces[0].b, s.h0()
    t, _ = s.groups()
    P = min(100 * t[:, 0].sum() / (b * h0), 2.5)
    legs, dia, sv = stirrups
    rho_sv = legs * math.pi * dia * dia / 4 / (b * sv)
    r = l / s.h
    Vu = 0.5e-4 * alpha1 * (14 - r) * b * h0 * math.sqrt((2 + 0.6 * P) * math.sqrt(c.fcuk) * rho_sv * sst.fsv)
    Vmax = 0.33e-4 * (r + 10.3) * math.sqrt(c.fcuk) * b * h0
    chk = Check("cap beam shear", gamma0 * Vd, Vu, "kN", "JTG 3362-2018 8.4.5",
                {"P": P, "rho_sv": rho_sv, "l_h": r, "V_section_max": Vmax, "alpha1": alpha1}, section=sec,
                hogging=hogging)
    _short(l, s.h, chk)
    if gamma0 * abs(Vd) > Vmax:
        chk.failed.append(f"section too small: γ0Vd > 0.33e-4 (l/h + 10.3) √fcu,k b h0 = {Vmax:.0f} kN (8.4.4)")
    if rho_sv < sst.rho_sv_min:
        chk.warnings.append(f"stirrup ratio {rho_sv:.4%} < minimum {sst.rho_sv_min:.2%} (9.3.12)")
    return chk


def cap_cantilever_tie(Fd: float, x: float, bc: float, h0: float, As: float, Ap: float = 0.0, *, h: float,
                       gamma0: float = 1.0, steel="HRB400", tendon="strand1860") -> Check:
    """top tie of a cap beam cantilever under a vertical force no farther than the depth from the column (8.4.6,
    figure 8.4.6): Tt,d = (x + bc/2) / z × Fd (8.4.6-2), z = 0.9 h0, γ0 Tt,d ≤ fsd As + fpd Ap (8.4.6-1). x: from the force
    to the column edge (a round column as a square of side 0.8 D); bc: the column's supporting width (square: its side;
    round: 0.8 D); Fd from the basic combination. x > h: an ordinary member (chapters 5–7, 8.4.6 item 1)."""
    z = 0.9 * h0
    T = (x + bc / 2) / z * Fd
    chk = tie_check(T, As, Ap, gamma0=gamma0, steel=steel, tendon=tendon, name="cap cantilever top tie",
                    clause="JTG 3362-2018 8.4.6")
    chk.details.update(T=T, z=z)
    if x > h:
        chk.failed.append(f"x = {x:.0f} mm > h = {h:.0f} mm: an ordinary member (chapters 5–7), not 8.4.6 item 2")
    return chk


def pier_head_tie(Fd: float, s: float, b_prime: float, h: float, b: float, As: float, *, gamma0: float = 1.0,
                  steel="HRB400") -> Check:
    """top transverse tie of the head of a single-column pier with two bearings (8.4.7, figure 8.4.7):
    Tt,d = 0.45 Fd (2s − b') / h (8.4.7-2), γ0 Tt,d ≤ fsd As (8.4.7-1). Fd: vertical design force (basic combination;
    the force of one bearing, figure 8.4.7); s: bearing centre distance; h: height of the flaring part, taken as b where it
    exceeds b (the head's top width); b': width of the head or pier at depth h (as used) below the top; As within the
    top 2h/9."""
    hh = min(h, b)
    T = 0.45 * Fd * (2 * s - b_prime) / hh
    chk = tie_check(T, As, gamma0=gamma0, steel=steel, name="pier head top tie", clause="JTG 3362-2018 8.4.7")
    chk.details.update(T=T, h_used=hh, As_zone=2 * hh / 9)
    return chk


def cap_beam_C3(l: float, h: float) -> float:
    """C3 of 6.4.3 for cap beams with 2.5 < l/h ≤ 5.0 (8.4.8): (0.4 l/h + 1) / 3; pass to jtg.crack_width(C3=…)"""
    if cap_beam_class(l, h) != "short":
        raise ValueError("8.4.8 covers 2.5 < l/h ≤ 5.0")
    return (0.4 * l / h + 1) / 3


# ── 8.5 pile caps ────────────────────────────────────────────────────────────
def pile_forces(Fd: float, Mx: float, My: float, xy) -> np.ndarray:
    """vertical design force of every pile on the cap (8.5.1): Nid = Fd/n + Mxd yi/Σyi² + Myd xi/Σxi², xy: pile centres
    (m), measured from the centroid of the pile group (taken as their mean); Fd kN (compression +), Mxd, Myd kN·m about
    the group's x and y axes (signs: a positive Mxd increases the force of the piles at +y, a positive Myd of those at
    +x). High pile caps under large horizontal forces need a finer method (commentary)."""
    p = np.asarray(xy, float)
    p = p - p.mean(axis=0)
    sx, sy = (p[:, 0] ** 2).sum(), (p[:, 1] ** 2).sum()
    N = np.full(len(p), Fd / len(p))
    if sy > 0:
        N += Mx * p[:, 1] / sy
    if sx > 0:
        N += My * p[:, 0] / sx
    return N


def pile_cap_width(B: float, *, a: float, D: float, n: int, spacing: float) -> float:
    """computed width bs of a pile cap section (8.5.2): the full width B when the piles are at most 3 D apart (D the pile
    side or diameter), else 2a + 3D (n − 1) (8.5.2-1), a: the edge pile centre to the cap edge, n: piles along the
    section (program: at most B)"""
    return B if spacing <= 3 * D else min(2 * a + 3 * D * (n - 1), B)


def pile_cap_moment(N_rows, distances) -> float:
    """design moment at a section of a pile cap (8.5.2-2, -3): Σ Nid × the distance of row i from the section (m), Nid
    = the number of piles in the row × its largest pile force (kN)"""
    return float(np.dot(N_rows, distances))


def pile_cap_shear(Vd: float, *, As: float, bs: float, h0: float, a: float, concrete, gamma0: float = 1.0) -> Check:
    """inclined section of a pile cap whose outer piles are farther from the pier than the cap depth (8.5.3):
    γ0 Vd ≤ 0.9·10⁻⁴ (2 + 0.6P) √fcu,k / m × bs h0, m = a/h0 ≥ 0.5 (a: from the pier edge to the edge of the pile row
    outside the section; round piles as squares of 0.8 D), P = 100 As/(bs h0) ≤ 2.5. Vd: the piles outside the section,
    each row as its number of piles × its largest pile force. Several possible sections: check each."""
    c = _c(concrete)
    P = min(100 * As / (bs * h0), 2.5)
    m = max(a / h0, 0.5)
    R = 0.9e-4 * (2 + 0.6 * P) * math.sqrt(c.fcuk) / m * bs * h0
    return Check("pile cap shear", gamma0 * Vd, R, "kN", "JTG 3362-2018 8.5.3", {"P": P, "m": m})


def pile_cap_stm(N1d: float, N2d: float, x1: float, x2: float, h0: float, *, b: float, s: float, d: float, As: float,
                 bs: float, concrete, gamma0: float = 1.0, steel="HRB400") -> list[Check]:
    """strut-and-tie model of a pile cap whose outer piles are no farther from the pier than the cap depth (8.5.4, figure
    8.5.4): the pier's load as two forces a = 0.15 h0 inside its edges, the piles' at their centres;
    θi = atan(h0 / (a + xi)) (xi: pile centre to the pier edge), Ci,d = Nid / sin θi, Ti,d = Nid / tan θi, Nid = the
    number of piles in row 1 or row 2 of figure 8.5.4 × that row's largest pile force (8.5.1).
    Strut: γ0 Ci,d ≤ t bs fce,d (8.5.4-1), t = b sin θi + ha cos θi (-4), ha = s + 6d (-5), fce,d = βc fcd /
    (0.8 + 170 ε1) ≤ 0.85 βc fcd (-2), ε1 = Ti,d/(As Es) + (Ti,d/(As Es) + 0.002) cot² θi (-3) with Ti,d the larger of
    T1,d, T2,d (its one definition, under 8.5.4-6: one continuous tie); b: the pile's supporting width (square side,
    round 0.8 D); s: top layer of tie bars to the cap bottom; d: tie bar diameter (weighted mean). The code takes the
    larger of C1,d, C2,d; the program checks both struts, each with its own θ and C (the larger C included). Tie:
    γ0 max(T1,d, T2,d) ≤ fsd As (8.5.4-6), As within bs, As/(bs h0) ≥ 0.15 % (program: on bs h0)."""
    c, st = _c(concrete), _s(steel)
    a = 0.15 * h0
    ha = s + 6 * d
    out = []
    th12 = [math.atan(h0 / (a + x)) for x in (x1, x2)]
    Ts = [N / math.tan(th) for N, th in zip((N1d, N2d), th12)]
    e_s = max(Ts) * 1e3 / (As * st.Es)  # the one tie
    for k, (N, th) in enumerate(zip((N1d, N2d), th12), start=1):
        C = N / math.sin(th)
        t = b * math.sin(th) + ha * math.cos(th)
        e1 = eps1_of(e_s, math.degrees(th))
        fce = strut_strength(c, e1)
        out.append(Check(f"pile cap strut {k}", gamma0 * C, t * bs * fce / 1e3, "kN", "JTG 3362-2018 8.5.4-1",
                         {"theta": math.degrees(th), "t": t, "ha": ha, "fce_d": fce, "eps1": e1, "a": a,
                          "T": max(Ts), "beta_c": beta_c(c)}))
    tie = Check("pile cap tie", gamma0 * max(Ts), st.fsd * As / 1e3, "kN", "JTG 3362-2018 8.5.4-6",
                {"T1": Ts[0], "T2": Ts[1], "rho": As / (bs * h0)})
    if As / (bs * h0) < 0.0015:
        tie.warnings.append(f"tie steel {As / (bs * h0):.3%} < 0.15 % within bs (8.5.4)")
    out.append(tie)
    return out


def _alpha(a: float, h0: float, k: float):
    """(a clamped to 0.2 h0 … h0, λ, k / (λ + 0.2)) for 8.5.5"""
    a = min(max(a, 0.2 * h0), h0)
    lam = a / h0
    return a, lam, k / (lam + 0.2)


def pile_cap_punching_column(Fld: float, *, bx: float, by: float, ax: float, ay: float, h0: float, concrete,
                             gamma0: float = 1.0) -> Check:
    """punching of a column or pier through a pile cap (8.5.5 item 1): γ0 Fld ≤ 0.6 ftd h0 [2 αpx (by + ay) + 2 αpy
    (bx + ax)] (8.5.5-1), αp = 1.2 / (λ + 0.2) (-2, -3), λ = a/h0 with a (column edge to pile edge) within 0.2 h0 … h0.
    Fld: the column force less the pile reactions inside the cone; bx, by: the column sides (round: 0.8 D square); h0: at
    the column edge for a cap of varying depth (note)."""
    c = _c(concrete)
    ax_, lx, apx = _alpha(ax, h0, 1.2)
    ay_, ly, apy = _alpha(ay, h0, 1.2)
    R = 0.6 * c.ftd * h0 * (2 * apx * (by + ay_) + 2 * apy * (bx + ax_)) / 1e3
    return Check("pile cap punching (column)", gamma0 * Fld, R, "kN", "JTG 3362-2018 8.5.5-1",
                 {"ax": ax_, "ay": ay_, "lambda_x": lx, "lambda_y": ly, "alpha_px": apx, "alpha_py": apy})


def pile_cap_punching_corner(Fld: float, *, bx: float, by: float, ax: float, ay: float, h0: float, concrete,
                             gamma0: float = 1.0) -> Check:
    """upward punching of a corner pile outside the column's cone (8.5.5 item 2-1): γ0 Fld ≤ 0.6 ftd h0 [α'px (by + ay/2)
    + α'py (bx + ax/2)] (8.5.5-4), α'p = 0.8 / (λ + 0.2); bx, by: cap edge to the inner edge of the pile; a: pile edge to
    the column edge, within 0.2 h0 … h0; h0 at the cap edge (note); round piles as 0.8 D squares."""
    c = _c(concrete)
    ax_, lx, apx = _alpha(ax, h0, 0.8)
    ay_, ly, apy = _alpha(ay, h0, 0.8)
    R = 0.6 * c.ftd * h0 * (apx * (by + ay_ / 2) + apy * (bx + ax_ / 2)) / 1e3
    return Check("pile cap punching (corner pile)", gamma0 * Fld, R, "kN", "JTG 3362-2018 8.5.5-4",
                 {"ax": ax_, "ay": ay_, "alpha_px": apx, "alpha_py": apy})


def pile_cap_punching_edge(Fld: float, *, bp: float, bx: float, ax: float, h0: float, b: float, concrete,
                           gamma0: float = 1.0) -> Check:
    """upward punching of an edge pile (8.5.5 item 2-2), when bp + 2 h0 ≤ b: γ0 Fld ≤ 0.6 ftd h0 [α'px (bp + h0) +
    0.667 (2 bx + ax)] (8.5.5-7); bp: the square pile's side (round 0.8 D); bx: cap edge to the inner edge of the pile;
    a: pile edge to the column edge within 0.2 h0 … h0; b: figure 8.5.5b (the cap dimension along the edge)"""
    c = _c(concrete)
    ax_, lx, apx = _alpha(ax, h0, 0.8)
    R = 0.6 * c.ftd * h0 * (apx * (bp + h0) + 0.667 * (2 * bx + ax_)) / 1e3
    chk = Check("pile cap punching (edge pile)", gamma0 * Fld, R, "kN", "JTG 3362-2018 8.5.5-7",
                {"ax": ax_, "alpha_px": apx})
    if bp + 2 * h0 > b:
        chk.failed.append(f"bp + 2h0 = {bp + 2 * h0:.0f} > b = {b:.0f} mm: 8.5.5-7 does not apply")
    return chk
