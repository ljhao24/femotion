"""Effective flange widths of JTG 3362-2018: T and I girders (4.3.3) and box girders (4.3.4).

Lengths in any one unit (mm with the section builders). 4.3.2: flexural resistance and stresses use the effective width
of the compression flange (commentary: at transfer, the axial stress of a prestressing force may use the full width).
Pass the result to jtg.tee(bf=…) or jtg.box(b_top=…).

    from femotion.codes import jtg
    bf = jtg.tee_flange_width(L=30_000, b=200, hf=180, spacing=2200, region="end span")        # inner girder
    bf_out = jtg.tee_flange_width_outer(bf, b=200, overhang=900, t_overhang=160)
    bm = jtg.box_flange_width(bi=4500, l=40_000, x=20_000, kind="end span", right=40_000)    # one flange part

Clauses, formulas, table 4.3.4: docs/codes/JTG_3362-2018.md.
"""
from __future__ import annotations

import math

import numpy as np

# share of the span for the T / I flange width (4.3.3 item 1-1)
_SPAN_SHARE = {"simple": 1 / 3, "end span": 0.27, "interior span": 0.2}


def tee_flange_width(*, b: float, hf: float, spacing: float, L: float | None = None, region: str = "simple",
                     spans: tuple[float, float] | None = None, bh: float = 0.0, hh: float = 0.0) -> float:
    """effective width b'f of the compression flange of an inner T or I girder (4.3.3 item 1): the least of
    1) a share of the computed span L: simply supported L/3; continuous girders: positive moments of an interior span
    0.2 L ("interior span"), of an end span 0.27 L ("end span"); negative moments over an interior support 0.07 (L1 + L2),
    region="support" with spans=(L1, L2);
    2) the girder spacing (average distance to the neighbours);
    3) b + 2 bh + 12 h'f with b the web, bh the haunch length, h'f the thickness of the compression flange overhang;
    bh is replaced by 3 hh (hh: haunch depth at its root) when hh / bh < 1/3."""
    if region == "support":
        if spans is None:
            raise ValueError("region='support': pass spans=(L1, L2), the two spans next to the support")
        share = 0.07 * (spans[0] + spans[1])
    else:
        if region not in _SPAN_SHARE:
            raise ValueError(f"region: one of {', '.join(_SPAN_SHARE)} or 'support'")
        if L is None:
            raise ValueError("pass the computed span L")
        share = _SPAN_SHARE[region] * L
    if bh > 0 and hh / bh < 1 / 3:
        bh = 3 * hh
    return min(share, spacing, b + 2 * bh + 12 * hf)


def tee_flange_width_outer(inner: float, *, b: float, overhang: float, t_overhang: float) -> float:
    """effective flange width of an outer girder (4.3.3 item 2): half the effective width of the neighbouring inner
    girder, plus half the web, plus the smaller of 6 × the mean thickness of the outer overhang and its actual width"""
    return inner / 2 + b / 2 + min(6 * t_overhang, overhang)


# ── box girders (4.3.4) ──────────────────────────────────────────────────────
def rho_f(r: float) -> float:
    """ρf (4.3.4-2), r = bi / li: span parts of simply supported and continuous girders, interior spans of cantilevers"""
    return -6.44 * r ** 4 + 10.10 * r ** 3 - 3.56 * r ** 2 - 1.44 * r + 1.08


def rho_s(r: float) -> float:
    """ρs (4.3.4-4), r = bi / li: supports of simply supported girders, end and interior supports of continuous
    girders, cantilever parts"""
    return 21.86 * r ** 4 - 38.01 * r ** 3 + 24.57 * r ** 2 - 7.67 * r + 1.27


R_WIDE = 0.7  # commentary to 4.3.4, figure 4-9: from bi / li = 0.7 on, bmi = 0.173 li (span parts), 0.104 li (supports)


def box_flange_coefficient(bi: float, li: float, *, support: bool = False, h: float | None = None) -> float:
    """ρ = bmi / bi of one flange part (4.3.4): ρs at supports (support=True), else ρf, with r = bi / li, up to r = 0.7;
    from 0.7 on the constant widths of the commentary's figure 4-9 (bmi,f = 0.247 × 0.7 li = 0.173 li, bmi,s = 0.148 ×
    0.7 li = 0.104 li). Program convention: ρ ≤ 1 (the fitted curves exceed 1 for very small bi / li; figure 4-9 stops at
    bmi / bi = 1). The actual width counts when the girder depth h ≥ bi / 0.3 (4.3.4, last paragraph)."""
    if h is not None and h >= bi / 0.3:
        return 1.0
    r = bi / li
    if r >= R_WIDE:
        return (0.104 if support else 0.173) / r
    return min(rho_s(r) if support else rho_f(r), 1.0)


def box_theoretical_span(l: float, kind: str, *, left: float | None = None, right: float | None = None):
    """(li at the left support, li of the span part, li at the right support) of table 4.3.4 for a span of computed
    length l: simply supported (l, l, l); continuous end span with its end support on the left (0.8 l, 0.8 l,
    0.2 (l + right)); continuous interior span (0.2 (l + left), 0.6 l, 0.2 (l + right)); cantilever (1.5 l for all).
    left, right: the neighbouring spans (nan where not given)."""
    nb = lambda o: 0.2 * (l + o) if o is not None else math.nan
    if kind == "simple":
        return l, l, l
    if kind == "end span":
        return 0.8 * l, 0.8 * l, nb(right)
    if kind == "interior span":
        return nb(left), 0.6 * l, nb(right)
    if kind == "cantilever":
        return 1.5 * l, 1.5 * l, 1.5 * l
    raise ValueError("kind: 'simple', 'end span', 'interior span', 'cantilever'")


def box_flange_width(bi: float, l: float, x: float, *, kind: str = "simple", left: float | None = None,
                     right: float | None = None, h: float | None = None) -> float:
    """effective width bmi (4.3.4) of one flange part of actual width bi (figure 4.3.4: each top or bottom flange part
    beside a web) at x along a span of computed length l (from its left support):
    kind="simple": ρs bi at both supports (li = l), ρf bi (li = l) over the middle part l − 2a;
    kind="end span" (end support at x = 0, interior support at x = l, the next span `right`): ρs (li = 0.8 l) at the end
    support, ρf (li = 0.8 l) over l − a − c, ρs (li = 0.2 (l + right)) at the interior support;
    kind="interior span": ρs (li = 0.2 (l + left)) and ρs (li = 0.2 (l + right)) at the two supports, ρf (li = 0.6 l) over
    l − 2c.
    a = bi (the actual width of the flange part), at most 0.25 l; c = 0.1 l; within a and c straight-line interpolation
    between ρs bi and ρf bi (table 4.3.4 notes). Cantilevers: box_flange_coefficient with li = 1.5 l."""
    a, c = min(bi, 0.25 * l), 0.1 * l
    co = lambda li, sup: box_flange_coefficient(bi, li, support=sup, h=h) * bi
    if kind == "simple":
        sup_l = sup_r = co(l, True)
        mid = co(l, False)
        zl, zr = a, a
    elif kind == "end span":
        if right is None:
            raise ValueError("end span: pass right=, the span beyond the interior support")
        sup_l, mid, sup_r = co(0.8 * l, True), co(0.8 * l, False), co(0.2 * (l + right), True)
        zl, zr = a, c
    elif kind == "interior span":
        if left is None or right is None:
            raise ValueError("interior span: pass left= and right=, the neighbouring spans")
        sup_l, mid, sup_r = co(0.2 * (l + left), True), co(0.6 * l, False), co(0.2 * (l + right), True)
        zl, zr = c, c
    else:
        raise ValueError("kind: 'simple', 'end span' or 'interior span'")
    if not 0 <= x <= l:
        raise ValueError("0 ≤ x ≤ l")
    return float(np.interp(x, [0.0, zl, l - zr, l], [sup_l, mid, mid, sup_r]))
