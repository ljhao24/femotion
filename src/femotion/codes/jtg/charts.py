"""Charts along a girder for design calculation reports (计算书): the moment envelope against the resisting moment
diagram (抵抗弯矩图), the shear envelope against the stirrup zones with the inclined sections of 5.2.8–5.2.11 on the
girder's elevation, utilisation along the girder, edge stresses, deflection and crack width against their limits, and
charts that viz.stack puts on one shared x axis. Plain SVG through femotion.viz (no plotting library).

Units as the rest of femotion.codes.jtg: positions along the girder in m, moments kN·m, shears kN, stresses MPa,
deflections and crack widths mm, section dimensions mm.

    from femotion import viz
    from femotion.codes import jtg
    from femotion.codes.jtg import charts
    zones = [(0, 11.2, span_sec), (11.2, 20.8, pier_sec), (20.8, 32, span_sec)]
    fm = charts.moment_chart(x, Mmax, Mmin, gamma0=1.0, resistance=charts.flexural_resistance(zones),
                             shaved=Mmin_r, supports=[0, 16, 32])
    Vu = charts.shear_resistance(zones, stirrups, supports=supports, x=x, Md=Md)
    fv = charts.shear_chart(x, Vmax, Vmin, resistance=Vu, search=search, supports=[0, 16, 32])
    el = charts.beam_elevation(0, 32, zones, supports=supports, search=search, stirrups=stirrups)
    page = viz.html([viz.stack([el, fv]), fm])

Program conventions:
- Design envelopes (basic combination) are passed as the combinations give them, without γ0; the charts plot γ0 Sd
  against the resistance and say so in the legend. Further envelopes (`others`: characteristic, frequent, …) are
  plotted as given.
- Moments are drawn on the tension side, sagging below the axis (the custom of Chinese design reports); the tick
  labels keep the sign of the moment (sagging +). tension_side=False draws sagging up. Shears, stresses and
  utilisation: positive up; deflections: downward positive, drawn down.
- Resistances along the girder are Profiles: steps (x0, x1, value) for zones of one section or one stirrup layout,
  or sampled curves. At a zone boundary the value of smaller magnitude counts (the weaker side).
- Utilisation γ0 Sd / R at every data point: sagging γ0 Mmax / Mu where Mmax > 0, hogging γ0 |Mmin| / |Mu⁻| where
  Mmin < 0, the larger of the two; positions outside every resistance zone are not checked.
- The resistance of a stirrup zone (shear_resistance) is jtg.shear with that zone's stirrups and section, α1 by the
  half spans of 5.2.9 and the tension face from the sign of the accompanying moment; bent-up bars are left out (they
  count per inclined section, shear_search). The inclined sections themselves come from jtg.shear_search: drawn on
  the elevation from the tension face at x_t to the compression face at x_c, the governing one marked on the shear
  chart with its own γ0 Vd and Vu.
- Each chart is a viz.Figure; `height` is the height of the plot box in px (Figure(plot_height=…): the figure grows
  around it for the title, legend, tick labels and notes). Style and conventions shared with the other drawings:
  docs/figures.md. Results are kept on the figure: `fig.critical` (moment_chart), `fig.governing`
  (shear_chart), `fig.maxima` (utilization_chart), `fig.extremes` (the others).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from ...viz import COLORS as C
from ...viz import (TEXT_SIZE, Figure, beyond, dim_chain, number, split, support_symbol, svg_line, svg_poly, svg_text,
                   text_box, text_width)
from .rc import RCSection, flexure as _flexure, shear as _shear

# classes of these charts besides those of femotion.viz (the "ch-…" names only tag elements for scripts and tests)
CSS = f"""
.fm-env2 {{ fill: none; stroke: {C['muted']}; stroke-width: 1.2; stroke-dasharray: 5 3; stroke-linejoin: round; }}
.fm-env3 {{ fill: none; stroke: {C['muted']}; stroke-width: 1.2; stroke-dasharray: 1.5 2.5; stroke-linejoin: round; }}
.fm-env4 {{ fill: none; stroke: {C['load']}; stroke-width: 1.2; stroke-dasharray: 8 3 2 3; stroke-linejoin: round; }}
.fm-band1 {{ fill: {C['s2']}; fill-opacity: 0.10; stroke: none; }}
.fm-band2 {{ fill: {C['compression']}; fill-opacity: 0.16; stroke: none; }}
.fm-edge1 {{ fill: none; stroke: {C['s2']}; stroke-width: 1.6; stroke-linejoin: round; }}
.fm-edge2 {{ fill: none; stroke: {C['s3']}; stroke-width: 1.6; stroke-linejoin: round; }}
.fm-incl-grid {{ fill: none; stroke: {C['hairline']}; stroke-width: 0.7; }}
.fm-incl {{ fill: none; stroke: {C['concrete_edge']}; stroke-width: 1.0; }}
.fm-incl-gov {{ fill: none; stroke: {C['warn']}; stroke-width: 2.2; stroke-linecap: round; }}
.fm-bent {{ fill: none; stroke: {C['s2']}; stroke-width: 1.3; }}
"""
SERIES = ("s1", "s2", "s3", "s4", "s5", "s6")
OTHERS = ("env2", "env3", "env4", "s6")

TEXT = {
    "zh": {
        "x": "x (m)", "M": "弯矩 M (kN·m)", "V": "剪力 V (kN)", "u": "利用率", "sigma": "正应力 σ (MPa，压为正)",
        "sigma_t": "正应力 σ (MPa，拉为正)", "f": "挠度 f (mm)", "w": "裂缝宽度 Wcr (mm)",
        "m_title": "弯矩包络图与抵抗弯矩图", "v_title": "剪力包络图与抗剪承载力", "e_title": "斜截面抗剪验算位置",
        "u_title": "验算利用率沿梁分布", "s_title": "截面上、下缘正应力", "f_title": "挠度", "w_title": "裂缝宽度",
        "Md": "γ0Md 包络（γ0 = {g}）", "Vd": "γ0Vd 包络（γ0 = {g}）", "Mu": "抵抗弯矩 Mu", "before": "削峰前",
        "Vu": "箍筋分段抗剪承载力 Vu", "mk": "控制截面 γ0Md / Mu", "mkv": "最不利斜截面 γ0Vd / Vu",
        "vlim": "截面上限", "vlim_short": "上限",
        "crit": "控制截面 x = {x} m（{sign}）：γ0Md = {d} kN·m，Mu = {r} kN·m，利用率 {u}",
        "crit_margin": "；最小富余 Mu − γ0Md = {m} kN·m", "sag": "正弯矩", "hog": "负弯矩",
        "ok": " → 满足", "nok": " → 不满足",
        "gov": "最不利斜截面 x_{{c}} = {xc} m，x_{{t}} = {xt} m，C = {C} m：γ0Vd = {d} kN，Vu = {r} kN，比值 {u}",
        "lim": "端支点 x = {x} m 截面上限（5.2.11）：γ0V0d = {d} kN，0.51×10⁻³√fcu,k·b·h0 = {r} kN",
        "grid": "斜截面（逐点搜索）", "pos": "5.2.8 规定位置", "govl": "最不利斜截面", "bent": "弯起钢筋",
        "gov_short": "最不利 {u}", "scale": "竖向未按比例；斜截面从受拉边 x_{t} 画到受压端 x_{c}",
        "legs": {1: "单肢", 2: "双肢", 3: "三肢", 4: "四肢", 6: "六肢"}, "leg": "{n}肢",
        "umax": "{label}（最大 {r}）", "unote": "{label}：最大 {r}（x = {x} m）", "limit": "限值", "top": "上缘",
        "bottom": "下缘", "edge": "{label}：{lo} ～ {hi} MPa", "over": "{label} 超出 {lim}：x = {x} m 处 {v}",
        "within": "均在限值以内", "fmax": "{label}：最大 {v} mm（x = {x} m）", "defl": "挠度",
        "crack": "裂缝宽度", "sep": "；",
    },
    "en": {
        "x": "x (m)", "M": "M (kN·m)", "V": "V (kN)", "u": "utilisation", "sigma": "σ (MPa, compression +)",
        "sigma_t": "σ (MPa, tension +)", "f": "f (mm)", "w": "Wcr (mm)",
        "m_title": "Moment envelope and resisting moment", "v_title": "Shear envelope and shear resistance",
        "e_title": "Inclined sections checked", "u_title": "Utilisation along the girder",
        "s_title": "Edge stresses", "f_title": "Deflection", "w_title": "Crack width",
        "Md": "γ0Md envelope (γ0 = {g})", "Vd": "γ0Vd envelope (γ0 = {g})", "Mu": "resisting moment Mu",
        "before": "before peak shaving", "Vu": "Vu of the stirrup zones", "mk": "governing γ0Md / Mu",
        "mkv": "governing inclined section γ0Vd / Vu", "vlim": "section limit", "vlim_short": "limit",
        "crit": "governing section x = {x} m ({sign}): γ0Md = {d} kN·m, Mu = {r} kN·m, utilisation {u}",
        "crit_margin": "; least margin Mu − γ0Md = {m} kN·m", "sag": "sagging", "hog": "hogging",
        "ok": " → OK", "nok": " → NOT OK",
        "gov": "governing inclined section x_{{c}} = {xc} m, x_{{t}} = {xt} m, C = {C} m: γ0Vd = {d} kN, Vu = {r} kN, "
               "ratio {u}",
        "lim": "section limit at the end support x = {x} m (5.2.11): γ0V0d = {d} kN, 0.51e-3 √fcu,k b h0 = {r} kN",
        "grid": "inclined sections (search)", "pos": "5.2.8 positions", "govl": "governing", "bent": "bent-up bars",
        "gov_short": "governing {u}", "scale": "not to scale vertically; inclined sections from the tension face at "
                                              "x_{t} to the compression end x_{c}",
        "legs": {}, "leg": "{n} legs",  # zh: 双肢, 四肢 …
        "umax": "{label} (max {r})", "unote": "{label}: max {r} (x = {x} m)", "limit": "limit", "top": "top edge",
        "bottom": "bottom edge", "edge": "{label}: {lo} … {hi} MPa", "over": "{label} beyond {lim}: {v} at x = {x} m",
        "within": "all within the limits", "fmax": "{label}: max {v} mm (x = {x} m)", "defl": "deflection",
        "crack": "crack width", "sep": "; ",
    },
}


def _text(lang: str) -> dict:
    if lang not in TEXT:
        raise ValueError("lang: 'zh' or 'en'")
    return TEXT[lang]


def _ratio(u: float, digits: int) -> str:
    return f"{u:.{digits}f}" if math.isfinite(u) else "∞"


# ── quantities along the girder ──────────────────────────────────────────────
@dataclass
class Profile:
    """a quantity along the girder: steps, rows of (x0, x1, value) (the resistance of reinforcement or stirrup zones),
    or a curve sampled at x. Calling it gives the value at positions x: at a zone boundary the one of smaller
    magnitude, NaN outside every zone (steps) or outside the sampled range (curve)."""
    rows: list | None = None
    x: np.ndarray | None = None
    v: np.ndarray | None = None

    @classmethod
    def steps(cls, rows) -> "Profile":
        rows = sorted((float(a), float(b), float(v)) for a, b, v in rows)
        if not rows or any(b <= a for a, b, _ in rows):
            raise ValueError("steps: rows of (x0, x1, value) with x0 < x1")
        return cls(rows=rows)

    @classmethod
    def curve(cls, x, v) -> "Profile":
        x, v = np.asarray(x, float), np.asarray(v, float)
        if x.ndim != 1 or x.shape != v.shape or len(x) < 2 or np.any(np.diff(x) < 0):
            raise ValueError("curve: x non-decreasing and v of the same length (at least 2 points)")
        return cls(x=x, v=v)

    @property
    def is_steps(self) -> bool:
        return self.rows is not None

    @property
    def extent(self) -> tuple[float, float]:
        if self.is_steps:
            return self.rows[0][0], max(b for _, b, _ in self.rows)
        return float(self.x[0]), float(self.x[-1])

    def values(self) -> np.ndarray:
        return np.array([v for *_, v in self.rows]) if self.is_steps else self.v

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, float))
        if not self.is_steps:
            out = np.interp(x, self.x, self.v)
            out[(x < self.x[0] - 1e-9) | (x > self.x[-1] + 1e-9)] = np.nan
            return out
        a0, a1 = self.extent
        tol = 1e-9 * max(1.0, abs(a1 - a0))
        out = np.full(x.shape, np.nan)
        for i, xi in enumerate(x):
            hit = [v for a, b, v in self.rows if a - tol <= xi <= b + tol]
            if hit:
                out[i] = min(hit, key=abs)
        return out

    def lines(self) -> list[list[tuple[float, float]]]:
        """polylines to draw: contiguous steps joined by vertical jumps"""
        if not self.is_steps:
            return split(self.x, self.v)
        out, cur = [], []
        tol = 1e-9 * max(1.0, abs(self.extent[1] - self.extent[0]))
        for a, b, v in self.rows:
            if cur and abs(cur[-1][0] - a) <= tol:
                cur += [(a, v), (b, v)]
            else:
                if cur:
                    out.append(cur)
                cur = [(a, v), (b, v)]
        out.append(cur)
        return out


def _as_profile(v, what: str) -> Profile | None:
    if v is None or isinstance(v, Profile):
        return v
    try:
        return Profile.steps(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: a Profile or rows of (x0, x1, value)") from e


@dataclass
class FlexuralResistance:
    """the resisting moment along the girder: sagging Mu ≥ 0 and hogging −Mu ≤ 0 (with its sign), each a Profile;
    `zones` keeps the (x0, x1, RCSection) rows it was built from"""
    sagging: Profile
    hogging: Profile
    zones: list = field(default_factory=list)

    @classmethod
    def from_arrays(cls, x, Mu_sagging, Mu_hogging) -> "FlexuralResistance":
        """sampled resistances (e.g. of a prestressed girder, one section per node); the hogging one in either sign"""
        return cls(Profile.curve(x, np.abs(np.asarray(Mu_sagging, float))),
                   Profile.curve(x, -np.abs(np.asarray(Mu_hogging, float))))

    def at(self, x) -> tuple[np.ndarray, np.ndarray]:
        return self.sagging(x), self.hogging(x)


def _mu(sec: RCSection, sign: float, kw: dict) -> float:
    """Mu of jtg.flexure; 0 when the face in tension has no steel (e.g. hogging over a simply supported girder)"""
    try:
        return _flexure(sec, sign, **kw).resistance
    except ValueError as e:
        if "no steel" in str(e):
            return 0.0
        raise


def flexural_resistance(zones, **flexure_kw) -> FlexuralResistance:
    """the resisting moment diagram (材料图) of reinforcement zones: rows of (x0, x1, RCSection) → Mu of each zone by
    jtg.flexure, sagging and hogging (the section turned over), as steps. flexure_kw go to jtg.flexure (pc_class,
    xi_b_bonded_only); the resistance is the one jtg.flexure reports, its other conditions (x ≤ ξb h0, minimum
    reinforcement) are not part of the diagram; a face without tension steel resists nothing (0)."""
    rows_s, rows_h, keep, cache = [], [], [], {}
    for a, b, sec in zones:
        if not isinstance(sec, RCSection):
            raise ValueError("zones: rows of (x0, x1, RCSection)")
        if id(sec) not in cache:
            cache[id(sec)] = (_mu(sec, 1.0, flexure_kw), _mu(sec, -1.0, flexure_kw))
        p, n = cache[id(sec)]
        rows_s.append((a, b, p))
        rows_h.append((a, b, -abs(n)))
        keep.append((float(a), float(b), sec))
    return FlexuralResistance(Profile.steps(rows_s), Profile.steps(rows_h), keep)


def _as_flexural(r) -> FlexuralResistance | None:
    if r is None or isinstance(r, FlexuralResistance):
        return r
    rows = list(r)
    if rows and all(len(row) == 3 and isinstance(row[2], RCSection) for row in rows):
        return flexural_resistance(rows)
    if rows and all(len(row) == 4 for row in rows):
        return FlexuralResistance(Profile.steps([(a, b, abs(p)) for a, b, p, _ in rows]),
                                  Profile.steps([(a, b, -abs(n)) for a, b, _, n in rows]))
    raise ValueError("resistance: a FlexuralResistance, rows of (x0, x1, RCSection) or rows of (x0, x1, Mu⁺, Mu⁻)")


def _ratios(x, dmax, dmin, res: FlexuralResistance):
    """(ratio, sagging ratio, hogging ratio, sagging margin, hogging margin) at every point; NaN where not checked"""
    pos, neg = res.at(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where(dmax > 0, np.where(pos > 0, dmax / pos, np.inf), 0.0)
        rn = np.where(dmin < 0, np.where(neg < 0, dmin / neg, np.inf), 0.0)
    rp[np.isnan(pos)] = np.nan
    rn[np.isnan(neg)] = np.nan
    mp = np.where(dmax > 0, pos - dmax, np.inf)
    mn = np.where(dmin < 0, dmin - neg, np.inf)  # |Mu⁻| − γ0|Mmin|
    mp[np.isnan(pos)] = np.nan
    mn[np.isnan(neg)] = np.nan
    both = np.where(np.isnan(rp) & np.isnan(rn), np.nan, np.fmax(rp, rn))
    return both, rp, rn, mp, mn


def flexure_utilization(x, Mmax, Mmin, resistance, gamma0: float = 1.0) -> np.ndarray:
    """γ0 Md / Mu at every point (the larger of sagging γ0 Mmax / Mu where Mmax > 0 and hogging γ0 |Mmin| / |Mu⁻|
    where Mmin < 0; 0 where neither acts; inf without resistance; NaN outside every resistance zone). Mmax, Mmin
    without γ0 (the basic combination, after peak shaving where it applies)."""
    x, Mmax, Mmin = _arrays(x, Mmax, Mmin)
    res = _as_flexural(resistance)
    return _ratios(x, gamma0 * Mmax, gamma0 * Mmin, res)[0]


def shear_utilization(search) -> tuple[np.ndarray, np.ndarray]:
    """(x_c, ratio) of the inclined sections of a jtg.shear_search, sorted by x_c, the highest ratio where several
    share one compression end"""
    best: dict[float, float] = {}
    for s in search.sections:
        k = round(s.x_c, 9)
        best[k] = max(best.get(k, -math.inf), s.ratio)
    xs = np.array(sorted(best))
    return xs, np.array([best[k] for k in xs])


def _norm_supports(supports) -> list[tuple[float, str, float]]:
    """floats (the outer ones "end", the others "interior") or rows (x, kind[, diaphragm width]) as shear_search"""
    rows = list(supports or ())
    if not rows:
        return []
    if all(isinstance(s, (int, float, np.floating, np.integer)) for s in rows):
        xs = sorted(float(s) for s in rows)
        return [(v, "end" if v in (xs[0], xs[-1]) else "interior", 0.0) for v in xs]
    out = []
    for s in rows:
        if isinstance(s, (int, float, np.floating, np.integer)):
            out.append((float(s), "interior", 0.0))
        else:
            out.append((float(s[0]), str(s[1]), float(s[2]) if len(s) > 2 else 0.0))
    return sorted(out)


def _alpha1_auto(sup):
    """5.2.9 α1 as jtg.shear_search: 0.9 in the half span next to an interior support and on a cantilever arm"""
    spans = list(zip(sup[:-1], sup[1:]))

    def f(u):
        for (a, ka, _), (b, kb, _) in spans:
            if a - 1e-9 <= u <= b + 1e-9:
                if "free" in (ka, kb) and ({ka, kb} & {"end", "interior"}):
                    return 0.9
                mid = (a + b) / 2
                if (ka == "interior" and u <= mid + 1e-9) or (kb == "interior" and u >= mid - 1e-9):
                    return 0.9
                return 1.0
        return 1.0
    return f, [(a + b) / 2 for (a, _, _), (b, _, _) in spans]


def _zone_fn(v, const, name):
    if const(v):
        return (lambda u: v), []
    rows = sorted((float(a), float(b), val) for a, b, val in v)

    def f(u):
        for a, b, val in rows:
            if a - 1e-9 <= u <= b + 1e-9:
                return val
        raise ValueError(f"{name}: no zone contains x = {u:.3f} m")
    return f, rows


def shear_resistance(section, stirrups, *, supports=(), x=None, Md=None, alpha1="auto", stirrup_steel=None,
                     shear_kw=None, span=None) -> Profile:
    """Vu of the stirrup zones along the girder as steps (kN): jtg.shear with the zone's stirrups and section, Vd = 0
    (only the resistance is read). section: an RCSection or rows (x0, x1, RCSection); stirrups: (legs, diameter,
    spacing) or rows (x0, x1, (legs, diameter, spacing)), as jtg.shear_search. alpha1 "auto": 0.9 in the half span next
    to an interior support and on a cantilever arm, 1.0 elsewhere (5.2.9; supports as shear_search, or plain positions
    with the outer ones as end supports), or a number. x, Md: the accompanying moment (sagging +): where it is negative
    the top is the tension face (h0 and P from the top steel); without it the girder is taken as sagging throughout.
    shear_kw: further arguments of jtg.shear (pc_class, alpha3, …). span: (x0, x1) when neither section nor stirrups
    are given as zones (default: the supports, then x). Steps change at zone boundaries, at the middle of spans next to
    interior supports and where Md changes sign."""
    sec_at, sec_rows = _zone_fn(section, lambda v: isinstance(v, RCSection), "section")
    st_at, st_rows = _zone_fn(stirrups, lambda v: isinstance(v, (tuple, list)) and len(v) == 3
                              and all(isinstance(e, (int, float)) for e in v), "stirrups")
    sup = _norm_supports(supports)
    bounds = [r[0] for r in sec_rows + st_rows] + [r[1] for r in sec_rows + st_rows]
    if span is not None:
        lo, hi = map(float, span)
    elif sec_rows or st_rows:
        lo = max(min(r[0] for r in rows) for rows in (sec_rows, st_rows) if rows)
        hi = min(max(r[1] for r in rows) for rows in (sec_rows, st_rows) if rows)
    elif len(sup) >= 2:
        lo, hi = sup[0][0], sup[-1][0]
    elif x is not None:
        lo, hi = float(np.min(x)), float(np.max(x))
    else:
        raise ValueError("shear_resistance: give zones, supports, x or span for the extent")
    if alpha1 == "auto":
        a1, mids = _alpha1_auto(sup) if len(sup) >= 2 else ((lambda u: 1.0), [])
        bounds += mids
    else:
        a1 = alpha1 if callable(alpha1) else (lambda u, v=float(alpha1): v)
    hog_at = lambda u: False  # noqa: E731
    if Md is not None:
        if x is None:
            raise ValueError("Md needs x")
        xa, Ma = np.asarray(x, float), np.asarray(Md, float)
        tol_m = 1e-9 * max(1.0, float(np.abs(Ma).max()))
        for k in range(len(xa) - 1):
            m0, m1 = Ma[k], Ma[k + 1]
            if (m0 < -tol_m) != (m1 < -tol_m) and xa[k + 1] > xa[k]:
                bounds.append(xa[k] + (xa[k + 1] - xa[k]) * m0 / (m0 - m1) if m0 != m1 else xa[k])
        hog_at = lambda u: float(np.interp(u, xa, Ma)) < -tol_m  # noqa: E731
    kw_at = shear_kw if callable(shear_kw) else (lambda u: dict(shear_kw or {}))
    tol = 1e-9 * max(1.0, hi - lo)
    pts = sorted({lo, hi} | {float(b) for b in bounds if lo + tol < b < hi - tol})
    rows = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a <= tol:
            continue
        u = (a + b) / 2
        v = _shear(sec_at(u), 0.0, st_at(u), hogging=hog_at(u), alpha1=a1(u), stirrup_steel=stirrup_steel,
                   **kw_at(u)).resistance
        if rows and abs(rows[-1][2] - v) <= 1e-12 * max(1.0, abs(v)):
            rows[-1] = (rows[-1][0], b, v)
        else:
            rows.append((a, b, v))
    return Profile.steps(rows)


# ── drawing helpers ─────────────────────────────────────────────────────────
def _arrays(x, *ys):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x: positions along the girder (at least 2)")
    out = [x]
    for y in ys:
        y = np.asarray(y, float)
        if y.shape != x.shape:
            raise ValueError("the values must have the length of x")
        out.append(y)
    return out


def _clamp(v, lo, hi):
    """values kept finite and within a little beyond the box (the clip path hides the rest)"""
    span = hi - lo
    v = np.asarray(v, float)
    return np.where(np.isnan(v), np.nan, np.clip(v, lo - 0.2 * span, hi + 0.2 * span))


def _y_range(arrays, plot_px: float, pad_px: float = 20.0, zero: bool = True, floor=None, ceil=None):
    vals = np.concatenate([np.ravel(np.asarray(a, float)) for a in arrays if a is not None and np.size(a)] or [[0.0]])
    vals = vals[np.isfinite(vals)]
    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:  # flat: one unit of room
        hi += 1.0
    pad = (hi - lo) * pad_px / max(plot_px - 2 * pad_px, 40.0)
    lo2, hi2 = lo - pad, hi + pad
    if floor is not None and lo >= floor - 1e-12:
        lo2 = floor  # nothing below it: no padding under it
    if ceil is not None and hi <= ceil + 1e-12:
        hi2 = ceil
    return lo2, hi2


def _chart(width, plot_px, box, title, xlabel, ylabel, yfmt=None) -> Figure:
    """a chart with its plot box plot_px high"""
    fig = Figure(width=width, plot_height=plot_px, equal=False, title=title, css=CSS)
    return fig.set_box(*box).axes(xlabel=xlabel, ylabel=ylabel, yfmt=yfmt)


def _verdict(ok: bool) -> str:
    return "ok" if ok else "fail"


def _step_label(fig: Figure, a: float, b: float, yw: float, s: str, x0: float, x1: float, priority: int = 1):
    """the value of a step (a resistance zone) outside it, away from the axis: at its middle, else a quarter in"""
    a, b = max(a, x0), min(b, x1)
    dy = -9.0 * (1 if yw >= 0 else -1)
    fig.label(((a + b) / 2, yw), s, dy=dy, room=(a, b), priority=priority,
              alt=[((a + (b - a) * f, yw), 0, dy, "middle") for f in (0.3, 0.7, 0.15, 0.85)])


def _peak_label(fig: Figure, xp: float, yw: float, s: str, priority: int = 2):
    """a peak value next to its point, inside the diagram (towards the axis) first, then beside or outside it"""
    sg = 1 if yw >= 0 else -1
    p = (xp, yw)
    fig.label(p, s, dy=10 * sg, priority=priority,
              alt=[(p, 6, 8 * sg, "start"), (p, -6, 8 * sg, "end"), (p, 0, -10 * sg, "middle"), (p, 7, 0, "start"),
                   (p, -7, 0, "end")])


def _support_x(supports) -> list[float]:
    return [s[0] for s in _norm_supports(supports)]


def _marker(fig: Figure, xk: float, yd: float, yr: float, text: str, *, left: bool, d_cls="mk-d", r_cls="mk-r",
            tag: str = "ch-crit"):
    """the governing point: a dot on the demand, an open circle on the resistance, a dashed connector and a label"""
    fig.polyline([(xk, yd), (xk, yr)], f"conn {tag}", fit=False)
    fig.dot((xk, yd), 3.4, cls=f"{d_cls} {tag}-d", fit=False)
    fig.dot((xk, yr), 3.4, cls=f"{r_cls} {tag}-r", fit=False)
    fig.mark((xk, yd), 5.0).mark((xk, yr), 5.0)
    mid = (xk, (yd + yr) / 2)
    first, other = [(mid, -7, 0, "end"), (mid, 7, 0, "start")][::1 if left else -1]
    fig.label(mid, text, "label", dx=first[1], anchor=first[3], priority=5, required=True,
              alt=[other, ((xk, yr), 0, -12 if yr >= yd else 12, "middle")])


def _spans(sx, x0, x1):
    pts = sorted({x0, x1} | {v for v in sx if x0 < v < x1})
    return list(zip(pts[:-1], pts[1:]))


# ── moment ──────────────────────────────────────────────────────────────────
def moment_chart(x, Mmax, Mmin, *, gamma0: float = 1.0, resistance=None, shaved=None, others=(), supports=(),
                 tension_side: bool = True, critical: str = "ratio", annotate: bool = True, lang: str = "zh",
                 width: float = 720, height: float = 220, title: str | None = None, xlim=None, xlabel: bool = True,
                 legend: bool = True, notes: bool = True) -> Figure:
    """moment envelope and resisting moment diagram (弯矩包络图与抵抗弯矩图).
    x (m), Mmax, Mmin (kN·m, sagging +): the design envelope of the basic combination without γ0; plotted as γ0 Md.
    resistance: a FlexuralResistance (flexural_resistance(zones), FlexuralResistance.from_arrays), rows of
    (x0, x1, RCSection) or rows of (x0, x1, Mu⁺, Mu⁻): Mu drawn as steps (sagging on the sagging side, hogging on
    the other), compared with γ0 Md. shaved: the envelope after peak shaving over the interior supports
    (jtg.peak_shaving, JTG 3362 4.3.5): Mmin (an array) or (Mmax, Mmin); the shaved one is the design envelope, the
    original is drawn dashed where they differ. others: further envelopes as given, rows of (label, Mmax, Mmin)
    (characteristic, frequent …; not scaled by γ0). supports: positions (or rows as jtg.shear_search) for the dotted
    support lines, the peak labels per span and over the supports. tension_side: sagging drawn below the axis (the
    custom of Chinese design reports). critical: "ratio" marks the section of the largest γ0 Md / Mu, "margin" the
    least Mu − γ0 Md. Returns the Figure; fig.critical = {x, ratio, demand, resistance, margin, sign} (demand and
    resistance with their sign), fig.utilization the ratio at every x."""
    L = _text(lang)
    x, Mmax, Mmin = _arrays(x, Mmax, Mmin)
    smax, smin = (None, shaved) if shaved is None or not isinstance(shaved, tuple) else shaved
    Mmax_e = Mmax if smax is None else _arrays(x, smax)[1]
    Mmin_e = Mmin if smin is None else _arrays(x, smin)[1]
    dmax, dmin = gamma0 * Mmax_e, gamma0 * Mmin_e
    res = _as_flexural(resistance)
    if critical not in ("ratio", "margin"):
        raise ValueError("critical: 'ratio' or 'margin'")
    s = -1.0 if tension_side else 1.0
    others = [(lab, *_arrays(x, a, b)[1:]) for lab, a, b in others]
    x0, x1 = (float(x.min()), float(x.max())) if xlim is None else map(float, xlim)
    vals = [dmax, dmin, gamma0 * Mmax, gamma0 * Mmin] + [v for _, a, b in others for v in (a, b)]
    if res is not None:
        vals += [res.sagging.values(), res.hogging.values()]
    lo, hi = _y_range([s * np.asarray(v) for v in vals], height, pad_px=22)
    fig = _chart(width, height, (x0, lo, x1, hi), L["m_title"] if title is None else title, L["x"] if xlabel else "",
                 L["M"], yfmt=lambda v: number(s * v + 0.0))
    sx = _support_x(supports)
    fig.vlines(sx)
    fig.polygon(list(zip(x, s * dmax)) + list(zip(x[::-1], s * dmin[::-1])), "area-demand ch-band", fit=False)
    if smax is not None or smin is not None:
        for orig, new in ((gamma0 * Mmax, dmax), (gamma0 * Mmin, dmin)):
            diff = np.abs(orig - new) > 1e-9 * max(1.0, float(np.abs(orig).max()))
            if diff.any():
                d = np.convolve(diff.astype(float), [1, 1, 1], "same") > 0
                for part in split(x, np.where(d, s * orig, np.nan)):
                    fig.polyline(part, "demand2 ch-md-orig", fit=False)
    for k, (_, a, b) in enumerate(others):
        cls = OTHERS[k % len(OTHERS)]
        fig.polyline(zip(x, s * a), f"{cls} ch-other", fit=False)
        fig.polyline(zip(x, s * b), f"{cls} ch-other", fit=False)
    if res is not None:
        for prof, tag in ((res.sagging, "ch-mu-pos"), (res.hogging, "ch-mu-neg")):
            for part in prof.lines():
                fig.polyline([(u, s * v) for u, v in part], f"resist {tag}", fit=False)
            if annotate and prof.is_steps:
                for a, b, v in prof.rows:
                    if v == 0:
                        continue
                    _step_label(fig, a, b, s * v, number(v), x0, x1)
    fig.polyline(zip(x, s * dmax), "demand ch-md-max", fit=False)
    fig.polyline(zip(x, s * dmin), "demand ch-md-min", fit=False)

    if annotate:
        peaks = []
        for a, b in _spans(sx, x0, x1):
            m = (x >= a - 1e-9) & (x <= b + 1e-9)
            if m.any() and dmax[m].max() > 0:
                k = np.flatnonzero(m)[int(np.argmax(dmax[m]))]
                peaks.append((x[k], dmax[k]))
        inner = [v for v in sx if x0 < v < x1]
        for v in inner:
            near = np.abs(x - v) <= 0.15 * max(min([abs(v - w) for w in sx if w != v] or [x1 - x0]), 1e-9)
            if near.any() and dmin[near].min() < 0:
                k = np.flatnonzero(near)[int(np.argmin(dmin[near]))]
                peaks.append((x[k], dmin[k]))
        if not inner and dmin.min() < 0:
            k = int(np.argmin(dmin))
            peaks.append((x[k], dmin[k]))
        big = max(float(np.abs(np.r_[dmax, dmin]).max()), 1e-12)
        for xp, v in peaks:
            if abs(v) >= 0.05 * big:
                _peak_label(fig, xp, s * v, number(v))

    crit, lines = None, []
    ratio = None
    if res is not None:
        ratio, rp, rn, mp, mn = _ratios(x, dmax, dmin, res)
        valid = ~np.isnan(ratio)
        if valid.any() and ratio[valid].max() > 0:
            if critical == "ratio":
                rr_ = np.where(np.isnan(ratio), -1.0, ratio)
                k = int(np.argmax(rr_))
                if np.isinf(rr_[k]):  # no resistance: the largest moment without it
                    cand = np.flatnonzero(np.isinf(rr_))
                    k = int(cand[np.argmax(np.where(np.isinf(rp), dmax, -dmin)[cand])])
                sag = np.nan_to_num(rp[k], nan=-1.0) >= np.nan_to_num(rn[k], nan=-1.0)
            else:
                mp_, mn_ = np.nan_to_num(mp, nan=np.inf), np.nan_to_num(mn, nan=np.inf)
                k = int(np.argmin(np.fmin(mp_, mn_)))
                sag = mp_[k] <= mn_[k]
            dem = float(dmax[k] if sag else dmin[k])
            rr = float(res.sagging(x[k])[0] if sag else res.hogging(x[k])[0]) + 0.0
            u = float(rp[k] if sag else rn[k])
            crit = {"x": float(x[k]), "ratio": u, "demand": dem, "resistance": rr, "margin": abs(rr) - abs(dem),
                    "sign": "sagging" if sag else "hogging", "index": k}
            left = x[k] > x0 + 0.72 * (x1 - x0)
            _marker(fig, x[k], s * dem, s * rr, _ratio(u, 2), left=left)
            txt = L["crit"].format(x=f"{x[k]:.2f}", sign=L["sag" if sag else "hog"], d=f"{dem:.1f}", r=f"{rr:.1f}",
                                   u=_ratio(u, 3))
            if critical == "margin":
                txt += L["crit_margin"].format(m=f"{crit['margin']:.1f}")
            lines.append((txt + (L["ok"] if u <= 1 else L["nok"]), _verdict(u <= 1)))
    if legend:
        ents = [("demand", L["Md"].format(g=number(gamma0)), "line")]
        if smax is not None or smin is not None:
            ents.append(("demand2", L["before"], "line"))
        ents += [(OTHERS[k % len(OTHERS)], lab, "line") for k, (lab, _, _) in enumerate(others)]
        if res is not None:
            ents.append(("resist", L["Mu"], "line"))
        if crit:
            ents.append(("mk-d", L["mk"], "dot"))
        fig.legend(ents)
    if notes:
        fig.notes(lines)
    fig.critical, fig.utilization = crit, ratio
    return fig


# ── shear ───────────────────────────────────────────────────────────────────
def _limit_x(chk) -> float | None:
    if "x" in chk.details:
        return float(chk.details["x"])
    m = re.search(r"x = (-?[\d.]+(?:e-?\d+)?)", chk.name)
    return float(m.group(1)) if m else None


def _sign_at(x, dmax, dmin, u) -> float:
    """the sign of the larger of the two envelopes at u (both sides of a jump counted)"""
    near = np.abs(x - u) <= 1e-9 * max(1.0, abs(u))
    if near.any():
        a, b = dmax[near].max(), dmin[near].min()
    else:
        a, b = float(np.interp(u, x, dmax)), float(np.interp(u, x, dmin))
    return 1.0 if abs(a) >= abs(b) else -1.0


def shear_chart(x, Vmax, Vmin, *, gamma0: float = 1.0, resistance=None, search=None, supports=(),
                annotate: bool = True, lang: str = "zh", width: float = 720, height: float = 220,
                title: str | None = None, xlim=None, xlabel: bool = True, legend: bool = True,
                notes: bool = True) -> Figure:
    """shear envelope against the shear resistance (剪力包络图与抗剪承载力).
    x (m), Vmax, Vmin (kN): the design envelope of the basic combination without γ0 (a position may appear twice, the
    two sides of a support); plotted as γ0 Vd. resistance: Vu of the stirrup zones (shear_resistance, a Profile, or rows
    (x0, x1, Vu)), drawn as ± steps. search: a jtg.shear_search result: the governing inclined section marked at its
    compression end x_c (γ0 Vd and its Vu, with the stirrups it crosses and the bent-up bars), and the section limit of
    5.2.11 at the end supports (a short bar at 0.51e-3 √fcu,k b h0 on the side of the support shear, a dot at γ0 V0d).
    Draw the inclined sections on the girder with beam_elevation. Returns the Figure; fig.governing = {x_c, x_t, C,
    ratio, demand, resistance}, fig.limits = [{x, demand, resistance}]."""
    L = _text(lang)
    x, Vmax, Vmin = _arrays(x, Vmax, Vmin)
    dmax, dmin = gamma0 * Vmax, gamma0 * Vmin
    vu = _as_profile(resistance, "resistance")
    x0, x1 = (float(x.min()), float(x.max())) if xlim is None else map(float, xlim)
    gov = search.governing if search is not None and search.sections else None
    lims = [(c, _limit_x(c)) for c in (search.limits if search is not None else [])]
    lims = [(c, u) for c, u in lims if u is not None]
    vals = [dmax, dmin]
    if vu is not None:
        vals += [vu.values(), -vu.values()]
    vals += [[c.resistance, -c.resistance] for c, _ in lims]
    if gov is not None:
        vals.append([gov.check.resistance, -gov.check.resistance])
    lo, hi = _y_range(vals, height, pad_px=22)
    fig = _chart(width, height, (x0, lo, x1, hi), L["v_title"] if title is None else title, L["x"] if xlabel else "",
                 L["V"])
    sx = _support_x(supports)
    fig.vlines(sx)
    fig.polygon(list(zip(x, dmax)) + list(zip(x[::-1], dmin[::-1])), "area-demand ch-band", fit=False)
    if vu is not None:
        for sg, tag in ((1.0, "ch-vu-pos"), (-1.0, "ch-vu-neg")):
            for part in vu.lines():
                fig.polyline([(u, sg * v) for u, v in part], f"resist {tag}", fit=False)
        if annotate and vu.is_steps:
            for a, b, v in vu.rows:
                _step_label(fig, a, b, v, number(v), x0, x1)
    fig.polyline(zip(x, dmax), "demand ch-vd-max", fit=False)
    fig.polyline(zip(x, dmin), "demand ch-vd-min", fit=False)
    lines, limits = [], []
    w = 0.045 * (x1 - x0)
    for c, u in lims:
        sg = _sign_at(x, dmax, dmin, u)
        dirn = 1.0 if u <= (x0 + x1) / 2 else -1.0
        fig.polyline([(u, sg * c.resistance), (u + dirn * w, sg * c.resistance)], "limit ch-vlim", fit=False)
        fig.dot((u, sg * c.demand), 3.0, cls="mk-s ch-vlim-d", fit=False)
        fig.mark((u, sg * c.demand), 4.0)
        q = (u + dirn * w, sg * c.resistance)
        anc = "start" if dirn > 0 else "end"
        fig.label(q, f"{L['vlim_short']} {number(c.resistance)}", dx=3 * dirn, dy=-9 * sg, anchor=anc, priority=3,
               alt=[(q, 3 * dirn, 9 * sg, anc), (q, 4 * dirn, 0, anc), ((u, sg * c.resistance), 0, -9 * sg, anc)])
        limits.append({"x": u, "demand": c.demand, "resistance": c.resistance})
        lines.append((L["lim"].format(x=number(u), d=f"{c.demand:.1f}", r=f"{c.resistance:.1f}") +
                      (L["ok"] if c.ok else L["nok"]), _verdict(c.ok)))
    governing = None
    if gov is not None:
        sg = _sign_at(x, dmax, dmin, gov.x_c)
        d, r = gov.check.demand, gov.check.resistance
        governing = {"x_c": gov.x_c, "x_t": gov.x_t, "C": gov.C, "ratio": gov.ratio, "demand": d, "resistance": r,
                     "source": gov.source}
        _marker(fig, gov.x_c, sg * abs(d), sg * r, _ratio(gov.ratio, 2), left=gov.x_c > x0 + 0.72 * (x1 - x0))
        lines.insert(0, (L["gov"].format(xc=f"{gov.x_c:.3f}", xt=f"{gov.x_t:.3f}", C=f"{gov.C:.3f}", d=f"{abs(d):.1f}",
                                         r=f"{r:.1f}", u=_ratio(gov.ratio, 3)) + (L["ok"] if gov.ok else L["nok"]),
                         _verdict(gov.ok)))
    if annotate:  # the support shears, inside the envelope next to the jump
        big = max(float(np.abs(np.r_[dmax, dmin]).max()), 1e-12)
        for v in (sx or [x0, x1]):
            idx = np.flatnonzero(np.abs(x - v) <= 1e-6 * max(1.0, x1 - x0))
            if not len(idx):
                continue
            for arr, pick, positive in ((dmax, np.argmax, True), (dmin, np.argmin, False)):
                k = idx[int(pick(arr[idx]))]
                if abs(arr[k]) < 0.08 * big or (arr[k] > 0) != positive:
                    continue
                side = 1.0 if v <= x0 + 1e-9 else -1.0 if v >= x1 - 1e-9 else (1.0 if len(idx) > 1 and k == idx[-1]
                                                                                 else -1.0)
                p, sgn, anc = (v, arr[k]), np.sign(arr[k]), "start" if side > 0 else "end"
                fig.label(p, number(arr[k]), dx=5 * side, dy=10 * sgn, anchor=anc, priority=2,
                       alt=[(p, 5 * side, -9 * sgn, anc), (p, 5 * side, 22 * sgn, anc)])
    if legend:
        ents = [("demand", L["Vd"].format(g=number(gamma0)), "line")]
        if vu is not None:
            ents.append(("resist", L["Vu"], "line"))
        if governing:
            ents.append(("mk-d", L["mkv"], "dot"))
        if lims:
            ents.append(("limit", L["vlim"], "line"))
        fig.legend(ents)
    if notes:
        fig.notes(lines)
    fig.governing, fig.limits = governing, limits
    return fig


# ── elevation with the inclined sections ────────────────────────────────────
def _depths(section, x0, x1) -> list[tuple[float, float, float]]:
    """rows (x0, x1, h in m) from a depth in mm, an RCSection or rows (x0, x1, RCSection or depth)"""
    h_of = lambda v: (v.h if isinstance(v, RCSection) else float(v)) / 1e3  # noqa: E731
    if isinstance(section, (RCSection, int, float)):
        return [(x0, x1, h_of(section))]
    return sorted((float(a), float(b), h_of(v)) for a, b, v in section)


def _stirrup_text(st, L) -> str:
    n, d, s = st
    return f"Φ{number(d)}@{number(s)} {L['legs'].get(int(n), L['leg'].format(n=number(n)))}"


def beam_elevation(x0: float, x1: float, section, *, supports=(), search=None, stirrups=None, bent=None,
                   lang: str = "zh", width: float = 720, depth: float = 34, title: str | None = None,
                   grid: bool = True, legend: bool = True, notes: bool = True) -> Figure:
    """the girder's elevation between x0 and x1 (m) for the shear checks: outline (depth from the section, mm: a number,
    an RCSection or rows (x0, x1, RCSection or depth); drawn `depth` px deep, not to scale), supports (glyphs under the
    soffit; rows as jtg.shear_search: a diaphragm width is drawn dashed), the inclined sections of a jtg.shear_search
    from the tension face at x_t to the compression face at x_c (the tension face is the top where the accompanying
    moment is hogging): the searched ones faint (grid=False leaves them out), those at the 5.2.8 positions dark, the
    governing one bold red; bent-up bars (a BentBarDesign or rows (x at the bottom bend, x at the top bend, …) as
    jtg.shear_search(bent=…)) between the steel levels; the stirrup zones (rows (x0, x1, (legs, diameter, spacing)))
    dimensioned under the girder. Share the x range with the charts below it in a stack."""
    L = _text(lang)
    rows = _depths(section, x0, x1)
    hmax = max(h for *_, h in rows)
    h_at = lambda u: next((h for a, b, h in rows if a - 1e-9 <= u <= b + 1e-9), hmax)  # noqa: E731
    fig = Figure(width=width, plot_height=depth, equal=False, title=L["e_title"] if title is None else title, css=CSS)
    fig.set_box(x0, -hmax, x1, 0.0)
    outline = [(x0, 0.0), (x1, 0.0)]
    for a, b, h in sorted(rows, key=lambda r: -r[0]):
        outline += [(min(b, x1), -h), (max(a, x0), -h)]
    fig.raw(lambda T: svg_poly([T(*p) for p in outline], "concrete ch-beam"))
    sup = [(u, kind, wd) for u, kind, wd in _norm_supports(supports) if kind != "free"]
    for k, (u, kind, wd) in enumerate(sup):
        def glyph(T, u=u, k=k):  # the first support a pin, the others rollers, as on a load drawing
            px, py = T(u, -h_at(u))
            return support_symbol("pin" if k == 0 else "roller", px, py)
        fig.raw(lambda T, g=glyph: g(T)[0], extent=lambda T, g=glyph: [g(T)[1]])
        if wd > 0:
            for v in (u - wd / 2, u + wd / 2):
                fig.raw(lambda T, v=v: svg_line(*T(v, 0), *T(v, -h_at(v)), "guide"))
    if bent is not None:
        brows = bent.bent() if hasattr(bent, "bent") else list(bent)
        for xb, xt, *_ in brows:
            p, q = (float(xb), -0.88 * h_at(xb)), (float(xt), -0.12 * h_at(xt))
            fig.raw(lambda T, p=p, q=q: svg_line(*T(*p), *T(*q), "bent ch-bent"))
    gov = search.governing if search is not None and search.sections else None

    def incl(sec, cls):
        hog = sec.Md < 0
        ht, hc = h_at(sec.x_t), h_at(sec.x_c)
        p = (sec.x_t, 0.0 if hog else -ht)
        q = (sec.x_c, -hc if hog else 0.0)
        fig.raw(lambda T: svg_poly([T(*p), T(*q)], cls, closed=False))

    if search is not None:
        for sec in search.sections:
            if sec is gov:
                continue
            if sec.source == "grid":
                if grid:
                    incl(sec, "incl-grid ch-incl")
            else:
                incl(sec, ("incl" if sec.ok else "incl-gov") + " ch-incl")
        if gov is not None:
            incl(gov, "incl-gov ch-incl-gov")
            fig.label(((gov.x_c + gov.x_t) / 2, 0.0), L["gov_short"].format(u=_ratio(gov.ratio, 2)), dy=-9,
                      priority=5, inside=False, required=True)
    zone_rows = sorted((float(a), float(b), st) for a, b, st in stirrups) if stirrups else []

    def zone_layout(T):
        """the dimension line of the stirrup zones under the supports, the zones' texts staggered where they meet"""
        yb = T(x0, -hmax)[1] + (26 if sup else 14)
        placed, out = [], []
        size = TEXT_SIZE["small"]
        for a, b, st in zone_rows:
            s = _stirrup_text(st, L)
            cx = (T(max(a, x0), 0)[0] + T(min(b, x1), 0)[0]) / 2
            w = text_width(s, size)
            row = 0
            while any(r == row and not (cx + w / 2 + 6 < l0 or cx - w / 2 - 6 > l1) for r, l0, l1 in placed):
                row += 1
            placed.append((row, cx - w / 2, cx + w / 2))
            out.append((s, cx, yb + 12 + 14 * row))
        return yb, out

    if zone_rows:
        def zones(T):
            yb, items = zone_layout(T)
            pos = sorted({T(min(max(v, x0), x1), 0)[0] for a, b, _ in zone_rows for v in (a, b)})
            svg, boxes = dim_chain(pos, yb, [None] * (len(pos) - 1))
            svg += "".join(svg_text(cx, ty, s, "small ch-zone") for s, cx, ty in items)
            return svg, boxes + [text_box(cx, ty, s, TEXT_SIZE["small"]) for s, cx, ty in items]
        fig.raw(lambda T: zones(T)[0], extent=lambda T: zones(T)[1])
    if legend and search is not None:
        ents = []
        if grid and any(s.source == "grid" for s in search.sections):
            ents.append(("incl-grid", L["grid"], "line"))
        if any(s.source != "grid" for s in search.sections):
            ents.append(("incl", L["pos"], "line"))
        ents.append(("incl-gov", L["govl"], "line"))
        if bent is not None:
            ents.append(("bent", L["bent"], "line"))
        fig.legend(ents)
    if notes:
        fig.notes([(L["scale"], "small")])
    return fig


# ── utilisation ─────────────────────────────────────────────────────────────
def utilization_chart(series, *, limit: float = 1.0, supports=(), xlim=None, ymax: float | None = None,
                      lang: str = "zh", width: float = 720, height: float = 150, title: str | None = None,
                      xlabel: bool = True, legend: bool = True, notes: bool = True) -> Figure:
    """utilisation of several checks along the girder (利用率): series = rows of (label, x, ratio) — e.g. ("受弯", x,
    flexure_utilization(…)), ("抗剪", *shear_utilization(search)), ("裂缝宽度", x, w / w_lim) — each with its own x
    (NaN or inf breaks a line; inf is drawn at the top of the box). The line at `limit` (1.0); the parts beyond it red
    with a tinted area. The legend gives each series' maximum; the notes where it is.
    fig.maxima = [{label, x, ratio}]."""
    L = _text(lang)
    ser = []
    for lab, xs, r in series:
        xs, r = _arrays(xs, r) if np.size(xs) >= 2 else (np.atleast_1d(np.asarray(xs, float)),
                                                         np.atleast_1d(np.asarray(r, float)))
        ser.append((lab, xs, r))
    if not ser:
        raise ValueError("series: rows of (label, x, ratio)")
    allx = np.concatenate([s[1] for s in ser])
    x0, x1 = (float(np.nanmin(allx)), float(np.nanmax(allx))) if xlim is None else map(float, xlim)
    fin = np.concatenate([r[np.isfinite(r)] for _, _, r in ser] + [np.array([0.0])])
    top = ymax if ymax is not None else max(1.15 * limit, 1.08 * float(fin.max()))
    lo, hi = 0.0, top
    fig = _chart(width, height, (x0, lo, x1, hi), L["u_title"] if title is None else title, L["x"] if xlabel else "",
                 L["u"])
    fig.vlines(_support_x(supports))
    fig.polyline([(x0, limit), (x1, limit)], "limit ch-ulimit", fit=False)
    maxima, lines, ents = [], [], []
    for k, (lab, xs, r) in enumerate(ser):
        cls = SERIES[k % len(SERIES)]
        rc = _clamp(r, lo, hi)
        for part in split(xs, rc):
            if len(part) == 1:
                fig.dot(part[0], 2.6, cls=f"mk-s ch-u{k}", fit=False)
            else:
                fig.polyline(part, f"{cls} ch-u{k}", fit=False)
        for part in beyond(xs, rc, limit):
            fig.polygon(part + [(part[-1][0], limit), (part[0][0], limit)], "area-demand ch-over", fit=False)
            fig.polyline(part, f"warn ch-over{k}", fit=False)
        ok = ~np.isnan(r)
        if ok.any():
            j = int(np.flatnonzero(ok)[np.argmax(r[ok])])
            maxima.append({"label": lab, "x": float(xs[j]), "ratio": float(r[j])})
            p = (xs[j], min(rc[j], hi))  # an infinite ratio at the top of the box
            fig.dot(p, 3.2, cls=f"{'mk-d' if r[j] > limit else 'mk-s'} ch-umax", fit=False)
            fig.mark(p, 4.0)
            ents.append((cls, L["umax"].format(label=lab, r=_ratio(r[j], 2)), "line"))
            lines.append(L["unote"].format(label=lab, r=_ratio(r[j], 3), x=f"{xs[j]:.2f}"))
        else:
            ents.append((cls, lab, "line"))
    fig.label((x1, limit), number(limit), dx=-3, dy=-8, anchor="end", priority=1,
           alt=[((x0, limit), 3, -8, "start"), ((x1, limit), -3, 8, "end")])
    if legend:
        fig.legend(ents)
    if notes and lines:
        warn = any(m["ratio"] > limit for m in maxima)
        fig.notes([(L["sep"].join(lines), "warn" if warn else "")])
    fig.maxima = maxima
    return fig


# ── serviceability ──────────────────────────────────────────────────────────
def _limit_rows(limits, x0, x1, default_kind):
    """limits → rows (x0, x1, value, label, kind): a number, (value, label[, kind]) or
    (x0, x1, value[, label[, kind]])"""
    if limits is None:
        return []
    if isinstance(limits, (int, float)):
        limits = [limits]
    out = []
    for lm in limits:
        if isinstance(lm, (int, float)):
            out.append((x0, x1, float(lm), None, default_kind(float(lm))))
        elif len(lm) >= 3 and isinstance(lm[1], (int, float)) and not isinstance(lm[2], str):
            v = float(lm[2])
            out.append((float(lm[0]), float(lm[1]), v, lm[3] if len(lm) > 3 else None,
                        lm[4] if len(lm) > 4 else default_kind(v)))
        else:
            v = float(lm[0])
            out.append((x0, x1, v, lm[1] if len(lm) > 1 else None, lm[2] if len(lm) > 2 else default_kind(v)))
    for r in out:
        if r[4] not in ("upper", "lower"):
            raise ValueError("limit kind: 'upper' (the value must stay below) or 'lower'")
    return out


def _series(v, x, default_label):
    """an array, (max, min) pair, or rows / dict of label → array or pair → [(label, [arrays])]"""
    if isinstance(v, dict):
        v = list(v.items())
    if isinstance(v, list) and v and isinstance(v[0], tuple) and isinstance(v[0][0], str):
        return [(lab, _parts(a, x)) for lab, a in v]
    return [(default_label, _parts(v, x))]


def _parts(a, x):
    if isinstance(a, tuple):
        return [_arrays(x, b)[1] for b in a]
    return [_arrays(x, a)[1]]


def _limits_draw(fig, rows, sign, L, unit=""):
    for a, b, v, lab, _ in rows:
        fig.polyline([(a, sign * v), (b, sign * v)], "limit ch-limit", fit=False)
        text = (lab or L["limit"]) + " = " + number(v)
        dy = -8 if sign * v >= 0 else 8
        fig.label((b, sign * v), text + unit, dx=-3, dy=dy, anchor="end", room=(a, b), priority=1,
               alt=[((a, sign * v), 3, dy, "start"), ((b, sign * v), -3, -dy, "end"), ((a, sign * v), 3, -dy, "start")])


def _over_lines(fig, x, y, rows, sign):
    """red parts of y (drawn values) beyond the limit rows; returns [(label, x, v)] of the worst exceedance"""
    worst = []
    for a, b, v, lab, kind in rows:
        m = (x >= a - 1e-9) & (x <= b + 1e-9)
        if not m.any():
            continue
        xs, ys = x[m], y[m]
        up = (kind == "upper") == (sign > 0)
        for part in beyond(xs, ys, sign * v, above=up):
            fig.polyline(part, "warn ch-over", fit=False)
        real = sign * ys
        bad = real > v + 1e-12 if kind == "upper" else real < v - 1e-12
        if bad.any():
            j = int(np.argmax(np.where(bad, np.abs(real - v), -1)))
            worst.append((lab, float(xs[j]), float(real[j]), v))
    return worst


def stress_chart(x, *, top=None, bottom=None, limits=(), compression_positive: bool = True, supports=(), xlim=None,
                 lang: str = "zh", width: float = 720, height: float = 180, title: str | None = None,
                 xlabel: bool = True, legend: bool = True, notes: bool = True, labels=None) -> Figure:
    """normal stresses at the top and bottom edges along the girder (MPa), e.g. of a prestressed girder in service:
    top, bottom: an array or the envelope (max, min) of arrays (e.g. Check.details["edge_stress"] of
    jtg.service_stresses under Mk,max and Mk,min). compression_positive: the sign convention of the values (JTG's, as
    service_stresses gives them); drawn positive up. limits: numbers, (value, label[, kind]) or (x0, x1, value[,
    label[, kind]]) rows in the same convention, e.g. (0.5 fck, "0.5fck"), (−0.7 ftk, "−0.7ftk"); kind "upper" (the
    stress must stay below: compression limits) or "lower"; by default a positive value is upper and a negative one
    lower, and 0 lower with compression +, upper with tension +. Values beyond a limit are drawn red. labels: names of
    the two edges. fig.extremes = {edge: (min, x, max, x)}, fig.exceeded = [(limit label, x, value, limit)]."""
    L = _text(lang)
    x = _arrays(x)[0]
    names = labels or (L["top"], L["bottom"])
    edges = [(n, _parts(v, x)) for n, v in zip(names, (top, bottom)) if v is not None]
    if not edges:
        raise ValueError("give top and / or bottom")
    x0, x1 = (float(x.min()), float(x.max())) if xlim is None else map(float, xlim)
    zero_kind = "lower" if compression_positive else "upper"
    rows = _limit_rows(limits, x0, x1, lambda v: "upper" if v > 0 else "lower" if v < 0 else zero_kind)
    vals = [a for _, parts in edges for a in parts] + [[r[2] for r in rows]]
    lo, hi = _y_range(vals, height, pad_px=20)
    fig = _chart(width, height, (x0, lo, x1, hi), L["s_title"] if title is None else title, L["x"] if xlabel else "",
                 L["sigma"] if compression_positive else L["sigma_t"])
    fig.vlines(_support_x(supports))
    _limits_draw(fig, rows, 1.0, L)
    ext, over, ents = {}, [], []
    for k, (name, parts) in enumerate(edges):
        cls, band = ("edge1", "band1") if k == 0 else ("edge2", "band2")
        if len(parts) == 2:
            a, b = parts
            fig.polygon(list(zip(x, a)) + list(zip(x[::-1], b[::-1])), f"{band} ch-band{k}", fit=False)
        for j, a in enumerate(parts):
            fig.polyline(zip(x, a), f"{cls} ch-edge{k}-{j}", fit=False)
            over += _over_lines(fig, x, a, rows, 1.0)
        allv = np.concatenate(parts)
        allx = np.concatenate([x] * len(parts))
        i0, i1 = int(np.nanargmin(allv)), int(np.nanargmax(allv))
        ext[name] = (float(allv[i0]), float(allx[i0]), float(allv[i1]), float(allx[i1]))
        ents.append((cls, name, "line"))
    if legend:
        ents.append(("limit", L["limit"], "line"))
        fig.legend(ents)
    if notes:
        lines = [(L["sep"].join(L["edge"].format(label=n, lo=f"{e[0]:.2f}", hi=f"{e[2]:.2f}") for n, e in ext.items()),
                  "")]
        if over:
            lines.append((L["sep"].join(L["over"].format(label=lab or L["limit"], lim=number(v), x=f"{xx:.2f}",
                                                         v=f"{val:.2f}") for lab, xx, val, v in over), "warn"))
        elif rows:
            lines.append((L["within"], "ok"))
        fig.notes(lines)
    fig.extremes, fig.exceeded = ext, over
    return fig


def _quantity_chart(x, series, limits, *, down, default_label, axis, title, unit, supports, xlim, lang, width, height,
                    xlabel, legend, notes):
    L = _text(lang)
    x = _arrays(x)[0]
    ser = _series(series, x, default_label)
    x0, x1 = (float(x.min()), float(x.max())) if xlim is None else map(float, xlim)
    rows = _limit_rows(limits, x0, x1, lambda v: "upper")
    sg = -1.0 if down else 1.0
    vals = [sg * a for _, parts in ser for a in parts] + [[sg * r[2] for r in rows]]
    lo, hi = _y_range(vals, height, pad_px=20, floor=None if down else 0.0, ceil=0.0 if down else None)
    fig = _chart(width, height, (x0, lo, x1, hi), title, L["x"] if xlabel else "", axis,
                 yfmt=lambda v: number(sg * v + 0.0))
    sx = _support_x(supports)
    fig.vlines(sx)
    _limits_draw(fig, [(a, b, v, lab, kind) for a, b, v, lab, kind in rows], sg, L)
    ext, over, ents, lines = {}, [], [], []
    for k, (lab, parts) in enumerate(ser):
        cls = SERIES[k % len(SERIES)]
        for j, a in enumerate(parts):
            if k == 0 and len(ser) == 1:
                fig.polygon([(x[0], 0.0)] + list(zip(x, sg * np.nan_to_num(a))) + [(x[-1], 0.0)],
                            "area-resist ch-area", fit=False)
            for part in split(x, sg * a):
                fig.polyline(part, f"{cls} ch-q{k}-{j}", fit=False)
            over += _over_lines(fig, x, sg * a, rows, sg)
        allv = np.concatenate(parts)
        allx = np.concatenate([x] * len(parts))
        if np.isfinite(allv).any():
            i = int(np.nanargmax(np.where(np.isfinite(allv), allv, np.nan)))
            ext[lab] = (float(allv[i]), float(allx[i]))
            for a, b in _spans(sx, x0, x1) if len(ser) == 1 else [(x0, x1)]:
                m = (allx >= a - 1e-9) & (allx <= b + 1e-9) & np.isfinite(allv)
                if m.any():
                    j = np.flatnonzero(m)[int(np.argmax(allv[m]))]
                    if allv[j] > 0:
                        fig.dot((allx[j], sg * allv[j]), 2.8, cls="mk-s ch-qmax", fit=False)
                        fig.mark((allx[j], sg * allv[j]), 3.5)
                        _peak_label(fig, allx[j], sg * allv[j], number(allv[j], 3 if unit == "w" else None))
            lines.append(L["fmax"].format(label=lab, v=number(allv[i], 3 if unit == "w" else None),
                                          x=f"{allx[i]:.2f}"))
        ents.append((cls, lab, "line"))
    if legend:
        if rows:
            ents.append(("limit", L["limit"], "line"))
        fig.legend(ents)
    if notes:
        out = [(L["sep"].join(lines), "")] if lines else []
        if over:
            out.append((L["sep"].join(L["over"].format(label=lab or L["limit"], lim=number(v), x=f"{xx:.2f}",
                                                       v=number(val, 3 if unit == "w" else None))
                                      for lab, xx, val, v in over), "warn"))
        elif rows:
            out.append((L["within"], "ok"))
        fig.notes(out)
    fig.extremes, fig.exceeded = ext, over
    return fig


def deflection_chart(x, f, *, limit=None, supports=(), xlim=None, lang: str = "zh", width: float = 720,
                     height: float = 140, title: str | None = None, xlabel: bool = True, legend: bool = True,
                     notes: bool = True, label: str | None = None) -> Figure:
    """deflection along the girder (mm, downward +, drawn down), e.g. ηθ × the frequent live-load deflection of 6.5.3:
    f an array, or rows / dict of label → array for several curves. limit: a number, (value, label) or rows (x0, x1,
    value[, label]) per span (e.g. L/600 of each span, jtg.deflection_limit). The largest deflection of each span is
    labelled; beyond the limit red. fig.extremes = {label: (max, x)}, fig.exceeded."""
    L = _text(lang)
    return _quantity_chart(x, f, limit, down=True, default_label=label or L["defl"], axis=L["f"],
                           title=L["f_title"] if title is None else title, unit="f", supports=supports, xlim=xlim,
                           lang=lang, width=width, height=height, xlabel=xlabel, legend=legend, notes=notes)


def crack_chart(x, w, *, limit=None, supports=(), xlim=None, lang: str = "zh", width: float = 720,
                height: float = 140, title: str | None = None, xlabel: bool = True, legend: bool = True,
                notes: bool = True, label: str | None = None) -> Figure:
    """crack width along the girder (mm): w an array (NaN where not checked) or rows / dict of label → array (e.g.
    the sagging and hogging zones); limit as deflection_chart (jtg.crack_width_limit). fig.extremes, fig.exceeded."""
    L = _text(lang)
    return _quantity_chart(x, w, limit, down=False, default_label=label or L["crack"], axis=L["w"],
                           title=L["w_title"] if title is None else title, unit="w", supports=supports, xlim=xlim,
                           lang=lang, width=width, height=height, xlabel=xlabel, legend=legend, notes=notes)
