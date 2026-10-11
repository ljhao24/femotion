"""Torsion of reinforced and prestressed concrete members, JTG 3362-2018 5.5, with the torsion detailing of 9.3.13.

Units as femotion.codes.jtg.rc: mm, mm², MPa; Td in kN·m, Vd in kN. Clauses and formulas: docs/codes/JTG_3362-2018.md.
The code covers equilibrium torsion of rectangles, single-cell boxes with walls t2 ≥ 0.1 b and t1 ≥ 0.1 h, and T, I and
flanged box sections split into rectangles (5.5.5): the web (full depth) or the box carries a share of Td by its
plastic torsion modulus and is checked for shear with torsion (5.5.4); each flange carries its share as pure torsion
(5.5.1). Thin-walled bridge box girders whose walls are thinner than 5.5.1 allows are outside 5.5 (the check says so).

    from femotion.codes import jtg
    sec = jtg.rect(400, 800, concrete="C40", bottom=[(4, 25, 50)], top=[(2, 16, 45)])
    r = jtg.torsion(sec, Td=95.0, Vd=260.0, stirrups=(2, 12, 100), Ast=6 * 491, core=(320, 720))
    w = jtg.torsion_moduli(jtg.tee(200, 1300, 1600, 150, concrete="C40", bottom=[(6, 32, 90)]))
    f = jtg.torsion_flange(tsec, Td=40.0, which="top", stirrups=(2, 10, 150), Ast=8 * 113, core=(1500, 90))

stirrups = (legs in the section, diameter, spacing): ρsv = legs Asv1 / (b sv) for shear, Asv1 the area of one leg of the
closed outer stirrup for torsion. Ast: the longitudinal steel placed symmetrically round the rectangle for torsion
(prestressing steel not counted, commentary to 5.5.1). core = (bcor, hcor): the core inside the stirrups.
"""
from __future__ import annotations

import math

from .materials import Rebar, rebar as _rebar
from .rc import Check, RCSection


def _rect_wt(b: float, h: float) -> float:
    """5.5.2-1, b the shorter side"""
    b, h = min(b, h), max(b, h)
    return b * b / 6 * (3 * h - b)


def _box_dims(sec: RCSection, t_web: float | None):
    """(b, h, t1, t2) of 5.5.1 for a box built by jtg.box: b the shorter, h the longer outer side; t1 the thickness of the
    walls along h (long sides), t2 of those along b. Single cell (two webs of webs/2) unless t_web is given; the thinner
    of top and bottom slab (program convention, the smaller Wt and βa)"""
    top, web, bot = sec.pieces[:3]
    B, h = bot.b, sec.h
    tw = web.b / 2 if t_web is None else t_web
    ts = min(top.bottom - top.top, bot.bottom - bot.top)
    if h >= B:
        return B, h, tw, ts
    return h, B, ts, tw


def torsion_moduli(sec: RCSection, *, t_web: float | None = None) -> dict:
    """plastic torsion moduli (mm³, 5.5.2, 5.5.5): "body" the rectangle (5.5.2-1) or the box (5.5.2-2), "top" and
    "bottom" the flanges outside the web (5.5.5-4, -5, overhangs capped at b'f ≤ b + 6 h'f, bf ≤ b + 6 hf), "total"
    (5.5.5-6, -7), "beta_a" (box), "dims" (b, h, t1, t2 of the body). Rectangles, T and I sections (jtg.tee), boxes with
    or without cantilevers (jtg.box, single cell unless t_web)."""
    out = {"top": 0.0, "bottom": 0.0, "beta_a": 1.0, "box": False, "capped": []}
    if sec.kind == "rect":
        b, h = sec.pieces[0].b, sec.h
        out["body"] = _rect_wt(b, h)
        out["dims"] = (min(b, h), max(b, h), None, None)
    elif sec.kind == "tee":
        if not (2 <= len(sec.pieces) <= 3 and all(p.sign == 1 and hasattr(p, "b") for p in sec.pieces)
                and sec.pieces[1].b < sec.pieces[0].b and (len(sec.pieces) == 2 or sec.pieces[1].b < sec.pieces[2].b)):
            raise NotImplementedError("torsion: T and I sections built by jtg.tee (flange, web[, bottom flange])")
        top, web = sec.pieces[0], sec.pieces[1]
        b = web.b
        out["body"] = _rect_wt(b, sec.h)
        out["dims"] = (min(b, sec.h), max(b, sec.h), None, None)
        flanges = [("top", top)] + ([("bottom", sec.pieces[2])] if len(sec.pieces) > 2 else [])
        for name, p in flanges:
            hf = p.bottom - p.top
            over = p.b - b
            if over > 6 * hf:
                out["capped"].append(f"{name} flange overhang {over:.0f} mm counted as 6 hf = {6 * hf:.0f} mm (5.5.5)")
                over = 6 * hf
            out[name] = hf * hf / 2 * max(over, 0.0)
    elif sec.kind == "box":
        b, h, t1, t2 = _box_dims(sec, t_web)
        out["box"] = True
        out["web_t"] = sec.pieces[1].b / 2 if t_web is None else t_web
        out["dims"] = (b, h, t1, t2)
        out["body"] = b * b / 6 * (3 * h - b) - (b - 2 * t1) ** 2 / 6 * (3 * (h - 2 * t2) - (b - 2 * t1))
        out["beta_a"] = min(1.0, 4 * t2 / b, 4 * t1 / h)
        top, bot = sec.pieces[0], sec.pieces[2]
        hf = top.bottom - top.top
        over = top.b - bot.b
        if over > 6 * hf:
            out["capped"].append(f"cantilevers {over:.0f} mm counted as 6 hf = {6 * hf:.0f} mm (5.5.5)")
            over = 6 * hf
        out["top"] = hf * hf / 2 * max(over, 0.0)
    else:
        raise NotImplementedError("torsion of rectangles, T / I sections and boxes (jtg.rect, jtg.tee, jtg.box); "
                                  "not composite sections (8.1 has no torsion clause)")
    out["total"] = out["body"] + out["top"] + out["bottom"]
    return out


def _steel(sec, stirrup_steel):
    return sec.steel if stirrup_steel is None else (stirrup_steel if isinstance(stirrup_steel, Rebar) else _rebar(stirrup_steel))


def _prestress(sec: RCSection, h: float, zeta: float):
    """(Np0 capped at 0.3 fcd A0, A0, ep0, counted?) for 5.5.1: the prestress term counts when ep0 ≤ h/6 and ζ ≥ 1.7"""
    if not sec.prestressed:
        return 0.0, 1.0, 0.0, False
    A0, z0, _ = sec.props("transformed")
    N, zN = sec.prestress_resultant("p0")
    Np0 = min(N, 0.3 * sec.concrete.fcd * A0)
    ep0 = abs(zN - z0)
    return Np0, A0, ep0, (ep0 <= h / 6 and zeta >= 1.7)


def _zeta(fsd, Ast, sv, fsv, Asv1, Ucor):
    return fsd * Ast * sv / (fsv * Asv1 * Ucor)


def _torsion_part(chk: Check, sec: RCSection, Wt: float, beta_a: float, h_sec: float, stirrups, Ast, core, sst,
                  beta_t: float | None, q_shear: float = 0.0):
    """resistance (N·mm) of one rectangle or box under (shear with) torsion: 5.5.1-1 (beta_t None) or 5.5.4-2. ζ from the
    stirrups provided; the stirrups' term from what is left of the outer leg after the share shear needs (q_shear,
    Asv per leg per length, 5.5.6 item 2: stirrups = shear + torsion)"""
    c, st = sec.concrete, sec.steel
    legs, dia, sv = stirrups
    Asv1 = math.pi * dia * dia / 4
    bcor, hcor = core
    Acor, Ucor = bcor * hcor, 2 * (bcor + hcor)
    fsv = sst.fsv
    zeta_raw = _zeta(st.fsd, Ast, sv, fsv, Asv1, Ucor)
    Np0, A0, ep0, use_p = _prestress(sec, h_sec, zeta_raw)
    zeta = min(zeta_raw, 1.7)
    if zeta_raw < 0.6:
        chk.failed.append(f"ζ = {zeta_raw:.2f} < 0.6: too little longitudinal torsion steel for the stirrups (5.5.1)")
    if zeta_raw > 1.7:
        chk.notes.append(f"ζ = {zeta_raw:.2f} > 1.7: taken as 1.7 (5.5.1)")
    p_term = 0.05 * Np0 / A0 if use_p else 0.0
    if sec.prestressed:
        chk.notes.append("prestress term 0.05 Np0/A0 Wt counted (ep0 ≤ h/6, ζ ≥ 1.7)" if use_p else
                         f"prestress not counted (ep0 = {ep0:.0f} mm vs h/6 = {h_sec / 6:.0f} mm, ζ = {zeta_raw:.2f}): "
                         "as reinforced concrete (5.5.1)")
    conc = 0.35 * beta_a * c.ftd + p_term
    if beta_t is not None:
        conc *= beta_t
    q_prov = Asv1 / sv
    q_t = max(q_prov - q_shear, 0.0)
    Tu = conc * Wt + 1.2 * math.sqrt(zeta) * fsv * q_t * Acor
    chk.details.update(zeta=zeta_raw, zeta_used=zeta, Acor=Acor, Ucor=Ucor, Asv1=Asv1, fsv=fsv, Np0=Np0 / 1e3,
                       ep0=ep0, prestress_counted=use_p, Wt=Wt, beta_a=beta_a, q_provided=q_prov, q_shear=q_shear,
                       q_torsion=q_t)
    return Tu, use_p, zeta


def _detailing(chk: Check, sec, sst, stirrups, Ast, b_part, h_part, beta_t, b_sv):
    """9.3.13 items 3 and 4: minimum stirrup ratio ρsv = Asv / (b sv) on the width b_sv, minimum longitudinal torsion
    steel Ast / (b h) on the rectangle's sides (beta_t None: pure torsion)"""
    c = sec.concrete
    legs, dia, sv = stirrups
    rho_sv = legs * math.pi * dia * dia / 4 / (sv * b_sv)
    if beta_t is None:
        need_sv = 0.055 * c.fcd / sst.fsv
        need_st = 0.08 * c.fcd / sec.steel.fsd
    else:
        cc = sst.rho_sv_min
        need_sv = (2 * beta_t - 1) * (0.055 * c.fcd / sst.fsv - cc) + cc
        need_st = 0.08 * (2 * beta_t - 1) * c.fcd / sec.steel.fsd
    rho_st = Ast / (b_part * h_part)
    chk.details.update(rho_sv=rho_sv, rho_sv_min=need_sv, rho_st=rho_st, rho_st_min=need_st)
    if rho_sv < need_sv - 1e-12:
        chk.warnings.append(f"stirrup ratio {rho_sv:.4%} < {need_sv:.4%} (9.3.13 item 3)")
    if rho_st < need_st - 1e-12:
        chk.warnings.append(f"longitudinal torsion steel Ast/(bh) = {rho_st:.4%} < {need_st:.4%} (9.3.13 item 4)")


def torsion(sec: RCSection, Td: float, Vd: float = 0.0, *, stirrups, Ast: float, core, gamma0: float = 1.0,
            alpha1: float = 1.0, alpha3: float | None = None, pc_class: str | None = None, stirrup_steel=None,
            hogging: bool = False, t_web: float | None = None) -> Check:
    """torsion resistance of a rectangle, a box, or the web / box body of a T, I or flanged box section (which carries
    Twd = Wtw / Wt × Td, 5.5.5-1). Vd = 0: pure torsion (5.5.1); Vd > 0: shear with torsion (5.5.4), βt =
    1.5 / (1 + 0.5 Vd Wt / (Td b h0)) within 0.5 … 1.0 (5.5.4-3; boxes: βa Wt there, program reading as GB 50010).
    One set of stirrups (5.5.6 item 2: the stirrups are the sum of what shear and torsion need): the share shear needs is
    found from 5.5.4-1 (ρsv needed, spread over all legs) and taken off the outer leg; what is left enters 5.5.4-2. So the
    returned check (γ0 Td against Tu) holds exactly when the stirrups cover both; if shear alone needs more than the
    stirrups provide, that is listed (details "Vu" is 5.5.4-1 with all the stirrups, "q_shear", "q_torsion": Asv1/sv).
    Also: the section limits of 5.5.3 with the whole Td and Wt (γ0Vd/(bh0) + γ0Td/Wt ≤ 0.51 √fcu,k; at most 0.50 α2 ftd:
    detailing reinforcement suffices, 9.3.13), ζ (5.5.1-2, stirrups as provided) within 0.6 … 1.7, the prestress term
    0.05 Np0/A0 Wt only where ep0 ≤ h/6 and ζ ≥ 1.7 (then ζ = 1.7; Np0 ≤ 0.3 fcd A0; α2 = 1 otherwise), the box limits
    t2 ≥ 0.1 b, t1 ≥ 0.1 h and βa (5.5.1), the web b / hw ≥ 0.15 of T and I sections (5.5.5 note) and the minimum
    stirrups and longitudinal steel of 9.3.13 (ρsv on the shear width b). α1, α3, pc_class, hogging and stirrup_steel as
    jtg.shear (α2 of 5.2.9). Flanges: torsion_flange."""
    c = sec.concrete
    sst = _steel(sec, stirrup_steel)
    w = torsion_moduli(sec, t_web=t_web)
    b_body, h_body, t1, t2 = w["dims"]
    chk = Check("torsion" + (" with shear" if Vd else "") + (" (web)" if sec.kind == "tee" else ""), 0.0, 0.0, "kN·m",
                "JTG 3362-2018 5.5.4, 5.5.6" if Vd else "JTG 3362-2018 5.5.1", {"moduli": w}, list(w["capped"]))
    if w["box"] and t_web is None:
        chk.notes.append(f"box: one cell assumed, webs {w['web_t']:.0f} mm each (pass t_web otherwise)")
    Wt_tot, Wtw = w["total"], w["body"]
    T_body = Td * Wtw / Wt_tot
    chk.demand = gamma0 * T_body
    chk.details["T_share"] = T_body
    if w["box"] and (t2 < 0.1 * b_body - 1e-9 or t1 < 0.1 * h_body - 1e-9):
        chk.failed.append(f"box walls t1 = {t1:.0f}, t2 = {t2:.0f} mm: 5.5.1 needs t2 ≥ 0.1 b = {0.1 * b_body:.0f} and "
                          f"t1 ≥ 0.1 h = {0.1 * h_body:.0f} mm (outside 5.5)")
    if sec.kind == "tee":
        hw = sec.pieces[1].bottom - sec.pieces[1].top
        if sec.pieces[1].b / hw < 0.15:
            chk.failed.append(f"web b / hw = {sec.pieces[1].b / hw:.3f} < 0.15 (5.5.5 note)")
    s = sec.flipped() if hogging else sec
    b, h0 = s.web(), s.h0(external=False)  # bonded steel, as jtg.shear
    beta_a = w["beta_a"]
    legs, dia, sv = stirrups
    beta_t = None
    if Vd:
        Wt_bt = beta_a * Wtw if w["box"] else Wtw
        # Td → 0: 5.5.4-3 tends to its lower limit 0.5 (shear alone)
        beta_t = min(max(1.5 / (1 + 0.5 * abs(Vd) * 1e3 * Wt_bt / (abs(T_body) * 1e6 * b * h0)), 0.5), 1.0) if T_body else 0.5
        chk.details["beta_t"] = beta_t
    # α2 (5.5.3-2, 5.5.4-1): 5.2.9 when the prestress term counts, else 1
    zeta_p = _zeta(sec.steel.fsd, Ast, sv, sst.fsv, math.pi * dia * dia / 4, 2 * (core[0] + core[1]))
    alpha2 = 1.0
    if sec.prestressed and _prestress(sec, sec.h, zeta_p)[3]:
        if pc_class is None:
            if Vd:
                raise ValueError("prestressed sections in shear with torsion need pc_class='full', 'A' or 'B' (α2)")
            chk.notes.append("pc_class not given: α2 = 1.0 in 5.5.3-2")
        else:
            from .rc import shear as _shear
            alpha2 = _shear(s, 0.0, stirrups, pc_class=pc_class, alpha1=alpha1, stirrup_steel=sst).details["alpha2"]
    q_shear = 0.0
    if Vd:
        rho_sv = legs * math.pi * dia * dia / 4 / (sv * b)
        pb, _ = s.tendon_masks()
        t_, _ = s.groups()
        P = min(100 * (t_[:, 0].sum() + s.tendons[pb, 0].sum()) / (b * h0), 2.5)
        a3 = alpha3 if alpha3 is not None else (1.1 if s.width(0.01 * s.h) > b * 1.01 else 1.0)
        coef = 0.5e-4 * alpha1 * alpha2 * a3 * (10 - 2 * beta_t) * b * h0
        K = (2 + 0.6 * P) * math.sqrt(c.fcuk) * sst.fsv
        Vu = coef * math.sqrt(K * rho_sv)
        rho_need = (gamma0 * abs(Vd) / coef) ** 2 / K
        q_shear = rho_need * b / legs
        chk.details.update(Vu=Vu, shear_ratio=gamma0 * abs(Vd) / Vu, P=P, alpha1=alpha1, alpha3=a3, rho_sv_shear=rho_need)
        if gamma0 * abs(Vd) > Vu:
            chk.failed.append(f"shear with torsion: γ0Vd = {gamma0 * abs(Vd):.1f} > Vu = {Vu:.1f} kN with all the "
                              "stirrups (5.5.4-1)")
    Tu, use_p, _ = _torsion_part(chk, sec, Wtw, beta_a, sec.h, stirrups, Ast, core, sst, beta_t, q_shear)
    chk.resistance = Tu / 1e6
    # 5.5.3 section limits, whole section
    tau = gamma0 * abs(Vd) * 1e3 / (b * h0) + gamma0 * abs(Td) * 1e6 / Wt_tot
    fcv = 0.51 * math.sqrt(c.fcuk)
    chk.details.update(tau_section=tau, f_cv=fcv, b=b, h0=h0, alpha2=alpha2)
    if tau > fcv:
        chk.failed.append(f"section too small: γ0Vd/(bh0) + γ0Td/Wt = {tau:.2f} > 0.51 √fcu,k = {fcv:.2f} MPa (5.5.3-1)")
    elif tau <= 0.50 * alpha2 * c.ftd:
        chk.notes.append(f"γ0Vd/(bh0) + γ0Td/Wt = {tau:.2f} ≤ 0.50 α2 ftd = {0.5 * alpha2 * c.ftd:.2f} MPa: torsion "
                         "detailing reinforcement suffices (5.5.3-2, 9.3.13)")
    _detailing(chk, sec, sst, stirrups, Ast, b_body, h_body, beta_t, b)
    chk.warnings += sec.material_rules()
    return chk


def torsion_flange(sec: RCSection, Td: float, which: str = "top", *, stirrups, Ast: float, core,
                   gamma0: float = 1.0, stirrup_steel=None, t_web: float | None = None) -> Check:
    """a flange of a T, I or flanged box section as a pure torsion member (5.5.5 item 4, 5.5.1): its share
    T'fd = W'tf / Wt × Td (5.5.5-2, -3) against 5.5.1-1 on the flange rectangle (overhang × thickness, overhang capped
    at 6 hf), with the stirrup and longitudinal minimums of 9.3.13 for pure torsion. which: "top" or "bottom"."""
    w = torsion_moduli(sec, t_web=t_web)
    if w.get(which, 0.0) <= 0:
        raise ValueError(f"no {which} flange outside the web")
    sst = _steel(sec, stirrup_steel)
    if sec.kind == "tee":
        p = sec.pieces[0] if which == "top" else sec.pieces[2]
        hf, over = p.bottom - p.top, p.b - sec.pieces[1].b
    else:
        top, bot = sec.pieces[0], sec.pieces[2]
        hf, over = top.bottom - top.top, top.b - bot.b
    over = min(over, 6 * hf)
    T = Td * w[which] / w["total"]
    chk = Check(f"torsion ({which} flange)", gamma0 * T, 0.0, "kN·m", "JTG 3362-2018 5.5.5, 5.5.1",
                {"moduli": w, "T_share": T}, list(w["capped"]))
    Tu, _, _ = _torsion_part(chk, sec, w[which], 1.0, sec.h, stirrups, Ast, core, sst, None)
    chk.resistance = Tu / 1e6
    _detailing(chk, sec, sst, stirrups, Ast, min(hf, over), max(hf, over), None, min(hf, over))
    return chk
