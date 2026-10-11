"""Calculation reports (计算书) of JTG checks, on the document layer femotion.report.

    from femotion.report import Report
    from femotion.codes import jtg
    from femotion.codes.jtg import report as jr

    rep = Report("两跨连续 T 梁计算书", project="某大桥", member="主梁 T 梁")
    ch = rep.chapter("持久状况承载能力极限状态")
    jr.write_check(ch, jtg.flexure(sec, Md, gamma0), where="x = 25.50 m（跨中）")    # one check: one section
    jr.write_check(ch, jtg.shear(sec, Vd, (2, 12, 100), gamma0=gamma0), where="x = 0.55 m（端支点 h/2 处）")
    jr.summary(rep.chapter("验算汇总"))                    # every check written so far, the warnings, the conclusion
    rep.save("girder.html")

write_check writes a Check as a numbered section: the location (given by the caller), the clause, the calculation with
the formula first in symbols, then with the values substituted, then the result (in N, mm, MPa; moments in kN·m), the
verdict (γ0Sd ≤ R, or the value against its limit, with the ratio), what failed, the warnings (detailing, minimum
reinforcement, materials: they do not change the verdict) apart, the notes, and the section drawn with the check over
it. The substituted values are tagged (data-sym) and the formulas are the code's, so the numbers shown can be checked by
hand; each template recomputes the resistance from the formula it writes and falls back to the general form (the
intermediate values of Check.details as a table) when it does not reproduce the check exactly — the case for sections
outside the formula's premises (bars between the edge rows, over-reinforced sections, circles by the strain method …),
for which the program uses the unified algorithm of docs/codes/JTG_3362-2018.md §10.

Templates: flexure (rectangular, T, I and box sections, compression zone in the flange or the web, x < 2a′ by 5.2.4,
internal and external tendons), shear (5.2.9 with 5.2.11, 5.2.12; slabs without stirrups), axial compression (5.3.1),
eccentric compression (5.3.4, 5.3.5 large and small eccentricity; circles by 5.3.8), tension (5.4.1, 5.4.2 small
eccentricity), torsion (5.5.1, 5.5.4), crack width (6.4.3; σss of 6.4.4-2 for flexural members), crack resistance of
normal and inclined sections (6.3), stresses in service (7.1.5, 7.1.6), deflection (6.5.3), transient stresses (7.2),
local compression (5.7), punching (5.6.1, 5.6.2, 5.6.3), bearings (8.7.3, 8.7.4, 8.7.5) and expansion joints (8.8.2),
detailing reports (a table of the rules). Any other check is written in the general form.

Design data: design_basis (the codes), materials (from the sections), section_properties, load_cases (LoadCase),
live_load, combinations. Charts and drawings come from femotion.codes.jtg.charts, .figures and femotion.frame.figures.
Language: lang="zh" (default: the report's) or "en"; the check names and the common messages are translated, the rest of
the program's messages are kept as they are.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from ...report import Cell, Markup, Node, fixed, m
from ...viz import number
from .figures import _RULES, _clause, _warning
from .rc import Check, Rect

__all__ = ["write_check", "summary", "design_basis", "materials", "section_properties", "load_cases", "live_load",
           "combinations", "shear_search_table", "stirrup_design_block", "bent_bar_table", "check_name", "clause_text",
           "message", "Entry"]

# ── words ───────────────────────────────────────────────────────────────────
_L = {
    "zh": {"where": "验算位置", "clause": "依据", "calc": "计算过程", "units": "以下各式中力以 N、长度以 mm、应力以 MPa 计，"
           "弯矩换算为 kN·m、力换算为 kN。", "ok": "满足要求", "not": "不满足要求", "ratio": "利用率",
           "failed": "不满足的条件", "warnings": "构造及最小配筋提示（不影响承载力验算结论）", "notes": "说明",
           "details": "计算参数", "param": "参数", "value": "数值", "unit": "单位", "fig": "{w}截面验算图",
           "verdict": "结论", "no": "序号", "item": "验算项目", "loc": "位置", "demand": "作用效应",
           "resist": "抗力或限值", "summary_cap": "验算结果汇总", "warn_cap": "构造检查提示汇总", "content": "内容",
           "all_ok": "共 {n} 项验算，全部满足要求。", "some_fail": "共 {n} 项验算，其中 {k} 项不满足要求（表中红色行）。",
           "warn_n": "另有构造检查提示 {k} 条（见", "warn_n2": "），不影响承载力和正常使用验算的结论，应在构造设计中处理。",
           "no_warn": "构造检查无提示。", "rules": "构造规则", "actual": "实际值", "limit": "限值", "rule_ok": "满足",
           "rule_fail": "不满足", "det_ok": "各项构造满足", "det_warn": "{k} 项提示", "general": "程序按截面平衡的统一算法"
           "（平截面假定，钢筋应力按 5.1.5 并受 5.1.3 限制）求解，主要中间结果如下。", "mm": "mm",
           "and": "，", "gov_edge": "控制边缘", "top": "上缘", "bottom": "下缘", "edge": "边缘",
           "detailing_report": "构造检查", "checks": "项"},
    "en": {"where": "Location", "clause": "Clause", "calc": "Calculation", "units": "Forces in N, lengths in mm,"
                                                                                    " stresses in "
           "MPa; moments converted to kN·m, forces to kN.", "ok": "OK", "not": "NOT OK", "ratio": "ratio",
           "failed": "Conditions not met", "warnings": "Detailing and minimum-reinforcement warnings (they do not"
                                                       " change "
           "the verdict)", "notes": "Notes", "details": "Values", "param": "Quantity", "value": "Value", "unit": "Unit",
           "fig": "Section at {w}", "verdict": "Verdict", "no": "No.", "item": "Check", "loc": "Location",
           "demand": "Effect", "resist": "Resistance or limit", "summary_cap": "Summary of the checks",
           "warn_cap": "Detailing warnings", "content": "Warning", "all_ok": "{n} checks, all met.",
           "some_fail": "{n} checks, {k} not met (red rows).", "warn_n": "{k} detailing warning(s) (see ",
           "warn_n2": "); they do not change the verdicts and are to be resolved in the detailing.",
           "no_warn": "No detailing warnings.", "rules": "Rule", "actual": "Value", "limit": "Limit", "rule_ok": "met",
           "rule_fail": "not met", "det_ok": "rules met", "det_warn": "{k} warning(s)",
           "general": "The program solves the "
           "section by the unified algorithm (plane sections, steel stresses by 5.1.5 within 5.1.3); the main values:",
           "mm": "mm", "and": ", ", "gov_edge": "governing edge", "top": "top", "bottom": "bottom", "edge": "edge",
           "detailing_report": "Detailing", "checks": "checks"},
}

# check names (regular expressions on Check.name) → (zh, en); groups as \1 …
_NAMES = [
    (r"flexure", "正截面抗弯承载力", "Flexural resistance"),
    (r"flexure \(hogging\)", "正截面抗弯承载力（负弯矩）", "Flexural resistance (hogging)"),
    (r"shear", "斜截面抗剪承载力", "Shear resistance"),
    (r"shear \(hogging side\)", "斜截面抗剪承载力（负弯矩区）", "Shear resistance (hogging side)"),
    (r"shear \(slab without stirrups\)", "不配箍筋板的斜截面抗剪", "Shear of a slab without stirrups"),
    (r"shear \(circular section\)", "圆形截面抗剪承载力", "Shear resistance (circular section)"),
    (r"shear \(composite member\)", "组合构件斜截面抗剪承载力", "Shear resistance (composite member)"),
    (r"shear \(precast girder, stage (\d+)\)", r"预制梁斜截面抗剪承载力（第 \1 阶段）", r"Shear (precast girder, stage \1)"),
    (r"shear share ξ = (\S+) of the concrete and stirrups", r"箍筋设计回代（混凝土和箍筋承担 ξ = \1）",
     r"Stirrups back-substituted (share ξ = \1)"),
    (r"section limit at the end support x = (\S+)", r"端支点截面尺寸限制（x = \1 m）", r"Section limit at the end support x = \1"
                                                                            r" m"),
    (r"axial compression", "轴心受压承载力", "Axial compression"),
    (r"axial compression \(spiral column\)", "螺旋箍筋柱轴心受压承载力", "Axial compression (spiral column)"),
    (r"eccentric compression", "偏心受压承载力", "Eccentric compression"),
    (r"biaxial eccentric compression(.*)", r"双向偏心受压承载力\1", r"Biaxial eccentric compression\1"),
    (r"biaxial eccentric tension", "双向偏心受拉承载力", "Biaxial eccentric tension"),
    (r"tension", "受拉承载力", "Tension"),
    (r"tension \(hogging side\)", "受拉承载力（负弯矩侧）", "Tension (hogging side)"),
    (r"torsion", "受扭承载力", "Torsion"),
    (r"torsion with shear", "剪扭承载力", "Torsion with shear"),
    (r"torsion \(web\)", "受扭承载力（腹板）", "Torsion (web)"),
    (r"torsion with shear \(web\)", "剪扭承载力（腹板）", "Torsion with shear (web)"),
    (r"torsion \((top|bottom) flange\)", r"受扭承载力（\1 翼缘）", r"Torsion (\1 flange)"),
    (r"crack width", "裂缝宽度", "Crack width"),
    (r"crack width \(composite\)", "裂缝宽度（组合构件）", "Crack width (composite)"),
    (r"deflection \(long term, live load\)", "长期挠度（汽车荷载）", "Long-term live-load deflection"),
    (r"crack resistance \(normal section, full prestress, (top|bottom) edge\)", r"正截面抗裂（全预应力，\1）",
     r"Crack resistance (normal section, full prestress, \1 edge)"),
    (r"crack resistance \(normal section, class A, (top|bottom) edge\)", r"正截面抗裂（A 类构件，\1）",
     r"Crack resistance (normal section, class A, \1 edge)"),
    (r"crack resistance \(normal section, class B under self-weight, (top|bottom) edge\)",
     r"正截面抗裂（B 类构件自重作用，\1）", r"Crack resistance (class B under self-weight, \1 edge)"),
    (r"crack resistance \(inclined section\)", "斜截面抗裂", "Crack resistance (inclined section)"),
    (r"crack resistance \(inclined section, composite\)", "斜截面抗裂（组合构件）",
     "Crack resistance (inclined section, composite)"),
    (r"crack resistance \((.*)\)", r"正截面抗裂（\1）", r"Crack resistance (\1)"),
    (r"principal compressive stress \(service\)", "使用阶段混凝土主压应力", "Principal compressive stress in service"),
    (r"principal compressive stress \(service, composite\)", "使用阶段混凝土主压应力（组合构件）",
     "Principal compressive stress in service (composite)"),
    (r"concrete compression in service \((top|bottom) edge\)", r"使用阶段混凝土法向压应力（\1）",
     r"Concrete compression in service (\1 edge)"),
    (r"concrete compression in service \(cracked section\)", "使用阶段混凝土法向压应力（开裂截面）",
     "Concrete compression in service (cracked section)"),
    (r"concrete compression in service \((.*)\)", r"使用阶段混凝土法向压应力（\1）", r"Concrete compression in service (\1)"),
    (r"transient stresses \(reinforced concrete\)", "短暂状况混凝土和钢筋应力（钢筋混凝土）",
     "Transient stresses (reinforced concrete)"),
    (r"transient shear stresses \(reinforced concrete\)", "短暂状况剪应力（钢筋混凝土）",
     "Transient shear stresses (reinforced concrete)"),
    (r"transient stresses \(prestressed, (top|bottom) edge in compression\)", r"短暂状况混凝土应力（预应力，\1受压）",
     r"Transient stresses (prestressed, \1 edge in compression)"),
    (r"local compression", "局部承压", "Local compression"),
    (r"local compression \(no indirect reinforcement\)", "局部承压（未配间接钢筋）",
     "Local compression (no indirect reinforcement)"),
    (r"punching \(slab, 5\.6\.1\)", "冲切（板，不配抗冲切钢筋）", "Punching (slab, 5.6.1)"),
    (r"punching \(slab with punching steel, 5\.6\.2\)", "冲切（板，配抗冲切钢筋）", "Punching (slab with punching steel)"),
    (r"punching \(footing, 5\.6\.3\)", "冲切（扩大基础）", "Punching (footing)"),
    (r"pile cap punching \(column\)", "承台冲切（墩柱）", "Pile cap punching (column)"),
    (r"pile cap punching \(corner pile\)", "承台冲切（角桩）", "Pile cap punching (corner pile)"),
    (r"pile cap punching \(edge pile\)", "承台冲切（边桩）", "Pile cap punching (edge pile)"),
    (r"pile cap shear", "承台斜截面抗剪", "Pile cap shear"),
    (r"pile cap strut (\S+)", r"承台撑杆 \1", r"Pile cap strut \1"),
    (r"pile cap tie", "承台拉杆", "Pile cap tie"),
    (r"cap beam flexure", "盖梁正截面抗弯", "Cap beam flexure"),
    (r"cap beam flexure \(hogging\)", "盖梁正截面抗弯（负弯矩）", "Cap beam flexure (hogging)"),
    (r"cap beam shear", "盖梁斜截面抗剪", "Cap beam shear"),
    (r"composite joint shear \(beam\)", "叠合面抗剪（梁）", "Composite joint shear (beam)"),
    (r"composite joint shear \(slab\)", "叠合面抗剪（板）", "Composite joint shear (slab)"),
    (r"bearing area \(8\.7\.3-1\)", "支座有效承压面积", "Bearing area"),
    (r"rubber thickness, shear without braking \((\S+)\)", "橡胶层总厚度（剪切变形，不计制动力）",
     "Rubber thickness for shear (without braking)"),
    (r"rubber thickness, shear with braking \((\S+)\)", "橡胶层总厚度（剪切变形，计入制动力）",
     "Rubber thickness for shear (with braking)"),
    (r"rubber thickness, stability \((\S+)\)", "橡胶层总厚度（稳定）", "Rubber thickness for stability"),
    (r"compression of the bearing \(8\.7\.3-8, -9\)", "支座竖向平均压缩变形", "Compression of the bearing"),
    (r"steel plates \(8\.7\.3-10\)", "加劲钢板厚度", "Steel plates"),
    (r"sliding without braking \(8\.7\.4-1\)", "支座抗滑稳定（不计制动力）", "Sliding (without braking)"),
    (r"sliding with braking \(8\.7\.4-2\)", "支座抗滑稳定（计入制动力）", "Sliding (with braking)"),
    (r"PTFE friction without braking \(8\.7\.5-1\)", "四氟滑板支座摩阻（不计制动力）", "PTFE friction (without braking)"),
    (r"PTFE friction with braking \(8\.7\.5-2\)", "四氟滑板支座摩阻（计入制动力）", "PTFE friction (with braking)"),
    (r"concrete hinge, compression \(8\.6\.1\)", "混凝土铰抗压", "Concrete hinge, compression"),
    (r"concrete hinge, transverse tension \(8\.6\.2\)", "混凝土铰横向受拉", "Concrete hinge, transverse tension"),
    (r"expansion joint \(8\.8\.2\)", "伸缩装置伸缩量", "Expansion joint movement range"),
    (r"lifting loops \(9\.8\.2\)", "吊环应力", "Lifting loops"),
    (r"in-plane principal tension \(table 6\.1\.3\)", "箱梁面内主拉应力", "In-plane principal tension"),
    (r"in-plane principal compression", "箱梁面内主压应力", "In-plane principal compression"),
    (r"transverse normal stress \((.*)\)", r"横向正应力（\1）", r"Transverse normal stress (\1)"),
    (r"durability, environment (\S+) \(4\.5\)", r"耐久性（\1类环境）", r"Durability, environment \1"),
    (r"section detailing (.*)", "截面构造", "Section detailing"),
    (r"stirrups \(9\.3\.12, 9\.1\.6\)", "箍筋构造", "Stirrup detailing"),
    (r"bars over an end support \(9\.3\.9\)", "端支点处纵筋构造", "Bars over an end support"),
    (r"lap splice of (\S+) mm (\S+) \((.*)\)", r"\2 \1 mm 钢筋绑扎搭接", r"Lap splice of \1 mm \2"),
    (r"slab detailing \(9\.2\)", "板的构造", "Slab detailing"),
    (r"girder top slab \((.*)\)", "梁顶板（翼缘板）构造", "Girder top slab"),
]
_EDGE_ZH = {"top": "上缘", "bottom": "下缘", "precast top": "预制梁顶缘", "precast bottom": "预制梁底缘",
            "topping top": "现浇层顶缘"}

# messages (warnings, notes, failed) → zh, beyond femotion.codes.jtg.figures._warning
_MSG_ZH = [
    (r"x = (\S+) < 2a' = (\S+) mm: compression bars ignored \(singly reinforced\) governs \((\S+)\)",
     lambda g: f"x = {g[1]} mm < 2a′ = {g[2]} mm：不计受压钢筋（按单筋截面）的承载力较大，取用（{g[3]}）"),
    (r"x = (\S+) < 2a' = (\S+) mm: moments about the compression steel governs \((\S+)\)",
     lambda g: f"x = {g[1]} mm < 2a′ = {g[2]} mm：按对受压钢筋合力点取矩计算（{g[3]}）"),
    (r"γ0Vd ≤ (\S+) kN: detailing stirrups per 9\.3\.12 suffice \(5\.2\.12\)",
     lambda g: f"γ0Vd ≤ {g[1]} kN：可只按 9.3.12 构造配置箍筋（5.2.12）"),
    (r"section too small: γ0Vd > 0\.51e-3 √fcu,k b h0 = (\S+) kN \(5\.2\.11\)",
     lambda g: f"截面尺寸不足：γ0Vd > 0.51×10⁻³√fcu,k b h0 = {g[1]} kN（5.2.11）"),
    (r"stirrup ratio (\S+) < minimum (\S+) \(9\.3\.12\)", lambda g: f"配箍率 {g[1]} < 最小配箍率 {g[2]}（9.3.12）"),
    (r"stirrup diameter (\S+) mm < 8 mm \(9\.3\.12\)", lambda g: f"箍筋直径 {g[1]} mm < 8 mm（9.3.12）"),
    (r"stirrup spacing (\S+) mm > (\S+) mm \(h/2 and 400 mm, 9\.3\.12\)",
     lambda g: f"箍筋间距 {g[1]} mm > {g[2]} mm（不大于 h/2 且不大于 400 mm，9.3.12）"),
    (r"spacing (\S+) mm: within one beam depth \((\S+) mm\) of the support centre at most 100 mm \(9\.3\.12 item 3\)",
     lambda g: f"箍筋间距 {g[1]} mm：支座中心向跨径方向一倍梁高（{g[2]} mm）范围内不宜大于 100 mm（9.3.12 第 3 款）"),
    (r"stirrup diameter (\S+) mm < 1/4 of the main bars \((\S+) mm\) \(9\.3\.12 item 1\)",
     lambda g: f"箍筋直径 {g[1]} mm < 主筋直径（{g[2]} mm）的 1/4（9.3.12 第 1 款）"),
    (r"over-reinforced: x = (\S+) > ξb h0 = (\S+) mm", lambda g: f"超筋：x = {g[1]} > ξb h0 = {g[2]} mm（5.2.2-3）"),
    (r"longitudinal reinforcement (\S+) < minimum (\S+) \(9\.1\.12\)",
     lambda g: f"全部纵向钢筋配筋率 {g[1]} < 最小配筋率 {g[2]}（9.1.12）"),
    (r"(tension side|compression side) reinforcement (\S+) < minimum 0\.2 % \(9\.1\.12\)",
     lambda g: f"{'受拉' if g[1].startswith('tension') else '受压'}一侧钢筋配筋率 {g[2]} < 0.2%（9.1.12）"),
    (r"e0 = (\S+) mm raised to the minimum (\S+) mm \(5\.3\.9\)",
     lambda g: f"e0 = {g[1]} mm 小于最小值，取 {g[2]} mm（5.3.9）"),
    (r"side faces not checked \(side_cover=…\); the horizontal room takes the bottom cover of the outermost steel",
     lambda g: "未给出侧面保护层（side_cover），侧面不检查；水平方向按底面最外侧钢筋保护层布置"),
    (r"9\.3\.3 horizontal clear spacing: bars spread evenly between the stirrups across the width at their depth, the "
     r"largest the layout allows \(program convention\)",
     lambda g: "9.3.3 水平净距：同排钢筋在箍筋内沿该深度的宽度均匀布置，取布置所能达到的最大净距（程序约定）"),
    (r"9\.3\.7 tension zone (below|above) depth (\S+), the cracked section's axis \(reinforced concrete\); a gap "
     r"reaching into it takes the tension limit",
     lambda g: f"9.3.7 受拉区取开裂截面中和轴（距顶面 {g[2]} mm）以{'下' if g[1] == 'below' else '上'}；跨入受拉区的间距按受拉区限值检查"),
    (r"4\.5\.4: cover per 9\.1\.1 \(section_detailing\); impermeability and frost-resistance grades where required; "
     r"forms and details that drain and ventilate \(not checked by numbers\)",
     lambda g: "4.5.4：保护层按 9.1.1 检查（截面构造）；需要时的抗渗、抗冻等级以及利于排水、通风的构造不作数值检查"),
    (r"5\.2\.13-1 with ξ = (\S+)", lambda g: f"按式（5.2.13-1），ξ = {g[1]}"),
    (r"the shear at the support centre \(V0d\), b and h0 of the support section \(program convention, the "
     r"near-end-support segment of 5\.2\.11\)", lambda g: "取支点中心处剪力 V0d 和支点截面的 b、h0（程序约定：5.2.11 的端支点附近梁段）"),
    (r"α1 = (\S+) \(near an end support, 5\.2\.9\)", lambda g: f"α1 = {g[1]}（端支点附近，5.2.9）"),
    (r"α1 = (\S+) \(near an interior support, 5\.2\.9\)", lambda g: f"α1 = {g[1]}（中间支点附近，5.2.9）"),
]
_DETAIL_RULES_ZH = [
    (r"stirrup diameter", "箍筋直径"), (r"stirrup diameter \(≥ main bar / 4\)", "箍筋直径（不小于主筋直径的 1/4）"),
    (r"stirrup ratio ρsv", "配箍率 ρsv"), (r"stirrup spacing", "箍筋间距"),
    (r"stirrup spacing within h of the support", "支座中心一倍梁高范围内箍筋间距"),
    (r"stirrup hook bend diameter", "箍筋弯钩弯弧内直径"), (r"stirrup hook straight part", "箍筋弯钩平直段长度"),
]


def _lang(node: Node | None, lang: str | None) -> str:
    if lang is not None:
        return lang
    return node.report.lang if node is not None else "zh"


def check_name(chk: Check, lang: str = "zh") -> str:
    """the check's name in the language (the program's English name where no translation is known)"""
    hit = None
    for pat, zh, en in _NAMES:
        mt = re.fullmatch(pat, chk.name)
        if mt:
            hit = mt.expand(zh if lang == "zh" else en)
    if hit is None:
        return chk.name
    if lang == "zh":
        for k, v in _EDGE_ZH.items():
            hit = hit.replace(f"（{k}）", f"（{v}）").replace(f"，{k}）", f"，{v}）").replace(f"{k}受压", f"{v}受压")
            hit = hit.replace(f"（{k} 翼缘）", f"（{v.rstrip('缘')}翼缘）")
    return hit


def clause_text(clause: str, lang: str = "zh") -> str:
    """'JTG 3362-2018 5.2.2 / 5.2.3' → 'JTG 3362—2018 第 5.2.2、5.2.3 条' (zh)"""
    s = str(clause)
    if lang != "zh":
        return s
    mt = re.match(r"(JTG \S+?)-(\d{4})\s*(.*)", s)
    if not mt:
        return _clause(s, lang)
    code, year, rest = mt.groups()
    rest = rest.replace(" / ", "、").replace("–", "～")
    rest = _clause(rest, lang)
    return f"{code}—{year} {rest}".strip()


def _rule_zh(name: str) -> str:
    for pat, zh, _ in _RULES:
        mt = re.fullmatch(pat, name)
        if mt:
            face = {"bottom": "底", "top": "顶"}.get(mt.group(1), "") if mt.groups() else ""
            zone = {"tension": "受拉", "compression": "受压"}.get(mt.group(1), "") if mt.groups() else ""
            return zh.format(face=face, zone=zone)
    for pat, zh in _DETAIL_RULES_ZH:
        if re.fullmatch(pat, name):
            return zh
    return name


def message(s: str, lang: str = "zh") -> str:
    """a warning, note or failed condition of a check in the language (zh: the common ones translated)"""
    s = str(s)
    if lang != "zh":
        return s
    t = _warning(s, lang)
    if t != s:
        return t
    for pat, f in _MSG_ZH:
        mt = re.fullmatch(pat, s)
        if not mt:
            continue
        if f is not None:
            return f(mt)
    # detailing rules: "<rule> <value> <|> <limit> (<clause>)"
    mt = re.fullmatch(r"(.+?) (−?-?[\d.]+(?: mm|%)?) ([<>]) (\S+(?: mm)?) \(([^()]*)\)", s)
    if mt:
        rule = _rule_zh(mt.group(1))
        if rule != mt.group(1):
            sep = "" if rule[-1] in "）》】" else " "
            return f"{rule}{sep}{mt.group(2)} {mt.group(3)} {mt.group(4)}（{_clause(mt.group(5), 'zh')}）"
    return s


# ── writing one check ───────────────────────────────────────────────────────
def _nv(v: float, sig: int = 4) -> str:
    """a value substituted into a formula: whole numbers from 1000, else `sig` significant figures (real minus sign)"""
    v = float(v)
    if not math.isfinite(v):
        return "∞"
    a = abs(v)
    if a >= 1000 or a == 0:
        return fixed(v, 0)
    d = max(sig - 1 - int(math.floor(math.log10(a))), 0)
    s = fixed(v, d)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


class _W:
    """what a template writes into: the node of the check, the language, helpers for formulas"""

    def __init__(self, node: Node, chk: Check, lang: str):
        self.node, self.chk, self.lang = node, chk, lang
        self.sec = chk.section
        self.s = None
        if self.sec is not None and hasattr(self.sec, "flipped"):
            self.s = self.sec.flipped() if chk.hogging else self.sec
        self.units_said = False
        self.hide_notes: list = []  # notes the template has written in its own words
        self.plain_figure = False  # draw the section without the check (its x differs from the one written)

    def t(self, zh: str, en: str) -> str:
        return zh if self.lang == "zh" else en

    def units(self):
        """the units of the formulas, once per chapter"""
        if self.units_said:
            return
        self.units_said = True
        ch = self.node
        while ch.parent is not None:
            ch = ch.parent
        said = self.node.report.store.setdefault("jtg.units_said", set())
        if id(ch) in said:
            return
        said.add(id(ch))
        self.node.paragraph(_L[self.lang]["units"], cls="muted")

    def p(self, *parts):
        self.node.paragraph(*parts)

    def eq(self, rows, tag=None, where=None):
        self.node.formula(rows, tag=tag, where=where)

    @staticmethod
    def v(key: str, value: float, nd: int | None = None) -> str:
        """a substituted value: \\v{key}{text}"""
        txt = _nv(value) if nd is None else fixed(value, nd)
        return f"\\v{{{key}}}{{{txt}}}"

    def head(self, text: str):
        self.node.paragraph(text, cls="lead")


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


# ── section data for the templates ──────────────────────────────────────────
@dataclass
class _Steel:
    As: float
    ds: float
    Asc: float
    asc: float
    Ap: float
    dp: float
    Apc: float
    apc: float
    sp0c: float
    Fex: float
    Aex: float
    sex: float
    dex: float
    edge_only: bool


def _steel(s) -> _Steel:
    bot, top = s.edge_masks()
    pb, pt = s.tendon_masks()

    def mean(a, d):
        return float(np.average(d, weights=a)) if a.sum() > 0 else 0.0
    bars, td, ex = s.bars, s.tendons, s.external
    As, Asc = bars[bot, 0].sum(), bars[top, 0].sum()
    Ap, Apc = (td[pb, 0].sum(), td[pt, 0].sum()) if len(td) else (0.0, 0.0)
    ex_ok = bool((ex[:, 1] > s.zg).all()) if len(ex) else True
    Fex = float((ex[:, 2] * ex[:, 0]).sum()) if len(ex) else 0.0
    return _Steel(float(As), mean(bars[bot, 0], bars[bot, 1]), float(Asc), mean(bars[top, 0], bars[top, 1]),
                  float(Ap), mean(td[pb, 0], td[pb, 1]) if Ap else 0.0, float(Apc),
                  mean(td[pt, 0], td[pt, 1]) if Apc else 0.0, mean(td[pt, 0], td[pt, 2]) if Apc else 0.0,
                  Fex, float(ex[:, 0].sum()) if len(ex) else 0.0, Fex / ex[:, 0].sum() if len(ex) else 0.0,
                  mean(ex[:, 0] * ex[:, 2], ex[:, 1]) if len(ex) else 0.0,
                  bool((bot | top).all()) and (bool((pb | pt).all()) if len(td) else True) and ex_ok)


def _flange(s):
    """(b'f, h'f, b) of the compressed top of a section of stacked solid rectangles: the top rectangle and the one under
    it (a rectangle: b'f = b, h'f = h); None for other shapes or when the top is not wider than what is under it"""
    if not all(isinstance(p, Rect) and p.sign > 0 for p in s.pieces):
        return None
    top = [p for p in s.pieces if abs(p.top) < 1e-9]
    if len(top) != 1:
        return None
    p0 = top[0]
    if p0.bottom >= s.h - 1e-9:
        return p0.b, s.h, p0.b
    nxt = [p for p in s.pieces if abs(p.top - p0.bottom) < 1e-9]
    if len(nxt) != 1:
        return None
    if p0.b < nxt[0].b - 1e-9:
        return p0.b, p0.bottom, None  # narrower on top (a turned T): a rectangle while x ≤ h'f
    return p0.b, p0.bottom, nxt[0].b


def _sum(terms) -> str:
    return " + ".join(terms) if terms else "0"


# ── templates ───────────────────────────────────────────────────────────────
def _tpl_flexure(w: _W) -> bool:
    chk, s = w.chk, w.s
    d = chk.details
    if s is None or s.kind not in ("rect", "tee", "box") or chk.failed:
        return False
    if "clause_5_2_7" in d:
        return False
    g = _steel(s)
    fl = _flange(s)
    if not g.edge_only or fl is None:
        return False
    bf, hf, b = fl
    c, st, t = s.concrete, s.steel, s.tendon
    fcd, fsd, fsdc = c.fcd, st.fsd, st.fsd_c
    fpd = t.fpd if (t is not None and g.Ap) else 0.0
    fpdc = t.fpd_c if (t is not None and g.Apc) else 0.0
    h0 = d["h0"]
    T = fsd * g.As + fpd * g.Ap + g.Fex
    Cs = fsdc * g.Asc + (fpdc - g.sp0c) * g.Apc
    flange = b is not None and bf > b + 1e-9
    if b is None:
        b = bf
    in_fl = T - Cs <= fcd * bf * hf + 1e-9 or not flange
    x = (T - Cs) / (fcd * bf) if in_fl else (T - Cs - fcd * (bf - b) * hf) / (fcd * b)
    aref, comp_p = s.comp_ref()
    has_c = (g.Asc > 0 or g.Apc > 0) and aref is not None
    if (x <= 0 and not has_c) or (not flange and x > hf + 1e-9):
        return False
    no_x = x <= 0  # the compression steel alone balances the tension steel: x < 2a' whatever the concrete
    case_b = has_c and x < 2 * aref
    if case_b != bool(chk.notes and any("2a'" in n for n in chk.notes)):
        return False

    def conc_M(xx, inside):
        if inside:
            return fcd * bf * xx * (h0 - xx / 2)
        return fcd * (b * xx * (h0 - xx / 2) + (bf - b) * hf * (h0 - hf / 2))
    if not case_b:
        Mu = conc_M(x, in_fl) + fsdc * g.Asc * (h0 - g.asc) + (fpdc - g.sp0c) * g.Apc * (h0 - g.apc)
    else:
        Mab = s.design_tension_moment(aref, comp_p)
        Msingle = 0.0
        x1 = None
        if not g.Apc:
            in1 = T <= fcd * bf * hf + 1e-9 or not flange
            x1 = T / (fcd * bf) if in1 else (T - fcd * (bf - b) * hf) / (fcd * b)
            if x1 <= d["xi_b"] * h0:
                Msingle = conc_M(x1, in1)
        Mu = max(Mab, Msingle)
    if not _close(Mu / 1e6, chk.resistance):
        return False
    if case_b:
        w.hide_notes = [n_ for n_ in chk.notes if "2a'" in n_]
        w.plain_figure = True  # the drawing's zone is the strain solution's x, not the x written here
    v, tt = w.v, w.t
    w.units()
    # the steel on each side, symbols and values
    ten_s = [("f_{sd}A_{s}", f"{v('fsd', fsd)} × {v('As', g.As)}")]
    if g.Ap:
        ten_s.append(("f_{pd}A_{p}", f"{v('fpd', fpd)} × {v('Ap', g.Ap)}"))
    if g.Fex:
        ten_s.append(("σ_{pe,ex}A_{ex}", f"{v('sex', g.sex)} × {v('Aex', g.Aex)}"))
    com_s = []
    if g.Asc:
        com_s.append(("f'_{sd}A'_{s}", f"{v('fsdc', fsdc)} × {v('Asc', g.Asc)}"))
    if g.Apc:
        com_s.append(("(f'_{pd} - σ'_{p0})A'_{p}", f"({v('fpdc', fpdc)} - {v('sp0c', g.sp0c)}) × {v('Apc', g.Apc)}"))
    sym_T, val_T = " + ".join(a for a, _ in ten_s), " + ".join(b_ for _, b_ in ten_s)
    sym_C, val_C = " + ".join(a for a, _ in com_s), " + ".join(b_ for _, b_ in com_s)
    k = 1
    if flange:
        w.head(tt(f"（{k}）判别 T 形截面类型（5.2.3）", f"({k}) Flange or web (5.2.3)"))
        k += 1
        rows = [(sym_T, val_T), ("", f"{fixed(T / 1e3, 1)}\\,\\u{{kN}}")]
        rhs_sym = "f_{cd}b'_{f}h'_{f}" + (f" + {sym_C}" if com_s else "")
        rhs_val = f"{v('fcd', fcd)} × {v('bf', bf)} × {v('hf', hf)}" + (f" + {val_C}" if com_s else "")
        rows += [(rhs_sym, rhs_val), ("", f"{fixed((fcd * bf * hf + Cs) / 1e3, 1)}\\,\\u{{kN}}")]
        w.eq(rows)
        if in_fl:
            w.p(tt(f"前者不大于后者，受压区在翼缘内，按宽度 $b'_{{f}}$ = {_nv(bf)} mm 的矩形截面计算（式 5.2.3-1）。",
                   f"The first does not exceed the second: the compression zone lies in the flange, a rectangle "
                   f"$b'_{{f}}$ = {_nv(bf)} mm wide (5.2.3-1)."))
        else:
            w.p(tt("前者大于后者，受压区进入腹板，按式（5.2.3-2）、（5.2.3-3）计算。",
                   "The first exceeds the second: the compression zone reaches into the web (5.2.3-2, -3)."))
    w.head(tt(f"（{k}）受压区高度", f"({k}) Depth of the compression zone"))
    k += 1
    if no_x:
        w.eq([(sym_T, f"{val_T} = {fixed(T / 1e3, 1)}\\,\\u{{kN}}"),
              ("", "<=", f"{sym_C} = {val_C} = {fixed(Cs / 1e3, 1)}\\,\\u{{kN}}")], tag="5.2.2-2")
        w.p(tt("受拉钢筋的拉力不大于受压钢筋按抗压强度设计值的压力，受压钢筋达不到抗压强度设计值，x < 2a′，按 5.2.4 条对受压钢筋合力点取矩计算。",
               "The tension steel's force does not exceed the compression steel's at its design strength: the "
               "compression steel cannot reach it, x < 2a′; moments about the compression steel (5.2.4)."))
    elif in_fl:
        num_sym = sym_T + (f" - ({sym_C})" if len(com_s) > 1 else (f" - {sym_C}" if com_s else ""))
        num_val = val_T + (f" - ({val_C})" if len(com_s) > 1 else (f" - {val_C}" if com_s else ""))
        bsym = "f_{cd}b'_{f}" if flange else "f_{cd}b"
        bval = f"{v('fcd', fcd)} × {v('bf' if flange else 'b', bf)}"
        tag = ("5.2.5-4" if flange else "5.2.5-2") if g.Fex else "5.2.2-2"
        w.eq([("x", f"\\frac{{{num_sym}}}{{{bsym}}}"), ("", f"\\frac{{{num_val}}}{{{bval}}}"),
              ("", f"{v('x', x, 1)}\\,\\u{{mm}}")], tag=tag)
    else:
        num_sym = sym_T + (f" - {sym_C}" if com_s else "") + " - f_{cd}(b'_{f} - b)h'_{f}"
        num_val = val_T + (f" - {val_C}" if com_s else "") + \
            f" - {v('fcd', fcd)} × ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)}"
        w.eq([("x", f"\\frac{{{num_sym}}}{{f_{{cd}}b}}"), ("", f"\\frac{{{num_val}}}{{{v('fcd', fcd)} × {v('b', b)}}}"),
              ("", f"{v('x', x, 1)}\\,\\u{{mm}}")], tag="5.2.5-6" if g.Fex else "5.2.3-3")
    xb = d["xi_b"]
    ok_x = x <= xb * h0 + 1e-9
    if not no_x:
        w.eq([("ξ", f"\\frac{{x}}{{h_{{0}}}} = \\frac{{{v('x', x, 1)}}}{{{v('h0', h0)}}} = {fixed(x / h0, 3)}",),
              ("", "<=" if ok_x else ">", f"ξ_{{b}} = {v('xib', xb)}")], tag="5.2.2-3")
    if no_x:
        pass
    elif not has_c:
        w.p(tt("受压区未计入纵向钢筋，不需验算 x ≥ 2a′。", "No compression steel counted: x ≥ 2a′ does not apply."))
    elif not case_b:
        w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"), ("", ">=", f"2a'_{{s}} = 2 × {v('aref', aref)} = "
                                                         f"{_nv(2 * aref)}\\,\\u{{mm}}")],
             tag="5.2.2-4" if comp_p else "5.2.2-5")
    else:
        w.eq([("x", f"{v('x', x, 1)}\\,\\u{{mm}}"), ("", "<", f"2a' = 2 × {v('aref', aref)} = "
                                                        f"{_nv(2 * aref)}\\,\\u{{mm}}")])
        w.p(tt("不满足 x ≥ 2a′，按 5.2.4 条对受压钢筋合力点取矩计算。", "x < 2a′: moments about the compression steel (5.2.4)."))
    w.head(tt(f"（{k}）正截面抗弯承载力", f"({k}) Flexural resistance"))
    if not case_b:
        if in_fl:
            csym = "f_{cd}" + ("b'_{f}" if flange else "b") + "x(h_{0} - \\frac{x}{2})"
            cval = f"{v('fcd', fcd)} × {v('bf' if flange else 'b', bf)} × {v('x', x, 1)} × ({v('h0', h0)} - " \
                   f"\\frac{{{v('x', x, 1)}}}{{2}})"
        else:
            csym = "f_{cd}[bx(h_{0} - \\frac{x}{2}) + (b'_{f} - b)h'_{f}(h_{0} - \\frac{h'_{f}}{2})]"
            cval = (f"{v('fcd', fcd)} × [{v('b', b)} × {v('x', x, 1)} × ({v('h0', h0)} - \\frac{{{v('x', x, 1)}}}{{2}})"
                    f" + ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)} × ({v('h0', h0)} -"
                    f" \\frac{{{v('hf', hf)}}}{{2}})]")
        msym, mval = [csym], [cval]
        if g.Asc:
            msym.append("f'_{sd}A'_{s}(h_{0} - a'_{s})")
            mval.append(f"{v('fsdc', fsdc)} × {v('Asc', g.Asc)} × ({v('h0', h0)} - {v('asc', g.asc)})")
        if g.Apc:
            msym.append("(f'_{pd} - σ'_{p0})A'_{p}(h_{0} - a'_{p})")
            mval.append(f"({v('fpdc', fpdc)} - {v('sp0c', g.sp0c)}) × {v('Apc', g.Apc)} × ({v('h0', h0)} - "
                        f"{v('apc', g.apc)})")
        tag = ("5.2.5-3" if (flange and in_fl) else "5.2.5-5" if flange else "5.2.5-1") if g.Fex else \
            ("5.2.3-2" if (flange and not in_fl) else "5.2.2-1")
        w.eq([("M_{u}", " + ".join(msym)), ("", " + ".join(mval)),
              ("", f"{v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")], tag=tag)
    else:
        h = s.h
        terms_s, terms_v = [], []
        if g.Ap:
            terms_s.append("f_{pd}A_{p}(h - a_{p} - a')")
            terms_v.append(f"{v('fpd', fpd)} × {v('Ap', g.Ap)} × ({v('h', h)} - {v('ap', h - g.dp)} - "
                           f"{v('aref', aref)})")
        terms_s.append("f_{sd}A_{s}(h - a_{s} - a')")
        terms_v.append(f"{v('fsd', fsd)} × {v('As', g.As)} × ({v('h', h)} - {v('as', h - g.ds)} - {v('aref', aref)})")
        if g.Fex:
            terms_s.append("σ_{pe,ex}A_{ex}(h - a_{p,ex} - a')")
            terms_v.append(f"{v('sex', g.sex)} × {v('Aex', g.Aex)} × ({v('h', h)} - {v('aex', h - g.dex)} - "
                           f"{v('aref', aref)})")
        if g.Apc and not comp_p:
            terms_s.append("- (f'_{pd} - σ'_{p0})A'_{p}(a'_{p} - a'_{s})")
            terms_v.append(f"- ({v('fpdc', fpdc)} - {v('sp0c', g.sp0c)}) × {v('Apc', g.Apc)} × ({v('apc', g.apc)} - "
                           f"{v('asc', g.asc)})")
        tag = ("5.2.6-1" if comp_p else "5.2.6-2") if g.Fex else ("5.2.4-1" if comp_p else "5.2.4-2")
        w.eq([("M_{u1}", " + ".join(terms_s).replace("+ -", "-")), ("", " + ".join(terms_v).replace("+ -", "-")),
              ("", f"{v('Mu1', Mab / 1e6, 1)}\\,\\u{{kN·m}}")], tag=tag)
        if x1 is not None:
            in1 = T <= fcd * bf * hf + 1e-9 or not flange
            w.p(tt("受压区没有预应力钢筋，另按不计受压普通钢筋的单筋截面计算（程序约定，二者取大值）：",
                   "Without compression tendons the section is also taken as singly reinforced (program convention: "
                   "the larger governs):"))
            bs = "b'_{f}" if flange else "b"
            if in1:
                w.eq([("x_{1}", f"\\frac{{{sym_T}}}{{f_{{cd}}{bs}}} = \\frac{{{val_T}}}{{{v('fcd', fcd)} × "
                                f"{v('bf' if flange else 'b', bf)}}} = {v('x1', x1, 1)}\\,\\u{{mm}}"),
                      ("", "<=" if x1 <= xb * h0 else ">", f"ξ_{{b}}h_{{0}} = {fixed(xb * h0, 1)}\\,\\u{{mm}}")])
                if Msingle:
                    w.eq([("M_{u2}", f"f_{{cd}}{bs}x_{{1}}(h_{{0}} - \\frac{{x_{{1}}}}{{2}})"),
                          ("", f"{v('fcd', fcd)} × {v('bf' if flange else 'b', bf)} × {v('x1', x1, 1)} × "
                               f"({v('h0', h0)} - \\frac{{{v('x1', x1, 1)}}}{{2}})"),
                          ("", f"{v('Mu2', Msingle / 1e6, 1)}\\,\\u{{kN·m}}")])
            else:
                w.eq([("x_{1}", f"{v('x1', x1, 1)}\\,\\u{{mm}}"),
                      ("M_{u2}", f"{v('Mu2', Msingle / 1e6, 1)}\\,\\u{{kN·m}}")])
        w.eq([("M_{u}", f"\\op{{max}}(M_{{u1}}, M_{{u2}}) = {v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")] if x1 is not None
             else [("M_{u}", f"M_{{u1}} = {v('Mu', Mu / 1e6, 1)}\\,\\u{{kN·m}}")])
    if "rho" in d:
        r_ok = d["rho"] >= d["rho_min"] - 1e-12
        w.eq([("ρ", f"\\frac{{A_{{s}}}}{{bh_{{0}}}} = \\frac{{{v('As', g.As)}}}{{{v('bweb', d['b'])} × {v('h0', h0)}}}"
                    f" = "
                    f"{fixed(100 * d['rho'], 3)}\\%"),
              ("", ">=" if r_ok else "<", f"ρ_{{min}} = \\op{{max}}(45f_{{td}}/f_{{sd}}, 0.20)\\% = "
                                          f"{fixed(100 * d['rho_min'], 3)}\\%")], tag="9.1.12")
    if "Mcr" in d:
        w.eq([("M_{ud}/M_{cr}", f"{fixed(chk.resistance, 1)}/{fixed(d['Mcr'], 1)} ="
                                f" {fixed(chk.resistance / d['Mcr'], 3)}"),
              ("", ">=" if chk.resistance >= d["Mcr"] else "<", "1.0")], tag="9.1.13")
    return True


def _tpl_shear(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None or "Vcs" not in d or "Asv" not in d:
        return False
    c = s.concrete
    b, h0, P = d["b"], d["h0"], d["P"]
    P_raw = 100 * (d["As"] + d["Ap"]) / (b * h0)
    if not _close(min(P_raw, 2.5), P):
        return False
    rho_pv = d["rho_pv"]
    Vcs = 0.45e-3 * d["alpha1"] * d["alpha2"] * d["alpha3"] * b * h0 * math.sqrt(
        (2 + 0.6 * P) * math.sqrt(c.fcuk) * (d["rho_sv"] * d["fsv"] + 0.6 * rho_pv * d["fpv"]))
    if not _close(Vcs, d["Vcs"]):
        return False
    v, tt = w.v, w.t
    w.units()
    w.head(tt("（1）计算参数", "(1) Parameters"))
    Ap_t = f" + {v('Ap', d['Ap'])}" if d["Ap"] else ""
    Ap_s = " + A_{p}" if d["Ap"] else ""
    rowsP = [("P", f"100\\frac{{A_{{s}}{Ap_s}}}{{bh_{{0}}}} = 100 × \\frac{{{v('As', d['As'])}{Ap_t}}}"
                   f"{{{v('b', b)} × {v('h0', h0)}}} = {fixed(P_raw, 3)}")]
    if P_raw > 2.5:
        rowsP.append(("", ">", "2.5，\\t{取} P = \\v{P}{2.5}" if w.lang == "zh" else "2.5, P = \\v{P}{2.5}"))
    else:
        rowsP[0] = ("P", rowsP[0][1].replace(f"= {fixed(P_raw, 3)}", f"= {v('P', P)}"))
    w.eq(rowsP)
    if _close(d["rho_sv"], d["rho_sv_provided"]):
        w.eq([("ρ_{sv}", f"\\frac{{A_{{sv}}}}{{s_{{v}}b}} = \\frac{{{d['legs']} × π × {_nv(d['dia'])}^{{2}}/4}}"
                         f"{{{v('sv', d['sv'])} × {v('b', b)}}} = \\frac{{{v('Asv', d['Asv'])}}}{{{v('sv', d['sv'])} × "
                         f"{v('b', b)}}} = {v('rho_sv', d['rho_sv'])}")])
    else:
        w.p(tt("斜截面跨越两个箍筋区段，配箍率取其范围内的平均值：", "The inclined section crosses two stirrup zones; "
               "the mean stirrup ratio over it:"), m(f"ρ_{{sv}} = {v('rho_sv', d['rho_sv'])}"))
    a3_txt = tt("受压翼缘", "compression flange") if d["alpha3"] > 1.0 else tt("矩形或受拉翼缘", "no compression flange")
    w.p(tt("系数：", "Factors: "), m(f"α_{{1}} = {v('a1', d['alpha1'])}"), "，" if w.lang == "zh" else ", ",
        m(f"α_{{2}} = {v('a2', d['alpha2'])}"), "，" if w.lang == "zh" else ", ",
        m(f"α_{{3}} = {v('a3', d['alpha3'])}"), f"（{a3_txt}）" if w.lang == "zh" else f" ({a3_txt})",
        "；" if w.lang == "zh" else "; ", m(f"f_{{cu,k}} = {v('fcuk', c.fcuk)}\\,\\u{{MPa}}"),
        "，" if w.lang == "zh" else ", ", m(f"f_{{sv}} = {v('fsv', d['fsv'])}\\,\\u{{MPa}}"),
        "。" if w.lang == "zh" else ".")
    w.head(tt("（2）斜截面抗剪承载力（5.2.9）", "(2) Shear resistance (5.2.9)"))
    pv_s = " + 0.6ρ_{pv}f_{pv}" if rho_pv else ""
    pv_v = f" + 0.6 × {v('rho_pv', rho_pv)} × {v('fpv', d['fpv'])}" if rho_pv else ""
    K = (2 + 0.6 * P) * math.sqrt(c.fcuk) * (d["rho_sv"] * d["fsv"] + 0.6 * rho_pv * d["fpv"])
    w.eq([("V_{cs}", f"0.45×10^{{-3}}α_{{1}}α_{{2}}α_{{3}}bh_{{0}}\\sqrt{{(2 + 0.6P)\\sqrt{{f_{{cu,k}}}}"
                     f"(ρ_{{sv}}f_{{sv}}{pv_s})}}")], tag="5.2.9-2")
    w.eq([("(2 + 0.6P)\\sqrt{f_{cu,k}}(ρ_{sv}f_{sv}" + pv_s + ")",
           f"(2 + 0.6 × {v('P', P)}) × \\sqrt{{{v('fcuk', c.fcuk)}}} × ({v('rho_sv', d['rho_sv'])} × "
           f"{v('fsv', d['fsv'])}{pv_v}) = {v('K', K)}"),
          ("V_{cs}", f"0.45×10^{{-3}} × {v('a1', d['alpha1'])} × {v('a2', d['alpha2'])} × {v('a3', d['alpha3'])} × "
                     f"{v('b', b)} × {v('h0', h0)} × \\sqrt{{{v('K', K)}}} = {v('Vcs', Vcs, 1)}\\,\\u{{kN}}")])
    parts = [("V_{cs}", Vcs)]
    st = s.steel
    if d["Vsb"]:
        rows = " + ".join(f"{_nv(A)} × \\op{{sin}}{_nv(th)}°" for A, th in d["bent"])
        w.eq([("V_{sb}", "0.75×10^{-3}f_{sd}ΣA_{sb}\\op{sin}θ_{s}"),
              ("", f"0.75×10^{{-3}} × {v('fsd', st.fsd)} × ({rows}) = {v('Vsb', d['Vsb'], 1)}\\,\\u{{kN}}")],
             tag="5.2.9-3")
        parts.append(("V_{sb}", d["Vsb"]))
    if d["Vpb"]:
        rows = " + ".join(f"{_nv(A)} × \\op{{sin}}{_nv(th)}°" for A, th in d["bent_tendons"])
        w.eq([("V_{pb}", "0.75×10^{-3}f_{pd}ΣA_{pb}\\op{sin}θ_{p}"),
              ("", f"0.75×10^{{-3}} × {v('fpd', s.tendon.fpd)} × ({rows}) = {v('Vpb', d['Vpb'], 1)}\\,\\u{{kN}}")],
             tag="5.2.9-4")
        parts.append(("V_{pb}", d["Vpb"]))
    if d["Vpb_ex"]:
        w.eq([("V_{pb,ex}", f"0.75×10^{{-3}}Σσ_{{pe,ex}}A_{{ex}}\\op{{sin}}θ_{{ex}} = {v('Vpbex', d['Vpb_ex'], 1)}"
                            "\\,\\u{kN}")], tag="5.2.9-5")
        parts.append(("V_{pb,ex}", d["Vpb_ex"]))
    if len(parts) > 1:
        w.eq([("V_{u}", " + ".join(p for p, _ in parts) + " = " + " + ".join(fixed(val, 1) for _, val in parts)
               + f" = {v('Vu', chk.resistance, 1)}\\,\\u{{kN}}")], tag="5.2.9-1")
    else:
        w.eq([("V_{u}", f"V_{{cs}} = {v('Vu', chk.resistance, 1)}\\,\\u{{kN}}")], tag="5.2.9-1")
    w.head(tt("（3）截面尺寸和构造配箍条件", "(3) Section limit and the detailing threshold"))
    dem = abs(chk.demand)
    vmax, vmin = d["V_section_max"], d["V_detailing"]
    w.eq([("0.51×10^{-3}\\sqrt{f_{cu,k}}bh_{0}", f"0.51×10^{{-3}} × \\sqrt{{{v('fcuk', c.fcuk)}}} × {v('b', b)} × "
                                                 f"{v('h0', h0)} = {v('Vmax', vmax, 1)}\\,\\u{{kN}}"),
          ("", ">=" if dem <= vmax else "<", f"γ_{{0}}V_{{d}} = {fixed(dem, 1)}\\,\\u{{kN}}")], tag="5.2.11")
    w.eq([("0.50×10^{-3}α_{2}f_{td}bh_{0}", f"0.50×10^{{-3}} × {v('a2', d['alpha2'])} × {v('ftd', c.ftd)} × "
                                            f"{v('b', b)} × {v('h0', h0)} = {v('Vmin', vmin, 1)}\\,\\u{{kN}}"),
          ("", ">=" if dem <= vmin else "<", f"γ_{{0}}V_{{d}} = {fixed(dem, 1)}\\,\\u{{kN}}")], tag="5.2.12")
    w.p(tt("截面尺寸满足 5.2.11 条。" if dem <= vmax else "截面尺寸不满足 5.2.11 条。",
           "The section limit of 5.2.11 is met." if dem <= vmax else "The section limit of 5.2.11 is not met."),
        tt("γ0Vd 不大于 5.2.12 条的限值，可只按构造配置箍筋。" if dem <= vmin else "γ0Vd 大于 5.2.12 条的限值，需按计算配置箍筋（上式）。",
           " γ0Vd is within 5.2.12: detailing stirrups suffice." if dem <= vmin else
           " γ0Vd exceeds 5.2.12: the stirrups are designed by calculation (above)."))
    return True


def _tpl_section_limit(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if w.sec is None or "V0d" not in d:
        return False
    c = w.sec.concrete
    R = 0.51e-3 * math.sqrt(c.fcuk) * d["b"] * d["h0"]
    if not _close(R, chk.resistance):
        return False
    v = w.v
    w.units()
    w.eq([("0.51×10^{-3}\\sqrt{f_{cu,k}}bh_{0}", f"0.51×10^{{-3}} × \\sqrt{{{v('fcuk', c.fcuk)}}} × {v('b', d['b'])} × "
                                                 f"{v('h0', d['h0'])} = {v('R', R, 1)}\\,\\u{{kN}}")], tag="5.2.11")
    return True


def _tpl_slab_shear(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None:
        return False
    c = s.concrete
    R = 1.25 * 0.50e-3 * d["alpha2"] * c.ftd * d["b"] * d["h0"]
    if not _close(R, chk.resistance):
        return False
    v = w.v
    w.units()
    w.eq([("V_{u}", "1.25 × 0.50×10^{-3}α_{2}f_{td}bh_{0}"),
          ("", f"1.25 × 0.50×10^{{-3}} × {v('a2', d['alpha2'])} × {v('ftd', c.ftd)} × {v('b', d['b'])} × "
               f"{v('h0', d['h0'])} = {v('Vu', R, 1)}\\,\\u{{kN}}")], tag="5.2.12")
    return True


def _tpl_axial(w: _W) -> bool:
    chk, s, d = w.chk, w.sec, w.chk.details
    if s is None or "A_used" not in d:
        return False
    c, st = s.concrete, s.steel
    Nu = 0.9 * d["phi"] * (c.fcd * d["A_used"] + st.fsd_c * d["As"]) / 1e3
    if not _close(Nu, chk.resistance):
        return False
    v, tt = w.v, w.t
    w.units()
    if d.get("l0"):
        sym = {"b": "b", "d": "d", "i": "i"}[d["slenderness_by"]]
        lam = d["l0"] / d["slenderness_dim"]
        w.eq([(f"l_{{0}}/{sym}", f"{v('l0', d['l0'])}/{v('dim', d['slenderness_dim'])} = {fixed(lam, 2)}"),
              ("φ", f"{v('phi', d['phi'])}" + tt("\\t{（表 5.3.1，线性内插）}", "\\t{ (table 5.3.1, interpolated)}"))])
    else:
        w.p(tt("未给出计算长度，取 φ = 1.0。", "No effective length given: φ = 1.0."))
    if d["rho"] > 0.03:
        w.p(tt("纵向钢筋配筋率大于 3%，A 取净截面面积 A_{n} = A − A'_{s}。",
               "Longitudinal steel above 3 %: A is the net area An = A − A's."))
    w.eq([("N_{u}", "0.90φ(f_{cd}A + f'_{sd}A'_{s})"),
          ("", f"0.90 × {v('phi', d['phi'])} × ({v('fcd', c.fcd)} × {v('A', d['A_used'])} + {v('fsdc', st.fsd_c)} × "
               f"{v('As', d['As'])})"), ("", f"{v('Nu', Nu, 1)}\\,\\u{{kN}}")], tag="5.3.1")
    return True


def _tpl_eccentric(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None or "e0" not in d:
        return False
    v, tt = w.v, w.t
    c, st = s.concrete, s.steel
    Nd, l0 = d.get("Nd"), d.get("l0")
    if Nd is None:
        return False
    e, h0 = d["e"], d["h0"]
    if "alpha" in d:  # 5.3.8, solid circle
        r = s.h / 2
        A = math.pi * r * r
        As = s.bars[:, 0].sum()
        a = d["alpha"]
        at = max(1.25 - 2 * a, 0.0)
        N = a * c.fcd * A * (1 - math.sin(2 * math.pi * a) / (2 * math.pi * a)) + (a - at) * st.fsd * As
        if not _close(N / 1e3, chk.resistance):
            return False
        w.units()
        _ecc_head(w, d, s, l0, h0, circle=True)
        w.head(tt("（3）承载力（5.3.8，沿周边均匀配筋的圆形截面）", "(3) Resistance (5.3.8, circle with bars evenly round)"))
        w.p(tt("由式（5.3.8-1）、（5.3.8-2）联立求得对应于受压区混凝土截面面积的圆心角与 2π 的比值：",
               "From 5.3.8-1 and -2 together, the angle ratio of the compression zone:"), m(f"α = {v('alpha', a)}"),
            "，" if w.lang == "zh" else ", ", m(f"α_{{t}} = 1.25 - 2α = {v('alphat', at)}"),
            "。" if w.lang == "zh" else ".")
        w.eq([("N_{u}", "αf_{cd}A(1 - \\frac{\\op{sin}2πα}{2πα}) + (α - α_{t})f_{sd}A_{s}"),
              ("", f"{v('alpha', a)} × {v('fcd', c.fcd)} × {v('A', A)} × (1 - \\frac{{\\op{{sin}}(2π ×"
                   f" {v('alpha', a)})}}"
                   f"{{2π × {v('alpha', a)}}}) + ({v('alpha', a)} - {v('alphat', at)}) × {v('fsd', st.fsd)} × "
                   f"{v('As', As)}"), ("", f"{v('Nu', N / 1e3, 1)}\\,\\u{{kN}}")], tag="5.3.8-1")
        return True
    if s.kind not in ("rect", "tee", "box") or s.prestressed or chk.notes and any(
            "reverse" in n or "x < 2a'" in n or "between the gross" in n for n in chk.notes):
        return False
    g = _steel(s)
    fl = _flange(s)
    if not g.edge_only or fl is None or "x" not in d:
        return False
    bf, hf, b = fl
    flange = b is not None and bf > b + 1e-9
    b = bf if b is None else b
    x = d["x"]
    xc = min(x, s.h)
    if xc <= hf + 1e-9 or not flange:
        Cc = c.fcd * bf * xc
        Mc = Cc * (h0 - xc / 2)
    else:
        Cc = c.fcd * (b * xc + (bf - b) * hf)
        Mc = c.fcd * (b * xc * (h0 - xc / 2) + (bf - b) * hf * (h0 - hf / 2))
    if not flange and xc > hf + 1e-9 and _flange(s)[2] is not None:
        return False
    large = d["case"].startswith("large")
    if large:
        ss = st.fsd
    else:
        ss = float(np.clip(c.eps_cu * st.Es * (c.beta * h0 / x - 1), -st.fsd_c, st.fsd))
        bot, _ = s.edge_masks()
        if len(np.unique(np.round(s.bars[bot, 1], 6))) > 1:
            return False
    sc = st.fsd_c if (g.Asc and x >= 2 * g.asc - 1e-9) else None
    if g.Asc and sc is None:
        return False
    Nu = Cc + (sc or 0.0) * g.Asc - ss * g.As
    if not _close(Nu / 1e3, chk.resistance):
        return False
    es = e + (h0 - s.zg)
    w.units()
    _ecc_head(w, d, s, l0, h0, circle=False)
    w.head(tt("（3）大小偏心判别", "(3) Large or small eccentricity"))
    w.p(tt("由式（5.3.4-1）、（5.3.4-2）联立（钢筋应力按 5.1.5 条）解得受压区高度：",
           "From 5.3.4-1 and -2 together (steel stresses by 5.1.5), the depth of the compression zone:"),
        m(f"x = {v('x', x, 1)}\\,\\u{{mm}}"), "。" if w.lang == "zh" else ".")
    xb = d["xi_b"]
    w.eq([("ξ", f"\\frac{{x}}{{h_{{0}}}} = \\frac{{{v('x', x, 1)}}}{{{v('h0', h0)}}} = {fixed(x / h0, 3)}"),
          ("", "<=" if large else ">", f"ξ_{{b}} = {v('xib', xb)}")], tag="5.3.3")
    if large:
        w.p(tt("为大偏心受压，受拉钢筋应力取 ", "Large eccentricity: the tension steel at "),
            m(f"σ_{{s}} = f_{{sd}} = {v('ss', ss)}\\,\\u{{MPa}}"), "。" if w.lang == "zh" else ".")
    else:
        w.p(tt("为小偏心受压，受拉一侧钢筋应力按式（5.1.5-1）：", "Small eccentricity: the far steel by 5.1.5-1:"))
        w.eq([("σ_{s}", "ε_{cu}E_{s}(\\frac{βh_{0}}{x} - 1)"),
              ("", f"{v('ecu', c.eps_cu)} × {v('Es', st.Es)} × (\\frac{{{v('beta', c.beta)} × {v('h0', h0)}}}"
                   f"{{{v('x', x, 1)}}} - 1) = {v('ss', ss)}\\,\\u{{MPa}}")], tag="5.1.5-1")
    w.head(tt("（4）偏心受压承载力", "(4) Resistance"))
    inside = xc <= hf + 1e-9 or not flange
    csym = ("f_{cd}b'_{f}x" if flange else "f_{cd}bx") if inside else "f_{cd}[bx + (b'_{f} - b)h'_{f}]"
    xv = v('x', xc, 1)
    cval = f"{v('fcd', c.fcd)} × {v('bf' if flange else 'b', bf)} × {xv}" if inside else \
        f"{v('fcd', c.fcd)} × [{v('b', b)} × {xv} + ({v('bf', bf)} - {v('b', b)}) × {v('hf', hf)}]"
    sym = [csym] + (["f'_{sd}A'_{s}"] if g.Asc else []) + ["- σ_{s}A_{s}"]
    val = [cval] + ([f"{v('fsdc', sc)} × {v('Asc', g.Asc)}"] if g.Asc else []) + \
        [f"- {v('ss', ss)} × {v('As', g.As)}" if ss >= 0 else f"+ {_nv(-ss)} × {v('As', g.As)}"]
    w.eq([("N_{u}", " + ".join(sym).replace("+ -", "-")), ("", " + ".join(val).replace("+ -", "-").replace("+ +", "+")),
          ("", f"{v('Nu', Nu / 1e3, 1)}\\,\\u{{kN}}")], tag="5.3.5-1" if flange else "5.3.4-1")
    Mr = Mc + (sc or 0.0) * g.Asc * (h0 - g.asc)
    w.p(tt("校核弯矩平衡（对受拉钢筋合力点取矩，", "Moment equilibrium about the tension steel ("),
        m(f"e_{{s}} = ηe_{{0}} + h_{{0}} - y_{{c}} = {v('e', e, 1)} + {v('h0', h0)} - {v('yc', s.zg)} = "
          f"{fixed(es, 1)}\\,\\u{{mm}}"), tt("）：", "):"))
    w.eq([("N_{u}e_{s}", f"{fixed(Nu / 1e3, 1)} × {fixed(es, 1)}×10^{{-3}} = {fixed(Nu * es / 1e6, 1)}\\,\\u{{kN·m}}"),
          ("M_{R}", ("f_{cd}b'_{f}x(h_{0} - \\frac{x}{2})" if (inside and flange) else "f_{cd}bx(h_{0} - \\frac{x}{2})"
                     if inside else "f_{cd}[bx(h_{0} - \\frac{x}{2}) + (b'_{f} - b)h'_{f}(h_{0} - \\frac{h'_{f}}{2})]")
           + (" + f'_{sd}A'_{s}(h_{0} - a'_{s})" if g.Asc else "") + f" = {fixed(Mr / 1e6, 1)}\\,\\u{{kN·m}}")],
         tag="5.3.4-2")
    return True


def _ecc_head(w: _W, d, s, l0, h0, circle: bool):
    v, tt = w.v, w.t
    w.head(tt("（1）初始偏心距", "(1) Initial eccentricity"))
    Md, Nd = abs(d["Md"]), d["Nd"]
    rows = [("e_{0}", f"\\frac{{M_{{d}}}}{{N_{{d}}}} = \\frac{{{fixed(Md, 1)}}}{{{fixed(Nd, 1)}}} × 10^{{3}} = "
                      f"{fixed(d['e0_from_forces'], 1)}\\,\\u{{mm}}")]
    if d["e0"] > d["e0_from_forces"] + 1e-9:
        rows.append(("", "<", f"\\op{{max}}(20, h/30) \\t{{，取}} e_{{0}} = {v('e0', d['e0'], 1)}\\,\\u{{mm}}"
                     if w.lang == "zh" else f"\\op{{max}}(20, h/30): e_{{0}} = {v('e0', d['e0'], 1)}\\,\\u{{mm}}"))
    else:
        rows[0] = ("e_{0}", rows[0][1].replace(f"= {fixed(d['e0_from_forces'], 1)}\\,", f"= {v('e0', d['e0'], 1)}\\,"))
    w.eq(rows, tag="5.3.9")
    w.head(tt("（2）偏心距增大系数", "(2) Moment magnifier"))
    eta = d["eta"]
    if eta == 1.0:
        w.p(tt("长细比不大于 17.5（或未给出计算长度），取 ", "Slenderness l0/i ≤ 17.5 (or no l0): "),
            m(f"η = {v('eta', 1.0)}"), "。" if w.lang == "zh" else ".")
    else:
        h = s.h
        e0 = d["e0"]
        z1r, z2r = 0.2 + 2.7 * e0 / h0, 1.15 - 0.01 * l0 / h
        z1, z2 = min(z1r, 1.0), min(z2r, 1.0)
        def cap(r, z, k):
            return (f"{fixed(r, 3)} > 1.0\\t{{，取}} ζ_{{{k}}} = {v('z' + k, z)}" if w.lang == "zh" else
                    f"{fixed(r, 3)} > 1.0: ζ_{{{k}}} = {v('z' + k, z)}")
        r1 = cap(z1r, z1, "1") if z1r > 1.0 else v('z1', z1)
        r2 = cap(z2r, z2, "2") if z2r > 1.0 else v('z2', z2)
        w.eq([("ζ_{1}", f"0.2 + 2.7\\frac{{e_{{0}}}}{{h_{{0}}}} = 0.2 + 2.7 ×"
                        f" \\frac{{{v('e0', e0, 1)}}}{{{v('h0', h0)}}} = {r1}"),
              ("ζ_{2}", f"1.15 - 0.01\\frac{{l_{{0}}}}{{h}} = 1.15 - 0.01 × \\frac{{{v('l0', l0)}}}{{{v('h', h)}}} ="
                        f" {r2}"),
              ("η", f"1 + \\frac{{1}}{{1300e_{{0}}/h_{{0}}}}(\\frac{{l_{{0}}}}{{h}})^{{2}}ζ_{{1}}ζ_{{2}} = 1 + "
                    f"\\frac{{1}}{{1300 × {v('e0', e0, 1)}/{v('h0', h0)}}} ×"
                    f" (\\frac{{{v('l0', l0)}}}{{{v('h', h)}}})^{{2}}"
                    f" × {v('z1', z1)} × {v('z2', z2)} = {v('eta', eta)}")], tag="5.3.9")
    w.eq([("e", f"ηe_{{0}} = {v('eta', eta)} × {v('e0', d['e0'], 1)} = {v('e', d['e'], 1)}\\,\\u{{mm}}")])


def _tpl_tension(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if s is None or "case" not in d:
        return False
    v, tt = w.v, w.t
    ft = d["fsd_tension"]
    if d["case"] == "axial tension":
        if s.prestressed:
            return False
        As = s.bars[:, 0].sum()
        if not _close(ft * As / 1e3, chk.resistance):
            return False
        w.units()
        w.eq([("N_{u}", f"f_{{sd}}ΣA_{{s}} = {v('fsd', ft)} × {v('As', As)} = "
                        f"{v('Nu', ft * As / 1e3, 1)}\\,\\u{{kN}}")],
             tag="5.4.1")
        return True
    if d["case"] == "small eccentricity" and not s.prestressed:
        w.units()
        w.p(tt("轴向拉力作用在两侧钢筋合力点之间，为小偏心受拉：", "The load lies between the two steel groups: small "
               "eccentricity:"), m(f"e_{{0}} = {v('e0', d['e0'], 1)}\\,\\u{{mm}}"), "，" if w.lang == "zh" else ", ",
            m(f"e = {v('e', d['e'], 1)}\\,\\u{{mm}}"), "，" if w.lang == "zh" else ", ",
            m(f"e' = {v('ep', d['e_prime'], 1)}\\,\\u{{mm}}"), "。" if w.lang == "zh" else ".")
        w.eq([("N_{u1}", f"\\frac{{Σf_{{sd}}A_{{si}}(h_{{0}} - a_{{si}})}}{{e}} ="
                         f" {v('N1', d['N_5_4_2_1'], 1)}\\,\\u{{kN}}"),
              ("N_{u2}", f"\\frac{{Σf_{{sd}}A_{{si}}(a_{{si}} - a'_{{s}})}}{{e'}} ="
                         f" {v('N2', d['N_5_4_2_2'], 1)}\\,\\u{{kN}}"),
              ("N_{u}", f"\\op{{min}}(N_{{u1}}, N_{{u2}}) = {v('Nu', chk.resistance, 1)}\\,\\u{{kN}}")],
             tag="5.4.2-1, -2")
        w.p(tt(f"钢筋抗拉强度取 {_nv(ft)} MPa。", f"Bars at {_nv(ft)} MPa."))
        return True
    return False


def _tpl_torsion(w: _W) -> bool:
    chk, s, d = w.chk, w.sec, w.chk.details
    if s is None or "Wt" not in d or "zeta_used" not in d or s.prestressed:
        return False
    if d.get("prestress_counted"):
        return False
    c = s.concrete
    bt = d.get("beta_t")
    q = d["q_torsion"] if bt is not None else d.get("q_provided")
    if q is None:
        return False
    conc = 0.35 * d["beta_a"] * (1.5 - bt if bt is not None else 1.0) * c.ftd * d["Wt"]
    stl = 1.2 * math.sqrt(d["zeta_used"]) * d["fsv"] * q * d["Acor"]
    Tu = (conc + stl) / 1e6
    if not _close(Tu, chk.resistance, 1e-5):
        return False
    v, tt = w.v, w.t
    w.units()
    if bt is not None:
        w.eq([("β_{t}", f"\\frac{{1.5}}{{1 + 0.5\\frac{{V_{{d}}W_{{t}}}}{{T_{{d}}bh_{{0}}}}}} \\t{{（0.5～1.0）}} = "
                        f"{v('beta_t', bt)}" if w.lang == "zh" else
                        f"\\frac{{1.5}}{{1 + 0.5\\frac{{V_{{d}}W_{{t}}}}{{T_{{d}}bh_{{0}}}}}} (0.5…1.0) ="
                        f" {v('beta_t', bt)}")],
             tag="5.5.4-3")
    w.eq([("ζ", f"\\frac{{f_{{sd}}A_{{st}}s_{{v}}}}{{f_{{sv}}A_{{sv1}}U_{{cor}}}} = {v('zeta', d['zeta_used'])}")],
         tag="5.5.1-2")
    q_sym = "\\frac{A_{sv1}}{s_{v}}"
    if bt is None:
        w.eq([("T_{u}", f"0.35β_{{a}}f_{{td}}W_{{t}} + 1.2\\sqrt{{ζ}}f_{{sv}}{q_sym}A_{{cor}}"),
              ("", f"0.35 × {v('beta_a', d['beta_a'])} × {v('ftd', c.ftd)} × {v('Wt', d['Wt'])} + 1.2 × "
                   f"\\sqrt{{{v('zeta', d['zeta_used'])}}} × {v('fsv', d['fsv'])} × {v('q', q)} ×"
                   f" {v('Acor', d['Acor'])}"),
              ("", f"{v('Tu', Tu, 1)}\\,\\u{{kN·m}}")], tag="5.5.1-1")
    else:
        w.p(tt("箍筋按 5.5.6 条：先扣除抗剪（式 5.5.4-1）所需的部分，余下的用于抗扭：",
               "Stirrups by 5.5.6: the part shear needs (5.5.4-1) taken off, the rest for torsion:"),
            m(f"A_{{sv1}}/s_{{v}} = {v('q', q)}\\,\\u{{mm^{{2}}/mm}}"), "。" if w.lang == "zh" else ".")
        w.eq([("T_{u}", f"0.35β_{{a}}(1.5 - β_{{t}})f_{{td}}W_{{t}} + 1.2\\sqrt{{ζ}}f_{{sv}}{q_sym}A_{{cor}}"),
              ("", f"0.35 × {v('beta_a', d['beta_a'])} × (1.5 - {v('beta_t', bt)}) × {v('ftd', c.ftd)} × "
                   f"{v('Wt', d['Wt'])} + 1.2 × \\sqrt{{{v('zeta', d['zeta_used'])}}} × {v('fsv', d['fsv'])} × "
                   f"{v('q', q)} × {v('Acor', d['Acor'])}"),
              ("", f"{v('Tu', Tu, 1)}\\,\\u{{kN·m}}")], tag="5.5.4-2")
    return True


def _tpl_crack_width(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    need = ("sigma_ss", "C1", "C2", "C3", "c", "d", "rho_te_used", "E")
    if not all(k in d for k in need):
        return False
    W = d["C1"] * d["C2"] * d["C3"] * d["sigma_ss"] / d["E"] * (d["c"] + d["d"]) / (0.36 + 1.7 * d["rho_te_used"])
    if not _close(W, chk.demand):
        return False
    v, tt = w.v, w.t
    w.units()
    flex = d.get("member") == "flexural" and d.get("N", 0) == 0 and s is not None and not s.prestressed \
        and "z" not in d and "sigma_s_t" not in d and d.get("As_te")
    w.head(tt("（1）钢筋应力", "(1) Steel stress"))
    if flex and _close(abs(d["Ms"]) * 1e6 / (0.87 * d["As_te"] * d["h0"]), d["sigma_ss"]):
        w.eq([("σ_{ss}", "\\frac{M_{s}}{0.87A_{s}h_{0}}"),
              ("", f"\\frac{{{v('Ms', abs(d['Ms']), 1)}×10^{{6}}}}{{0.87 × {v('As', d['As_te'])} ×"
                   f" {v('h0', d['h0'])}}}"),
              ("", f"{v('sss', d['sigma_ss'])}\\,\\u{{MPa}}")], tag="6.4.4-2")
    else:
        w.p(tt("按 6.4.4 条（构件受力性质、轴力和温度作用见计算参数）求得：", "By 6.4.4 (the nature of the forces, axial "
               "force and temperature in the values below):"), m(f"σ_{{ss}} = {v('sss', d['sigma_ss'])}\\,\\u{{MPa}}"),
            "。" if w.lang == "zh" else ".")
    w.head(tt("（2）系数", "(2) Factors"))
    rows = []
    if "Ms" in d and d.get("Ms") and "Ml" in d and d.get("member") != "tension":
        r = d["Ml"] / d["Ms"]
        if _close(1 + 0.5 * max(r, 0.0), d["C2"]):
            rows.append(("C_{2}", f"1 + 0.5\\frac{{M_{{l}}}}{{M_{{s}}}} = 1 + 0.5 ×"
                                  f" \\frac{{{v('Ml', abs(d['Ml']), 1)}}}"
                                  f"{{{v('Ms', abs(d['Ms']), 1)}}} = {v('C2', d['C2'])}"))
    if not rows:
        rows.append(("C_{2}", f"{v('C2', d['C2'])}"))
    if "Ate" in d and d.get("As_te"):
        rows.append(("ρ_{te}", f"\\frac{{A_{{s}}}}{{A_{{te}}}} ="
                               f" \\frac{{{v('As', d['As_te'])}}}{{{v('Ate', d['Ate'])}}} = "
                               f"{fixed(d['rho_te'], 4)}"))
    else:
        rows.append(("ρ_{te}", f"{fixed(d['rho_te'], 4)}"))
    if not _close(d["rho_te"], d["rho_te_used"]):
        rows.append(("", "→", f"{v('rho_te', d['rho_te_used'])}" + tt("\\t{（限于 0.01～0.1）}", "\\t{ (within 0.01…0.1)}")))
    else:
        rows[-1] = (rows[-1][0], rows[-1][1].rsplit("=", 1)[0] + f"= {v('rho_te', d['rho_te_used'])}"
                    if "=" in rows[-1][1] else f"{v('rho_te', d['rho_te_used'])}")
    w.eq(rows)
    w.p(m(f"C_{{1}} = {v('C1', d['C1'])}"), "，" if w.lang == "zh" else ", ", m(f"C_{{3}} = {v('C3', d['C3'])}"),
        "，" if w.lang == "zh" else ", ", m(f"c = {v('c', d['c'])}\\,\\u{{mm}}"), "，" if w.lang == "zh" else ", ",
        m(f"d = {v('d', d['d'])}\\,\\u{{mm}}"), "，" if w.lang == "zh" else ", ",
        m(f"E_{{s}} = {v('Es', d['E'])}\\,\\u{{MPa}}"), "。" if w.lang == "zh" else ".")
    w.head(tt("（3）最大裂缝宽度", "(3) Largest crack width"))
    w.eq([("W_{cr}", "C_{1}C_{2}C_{3}\\frac{σ_{ss}}{E_{s}}(\\frac{c + d}{0.36 + 1.7ρ_{te}})"),
          ("", f"{v('C1', d['C1'])} × {v('C2', d['C2'])} × {v('C3', d['C3'])} × \\frac{{{v('sss', d['sigma_ss'])}}}"
               f"{{{v('Es', d['E'])}}} × \\frac{{{v('c', d['c'])} + {v('d', d['d'])}}}{{0.36 + 1.7 × "
               f"{v('rho_te', d['rho_te_used'])}}}"),
          ("", f"{v('Wcr', W, 3)}\\,\\u{{mm}}")], tag="6.4.3")
    return True


def _tpl_crack_resistance(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "edge" not in d or "top" not in d or "bottom" not in d:
        return False
    v, tt = w.v, w.t
    w.units()
    E = _L[w.lang]
    rows = []
    for e in ("top", "bottom"):
        r = d[e]
        rows.append([E[e], r["W0"] / 1e6, r["sigma_st"], r["sigma_lt"], r["sigma_pc"], r["demand"], r["limit"]])
    w.node.table(rows, header=[E["edge"], "W_{0}", "σ_{st}", "σ_{lt}", "σ_{pc}", chk.symbol, tt("限值", "limit")],
                 units=["", "×10^{6} mm³", "MPa", "MPa", "MPa", "MPa", "MPa"], digits=[None, 1, 2, 2, 2, 2, 2],
                 compact=True)
    g = d[d["edge"]]
    if "Ms" in d and d["Ms"] is not None:
        w.eq([("σ_{st}", f"\\frac{{M_{{s}}}}{{W_{{0}}}} ="
                         f" \\frac{{{v('Ms', d['Ms'], 1)}×10^{{6}}}}{{{v('W0', g['W0'])}}} = "
                         f"{v('sst', g['sigma_st'], 2)}\\,\\u{{MPa}}" + tt("\\t{（拉为正）}", "\\t{ (tension +)}"))],
             tag="6.3.2")
    if "class A" in chk.name:
        ftk = w.s.concrete.ftk if w.s is not None else None
        w.eq([("σ_{st} - σ_{pc}", f"{v('sst', g['sigma_st'], 2)} - {v('spc', g['sigma_pc'], 2)} = "
                                  f"{fixed(g['sigma_st'] - g['sigma_pc'], 2)}\\,\\u{{MPa}}"),
              ("", "<=" if chk.ok else ">", f"0.7f_{{tk}} = 0.7 × {v('ftk', ftk)} ="
                                            f" {fixed(0.7 * ftk, 2)}\\,\\u{{MPa}}")],
             tag="6.3.1-3")
        w.eq([("σ_{lt} - σ_{pc}", f"{fixed(g['sigma_lt'], 2)} - {fixed(g['sigma_pc'], 2)} = "
                                  f"{fixed(g['sigma_lt'] - g['sigma_pc'], 2)}\\,\\u{{MPa}}"),
              ("", "<=" if g["sigma_lt"] <= g["sigma_pc"] + 1e-9 else ">", "0")], tag="6.3.1-4")
    elif "full prestress" in chk.name:
        f = g["limit"] / g["sigma_pc"] if g["sigma_pc"] else 1.0
        w.eq([("σ_{st} - " + fixed(f, 2) + "σ_{pc}", f"{v('sst', g['sigma_st'], 2)} - {fixed(f, 2)} × "
                                                    f"{v('spc', g['sigma_pc'], 2)} ="
                                                    f" {fixed(g['sigma_st'] - g['limit'], 2)}"),
              ("", "<=" if chk.ok else ">", "0")], tag=chk.clause.split()[-1])
    else:
        w.eq([("σ_{G} - σ_{pc}", f"{fixed(g['demand'], 2)} - {fixed(g['sigma_pc'], 2)} = "
                                 f"{fixed(g['demand'] - g['sigma_pc'], 2)}"), ("", "<=" if chk.ok else ">", "0")],
             tag="6.3.1")
    return True


def _tpl_principal(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "at" not in d or "points" not in d:
        return False
    a = d["at"]
    v, tt = w.v, w.t
    w.units()
    rows = [[p["z"], p["b"], p["sigma_cx"], p["sigma_cy"], p["tau"], p["sigma_tp"], p["sigma_cp"]] for p in d["points"]]
    k = [i for i, p in enumerate(d["points"]) if p is a or p["z"] == a["z"]]
    w.node.table(rows, header=[tt("计算点深度", "depth"), "b", "σ_{cx}", "σ_{cy}", "τ", "σ_{tp}", "σ_{cp}"],
                 units=["mm", "mm", "MPa", "MPa", "MPa", "MPa", "MPa"], digits=[0, 0, 2, 2, 2, 2, 2], compact=True,
                 row_cls=["total" if i in k else "" for i in range(len(rows))])
    sx, sy, tau = a["sigma_cx"], a["sigma_cy"], a["tau"]
    tp = math.hypot((sx - sy) / 2, tau) - (sx + sy) / 2
    cp = math.hypot((sx - sy) / 2, tau) + (sx + sy) / 2
    w.p(tt(f"控制点距顶面 {_nv(a['z'])} mm（σcx、σcy 压为正，σtp 拉为正）：",
           f"Governing point {_nv(a['z'])} mm below the top (σcx, σcy compression +, σtp tension +):"))
    if "tension" in chk.name or "inclined" in chk.name:
        w.eq([("σ_{tp}", "\\sqrt{(\\frac{σ_{cx} - σ_{cy}}{2})^{2} + τ^{2}} - \\frac{σ_{cx} + σ_{cy}}{2}"),
              ("", f"\\sqrt{{(\\frac{{{v('scx', sx, 2)} - {v('scy', sy, 2)}}}{{2}})^{{2}} + {v('tau', tau, 2)}^{{2}}}}"
                   f" - "
                   f"\\frac{{{v('scx', sx, 2)} + {v('scy', sy, 2)}}}{{2}} = {v('stp', tp, 2)}\\,\\u{{MPa}}")],
             tag="6.3.3-1")
    else:
        w.eq([("σ_{cp}", "\\frac{σ_{cx} + σ_{cy}}{2} + \\sqrt{(\\frac{σ_{cx} - σ_{cy}}{2})^{2} + τ^{2}}"),
              ("", f"\\frac{{{v('scx', sx, 2)} + {v('scy', sy, 2)}}}{{2}} + \\sqrt{{(\\frac{{{v('scx', sx, 2)} - "
                   f"{v('scy', sy, 2)}}}{{2}})^{{2}} + {v('tau', tau, 2)}^{{2}}}} = {v('scp', cp, 2)}\\,\\u{{MPa}}")],
             tag="6.3.3-1")
    return True


def _tpl_service(w: _W) -> bool:
    chk, s, d = w.chk, w.s, w.chk.details
    if "sigma_kc" not in d or "edge_stress" not in d or s is None:
        return False
    v, tt = w.v, w.t
    E = _L[w.lang]
    w.units()
    rows = []
    for e in ("top", "bottom"):
        rest = d["edge_stress"][e] - d["sigma_kc"][e] - d["sigma_pt"][e]
        rows.append([E[e], d["sigma_kc"][e], d["sigma_pt"][e], rest, d["edge_stress"][e], chk.resistance])
    w.node.table(rows, header=[E["edge"], "σ_{kc}", "σ_{pt}", tt("轴力、温度", "N, temperature"), "σ_{kc} + σ_{pt}",
                               "0.5f_{ck}"], units=["", "MPa", "MPa", "MPa", "MPa", "MPa"],
                 digits=[None, 2, 2, 2, 2, 2],
                 compact=True)
    gov = "top" if "top edge" in chk.name else "bottom"
    w.eq([("σ_{kc}", f"\\frac{{M_{{k}}y_{{0}}}}{{I_{{0}}}} = {v('skc', d['sigma_kc'][gov], 2)}\\,\\u{{MPa}}"),
          ("σ_{kc} + σ_{pt}", f"{v('sigma', d['edge_stress'][gov], 2)}\\,\\u{{MPa}}"),
          ("", "<=" if chk.ok else ">", f"0.5f_{{ck}} = 0.5 × {v('fck', s.concrete.fck)} = {fixed(chk.resistance, 2)}"
                                        "\\,\\u{MPa}")], tag="7.1.5-1")
    sp = np.asarray(d["tendon_stress"])
    if len(sp):
        i = int(np.argmax(sp))
        lim = d.get("tendon_limit")
        w.eq([("σ_{pe} + σ_{p}", f"{v('sp', sp[i], 1)}\\,\\u{{MPa}}"),
              ("", "<=" if (lim is None or sp[i] <= lim) else ">", f"{fixed(lim, 1)}\\,\\u{{MPa}}" if lim else "—")],
             tag="7.1.5-2")
    return True


def _tpl_deflection(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "L" not in d:
        return False
    v, tt = w.v, w.t
    k = 300 if d.get("cantilever") else 600
    w.eq([("f", f"{v('f', chk.demand, 3)}\\,\\u{{mm}}"),
          ("", "<=" if chk.ok else ">", f"\\frac{{L}}{{{k}}} = \\frac{{{v('L', d['L'])}}}{{{k}}} = "
                                        f"{fixed(chk.resistance, 3)}\\,\\u{{mm}}")], tag="6.5.3")
    w.p(tt("f 为按频遇组合（汽车荷载不计冲击）求得、乘以长期增长系数 ηθ 后的挠度，扣除结构自重产生的长期挠度。",
           "f: the deflection under the frequent live load without impact × ηθ, the self-weight part taken off."))
    return True


def _tpl_transient(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "top" not in d or "fck_t" not in d:
        return False
    v, tt = w.v, w.t
    gov = "top" if "top edge" in chk.name else "bottom"
    w.p(tt("预加应力和构件自重（含动力系数）等施工荷载作用下，截面边缘混凝土法向应力（压为正）：",
           "Edge stresses (compression +) under the prestress and the construction loads:"),
        m(f"σ_{{top}} = {v('stop', d['top'], 2)}\\,\\u{{MPa}}"), "，" if w.lang == "zh" else ", ",
        m(f"σ_{{bottom}} = {v('sbot', d['bottom'], 2)}\\,\\u{{MPa}}"), "。" if w.lang == "zh" else ".")
    w.eq([("σ^{t}_{cc}", f"{v('scc', d[gov], 2)}\\,\\u{{MPa}}"),
          ("", "<=" if chk.ok else ">", f"0.70f'_{{ck}} = 0.70 × {v('fckt', d['fck_t'])} = {fixed(chk.resistance, 2)}"
                                        "\\,\\u{MPa}")], tag="7.2.8-1")
    if "sigma_tct" in d:
        w.p(tt("预拉区：", "Pretensioned zone: "), m(f"σ^{{t}}_{{ct}} = {fixed(d['sigma_tct'], 2)}\\,\\u{{MPa}}"),
            tt(f"，f'tk = {_nv(d['ftk_t'])} MPa（7.2.8 条第 2 款，按应力配置预拉区纵筋）。",
               f", f'tk = {_nv(d['ftk_t'])} MPa (7.2.8 item 2)."))
    return True


def _tpl_local(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    if "beta" not in d or "Aln" not in d:
        return False
    v = w.v
    w.units()
    w.eq([("β", f"\\sqrt{{\\frac{{A_{{b}}}}{{A_{{l}}}}}} ="
                f" \\sqrt{{\\frac{{{v('Ab', d['Ab'])}}}{{{v('Al', d['Al'])}}}}} = "
                f"{v('beta', d['beta'])}")], tag="5.7.1")
    w.eq([("F_{u1}", "1.3η_{s}βf_{cd}A_{ln}"),
          ("", f"1.3 × {v('eta_s', d['eta_s'])} × {v('beta', d['beta'])} × {v('fcd', d['fcd'])} × {v('Aln', d['Aln'])}"
               f" = "
               f"{v('F1', d['F_section_max'], 1)}\\,\\u{{kN}}")], tag="5.7.1-1")
    if "rho_v" in d:
        w.eq([("β_{cor}", f"\\sqrt{{\\frac{{A_{{cor}}}}{{A_{{l}}}}}} = \\sqrt{{\\frac{{{v('Acor', d['Acor'])}}}"
                          f"{{{v('Al', d['Al'])}}}}} = {v('beta_cor', d['beta_cor'])}"),
              ("ρ_{v}", f"{v('rho_v', d['rho_v'])}")], tag="5.7.2")
        w.eq([("F_{u2}", "0.9(η_{s}βf_{cd} + kρ_{v}β_{cor}f_{sd})A_{ln}"),
              ("", f"0.9 × ({v('eta_s', d['eta_s'])} × {v('beta', d['beta'])} × {v('fcd', d['fcd'])} +"
                   f" {v('k', d['k'])} × "
                   f"{v('rho_v', d['rho_v'])} × {v('beta_cor', d['beta_cor'])} × {v('fsd', d['fsd'])}) ×"
                   f" {v('Aln', d['Aln'])}"),
              ("", f"{v('F2', d['F_resistance'], 1)}\\,\\u{{kN}}")], tag="5.7.2-1")
        w.eq([("F_{u}", f"\\op{{min}}(F_{{u1}}, F_{{u2}}) = {v('Fu', chk.resistance, 1)}\\,\\u{{kN}}")])
    return True


def _tpl_punching(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    v = w.v
    if "Um" in d and "beta_h" in d:
        w.units()
        sp = d["sigma_pc_m"]
        sp_s = " + 0.15σ_{pc,m}" if sp else ""
        sp_v = f" + 0.15 × {v('spc', sp)}" if sp else ""
        if "V_concrete" not in d:
            R = (0.7 * d["beta_h"] * d["ftd"] + 0.15 * sp) * d["Um"] * d["h0"] / 1e3
            if not _close(R, chk.resistance):
                return False
            w.eq([("F_{u}", f"(0.7β_{{h}}f_{{td}}{sp_s})U_{{m}}h_{{0}}"),
                  ("", f"(0.7 × {v('beta_h', d['beta_h'])} × {v('ftd', d['ftd'])}{sp_v}) × {v('Um', d['Um'])} × "
                       f"{v('h0', d['h0'])} = {v('Fu', R, 1)}\\,\\u{{kN}}")], tag="5.6.1")
        else:
            w.eq([("F_{u}", f"(0.35β_{{h}}f_{{td}}{sp_s})U_{{m}}h_{{0}} + V_{{s}}"),
                  ("", f"(0.35 × {v('beta_h', d['beta_h'])} × {v('ftd', d['ftd'])}{sp_v}) × {v('Um', d['Um'])} × "
                       f"{v('h0', d['h0'])} + {v('Vs', d['V_steel'], 1)}"),
                  ("", f"{v('Fu', chk.resistance, 1)}\\,\\u{{kN}}")], tag=chk.clause.split()[-1])
        return True
    if "bm" in d and "beta_h" in d:
        c = None
        w.units()
        w.eq([("F_{ld}", f"p_{{s}}A = {v('Fld', d['Fld'], 1)}\\,\\u{{kN}}"),
              ("b_{m}", f"\\frac{{b_{{t}} + b_{{b}}}}{{2}} = {v('bm', d['bm'])}\\,\\u{{mm}}"),
              ("F_{u}", f"0.7β_{{h}}f_{{td}}b_{{m}}h_{{0}} = {v('Fu', chk.resistance, 1)}\\,\\u{{kN}}")], tag="5.6.3")
        del c
        return True
    return False


def _tpl_devices(w: _W) -> bool:
    chk, d = w.chk, w.chk.details
    v = w.v
    n = chk.name
    if n.startswith("bearing area"):
        w.eq([("\\frac{R_{ck}}{A_{e}}", f"\\frac{{{v('Rck', d['Rck'], 1)}×10^{{3}}}}{{{v('Ae', d['Ae'])}}} = "
                                        f"{v('sig', chk.demand, 2)}\\,\\u{{MPa}}"),
              ("", "<=" if chk.ok else ">", f"σ_{{c}} = {fixed(chk.resistance, 1)}\\,\\u{{MPa}}")], tag="8.7.3-1")
        return True
    if n.startswith("rubber thickness, shear"):
        k = "2" if "without" in n else "1.43"
        lhs = f"{k}\\sqrt{{Δ_{{l}}^{{2}} + Δ_{{t}}^{{2}}}}" if d.get("dt") else f"{k}Δ_{{l}}"
        val = (f"{k} × \\sqrt{{{v('dl', d['dl'], 2)}^{{2}} + {v('dt', d['dt'], 2)}^{{2}}}}" if d.get("dt") else
               f"{k} × {v('dl', d['dl'], 2)}")
        w.eq([(lhs, f"{val} = {v('req', chk.demand, 2)}\\,\\u{{mm}}"),
              ("", "<=" if chk.ok else ">", f"t_{{e}} = {fixed(chk.resistance, 1)}\\,\\u{{mm}}")],
             tag=chk.clause.split()[-1])
        return True
    if n.startswith("rubber thickness, stability"):
        w.eq([("\\frac{l_{a}}{10}", f"{fixed(d['te_min'], 1)}\\,\\u{{mm}}"),
              ("", "<=", f"t_{{e}} = {v('te', d['te'], 1)}\\,\\u{{mm}}"),
              ("", "<=", f"\\frac{{l_{{a}}}}{{5}} = \\frac{{{v('la', d['la'])}}}{{5}} ="
                         f" {fixed(d['te_max'], 1)}\\,\\u{{mm}}")],
             tag=chk.clause.split()[-1])
        return True
    if n.startswith("compression of the bearing") and "Rck" in d:
        w.eq([("δ_{c,m}", "\\frac{R_{ck}t_{e}}{A_{e}E_{e}} + \\frac{R_{ck}t_{e}}{A_{e}E_{b}}"),
              ("", f"\\frac{{{v('Rck', d['Rck'], 1)}×10^{{3}} × {v('te', d['te'])}}}{{{v('Ae', d['Ae'])} ×"
                   f" {v('Ee', d['Ee'])}}}"
                   f" + \\frac{{{v('Rck', d['Rck'], 1)}×10^{{3}} × {v('te', d['te'])}}}{{{v('Ae', d['Ae'])} × "
                   f"{v('Eb', d['Eb'])}}} = {v('dcm', d['delta_cm'], 3)}\\,\\u{{mm}}"),
              ("", "<=" if chk.demand <= chk.resistance + 1e-9 else ">", f"0.07t_{{e}} = {fixed(chk.resistance, 3)}"
                                                                          "\\,\\u{mm}")], tag="8.7.3-8, -9")
        w.eq([("θ\\frac{l_{a}}{2}", f"{fixed(d['uplift_limit'], 3)}\\,\\u{{mm}}"),
              ("", "<=" if d["delta_cm"] >= d["uplift_limit"] - 1e-12 else ">", "δ_{c,m}")], tag="8.7.3-8")
        return True
    if n.startswith("steel plates") and "Rck" in d:
        w.eq([("t_{s}", "\\frac{K_{p}R_{ck}(t_{es,u} + t_{es,l})}{A_{e}σ_{s}}"),
              ("", f"\\frac{{{v('Kp', d['Kp'])} × {v('Rck', d['Rck'], 1)}×10^{{3}} × {v('tes', d['tes_u_plus_l'])}}}"
                   f"{{{v('Ae', d['Ae'])} × {v('sigma_s', d['sigma_s'])}}} = {v('ts_req', chk.demand, 2)}\\,\\u{{mm}}"),
              ("", "<=" if chk.ok else ">", f"t_{{s}} = {fixed(chk.resistance, 1)}\\,\\u{{mm}}")], tag="8.7.3-10")
        return True
    if n.startswith("sliding"):
        R = d["R"]
        fb = " + F_{bk}" if "Fbk" in d else ""
        fbv = f" + {v('Fbk', d['Fbk'], 1)}" if "Fbk" in d else ""
        w.eq([("1.4G_{e}A_{g}\\frac{Δ_{l}}{t_{e}}" + fb,
               f"1.4 × {v('Ge', d['Ge'])} × {v('Ag', d['Ag'])} × \\frac{{{v('dl', d['dl'], 2)}}}{{{v('te', d['te'])}}}"
               f" × 10^{{-3}}"
               f"{fbv} = {v('H', chk.demand, 1)}\\,\\u{{kN}}"),
              ("", "<=" if chk.ok else ">", f"μR = {v('mu', d['mu'])} × {v('R', R, 1)} = {fixed(chk.resistance, 1)}"
                                            "\\,\\u{kN}")], tag=chk.clause.split()[-1])
        return True
    if n.startswith("PTFE friction"):
        w.eq([("μ_{f}R", f"{v('muR', chk.demand, 1)}\\,\\u{{kN}}"),
              ("", "<=" if chk.ok else ">", f"G_{{e}}A_{{g}}\\op{{tan}}α = {fixed(chk.resistance, 1)}\\,\\u{{kN}}"
                                            f"\\t{{（tan α = {_nv(d['tan_alpha'])}）}}")], tag=chk.clause.split()[-1])
        return True
    if n.startswith("expansion joint"):
        b = d["beta"]
        w.eq([("C^{+}", f"β(Δl_{{t}}^{{+}} + Δl_{{b}}^{{+}}) = {v('beta', b)} × ({v('tp', d['t_plus'], 1)} + "
                        f"{v('br', d['brake'], 1)}) = {v('Cp', d['C_plus'], 1)}\\,\\u{{mm}}"),
              ("C^{-}", f"β(Δl_{{t}}^{{-}} + Δl_{{s}}^{{-}} + Δl_{{c}}^{{-}} + Δl_{{b}}^{{-}}) = {v('beta', b)} × "
                        f"({v('tm', d['t_minus'], 1)} + {v('sh', d['shrinkage'], 1)} + {v('cr', d['creep'], 1)} + "
                        f"{v('br', d['brake'], 1)}) = {v('Cm', d['C_minus'], 1)}\\,\\u{{mm}}"),
              ("C^{+} + C^{-}", f"{fixed(chk.demand, 1)}\\,\\u{{mm}}"),
              ("", "<=" if chk.ok else ">", f"C = {fixed(chk.resistance, 1)}\\,\\u{{mm}}")], tag="8.8.2-7")
        return True
    return False


def _tpl_detailing(w: _W) -> bool:
    chk = w.chk
    if not chk.detailing:
        return False
    tt = w.t
    rows = []
    flagged = set()
    for k, val in chk.details.items():
        if isinstance(val, dict) and "value" in val and "limit" in val:
            bad = any(x.startswith(k + " ") or x.startswith(k + ":") for x in chk.warnings)
            if bad:
                flagged.add(k)
            name = _rule_zh(k) if w.lang == "zh" else k
            rows.append([name, val["value"], val["limit"],
                         Cell(_L[w.lang]["rule_fail"] if bad else _L[w.lang]["rule_ok"], "fail" if bad else "ok")])
    if rows:
        w.node.table(rows, header=[_L[w.lang]["rules"], _L[w.lang]["actual"], _L[w.lang]["limit"],
                                   _L[w.lang]["verdict"]], compact=True,
                     row_cls=["fail" if r[3].cls == "fail" else "" for r in rows], align="lrrc")
    other = {k: val for k, val in chk.details.items() if not (isinstance(val, dict) and "value" in val)}
    if other:
        _details_table(w, other)
    if not rows and not other and not chk.warnings:
        w.p(tt("各项构造要求满足。", "All rules met."))
    return True


# ── the general form ────────────────────────────────────────────────────────
_KEYS = {
    "x": ("x", "mm"), "h0": ("h_{0}", "mm"), "xi": ("ξ", ""), "xi_b": ("ξ_{b}", ""), "b": ("b", "mm"),
    "rho": ("ρ", ""), "rho_min": ("ρ_{min}", ""), "P": ("P", ""), "rho_sv": ("ρ_{sv}", ""), "alpha1": ("α_{1}", ""),
    "alpha2": ("α_{2}", ""), "alpha3": ("α_{3}", ""), "Vcs": ("V_{cs}", "kN"), "Vsb": ("V_{sb}", "kN"),
    "Vpb": ("V_{pb}", "kN"), "fsv": ("f_{sv}", "MPa"), "sigma_ss": ("σ_{ss}", "MPa"), "C1": ("C_{1}", ""),
    "C2": ("C_{2}", ""), "C3": ("C_{3}", ""), "c": ("c", "mm"), "d": ("d", "mm"), "rho_te": ("ρ_{te}", ""),
    "E": ("E_{s}", "MPa"), "W0": ("W_{0}", "mm³"), "e0": ("e_{0}", "mm"), "eta": ("η", ""), "e": ("e", "mm"),
    "phi": ("φ", ""), "Mcr": ("M_{cr}", "kN·m"), "sigma_pc": ("σ_{pc}", "MPa"), "sigma_st": ("σ_{st}", "MPa"),
    "sigma_lt": ("σ_{lt}", "MPa"), "Np0": ("N_{p0}", "kN"), "beta": ("β", ""), "Ab": ("A_{b}", "mm²"),
    "Al": ("A_{l}", "mm²"), "Aln": ("A_{ln}", "mm²"), "fcd": ("f_{cd}", "MPa"), "fsd": ("f_{sd}", "MPa"),
    "ftd": ("f_{td}", "MPa"), "Um": ("U_{m}", "mm"), "z": ("z", "mm"), "Ate": ("A_{te}", "mm²"), "As": ("A_{s}", "mm²"),
    "Ap": ("A_{p}", "mm²"), "Asv": ("A_{sv}", "mm²"), "sv": ("s_{v}", "mm"), "a_s": ("a_{s}", "mm"),
    "Ms": ("M_{s}", "kN·m"), "Ml": ("M_{l}", "kN·m"), "N": ("N", "kN"), "Mk": ("M_{k}", "kN·m"), "l0": ("l_{0}", "mm"),
    "Nd": ("N_{d}", "kN"), "Md": ("M_{d}", "kN·m"), "Wt": ("W_{t}", "mm³"), "Acor": ("A_{cor}", "mm²"),
    "zeta": ("ζ", ""), "beta_t": ("β_{t}", ""), "V_section_max": ("0.51×10^{-3}√f_{cu,k}bh_{0}", "kN"),
    "V_detailing": ("0.50×10^{-3}α_{2}f_{td}bh_{0}", "kN"), "L": ("L", "mm"), "te": ("t_{e}", "mm"),
    "Ae": ("A_{e}", "mm²"), "Rck": ("R_{ck}", "kN"),
}


# named values of the detailing reports and other checks → (zh, unit)
_PARAM_ZH = {
    "min_class_4.5.3": ("表 4.5.3 要求的最低强度等级 C", ""), "cmin": ("最小保护层厚度 $c_{min}$", "mm"),
    "layers (bottom)": ("底排钢筋层数", ""), "layers (top)": ("顶排钢筋层数", ""),
    "spacing limit (bottom)": ("底排钢筋净距限值", "mm"), "spacing limit (top)": ("顶排钢筋净距限值", "mm"),
    "9.3.7 neutral axis": ("9.3.7 受拉区上界（距顶面）", "mm"), "environment": ("环境类别", ""),
    "member": ("构件类别", ""), "life": ("设计使用年限", "年"), "class": ("混凝土强度等级 C", ""),
}


def _fmt_val(val):
    if isinstance(val, (bool, np.bool_)):
        return "✓" if val else "✗"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, np.ndarray):
        a = val.ravel()
        if a.size <= 6 and np.issubdtype(a.dtype, np.number):
            return "、".join(number(float(x)) for x in a)
        if np.issubdtype(a.dtype, np.number) and a.size:
            return f"{a.size} 项，{number(float(np.nanmin(a)))} … {number(float(np.nanmax(a)))}"
        return f"{a.size} 项"
    if isinstance(val, (list, tuple)):
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in val) and len(val) <= 6:
            return "、".join(number(float(x)) for x in val)
        s = ", ".join(str(x) if not isinstance(x, float) else number(x) for x in val)
        return s if len(s) < 120 else s[:117] + "…"
    if val is None:
        return "—"
    s = str(val)
    return s if len(s) < 120 else s[:117] + "…"


def _flatten(d: dict, pre: str = "", depth: int = 0):
    for k, val in d.items():
        key = f"{pre}{k}"
        if isinstance(val, dict) and depth < 1:
            yield from _flatten(val, key + ".", depth + 1)
        elif isinstance(val, list) and val and isinstance(val[0], dict):
            yield key, f"[{len(val)}]"
        else:
            yield key, val


def _details_table(w: _W, details: dict):
    rows = []
    for k, val in _flatten(details):
        sym, unit = _KEYS.get(k.split(".")[-1], (None, ""))
        name = f"${sym}$" if sym else k
        if not sym and w.lang == "zh" and k in _PARAM_ZH:
            name, unit = _PARAM_ZH[k]
        if isinstance(val, str) and w.lang == "zh":
            val = {"superstructure": "上部结构", "substructure": "下部结构", "foundation": "基础"}.get(val, val)
        if "." in k and sym:
            name = f"{k.rsplit('.', 1)[0]}：${sym}$" if w.lang == "zh" else f"{k.rsplit('.', 1)[0]}: ${sym}$"
        rows.append([name, Cell(_fmt_val(val), digits=None), unit])
    if rows:
        L = _L[w.lang]
        w.node.table(rows, header=[L["param"], L["value"], L["unit"]], compact=True, align="lrl", keep=len(rows) <= 30)


def _tpl_general(w: _W) -> bool:
    if w.chk.section is not None and w.chk.name.startswith(("flexure", "eccentric", "tension")):
        w.p(_L[w.lang]["general"])
    _details_table(w, w.chk.details)
    return True


_TEMPLATES = [
    (lambda c: c.detailing, _tpl_detailing),
    (lambda c: c.name.startswith("flexure"), _tpl_flexure),
    (lambda c: c.name.startswith("shear (slab without"), _tpl_slab_shear),
    (lambda c: c.name.startswith("section limit"), _tpl_section_limit),
    (lambda c: c.name.startswith(("shear", "section limit")) and "Vcs" in c.details, _tpl_shear),
    (lambda c: c.name == "axial compression", _tpl_axial),
    (lambda c: "eccentric compression" in c.name, _tpl_eccentric),
    (lambda c: c.name.startswith("tension"), _tpl_tension),
    (lambda c: c.name.startswith("torsion"), _tpl_torsion),
    (lambda c: c.name.startswith("crack width"), _tpl_crack_width),
    (lambda c: c.name.startswith("crack resistance (normal") or (c.name.startswith("crack resistance (") and
                                                                 "edge" in c.details and "top" in c.details),
     _tpl_crack_resistance),
    (lambda c: "principal" in c.name or c.name.startswith("crack resistance (inclined"), _tpl_principal),
    (lambda c: c.name.startswith("concrete compression in service"), _tpl_service),
    (lambda c: c.name.startswith("deflection"), _tpl_deflection),
    (lambda c: c.name.startswith("transient stresses (prestressed"), _tpl_transient),
    (lambda c: c.name.startswith("local compression"), _tpl_local),
    (lambda c: "punching" in c.name and not c.name.startswith("pile cap"), _tpl_punching),
    (lambda c: c.clause.startswith("JTG 3362-2018 8.7") or c.clause.startswith("JTG 3362-2018 8.8"), _tpl_devices),
]


# ── the verdict ─────────────────────────────────────────────────────────────
_SYM = [("flexure", "M_{u}", "γ_{0}M_{d}"), ("cap beam flexure", "M_{u}", "γ_{0}M_{d}"),
        ("shear", "V_{u}", "γ_{0}V_{d}"), ("section limit", "0.51×10^{-3}\\sqrt{f_{cu,k}}bh_{0}", "γ_{0}V_{d}"),
        ("axial", "N_{u}", "γ_{0}N_{d}"), ("eccentric", "N_{u}", "γ_{0}N_{d}"), ("biaxial", "N_{u}", "γ_{0}N_{d}"),
        ("tension", "N_{u}", "γ_{0}N_{d}"), ("torsion", "T_{u}", "γ_{0}T_{d}"),
        ("local compression", "F_{u}", "γ_{0}F_{ld}"), ("punching", "F_{u}", "γ_{0}F_{ld}"),
        ("pile cap punching", "F_{u}", "γ_{0}F_{ld}")]
_LIMIT_SYM = {"Wcr": ("W_{cr}", "[W_{cr}]"), "f": ("f", "[f]"), "σst": ("σ_{st}", "[σ]"), "σtp": ("σ_{tp}", "[σ_{tp}]"),
              "σcp": ("σ_{cp}", "0.6f_{ck}"), "σcc": ("σ_{cc}", "0.5f_{ck}"),
              "σkc + σpt": ("σ_{kc} + σ_{pt}", "0.5f_{ck}"),
              "σG": ("σ_{G}", "σ_{pc}"), "σtcc": ("σ^{t}_{cc}", "0.70f'_{ck}")}


def _sym_math(s: str) -> str:
    """a plain symbol of the program ("σkc + σpt", "ts,req", "2Δl") in the math language"""
    s = s.replace("γ0", "γ_{0}")
    return re.sub(r"(?<=[A-Za-zσδμ])([a-z0-9,]+(?:,[a-z]+)?)(?![A-Za-z(])", lambda x: "_{" + x.group(1) + "}", s)


def _verdict_html(chk: Check, lang: str) -> list:
    """the parts of the verdict line"""
    L = _L[lang]
    if chk.limit_check:
        a, b = _LIMIT_SYM.get(chk.symbol, (_sym_math(chk.symbol), "[" + _sym_math(chk.symbol) + "]"))
        p = 3 if chk.unit == "mm" else 2
        u = f"\\,\\u{{{chk.unit}}}" if chk.unit else ""
        rel = "<=" if chk.demand <= chk.resistance + 1e-9 * max(abs(chk.resistance), 1.0) else ">"
        parts = [m(f"{a} = {fixed(chk.demand, p)}{u} {rel} {b} = {fixed(chk.resistance, p)}{u}")]
        if chk.resistance > 0:
            parts.append(f"{'，' if lang == 'zh' else ', '}{L['ratio']} {chk.ratio:.3f}")
        return parts
    r_sym, d_sym = "R", "γ_{0}S_{d}"
    for k, rs, ds in _SYM:
        if chk.name.startswith(k):
            r_sym, d_sym = rs, ds
    u = f"\\,\\u{{{chk.unit}}}" if chk.unit else ""
    rel = "<=" if abs(chk.demand) <= chk.resistance else ">"
    return [m(f"{d_sym} = {fixed(abs(chk.demand), 1)}{u} {rel} {r_sym} = {fixed(chk.resistance, 1)}{u}"),
            f"{'，' if lang == 'zh' else ', '}{L['ratio']} {chk.ratio:.3f}"]


# ── public ──────────────────────────────────────────────────────────────────
@dataclass
class Entry:
    """a check written into a report: where, the check, its title and the id of its section"""
    where: str
    check: Check
    title: str
    id: str


_FIGURE_CHECKS = ("flexure", "eccentric compression", "axial compression", "tension", "concrete compression in service",
                  "crack resistance (normal section")


def write_check(node: Node, chk: Check, *, where: str | None = None, title: str | None = None, id: str | None = None,
                lang: str | None = None, figure: bool | str = "auto", figure_kw: dict | None = None,
                formulas: bool = True, details: bool = False) -> Node:
    """write a check as a numbered section under `node` (see the module notes) and record it for summary(). where: the
    location ("x = 25.50 m（跨中）"); title: default the location and the check's name; id: for cross-references
    (default "chk:<n>"); figure: the section drawn with the check (codes.jtg.figures.section_figure, figure_kw passed
    on): True, False, or "auto" (default) for the checks the drawing shows something of (flexure, compression,
    tension, stresses, normal-section crack resistance, section detailing); formulas: False writes the general form
    only; details: also the table of Check.details. Returns the new section."""
    lang = _lang(node, lang)
    L = _L[lang]
    store = node.report.store.setdefault("jtg.checks", [])
    uid = id or f"chk:{len(store) + 1}"
    name = check_name(chk, lang)
    head = title or (f"{where} {name}" if (where and lang == "zh") else (f"{name}, {where}" if where else name))
    sec = node.section(head, id=uid)
    w = _W(sec, chk, lang)
    info = []
    if where:
        info.append(f"{L['where']}：{where}" if lang == "zh" else f"{L['where']}: {where}")
    info.append(f"{L['clause']}：{clause_text(chk.clause, lang)}" if lang == "zh" else
                f"{L['clause']}: {clause_text(chk.clause, lang)}")
    sec.paragraph("；".join(info) + "。" if lang == "zh" else "; ".join(info) + ".", cls="muted")
    done = False
    if formulas and not chk.detailing:
        for pred, tpl in _TEMPLATES:
            try:
                if pred(chk):
                    mark = len(sec.blocks)
                    if tpl(w):
                        done = True
                        break
                    del sec.blocks[mark:]
                    w.units_said = False
            except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError, AttributeError):
                del sec.blocks[mark:]
                w.units_said = False
    elif chk.detailing:
        done = _tpl_detailing(w)
    if not done:
        _tpl_general(w)
    elif details and not chk.detailing:
        _details_table(w, chk.details)
    # the verdict
    if chk.detailing:
        if chk.warnings:
            sec.box("warn", title=L["warnings"], items=[message(x, lang) for x in chk.warnings])
        else:
            sec.box("ok", title=L["det_ok"] if lang == "en" else "构造满足要求")
    else:
        ok = chk.ok
        sec.box("ok" if ok else "fail", *_verdict_html(chk, lang),
                title=(L["ok"] if ok else L["not"]) + ("" if lang == "en" else ""))
        if chk.failed:
            sec.box("fail", title=L["failed"], items=[message(x, lang) for x in chk.failed])
        if chk.warnings:
            sec.box("warn", title=L["warnings"], items=[message(x, lang) for x in chk.warnings])
    notes = [x for x in chk.notes if x not in w.hide_notes]
    if notes:
        sec.box("note", title=L["notes"], items=[message(x, lang) for x in notes])
    if figure == "auto":
        figure = chk.name.startswith(_FIGURE_CHECKS) or chk.name.startswith("section detailing")
    if figure and chk.section is not None:
        from .figures import section_figure
        kw = {"lang": lang, "title": "", "width": 560}
        kw.update(figure_kw or {})
        try:
            fig = section_figure(chk.section, check=None if w.plain_figure else chk, **kw)
        except (TypeError, ValueError, NotImplementedError):
            fig = None
        if fig is not None:
            cap = (f"{where} {name}" if where else name) if lang == "zh" else (f"{name}, {where}" if where else name)
            sec.figure(fig, caption=cap, width="560px")
    store.append(Entry(where or "", chk, name, uid))
    return sec


def summary(node: Node, entries=None, *, lang: str | None = None, id: str = "tab:summary",
            warn_id: str = "tab:warnings", caption: str | None = None) -> Node:
    """the summary of the checks (default: every check written with write_check so far): one table of all checks
    (location, check, clause, γ0Sd, R or limit, ratio, verdict; those not met in red, each linked to its section), one
    of the warnings, and the conclusion. entries: Entry records or (where, Check) pairs"""
    lang = _lang(node, lang)
    L = _L[lang]
    rep = node.report
    if entries is None:
        entries = list(rep.store.get("jtg.checks", []))
    ents = [e if isinstance(e, Entry) else Entry(e[0], e[1], check_name(e[1], lang), "") for e in entries]
    rows, cls = [], []
    short = {"zh": ("满足", "不满足"), "en": ("OK", "NOT OK")}[lang]

    def cl(c):
        return clause_text(c.clause, lang).replace("JTG 3362—2018 ", "").replace("JTG 3362-2018 ", "")
    for k, e in enumerate(ents, 1):
        c = e.check
        sec_ref = rep.ref(e.id, bare=True) if e.id else k  # the section the check is written in, linked
        if c.detailing:
            verdict = L["det_warn"].format(k=len(c.warnings)) if c.warnings else short[0]
            rows.append([sec_ref, e.where, e.title, cl(c), "", "", "", "",
                         Cell(verdict, "warn" if c.warnings else "ok")])
            cls.append("warn" if c.warnings else "")
            continue
        p = 3 if c.unit == "mm" else (2 if c.limit_check else 1)
        ok = c.ok
        rows.append([sec_ref, e.where, e.title, cl(c), Cell(abs(c.demand) if not c.limit_check else c.demand, digits=p),
                     Cell(c.resistance, digits=p), Cell(c.unit, "unit"),
                     Cell(c.ratio if math.isfinite(c.ratio) else "∞", digits=3),
                     Cell(short[0] if ok else short[1], "ok" if ok else "fail")])
        cls.append("" if ok else "fail")
    head = ["章节" if lang == "zh" else "Section", L["loc"], L["item"], L["clause"], L["demand"], L["resist"], L["unit"],
            L["ratio"], L["verdict"]]
    node.table(rows, header=head, caption=caption or L["summary_cap"], id=id, row_cls=cls,
               align="llllrrlrc", keep=False, cls="summary fixed",
               widths=["4.6em", "17%", None, "11%", "5.2em", "5.2em", "3.6em", "4em", "4.4em"])
    n = sum(1 for e in ents if not e.check.detailing)
    bad = sum(1 for e in ents if not e.check.detailing and not e.check.ok)
    warns = [(e, w) for e in ents for w in e.check.warnings]
    if warns:
        wrows = [[k, rep.ref(e.id, bare=True) if e.id else "", e.where, e.title, message(w, lang)]
                 for k, (e, w) in enumerate(warns, 1)]
        node.table(wrows, header=[L["no"], "章节" if lang == "zh" else "Sect.", L["loc"], L["item"], L["content"]],
                   caption=L["warn_cap"], id=warn_id, align="cllll", keep=False, cls="summary fixed",
                   widths=["2.6em", "4.8em", "15%", "22%", None])
    concl = L["all_ok"].format(n=n) if not bad else L["some_fail"].format(n=n, k=bad)
    if warns:
        node.box("ok" if not bad else "fail", concl, L["warn_n"].format(k=len(warns)), rep.ref(warn_id), L["warn_n2"],
                 title="结论" if lang == "zh" else "Conclusion")
    else:
        node.box("ok" if not bad else "fail", concl, L["no_warn"], title="结论" if lang == "zh" else "Conclusion")
    return node


# ── design data ─────────────────────────────────────────────────────────────
CODES = {
    "zh": [("JTG D60—2015", "《公路桥涵设计通用规范》"), ("JTG 3362—2018", "《公路钢筋混凝土及预应力混凝土桥涵设计规范》")],
    "en": [("JTG D60-2015", "General Specifications for Design of Highway Bridges and Culverts"),
           ("JTG 3362-2018", "Specifications for Design of Highway Reinforced Concrete and Prestressed Concrete"
                             " Bridges "
                             "and Culverts")],
}


def design_basis(node: Node, extra=(), *, lang: str | None = None) -> Node:
    """the codes the report follows (JTG D60-2015, JTG 3362-2018 and `extra` rows of (number, title))"""
    lang = _lang(node, lang)
    items = [f"{a}{b}" if lang == "zh" else f"{a}: {b}" for a, b in list(CODES[lang]) + list(extra)]
    node.list(items, ordered=True)
    return node


def _bars_text(sec, lang: str) -> list:
    """the rows of bars of a section: 'n Φd' and the depth"""
    out = []
    for (a, z), dd in zip(sec.bars, sec.bar_d):
        if dd and not math.isnan(dd):
            n = a / (math.pi * dd * dd / 4)
            txt = f"{n:.0f}Φ{dd:g}" if abs(n - round(n)) < 0.02 else f"A = {a:.0f} mm²（Φ{dd:g}）"
        else:
            txt = f"A = {a:.0f} mm²"
        out.append((txt, z))
    return out


def materials(node: Node, sections, *, lang: str | None = None, caption: str | None = None,
              id: str | None = "tab:materials") -> Node:
    """the material values of the sections' concrete, bars and tendons (tables 3.1.3–3.1.5, 3.2.2–3.2.4)"""
    lang = _lang(node, lang)
    secs = list(sections.values()) if isinstance(sections, dict) else list(sections)
    concretes, bars, tendons = {}, {}, {}
    for s in secs:
        s = getattr(s, "precast", s)
        concretes[s.concrete.grade] = s.concrete
        bars[s.steel.name] = s.steel
        if getattr(s, "tendon", None) is not None and s.has_internal:
            tendons[s.tendon.name] = s.tendon
        if getattr(s, "ext_tendon", None) is not None and s.has_external:
            tendons[s.ext_tendon.name] = s.ext_tendon
    zh = lang == "zh"
    rows = []
    for g, c in sorted(concretes.items()):
        nm = f"{'混凝土' if zh else 'Concrete'} C{g}"
        for label, sym, val in [("轴心抗压强度标准值" if zh else "characteristic compressive strength", "f_{ck}", c.fck),
                                ("轴心抗拉强度标准值" if zh else "characteristic tensile strength", "f_{tk}", c.ftk),
                                ("轴心抗压强度设计值" if zh else "design compressive strength", "f_{cd}", c.fcd),
                                ("轴心抗拉强度设计值" if zh else "design tensile strength", "f_{td}", c.ftd),
                                ("弹性模量" if zh else "modulus of elasticity", "E_{c}", c.Ec)]:
            rows.append([nm, label, f"${sym}$", Cell(val, digits=None if val < 1e4 else 0), "MPa"])
            nm = ""
    for n_, st in bars.items():
        nm = f"{'普通钢筋' if zh else 'Reinforcing bars'} {n_}"
        for label, sym, val in [("抗拉强度标准值" if zh else "characteristic strength", "f_{sk}", st.fsk),
                                ("抗拉强度设计值" if zh else "design tensile strength", "f_{sd}", st.fsd),
                                ("抗压强度设计值" if zh else "design compressive strength", "f'_{sd}", st.fsd_c),
                                ("弹性模量" if zh else "modulus of elasticity", "E_{s}", st.Es)]:
            rows.append([nm, label, f"${sym}$", Cell(val, digits=0), "MPa"])
            nm = ""
    for n_, t in tendons.items():
        nm = f"{'预应力钢筋' if zh else 'Prestressing steel'} {n_}"
        for label, sym, val in [("抗拉强度标准值" if zh else "characteristic strength", "f_{pk}", t.fpk),
                                ("抗拉强度设计值" if zh else "design tensile strength", "f_{pd}", t.fpd),
                                ("抗压强度设计值" if zh else "design compressive strength", "f'_{pd}", t.fpd_c),
                                ("弹性模量" if zh else "modulus of elasticity", "E_{p}", t.Ep),
                                ("张拉控制应力限值" if zh else "largest jacking stress", "σ_{con}", t.sigma_con_max())]:
            rows.append([nm, label, f"${sym}$", Cell(val, digits=0), "MPa"])
            nm = ""
    node.table(rows, header=["材料" if zh else "Material", "项目" if zh else "Property", "符号" if zh else "Symbol",
                             "数值" if zh else "Value", "单位" if zh else "Unit"],
               caption=caption or ("材料参数" if zh else "Materials"), id=id, align="lllrl")
    return node


def section_properties(node: Node, sections: dict, *, lang: str | None = None, caption: str | None = None,
                       id: str | None = "tab:sections", transformed: bool = True) -> Node:
    """the gross (and transformed) section properties and the reinforcement of each section: {name: RCSection}"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    rows = []
    for name, s in sections.items():
        A0, z0, I0 = s.props("transformed")
        bot, top = s.edge_masks()
        r = [name, s.h, s.web(), s.area, s.zg, Markup(m(_sci_math(s.I)))]
        if transformed:
            r += [A0, z0, Markup(m(_sci_math(I0)))]
        r += [s.bars[bot, 0].sum(), s.bars[top, 0].sum(), s.h0()]
        rows.append(r)
    head = ["截面" if zh else "Section", "h", "b", "A", "y_{c}", "I"]
    units = ["", "mm", "mm", "mm²", "mm", "mm⁴"]
    if transformed:
        head += ["A_{0}", "y_{0}", "I_{0}"]
        units += ["mm²", "mm", "mm⁴"]
    head += ["A_{s}", "A'_{s}", "h_{0}"]
    units += ["mm²", "mm²", "mm"]
    node.table(rows, header=head, units=units, caption=caption or ("截面几何特性及配筋" if zh else "Section properties"),
               id=id, digits=0, align="l" + "r" * (len(head) - 1),
               notes=["y_{c}、y_{0}：毛截面、换算截面重心至截面顶面的距离；A_{s}、A'_{s}：受拉、受压边排纵筋面积（正弯矩）；"
                      "h_{0}：截面有效高度。" if zh else
                      "y_{c}, y_{0}: depth of the gross and transformed centroid; A_{s}, A'_{s}: the tension and"
                      " compression "
                      "edge rows (sagging); h_{0}: effective depth."])
    for name, s in sections.items():
        bt = _bars_text(s, lang)
        if bt:
            node.paragraph(f"{name}：" if zh else f"{name}: ",
                           "；".join(f"{t}（距顶面 {z:.0f} mm）" if zh else f"{t} at {z:.0f} mm" for t, z in bt),
                           "。" if zh else ".")
    return node


def _sci_math(v: float) -> str:
    e = int(math.floor(math.log10(abs(v)))) if v else 0
    return f"{v / 10 ** e:.4f}×10^{{{e}}}"


def load_cases(node: Node, cases, *, model=None, lang: str | None = None, caption: str | None = None,
               id: str | None = "tab:loads") -> Node:
    """a table of the load cases (femotion.frame.loads.LoadCase): case, kind of load, value, where. model: for the
    self-weight per length (Section.mass × g) and node positions"""
    from ...frame.loads import (BarStrain, Imposed, LineLoad, NodalLoad, PointLoad, SelfWeight, Settlement,
                                TendonLoad)
    lang = _lang(node, lang)
    zh = lang == "zh"
    rows = []
    for case in cases:
        first = True
        for it in case.items:
            nm = case.name if first else ""
            first = False
            if isinstance(it, SelfWeight):
                q = None
                if model is not None:
                    ms = {round(sec.mass, 9) for (_, _, sec, _, _) in model.beams}
                    if len(ms) == 1:
                        q = ms.pop() * it.g * it.factor
                kind = "结构自重" if zh else "self-weight"
                val = (f"$q = m g$ = {q:.2f} kN/m" if q is not None else "$m g$") if zh else \
                    (f"$q = m g$ = {q:.2f} kN/m" if q is not None else "$m g$")
                where = "全部构件" if (zh and it.beams is None) else ("all members" if it.beams is None else
                                                                     f"{len(it.beams)} 个单元" if zh else
                                                                     f"{len(it.beams)} elements")
            elif isinstance(it, LineLoad):
                w0, w1 = float(np.linalg.norm(it.w0)), float(np.linalg.norm(it.w1))
                kind = "分布荷载" if zh else "distributed load"
                val = f"$q$ = {w0:.2f} kN/m" if math.isclose(w0, w1) else f"$q$ = {w0:.2f} → {w1:.2f} kN/m"
                if it.start == 0 and it.end is None:
                    where = "全长" if zh else "whole length"
                else:
                    end = ("末端" if zh else "end") if it.end is None else f"{it.end:g} m"
                    where = f"s = {it.start:g} m ～ {end}"
            elif isinstance(it, PointLoad):
                kind = "集中荷载" if zh else "point load"
                val = f"$P$ = {float(np.linalg.norm(it.P)):.1f} kN"
                where = f"s = {it.s:g} m"
            elif isinstance(it, NodalLoad):
                kind = "节点荷载" if zh else "nodal load"
                val = f"F = {number(float(np.linalg.norm(it.F)))} kN, M = {number(float(np.linalg.norm(it.M)))} kN·m"
                where = f"{'节点' if zh else 'node'} {it.node}"
            elif isinstance(it, Imposed):
                kind = "强迫变形（温度、收缩）" if zh else "imposed strain"
                k = np.atleast_1d(it.curvature)
                e = np.atleast_1d(it.strain)
                val = f"κ = {number(float(k.max()))} 1/m, ε = {number(float(e.max()))}"
                where = f"{len(it.beams)} {'个单元' if zh else 'elements'}"
            elif isinstance(it, TendonLoad):
                kind = "预加力" if zh else "prestress"
                f_ = np.atleast_1d(it.tendon.force)
                val = f"N = {number(float(f_.max()))} kN"
                where = f"{len(it.tendon.beams)} {'个单元' if zh else 'elements'}"
            elif isinstance(it, Settlement):
                kind = "支座沉降" if zh else "settlement"
                val = f"Δ = {number(float(it.d[2]) * 1e3)} mm"
                where = f"{'节点' if zh else 'node'} {it.node}"
            elif isinstance(it, BarStrain):
                kind = "初应变" if zh else "initial strain"
                val = f"ε0 = {number(float(np.atleast_1d(it.eps0).max()))}"
                where = f"{len(it.bars)} {'根杆' if zh else 'bars'}"
            else:
                kind, val, where = type(it).__name__, "", ""
            if it.label and not re.search(r"\d", it.label):  # a label repeating the value is left out
                val = f"{val}（{it.label}）" if zh else f"{val} ({it.label})"
            rows.append([nm, kind, val, where])
    node.table(rows, header=["工况" if zh else "Case", "类型" if zh else "Kind", "荷载" if zh else "Load",
                             "作用位置" if zh else "Where"], caption=caption or ("荷载工况" if zh else "Load cases"),
               id=id, align="llll")
    return node


def live_load(node: Node, live, *, mu=None, lang: str | None = None, caption: str | None = None,
              id: str | None = "tab:live", notes=None) -> Node:
    """the lane load of D60 4.3.1 behind a jtg.LiveLoad: grade, qk, Pk (×1.2 for shear), lanes and reduction factors,
    distribution and impact factors. mu: (label, μ) rows when the impact factor differs by effect"""
    from .actions import lane_load
    lang = _lang(node, lang)
    zh = lang == "zh"
    q, P = lane_load(live.L0, live.grade)
    rows = [["公路等级" if zh else "class", "", f"公路—{live.grade}级" if zh else f"Class {live.grade}", ""],
            ["计算跨径" if zh else "computed span", "$L_{0}$", live.L0, "m"],
            ["均布荷载标准值" if zh else "uniform load", "$q_{k}$", q, "kN/m"],
            ["集中荷载标准值" if zh else "concentrated load", "$P_{k}$", P, "kN"],
            ["集中荷载（计算剪力）" if zh else "concentrated load for shear", "$1.2P_{k}$", 1.2 * P, "kN"],
            ["设计车道数" if zh else "design lanes", "", live.lanes, ""],
            ["横向车道布载系数" if zh else "transverse reduction", "", live.transverse if live.transverse is not None
             else "表 4.3.1-5", ""],
            ["荷载横向分布系数" if zh else "lateral distribution factor", "$m$", live.distribution, ""]]
    for label, mu_v in (mu or [("冲击系数" if zh else "impact factor", live.mu)]):
        rows.append([label, "$μ$", Cell(mu_v, digits=3), ""])
    node.table(rows, header=["项目" if zh else "Item", "符号" if zh else "Symbol", "数值" if zh else "Value",
                             "单位" if zh else "Unit"], caption=caption or ("汽车荷载" if zh else "Traffic load"), id=id,
               align="llrl", notes=notes)
    return node


def combinations(node: Node, *, gamma0: float = 1.0, crowd: bool = False, characteristic: bool = False,
                 lang: str | None = None, caption: str | None = None, id: str | None = "tab:combinations") -> Node:
    """the combinations used and their factors (D60 4.1.5, 4.1.6, 4.1.8; values of femotion.codes.jtg.actions)"""
    from .actions import PSI
    lang = _lang(node, lang)
    zh = lang == "zh"
    pf, pq = PSI["vehicle"]
    cf, cq = PSI["crowd"]
    g0 = f"$γ_{{0}}$ = {gamma0:.1f}"
    rows = [["基本组合" if zh else "basic", "4.1.5-1",
             m("γ_{0}S_{ud} = γ_{0}(Σγ_{Gi}G_{ik} + γ_{Q1}Q_{1k} + ψ_{c}Σγ_{Qj}Q_{jk})"),
             f"{g0}；$γ_{{G}}$ = 1.2（有利时 1.0）；$γ_{{Q1}}$ = 1.4（车道荷载，含冲击）"
             + ("；人群 $γ_{Qj}$ = 1.4，$ψ_{c}$ = 0.75" if crowd else "")
             if zh else f"{g0}; $γ_{{G}}$ = 1.2 (1.0 favourable); $γ_{{Q1}}$ = 1.4 (lane load with impact)"
             + ("; crowd $γ_{Qj}$ = 1.4, $ψ_{c}$ = 0.75" if crowd else "")],
            ["频遇组合" if zh else "frequent", "4.1.6-1", m("S_{fd} = ΣG_{ik} + ψ_{f1}Q_{1k} + Σψ_{qj}Q_{jk}"),
             f"汽车荷载 $ψ_{{f1}}$ = {pf:g}（不计冲击）" + (f"；人群 $ψ_{{q}}$ = {cq:g}" if crowd else "") if zh else
             f"traffic $ψ_{{f1}}$ = {pf:g} (no impact)" + (f"; crowd $ψ_{{q}}$ = {cq:g}" if crowd else "")],
            ["准永久组合" if zh else "quasi-permanent", "4.1.6-2", m("S_{qd} = ΣG_{ik} + Σψ_{qj}Q_{jk}"),
             f"汽车荷载 $ψ_{{q}}$ = {pq:g}（不计冲击）" + (f"；人群 $ψ_{{q}}$ = {cq:g}" if crowd else "") if zh else
             f"traffic $ψ_{{q}}$ = {pq:g} (no impact)" + (f"; crowd $ψ_{{q}}$ = {cq:g}" if crowd else "")]]
    if characteristic:
        rows.append(["标准值组合" if zh else "characteristic", "4.1.8", m("S = ΣG_{ik} + Q_{1k} + ΣQ_{jk}"),
                     "汽车荷载计入冲击" if zh else "traffic with impact"])
    node.table(rows, header=["组合" if zh else "Combination", "条文" if zh else "Clause",
                             "表达式" if zh else "Expression", "系数" if zh else "Factors"],
               caption=caption or ("作用组合（JTG D60—2015）" if zh else "Combinations (JTG D60-2015)"), id=id,
               align="llll", widths=["6em", "4.6em", "44%", None])
    return node


# ── shear along the girder ──────────────────────────────────────────────────
def shear_search_table(node: Node, search, *, lang: str | None = None, caption: str | None = None,
                       id: str | None = "tab:inclined", all_sections: bool = False) -> Node:
    """the inclined sections of jtg.shear_search: the 5.2.8 positions (all with all_sections) and the governing one,
    with x_c, x_t, C, m, γ0Vd, Vu and the ratio; the section limits at the end supports"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    gov = search.governing
    secs = [s for s in search.sections if all_sections or s.source != "grid" or s is gov]
    rows, cls = [], []
    for k, s in enumerate(secs, 1):
        src = _source(s.source, lang) + (("（控制）" if zh else " (governing)") if s is gov else "")
        rows.append([k, src, s.x_c, s.x_t, s.C, Cell(s.m, digits=2), Cell(abs(s.check.demand), digits=1),
                     Cell(s.check.resistance, digits=1), Cell(s.ratio, digits=3),
                     Cell("满足" if s.ok and zh else ("OK" if s.ok else ("不满足" if zh else "NOT OK")),
                          "ok" if s.ok else "fail")])
        cls.append("total" if s is gov else ("fail" if not s.ok else ""))
    for c in search.limits:
        rows.append(["", check_name(c, lang), "", "", "", "", Cell(abs(c.demand), digits=1),
                     Cell(c.resistance, digits=1),
                     Cell(c.ratio, digits=3), Cell("满足" if c.ok and zh else ("OK" if c.ok else "不满足"),
                                                    "ok" if c.ok else "fail")])
        cls.append("sub" if c.ok else "fail")
    node.table(rows, header=["序号" if zh else "No.", "验算位置（5.2.8）" if zh else "Position (5.2.8)", "x_{c}", "x_{t}",
                             "C", "m", "γ_{0}V_{d}", "V_{u}", "利用率" if zh else "ratio", "结论" if zh else "verdict"],
               units=["", "", "m", "m", "m", "", "kN", "kN", "", ""], caption=caption or (
                   "斜截面抗剪验算（5.2.8～5.2.11）" if zh else "Inclined sections (5.2.8–5.2.11)"), id=id,
               digits=[None, None, 3, 3, 3, 2, 1, 1, 3, None], row_cls=cls, align="clrrrrrrrc")
    return node


def _source(src: str, lang: str) -> str:
    if lang != "zh":
        return src
    s = src
    s = re.sub(r"h/2 from end support x = (\S+)", r"端支点（x = \1 m）h/2 处", s)
    s = re.sub(r"diaphragm edge, support x = (\S+)", r"中支点（x = \1 m）横隔梁边缘", s)
    s = re.sub(r"stirrup change x = (\S+)", r"箍筋变化处 x = \1 m", s)
    s = re.sub(r"bend x = (\S+)", r"弯起点 x = \1 m", s)
    return s.replace("grid", "搜索网格")


def stirrup_design_block(node: Node, design, Vd: float, *, where: str = "", lang: str | None = None,
                         gamma0: float = 1.0) -> Node:
    """the stirrup design of jtg.stirrup_design (5.2.13-1) with the values substituted, and the check
    back-substituted"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    d = design.details
    sec = node.section("箍筋设计（5.2.13）" if zh else "Stirrup design (5.2.13)", id="sec:stirrups")
    sec.paragraph((f"设计剪力取{where}的 " if zh else f"Design shear at {where}: "),
                  m(f"V'_{{d}} = {fixed(Vd, 1)}\\,\\u{{kN}}"), "，" if zh else ", ",
                  "混凝土和箍筋共同承担不少于 60%（ξ = 0.6），其余由弯起钢筋承担。" if zh else
                  "the concrete and stirrups take at least 60 % (ξ = 0.6), the bent-up bars the rest.")
    Asv = design.legs * math.pi * design.dia ** 2 / 4
    xi = 0.6 if "spacing_exact_5_2_9" in d else None
    if design.spacing_required is not None:
        sec.formula([("s_{v}", "\\frac{0.2×10^{-6}α_{1}^{2}α_{3}^{2}(2 + 0.6P)\\sqrt{f_{cu,k}}A_{sv}f_{sv}bh_{0}^{2}}"
                               "{(ξγ_{0}V_{d})^{2}}"),
                     ("", f"\\frac{{0.2×10^{{-6}} × {_nv(d['alpha1'])}^{{2}} × {_nv(d['alpha3'])}^{{2}} × (2 + 0.6 × "
                          f"{_nv(d['P'])}) × \\sqrt{{{_nv(design.check.section.concrete.fcuk)}}} × {_nv(Asv)} × "
                          f"{_nv(d['fsv'])} × {_nv(d['b'])} × {_nv(d['h0'])}^{{2}}}}{{({_nv(d['demand'])})^{{2}}}}"),
                     ("", f"{fixed(design.spacing_required, 1)}\\,\\u{{mm}}")], tag="5.2.13-1")
    sec.paragraph(("取 " if zh else "Adopted: "), m(f"{design.legs}\\t{{肢}} Φ{_nv(design.dia)}@{_nv(design.spacing)}"
                                                    if zh else f"{design.legs} legs Φ{_nv(design.dia)} @ "
                                                    f"{_nv(design.spacing)}"),
                  "（满足 9.3.12 的间距和最小配箍率要求）。" if zh else " (within 9.3.12).")
    del xi
    write_check(sec, design.check, where=where, figure=False, lang=lang)
    return sec


def bent_bar_table(node: Node, design, *, lang: str | None = None, caption: str | None = None,
                   id: str | None = "tab:bent") -> Node:
    """the rows of bent-up bars of jtg.bent_bar_design (5.2.13-2)"""
    lang = _lang(node, lang)
    zh = lang == "zh"
    node.paragraph(("按式（5.2.13-2）" if zh else "By 5.2.13-2 "),
                   m("A_{sb} = \\frac{γ_{0}V_{sb}}{0.75×10^{-3}f_{sd}\\op{sin}θ_{s}}"),
                   (f"，θs = {design.theta:g}°，分配线 ξV′d = {design.V_line:.1f} kN。" if zh else
                    f", θs = {design.theta:g}°, the line ξ V'd = {design.V_line:.1f} kN."))
    rows = [[k, r.x_top, r.x_bottom, r.x_read, Cell(r.Vsb, digits=1), Cell(r.Asb, digits=0)]
            for k, r in enumerate(design.rows, 1)]
    node.table(rows, header=["排" if zh else "Row", "上弯点" if zh else "top bend", "下弯点" if zh else "bottom bend",
                             "取值位置" if zh else "read at", "V_{sb}", "A_{sb}"],
               units=["", "m", "m", "m", "kN", "mm²"], digits=[None, 3, 3, 3, 1, 0],
               caption=caption or ("弯起钢筋设计（5.2.13）" if zh else "Bent-up bars (5.2.13)"), id=id, align="crrrrr")
    return node


def write_checks(node: Node, items, **kw) -> list[Node]:
    """write_check for each (where, Check) pair"""
    return [write_check(node, c, where=w, **kw) for w, c in items]


__all__.append("write_checks")
