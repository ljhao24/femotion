"""A calculation report (计算书) of the reinforced-concrete girder of examples/jtg_rc_girder.py, as one HTML file
(femotion.report, femotion.codes.jtg.report), illustrative values.

One T girder of a two-span continuous deck (2 × 16 m, C40, HRB400, one of five girders) carrying Class I lane load,
rebuilt here: the same model, loads and checks as jtg_rc_girder.py, written up chapter by chapter —
    1 basis (codes, design parameters); 2 design data (materials, sections with their properties and bars, loads,
    combinations); 3 the model (elevation, supports, elements, frequencies and impact factors); 4 actions and their
    effects (load cases drawn, influence lines with the worst placements, envelopes); 5 ultimate limit state (flexure at
    the governing sections with the pier moment reduced for the bearing, the governing inclined section, stirrups and
    bent-up bars), with the formulas and the values substituted; 6 serviceability (crack widths, deflection and
    pre-camber); 7 detailing (durability, cover and spacing, stirrups); 8 summary of every check, the warnings and the
    conclusion.
Open the HTML in a browser; print it (A4) to get the PDF, or pass --pdf to print it with Chrome (the contents then
carry page numbers).

    uv run python examples/jtg_report_rc_girder.py [--out jtg_report_rc_girder.html] [--lang zh|en] [--pdf out.pdf]
"""
import argparse

import numpy as np

from femotion import viz
from femotion.codes import jtg
from femotion.codes.jtg import charts
from femotion.codes.jtg import report as jr
from femotion.codes.jtg.figures import section_figure
from femotion.frame import Model, Section, Solver
from femotion.frame.figures import elevation_figure, influence_figure, load_figure
from femotion.frame.influence import InfluenceLines, chain_sections
from femotion.frame.loads import LoadCase
from femotion.report import Report, fixed

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="jtg_report_rc_girder.html")
ap.add_argument("--lang", default="zh", choices=["zh", "en"])
ap.add_argument("--pdf", default=None, help="also print to this PDF with headless Chrome (TOC with page numbers)")
opt = ap.parse_args()
lang = opt.lang
zh = lang == "zh"


def T(z, e):
    return z if zh else e


# ── the girder, as examples/jtg_rc_girder.py ────────────────────────────────
L, n = 16.0, 32  # span (m), elements per span
gamma0 = jtg.GAMMA_0[2]  # safety class 2
span_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
pier_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(6, 28, 95)], top=[(14, 25, 60)])
stirrups_span, stirrups_end, stirrups_pier = (2, 10, 200), (2, 12, 100), (4, 10, 100)
# the same bars as they are placed, for the detailing rules: the 10Φ28 of the span (one point at 110 mm in the
# calculation) are welded frames of five layers of 2Φ28 in contact, the 6Φ28 of the pier three layers; the centroids
# stay at 110 and 95 mm, so the resistances are those of the sections above
span_frames = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400",
                      bottom=[(2, 28, 54 + 28 * k) for k in range(5)], top=[(8, 20, 50)])
pier_frames = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400",
                      bottom=[(2, 28, 67 + 28 * k) for k in range(3)], top=[(14, 25, 60)])

c = span_sec.concrete
area, I = span_sec.area * 1e-6, span_sec.I * 1e-12  # noqa: E741
m_girder, m_sdl = area * 2.6, 0.55  # t/m: the girder and its share of pavement and railing
mass = m_girder + m_sdl
sec = Section(E=c.Ec * 1e3, A=area, Iy=I, Iz=10 * I, J=0.01, mass=mass)
model = Model()
xs = np.linspace(0, 2 * L, 2 * n + 1)
nodes = [model.node((x, 0, 0)) for x in xs]
beams = [model.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
model.fix(nodes[0], (1, 1, 1, 1, 0, 0))
model.fix(nodes[n], (0, 1, 1, 1, 0, 0))
model.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
sv = Solver(model)

f, Phi = sv.modes(8)
vert = [k for k in range(8) if np.abs(Phi[2::6, k]).max() > np.abs(Phi[1::6, k]).max()]
f1, f2 = f[vert[0]], f[vert[1]]
mu1, mu2 = jtg.impact_factor(f1), jtg.impact_factor(f2)

secs = chain_sections(beams)
il = InfluenceLines(sv, nodes, secs)
x = np.array([xs[b + (e == "j")] for b, e in secs])
live = lambda mu: jtg.LiveLoad(L0=L, lanes=1, transverse=1.0, distribution=0.55, mu=mu)  # noqa: E731

# permanent loads: Section.mass × g split into the girder and the superimposed dead load (the sum is the one case of
# jtg_rc_girder.py)
self_w = LoadCase(T("结构自重", "self-weight")).self_weight(
    m_girder / mass, label=T(f"主梁 {m_girder * 9.81:.2f} kN/m", f"girder {m_girder * 9.81:.2f} kN/m"))
sdl = LoadCase(T("二期恒载", "superimposed dead load")).line(
    beams, m_sdl * 9.81, label=T(f"桥面铺装及栏杆 {m_sdl * 9.81:.2f} kN/m", f"pavement and railing {m_sdl * 9.81:.2f} kN/m"))
dead = self_w + sdl
dead.name = T("永久作用（自重 + 二期恒载）", "permanent (self-weight + superimposed)")
G = dead.solve(sv)
G_M, G_V = G.section_forces(secs, "My"), G.section_forces(secs, "Vz")
Qs, Qh = live(mu1).effects(il, "My"), live(mu2).effects(il, "My")
Q_M = (Qs[0], Qh[1])
Q_V = live(mu1).effects(il, "Vz")
Mmax, Mmin = jtg.basic_combination(G_M, Q_M)
Vmax, Vmin = jtg.basic_combination(G_V, Q_V)

k_span, k_pier = int(np.argmax(Mmax)), int(np.argmin(Mmin))
k_end = int(np.argmin(np.abs(x - span_sec.h / 2000)))
k_pierV = int(np.argmax(np.maximum(np.abs(Vmax), np.abs(Vmin))))
near_pier = lambda k: abs(x[k] - L) < 0.3 * L  # noqa: E731

# the pier moment reduced for the bearing (4.3.5)
kL, kR = 2 * (n - 1) + 1, 2 * n
R_G = abs(G_V[kL] - G_V[kR])
pier_lane = live(mu2).layouts(il, "My", k_pier)[1]
R_Q = float(pier_lane.load_case(il).solve(sv).reactions()[6 * nodes[n] + 2])
R_d = 1.2 * R_G + 1.4 * R_Q
y_c = (pier_sec.h - pier_sec.zg) / 1e3
a_sp = jtg.spread_length(0.45, y_c)
Me, red = jtg.support_moment_reduction(Mmin[k_pier], R_d, a_sp)
Mmin_r = jtg.peak_shaving(x, Mmin, L, R_d, a_sp)

# shear along the girder
Vd = np.maximum(np.abs(Vmax), np.abs(Vmin))
Md = np.where(np.abs(Mmin_r) > np.abs(Mmax), Mmin_r, Mmax)
zones = [(0.0, 0.7 * L, span_sec), (0.7 * L, 1.3 * L, pier_sec), (1.3 * L, 2 * L, span_sec)]
stirrups = [(0.0, 3.0, stirrups_end), (3.0, 0.7 * L, stirrups_span), (0.7 * L, 1.3 * L, stirrups_pier),
            (1.3 * L, 2 * L - 3.0, stirrups_span), (2 * L - 3.0, 2 * L, stirrups_end)]
supports = [(0.0, "end"), (L, "interior", 0.4), (2 * L, "end")]
search = jtg.shear_search(x, Vd, Md, zones, stirrups, supports=supports, gamma0=gamma0)
half = x <= 0.5 * L + 1e-9
Vd_h2 = float(np.interp(span_sec.h / 2e3, x[half], Vd[half]))
st_design = jtg.stirrup_design(span_sec, Vd_h2, legs=2, dia=10, xi=0.6, gamma0=gamma0)
bent = jtg.bent_bar_design(x[half], Vd[half], support=0.0, h=span_sec.h, z=span_sec.h - 110 - 50, xi=0.6,
                           gamma0=gamma0)

# serviceability
Q0 = live(0.0).effects(il, "My")
Ms_max, Ms_min = jtg.frequent_combination(G_M, Q0)
Ml_max, Ml_min = jtg.quasi_permanent_combination(G_M, Q0)
ks, kp = int(np.argmax(Ms_max)), int(np.argmin(Ms_min))
sec_at = lambda u: pier_sec if abs(u - L) < 0.3 * L else span_sec  # noqa: E731
w_sag = np.full(len(x), np.nan)
w_hog = np.full(len(x), np.nan)
for k in range(len(x)):
    if Ms_max[k] > 1.0:
        w_sag[k] = jtg.crack_width(sec_at(x[k]), Ms_max[k], Ml_max[k], env="I").demand
    if Ms_min[k] < -1.0:
        w_hog[k] = jtg.crack_width(sec_at(x[k]), Ms_min[k], Ml_min[k], env="I").demand
w_lim = jtg.crack_width_limit("I")

m2 = Model()
nodes2 = [m2.node((xx, 0, 0)) for xx in xs]
for e in range(len(beams)):
    Mse = max((Ms_max[j] if abs(Ms_max[j]) >= abs(Ms_min[j]) else Ms_min[j] for j in (2 * e, 2 * e + 1)), key=abs)
    B = jtg.stiffness(pier_sec if near_pier(2 * e) else span_sec, Mse)[0]
    m2.beam(nodes2[e], nodes2[e + 1], Section(E=c.Ec * 1e3, A=area, Iy=B * 1e-9 / (c.Ec * 1e3), Iz=10 * I, J=0.01,
                                              mass=mass))
m2.fix(nodes2[0], (1, 1, 1, 1, 0, 0))
m2.fix(nodes2[n], (0, 1, 1, 1, 0, 0))
m2.fix(nodes2[-1], (0, 1, 1, 1, 0, 0))
sv2 = Solver(m2)
fG = -dead.solve(sv2).U[2::6] * 1e3
D = np.zeros((len(xs), len(xs)))
for j in range(len(xs)):
    F = np.zeros(m2.ndof)
    F[6 * j + 2] = -1.0
    D[:, j] = -sv2.solve(F)[2::6] * 1e3
q_k, P_k = jtg.lane_load(L)
dxs = np.diff(xs)
pos = np.maximum(D, 0.0)
fQ = 0.7 * live(0.0).factor() * np.array([q_k * float(((r[:-1] + r[1:]) / 2 * dxs).sum()) + P_k * r.max() for r in pos])
eta_t = jtg.long_term_factor(c)
f_long = eta_t * fQ
# pre-camber at the largest deflection of each span (the spans alike but for rounding)
i_spans = [int(np.argmax(fQ[: n + 1])), n + int(np.argmax(fQ[n:]))]
cambers = [(i, eta_t * fG[i], eta_t * fQ[i], jtg.precamber_rc(eta_t * fG[i], eta_t * fQ[i], L * 1e3)) for i in i_spans]
i_max, fG_l, fQ_l, camber = max(cambers, key=lambda t: t[3])

# ── the report ──────────────────────────────────────────────────────────────
W = 680  # px, the width of the drawings
rep = Report(T("两跨连续钢筋混凝土 T 梁计算书", "Two-span continuous reinforced-concrete T girder: calculation report"),
             project=T("示例公路桥（2 × 16 m 连续 T 梁）", "Example highway bridge (2 × 16 m continuous T girders)"),
             member=T("主梁（中梁）", "main girder (inner)"), number="FM-JTG-RC-001",
             author=T("（编制人）", "(author)"), checker=T("（校核人）", "(checker)"), approver="",
             date="2026-10-12", version="A", organisation=T("femotion 示例", "femotion example"),
             subtitle=T("持久状况承载能力极限状态和正常使用极限状态验算", "Ultimate and serviceability limit states"),
             lang=lang)
kw = dict(lang=lang, width=W)
loc = lambda u, what="": (f"x = {u:.2f} m" + (f"（{what}）" if what and zh else f" ({what})" if what else ""))  # noqa

# 1 basis
ch = rep.chapter(T("设计依据", "Basis of design"))
s = ch.section(T("规范", "Codes"))
jr.design_basis(s)
s.paragraph(T("式号未注明规范的为 JTG 3362—2018 的公式，注“D60”的为 JTG D60—2015 的公式；括号内带“条”字的为条文号。",
              "Formula numbers are those of JTG 3362-2018 unless marked D60 (JTG D60-2015); \"cl.\" marks a clause."),
            cls="muted")
s = ch.section(T("设计标准", "Design parameters"))
s.kv([(T("设计安全等级", "design safety class"), T("二级", "2"), ""),
      (T("结构重要性系数", "importance factor") + " $γ_{0}$", f"{gamma0:.1f}", ""),
      (T("汽车荷载", "traffic load"), T("公路—Ⅰ级（车道荷载）", "Class I (lane load)"), ""),
      (T("设计使用年限", "design working life"), 100, T("年", "years")),
      (T("环境类别", "environment"), T("Ⅰ类（一般环境）", "I (general)"), ""),
      (T("裂缝宽度限值", "crack width limit"), w_lim, "mm")], cols=2)
s.paragraph(T("本计算书为示例：梁、荷载和配筋取自 femotion 的示例 examples/jtg_rc_girder.py，数值仅作说明之用。",
              "An example: the girder, loads and reinforcement of femotion's examples/jtg_rc_girder.py, for "
              "illustration only."), cls="muted")

# 2 design data
ch = rep.chapter(T("设计资料", "Design data"), page_break=False)
s = ch.section(T("结构概况", "The structure"))
s.paragraph(T(f"上部结构为两跨等跨连续 T 梁，跨径 2 × {L:g} m，横向五片主梁，本计算书验算其中一片中梁。主梁梁高 1100 mm，腹板宽"
              f" 200 mm，翼缘有效宽度 1600 mm、厚 160 mm；中支点设宽 400 mm 的横隔梁，支座长 450 mm。汽车荷载横向分布系数取 0.55"
              f"（已计入车道数），桥面铺装及栏杆分摊到本梁 {m_sdl:g} t/m。",
              f"Two equal continuous spans of {L:g} m, five T girders across, one inner girder checked here: depth "
              f"1100 mm, web 200 mm, effective flange 1600 × 160 mm; a 400 mm diaphragm over the pier, 450 mm"
              f" bearings. "
              f"Lateral distribution factor 0.55 (lanes counted), pavement and railing {m_sdl:g} t/m on this girder."))
s = ch.section(T("材料", "Materials"))
jr.materials(s, [span_sec])
s = ch.section(T("截面", "Sections"))
s.figure([section_figure(span_sec, lang=lang, title="", width=330, stirrup_d=12),
          section_figure(pier_sec, lang=lang, title="", width=330, stirrup_d=10)],
         caption=T("主梁截面配筋", "Girder sections"), id="fig:sections",
         subcaptions=[T(f"跨中截面（0 ～ {0.7 * L:g} m、{1.3 * L:g} ～ {2 * L:g} m）",
                        f"span section (0–{0.7 * L:g} m, {1.3 * L:g}–{2 * L:g} m)"),
                      T(f"中支点截面（{0.7 * L:g} ～ {1.3 * L:g} m）", f"pier section ({0.7 * L:g}–{1.3 * L:g} m)")])
jr.section_properties(s, {T("跨中截面", "span"): span_sec, T("中支点截面", "pier"): pier_sec},
                      hogging=(T("中支点截面", "pier"),))
s.paragraph(T("箍筋：端支点附近 0 ～ 3 m 为 2 肢 Φ12@100，跨中段 2 肢 Φ10@200，中支点两侧各 0.3L 范围内 4 肢 Φ10@100。",
              "Stirrups: 2 legs Φ12 @ 100 over 0–3 m at the end supports, 2 legs Φ10 @ 200 in the spans, 4 legs "
              "Φ10 @ 100 within 0.3 L each side of the pier."))
s = ch.section(T("作用", "Actions"), id="sec:actions")
jr.load_cases(s, [self_w, sdl], model=model)
jr.live_load(s, live(mu1), mu=[(T("冲击系数（正弯矩、剪力）", "impact factor (sagging, shear)"), mu1),
                               (T("冲击系数（负弯矩）", "impact factor (hogging)"), mu2)],
             notes=[T("冲击系数由结构基频按 JTG D60—2015 4.3.2 计算，见 ", "Impact factors from the frequencies (D60 "
                      "4.3.2), see ") + rep.ref("sec:freq") + ("。" if zh else ".")])
jr.combinations(s, gamma0=gamma0)

# 3 the model
ch = rep.chapter(T("计算模型", "Model"), page_break=False)
s = ch.section(T("结构离散", "Discretisation"))
s.paragraph(T(f"按平面杆系计算，主梁离散为 {len(beams)} 个梁单元（单元长 {L / n:g} m），节点 {len(nodes)} 个。左端支点为固定铰支座，"
              f"中支点和右端支点为活动铰支座（见", f"A plane frame of {len(beams)} beam elements ({L / n:g} m) and "
              f"{len(nodes)} nodes; a fixed pin at the left end, rollers at the pier and the right end (see "),
            rep.ref("fig:model"), T("）。截面刚度取毛截面 $E_{c}I$，自重和二期恒载按一致荷载精确计入单元端。",
                                   "). Gross stiffness EcI; permanent loads exact at the element ends."))
s.figure(elevation_figure(model, title="", **kw), caption=T("计算模型", "The model"), id="fig:model")
s.kv([("$A$", area, "m²"), ("$I$", I, "m⁴"), ("$E_{c}$", c.Ec, "MPa"), (T("每延米质量", "mass per metre"), mass, "t/m")],
     cols=2)
s.paragraph(T("注：JTG 3362—2018 4.3.1 规定钢筋混凝土构件内力分析取 $0.8E_{c}I$；本梁为同一截面、同一材料，刚度同比例折减不改变荷载产生的内力，"
              "故取 $E_{c}I$（基频计算按 JTG D60—2015 4.3.2 亦取 $E_{c}I$）。",
              "JTG 3362 4.3.1 asks for 0.8 EcI of reinforced concrete members; on a girder of one section and one"
              " material "
              "this scales all stiffness alike and leaves the forces from loads unchanged, so EcI is used (also for"
              " the "
              "frequencies, D60 4.3.2)."), cls="muted")
s = ch.section(T("动力特性与冲击系数", "Frequencies and impact factors"), id="sec:freq")
s.formula([("μ", "0.1767\\op{ln}\\,f - 0.0157"),
           ("μ_{1}", f"0.1767 × \\op{{ln}}\\,{fixed(f1, 3)} - 0.0157 = {fixed(mu1, 3)}"),
           ("μ_{2}", f"0.1767 × \\op{{ln}}\\,{fixed(f2, 3)} - 0.0157 = {fixed(mu2, 3)}")], tag="D60 4.3.2")
s.paragraph(T(f"结构第一、二阶竖弯频率 f₁ = {f1:.3f} Hz、f₂ = {f2:.3f} Hz（有限元特征值分析）；按 JTG D60—2015 条文说明，正弯矩和剪力"
              f"用 f₁，负弯矩用 f₂。", f"First two vertical frequencies f1 = {f1:.3f} Hz, f2 = {f2:.3f} Hz (eigenvalue "
              "analysis); f1 for sagging moment and shear, f2 for hogging moment (D60 commentary)."))

# 4 actions and effects
ch = rep.chapter(T("作用和作用效应", "Actions and their effects"))
s = ch.section(T("永久作用", "Permanent actions"))
s.figure(load_figure(model, self_w, title="", **kw), caption=T("结构自重", "Self-weight"), id="fig:self")
s.figure(load_figure(model, sdl, title="", **kw), caption=T("二期恒载", "Superimposed dead load"), id="fig:sdl")
s = ch.section(T("汽车荷载", "Traffic"))
s.paragraph(T("汽车荷载（车道荷载）按影响线最不利布置：均布荷载布满影响线同号区段，集中荷载作用于影响线峰值处（计算剪力时集中荷载乘 1.2）。"
              f"图中车道荷载已乘冲击系数和横向分布系数：正弯矩和剪力为 $(1 + μ_{{1}})m$ = (1 + {mu1:.3f}) × 0.55 = "
              f"{(1 + mu1) * 0.55:.4f}，负弯矩为 $(1 + μ_{{2}})m$ = (1 + {mu2:.3f}) × 0.55 = {(1 + mu2) * 0.55:.4f}。",
              "The lane load in its worst placement on the influence lines: q over the parts of one sign, P at the"
              " peak (× 1.2 for shear). The loads drawn include impact and distribution: $(1 + μ_{1})m$ = "
              f"(1 + {mu1:.3f}) × 0.55 = {(1 + mu1) * 0.55:.4f} for sagging moment and shear, $(1 + μ_{{2}})m$ = "
              f"(1 + {mu2:.3f}) × 0.55 = {(1 + mu2) * 0.55:.4f} for hogging."))
k_end0 = 0
v_sense = 1 if abs(Q_V[1][k_end0]) > abs(Q_V[0][k_end0]) else 0
for lay, cap, uid in [
        (live(mu1).layouts(il, "My", k_span)[0], T(f"跨中（x = {x[k_span]:g} m）最大正弯矩的车道荷载布置",
                                                   f"Lane load for the largest sagging moment at x = {x[k_span]:g} m"),
         "fig:il-span"),
        (pier_lane, T("中支点最大负弯矩的车道荷载布置", "Lane load for the largest hogging moment at the pier"),
         "fig:il-pier"),
        (live(mu1).layouts(il, "Vz", k_end0)[v_sense], T("端支点最大剪力的车道荷载布置",
                                                         "Lane load for the largest shear at the end support"),
         "fig:il-end")]:
    s.figure(influence_figure(il, lay, title="", **kw), caption=cap, id=uid)
s = ch.section(T("作用效应组合", "Combined effects"))
at_x = lambda u: int(np.argmin(np.abs(x - u)))  # noqa: E731  (the first section at x: the j end of the element before)
key_k = [0, at_x(0.25 * L), at_x(0.4 * L), at_x(0.75 * L), 2 * n - 1, 2 * n, k_span]
rows = [[loc(x[k], T("中支点左", "left of pier") if k == 2 * n - 1 else (T("中支点右", "right of pier")
                                                                        if k == 2 * n else "")),
         G_M[k], Q_M[0][k], Q_M[1][k], Mmax[k], Mmin[k], Vmax[k], Vmin[k]] for k in key_k]
s.table(rows, header=[T("截面", "Section"), "M_{G}", "M_{Q,max}", "M_{Q,min}", "M_{d,max}", "M_{d,min}", "V_{d,max}",
                      "V_{d,min}"], units=["", "kN·m", "kN·m", "kN·m", "kN·m", "kN·m", "kN", "kN"], digits=1,
        caption=T("控制截面作用效应（基本组合，未计 γ0）", "Effects at the governing sections (basic combination, without γ0)"),
        id="tab:effects", notes=[T("汽车荷载效应已计冲击系数和横向分布系数；M、V 正负号同杆系计算结果（正弯矩下缘受拉）。",
                                   "Traffic effects with impact and distribution; signs as the frame results (sagging "
                                   "+).")])
s.figure(charts.moment_chart(x, Mmax, Mmin, gamma0=gamma0, shaved=Mmin_r, supports=supports, title="", **kw),
         caption=T("基本组合弯矩包络图（$γ_{0}M_{d}$）", "Moment envelope of the basic combination ($γ_{0}M_{d}$)"), id="fig:menv")
s.figure(charts.shear_chart(x, Vmax, Vmin, gamma0=gamma0, supports=supports, title="", **kw),
         caption=T("基本组合剪力包络图（$γ_{0}V_{d}$）", "Shear envelope of the basic combination ($γ_{0}V_{d}$)"), id="fig:venv")
s.paragraph(T("正常使用极限状态：汽车荷载不计冲击，频遇组合 $ψ_{f1}$ = 0.7、准永久组合 $ψ_{q}$ = 0.4（JTG D60—2015 4.1.6）。",
              "Serviceability: traffic without impact, frequent ψf1 = 0.7, quasi-permanent ψq = 0.4 (D60 4.1.6)."))
rows = [[loc(x[k]), Ms_max[k], Ms_min[k], Ml_max[k], Ml_min[k]] for k in (ks, kp)]
s.table(rows, header=[T("截面", "Section"), "M_{s,max}", "M_{s,min}", "M_{l,max}", "M_{l,min}"],
        units=["", "kN·m", "kN·m", "kN·m", "kN·m"], digits=1,
        caption=T("频遇组合和准永久组合弯矩", "Frequent and quasi-permanent moments"), id="tab:sls")

# 5 ULS
ch = rep.chapter(T("持久状况承载能力极限状态", "Ultimate limit state"))
s = ch.section(T("正截面抗弯", "Flexure"))
s.paragraph(T("纵向受拉钢筋：跨中段（0 ～ 0.7L、1.3L ～ 2L）用跨中截面配筋，中支点两侧各 0.3L 范围用中支点截面配筋。抵抗弯矩图与弯矩包络图见",
              "Span reinforcement over 0–0.7 L and 1.3 L–2 L, pier reinforcement within 0.3 L each side of the pier; "
              "resisting moments against the envelope in "), rep.ref("fig:mres"), "。" if zh else ".")
Mu = charts.flexural_resistance(zones)
fm = charts.moment_chart(x, Mmax, Mmin, gamma0=gamma0, resistance=Mu, shaved=Mmin_r, supports=supports, title="", **kw)
s.figure(fm, caption=T("弯矩包络图与抵抗弯矩图", "Moment envelope and resisting moments"), id="fig:mres")
ss = s.section(T("中支点负弯矩的削峰（4.3.5）", "The pier moment reduced for the bearing (4.3.5)"))
ss.paragraph(T("与最大负弯矩同一荷载布置下的中支点反力：自重取支点两侧剪力之差，汽车荷载按该最不利布置作为荷载工况求解。",
               "The pier reaction of the same placement: self-weight from the jump of its shear, the lane load solved"
               " in "
               "its hogging placement."))
ss.formula([("R", f"1.2R_{{G}} + 1.4R_{{Q}} = 1.2 × {fixed(R_G, 1)} + 1.4 × {fixed(R_Q, 1)} ="
                  f" {fixed(R_d, 1)}\\,\\u{{kN}}"),
            ("a", f"b + 2y = 0.45 + 2 × {fixed(y_c, 3)} = {fixed(a_sp, 3)}\\,\\u{{m}}"),
            ("M'", "=", f"\\frac{{qa^{{2}}}}{{8}} = \\frac{{Ra}}{{8}} = \\frac{{{fixed(R_d, 1)} ×"
                        f" {fixed(a_sp, 3)}}}{{8}} = "
                        f"{fixed(red['M_prime'], 1)}\\,\\u{{kN·m}}", "4.3.5-2")])
ss.formula([("M - M'", "=", f"-({fixed(abs(Mmin[k_pier]), 1)} - {fixed(red['M_prime'], 1)}) = "
                             f"{fixed(Mmin[k_pier] + red['M_prime'], 1)}\\,\\u{{kN·m}}", "4.3.5-1"),
            ("0.9M", f"0.9 × ({fixed(Mmin[k_pier], 1)}) = {fixed(0.9 * Mmin[k_pier], 1)}\\,\\u{{kN·m}}"),
            ("M_{e}", f"{fixed(Me, 1)}\\,\\u{{kN·m}}" + (T("\\t{（削减量不超过 10%，取 0.9M）}", "\\t{ (at most 10 %: 0.9M)}")
                                                     if red["capped"] else ""))])
jr.write_check(s, jtg.flexure(span_sec, Mmax[k_span], gamma0), where=loc(x[k_span], T("跨中最大正弯矩", "largest sagging")),
               id="chk:flex-span", figure_kw={"stirrup_d": 12})
jr.write_check(s, jtg.flexure(pier_sec, Mmin_r[k_pier], gamma0),
               where=loc(x[k_pier], T("中支点，削峰后", "pier, reduced")), id="chk:flex-pier",
               figure_kw={"stirrup_d": 10})
crit = fm.critical
s.paragraph(T(f"沿梁各截面受弯利用率最大值 {crit['ratio']:.3f}，位于 x = {crit['x']:.2f} m。",
              f"Largest flexure ratio along the girder {crit['ratio']:.3f} at x = {crit['x']:.2f} m."))

s = ch.section(T("斜截面抗剪", "Shear"))
s.paragraph(T("按 5.2.8 条确定验算位置（端支点 h/2 处、箍筋变化处、中支点横隔梁边缘），按 5.2.10 条取斜截面水平投影长度，并在网格上搜索最不利斜截面；"
              "端支点另按 5.2.11 条验算截面尺寸。结果见", "Positions by 5.2.8 (h/2 from the end supports, stirrup changes, "
              "diaphragm edges), projections by 5.2.10, and a search for the governing inclined section; the section "
              "limit of 5.2.11 at the end supports. See "), rep.ref("fig:shear"), T("和", " and "),
            rep.ref("tab:inclined"), "。" if zh else ".")
Vu = charts.shear_resistance(zones, stirrups, supports=supports, x=x, Md=Md)
el = charts.beam_elevation(0.0, 2 * L, zones, search=search, stirrups=stirrups, supports=supports, title="", lang=lang,
                           width=W)
fv = charts.shear_chart(x, Vmax, Vmin, gamma0=gamma0, resistance=Vu, search=search, supports=supports, title="", **kw)
s.figure(viz.stack([el, fv]), caption=T("斜截面位置与剪力包络、抗剪承载力", "Inclined sections, shear envelope and resistance"),
         id="fig:shear")
jr.shear_search_table(s, search)
gov = search.governing
jr.write_check(s, gov.check, where=T(f"控制斜截面 x_{{c}} = {gov.x_c:.3f} m", f"governing inclined section x_{{c}} ="
                                                                         f" {gov.x_c:.3f} m"),
               id="chk:shear-gov")
jr.write_check(s, jtg.shear(span_sec, max(abs(Vmax[k_end]), abs(Vmin[k_end])), stirrups_end, gamma0=gamma0),
               where=loc(x[k_end], T("端支点 h/2 处", "h/2 from the end support")), id="chk:shear-end")
jr.write_check(s, jtg.shear(pier_sec, max(abs(Vmax[k_pierV]), abs(Vmin[k_pierV])), stirrups_pier, gamma0=gamma0,
                            hogging=near_pier(k_pierV), alpha1=0.9 if near_pier(k_pierV) else 1.0),
               where=loc(x[k_pierV], T("中支点", "pier")), id="chk:shear-pier")
for lim in search.limits[:1]:
    jr.write_check(s, lim, id="chk:limit-end")
jr.stirrup_design_block(s, st_design, Vd_h2, where=T("端支点 h/2 处", "h/2 from the end support"))
ss = s.section(T("弯起钢筋设计（5.2.13）", "Bent-up bars (5.2.13)"))
jr.bent_bar_table(ss, bent)

# 6 SLS
ch = rep.chapter(T("持久状况正常使用极限状态", "Serviceability limit state"))
s = ch.section(T("裂缝宽度", "Crack width"))
fw = charts.crack_chart(x, [(T("正弯矩区", "sagging"), w_sag), (T("负弯矩区", "hogging"), w_hog)],
                        limit=[(w_lim, T("限值（Ⅰ类环境）", "limit (class I)"))], supports=supports, title="", **kw)
s.figure(fw, caption=T("频遇组合下的最大裂缝宽度", "Largest crack width, frequent combination"), id="fig:crack")
jr.write_check(s, jtg.crack_width(span_sec, Ms_max[ks], Ml_max[ks], env="I"), where=loc(x[ks], T("跨中", "span")),
               id="chk:crack-span")
jr.write_check(s, jtg.crack_width(pier_sec, Ms_min[kp], Ml_min[kp], env="I"), where=loc(x[kp], T("中支点", "pier")),
               id="chk:crack-pier")
s = ch.section(T("挠度和预拱度", "Deflection and pre-camber"))
s.paragraph(T("各单元刚度按其频遇组合弯矩取用（6.5.2：小于开裂弯矩取 $B_{0}$，大于时取开裂后的等效刚度），按结构力学方法求挠度；汽车荷载频遇值（$ψ_{f1}$ = 0.7，"
              f"不计冲击）的挠度乘以长期增长系数 $η_{{θ}}$ = {eta_t:.2f}（6.5.3）。",
              "The stiffness of each element from its frequent moment (6.5.2), deflections by structural mechanics;"
              " the "
              f"frequent live-load deflection (ψf1 = 0.7, no impact) × ηθ = {eta_t:.2f} (6.5.3)."))
f_lim = jtg.deflection_limit(L * 1e3)
ff = charts.deflection_chart(xs, f_long, limit=[(0, L, f_lim, "L/600"), (L, 2 * L, f_lim, "L/600")],
                             label=T("ηθ × 汽车荷载频遇值挠度", "ηθ × frequent live-load deflection"), supports=supports,
                             title="", **kw)
s.figure(ff, caption=T("汽车荷载（频遇值）长期挠度", "Long-term live-load deflection"), id="fig:defl")
f_max, x_max = next(iter(ff.extremes.values()))  # the largest value the chart labels
i_c = int(np.argmin(np.abs(xs - x_max)))
jr.write_check(s, jtg.deflection_check(f_long[i_c], L * 1e3), where=loc(xs[i_c]), id="chk:defl")
ss = s.section(T("预拱度（6.5.5）", "Pre-camber (6.5.5)"))
ss.formula([("f_{G} + f_{Q}", f"{fixed(fG_l, 2)} + {fixed(fQ_l, 2)} = {fixed(fG_l + fQ_l, 2)}\\,\\u{{mm}}"),
            ("", ">" if fG_l + fQ_l > L * 1e3 / 1600 else "<=",
             f"\\frac{{L}}{{1600}} = {fixed(L * 1e3 / 1600, 2)}\\,\\u{{mm}}"),
            ("Δ", f"f_{{G}} + \\frac{{f_{{Q}}}}{{2}} = {fixed(fG_l, 2)} + \\frac{{{fixed(fQ_l, 2)}}}{{2}} = "
                  f"{fixed(camber, 2)}\\,\\u{{mm}}")], tag=T("6.5.5 条", "cl. 6.5.5"))
ss.paragraph(T(f"上式为 x = {xs[i_max]:.2f} m 处（两跨中较大者）。各跨预拱度："
               + "；".join(f"x = {xs[i]:.2f} m 处 {cb:.2f} mm" for i, _, _, cb in cambers) + "，沿顺桥向做成平顺曲线。",
               f"The above at x = {xs[i_max]:.2f} m (the larger of the spans). Pre-camber: "
               + "; ".join(f"{cb:.2f} mm at x = {xs[i]:.2f} m" for i, _, _, cb in cambers)
               + ", as a smooth curve along the span."))

# 7 detailing
ch = rep.chapter(T("构造检查", "Detailing"))
ch.paragraph(T("构造要求不满足时只给出提示，不改变承载力和正常使用验算的结论；提示汇总见", "Unmet detailing rules are warnings and do "
               "not change the verdicts of the checks; all of them in "), rep.ref("tab:warnings"), "。" if zh else ".")
jr.write_check(ch, jtg.durability(span_sec, "I", life=100), where=T("主梁", "girder"))
ch.paragraph(T("构造检查按钢筋的实际排列：跨中截面的 10Φ28 为 5 层 2Φ28 焊接骨架（层间相贴），中支点截面的 6Φ28 为 3 层 2Φ28 "
               "焊接骨架；合力点仍分别距底面 110 mm、95 mm，与承载力计算所用的截面一致。",
               "The detailing rules take the bars as placed: the 10Φ28 of the span are welded frames of five layers of"
               " 2Φ28 in contact, the 6Φ28 of the pier three layers; their centroids stay at 110 and 95 mm, as in the"
               " resistance."))
jr.write_check(ch, jtg.section_detailing(span_frames, env="I", life=100, stirrup_d=12, welded_frame=True, main_rows=5),
               where=T("跨中截面", "span section"), figure_kw={"stirrup_d": 12})
jr.write_check(ch, jtg.section_detailing(pier_frames, env="I", life=100, stirrup_d=10, hogging=True, welded_frame=True,
                                         main_rows=(3, 1)),
               where=T("中支点截面", "pier section"), figure_kw={"stirrup_d": 10})
jr.write_check(ch, jtg.stirrup_detailing(span_sec, stirrups_span, hook=(50, 50, 135), support_spacing=100),
               where=T("跨中段箍筋", "stirrups in the span"))

# 8 summary
ch = rep.chapter(T("验算汇总和结论", "Summary and conclusion"))
fu = charts.utilization_chart([
    (T("受弯", "flexure"), x, charts.flexure_utilization(x, Mmax, Mmin_r, Mu, gamma0)),
    (T("抗剪（斜截面）", "shear (inclined sections)"), *charts.shear_utilization(search)),
    (T("裂缝宽度", "crack width"), x, np.fmax(np.nan_to_num(w_sag, nan=0), np.nan_to_num(w_hog, nan=0)) / w_lim),
    (T("挠度", "deflection"), xs, f_long / f_lim)], supports=supports, title="", **kw)
ch.figure(fu, caption=T("各项验算利用率沿梁分布", "Utilisation along the girder"), id="fig:util")
jr.summary(ch)

rep.save(opt.out)
print(f"written {opt.out}")
if opt.pdf:
    rep.pdf(opt.pdf, html_path=opt.out)
    print(f"printed {opt.pdf}, contents with page numbers ({len(rep.pages)} headings)")
