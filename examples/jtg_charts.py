"""Charts along a girder for a design calculation report (femotion.codes.jtg.charts), illustrative values.

Part 1, the reinforced-concrete T girder of examples/jtg_rc_girder.py (two spans of 16 m, C40, HRB400, Class I lane
load), rebuilt here: the girder's elevation with the inclined sections checked and the stirrup zones over the shear
envelope and the stirrup zones' resistance; the moment envelope with the resisting moment diagram of the span and pier
reinforcement and the pier moment before and after peak shaving (4.3.5); the utilisation of flexure, shear, crack width
and deflection; crack width and long-term deflection against their limits.
Part 2, the post-tensioned I girder of examples/jtg_pc_girder.py (two spans of 30 m, C50, class A), its prestress
rebuilt here without the temperature gradient: the top and bottom edge stresses in service (characteristic
combination, 0.5 fck, 7.1.5) and under the frequent combination (−0.7 ftk, 6.3.1).

    uv run python examples/jtg_charts.py [out.html] [--lang en] [--width 380]
"""
import argparse
import math

import numpy as np

from femotion import viz
from femotion.codes import jtg
from femotion.codes.jtg import charts
from femotion.codes.jtg import prestress as ps
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces
from femotion.frame.prestress import Tendon, effects

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="jtg_charts.html")
ap.add_argument("--lang", default="zh", choices=("zh", "en"))
ap.add_argument("--width", type=float, default=720.0)
opt = ap.parse_args()
out, lang, width = opt.out, opt.lang, opt.width
zh = lang == "zh"


def frame(xs, sec, n):
    m = Model()
    nodes = [m.node((x, 0, 0)) for x in xs]
    beams = [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
    m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
    m.fix(nodes[n], (0, 1, 1, 1, 0, 0))
    m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
    return Solver(m), nodes, beams


def impact(sv):
    f, Phi = sv.modes(8)
    vert = [k for k in range(8) if np.abs(Phi[2::6, k]).max() > np.abs(Phi[1::6, k]).max()]
    return jtg.impact_factor(f[vert[0]]), jtg.impact_factor(f[vert[1]])


# ── part 1: the reinforced-concrete girder ───────────────────────────────────
L, n = 16.0, 32
gamma0 = jtg.GAMMA_0[2]
span_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
pier_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(6, 28, 95)], top=[(14, 25, 60)])
c = span_sec.concrete
area, I = span_sec.area * 1e-6, span_sec.I * 1e-12  # noqa: E741
xs = np.linspace(0, 2 * L, 2 * n + 1)
sv, nodes, beams = frame(xs, Section(E=c.Ec * 1e3, A=area, Iy=I, Iz=10 * I, J=0.01, mass=area * 2.6 + 0.55), n)
mu1, mu2 = impact(sv)
secs = chain_sections(beams)
il = InfluenceLines(sv, nodes, secs)
live = lambda mu: jtg.LiveLoad(L0=L, lanes=1, transverse=1.0, distribution=0.55, mu=mu)  # noqa: E731
G_M, G_V = uniform_load_forces(sv, secs, "My"), uniform_load_forces(sv, secs, "Vz")
Q_M = (live(mu1).effects(il, "My")[0], live(mu2).effects(il, "My")[1])
Mmax, Mmin = jtg.basic_combination(G_M, Q_M)
Vmax, Vmin = jtg.basic_combination(G_V, live(mu1).effects(il, "Vz"))
x = np.array([xs[b + (e == "j")] for b, e in secs])

# the pier moment reduced for the bearing width (4.3.5), as in jtg_rc_girder.py
k_pier = int(np.argmin(Mmin))
kL, kR = 2 * (n - 1) + 1, 2 * n
R_G = abs(G_V[kL] - G_V[kR])
R_Q = float(live(mu2).layouts(il, "My", k_pier)[1].load_case(il).solve(sv).reactions()[6 * nodes[n] + 2])
R_d = 1.2 * R_G + 1.4 * R_Q
a = jtg.spread_length(0.45, (pier_sec.h - pier_sec.zg) / 1e3)
Mmin_r = jtg.peak_shaving(x, Mmin, L, R_d, a)

# reinforcement zones (pier reinforcement over 0.3 L each side), stirrup zones, supports
zones = [(0.0, 0.7 * L, span_sec), (0.7 * L, 1.3 * L, pier_sec), (1.3 * L, 2 * L, span_sec)]
stirrups = [(0.0, 3.0, (2, 12, 100)), (3.0, 0.7 * L, (2, 10, 200)), (0.7 * L, 1.3 * L, (4, 10, 100)),
            (1.3 * L, 2 * L - 3.0, (2, 10, 200)), (2 * L - 3.0, 2 * L, (2, 12, 100))]
supports = [(0.0, "end"), (L, "interior", 0.4), (2 * L, "end")]
Vd = np.maximum(np.abs(Vmax), np.abs(Vmin))
Md = np.where(np.abs(Mmin_r) > np.abs(Mmax), Mmin_r, Mmax)
search = jtg.shear_search(x, Vd, Md, zones, stirrups, supports=supports, gamma0=gamma0)
Mu = charts.flexural_resistance(zones)
Vu = charts.shear_resistance(zones, stirrups, supports=supports, x=x, Md=Md)

# serviceability: crack widths at every section (6.4), long-term live-load deflection at every node (6.5.3)
Q0 = live(0.0).effects(il, "My")
Ms_max, Ms_min = jtg.frequent_combination(G_M, Q0)
Ml_max, Ml_min = jtg.quasi_permanent_combination(G_M, Q0)
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
    Me = max((Ms_max[j] if abs(Ms_max[j]) >= abs(Ms_min[j]) else Ms_min[j] for j in (2 * e, 2 * e + 1)), key=abs)
    B = jtg.stiffness(sec_at(xs[e] + 0.25), Me)[0]
    m2.beam(nodes2[e], nodes2[e + 1], Section(E=c.Ec * 1e3, A=area, Iy=B * 1e-9 / (c.Ec * 1e3), Iz=10 * I, J=0.01,
                                              mass=area * 2.6 + 0.55))
m2.fix(nodes2[0], (1, 1, 1, 1, 0, 0))
m2.fix(nodes2[n], (0, 1, 1, 1, 0, 0))
m2.fix(nodes2[-1], (0, 1, 1, 1, 0, 0))
sv2 = Solver(m2)
D = np.zeros((len(xs), len(xs)))
for j in range(len(xs)):
    F = np.zeros(m2.ndof)
    F[6 * j + 2] = -1.0
    D[:, j] = -sv2.solve(F)[2::6] * 1e3
q, P = jtg.lane_load(L)
dxs = np.diff(xs)
pos = np.maximum(D, 0.0)
fQ = 0.7 * live(0.0).factor() * np.array([q * float(((r[:-1] + r[1:]) / 2 * dxs).sum()) + P * r.max() for r in pos])
f_long = jtg.long_term_factor(c) * fQ
f_lim = jtg.deflection_limit(L * 1e3)

# the charts: the elevation stacked over the shear chart, the moment chart, the utilisation, crack width, deflection
label = (lambda z, e: z if zh else e)  # noqa: E731
kw = dict(supports=supports, lang=lang, width=width)
el = charts.beam_elevation(0.0, 2 * L, zones, search=search, stirrups=stirrups, **kw)
fv = charts.shear_chart(x, Vmax, Vmin, gamma0=gamma0, resistance=Vu, search=search, **kw)
fm = charts.moment_chart(x, Mmax, Mmin, gamma0=gamma0, resistance=Mu, shaved=Mmin_r, **kw)
w_env = np.fmax(np.nan_to_num(w_sag, nan=-1), np.nan_to_num(w_hog, nan=-1))
w_env = np.where(w_env < 0, np.nan, w_env)
fu = charts.utilization_chart([
    (label("受弯", "flexure"), x, charts.flexure_utilization(x, Mmax, Mmin_r, Mu, gamma0)),
    (label("抗剪（斜截面）", "shear (inclined sections)"), *charts.shear_utilization(search)),
    (label("裂缝宽度", "crack width"), x, w_env / w_lim),
    (label("挠度", "deflection"), xs, f_long / f_lim),
], **kw)
fw = charts.crack_chart(x, [(label("正弯矩区", "sagging"), w_sag), (label("负弯矩区", "hogging"), w_hog)],
                        limit=[(w_lim, label("限值（Ⅰ类环境）", "limit (class I)"))], xlabel=False, **kw)
ff = charts.deflection_chart(xs, f_long, limit=[(0, L, f_lim, "L/600"), (L, 2 * L, f_lim, "L/600")],
                             label=label("ηθ × 频遇汽车荷载挠度", "ηθ × frequent live-load deflection"), **kw)

# ── part 2: the prestressed girder's edge stresses (as examples/jtg_pc_girder.py, without the temperature gradient) ─
L2, n2 = 30.0, 60
geo = dict(b=200, h=2000, bf=2200, hf=180, bf_bottom=800, hf_bottom=300, concrete="C50", steel="HRB400")
bars = dict(bottom=[(10, 20, 60)], top=[(16, 16, 50)])
strand = jtg.tendon("strand", 1860)
n_t, Ap = 4, 4 * 9 * 139.0
duct = math.sqrt(n_t) * 90.0
gross = jtg.tee(**geo, **bars)
c2 = gross.concrete
aEP = strand.Ep / c2.Ec
xs2 = np.linspace(0, 2 * L2, 2 * n2 + 1)
xi = np.minimum(xs2, 2 * L2 - xs2) / L2
e = xi * (-0.6) + 4 * 1.0 * xi * (1 - xi)
slope = np.gradient(e, xs2)
theta = np.abs(np.r_[0, np.cumsum(np.abs(np.diff(np.arctan(slope))))])
theta = np.where(xs2 <= L2, theta, theta[::-1])
dist = np.minimum(xs2, 2 * L2 - xs2)
s_con = strand.sigma_con_max()
kf, muf = ps.DUCT_FRICTION["metal corrugated"][0], 0.25
l1 = np.array([ps.friction_loss(s_con, muf, th, kf, d) for th, d in zip(theta, dist)])
l2, _ = ps.anchorage_loss_reverse(s_con, s_con - l1[n2], L2 * 1000, ps.ANCHOR_SET["wedge"], strand.Ep, dist * 1000)
An, zn, In = gross.replace(tendons=np.array([[Ap, gross.zg, 0, math.pi * duct ** 2 / 4]]), tendon=strand).props("net")
dp = gross.zg + e * 1000
ep_n = dp - zn
d_pc = ps.concrete_stress((s_con - l1 - l2) * Ap / n_t, ep_n, ep_n, An, In)
l4 = np.array([ps.elastic_shortening_simplified(n_t, aEP, s) for s in d_pc])
lI = l1 + l2 + l4
l5 = np.array([ps.relaxation_loss(strand, s_con - v) for v in lI])
h_not = 2 * gross.area / (2 * geo["h"] + 2 * geo["bf"])
eps_cs, phi = ps.shrinkage_strain(c2, 75, h_not, t=3650, t0=7), ps.creep_coefficient(c2, 75, h_not, t=3650, t0=7)
bot, _ = gross.edge_masks()
As_b, es_b = gross.bars[bot, 0].sum(), gross.bars[bot, 1].mean() - zn
s_pcI = ps.concrete_stress((s_con - lI) * Ap, ep_n, ep_n, An, In)
l6 = np.array([ps.creep_shrinkage_loss(strand.Ep, eps_cs, aEP, s_pcI[i], phi,
                                       *(ps.steel_ratios(An, In, Ap, ep_n[i], As_b, es_b) if ep_n[i] > 0 else
                                         ps.steel_ratios(An, In, Ap, -ep_n[i])), fcu_transfer=0.9 * c2.fcuk)
               for i in range(len(xs2))])
s_pe = s_con - lI - l5 - l6

area2, I2 = gross.area * 1e-6, gross.I * 1e-12
sv_p, nodes_p, beams_p = frame(xs2, Section(E=c2.Ec * 1e3, A=area2, Iy=I2, Iz=10 * I2, J=0.05,
                                            mass=area2 * 2.6 + 1.2), n2)
secs2 = chain_sections(beams_p)
x2 = np.array([xs2[b + (end == "j")] for b, end in secs2])
node_of = np.array([b + (end == "j") for b, end in secs2])
M_sec = effects(sv_p, Tendon(beams_p, e, (s_pe[:-1] + s_pe[1:]) / 2 * Ap / 1e3), secs2, "My")[2]
mu1p, mu2p = impact(sv_p)
il2 = InfluenceLines(sv_p, nodes_p, secs2)
live2 = lambda mu: jtg.LiveLoad(L0=L2, lanes=1, transverse=1.0, distribution=0.5, mu=mu)  # noqa: E731
G2 = uniform_load_forces(sv_p, secs2, "My")
C2 = jtg.CrowdLoad(L0=L2, width=3.0, distribution=0.2).effects(il2, "My")
Mk = jtg.characteristic_combination(G2, (live2(mu1p).effects(il2, "My")[0], live2(mu2p).effects(il2, "My")[1]), [C2])
Ms = jtg.frequent_combination(G2, live2(0.0).effects(il2, "My"), [C2], kinds=["crowd"])


def girder_at(k):
    """the section at chart point k with its tendon (σp0 with the secondary moment); end blocks within 2 m of the
    supports"""
    i = node_of[k]
    s_pc = ps.concrete_stress(s_pe[i] * Ap, ep_n[i], ep_n[i], An, In, M=M_sec[k] * 1e6)
    sp0 = ps.sigma_p0(s_con, lI[i] + l5[i] + l6[i], post=True, alpha_EP=aEP, sigma_pc=s_pc)
    end = min(abs(xs2[i]), abs(xs2[i] - L2), abs(xs2[i] - 2 * L2)) <= 2.0
    g = dict(geo, b=550) if end else geo
    return jtg.tee(**g, **bars, tendons=[(Ap, dp[i], sp0, duct)], tendon=strand, sigma_pe=s_pe[i], sigma_l6=l6[i])


# edge stresses, compression +: characteristic (7.1.3, prestress after all losses with Mp2) and frequent (6.3.1:
# σpc − σst at each edge)
char = {"top": np.zeros((2, len(x2))), "bottom": np.zeros((2, len(x2)))}
freq = {"top": np.zeros((2, len(x2))), "bottom": np.zeros((2, len(x2)))}
for k in range(len(x2)):
    s = girder_at(k)
    for j in (0, 1):
        d = jtg.service_stresses(s, Mk[j][k], pc_class="A", M_p2=M_sec[k]).details["edge_stress"]
        r = jtg.crack_resistance(s, Ms[j][k], Ms[j][k], pc_class="A", M_p2=M_sec[k]).details
        for edge in ("top", "bottom"):
            char[edge][j, k] = d[edge]
            freq[edge][j, k] = r[edge]["sigma_pc"] - r[edge]["sigma_st"]
env = lambda a: (a.max(axis=0), a.min(axis=0))  # noqa: E731
kw2 = dict(supports=[0.0, L2, 2 * L2], lang=lang, width=width)
fs1 = charts.stress_chart(x2, top=env(char["top"]), bottom=env(char["bottom"]), limits=[(0.5 * c2.fck, "0.5fck")],
                          xlabel=False, title=label("标准值组合：上、下缘正应力（7.1.5）",
                                                    "Characteristic combination: edge stresses (7.1.5)"), **kw2)
fs2 = charts.stress_chart(x2, top=env(freq["top"]), bottom=env(freq["bottom"]), limits=[(-0.7 * c2.ftk, "−0.7ftk")],
                          title=label("频遇组合：上、下缘正应力 σpc − σst（6.3.1，A 类）",
                                      "Frequent combination: edge stresses σpc − σst (6.3.1, class A)"), **kw2)

h = (lambda z, en: f"<h2>{z if zh else en}</h2>")  # noqa: E731
page = viz.html([
    h("一、钢筋混凝土连续 T 梁（2 × 16 m）", "1. Reinforced-concrete continuous T girder (2 × 16 m)"),
    h("抗剪：斜截面与剪力包络", "Shear: inclined sections and shear envelope"), charts.stack([el, fv]),
    h("受弯：弯矩包络与抵抗弯矩图", "Flexure: moment envelope and resisting moment"), fm,
    h("验算利用率", "Utilisation"), fu,
    h("正常使用：裂缝宽度与挠度", "Serviceability: crack width and deflection"), charts.stack([fw, ff]),
    h("二、预应力混凝土连续 I 梁（2 × 30 m，A 类）", "2. Prestressed continuous I girder (2 × 30 m, class A)"),
    charts.stack([fs1, fs2]),
], title=label("沿梁验算图", "Checks along the girder"), css="h2{font-size:16px;margin:28px 0 4px}")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(page)
print(f"governing flexure: {fm.critical}")
print(f"governing inclined section: {fv.governing}")
print(f"utilisation maxima: {fu.maxima}")
print(f"edge stresses (characteristic): {fs1.extremes}; frequent: {fs2.extremes}, beyond limits: {fs2.exceeded}")
print(f"written {out}")
