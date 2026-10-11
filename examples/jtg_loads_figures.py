"""Load cases and their drawings on the girder of examples/jtg_rc_girder.py (illustrative values).

One T girder of a two-span continuous deck (2 × 16 m, C40), one of five girders: the loads as load cases
(femotion.frame.loads), the worst placements of the moving loads behind the envelopes (InfluenceLines.lane_layout,
axle_layout through jtg.LiveLoad.layouts, vehicle_layouts and jtg.CrowdLoad.layouts), and an HTML page with
    the load arrangement of every case: self-weight, superimposed dead load, temperature gradient, settlement of the pier;
    influence lines with the worst placement of the lane load and of the vehicle load for the largest sagging moment
    in the span, the largest hogging moment over the pier and the largest shear at the end support; the crowd on the
    sidewalks for the span moment.

    uv run python examples/jtg_loads_figures.py [--out jtg_loads_figures.html] [--lang zh|en] [--width 720]
"""
import argparse

import numpy as np

from femotion.codes import jtg
from femotion.frame import Model, Section, Solver
from femotion.frame.figures import influence_figure, load_figure
from femotion.frame.influence import InfluenceLines, chain_sections
from femotion.frame.loads import LoadCase
from femotion.viz import html

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="jtg_loads_figures.html")
ap.add_argument("--lang", default="zh", choices=["zh", "en"])
ap.add_argument("--width", type=float, default=720)
a = ap.parse_args()
zh = a.lang == "zh"

# the girder of jtg_rc_girder.py; Section.mass is the girder alone here, the pavement and railing share is its own case
L, n = 16.0, 32
span_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
c = span_sec.concrete
area, I = span_sec.area * 1e-6, span_sec.I * 1e-12
sec = Section(E=c.Ec * 1e3, A=area, Iy=I, Iz=10 * I, J=0.01, mass=area * 2.6)
m = Model()
xs = np.linspace(0, 2 * L, 2 * n + 1)
nodes = [m.node((x, 0, 0)) for x in xs]
beams = [m.beam(p, q, sec) for p, q in zip(nodes[:-1], nodes[1:])]
m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
m.fix(nodes[n], (0, 1, 1, 1, 0, 0))
m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
sv = Solver(m)
secs = chain_sections(beams)
x = np.array([xs[b + (e == "j")] for b, e in secs])

# ── permanent and other actions as load cases ──
sdl_q = 0.55 * 9.81  # pavement and railing share of this girder (kN/m)
grad = jtg.temperature_gradient(span_sec.h, surfacing="asphalt", asphalt=100)  # D60 4.3.12: T1 = 14, T2 = 5.5 °C
_, kappa = jtg.thermal_stresses(span_sec, grad).free_strain()
cases = {
    "self": LoadCase("自重" if zh else "self-weight").self_weight(),
    "sdl": LoadCase("二期恒载" if zh else "superimposed dead load").line(
        beams, sdl_q, label=(f"桥面铺装及栏杆 {sdl_q:.1f} kN/m" if zh else f"pavement and railing {sdl_q:.1f} kN/m")),
    "temp": LoadCase("温度梯度（正）" if zh else "temperature gradient (positive)").imposed(
        beams, curvature=kappa, label=(f"T₁ = {grad.t[0]:g} °C，T₂ = {grad.t[1]:g} °C，κ = {kappa * 1e4:.2f}×10⁻⁴ 1/m" if zh else
                                       f"T1 = {grad.t[0]:g} °C, T2 = {grad.t[1]:g} °C, κ = {kappa * 1e4:.2f}×10⁻⁴ 1/m")),
    "settle": LoadCase("中支点沉降 5 mm" if zh else "settlement of the pier, 5 mm").settlement(nodes[n], -0.005),
}
k_pier = 2 * n - 1  # the j end of the element left of the pier
k_mid = n // 2 - 1  # the j end of the element left of x = 8 m
print(f"{'case':<28} {'M at 8 m':>10} {'M at pier':>10}   kN·m")
res = {}
for key, case in cases.items():
    M = case.section_forces(sv, secs, "My")
    res[key] = M
    print(f"{case.name:<28} {M[k_mid]:10.1f} {M[k_pier]:10.1f}")
dead = cases["self"] + cases["sdl"]
G_M = dead.section_forces(sv, secs, "My")
print(f"{'self-weight + superimposed':<28} {G_M[k_mid]:10.1f} {G_M[k_pier]:10.1f}   (as jtg_rc_girder.py: one case "
      "with the pavement in Section.mass)")
assert np.allclose(G_M, res["self"] + res["sdl"])

# ── moving loads: the placements behind the envelopes ──
f, Phi = sv.modes(8)
vert = [k for k in range(8) if np.abs(Phi[2::6, k]).max() > np.abs(Phi[1::6, k]).max()]
mu1, mu2 = jtg.impact_factor(f[vert[0]]), jtg.impact_factor(f[vert[1]])
il = InfluenceLines(sv, nodes, secs)
live = lambda mu: jtg.LiveLoad(L0=L, lanes=1, transverse=1.0, distribution=0.55, mu=mu)  # noqa: E731
crowd = jtg.CrowdLoad(L0=L, width=1.0, distribution=0.6)  # 0.6 m² of crowd per m on this girder (illustrative)
Mmax = live(mu1).effects(il, "My")[0]
Mmin = live(mu2).effects(il, "My")[1]
Vend = live(mu1).effects(il, "Vz")
k_span = int(np.argmax(Mmax[: 2 * n]))  # the largest sagging moment of the first span
k_end = 0  # just right of the end support
v_sense = 1 if abs(Vend[1][k_end]) > abs(Vend[0][k_end]) else 0  # the governing sign
picks = [
    (f"第一跨最大正弯矩（x = {x[k_span]:g} m）" if zh else f"largest sagging moment (x = {x[k_span]:g} m)",
     live(mu1).layouts(il, "My", k_span)[0], live(mu1).vehicle_layouts(il, "My", k_span)[0]),
    ("中支点最大负弯矩" if zh else "largest hogging moment over the pier",
     live(mu2).layouts(il, "My", k_pier)[1], live(mu2).vehicle_layouts(il, "My", k_pier)[1]),
    ("端支点最大剪力" if zh else "largest shear at the end support",
     live(mu1).layouts(il, "Vz", k_end)[v_sense], live(mu1).vehicle_layouts(il, "Vz", k_end)[v_sense]),
]
print(f"\nμ1 = {mu1:.3f}, μ2 = {mu2:.3f};  worst placements (with impact, distribution 0.55):")
for title, lane, veh in picks:
    print(f"  {title}: lane load {lane.value:.1f} (q over {[(round(s0, 2), round(s1, 2)) for s0, s1 in lane.segments]},"
          f" P at {lane.P_at} m), vehicle {veh.value:.1f} (lead axle at {veh.lead} m, heading {veh.heading:+d})")
    # the placement solved as an ordinary load case: exact where the line is linear between the nodes, O(h²) here
    rep = lane.load_case(il).section_forces(sv, [secs[lane.section]], lane.comp)[0]
    print(f"      the lane placement solved as a load case: {rep:.1f}")
crowd_lay = crowd.layouts(il, "My", k_span)[0]

# ── the page ──
W = a.width
parts = [f"<h1>{'荷载工况与影响线加载' if zh else 'Load cases and influence-line loading'}</h1>",
         "<p>" + ("两跨 16 m 连续 T 梁（五梁之一，示意）。荷载布置按工况绘出；移动荷载按影响线最不利布置，"
                  "上栏为加载位置，下栏为截面影响线（正值在上，加载区段加深）。" if zh else
                  "Two-span 16 m continuous T girder (one of five, illustrative). Loads by case; moving loads in their "
                  "worst placement above and the section's influence line below (positive up, loaded parts darker).")
         + "</p>",
         f"<h2>{'工况' if zh else 'Load cases'}</h2>"]
for key in ("self", "sdl", "temp", "settle"):
    parts.append(load_figure(m, cases[key], width=W, lang=a.lang))
parts.append(load_figure(m, dead, width=W, lang=a.lang, title="自重 + 二期恒载" if zh else "self-weight + superimposed"))
parts.append(f"<h2>{'影响线加载' if zh else 'Influence lines'}</h2>")
for title, lane, veh in picks:
    parts.append(f"<h3>{title}</h3>")
    parts.append(influence_figure(il, lane, width=W, lang=a.lang))
    parts.append(influence_figure(il, veh, width=W, lang=a.lang))
parts.append(f"<h3>{'人群荷载' if zh else 'Crowd load'}</h3>")
parts.append(influence_figure(il, crowd_lay, width=W, lang=a.lang))
with open(a.out, "w", encoding="utf-8") as fh:
    fh.write(html(parts, title="荷载工况与影响线加载" if zh else "Load cases and influence lines",
                  css=f"body{{max-width:{W + 48:.0f}px}} h1{{font-size:20px}} h2{{font-size:17px;margin-top:28px}} "
                      "h3{font-size:14px;margin:18px 0 0} p{font-size:13px;line-height:1.6}"))
print(f"\nwritten {a.out}")
