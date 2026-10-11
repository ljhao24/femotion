"""A single-cell box girder across the bridge: JTG 3362-2018 4.2, 5.6 and table 6.1.3 (illustrative values).

A 12.5 m wide, 2.5 m deep C50 box: webs 5.8 m apart, 0.25 m top slab with transverse tendons (BM15-3 flat tendons
every 0.75 m), 0.22 m bottom slab reinforced only, 3.1 m cantilevers 0.55 → 0.20 m, haunches; 100 mm asphalt, barriers
0.5 m wide, carriageway 11.5 m with two-way traffic (two design lanes).
    the slice as a closed frame on the web feet (program convention) → self-weight, surfacing, barriers, vehicle
    wheels spread by 4.2.3 / 4.2.5 and placed for the largest effect (μ = 0.3), transverse tendons (primary and
    secondary), the temperature gradient → top slab: flexure and shear (5.2.2, 5.2.12), transverse normal stresses
    at both faces (class A, 6.3.1), punching under a wheel (5.6.1) → bottom slab: crack widths (6.4) → in-plane
    principal stresses of the top slab and a web with the shear flow of V and T (6.3.3, 6.3.1-8) → the cantilever
    by the strip formulas of 4.2.5 as a check on the frame.

    uv run python examples/jtg_box_transverse.py
"""
import math

import numpy as np

from femotion.codes import jtg
from femotion.codes.jtg import slab
from femotion.codes.jtg import transverse as tr

gamma0 = jtg.GAMMA_0[1]
box = tr.BoxGeometry(B=12.5, H=2.5, webs=(-2.9, 2.9), t_top=0.25, t_bottom=0.22, t_web=0.5, t_root=0.55, t_tip=0.20,
                     haunch_top=(0.9, 0.3), haunch_bottom=(0.5, 0.2), carriageway=(-5.75, 5.75))
fr = tr.TransverseFrame(box, concrete="C50")
print(tr.box_detailing(box, transverse_prestress=True))  # 9.3.2 plate dimensions
h_asph = 0.10  # surfacing thickness for the 45° spread (m)

# ── actions per metre of girder ──
G = fr.self_weight() + fr.surface(24 * h_asph, -5.75, 5.75) + fr.line(10.0, -6.0) + fr.line(10.0, 6.0)
Q = {t: fr.wheels(h=h_asph, target=t, two_way=True) for t in ("M", "V", "top", "bottom")}  # with impact (ULS)
Q0 = {t: fr.wheels(h=h_asph, target=t, two_way=True, mu=0.0) for t in ("top", "bottom")}  # no impact (SLS, 6.1.1)
grad = jtg.temperature_gradient(box.H * 1e3, surfacing="asphalt", asphalt=100)
Tp, Tn = fr.temperature(grad), fr.temperature(grad.scaled(jtg.NEGATIVE_GRADIENT))
Ap, sigma_pe, sigma_p0 = 3 * 140.0 / 0.75, 1050.0, 1080.0  # per metre: 3 strands every 0.75 m; after all losses (MPa)
duct = math.sqrt(4 * 70 * 25 / 0.75 / math.pi)  # the flat ducts (70 × 25 every 0.75 m) as one round one per metre (mm)
xs = np.linspace(-2.9, 2.9, 13)
profile = [(-6.2, 0.07)] + [(x, 0.12 + 0.05 * (1 - (x / 2.9) ** 2)) for x in xs] + [(6.2, 0.07)]  # depth (m)
P = fr.tendon(profile, force=Ap * sigma_pe / 1e3)
print(f"vehicle: {Q['M'].details['lanes']} lanes, μ = {Q['M'].details['mu']}; " + "; ".join(Q["M"].notes))
print(fr.table(G, "permanent (self-weight, surfacing, barriers)"))
print(fr.table(Q["M"].min, "vehicle, most negative M (with impact)"))
print(fr.table(P.secondary, "transverse tendons: secondary"))


def depth_at(x):
    return float(np.interp(x, *np.array(profile).T)) * 1e3  # tendon depth below the deck (mm)


def strip(cs, x, tendons=True):
    """one metre of the top slab at a control section: check depth by 4.2.6, Φ16 @ 100 both faces"""
    td = [(Ap, depth_at(x), sigma_p0, duct)] if tendons else []
    return jtg.rect(1000, cs.h * 1e3, concrete="C50", top=[(10, 16, 40)], bottom=[(10, 16, 40)], tendons=td,
                    sigma_pe=sigma_pe if tendons else None)


# ── top slab ──
print("\ntop slab (class A across the bridge)")
for name, x, face in (("top slab, web 0 face, cell 0", box.faces(0)[1], "top"),
                      ("top slab, cell 0 middle", 0.0, "bottom"), ("left cantilever root", box.faces(0)[0], "top")):
    i = G.index(name)
    cs = fr.sections[i]
    sec = strip(cs, x)
    env = Q["M"].min if face == "top" else Q["M"].max
    T_unf = Tp if (Tp.stress()[i, 0 if face == "top" else 1] > Tn.stress()[i, 0 if face == "top" else 1]) else Tn
    Md = 1.2 * G.M[i] + 1.8 * env.M[i] + 0.75 * 1.4 * T_unf.M[i] + 1.2 * P.secondary.M[i]
    Vd = 1.2 * abs(G.V[i]) + 1.8 * abs(Q["V"].max.V[i] if abs(Q["V"].max.V[i]) > abs(Q["V"].min.V[i]) else Q["V"].min.V[i])
    print(f"  {name}: h = {cs.h * 1e3:.0f} mm, tendon {depth_at(x):.0f} mm deep")
    print("    " + str(jtg.flexure(sec, Md, gamma0, pc_class="A")))
    print("    " + str(jtg.shear(sec, Vd, gamma0=gamma0, slab=True, pc_class="A", hogging=Md < 0)))
    fs = G + 0.7 * Q0[face].max  # frequent combination: vehicle ψf = 0.7, no impact
    ql = G + 0.4 * Q0[face].max  # quasi-permanent
    extra = 0.8 * T_unf.stress()[i]  # temperature gradient ψ = 0.8
    print("    " + str(tr.transverse_normal_stresses(sec, fs.N[i], fs.M[i], ql.N[i], ql.M[i], pc_class="A",
                                                       N_p2=P.secondary.N[i], M_p2=P.secondary.M[i], extra=extra)))

# punching under a rear wheel on the top slab (5.6.1): the contact spread through the asphalt at 45°; the wheel of one
# vehicle × 1.2 (table 4.3.1-5, as the program applies it to local loading), × 1.3 impact, × 1.8 (D60 4.1.5)
h0 = 250 - 45
Um = slab.punching_perimeter(h0, bx=(0.6 + 2 * h_asph) * 1e3, by=(0.2 + 2 * h_asph) * 1e3)
print("  " + str(slab.punching(1.8 * 1.3 * 1.2 * 70.0, Um=Um, h0=h0, h=250, concrete="C50",
                               sigma_pc=Ap * sigma_pe / 250e3, gamma0=gamma0)))  # σpc,m above 3.5: taken as 3.5

# ── bottom slab: reinforced concrete across the bridge (6.4) ──
print("\nbottom slab (reinforced concrete across the bridge)")
for name, face in (("bottom slab, web 0 face, cell 0", "top"), ("bottom slab, cell 0 middle", "bottom")):
    i = G.index(name)
    sec = jtg.rect(1000, fr.sections[i].h * 1e3, concrete="C50", top=[(8, 16, 40)], bottom=[(8, 16, 40)])
    fs, ql = G + 0.7 * Q0[face].max, G + 0.4 * Q0[face].max
    fs = fs + P.secondary
    ql = ql + P.secondary
    print(f"  {name}: Ns = {fs.N[i]:.1f} kN/m, Ms = {fs.M[i]:.2f} kN·m/m")
    print("    " + str(tr.transverse_crack_width(sec, fs.N[i], fs.M[i], ql.N[i], ql.M[i], env="I")))

# ── in-plane principal stresses (frequent combination): σx along the girder from the longitudinal analysis ──
Vs, Ts = 4200.0, 2600.0  # frequent shear (kN) and torque (kN·m) of the girder at this section (illustrative)
f = tr.shear_flow(box, Vs, Ts)
# σy, σcy: only the actions 6.3.3-3 lists (transverse prestress, transverse temperature gradient, vehicles; vertical
# prestress there is none), the temperature gradient whichever (or neither) gives less compression
def sigma_y(name, h, vehicle=True):
    k = G.index(name)
    t = min(0.0, -0.8 * Tp.N[k] / h / 1e3, -0.8 * Tn.N[k] / h / 1e3)
    v = -0.7 * Q0["top"].max.N[k] / h / 1e3 if vehicle else 0.0
    return -P.total.N[k] / h / 1e3 + t + v


i = G.index("top slab, web 0 face, cell 0")
sx_top = 5.2  # MPa, longitudinal compression in the top slab (illustrative)
pts_top = [(sx_top, sigma_y("top slab, web 0 face, cell 0", fr.sections[i].h), f["top slab at web"]["tau"])]
print("\nin-plane principal stresses")
print("  top slab at the web: " + str(tr.inplane_principal(pts_top, concrete="C50", pc_class="A")))
# the web: σcy,l of the vehicles is not taken from the frame, whose web force is the share of the load its support
# calls for, not the stress at the centroid (left at 0, on the safe side)
sy_web = sigma_y("web 0 middle", box.t_web, vehicle=False)
pts_web = [(3.6, sy_web, f["web at centroid"]["tau"])]  # σx at the web's centroid from the longitudinal analysis
print("  web at the centroid: " + str(tr.inplane_principal(pts_web, concrete="C50", pc_class="A")))
print(f"  (σcy = {sy_web:.2f} MPa, τ = {f['web at centroid']['tau']:.2f} MPa: the same values go to "
      "jtg.principal_stresses(..., sigma_cy=…, tau_t=…) on the girder's section)")

# ── the cantilever by the strip formulas of 4.2.5 ──
L = box.cantilevers()[0]
c = slab.cantilever_slab(L, box.t_root, box.t_tip, h=h_asph, curb=L - 0.5, surfacing=(2.4, 0.0, L - 0.5),
                         line_loads=[(10.0, L - 0.25)], lanes=2)
i = G.index("left cantilever root")
print(f"\ncantilever root by 4.2.5: M = {c.M_support[0]:.1f} + {c.M_support[1]:.1f} kN·m/m; frame: "
      f"{G.M[i]:.1f} + {Q['M'].min.M[i]:.1f} kN·m/m")
