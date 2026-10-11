"""Prestressed concrete girder to JTG D60-2015 / JTG 3362-2018 (illustrative values).

One post-tensioned I girder of a two-span continuous deck (2 × 30 m, C50, 4 tendons of 9 × φs15.2 strand 1860, jacked
at both ends), Class I lane load and crowd, 100 mm asphalt surfacing, designed as class A:
    effective flange width → tendon profile → losses along the tendon (friction, anchorage set with reverse friction,
    elastic shortening, relaxation, creep and shrinkage) → σpe, σp0 → secondary moments of the prestress and of the
    temperature gradient on the frame → basic combination with both → flexure (span, pier), shear (α2 for prestressed
    members, bent tendons) → serviceability (normal and inclined crack resistance with the temperature gradient), stresses
    in service (characteristic combination with impact); a class B what-if with fewer strands (crack width and stresses
    on the cracked section, with the temperature gradient) → stresses at transfer, deflection, camber and pre-camber →
    anchorage zone: local compression under the anchors, bursting and spalling forces.

    uv run python examples/jtg_pc_girder.py

Simplifications (illustrative example, not a design): the tendon profile has a kink over the pier instead of reversed
curves; ρ, ρps of 6.2.7 take the bottom bars in the span zone only; σpc for σl6 is taken at the tendon and without the
secondary moment; Np of 6.1.7-3 leaves out σl6·As; the end blocks (550 mm web) are used for the section checks only,
the losses and σp0 use the net section of the span; the two ends' reverse-friction zones (G.0.3) do not overlap here.
"""
import math

import numpy as np

from femotion.codes import jtg
from femotion.codes.jtg import prestress as ps
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces
from femotion.frame.prestress import Tendon, effects, equivalent_loads
from femotion.frame.thermal import effects as thermal_effects
from femotion.codes.jtg import regions

L, n = 30.0, 60  # span (m), elements per span
geo = dict(b=200, h=2000, bf=2200, hf=180, bf_bottom=800, hf_bottom=300, concrete="C50", steel="HRB400")
bars = dict(bottom=[(10, 20, 60)], top=[(16, 16, 50)])
bars_end = dict(bottom=[(10, 20, 60), (6, 20, 110)], top=[(16, 16, 50)])  # end blocks: the pretensioned zone at transfer (7.2.8)
strand = jtg.tendon("strand", 1860)
n_tendons, Ap = 4, 4 * 9 * 139.0  # mm²
duct = math.sqrt(n_tendons) * 90.0  # one equivalent duct for the four 90 mm ducts (same area)
gamma0 = jtg.GAMMA_0[1]  # safety class 1 (large bridge)

# effective flange width (4.3.3): girders at 2.2 m, no haunch; 0.27 L of the end spans, 0.07 (L1 + L2) over the pier
bf_span = jtg.tee_flange_width(b=geo["b"], hf=geo["hf"], spacing=2200, L=L * 1e3, region="end span")
bf_pier = jtg.tee_flange_width(b=geo["b"], hf=geo["hf"], spacing=2200, region="support", spans=(L * 1e3, L * 1e3))
assert bf_span == bf_pier == geo["bf"]  # the girder spacing governs: the whole flange counts

gross = jtg.tee(**geo, **bars)
c = gross.concrete
aEP = strand.Ep / c.Ec

# ── tendon profile: e below the centroid (m), 0 at the abutments, 0.6 m above it over the pier ──
xs = np.linspace(0, 2 * L, 2 * n + 1)


def ecc(x):
    xi = np.minimum(x, 2 * L - x) / L
    return xi * (-0.6) + 4 * 1.0 * xi * (1 - xi)


e = ecc(xs)
slope = np.gradient(e, xs)
theta = np.abs(np.r_[0, np.cumsum(np.abs(np.diff(np.arctan(slope))))])  # angle turned from the left end
theta = np.where(xs <= L, theta, theta[::-1])  # jacked at both ends: from the nearer end
dist = np.minimum(xs, 2 * L - xs)  # m

# ── losses (MPa) at every node ──
s_con = strand.sigma_con_max()  # 0.75 fpk
k, mu = ps.DUCT_FRICTION["metal corrugated"][0], 0.25
l1 = np.array([ps.friction_loss(s_con, mu, th, k, x) for th, x in zip(theta, dist)])
s_mid = s_con - l1[n]  # stress at the middle of the tendon after friction (each half acts as jacking end → fixed end)
l2, lf = ps.anchorage_loss_reverse(s_con, s_mid, L * 1000, ps.ANCHOR_SET["wedge"], strand.Ep, dist * 1000)

An, zn, In = gross.replace(tendons=np.array([[Ap, gross.zg, 0, math.pi * duct ** 2 / 4]]), tendon=strand).props("net")
dp = gross.zg + e * 1000  # tendon depth from the top (mm)
ep_n = dp - zn  # below the net centroid
one = (s_con - l1 - l2) * Ap / n_tendons  # force of one tendon after friction and set (N)
d_pc = ps.concrete_stress(one, ep_n, ep_n, An, In)  # at the tendon centroid, from one tendon
l4 = np.array([ps.elastic_shortening_simplified(n_tendons, aEP, s) for s in d_pc])
lI = l1 + l2 + l4
l5 = np.array([ps.relaxation_loss(strand, s_con - a) for a in lI])

h_notional = 2 * gross.area / (2 * geo["h"] + 2 * geo["bf"])  # 2A/u (mm), outline of the girder roughly
eps_cs = ps.shrinkage_strain(c, 75, h_notional, t=3650, t0=7)
phi = ps.creep_coefficient(c, 75, h_notional, t=3650, t0=7)
bot, _ = gross.edge_masks()
As_b = gross.bars[bot, 0].sum()
es_b = gross.bars[bot, 1].mean() - zn
s_pcI = ps.concrete_stress((s_con - lI) * Ap, ep_n, ep_n, An, In)  # first-batch prestress only, self-weight left out
l6 = np.zeros_like(xs)
for i in range(len(xs)):
    if ep_n[i] > 0:  # tendon in the bottom zone, with the bottom bars
        rho, rps = ps.steel_ratios(An, In, Ap, ep_n[i], As_b, es_b)
    else:
        rho, rps = ps.steel_ratios(An, In, Ap, -ep_n[i])
    l6[i] = ps.creep_shrinkage_loss(strand.Ep, eps_cs, aEP, s_pcI[i], phi, rho, rps, fcu_transfer=0.9 * c.fcuk)
lII = l5 + l6
s_pe = s_con - lI - lII

print(f"σcon = {s_con:.0f} MPa, reverse-friction length lf = {lf / 1000:.1f} m, εcs = {eps_cs:.2e}, φ = {phi:.2f}")
print(f"{'x (m)':>6} {'σl1':>6} {'σl2':>6} {'σl4':>6} {'σl5':>6} {'σl6':>6} {'σpe':>7}   MPa")
for i in range(0, n + 1, 10):
    print(f"{xs[i]:6.1f} {l1[i]:6.1f} {l2[i]:6.1f} {l4[i]:6.1f} {l5[i]:6.1f} {l6[i]:6.1f} {s_pe[i]:7.1f}")

# ── frame: self-weight, live load, prestress ──
area, I = gross.area * 1e-6, gross.I * 1e-12
sec = Section(E=c.Ec * 1e3, A=area, Iy=I, Iz=10 * I, J=0.05, mass=area * 2.6 + 1.2)
m = Model()
nodes = [m.node((x, 0, 0)) for x in xs]
beams = [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
m.fix(nodes[n], (0, 1, 1, 1, 0, 0))
m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
sv = Solver(m)
secs = chain_sections(beams)
x_sec = np.array([xs[b + (end == "j")] for b, end in secs])
node_of = np.array([b + (end == "j") for b, end in secs])

P = (s_pe[:-1] + s_pe[1:]) / 2 * Ap / 1e3  # kN per element
td = Tendon(beams, e, P)
M_tot, M_pri, M_sec = effects(sv, td, secs, "My")
V_tot, V_pri, V_sec = effects(sv, td, secs, "Vz")

f, Phi = sv.modes(8)
vert = [i for i in range(8) if np.abs(Phi[2::6, i]).max() > np.abs(Phi[1::6, i]).max()]
mu1, mu2 = jtg.impact_factor(f[vert[0]]), jtg.impact_factor(f[vert[1]])
il = InfluenceLines(sv, nodes, secs)
live = lambda mu_: jtg.LiveLoad(L0=L, lanes=1, transverse=1.0, distribution=0.5, mu=mu_)
G_M, G_V = uniform_load_forces(sv, secs, "My"), uniform_load_forces(sv, secs, "Vz")
Q_M = (live(mu1).effects(il, "My")[0], live(mu2).effects(il, "My")[1])
Q_V = live(mu1).effects(il, "Vz")
# crowd (D60 4.3.6) on two 1.5 m sidewalks, 0.2 of it on this girder
crowd = jtg.CrowdLoad(L0=L, width=3.0, distribution=0.2)
C_M, C_V = crowd.effects(il, "My"), crowd.effects(il, "Vz")

# temperature gradient (D60 4.3.12, 100 mm asphalt: T1 = 14, T2 = 5.5 °C) and its secondary moment on the continuous
# girder (3362 4.3.10, D.0.2): the free curvature of the span section imposed on every element
grad = jtg.temperature_gradient(geo["h"], surfacing="asphalt", asphalt=100)
_, kappa = jtg.thermal_stresses(gross, grad).free_strain()
Mt2 = thermal_effects(sv, beams, secs, "My", curvature=kappa)  # positive gradient; the negative one is −0.5 ×
T_M = (np.maximum(Mt2, jtg.NEGATIVE_GRADIENT * Mt2), np.minimum(Mt2, jtg.NEGATIVE_GRADIENT * Mt2))

# secondary effect of prestress as a permanent action, γ 1.2 / 1.0 (D60 table 4.1.5-2, 预加力); crowd and temperature
# gradient as other variable actions (γ 1.4 × ψc 0.75)
Mmax, Mmin = jtg.basic_combination([G_M, (M_sec, (1.2, 1.0))], Q_M, [C_M, T_M])
Vmax, Vmin = jtg.basic_combination([G_V, (V_sec, (1.2, 1.0))], Q_V, [C_V])
k_pier = int(np.argmin(np.abs(x_sec - L)))
print(f"\nsecondary moment at the pier: prestress {M_sec[k_pier]:.0f} kN·m (primary {M_pri[k_pier]:.0f}, total "
      f"{M_tot[k_pier]:.0f}); temperature gradient {Mt2[k_pier]:.0f} kN·m (T1 = {grad.t[0]:g}, T2 = {grad.t[1]:g} °C)")


def girder_at(i):
    """the girder section at node i with its tendon (σp0 from 6.1.6-5, including the secondary moment, 6.1.6-4; σpe
    after all losses). The web is thickened to 550 mm within 2 m of the supports (end blocks; the frame keeps the span
    section throughout)"""
    M2 = M_sec[np.argmin(np.abs(x_sec - xs[i]))] * 1e6  # N·mm, sagging +
    s_pc = ps.concrete_stress(s_pe[i] * Ap, ep_n[i], ep_n[i], An, In, M=M2)
    sp0 = ps.sigma_p0(s_con, lI[i] + lII[i], post=True, alpha_EP=aEP, sigma_pc=s_pc)
    near_support = min(abs(xs[i]), abs(xs[i] - L), abs(xs[i] - 2 * L)) <= 2.0
    g, b_ = (dict(geo, b=550), bars_end) if near_support else (geo, bars)
    return jtg.tee(**g, **b_, tendons=[(Ap, dp[i], sp0, duct)], tendon=strand, sigma_pe=s_pe[i],
                   sigma_l6=l6[i]), sp0


print("\nchecks")
k_span = int(np.argmax(np.where(x_sec < L, Mmax, -np.inf)))
for k_, M in [(k_span, Mmax[k_span]), (k_pier, Mmin[k_pier])]:
    s, sp0 = girder_at(node_of[k_])
    r = jtg.flexure(s, M, gamma0, pc_class="A")
    print(f"  x = {x_sec[k_]:5.1f} m  σp0 = {sp0:.0f} MPa  {r}")
    print(f"               Mcr = {r.details['Mcr']:.0f} kN·m, x = {r.details['x']:.0f} mm, ξb = {r.details['xi_b']}")
h_half = geo["h"] / 2000
for x_v, hog, a1 in [(h_half, False, 1.0), (L - 1.0, True, 0.9)]:
    k_ = int(np.argmin(np.abs(x_sec - x_v)))
    s, _ = girder_at(node_of[k_])
    ang = abs(math.degrees(math.atan(slope[node_of[k_]])))
    v = jtg.shear(s, max(abs(Vmax[k_]), abs(Vmin[k_])), (2, 12, 150), gamma0=gamma0, pc_class="A", hogging=hog,
                  alpha1=a1, bent_tendons=[(Ap, ang)])
    print(f"  x = {x_sec[k_]:5.1f} m  {v}  (α2 = {v.details['alpha2']}, Vpb = {v.details['Vpb']:.0f} kN)")

# ── serviceability (JTG 3362-2018 chapter 6, class A) and stresses (chapter 7) ──
# The prestress is in the section (σp0, σpe, σl6); its secondary moment and shear go in separately (M_p2, V_p2), so the
# combinations below hold the self-weight and the live load only. Chapter 6: no impact (6.1.1); 7.1: with impact.
# The temperature gradient (its self-equilibrating and secondary stresses) goes in through temperature=, not through the
# moments: ψ = 0.8 in the frequent combination, none in the quasi-permanent one (6.3.2), 1.0 in the characteristic one.
Q0_M, Q0_V = live(0.0).effects(il, "My"), live(0.0).effects(il, "Vz")
Ms_max, Ms_min = jtg.frequent_combination(G_M, Q0_M, [C_M], kinds=["crowd"])
Ml_max, Ml_min = jtg.quasi_permanent_combination(G_M, Q0_M, [C_M], kinds=["crowd"])
Vs_max, Vs_min = jtg.frequent_combination(G_V, Q0_V, [C_V], kinds=["crowd"])
Mk_max, Mk_min = jtg.characteristic_combination(G_M, Q_M, [C_M])
Vk_max, Vk_min = jtg.characteristic_combination(G_V, Q_V, [C_V])
print("\nserviceability and stresses (class A, precast girder)")
for k_, Ms, Ml, Mk in [(k_span, Ms_max, Ml_max, Mk_max), (k_pier, Ms_min, Ml_min, Mk_min)]:
    s, _ = girder_at(node_of[k_])
    th = jtg.thermal_stresses(s, grad, M_sec=Mt2[k_])
    print(f"  x = {x_sec[k_]:5.1f} m  Mp2 = {M_sec[k_]:.0f} kN·m, Mt' = {Mt2[k_]:.0f} kN·m")
    r = jtg.crack_resistance(s, Ms[k_], Ml[k_], pc_class='A', M_p2=M_sec[k_], temperature=th)
    print(f"      {r}  (temperature {r.details['sigma_t']:+.2f} MPa at that edge)")
    print(f"      {jtg.service_stresses(s, Mk[k_], pc_class='A', M_p2=M_sec[k_], temperature=th)}")
# the same girder as a partially prestressed class B member (what-if: 6 of the 9 strands in each tendon, the losses
# kept, σp0 with the reduced precompression; the secondary moment of the prestress scales with the force): crack width
# (6.4.4 item 3) and stresses in service (7.1.4) on the cracked section, with the temperature gradient (its secondary
# moment in the moment, its self-equilibrating stress on the cracked section, appendix D commentary). A member allowed
# to crack takes 0.8 Ec I in the frame (4.3.1): with every element scaled alike the temperature's secondary moment is
# 0.8 × that of the class A frame. Illustrative: over the pier this variant exceeds the 0.10 mm of table 6.4.2.
print("\nclass B what-if (6 of 9 strands per tendon)")
f_B = 6 / 9
for k_, Ms, Ml, Mk in [(k_span, Ms_max, Ml_max, Mk_max), (k_pier, Ms_min, Ml_min, Mk_min)]:
    i = node_of[k_]
    s, _ = girder_at(i)
    s_pc = ps.concrete_stress(s_pe[i] * Ap * f_B, ep_n[i], ep_n[i], An, In, M=M_sec[k_] * f_B * 1e6)
    td = s.tendons * [f_B, 1, 1, f_B]
    td[:, 2] = ps.sigma_p0(s_con, lI[i] + lII[i], post=True, alpha_EP=aEP, sigma_pc=s_pc)
    sB = s.replace(tendons=td)
    th = jtg.thermal_stresses(sB, grad, M_sec=0.8 * Mt2[k_])
    kw = dict(pc_class="B", M_p2=M_sec[k_] * f_B, temperature=th)
    w = jtg.crack_width(sB, Ms[k_], Ml[k_], env="I", tendon_bundles=[(n_tendons, 6, 15.2)], **kw)
    print(f"  x = {x_sec[k_]:5.1f} m  {w}  (gradient: {w.details.get('temperature')})")
    sv_ = jtg.service_stresses(sB, Mk[k_], **kw)
    print(f"               {sv_}  (gradient: {sv_.details['temperature']}, "
          f"tendon {sv_.details['tendon_stress'].max():.0f} MPa)")
# principal stresses along the girder (6.3.1 item 2, 7.1.6): every section from h/2 off the supports, each shear extreme
# with the moment extreme of the same combination (envelopes: the pairs are not concurrent loadings)
worst_t = worst_c = None
for k_ in range(len(secs)):
    if min(abs(x_sec[k_]), abs(x_sec[k_] - L), abs(x_sec[k_] - 2 * L)) < h_half:
        continue
    s, _ = girder_at(node_of[k_])
    for Vs, Ms, Vk, Mk in [(Vs_max[k_], Ms_max[k_], Vk_max[k_], Mk_max[k_]), (Vs_min[k_], Ms_min[k_], Vk_min[k_], Mk_min[k_])]:
        # the inclined tendon relieves the shear where its own (primary) shear opposes the load shear (6.3.3-5)
        ang = math.degrees(math.atan(abs(slope[node_of[k_]]))) * (1 if V_pri[k_] * Vs < 0 else -1)
        kw = dict(M_p2=M_sec[k_], V_p2=V_sec[k_], bent_tendons=[(Ap, ang)],
                  temperature=jtg.thermal_stresses(s, grad, M_sec=Mt2[k_]))
        t = jtg.inclined_crack_resistance(s, Ms, Vs, pc_class="A", **kw)
        p_ = jtg.principal_compression(s, Mk, Vk, stirrups=(2, 12, 150), **kw)
        if worst_t is None or t.ratio > worst_t[1].ratio:
            worst_t = (x_sec[k_], t)
        if worst_c is None or p_.ratio > worst_c[1].ratio:
            worst_c = (x_sec[k_], p_)
for xx, r in (worst_t, worst_c):
    print(f"  x = {xx:5.1f} m  {r}  (at {r.details['at']['z']:.0f} mm from the top)")

# at transfer: σpe = σcon − σlI, the girder's own weight only (no deck finishes yet), net section before grouting,
# f'cu = 0.9 fcu,k; the secondary moment of the transfer force
w_girder = np.zeros((len(beams), 3))
w_girder[:, 2] = -area * 2.6 * 9.81  # kN/m
G_M_t = uniform_load_forces(sv, secs, "My", w=w_girder)
P_t = (s_con - lI[:-1] / 2 - lI[1:] / 2) * Ap / 1e3
M_sec_t = effects(sv, Tendon(beams, e, P_t), secs, "My")[2]
for k_ in (k_span, k_pier):
    i = node_of[k_]
    s, _ = girder_at(i)
    s_t = s.replace(sigma_pe=s_con - lI[i], sigma_l6=0.0)
    print(f"  transfer x = {x_sec[k_]:5.1f} m  {jtg.transient_pc(s_t, G_M_t[k_], fcu=0.9 * c.fcuk, M_p2=M_sec_t[k_])}")

# deflection (6.5): class A, B0 = 0.95 Ec I0 throughout (span section); camber from the prestress after all losses on
# Ec I0 × 2.0 (6.5.4)
A0, z0, I0 = jtg.tee(**geo, **bars, tendons=[(Ap, dp[n // 2], 1000.0, duct)], tendon=strand).props("transformed")


def frame(EI_factor):
    mm = Model()
    nn = [mm.node((xx, 0, 0)) for xx in xs]
    bb = [mm.beam(a, b, Section(E=c.Ec * 1e3, A=area, Iy=EI_factor * I0 * 1e-12, Iz=10 * I, J=0.05,
                                mass=area * 2.6 + 1.2)) for a, b in zip(nn[:-1], nn[1:])]
    mm.fix(nn[0], (1, 1, 1, 1, 0, 0))
    mm.fix(nn[n], (0, 1, 1, 1, 0, 0))
    mm.fix(nn[-1], (0, 1, 1, 1, 0, 0))
    return Solver(mm), bb


sv_B, _ = frame(0.95)
sv_E, beams_E = frame(1.0)
eta = jtg.long_term_factor(c)
i_m = int(np.argmin(np.abs(xs - 0.4 * L)))
fG = -uniform_load_forces(sv_B, secs, None)[1][2::6][i_m] * 1e3
infl = np.array([-sv_B.solve(np.eye(sv_B.m.ndof)[6 * j + 2] * -1.0)[2::6][i_m] for j in range(len(xs))]) * 1e3
q, Pk = jtg.lane_load(L)
pos = np.maximum(infl, 0.0)
area_pos = float(((pos[:-1] + pos[1:]) / 2 * np.diff(xs)).sum())
# vehicle (ψf 0.7, no impact) and crowd (another variable action: ψq 0.4) of the frequent combination (6.5.3)
fQ = 0.7 * live(0.0).factor() * (q * area_pos + Pk * pos.max()) + 0.4 * crowd.intensity() * crowd.width * \
    crowd.distribution * area_pos
f_pe = sv_E.solve(equivalent_loads(sv_E, Tendon(beams_E, e, P)))[2::6][i_m] * 1e3  # upward +
print(f"  deflection at x = {xs[i_m]:.1f} m (ηθ = {eta:.3f}): {jtg.deflection_check(eta * fQ, L * 1e3)}")
f_s, f_c = eta * (fG + fQ), jtg.CAMBER_LONG_TERM * f_pe
print(f"  long-term: frequent combination {f_s:.1f} mm down, prestress camber {f_c:.1f} mm up → pre-camber "
      f"{jtg.precamber_pc(f_s, f_c):.1f} mm (6.5.5)")

# ── anchorage zone at the girder end (5.7, 8.2): the four anchors one above the other at 450 mm in the 550 mm end block,
# plates 230 × 230 with a 110 mm trumpet, spirals φ12 (core 250 mm, pitch 50 mm, 6 turns), f'cu = 0.9 fcu,k at jacking
Pd = regions.anchor_force(s_con * Ap / n_tendons / 1e3)  # 1.2 × the jacking force of one tendon (kN)
Al = 230.0 * 230.0
Ab = jtg.bearing_base_area(230, 230, cx=(550 - 230) / 2, cy=(450 - 230) / 2)
print(f"\nanchorage zone (Pd = {Pd:.0f} kN per anchor)")
print("  " + str(jtg.local_bearing(c, Pd, Al=Al, Aln=Al - math.pi * 55 ** 2, Ab=Ab, gamma0=gamma0, fcu=0.9 * c.fcuk,
                                    spiral=(113.1, 250, 50), layers=6)))
# dense anchors (450 mm < 2 × 230 mm apart): the group as one, a = the outer edges of the outer plates (8.2.2)
alpha = -math.degrees(math.atan(slope[0]))  # at the centroid and leaving it: negative (figure 8-5)
Tb, db = regions.bursting_force(n_tendons * Pd, a=3 * 450 + 230, h=geo["h"], e=0.0, alpha=alpha)
print(f"  bursting force {Tb:.0f} kN at {db:.0f} mm from the end face; spalling {regions.spalling_force([Pd] * 4):.0f} kN")
print("  " + str(regions.tie_check(Tb, As=16 * math.pi * 16 ** 2 / 4, gamma0=gamma0, name="bursting steel (16 legs φ16)")))
