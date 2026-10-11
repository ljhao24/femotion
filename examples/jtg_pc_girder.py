"""Prestressed concrete girder to JTG D60-2015 / JTG 3362-2018, ultimate limit state (illustrative values).

One post-tensioned I girder of a two-span continuous deck (2 × 30 m, C50, 4 tendons of 9 × φs15.2 strand 1860, jacked
at both ends), Class I lane load:
    tendon profile → losses along the tendon (friction, anchorage set with reverse friction, elastic shortening,
    relaxation, creep and shrinkage) → σpe, σp0 → secondary moments on the frame → basic combination with the
    secondary effect → flexure (span, pier), shear (α2 for prestressed members, bent tendons).

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
from femotion.frame.prestress import Tendon, effects

L, n = 30.0, 60  # span (m), elements per span
geo = dict(b=200, h=2000, bf=2200, hf=180, bf_bottom=800, hf_bottom=300, concrete="C50", steel="HRB400")
bars = dict(bottom=[(8, 16, 60)], top=[(16, 16, 50)])
strand = jtg.tendon("strand", 1860)
n_tendons, Ap = 4, 4 * 9 * 139.0  # mm²
duct = math.sqrt(n_tendons) * 90.0  # one equivalent duct for the four 90 mm ducts (same area)
gamma0 = jtg.GAMMA_0[1]  # safety class 1 (large bridge)

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
# secondary effect of prestress as a permanent action, γ 1.2 / 1.0 (D60 table 4.1.5-2, 预加力)
Mmax, Mmin = jtg.basic_combination([G_M, (M_sec, (1.2, 1.0))], Q_M)
Vmax, Vmin = jtg.basic_combination([G_V, (V_sec, (1.2, 1.0))], Q_V)
k_pier = int(np.argmin(np.abs(x_sec - L)))
print(f"\nsecondary moment at the pier: {M_sec[k_pier]:.0f} kN·m (primary {M_pri[k_pier]:.0f}, total {M_tot[k_pier]:.0f})")


def girder_at(i):
    """the girder section at node i with its tendon (σp0 from 6.1.6-5, including the secondary moment, 6.1.6-4). The
    web is thickened to 550 mm within 2 m of the supports (end blocks; the frame keeps the span section throughout)"""
    M2 = M_sec[np.argmin(np.abs(x_sec - xs[i]))] * 1e6  # N·mm, sagging +
    s_pc = ps.concrete_stress(s_pe[i] * Ap, ep_n[i], ep_n[i], An, In, M=M2)
    sp0 = ps.sigma_p0(s_con, lI[i] + lII[i], post=True, alpha_EP=aEP, sigma_pc=s_pc)
    near_support = min(abs(xs[i]), abs(xs[i] - L), abs(xs[i] - 2 * L)) <= 2.0
    g = dict(geo, b=550) if near_support else geo
    return jtg.tee(**g, **bars, tendons=[(Ap, dp[i], sp0, duct)], tendon=strand), sp0


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
