"""Box girder with internal and external tendons to JTG D60-2015 / JTG 3362-2018 (illustrative values).

One simply supported single-cell box (40 m, C50, precast segments with epoxy joints) prestressed by 8 internal tendons
of 12 × φs15.2 in the bottom slab and 10 external tendons of 19 × φs15.2 in HDPE ducts, anchored at the end diaphragms
on the centroid, deviated at the third points to just above the bottom slab, with positioners halfway between; three
lanes of Class I lane load, fully prestressed:
    jacking stresses (6.1.4) → losses of both kinds of tendon (friction at the deviators only for the external ones,
    table 6.2.2) → σpe, σpe,ex, σp0 with σpc of the mixed member (6.1.6-7, 6.1.7-5) → flexure at midspan (5.2.5) → shear
    near the support with the inclined external tendons (5.2.9-5) → crack resistance (6.3.1-2 for segmental members,
    6.3.1-6), stresses in service with σpe,ex ≤ 0.60 fpk (7.1.5-3), stresses at transfer → detailing (9.4.23–9.4.27).

    uv run python examples/jtg_external_girder.py

Simplifications (illustrative, not a design): one section all along (no thickened webs at the supports); the anchorage
set of both kinds of tendon is spread uniformly over half the tendon from each jacked end (6.2.3, friction at the
deviators neglected for it); elastic shortening by appendix H with all 18 tendons and the stress at their resultant;
σpc for σl6 from the first-batch prestress and the girder's self-weight (6.2.7 allows it), and ρ of 6.2.7 with the bonded
steel only (the external
tendons do not restrain the concrete locally; 6.2.7 does not say); simply supported, so no secondary moments (Mp2,ex = 0).
"""
import math

from femotion.codes import jtg
from femotion.codes.jtg import prestress as ps

L = 40.0  # span (m)
geo = dict(B=5000, h=2400, t_top=250, t_bottom=220, webs=1000, b_top=12000, concrete="C50", steel="HRB400",
           bottom=[(50, 16, 60)], top=[(100, 16, 50)])
strand = jtg.tendon("strand", 1860)
Ap, d_int = 8 * 12 * 139.0, 2290.0  # internal tendons in the bottom slab (mm², depth)
Aex, d_dev = 10 * 19 * 139.0, 2030.0  # external tendons and their depth at the deviators
duct = math.sqrt(8) * 90.0  # one equivalent duct for the eight 90 mm ducts
gamma0 = jtg.GAMMA_0[1]

rc = jtg.box(**geo)
c = rc.concrete
zg = rc.zg
aEP = strand.Ep / c.Ec
x_dev = L / 3  # deviators at the third points (m)
theta_dev = math.atan((d_dev - zg) / (x_dev * 1000))  # the tendon turns by this at a deviator

# ── jacking (6.1.4) and losses (6.2) at midspan ──
s_con_i = strand.sigma_con_max()  # 0.75 fpk, internal
s_con_e = strand.sigma_con_max(external=True)  # 0.70 fpk, external (6.1.4-2)
l1_i = ps.friction_loss(s_con_i, 0.25, 0.0, ps.DUCT_FRICTION["metal corrugated"][0], L / 2)
mu_ex = ps.DUCT_FRICTION_EXTERNAL["HDPE"][0][1]  # bare strand in HDPE: 0.12–0.15, upper value
l1_e = ps.friction_loss_external(s_con_e, mu_ex, [theta_dev])  # one deviator between the jacking end and midspan
l2_i = ps.anchorage_loss_straight(ps.ANCHOR_SET["wedge"], L / 2 * 1000, strand.Ep)
l2_e = l2_i

An, zn, In = rc.replace(tendons=[[Ap, d_int, 0.0, math.pi * duct ** 2 / 4]], tendon=strand).props("net")
P1 = ((s_con_i - l1_i - l2_i) * Ap + (s_con_e - l1_e - l2_e) * Aex) / 18  # one tendon after friction and set
d_res = ((s_con_i - l1_i - l2_i) * Ap * d_int + (s_con_e - l1_e - l2_e) * Aex * d_dev) / (P1 * 18)
l4 = ps.elastic_shortening_simplified(18, aEP, ps.concrete_stress(P1, d_res - zn, d_res - zn, An, In))
lI_i, lI_e = l1_i + l2_i + l4, l1_e + l2_e + l4  # table 6.2.8: the same combination for both kinds
l5_i = ps.relaxation_loss(strand, s_con_i - lI_i)
l5_e = ps.relaxation_loss(strand, s_con_e - lI_e)

h_n = 2 * rc.area / (2 * 12000 + 2 * 5000 + 2 * 2 * 2400)  # notional size 2A/u, outline of the box roughly
eps_cs = ps.shrinkage_strain(c, 70, h_n, t=3650, t0=10)
phi = ps.creep_coefficient(c, 70, h_n, t=3650, t0=10)
NI = (s_con_i - lI_i) * Ap + (s_con_e - lI_e) * Aex
eI = ((s_con_i - lI_i) * Ap * d_int + (s_con_e - lI_e) * Aex * d_dev) / NI - zn
bot, _ = rc.edge_masks()
As_b, es_b = rc.bars[bot, 0].sum(), rc.bars[bot, 1].mean() - zn
l6 = {}
Mg1 = rc.area * 1e-6 * 26.0 * L * L / 8  # the girder's self-weight at midspan (kN·m)
for name, d, A_t in (("internal", d_int, Ap), ("external", d_dev, 0.0)):
    s_pc = ps.concrete_stress(NI, eI, d - zn, An, In, M=Mg1 * 1e6)
    rho, rps = ps.steel_ratios(An, In, A_t, d - zn, As_b, es_b) if A_t else ps.steel_ratios(An, In, 0.0, 0.0, As_b, es_b)
    l6[name] = ps.creep_shrinkage_loss(strand.Ep, eps_cs, aEP, s_pc, phi, rho, rps, fcu_transfer=45)
s_pe = s_con_i - lI_i - l5_i - l6["internal"]
s_pe_ex = s_con_e - lI_e - l5_e - l6["external"]
print(f"σcon {s_con_i:.0f} / {s_con_e:.0f} MPa (internal / external); at midspan:")
print(f"  internal σl1 {l1_i:.1f}, σl2 {l2_i:.1f}, σl4 {l4:.1f}, σl5 {l5_i:.1f}, σl6 {l6['internal']:.1f} → σpe {s_pe:.0f} MPa")
print(f"  external σl1 {l1_e:.1f}, σl2 {l2_e:.1f}, σl4 {l4:.1f}, σl5 {l5_e:.1f}, σl6 {l6['external']:.1f} → σpe,ex "
      f"{s_pe_ex:.0f} MPa = {s_pe_ex / strand.fpk:.3f} fpk")


def section(d_ex, sp_ex, sp_in, sp0=None):
    """the box with both kinds of tendon; σp0 of the internal tendon from 6.1.6-5 with σpc of Np,ex (6.1.6-7, note 3)"""
    s = jtg.box(**geo, tendons=[(Ap, d_int, sp_in, duct)], sigma_pe=sp_in, external=[(Aex, d_ex, sp_ex)])
    if sp0 is None:
        sp0 = ps.sigma_p0(sp_in, 0.0, post=True, alpha_EP=aEP, sigma_pc=float(s.prestress_stress(d_int)))
    return s.replace(tendons=[[Ap, d_int, sp0, math.pi * duct ** 2 / 4]])


mid = section(d_dev, s_pe_ex, s_pe)
print(f"  σp0 of the internal tendons {mid.tendons[0, 2]:.0f} MPa; Np,ex = {mid.prestress_resultant('pe')[0] / 1e3:.0f} kN")

# ── actions (D60): self-weight, surfacing and barriers, three lanes of Class I lane load ──
g = rc.area * 1e-6 * 26.0 + 40.0  # kN/m
I = rc.props("transformed")[2] * 1e-12
f1 = jtg.simple_beam_frequency(L, c.Ec * 1e3 * I, g / 9.81)
mu = jtg.impact_factor(f1)
lanes = 3 * jtg.transverse_reduction(3)
qk, Pk = jtg.lane_load(L)
MG, MQ = g * L * L / 8, lanes * (qk * L * L / 8 + Pk * L / 4)
Md = jtg.basic_combination(MG, ((1 + mu) * MQ, 0.0))[0]
Ms = jtg.frequent_combination(MG, (MQ, 0.0))[0]
Ml = jtg.quasi_permanent_combination(MG, (MQ, 0.0))[0]
Mk = jtg.characteristic_combination(MG, ((1 + mu) * MQ, 0.0))[0]
print(f"\nf1 = {f1:.2f} Hz, μ = {mu:.3f}; midspan MG = {MG:.0f}, MQ = {MQ:.0f} kN·m (without impact)")

print(jtg.flexure(mid, float(Md), gamma0, pc_class="full"))
print(jtg.crack_resistance(mid, float(Ms), float(Ml), pc_class="full", segmental=True))
sv = jtg.service_stresses(mid, float(Mk), pc_class="full")
print(sv, f"(external tendons {sv.details['external_stress'][0]:.0f} ≤ {sv.details['external_limit']:.0f} MPa)")

# ── shear h/2 from the support: the external tendons rise towards the anchorage there ──
x = geo["h"] / 2 / 1000
d_x = zg + (d_dev - zg) * x / x_dev
near = section(d_x, s_pe_ex, s_pe)
xQ = L - x
Vd = jtg.basic_combination(g * (L / 2 - x), ((1 + mu) * lanes * (qk * xQ * xQ / (2 * L) + 1.2 * Pk * xQ / L), 0.0))[0]
Vs = jtg.frequent_combination(g * (L / 2 - x), (lanes * (qk * xQ * xQ / (2 * L) + 1.2 * Pk * xQ / L), 0.0))[0]
v = jtg.shear(near, float(Vd), (4, 16, 100), gamma0=gamma0, pc_class="full",
              bent_external=[(Aex, math.degrees(theta_dev))])
print(v, f"(Vpb,ex = {v.details['Vpb_ex']:.0f} kN)")
Ms_x = g * x * (L - x) / 2 + 0.7 * lanes * (qk * x * (L - x) / 2 + Pk * x * (L - x) / L)
print(jtg.inclined_crack_resistance(near, Ms_x, float(Vs), pc_class="full", precast=False,
                                    bent_external=[(Aex, math.degrees(theta_dev))]))

# ── transfer: first-batch losses, self-weight of the girder, before the internal ducts are grouted ──
at_transfer = section(d_dev, s_con_e - lI_e, s_con_i - lI_i, sp0=s_con_i - lI_i)
print(jtg.transient_pc(at_transfer, Mg1, fcu=45, grouted=False))

# ── detailing: φs15.2 strand (wires of about 5 mm), deviator pipes bent to 5 m, positioners halfway ──
print(jtg.external_detailing(wire_d=5.0, radius=5000, free_length=x_dev / 2 * 1000, hoop_d=22, hoop_spacing=80,
                             crossbeam=1200))
