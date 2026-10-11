"""Precast prestressed girder with a cast-in-place deck: a composite flexural member of JTG 3362-2018 8.1 (illustrative).

A simply supported 30 m post-tensioned I girder (C50, 4 tendons of 9 × φs15.2 strand 1860, class A) at 2.2 m spacing
carries its 180 mm C40 deck slab while it hardens (stage 1), then acts with it under the deck finishes and the Class I
lane load (stage 2):
    effective width of the deck (4.3.3) → the composite section → stresses at transfer (7.2.8, precast girder alone) →
    two-stage design moments and shears (8.1.4, 8.1.5) → flexure of the precast girder and of the composite member →
    shear (lower concrete, α2 = 1.0) and the joint (8.1.6) → normal and inclined crack resistance (8.1.8, 8.1.9) →
    stresses and principal stresses in service (8.1.17) → stiffness, long-term deflection and camber (8.1.13–8.1.15) →
    differential shrinkage taken as a temperature drop in the deck (8.1.3, the drop assumed); what-ifs: the girders made
    continuous over a pier for stage 2 (negative-moment region, deck tendons stressed on the composite girder, 8.1.8) and
    a class B girder (6.3.1 item 1-3 in the two-stage form of 8.1.8, 8.1.9).

    uv run python examples/jtg_composite_girder.py

Simplifications (illustrative, not a design): the effective prestress, σp0 and the transfer losses are given at mid-span
instead of computed along the tendon (see examples/jtg_pc_girder.py for that); the lane load acts on the spine of one
girder through a lateral distribution factor; near the supports the web is thickened to 500 mm (end blocks) and
the tendons rise at 6° through the centroid.
"""
import math

from femotion.codes import jtg

L = 30.0  # computed span (m)
gamma0 = jtg.GAMMA_0[1]
strand = jtg.tendon("strand", 1860)
Ap = 4 * 9 * 139.0
dp = 1380.0  # tendon centroid at mid-span (mm from the precast top)
s_con = strand.sigma_con_max()
s_pe, s_p0, s_transfer = 1050.0, 1120.0, s_con - 120.0  # MPa (assumed: after all losses, at zero concrete stress, at transfer)
pre = jtg.tee(200, 1600, 1600, 160, bf_bottom=700, hf_bottom=300, concrete="C50", bottom=[(8, 16, 50)],
              top=[(10, 12, 40)], tendons=[(Ap, dp, s_p0, 2 * 80)], tendon=strand, sigma_pe=s_pe)
b_deck = jtg.tee_flange_width(b=200, hf=180 + 160, spacing=2200, L=L * 1e3)  # deck and flange together over the web
comp = jtg.composite(pre, b=b_deck, t=180, concrete="C40", top=[(14, 12, 50)])
pre_end = pre.replace(pieces=jtg.tee(500, 1600, 1600, 160, bf_bottom=700, hf_bottom=300).pieces,
                      tendons=[[Ap, 950.0, s_p0, math.pi * 80 ** 2]])  # end block: 500 mm web, tendons near the centroid
comp_end = jtg.composite(pre_end, b=b_deck, t=180, concrete="C40", top=[(14, 12, 50)])
print(f"deck effective width {b_deck:.0f} mm; modulus ratio n = {comp.n:.3f}; composite depth {comp.h:.0f} mm")

# ── actions at mid-span (kN·m) and at h/2 from the supports (kN) ──
g_pre = pre.area * 1e-6 * 26.0  # kN/m
g_deck = 2.2 * 0.18 * 26.0 + 0.6  # slab over the girder spacing + formwork left in place
q_constr = 1.5 * 2.2  # construction load of stage 1
g2 = 2.2 * 0.10 * 24.0 + 2.0  # 100 mm asphalt + barrier share
Mw = lambda w: w * L ** 2 / 8
Vw = lambda w, x: w * (L / 2 - x)
x_v = comp.h / 2e3
M1Gk, M1Qk, M2Gk = Mw(g_pre + g_deck), Mw(q_constr), Mw(g2)
qk, Pk = jtg.lane_load(L)
live = jtg.LiveLoad(L0=L, lanes=1, transverse=1.0, distribution=0.55, mu=jtg.impact_factor(4.0))
M_lane = (qk * L ** 2 / 8 + Pk * L / 4) * live.factor()  # with impact
M_lane0 = M_lane / (1 + live.mu)  # without impact (chapter 6)
V_lane = (qk * (L - x_v) ** 2 / (2 * L) + 1.2 * Pk * (L - x_v) / L) * live.factor()
V1Gk, V1Qk, V2Gk = Vw(g_pre + g_deck, x_v), Vw(q_constr, x_v), Vw(g2, x_v)

# ── at transfer: the precast girder under its own weight, f'cu = 0.9 fcu,k (7.2.8) ──
print("\nprecast girder at transfer")
print("  " + str(jtg.transient_pc(pre.replace(sigma_pe=s_transfer), Mw(g_pre), fcu=45.0)))

# ── ultimate limit state (8.1.4, 8.1.5) ──
M1d, Md = jtg.composite_design_moments(M1Gk, M1Qk, M2Gk, 1.4 * M_lane)
V1d, Vd = jtg.composite_design_moments(V1Gk, V1Qk, V2Gk, 1.4 * V_lane)
print("\nultimate limit state")
print("  precast girder (stage 1): " + str(jtg.flexure(pre, M1d, pc_class="A")))
print("  composite member:         " + str(jtg.flexure(comp.ultimate, Md, gamma0, pc_class="A")))
for r in jtg.composite_shear(comp_end, V1d, Vd, (2, 12, 150), gamma0=gamma0, pc_class="A", bent_tendons=[(Ap, 6.0)]):
    print("  " + str(r))
print("  " + str(jtg.interface_shear(comp_end, Vd, (2, 12, 150), gamma0=gamma0)))

# ── serviceability (8.1.8, 8.1.9): ψf 0.7 for the vehicle, ψq 0.4 ──
M2s, M2l = M2Gk + 0.7 * M_lane0, M2Gk + 0.4 * M_lane0
print("\nserviceability (class A)")
for r in jtg.composite_crack_resistance(comp, M1k=M1Gk + M1Qk, M1Gk=M1Gk, M2s=M2s, M2l=M2l, pc_class="A"):
    print("  " + str(r))
V2s = V2Gk + 0.7 * V_lane / (1 + live.mu)
M_at = lambda w, x: w * x * (L - x) / 2
r = jtg.composite_inclined_crack_resistance(comp_end, M_at(g_pre + g_deck, x_v), V1Gk, M_at(g2, x_v), V2s, pc_class="A",
                                            bent_tendons=[(Ap, 6.0)])
print(f"  {r}  (at {r.details['at']['z']:.0f} mm from the deck top, near the support)")
# in service the construction loads of stage 1 are gone: M1k = M1Gk (8.1.17, 7.1)
print("  " + str(jtg.composite_service_stresses(comp, M1k=M1Gk, M2k=M2Gk + M_lane, pc_class="A")))
V2k = V2Gk + V_lane
r = jtg.composite_principal_compression(comp_end, M_at(g_pre + g_deck, x_v), V1Gk, M_at(g2, x_v), V2k,
                                        stirrups=(2, 12, 150), bent_tendons=[(Ap, 6.0)])
print(f"  {r}  (at {r.details['at']['z']:.0f} mm)")

# ── deflection and camber (8.1.12–8.1.16): stage 2 live load on 0.80 Ec1 I0, camber on Ec1 I01 × 1.75 ──
B2, _ = jtg.composite_stiffness(comp, M2s, pc_class="A")  # N·mm²
Ls = L * 1e3
f_live = 0.7 * live.factor() / (1 + live.mu) * (5 * qk * Ls ** 4 / (384 * B2) + Pk * 1e3 * Ls ** 3 / (48 * B2))
eta = jtg.composite_long_term_factor(pre.concrete)
print(f"  {jtg.deflection_check(eta * f_live, Ls)}  (ηθ = {eta:.3f})")
_, z01, I01 = pre.props("transformed")
camber = 5 * s_pe * Ap * (dp - z01) * Ls ** 2 / (48 * pre.concrete.Ec * I01)  # parabolic tendon, e at the ends 0
print(f"  long-term camber {jtg.COMPOSITE_CAMBER_LONG_TERM * camber:.1f} mm (× {jtg.COMPOSITE_CAMBER_LONG_TERM})")

# ── differential shrinkage (8.1.3) as a uniform drop in the deck; −5 °C is an assumed value, not from the code ──
th = jtg.thermal_stresses(comp.elastic, jtg.shrinkage_difference_profile(comp, -5.0))
print(f"\ndifferential shrinkage (deck −5 °C): deck top {comp.n * th.sigma(0):+.2f} MPa, precast bottom "
      f"{th.sigma(comp.h):+.2f} MPa (compression +)")

# ── what-if: the girders made continuous over a pier after the deck has hardened (2 × 30 m): negative-moment region ──
# Stage 1 is as before (each precast girder simply supported: no moment at the pier); stage 2 (deck finishes and the lane
# load) acts on the continuous composite girders and puts the deck in tension over the pier: −w L²/8 for the uniform
# loads, −0.0962 Pk L for the point load at its worst place (two equal spans, elastic). The deck carries negative-moment
# tendons stressed on the composite girder (2 × 5 φs15.2 at 90 mm, σpe 1000 MPa, flat ducts as 60 mm round ones);
# their secondary moment is left out (illustrative). 8.1.8 places the crack-check edge in the precast girder; the deck's
# own edge is checked the same way with the deck's ftk and no precompression from the girder's tendons (program
# convention).
qk_, Pk_ = jtg.lane_load(L)
M_lane0_p = -(qk_ * L ** 2 / 8 + 0.0962 * Pk_ * L) * live.factor() / (1 + live.mu)
M2Gk_p = -g2 * L ** 2 / 8
M2s_p, M2l_p = M2Gk_p + 0.7 * M_lane0_p, M2Gk_p + 0.4 * M_lane0_p
deck_tendons = [(5 * 139.0, 90.0, 1000.0, 60.0)] * 2
sigma_deck = jtg.composite_prestress_stress(comp_end, deck_tendons)
print(f"\nnegative-moment region over a pier (stage 2: M2s = {M2s_p:.0f}, M2l = {M2l_p:.0f} kN·m)")
bare = jtg.composite_crack_resistance(comp_end, M1k=0.0, M1Gk=0.0, M2s=M2s_p, M2l=M2l_p, pc_class="A")[1]
print(f"  without deck tendons: {bare}")
r = jtg.composite_crack_resistance(comp_end, M1k=0.0, M1Gk=0.0, M2s=M2s_p, M2l=M2l_p, pc_class="A", sigma_pc2=sigma_deck)[1]
print(f"  with deck tendons:    {r}  (deck top precompression {sigma_deck(0.0):.2f} MPa)")
print("  " + str(jtg.composite_service_stresses(comp_end, M1k=0.0, M2k=M2Gk_p + M_lane0_p * (1 + live.mu), pc_class="A",
                                                 sigma_pc2=sigma_deck)))

# ── what-if: a class B girder with 3 of the 4 tendons (6.3.1 item 1-3 in the two-stage form of 8.1.8; 8.1.9) ──
pre_B = pre.replace(tendons=pre.tendons * [0.75, 1, 1, 0.75])
comp_B = jtg.composite(pre_B, b=b_deck, t=180, concrete="C40", top=[(14, 12, 50)])
print("\nclass B what-if (3 of 4 tendons)")
for r in jtg.composite_crack_resistance(comp_B, M1k=M1Gk + M1Qk, M1Gk=M1Gk, M2s=M2s, M2l=M2l, pc_class="B", M2Gk=M2Gk):
    print("  " + str(r))
comp_end_B = jtg.composite(pre_end.replace(tendons=pre_end.tendons * [0.75, 1, 1, 0.75]), b=b_deck, t=180,
                           concrete="C40", top=[(14, 12, 50)])
r = jtg.composite_inclined_crack_resistance(comp_end_B, M_at(g_pre + g_deck, x_v), V1Gk, M_at(g2, x_v), V2s, pc_class="B",
                                            bent_tendons=[(0.75 * Ap, 6.0)])
print(f"  {r}  (class A limits, 6.3.1-7)")
