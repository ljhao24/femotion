"""Bearings, expansion joints and a concrete hinge to JTG 3362-2018 8.6–8.8, 9.7, 9.8 (illustrative values).

A 3 × 30 m continuous girder (five precast beams, made continuous) in a cold region (D60 appendix A): laminated
bearings on the two piers, PTFE sliding laminated bearings on the abutments, an expansion joint at each abutment:
    stiffness of bearings and piers in series → stationary point and the braking force shared by stiffness (D60
    4.3.5) → movements at the pier bearings (temperature, shrinkage, creep with appendix C; 8.7.1 item 5) →
    laminated bearing selection (8.7.3) and sliding (8.7.4) → PTFE bearing at the abutment (8.7.3-1, 8.7.5) →
    joint movements and selection (8.8.2), installation width (8.8.3) → detailing (9.7) → a concrete hinge of a
    small arch (8.6, 9.8.3) → lifting loops of a precast slab (9.8.2).

σc, μ, μf and the plate steel are the designer's values from JT/T 4 (the library has no defaults for them).

    uv run python examples/jtg_devices.py
"""
import numpy as np

from femotion.codes import jtg
from femotion.codes.jtg import devices as dv

SIGMA_C, MU, MU_F, FY_PLATE = 10.0, 0.3, 0.06, 235.0  # MPa, -, -, MPa: designer's values (JT/T 4, plate steel)
spans, n_beams = (30.0, 30.0, 30.0), 5
x_sup = np.r_[0.0, np.cumsum(spans)]  # support lines (m): abutment, pier, pier, abutment
c50 = jtg.concrete("C50")
T_set = (10.0, 20.0)  # °C, installation temperature range of bearings and joints

# ── bearings and piers in series; abutments on PTFE sliding bearings carry no horizontal stiffness (conservative) ──
pier_brg = dv.Laminated(350, 450, t1=8, n=4, ts=3)  # 350 × 450, te = 37 mm
k_pier = 40.0  # kN/mm, a pier's top under a horizontal force (from its own model)
k = np.array([0.0, dv.series(n_beams * pier_brg.k, k_pier), dv.series(n_beams * pier_brg.k, k_pier), 0.0])
x0 = dv.stationary_point(x_sup, k)
F_brake = jtg.braking_force(sum(spans), lanes=2, L0=30.0)  # one direction, two lanes (D60 4.3.5)
F_sup = dv.distribute(F_brake, k)
u_deck = F_brake / k.sum()  # the deck moves as one body
print(f"bearing {pier_brg.la:g} × {pier_brg.lb:g}: te = {pier_brg.te:g} mm, S = {pier_brg.S:.2f}, "
      f"Ee = {pier_brg.Ee:.0f} MPa, k = {pier_brg.k:.2f} kN/mm")
print(f"stationary point x0 = {x0:.1f} m; braking force {F_brake:.0f} kN → {F_sup[1]:.0f} kN on each pier, "
      f"deck moves {u_deck:.1f} mm")

# ── a pier bearing (8.7.3, 8.7.4) ──
Fbk = F_sup[1] / n_beams  # kN on one bearing
mv_pier = dv.movements(abs(x_sup[1] - x0) * 1e3, T_set=T_set, zone="cold", concrete=c50, RH=70, h=250, t0=60,
                       sigma_pc=7.0)  # t0: the beams sit on the bearings at 60 days
# the bearings take the whole movement (conservative; with the pier in series they take k_pier / (k_pier + 5 k_b))
RGk, RQk = 1050.0, 420.0  # kN, self-weight and vehicle with impact (characteristic, 8.7.1 item 3)
print(f"pier bearing, l = {mv_pier.l / 1e3:g} m: expansion {mv_pier.t_plus:.1f} mm; contraction {mv_pier.t_minus:.1f} "
      f"+ shrinkage {mv_pier.shrinkage:.1f} + creep {mv_pier.creep:.1f} mm → Δl = {mv_pier.bearing_shear():.1f} mm")
for chk in dv.laminated_bearing(pier_brg, RGk + RQk, sigma_c=SIGMA_C, dl=mv_pier.bearing_shear(), Fbk=Fbk,
                                theta=0.0015, fy_plate=FY_PLATE):
    print("  " + str(chk))
for chk in dv.laminated_sliding(pier_brg, mu=MU, RGk=RGk, RQk=RQk, dl=mv_pier.bearing_shear(), Fbk=Fbk):
    print("  " + str(chk))

# ── an abutment bearing: PTFE sliding laminated bearing (8.7.3-1, 8.7.5) ──
abut_brg = dv.Laminated(250, 350, t1=8, n=3, ts=3)
RGa, RQa = 420.0, 300.0
print("abutment bearing (PTFE sliding):")
print("  " + str(dv.laminated_bearing(abut_brg, RGa + RQa, sigma_c=SIGMA_C, dl=0.0, fy_plate=FY_PLATE)[0]))
for chk in dv.ptfe_sliding(abut_brg, mu_f=MU_F, RGk=RGa, RQk=RQa):
    print("  " + str(chk))
print("  " + str(dv.bearing_detailing(kind="ptfe", rubber="CR", site_T=(-22, 38), n_longitudinal=1, n_transverse=1)))

# ── expansion joint at an abutment (8.8.2, 8.8.3, 9.7.7) ──
mv_joint = dv.movements((x_sup[-1] - x0) * 1e3, T_set=T_set, zone="cold", concrete=c50, RH=70, h=250, t0=120,
                        sigma_pc=7.0, brake=u_deck)  # installed at 120 days
for C in (80.0, 160.0, 240.0):  # modular joints with one, two, three gaps of 80 mm, Bmin 20 mm per gap
    joint = jtg.expansion_joint(mv_joint, C=C, beta=1.3, Bmin=C / 4, kind="modular")
    if joint.ok:
        break
print(f"joint, l = {mv_joint.l / 1e3:g} m: C⁺ = {joint.details['C_plus']:.1f} mm, "
      f"C⁻ = {joint.details['C_minus']:.1f} mm → modular {C:.0f}")
print("  " + str(joint))
lo, hi = joint.details["installation"]
print(f"  installation width {lo:.0f} … {hi:.0f} mm (8.8.3)")

# ── a concrete hinge at the springing of a small arch (8.6, 9.8.3) ──
Fhd, gamma0 = 2400.0, 1.1
hc = jtg.hinge_compression("C40", Fhd, l=7000, r1=2000, a=600, gamma0=gamma0)
print(f"concrete hinge: contact width b = {hc.details['b']:.1f} mm, β = {hc.details['beta']:.2f}")
print("  " + str(hc))
print("  " + str(jtg.hinge_transverse(Fhd, a=600, h=600, b=hc.details["b"], As=12 * 314.2, steel="HRB400",
                                      gamma0=gamma0, As_side=5 * 314.2)))

# ── lifting loops of a 2.4 t precast deck slab (9.8.2) ──
print(dv.lifting_loop(24.0, 16, n=4, embed=600, inner_diameter=60))
