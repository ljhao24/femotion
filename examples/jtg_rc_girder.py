"""Checks of a reinforced-concrete girder to the Chinese highway codes (illustrative values).

One T girder of a two-span continuous deck (2 × 16 m, C40, HRB400) carrying Class I lane load:
    frame model → frequencies → influence lines → self-weight and lane-load envelopes → basic
    combination → flexure (sagging and hogging, the pier moment reduced for the bearing width, 4.3.5) and shear checks →
    the governing inclined section along the girder (5.2.8–5.2.11) → stirrup and bent-up bar design (5.2.13) →
    serviceability: frequent and quasi-permanent combinations (no impact), crack widths, stiffness of every element
    from its frequent moment, long-term deflection and pre-camber.

    uv run python examples/jtg_rc_girder.py
"""
import numpy as np

from femotion.codes import jtg
from femotion.frame import Model, Section, Solver
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces

L, n = 16.0, 32  # span (m), elements per span
gamma0 = jtg.GAMMA_0[2]  # safety class 2

# the section, in mm (effective flange width 1600 mm)
span_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400",
                   bottom=[(10, 28, 110)], top=[(8, 20, 50)])  # two rows of 5
pier_sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", steel="HRB400",
                   bottom=[(6, 28, 95)], top=[(14, 25, 60)])
stirrups_span, stirrups_end, stirrups_pier = (2, 10, 200), (2, 12, 100), (4, 10, 100)

# frame model, in m / kN / t
c = span_sec.concrete
area, I = span_sec.area * 1e-6, span_sec.I * 1e-12
mass = area * 2.6 + 0.55  # girder + pavement and railing share (t/m)
# Ec I: the frequencies for the impact factor take Ec I (JTG D60-2015 4.3.2). JTG 3362 4.3.1 asks for 0.8 Ec I of a
# reinforced concrete member in the force analysis (jtg.frame_section(span_sec, member="rc", …)); on a girder of one
# section and one material it scales all the stiffness alike and leaves the forces from loads unchanged (the forces
# from imposed deformations, temperature gradient or settlement, scale with EI: full Ec I is on the safe side).
sec = Section(E=c.Ec * 1e3, A=area, Iy=I, Iz=10 * I, J=0.01, mass=mass)
m = Model()
xs = np.linspace(0, 2 * L, 2 * n + 1)
nodes = [m.node((x, 0, 0)) for x in xs]
beams = [m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])]
m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
m.fix(nodes[n], (0, 1, 1, 1, 0, 0))
m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
sv = Solver(m)

# frequencies: the first two vertical modes (f1 for sagging and shear, f2 for hogging, as JTG D60-2015 4.3.2 explains)
f, Phi = sv.modes(8)
vert = [k for k in range(8) if np.abs(Phi[2::6, k]).max() > np.abs(Phi[1::6, k]).max()]
f1, f2 = f[vert[0]], f[vert[1]]
mu1, mu2 = jtg.impact_factor(f1), jtg.impact_factor(f2)
print(f"f1 = {f1:.2f} Hz (μ = {mu1:.3f}), f2 = {f2:.2f} Hz (μ = {mu2:.3f})")

secs = chain_sections(beams)
il = InfluenceLines(sv, nodes, secs)

# one girder of a five-girder deck: lateral distribution factor 0.55 (it already counts the lanes)
live = lambda mu: jtg.LiveLoad(L0=L, lanes=1, transverse=1.0, distribution=0.55, mu=mu)
G_M = uniform_load_forces(sv, secs, "My")
G_V = uniform_load_forces(sv, secs, "Vz")
Qs = live(mu1).effects(il, "My")
Qh = live(mu2).effects(il, "My")
Q_M = (Qs[0], Qh[1])  # sagging with μ(f1), hogging with μ(f2)
Q_V = live(mu1).effects(il, "Vz")
Mmax, Mmin = jtg.basic_combination(G_M, Q_M)
Vmax, Vmin = jtg.basic_combination(G_V, Q_V)

x = np.array([xs[b + (e == "j")] for b, e in secs])
print(f"\n{'x (m)':>7} {'Mmax':>9} {'Mmin':>9} {'Vmax':>8} {'Vmin':>8}   kN·m / kN, basic combination (without γ0)")
for k in range(0, len(secs), 8):
    print(f"{x[k]:7.2f} {Mmax[k]:9.1f} {Mmin[k]:9.1f} {Vmax[k]:8.1f} {Vmin[k]:8.1f}")

print("\nchecks")
k_span = int(np.argmax(Mmax))
k_pier = int(np.argmin(Mmin))
k_end = int(np.argmin(np.abs(x - span_sec.h / 2000)))  # JTG 3362 5.2.8: h/2 from the end support centre
k_pierV = int(np.argmax(np.maximum(np.abs(Vmax), np.abs(Vmin))))  # at the pier itself (5.2.8 allows the diaphragm edge)
near_pier = lambda k: abs(x[k] - L) < 0.3 * L  # pier reinforcement over 0.3 L each side
results = [
    (x[k_span], jtg.flexure(span_sec, Mmax[k_span], gamma0)),
    (x[k_pier], jtg.flexure(pier_sec, Mmin[k_pier], gamma0)),
    (x[k_end], jtg.shear(span_sec, max(abs(Vmax[k_end]), abs(Vmin[k_end])), stirrups_end, gamma0=gamma0)),
    (x[k_pierV], jtg.shear(pier_sec, max(abs(Vmax[k_pierV]), abs(Vmin[k_pierV])), stirrups_pier, gamma0=gamma0,
                           hogging=near_pier(k_pierV), alpha1=0.9 if near_pier(k_pierV) else 1.0)),
]
for xx, r in results:
    print(f"  x = {xx:5.2f} m  {r}")
    for note in r.notes:
        print(f"               note: {note}")

# every section along the girder: the worst flexure ratio
ratios = []
for k in range(len(secs)):
    sec_k = pier_sec if near_pier(k) else span_sec
    worst = 0.0
    if Mmax[k] > 0:
        worst = max(worst, jtg.flexure(sec_k, Mmax[k], gamma0).ratio)
    if Mmin[k] < 0:
        worst = max(worst, jtg.flexure(sec_k, Mmin[k], gamma0).ratio)
    ratios.append(worst)
k = int(np.argmax(ratios))
print(f"\nhighest flexure ratio along the girder: {ratios[k]:.3f} at x = {x[k]:.2f} m")

# ── the pier moment reduced for the width of the bearing (4.3.5) ──
# the design reaction that goes with the largest hogging moment: self-weight from the jump of its shear, the lane load
# placed as for Mmin at the pier (q on the negative part of the moment influence line, P at its lowest ordinate)
kL, kR = 2 * (n - 1) + 1, 2 * n  # the sections just left and right of the pier
R_G = abs(G_V[kL] - G_V[kR])
Fu = np.zeros((sv.m.ndof, len(nodes)))
for j, nd in enumerate(nodes):
    Fu[6 * nd + 2, j] = -1.0
R_line = np.array([(sv.K @ sv.solve(Fu[:, j]))[6 * nodes[n] + 2] for j in range(len(nodes))])  # reaction per unit load
s_M, eta_M = il.line(k_pier, "My")
q_lane, P_lane = jtg.lane_load(L)
neg = np.interp(xs, s_M, eta_M) < 0
R_Q = (q_lane * float(np.trapezoid(np.where(neg, R_line, 0.0), xs)) + P_lane * R_line[int(np.argmin(eta_M))]) \
    * live(mu2).factor()
R_d = 1.2 * R_G + 1.4 * R_Q
y_c = (pier_sec.h - pier_sec.zg) / 1e3  # soffit to the centroid axis (m)
a = jtg.spread_length(0.45, y_c)  # 450 mm long bearing, 45° spread up to the centroid
Me, d = jtg.support_moment_reduction(Mmin[k_pier], R_d, a)
Mmin_r = jtg.peak_shaving(x, Mmin, L, R_d, a)
print(f"\npier moment (4.3.5): R = 1.2 × {R_G:.0f} + 1.4 × {R_Q:.0f} = {R_d:.0f} kN, a = {a:.2f} m, "
      f"M' = R a / 8 = {d['M_prime']:.1f} kN·m: "
      f"{Mmin[k_pier]:.1f} → {Me:.1f} kN·m{' (0.9 M governs)' if d['capped'] else ''}")
print(f"  {jtg.flexure(pier_sec, Mmin_r[k_pier], gamma0)}")

# ── the governing inclined section along the girder (5.2.8–5.2.11) ──
Vd = np.maximum(np.abs(Vmax), np.abs(Vmin))
Md = np.where(np.abs(Mmin_r) > np.abs(Mmax), Mmin_r, Mmax)  # the moment of the governing sign as the accompanying one
sections = [(0.0, 0.7 * L, span_sec), (0.7 * L, 1.3 * L, pier_sec), (1.3 * L, 2 * L, span_sec)]
stirrups = [(0.0, 3.0, stirrups_end), (3.0, 0.7 * L, stirrups_span), (0.7 * L, 1.3 * L, stirrups_pier),
            (1.3 * L, 2 * L - 3.0, stirrups_span), (2 * L - 3.0, 2 * L, stirrups_end)]
supports = [(0.0, "end"), (L, "interior", 0.4), (2 * L, "end")]  # a 400 mm diaphragm over the pier
search = jtg.shear_search(x, Vd, Md, sections, stirrups, supports=supports, gamma0=gamma0)
print("\ninclined sections (5.2.8 positions, then the governing one of the search):")
print("\n".join("  " + line for line in str(search).splitlines()))

# ── design of the stirrups and bent-up bars next to the end support (5.2.13) ──
half = x <= 0.5 * L + 1e-9  # the half of the end span next to the end support
x1, V1, M1 = x[half], Vd[half], Md[half]
Vd_h2 = float(np.interp(span_sec.h / 2e3, x1, V1))
st = jtg.stirrup_design(span_sec, Vd_h2, legs=2, dia=10, xi=0.6, gamma0=gamma0)
z = span_sec.h - 110 - 50  # between the bottom and top bar levels (mm)
bb = jtg.bent_bar_design(x1, V1, support=0.0, h=span_sec.h, z=z, xi=0.6, gamma0=gamma0)
print(f"\nnext to the end support, V'd = {Vd_h2:.1f} kN at h/2: {st}")
print(bb)
design = jtg.shear_search(x1, V1, M1, span_sec, (2, 10, st.spacing), supports=[(0.0, "end")], bent=bb.bent(),
                          gamma0=gamma0)
print(f"  inclined sections with these stirrups and bars, governing: {design.governing}")

# ── serviceability (JTG 3362-2018 chapter 6): frequent and quasi-permanent combinations, vehicle load without impact ──
Q0 = live(0.0).effects(il, "My")
Ms_max, Ms_min = jtg.frequent_combination(G_M, Q0)
Ml_max, Ml_min = jtg.quasi_permanent_combination(G_M, Q0)
print("\nserviceability")
ks, kp = int(np.argmax(Ms_max)), int(np.argmin(Ms_min))
for k, Ms, Ml, s_ in [(ks, Ms_max, Ml_max, span_sec), (kp, Ms_min, Ml_min, pier_sec)]:
    print(f"  x = {x[k]:5.2f} m  Ms = {Ms[k]:6.1f}, Ml = {Ml[k]:6.1f} kN·m  {jtg.crack_width(s_, Ms[k], Ml[k], env='I')}")

# stiffness of every element from its frequent moment (6.5.2: B0 below Mcr, the cracked equivalent above), then the
# deflections by structural mechanics on a frame with EI = B (6.5.1)
m2 = Model()
nodes2 = [m2.node((xx, 0, 0)) for xx in xs]
for e in range(len(beams)):
    Me = max((Ms_max[j] if abs(Ms_max[j]) >= abs(Ms_min[j]) else Ms_min[j] for j in (2 * e, 2 * e + 1)), key=abs)
    B = jtg.stiffness(pier_sec if near_pier(2 * e) else span_sec, Me)[0]  # N·mm²
    m2.beam(nodes2[e], nodes2[e + 1], Section(E=c.Ec * 1e3, A=area, Iy=B * 1e-9 / (c.Ec * 1e3), Iz=10 * I, J=0.01,
                                              mass=mass))
m2.fix(nodes2[0], (1, 1, 1, 1, 0, 0))
m2.fix(nodes2[n], (0, 1, 1, 1, 0, 0))
m2.fix(nodes2[-1], (0, 1, 1, 1, 0, 0))
sv2 = Solver(m2)
_, Ug = uniform_load_forces(sv2, secs, None)
fG = -Ug[2::6] * 1e3  # self-weight deflection (mm, down +)
D = np.zeros((len(xs), len(xs)))  # D[i, j]: deflection at node i under a unit load at node j (symmetric)
for j in range(len(xs)):
    F = np.zeros(m2.ndof)
    F[6 * j + 2] = -1.0
    D[:, j] = -sv2.solve(F)[2::6] * 1e3
q, P = jtg.lane_load(L)
dx = np.diff(xs)


def lane_deflection(row):
    pos = np.maximum(row, 0.0)
    return q * float(((pos[:-1] + pos[1:]) / 2 * dx).sum()) + P * pos.max()


fQ = 0.7 * live(0.0).factor() * np.array([lane_deflection(D[i]) for i in range(len(xs))])  # ψf1 = 0.7, no impact
eta = jtg.long_term_factor(c)
i_max = int(np.argmax(fQ[: n + 1]))
print(f"  ηθ = {eta:.2f};  live-load deflection (frequent, long term) at x = {xs[i_max]:.2f} m: "
      f"{jtg.deflection_check(eta * fQ[i_max], L * 1e3)}")
fG_l, fQ_l = eta * fG[i_max], eta * fQ[i_max]
print(f"  long-term deflection: self-weight {fG_l:.1f} mm + frequent live {fQ_l:.1f} mm → pre-camber "
      f"{jtg.precamber_rc(fG_l, fQ_l, L * 1e3):.1f} mm (6.5.5)")
