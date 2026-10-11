"""Pier cap, columns and pile caps to JTG 3362-2018 chapter 8 and appendix B (illustrative values).

A two-column bent: a 1.6 × 1.6 m cap beam (C35) over two φ1.5 m columns 6.0 m apart, 2.0 m cantilevers, four bearings;
each column on a 2.0 m deep two-pile cap (C30, φ1.2 m piles 3.6 m apart). Design forces from the basic combination are
given (bearing reactions of the two spans), the frequent ones as a ratio of them:
    cap beam on the rigid frame (8.4.1) → class by l/h (8.4.2) → flexure with the deep-beam lever arm (8.4.3), shear
    (8.4.4, 8.4.5), crack width with C3 (8.4.8) → torsion from the unequal reactions of the two spans (5.5) →
    cantilever top tie (8.4.6), the same cantilever as a strut-and-tie model (appendix B: strut and node checks) →
    local compression under a bearing (5.7) → pile forces (8.5.1), pile cap strut-and-tie (8.5.4), punching (8.5.5) →
    columns: effective lengths (appendix E), biaxial bending of a circular column (5.3.8, appendix F) → a pile's shear
    (3362 has no rule for circles: GB 50010-2010 6.3.15) → a tie beam between the pile caps in tension (5.4.2).

    uv run python examples/jtg_substructure.py
"""
import math

import numpy as np

from femotion.codes import jtg
from femotion.codes.jtg import regions as rg
from femotion.frame import Model, Section, Solver

gamma0 = jtg.GAMMA_0[1]
l, h, b = 6.0, 1.6, 1.6  # cap beam span between column centres, depth, width (m)
cant, D = 2.0, 1.5  # cantilever length, column diameter (m)
H_col = 8.0  # column height to the pile cap (m)
x_bear = [-cant + 0.5, 1.5, l - 1.5, l + cant - 0.5]  # bearings along the cap (m, from the left column)
R_span1, R_span2 = 1500.0, 1100.0  # design reactions of the two spans on each bearing line (kN): the bearings sit
e_bear = 0.45  # 0.45 m either side of the cap's axis, so the difference twists the cap (m)
Fd = R_span1 + R_span2  # vertical design force of one bearing position (kN)
frequent = 0.72  # frequent / basic combination (illustrative)

cap = jtg.rect(b * 1e3, h * 1e3, concrete="C35", bottom=[(20, 28, 75)], top=[(24, 28, 75)])
c = cap.concrete

# ── the bent as a rigid frame (8.4.1): cap beam and columns, fixed at the pile caps ──
Ec = c.Ec * 1e3  # kPa
s_cap = Section(E=Ec, A=b * h, Iy=b * h ** 3 / 12, Iz=h * b ** 3 / 12, J=0.14 * b ** 4)
s_col = Section(E=Ec, A=math.pi * D ** 2 / 4, Iy=math.pi * D ** 4 / 64, Iz=math.pi * D ** 4 / 64, J=math.pi * D ** 4 / 32)
m = Model()
xs = np.unique(np.r_[np.linspace(-cant, l + cant, 41), x_bear, 0.0, l])
cap_nodes = [m.node((x, 0, 0)) for x in xs]
cap_beams = [m.beam(a, bb, s_cap) for a, bb in zip(cap_nodes[:-1], cap_nodes[1:])]
for xc in (0.0, l):
    top = cap_nodes[int(np.argmin(np.abs(xs - xc)))]
    base = m.node((xc, 0, -H_col))
    m.beam(base, top, s_col, ref=(1, 0, 0))
    m.fix(base)
sv = Solver(m)
F = np.zeros(m.ndof)
for xb in x_bear:
    F[6 * cap_nodes[int(np.argmin(np.abs(xs - xb)))] + 2] = -Fd
ef = sv.beam_end_forces(sv.solve(F))
M = np.array([ef[e][4] for e in cap_beams] + [-ef[cap_beams[-1]][10]])  # sagging +, at every cap node
V = np.array([-ef[e][2] for e in cap_beams])
i_mid = int(np.argmin(np.abs(xs - l / 2)))
k_face = int(np.searchsorted(xs, 0.8 * D / 2 + h / 2)) - 1  # the element h/2 from the column face (square of 0.8 D)
print(f"cap beam: l/h = {l / h:.2f} → {rg.cap_beam_class(l * 1e3, h * 1e3)} (8.4.2)")
print(f"  M at mid-span {M[i_mid]:.0f} kN·m, over the column {M[np.argmin(np.abs(xs))]:.0f} kN·m, "
      f"V near the column {V[k_face]:.0f} kN")

# ── cap beam between the columns (8.4.3–8.4.5, 8.4.8) ──
print("  " + str(rg.cap_flexure(cap, M[i_mid], l=l * 1e3, gamma0=gamma0)))
print("  " + str(rg.cap_flexure(cap, M[np.argmin(np.abs(xs))], l=l * 1e3, gamma0=gamma0)))
print("  " + str(rg.cap_shear(cap, abs(V[k_face]), l * 1e3, (6, 16, 100), gamma0=gamma0, alpha1=0.9)))
C3 = rg.cap_beam_C3(l * 1e3, h * 1e3)
print("  " + str(jtg.crack_width(cap, frequent * M[i_mid], 0.8 * frequent * M[i_mid], C3=C3)) + f"  (C3 = {C3:.3f})")

# ── torsion near the column: the unequal reactions of the two spans, e = 0.45 m off the axis (5.5); the same stirrups
# carry the shear of 5.5.4-1 and the torsion (5.5.6), the deep-beam shear check above is 8.4.5 ──
Td = 2 * (R_span1 - R_span2) * e_bear / 2  # two bearing lines between the columns, half to each column
core = (b * 1e3 - 2 * 60, h * 1e3 - 2 * 60)  # inside the stirrups
print("  " + str(jtg.torsion(cap, Td, abs(V[k_face]), stirrups=(6, 16, 100), Ast=18 * math.pi * 28 ** 2 / 4,
                             core=core, gamma0=gamma0, alpha1=0.9)))

# ── cantilever: the bearing 0.5 m from the free end (8.4.6) and the same as a strut-and-tie model (appendix B) ──
bc = 0.8 * D * 1e3  # the column as a square of 0.8 D
x = (cant - 0.5) * 1e3 - bc / 2  # from the bearing to the column edge (mm)
h0 = h * 1e3 - 75
As_top = 24 * math.pi * 28 ** 2 / 4
print(f"\ncantilever (x = {x:.0f} mm ≤ h)")
print("  " + str(rg.cap_cantilever_tie(Fd, x, bc, h0, As_top, h=h * 1e3, gamma0=gamma0)))
z = 0.9 * h0
stm = jtg.StrutTie(c)
P = stm.node(-(x + bc / 2), z, load=(0, -Fd), bearing=600 * 700)  # bearing plate 600 × 700 mm
T = stm.node(0, z, support="x")  # the tie runs on over the column into the span
B = stm.node(0, 0, support="xz", bearing=bc * bc)  # the column's centre at the bottom
stm.tie(P, T, As=As_top, name="top tie")
stm.strut(P, B, area=b * 1e3 * 500, name="diagonal strut")  # 500 mm wide over the cap's width
for r in stm.check(gamma0):
    print("  " + str(r))
print(f"  tie force {stm.N[0]:.0f} kN = (x + bc/2) / z × Fd = {(x + bc / 2) / z * Fd:.0f} kN (8.4.6-2)")

# ── local compression under the bearing of span 1 at the free end (5.7): plate 600 (across) × 400 (along the bridge),
# centred 0.45 m off the cap's axis (plate edge 150 mm from the cap's side) and 0.5 m from the cap's end (200 mm);
# mesh φ12 at 60 mm, 6 layers ──
Ab = jtg.bearing_base_area(600, 400, cx=200, cy=(b * 1e3 / 2 - e_bear * 1e3) - 200)
print("  " + str(jtg.local_bearing(c, R_span1, Al=600 * 400, Ab=Ab, gamma0=gamma0, mesh=(5, 113, 560, 7, 113, 360, 60),
                                    layers=6)))

# ── pile cap under one column (8.5): two piles 3.6 m apart along the bent ──
col_beams = [k for k, bm in enumerate(m.beams) if bm[2] is s_col]
N_col = max(ef[k][0] for k in col_beams) + 25 * 6.0 * 2.0 * 2.0 * 1.2  # column force + cap weight × 1.2 (kN)
M_col = max(abs(ef[k][4]) for k in col_beams)  # the column's base moment (kN·m)
Nd = rg.pile_forces(N_col, 0.0, M_col, [(-1.8, 0.0), (1.8, 0.0)])
print(f"\npile cap: column {N_col:.0f} kN, {M_col:.0f} kN·m → piles {Nd.round(0)} kN (8.5.1)")
h_cap, h0c = 2000.0, 2000.0 - 120
x1 = 1800 - 1200 / 2  # pile centre to the column edge (mm) ≤ h: strut-and-tie (8.5.4)
pc = jtg.concrete("C30")
for r in rg.pile_cap_stm(Nd[0], Nd[1], x1, x1, h0c, b=0.8 * 1200, s=100, d=28, As=22 * 616, bs=2000, concrete=pc,
                         gamma0=gamma0):
    print("  " + str(r))
# punching of the column: the cone reaches the piles, nothing to deduct; ax = column edge to pile edge
print("  " + str(rg.pile_cap_punching_column(N_col, bx=1200, by=1200, ax=x1 - 0.8 * 1200 / 2, ay=h0c, h0=h0c,
                                              concrete=pc, gamma0=gamma0)))

# ── columns (appendix E, 5.3.8, appendix F): across the bridge the bent sways, each column fixed at its pile cap and
# held against rotation at the top by the cap beam (antisymmetric sway: 6 E I / l at each column head), with no lateral
# support (E.0.2, KF = 0); along the bridge each column is a cantilever (the bearings let the deck slide) ──
col = jtg.circle(D * 1e3, 28, 28, 75, concrete="C35")
EI_col, EI_cap = col.concrete.Ec * col.I, c.Ec * cap.I  # N·mm²
l0_t = jtg.effective_length(H_col * 1e3, K_theta=6 * EI_cap / (l * 1e3), K_F=0.0, EI=EI_col)
l0_l = jtg.effective_length(H_col * 1e3, ends="fixed-free")
M_long = 450.0  # longitudinal moment at the column base from braking and bearing friction (illustrative, kN·m)
print(f"\ncolumn: l0 = {l0_t / 1e3:.2f} m across (E.0.2), {l0_l / 1e3:.2f} m along (E.0.1)")
print("  " + str(jtg.biaxial(col, None, N_col, M_col, M_long, l0x=l0_t, l0y=l0_l, gamma0=gamma0)))
print("  " + str(jtg.eccentric(col, N_col, math.hypot(M_col, M_long), max(l0_t, l0_l), gamma0=gamma0, method="table")))

# ── a pile (φ1.2 m, 20Φ25, spiral Φ10 at 150 mm = 2 legs) under a horizontal force (illustrative): 3362 has no shear
# rule for circular sections; b = 1.76 r, h0 = 1.6 r of GB 50010-2010 6.3.15 in the formulas of 5.2.9; compression
# is not counted, tension is deducted ──
pile = jtg.circle(1200, 20, 25, 70, concrete="C30")
print("\npile: " + str(jtg.shear_circle(pile, 420.0, (2, 10, 150), gamma0=gamma0, pile=True, Nd=Nd[0])))
# the same pile pulled by an uplift of 800 kN (illustrative): Vcs − 0.2 γ0 N after GB 50010 6.3.14, whose minimum
# fsv Asv h0 / s ≥ 0.36 ftd b h0 asks for denser hoops (spiral Φ12 at 100 mm)
print("pile in tension: " + str(jtg.shear_circle(pile, 420.0, (2, 12, 100), gamma0=gamma0, pile=True, Nd=-800.0)))

# ── a tie beam between the two pile caps, 0.8 × 1.0 m, pulled apart by the piles' horizontal forces with a moment
# from the caps' rotation (illustrative values): eccentric tension (5.4.2) ──
tie = jtg.rect(800, 1000, concrete="C30", top=[(5, 22, 60)], bottom=[(5, 22, 60)])
for Nt, Mt in ((380.0, 60.0), (380.0, 260.0)):
    r = jtg.tension(tie, Nt, Mt, gamma0=gamma0)
    print(f"tie beam ({r.details['case']}): " + str(r))
