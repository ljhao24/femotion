"""Ultimate-limit-state check of a reinforced-concrete girder to the Chinese highway codes (illustrative values).

One T girder of a two-span continuous deck (2 × 16 m, C40, HRB400) carrying Class I lane load:
    frame model → frequencies → influence lines → self-weight and lane-load envelopes → basic combination →
    flexure (sagging and hogging) and shear checks along the girder.

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
