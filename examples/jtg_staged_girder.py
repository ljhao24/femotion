"""Precast girders made continuous, staged and time-dependent, to JTG 3362-2018 (illustrative values).

One post-tensioned T girder of a three-span deck (3 × 30 m, C50, RH 70 %), built as is usual for such decks:
    day 0    girders 7 days old: positive-moment tendons stressed; each girder a simple span under its own weight
    day 60   wet joints over the piers, hogging tendons stressed across them, permanent bearings (one fixed): continuous
    day 90   surfacing and barriers
    … until 10 years after completion, creep and shrinkage of Appendix C, later prestress losses of 6.2.6 and 6.2.7
→ pier moment and reactions at each stage and how creep changes them (the creep secondary moment), against the closed
  forms of the commentary to 4.3.8 (4-19, 4-20) → mid-span deflections of each stage, since built and total → pre-camber
  of 6.5.5 from the staged long-term deflections, against the simplified way of 6.5.3–6.5.5 (ηθ, 2.0).

    uv run python examples/jtg_staged_girder.py

Simplifications (illustrative, not a design): one σpc, ρ and ρps per tendon (at mid-span / over the pier) for σl6; the
joints are stiff links (they do not creep); the girder section is the span section throughout; Ec I of the gross section.
"""
import copy

import numpy as np

from femotion.codes import jtg
from femotion.codes.jtg import creep as jc
from femotion.codes.jtg import prestress as ps
from femotion.frame import Model, Section, Solver
from femotion.frame.prestress import Tendon
from femotion.frame.stages import Stage, analyse

L, n, RH = 30.0, 30, 70.0  # span (m), elements per span, relative humidity (%)
girder = jtg.tee(b=200, h=1800, bf=2400, hf=180, bf_bottom=700, hf_bottom=250, concrete="C50")
c = girder.concrete
A, zg, I = girder.area * 1e-6, girder.zg * 1e-3, girder.I * 1e-12  # m², m from the top, m⁴
h_notional = 2 * girder.area / (2 * 1800 + 2 * 2400)  # 2A/u (mm), girder outline roughly
cr = jc.concrete_creep(c, RH, h_notional)
strand = jtg.tendon("strand", 1860)
aEP = strand.Ep / c.Ec
sec = Section(E=c.Ec * 1e3, A=A, Iy=I, Iz=10 * I, J=0.05, mass=A * 2.6)  # t/m: the girder's own weight

# ── model: each span on its own bearings, two nodes at every pier, the joint a stiff link ──
m = Model()
deck, spans, piers = [], [], []
for s in range(3):
    nodes = [m.node((x, 0, 0)) for x in np.linspace(s * L, (s + 1) * L, n + 1)]
    spans.append([m.beam(a, b, sec) for a, b in zip(nodes[:-1], nodes[1:])])
    deck += spans[-1]
    m.fix(nodes[0], (1, 1, 1, 1, 0, 0))
    m.fix(nodes[-1], (0, 1, 1, 1, 0, 0))
    if s:
        m.link(piers[-1][0], nodes[0], k=(1e13,) * 3, kr=(1e13,) * 3)
        piers[-1] = (piers[-1][0], nodes[0])
    piers.append((nodes[-1], None))
piers = piers[:-1]
mid = [n // 2, n + 1 + n // 2, 2 * (n + 1) + n // 2]  # mid-span nodes

# ── tendons: forces after the instantaneous losses, the later losses of 6.2.6, 6.2.7 and table C.3.1 ──
s_con = strand.sigma_con_max()
xi = np.linspace(0, 1, n + 1)
e_pos = 4 * (girder.h - girder.zg - 150) * 1e-3 * xi * (1 - xi)  # parabola, 150 mm above the bottom at mid-span
An, zn, In = girder.props("net")
Ap_pos, Ap_neg = 3 * 9 * 139.0, 2 * 5 * 139.0  # mm²
s_pos = s_con * 0.88  # after friction, anchorage set and elastic shortening (first batch), illustrative
s_neg = s_con * 0.85


def losses(Ap, s_i, e_mm, M_g=0.0, As=0.0):
    """loss function of one tendon: σl5 of 6.2.6 at σpe = s_i, σl6 of 6.2.7 with σpc at the tendon (N, mm)"""
    l5 = ps.relaxation_loss(strand, s_i)
    s_pc = ps.concrete_stress(s_i * Ap, e_mm, e_mm, An, In, M=M_g)
    rho, rps = ps.steel_ratios(An, In, Ap, abs(e_mm), As, abs(e_mm))
    return jc.later_losses(Ap, l5, cr, 7.0, Ep=strand.Ep, alpha_EP=aEP, sigma_pc=s_pc, rho=rho, rho_ps=rps,
                           fcu_transfer=0.9 * c.fcuk)


Mg_mid = sec.mass * 9.81 * L ** 2 / 8 * 1e6  # N·mm, girder weight on the simple span
td_pos = [Tendon(sp, e_pos, s_pos * Ap_pos / 1e3) for sp in spans]
loss_pos = losses(Ap_pos, s_pos, e_pos.max() * 1e3, Mg_mid)
# hogging tendons: straight, 120 mm below the top, 6 m either side of each joint
e_neg = -(girder.zg - 120) * 1e-3
td_neg = [Tendon(spans[s][-6:] + spans[s + 1][:6], np.full((12, 2), e_neg), s_neg * Ap_neg / 1e3) for s in range(2)]
loss_neg = losses(Ap_neg, s_neg, e_neg * 1e3)

stages = [
    Stage(0.0, "girders stressed", beams=deck, age=7, tendons=[(t, loss_pos) for t in td_pos]),
    Stage(60.0, "made continuous", links=[0, 1], tendons=[(t, loss_neg) for t in td_neg],
          release={p2: (1, 0, 0, 0, 0, 0) for _, p2 in piers}),  # one fixed bearing for the whole deck
    Stage(90.0, "surfacing", w={b: (0.0, 0.0, -12.0) for b in deck}),  # kN/m on this girder
]
t_end = 90.0 + 3650.0
res = analyse(m, stages, creep=cr, t_end=t_end, times=(90 + 365.0,))
el = analyse(m, stages, creep=None, t_end=t_end, times=(90 + 365.0,))  # elastic: the same stages and losses

pier = [(spans[0][-1], "j"), (spans[1][-1], "j")]
span = [(spans[0][n // 2], "i"), (spans[1][n // 2], "i")]
print(f"girder: A = {A:.3f} m², I = {I:.3f} m⁴, h = 2A/u = {h_notional:.0f} mm, φ(10 y, 7 d) = "
      f"{jc.redistribution_phi(cr, t_end + 7, 7):.2f}, εcs = {cr.shrinkage(np.array([t_end + 7]), np.array([7.0]))[0]:.2e}")
print(f"tendons after transfer: positive {td_pos[0].force:.0f} kN, hogging {td_neg[0].force:.0f} kN; lost after 10 years "
      f"{loss_pos(t_end)[()]:.0f} and {loss_neg(t_end - 60)[()]:.0f} kN")
print(f"\n{'state':<28}{'pier M':>9}{'span1 M':>9}{'span2 M':>9}{'R abut':>8}{'R pier':>8}   kN·m, kN")
rows = [("girders stressed (day 0)", res.after(0)), ("before the joints (day 60)", res.index(60, before=True)),
        ("made continuous (day 60)", res.after(1)), ("surfacing (day 90)", res.after(2)),
        ("1 year after completion", res.index(90 + 365)), ("10 years after completion", res.index(t_end))]
for name, k in rows:
    Mp, Ms = res.section_forces(k, pier, "My"), res.section_forces(k, span, "My")
    R = res.reactions(k)
    Rp = R[piers[0][0], 2] + R[piers[0][1], 2]
    print(f"{name:<28}{Mp[0]:9.0f}{Ms[0]:9.0f}{Ms[1]:9.0f}{R[0, 2]:8.0f}{Rp:8.0f}")
k_c, k_e = res.after(2), res.index(t_end)
dM = res.section_forces(k_e, pier, "My") - res.section_forces(k_c, pier, "My")
print(f"creep and shrinkage from completion to 10 years: pier moment {dM[0]:+.0f} kN·m, abutment reaction "
      f"{res.reactions(k_e)[0, 2] - res.reactions(k_c)[0, 2]:+.0f} kN")

# ── the closed forms of the commentary: what was applied before the joints (girder weight, positive tendons) ──
pre = analyse(m, [Stage(0.0, beams=deck, age=7, tendons=[(t, loss_pos) for t in td_pos]),
                  Stage(60.0, links=[0, 1], release={p2: (1, 0, 0, 0, 0, 0) for _, p2 in piers})], creep=cr, t_end=t_end)
m1 = copy.deepcopy(m)
m1.links = []
m2 = copy.deepcopy(m)
for _, p2 in piers:
    m2.fixed[p2][0] = False
phi = jc.redistribution_phi(cr, t_end + 7, 67, 7)  # loaded at 7 d, made continuous at 67 d (4-19, 4-20)
Pe = td_pos[0].force - loss_pos(t_end)[()]
tr = jc.system_transformation(Solver(m1), Solver(m2), pier, "My", phi,
                              tendons=[Tendon(sp, e_pos, Pe) for sp in spans])
M_step = pre.section_forces(pre.index(t_end), pier, "My")[0]
M_cf = tr["dead"].final[0] + tr["prestress"].final[0]
print(f"\nwhat was on the girders before the joints, pier moment at 10 years (φ(t,τ0) − φ(τ,τ0) = {phi:.2f}):")
print(f"  step by step (Appendix C) {M_step:.0f} kN·m;  closed forms 4-19 + 4-20 (Dischinger) {M_cf:.0f} kN·m "
      f"(weight {tr['dead'].final[0]:.0f}, prestress {tr['prestress'].final[0]:.0f}); were the loads applied on the "
      f"continuous girder: {tr['dead'].second[0] + tr['prestress'].second[0]:.0f}")

# ── deflections (upward +, mm) and pre-camber ──
print(f"\n{'mid-span deflection (mm, up +)':<34}{'span 1':>8}{'span 2':>8}{'elastic: span 1':>17}{'span 2':>8}")
for name, k in rows:  # since built (here every deck node is built on day 0, so total = since built)
    u = res.displacements(k)[:, 2] * 1e3
    ue = el.displacements(el.index(res.t[k], before=not res.event[k]))[:, 2] * 1e3
    print(f"  {name:<32}{u[mid[0]]:8.1f}{u[mid[1]]:8.1f}{ue[mid[0]]:17.1f}{ue[mid[1]]:8.1f}")
# the frequent live load (vehicle, ψf 0.7, no impact) on the continuous girder, 0.6 of a lane on this girder
m_f = copy.deepcopy(m2)
sv = Solver(m_f)
q, Pk = jtg.lane_load(L)
eta = jtg.long_term_factor(c)
no_pre = analyse(m, [Stage(0.0, beams=deck), Stage(60.0, links=[0, 1], release=stages[1].release),
                     Stage(90.0, w=stages[2].w)], t_end=t_end)  # elastic, weight and surfacing only
for s_, i in ((0, mid[0]), (1, mid[1])):
    F = np.zeros(m.ndof)
    F[6 * i + 2] = -1.0
    infl = -sv.solve(F)[2::6]  # downward deflection at i per unit load at each node (Maxwell)
    pos = np.maximum(infl, 0.0)
    area = sum(float((pos[a:b][:-1] + pos[a:b][1:]).sum() / 2 * L / n) for a, b in ((0, n + 1), (n + 1, 2 * n + 2), (2 * n + 2, 3 * n + 3)))
    fQ = 0.7 * 0.6 * (q * area + Pk * pos.max()) * 1e3  # mm down
    u_lt = res.displacements(k_e)[i, 2] * 1e3  # staged, 10 years: weight, surfacing, prestress, creep, shrinkage, losses
    staged = max(-(u_lt - eta * fQ), 0.0)
    # simplified (6.5.3–6.5.5): elastic staged results, loads on B0 = 0.95 EcI0 × ηθ, prestress camber × 2.0
    fG = -no_pre.displacements(no_pre.index(t_end))[i, 2] * 1e3 / 0.95
    f_pe = el.displacements(el.index(t_end))[i, 2] * 1e3 + fG * 0.95
    simple = jtg.precamber_pc(eta * (fG + fQ), jtg.CAMBER_LONG_TERM * f_pe)
    print(f"  span {s_ + 1}: frequent live load {fQ:.1f} mm (× ηθ = {eta:.3f}); staged 10-year {u_lt:+.1f} mm → pre-camber "
          f"{staged:.1f} mm;  6.5.3–6.5.5 (dead {fG:.1f} down, prestress {f_pe:.1f} up) → {simple:.1f} mm")
