"""Illustrative cable-stayed bridge as a frame ("fishbone") model. Not a real bridge; dimensions are rounded values
typical of twin-tower cable-stayed bridges below a kilometre: 500 m main span, 210 m side spans (one auxiliary pier
each), inverted-Y concrete towers (top 175 m above water), a 30 m wide steel box deck, two fan-shaped cable planes with
20 pairs per tower side (160 cables).

Coordinates: x along the bridge (0 at mid-span), y across, z up (water level z = 0). Units m, kN, t.
The deck is a single spine; at every cable anchor two rigid arms reach out to the cable planes (the fish bones).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frame.linear import Solver, pretension_influence
from ..frame.model import Model, Section

E_STEEL = 2.06e8
E_CABLE = 1.95e8
E_CONC = 3.45e7

MAIN = 500.0
SIDE = 210.0
XT = MAIN / 2  # tower x (±)
XEND = XT + SIDE
XAUX = XT + 130.0  # auxiliary piers
ZD = 50.0  # deck (girder centroid) level
BHALF = 15.0  # cable plane to bridge axis
NCAB = 20
D_MAIN = [18 + 12 * k for k in range(NCAB)]  # main-span anchor distances from the tower
D_SIDE = [18 + 10 * k for k in range(NCAB)]  # side span
Z_ANCH = [120 + 2.6 * k for k in range(NCAB)]  # tower anchor levels (short cables lowest)
Z_TOP = 175.0
Z_JUNC = 115.0  # where the two legs meet
Y_BASE = 34.0  # transverse position of the leg feet
Z_XB = 46.0  # lower cross beam

DECK = Section(E=E_STEEL, A=1.30, Iy=3.0, Iz=90.0, J=7.0, mass=18.0, rot_mass=18.0 * 75.0)
ARM = Section(E=E_STEEL * 50, A=10.0, Iy=10.0, Iz=10.0, J=10.0)
LEG = Section(E=E_CONC, A=27.8, Iy=203.0, Iz=122.0, J=180.0, mass=27.8 * 2.6)
SHAFT = Section(E=E_CONC, A=37.4, Iy=462.0, Iz=315.0, J=420.0, mass=37.4 * 2.6)
XBEAM = Section(E=E_CONC, A=20.0, Iy=80.0, Iz=80.0, J=60.0, mass=20.0 * 2.6)


def cable_area(k: int) -> float:
    """cable area (m²): about 40 cm² for short cables, 95 cm² for long ones (roughly proportioned to cable force)"""
    return (40 + 55 * (k / (NCAB - 1)) ** 1.2) * 1e-4


@dataclass
class Cable:
    e: int  # bar element id
    tower: int  # -1 / +1
    span: str  # 'main' / 'side'
    k: int  # 0 = closest to the tower
    plane: int  # -1 / +1 (y direction)
    deck_node: int
    tower_node: int
    group: int  # optimisation group (symmetric pairs merged): main k → k, side k → 20 + k


@dataclass
class Bridge:
    m: Model
    deck_nodes: list  # deck spine nodes (sorted by x)
    deck_x: np.ndarray
    cables: list
    towers: dict  # tower -> dict(nodes...)
    anchors: dict  # (deck x) -> (left node, right node)


def build() -> Bridge:
    m = Model()
    # ── deck spine: a node every 2 m ─────────────────────────────────────
    xs = np.arange(-XEND, XEND + 1e-6, 2.0)
    deck = [m.node((x, 0.0, ZD)) for x in xs]
    for a, b in zip(deck[:-1], deck[1:]):
        m.beam(a, b, DECK, ref=(0, 0, 1), tag="deck")
    xi = {round(x, 3): n for x, n in zip(xs, deck)}

    def dn(x):
        return xi[round(x, 3)]

    # ── towers ──────────────────────────────────────────────────────────
    towers = {}
    for s in (-1, 1):
        xT = s * XT
        legs = {}
        for side in (-1, 1):
            # tower legs: foot z=0 → lower cross beam z=46 → junction z=115, straight
            zs = list(np.linspace(0, Z_XB, 6)) + list(np.linspace(Z_XB, Z_JUNC, 10)[1:-1])
            nodes = [m.node((xT, side * Y_BASE * (1 - z / Z_JUNC), z)) for z in zs]
            m.fix(nodes[0])
            legs[side] = nodes
        junc = m.node((xT, 0.0, Z_JUNC))
        for side in (-1, 1):
            ns = legs[side] + [junc]
            for a, b in zip(ns[:-1], ns[1:]):
                m.beam(a, b, LEG, ref=(1, 0, 0), tag="tower")
        # lower cross beam (between the leg nodes at z=46, three nodes in between)
        xl, xr = legs[-1][5], legs[1][5]
        pl, pr = m.X[xl], m.X[xr]
        mids = [m.node(pl + (pr - pl) * t) for t in (0.25, 0.5, 0.75)]
        xbn = [xl] + mids + [xr]
        for a, b in zip(xbn[:-1], xbn[1:]):
            m.beam(a, b, XBEAM, ref=(1, 0, 0), tag="xbeam")
        # the deck rests on the cross beam: fixed vertically and laterally, free longitudinally (semi-floating), torsion held by the bearings
        m.link(dn(xT), mids[1], k=(2e4, 1e9, 1e9), kr=(1e10, 0, 0), tag="bearing")
        # upper shaft: junction → anchors → top
        shaft = [junc] + [m.node((xT, 0.0, z)) for z in Z_ANCH] + [m.node((xT, 0.0, Z_TOP))]
        for a, b in zip(shaft[:-1], shaft[1:]):
            m.beam(a, b, SHAFT, ref=(1, 0, 0), tag="tower")
        towers[s] = dict(legs=legs, junc=junc, shaft=shaft, xbeam=xbn, top=shaft[-1])

    # ── auxiliary and end piers: vertical, lateral and torsional restraint ──
    for x in (-XEND, -XAUX, XAUX, XEND):
        m.fix(dn(x), (0, 1, 1, 1, 0, 0))

    # ── rigid arms + cables ─────────────────────────────────────────────
    anchors = {}
    cables = []
    for s in (-1, 1):
        xT = s * XT
        for span, dlist, sign in (("main", D_MAIN, -s), ("side", D_SIDE, s)):
            for k, d in enumerate(dlist):
                x = xT + sign * d
                key = round(x, 3)
                if key not in anchors:
                    sp_n = dn(x)
                    l = m.node((x, -BHALF, ZD))
                    r = m.node((x, BHALF, ZD))
                    m.beam(sp_n, l, ARM, ref=(0, 0, 1), tag="arm")
                    m.beam(sp_n, r, ARM, ref=(0, 0, 1), tag="arm")
                    anchors[key] = (l, r)
                l, r = anchors[key]
                tn = towers[s]["shaft"][1 + k]
                A = cable_area(k)
                sec = Section(E=E_CABLE, A=A, mass=7.85 * A * 1.15)
                for plane, dnode in ((-1, l), (1, r)):
                    e = m.truss(dnode, tn, sec, tag="cable")
                    cables.append(Cable(e, s, span, k, plane, dnode, tn, k if span == "main" else NCAB + k))
    # longitudinal: a spring at one end pier (the static stiffness of a damper) prevents rigid-body drift
    m.link(dn(-XEND), m.node((-XEND - 1.0, 0, ZD)), k=(5e4, 0, 0), tag="damper")
    m.fix(len(m.X) - 1)
    return Bridge(m=m, deck_nodes=deck, deck_x=xs, cables=cables, towers=towers, anchors=anchors)


def deck_beams(br: Bridge) -> list[int]:
    return [e for e, b in enumerate(br.m.beams) if b[4] == "deck"]


def optimize_pretension(br: Bridge, sv: Solver, U0: np.ndarray, w_tol: float = 0.01, n_tol: float = 0.10, verbose=True):
    """completed-bridge cable forces (rigidly supported continuous beam method + zero displacement):
    1. take the deck alone with a rigid vertical support at every anchor and pier and compute the reactions under self-weight
       (including half the weight of each cable);
    2. split each anchor reaction equally between the two cable planes as the vertical cable component → target forces;
    3. in the full model, solve symmetric groups (40) of initial strains so that under dead load + tensioning the anchor
       deflections and the longitudinal tower-top displacements are ≈ 0 and group forces ≈ targets (weighted least squares,
       displacements normalised by w_tol m and forces by n_tol × target).
    U0: displacements of the full bridge under self-weight alone. Returns (eps_opt per cable, target force T per cable)."""
    m = br.m
    ncab = len(m.trusses)
    deck = np.array(br.deck_nodes)
    db = deck_beams(br)
    tops = [br.towers[s]["top"] for s in (-1, 1)]
    cb = Model()
    cb_nodes = [cb.node(m.X[n]) for n in deck]
    for a, bb in zip(cb_nodes[:-1], cb_nodes[1:]):
        cb.beam(a, bb, m.beams[db[0]][2], ref=(0, 0, 1))
    xs = np.round(br.deck_x, 3)
    sup_x = set(round(c_x, 3) for c_x in (-XEND, -XAUX, -XT, XT, XAUX, XEND))
    anc_x = set(br.anchors.keys())
    for k, x in enumerate(xs):
        if x in sup_x or x in anc_x:
            cb.fix(cb_nodes[k], (0, 1, 1, 1, 0, 0))
    cb.fix(cb_nodes[len(xs) // 2], (1, 0, 0, 0, 0, 0))
    cs = Solver(cb)
    Fcb = cs.gravity()
    # half of each cable's weight goes to its deck anchor (consistent with the lumped masses of the full model)
    idx = {round(x, 3): k for k, x in enumerate(np.round(br.deck_x, 3))}
    for c in br.cables:
        L_ = np.linalg.norm(m.X[c.tower_node] - m.X[c.deck_node])
        Fcb[6 * cb_nodes[idx[round(m.X[c.deck_node][0], 3)]] + 2] -= m.trusses[c.e][2].mass * L_ / 2 * 9.81
    Ucb = cs.solve(Fcb)
    R = cs.K @ Ucb - Fcb  # reactions (upwards positive): K U = F + r
    Rz = {x: R[6 * cb_nodes[k] + 2] for k, x in enumerate(xs)}
    T = np.zeros(ncab)
    for c in br.cables:
        xi_ = round(m.X[c.deck_node][0], 3)
        u = m.X[c.tower_node] - m.X[c.deck_node]
        u /= np.linalg.norm(u)
        T[c.e] = 0.5 * Rz[xi_] / u[2]
    groups = sorted({c.group for c in br.cables})
    members = [[c.e for c in br.cables if c.group == g] for g in groups]
    rep = {g: next(c.e for c in br.cables if c.group == g and c.tower == 1 and c.plane == 1) for g in groups}
    repv = [rep[g] for g in groups]
    N0_ = sv.truss_forces(U0)
    anc_nodes = [deck[list(np.round(br.deck_x, 3)).index(xk)] for xk in sorted(anc_x)]
    rows = [6 * n + 2 for n in anc_nodes] + [6 * t for t in tops]
    Aw, AN = pretension_influence(sv, members, rows=rows)
    B = AN[repv]
    tgt = T[repv]
    cur = N0_[repv]
    w0r = U0[rows]
    sw = np.r_[np.full(len(anc_nodes), 1 / w_tol), np.full(2, 1 / w_tol)]
    sn = 1 / (n_tol * tgt)
    A = np.vstack([Aw * sw[:, None], B * sn[:, None]])
    bvec = np.r_[-w0r * sw, (tgt - cur) * sn]
    x = np.linalg.lstsq(A, bvec, rcond=None)[0]
    eps_opt = np.zeros(ncab)
    for c in br.cables:
        eps_opt[c.e] = 1e-3 * x[groups.index(c.group)]
    if verbose:
        print("target cable forces (kN), main span:", [round(T[rep[g]]) for g in groups[:NCAB]])
        print("target cable forces (kN), side span:", [round(T[rep[g]]) for g in groups[NCAB:]])
    return eps_opt, T


if __name__ == "__main__":
    b = build()
    print("nodes", len(b.m.X), "dofs", b.m.ndof, "beams", len(b.m.beams), "cables", len(b.m.trusses))
