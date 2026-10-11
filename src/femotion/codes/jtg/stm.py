"""Strut-and-tie models of JTG 3362-2018 appendix B (拉压杆模型), for disturbed regions (4.1.9): deep cap beams,
pile caps, anchorage zones, diaphragms, corbels.

Units: mm, mm², MPa; forces kN (tension +). Clauses: docs/codes/JTG_3362-2018.md.

A model is drawn in its plane (x across, z up): nodes, struts and ties, supports and the design loads on the boundary of
the region (B.1.2). solve() finds the member forces from the equilibrium of every node. Many strut-and-tie models are
mechanisms that are in equilibrium only for the loads they were drawn for (the geometry follows the force flow, as in
figure 8-15 of the commentary); they are solved from equilibrium alone, and loads they cannot carry are reported.
Statically indeterminate models are solved by stiffness (femotion.frame, member EA as given, default equal).

    from femotion.codes import jtg
    m = jtg.StrutTie("C35", steel="HRB400")
    A, B = m.node(0, 0, support="xz"), m.node(3000, 0, support="z")
    C = m.node(1500, 1200, load=(0, -5000))
    m.strut(A, C, area=600 * 500)          # effective strut area Acs (figures B.3.3-1 … -3, strut_width)
    m.strut(B, C, area=600 * 500)
    m.tie(A, B, As=12 * 491)
    for r in m.check(gamma0=1.1): print(r)

Checks (B.3): ties γ0 T ≤ fsd As + fpd Ap (B.3.2); struts γ0 C ≤ fce,d Acs (+ f'sd Ass, B.3.3-4) with
fce,d = βc fcd / (0.8 + 170 ε1) ≤ 0.85 βc fcd, ε1 = εs + (εs + 0.002) cot² θs (B.3.3-2, -3), taken with the tie at either
end of the strut that gives the largest ε1 (θs the angle between the strut and that tie; every such angle ≥ 25°, B.2.3);
node faces γ0 S ≤ βn fcd An (B.3.4, table B.3.4) where their areas are given; the surface mesh of B.3.5 when given.
Strut and node areas are the user's (figures B.3.3-1 … -3: strut_width for the width; the thickness out of the plane at
most 6d either side of the anchored bars at a node formed by anchored bars, figure B.3.3-3b).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .materials import Concrete, Rebar, Tendon, concrete as _concrete, rebar as _rebar, tendon as _tendon
from .rc import Check


def beta_c(c: Concrete) -> float:
    """βc (B.3.3): 1.30 for C25–C50, 1.35 for C55–C80"""
    return 1.30 if c.grade <= 50 else 1.35


BETA_N = {"CCC": 0.85, "CCT": 0.75, "CTT": 0.65}  # × βc, table B.3.4


def strut_width(L: float, theta: float, h: float = 0.0) -> float:
    """width of a strut where it meets a node (figures B.3.3-1 … -3, 8.5.4), θ in degrees: L sin θ + h cos θ. CCC node
    under a load and against a strut (B.3.3-1): L = Lb the bearing plate width, h = hs the height of the node held by
    the other strut; CCT node over a bearing (B.3.3-2): L = Lb, h = ha the height held by the bars (from the bearing face
    to 6d above the top bar layer, figure B.3.3-2); CTT node formed by anchored bars (B.3.3-3): L = La the anchorage
    length of the node (bar spacing S plus 6d each side), h = 0. The thickness out of the plane is the user's: at a node
    formed by anchored bars at most 6d either side of each bar (figure B.3.3-3b)."""
    t = math.radians(theta)
    return L * math.sin(t) + h * math.cos(t)


def strut_strength(c: Concrete, eps1: float | None) -> float:
    """fce,d = βc fcd / (0.8 + 170 ε1) ≤ 0.85 βc fcd (B.3.3-2); eps1 None: no tie at either end (CCC nodes, 0.85 βc fcd,
    commentary to B.3.2–B.3.4)"""
    cap = 0.85 * beta_c(c) * c.fcd
    if eps1 is None:
        return cap
    return min(beta_c(c) * c.fcd / (0.8 + 170 * eps1), cap)


def eps1_of(eps_s: float, theta: float) -> float:
    """ε1 = εs + (εs + 0.002) cot² θs (B.3.3-3), θs in degrees"""
    return eps_s + (eps_s + 0.002) / math.tan(math.radians(theta)) ** 2


@dataclass
class _Member:
    i: int
    j: int
    kind: str  # "strut" or "tie"
    area: float = 0.0  # strut: effective area Acs (mm²)
    Ass: float = 0.0  # strut: bars along it (projected area, B.3.3-4)
    ends: tuple = (None, None)  # strut: node face areas An at i and j
    As: float = 0.0
    Ap: float = 0.0
    fpe: float = 0.0  # tie with prestressing steel: effective prestress (MPa)
    decompressed: bool = True
    EA: float = 1.0
    name: str = ""


@dataclass
class StrutTie:
    concrete: Concrete | str | int = "C30"
    steel: Rebar | str = "HRB400"
    tendon: Tendon | str = "strand1860"
    X: list = field(default_factory=list)  # (x, z) mm
    fixed: list = field(default_factory=list)  # per node: (x fixed, z fixed)
    loads: list = field(default_factory=list)  # per node: (Fx, Fz) kN
    bearing: list = field(default_factory=list)  # per node: bearing face area (mm²) or None
    kinds: list = field(default_factory=list)  # per node: node type given by the user, or None
    members: list = field(default_factory=list)
    N: np.ndarray | None = None  # member forces (kN, tension +)
    R: np.ndarray | None = None  # reactions (n_nodes, 2) kN

    def __post_init__(self):
        if not isinstance(self.concrete, Concrete):
            self.concrete = _concrete(self.concrete)
        if not isinstance(self.steel, Rebar):
            self.steel = _rebar(self.steel)
        if not isinstance(self.tendon, Tendon):
            self.tendon = _tendon(self.tendon)

    # ── model ──
    def node(self, x: float, z: float, *, support: str = "", load=(0.0, 0.0), bearing: float | None = None,
             kind: str | None = None) -> int:
        """a node at (x, z) mm; support: "x", "z" or "xz" (reaction components); load (Fx, Fz) kN; bearing: area of the
        bearing face (plate, pile top) for the node check of the reaction or load (its resultant). kind: "CCC", "CCT" or
        "CTT" to set the node type of table B.3.4 (default: from the ties meeting there, reactions and loads taken as
        compressions — give kind where a reaction or load pulls on the node)"""
        if kind is not None and kind not in BETA_N:
            raise ValueError(f"node kind: one of {', '.join(BETA_N)}")
        self.X.append((float(x), float(z)))
        self.fixed.append(("x" in support, "z" in support))
        self.loads.append((float(load[0]), float(load[1])))
        self.bearing.append(bearing)
        self.kinds.append(kind)
        return len(self.X) - 1

    def strut(self, i: int, j: int, *, area: float, Ass: float = 0.0, ends=(None, None), EA: float = 1.0,
              name: str = "") -> int:
        """a concrete strut with effective area Acs (mm², thickness × width at its narrower end, figures B.3.3-1 … -3),
        bars along it Ass (projected on its axis), node face areas An at its ends"""
        self.members.append(_Member(i, j, "strut", area=area, Ass=Ass, ends=tuple(ends), EA=EA, name=name))
        return len(self.members) - 1

    def tie(self, i: int, j: int, *, As: float = 0.0, Ap: float = 0.0, fpe: float = 0.0, decompressed: bool = True,
            EA: float = 1.0, name: str = "") -> int:
        """a tie of reinforcing bars As and / or prestressing steel Ap (fpe its effective prestress; decompressed: the
        concrete round it is decompressed, for εs of the struts, B.3.3)"""
        self.members.append(_Member(i, j, "tie", As=As, Ap=Ap, fpe=fpe, decompressed=decompressed, EA=EA, name=name))
        return len(self.members) - 1

    def _unit(self, k):
        m = self.members[k]
        d = np.subtract(self.X[m.j], self.X[m.i])
        L = float(np.hypot(*d))
        return d / L, L

    # ── statics ──
    def solve(self, tol: float = 1e-6):
        """member forces N (kN, tension +) and reactions R (kN) from nodal equilibrium; statically indeterminate models
        by stiffness (member EA). Raises when the loads cannot be carried (a mechanism for these loads): the out-of-balance
        force left by the best fit must stay below tol × the largest load."""
        n, nm = len(self.X), len(self.members)
        rx = [(a, d) for a, f in enumerate(self.fixed) for d in (0, 1) if f[d]]
        E = np.zeros((2 * n, nm + len(rx)))
        for k, m in enumerate(self.members):
            u, _ = self._unit(k)
            E[2 * m.i:2 * m.i + 2, k] += u
            E[2 * m.j:2 * m.j + 2, k] -= u
        for c, (a, d) in enumerate(rx):
            E[2 * a + d, nm + c] = 1.0
        F = np.array(self.loads, float).ravel()
        rank = np.linalg.matrix_rank(E)
        if rank == E.shape[1]:
            sol, *_ = np.linalg.lstsq(E, -F, rcond=None)
            res = float(np.abs(E @ sol + F).max())
            if res > tol * max(1.0, np.abs(F).max()):
                raise ValueError(f"the model cannot carry these loads (out of balance by {res:.3g} kN, a mechanism for "
                                 "them): add members or supports, or move the nodes along the force flow")
            N, r = sol[:nm], sol[nm:]
        else:
            N, r = self._stiffness(rx, F)
        R = np.zeros((n, 2))
        for c, (a, d) in enumerate(rx):
            R[a, d] = r[c]
        self.N, self.R = N, R
        return N

    def _stiffness(self, rx, F):
        from ...frame import Model, Section, Solver
        fm = Model()
        for x, z in self.X:
            fm.node((x / 1e3, 0.0, z / 1e3))
        for m in self.members:
            fm.truss(m.i, m.j, Section(E=m.EA, A=1.0))
        for a in range(len(self.X)):
            fm.fix(a, (self.fixed[a][0], 1, self.fixed[a][1], 1, 1, 1))
        Fg = np.zeros(fm.ndof)
        for a, (fx, fz) in enumerate(self.loads):
            Fg[6 * a], Fg[6 * a + 2] = fx, fz
        try:
            sv = Solver(fm)
        except RuntimeError as err:
            raise ValueError("the strut-and-tie model is a mechanism") from err
        U = sv.solve(Fg)
        N = sv.truss_forces(U)
        Rg = sv.K @ U - Fg
        return N, np.array([Rg[6 * a + (0 if d == 0 else 2)] for a, d in rx])

    # ── checks ──
    def _node_kind(self, a: int) -> str:
        """CCC, CCT or CTT: as given, else by the directions of the ties meeting at node a (a tie running straight through
        the node counts once; reactions and loads count as compressions)"""
        if self.kinds[a] is not None:
            return self.kinds[a]
        dirs = []
        for k, m in enumerate(self.members):
            if a in (m.i, m.j) and m.kind == "tie":
                u, _ = self._unit(k)
                if not any(abs(abs(float(u @ v)) - 1.0) < 1e-9 for v in dirs):
                    dirs.append(u)
        nt = len(dirs)
        compression = (any(a in (m.i, m.j) and m.kind == "strut" for m in self.members) or any(self.fixed[a])
                       or self.loads[a] != (0.0, 0.0))
        if not compression:
            return "TTT"
        return ("CCC", "CCT", "CTT")[min(nt, 2)]

    def _angle(self, k1: int, k2: int) -> float:
        """acute angle between two members (degrees)"""
        u1, _ = self._unit(k1)
        u2, _ = self._unit(k2)
        return math.degrees(math.acos(min(abs(float(u1 @ u2)), 1.0)))

    def _eps_s(self, k: int) -> float:
        """εs of a tie (B.3.3): bars N/(As Es) with the tie's design force; prestressing steel 0 before the concrete round
        it is decompressed, (fpd − fpe)/Ep after. A tie of both: the larger of the two, the bar strain taken as if the
        bars carried the whole force (program convention: B.3.3 gives only the two pure cases)"""
        m = self.members[k]
        if m.As <= 0 and m.Ap <= 0:
            raise ValueError(f"tie {k}: no steel")
        e = []
        if m.Ap > 0:
            e.append((self.tendon.fpd - m.fpe) / self.tendon.Ep if m.decompressed else 0.0)
        if m.As > 0:
            e.append(max(self.N[k], 0.0) * 1e3 / (m.As * self.steel.Es))
        return max(e)

    def check(self, gamma0: float = 1.0, *, mesh=None) -> list[Check]:
        """B.3 checks of every member and of the node faces whose areas are given (the model is solved again for the
        current loads). mesh: (ratio x, ratio z, spacing mm) of the orthogonal surface mesh of B.3.5 (≥ 0.3 % each way,
        spacing ≤ 300 mm). A bearing face is checked against the resultant reaction (or load) of its node."""
        self.solve()
        c = self.concrete
        out = []
        for k, m in enumerate(self.members):
            N = float(self.N[k])
            label = m.name or f"{m.kind} {k} ({m.i}–{m.j})"
            if m.kind == "tie":
                R = (self.steel.fsd * m.As + self.tendon.fpd * m.Ap) / 1e3
                chk = Check(label, gamma0 * N, R, "kN", "JTG 3362-2018 B.3.2",
                            {"As": m.As, "Ap": m.Ap, "N": N})
                if N < -1e-9:
                    chk.failed.append(f"tie in compression ({N:.1f} kN): the model does not follow the force flow")
            else:
                ties = [(t, a) for a in (m.i, m.j) for t, mt in enumerate(self.members)
                        if mt.kind == "tie" and a in (mt.i, mt.j)]
                eps1, worst = None, None
                angles = []
                for t, a in ties:
                    th = self._angle(k, t)
                    angles.append(th)
                    e1 = eps1_of(self._eps_s(t), th)
                    if eps1 is None or e1 > eps1:
                        eps1, worst = e1, (t, a, th)
                fce = strut_strength(c, eps1)
                R = (fce * m.area + self.steel.fsd_c * m.Ass) / 1e3
                d = {"N": N, "Acs": m.area, "fce_d": fce, "eps1": eps1, "beta_c": beta_c(c)}
                if worst is not None:
                    d.update(theta_s=worst[2], governing_tie=worst[0])
                chk = Check(label, -gamma0 * N, R, "kN",
                            "JTG 3362-2018 B.3.3-4" if m.Ass else "JTG 3362-2018 B.3.3-1", d)
                if N > 1e-9:
                    chk.failed.append(f"strut in tension ({N:.1f} kN): the model does not follow the force flow")
                if angles and min(angles) < 25 - 1e-9:
                    chk.failed.append(f"angle between strut and tie {min(angles):.1f}° < 25° (B.2.3)")
                for a, An in zip((m.i, m.j), m.ends):
                    if An is not None:
                        out.append(self._node_check(a, An, abs(N), gamma0, f"node {a} face of {label}"))
            out.append(chk)
        for a, An in enumerate(self.bearing):
            if An is not None:
                S = float(np.hypot(*self.R[a])) if any(self.fixed[a]) else float(np.hypot(*self.loads[a]))
                out.append(self._node_check(a, An, S, gamma0, f"node {a} bearing face"))
        if mesh is not None:
            rx, rz, s = mesh
            chk = Check.report("surface mesh (B.3.5)", "JTG 3362-2018 B.3.5", {"ratio_x": rx, "ratio_z": rz, "spacing": s})
            if min(rx, rz) < 0.003:
                chk.warnings.append(f"mesh ratio {min(rx, rz):.3%} < 0.3 % (B.3.5)")
            if s > 300:
                chk.warnings.append(f"mesh spacing {s:g} mm > 300 mm (B.3.5)")
            out.append(chk)
        return out

    def _node_check(self, a: int, An: float, S: float, gamma0: float, label: str) -> Check:
        c = self.concrete
        kind = self._node_kind(a)
        if kind == "TTT":
            raise ValueError(f"node {a}: TTT nodes are not in table B.3.4")
        bn = BETA_N[kind] * beta_c(c)
        return Check(label, gamma0 * S, bn * c.fcd * An / 1e3, "kN", "JTG 3362-2018 B.3.4",
                     {"node_type": kind, "beta_n": bn, "An": An})
