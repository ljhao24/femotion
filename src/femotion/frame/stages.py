"""Construction stages and time-dependent analysis of frames (linear creep and shrinkage, step by step).

Members are cast or erected stage by stage, each with its own concrete age; supports and links are added and released
(system transformation, e.g. simple spans made continuous); every stage brings its loads: the weight of the new members,
superimposed dead loads, nodal loads, tendons (femotion.frame.prestress) with the later part of their losses, imposed
strains and curvatures, bar pretension. Between stages the time is cut into steps growing geometrically, and in each step
the creep and shrinkage increments of every concrete beam are turned into restraint forces and solved on the structure
of that moment.

    from femotion.frame.stages import Stage, analyse
    from femotion.codes.jtg.creep import concrete_creep
    cr = concrete_creep(jtg.concrete(50), RH=70, h=300)            # Appendix C of JTG 3362-2018
    res = analyse(m, [Stage(0, "spans", beams=deck, age=28),       # precast girders on their bearings, own weight
                      Stage(30, "joint", links=[joint]),           # made continuous
                      Stage(40, "surfacing", w={b: (0, 0, -25) for b in deck})], creep=cr, t_end=40 + 3650)
    k = res.index(3690)
    M = res.section_forces(k, chain_sections(deck), "My")       # sagging +, as femotion.frame.post.chain_force
    u = res.displacements(k)                                      # since each node was built: camber, formwork levels

Material law (linear creep, superposition; JTG 3362-2018 4.3.9): the deformation of a concrete beam is
    elastic + creep + shrinkage,  ε(t) = Σ Δσj (1 + φ(t, tj)) / E + εcs(t, tact)
over the stress increments Δσj applied at ages tj since the beam was built, with E the (28-day) modulus of its Section
and φ, εcs from its Creep. On a beam element the "stress" is its set of end forces: an element with uniform section
and creep properties creeps in proportion to its elastic deformation, so the creep of the end-force increments Δpj is
the restrained force −φ Δpj; loads inside the span (self-weight, w) keep their fixed-end part, which creeps without
moving the element ends (only the end-force part is kept in the history).

Integration (a step-by-step method after Bažant, program implementation): within a time step the stress changes
linearly in log(t − ta), ta the last stage (the steps are geometric in t − ta; the first step after a stage follows
(t − ta)^0.3, the start of the creep curves), so the creep coefficient of step j's increment at time t is its average
over the step, φ̄j(t) (Gauss–Legendre for past steps, a rule graded towards the end for the current one, where
φ(t, s) ~ (t − s)^0.3). In step k (tk−1 → tk) every concrete beam gets the effective modulus E″ = E / (1 + φ̄k(tk)) and
the restrained forces
    Δrk = [−Σj<k (φ̄j(tk) − φ̄j(tk−1)) Δpj + Δf_cs,k] / (1 + φ̄k(tk)),
Δf_cs,k the end forces of the step's shrinkage fully restrained (axial tension EA Δεcs), then K″ Δu = residual. Stage
events are steps of zero length (elastic). Second-order accurate in the step size, checked against independent
Volterra solutions with non-parallel creep; the steps grow geometrically from `first` days after each stage with
`per_decade` steps per decade of elapsed time (the defaults, 0.01 d and 8, are within ~1e-4 of the converged
solutions; refine both for convergence studies). Memory: two (steps × beams × 12) arrays (history, end forces).
Steel members (beams without a Creep, bars, links) are elastic. Tendons act through their equivalent loads and do not
add stiffness; the change of their force after transfer comes from the loss function (e.g. JTG 6.2.6, 6.2.7).

Displacements: a node does not move before it is built; `U` holds the displacement since then (what pre-camber and
formwork levels need), and `displacements(k, total=True)` adds the displacement the node inherited when built: the
rigid motion of the built part it was attached to (a cast segment follows the tangent of the cantilever tip), nothing
on a support, and between two ends (a closure, a segment from a tip to a support) linear between them (_inherit).
Supports added later hold the node where it is; released supports hand their reaction back to the structure; removed
members hand their end forces back and take their span loads with them.

Signs and units as femotion.frame (m, kN, kPa, days); section forces as femotion.frame.post.chain_force.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .influence import _fixed_end, _sign
from .linear import _T12, beam_k
from .model import Model, local_axes
from .post import _COMP
from .prestress import Tendon, equivalent_loads
from .thermal import restrained_end_forces


@dataclass
class Creep:
    """time behaviour of one concrete. phi(t, t0): creep coefficient at age t of concrete loaded at age t0; shrinkage(t,
    t0): shrinkage strain from age t0 to age t (shortening +). Both take arrays of ages (days) and must return 0 where
    t ≤ t0. femotion.codes.jtg.creep.concrete_creep gives JTG 3362-2018 Appendix C."""
    phi: Callable
    shrinkage: Callable | None = None


def parallel_creep(phi_base: Callable, shrinkage: Callable | None = None) -> Creep:
    """the rate-of-creep (Dischinger) form φ(t, t′) = φb(t) − φb(t′): every creep curve parallel to the base curve φb
    (a function of age, e.g. the curve of the first loading). The closed forms of JTG 3362-2018 commentary (4-8)…(4-20)
    are exact for it; real concrete loaded later creeps more than this"""
    def phi(t, t0):
        t, t0 = np.broadcast_arrays(np.asarray(t, float), np.asarray(t0, float))
        return np.where(t > t0, np.maximum(phi_base(t) - phi_base(t0), 0.0), 0.0)
    return Creep(phi, shrinkage)


@dataclass
class Stage:
    """what happens at one moment of the construction (time in days on the clock of the analysis)"""
    time: float
    name: str = ""
    beams: Sequence[int] = ()  # beams cast or erected now
    age: float = 7.0  # concrete age of these beams now (d): their creep and shrinkage count from here
    trusses: Sequence[int] = ()  # bars (cables) installed now
    links: Sequence[int] = ()  # links (bearings, joints, hinges) installed now
    remove_beams: Sequence[int] = ()  # removed now: their end forces go back onto the structure, their span loads leave
    remove_trusses: Sequence[int] = ()
    remove_links: Sequence[int] = ()
    fix: dict = field(default_factory=dict)  # node -> six flags: supports added (they hold the node where it is)
    release: dict = field(default_factory=dict)  # node -> six flags: supports released (their reactions go back)
    self_weight: bool = True  # weight of the beams and bars installed now (Section.mass × g)
    w: dict = field(default_factory=dict)  # beam -> global load per length (3,) added now (kN/m), e.g. surfacing
    F: np.ndarray | None = None  # nodal loads added now (ndof; kN, kN·m)
    tendons: Sequence = ()  # Tendon stressed now, or (Tendon, loss): loss(dt) the force lost per element (kN) dt days later
    eps0: dict = field(default_factory=dict)  # bar -> added initial strain (pretension +), e.g. cable tensioning
    strain: dict = field(default_factory=dict)  # beam -> imposed axial strain applied now (elongation +), lasting
    curvature: dict = field(default_factory=dict)  # beam -> imposed curvature applied now (sagging +), lasting
    # strain and curvature stay in the history and creep relaxes their restraint forces: for lasting actions such as
    # differential shrinkage. Temperature is a short-term variable action: solve it elastically on the structure of the
    # moment (femotion.frame.thermal) and combine, rather than putting it here.


@dataclass
class History:
    """states after every step: stage events (elastic) and time steps (creep, shrinkage, later losses)"""
    t: np.ndarray  # time of each state (d)
    stage: np.ndarray  # index of the last stage applied
    event: np.ndarray  # True: the state right after a stage; False: the end of a time step
    U: np.ndarray  # (n, ndof) displacements since each node was built
    forces: np.ndarray  # (n, n_beams, 12) local end forces (section forces as femotion.frame.post); 0 when not built
    N: np.ndarray  # (n, n_trusses) bar forces (tension +)
    R: np.ndarray  # (n, ndof) support reactions (loads the supports apply to the structure)
    built: np.ndarray  # state index at which each node was built (−1: never)
    U0: np.ndarray  # (ndof) displacement each node inherited when it was built
    names: list

    def index(self, t: float, before: bool = False) -> int:
        """the state at time t: after the stages applied at t, or (before=True) at the end of the time step reaching t"""
        hit = np.where(np.abs(self.t - t) <= 1e-9 * max(1.0, abs(t)))[0]
        if not len(hit):
            raise ValueError(f"no state at t = {t}: pass it in analyse(times=...)")
        if before:
            pre = hit[~self.event[hit]]
            return int(pre[-1] if len(pre) else hit[0])
        return int(hit[-1])

    def after(self, stage) -> int:
        """the state right after a stage (its index or name)"""
        s = self.names.index(stage) if isinstance(stage, str) else int(stage)
        k = np.where(self.event & (self.stage == s))[0]
        return int(k[0])

    def section_forces(self, k: int, sections, comp: str) -> np.ndarray:
        """section forces of state k at (beam, "i" | "j") sections, components and signs of femotion.frame.post"""
        a, b = _COMP[comp]
        ef = self.forces[k]
        return np.array([_sign(comp, end) * ef[e][a if end == "i" else b] for e, end in sections])

    def displacements(self, k: int, total: bool = False) -> np.ndarray:
        """(n_nodes, 6) displacements of state k: since each node was built, or (total=True) including what it
        inherited when built"""
        U = self.U[k].copy()
        if total:
            on = np.repeat((self.built >= 0) & (self.built <= k), 6)
            U[on] += self.U0[on]
        return U.reshape(-1, 6)

    def reactions(self, k: int) -> np.ndarray:
        """(n_nodes, 6) support reactions of state k"""
        return self.R[k].reshape(-1, 6)


class _Geometry:
    """what the load helpers of femotion.frame (prestress, thermal) read from a Solver: the model and the beam frames"""

    def __init__(self, m: Model):
        self.m = m
        self._beam_cache = []
        for (i, j, sec, ref, tag) in m.beams:
            L, R = local_axes(m.X[i], m.X[j], ref)
            self._beam_cache.append((L, R, _T12(R)))


def _time_steps(stages, t_end, times, first, per_decade):
    """steps (t, stage index or −1, time of the last stage): every stage is a step of zero length, then steps growing
    geometrically after it"""
    ev = [float(s.time) for s in stages]
    if any(b < a for a, b in zip(ev[:-1], ev[1:])):
        raise ValueError("stages must be in time order")
    t_end = max(float(t_end if t_end is not None else ev[-1]), ev[-1])
    r = 10.0 ** (1.0 / per_decade)
    steps = []
    marks = sorted(set(ev))
    for n, ta in enumerate(marks):
        steps += [(ta, s, ta) for s, te in enumerate(ev) if te == ta]
        tb = marks[n + 1] if n + 1 < len(marks) else t_end
        D = tb - ta
        if D <= 0:
            continue
        d = first * r ** np.arange(int(np.ceil(np.log(max(D / first, 1.0)) / np.log(r))) + 1)
        d = d[d < D / r ** 0.5]
        extra = [x - ta for x in times if ta < x < tb]
        steps += [(ta + di, -1, ta) for di in np.unique(np.r_[d, extra, D])]
    return steps


# quadrature of the creep kernel over a time step. Within a step the stress is taken linear in log(t − ta), ta the last
# stage (the step grid is geometric in t − ta; linear in t in the first step after a stage): Gauss–Legendre for the steps
# behind, where the kernel is smooth, and for the current step a rule graded towards its end, which turns the
# (t − s)^0.3 of the creep curves into a smooth integrand
_GX, _GW = np.polynomial.legendre.leggauss(4)
_GX, _GW = (_GX + 1) / 2, _GW / 2
_CU, _CW = np.polynomial.legendre.leggauss(8)
_CU, _CW = (_CU + 1) / 2, _CW / 2
_P = 10.0 / 3.0
_CW = _CW * _P * _CU ** (_P - 1)
_CX = 1.0 - _CU ** _P  # position in the step (0 at its start, 1 at its end), crowded towards the end


def _in_step(t0, t1, ta, x):
    """times at the fractions x of the step t0 → t1 (stress linear in log(t − ta); linear in t when t0 = ta)"""
    if t0 <= ta:
        return t0 + (t1 - t0) * x ** (1.0 / 0.3)
    return ta + (t0 - ta) * ((t1 - ta) / (t0 - ta)) ** x


def analyse(m: Model, stages: Sequence[Stage], creep: Creep | dict | None = None, t_end: float | None = None, *,
            times: Sequence[float] = (), first: float = 0.01, per_decade: int = 8, g: float = 9.81) -> History:
    """staged, time-dependent linear analysis of the frame model m.
    stages: in time order; m.fixed are the supports present from the start. creep: one Creep for every beam, a dict
    beam -> Creep (beams left out are elastic, e.g. steel), or None (no creep or shrinkage). t_end: last time (d);
    times: extra output times; first, per_decade: the time steps after each stage (see the module docstring)."""
    nb, nt, nl, ndof, nn = len(m.beams), len(m.trusses), len(m.links), m.ndof, len(m.X)
    geo = _Geometry(m)
    cr = [(creep.get(e) if isinstance(creep, dict) else creep) for e in range(nb)]
    # element data
    Tb = np.array([c[2] for c in geo._beam_cache]).reshape(nb, 12, 12)
    kl = np.array([beam_k(sec, c[0]) for (_, _, sec, _, _), c in zip(m.beams, geo._beam_cache)]).reshape(nb, 12, 12)
    kg = np.einsum("eji,ejk,ekl->eil", Tb, kl, Tb)
    bd = np.array([np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6] for (i, j, *_) in m.beams], int).reshape(nb, 12)
    b_rows, b_cols = np.repeat(bd, 12, axis=1), np.tile(bd, (1, 12))
    tu, tk, td_ = np.zeros((nt, 3)), np.zeros((nt, 6, 6)), np.zeros((nt, 6), int)
    for e, (i, j, sec, tag) in enumerate(m.trusses):
        d = m.X[j] - m.X[i]
        L = np.linalg.norm(d)
        tu[e] = d / L
        k3 = sec.E * sec.A / L * np.outer(tu[e], tu[e])
        tk[e] = np.block([[k3, -k3], [-k3, k3]])
        td_[e] = np.r_[6 * i:6 * i + 3, 6 * j:6 * j + 3]
    t_EA_L = np.array([sec.E * sec.A / np.linalg.norm(m.X[j] - m.X[i]) for (i, j, sec, tag) in m.trusses])
    lk = np.array([np.r_[k, kr if kr is not None else (0.0, 0.0, 0.0)] for (_, _, k, kr, _) in m.links], float).reshape(nl, 6)
    ln = np.array([(i, j) for (i, j, *_) in m.links], int).reshape(nl, 2)

    steps = _time_steps(stages, t_end, times, first, per_decade)
    ns = len(steps)
    Tk = np.array([s[0] for s in steps])
    # where and how much of each step's stress increment counts: a stage at its instant, a time step spread linearly
    QS, QW = np.repeat(Tk[:, None], len(_GX), axis=1), np.zeros((ns, len(_GX)))
    QW[:, 0] = 1.0
    for k in range(1, ns):
        if steps[k][1] < 0:
            QS[k] = _in_step(Tk[k - 1], Tk[k], steps[k][2], _GX)
            QW[k] = _GW

    on_b, on_t, on_l = np.zeros(nb, bool), np.zeros(nt, bool), np.zeros(nl, bool)
    sup = np.zeros(ndof, bool)
    for nd, f in m.fixed.items():
        sup[6 * nd:6 * nd + 6] = f
    U, U0 = np.zeros(ndof), np.zeros(ndof)
    built = np.full(nn, -1)
    P = np.zeros((nb, 12))  # end forces from the element deformation (incl. restrained parts)
    FE = np.zeros((nb, 12))  # fixed-end forces of span loads
    N = np.zeros(nt)
    FL = np.zeros((nl, 6))  # link forces k (uj − ui)
    t_weight = np.zeros(nt)  # half weight of each bar applied at its ends
    Fext = np.zeros(ndof)
    dP = np.zeros((ns, nb, 12))  # stress (end force) increments of every step, for the creep history
    groups = {}  # (creep id, cast time, first step) -> dict(beams, creep, Tc, age, j0, prev)
    tendons = []  # (Tendon, loss, time stressed)
    out = dict(U=np.zeros((ns, ndof)), forces=np.zeros((ns, nb, 12)), N=np.zeros((ns, nt)), R=np.zeros((ns, ndof)),
               stage=np.zeros(ns, int))
    cur = -1

    def f_int():
        f = np.zeros(ndof)
        np.add.at(f, bd[on_b].ravel(), np.einsum("eji,ej->ei", Tb[on_b], P[on_b]).ravel())
        for e in np.where(on_t)[0]:
            f[td_[e]] += N[e] * np.r_[-tu[e], tu[e]]
        for e in np.where(on_l)[0]:
            i, j = ln[e]
            f[6 * i:6 * i + 6] -= FL[e]
            f[6 * j:6 * j + 6] += FL[e]
        return f

    def span_load(e, wv):
        if not on_b[e]:
            raise ValueError(f"load on beam {e}, which is not built")
        L, R, T = geo._beam_cache[e]
        fe = _fixed_end(R @ np.asarray(wv, float), L)
        FE[e] += fe
        Fext[bd[e]] += T.T @ fe

    for k, (t, s, ta) in enumerate(steps):
        scale = np.ones(nb)
        node_was = np.zeros(nn, bool)
        node_was[built >= 0] = True
        if s >= 0:  # ── a stage: elastic step ──
            st = stages[s]
            cur = s
            for e in st.remove_beams:
                on_b[e] = False
                Fext[bd[e]] -= Tb[e].T @ FE[e]
            for e in st.remove_trusses:
                on_t[e] = False
                i, j = m.trusses[e][:2]
                Fext[6 * i + 2] += t_weight[e]
                Fext[6 * j + 2] += t_weight[e]
            for e in st.remove_links:
                on_l[e] = False
            for e in st.beams:
                if on_b[e] or P[e].any():
                    raise ValueError(f"beam {e} is built twice")
                on_b[e] = True
                c = cr[e]
                if c is not None:
                    key = (id(c), t - st.age, k)
                    gr = groups.setdefault(key, dict(beams=[], creep=c, Tc=t - st.age, age=st.age, j0=k, prev=np.zeros(0)))
                    gr["beams"].append(e)
                if st.self_weight:
                    span_load(e, (0.0, 0.0, -m.beams[e][2].mass * g))
            for e in st.trusses:
                on_t[e] = True
                if st.self_weight:
                    i, j, sec, tag = m.trusses[e]
                    t_weight[e] = sec.mass * np.linalg.norm(m.X[j] - m.X[i]) / 2 * g
                    Fext[6 * i + 2] -= t_weight[e]
                    Fext[6 * j + 2] -= t_weight[e]
            for e in st.links:
                on_l[e] = True
            for nd, f in st.fix.items():
                sup[6 * nd:6 * nd + 6] |= np.asarray(f, bool)
            for nd, f in st.release.items():
                sup[6 * nd:6 * nd + 6] &= ~np.asarray(f, bool)
            for e, wv in st.w.items():
                span_load(e, wv)
            if st.F is not None:
                Fext += st.F
            for item in st.tendons:
                td, loss = item if isinstance(item, tuple) else (item, None)
                if not on_b[list(td.beams)].all():
                    raise ValueError("a tendon runs along beams that are not built")
                Fext += equivalent_loads(geo, td)
                tendons.append((td, loss, t))
            for e, eps in st.eps0.items():
                N[e] += m.trusses[e][2].E * m.trusses[e][2].A * eps
            imp = sorted(set(st.strain) | set(st.curvature))
            if imp:
                r = restrained_end_forces(geo, imp, [st.strain.get(e, 0.0) for e in imp], [st.curvature.get(e, 0.0) for e in imp])
                P[imp] += r[imp]
                dP[k, imp] += r[imp]
            for gr in groups.values():  # no time passes: the history rows only grow by φ(t, t) = 0
                if gr["j0"] <= k:
                    gr["prev"] = np.r_[gr["prev"], 0.0]
        else:  # ── a time step: creep, shrinkage, later losses ──
            for gr in groups.values():
                ids = [e for e in gr["beams"] if on_b[e]]
                j0, c, Tc = gr["j0"], gr["creep"], gr["Tc"]
                n = k - j0
                past = np.asarray(c.phi(np.full(n * len(_GX), t - Tc), QS[j0:k].ravel() - Tc), float).reshape(n, -1)
                cur_s = _in_step(Tk[k - 1], t, ta, _CX)
                row = np.r_[(past * QW[j0:k]).sum(axis=1),
                            float((np.asarray(c.phi(np.full(len(_CX), t - Tc), cur_s - Tc), float) * _CW).sum())]
                dphi, phikk = row[:-1] - gr["prev"], row[-1]
                gr["prev"] = row
                if not ids:
                    continue
                r = -np.einsum("j,jef->ef", dphi, dP[j0:k, ids])
                if c.shrinkage is not None:
                    sh = np.asarray(c.shrinkage(np.array([t - Tc, Tk[k - 1] - Tc]), np.full(2, gr["age"])), float)
                    de = float(sh[0] - sh[1])
                    if de:
                        r += restrained_end_forces(geo, ids, strain=-de)[ids]
                r /= 1.0 + phikk
                P[ids] += r
                dP[k, ids] += r
                scale[ids] = 1.0 / (1.0 + phikk)
            for td, loss, ts in tendons:
                if loss is None:
                    continue
                dl = np.asarray(loss(t - ts), float) - np.asarray(loss(Tk[k - 1] - ts), float)
                if np.any(dl):
                    Fext += equivalent_loads(geo, Tendon(td.beams, td.ecc, -dl))

        # ── assemble the structure of this step ──
        wb = on_b * scale
        rows, cols, vals = [b_rows[on_b].ravel()], [b_cols[on_b].ravel()], [(kg[on_b] * wb[on_b, None, None]).ravel()]
        for e in np.where(on_t)[0]:
            rows.append(np.repeat(td_[e], 6)); cols.append(np.tile(td_[e], 6)); vals.append(tk[e].ravel())
        for e in np.where(on_l)[0]:
            i, j = ln[e]
            for a in range(6):
                if lk[e, a]:
                    p, q = 6 * i + a, 6 * j + a
                    rows.append(np.array([p, q, p, q])); cols.append(np.array([p, q, q, p]))
                    vals.append(lk[e, a] * np.array([1.0, 1.0, -1.0, -1.0]))
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsc()
        diag = K.diagonal()
        node_on = np.abs(diag.reshape(nn, 6)[:, :3]).max(axis=1) > 0
        fixed = sup | (np.abs(diag) < 1e-9)
        free = np.where(~fixed)[0]
        if s >= 0 and st.F is not None:
            loaded = np.abs(np.asarray(st.F, float)).reshape(nn, 6).max(axis=1) > 0
            if (loaded & ~node_on).any():
                raise ValueError(f"nodal loads on nodes not built: {np.where(loaded & ~node_on)[0].tolist()}")
        # nodes built now inherit the rigid motion of the built part they hang on
        new = np.where(node_on & ~node_was)[0]
        if len(new):
            _inherit(m, new, node_was, sup, on_b, on_l, on_t, ln, U, U0)
            built[new] = k
        res = Fext - f_int()
        dU = np.zeros(ndof)
        if len(free):
            name = stages[cur].name or f"stage {cur}"
            try:
                lu = spla.splu(K[free][:, free])
            except RuntimeError as err:
                raise RuntimeError(f"the structure is unstable at t = {t:g} d (after {name})") from err
            piv = np.abs(lu.U.diagonal())
            if piv.min() < 1e-13 * piv.max():
                raise RuntimeError(f"the structure is unstable at t = {t:g} d (after {name})")
            dU[free] = lu.solve(res[free])
        U += dU
        due = np.einsum("eij,ej->ei", Tb, dU[bd])
        dp = np.einsum("eij,ej->ei", kl, due) * wb[:, None]
        P += dp
        dP[k] += dp
        if nt:
            N += on_t * t_EA_L * np.einsum("ei,ei->e", tu, dU[td_[:, 3:]] - dU[td_[:, :3]])
        if nl:
            FL += on_l[:, None] * lk * (dU.reshape(nn, 6)[ln[:, 1]] - dU.reshape(nn, 6)[ln[:, 0]])
        out["U"][k] = U
        out["forces"][k] = (P - FE) * on_b[:, None]
        out["N"][k] = N * on_t
        R = f_int() - Fext
        R[~fixed] = 0.0
        out["R"][k] = R
        out["stage"][k] = cur
    return History(t=Tk, event=np.array([s[1] >= 0 for s in steps]), built=built, U0=U0,
                   names=[s.name for s in stages], **out)


def _inherit(m: Model, new, node_was, sup, on_b, on_l, on_t, ln, U, U0):
    """displacement inherited by the nodes built now (program convention, independent of the numbering): a new node
    restrained by a support in any translation inherits nothing; every other new node takes, from each source it reaches
    through members built now without passing another source, that source's rigid motion carried to it (translation +
    rotation × arm; through bars the translation only), weighted by the inverse distance along those members. Sources:
    the nodes built before and the supported new nodes (rest, 0). One built neighbour: the tangent of a cantilever; a
    segment from a cantilever tip to a support or a closure between two tips: linear between the two ends."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    nn = len(m.X)
    new = np.asarray(new, int)
    on_sup = sup.reshape(nn, 6)[:, :3].any(axis=1)
    rest = [int(n) for n in new if on_sup[n]]
    free_new = [int(n) for n in new if not on_sup[n]]
    if not free_new:
        return
    Ut = U + U0 * np.repeat(node_was, 6)
    edges = [(i, j, True) for (i, j, *_), on in zip(m.beams, on_b) if on]
    edges += [(int(i), int(j), True) for (i, j), on in zip(ln, on_l) if on]
    edges += [(i, j, False) for (i, j, *_), on in zip(m.trusses, on_t) if on]
    is_new = np.zeros(nn, bool)
    is_new[free_new] = True
    # a source reaches the new nodes only through a member to one of them
    near = {a for i, j, _ in edges for a, b in ((i, j), (j, i)) if is_new[b] and not is_new[a]}
    built = sorted(n for n in near if node_was[n])
    if not built:
        return  # standing on supports only: nothing inherited
    sources = built + sorted(n for n in near if n in set(rest))
    num = np.zeros((nn, 6))
    den_t, den_r = np.zeros(nn), np.zeros(nn)
    for s in sources:
        # graph of the free new nodes and this one source
        r, c, wgt, rigid = [], [], [], False
        for i, j, rg in edges:
            for a, b in ((i, j), (j, i)):
                if is_new[b] and (is_new[a] or a == s):
                    r.append(a), c.append(b), wgt.append(max(np.linalg.norm(m.X[a] - m.X[b]), 1e-9))
                    rigid |= rg and a == s
        if not any(a == s for a in r):
            continue
        d = dijkstra(coo_matrix((wgt, (r, c)), shape=(nn, nn)).tocsr(), indices=s)
        reach = np.where(np.isfinite(d) & is_new)[0]
        if not len(reach):
            continue
        if s in rest:
            u_s, rot = np.zeros(6), False
        else:
            u_s, rot = Ut[6 * s:6 * s + 6], rigid
        wt = 1.0 / np.maximum(d[reach], 1e-9)
        arm = m.X[reach] - m.X[s]
        tr = u_s[:3][None, :] + (np.cross(u_s[3:][None, :], arm) if rot else 0.0)
        num[reach, :3] += wt[:, None] * tr
        den_t[reach] += wt
        if s not in rest and rot:
            num[reach, 3:] += wt[:, None] * u_s[3:][None, :]
            den_r[reach] += wt
    for n in free_new:
        if den_t[n] > 0:
            U0[6 * n:6 * n + 3] = num[n, :3] / den_t[n]
        if den_r[n] > 0:
            U0[6 * n + 3:6 * n + 6] = num[n, 3:] / den_r[n]
