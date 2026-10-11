"""Creep and shrinkage effects on structures, JTG 3362-2018 4.3.8, 4.3.9, 6.5.6 and Appendix C.

Closed forms of the commentary to 4.3.8 (p163–165) on arrays of any linear effect (moments, shears, reactions):

    C = average_effective_factor(Pe, Pi)                        # (4-7): no system transformation
    M2 = long_term_secondary(M2_transfer, Pe, Pi)               # elastic secondary effect at transfer × C
    phi = redistribution_phi(cr, t=3650, tau=60, tau0=28)       # φ(t, τ0) − φ(τ, τ0); φ(t, τ) when tau0 is None
    Mgt = creep_redistribution(M1g, M2g, phi)                   # (4-12), (4-19): loads applied before the transformation
    Mpt = prestress_creep_redistribution(M1p, M2p_sec, M1p_sec, phi)   # (4-17), (4-18), (4-20)
    tr = system_transformation(sv1, sv2, sections, "My", phi, tendons=[td])   # both structures solved by femotion.frame

They rest on the rate-of-creep (Dischinger) assumption, all creep curves parallel; the step-by-step analysis of
femotion.frame.stages (6.5.6: creep and shrinkage increments of every stage summed with each member's own ages) needs no
such assumption and takes the Appendix C concrete from concrete_creep:

    cr = concrete_creep(jtg.concrete(50), RH=70, h=300)         # femotion.frame.stages.Creep
    loss = later_losses(Ap, l5, cr, t0=7, Ep=…, alpha_EP=…, sigma_pc=…, rho=…, rho_ps=…)   # tendon force lost after transfer

Units: φ dimensionless, ages in days, h in mm, stresses MPa, Ap mm², forces kN. Clauses: docs/codes/JTG_3362-2018.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...frame.influence import section_forces, uniform_load_forces
from ...frame.linear import Solver
from ...frame.post import _COMP
from ...frame.prestress import effects as _prestress_effects
from ...frame.prestress import equivalent_loads
from ...frame.stages import Creep
from . import prestress as ps
from .materials import Concrete


# ── 4-7: no system transformation ───────────────────────────────────────────
def average_effective_factor(Pe, Pi, weights=None) -> float:
    """C = Pe / Pi (commentary 4-7): Pe mean tendon force after all losses, Pi after the instantaneous (first-batch)
    losses. Arrays (force per element) are averaged with the weights, e.g. element lengths (program convention)"""
    Pe, Pi = np.asarray(Pe, float), np.asarray(Pi, float)
    w = np.ones(Pe.shape) if weights is None else np.broadcast_to(np.asarray(weights, float), Pe.shape)
    return float((w * Pe).sum() / (w * Pi).sum())


def long_term_secondary(S_transfer, Pe, Pi, weights=None):
    """total secondary effect of prestress after relaxation, creep and shrinkage on a continuous girder built without
    system transformation: the elastic secondary effect at transfer (forces after the instantaneous losses) × C (4-7)"""
    return np.asarray(S_transfer, float) * average_effective_factor(Pe, Pi, weights)


# ── 4-8…4-20: system transformation ─────────────────────────────────────────
def _phi_fn(creep):
    return creep.phi if isinstance(creep, Creep) else creep


def redistribution_phi(creep, t: float, tau: float, tau0: float | None = None) -> float:
    """exponent of the creep redistribution: φ(t, τ) when the members are loaded and the system transformed at the same
    age τ (4-12, 4-13…4-18), φ(t, τ0) − φ(τ, τ0) when loaded at age τ0 and transformed at τ (4-19, 4-20); t the age
    considered. creep: a femotion.frame.stages.Creep (e.g. concrete_creep) or a function φ(t, t0) of ages (d)"""
    f = _phi_fn(creep)
    one = lambda a, b: float(np.asarray(f(np.array([float(a)]), np.array([float(b)])))[0])  # noqa: E731
    if tau0 is None:
        return one(t, tau)
    if tau < tau0:
        raise ValueError("the transformation (tau) cannot come before the loading (tau0)")
    return one(t, tau0) - one(tau, tau0)


def creep_redistribution(S1, S2, phi):
    """St = S1 + (S2 − S1)(1 − e^−φ) (4-12; 4-19 with φ = φ(t, τ0) − φ(τ, τ0)): S1 effect of a load on the earlier
    structure (simply supported, cantilevers), S2 effect of the same load computed on the later structure"""
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    return S1 + (S2 - S1) * (1.0 - np.exp(-phi))


def prestress_creep_redistribution(S1, S2_sec, S1_sec, phi):
    """Spt = S1 + (S2′ − S1′)(1 − e^−φ) (4-17; 4-20 with φ = φ(t, τ0) − φ(τ, τ0)): S1 total effect of the prestress
    applied on the earlier structure, S2′ and S1′ its elastic secondary effects computed on the later and on the earlier
    structure; S1′ = 0 for a statically determinate earlier structure, which gives 4-18"""
    S1, S2_sec, S1_sec = (np.asarray(a, float) for a in (S1, S2_sec, S1_sec))
    return S1 + (S2_sec - S1_sec) * (1.0 - np.exp(-phi))


@dataclass
class Transfer:
    """one load across a system transformation"""
    first: np.ndarray  # on the earlier structure: M1g (4-12), M1pt (4-15)
    second: np.ndarray  # what creep drives it towards: the same load on the later structure, M2g (4-12), M2pt (4-14)
    final: np.ndarray  # after creep: Mgt, Mpt

    @property
    def creep(self) -> np.ndarray:
        """the creep secondary effect, final − first"""
        return self.final - self.first


def _reactions(sv: Solver, U: np.ndarray, F: np.ndarray, sections) -> np.ndarray:
    R = sv.K @ U - F
    return np.array([R[6 * int(n) + int(d)] for n, d in sections])


def system_transformation(sv1: Solver, sv2: Solver, sections, comp: str, phi: float, *, w=None, tendons=None,
                          g: float = 9.81) -> dict:
    """creep redistribution of what was applied before a system transformation, both structures solved by
    femotion.frame: sv1 the earlier structure, sv2 the later one (the same model with the joints made continuous,
    restraints or supports added). Loads: the self-weight of the beams (or w, global load per length (n_beams, 3)), and
    the tendons (femotion.frame.prestress.Tendon) stressed on the earlier structure. sections: (beam, "i" | "j") for comp
    in N, Vy, Vz, T, My, Mz, or (node, dof 0–5) with comp = "R" for support reactions (a support of the later structure
    only has 0 on the earlier one; supports removed at the transformation are not covered).
    Returns {"dead": Transfer} (4-12, 4-19) and, with tendons, {"prestress": Transfer} (4-17, 4-18, 4-20)."""
    out = {}

    def dead(sv):
        ef, U = uniform_load_forces(sv, [], None, w=w, g=g, with_trusses=False)
        if comp == "R":
            feq = sv.beam_end_forces(U) - ef
            F = np.zeros(sv.m.ndof)
            for e, (i, j, *_) in enumerate(sv.m.beams):
                F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] += sv._beam_cache[e][2].T @ feq[e]
            return _reactions(sv, U, F, sections)
        return section_forces(sv, U, sections, comp, end_forces=ef)

    S1, S2 = dead(sv1), dead(sv2)
    out["dead"] = Transfer(S1, S2, creep_redistribution(S1, S2, phi))
    if tendons:
        if comp == "R":
            def sec_r(sv):
                F = sum(equivalent_loads(sv, t) for t in tendons)
                return _reactions(sv, sv.solve(F), F, sections)  # the primary state has no reactions
            tot1, s1, s2 = sec_r(sv1), sec_r(sv1), sec_r(sv2)
        else:
            if comp not in _COMP:
                raise ValueError(f"comp: one of {list(_COMP)} or 'R'")
            tot1, _, s1 = _prestress_effects(sv1, list(tendons), sections, comp)
            _, _, s2 = _prestress_effects(sv2, list(tendons), sections, comp)
        out["prestress"] = Transfer(tot1, tot1 - s1 + s2, prestress_creep_redistribution(tot1, s2, s1, phi))
    return out


# ── Appendix C concrete for femotion.frame.stages ───────────────────────────
def _rh_beta(c: Concrete, RH: float, method: str) -> float:
    """the RH used in βH: the actual RH with the formulas (C.2.1), 55 % or 80 % with the tables (C.2.3), as
    prestress.creep_coefficient"""
    return RH if method == "formula" and c.grade <= 50 else (55.0 if RH < 70 else 80.0)


def concrete_creep(c: Concrete, RH: float, h: float, *, ts: float = 3.0, method: str = "table",
                   fly_ash: float | None = None) -> Creep:
    """the creep coefficient φ(t, t0) (C.2.1-1, C.2.3, C.2.4) and shrinkage εcs(t, t0) (C.1.3) of Appendix C as a
    femotion.frame.stages.Creep: concrete class c, annual mean relative humidity RH (%), notional size h = 2A/u (mm),
    ts age at the end of curing (3–7 d); method and fly_ash as femotion.codes.jtg.prestress.creep_coefficient. The same
    values as prestress.creep_coefficient and shrinkage_strain, evaluated on arrays: φ0(t0) once per loading age,
    βc(t − t0) on the whole array. Shrinkage before ts is not counted (from max(t0, ts))."""
    rh_b = _rh_beta(c, RH, method)
    bH = ps.beta_H(rh_b, h)
    eps0 = ps.nominal_shrinkage(c, RH, method)
    keys, vals = np.empty(0), np.empty(0)  # φ0(t0) (with the fly-ash factor) of the loading ages seen so far, sorted

    def phi0(t0):
        nonlocal keys, vals
        idx = np.minimum(np.searchsorted(keys, t0), max(len(keys) - 1, 0))
        hit = keys[idx] == t0 if len(keys) else np.zeros(t0.shape, bool)
        if not hit.all():
            new = np.unique(t0[~hit])
            v = [ps.creep_coefficient(c, RH, h, a + 1.0, a, method, fly_ash) / ps.beta_c(1.0, rh_b, h) for a in new]
            keys, vals = np.r_[keys, new], np.r_[vals, v]
            o = np.argsort(keys)
            keys, vals = keys[o], vals[o]
            idx = np.searchsorted(keys, t0)
        return vals[idx]

    def phi(t, t0):
        t, t0 = np.broadcast_arrays(np.asarray(t, float), np.asarray(t0, float))
        out = np.zeros(t.shape)
        ok = t > t0
        if ok.any():
            d = t[ok] - t0[ok]
            out[ok] = phi0(t0[ok]) * (d / (bH + d)) ** 0.3  # C.2.1-6
        return out

    bs = np.vectorize(ps.beta_s, otypes=[float])

    def shrinkage(t, t0):
        t, t0 = np.broadcast_arrays(np.asarray(t, float), np.asarray(t0, float))
        a = np.maximum(t0, ts)
        return np.where(t > a, eps0 * (bs(np.maximum(t - ts, 0.0), h) - bs(a - ts, h)), 0.0)

    return Creep(phi, shrinkage)


def later_losses(Ap, l5, creep: Creep, t0: float, *, Ep: float, alpha_EP: float, sigma_pc, rho, rho_ps,
                 fcu_transfer: float | None = None):
    """the tendon force lost after transfer, as the loss function of femotion.frame.stages: dt (days after stressing at
    concrete age t0) → (σl5 × r(dt) + σl6(t0 + dt)) Ap / 1000 (kN) per element. σl5: the relaxation loss of 6.2.6 (the
    final value, MPa) with r the ratio of table C.3.1 (prestress.relaxation_ratio; from 0 at transfer to 0.50 at 2 d
    linearly, program convention); σl6(t) of 6.2.7 with φ(t, t0) and εcs(t, t0) of `creep` (normally concrete_creep of
    the girder). Ap (mm²), l5, sigma_pc (MPa), rho, rho_ps: one value or one per element. Post-tensioned tendons
    (σlII = σl5 + σl6, table 6.2.8)."""
    Ap, l5, spc, rho, rps = (np.asarray(a, float) for a in (Ap, l5, sigma_pc, rho, rho_ps))

    def loss(dt):
        if dt <= 0:
            return np.zeros(np.broadcast(Ap, l5, spc, rho, rps).shape)
        r = ps.relaxation_ratio(dt) if dt >= 2 else 0.25 * dt
        a, b = np.array([t0 + dt]), np.array([t0])
        phi = float(creep.phi(a, b)[0])
        eps = float(creep.shrinkage(a, b)[0]) if creep.shrinkage is not None else 0.0
        l6 = np.vectorize(lambda s, p, q: ps.creep_shrinkage_loss(Ep, eps, alpha_EP, s, phi, p, q, fcu_transfer))(spc, rho, rps)
        return (l5 * r + l6) * Ap / 1e3
    return loss
