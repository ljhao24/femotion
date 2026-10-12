"""OpenSees back-end for solids (OpenSeesPy ≥ 3.8; extra "opensees"): nonlinear analysis of concrete with smeared cracks —
where cracks appear, how they grow and how wide they open — with embedded reinforcement, prestressing tendons, anchor bolts
and bonded or anchored strips, staged loading and crack-width read-outs. The guide, its validation and the OpenSees
pitfalls met on the way are in docs/continuum-nonlinear.md; this docstring is the summary.

    from femotion.mesh import structured
    from femotion.continuum.opensees import OpsSolid, ConcreteDamage, steel
    m = structured.box(3000, 200, 300, 60, 4, 6)                         # mm
    c = ConcreteDamage.from_strength(fc=38.0, units="N-mm")              # mean strengths, fib Model Code 2010 defaults
    s = OpsSolid(m, c)
    s.bar([(50, 100, 40), (2950, 100, 40)], area=402.0, material=steel(500.0, 2.0e5))    # bonded bar, any path
    s.fix(lambda x: np.isclose(x[0], 100) & np.isclose(x[2], 0), comps=(1, 2))
    s.fix(lambda x: np.isclose(x[0], 2900) & np.isclose(x[2], 0), comps=(2,))
    s.fix(lambda x: np.isclose(x[0], 1500) & np.isclose(x[2], 0), comps=(0,))   # holds the rotation about z too
    F = s.traction(lambda x: np.isclose(x[2], 300) & (abs(x[0] - 1500) < 100), (0, 0, -1.0))
    s.static(s.self_weight(25e-6), steps=2)                             # stage 1, held
    s.push(F, node=(1500, 100, 0), dof=2, target=-6.0, steps=60)         # stage 2, displacement control
    s.crack_widths(); s.cracks_along((0, 100, 1), (3000, 100, 1))       # per cell; individual cracks along a line
    s.export("out/beam.pvd")                                             # ParaView series; s.to_result() for fem-result/1

Units: any consistent set; only the defaults of ConcreteDamage.from_strength need units= ("N-mm" | "kN-m" | "N-m" | "MN-m").

- Solids: hex8 → stdBrick (or bbarBrick), tet4 → FourNodeTetrahedron; materials by cell group: ConcreteDamage
  (ASDConcrete3D, damage-plasticity; its softening laws regularised by the crack band of each cell, Bažant & Oh 1983),
  ElasticIso / ElasticOrtho (or femotion.continuum.Isotropic / Orthotropic).
- Bars (Truss) along any polyline, tied to the cells they cross with the cells' own interpolation (trilinear in
  hexahedra, built from nested ASDEmbeddedNodeElement ties); unbonded ones slide over their deviators; prestress through
  InitStressMaterial. Dowels (anchor bolts) are beams tied to the solid. Supports, ties, springs, Winkler faces, imposed
  displacements (through stiff springs; sub-models from a femotion.continuum CaseResult).
- Stages (static: load control; push: displacement control) accumulate and are held. Concrete integrates with IMPL-EX by
  default (two Newton iterations per step; holds and prestress are settled without extrapolation), or implicitly.
- Cracks: the inelastic strain ε − C₀⁻¹σ per cell gives the crack strain and normal; width w = ε_cr · h; along a line
  the cracks are cut at the valleys of the profile. Widths are of the size of code formulas, not predictions (smeared
  cracks, perfect bond).

OpenSees holds one global model, shared with femotion.frame.opensees: only the model built last is valid, calling an older
one raises. Licence: OpenSees is free for education, research and non-profit use (copyright The Regents of the University
of California); commercial use needs a licence from the Regents. ASDConcrete3D and ASDEmbeddedNodeElement are part of it.
"""
from __future__ import annotations

import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..mesh.core import Mesh, orient
from ..post.stress import principal_3d
from .embed import resample
from .materials import Isotropic, Orthotropic

LOG = os.path.join(tempfile.gettempdir(), "femotion_opensees_solid.log")

# stress unit → MPa, force/length unit → N/mm (for the empirical defaults of ConcreteDamage.from_strength)
UNITS = {"N-mm": (1.0, 1.0), "MN-m": (1.0, 1e3), "kN-m": (1e-3, 1.0), "N-m": (1e-6, 1e-3)}

_HEX_SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
# a hexahedron (VTK order) as six tetrahedra around its 0-6 diagonal
HEX_TETS = np.array([(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)])
_G2 = np.array([-1.0, 1.0]) / math.sqrt(3.0)
# integration points of stdBrick / bbarBrick in their order: point 4i + 2j + k at (g_i, g_j, g_k)
_GAUSS8 = np.array([[a, b, c] for a in _G2 for b in _G2 for c in _G2])


def _ops():
    from ..frame.opensees import _ops as ops  # one lazy import of OpenSeesPy for the whole package

    return ops()


def _registry() -> list:
    from ..frame.opensees import _ACTIVE  # shared with OpsFrame: OpenSees has a single global domain

    return _ACTIVE


# ── materials ────────────────────────────────────────────────────────────
@dataclass
class ElasticIso:
    """isotropic linear elasticity (nDMaterial ElasticIsotropic); alpha: coefficient of thermal expansion (per degree,
    used by OpsSolid.thermal)"""
    E: float
    nu: float
    rho: float = 0.0
    alpha: float = 0.0


@dataclass
class ElasticOrtho:
    """orthotropic linear elasticity in the global axes, with the parameters of OpenSees' nDMaterial ElasticOrthotropic:
    E = (Ex, Ey, Ez), nu = (νxy, νyz, νzx), G = (Gxy, Gyz, Gzx), where νij is the contraction along j under stress along i
    (so νzx is the contraction along x under stress along z, = ν13·E3/E1 in the notation of
    femotion.continuum.Orthotropic, whose nu = (ν12, ν13, ν23) and G = (G12, G13, G23)). OpsSolid also takes
    femotion.continuum.Isotropic and Orthotropic (material axes along the global ones) and converts them.
    alpha = (αx, αy, αz): coefficients of thermal expansion (per degree, used by OpsSolid.thermal)"""
    E: tuple
    nu: tuple
    G: tuple
    rho: float = 0.0
    alpha: tuple = (0.0, 0.0, 0.0)


def _native(m):
    """materials of femotion.continuum (Isotropic, Orthotropic with axes along the global ones) as OpenSees ones"""
    if isinstance(m, Isotropic):
        return ElasticIso(m.E, m.nu, alpha=m.alpha)
    if isinstance(m, Orthotropic):
        if not np.allclose(np.abs(m.axes), np.eye(3), atol=1e-9):
            raise ValueError("Orthotropic with rotated axes: OpenSees' ElasticOrthotropic works in the global axes only")
        E1, E2, E3 = m.E
        n12, n13, n23 = m.nu
        G12, G13, G23 = m.G
        return ElasticOrtho(E=(E1, E2, E3), nu=(n12, n23, n13 * E3 / E1), G=(G12, G23, G13), alpha=tuple(m.alpha))
    return m


def _area(w, s) -> float:
    return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(w)))


@dataclass
class ConcreteDamage:
    """concrete for ASDConcrete3D, with laws defined independently of the mesh.
    t_pre: pre-peak tension (strain, stress) after the elastic line, ending at the peak (ε_t,peak, ft);
    t_post: post-peak tension (crack opening w from 0 at the peak, stress), area = Gf;
    c_pre: pre-peak compression (strain, stress) as positive numbers, ending at the peak (εc1, fc);
    c_post: post-peak compression (crushing displacement w, stress), area to the residual = Gc.
    plastic_t / plastic_c: share of the post-peak (tension) or inelastic (compression) strain that stays on unloading,
    the rest is stiffness damage. implex (default): IMPL-EX integration (Oliver et al. 2008): the damage of a step is
    extrapolated from the previous step, so each step is linear (two Newton iterations) and never fails to converge;
    the error is of the order of one step, so use steps small enough that a step changes the cracking little (≈ 100+ to
    failure). implex=False: implicit secant iterations, exact at convergence but slow and fragile once cracks spread;
    tangent: numerical tangent instead of the secant (implicit only); crack_planes: (nct, ncc, smoothing angle °) for anisotropic
    (multi-directional) cracking; Kc, cdf, eta: ASDConcrete3D parameters (triaxial shape, dilatancy, viscosity).
    alpha: coefficient of thermal expansion per degree (OpsSolid.thermal), 1e-5 as in JTG D60-2015 4.3.12 and
    EN 1992-1-1 3.1.3(5)."""
    E: float
    nu: float
    ft: float
    fc: float
    Gf: float
    Gc: float
    t_pre: tuple = ()
    t_post: tuple = ()
    c_pre: tuple = ()
    c_post: tuple = ()
    plastic_t: float = 0.2
    plastic_c: float = 0.7
    rho: float = 0.0
    implex: bool = True
    tangent: bool = False
    crack_planes: tuple | None = None
    Kc: float = 2.0 / 3.0
    cdf: float = 0.0
    eta: float = 0.0
    alpha: float = 1.0e-5

    @classmethod
    def from_strength(cls, fc: float, ft: float | None = None, E: float | None = None, nu: float = 0.2,
                      Gf: float | None = None, Gc: float | None = None, *, units: str = "N-mm",
                      softening: str = "mc2010", prepeak: str = "mc2010", eps_c1: float | None = None,
                      residual_c: float = 0.1, **kw) -> "ConcreteDamage":
        """concrete from its mean compressive strength fc (= fcm; analyses of cracking use mean values). Defaults, all in
        MPa and N/mm and converted with `units`:
        - ft = fctm, fib Model Code 2010 eq. 5.1-3a/b with fck = fcm − 8: 0.3 fck^(2/3) up to C50, else 2.12 ln(1 + 0.1 fcm);
        - E = Eci = 21 500 (fcm/10)^(1/3) (MC2010 eq. 5.1-21, quartzite aggregate);
        - Gf = 73 fcm^0.18 N/m (MC2010 eq. 5.1-9); Gc = 250 Gf (Rijkswaterstaat RTD 1016-1:2017, after Nakamura and Higai);
        - εc1 = 0.7 fcm^0.31 ‰ ≤ 2.8 ‰ (EN 1992-1-1 table 3.1).
        softening: "mc2010" (bilinear, MC2010 eq. 5.1-10: 0.2 ft at w1 = Gf/ft, zero at 5 w1), "hordijk" (exponential,
        Hordijk 1991, wc = 5.136 Gf/ft) or "linear" (zero at 2 Gf/ft); every branch is scaled to an area of exactly Gf.
        prepeak (tension): "mc2010" (linear to 0.9 ft, then to ft at 0.15 ‰, eq. 5.1-8) or "linear".
        Compression: linear to 0.4 fc, Sargin's curve (MC2010 eq. 5.1-26) to fc at εc1, then linear to residual_c·fc over
        an area Gc, constant beyond. Extra keywords go to the dataclass (plastic_t, implex, crack_planes, rho, …).
        With ν = 0.2 a crack band restrained by elastic neighbours dissipates about 0.93 Gf (ASDConcrete3D's equivalent
        strain comes from the effective stress); pass Gf = Gf_target / 0.933 where the band's energy should be Gf."""
        if units not in UNITS:
            raise ValueError(f"units: one of {', '.join(UNITS)}")
        ks, kg = UNITS[units]
        fcm = fc * ks  # MPa
        if ft is None:
            fck = fcm - 8.0
            ft = (0.3 * max(fck, 0.0) ** (2 / 3) if fck <= 50 else 2.12 * math.log(1 + 0.1 * fcm)) / ks
        if E is None:
            E = 21500.0 * (fcm / 10.0) ** (1 / 3) / ks
        if Gf is None:
            Gf = 0.073 * fcm**0.18 / kg
        if Gc is None:
            Gc = 250.0 * Gf
        # tension
        e_lin = ft / E
        if prepeak == "mc2010":
            e_pk = max(1.5e-4, 1.1 * e_lin)
            t_pre = ((0.9 * e_lin, 0.9 * ft), (e_pk, ft))
        elif prepeak == "linear":
            t_pre = ((e_lin, ft),)
        else:
            raise ValueError("prepeak: 'mc2010' or 'linear'")
        res = 1e-3 * ft
        if softening == "mc2010":
            w1 = Gf / ft
            w, s = np.array([0.0, w1, 5 * w1]), np.array([ft, 0.2 * ft, res])
        elif softening == "hordijk":
            wc = 5.136 * Gf / ft
            x = np.linspace(0, 1, 25) ** 1.5
            c1, c2 = 3.0, 6.93
            s = ft * ((1 + (c1 * x) ** 3) * np.exp(-c2 * x) - x * (1 + c1**3) * math.exp(-c2))
            w, s = wc * x, np.maximum(s, res)
        elif softening == "linear":
            w, s = np.array([0.0, 2 * Gf / ft]), np.array([ft, res])
        else:
            raise ValueError("softening: 'mc2010', 'hordijk' or 'linear'")
        w = w * Gf / _area(w, s)  # exactly Gf
        t_post = tuple(zip(w.tolist(), s.tolist()))
        # compression
        if eps_c1 is None:
            eps_c1 = min(0.7 * fcm**0.31, 2.8) * 1e-3
        k = E * eps_c1 / fc
        if k <= 1.05:
            eps_c1 = 1.05 * fc / E
            k = 1.05
        sargin = lambda eta: fc * (k * eta - eta**2) / (1 + (k - 2) * eta)  # noqa: E731
        lo, hi = 1e-9, 1.0
        for _ in range(60):  # η where Sargin's curve reaches 0.45 fc
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if sargin(mid) < 0.45 * fc else (lo, mid)
        etas = np.linspace(hi, 1.0, 8)
        c_pre = ((0.4 * fc / E, 0.4 * fc),) + tuple((float(eta * eps_c1), float(sargin(eta))) for eta in etas)
        r = residual_c
        c_post = ((0.0, fc), (2 * Gc / ((1 + r) * fc), r * fc))
        return cls(E=E, nu=nu, ft=ft, fc=fc, Gf=Gf, Gc=Gc, t_pre=t_pre, t_post=t_post, c_pre=c_pre, c_post=c_post, **kw)

    # derived quantities
    @property
    def eps_t_peak(self) -> float:
        return self.t_pre[-1][0] if self.t_pre else self.ft / self.E

    @property
    def eps_in_peak(self) -> float:
        """inelastic tensile strain at the peak (pre-peak micro-cracking), subtracted from the crack strain"""
        return self.eps_t_peak - self.ft / self.E

    @property
    def eps_c_peak(self) -> float:
        return self.c_pre[-1][0] if self.c_pre else 2 * self.fc / self.E

    def h_max(self) -> float:
        """largest crack band without snap-back of the material law: min E·Δw/Δσ over the softening segments"""
        out = math.inf
        for post in (self.t_post, self.c_post):
            w = np.array([p[0] for p in post])
            s = np.array([p[1] for p in post])
            drop = s[:-1] - s[1:]
            ok = drop > 0
            if ok.any():
                out = min(out, float(np.min(self.E * np.diff(w)[ok] / drop[ok])))
        return out

    def laws(self, h: float) -> dict:
        """the ASDConcrete3D lists Te, Ts, Td, Ce, Cs, Cd for a crack band h (h above h_max is replaced by 0.99 h_max)"""
        h = min(h, 0.99 * self.h_max())
        E = self.E

        def build(pre, post, peak_e, peak_s, plastic, tension):
            e = [0.0] + [p[0] for p in pre]
            s = [0.0] + [p[1] for p in pre]
            e_pre = peak_e - peak_s / E  # inelastic strain at the peak
            for w, sig in post[1:]:
                e.append(sig / E + e_pre + w / h)
                s.append(sig)
            e, s = np.array(e), np.array(s)
            ein = np.maximum(e - s / E, 0.0)
            if tension:  # pre-peak: plastic (as ASDConcrete3D's own laws); post-peak: plastic share of the crack strain
                epl = np.where(np.arange(len(e)) <= len(pre), ein, e_pre + plastic * (ein - e_pre))
            else:
                epl = plastic * ein
            q = E * (e - epl)
            d = np.where(q > 0, 1 - s / np.where(q > 0, q, 1), 0.0)
            d[0] = 0.0
            # damage up to 1 − 1e-9: the effective stress q = σ/(1 − d) of the last points (σ ≈ 1e-3 ft) must stay what the
            # plastic share asks for, or unloading from wide cracks would be almost all plastic
            return e.tolist(), s.tolist(), np.clip(d, 0.0, 1.0 - 1e-9).tolist()

        Te, Ts, Td = build(self.t_pre, self.t_post, self.eps_t_peak, self.ft, self.plastic_t, True)
        Ce, Cs, Cd = build(self.c_pre, self.c_post, self.eps_c_peak, self.fc, self.plastic_c, False)
        return dict(Te=Te, Ts=Ts, Td=Td, Ce=Ce, Cs=Cs, Cd=Cd, h=h)

    def ops_args(self, h: float) -> list:
        L = self.laws(h)
        a = [self.E, self.nu, "-rho", self.rho, "-Kc", self.Kc, "-cdf", self.cdf]
        for k in ("Te", "Ts", "Td", "Ce", "Cs", "Cd"):
            a += ["-" + k] + [float(v) for v in L[k]]
        if self.implex:
            a.append("-implex")
        if self.tangent:
            a.append("-tangent")
        if self.eta > 0:
            a += ["-eta", self.eta]
        if self.crack_planes:
            a += ["-crackPlanes", int(self.crack_planes[0]), int(self.crack_planes[1]), float(self.crack_planes[2])]
        return a

    def uniaxial(self, h: float, tension: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """the uniaxial (strain, stress) law the material follows for band h (for plots and checks)"""
        L = self.laws(h)
        return (np.array(L["Te"]), np.array(L["Ts"])) if tension else (np.array(L["Ce"]), np.array(L["Cs"]))


# uniaxial materials of bars: OpenSees uniaxialMaterial type and parameters (without the tag), as femotion.frame.opensees.Fiber
def steel(fy: float, E: float, b: float = 0.01) -> tuple:
    """reinforcing or prestressing steel, Steel02 (Giuffré-Menegotto-Pinto, hardening ratio b)"""
    return ("Steel02", float(fy), float(E), float(b), 18.0, 0.925, 0.15)


def elastic(E: float) -> tuple:
    return ("Elastic", float(E))


def elastic_pp(fy: float, E: float) -> tuple:
    """elastic-perfectly plastic"""
    return ("ElasticPP", float(E), float(fy) / float(E))


def _bar_E(mat: tuple) -> float:
    """the initial modulus of a bar material tuple (for the stiffness of its series spring), 0 if unknown"""
    k = {"Elastic": 1, "ElasticPP": 1, "FRP": 1, "Steel01": 2, "Steel02": 2, "Steel4": 2, "ReinforcingSteel": 3,
         "Hysteretic": None}.get(mat[0])
    try:
        if k is not None:
            return float(mat[k])
        if mat[0] == "Hysteretic":
            return float(mat[1]) / float(mat[2])
    except (IndexError, ValueError, ZeroDivisionError):
        pass
    return 0.0


def frp(E: float, eps_u: float) -> tuple:
    """fibre-reinforced polymer strip or bar: linear elastic, ruptures (stress → 0 for good) beyond eps_u (tension or
    compression)"""
    return ("FRP", float(E), float(eps_u))


@dataclass
class BondSlip:
    """bond stress – slip law τ(s) of a bar in concrete for bar(..., bond=BondSlip(...)): the envelope through the points
    (s_i, τ_i), s > 0 increasing (up to 7; τ stays at the last value beyond), the same in both directions; unloading
    and reloading as OpenSees' HystereticSM (Mazzoni 2023) with its defaults. Lengths and stresses in the model's units.
    plastic: an elastic – perfectly plastic law instead (ElasticPP: τ_1 reached at s_1, then constant), whose unloading
    is elastic — the rigid-plastic bond of the tension chord model when s_1 is small."""
    s: tuple
    tau: tuple
    plastic: bool = False

    def __post_init__(self):
        s, t = np.asarray(self.s, float), np.asarray(self.tau, float)
        if len(s) != len(t) or not 1 <= len(s) <= 7 or np.any(np.diff(s) <= 0) or s[0] <= 0:
            raise ValueError("BondSlip: 1 to 7 points with slips s > 0 increasing")
        if self.plastic and len(s) != 1:
            raise ValueError("BondSlip(plastic=True): one point (s_1, τ_1)")

    @classmethod
    def rigid_plastic(cls, tau: float, s_el: float) -> "BondSlip":
        """constant bond stress tau once the slip exceeds s_el (keep s_el small against the expected slips): the bond of
        the tension chord model (Marti et al. 1998: τ_b0 = 2 f_ct before the steel yields)"""
        return cls((float(s_el),), (float(tau),), plastic=True)

    @classmethod
    def mc2010(cls, fcm: float, *, units: str = "N-mm", failure: str = "pull-out", good: bool = True,
               confined: bool = False, clear_rib: float | None = None, n: int = 5) -> "BondSlip":
        """fib Model Code 2010, 6.1.1, table 6.1-1, ribbed bars before yielding: τ = τmax (s/s1)^0.4 up to s1 (sampled at
        n points), τmax to s2, linear to τbf at s3.
        failure="pull-out": the bar pulls out of well-confined concrete — a cover of at least 5 φ and a clear spacing of
        at least 10 φ, or closely spaced transverse reinforcement (CEB-FIP MC90 3.1.1; MC2010 the same in substance); τmax = 2.5 √fcm (good bond) or 1.25 √fcm, s1 = 1.0 /
        1.8 mm, s2 = 2.0 / 3.6 mm, s3 = clear rib spacing, τbf = 0.4 τmax.
        failure="splitting": the cover splits first — every bar with a smaller cover or spacing (most slabs and the
        cover zone of beams): τmax = 7.0 (fcm/20)^0.25 (good bond, no stirrups), 8.0 (good, with stirrups:
        confined=True), 5.0 / 5.5 (other bond conditions); s1 = the pull-out slip at that τmax, s2 = s1; without
        stirrups s3 = 1.2 s1 and τbf = 0, with stirrups s3 = 0.5 × clear rib spacing and τbf = 0.4 τmax.
        good=True: good bond conditions (bars at 45–90° to the horizontal during concreting, or flatter bars at most 250 mm
        above the bottom or at least 300 mm below the top of the pour); clear_rib: the clear distance between the bar's
        ribs (needed for pull-out and for splitting with stirrups; neither code gives a value: from the bar's rib
        geometry). Mean values, scatter about 30 % (MC90 3.1.1)"""
        if units not in UNITS:
            raise ValueError(f"units: one of {', '.join(UNITS)}")
        ks, kl = UNITS[units][0], UNITS[units][1] / UNITS[units][0]  # MPa per unit stress; mm per unit length
        f = fcm * ks
        needs = failure == "pull-out" or (failure == "splitting" and confined)
        if needs and clear_rib is None:
            raise ValueError("give clear_rib, the clear distance between the bar's ribs (s3 of MC2010 table 6.1-1)")
        cr = None if clear_rib is None else clear_rib * kl
        tpo, s1po, s2po = (2.5 * math.sqrt(f), 1.0, 2.0) if good else (1.25 * math.sqrt(f), 1.8, 3.6)
        if failure == "pull-out":
            tmax, s1, s2, s3, tbf = tpo, s1po, s2po, cr, 0.4 * tpo
        elif failure == "splitting":
            c = {(True, False): 7.0, (True, True): 8.0, (False, False): 5.0, (False, True): 5.5}[(good, confined)]
            tmax = c * (f / 20.0) ** 0.25
            s1 = s1po * (tmax / tpo) ** (1 / 0.4)
            s2 = s1
            s3, tbf = (0.5 * cr, 0.4 * tmax) if confined else (1.2 * s1, 0.0)
        else:
            raise ValueError("failure: 'pull-out' or 'splitting'")
        ss = [s1 * (k / n) ** 2 for k in range(1, n + 1)]
        tt = [tmax * (x / s1) ** 0.4 for x in ss]
        if s2 > s1:
            ss.append(s2)
            tt.append(tmax)
        s3 = max(s3, ss[-1] * 1.001)
        ss.append(s3)
        tt.append(max(tbf, 1e-3 * tmax))
        return cls(tuple(x / kl for x in ss), tuple(x / ks for x in tt))

    @classmethod
    def mc2010_plain(cls, fcm: float, *, units: str = "N-mm", kind: str = "hot-rolled", good: bool = True,
                     n: int = 5) -> "BondSlip":
        """fib Model Code 2010, 6.1.1, table 6.1-2, smooth (plain) bars — the plain round bars of older structures:
        τ = τmax (s/s1)^0.5 up to s1 = s2 = s3, then constant τbf = τmax. kind="hot-rolled": τmax = 0.3 √fcm (good bond
        conditions) or 0.15 √fcm (all others), s1 = 0.1 mm; "cold-drawn" (wire): 0.1 √fcm or 0.05 √fcm, s1 = 0.01 mm.
        Values as reproduced from the table by Cairns (2021, Structural Concrete 22:666–675, table 1; the MC2010 text was
        not at hand); CEB-FIP MC90 table 3.1.2 has the same numbers with √fck, for confined and unconfined concrete, and
        only where the concrete is not in lateral tension"""
        if units not in UNITS:
            raise ValueError(f"units: one of {', '.join(UNITS)}")
        ks, kl = UNITS[units][0], UNITS[units][1] / UNITS[units][0]
        f = fcm * ks
        if kind == "hot-rolled":
            tmax, s1 = (0.3 if good else 0.15) * math.sqrt(f), 0.1
        elif kind == "cold-drawn":
            tmax, s1 = (0.1 if good else 0.05) * math.sqrt(f), 0.01
        else:
            raise ValueError("kind: 'hot-rolled' or 'cold-drawn'")
        ss = [s1 * (k / n) ** 2 for k in range(1, n + 1)] + [100 * s1]
        tt = [tmax * (x / s1) ** 0.5 for x in ss[:-1]] + [tmax]
        return cls(tuple(x / kl for x in ss), tuple(x / ks for x in tt))

    def ops_material(self, ops, tag: int, scale: float) -> int:
        """the uniaxial force – slip material of one bond spring: τ × scale (perimeter × tributary length)"""
        s, t = np.asarray(self.s, float), np.asarray(self.tau, float) * scale
        if self.plastic:
            ops.uniaxialMaterial("ElasticPP", tag, float(t[0] / s[0]), float(s[0]))
            return tag
        env = [v for a, b in zip(t, s) for v in (float(a), float(b))]
        ops.uniaxialMaterial("HystereticSM", tag, "-posEnv", *env, "-negEnv", *[-v for v in env])
        return tag


# ── geometry: shape functions, point location ─────────────────────────────
def _hex_N(xi) -> np.ndarray:
    xi = np.atleast_2d(xi)
    return np.prod(1 + xi[:, None, :] * _HEX_SIGNS[None], axis=2) / 8


def _hex_dN(xi) -> np.ndarray:
    """(n, 8, 3) derivatives of the shape functions w.r.t. ξ, η, ζ"""
    xi = np.atleast_2d(xi)
    a = 1 + xi[:, None, :] * _HEX_SIGNS[None]
    d = np.empty_like(a)
    S = _HEX_SIGNS[None]
    d[..., 0] = S[..., 0] * a[..., 1] * a[..., 2] / 8
    d[..., 1] = a[..., 0] * S[..., 1] * a[..., 2] / 8
    d[..., 2] = a[..., 0] * a[..., 1] * S[..., 2] / 8
    return d


def _hex_natural(X: np.ndarray, p: np.ndarray, iters: int = 30) -> np.ndarray:
    """natural coordinates of p in the hexahedron with corners X (8, 3), by Newton's method"""
    xi = np.zeros(3)
    for _ in range(iters):
        r = _hex_N(xi)[0] @ X - p
        J = X.T @ _hex_dN(xi)[0]
        try:
            d = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return np.full(3, np.inf)
        xi = xi - d
        if np.abs(d).max() < 1e-12:
            break
        if np.abs(xi).max() > 10:
            return xi
    return xi


def _bary(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.c_[X[1] - X[0], X[2] - X[0], X[3] - X[0]]
    lam = np.linalg.solve(T, p - X[0])
    return np.r_[1 - lam.sum(), lam]


class CellLocator:
    """finds the solid cell containing a point and its interpolation weights (hex8 / tet4 meshes)"""

    def __init__(self, mesh: Mesh):
        from scipy.spatial import cKDTree

        self.mesh = mesh
        self.kinds = [k for k in ("hex8", "tet4") if k in mesh.cells]
        if not self.kinds:
            raise ValueError("CellLocator needs hex8 or tet4 cells")
        self.conn = [mesh.cells[k] for k in self.kinds]
        self.off = np.cumsum([0] + [len(c) for c in self.conn])
        P = mesh.points
        cen = np.concatenate([P[c].mean(axis=1) for c in self.conn])
        self.rad = np.concatenate([np.linalg.norm(P[c] - P[c].mean(axis=1, keepdims=True), axis=2).max(axis=1)
                                   for c in self.conn])
        self.lo = np.concatenate([P[c].min(axis=1) for c in self.conn])
        self.hi = np.concatenate([P[c].max(axis=1) for c in self.conn])
        self.tree = cKDTree(cen)
        self.scale = float(np.linalg.norm(P.max(0) - P.min(0)))

    def cell_nodes(self, e: int) -> tuple[str, np.ndarray]:
        b = int(np.searchsorted(self.off, e, side="right") - 1)
        return self.kinds[b], self.conn[b][e - self.off[b]]

    def _local(self, e: int, p: np.ndarray):
        kind, nodes = self.cell_nodes(e)
        X = self.mesh.points[nodes]
        if kind == "hex8":
            xi = _hex_natural(X, p)
            return kind, nodes, xi, float(np.abs(xi).max() - 1)
        lam = _bary(X, p)
        return kind, nodes, lam, float(-lam.min())

    def find(self, p, tol: float = 1e-6):
        """(cell, kind, nodes, local coordinates) or None when p is outside the mesh (tolerance tol in natural units)"""
        p = np.asarray(p, float)
        n = len(self.rad)
        k = min(16, n)
        _, idx = self.tree.query(p, k=k)
        cand = list(np.atleast_1d(idx))
        best = None
        for sweep in range(2):
            pad = 1e-9 * self.scale
            for e in cand:
                if np.any(p < self.lo[e] - pad) or np.any(p > self.hi[e] + pad):
                    continue
                kind, nodes, loc, out = self._local(int(e), p)
                if out <= tol:
                    return int(e), kind, nodes, loc
                if best is None or out < best[0]:
                    best = (out, int(e), kind, nodes, loc)
            if sweep == 0:
                cand = [e for e in self.tree.query_ball_point(p, r=1.01 * float(self.rad.max())) if e not in set(cand)]
        return None

    def weights(self, p, tol: float = 1e-6):
        """(node ids, interpolation weights) of the shape functions at p, or None outside"""
        f = self.find(p, tol)
        if f is None:
            return None
        e, kind, nodes, loc = f
        w = _hex_N(np.clip(loc, -1, 1))[0] if kind == "hex8" else np.clip(loc, 0, 1) / np.clip(loc, 0, 1).sum()
        return nodes, w


def interpolate(mesh: Mesh, U: np.ndarray, points, tol: float = 1e-6) -> np.ndarray:
    """values of a nodal field U (N, …) of a hex8 / tet4 mesh at arbitrary points (shape-function interpolation); NaN
    outside the mesh. For sub-models: the displacements of a global model at the boundary nodes of a local one."""
    loc = CellLocator(mesh)
    U = np.asarray(U, float)
    out = np.full((len(points),) + U.shape[1:], np.nan)
    for i, p in enumerate(np.asarray(points, float)):
        w = loc.weights(p, tol)
        if w is not None:
            out[i] = np.tensordot(w[1], U[w[0]], axes=1)
    return out


def displacement_source(source, U=None):
    """a function x (n, 3) → u (n, 3) from: an object with sample(points)["u"] (femotion.continuum CaseResult: the
    model's own interpolation, quadratic cells included), any callable u(points), or a hex8 / tet4 Mesh with nodal
    displacements U (N, 3) (trilinear / linear interpolation between its nodes)"""
    if U is not None:
        return lambda P: interpolate(source, U, P)
    if hasattr(source, "sample"):
        return lambda P: np.asarray(source.sample(np.asarray(P, float))["u"], float)
    if callable(source):
        return lambda P: np.asarray(source(np.asarray(P, float)), float)
    raise TypeError("source: a CaseResult (with .sample), a callable u(points), or a Mesh with U")


def submodel_boundary(source, U, local: "OpsSolid", where) -> tuple[np.ndarray, np.ndarray]:
    """imposed displacements for a sub-model, ready for OpsSolid.static(imposed=…): the nodes of `local` selected by
    `where` (its cut boundary) and the displacements of the global model there. source: a CaseResult of
    femotion.continuum (U=None; recommended — it interpolates with the global model's own shape functions, so a
    coarse quadratic model keeps its accuracy), a callable u(points) (U=None), or a hex8 / tet4 Mesh with nodal
    displacements U (N, 3) — corner nodes only, linear interpolation between them, so quadratic global models lose their
    accuracy that way. The local mesh must lie inside the global one, in the same coordinates. Same as
    local.boundary_from(source, where, U)"""
    return local.boundary_from(source, where, U)


def cell_aspect(mesh: Mesh) -> np.ndarray:
    """elongation of every solid cell (hex8 then tet4): hexahedra, the longest over the shortest of the mean edge lengths
    along their three directions; tetrahedra, the longest over the shortest edge"""
    out = []
    for k in [k for k in ("hex8", "tet4") if k in mesh.cells]:
        X = mesh.points[mesh.cells[k]]
        if k == "hex8":
            dirs = [[(0, 1), (3, 2), (4, 5), (7, 6)], [(0, 3), (1, 2), (4, 7), (5, 6)], [(0, 4), (1, 5), (2, 6), (3, 7)]]
            L = np.stack([np.mean([np.linalg.norm(X[:, b] - X[:, a], axis=1) for a, b in d], axis=0) for d in dirs], 1)
        else:
            L = np.stack([np.linalg.norm(X[:, b] - X[:, a], axis=1) for a, b in
                          ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))], 1)
        out.append(L.max(axis=1) / L.min(axis=1))
    return np.concatenate(out)


def band_widths(mesh: Mesh, band="volume") -> np.ndarray:
    """crack band width h of every solid cell (hex8 then tet4): "volume" (∛V for hexahedra, ∛(6V) for tetrahedra; right
    for cells of about equal sides, see cell_aspect), ("projected", n) (extent of the cell along the crack normal n,
    Oliver 1989: one unit vector, or one per cell (n_cells, 3) where the expected crack direction varies) or an array"""
    kinds = [k for k in ("hex8", "tet4") if k in mesh.cells]
    if isinstance(band, (np.ndarray, list)) and not (isinstance(band, (list, tuple)) and band and band[0] == "projected"):
        h = np.asarray(band, float)
        if h.shape != (sum(mesh.n_cells(k) for k in kinds),):
            raise ValueError("band: one width per solid cell")
        return h
    out = []
    off = 0
    for k in kinds:
        nk = len(mesh.cells[k])
        if isinstance(band, (tuple, list)) and band[0] == "projected":
            n = np.asarray(band[1], float)
            n = n / np.linalg.norm(n, axis=-1, keepdims=True)
            X = mesh.points[mesh.cells[k]]
            d = np.einsum("cvi,ci->cv", X, np.broadcast_to(n[off:off + nk] if n.ndim == 2 else n, (nk, 3)))
            out.append(d.max(axis=1) - d.min(axis=1))
        elif band == "volume":
            V = mesh.measure(k)
            out.append(np.cbrt(V) if k == "hex8" else np.cbrt(6 * V))
        else:
            raise ValueError("band: 'volume', ('projected', n) or an array")
        off += nk
    return np.concatenate(out)


def _quad_face_rule(X: np.ndarray):
    """2×2 Gauss on a bilinear quadrilateral face X (4, 3): (points (4, 3), shape values (4, 4), weights·|J| (4,))"""
    pts, Ns, ws = [], [], []
    for a in _G2:
        for b in _G2:
            N = 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b), (1 + a) * (1 + b), (1 - a) * (1 + b)])
            dNa = 0.25 * np.array([-(1 - b), (1 - b), (1 + b), -(1 + b)])
            dNb = 0.25 * np.array([-(1 - a), -(1 + a), (1 + a), (1 - a)])
            pts.append(N @ X)
            Ns.append(N)
            ws.append(np.linalg.norm(np.cross(dNa @ X, dNb @ X)))
    return np.array(pts), np.array(Ns), np.array(ws)


def _tri_face_rule(X: np.ndarray):
    """3-point rule (exact for quadratics) on a triangle X (3, 3)"""
    A = 0.5 * np.linalg.norm(np.cross(X[1] - X[0], X[2] - X[0]))
    L = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
    return L @ X, L, np.full(3, A / 3)


def _sel(where, X: np.ndarray) -> np.ndarray:
    """indices selected by a predicate pred(x) → bool on coordinates x of shape (3, n), a bool mask or an index array"""
    if callable(where):
        return np.flatnonzero(np.asarray(where(X.T), bool))
    a = np.asarray(where)
    return np.flatnonzero(a) if a.dtype == bool else a.astype(np.int64).ravel()


# ── results ──────────────────────────────────────────────────────────────
@dataclass
class Step:
    """one converged step: stage name, step number in the stage, load factor of the stage, displacements (n_nodes, 3)
    of solid then bar nodes, reactions R (n, 3) at the supported nodes Rn (fixed, on springs, or with imposed
    displacements), and — when recorded — cell and bar results (dicts of arrays; see OpsSolid._read_fields)"""
    stage: str
    k: int
    lam: float
    U: np.ndarray
    R: np.ndarray
    Rn: np.ndarray
    cell: dict | None = None
    bar: dict | None = None
    dowel: dict | None = None
    watch: dict = field(default_factory=dict)


@dataclass
class Stage:
    name: str
    kind: str  # "static", "push", "hold" (no load increment)
    steps: list = field(default_factory=list)  # indices into OpsSolid.history
    converged: bool = True
    message: str = ""
    change: float = 0.0  # hold stages: displacement change of the last step / of the whole stage


VOIGT = ("11", "22", "33", "12", "23", "13")  # OpenSees' strain order = femotion's (xx, yy, zz, xy, yz, zx)


@dataclass
class Eigenstrain:
    """an initial (free, eigen-) strain that a stage adds: shrinkage, temperature, swelling, rust. Per cell and integration
    point (cells (m,), values (m, 8, 6): Voigt order xx yy zz xy yz zx, engineering shear; a tetrahedron uses slot 0) and
    per bar segment (segs (k,), seg_values (k,): along the bar). Made by OpsSolid.initial_strain / shrinkage / thermal /
    bar_strain; combine with + and scale with a number. The material sees σ = f(ε − ε0): OpenSees' nDMaterial
    InitStrain around every solid material, so cracked and crushed concrete respond to the free strain as they do to a
    load; a bar's free strain is the imposed elongation of a stiff elastic spring in series with each of its segments,
    so a yielding bar follows its own path too. size = (number of cells, number of bar segments) of the model it was
    made for (checked when a stage applies it)"""
    cells: np.ndarray
    values: np.ndarray
    segs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    seg_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    size: tuple | None = None

    @staticmethod
    def _merge(ia, va, ib, vb):
        ids = np.union1d(ia, ib).astype(np.int64)
        out = np.zeros((len(ids),) + (va.shape[1:] if len(va) else vb.shape[1:]))
        np.add.at(out, np.searchsorted(ids, ia), va)
        np.add.at(out, np.searchsorted(ids, ib), vb)
        return ids, out

    def __add__(self, other: "Eigenstrain") -> "Eigenstrain":
        if self.size is not None and other.size is not None and self.size != other.size:
            raise ValueError(f"free strains of two models ({self.size} and {other.size} cells and bar segments)")
        c, v = self._merge(self.cells, self.values.reshape(-1, 8, 6), other.cells, other.values.reshape(-1, 8, 6))
        s, w = self._merge(self.segs, self.seg_values, other.segs, other.seg_values)
        return Eigenstrain(c, v, s, w, self.size if self.size is not None else other.size)

    def __mul__(self, k: float) -> "Eigenstrain":
        return Eigenstrain(self.cells, self.values * float(k), self.segs, self.seg_values * float(k), self.size)

    __rmul__ = __mul__

    def __neg__(self) -> "Eigenstrain":
        return self * -1.0

    def cell_mean(self, nc: int) -> np.ndarray:
        """(nc, 6): the cell averages over the integration points (zero for cells it does not touch)"""
        out = np.zeros((nc, 6))
        out[self.cells] = self.values.mean(axis=1)
        return out


# ── model ────────────────────────────────────────────────────────────────
class OpsSolid:
    """femotion Mesh (hex8 and / or tet4) → OpenSees solid model, built when the first analysis starts.

    materials: one material for every cell, or {cell group: material} covering all cells (ElasticIso, ElasticOrtho,
    ConcreteDamage). element: "stdBrick" (default) or "bbarBrick" for hexahedra; tetrahedra use FourNodeTetrahedron.
    band: crack band width per cell (see band_widths). penalty: stiffness of the embedding ties and displacement drivers
    relative to the largest Young's modulus. Convergence: the norm of the displacement correction must fall below rtol ×
    the norm of the displacement increment of the previous converged step (adaptive, unit-free; at least 1e-10 × the
    model size, the round-off of the penalty ties), within `iters` iterations. numberer: equation ordering for the
    sparse solver (UmfPack): "AMD" (default; 30 % faster on bulky blocks) or "RCM" (10 % faster on long slender ones)."""

    def __init__(self, mesh: Mesh, materials, element: str = "stdBrick", band="volume", penalty: float = 1e6,
                 rtol: float = 1e-4, iters: int = 50, numberer: str = "AMD", log: str | None = LOG):
        kinds = [k for k in ("hex8", "tet4") if k in mesh.cells]
        if not kinds:
            raise ValueError("OpsSolid needs hex8 or tet4 cells")
        if element not in ("stdBrick", "bbarBrick"):
            raise ValueError("element: 'stdBrick' or 'bbarBrick'")
        cells = {k: orient(mesh.points, k, mesh.cells[k]) for k in kinds}
        self.mesh = Mesh(mesh.points, cells, mesh.point_groups,
                         {g: {k: v for k, v in d.items() if k in kinds} for g, d in mesh.cell_groups.items()})
        self.kinds = kinds
        self.off = {k: o for k, o in zip(kinds, np.cumsum([0] + [len(cells[k]) for k in kinds]))}
        self.nc = sum(len(cells[k]) for k in kinds)
        self.element = element
        self.log = log
        self.rtol, self.iters, self.numberer = rtol, iters, numberer
        self._tol = None  # absolute tolerance of the next step
        self._dt = None
        self.hold_tol, self.hold_steps = 1e-4, 200
        self.N = mesh.n_points
        self.scale = float(np.linalg.norm(mesh.points.max(0) - mesh.points.min(0)))
        # materials per cell
        self.mat_of = np.full(self.nc, -1, np.int64)
        if isinstance(materials, dict):
            self.materials = list(materials.values())
            for i, (g, _) in enumerate(materials.items()):
                if g not in self.mesh.cell_groups:
                    raise KeyError(f"cell group {g!r} not in the mesh")
                for k, ids in self.mesh.cell_groups[g].items():
                    self.mat_of[self.off[k] + np.asarray(ids, np.int64)] = i
            if (self.mat_of < 0).any():
                raise ValueError(f"{int((self.mat_of < 0).sum())} cells have no material (cell groups do not cover the mesh)")
        else:
            self.materials = [materials]
            self.mat_of[:] = 0
        self.materials = [_native(m) for m in self.materials]
        self.implex = any(isinstance(m, ConcreteDamage) and m.implex for m in self.materials)
        self.h = band_widths(self.mesh, band)
        if isinstance(band, str) and band == "volume":
            conc = np.isin(self.mat_of, [i for i, m in enumerate(self.materials) if isinstance(m, ConcreteDamage)])
            asp = cell_aspect(self.mesh)[conc]
            lim = {"hex8": 1.5, "tet4": 3.0}
            kind_of = np.concatenate([np.full(len(self.mesh.cells[k]), lim[k]) for k in kinds])[conc]
            bad = int((asp > kind_of).sum())
            if bad:
                warnings.warn(f"{bad} concrete cells are elongated (aspect up to {asp.max():.1f}): band='volume' (∛V) is then "
                              "the band of no crack direction, and the energy a crack dissipates is off by up to the aspect "
                              "ratio; give band=('projected', n) with the expected crack normal (one vector, or one per cell)",
                              stacklevel=2)
        self.E_ref = max(m.E if not isinstance(m, ElasticOrtho) else max(m.E) for m in self.materials)
        self.penalty = penalty
        self.loc = CellLocator(self.mesh)
        # bars
        self._bxyz: list[np.ndarray] = []  # coordinates of extra (bar) nodes
        self._bkey: dict = {}
        self._tie: dict[int, tuple] = {}  # extra node → (cell, kind, nodes, local coordinates) of the cell it is tied to
        self.bars: list[dict] = []
        self._segs: list[tuple] = []  # (node i, node j, bar)
        self.dowels: list[dict] = []
        self._dsegs: list[tuple] = []  # (node i, node j, dowel)
        self._dkey: dict = {}  # coordinates → dowel node (6 dof)
        self._six: set[int] = set()  # extra nodes with 6 dof (dowels)
        self._rot: set[int] = set()  # dowel nodes whose rotation is tied to the solid's
        self._links: dict[int, int] = {}  # bar node → dowel node it is pinned to (equal translations)
        self._sliders: list[tuple] = []  # (tendon node, anchor node, sliding direction) at deviators of unbonded bars
        self._bonds: list[dict] = []  # bond springs of bars with a bond-slip law: anchor, node, t, L, bar, s
        self._akey: dict = {}  # coordinates → anchor node of bond springs
        # supports
        self.fixed: dict[int, list[int]] = {}
        self._springs: list[tuple] = []  # (node, (kx, ky, kz))
        self._ties: list[tuple] = []  # (master, slaves, comps)
        self._driven: set[int] = set()
        self._driven_masks: list[tuple] = []
        self._react: dict[int, list[tuple]] = {}  # model node → [(OpenSees node tag, component mask)] carrying its reaction
        self._faces = None
        # state
        self.built = False
        self.history: list[Step] = []
        self.stages: list[Stage] = []
        self._pat = self._ts = 0
        self.h_clipped = 0
        self._e0 = self._e0b = None  # committed free strains: (nc, 8, 6) per integration point, (n_segments,)
        self._watch: dict = {}  # name → fn(self), evaluated after every converged step
        self._eig = None  # the free strain of the running stage
        self._ptag = 0

    # ── geometry helpers ─────────────────────────────────────────────────
    @property
    def n_nodes(self) -> int:
        """solid nodes then extra bar nodes"""
        return self.N + len(self._bxyz)

    @property
    def points(self) -> np.ndarray:
        return np.concatenate([self.mesh.points, np.array(self._bxyz).reshape(-1, 3)])

    def cell_conn(self) -> list[tuple[str, np.ndarray]]:
        return [(k, self.mesh.cells[k]) for k in self.kinds]

    def faces(self) -> tuple[np.ndarray, np.ndarray]:
        """boundary faces (hexahedral meshes: quadrilaterals, tetrahedral: triangles; outward normals) and their cell"""
        if self._faces is None:
            if len(self.kinds) > 1:
                raise NotImplementedError("boundary faces of mixed hex8 / tet4 meshes")
            f, owner = self.mesh.surface()
            self._faces = (f, owner + self.off[self.kinds[0]])
        return self._faces

    def face_centroids(self) -> np.ndarray:
        f, _ = self.faces()
        return self.mesh.points[f].mean(axis=1)

    def nodes(self, where) -> np.ndarray:
        """solid node ids selected by a predicate pred(x) on coordinates (3, N), a bool mask or ids"""
        return _sel(where, self.mesh.points)

    def nearest(self, p) -> int:
        return self.mesh.nearest(p)

    # ── reinforcement ────────────────────────────────────────────────────
    def _key(self, p):
        tol = 1e-7 * self.scale
        return tuple(np.round(np.asarray(p, float) / tol).astype(np.int64))

    def _new_node(self, p, tie) -> int:
        j = self.N + len(self._bxyz)
        self._bxyz.append(np.asarray(p, float))
        if tie is not None:
            self._tie[j] = tie
        return j

    def _node_at(self, p, outside: str) -> int:
        """the node of a bar point: pinned to a dowel node there, else a mesh node there, else a node of another bar
        there, else a new node tied to the solid (or free, outside it)"""
        p = np.asarray(p, float)
        key = self._key(p)
        if key in self._bkey:
            return self._bkey[key]
        if key in self._dkey:  # a strip or tendon ending on an anchor: its own node, pinned to the dowel's
            j = self._new_node(p, None)
            self._links[j] = self._dkey[key]
            self._bkey[key] = j
            return j
        n = self.mesh.nearest(p)
        if np.linalg.norm(self.mesh.points[n] - p) <= 1e-7 * self.scale:
            self._bkey[key] = n
            return n
        tie = self.loc.find(p)
        if tie is None and outside != "free":
            raise ValueError(f"bar point {p.tolist()} lies outside the solid (pass outside='free' for a node held only "
                             "by other bars, springs or supports)")
        j = self._new_node(p, tie)
        self._bkey[key] = j
        return j

    def bar(self, points, area: float, material: tuple, *, bond: bool = True, prestress: float = 0.0,
            seg: float | None = None, name: str | None = None, outside: str = "error", slide: bool = True,
            alpha: float = 0.0, perimeter: float | None = None, free_strain: bool | None = None) -> int:
        """a reinforcing bar, tendon, anchor or strip along the polyline `points` (n, 3); returns its index.
        material: uniaxial material tuple (steel, elastic, elastic_pp, frp, or any ("OpenSeesType", *params)).
        bond=True: perfectly bonded, cut into segments of at most `seg` (default half the median crack band) with every
        node tied to the solid (a run outside the solid, outside="free", stays one segment between its given points).
        bond=False: unbonded, attached only at the given points: the end points are anchored; the points in between are
        deviators over which the tendon slides without friction (slide=True: one force along the whole tendon, the
        node is held across the tendon only, along the bisector of the two spans it is free) or that hold it
        (slide=False: an anchor at every point, a force per span). prestress: initial stress σ0 (tension +), in full
        from the first analysis step, balanced by the solid (elastic shortening included): a pretensioned bonded tendon,
        or an unbonded one; a post-tensioned tendon grouted afterwards is not modelled. Points that coincide with mesh
        nodes or with nodes of other bars share them (a strip ending on an anchor head); outside="free" allows points
        outside the solid (they must then be held by other members, springs or supports). alpha: coefficient of thermal
        expansion per degree for thermal() (steel 1.2e-5; 0 by default). free_strain: give the bar a stiff elastic
        spring in series with each segment, whose imposed elongation is the bar's free strain (bar_strain, thermal);
        default: when alpha ≠ 0. Without it the bar takes no free strain once the model is built.
        bond=BondSlip(...): bonded with slip — the bar has its own nodes (every `seg`), each joined to the solid at the
        same point by a spring along the bar with the bond law × perimeter × tributary length, and stiff across it; the
        perimeter defaults to that of a round bar of the given area. Slip and bond stress: bar_results(...)["bond"]."""
        if self.built:
            raise RuntimeError("the OpenSees model is already built: add bars before the first analysis")
        P0 = np.asarray(points, float).reshape(-1, 3)
        if len(P0) < 2:
            raise ValueError("a bar needs at least two points")
        slip = isinstance(bond, BondSlip)
        if slip:  # own nodes, each joined to an anchor point of the solid by a bond spring
            P = resample(P0, seg or 0.5 * float(np.median(self.h)))
            given = np.zeros(len(P), bool)
            given[[int(np.argmin(np.linalg.norm(P - q, axis=1))) for q in P0]] = True
            P = P[given | np.array([self.loc.find(q) is not None for q in P])]
            ids = [self._new_node(p, None) for p in P]
            L = np.linalg.norm(np.diff(P, axis=0), axis=1)
            trib = np.zeros(len(P))  # bonded length per node: the inside parts of the half segments next to it
            for k in range(len(L)):
                lo, hi = self._inside_parts(P[k], P[k + 1])
                trib[k] += L[k] * lo
                trib[k + 1] += L[k] * hi
            T = np.diff(P, axis=0) / L[:, None]
            tan = np.r_[T[:1], T[:-1] + T[1:], T[-1:]]
            tan /= np.linalg.norm(tan, axis=1)[:, None]
            sa = np.r_[0.0, np.cumsum(L)]
            per = float(perimeter) if perimeter is not None else math.sqrt(4 * math.pi * float(area))
            bonds = []
            for k, (p, j) in enumerate(zip(P, ids)):
                a = self._anchor_at(p, outside)
                if a is None:
                    continue
                bonds.append(len(self._bonds))
                self._bonds.append(dict(anchor=a, node=j, t=tan[k], L=float(trib[k]), bar=len(self.bars), s=float(sa[k])))
        elif bond:
            P = resample(P0, seg or 0.5 * float(np.median(self.h)))
            given = np.zeros(len(P), bool)
            given[[int(np.argmin(np.linalg.norm(P - q, axis=1))) for q in P0]] = True
            inside = np.array([self.loc.find(q) is not None for q in P])
            P = P[given | inside]  # no free nodes in the middle of a run outside the solid (they would hang on two bars)
            ids = [self._node_at(p, outside) for p in P]
        else:
            P = P0
            ids = []
            for k, p in enumerate(P):
                if 0 < k < len(P) - 1 and slide:
                    anchor = self._node_at(p, outside)
                    j = self._new_node(p, None)  # the tendon's own node, held across the tendon by the anchor
                    t0 = (P[k] - P[k - 1]) / np.linalg.norm(P[k] - P[k - 1])
                    t1 = (P[k + 1] - P[k]) / np.linalg.norm(P[k + 1] - P[k])
                    self._sliders.append((j, anchor, (t0 + t1) / np.linalg.norm(t0 + t1)))
                    ids.append(j)
                else:
                    ids.append(self._node_at(p, outside))
        b = len(self.bars)
        segs = []
        for i, j in zip(ids[:-1], ids[1:]):
            if i != j:
                segs.append(len(self._segs))
                self._segs.append((i, j, b))
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        self.bars.append(dict(name=name or f"bar{b}", nodes=ids, points=P, s=s, area=float(area), material=material,
                              prestress=float(prestress), bond=bond, segs=np.array(segs, np.int64), alpha=float(alpha),
                              bonds=np.array(bonds if slip else [], np.int64), perimeter=per if slip else None,
                              free_strain=bool(alpha != 0) if free_strain is None else bool(free_strain)))
        return b

    def _inside_parts(self, a, b, n: int = 16) -> tuple[float, float]:
        """the shares of the segment a–b that lie inside the solid in its first and its second half: sampled at n + 1
        points, the crossings of the surface found by bisection"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        inside = lambda t: self.loc.find(a + t * (b - a)) is not None  # noqa: E731
        ts = np.linspace(0.0, 1.0, n + 1)
        flags = [inside(t) for t in ts]
        cuts = [0.0]
        for t0, t1, f0, f1 in zip(ts[:-1], ts[1:], flags[:-1], flags[1:]):
            if f0 != f1:
                lo_, hi_ = t0, t1
                for _ in range(40):
                    mid = 0.5 * (lo_ + hi_)
                    lo_, hi_ = (mid, hi_) if inside(mid) == f0 else (lo_, mid)
                cuts.append(0.5 * (lo_ + hi_))
        cuts.append(1.0)
        first = second = 0.0
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            if not inside(0.5 * (t0 + t1)):
                continue
            first += max(0.0, min(t1, 0.5) - t0)
            second += max(0.0, t1 - max(t0, 0.5))
        return first, second

    def _anchor_at(self, p, outside: str):
        """the point of the solid a bond spring holds: a mesh node there, else a node tied to the cell (shared by the bond
        springs of bars crossing there); None outside the solid with outside='free'"""
        key = self._key(p)
        if key in self._akey:
            return self._akey[key]
        n = self.mesh.nearest(p)
        if np.linalg.norm(self.mesh.points[n] - p) <= 1e-7 * self.scale:
            self._akey[key] = n
            return n
        tie = self.loc.find(p)
        if tie is None:
            if outside != "free":
                raise ValueError(f"bar point {np.round(p, 9).tolist()} lies outside the solid (pass outside='free' for "
                                 "a run outside it, without bond)")
            return None
        j = self._new_node(p, tie)
        self._akey[key] = j
        return j

    def dowel(self, points, *, d: float | None = None, E: float | None = None, nu: float = 0.3, section=None,
              fiber=None, seg: float | None = None, rot: bool = False, name: str | None = None) -> int:
        """an anchor bolt, dowel or stud as beam elements along the polyline `points`: it carries axial force, shear and
        bending, so a force on its head reaches the concrete along the embedded length (dowel action), not at one point.
        Section: a solid circle of diameter d and Young's modulus E (shear modulus E / 2(1 + ν)), or section=(E, A, Iy,
        Iz, G, J); fiber=femotion.frame.opensees.Fiber for a yielding anchor (forceBeamColumn, 5 Lobatto points).
        Cut into elements of at most `seg` (default half the median crack band); every node inside the solid is tied to
        it in translation (the dowel bears on the concrete and bends on its own; a torsional spring as stiff as one of its
        elements holds its twist at the first embedded node); rot=True also ties its rotation to the rotation of the
        solid (a fully bonded stiff insert; tied to the sub-tetrahedron that holds the node, linear). Nodes outside the solid (a head standing proud of the
        surface) stay free. Bars whose points coincide with dowel nodes are pinned to them (define the dowels first): a
        prestressed strip anchored on the bolt heads. Returns the dowel's index."""
        if self.built:
            raise RuntimeError("the OpenSees model is already built: add dowels before the first analysis")
        P = np.asarray(points, float).reshape(-1, 3)
        if section is None:
            if d is None or E is None:
                raise ValueError("dowel: give d and E (solid circle) or section=(E, A, Iy, Iz, G, J)")
            A, Id = math.pi * d**2 / 4, math.pi * d**4 / 64
            section = (E, A, Id, Id, E / (2 * (1 + nu)), 2 * Id)
        P = resample(P, seg or 0.5 * float(np.median(self.h)))
        ids = []
        for p in P:
            key = self._key(p)
            if key in self._bkey:
                raise ValueError("a dowel point coincides with a node of a bar defined earlier: define dowels first")
            if key in self._dkey:
                ids.append(self._dkey[key])
                continue
            j = self._new_node(p, self.loc.find(p))
            self._six.add(j)
            if rot:
                self._rot.add(j)
            self._dkey[key] = j
            ids.append(j)
        k = len(self.dowels)
        segs = []
        for i, j in zip(ids[:-1], ids[1:]):
            segs.append(len(self._dsegs))
            self._dsegs.append((i, j, k))
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        self.dowels.append(dict(name=name or f"dowel{k}", nodes=ids, points=P, s=s, section=tuple(map(float, section)),
                                fiber=fiber, segs=np.array(segs, np.int64)))
        return k

    # ── supports ─────────────────────────────────────────────────────────
    def fix(self, where, comps=(0, 1, 2), nodes=None):
        """fix components (0, 1, 2 = x, y, z) of solid nodes selected by `where` (predicate on coordinates, mask or ids),
        or of any model nodes given by id in `nodes` (bar nodes included)"""
        sel = np.asarray(nodes, np.int64) if nodes is not None else self.nodes(where)
        if len(sel) == 0:
            raise ValueError("fix: no node selected")
        for n in sel:
            f = self.fixed.setdefault(int(n), [0, 0, 0])
            for c in comps:
                f[c] = 1
        return sel

    def spring(self, where, k, nodes=None):
        """elastic support: a spring of stiffness k = (kx, ky, kz) (0: none) from every selected node to the ground"""
        sel = np.asarray(nodes, np.int64) if nodes is not None else self.nodes(where)
        for n in sel:
            self._springs.append((int(n), tuple(float(v) for v in k)))
        return sel

    def tie(self, where, comps=(0, 1, 2), master=None, nodes=None) -> int:
        """tie components of the selected nodes to a master node (equalDOF: they move together, e.g. a rigid loading
        plate or a symmetric push); master: node id or point (default the first selected node). Returns the master id.
        Reactions are read on fixed components only, so tied nodes on supports report correct support forces"""
        sel = np.asarray(nodes, np.int64) if nodes is not None else self.nodes(where)
        m = int(sel[0]) if master is None else (int(master) if np.ndim(master) == 0 else self.nearest(master))
        self._ties.append((m, [int(n) for n in sel if n != m], tuple(int(c) for c in comps)))
        return m

    def face_spring(self, where, k):
        """distributed elastic support (Winkler) on boundary faces selected by a predicate on face centroids: k = (kx, ky,
        kz) per unit area, lumped to the nodes by their tributary areas"""
        f, _ = self.faces()
        sel = _sel(where, self.face_centroids())
        A = np.zeros(self.N)
        for fi in sel:
            X = self.mesh.points[f[fi]]
            _, Ns, ws = _quad_face_rule(X) if f.shape[1] == 4 else _tri_face_rule(X)
            np.add.at(A, f[fi], Ns.T @ ws)
        for n in np.flatnonzero(A > 0):
            self._springs.append((int(n), tuple(float(A[n] * v) for v in k)))
        return np.flatnonzero(A > 0)

    def boundary_from(self, source, where, U=None) -> tuple[np.ndarray, np.ndarray]:
        """imposed displacements on the nodes selected by `where` taken from a global model, for
        static(imposed=…): source a femotion.continuum CaseResult (recommended: its own shape functions, quadratic cells
        included), a callable u(points) → (n, 3), or a hex8 / tet4 Mesh with nodal displacements U (linear between its
        nodes; corner nodes of quadratic models lose their accuracy that way)"""
        nodes = self.nodes(where)
        vals = displacement_source(source, U)(self.mesh.points[nodes])
        if np.isnan(vals).any():
            raise ValueError(f"{int(np.isnan(vals).any(axis=1).sum())} boundary nodes lie outside the global model")
        return nodes, vals

    # ── loads: consistent nodal forces (n_nodes, 3) ──────────────────────
    def _F(self) -> np.ndarray:
        return np.zeros((self.n_nodes, 3))

    def nodal_force(self, where, F, total: bool = False, nodes=None) -> np.ndarray:
        """force F on every selected node (total=True: F shared equally among them)"""
        sel = np.asarray(nodes, np.int64) if nodes is not None else self.nodes(where)
        out = self._F()
        out[sel] += np.asarray(F, float) / (len(sel) if total else 1)
        return out

    def point_force(self, p, F) -> np.ndarray:
        """force F at any point inside the solid, spread to the nodes of its cell by the shape functions"""
        w = self.loc.weights(p)
        if w is None:
            raise ValueError(f"point {list(p)} is outside the solid")
        out = self._F()
        out[w[0]] += np.outer(w[1], np.asarray(F, float))
        return out

    def traction(self, where, t) -> np.ndarray:
        """surface traction (force per unit area) on boundary faces selected by a predicate on face centroids (3, F), a
        mask or face ids: a constant vector, or t(x) with x (3, n) → (3, n)"""
        f, _ = self.faces()
        sel = _sel(where, self.face_centroids())
        if len(sel) == 0:
            raise ValueError("traction: no face selected")
        out = self._F()
        for fi in sel:
            X = self.mesh.points[f[fi]]
            pts, Ns, ws = _quad_face_rule(X) if f.shape[1] == 4 else _tri_face_rule(X)
            tv = np.asarray(t(pts.T), float).T if callable(t) else np.broadcast_to(np.asarray(t, float), pts.shape)
            out[f[fi]] += Ns.T @ (tv * ws[:, None])
        return out

    def pressure(self, where, p) -> np.ndarray:
        """normal pressure p (positive pushing into the solid) on the selected boundary faces"""
        f, _ = self.faces()
        sel = _sel(where, self.face_centroids())
        out = self._F()
        for fi in sel:
            X = self.mesh.points[f[fi]]
            pts, Ns, ws = _quad_face_rule(X) if f.shape[1] == 4 else _tri_face_rule(X)
            n = np.cross(X[1] - X[0], X[2] - X[0])
            n /= np.linalg.norm(n)
            pv = np.asarray(p(pts.T), float) if callable(p) else np.full(len(pts), float(p))
            out[f[fi]] += Ns.T @ (-(pv * ws)[:, None] * n[None])
        return out

    def body_force(self, b, cells=None) -> np.ndarray:
        """body force per unit volume b = (bx, by, bz) (or per cell (n_cells, 3)) on all cells or on those selected by a
        predicate on cell centroids, a mask or ids; consistent nodal forces (2×2×2 Gauss on hexahedra)"""
        out = self._F()
        bb = np.asarray(b, float)
        bb = np.broadcast_to(bb, (self.nc, 3)) if bb.ndim == 1 else bb
        sel = np.zeros(self.nc, bool)
        sel[np.arange(self.nc) if cells is None else _sel(cells, self.cell_centroids())] = True
        P = self.mesh.points
        for k, conn in self.cell_conn():
            o = self.off[k]
            e = np.flatnonzero(sel[o:o + len(conn)])
            if not len(e):
                continue
            X = P[conn[e]]
            be = bb[o + e]
            if k == "tet4":
                V = np.abs(np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])) / 6
                f = np.repeat((V[:, None] * be)[:, None, :] / 4, 4, axis=1)
            else:
                f = np.zeros((len(e), 8, 3))
                for a in _G2:
                    for c in _G2:
                        for d in _G2:
                            xi = np.array([a, c, d])
                            J = np.einsum("nki,kj->nij", X, _hex_dN(xi)[0])
                            f += _hex_N(xi)[0][None, :, None] * (np.abs(np.linalg.det(J))[:, None] * be)[:, None, :]
            np.add.at(out, conn[e].ravel(), f.reshape(-1, 3))
        return out

    def self_weight(self, gamma, direction=(0.0, 0.0, -1.0)) -> np.ndarray:
        """self-weight: gamma = weight per unit volume, one value or {cell group: value} (cells in no listed group get 0);
        direction of gravity"""
        g = np.asarray(direction, float) / np.linalg.norm(direction)
        if isinstance(gamma, dict):
            w = np.zeros(self.nc)
            for name, v in gamma.items():
                for k, ids in self.mesh.cell_groups[name].items():
                    w[self.off[k] + np.asarray(ids, np.int64)] = float(v)
        else:
            w = np.full(self.nc, float(gamma))
        return self.body_force(w[:, None] * g[None])

    def cell_centroids(self) -> np.ndarray:
        return np.concatenate([self.mesh.points[c].mean(axis=1) for _, c in self.cell_conn()])

    # ── initial (free) strains: shrinkage, temperature, swelling, rust ───
    def cells_of(self, where=None) -> np.ndarray:
        """cell ids: all (None), cell group name(s), a predicate on cell centroids x (3, n), a mask or ids"""
        if where is None:
            return np.arange(self.nc)
        if isinstance(where, str) or (isinstance(where, (list, tuple)) and len(where)
                                      and all(isinstance(g, str) for g in where)):
            out = []
            for g in [where] if isinstance(where, str) else where:
                if g not in self.mesh.cell_groups:
                    raise KeyError(f"cell group {g!r} not in the mesh")
                out += [self.off[k] + np.asarray(ids, np.int64) for k, ids in self.mesh.cell_groups[g].items()]
            return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)
        return _sel(where, self.cell_centroids())

    def gauss_points(self, cells=None) -> np.ndarray:
        """(m, 8, 3) integration points of cells (see cells_of) in OpenSees' order: hexahedra 2 × 2 × 2 Gauss, point
        4i + 2j + k at natural coordinates (g_i, g_j, g_k), g = (−1, +1)/√3; a tetrahedron's one point (its centroid)
        fills all eight slots"""
        cells = self.cells_of(cells)
        out = np.zeros((len(cells), 8, 3))
        P = self.mesh.points
        for k, conn in self.cell_conn():
            o, n = self.off[k], len(conn)
            sel = (cells >= o) & (cells < o + n)
            if not sel.any():
                continue
            X = P[conn[cells[sel] - o]]
            out[sel] = np.einsum("gn,mnd->mgd", _hex_N(_GAUSS8), X) if k == "hex8" else X.mean(axis=1)[:, None, :]
        return out

    def _points_for(self, cells, at: str) -> np.ndarray:
        X = self.gauss_points(cells)
        if at not in ("gauss", "centroid"):
            raise ValueError("at: 'gauss' or 'centroid'")
        if at == "centroid" or self.element == "bbarBrick":
            X = np.repeat(X.mean(axis=1, keepdims=True), 8, axis=1)
        return X

    @staticmethod
    def _voigt_field(eps0, X: np.ndarray) -> np.ndarray:
        """a free strain (number, 6 Voigt components, 3 × 3 tensor, or a function of x (3, m, 8) returning one of these
        per point) → (m, 8, 6)"""
        m = X.shape[0]
        v = np.asarray(eps0(np.moveaxis(X, -1, 0)) if callable(eps0) else eps0, float)
        pts = (m, 8)
        if v.ndim == 0 or v.shape == pts:  # isotropic
            iso = np.broadcast_to(v, pts)
            return np.concatenate([np.repeat(iso[..., None], 3, axis=2), np.zeros(pts + (3,))], axis=2)
        if v.shape[0] == 6 and v.shape[1:] in ((), pts):
            return np.moveaxis(np.broadcast_to(v.reshape((6,) + (1, 1) * (v.ndim == 1) + v.shape[1:]), (6,) + pts), 0, -1).copy()
        if v.shape[:2] == (3, 3) and v.shape[2:] in ((), pts):
            T = np.broadcast_to(v.reshape((3, 3) + (1, 1) * (v.ndim == 2) + v.shape[2:]), (3, 3) + pts)
            if not np.allclose(T, np.swapaxes(T, 0, 1), atol=1e-14 + 1e-9 * np.abs(T).max()):
                raise ValueError("a strain tensor must be symmetric")
            out = np.stack([T[0, 0], T[1, 1], T[2, 2], 2 * T[0, 1], 2 * T[1, 2], 2 * T[2, 0]], axis=-1)
            return out
        raise ValueError(f"free strain of shape {v.shape}: a number, 6 Voigt components or a 3 × 3 tensor (per point: "
                         f"{pts}, (6,) + {pts} or (3, 3) + {pts})")

    def initial_strain(self, eps0, cells=None, *, at: str = "gauss") -> Eigenstrain:
        """a free strain ε0 of cells (cells_of: all, group name(s), predicate, mask, ids) for a stage:
        static(..., strain=...) ramps it with the stage's load factor and holds it afterwards; the stress is
        σ = f(ε − ε0). eps0: a number (isotropic ε0·I: swelling +, shrinkage −), 6 Voigt components (xx yy zz xy yz zx,
        engineering shear), a 3 × 3 tensor, or a function of the integration points x (3, m, 8) returning one of these
        per point ((m, 8), (6, m, 8), (3, 3, m, 8)). at="gauss": functions are evaluated at every integration point
        (fields varying within a cell, as femotion.continuum.Elastic does); "centroid": once per cell (always with
        bbarBrick, whose averaged volumetric strain needs a free strain constant over the cell)"""
        c = self.cells_of(cells)
        return Eigenstrain(c, self._voigt_field(eps0, self._points_for(c, at)), size=self._size())

    def _size(self) -> tuple:
        return (self.nc, len(self._segs))

    def concrete_cells(self) -> np.ndarray:
        """ids of the cells whose material is ConcreteDamage"""
        return np.flatnonzero(np.isin(self.mat_of, [i for i, m in enumerate(self.materials) if isinstance(m, ConcreteDamage)]))

    def shrinkage(self, eps_sh, cells=None, *, at: str = "gauss") -> Eigenstrain:
        """free shrinkage (eps_sh > 0 shortens; a number or a function of x like initial_strain's): ε0 = −eps_sh·I, on
        the concrete (ConcreteDamage) cells by default — not on steel, rust or bearing pads; give cells for others"""
        f = (lambda x: -np.asarray(eps_sh(x), float)) if callable(eps_sh) else -float(eps_sh)
        if cells is None:
            cells = self.concrete_cells()
            if not len(cells):
                raise ValueError("no ConcreteDamage cells: give the cells that shrink")
        return self.initial_strain(f, cells, at=at)

    def _alpha6(self, mi: int) -> np.ndarray:
        m = self.materials[mi]
        a = np.asarray(m.alpha, float) * np.ones(3)
        return np.r_[a, np.zeros(3)]

    def thermal(self, dT, cells=None, *, bars=True, at: str = "gauss") -> Eigenstrain:
        """a temperature change ΔT (a number or a function of x (3, …) → (…)): every cell expands by its material's
        alpha (ConcreteDamage: 1e-5 by default; ElasticIso, ElasticOrtho: 0 unless given — warned about, an error when no
        selected cell expands), every bar segment by its bar's alpha (bar(..., alpha=)) at ΔT of its midpoint.
        bars=True: the segments of all bars whose midpoint lies in the selected cells (all bars when cells is None); a
        list of bar indices (those bars, wherever they are); False: none. Dowels do not expand"""
        c = self.cells_of(cells)
        X = self._points_for(c, at)
        T = np.broadcast_to(np.asarray(dT(np.moveaxis(X, -1, 0)) if callable(dT) else dT, float), X.shape[:2])
        A = np.array([self._alpha6(mi) for mi in range(len(self.materials))])[self.mat_of[c]] if len(c) else np.zeros((0, 6))
        cold = np.unique(self.mat_of[c][~np.any(A[:, :3], axis=1)]) if len(c) else []
        if len(c) and len(cold) == len(np.unique(self.mat_of[c])):
            raise ValueError("no selected cell has a coefficient of thermal expansion: give alpha= to the materials")
        if len(cold):
            names = sorted({type(self.materials[k]).__name__ for k in cold})
            warnings.warn(f"{int(np.isin(self.mat_of[c], cold).sum())} heated cells ({', '.join(names)}) have alpha = 0 "
                          "and do not expand: give them alpha= (steel 1.2e-5) or leave them out of `cells`", stacklevel=2)
        vals = A[:, None, :] * T[..., None]
        out = Eigenstrain(c, vals, size=self._size())
        if bars is True:
            sel = range(len(self.bars))
            inside = None if cells is None else set(c.tolist())
        else:
            sel, inside = ([] if bars is False or bars is None else [int(b) for b in np.atleast_1d(bars)]), None
        P = self.points
        segs = []
        for b in sel:
            for k in self.bars[b]["segs"]:
                if inside is not None:
                    hit = self.loc.find(0.5 * (P[self._segs[k][0]] + P[self._segs[k][1]]))
                    if hit is None or hit[0] not in inside:
                        continue
                segs.append(int(k))
        if segs:
            mid = np.array([0.5 * (P[self._segs[k][0]] + P[self._segs[k][1]]) for k in segs])
            Tm = np.broadcast_to(np.asarray(dT(mid.T) if callable(dT) else dT, float), (len(segs),))
            ab = np.array([self.bars[self._segs[k][2]]["alpha"] for k in segs])
            zero = sorted({self.bars[self._segs[k][2]]["name"] for k in segs if self.bars[self._segs[k][2]]["alpha"] == 0})
            if zero and np.any(A[:, :3]):
                warnings.warn(f"bars {zero} have alpha = 0 and restrain the thermal strain of the concrete; give "
                              "bar(..., alpha=1.2e-5) for steel (or bars=False to leave them out)", stacklevel=2)
            keep = ab != 0
            for k in np.array(segs)[keep]:
                self._want_free_strain(self._segs[k][2])
            out = Eigenstrain(c, vals, np.array(segs, np.int64)[keep], (ab * Tm)[keep], size=self._size())
        return out

    def _want_free_strain(self, b: int):
        """bars take free strains through springs in series with their segments, which exist only if the bar was
        built with them: bar(..., free_strain=True), or any bar with alpha ≠ 0, or one given a free strain before the
        first analysis"""
        if self.bars[b]["free_strain"]:
            return
        if self.built:
            raise ValueError(f"bar {self.bars[b]['name']!r} was built without free strains: give bar(..., "
                             "free_strain=True) (or alpha) before the first analysis")
        self.bars[b]["free_strain"] = True

    def bar_strain(self, bars, eps0) -> Eigenstrain:
        """a free strain along bars (an index or a list): a number or a function of the segment midpoints x (3, k) → (k,).
        The bars must have been made with bar(..., free_strain=True) (or alpha ≠ 0) when the model is already built"""
        bl = [int(b) for b in np.atleast_1d(bars)]
        for b in bl:
            self._want_free_strain(b)
        segs = [int(k) for b in bl for k in self.bars[b]["segs"]]
        P = self.points
        mid = np.array([0.5 * (P[self._segs[k][0]] + P[self._segs[k][1]]) for k in segs]).reshape(-1, 3)
        v = np.broadcast_to(np.asarray(eps0(mid.T) if callable(eps0) else eps0, float), (len(segs),))
        return Eigenstrain(np.zeros(0, np.int64), np.zeros((0, 8, 6)), np.array(segs, np.int64), v.copy(),
                           size=self._size())

    # ── building the OpenSees model ──────────────────────────────────────
    def _live(self):
        if _registry()[0] is not self:
            raise RuntimeError("this OpsSolid is stale: OpenSees holds a single global model and a newer one replaced it")

    def _build(self):
        ops = _ops()
        self.ops = ops
        ops.wipe()
        _registry()[0] = self
        if self.log:
            ops.logFile(self.log, "-noEcho")
        ops.model("basic", "-ndm", 3, "-ndf", 3)
        P = self.points
        for n, p in enumerate(P):
            if n not in self._six:
                ops.node(n + 1, *map(float, p))
        if self._six:
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            for n in sorted(self._six):
                ops.node(n + 1, *map(float, P[n]))
            ops.model("basic", "-ndm", 3, "-ndf", 3)
        # solid materials: one per (material, crack-band bin)
        self.ops_mat = np.zeros(self.nc, np.int64)
        self.h_used = self.h.copy()
        tag = 0
        clipped = 0
        for mi, m in enumerate(self.materials):
            cells = np.flatnonzero(self.mat_of == mi)
            if isinstance(m, ConcreteDamage):
                hmax = m.h_max()
                key = np.round(np.log(self.h[cells]) / np.log(1.01)).astype(np.int64)
                for kv in np.unique(key):
                    tag += 1
                    hb = float(1.01 ** kv)
                    ops.nDMaterial("ASDConcrete3D", tag, *m.ops_args(hb))
                    sel = cells[key == kv]
                    self.ops_mat[sel] = tag
                    self.h_used[sel] = min(hb, 0.99 * hmax)
                    if hb > 0.99 * hmax:
                        clipped += len(sel)
            elif isinstance(m, ElasticOrtho):
                tag += 1
                ops.nDMaterial("ElasticOrthotropic", tag, *map(float, m.E), *map(float, m.nu), *map(float, m.G), float(m.rho))
                self.ops_mat[cells] = tag
            elif isinstance(m, ElasticIso):
                tag += 1
                ops.nDMaterial("ElasticIsotropic", tag, float(m.E), float(m.nu), float(m.rho))
                self.ops_mat[cells] = tag
            else:
                raise TypeError(f"unknown solid material {type(m).__name__}")
        self.h_clipped = clipped
        # every material inside OpenSees' InitStrain wrapper (zero until a stage applies a free strain): σ = f(ε + ε_init)
        self._wrap = {}
        for t in np.unique(self.ops_mat):
            tag += 1
            ops.nDMaterial("InitStrain", tag, int(t), 0.0)
            self._wrap[int(t)] = tag
        self.ops_mat = np.array([self._wrap[int(t)] for t in self.ops_mat], np.int64)
        if clipped:
            warnings.warn(f"{clipped} concrete cells are larger than the snap-back limit of their softening law and got the "
                          "law of h_max (they dissipate more than Gf); refine the mesh where cracks are expected",
                          stacklevel=3)
        # solid elements
        e = 0
        for k, conn in self.cell_conn():
            name = self.element if k == "hex8" else "FourNodeTetrahedron"
            for c in conn:
                ops.element(name, e + 1, *[int(v) + 1 for v in c], int(self.ops_mat[e]))
                e += 1
        self.ngp = {"hex8": 8, "tet4": 1}
        # bars
        mt = 100000
        self.bar_mat = []
        tag_e = self.nc
        self.seg_tag = np.zeros(len(self._segs), np.int64)
        for b in self.bars:
            mat = b["material"]
            mt += 3
            if mat[0] == "FRP":
                ops.uniaxialMaterial("Elastic", mt, float(mat[1]))
                ops.uniaxialMaterial("MinMax", mt + 1, mt, "-min", -float(mat[2]), "-max", float(mat[2]))
                use = mt + 1
            else:
                ops.uniaxialMaterial(mat[0], mt, *map(float, mat[1:]))
                use = mt
            if b["prestress"]:
                ops.uniaxialMaterial("InitStressMaterial", mt + 2, use, float(b["prestress"]))
                use = mt + 2
            b["ops_mat"] = use
        mt += 3  # past the last bar's three tags
        for s, (i, j, bi) in enumerate(self._segs):
            if self.bars[bi]["free_strain"]:
                continue  # built with its series spring below, once auxiliary nodes can be numbered
            tag_e += 1
            ops.element("Truss", tag_e, i + 1, j + 1, self.bars[bi]["area"], self.bars[bi]["ops_mat"])
            self.seg_tag[s] = tag_e
        # dowels: beam elements on 6-dof nodes (frame elements need the 6-dof model builder)
        self.dseg_tag = np.zeros(len(self._dsegs), np.int64)
        transf: dict = {}
        sec_tag: dict = {}
        if self._dsegs:
            ops.model("basic", "-ndm", 3, "-ndf", 6)
        for s_, (i, j, k) in enumerate(self._dsegs):
            dw = self.dowels[k]
            ax = P[j] - P[i]
            ax = ax / np.linalg.norm(ax)
            ref = np.array([0.0, 0.0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            vz = np.cross(ax, np.cross(ref, ax))
            vz /= np.linalg.norm(vz)
            key = tuple(np.round(vz, 9))
            if key not in transf:
                transf[key] = 200 + len(transf)
                ops.geomTransf("Linear", transf[key], *map(float, vz))
            tag_e += 1
            if dw["fiber"] is not None:
                from ..frame.opensees import build_fiber_section

                if k not in sec_tag:  # material tags 2 000 000 + 100 k: clear of the bars' (100 000 + …)
                    sec_tag[k] = build_fiber_section(ops, dw["fiber"], 300 + k, mat_base=2_000_000 + 100 * k)
                    ops.beamIntegration("Lobatto", 300 + k, sec_tag[k], 5)
                ops.element("forceBeamColumn", tag_e, i + 1, j + 1, transf[key], 300 + k)
            else:
                E_, A_, Iy, Iz, G_, J_ = dw["section"]
                ops.element("elasticBeamColumn", tag_e, i + 1, j + 1, A_, E_, G_, J_, Iy, Iz, transf[key])
            self.dseg_tag[s_] = tag_e
        # a dowel tied in translation only would turn freely about its own axis (a zero-energy mode): hold that rotation at
        # its first embedded node by a spring as stiff as one element of the dowel in torsion
        self._dowel_torsion = []
        for k, dw in enumerate(self.dowels):
            inside = [n for n in dw["nodes"] if n in self._tie]
            if not inside or any(n in self._rot for n in dw["nodes"]):
                continue
            n = inside[0]
            E_, A_, Iy, Iz, G_, J_ = dw["section"] if dw["fiber"] is None else (0, 0, 0, 0, dw["fiber"].GJ, 1.0)
            ax = dw["points"][-1] - dw["points"][0]
            ax = ax / np.linalg.norm(ax)
            y = np.cross(ax, (0.0, 0.0, 1.0) if abs(ax[2]) < 0.9 else (1.0, 0.0, 0.0))
            seg_len = float(np.min(np.diff(dw["s"]))) if len(dw["s"]) > 1 else 1.0
            self._dowel_torsion.append((n, ax, y / np.linalg.norm(y), G_ * J_ / seg_len))
        if self._dsegs:
            ops.model("basic", "-ndm", 3, "-ndf", 3)
        # embedding ties: the node follows the cell's own interpolation (trilinear in a hexahedron, linear in a
        # tetrahedron), built from ASDEmbeddedNodeElement penalty ties, see _embed
        used = {n for i, j, _ in self._segs for n in (i, j)} | {n for i, j, _ in self._dsegs for n in (i, j)}
        used |= {a for _, a, _ in self._sliders} | {bd["anchor"] for bd in self._bonds}
        self._next_node, self._next_ele = self.n_nodes, tag_e
        self.n_aux = 0
        for j, tie in self._tie.items():
            if j in used and j not in self._links:
                self._embed(j, tie)
        # deviators of unbonded tendons: tendon node and anchor joined across the tendon only
        k_pen = self.penalty * self.E_ref * float(np.median(self.h))
        for j, a, sd in self._sliders:
            y = np.cross(sd, (0.0, 0.0, 1.0) if abs(sd[2]) < 0.9 else (1.0, 0.0, 0.0))
            mt += 1
            ops.uniaxialMaterial("Elastic", mt, k_pen)
            self._next_ele += 1
            ops.element("zeroLength", self._next_ele, a + 1, j + 1, "-mat", mt, mt, "-dir", 2, 3,
                        "-orient", *map(float, sd), *map(float, y / np.linalg.norm(y)))
        # free strains of bars: segment i–j = Truss i–m + zeroLength m–j (m at j), the zeroLength along the bar an
        # InitStrainMaterial around a stiff Elastic (its imposed elongation is ε0·L), across it stiff. Elastic, so the
        # commit InitStrainMaterial makes on every parameter update is harmless; the bar's own (yielding) material only
        # ever sees converged states
        self._fs = {}  # segment → (zeroLength tag, segment length)
        P_ = self.points
        if any(self.bars[bi]["free_strain"] for _, _, bi in self._segs):
            mt += 1
            m_pen_fs = mt
            ops.uniaxialMaterial("Elastic", m_pen_fs, k_pen)
        for s_, (i, j, bi) in enumerate(self._segs):
            b = self.bars[bi]
            if not b["free_strain"]:
                continue
            L = float(np.linalg.norm(P_[j] - P_[i]))
            m = self._aux_node(P_[j])
            self._next_ele += 1
            ops.element("Truss", self._next_ele, i + 1, m, b["area"], b["ops_mat"])
            self.seg_tag[s_] = self._next_ele
            mt += 2
            ops.uniaxialMaterial("Elastic", mt - 1, self.penalty * max(_bar_E(b["material"]), self.E_ref) * b["area"] / L)
            ops.uniaxialMaterial("InitStrainMaterial", mt, mt - 1, 0.0)
            t = (P_[j] - P_[i]) / L
            y = np.cross(t, (0.0, 0.0, 1.0) if abs(t[2]) < 0.9 else (1.0, 0.0, 0.0))
            self._next_ele += 1
            ops.element("zeroLength", self._next_ele, m, j + 1, "-mat", mt, m_pen_fs, m_pen_fs, "-dir", 1, 2, 3,
                        "-orient", *map(float, t), *map(float, y / np.linalg.norm(y)))
            self._fs[s_] = (self._next_ele, L)
        # bond springs: along the bar the bond law, across it stiff
        if self._bonds:
            mt += 1
            m_pen = mt
            ops.uniaxialMaterial("Elastic", m_pen, k_pen)
        for bd in self._bonds:
            bar = self.bars[bd["bar"]]
            mt += 1
            bar["bond"].ops_material(ops, mt, bar["perimeter"] * bd["L"])
            t = bd["t"]
            y = np.cross(t, (0.0, 0.0, 1.0) if abs(t[2]) < 0.9 else (1.0, 0.0, 0.0))
            self._next_ele += 1
            ops.element("zeroLength", self._next_ele, bd["anchor"] + 1, bd["node"] + 1, "-mat", mt, m_pen, m_pen,
                        "-dir", 1, 2, 3, "-orient", *map(float, t), *map(float, y / np.linalg.norm(y)))
            bd["tag"] = self._next_ele
        tag_e = self._next_ele
        for j, dn in self._links.items():
            ops.equalDOF(dn + 1, j + 1, 1, 2, 3)
        free = [j for j in range(self.N, self.n_nodes) if j not in self._tie]
        # supports, springs, ties
        self._next_ele, self._next_mat = tag_e, mt
        for n, f in self.fixed.items():
            ops.fix(n + 1, *f, *([0, 0, 0] if n in self._six else []))
            self._react.setdefault(n, []).append((n + 1, np.array(f, float)))
        for n, k in self._springs:
            self._ground(n, [a for a in range(3) if k[a]], [k[a] for a in range(3) if k[a]])
        for n, ax, y, kt in self._dowel_torsion:
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            self._next_node += 1
            g = self._next_node
            ops.node(g, *map(float, self.points[n]))
            ops.fix(g, 1, 1, 1, 1, 1, 1)
            ops.model("basic", "-ndm", 3, "-ndf", 3)
            self._next_mat += 1
            ops.uniaxialMaterial("Elastic", self._next_mat, float(kt))
            self._next_ele += 1
            ops.element("zeroLength", self._next_ele, n + 1, g, "-mat", self._next_mat, "-dir", 4,
                        "-orient", *map(float, ax), *map(float, y))
        for m, slaves, comps in self._ties:
            for n in slaves:
                ops.equalDOF(m + 1, n + 1, *[c + 1 for c in comps])
        self._free_bar_nodes = free
        self.built = True
        # initial state
        self.history.append(Step("initial", 0, 0.0, np.zeros((self.n_nodes, 3)), np.zeros((len(self.supported), 3)),
                                 self.supported, *self._read_fields(), {n: f(self) for n, f in self._watch.items()}))

    @property
    def supported(self) -> np.ndarray:
        """nodes that carry reactions: fixed components, springs, imposed displacements"""
        return np.array(sorted(self._react), np.int64)

    def _tie_element(self, c: int, retained, rot: bool = False):
        """ASDEmbeddedNodeElement: node tag c follows the linear interpolation of 3 (triangle) or 4 (tetrahedron) retained
        node tags; the 4th as a string (OpenSeesPy drops an int there silently)"""
        ops = self.ops
        self._next_ele += 1
        r = [int(v) for v in retained]
        if len(r) == 4:  # positive volume: the element takes det J / 6 as it is, a negative one is a negative penalty
            Xr = np.array([ops.nodeCoord(t) for t in r])
            if np.dot(np.cross(Xr[1] - Xr[0], Xr[2] - Xr[0]), Xr[3] - Xr[0]) < 0:
                r[2], r[3] = r[3], r[2]
        args = r[:3] + ([str(r[3])] if len(r) == 4 else [])
        ops.element("ASDEmbeddedNodeElement", self._next_ele, int(c), *args, *(["-rot"] if rot else []),
                    "-K", float(self.penalty * self.E_ref))

    def _aux_node(self, x) -> int:
        self._next_node += 1
        self.ops.node(self._next_node, *map(float, x))
        self.n_aux += 1
        return self._next_node

    def _embed(self, j: int, tie):
        """tie extra node j to its cell. Tetrahedra: one tie (linear, exact). Hexahedra: the trilinear interpolation
        u = Σ N_i(ξ, η, ζ) u_i, built exactly from 2-point linear interpolations nested along the cell's edges — four
        auxiliary nodes on the edges along ξ, two between them along η, and node j between those along ζ — each tied on
        an edge of a tetrahedron whose other two nodes (cell corners) get weight 0. Tetrahedra, not triangles: the
        triangle tie of ASDEmbeddedNodeElement in 3D drops the global z component of the point's offset when it computes
        its local coordinates (it zeroes the third column of J⁻¹ instead of the normal row), so it is wrong for every
        triangle that is not horizontal. A dowel node with rot=True is tied with its rotation to the
        sub-tetrahedron that holds it (linear interpolation; ASDEmbeddedNodeElement has no 8-node tie)."""
        cell, kind, nodes, loc = tie
        P = self.mesh.points
        if kind == "tet4":
            self._tie_element(j + 1, nodes + 1, rot=j in self._rot)
            return
        X = P[nodes]
        if j in self._rot:
            p = self.points[j]
            best = max(HEX_TETS, key=lambda t: _bary(X[t], p).min())
            self._tie_element(j + 1, nodes[best] + 1, rot=True)
            return
        xi = np.clip(loc, -1.0, 1.0)
        corner = {tuple(sg.astype(int)): k for k, sg in enumerate(_HEX_SIGNS)}

        def others(a, b):
            """two corners (tags) that make a well-shaped tetrahedron with the segment a–b: the farthest from its line,
            then the farthest from the plane through it"""
            d = b - a
            k1 = int(np.argmax(np.linalg.norm(np.cross(X - a, d), axis=1)))
            nrm = np.cross(d, X[k1] - a)
            k2 = int(np.argmax(np.abs((X - a) @ nrm)))
            return int(nodes[k1]) + 1, int(nodes[k2]) + 1

        def between(ta, xa, tb, xb, t, tag=None):
            """a node at (1 − t)·a + t·b tied to a and b"""
            x = xa + t * (xb - xa)
            c = tag if tag is not None else self._aux_node(x)
            self._tie_element(c, [ta, tb, *others(xa, xb)])
            return c, x

        tx, ty, tz = (1 + xi) / 2
        lvl1 = {}
        for sy in (-1, 1):
            for sz in (-1, 1):
                i0, i1 = corner[(-1, sy, sz)], corner[(1, sy, sz)]
                lvl1[sy, sz] = between(int(nodes[i0]) + 1, X[i0], int(nodes[i1]) + 1, X[i1], tx)
        lvl2 = {}
        for sz in (-1, 1):
            (ta, xa), (tb, xb) = lvl1[-1, sz], lvl1[1, sz]
            lvl2[sz] = between(ta, xa, tb, xb, ty)
        (ta, xa), (tb, xb) = lvl2[-1], lvl2[1]
        between(ta, xa, tb, xb, tz, tag=j + 1)

    def _ground(self, n: int, comps, ks, fixed=(0, 1, 2)) -> int:
        """a ground node at node n joined to it by springs ks in comps (zeroLength), fixed in `fixed` (and in rotation for
        dowel nodes); registered as a reaction source"""
        ops = self.ops
        self._next_node += 1
        g = self._next_node
        six = n in self._six
        if six:
            ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(g, *map(float, self.points[n]))
        if six:
            ops.model("basic", "-ndm", 3, "-ndf", 3)
        ops.fix(g, *[int(c in fixed) for c in range(3)], *([1, 1, 1] if six else []))
        mats = []
        for k in ks:
            self._next_mat += 1
            ops.uniaxialMaterial("Elastic", self._next_mat, float(k))
            mats.append(self._next_mat)
        self._next_ele += 1
        ops.element("zeroLength", self._next_ele, n + 1, g, "-mat", *mats, "-dir", *[c + 1 for c in comps])
        mask = np.zeros(3)
        mask[list(comps)] = 1.0
        self._react.setdefault(n, []).append((g, mask))
        return g

    # ── analysis ─────────────────────────────────────────────────────────
    def _pattern(self, F=None, imposed=None):
        ops = self.ops
        self._ts += 1
        self._pat += 1
        ops.timeSeries("Linear", self._ts)
        ops.pattern("Plain", self._pat, self._ts)
        if F is not None:
            F = np.asarray(F, float)
            if F.shape[0] < self.n_nodes:
                F = np.vstack([F, np.zeros((self.n_nodes - F.shape[0], 3))])
            for n in np.flatnonzero(np.abs(F).max(axis=1) > 0):
                ops.load(int(n) + 1, *map(float, F[n]), *([0.0] * 3 if n in self._six else []))
        for g, comps, vals in (imposed or ()):
            for c, v in zip(comps, vals):
                ops.sp(g, c + 1, float(v))
        return self._pat

    def _drivers(self, imposed) -> list:
        """imposed displacements are applied through stiff springs from driven ground nodes (penalty × E × median band):
        an sp constraint on the node itself would put the whole increment into the first layer of cells in the predictor
        and, with softening, crack it"""
        nodes, U = imposed
        U = np.asarray(U, float).reshape(len(nodes), 3)
        k = self.penalty * self.E_ref * float(np.median(self.h))
        out = []
        for n, u in zip(np.asarray(nodes, np.int64), U):
            comps = [c for c in range(3) if not np.isnan(u[c])]
            if not comps:
                continue
            n = int(n)
            if n in self._driven:
                raise ValueError(f"node {n} already has imposed displacements (one imposed stage per node)")
            if n in self.fixed and any(self.fixed[n][c] for c in comps):
                raise ValueError(f"node {n} is fixed in an imposed component")
            self._driven.add(n)
            self._driven_masks.append((n, np.isin(np.arange(3), comps)))
            g = self._ground(n, comps, [k] * len(comps), fixed=[c for c in range(3) if c not in comps])
            out.append((g, comps, u[comps]))
        return out

    ALGORITHMS = (("KrylovNewton",), ("Newton",), ("NewtonLineSearch", 0.8), ("ModifiedNewton", "-initial"))
    ALGORITHMS_IMPLEX = (("Newton",), ("KrylovNewton",), ("NewtonLineSearch", 0.8), ("ModifiedNewton", "-initial"))

    @property
    def algorithms(self):
        return self.ALGORITHMS_IMPLEX if self.implex else self.ALGORITHMS

    def _implex_time(self, dt: float, first: bool = False):
        """IMPL-EX extrapolates the damage increment of the last step by dtime / dtime_committed, where dtime is OpenSees'
        pseudo-time increment: the load-factor increment, which displacement control changes in every iteration (the step
        is then no longer linear and diverges). The time is therefore set here, per material point, as the step size
        relative to the stage's nominal step; at the first step of a stage the committed time is made huge, so that the
        stage starts without extrapolation"""
        if not self.implex:  # set before every attempt: a failed attempt reverts the material's time to the committed one
            return
        ops = self.ops
        for k, o in self.off.items():
            n = len(self.mesh.cells[k])
            if not n:
                continue
            for ip in range(self.ngp[k]):
                ops.setParameter("-val", float(dt), "-eleRange", int(o) + 1, int(o) + n, "material", str(ip + 1), "dTime")
                if first:
                    ops.setParameter("-val", 1e30, "-eleRange", int(o) + 1, int(o) + n, "material", str(ip + 1), "dTimeCommit")
        self._dt = dt

    def free_rigid_modes(self) -> int:
        """how many of the six rigid-body motions the supports leave free (fixed components, springs, displacement
        drivers); a free one makes the stiffness singular, and round-off (the penalty ties of bars) then wrecks the
        iterations. Checked at the start of every stage"""
        rows = []
        P = self.points
        c = P.mean(axis=0)
        held = [(n, np.array(f, float)) for n, f in self.fixed.items()]
        held += [(n, np.array(k, float) != 0) for n, k in self._springs]
        held += [(n, m) for n, m in self._driven_masks]
        for n, mask in held:
            r = P[n] - c
            for i in np.flatnonzero(mask):
                e = np.zeros(3)
                e[i] = 1.0
                rows.append(np.r_[e, np.cross(r, e)])  # e·(ω × r) = ω·(r × e)
        if not rows:
            return 6
        A = np.array(rows) / np.r_[np.ones(3), np.full(3, max(self.scale, 1e-300))]
        return 6 - int(np.linalg.matrix_rank(A, tol=1e-9 * np.abs(A).max()))

    def loose_bar_nodes(self) -> list[int]:
        """bar nodes held in fewer than three directions: not tied to the solid, not on a dowel, not supported, and the
        bars (and deviator ties) on them span less than 3D — two bars in line hold a node only along their line"""
        held = set(self._tie) | set(self._links) | self._six | set(self.fixed) | {n for n, _ in self._springs}
        held |= {bd["node"] for bd in self._bonds}
        dirs: dict[int, list] = {}
        P = self.points
        for i, j, _ in self._segs:
            d = P[j] - P[i]
            for n in (i, j):
                dirs.setdefault(n, []).append(d / np.linalg.norm(d))
        for j, a, sd in self._sliders:
            y = np.cross(sd, (0.0, 0.0, 1.0) if abs(sd[2]) < 0.9 else (1.0, 0.0, 0.0))
            dirs.setdefault(j, []).extend([y, np.cross(sd, y)])
        return [n for n, d in dirs.items() if n >= self.N and n not in held and np.linalg.matrix_rank(np.array(d), 1e-6) < 3]

    def _check_supports(self):
        k = self.free_rigid_modes()
        if k:
            warnings.warn(f"the supports leave {k} rigid-body motion(s) free: the stiffness is singular and the analysis "
                          "will not converge (check that rotations about all three axes are held)", stacklevel=4)
        loose = self.loose_bar_nodes()
        if loose:
            warnings.warn(f"{len(loose)} bar node(s) are held in fewer than three directions (e.g. at "
                          f"{np.round(self.points[loose[0]], 6).tolist()}): the stiffness is singular there", stacklevel=4)

    def _setup(self):
        ops = self.ops
        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer(self.numberer)
        ops.system("UmfPack")
        ops.test("NormDispIncr", self._tol_now(), self.iters, 0)
        ops.algorithm(*self.algorithms[0])

    def _tol_now(self) -> float:
        floor = 1e-10 * self.scale
        return max(floor, self._tol if self._tol is not None else 1e-8 * self.scale)

    def _step(self, analyze, retry, depth: int = 0, max_depth: int = 6) -> bool:
        """one step; on failure the other algorithms, then two half steps (recursively, down to 1/2⁶)"""
        ops = self.ops
        if analyze() == 0:
            return True
        for alg in self.algorithms[1:]:
            ops.algorithm(*alg)
            if analyze() == 0:
                ops.algorithm(*self.algorithms[0])
                return True
        ops.algorithm(*self.algorithms[0])
        if depth >= max_depth:
            return False
        retry(0.5)
        ok = self._step(analyze, retry, depth + 1, max_depth) and self._step(analyze, retry, depth + 1, max_depth)
        retry(2.0)
        return ok

    def _run(self, stage: Stage, steps: int, set_inc, analyze, record, stop=None, lam_of=None, hold: bool = False,
             pre=None):
        """`steps` steps (fewer when `stop` says so). hold: a stage without load increment — under IMPL-EX every step
        starts without extrapolation, i.e. a fixed-point iteration on the damage towards the implicit equilibrium, and
        the stage ends once a step changes the displacements by less than hold_tol × their norm (implicitly: after
        one step, which is already in equilibrium)"""
        ops = self.ops
        d = [1.0]
        first = [True]

        def retry(s):
            d[0] *= s
            set_inc(d[0])

        def attempt():
            self._implex_time(d[0], first[0] or hold)
            if pre is not None:  # free strains at the load factor this attempt aims at
                pre(d[0])
            return analyze()

        settled = False
        for k in range(steps):
            ok = self._step(attempt, retry)
            first[0] = False
            if not ok:
                stage.converged = False
                stage.message = f"step {k + 1}/{steps} did not converge (log: {self.log})"
                break
            rec = record is True or (record and (k + 1) % int(record) == 0) or k == steps - 1
            self.history.append(self._snapshot(stage.name, k + 1, lam_of(), rec))
            dU = float(np.linalg.norm(self.history[-1].U - self.history[-2].U))
            if dU > 0 and not hold:
                self._tol = self.rtol * dU
                ops.test("NormDispIncr", self._tol_now(), self.iters, 0)
            stage.steps.append(len(self.history) - 1)
            if hold:
                total = float(np.linalg.norm(self.history[-1].U))
                stage.change = dU / total if total > 0 else 0.0
                if stage.change <= self.hold_tol or not self.implex:
                    settled = True
                    stage.message = (f"held: settled after {k + 1} step(s) (last change {stage.change:.1e} of the "
                                     "displacements)" if self.implex else "held: in equilibrium (implicit)")
                    break
            if stop is not None and stop(self.history[-1]):
                stage.message = f"stopped by the stop condition at step {k + 1}"
                break
        if hold and stage.converged and not settled:
            stage.message = (f"held for {steps} steps without settling (last change {stage.change:.1e} of the "
                             f"displacements, limit {self.hold_tol:.0e}): give more steps")
            warnings.warn(f"stage {stage.name!r}: {stage.message}", stacklevel=3)
        h = self.history[-1]
        if stage.steps and h.cell is None:  # the last step of a stage always carries its fields
            h.cell, h.bar, h.dowel = self._read_fields()
        self.stages.append(stage)
        if not stage.converged:
            warnings.warn(f"stage {stage.name!r}: {stage.message}", stacklevel=3)
        return stage

    def _settle_prestress(self):
        """IMPL-EX and prestressed bars: the initial stresses enter in full in the first step, so the damage they cause is
        a jump, not a rate, and extrapolating it would be wrong. Before the first stage they are therefore settled in a
        stage of their own ("prestress"): hold steps without extrapolation until the displacements stop changing"""
        if not (self.implex and any(b["prestress"] for b in self.bars)) or self.stages:
            return
        ops = self.ops
        self._setup()
        ops.integrator("LoadControl", 0.0)  # the pattern of the coming stage stays at λ = 0
        ops.analysis("Static")
        self._run(Stage("prestress", "hold"), self.hold_steps, lambda s: None, lambda: ops.analyze(1), False,
                  lam_of=lambda: 0.0, hold=True)

    def static(self, F=None, steps: int = 1, *, imposed=None, strain: Eigenstrain | None = None,
               name: str | None = None, record=True, stop=None) -> Stage:
        """load control: load pattern F (n_nodes, 3) (or None: prestress and initial stresses only), imposed
        displacements imposed=(node ids, values (n, 3), NaN = free) and free strains strain (an Eigenstrain from
        initial_strain, shrinkage, thermal, bar_strain) applied together in `steps` equal increments of the load factor
        λ, then held for the following stages (unload with −F; take a free strain back with −strain). record: True (cell
        and bar results every step), an integer k (every k-th step and the last) or False (the last step only);
        displacements and reactions are kept for every step. stop(step) → True ends the stage (the free strain keeps the
        λ reached)."""
        if not self.built:
            self._build()
        self._live()
        ops = self.ops
        hold = imposed is None and (F is None or not np.any(F)) and strain is None
        drv = self._drivers(imposed) if imposed is not None else None
        self._check_supports()
        p = self._pattern(F, drv)
        self._settle_prestress()  # at λ = 0 of this stage's pattern: imposed displacements held at zero meanwhile
        self._setup()
        ops.integrator("LoadControl", 1.0 / steps)
        ops.analysis("Static")
        st = Stage(name or f"stage{len(self.stages) + 1}", "hold" if hold else "static")
        pre = None
        if strain is not None:
            self._eig_begin(strain)
            pre = lambda d: self._eig_set(ops.getLoadFactor(p) + d / steps)  # noqa: E731  (λ before the step is the committed one)
        try:
            self._run(st, steps, lambda s: ops.integrator("LoadControl", s / steps), lambda: ops.analyze(1), record,
                      stop, lambda: ops.getLoadFactor(p), hold=hold, pre=pre)
        finally:
            if strain is not None:
                self._eig_end(ops.getLoadFactor(p))
        ops.loadConst("-time", 0.0)
        return st

    # free strains of a stage: OpenSees parameters on the InitStrain wrappers, updated before every attempt
    def _eig_begin(self, S: Eigenstrain):
        """one OpenSees parameter per group of material points that share the stage's (start value, increment) of a
        component — a uniform shrinkage of a whole group is a single parameter — so a step costs one update per group"""
        ops = self.ops
        if S.size is not None and S.size != self._size():
            raise ValueError(f"this free strain was made for another model ({S.size[0]} cells, {S.size[1]} bar segments; "
                             f"this one has {self.nc} and {len(self._segs)})")
        if self._e0 is None:
            self._e0 = np.zeros((self.nc, 8, 6))
            self._e0b = np.zeros(len(self._segs))
        cells = np.asarray(S.cells, np.int64)
        inc = np.asarray(S.values, float).reshape(len(cells), 8, 6)
        base = self._e0[cells]
        tet = np.zeros(len(cells), bool)
        if "tet4" in self.off:
            o = self.off["tet4"]
            tet = (cells >= o) & (cells < o + len(self.mesh.cells["tet4"]))
        same = tet | (np.all(inc == inc[:, :1], axis=(1, 2)) & np.all(base == base[:, :1], axis=(1, 2)))
        rc, rg, rk = [cells[same]], [np.zeros(int(same.sum()), np.int64)], [np.c_[base[same, 0], inc[same, 0]]]
        nu = ~same
        if nu.any():  # varying within the cell: one row per integration point
            rc.append(np.repeat(cells[nu], 8))
            rg.append(np.tile(np.arange(1, 9), int(nu.sum())))
            rk.append(np.c_[base[nu].reshape(-1, 6), inc[nu].reshape(-1, 6)])
        rc, rg, rk = np.concatenate(rc), np.concatenate(rg), np.concatenate(rk)
        params = []
        keys, inv = np.unique(rk, axis=0, return_inverse=True) if len(rk) else (np.zeros((0, 12)), np.zeros(0, np.int64))
        inv = np.asarray(inv).ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
        for u, key in enumerate(keys):
            b, i = key[:6], key[6:]
            if not i.any():
                continue
            iso = i[0] == i[1] == i[2] and not i[3:].any() and b[0] == b[1] == b[2]
            comps = [("eps0", 0)] if iso else [("eps0_" + VOIGT[k], k) for k in range(6) if i[k] != 0]
            rows = order[bounds[u]:bounds[u + 1]]
            for nm, k in comps:
                self._ptag += 1
                for r, row in enumerate(rows):
                    addr = ["element", int(rc[row]) + 1] + (["material", str(int(rg[row]))] if rg[row] else []) + [nm]
                    (ops.parameter if r == 0 else ops.addToParameter)(self._ptag, *addr)
                params.append((self._ptag, float(b[k]), float(i[k])))
        segs = np.asarray(S.segs, np.int64)
        if len(segs):
            miss = [int(k) for k in segs if int(k) not in self._fs]
            if miss:
                raise ValueError(f"free strain on {len(miss)} segment(s) of bars built without free strains "
                                 f"({self.bars[self._segs[miss[0]][2]]['name']!r}): give bar(..., free_strain=True)")
            Ls = np.array([self._fs[int(k)][1] for k in segs])
            sk = np.c_[self._e0b[segs] * Ls, np.asarray(S.seg_values, float) * Ls]  # imposed elongations
            keys, inv = np.unique(sk, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            for u, (b, i) in enumerate(keys):
                if i == 0:
                    continue
                self._ptag += 1
                for r, sg in enumerate(segs[inv == u]):
                    (ops.parameter if r == 0 else ops.addToParameter)(self._ptag, "element", int(self._fs[int(sg)][0]),
                                                                       "material", "1", "epsInit")
                params.append((self._ptag, float(b), float(i)))
        self._eig = dict(S=S, cells=cells, inc=inc, params=params, lam=0.0)

    def _eig_set(self, lam: float):
        """the free strain at load factor λ of the stage: ε0 = start + λ·increment; the wrappers take ε_init = −ε0"""
        upd = self.ops.updateParameter
        for t, b, i in self._eig["params"]:
            upd(t, -(b + lam * i))
        self._eig["lam"] = float(lam)

    def _eig_end(self, lam: float):
        """back to the committed λ (a failed attempt may have left its own), stored, parameters removed"""
        self._eig_set(lam)
        e = self._eig
        np.add.at(self._e0, e["cells"], lam * e["inc"])
        np.add.at(self._e0b, np.asarray(e["S"].segs, np.int64), lam * np.asarray(e["S"].seg_values, float))
        for t, _, _ in e["params"]:
            self.ops.remove("parameter", t)
        self._eig = None

    def _eig_now(self) -> tuple[np.ndarray, np.ndarray]:
        """current free strain: (nc, 6) cell averages, (n_segments,) along the bars"""
        nc, nb = self.nc, len(self._segs)
        if self._e0 is None:
            return np.zeros((nc, 6)), np.zeros(nb)
        c = self._e0.mean(axis=1)
        b = self._e0b.copy()
        e = self._eig
        if e is not None and e["lam"]:
            np.add.at(c, e["cells"], e["lam"] * e["inc"].mean(axis=1))
            np.add.at(b, np.asarray(e["S"].segs, np.int64), e["lam"] * np.asarray(e["S"].seg_values, float))
        return c, b

    def push(self, F, node, dof: int, target: float, steps: int = 50, *, name: str | None = None, record=True,
             stop=None) -> Stage:
        """displacement control: load pattern F scaled by λ so that component dof (0, 1, 2) of `node` (a node id, or a
        point → nearest solid node) moves by `target` in `steps` equal increments; follows the response past the peak.
        The step's λ is Step.lam; the loads stay applied afterwards."""
        if not self.built:
            self._build()
        self._live()
        ops = self.ops
        n = int(node) if np.ndim(node) == 0 else self.nearest(node)
        self._check_supports()
        p = self._pattern(F)
        self._settle_prestress()
        self._setup()
        inc = target / steps
        ops.integrator("DisplacementControl", n + 1, dof + 1, inc)
        ops.analysis("Static")
        st = Stage(name or f"stage{len(self.stages) + 1}", "push")
        self._run(st, steps, lambda s: ops.integrator("DisplacementControl", n + 1, dof + 1, inc * s),
                  lambda: ops.analyze(1), record, stop, lambda: ops.getLoadFactor(p))
        ops.loadConst("-time", 0.0)
        return st

    # ── reading results ──────────────────────────────────────────────────
    def watch(self, name: str, fn):
        """evaluate fn(self) on the live OpenSees model after every converged step (and for the initial state, when
        given before the first analysis); the values are in Step.watch[name], all of them in watched(name). For
        quantities the recorded fields do not hold, e.g. the forces of elements on some nodes
        (femotion.continuum.corrosion.interface_pressure)"""
        self._watch[name] = fn

    def watched(self, name: str) -> list:
        """the values of a watch over the history (None where it was not watched yet)"""
        return [h.watch.get(name) for h in self.history]

    def element_forces(self, cells) -> list:
        """the nodal resisting forces of solid cells at the current state, one (nodes of the cell, 3) array per cell in the
        cell's node order: the forces the nodes exert on the cell, in equilibrium with its stresses (to sum on a cut)"""
        self._live()
        return [np.asarray(self.ops.eleResponse(int(e) + 1, "forces"), float).reshape(-1, 3) for e in np.atleast_1d(cells)]

    def _snapshot(self, stage: str, k: int, lam: float, record: bool) -> Step:
        ops = self.ops
        U = np.array([ops.nodeDisp(n + 1)[:3] for n in range(self.n_nodes)])
        ops.reactions()
        sup = self.supported
        R = np.zeros((len(sup), 3))
        for i, n in enumerate(sup):  # reactions on the fixed components, of springs and of displacement drivers
            for tag, mask in self._react[int(n)]:
                R[i] += np.asarray(ops.nodeReaction(tag)[:3], float) * mask
        cell, bar, dowel = self._read_fields() if record else (None, None, None)
        return Step(stage, k, float(lam), U, R, sup, cell, bar, dowel, {n: f(self) for n, f in self._watch.items()})

    def _read_fields(self) -> tuple[dict, dict, dict]:
        """cell averages over the integration points: stress, strain (the strain the material sees, ε − ε0; engineering
        shear, order xx yy zz xy yz zx), eps0 (the free strain ε0; the strain of the displacements is strain + eps0),
        inelastic strain, crack strain and normal, crack width, cracked / crushed flags; bar axial force, strain (of the
        displacements) and eps0; dowel element forces"""
        ops = self.ops
        nc = self.nc
        S = np.zeros((nc, 6))
        Eps = np.zeros((nc, 6))
        Ein = np.zeros((nc, 6))
        xt = np.zeros(nc)
        xc = np.zeros(nc)
        e = 0
        for k, conn in self.cell_conn():
            ng = self.ngp[k]
            for _ in range(len(conn)):
                s = np.asarray(ops.eleResponse(e + 1, "stresses"), float).reshape(ng, 6)
                g = np.asarray(ops.eleResponse(e + 1, "strains"), float).reshape(ng, 6)
                m = self.materials[self.mat_of[e]]
                S[e], Eps[e] = s.mean(0), g.mean(0)
                Ein[e] = (g - self._compliance(m, s)).mean(0)
                if isinstance(m, ConcreteDamage):
                    x = np.array([ops.eleResponse(e + 1, "material", str(i + 1), "equivalentTotalStrain") for i in range(ng)])
                    xt[e], xc[e] = x[:, 0].max(), x[:, 1].max()
                e += 1
        T = np.zeros((nc, 3, 3))
        for a, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]):
            v = Ein[:, a] if a < 3 else Ein[:, a] / 2
            T[:, i, j] = v
            T[:, j, i] = v
        lam, vec = np.linalg.eigh(T)
        e1, n1 = lam[:, 2], vec[:, :, 2]
        big = np.abs(n1).argmax(axis=1)  # a normal has no sign: make its largest component positive
        n1 = n1 * np.sign(n1[np.arange(nc), big])[:, None]
        cracked = np.zeros(nc, bool)
        crushed = np.zeros(nc, bool)
        epre = np.zeros(nc)
        for mi, m in enumerate(self.materials):
            if isinstance(m, ConcreteDamage):
                sel = self.mat_of == mi
                cracked[sel] = xt[sel] > m.eps_t_peak * (1 + 1e-6)
                crushed[sel] = xc[sel] > m.eps_c_peak * (1 + 1e-6)
                epre[sel] = m.eps_in_peak
        ecr = np.where(cracked, np.maximum(e1 - epre, 0.0), 0.0)
        e0c, e0b = self._eig_now()
        cell = dict(stress=S, strain=Eps, eps_in=Ein, crack_strain=ecr, crack_normal=n1 * (ecr > 0)[:, None],
                    width=ecr * self.h_used, cracked=cracked, crushed=crushed, eq_strain=np.c_[xt, xc], eps0=e0c)
        nseg = len(self._segs)
        N = np.zeros(nseg)
        dL = np.zeros(nseg)
        for s in range(nseg):
            N[s] = ops.eleResponse(int(self.seg_tag[s]), "axialForce")[0]
            dL[s] = ops.eleResponse(int(self.seg_tag[s]), "basicDeformation")[0]
        L0, A = self._seg_geometry()
        slip = np.array([ops.eleResponse(bd["tag"], "basicDeformation")[0] for bd in self._bonds])
        fb = np.array([ops.eleResponse(bd["tag"], "basicForce")[0] for bd in self._bonds])
        pl = np.array([self.bars[bd["bar"]]["perimeter"] * bd["L"] for bd in self._bonds])
        bar = dict(N=N, stress=N / A, strain=dL / L0, eps0=e0b, slip=slip, tau=fb / pl if len(pl) else fb)
        nd = len(self._dsegs)
        lf = np.array([ops.eleResponse(int(t), "localForce") for t in self.dseg_tag]).reshape(nd, 12)
        dowel = dict(local=lf, N=lf[:, 6], V=np.hypot(lf[:, 1], lf[:, 2]),
                     M=np.maximum(np.hypot(lf[:, 4], lf[:, 5]), np.hypot(lf[:, 10], lf[:, 11])))
        return cell, bar, dowel

    def _seg_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        P = self.points
        if not self._segs:
            return np.zeros(0), np.zeros(0)
        L0 = np.array([np.linalg.norm(P[j] - P[i]) for i, j, _ in self._segs])
        A = np.array([self.bars[b]["area"] for _, _, b in self._segs])
        return L0, A

    # ── post-processing ──────────────────────────────────────────────────
    def recorded(self) -> list[int]:
        """history indices of the steps with cell and bar results"""
        return [i for i, h in enumerate(self.history) if h.cell is not None]

    def _rec(self, step) -> Step:
        h = step if isinstance(step, Step) else self.history[step]
        if h.cell is None:
            raise ValueError("this step has no cell results (see record= of the analysis)")
        return h

    def crack_widths(self, step=-1) -> np.ndarray:
        """crack width w = ε_cr · h of every solid cell (0 where not cracked)"""
        return self._rec(step).cell["width"]

    def width_history(self, *path, **kw) -> tuple[np.ndarray, np.ndarray]:
        """the largest crack width at every recorded step: of any cell, or of the cracks along a line or polyline
        (cracks_along(*path, **kw)). Returns (history indices, widths): interpolate the load at a width limit, e.g.
        np.interp(0.2, w, P[idx]) while the widths grow"""
        idx = np.array(self.recorded())
        if not path:
            return idx, np.array([self.history[i].cell["width"].max() for i in idx])
        return idx, np.array([self.cracks_along(*path, step=int(i), **kw)["max_width"] for i in idx])

    def first_crack(self) -> int | None:
        """history index of the first recorded step with a cracked cell (record every step to find it to one step)"""
        return next((i for i in self.recorded() if self.history[i].cell["cracked"].any()), None)

    def reaction(self, at=None, step=-1) -> np.ndarray:
        """resultant reaction (3,) at the supported nodes selected by `at` (predicate on coordinates, mask or node ids;
        default all supported nodes)"""
        h = self.history[step]
        if at is None:
            return h.R.sum(axis=0)
        sel = np.asarray(at, np.int64) if not callable(at) and np.asarray(at).dtype != bool else self.nodes(at)
        return h.R[np.isin(h.Rn, sel)].sum(axis=0)

    def curve(self, node, dof: int, load_dof: int | None = None, sign: float = -1.0, at=None, stages=None):
        """load–displacement curve over all steps (or those of the given stage names): component dof of `node` (id or
        point) and sign × the reaction resultant along load_dof (default dof) at the supports selected by `at` (default
        all; sign −1 turns reactions into the applied load)"""
        n = int(node) if np.ndim(node) == 0 else self.nearest(node)
        ld = dof if load_dof is None else load_dof
        idx = range(len(self.history)) if stages is None else \
            [0] + [i for st in self.stages if st.name in set(stages) for i in st.steps]
        u = np.array([self.history[i].U[n, dof] for i in idx])
        F = np.array([sign * self.reaction(at, i)[ld] for i in idx])
        return u, F

    def bar_results(self, bar: int, step=-1) -> dict:
        """axial force, stress and strain of the segments of one bar: s (distance of segment midpoints from its start), N,
        stress, strain (of the element's elongation; the prestrain of a prestressed bar comes on top), eps0 (its free
        strain); with a bond-slip law also bond=dict(s, slip (bar relative to the concrete, along the bar), tau (bond
        stress)) at its nodes"""
        h = self._rec(step)
        b = self.bars[bar]
        segs = b["segs"]
        P = self.points
        mids = np.array([0.5 * (P[self._segs[k][0]] + P[self._segs[k][1]]) for k in segs])
        s = np.linalg.norm(mids - b["points"][0], axis=1) if len(segs) else np.zeros(0)
        if len(segs) and len(b["points"]) > 2:  # along a polyline: project on the arc length
            s = np.array([self._arc(b, m) for m in mids])
        out = dict(s=s, x=mids, N=h.bar["N"][segs], stress=h.bar["stress"][segs], strain=h.bar["strain"][segs],
                   eps0=h.bar["eps0"][segs])
        if len(b["bonds"]):
            k = b["bonds"]
            out["bond"] = dict(s=np.array([self._bonds[i]["s"] for i in k]), slip=h.bar["slip"][k], tau=h.bar["tau"][k])
        return out

    def dowel_results(self, dowel: int, step=-1) -> dict:
        """forces along one dowel per element: s (distance of element midpoints from its first point), N (axial,
        tension +), V (resultant shear), M (largest resultant bending moment at the element ends), local (the 12 local end
        forces of OpenSees)"""
        h = self._rec(step)
        d = self.dowels[dowel]
        segs = d["segs"]
        P = self.points
        mids = np.array([0.5 * (P[self._dsegs[k][0]] + P[self._dsegs[k][1]]) for k in segs])
        s = np.array([self._arc(d, m) for m in mids])
        return dict(s=s, x=mids, N=h.dowel["N"][segs], V=h.dowel["V"][segs], M=h.dowel["M"][segs],
                    local=h.dowel["local"][segs])

    @staticmethod
    def _arc(b, x):
        P, s = b["points"], b["s"]
        best = (np.inf, 0.0)
        for i in range(len(P) - 1):
            d = P[i + 1] - P[i]
            t = np.clip(np.dot(x - P[i], d) / max(np.dot(d, d), 1e-300), 0, 1)
            r = np.linalg.norm(P[i] + t * d - x)
            if r < best[0]:
                best = (r, s[i] + t * (s[i + 1] - s[i]))
        return best[1]

    def cracks_along(self, *path, step=-1, ds: float | None = None, min_width: float = 0.0, split: float = 0.5) -> dict:
        """cracks crossed by a line (two points) or polyline. The line is sampled every ds (default an eighth of the
        smallest band); each sample takes the crack strain of its cell projected on the line, ε_cr (n·t)², so that the
        integral over a crack band is the opening of that crack along the line. Consecutive cells along the line form the
        profile; it is cut into individual cracks at cells without cracks and at valleys lower than `split` × the smaller
        of the two neighbouring peaks (a valley cell is shared half and half). Returns dict(s, x (samples), opening
        (projected crack strain per sample), cells (cell per sample, −1 outside), profile (per crossed cell: s, length,
        value), cracks=[dict(s, x, width, cells)] in order along the line (cracks wider than min_width), total (the
        whole opening along the line), max_width)."""
        h = self._rec(step)
        P = np.asarray(path[0] if len(path) == 1 else path, float).reshape(-1, 3)
        ds = ds or float(self.h_used.min()) / 8
        ss, pts, ts, cells, edges = self._line_geometry(P, ds)
        ecr, nrm = h.cell["crack_strain"], h.cell["crack_normal"]
        ok = cells >= 0
        val = np.zeros(len(pts))
        val[ok] = ecr[cells[ok]] * np.einsum("ij,ij->i", nrm[cells[ok]], ts[ok]) ** 2
        brk = np.flatnonzero(np.diff(cells) != 0) + 1
        groups = np.split(np.arange(len(pts)), brk)
        pc = np.array([cells[g[0]] for g in groups])
        pl = np.diff(np.array(edges))
        pv = np.array([float(np.mean(val[g])) for g in groups])
        psm = np.array([float(np.mean(ss[g])) for g in groups])
        pw = pv * pl  # opening carried by each crossed cell
        cracks = []
        on = np.r_[False, pv > 0, False]
        for a, b in zip(np.flatnonzero(on[1:] & ~on[:-1]), np.flatnonzero(~on[1:] & on[:-1])):
            v = pv[a:b]
            # peaks, then merge neighbours whose valley is not deep enough
            pk = [i for i in range(len(v)) if (i == 0 or v[i] >= v[i - 1]) and (i == len(v) - 1 or v[i] > v[i + 1])]
            cuts = []  # valley index between consecutive kept peaks
            kept = [pk[0]]
            for q in pk[1:]:
                p_ = kept[-1]
                vi = p_ + int(np.argmin(v[p_:q + 1]))
                if v[vi] < split * min(v[p_], v[q]):
                    cuts.append(vi)
                    kept.append(q)
                elif v[q] > v[p_]:
                    kept[-1] = q
            bounds = [0] + cuts + [len(v) - 1]
            for j, pkj in enumerate(kept):
                lo, hi = bounds[j], bounds[j + 1]
                wts = np.ones(hi - lo + 1)
                if j > 0:
                    wts[0] = 0.5
                if j < len(kept) - 1:
                    wts[-1] = 0.5
                w = float(np.sum(pw[a + lo:a + hi + 1] * wts))
                if w > min_width:
                    i = a + pkj
                    cracks.append(dict(s=float(psm[i]), x=pts[groups[i][len(groups[i]) // 2]], width=w,
                                       cells=np.unique(pc[a + lo:a + hi + 1])))
        return dict(s=ss, x=pts, opening=val, cells=cells, profile=dict(s=psm, length=pl, value=pv, cell=pc),
                    cracks=cracks, total=float(np.sum(pw)), max_width=max((c["width"] for c in cracks), default=0.0))

    def _line_geometry(self, P: np.ndarray, ds: float):
        """samples of a polyline (arc length, points, unit tangents), the cell of each sample, and the cell boundaries
        along it (found by bisection, so that every crossed cell counts with its exact length); kept per line, since they
        do not depend on the step"""
        key = (P.round(12).tobytes(), round(ds, 12))
        cache = self.__dict__.setdefault("_line_cache", {})
        if key in cache:
            return cache[key]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        total = float(cum[-1])

        def at(sv):
            i = min(int(np.searchsorted(cum, sv, side="right")) - 1, len(seg) - 1)
            t = (P[i + 1] - P[i]) / seg[i]
            return P[i] + t * (sv - cum[i]), t

        def cell_at(sv):
            f = self.loc.find(at(sv)[0])
            return f[0] if f is not None else -1

        n = max(1, int(math.ceil(total / ds)))
        ss = (np.arange(n) + 0.5) * total / n
        pts = np.array([at(v)[0] for v in ss])
        ts = np.array([at(v)[1] for v in ss])
        cells = np.array([cell_at(v) for v in ss])
        edges = [0.0]
        for i in np.flatnonzero(np.diff(cells) != 0) + 1:
            lo, hi, c0 = ss[i - 1], ss[i], cells[i - 1]
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if cell_at(mid) == c0 else (lo, mid)
            edges.append(0.5 * (lo + hi))
        edges.append(total)
        cache[key] = (ss, pts, ts, cells, edges)
        return cache[key]

    def surface_cracks(self, where=None, step=-1) -> dict:
        """crack widths on the boundary: for the faces selected by a predicate on face centroids (default all) the
        centroid, outward normal, owning cell, its crack width and crack normal"""
        h = self._rec(step)
        f, owner = self.faces()
        sel = np.arange(len(f)) if where is None else _sel(where, self.face_centroids())
        X = self.mesh.points[f[sel]]
        nv = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
        nv /= np.linalg.norm(nv, axis=1, keepdims=True)
        return dict(faces=sel, x=X.mean(axis=1), normal=nv, cell=owner[sel], width=h.cell["width"][owner[sel]],
                    crack_normal=h.cell["crack_normal"][owner[sel]])

    # ── export ───────────────────────────────────────────────────────────
    def result_mesh(self) -> Mesh:
        """solid cells and bar segments (line2) on all model nodes; cell fields of to_result are in this order"""
        cells = {k: c for k, c in self.cell_conn()}
        lines = [(i, j) for i, j, _ in self._segs] + [(i, j) for i, j, _ in self._dsegs]
        if lines:
            cells["line2"] = np.array(lines, np.int64)
        cg = {g: dict(d) for g, d in self.mesh.cell_groups.items()}
        for b in self.bars:
            if len(b["segs"]):
                cg[b["name"]] = {"line2": b["segs"]}
        for d in self.dowels:
            cg[d["name"]] = {"line2": len(self._segs) + d["segs"]}
        return Mesh(self.points, cells, dict(self.mesh.point_groups), cg)

    def to_result(self, steps=None, meta: dict | None = None):
        """fem-result/1 Result of the recorded steps (or the given history indices), one frame each: node field u; cell
        fields crack_width, crack_strain, crack_normal (3), cracked (0/1), stress (6, xx yy zz xy yz zx), s1 (largest
        principal stress) on solid cells, bar_stress and bar_force on bar segments, dowel_N, dowel_V, dowel_M on dowel
        elements (0 on the other kinds; line2 cells are the bar segments, then the dowel elements). meta carries t
        (frame number), lam and stage per frame"""
        from ..result import Result

        idx = self.recorded() if steps is None else list(steps)
        hs = [self._rec(i) for i in idx]
        nb, nd = len(self._segs), len(self._dsegs)
        nseg = nb + nd
        r = Result(self.result_mesh(), dict(meta or {}, t=list(range(len(idx))), lam=[h.lam for h in hs],
                                            stage=[h.stage for h in hs], format_note="femotion.continuum.opensees"))
        r.field("u", np.array([h.U for h in hs]), frames=True, label="displacement")

        def cellf(key, f=lambda v: v):
            out = []
            for h in hs:
                v = np.asarray(f(h.cell[key]), float)
                pad = np.zeros((nseg,) + v.shape[1:])
                out.append(np.concatenate([v, pad]))
            return np.array(out)

        def barf(key):
            return np.array([np.concatenate([np.zeros(self.nc), h.bar[key], np.zeros(nd)]) for h in hs])

        def dowf(key):
            return np.array([np.concatenate([np.zeros(self.nc + nb), h.dowel[key]]) for h in hs])

        r.field("crack_width", cellf("width"), loc="cell", frames=True, label="crack width (ε_cr·h)")
        r.field("crack_strain", cellf("crack_strain"), loc="cell", frames=True, label="crack opening strain")
        r.field("crack_normal", cellf("crack_normal"), loc="cell", frames=True, label="crack normal")
        r.field("cracked", cellf("cracked"), loc="cell", frames=True, label="cracked", dtype="u8")
        r.field("stress", cellf("stress"), loc="cell", frames=True, label="stress xx yy zz xy yz zx")
        r.field("s1", cellf("stress", lambda S: principal_3d(S)[0][:, 0]), loc="cell", frames=True,
                label="largest principal stress")
        if any(np.any(h.cell["eps0"]) for h in hs):
            r.field("eps0", cellf("eps0"), loc="cell", frames=True, label="free strain xx yy zz xy yz zx")
        if nb:
            r.field("bar_stress", barf("stress"), loc="cell", frames=True, label="bar stress")
            r.field("bar_force", barf("N"), loc="cell", frames=True, label="bar axial force")
        if nd:
            r.field("dowel_N", dowf("N"), loc="cell", frames=True, label="dowel axial force")
            r.field("dowel_V", dowf("V"), loc="cell", frames=True, label="dowel shear")
            r.field("dowel_M", dowf("M"), loc="cell", frames=True, label="dowel bending moment")
        return r

    def export(self, path, steps=None) -> list:
        """ParaView files of the recorded steps: path ending in .pvd → a VTU per step plus the collection; any other
        meshio format → the last recorded step only"""
        from pathlib import Path

        from ..io import export_result

        r = self.to_result(steps)
        if Path(path).suffix.lower() == ".pvd":
            return export_result(r, path)
        return export_result(r, path, frame=len(r.meta["t"]) - 1)

    def save_npz(self, path):
        """all steps as numpy arrays: lam, stage, U (steps, nodes, 3), R, and per recorded step the cell and bar fields
        (keys cell/<name>, bar/<name>, rec = their history indices)"""
        rec = self.recorded()
        d = dict(lam=np.array([h.lam for h in self.history]), stage=np.array([h.stage for h in self.history]),
                 U=np.array([h.U for h in self.history]), R=np.array([h.R for h in self.history]), rec=np.array(rec),
                 points=self.points, supported=self.supported, h=self.h_used, mat_of=self.mat_of)
        for key in self.history[rec[0]].cell:
            d[f"cell/{key}"] = np.array([self.history[i].cell[key] for i in rec])
        for key in self.history[rec[0]].bar:
            d[f"bar/{key}"] = np.array([self.history[i].bar[key] for i in rec])
        for key in self.history[rec[0]].dowel:
            d[f"dowel/{key}"] = np.array([self.history[i].dowel[key] for i in rec])
        np.savez_compressed(path, **d)
        return path

    @staticmethod
    def _compliance(m, s: np.ndarray) -> np.ndarray:
        """elastic strain C₀⁻¹ σ (engineering shear) of stresses s (n, 6)"""
        if isinstance(m, ElasticOrtho):
            Ex, Ey, Ez = m.E
            nxy, nyz, nzx = m.nu
            Gxy, Gyz, Gzx = m.G
            C = np.zeros((6, 6))
            C[:3, :3] = [[1 / Ex, -nxy / Ex, -nzx / Ez], [-nxy / Ex, 1 / Ey, -nyz / Ey], [-nzx / Ez, -nyz / Ey, 1 / Ez]]
            C[3, 3], C[4, 4], C[5, 5] = 1 / Gxy, 1 / Gyz, 1 / Gzx
            return s @ C.T
        E, nu = m.E, m.nu
        out = np.empty_like(s)
        tr = s[:, :3].sum(axis=1)
        out[:, :3] = ((1 + nu) * s[:, :3] - nu * tr[:, None]) / E
        out[:, 3:] = s[:, 3:] * 2 * (1 + nu) / E
        return out


__all__ = ["OpsSolid", "ConcreteDamage", "ElasticIso", "ElasticOrtho", "CellLocator", "interpolate", "band_widths",
           "cell_aspect", "displacement_source", "submodel_boundary", "steel", "elastic", "elastic_pp", "frp", "Step",
           "Stage", "Eigenstrain", "UNITS", "HEX_TETS", "VOIGT"]
