"""OpenSees back-end for solids (OpenSeesPy ≥ 3.8; extra "opensees"): nonlinear analysis of concrete with smeared cracks —
where cracks appear, how they grow and how wide they open — with embedded reinforcement, prestressing tendons, anchor bolts
and bonded or anchored strips, staged loading and crack-width read-outs. Guide and validation: docs/continuum-nonlinear.md.

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

Units: any consistent set (N, mm, MPa or kN, m, kPa …). Only the defaults of ConcreteDamage.from_strength, which come
from empirical formulas in MPa, need to know it (units="N-mm" | "kN-m" | "N-m" | "MN-m").

Model
- solids: hex8 → stdBrick (8-point trilinear, or bbarBrick), tet4 → FourNodeTetrahedron; materials by cell group
  (ConcreteDamage, ElasticIso, ElasticOrtho).
- ConcreteDamage → nDMaterial ASDConcrete3D (Petracca, Camata, ASDEA): damage-plasticity with a Lubliner-type surface and
  separate tension and compression laws; the post-peak branches are regularised by the crack band of each cell (below).
- bars (rebars, tendons, strips): Truss elements along any polyline, not tied to the mesh. Bonded bars are cut into short
  segments whose nodes are each tied to the solid by ASDEmbeddedNodeElement (penalty, linear tetrahedral interpolation:
  inside a hexahedron the node is tied to one of its six sub-tetrahedra); unbonded bars get nodes only at their given
  points (anchors, deviators) and one Truss per span. A bar node that coincides with a mesh node uses it directly.
  prestress=σ0 wraps the material in InitStressMaterial (tension +; the first analysis step balances it).
- dowels (anchor bolts, studs): beam elements on 6-dof nodes tied to the solid in translation (or also in rotation), so
  shear and bending reach the concrete along the embedded length; bars ending on a dowel node are pinned to it (a
  prestressed strip anchored on bolt heads).
- supports: fixed components, ties (equalDOF), nodal springs, distributed (Winkler) springs on boundary faces; imposed
  displacements through stiff springs from driven ground nodes (sub-models, stroke control).
- loads (consistent nodal forces): nodal, point (anywhere, by shape functions), traction / pressure on boundary faces,
  body force and self-weight.
- stages: static (load control) and push (displacement control) accumulate in call order, each held afterwards. The
  concrete integrates with IMPL-EX by default (two Newton iterations per step, never stalls; error of about one step);
  implex=False iterates implicitly. Steps that do not converge switch algorithm and then halve.

Crack band (Bažant & Oh 1983) and crack width
- Each concrete cell gets its own softening law: the tensile law is given as stress against crack opening w (fib MC2010
  bilinear by default, area Gf) and converted to strain with the band width h of the cell, ε = σ/E + ε_pre + w/h, so the
  energy dissipated per unit crack area is Gf whatever the mesh size; the compressive branch likewise with Gc. h is ∛V for
  hexahedra and ∛(6V) for tetrahedra (the side of the cube the tetrahedron would be a sixth of), or the extent of the cell
  along a given direction (band=("projected", n), Oliver 1989), or any array. Cells are grouped in 1 % bins of h, one
  OpenSees material per bin. A cell larger than h_max = the snap-back limit E·Δw/Δσ of the steepest softening segment is
  given the law of h_max (it dissipates Gf·h/h_max; a warning says how many).
- The crack strain is read from stresses and strains, not from internal variables: the inelastic strain tensor
  ε_in = ε − C₀⁻¹ σ at the integration points, averaged over the cell; its largest principal value minus the pre-peak part
  is the crack opening strain ε_cr, its direction the crack normal n; a cell counts as cracked once the tensile
  equivalent strain of the material passed the peak. The crack width is w = ε_cr · h (the opening of the band the law was
  regularised for). Along a line the profile of ε_cr (n·t)² is integrated cell by cell and cut into individual cracks at
  its valleys; the width of each is the opening of that crack projected on the line.
- Limits of the estimate: smeared cracks do not localise into single lines, so one physical crack is a band of one or
  two cells, and neighbouring cracks closer than ~2 h merge; the opening is the mean over the band, not the width at the
  surface; the steel is perfectly bonded (no slip, no tension-stiffening law), so crack spacing follows the mesh and the
  bar diameter does not enter; widths are order-of-magnitude checks against code formulas, not predictions to 0.01 mm.
- With ν > 0 a softening band restrained laterally by its elastic neighbours dissipates a few per cent less than Gf
  (ASDConcrete3D's equivalent strain comes from the effective stress): 0.93 Gf for ν = 0.2, Gf exactly for ν = 0.

Pitfalls found while writing this module (OpenSeesPy 3.8)
- ASDEmbeddedNodeElement takes the fourth retained node as an optional argument parsed from a *string*: passed as a
  Python int it is silently dropped and the node is tied to the triangle of the first three nodes. Pass str(node).
- Its default penalty 1e18·∛V leaves round-off of ~1e-5 in the displacements and stalls convergence tests; here the
  penalty is 1e6 × the largest Young's modulus (relative error ~1e-8).
- FourNodeTetrahedron with negative volume gives a negative stiffness without a warning; cells are re-oriented here.
- IMPL-EX extrapolates with OpenSees' pseudo-time increment, which displacement control changes in every iteration (the
  step then diverges): the material time is set per step here (setParameter dTime / dTimeCommit).
- An sp constraint on a node puts a whole displacement increment into the first cell layer in the predictor and cracks
  it; imposed displacements go through stiff springs instead.
- ASDConcrete3D's own "-fc" shortcut assumes MPa and N/mm, and its "-autoRegularization" and crack-width responses use
  the shortest node-to-node distance as the band; none is used here.
- geomTransf and frame elements need the 6-dof model builder when created; loads on 6-dof nodes take six components.

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


def _ops():
    from ..frame.opensees import _ops as ops  # one lazy import of OpenSeesPy for the whole package

    return ops()


def _registry() -> list:
    from ..frame.opensees import _ACTIVE  # shared with OpsFrame: OpenSees has a single global domain

    return _ACTIVE


# ── materials ────────────────────────────────────────────────────────────
@dataclass
class ElasticIso:
    """isotropic linear elasticity (nDMaterial ElasticIsotropic)"""
    E: float
    nu: float
    rho: float = 0.0


@dataclass
class ElasticOrtho:
    """orthotropic linear elasticity in the global axes, with the parameters of OpenSees' nDMaterial ElasticOrthotropic:
    E = (Ex, Ey, Ez), nu = (νxy, νyz, νzx), G = (Gxy, Gyz, Gzx), where νij is the contraction along j under stress along i
    (so νzx is the contraction along x under stress along z, = ν13·E3/E1 in the notation of
    femotion.continuum.Orthotropic, whose nu = (ν12, ν13, ν23) and G = (G12, G13, G23)). OpsSolid also takes
    femotion.continuum.Isotropic and Orthotropic (material axes along the global ones) and converts them"""
    E: tuple
    nu: tuple
    G: tuple
    rho: float = 0.0


def _native(m):
    """materials of femotion.continuum (Isotropic, Orthotropic with axes along the global ones) as OpenSees ones"""
    if isinstance(m, Isotropic):
        return ElasticIso(m.E, m.nu)
    if isinstance(m, Orthotropic):
        if not np.allclose(np.abs(m.axes), np.eye(3), atol=1e-9):
            raise ValueError("Orthotropic with rotated axes: OpenSees' ElasticOrthotropic works in the global axes only")
        E1, E2, E3 = m.E
        n12, n13, n23 = m.nu
        G12, G13, G23 = m.G
        return ElasticOrtho(E=(E1, E2, E3), nu=(n12, n23, n13 * E3 / E1), G=(G12, G23, G13))
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
    (multi-directional) cracking; Kc, cdf, eta: ASDConcrete3D parameters (triaxial shape, dilatancy, viscosity)."""
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
        an area Gc, constant beyond. Extra keywords go to the dataclass (plastic_t, implex, crack_planes, rho, …)."""
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


def frp(E: float, eps_u: float) -> tuple:
    """fibre-reinforced polymer strip or bar: linear elastic, ruptures (stress → 0 for good) beyond eps_u (tension or
    compression)"""
    return ("FRP", float(E), float(eps_u))


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


@dataclass
class Stage:
    name: str
    kind: str  # "static", "push", "hold" (no load increment)
    steps: list = field(default_factory=list)  # indices into OpsSolid.history
    converged: bool = True
    message: str = ""
    change: float = 0.0  # hold stages: displacement change of the last step / of the whole stage


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
            seg: float | None = None, name: str | None = None, outside: str = "error", slide: bool = True) -> int:
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
        outside the solid (they must then be held by other members, springs or supports)."""
        if self.built:
            raise RuntimeError("the OpenSees model is already built: add bars before the first analysis")
        P0 = np.asarray(points, float).reshape(-1, 3)
        if len(P0) < 2:
            raise ValueError("a bar needs at least two points")
        if bond:
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
                              prestress=float(prestress), bond=bond, segs=np.array(segs, np.int64)))
        return b

    def dowel(self, points, *, d: float | None = None, E: float | None = None, nu: float = 0.3, section=None,
              fiber=None, seg: float | None = None, rot: bool = False, name: str | None = None) -> int:
        """an anchor bolt, dowel or stud as beam elements along the polyline `points`: it carries axial force, shear and
        bending, so a force on its head reaches the concrete along the embedded length (dowel action), not at one point.
        Section: a solid circle of diameter d and Young's modulus E (shear modulus E / 2(1 + ν)), or section=(E, A, Iy,
        Iz, G, J); fiber=femotion.frame.opensees.Fiber for a yielding anchor (forceBeamColumn, 5 Lobatto points).
        Cut into elements of at most `seg` (default half the median crack band); every node inside the solid is tied to
        it in translation (the dowel bears on the concrete and bends on its own); rot=True also ties its rotation to the
        rotation of the solid (a fully bonded stiff insert). Nodes outside the solid (a head standing proud of the
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
        used |= {a for _, a, _ in self._sliders}
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
                                 self.supported, *self._read_fields()))

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

    def _run(self, stage: Stage, steps: int, set_inc, analyze, record, stop=None, lam_of=None, hold: bool = False):
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

    def static(self, F=None, steps: int = 1, *, imposed=None, name: str | None = None, record=True,
               stop=None) -> Stage:
        """load control: load pattern F (n_nodes, 3) (or None: prestress and initial stresses only) and imposed
        displacements imposed=(node ids, values (n, 3), NaN = free) applied in `steps` equal increments, then held for
        the following stages (unload with −F). record: True (cell and bar results every step), an integer k (every k-th
        step and the last) or False (the last step only); displacements and reactions are kept for every step.
        stop(step) → True ends the stage."""
        if not self.built:
            self._build()
        self._live()
        ops = self.ops
        hold = imposed is None and (F is None or not np.any(F))
        drv = self._drivers(imposed) if imposed is not None else None
        self._check_supports()
        p = self._pattern(F, drv)
        self._settle_prestress()  # at λ = 0 of this stage's pattern: imposed displacements held at zero meanwhile
        self._setup()
        ops.integrator("LoadControl", 1.0 / steps)
        ops.analysis("Static")
        st = Stage(name or f"stage{len(self.stages) + 1}", "hold" if hold else "static")
        self._run(st, steps, lambda s: ops.integrator("LoadControl", s / steps), lambda: ops.analyze(1), record, stop,
                  lambda: ops.getLoadFactor(p), hold=hold)
        ops.loadConst("-time", 0.0)
        return st

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
        return Step(stage, k, float(lam), U, R, sup, cell, bar, dowel)

    def _read_fields(self) -> tuple[dict, dict]:
        """cell averages over the integration points: stress, strain (engineering shear, order xx yy zz xy yz zx),
        inelastic strain, crack strain and normal, crack width, cracked / crushed flags; bar axial force and strain"""
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
        cell = dict(stress=S, strain=Eps, eps_in=Ein, crack_strain=ecr, crack_normal=n1 * (ecr > 0)[:, None],
                    width=ecr * self.h_used, cracked=cracked, crushed=crushed, eq_strain=np.c_[xt, xc])
        nseg = len(self._segs)
        N = np.zeros(nseg)
        dL = np.zeros(nseg)
        for s in range(nseg):
            N[s] = ops.eleResponse(int(self.seg_tag[s]), "axialForce")[0]
            dL[s] = ops.eleResponse(int(self.seg_tag[s]), "basicDeformation")[0]
        L0, A = self._seg_geometry()
        bar = dict(N=N, stress=N / A, strain=dL / L0)
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
        stress, strain (elastic strain of the element; the prestrain of a prestressed bar comes on top)"""
        h = self._rec(step)
        b = self.bars[bar]
        segs = b["segs"]
        P = self.points
        mids = np.array([0.5 * (P[self._segs[k][0]] + P[self._segs[k][1]]) for k in segs])
        s = np.linalg.norm(mids - b["points"][0], axis=1) if len(segs) else np.zeros(0)
        if len(segs) and len(b["points"]) > 2:  # along a polyline: project on the arc length
            s = np.array([self._arc(b, m) for m in mids])
        return dict(s=s, x=mids, N=h.bar["N"][segs], stress=h.bar["stress"][segs], strain=h.bar["strain"][segs])

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
           "Stage", "UNITS", "HEX_TETS"]
