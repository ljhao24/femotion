"""Corrosion of reinforcement in solid models: the free strain of a rust layer around a bar, conversions between
corrosion depth, mass loss and the free radial expansion of the rust, and the classical references for cover cracking
(thick-walled cylinder, ring models, empirical critical corrosion depths) used to check a model.

The rust layer is a thin ring of cells at the bar's surface (femotion.mesh.inclusions: ring=t); its free strain
(rust_strain) is the strain of the displacement field u = δ (ρ − r_in)/t in the ring — radial δ/t, hoop
δ (ρ − r_in)/(t ρ) — so the ring's outer surface moves out by δ when nothing resists it and the rust's own stiffness (the
ring's material) only matters under the pressure of the concrete. A stage of OpsSolid ramps δ from 0 to its value:

    from femotion.continuum.corrosion import rust_strain, depth_for_expansion
    s.static(steps=200, strain=s.initial_strain(rust_strain((0, y0, z0), (1, 0, 0), r - t, t, delta), "rust0"))

Units: lengths in any unit (consistent), stresses likewise; the empirical formulas state their own units.
"""
from __future__ import annotations

import math

import numpy as np


# ── geometry of the rust layer ─────────────────────────────────────────────
def mass_loss(x: float, d: float) -> float:
    """relative steel mass (= cross-section) loss of a bar of diameter d corroded uniformly to the depth x"""
    return 1.0 - (1.0 - 2.0 * x / d) ** 2


def depth_from_mass_loss(rho: float, d: float) -> float:
    """uniform corrosion depth of a bar of diameter d that lost the share rho of its mass"""
    return d / 2 * (1.0 - math.sqrt(1.0 - rho))


def free_expansion(x: float, d: float, n: float = 3.0, porous: float = 0.0) -> float:
    """free radial expansion δ of the surface of a bar of diameter d corroded to the depth x, when the rust occupies n
    times the volume of the steel it replaces (rust volume ratio: Fe3O4 2.08, α-FeOOH 2.91, Fe(OH)2 3.75, Fe(OH)3 4.2,
    Fe(OH)3·3H2O 6.4 after Caré et al. 2008; 2–4 for the mixtures found in concrete), less the thickness `porous` of the porous zone around the
    bar that the rust fills before it presses on the concrete (Liu & Weyers 1998: 12.5 μm). Exact for the ring
    (π(R² − r_i²) = n π(r² − r_i²)); for x ≪ d it is (n − 1) x − porous"""
    r = d / 2
    ri = r - x
    R = math.sqrt(ri * ri + n * (r * r - ri * ri))
    return max(R - r - porous, 0.0)


def depth_for_expansion(delta: float, d: float, n: float = 3.0, porous: float = 0.0) -> float:
    """the corrosion depth x whose rust expands the surface of a bar of diameter d by delta (free_expansion inverted)"""
    if n <= 1:
        raise ValueError("rust volume ratio n must exceed 1")
    r = d / 2
    R = r + delta + porous
    ri2 = (n * r * r - R * R) / (n - 1)
    if ri2 < 0:
        raise ValueError("expansion larger than a fully corroded bar gives")
    return r - math.sqrt(ri2)


def _axis_frame(p0, axis):
    p0 = np.asarray(p0, float)
    a = np.asarray(axis, float)
    return p0, a / np.linalg.norm(a)


def _polar(x, p0, a):
    """unit radial vector n (3, …), radius ρ (…) of points x (3, …) about the axis through p0 along a"""
    d = x - p0.reshape((3,) + (1,) * (x.ndim - 1))
    d = d - np.einsum("i,i...->...", a, d)[None] * a.reshape((3,) + (1,) * (x.ndim - 1))
    rho = np.linalg.norm(d, axis=0)
    n = d / np.where(rho > 0, rho, 1.0)[None]
    return n, rho


def cylindrical_strain(p0, axis, err, ett, ezz=0.0):
    """a free strain given in cylindrical components about the axis through p0 along `axis`: radial err, hoop ett,
    axial ezz (numbers or functions of the radius ρ). Returns eps0(x) for OpsSolid.initial_strain and
    femotion.continuum LoadCase.initial_strain: x (3, …) → Voigt (6, …), engineering shear"""
    p0, a = _axis_frame(p0, axis)

    def f(x):
        x = np.asarray(x, float)
        n, rho = _polar(x, p0, a)
        sh = (3,) + (1,) * (x.ndim - 1)
        t = np.cross(a.reshape(sh), n, axisa=0, axisb=0, axisc=0)
        er = np.asarray(err(rho) if callable(err) else err, float) * np.ones_like(rho)
        et = np.asarray(ett(rho) if callable(ett) else ett, float) * np.ones_like(rho)
        ez = np.asarray(ezz(rho) if callable(ezz) else ezz, float) * np.ones_like(rho)
        A = a.reshape(sh) * np.ones_like(rho)[None]
        T = er[None, None] * n[:, None] * n[None] + et[None, None] * t[:, None] * t[None] + ez[None, None] * A[:, None] * A[None]
        return np.stack([T[0, 0], T[1, 1], T[2, 2], 2 * T[0, 1], 2 * T[1, 2], 2 * T[2, 0]])

    return f


def rust_strain(p0, axis, r_in: float, t: float, delta: float):
    """free strain of a rust ring between the radii r_in and r_in + t about the bar axis (through p0 along `axis`) that
    moves its outer surface out by delta when unrestrained: the strain of u = δ (ρ − r_in)/t (radial δ/t, hoop
    δ (ρ − r_in)/(t ρ), axial 0). Being compatible, it leaves a free ring without stress; the core inside does not move"""
    return cylindrical_strain(p0, axis, delta / t,
                              lambda rho: delta * (np.clip(rho, r_in, r_in + t) - r_in) / (t * np.maximum(rho, 1e-300)))


# ── references for cover cracking ──────────────────────────────────────────
def lame_cylinder(a: float, b: float, E: float, nu: float, plane: str = "strain") -> dict:
    """thick-walled cylinder (inner radius a, outer b, free outside) under an inner pressure p, linear elastic (Lamé):
    compliance u(a)/p, hoop stress at the inner surface σθ(a)/p = (a² + b²)/(b² − a²), and p1 = ft (b² − a²)/(a² + b²),
    the pressure at which σθ(a) reaches ft (first cracking, uncracked elastic model of Tepfers 1979) — pass ft via
    first_crack(ft). plane: "strain" (long member) or "stress" (thin slice)"""
    if plane == "strain":
        c = (1 + nu) * a / (E * (b * b - a * a)) * ((1 - 2 * nu) * a * a + b * b)
    elif plane == "stress":
        c = a / (E * (b * b - a * a)) * ((1 - nu) * a * a + (1 + nu) * b * b)
    else:
        raise ValueError("plane: 'strain' or 'stress'")
    hoop = (a * a + b * b) / (b * b - a * a)
    return dict(compliance=c, hoop=hoop, p1=lambda ft: ft / hoop, delta1=lambda ft: ft / hoop * c)


def tepfers(c: float, d: float, ft: float) -> dict:
    """ring models of the cover around a bar of diameter d with the cover c (Tepfers 1979, Magazine of Concrete Research
    31(106):3–12): the radial pressure on the bar surface that splits the cover, for an uncracked elastic ring (the
    hoop stress at the bar reaches ft: first crack), a partly cracked elastic ring (the peak of a brittle ring, crack
    front at 0.486 (c + d/2)) and a plastic ring (hoop stress ft through the whole cover: upper bound)"""
    r, R = d / 2, c + d / 2
    return dict(uncracked=ft * (R * R - r * r) / (R * R + r * r), partly_cracked=ft * (c + d / 2) / (1.664 * d),
                plastic=2 * c * ft / d)


def alonso_1998(c: float, d: float) -> float:
    """corrosion depth (radius loss, mm) at the first visible crack of the cover c over a bar of diameter d, from
    accelerated tests (impressed current 10–100 μA/cm², c/d ≈ 1.25–4.4): x0 = 7.53 + 9.32 c/d μm (Alonso, Andrade,
    Rodriguez & Diez 1998, Materials and Structures 31:435–441; the first visible cracks of 0.05–0.1 mm came at 15–50 μm
    in the same group's tests)"""
    return (7.53 + 9.32 * c / d) * 1e-3


def liu_weyers_1998(c: float, d: float, ft: float, Ec: float, nu: float = 0.18, d0: float = 0.0125, creep: float = 0.0,
                    alpha: float = 0.57, rust_density: float = 3600 / 7850) -> dict:
    """critical rust of Liu & Weyers (1998, ACI Materials Journal 95(6):675–681; Liu 1996, PhD thesis, Virginia Tech,
    5.6): the rust first fills a porous zone d0 around the bar (12.5 μm), then presses on a thick-walled elastic
    cylinder (inner radius a = d/2 + d0, outer b = a + c, modulus Ec / (1 + creep)) until the pressure reaches the
    plastic ring's 2 c ft / (d + 2 d0). Lengths in mm. Returns the radial expansion ds beyond the porous zone, the steel
    mass loss and the corrosion depth; alpha: mass of the steel per mass of its rust (0.523 Fe(OH)3, 0.622 Fe(OH)2),
    rust_density: of the rust relative to steel"""
    a = d / 2 + d0
    b = a + c
    E = Ec / (1 + creep)
    ds = c * ft / E * ((a * a + b * b) / (b * b - a * a) + nu)
    loss = 4 * alpha * rust_density * (ds + d0) / (d * (1 - alpha * rust_density))  # W_st / (ρ_st π d²/4)
    return dict(expansion=ds, mass_loss=loss, depth=depth_from_mass_loss(loss, d), pressure=2 * c * ft / (d + 2 * d0))


def el_maaddawy_soudki_2007(c: float, d: float, fct: float, Ec: float, nu: float = 0.18, delta0: float = 0.010,
                            creep: float = 2.35) -> dict:
    """critical steel mass loss of El Maaddawy & Soudki (2007, Cement & Concrete Composites 29:168–175, eq. 12–19): a
    thick-walled cylinder (wall = the cover c, modulus Ec / (1 + creep), creep coefficient 2.35 in the paper) around a
    porous zone delta0 (10–20 μm), rust of half the steel's density (volume ratio ≈ 3.2), cracking at the plastic
    pressure 2 c fct / d. Lengths in mm. Returns the mass loss (fraction), depth and pressure"""
    E = Ec / (1 + creep)
    Dp = d + 2 * delta0
    psi = Dp * Dp / (2 * c * (c + Dp))
    m = 0.909 * (2 * c * fct * (1 + nu + psi) * Dp / (E * d * d) + 2 * delta0 / d)
    return dict(mass_loss=m, depth=depth_from_mass_loss(m, d), pressure=2 * c * fct / d)


def gbt51355_2019(c: float, d: float, fcu: float, corner: bool = False) -> float:
    """corrosion depth (mm) at which the cover cracks from corrosion, GB/T 51355-2019 (assessment of the durability of
    existing concrete structures), appendix C.0.4: corner bars of beams and columns 0.012 c/d + 0.00084 fcu + 0.018;
    bars of walls and slabs away from corners 0.015 (c/d)^1.55 + 0.0014 fcu + 0.016 (c, d in mm, fcu the in-situ cube
    strength in MPa); natural corrosion, cracks of about 0.1 mm according to its drafters"""
    if corner:
        return 0.012 * c / d + 0.00084 * fcu + 0.018
    return 0.015 * (c / d) ** 1.55 + 0.0014 * fcu + 0.016


__all__ = ["mass_loss", "depth_from_mass_loss", "free_expansion", "depth_for_expansion", "cylindrical_strain",
           "rust_strain", "lame_cylinder", "tepfers", "alonso_1998", "liu_weyers_1998", "el_maaddawy_soudki_2007",
           "gbt51355_2019"]
