"""Structural analysis rules of JTG 3362-2018 4.3 for frame models (femotion.frame): the flexural stiffness of members
in statically indeterminate structures (4.3.1), when a varying second moment must be modelled (4.3.6), and the
reduction of the negative moment over an interior support for the width of the bearing (4.3.5, "peak shaving").

Units: sections in mm (RCSection), frame models in m, kN, kPa; moments kN·m, reactions kN. Clauses and the program
conventions: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    s = jtg.frame_section(span_sec, member="rc", Iz=…, J=…, mass=…)        # 0.8 Ec I for members allowed to crack
    a = jtg.spread_length(0.45, 0.62)                                      # bearing 450 mm, soffit to centroid 620 mm
    Mr = jtg.peak_shaving(x, Mmin, x_support=16.0, R=1450.0, a=a)          # the hogging peak over the pier rounded
"""
from __future__ import annotations

import numpy as np

from .rc import RCSection

# 4.3.1: flexural stiffness factor on Ec I (I the gross second moment)
STIFFNESS_FACTOR = {"rc": 0.8, "B": 0.8, "A": 1.0, "full": 1.0}


def stiffness_factor(member: str) -> float:
    """factor on Ec I in the analysis of statically indeterminate structures (4.3.1): 0.8 for members allowed to crack
    (member "rc": reinforced concrete; "B": class B prestressed), 1.0 for members not allowed to crack ("full", "A":
    fully and class A prestressed); I the gross second moment"""
    key = member if member in STIFFNESS_FACTOR else member.upper() if member.upper() in STIFFNESS_FACTOR else None
    if key is None:
        raise ValueError(f"member: one of {', '.join(STIFFNESS_FACTOR)}")
    return STIFFNESS_FACTOR[key]


def frame_section(sec: RCSection, *, member: str = "rc", Iz: float = 0.0, J: float = 0.0, mass: float = 0.0,
                  G: float | None = None, rot_mass: float = 0.0):
    """a femotion.frame Section (m, kPa) of a concrete member for the analysis of a statically indeterminate structure:
    E = Ec, A the gross area, Iy (vertical bending) = factor × I of the gross section (4.3.1, stiffness_factor(member)).
    Iz (lateral bending, m⁴) gets the same factor (4.3.1 speaks of flexural stiffness in general); J (m⁴), G, mass (t/m)
    and rot_mass pass through unchanged."""
    from ...frame import Section
    k = stiffness_factor(member)
    return Section(E=sec.concrete.Ec * 1e3, A=sec.area * 1e-6, Iy=k * sec.I * 1e-12, Iz=k * Iz, J=J, G=G, mass=mass,
                   rot_mass=rot_mass)


def variable_inertia_needed(I_support: float, I_midspan: float) -> bool:
    """4.3.6: a haunched continuous beam, or one of constant depth with haunches over the supports, needs the varying
    second moment in the analysis unless I at the support / I at midspan ≤ 2"""
    return I_support / I_midspan > 2.0


def spread_length(bearing: float, y: float, *, circular: bool = False) -> float:
    """a of 4.3.5 (figure 4.3.5): the reaction spread at 45° both ways from the bearing up to the centroid axis G-G:
    a = bearing length along the beam + 2 y, y the height from the bearing face (the soffit) to the centroid axis; a
    circular bearing as a square of side 0.8 × its diameter. Any one length unit."""
    return (0.8 * bearing if circular else bearing) + 2 * y


def support_moment_reduction(M: float, R: float, a: float, *, floor: float = 0.9) -> tuple[float, dict]:
    """4.3.5: the negative moment over an interior support reduced for the width of the bearing: Me = M − M′ in
    magnitude, M′ = q a² / 8 = R a / 8 (4.3.5-2, q = R / a), but |Me| ≥ 0.9 |M|. M: the support moment (kN·m, hogging
    negative), R the support reaction (kN, ≥ 0), a (m, spread_length). Returns (Me kN·m, details); M ≥ 0 is returned
    unchanged (the clause is about the negative moment)."""
    if R < 0 or a < 0:
        raise ValueError("R and a must be ≥ 0")
    Mp = R * a / 8
    if M >= 0:
        return M, {"M_prime": Mp, "applied": 0.0, "capped": False}
    applied = min(Mp, (1 - floor) * abs(M))
    return M + applied, {"M_prime": Mp, "applied": applied, "capped": applied < Mp}


def peak_shaving(x, M, x_support: float, R: float, a: float, *, floor: float = 0.9) -> np.ndarray:
    """the moment diagram over an interior support with the reaction R spread evenly over a (q = R / a, figure 4.3.5):
    M(x) + k q (a/2 − |x − x_support|)² / 2 within a/2 of the support, unchanged outside. The parabola is the exact
    difference between a point reaction and the same reaction spread over a (statics), it equals M′ = R a / 8 at the
    support (4.3.5-2) and joins the original diagram with the same slope at both ends; k ≤ 1 scales it so that the support
    moment keeps at least 0.9 of its magnitude (4.3.5). x (m), M (kN·m, sagging +; a position may appear twice at the
    support), R (kN), a (m). A support moment ≥ 0 leaves M unchanged."""
    x = np.asarray(x, float)
    M = np.asarray(M, float)
    near = np.abs(x - x_support) <= 1e-9 * max(1.0, abs(x_support))
    Ms = float(M[near].min()) if near.any() else float(np.interp(x_support, x, M))
    if Ms >= 0 or R <= 0 or a <= 0:
        return M.copy()
    Me, d = support_moment_reduction(Ms, R, a, floor=floor)
    k = d["applied"] / d["M_prime"]
    u = np.abs(x - x_support)
    dM = np.where(u <= a / 2, (R / a) * (a / 2 - u) ** 2 / 2, 0.0)
    return M + k * dM
