"""Cross-sections of any shape (extra "sections"): sectionproperties meshes the section and computes the area, second moments,
St Venant torsion constant (warping analysis), modulus-weighted properties of composite sections and the mass; femotion turns
them into a frame Section for either solver, or into an OpenSees fiber section of the same shape.

    from sectionproperties.pre.library import rectangular_hollow_section
    from femotion.frame import sections
    from femotion.frame.opensees import Fiber
    box = rectangular_hollow_section(d=2.5, b=1.2, t=0.03, r_out=0.0, n_r=1)   # in m: depth d, width b
    sec = sections.elastic(box, E=2.06e8, nu=0.3, density=7.85)               # → femotion.frame.Section
    fib = sections.fiber(box, Fiber.steel(3.45e5))                             # → Fiber for OpsFrame(fiber={tag: fib})

Any shapely polygon works too (sectionproperties.pre.geometry.Geometry(Polygon(outline, [hole]))), e.g. a box girder with
cantilever slabs, and geometries add up (a + b) into composite sections.

Axes: draw the section as seen from the member's end node j looking back towards i (local x pointing at you). The drawing's x
(right) is the member's local y and its y (up) is local z, so with the default reference vector (global up) the drawing's up
is up: Iy is the second moment about the drawing's horizontal axis (vertical bending), Iz about its vertical axis. The beam
axis is the (modulus-weighted) centroid. femotion's beams take the local axes as principal axes and the centroid as the shear
centre; a section with a product of inertia gets a warning with its principal angle (rotate the reference vector by it).

Composite sections: give every sectionproperties geometry a Material (elastic_modulus in kPa, density in t/m³, consistent
with femotion's m, kN, t). Properties are modulus-weighted and transformed to the reference modulus E (default: the first
material's), G = E·G_ref/E_ref of that material, and mass and rotary inertia come from the densities.

Licence: sectionproperties is MIT; its mesher (Shewchuk's Triangle, through cytriangle) is free for non-commercial use.
"""
from __future__ import annotations

import warnings

from .model import Section
from .opensees import Fiber


def analyse(shape, mesh_size: float | None = None):
    """mesh a sectionproperties Geometry / CompoundGeometry and run its frame analysis (area, second moments, torsion constant).
    mesh_size: maximum triangle area, default area / 500 (torsion constant of a rectangle within 1e-4 of the exact series).
    An already-analysed sectionproperties Section is returned as it is."""
    from sectionproperties.analysis import Section as SpSection

    if isinstance(shape, SpSection):
        if shape.section_props.j is None:
            shape.calculate_frame_properties()
        return shape
    shape.create_mesh(mesh_sizes=mesh_size or shape.calculate_area() / 500)
    s = SpSection(shape)
    s.calculate_frame_properties()
    return s


def _check_principal(p):
    if abs(p.ixy_c) > 1e-6 * max(abs(p.ixx_c), abs(p.iyy_c)):
        warnings.warn(f"the section's principal axes are rotated by {p.phi:.3f}° from its drawing axes; femotion beams bend about "
                      "their local axes, so rotate the member's reference vector by this angle", stacklevel=3)


def _inertia(s, about: tuple[float, float]) -> tuple[float, float]:
    """density-weighted area and polar moment about a point, summed over the mesh (composite sections)"""
    cx, cy = about
    m = r = 0.0
    for el in s.elements:
        area, qx, qy, ixx, iyy, _, _, _, rho = el.geometric_properties()
        m += rho * area
        r += rho * (ixx + iyy - 2 * cy * qx - 2 * cx * qy + area * (cx**2 + cy**2))
    return m, r


def elastic(shape, E: float | None = None, nu: float | None = None, density: float = 0.0, mesh_size: float | None = None) -> Section:
    """section of any shape → femotion.frame.Section (A, Iy, Iz, J, G, mass per length, rotary inertia per length).
    shape: a sectionproperties Geometry / CompoundGeometry (meshed here) or an analysed sectionproperties Section.
    Without materials: E (kPa) is required, nu gives G (default: femotion's E / 2.6), density (t/m³) gives the mass.
    With materials: they define everything; E only picks a different reference modulus."""
    s = analyse(shape, mesh_size)
    p = s.section_props
    _check_principal(p)
    if s.is_composite():
        if density:
            raise ValueError("a composite section takes its densities from the sectionproperties materials")
        ref = s.materials[0]
        E = ref.elastic_modulus if E is None else E
        G = E * ref.shear_modulus / ref.elastic_modulus
        mass, rot = _inertia(s, (p.cx, p.cy))
        w = E  # sectionproperties weights by the modulus
    else:
        if E is None:
            raise ValueError("E is required for a section without materials")
        G = None if nu is None else E / (2 * (1 + nu))
        mass, rot = density * p.area, density * (p.ixx_c + p.iyy_c)
        w = 1.0
    return Section(E=float(E), A=float(p.ea / w), Iy=float(p.ixx_c / w), Iz=float(p.iyy_c / w), J=float(p.j / w),
                   G=None if G is None else float(G), mass=float(mass), rot_mass=float(rot))


def fiber(shape, materials, GJ: float | None = None, mesh_size: float | None = None) -> Fiber:
    """section of any shape → OpenSees fiber section: one fiber per triangle of the sectionproperties mesh, at its centroid
    measured from the (modulus-weighted) centroid of the section. materials: one material tuple (Fiber.steel(...),
    Fiber.concrete(...)) for the whole section, or {sectionproperties material name: tuple} for composite sections (e.g.
    concrete and bars from sectionproperties' concrete library). GJ: elastic torsional stiffness (fibers carry no torsion);
    default G·J from the analysis, with G = modulus / 2.6 for a single material tuple. A coarser mesh_size gives fewer fibers."""
    s = analyse(shape, mesh_size)
    p = s.section_props
    _check_principal(p)
    if isinstance(materials, tuple):
        mats, name = {"m": materials}, (lambda el: "m")
    else:
        mats, name = dict(materials), (lambda el: el.material.name)
        missing = {m.name for m in s.materials} - set(mats)
        if missing:
            raise KeyError(f"no fiber material for: {', '.join(sorted(missing))}")
    fibers = []
    for el in s.elements:
        area, qx, qy, *_ = el.geometric_properties()
        fibers.append((name(el), qy / area - p.cx, qx / area - p.cy, area))
    if GJ is None:
        if s.is_composite():
            ref = s.materials[0]
            GJ = p.j * ref.shear_modulus / ref.elastic_modulus
        elif isinstance(materials, tuple):
            GJ = Fiber.modulus(materials) / 2.6 * p.j
        else:
            raise ValueError("give GJ: the section has no sectionproperties materials to take a shear modulus from")
    return Fiber(mats, fibers=fibers, GJ=float(GJ))

