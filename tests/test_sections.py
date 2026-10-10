"""Sections of any shape through sectionproperties: closed-form properties, the axis mapping on a cantilever, composite
sections, and fiber sections in OpenSees."""
import numpy as np
import pytest

pytest.importorskip("sectionproperties")
from sectionproperties.pre.library import (  # noqa: E402
    angle_section, circular_hollow_section, concrete_rectangular_section, i_section, rectangular_hollow_section,
    rectangular_section)
from sectionproperties.pre.pre import Material  # noqa: E402

from femotion.frame import Model, Solver, sections  # noqa: E402
from femotion.frame.opensees import Fiber  # noqa: E402

CONCRETE = Material(name="concrete", elastic_modulus=3.45e7, poissons_ratio=0.2, yield_strength=3.24e4, density=2.5, color="lightgrey")
STEEL = Material(name="steel", elastic_modulus=2.06e8, poissons_ratio=0.3, yield_strength=3.45e5, density=7.85, color="grey")


def j_rect(b, h, n=200):
    """exact St Venant torsion constant of a b × h rectangle (b ≤ h), series solution"""
    k = np.arange(1, 2 * n, 2)
    return b**3 * h / 3 * (1 - 192 / np.pi**5 * b / h * np.sum(np.tanh(k * np.pi * h / (2 * b)) / k**5))


def test_rectangle_closed_form():
    b, h = 0.4, 1.0
    s = sections.elastic(rectangular_section(d=h, b=b), E=3.45e7, nu=0.2, density=2.5)
    assert s.A == pytest.approx(b * h, rel=1e-12)
    assert s.Iy == pytest.approx(b * h**3 / 12, rel=1e-12) and s.Iz == pytest.approx(h * b**3 / 12, rel=1e-12)
    assert s.J == pytest.approx(j_rect(b, h), rel=1e-4)
    assert s.g() == pytest.approx(3.45e7 / 2.4)
    assert s.mass == pytest.approx(2.5 * b * h) and s.rot_mass == pytest.approx(2.5 * (b * h**3 + h * b**3) / 12)


def test_axes_on_a_cantilever():
    """the drawing's up is the member's up: a tall section is stiff vertically and soft sideways"""
    b, h, L, P, E = 0.2, 0.6, 10.0, 10.0, 2e8
    sec = sections.elastic(rectangular_section(d=h, b=b), E=E)
    m = Model()
    ns = [m.node((L * k / 10, 0, 0)) for k in range(11)]
    for a, c in zip(ns[:-1], ns[1:]):
        m.beam(a, c, sec)
    m.fix(ns[0])
    sv = Solver(m)
    for dof, inertia in ((2, b * h**3 / 12), (1, h * b**3 / 12)):  # global z (vertical), global y (sideways)
        F = np.zeros(m.ndof)
        F[6 * ns[-1] + dof] = -P
        assert -sv.solve(F)[6 * ns[-1] + dof] == pytest.approx(P * L**3 / (3 * E * inertia), rel=1e-9)


def test_thin_walled_sections():
    tube = sections.elastic(circular_hollow_section(d=1.0, t=0.05, n=128), E=2e8)
    assert tube.J == pytest.approx(tube.Iy + tube.Iz, rel=1e-4)  # closed circular tube: J = polar moment
    box = sections.elastic(rectangular_hollow_section(d=1.0, b=2.0, t=0.04, r_out=0.0, n_r=1), E=2e8)
    assert box.J == pytest.approx(4 * (1.96 * 0.96) ** 2 / (2 * 1.96 / 0.04 + 2 * 0.96 / 0.04), rel=0.02)  # Bredt
    i = sections.elastic(i_section(d=1.0, b=0.5, t_f=0.04, t_w=0.02, r=0.0, n_r=1), E=2e8)
    assert i.J == pytest.approx((2 * 0.5 * 0.04**3 + 0.92 * 0.02**3) / 3, rel=0.05)  # open thin walls: Σ b t³ / 3


def test_composite_transformed_section():
    """concrete slab 1.0 × 0.2 on a steel web 0.1 × 1.0: modulus-weighted properties against hand calculation"""
    slab = rectangular_section(d=0.2, b=1.0, material=CONCRETE).shift_section(-0.5, 1.0)
    web = rectangular_section(d=1.0, b=0.1, material=STEEL).shift_section(-0.05, 0.0)
    parts = [(CONCRETE, 1.0, 0.2, 1.1), (STEEL, 0.1, 1.0, 0.5)]  # material, width, depth, centroid height
    EA = sum(mt.elastic_modulus * b * d for mt, b, d, _ in parts)
    yc = sum(mt.elastic_modulus * b * d * y for mt, b, d, y in parts) / EA
    EI = sum(mt.elastic_modulus * (b * d**3 / 12 + b * d * (y - yc) ** 2) for mt, b, d, y in parts)
    s = sections.elastic(slab + web)
    assert s.E == CONCRETE.elastic_modulus and s.g() == pytest.approx(3.45e7 / 2.4)
    assert s.E * s.A == pytest.approx(EA, rel=1e-12) and s.E * s.Iy == pytest.approx(EI, rel=1e-9)
    assert s.mass == pytest.approx(sum(mt.density * b * d for mt, b, d, _ in parts), rel=1e-12)
    rot = sum(mt.density * ((b * d**3 + d * b**3) / 12 + b * d * (y - yc) ** 2) for mt, b, d, y in parts)
    assert s.rot_mass == pytest.approx(rot, rel=1e-9)
    s2 = sections.elastic(slab + web, E=STEEL.elastic_modulus)  # another reference modulus, same rigidities
    assert s2.E * s2.A == pytest.approx(EA, rel=1e-12) and s2.E * s2.Iy == pytest.approx(EI, rel=1e-9)


def test_unsymmetric_section_warns():
    with pytest.warns(UserWarning, match="principal axes"):
        sections.elastic(angle_section(d=0.2, b=0.1, t=0.01, r_r=0.0, r_t=0.0, n_r=1), E=2e8)


def test_fibers_from_a_reinforced_concrete_section():
    rc = concrete_rectangular_section(d=1.0, b=0.6, dia_top=0.025, area_top=4.9e-4, n_top=3, c_top=0.05, dia_bot=0.032,
                                      area_bot=8.0e-4, n_bot=4, c_bot=0.05, conc_mat=CONCRETE, steel_mat=STEEL)
    mats = {"concrete": Fiber.concrete(3.24e4, Ec=3.45e7), "steel": Fiber.steel(4.0e5)}
    fb = sections.fiber(rc, mats)
    f = np.array([(y, z, a, m == "steel") for m, y, z, a in fb.fibers])
    assert f[f[:, 3] == 1, 2].sum() == pytest.approx(3 * 4.9e-4 + 4 * 8.0e-4, rel=1e-9)  # bar areas exact
    assert f[:, 2].sum() == pytest.approx(0.6, rel=1e-9)
    E = np.where(f[:, 3] == 1, 2.06e8, 3.45e7)
    assert abs((E * f[:, 2] * f[:, 1]).sum()) < 1e-9 * (E * f[:, 2]).sum()  # measured from the modulus-weighted centroid
    with pytest.raises(KeyError):
        sections.fiber(rc, {"concrete": mats["concrete"]})


def test_fiber_plastic_moment():
    """cantilever with a fiber section from sectionproperties: the base moment reaches Mp = fy·b·h²/4"""
    pytest.importorskip("openseespy")
    from femotion.frame.opensees import OpsFrame

    Lc, b, h, fy = 5.0, 0.1, 0.2, 3.45e5
    rect = rectangular_section(d=h, b=b)
    m = Model()
    ns = [m.node((Lc * k / 4, 0, 0)) for k in range(5)]
    for a, c in zip(ns[:-1], ns[1:]):
        m.beam(a, c, sections.elastic(rect, E=2e8), tag="steel")
    m.fix(ns[0])
    F = np.zeros(m.ndof)
    F[6 * ns[-1] + 2] = -1.0
    fb = sections.fiber(rect, Fiber.steel_pp(fy), mesh_size=b * h / 200)
    lam, H = OpsFrame(m, fiber={"steel": fb}).pushover(F, node=ns[-1], dof=2, target=-0.4, steps=80)
    assert lam.max() * Lc / (fy * b * h**2 / 4) == pytest.approx(1.0, rel=0.01)
    assert -H[1, 6 * ns[-1] + 2] == pytest.approx(lam[1] * Lc**3 / (3 * 2e8 * b * h**3 / 12), rel=0.01)
