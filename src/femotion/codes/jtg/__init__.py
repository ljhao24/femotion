"""Chinese highway bridge codes.

    actions      JTG D60-2015: lane and vehicle loads, impact factor, design lanes, reduction factors, crowd load,
                 basic, frequent, quasi-permanent and characteristic combinations
    temperature  JTG D60-2015 4.3.12 (expansion coefficients, effective temperatures, vertical gradient) and JTG 3362-2018
                 4.3.10 / appendix D (temperature stresses; secondary moments with femotion.frame.thermal)
    materials    JTG 3362-2018: concrete C25–C80, reinforcing and prestressing steel, ξb, σcon limits
    flange       JTG 3362-2018 4.3.3, 4.3.4: effective flange widths of T, I and box girders
    rc           JTG 3362-2018: ultimate-limit-state resistance of reinforced and prestressed concrete sections (flexure,
                 shear, axial and eccentric compression) on rectangles, T / I / box girders, solid and hollow circles;
                 cracking moment and minimum reinforcement of prestressed members
    torsion      JTG 3362-2018 5.5, 9.3.13: torsion and shear with torsion of rectangles, boxes, T and I sections
    bearing      JTG 3362-2018 5.7: local compression under anchorages and bearings
    prestress    JTG 3362-2018: prestress losses (friction, anchorage set with reverse friction, temperature,
                 elastic shortening, relaxation, creep and shrinkage with Appendix C), σpe, σp0; anchorage and
                 transfer lengths of pretensioned tendons
    sls          JTG 3362-2018 chapters 6 and 7: crack resistance of prestressed members (normal and inclined sections,
                 principal stresses, with the temperature gradient), crack width, cracked sections, flexural stiffness
                 and deflection, stresses in service and in construction stages
    composite    JTG 3362-2018 8.1: composite flexural members (precast girder + cast-in-place topping, two stages)
    stm          JTG 3362-2018 appendix B: strut-and-tie models (statics, ties, struts, nodes)
    regions      JTG 3362-2018 8.2–8.5: anchorage zones, diaphragms over bearings, cap beams, pile caps
    slab         JTG 3362-2018 4.2, 5.6, 9.2: deck slabs (spans, wheel load distribution widths, slabs between ribs and
                 cantilevers with the D60 vehicle), punching of slabs and footings, slab detailing
    transverse   JTG 3362-2018 table 6.1.3: box girders across the bridge as closed frames (wheels, transverse tendons,
                 temperature), shear flow of single cells, in-plane principal stresses, transverse stress checks

Typical use on a frame model (spine of a girder bridge), see examples/jtg_rc_girder.py:

    il = InfluenceLines(sv, deck_nodes, chain_sections(deck_beams))     # femotion.frame.influence
    G = uniform_load_forces(sv, il.sections, "My")                         # self-weight
    Q = jtg.LiveLoad(L0=20, lanes=2, mu=mu).effects(il, "My")              # lane load, (max, min)
    C = jtg.CrowdLoad(L0=20, width=1.5).effects(il, "My")                  # crowd on the sidewalks
    Mmax, Mmin = jtg.basic_combination(G, Q, [C])
    jtg.flexure(section, Mmax[k], gamma0=1.0)
    Q0 = jtg.LiveLoad(L0=20, lanes=2).effects(il, "My")                    # no impact
    Ms = jtg.frequent_combination(G, Q0, [C], kinds=["crowd"])[0]
    Ml = jtg.quasi_permanent_combination(G, Q0, [C], kinds=["crowd"])[0]
    jtg.crack_width(section, Ms[k], Ml[k], env="I")

Secondary effects of prestress on statically indeterminate structures: femotion.frame.prestress; of temperature and
shrinkage: femotion.frame.thermal. External and unbonded tendons are not covered.
"""
from .actions import (DYNAMIC_FACTOR, GAMMA_0, PSI, RAILING_LOADS, SIDEWALK_SLAB_LOAD, VEHICLE_AXLES,  # noqa: F401
                      VEHICLE_SPACINGS, CrowdLoad, LiveLoad, basic_combination, characteristic_combination,
                      continuous_beam_frequencies, crowd_load, design_lanes, frequent_combination, impact_factor,
                      lane_load, longitudinal_reduction, quasi_permanent_combination, simple_beam_frequency,
                      transverse_reduction)
from . import prestress, regions  # noqa: F401
from . import slab, transverse  # noqa: F401
from .bearing import bearing_base_area, fcd_at, local_bearing, mesh_ratio, spiral_ratio  # noqa: F401
from .composite import (COMPOSITE_CAMBER_LONG_TERM, Composite, composite, composite_crack_resistance,  # noqa: F401
                        composite_crack_width, composite_design_moments, composite_inclined_crack_resistance,
                        composite_long_term_factor, composite_principal_compression, composite_principal_stresses, composite_service_stresses,
                        composite_shear, composite_stiffness, interface_shear, interface_shear_slab,
                        shrinkage_difference_profile)
from .flange import (box_flange_coefficient, box_flange_width, box_theoretical_span, tee_flange_width,  # noqa: F401
                     tee_flange_width_outer)
from .materials import Concrete, Rebar, Tendon, concrete, rebar, strength_at, tendon, xi_b  # noqa: F401
from .rc import (Check, RCSection, axial, box, circle, cracking_moment, eccentric, eta_factor, flexure, rect, shear,  # noqa: F401
                 stability_factor, tee)
from .sls import (CAMBER_LONG_TERM, CRACK_WIDTH_LIMITS, crack_resistance, crack_width, crack_width_limit,  # noqa: F401
                  cracked_section, deflection_check, deflection_limit, inclined_crack_resistance, long_term_factor,
                  precamber_pc, precamber_rc, principal_compression, principal_stresses, service_stresses, stiffness,
                  transient_pc, transient_rc)
from .stm import StrutTie, strut_width  # noqa: F401
from .temperature import (EFFECTIVE_TEMPERATURE, EXPANSION, NEGATIVE_GRADIENT, Profile, ThermalStresses,  # noqa: F401
                          effective_temperature, temperature_gradient, thermal_stresses)
from .torsion import torsion, torsion_flange, torsion_moduli  # noqa: F401
