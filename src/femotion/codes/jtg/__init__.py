"""Chinese highway bridge codes.

    actions    JTG D60-2015: lane and vehicle loads, impact factor, design lanes, reduction factors, basic combination
    materials  JTG 3362-2018: concrete C25–C80, reinforcing and prestressing steel, ξb, σcon limits
    rc         JTG 3362-2018: ultimate-limit-state resistance of reinforced and prestressed concrete sections (flexure,
               shear, axial and eccentric compression) on rectangles, T / I / box girders, solid and hollow circles;
               cracking moment and minimum reinforcement of prestressed members
    prestress  JTG 3362-2018: prestress losses (friction, anchorage set with reverse friction, temperature,
               elastic shortening, relaxation, creep and shrinkage with Appendix C), σpe, σp0; anchorage and
               transfer lengths of pretensioned tendons
    sls        JTG 3362-2018 chapters 6 and 7: crack resistance of prestressed members (normal and inclined sections,
               principal stresses), crack width, cracked sections, flexural stiffness and deflection, stresses in
               service and in construction stages

Typical use on a frame model (spine of a girder bridge), see examples/jtg_rc_girder.py:

    il = InfluenceLines(sv, deck_nodes, chain_sections(deck_beams))     # femotion.frame.influence
    G = uniform_load_forces(sv, il.sections, "My")                         # self-weight
    Q = jtg.LiveLoad(L0=20, lanes=2, mu=mu).effects(il, "My")              # lane load, (max, min)
    Mmax, Mmin = jtg.basic_combination(G, Q)
    jtg.flexure(section, Mmax[k], gamma0=1.0)
    Ms = jtg.frequent_combination(G, jtg.LiveLoad(L0=20, lanes=2).effects(il, "My"))[0]      # no impact
    Ml = jtg.quasi_permanent_combination(G, jtg.LiveLoad(L0=20, lanes=2).effects(il, "My"))[0]
    jtg.crack_width(section, Ms[k], Ml[k], env="I")

Secondary effects of prestress on statically indeterminate structures: femotion.frame.prestress. External and unbonded
tendons, torsion and local bearing are not covered.
"""
from .actions import (DYNAMIC_FACTOR, GAMMA_0, PSI, VEHICLE_AXLES, VEHICLE_SPACINGS, LiveLoad,  # noqa: F401
                      basic_combination, characteristic_combination, continuous_beam_frequencies, design_lanes,
                      frequent_combination, impact_factor, lane_load, longitudinal_reduction, quasi_permanent_combination,
                      simple_beam_frequency, transverse_reduction)
from . import prestress  # noqa: F401
from .materials import Concrete, Rebar, Tendon, concrete, rebar, strength_at, tendon, xi_b  # noqa: F401
from .rc import (Check, RCSection, axial, box, circle, cracking_moment, eccentric, eta_factor, flexure, rect, shear,  # noqa: F401
                 stability_factor, tee)
from .sls import (CAMBER_LONG_TERM, CRACK_WIDTH_LIMITS, crack_resistance, crack_width, crack_width_limit,  # noqa: F401
                  cracked_section, deflection_check, deflection_limit, inclined_crack_resistance, long_term_factor,
                  precamber_pc, precamber_rc, principal_compression, principal_stresses, service_stresses, stiffness,
                  transient_pc, transient_rc)
