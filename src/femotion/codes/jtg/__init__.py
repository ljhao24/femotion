"""Chinese highway bridge codes.

    actions    JTG D60-2015: lane and vehicle loads, impact factor, design lanes, reduction factors, basic combination
    materials  JTG 3362-2018: concrete C25–C80, reinforcing steel, ξb
    rc         JTG 3362-2018: ultimate-limit-state resistance of reinforced-concrete sections (flexure, shear, axial and
               eccentric compression) on rectangles, T / I / box girders, solid and hollow circles

Typical use on a frame model (spine of a girder bridge), see examples/jtg_rc_girder.py:

    il = InfluenceLines(sv, deck_nodes, chain_sections(deck_beams))     # femotion.frame.influence
    G = uniform_load_forces(sv, il.sections, "My")                         # self-weight
    Q = jtg.LiveLoad(L0=20, lanes=2, mu=mu).effects(il, "My")              # lane load, (max, min)
    Mmax, Mmin = jtg.basic_combination(G, Q)
    jtg.flexure(section, Mmax[k], gamma0=1.0)

Prestressed concrete, serviceability (crack width, deflection, stresses), torsion and local bearing are not covered.
"""
from .actions import (GAMMA_0, VEHICLE_AXLES, VEHICLE_SPACINGS, LiveLoad, basic_combination, design_lanes,  # noqa: F401
                      impact_factor, lane_load, longitudinal_reduction, simple_beam_frequency, transverse_reduction)
from .materials import Concrete, Rebar, concrete, rebar, xi_b  # noqa: F401
from .rc import (Check, RCSection, axial, box, circle, eccentric, eta_factor, flexure, rect, shear,  # noqa: F401
                 stability_factor, tee)
