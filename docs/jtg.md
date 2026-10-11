# Chinese highway bridge codes (`femotion.codes.jtg`)

Ultimate-limit-state checks of reinforced-concrete highway bridges to

- **JTG D60-2015** *General Specifications for Design of Highway Bridges and Culverts*: traffic actions, impact,
  combinations;
- **JTG 3362-2018** *Specifications for Design of Highway Reinforced Concrete and Prestressed Concrete Bridges and
  Culverts*: materials and section resistance.

> **Status: alpha, checked against the official text.** Every table value, formula and clause number used here was
> compared with the official texts published by the Ministry of Transport (2026-10-11); the deviations found were fixed
> and each has a test. Clause numbers below refer to the 2018 and 2015 editions. The tests verify the implementation
> against hand calculations and closed forms. femotion does not replace a qualified engineer's design or review.

## Workflow

```python
from femotion.codes import jtg
from femotion.frame import Solver
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces

sv = Solver(model)                                      # spine model of the girder, m / kN / t
secs = chain_sections(deck_beams)                       # both ends of every deck element
il = InfluenceLines(sv, deck_nodes, secs)               # one factorisation, one batch of back-substitutions
G = uniform_load_forces(sv, secs, "My")                 # self-weight, exact at element ends
Q = jtg.LiveLoad(L0=16, lanes=1, transverse=1.0, distribution=0.55, mu=jtg.impact_factor(f1)).effects(il, "My")
Mmax, Mmin = jtg.basic_combination(G, Q)
sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])
print(jtg.flexure(sec, Mmax[k], gamma0=1.0))            # flexure: γ0Sd = …, R = …, ratio … → OK
```

`examples/jtg_rc_girder.py` runs the whole chain on a two-span continuous T girder.

Units: frame analysis in m, kN, t; sections in mm, mm², MPa; design forces in kN and kN·m.

## What is covered

| Part | Function | Clause |
|---|---|---|
| Lane load Class I / II, Pk × 1.2 for shear | `lane_load` | D60 4.3.1 |
| Vehicle load (550 kN, 5 axles) | `VEHICLE_AXLES`, `LiveLoad.vehicle_effects` | D60 4.3.1 |
| Design lanes, transverse and longitudinal reduction | `design_lanes`, `transverse_reduction`, `longitudinal_reduction` | D60 4.3.1 |
| Impact factor from frequency; frequency of a simple beam | `impact_factor`, `simple_beam_frequency` | D60 4.3.2 |
| Basic combination, γ0 | `basic_combination`, `GAMMA_0` | D60 4.1.5 |
| Influence lines, lane-load and axle-train envelopes | `femotion.frame.influence` | (generic) |
| Concrete C25–C80, HPB300 / HRB400 / HRBF400 / RRB400 / HRB500; β, εcu | `concrete`, `rebar` | 3362 3.1, 3.2, 5.1.4, 5.1.5 |
| ξb (table 5.2.1; none for ordinary bars in C75 / C80) | `xi_b` | 3362 5.2.1 |
| Flexure of rectangles, T, I, box girders, circles; x ≤ ξb h0; x < 2as′; ρmin | `flexure` | 3362 5.2.2, 5.2.3, 5.2.4, 9.1.12 |
| Shear (stirrups, bent bars), section limit, detailing threshold, ρsv,min | `shear` | 3362 5.2.9, 5.2.11, 5.2.12, 9.3.12 |
| Axial compression with φ (least radius of gyration), minimum longitudinal steel | `axial`, `stability_factor` | 3362 5.3.1, 5.3.10, 9.1.12 |
| Eccentric compression (rectangles, T, box, hollow circles by the general rule set; solid circles with ≥ 8 bars by the closed formula), η, e0 ≥ max(20 mm, h/30), x < 2as′, reverse failure, minimum longitudinal steel | `eccentric`, `eta_factor` | 3362 5.3.4–5.3.9, 9.1.12 |

Not covered yet: prestressed concrete (losses, creep and shrinkage, construction stages), serviceability (crack width,
deflection, stress limits), torsion, local bearing, punching, shear of circular sections, web reinforcement of 5.3.7,
crowd, wind and other variable actions beyond passing their effects to `basic_combination(others=...)` (wind: γQ = 1.1;
choosing which variable action leads is left to the user, D60 4.1.5), accidental and seismic combinations.

## The rule set behind the section resistance

All shapes go through one routine (`RCSection.forces`): plane sections, concrete tension ignored, a rectangular block of
fcd over x = β·xc, bar stress σi = εcu·Es·(β·h0i/x − 1) clipped to −fsd′ … fsd. β = 0.8 up to C50 and 0.74 at C80,
εcu = 0.0033 up to C50 and 0.0030 at C80, linear between. For ordinary reinforcement (tension bars yield when
ξ ≤ ξb, compression bars yield when x ≥ 2as′) this reproduces the closed formulas of 5.2.2 / 5.2.3 / 5.3.5 exactly; the
tests check this. Where bars do not yield (small eccentricity, high-grade steel) it is the code's general expression.

- **x < 2as′ in flexure** (5.2.4): 5.2.4 applies "when the compression bars are counted", so the section may also be
  taken as singly reinforced. The resistance is the larger of (a) moments about the compression bars, concrete
  ignored, and (b) the singly reinforced section, provided its own x ≤ ξb h0. This follows from the premise of 5.2.4;
  the code does not state "take the larger" in so many words. The result notes which governs. Eccentric compression
  (5.3.6) uses (a) only.
- **Over-reinforced** (x > ξb h0) members fail the check; their resistance from the rule set is reported for
  information only.
- **Eccentric compression**: e0 = Md/Nd, at least max(20 mm, h/30) (5.3.9; there is no separate accidental
  eccentricity in the 2018 edition). e = η·e0 from the gross centroid, x solved from moment equilibrium, Nu read off.
  With unequal bars, a load between the gross and the plastic centroid is resisted with the far face more compressed:
  the section is flipped and solved again. Small eccentricity with the load between the bar groups also gets the
  reverse-failure check 5.3.4-4 / 5.3.5-3 (moments about A′s of the whole concrete section and As; e0 there without
  the minimum, the less favourable reading); the smaller resistance governs. Solid circles with at least 8 bars use the
  5.3.8 closed formula by default (`method="code"`); `method="strain"` uses the rule set (they agree within a few
  per cent, tested).

## Review against the official text

The texts (JTG D60-2015, JTG 3362-2018, official PDFs from xxgk.mot.gov.cn) were read clause by clause against this
module on 2026-10-11. Found and fixed:

| Item | Was | Now (code) |
|---|---|---|
| e0 minimum | none | e0 ≥ max(20 mm, h/30) (5.3.9) |
| ξb | formula per class | table 5.2.1 (classes grouped; C75 / C80 with ordinary bars: no value, raises) |
| Circle formula | "≥ 6 bars", not checked | ≥ 8 bars (5.3.8 note), else the rule set |
| Reverse failure | not checked | 5.3.4-4, 5.3.5-3 |
| Compression members, minimum steel | not checked | all bars ≥ 0.5 % (0.6 % from C50), each side ≥ 0.2 % (9.1.12) |
| φ | radius of gyration in the bending plane | least radius of gyration; T / I / box need `i_min` or `b_min` (table 5.3.1) |
| More than two lanes | not enforced | effect not below two lanes' (D60 4.3.1 item 7) |
| Longitudinal reduction at L0 = 150 m | 0.97 | 1.0 (reduction only above 150 m) |
| Wind in the combination | not mentioned | γQ = 1.1 documented (D60 4.1.5) |
| Flexure, x < 2as′, singly reinforced alternative | no ξb check | its own x ≤ ξb h0 |
| Clause numbers | 9.3.13, 3.2.5 | 9.3.12, 3.2.4; frequency formula is in the commentary to 4.3.2 (eq. 4-3) |

Confirmed unchanged: all material values (C25–C80, the reinforcement grades), β, εcu, the bar stress formula, the
lane and vehicle loads, the impact factor, the lane and reduction tables, γ0, γG, γQ, ψc, the shear formulas, limits
and coefficients, the axial formula, the φ table, η with ζ1 and ζ2, the circle formula with αt = 1.25 − 2α.
Note: the official English edition of 3362-2018 has two misprints (C40 ftk printed as 2.51, αt as 1.25 − α); the values
here follow the Chinese text.
