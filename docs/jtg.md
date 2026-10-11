# Chinese highway bridge codes (`femotion.codes.jtg`)

Ultimate-limit-state checks of reinforced-concrete highway bridges to

- **JTG D60-2015** *General Specifications for Design of Highway Bridges and Culverts*: traffic actions, impact,
  combinations;
- **JTG 3362-2018** *Specifications for Design of Highway Reinforced Concrete and Prestressed Concrete Bridges and
  Culverts*: materials and section resistance.

> **Status: alpha, unverified transcription.** Every value and formula below was transcribed into code from the code
> texts as the author of this module knows them, not copied from an official copy, and has not yet been checked
> against the printed clauses. The tests verify that the formulas are *implemented* as written here (hand calculations,
> closed forms), not that they are *the code*. Check the "verify" items before using results for anything that matters.
> femotion does not replace a qualified engineer's design or review.

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
| Concrete C25–C80, HPB300 / HRB400 / HRBF400 / RRB400 / HRB500 | `concrete`, `rebar` | 3362 3.1, 3.2 |
| ξb | `xi_b` | 3362 5.2.1 |
| Flexure of rectangles, T, I, box girders, circles; x ≤ ξb h0; x < 2as′; ρmin | `flexure` | 3362 5.2.2, 5.2.3, 9.1.12 |
| Shear (stirrups, bent bars), section limit, detailing threshold, ρsv,min | `shear` | 3362 5.2.9, 5.2.11, 5.2.12, 9.3.13 |
| Axial compression with φ | `axial`, `stability_factor` | 3362 5.3.1 |
| Eccentric compression (rectangles, T, box, hollow circles by the general rule set; solid circles by the closed formula), η | `eccentric`, `eta_factor` | 3362 5.3.4–5.3.9 |

Not covered yet: prestressed concrete (losses, creep and shrinkage, construction stages), serviceability (crack width,
deflection, stress limits), torsion, local bearing, punching, the "reverse failure" check of small-eccentricity
members, shear of circular sections, crowd and other variable actions beyond passing their effects to
`basic_combination(others=...)`, accidental and seismic combinations.

## The rule set behind the section resistance

All shapes go through one routine (`RCSection.forces`): plane sections, concrete tension ignored, a rectangular block of
fcd over x = β·xc, bar stress σi = εcu·Es·(β·h0i/x − 1) clipped to −fsd′ … fsd. β = 0.8 up to C50 and 0.74 at C80,
εcu = 0.0033 up to C50 and 0.0030 at C80, linear between. For ordinary reinforcement (tension bars yield when
ξ ≤ ξb, compression bars yield when x ≥ 2as′) this reproduces the closed formulas of 5.2.2 / 5.2.3 / 5.3.5 exactly; the
tests check this. Where bars do not yield (small eccentricity, high-grade steel) it is the code's general expression.

- **x < 2as′ in flexure**: the code requires x ≥ 2as′ only when compression bars are counted. The resistance is the
  larger of (a) moments about the compression bars, concrete ignored, and (b) the section as singly reinforced. The
  result notes which governs.
- **Over-reinforced** (x > ξb h0) members fail the check; their resistance from the rule set is reported for
  information only.
- **Eccentric compression**: e = η·e0 from the gross centroid, x solved from moment equilibrium, Nu read off. Solid
  circles use the 5.3.8 closed formula by default (`method="code"`); `method="strain"` uses the rule set (they agree
  within a few per cent, tested).

## Values to verify against the code text

| Item | Value used | Note |
|---|---|---|
| Concrete fck, ftk, fcd, ftd, Ec, C25–C80 | `materials._CONCRETE` | e.g. C30: 20.1, 2.01, 13.8, 1.39, 3.00×10⁴ MPa |
| Rebar fsk, fsd, fsd′, Es | HPB300 300/250/250/2.1×10⁵; HRB400 400/330/330/2.0×10⁵; HRB500 500/415/400/2.0×10⁵ | |
| Stirrup strength cap | fsv = min(fsd, 330) | |
| ξb | β / (1 + fsd/(εcu Es)) rounded down to 0.01 | matches the table up to C50 (0.58 / 0.53 / 0.49); **above C50 the table groups classes and may differ by 0.01** |
| Lane load | qk = 10.5 kN/m; Pk = 270 (L0 ≤ 5) … 360 (L0 ≥ 50), 2(L0 + 130) between; Class II × 0.75; Pk × 1.2 for shear | |
| Impact factor | 0.05 (f < 1.5), 0.1767 ln f − 0.0157, 0.45 (f > 14) | |
| Transverse reduction | 1.20, 1.00, 0.78, 0.67, 0.60, 0.55, 0.52, 0.50 for 1–8 lanes | |
| Longitudinal reduction | 0.97 / 0.96 / 0.95 / 0.94 / 0.93 from 150 / 400 / 600 / 800 / 1000 m | |
| Design lanes | table 4.3.1-4 | |
| Basic combination | γG 1.2 / 1.0, γQ1 1.4 (lane) / 1.8 (vehicle), other variable × 1.4 × ψc 0.75; γ0 1.1 / 1.0 / 0.9 | |
| Shear | Vcs = α1 α2 α3 · 0.45×10⁻³ · b h0 √((2 + 0.6P) √fcu,k ρsv fsv), P = 100ρ ≤ 2.5; Vsb = 0.75×10⁻³ fsd ΣAsb sin θs | |
| Section limit / detailing | 0.51×10⁻³ √fcu,k b h0; 0.50×10⁻³ α2 ftd b h0 (× 1.25 for slabs) | |
| Minimum reinforcement | ρ ≥ max(45 ftd/fsd %, 0.20 %); ρsv ≥ 0.14 % (HPB300), 0.11 % (HRB400 / 500) | |
| φ table | l0/b 8 … 50, l0/d 7 … 43, l0/i 28 … 174 | |
| η | 1 + (l0/h)² ζ1 ζ2 / (1300 e0/h0), ζ1 = 0.2 + 2.7 e0/h0 ≤ 1, ζ2 = 1.15 − 0.01 l0/h ≤ 1, when l0/i > 17.5; h0 = r + rs for circles | **check whether 3362-2018 adds an accidental eccentricity ea to e0** (not applied here) |
| Solid circle formula | Nu = α fcd A (1 − sin 2πα / 2πα) + (α − αt) fsd As; Nu ηe0 = ⅔ fcd A r sin³πα / π + fsd As rs (sin πα + sin παt)/π; αt = 1.25 − 2α ≥ 0 | **check the clause number and form in 3362-2018** |
| Clause numbers | as in the tables above | from memory; may be off by a sub-clause |
