# Nonlinear solids: cracking of reinforced and prestressed concrete

`femotion.continuum.opensees` runs solid models (hexahedra and tetrahedra) on OpenSees with a damage-plasticity concrete
model, so that cracks appear, grow and open under the loads, and reads out where the cracks are, which way they run and
how wide they are. Reinforcing bars, prestressing tendons, bonded or anchored strips and anchor bolts are drawn as lines
through the solid, independent of the mesh. Loads are applied in stages (self-weight and prestress, then the service or
ultimate load), by load or by displacement control.

It is meant for local, three-dimensional questions that a frame model and a code formula cannot answer: splitting and
spalling around anchorages and anchor bolts, cracking in diaphragms, deviators, corbels and openings, crack patterns of
deep members, and the crack widths that go with them.

```python
import numpy as np
from femotion.mesh import structured
from femotion.continuum.opensees import OpsSolid, ConcreteDamage, ElasticIso, steel, frp

m = structured.box(3000, 200, 300, 120, 8, 12)                       # mm; any hex8 / tet4 Mesh with cell groups
c = ConcreteDamage.from_strength(38.0, units="N-mm")                 # mean strength; MC2010 defaults for the rest
s = OpsSolid(m, c)                                                   # or {cell group: material}
s.bar([(30, 100, 40), (2970, 100, 40)], area=402.0, material=steel(450.0, 2.0e5))   # bonded bar, any polyline
s.fix(lambda x: np.isclose(x[2], 0) & (np.abs(x[0] - 100) < 30), comps=(1, 2))
s.fix(lambda x: np.isclose(x[2], 0) & (np.abs(x[0] - 2900) < 30), comps=(2,))
s.fix(lambda x: np.isclose(x[0], 1500) & np.isclose(x[2], 0), comps=(0,))   # a line across: holds the rotation about z
s.static(s.self_weight(25e-6), steps=2, name="self-weight")          # stage 1, held afterwards
F = s.traction(lambda x: np.isclose(x[2], 300) & (np.abs(x[0] - 1500) < 60), (0, 0, -1.0))
s.push(F, (1500, 100, 0), dof=2, target=-8.0, steps=120, name="load")   # stage 2, displacement control

w = s.crack_widths()                       # per cell, last step: w = ε_cr · h
r = s.cracks_along((0, 100, 1), (3000, 100, 1))   # individual cracks crossed by a line, with their widths
u, P = s.curve((1500, 100, 0), 2)          # load–deflection
s.export("out/beam.pvd")                   # ParaView series; s.to_result() → fem-result/1 for animation
```

A complete example with all the comparisons below is `examples/continuum_crack_rc_beam.py`.

## Units

Any consistent set: N, mm, MPa (fracture energies in N/mm), or kN, m, kPa (kN/m = N/mm), etc. Only the defaults of
`ConcreteDamage.from_strength` come from empirical formulas in MPa and N/mm, so it needs `units=` ("N-mm", "kN-m",
"N-m", "MN-m"). Self-weight is given as a weight per unit volume (25 kN/m³ is 2.5e-5 in N, mm).

## The model

| Part | OpenSees | Notes |
|---|---|---|
| hexahedra (hex8) | `stdBrick` (2 × 2 × 2 Gauss), or `bbarBrick` | same element as `femotion.continuum.Elastic(order=1)` |
| tetrahedra (tet4) | `FourNodeTetrahedron` | cells re-oriented to positive volume (a negative one gives a negative stiffness silently) |
| `ConcreteDamage` | `nDMaterial ASDConcrete3D` | one material per crack-band bin (below) |
| `ElasticIso`, `ElasticOrtho` | `ElasticIsotropic`, `ElasticOrthotropic` (global axes) | bearing pads, steel plates, rock, FRP blocks |
| `bar(...)` | `Truss` + `ASDEmbeddedNodeElement` | bonded: a node every `seg` (default half the median band), each tied to the solid; unbonded: nodes at the given points only |
| `bar(..., prestress=σ0)` | `InitStressMaterial` around the bar's material | tension positive; the first step balances it, so the elastic shortening (and the indentation under point anchors) is included |
| `dowel(...)` | `elasticBeamColumn` (or `forceBeamColumn` on a `femotion.frame.opensees.Fiber` section) on 6-dof nodes + `ASDEmbeddedNodeElement` | anchor bolts and studs: shear and bending reach the concrete along the embedded length; `rot=True` also ties the rotation |
| `fix`, `tie`, `spring`, `face_spring` | `fix`, `equalDOF`, `zeroLength` to fixed ground nodes | Winkler springs per unit area are lumped by tributary area |
| `static(imposed=…)` | `zeroLength` drivers + `sp` on their ground node | see "Imposed displacements" |

Uniaxial materials of bars are tuples `("OpenSeesType", *parameters)` as in `femotion.frame.opensees.Fiber`, with helpers
`steel(fy, E, b)` (Steel02), `elastic(E)`, `elastic_pp(fy, E)` and `frp(E, eps_u)` (linear, ruptures for good beyond
eps_u). Bars that start or end on a dowel node are pinned to it (`equalDOF` in translation): a prestressed strip
anchored on bolt heads. Define the dowels before those bars. A bar point outside the solid raises an error unless
`outside="free"` (it must then be held by other members).

### Concrete

`ConcreteDamage` holds its laws independently of the mesh: the tensile law as stress against crack opening w after the
peak, the compressive law as stress against a crushing displacement after the peak, and the pre-peak curves as
stress–strain. `from_strength(fc, …)` fills them from the mean compressive strength (analyses of cracking use mean
values):

| Quantity | Default | Source |
|---|---|---|
| ft = fctm | 0.3 fck^(2/3) (fck = fcm − 8 ≤ 50), else 2.12 ln(1 + 0.1 fcm) | fib Model Code 2010, eq. 5.1-3a/b |
| E = Eci | 21 500 (fcm / 10)^(1/3) | MC2010 eq. 5.1-21 (quartzite) |
| Gf | 73 fcm^0.18 N/m | MC2010 eq. 5.1-9 |
| Gc | 250 Gf | Rijkswaterstaat RTD 1016-1:2017 (after Nakamura and Higai) |
| εc1 | 0.7 fcm^0.31 ‰ ≤ 2.8 ‰ | EN 1992-1-1 table 3.1 |
| tension pre-peak | linear to 0.9 ft, then to ft at 0.15 ‰ (`prepeak="mc2010"`), or linear (`"linear"`) | MC2010 eq. 5.1-8 |
| tension softening | bilinear: 0.2 ft at w1 = Gf/ft, 0 at 5 w1 (`"mc2010"`); exponential (`"hordijk"`, Hordijk 1991); linear | MC2010 eq. 5.1-10; each scaled to an area of exactly Gf |
| compression | linear to 0.4 fc, Sargin to fc at εc1, linear to 0.1 fc over Gc, constant beyond | MC2010 eq. 5.1-26 |
| unloading | 20 % of the post-peak tensile strain and 70 % of the inelastic compressive strain stay (the rest is stiffness damage) | `plastic_t`, `plastic_c` |

For C30 (fcm = 38 MPa): ft 2.90 MPa, E 33 550 MPa, Gf 0.141 N/mm, Gc 35.1 N/mm, εc1 2.16 ‰.

ASDConcrete3D (Petracca, Camata et al., ASDEA) splits the effective stress into tensile and compressive parts by the
principal directions; each part has a damage variable driven by an equivalent strain of a Lubliner-type surface, with
a share of plasticity. `crack_planes=(nct, ncc, angle)` makes the damage directional (several crack planes per point),
`Kc` shapes the triaxial compression meridian.

## Solving

**IMPL-EX (default, `implex=True`)** (Oliver, Huespe and Cante 2008): the damage of a step is extrapolated from the two
previous steps, so each step is linear in the concrete and Newton converges in two iterations; the true state is
recovered at the commit. It never stalls when cracks spread, but its error is of the order of one step: use steps that
change the cracking little (about 100 to failure), and expect the peak of a test with a uniform stress (a tension
prism) to be overshot by about one step.

**Implicit (`implex=False`)**: secant iterations to equilibrium at every step. Exact at convergence, but convergence is
linear (Krylov–Newton helps) and stalls when many points crack in one step; steps that do not converge are retried with
other algorithms and then halved (down to 1/2⁸). Use it for small models and for checks.

IMPL-EX extrapolates with the ratio of OpenSees' pseudo-time increments, and under displacement control that increment is
the load-factor increment, which changes in every iteration: the step is then not linear and diverges. Here the
pseudo-time of every material point is set to the step size relative to the stage's nominal step (`setParameter
dTime`), and at the first step of a stage the committed time is made huge, so that a stage starts without
extrapolation. With that, `push` (displacement control) works with IMPL-EX too.

Convergence: the norm of the displacement correction below `rtol` (1e-4) × the norm of the previous step's displacement
increment, but not below 1e-10 × the model size (the round-off left by the penalty ties).

Supports: every stage first counts the rigid-body motions the supports leave free (`free_rigid_modes()`) and warns.
A free one makes the stiffness singular; a symmetric elastic model may still solve, but the round-off of the penalty
ties of bars excites the free motion and no step converges (a beam held in x only along one edge, at a single point
across, can still turn about the vertical through its bearing).

**Imposed displacements** (`static(imposed=(nodes, values))`, sub-models, stroke control through a rigid plate made with
`tie`) are applied through stiff springs from driven ground nodes. An `sp` constraint on the node itself puts the whole
increment of a step into the first layer of cells while the predictor is formed; with softening concrete this cracks
that layer and the crack stays there (found with a tension prism: the crack formed at the loaded end, not in the weak
layer). Through springs the predictor is the smooth elastic field.

## Crack band and crack width

### Regularisation

Without regularisation the energy dissipated by a smeared crack is proportional to the size of the cells it runs
through. The crack band method (Bažant and Oh 1983) makes it mesh-independent: the softening law of each cell is
converted from crack opening to strain with the width h of the band of cells the crack runs in,

  ε = σ / E + ε_pre + w / h,

so that the energy per unit crack area is ∫σ dw = Gf whatever h. Here h is

- `band="volume"` (default): ∛V for hexahedra, ∛(6V) for tetrahedra (the side of the cube the tetrahedron would be a
  sixth of);
- `band=("projected", n)`: the extent of the cell along the unit vector n (Oliver 1989), when the crack direction is
  known beforehand, for example a splitting crack along a member;
- any array (one value per cell).

Cells are grouped in bins of 1 % of h, one OpenSees material per bin. A cell larger than the snap-back limit
h_max = min E·Δw/Δσ over the softening segments (for the MC2010 law 1.25 E Gf / ft² ≈ 700 mm for C30) gets the law of
h_max and dissipates more than Gf (`OpsSolid.h_clipped` counts them, with a warning). ASDConcrete3D's own
`-autoRegularization` uses the shortest node-to-node distance of the element as the band, and its `-fc` shortcut
assumes MPa and N/mm; neither is used.

Lateral restraint of the band: ASDConcrete3D's equivalent strain is computed from the effective stress. In a band that
softens between elastic neighbours, the neighbours prevent its lateral contraction, so the band carries lateral
effective tension and its equivalent strain grows faster than its axial strain. With ν = 0.2 a band dissipates about
6 % less than Gf (0.933 Gf in the tension prism, the same for every cell size); with ν = 0 it is Gf to 0.2 %. Raise Gf by
that amount if the energy matters more than the elastic Poisson effect.

### Crack strain, direction and width

The crack strain is read from stresses and strains at the integration points, not from internal variables, so it is the
same for any material: the inelastic strain tensor

  ε_in = ε − C₀⁻¹ σ

(C₀ the initial elasticity of the cell's material), averaged over the cell. Its largest principal value minus the
pre-peak part ε_pre is the crack opening strain ε_cr, its direction the crack normal n. A cell counts as cracked once the
tensile equivalent strain of the material (`equivalentTotalStrain`) has passed the peak; elsewhere ε_cr = 0 (so the
Poisson expansion of crushed concrete is not reported as cracks). Per cell

  w = ε_cr · h,

the opening of the band the law was regularised for (`crack_widths()`, field `crack_width`).

Along a line (`cracks_along(a, b)` or a polyline), the profile of ε_cr (n·t)² is integrated cell by cell (the cell
boundaries are found by bisection, so each cell counts with its exact length); the integral over one crack band is the
opening of that crack projected on the line. The profile is cut into individual cracks at uncracked cells and at valleys
lower than half the smaller neighbouring peak; a valley cell is shared between the two cracks. `surface_cracks(where)`
gives the widths of the cells under selected boundary faces (a crack map of the soffit, say).

### What the width means, and what it does not

- **Smeared, not discrete**: a crack is a band of one or two cells. Two real cracks closer than about 2 h merge into one
  band, and the width is the mean opening over the band, not the width at the surface of one crack.
- **Perfect bond**: bars are tied to the concrete without slip, so there is no transfer length and no tension-stiffening
  law: crack spacing follows from the mesh and the stress field, and the bar diameter does not enter. Crack widths are
  therefore of the same order as code formulas, not predictions to 0.01 mm. In the beam below, 25 and 50 mm cells give
  the same first-crack moment (13.5–14.3 vs 13.6–14.5 kN·m), crack spacing (95 vs 100–120 mm) and largest widths
  (0.20–0.23 vs 0.18–0.25 mm at 3–4.5 mm deflection), and both stay below the characteristic values of the codes, as
  mean cracks should.
- **Mesh**: crack paths follow the mesh; inclined cracks in a structured mesh zig-zag. Refine where cracks are expected
  and keep cells below h_max.
- **Time-dependent effects** (creep, shrinkage, fatigue) are not modelled; long-term widths need the code's long-term
  factors.

## Anchors, strips and local models

A prestressed strip anchored by bolts, for example, is three members:

```python
s = OpsSolid(mesh, concrete, band=("projected", (0, 1, 0)))   # splitting cracks along x: band across the member
a1 = s.dowel([(0.3, 0.2, -0.01), (0.3, 0.2, 0.15)], d=0.016, E=2e8)   # bolt: head 10 mm below the soffit, 150 mm deep
a2 = s.dowel([(1.7, 0.2, -0.01), (1.7, 0.2, 0.15)], d=0.016, E=2e8)
s.bar([(0.3, 0.2, -0.01), (1.7, 0.2, -0.01)], area=70e-6, material=frp(1.65e8, 0.016),
      prestress=1.1e6, bond=False, outside="free")          # strip pinned to the bolt heads
s.static(None, steps=5)                                      # the prestress is balanced in the first step
s.dowel_results(a1)                                          # shear and moment along the bolt
```

The bolt is a line of beam elements tied to the solid at its nodes: the bearing pressure over the bolt's diameter is not
resolved, and the cells right next to a bolt crack first, with widths that depend on the mesh (0.34 mm with 50 mm cells,
0.69 mm with 25 mm cells across the member in this example). Read the cracks a few cells away from the bolts (the
splitting crack along the member, say), compare two meshes, or model the bolt and its hole as solids where the contact
itself matters.

## Sub-models

`submodel_boundary(global_mesh, U, local, where)` interpolates the displacements U of a global model (any hex8 / tet4
mesh, for example `Elastic(...).nodal_displacement()` on `result_mesh()`) at the cut-boundary nodes of a local model,
ready for `local.static(imposed=…)`; `interpolate(mesh, U, points)` is the general version. Boundary forces instead of
displacements are ordinary nodal loads (`nodal_force`, `point_force`).

## Results

Every converged step keeps the displacements of all nodes (`Step.U`, solid nodes then bar and dowel nodes) and the
reactions of the supported nodes (`Step.R`, `Step.Rn`: fixed components, springs, displacement drivers). Recorded steps
(`record=True`, every k-th, or only the last) also keep, per cell: `stress`, `strain` (engineering shear; order xx yy zz
xy yz zx), `eps_in`, `crack_strain`, `crack_normal`, `width`, `cracked`, `crushed`, `eq_strain` (material equivalent
strains); per bar segment `N`, `stress`, `strain`; per dowel element `N`, `V`, `M`, `local`.

- `curve(node, dof, at=…)`: load–displacement (−1 × the reaction resultant at the supports selected by `at`, i.e. the
  applied load with its sign; `sign=1` gives the reaction);
  `reaction(at)`; `first_crack()`; `bar_results(k)`, `dowel_results(k)` along one member.
- `width_history()` or `width_history(a, b)`: the largest crack width per recorded step (any cell, or the cracks along a
  line), to interpolate the load at a width limit; `stop=lambda step: step.cell["width"].max() > 0.2` ends a stage
  there.
- `to_result()`: fem-result/1 (solids plus bars and dowels as line2 cells, one frame per recorded step);
  `export("x.pvd")`: a VTU per step and a ParaView collection; `save_npz()`: everything as NumPy arrays.

## Validation

All in `tests/test_continuum_opensees.py`.

| Check | Reference | Result |
|---|---|---|
| cantilever below cracking, hex8 and tet4, `ElasticIso` and `ConcreteDamage` | `femotion.continuum.Elastic(order=1)` (scikit-fem), same mesh | displacements 4e-13 (hex), 3e-12 (tet); cell stresses 5e-13 / 9e-7 (ASDConcrete3D's spectral split) |
| uniaxial tension prism, ν = 0, h = 100 and 50 mm (implicit) | input law: area Gf | dissipated energy 0.999 Gf and 1.002 Gf; peak within one load step of ft |
| same, ν = 0.2 | mesh objectivity | 0.933 Gf and 0.934 Gf (lateral restraint of the band, see above) |
| same, IMPL-EX | implicit | 1.03 Gf (all layers pass the peak by one step in a uniform field); same crack |
| same with tetrahedra (cubes cut in six), band ∛(6V) | input law | 0.999 Gf and 1.002 Gf (ν = 0), 0.933 Gf (ν = 0.2); width = end displacement |
| separated prism | end displacement less the elastic stretch | crack width along a line 0.29997 mm vs 0.29997 mm; cell width 0.2995 mm |
| bonded bar on and off mesh nodes | EcAg + EsAs (the bar overlaps the concrete) | −7e-8 and −9e-8; bar force uniform to 2e-6 |
| bonded prestressing tendon (InitStressMaterial) | −σ0 Ap / (Ag + n Ap), σ0 Ap (1 − n Ap / (Ag + n Ap)) | 4e-7 and 5e-7 away from the anchors |
| unbonded eccentric tendon anchored at nodes | scikit-fem under the two anchor forces (equivalent loads) | cell stresses equal to 7e-14; the tendon lost 7.8 % (shortening and indentation under the point anchors) |
| anchor bolt in a rigid block, rot=True | clamped cantilever Pa³/3EI | 1 + 1e-11; shear P and moment Pa at the surface |
| prestressed strip on two bolts | σ = σ0 + E ε of the strip from its end displacements; equilibrium | to 1e-6; head shear = strip force; reactions 0 |
| sub-model cut from a global linear model | the global field inside | 1.5e-8 |
| RC beam, linear pre-peak: first crack | ft Ig / (yb − zgp), transformed section at the bottom Gauss points: 10.24 kN·m (50 mm cells), 9.85 kN·m (25 mm); ft·W = 9.50 kN·m | 10.04 … 10.66 kN·m and 9.25 … 9.91 kN·m (bracketing load steps) |
| RC beam, MC2010 pre-peak: first crack | MC2010 flexural tensile strength fctm,fl = 1.31 fctm (h = 300) | 1.33 … 1.41 × ft Ig / (yb − zgp) |
| RC beam, load–deflection | uncracked and fully cracked beam (bending only), Branson's effective I in between | between the bounds at every step; 10–20 % stiffer than Branson in service (perfect bond: more tension stiffening), within 3 % at 0.8 fy |
| RC beam, steel stress in the constant-moment zone | cracked section n M (d − x) / Icr | 0.85 … 0.96 of it once cracked, rising with the load (concrete between cracks still in tension) |
| RC beam, crack widths at σs = 270 … 380 MPa (0.87 lever arm) | JTG 3362-2018 6.4.3 Wcr (via `femotion.codes.jtg.sls.crack_width`); Eurocode 2 7.3.4 wk | FE largest crack 0.18 … 0.25 mm vs Wcr 0.24 … 0.34, wk 0.25 … 0.38: FE/Wcr 0.69 … 0.83 (mean cracks against characteristic values); spacing 100 … 120 mm (EC2 sr,max / 1.7 = 143 mm) |

The RC beam: 200 × 300 mm, span 2400, loads at a = 900, 2 Ø16 at 40 mm, fcm 38 MPa, quarter model, 50 mm hexahedra,
line bearing, elastic pads under the bearing and the loading plate. Supporting the beam on a strip of node lines instead
of a line clamps its end: in the elastic range the 25 mm model was then 22 % stiffer than beam theory with shear, against
0.3 % on a line bearing.

## Performance

Apple M1 Max, OpenSeesPy 3.8 (UmfPack), IMPL-EX, two Newton iterations per step:

| Model | Cells | Steps | Time |
|---|---|---|---|
| RC beam, 50 mm | 312 + 52 bar segments | 100 | 13 s |
| RC beam, 25 mm | 2 496 + 104 | 100 | 179 s (`numberer="RCM"`; 199 s with AMD) |
| member with two anchor bolts and a prestressed strip, 50 mm | 3 840 + 2 dowels | 5 | 10 s |
| the same, 50 × 25 × 25 mm | 15 360 (52 000 equations) | 10 | 247 s (AMD; 352 s with RCM) |

The material update (eight points per hexahedron, about 13 µs each) and the sparse factorisation (UmfPack; the
equation ordering `numberer` matters: AMD for bulky blocks, RCM for long slender members) share the time; IMPL-EX needs two to three iterations per step. Reading the
results (two element responses per cell and one per integration point) adds about 0.2 s per step for 4 000 cells.

## OpenSees notes (3.8)

- `ASDEmbeddedNodeElement` takes the fourth retained node as an optional argument parsed from a string: passed as a
  Python int it is silently dropped and the node is tied to the triangle of the first three nodes (a wrong constraint,
  no warning). Pass `str(node)`.
- Its default penalty 1e18·∛V leaves round-off of about 1e-5 in the displacements and stalls convergence tests; here it
  is 1e6 × the largest Young's modulus (stiffness error ~3e-8).
- It ties a node to a tetrahedron with linear shape functions; inside a hexahedron the node is tied to the one of its six
  sub-tetrahedra (around the 0–6 diagonal) that contains it. Constant-strain states are exact; in bending the tie is
  slightly non-conforming.
- `geomTransf` and frame elements need the 6-dof model builder (`model -ndf 6`) at the time they are created; loads on
  6-dof nodes need six components.
- ASDConcrete3D's responses `crackWidth` and `crackPattern` multiply by the element's shortest node distance; here the
  width is computed from stresses and strains and the band of the law instead.
- IMPL-EX under displacement control: see "Solving".

## Licence

OpenSees is free for education, research and non-profit use; commercial use needs a licence from the Regents of the
University of California. ASDConcrete3D and ASDEmbeddedNodeElement are part of OpenSees. OpenSeesPy is installed from
its own distribution (extra `opensees`), not bundled.

## References

- Bažant, Z. P., Oh, B. H. (1983). Crack band theory for fracture of concrete. Matériaux et Constructions 16, 155–177.
- Oliver, J. (1989). A consistent characteristic length for smeared cracking models. IJNME 28, 461–474.
- Oliver, J., Huespe, A. E., Cante, J. C. (2008). An implicit/explicit integration scheme to increase computability of
  non-linear material and contact/friction problems. CMAME 197, 1865–1889.
- Hordijk, D. A. (1991). Local approach to fatigue of concrete. PhD thesis, Delft University of Technology.
- fib Model Code for Concrete Structures 2010 (2013), sections 5.1 and 7.6.
- Hendriks, M. A. N. et al. (2017). Guidelines for nonlinear finite element analysis of concrete structures,
  Rijkswaterstaat RTD 1016-1:2017.
- Petracca, M., Camata, G. et al. ASDConcrete3D, OpenSees documentation.
- JTG 3362-2018, 6.4.3; EN 1992-1-1:2004, 7.3.4.
