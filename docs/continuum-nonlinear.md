# Nonlinear solids: cracking of reinforced and prestressed concrete

`femotion.continuum.opensees` runs solid models (hexahedra and tetrahedra) on OpenSees with a damage-plasticity concrete
model, so that cracks appear, grow and open under the loads, and reads out where the cracks are, which way they run and
how wide they are. Reinforcing bars, prestressing tendons, bonded or anchored strips and anchor bolts are drawn as lines
through the solid, independent of the mesh. Loads are applied in stages (self-weight and prestress, then the service or
ultimate load), by load or by displacement control.

It is meant for local, three-dimensional questions that a frame model and a code formula cannot answer: splitting and
spalling around anchorages and anchor bolts, cracking in diaphragms, deviators, corbels and openings, crack patterns of
deep members, cracks from restrained shrinkage and temperature, the cover split by rusting bars, and the crack widths
that go with them.

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
| `ElasticIso`, `ElasticOrtho` (or `femotion.continuum.Isotropic`, `Orthotropic` with the global axes) | `ElasticIsotropic`, `ElasticOrthotropic` (global axes) | bearing pads, steel plates, rock, FRP blocks; Poisson's ratios below |
| `bar(...)` | `Truss` + ties to the solid (below) | bonded: a node every `seg` (default half the median band), each tied to the solid (a run outside the solid stays one bar); unbonded: attached at the given points only |
| `bar(..., bond=False)` | one `Truss` per span; deviators: `zeroLength` across the tendon | the end points are anchored; at the points in between the tendon slides without friction (one force along the whole tendon); `slide=False` anchors it there too |
| `bar(..., prestress=σ0)` | `InitStressMaterial` around the bar's material | tension positive, in full from the first step and balanced by the solid (elastic shortening, and the indentation under point anchors, included): a pretensioned bonded tendon or an unbonded one; a post-tensioned tendon grouted afterwards is not modelled |
| `dowel(...)` | `elasticBeamColumn` (or `forceBeamColumn` on a `femotion.frame.opensees.Fiber` section) on 6-dof nodes, tied to the solid | anchor bolts and studs: shear and bending reach the concrete along the embedded length; their twist is held by a torsional spring at the first embedded node; `rot=True` also ties the rotation |
| `bar(..., bond=BondSlip(...))` | `Truss` on its own nodes + a `zeroLength` per node to an anchor point tied to the solid | bond–slip law along the bar (`HystereticSM`, or `ElasticPP` for a rigid-plastic law), stiff across it; see "Bond-slip" |
| every solid and bar material | wrapped in `nDMaterial InitStrain` / `uniaxialMaterial InitStrainMaterial` | free strains (shrinkage, temperature, rust): σ = f(ε − ε0), see "Free strains" |
| `fix`, `tie`, `spring`, `face_spring` | `fix`, `equalDOF`, `zeroLength` to fixed ground nodes | Winkler springs per unit area are lumped by tributary area |
| `static(imposed=…)` | `zeroLength` drivers + `sp` on their ground node | see "Imposed displacements" |

A mesh may mix hexahedra and tetrahedra (cell results are in that order), but boundary-face loads and supports
(`traction`, `pressure`, `face_spring`, `surface_cracks`) need one kind.

**How bars are tied to the solid.** A bar node follows the interpolation of the cell it lies in, exactly: linear in a
tetrahedron (one `ASDEmbeddedNodeElement`), trilinear in a hexahedron. OpenSees has no 8-node tie, so the trilinear
value u = Σ Nᵢ(ξ, η, ζ) uᵢ is built from 2-point linear interpolations nested along the cell's edges — four auxiliary
nodes on the edges along ξ, two between them along η, the bar node between those along ζ — each an
`ASDEmbeddedNodeElement` on a tetrahedron edge (the other two corners get weight 0, the tetrahedron is ordered to a
positive volume). In pure bending a bar then has the strain of the solid at its level everywhere (tying it to one of
the six sub-tetrahedra instead, as first released, made the segment strains alternate by ±17 … 33 % around the right
mean). Dowel nodes with `rot=True` are tied, rotation included, to the sub-tetrahedron that holds them (linear).

`ElasticOrtho(E, nu, G)` takes OpenSees' parameters: ν = (νxy, νyz, νzx), νij the contraction along j under stress
along i, so νzx = ν13·E3/E1 in the notation of `femotion.continuum.Orthotropic` (ν = (ν12, ν13, ν23), G = (G12, G13,
G23)); an `Orthotropic` passed directly is converted (its axes must be the global ones).

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
| unloading | 20 % of the post-peak tensile strain and 70 % of the inelastic compressive strain stay (the rest is stiffness damage), on every branch of the law (the damage of the last points may reach 1 − 1e-9) | `plastic_t`, `plastic_c` |

For C30 (fcm = 38 MPa): ft 2.90 MPa, E 33 550 MPa, Gf 0.141 N/mm, Gc 35.1 N/mm, εc1 2.16 ‰.

ASDConcrete3D (Petracca, Camata et al., ASDEA) splits the effective stress into tensile and compressive parts by the
principal directions; each part has a damage variable driven by an equivalent strain of a Lubliner-type surface, with
a share of plasticity. `crack_planes=(nct, ncc, angle)` makes the damage directional (several crack planes per point),
`Kc` shapes the triaxial compression meridian.

## Solving

**IMPL-EX (default, `implex=True`)** (Oliver, Huespe and Cante 2008): the damage of a step is extrapolated from the two
previous steps, so each step is linear in the concrete and Newton converges in two iterations; the true state is
recovered at the commit. It never stalls when cracks spread, but its error is of the order of one step: use steps that
change the cracking little (about 100 to failure). In a uniform stress field (a tension prism) every layer passes the
peak by a step before one localises, which costs 8–10 % more energy than the implicit solution; in a non-uniform one
(the notched beam below) the difference is about 1 %.

**Stages without a load increment** (`static(None)`, zero loads: holds) and **prestress**. Extrapolation is wrong when
the damage of the last step was not a rate: the prestress of `InitStressMaterial` enters in full in the first step.
Under IMPL-EX, therefore:
- before the first stage, prestressed bars are settled in a stage of their own, `"prestress"`: steps at λ = 0 without
  extrapolation — a fixed-point iteration on the damage, each step a secant solve with the damage of the last — until a
  step changes the displacements by less than `hold_tol` (1e-4) of their norm (at most `hold_steps` = 200);
- every step of a hold stage is taken the same way, and the stage ends once it has settled (`Stage.message`,
  `Stage.change`; a warning when `steps` were not enough).

So a prestress-only stage gives the implicit result whatever `steps` is (anchored strip below: crack width within 4 %,
strip force within 0.01 %; first released, extrapolating, it gave −89 % with 1 step and +23 % with 5). Implicitly
(`implex=False`) a hold is one Newton solve.

**Implicit (`implex=False`)**: secant iterations to equilibrium at every step. Exact at convergence, but convergence is
linear (Krylov–Newton helps) and stalls when many points crack in one step; steps that do not converge are retried with
other algorithms and then halved (down to 1/2⁶). Use it for small models and for checks.

IMPL-EX extrapolates with the ratio of OpenSees' pseudo-time increments, and under displacement control that increment is
the load-factor increment, which changes in every iteration: the step is then not linear and diverges. Here the
pseudo-time of every material point is set to the step size relative to the stage's nominal step (`setParameter
dTime`, before every attempt, since a failed attempt reverts it), and at the first step of a stage (and in holds) the
committed time is made huge, so that the step starts without extrapolation. With that, `push` (displacement control)
works with IMPL-EX too.

Convergence: the norm of the displacement correction below `rtol` (1e-4) × the norm of the previous step's displacement
increment, but not below 1e-10 × the model size (the round-off left by the penalty ties).

Supports: every stage first counts the rigid-body motions the supports leave free (`free_rigid_modes()`) and the bar
nodes held in fewer than three directions (`loose_bar_nodes()`: not tied to the solid, not on a dowel or a support,
with bars in line or a single bar — a free end outside the solid), and warns. Either makes the stiffness singular; a
symmetric elastic model may still solve, but the round-off of the penalty ties excites the free motion and no step
converges (a beam held in x only along one edge can still turn about the vertical through its bearing). The checks
cover the whole model and these nodes, not mechanisms inside it (a part joined to the rest by nothing).

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
  sixth of). Right for cells of about equal sides only: in a cell twice as long across the crack as along its normal
  the crack dissipates 63 % of Gf (test below), and a 10 × 100 × 10 mm cell (one cell through the thickness) halves the
  softening branch of a notched beam (independent review). OpsSolid warns when concrete hexahedra are elongated by
  more than 1.5 (tetrahedra: edges by more than 3);
- `band=("projected", n)`: the extent of the cell along the crack normal n (Oliver 1989) — one unit vector, or one per
  cell (n_cells, 3) where the expected direction varies (a splitting crack along a member, flexural cracks across it).
  ASDConcrete3D fixes the law of a point before it cracks, so the band cannot follow the crack as it forms: give the
  direction expected, from the stress field of a linear run if need be;
- any array (one value per cell).

Cells are grouped in bins of 1 % of h, one OpenSees material per bin. A cell larger than the snap-back limit
h_max = min E·Δw/Δσ over the softening segments (for the MC2010 law 1.25 E Gf / ft² ≈ 700 mm for C30) gets the law of
h_max and dissipates more than Gf (`OpsSolid.h_clipped` counts them, with a warning). ASDConcrete3D's own
`-autoRegularization` uses the shortest node-to-node distance of the element as the band, and its `-fc` shortcut
assumes MPa and N/mm; neither is used.

Lateral restraint of the band: ASDConcrete3D's equivalent strain is computed from the effective stress. In a band that
softens between elastic neighbours, the neighbours prevent its lateral contraction, so the band carries lateral
effective tension and its equivalent strain grows faster than its axial strain. With ν = 0.2 a band dissipates about
6 % less than Gf (0.933 Gf in the tension prism, the same for every cell size); with ν = 0 it is Gf to 0.2 %. Where the
whole band's energy should be Gf, give `Gf=Gf_target / 0.933` for ν = 0.2. In bending (the notched beam below) the
cells next to the crack band crack a little too and make up for it.

### Crack strain, direction and width

The crack strain is read from stresses and strains at the integration points, not from internal variables, so it is the
same for any material: the inelastic strain tensor

  ε_in = ε − C₀⁻¹ σ

(C₀ the initial elasticity of the cell's material), averaged over the cell (and thereby closing on unloading: a crack
opened to w and unloaded keeps plastic_t·w, 0.2 w by default). Its largest principal value minus the
pre-peak part ε_pre is the crack opening strain ε_cr, its direction the crack normal n. A cell counts as cracked once the
tensile equivalent strain of the material (`equivalentTotalStrain`) has passed the peak; elsewhere ε_cr = 0 (so the
Poisson expansion of crushed concrete is not reported as cracks). Per cell

  w = ε_cr · h,

the opening of the band the law was regularised for (`crack_widths()`, field `crack_width`).

Along a line (`cracks_along(a, b)` or a polyline), the profile of ε_cr (n·t)² is integrated cell by cell (the cell
boundaries are found by bisection, so each cell counts with its exact length); the integral over one crack band is the
opening of that crack projected on the line. The profile is cut into individual cracks at uncracked cells and at valleys
lower than half the smaller neighbouring peak; a valley cell is shared between the two cracks. The line geometry (cells,
boundaries) is computed once per line and reused for every step. `surface_cracks(where)` gives the widths of the cells
under selected boundary faces (a crack map of the soffit, say). Two readings differ where the band of the law differs
from the cell's extent along the line: the cell width ε_cr·h uses the band of the law (h_max for clipped cells, ∛V for
non-cubic cells with `band="volume"`), the line integral the kinematic length.

### What the width means, and what it does not

- **Smeared, not discrete**: a crack is a band of one or two cells. Two real cracks closer than about 2 h merge into one
  band, and the width is the mean opening over the band, not the width at the surface of one crack.
- **Perfect bond** (the default `bond=True`): bars are tied to the concrete without slip, so there is no transfer length
  and no tension-stiffening law: crack spacing follows from the mesh and the stress field, and the bar diameter does not
  enter. With a bond-slip law (`bond=BondSlip(...)`, below) the spacing follows from the transfer length instead. Crack widths are
  therefore of the size of code formulas, not predictions to 0.01 mm. In the beam below, compared on the same
  (short-term) basis as the analysis, the largest FE crack is 0.90–1.09 × the short-term maximum width of JTG 3362-2018
  6.4.3 and 0.73–0.88 × the short-term wk of Eurocode 2 once cracking has stabilised; the independent review found the
  surface widths of a reinforced tie between the mean and the maximum widths of fib MC2010 7.6. Individual crack
  positions and spacings (100–130 mm here, about half of MC2010's and EC2's mean spacing in the tie) follow the mesh:
  read them as an illustration.
- **Mesh**: crack paths follow the mesh; inclined cracks in a structured mesh zig-zag. Refine where cracks are expected
  and keep cells below h_max.
- **Time-dependent effects**: shrinkage and temperature enter as free strains of a stage (below), with the value the
  user gives; creep and fatigue are not modelled, so long-term widths under sustained load need the code's long-term
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
resolved, and the cells right next to a bolt crack first, with widths that depend on the mesh (in this example, exactly
as written on a 2.0 × 0.4 × 0.6 m member: largest width 0.089 mm, 32 cracked cells, strip force 59.9 kN with 50 mm
cells; 0.215 mm, 100 cracked cells, 58.0 kN with 50 × 25 × 25 mm cells). Read the cracks a few cells away from the
bolts (the splitting crack along the member, say), compare two meshes, or model the bolt and its hole as solids where
the contact itself matters. The prestress is settled before the stage (see "Solving"), so `steps` does not change the
result.

## Sub-models

`local.boundary_from(source, where)` takes the displacements of a global model at the cut-boundary nodes of a local
one, ready for `local.static(imposed=…)`. The source is best a `femotion.continuum` result:

```python
r = global_model.result("自重")                      # CaseResult of femotion.continuum.Elastic (any order)
local.static(imposed=local.boundary_from(r, cut))    # its own shape functions: quadratic models keep their accuracy
```

A callable u(points) → (n, 3) works too, and so does a hex8 / tet4 mesh with nodal displacements
(`boundary_from(mesh, cut, U)`, linear between its nodes): that loses the accuracy of quadratic global models, whose
`result_mesh()` has the corner nodes only (3.6 % off on a coarse quadratic cantilever). `submodel_boundary(source, U,
local, where)` is the same as a function; `interpolate(mesh, U, points)` interpolates a nodal field at any points.
Boundary forces instead of displacements are ordinary nodal loads (`nodal_force`, `point_force`).

## Free strains: shrinkage, temperature, rust

A free strain ε0 (eigenstrain) is a strain the material would take without stress: shrinkage, a temperature change,
swelling, the growth of rust. Restrained, it stresses the member and cracks it; free, it only moves it.

```python
sh = s.shrinkage(3e-4, "slab")                        # ε_sh > 0 shortens; cells by group name(s), predicate, mask, ids
dif = s.shrinkage(4e-4, "bottom slab") + s.shrinkage(2e-4, ["webs", "top slab"])   # differential shrinkage
th = s.thermal(lambda x: 15.0 * x[2] / 0.25)          # ΔT(x): every cell by its material's α, every bar by its own
e0 = s.initial_strain([2e-4, 0, 0, 0, 0, 0], "web")   # a number (isotropic), 6 Voigt components, a 3 × 3 tensor, or eps0(x)
s.static(strain=sh, steps=40, name="shrinkage")       # ramped with the load factor λ of the stage, held afterwards
s.static(F, steps=40, strain=th + 0.5 * e0)           # with loads in the same stage; Eigenstrain objects add and scale
s.static(strain=-th, steps=20)                        # taken back
```

Functions of the position take the coordinates of the integration points, x of shape (3, m, 8), and return a value per
point ((m, 8)), 6 Voigt components ((6, m, 8), engineering shear, order xx yy zz xy yz zx) or a tensor ((3, 3, m, 8)) —
the convention of `femotion.continuum` `LoadCase.initial_strain`, so the same function drives both. `at="centroid"`
evaluates once per cell (always with `bbarBrick`, whose averaged volumetric strain needs a free strain constant over the
cell). Coefficients of thermal expansion: `ElasticIso(alpha=)`, `ElasticOrtho(alpha=(αx, αy, αz))`,
`ConcreteDamage.alpha` (1e-5 per degree by default, JTG D60-2015 4.3.12 and EN 1992-1-1 3.1.3), `bar(..., alpha=)`
(0 by default: give 1.2e-5 for steel). `thermal` warns when heated cells or bars have α = 0 (a steel core of
`ElasticIso` left at α = 0 would not expand and would restrain the concrete around it) and raises when no selected cell
has one. Which bars a `thermal` heats: with `cells` given, only the segments whose midpoint lies in those cells
(`bars=True`), the bars listed (`bars=[k, …]`), or none (`bars=False`). Dowels (anchor bolts) do not expand.
`shrinkage` takes the concrete (`ConcreteDamage`) cells by default — not a steel core, a rust ring or a bearing pad;
`initial_strain` takes all cells unless told otherwise. `bar_strain(bars, eps0)` gives a free strain along bars alone.
An `Eigenstrain` remembers the size of the model it was made for and a stage of another model refuses it.

**How.** Every solid material is wrapped in OpenSees' `nDMaterial InitStrain` (class InitStrainNDMaterial, OpenSees ≥
3.7; the interpreter knows it as "InitStrain", not "InitStrainNDMaterial"), which passes ε + ε_init to the wrapped
material. With ε_init = −ε0 the material sees ε − ε0: cracked or crushed, it answers a free strain exactly as it
answers a load, and there is no approximation to quantify. The value is an OpenSees parameter (`eps0`, `eps0_11` … of
each material point): before every attempt of a step it is set to start + λ·increment for the λ that step aims at, so a
halved step sets the halved value and a failed attempt leaves nothing behind (the stage ends by setting the committed λ
again). Material points that share a (start, increment) pair share one parameter: a uniform shrinkage of a group costs
one update per step, a field that varies from point to point one per point (about 0.2 μs each; creating them costs
about 1.5 μs per point and stage). Hold stages and `push` stages keep the free strain as it is; a stage stopped early by
`stop` keeps the share it reached.

Bars: OpenSees' uniaxial `InitStrainMaterial` commits the material it wraps whenever its parameter changes, at the
strain of the last step plus the new free strain — a state off the path; a yielding steel would keep it in its history
(30 % off with one step, 2 % with twenty in an independent review). So the free strain of a bar is put into a spring in
series with each of its segments instead: segment i–j becomes a `Truss` i–m and a `zeroLength` m–j (m at j) whose
material along the bar is `InitStrainMaterial` around a stiff `Elastic` (10⁶ × the segment's EA/L), so its imposed
elongation ε0·L is the free strain and its commit is harmless; the bar's own material only ever sees converged states.
Only bars that may get a free strain have the springs: `bar(..., free_strain=True)`, or any bar with alpha ≠ 0, or a
bar given a `bar_strain` before the first analysis (after it, a bar without them raises).

**Why not equivalent loads.** ∫Bᵀ D ε0 as nodal loads (what `femotion.continuum.Elastic` does, exact for linear
materials) leaves the material seeing f(ε) and corrects the stress afterwards, by −D ε0. A member held at both ends
that shrinks then gets no strain, no material stress and no crack at all — the restraint stress exists only in
post-processing. Restraint cracking needs σ = f(ε − ε0) inside the material, which the wrapper gives.

**Results.** `strain` is the strain the material sees, ε − ε0, for cells and bars alike (so that the inelastic strain
and the crack strain keep their meaning); `eps0` is ε0, and the strain of the displacements is their sum. `to_result()`
adds a cell field `eps0` when there is one.

**Creep.** Free strains act on the short-term moduli. Restrained shrinkage and cooling build up over months while creep
relaxes the restraint stress, so a short-term analysis overestimates them and the cracks they make; use an effective
modulus E / (1 + φ) (or the shrinkage reduced accordingly) for long-term restraint.

### Bond-slip

`bar(points, area, material, bond=BondSlip(...))` gives the bar its own nodes (every `seg`, as a bonded bar), each joined
to an anchor point of the solid at the same place by a `zeroLength` spring: along the bar the bond law times the
perimeter (default that of a round bar of the given area, or `perimeter=`) times the node's bonded length (only the
parts of the neighbouring half segments inside the solid), across it stiff. Laws:

| Law | OpenSees | Use |
|---|---|---|
| `BondSlip(s, tau)`: up to 7 points of the envelope, the same both ways | `HystereticSM` (Mazzoni 2023) | any measured or code law |
| `BondSlip.mc2010(fcm, failure="pull-out" / "splitting", good=, confined=, clear_rib=)` | `HystereticSM` | fib Model Code 2010, 6.1.1, table 6.1-1, ribbed bars before yielding: τ = τmax (s/s1)^0.4 up to s1 (five points), τmax to s2, linear to τbf at s3. **Pull-out** (the bar pulls out of well-confined concrete: cover ≥ 5 φ and clear spacing ≥ 10 φ, or heavy confinement): τmax = 2.5 √fcm (good bond) / 1.25 √fcm, s1 = 1.0 / 1.8 mm, s2 = 2.0 / 3.6 mm, s3 = clear rib spacing, τbf = 0.4 τmax. **Splitting** (the cover splits first: smaller covers and spacings — most slabs): τmax = 7.0 / 8.0 (fcm/20)^0.25 for good bond without / with stirrups, 5.0 / 5.5 otherwise, s1 = s2 = the pull-out slip at that τmax, s3 = 1.2 s1 and τbf = 0 without stirrups, s3 = 0.5 × clear rib spacing and τbf = 0.4 τmax with. `clear_rib` (the clear distance between the bar's ribs) must be given where s3 needs it |
| `BondSlip.mc2010_plain(fcm, kind="hot-rolled" / "cold-drawn", good=)` | `HystereticSM` | MC2010 table 6.1-2, smooth (plain) bars — the plain round bars of older bridges; see below |
| `BondSlip.rigid_plastic(tau, s_el)` | `ElasticPP` | constant bond stress once the slip exceeds s_el: the tension chord model (Marti et al. 1998: τb0 = 2 fct) |

`bar_results(k)["bond"]` gives slip and bond stress at the bar's nodes. With slip, cracks space themselves by the
transfer length — the restrained tie below lands in the range of the tension chord model — instead of by the mesh.

### Rust expanding around a bar

Corroding steel turns into rust of 2 to 6 times its volume (Fe3O4 2.08, α-FeOOH 2.91, Fe(OH)2 3.75, Fe(OH)3 4.2,
Fe(OH)3·3H2O 6.4; Caré et al. 2008, as tabulated by Noubactep et al. 2012), which pushes the concrete outwards until the cover splits along the bar. Modelled
here with the bar resolved by the mesh and a thin rust ring at its surface:

```python
from femotion.mesh.inclusions import bar_block, hoop_normals
from femotion.continuum.corrosion import rust_strain, depth_for_expansion
m = bar_block([0.0, 10.0], 300, 80, [(150, 39, 12)], h=6, ring=0.5)   # a slice; groups concrete, steel, rust, rust0 …
s = OpsSolid(m, {"concrete": c, "steel": ElasticIso(2e5, 0.3), "rust": ElasticIso(2e5, 0.3)},
             band=("projected", hoop_normals(m, [(150, 39, 12)])))           # radial cracks: hoop direction
s.static(strain=s.initial_strain(rust_strain((0, 150, 39), (1, 0, 0), 5.5, 0.5, 0.02), "rust0"), steps=100)
x = depth_for_expansion(0.02, 12, n=3.0)                                       # corrosion depth for δ = 0.02 mm
```

- **Mesh** (`femotion.mesh.inclusions`): `bar_block(xs, width, height, bars, h=, ring=)` sweeps a rectangular section
  with round bars along x; each bar sits in a square tile meshed as an O-grid (core square, rings following the circle,
  layers growing outwards to the tile), the tiles fit into a tensor grid of plain blocks. Tiles are as large as fits
  (to the nearest edge, or half way to the next bar), so the cover is meshed radially; `m` cells per tile side (4m
  around the bar). Bars in one row share its height and bars in one column its position; staggered tiles raise.
  `cylinder(xs, d, R, ring=)` is the thick-walled cylinder of the classical models; `hoop_normals` the hoop direction per
  cell, the crack direction of radial cracks for `band=("projected", n)`.
- **Free strain** (`femotion.continuum.corrosion.rust_strain`): the strain of the displacement u = δ (ρ − r_in)/t in the
  ring (radial δ/t, hoop δ (ρ − r_in)/(t ρ)). Being compatible it leaves an unrestrained ring without stress, and its
  outer surface moves out by δ, the free expansion of the rust. The ring's material is the rust's stiffness under the
  pressure of the concrete: as stiff as steel it is rigid in compression (the upper bound of the push, the usual
  assumption of analytical models), but it also carries hoop tension across the radial cracks, which real rust does
  not (about 7–9 % more pressure in the cylinder below, the same expansion at cracking — an independent review; a
  no-tension ring, e.g. `ConcreteDamage` with a tiny ft, removes it); softer, it gives way. A free strain is driven
  like a displacement through the ring's stiffness, so the stage follows the drop of pressure once the cover cracks — a
  pressure on the hole would snap through there.
- **Pressure** (`interface_pressure`): read the pressure of the rust on the concrete from the equilibrium of the
  concrete — the forces its cells exert on the nodes of the interface (`OpsSolid.element_forces`), over the area a
  uniform pressure would load — as a watch evaluated after every step (`s.watch("p", interface_pressure(p0, axis, r))`,
  then `s.watched("p")`). It is exact to 0.3 % on any mesh (below). The stresses of the thin rust ring are not a
  measure of it: a coarse polygon forced to a large free strain puts several MPa of error into them (12–22 % low with
  24–32 cells round the bar).
- **Conversions**: `free_expansion(x, d, n, porous)` — the free expansion of a bar of diameter d corroded uniformly to
  the depth x when its rust takes n times the steel's volume, exact for the ring, (n − 1) x for x ≪ d, less a porous
  zone the rust fills first (Liu & Weyers: 12.5 μm); `depth_for_expansion` inverts it; `mass_loss`,
  `depth_from_mass_loss`; `cylindrical_strain` for other fields in cylindrical components (functions of the radius;
  rust on one side of the bar only is a function of position given to `initial_strain` directly).
- **References** for checks: `lame_cylinder` (elastic cylinder: compliance, first crack), `tepfers` (Tepfers 1979: the
  splitting pressure of an uncracked elastic, a partly cracked elastic and a plastic ring), and the critical corrosion of
  `liu_weyers_1998` and `el_maaddawy_soudki_2007` (thick-walled cylinders with a porous zone, cracking at the plastic
  pressure 2 c ft / d; their defaults are the papers' own creep coefficients, 2.0 and 2.35 — pass creep=0 to compare
  with a short-term analysis), `alonso_1998` (accelerated tests: x0 = 7.53 + 9.32 c/d μm for the first visible crack)
  and `gbt51355_2019` (the critical depths as given for GB/T 51355-2019, C.0.4 — **coefficients still to be checked
  against the original text**, as are the bar type and the strength they refer to).

What the model gives is the mechanics after the rust is there: how much free expansion splits the cover and how the
cracks open with more. What it does not give: how much rust stays in pores and cracks instead of pressing (a porous
zone of fixed thickness only), how fast it forms, creep of the concrete under the rust pressure, and non-uniform
corrosion unless it is given as a field. The empirical depths come from tests that include all of that.

**Example** (`examples/continuum_corrosion_slab.py`): a 10 mm plane-strain slice of an 80 mm slab, 200 mm wide, a Ø12
bar with 25 mm cover, transverse bars Ø8 @ 100 resting on it (smeared over their spacing, MC2010 splitting bond-slip),
fcm 33 MPa, δ up to 0.4 mm. **What holds the slab together across the bar decides everything**, so it runs with three
boundary conditions at the sides:

- `roller`: symmetry planes (u_y = 0 over the full depth, the transverse bars held there) — the same bar corrodes every
  200 mm, or the slab around is much stiffer than the cover: a continuous slab;
- `spring`: springs E / L per unit area on the sides (and E_s A_s / L on the transverse bars), L = 1 m — the slab goes
  on for 1 m to a rigid restraint (a web), the corroding bar alone;
- `open`: no restraint in the plane — a free strip: the extreme, an upper bound of the soffit crack, not a slab.

| Soffit crack | `roller` | `spring` (L = 1 m) | `open` (free strip, upper bound) |
|---|---|---|---|
| first crack at the bar | δ 1.5 μm | 1.5 μm | 1.5 μm |
| 0.05 mm | δ 42 μm, x 21 μm (27 with a 12.5 μm porous zone) | δ 16, x 8 (14) μm | δ 14, x 7 (14) μm |
| 0.2 mm | not reached: at most 0.05 mm | δ 61, x 31 (37) μm | δ 51, x 26 (32) μm |
| 0.5 mm | — | δ 199, x 102 (109) μm | δ 128, x 65 (71) μm |
| 1.0 mm | — | not reached: at most 0.56 mm (δ ≈ 270 μm), then it closes | δ 269, x 139 (146) μm (4.6 % mass loss) |
| what opens at δ = 0.4 mm | the cover delaminates along the bar layer: 0.50 mm | delamination 0.57 mm, soffit 0.41 mm, top 0.13 mm | the soffit crack: 1.33 mm |
| pressure of the rust on the concrete | up to 27 MPa (confined) | up to 21 MPa | up to 10.8 MPa (Tepfers' plastic ring: 10.7) |

x from δ with n = 3; with the transverse bars under the corroding bar (`--layout inner`, cover 33 mm over it) the soffit
crack stays at 0.04 mm (`roller`) or reaches 0.30 mm (`spring`), the cover delaminating by 0.7–0.9 mm; only the free
strip shows the soffit crack stopping at 0.33 mm and the cracks turning up to the top face (0.77 mm) — a folding of the
free strip about the yielding transverse bars, which a slab does not have.

For the first hairline crack on the soffit (0.05 mm) the FE needs x = 7–21 μm (14–27 μm with a porous zone); the
references give 7.1 μm (Liu & Weyers with their creep coefficient 2.0; 5.3 without creep), 8.4–13.0 μm (El Maaddawy &
Soudki, creep 2.35, porous zone 10–20 μm; 5.7 without creep), 27 μm (Alonso et al., accelerated tests, first visible
crack) and 105 μm (the GB/T 51355-2019 formula as given, natural corrosion, coefficients to be checked). The FE, with
rigid rust that all stays at the bar and no creep, belongs at the lower end.

Read the widths as a range, not a prediction: in this model a millimetre-wide crack along a bar on a soffit appears
from corrosion alone only where almost nothing holds the slab together across that crack (the free strip). With the
restraint of a continuous slab the corrosion of a layer of bars delaminates the cover, and a single corroding bar in a
slab tied to webs a metre away opens a soffit crack of at most about 0.5 mm within δ = 0.4 mm (x ≈ 0.2 mm, 7 % mass
loss). A wider crack then points to a second cause — restraint of shrinkage or temperature, loads — or to restraint
much weaker than assumed; check the restraint of the real structure (and its range) before concluding.

## Results

Every converged step keeps the displacements of all nodes (`Step.U`, solid nodes then bar and dowel nodes) and the
reactions of the supported nodes (`Step.R`, `Step.Rn`: fixed components, springs, displacement drivers). Recorded steps
(`record=True`, every k-th, or only the last) also keep, per cell: `stress`, `strain` (the strain the material sees,
ε − ε0; engineering shear; order xx yy zz xy yz zx), `eps0` (free strain), `eps_in`, `crack_strain`, `crack_normal`,
`width`, `cracked`, `crushed`, `eq_strain` (material equivalent strains); per bar segment `N`, `stress`, `strain`,
`eps0`; per bond spring `slip`, `tau`; per dowel element `N`, `V`, `M`, `local`.

- `curve(node, dof, at=…)`: load–displacement (−1 × the reaction resultant at the supports selected by `at`, i.e. the
  applied load with its sign; `sign=1` gives the reaction);
  `reaction(at)`; `first_crack()`; `bar_results(k)`, `dowel_results(k)` along one member.
- `width_history()` or `width_history(a, b)`: the largest crack width per recorded step (any cell, or the cracks along a
  line), to interpolate the load at a width limit; `stop=lambda step: step.cell["width"].max() > 0.2` ends a stage
  there (with `record` other than every step, unrecorded steps have `cell = None`: test for it).
- `to_result()`: fem-result/1 (solids plus bars and dowels as line2 cells, one frame per recorded step);
  `export("x.pvd")`: a VTU per step and a ParaView collection; `save_npz()`: everything as NumPy arrays.

## Validation

Loads, bars and cracks in `tests/test_continuum_opensees.py`; the notched beam, the reinforced tie and the comparisons on a common basis
were added after an independent review (its own scripts and a discrete cohesive-crack model).

| Check | Reference | Result |
|---|---|---|
| cantilever below cracking, hex8 and tet4, `ElasticIso` and `ConcreteDamage` | `femotion.continuum.Elastic(order=1)` (scikit-fem), same mesh | displacements 4e-13 (hex), 3e-12 (tet); cell stresses 5e-13 / 9e-7 (ASDConcrete3D's spectral split) |
| orthotropic block under uniaxial stress along x, y, z; `continuum.Orthotropic` converted | its compliance matrix | 1e-8, every column |
| uniaxial tension prism, ν = 0, h = 100 and 50 mm (implicit) | input law: area Gf | dissipated energy 0.999 Gf and 1.002 Gf; peak within one load step of ft |
| same, ν = 0.2 | mesh objectivity | 0.933 Gf and 0.934 Gf (lateral restraint of the band, see above) |
| same, IMPL-EX | implicit | 1.03 Gf (ν = 0.2; 1.08 Gf with ν = 0): 8–10 % above the implicit solution, every layer passing the peak by one step in the uniform field; same crack |
| same with tetrahedra (cubes cut in six), band ∛(6V) | input law | 0.999 Gf and 1.002 Gf (ν = 0); width = end displacement |
| IMPL-EX with tetrahedra and with bbarBrick | separation, energy | separated; within the IMPL-EX overshoot above |
| cells 50 × 100 × 100 across the crack, `band="volume"` / `("projected", n)` | h/∛V = 0.63; Gf | 0.63 Gf with a warning / 1.00 Gf |
| separated prism | end displacement less the elastic stretch | crack width along a line 0.29997 mm vs 0.29997 mm; cell width 0.2995 mm |
| crack opened to 0.5, 2.5 and 4 w1, unloaded to zero stress | plastic_t = 0.2 of w stays open | 0.20 (± 0.01) on both branches of the law |
| **notched three-point bend beam** (100 × 100 mm, notch 50, span 790; 10 mm cells, implicit) | independent discrete cohesive-crack model (scikit-fem plane stress, cohesive law on the ligament, Newton under CMOD control) | peak +0.1 %; load at CMOD 0.1 … 0.3 mm within 1.3 % |
| bonded bar on and off mesh nodes | EcAg + EsAs (the bar overlaps the concrete) | −7e-8 and −1.3e-7; bar force uniform to 4e-6 |
| **bar in a hexahedral beam in pure bending** (off the nodes, 25 mm cells) | scikit-fem's embedded bar (`continuum.embed.Bar`, the same trilinear interpolation); transformed section n M y / I | 1.3e-5 (penalty ties); uniform along the bar to 1e-5; beam theory within 3 % (8 trilinear cells over the depth) |
| bonded prestressing tendon (InitStressMaterial) | −σ0 Ap / (Ag + n Ap), σ0 Ap (1 − n Ap / (Ag + n Ap)) | 4e-7 and 5e-7 away from the anchors |
| unbonded eccentric tendon anchored at nodes | scikit-fem under the two anchor forces (equivalent loads) | cell stresses equal to 1e-12; the tendon lost 7.8 % (shortening and indentation under the point anchors) |
| **unbonded tendon over a frictionless deviator**, prestress and a load on one half | scikit-fem with `continuum.embed.Bar(bonded=False)` (one force from the summed elongations) | force 1e-10, displacements 3e-10; one force along the tendon (force increments 7.32 / 7.32 kN; `slide=False`: 9.28 / 5.36 kN) |
| **prestress-only stage under IMPL-EX** (strip on two bolts, 480 cells), steps = 1 and 5 | implicit solution | crack width +4 %, strip force 0.01 %, same cracked cells |
| anchor bolt in a rigid block, rot=True | clamped cantilever Pa³/3EI | 1 + 1e-11; shear P and moment Pa at the surface |
| prestressed strip on two bolts | σ = σ0 + E ε of the strip from its end displacements; equilibrium | to 1e-6; head shear = strip force; reactions 0 |
| sub-model cut from a global linear model | the global field inside | 1.5e-8 |
| sub-model driven by a coarse quadratic model (`boundary_from(CaseResult)`) | the global model's own interpolation | boundary values exact; inside 5e-5; linear interpolation between corner nodes would be 3.6 % off |
| RC beam, linear pre-peak: first crack | ft Ig / (yb − zgp), transformed section at the bottom Gauss points: 10.24 kN·m (50 mm cells), 9.85 kN·m (25 mm); ft·W = 9.50 kN·m | 10.03 … 10.65 kN·m and 9.25 … 9.91 kN·m (bracketing load steps) |
| RC beam, MC2010 pre-peak: first crack | MC2010 flexural tensile strength: fctm,fl·W = 1.31 ft·W = 12.42 kN·m | 14.00 … 14.43 kN·m = 1.13 … 1.16 × fctm,fl·W (no self-weight in the test beam; the example adds its 1.07 kN·m) |
| RC beam, load–deflection | uncracked and fully cracked beam (bending only), Branson's effective I in between | between the bounds at every step; 10–20 % stiffer than Branson in service (perfect bond: more tension stiffening), within 3 % at 0.8 fy |
| RC beam, steel stress in the constant-moment zone | cracked section n M (d − x) / Icr | 0.80 … 0.93 of it once cracking has stabilised, rising with the load (concrete between cracks still in tension) |
| RC beam, crack widths at σs = 260 … 360 MPa (cracked section), on the basis of the analysis (short-term, monotonic) | JTG 3362-2018 6.4.3 Wcr with C2 = 1 (`femotion.codes.jtg.sls.crack_width`, Ml = 0); Eurocode 2 7.3.4 wk with kt = 0.6, hc,ef = min(2.5 (h − d), (h − x)/3, h/2) | FE largest crack 0.90 … 1.09 × Wcr and 0.73 … 0.88 × wk; spacing 106 … 131 mm |
| reinforced tie 100 × 100 × 1000, Ø16 (independent review, 25 and 12.5 mm cells) | fib MC2010 7.6 (mean and maximum widths), EC2 | surface widths between MC2010's mean and maximum, below EC2's wk; spacing about half of the codes' mean spacing, and dependent on the mesh while cracks form |

Free strains, bond-slip and rust (`tests/test_continuum_eigenstrain.py`):

| Check | Reference | Result |
|---|---|---|
| uniform temperature on a statically supported block with a bar of the same α | free expansion: no stress, u = α ΔT x | stresses 1e-7 of E α ΔT (round-off of the bar's penalty ties); displacement 2e-7 |
| block held on all faces, isotropic and anisotropic ε0, two stages added | σ = −E ε0 / (1 − 2ν); −D ε0 | 1e-9 |
| free strain field with all six components varying inside the cells; a temperature field (integration points) | `femotion.continuum.Elastic(order=1)` with the same field as equivalent loads, same mesh | displacements 2e-13, cell stresses 5e-13 (`ElasticIso`); ASDConcrete3D below cracking: displacements 1e-13 (IMPL-EX; 3e-6 implicit), stresses 7e-5 (its own departures from linearity) |
| bimetal strip 100 × 10 mm (ν = 0), heated, 50 and 100 cells along | Timoshenko (1925) | curvature 0.978 and 0.995 (0.9986 with 200): the 8-node brick locks in bending as cells get long; converges |
| heated prism with a bar of another α | composite strain (Ec Ac αc + Es As αs) ΔT / (Ec Ac + Es As) | bar force 9e-5 in the middle third; end displacement −0.16 % (the bar hands its force over near the ends) |
| **Steel01 bar yielding under a free strain of its own**, unbonded beside an elastic block (EsAs/EcAc = 0.33 and 1.33), −6e-3 and back | the same path in 40 steps (bilinear steel on a monotonic path: no step dependence) | 1, 5 and 40 steps agree to 1e-6, loading and unloading (with `InitStrainMaterial` on the bar: −30 % in 1 step, −2 % in 20) |
| `thermal(20, "top")` with bars in both halves; α = 0 materials | only the bars inside the selected cells heat | bottom bars 0, top bars α ΔT; warning for α = 0 cells, error when none expands; `shrinkage()` on concrete cells only |
| **prism held at both ends shrinks and cracks** (weak layer, ν = 0) | the crack band law: ∫σ dw = Gf with w = ε_sh L − σ L / E | 0.999 Gf (implicit), 1.03 Gf (IMPL-EX); peak within a step of ft; cell width = w to 0.2 % (with ν = 0.2: 0.934 Gf, the lateral restraint of the band, as under loads; the same prism pulled at its end instead of shrinking gives the same reactions to 4e-7 MPa — independent review) |
| bar with a rigid-plastic bond law pulled out of a stiff block; a bar entering the solid between two of its nodes | τ π d L (bonded length inside the solid only, the surface crossing found by bisection) | 1e-8 (point location tolerance); every spring at τ; bonded length 200.000 mm of 200 |
| **RC tie 100 × 100 × 600, Ø16, fixed at both ends, ε_sh = 6e-4** (bond τb0 = 2 fct, strength ±5 % between sections, 20 mm cells) | tension chord model (Marti et al. 1998): spacing within [s_rm0/2, s_rm0] = [97, 195] mm — the same rigid-plastic bond as the FE, so this checks the implementation of the assumption, not real crack spacings; Σw = L (ε_sh − R / Ec Ac) (kinematics, R including the bar's end reaction); EN 1992-1-1 (7.11) s_r,max = 413 mm and EN 1992-3 annex M (M.1) w_k = 0.167 mm for reference | spacings 100–142 mm, mean 125; Σw 0.330 vs 0.336 mm (−2 %); widths 0.05–0.09 mm, below w_k, which assumes the restraint force at the cracking force (here it has dropped to 42 % of it) |
| O-grid sections | areas | section exact to 1e-9; bar = polygon of 4m sides, ring = its annulus |
| **elastic rust ring in a concrete cylinder** (d 12, R 40, ring 0.6 mm of 20 GPa, plane strain, 24, 32 and 64 cells around) | exact three-layer Lamé solution with the ring's free strain | interface pressure from the concrete's equilibrium (`interface_pressure`) +0.31 / +0.26 / +0.21 %, the same at every node; radial displacement of the bar surface −0.1 %, of the outside +0.02 %; hoop stress next to the bar 0.03 % (the stresses of the thin ring itself are 22 / 12 / 3 % low: a tool to impose δ, not a result) |
| **rust cracking a cylinder of cover** (d 12, c 25, C30, ring as stiff as steel, 24 and 32 cells around) | first crack: Lamé hoop stress at the bar = ft (δ = 0.613 μm); pressure: Tepfers (1979) partly cracked elastic ring 4.50 MPa, plastic ring 12.07 MPa; thick-walled models (Liu & Weyers 1998, El Maaddawy & Soudki 2007, no creep): the plastic pressure on an uncracked elastic cylinder, δ = 2.76 μm | first crack at 0.65 / 0.61 μm (within a step); pressure plateau 11.86 / 11.87 MPa = 0.98 × the plastic ring, from δ ≈ 6 μm on, slowly falling (11.4 MPa at 20 μm); the outer surface cracks on the plateau at δ = 5.78 / 5.75 μm = 2.1 × the thick-wall value |

In the axisymmetric cylinder every cell around the bar cracks at once and each crack opens by 2πu/(4m), far less than
w1 = Gf/ft ≈ 0.05 mm: the cracked zone still carries about ft, and the cover behaves like Tepfers' plastic ring — right
for a cover much smaller than the characteristic length E Gf/ft² (≈ 560 mm for C30). The pressure does not depend on
the number of cells around (0.1 %; the reviewer's random strengths, localising into four to seven cracks, gave
11.5–11.9 MPa). Read the pressure from the equilibrium of the concrete (`interface_pressure`), not from the rust
ring's stresses, which carry the error of a coarse polygon forced to a large free strain (earlier versions of this
guide reported 7.1 / 9.1 MPa read that way).

The RC beam: 200 × 300 mm, span 2400, loads at a = 900, 2 Ø16 at 40 mm, fcm 38 MPa, quarter model, 50 mm hexahedra,
line bearing, elastic pads under the bearing and the loading plate. Supporting the beam on a strip of node lines instead
of a line clamps its end: in the elastic range the 25 mm model was then 22 % stiffer than beam theory with shear, against
0.3 % on a line bearing. The widths were first compared with long-term values (JTG with C2 = 1.5, EC2 with kt = 0.4 and
hc,ef without (h − x)/3), which made the FE look like "mean cracks below characteristic values"; on the short-term basis
of the analysis it is the size of the codes' maximum widths.

## Performance

Apple M1 Max, OpenSeesPy 3.8 (UmfPack), IMPL-EX, two Newton iterations per step:

| Model | Cells | Steps | Time |
|---|---|---|---|
| RC beam, 50 mm | 312 + 52 bar segments | 100 | 17 s |
| RC beam, 25 mm | 2 496 + 104 | 100 | 179 s (`numberer="RCM"`; 199 s with AMD) ¹ |
| member with two anchor bolts and a prestressed strip, 50 mm | 3 840 + 2 dowels | 17 settling steps + 1 | 34 s ² |
| the same, 50 × 25 × 25 mm | 15 360 (52 000 equations) | 10 | 247 s (AMD; 352 s with RCM) ¹; 29 settling steps + 1: 998 s ² |

| restrained RC tie with a bond-slip bar (free strain) | 600 + 30 bond springs | 150 | 9 s |
| rust in a thick-walled cylinder, 24 cells around (pressure watched every step) | 504 | 90 | 9 s |
| corroding slab, plane-strain slice, per boundary condition | 1 150–1 322 + 35 bond springs | 230 | 76–129 s |
| the same slab as a 3D slice, 4 cells along the bar | 6 184 | — | 2.5 s per Newton iteration ³ |

¹ measured before the trilinear ties, which add six auxiliary nodes and seven tie elements per bar node (a few per cent
of the equations); ² with another heavy job on the machine, so an upper bound. The prestress of a stage settles in 10–20
steps under IMPL-EX (see "Solving"). ³ UmfPack's factorisation of 24 000 equations dominates (SuperLU 2.3 times and
SparseSYM 50 times slower; MUMPS is not in OpenSeesPy for macOS): model a bar's surroundings as a slice where the
problem allows. Free strains add one parameter update per group of points per step (0.2 μs each).

The material update (eight points per hexahedron, about 13 µs each) and the sparse factorisation (UmfPack; the
equation ordering `numberer` matters: AMD for bulky blocks, RCM for long slender members) share the time; IMPL-EX needs
two to three iterations per step. Reading the results (two element responses per cell and one per integration point)
adds about 0.2 s per step for 4 000 cells.

## OpenSees notes (3.8)

- `ASDEmbeddedNodeElement` takes the fourth retained node as an optional argument parsed from a string: passed as a
  Python int it is silently dropped and the node is tied to the triangle of the first three nodes (a wrong constraint,
  no warning). Pass `str(node)`.
- Its default penalty 1e18·∛V leaves round-off of about 1e-5 in the displacements and stalls convergence tests; here it
  is 1e6 × the largest Young's modulus (stiffness error ~3e-8).
- Its triangle tie in 3D (three retained nodes) is wrong for every triangle that is not horizontal: computing the
  point's local coordinates it zeroes the third *column* of J⁻¹ (the global z component of the offset) instead of the
  normal row, so a point between two nodes at different heights gets the weights of a point at the first node's height.
  Only tetrahedra are used here (a point on an edge of a tetrahedron gives the 2-point interpolation exactly).
- The tetrahedron tie takes det J / 6 as its volume without the absolute value: a negatively ordered tetrahedron gives
  a negative penalty. The retained nodes are ordered to a positive volume here.
- `equationConstraint` (a multi-point linear constraint, which would give the trilinear tie directly) works only with
  the Penalty and Lagrange handlers, and `wipe` does not remove these constraints: they stay in the domain and corrupt
  every later model of the Python process. Not used here.
- `geomTransf` and frame elements need the 6-dof model builder (`model -ndf 6`) at the time they are created; loads on
  6-dof nodes need six components.
- ASDConcrete3D's responses `crackWidth` and `crackPattern` multiply by the element's shortest node distance; here the
  width is computed from stresses and strains and the band of the law instead.
- IMPL-EX under displacement control, and with loads that do not change (holds, prestress): see "Solving".
- A failed step reverts ASDConcrete3D's time increment to the committed one: the pseudo-time is set again before every
  attempt.
- The initial-strain wrapper exists in OpenSeesPy 3.8 as `nDMaterial InitStrain tag base eps0 <5 more>` (class
  InitStrainNDMaterial, parameters `eps0`, `eps0_11` … `eps0_13`); `nDMaterial InitStrainNDMaterial` is "unknown". It
  adds its value to the strain (σ = f(ε + ε_init)), so a free strain is passed with the opposite sign.
  `InitStressNDMaterial` around `ElasticIsotropic` aborts the process ("getCopy — subclass responsibility").
- `uniaxialMaterial InitStrainMaterial` commits the wrapped material at the shifted strain when its parameter `epsInit`
  changes (`InitStrainMaterial::updateParameter` calls `commitState`), unlike the nD version: bars therefore take their
  free strain through a stiff elastic spring in series (see "Free strains"). `SeriesMaterial` would hide the parameter
  (it does not forward `setParameter`).
- `setParameter -val v -ele e … ` costs about 0.9 μs a call, `updateParameter` 0.2 μs: per-point free strains are cheap.
- Bond laws: `uniaxialMaterial MultiLinear` jumps on unloading (from the softening plateau at 8 to −15 within 0.5 of
  the slip in a probe); `ElasticMultiLinear` unloads along its loading curve; `HystereticSM` follows a multi-point
  envelope and unloads like Hysteretic, so it is used (it prints a banner when first created).
- `system SparseSYM` was 50 times and `SuperLU` 2.3 times slower than UmfPack on a 24 000-equation solid.

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
- JTG 3362-2018, 6.4.3; EN 1992-1-1:2004, 7.3.4; EN 1992-3:2006, annex M; JTG D60-2015, 4.3.12.
- Timoshenko, S. (1925). Analysis of bi-metal thermostats. J. Opt. Soc. Am. 11(3), 233–255.
- Marti, P., Alvarez, M., Kaufmann, W., Sigrist, V. (1998). Tension chord model for structural concrete. Structural
  Engineering International 8(4), 287–298.
- fib Model Code 2010, 6.1.1 (bond and anchorage, table 6.1-1).
- Tepfers, R. (1979). Cracking of concrete cover along anchored deformed reinforcing bars. Magazine of Concrete Research
  31(106), 3–12.
- Alonso, C., Andrade, C., Rodriguez, J., Diez, J. M. (1998). Factors controlling cracking of concrete affected by
  reinforcement corrosion. Materials and Structures 31, 435–441 (and Alonso et al. 1994, Hormigón y Acero 45(194): first
  visible cracks of 0.05–0.1 mm at 15–50 μm, growth 0.003–0.018 mm/μm).
- Liu, Y., Weyers, R. E. (1998). Modeling the time-to-corrosion cracking in chloride contaminated reinforced concrete
  structures. ACI Materials Journal 95(6), 675–681; Liu, Y. (1996), PhD thesis, Virginia Tech, 5.6.
- El Maaddawy, T., Soudki, K. (2007). A model for prediction of time from corrosion initiation to corrosion cracking.
  Cement & Concrete Composites 29, 168–175.
- Caré, S., Nguyen, Q. T., L'Hostis, V., Berthaud, Y. (2008). Mechanical properties of the rust layer induced by
  impressed current method in reinforced mortar. Cement and Concrete Research 38, 1079–1091 (volume ratios of rusts).
- Noubactep, C., Caré, S., Crane, R. (2012). Nanoscale metallic iron for environmental remediation. Water, Air, & Soil
  Pollution 223, 1363–1382 (table 5).
- GB/T 51355-2019 既有混凝土结构耐久性评定标准, appendix C.0.4.
- Mazzoni, S. (2023). HystereticSM material, OpenSees documentation.
