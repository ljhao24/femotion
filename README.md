# femotion

[![test](https://github.com/ljhao24/femotion/actions/workflows/test.yml/badge.svg)](https://github.com/ljhao24/femotion/actions/workflows/test.yml)

**Finite-element and simulation results made for animation.** Compute with mature open-source solvers, write a compact
result file, and draw it frame by frame — deformation, stresses, force flow, mode shapes, topology optimisation
iterations, flow around a bridge deck.

[中文说明](README.zh-CN.md)

femotion is made for engineering animation in which every deflection, stress colour and vortex on screen has to be
*computed*, not drawn by hand. It is organised around three ideas:

1. **Lean on mature solvers.** OpenSees, scikit-fem, gmsh, pyamg and SciPy do the heavy lifting. femotion adds a
   uniform model format, glue, and a fast linear frame solver that is cross-checked against OpenSees.
2. **One result file between computing and drawing.** Solvers write `fem-result/1` (a JSON description plus a
   little-endian binary blob, see [docs/result-format.md](docs/result-format.md)); the JavaScript packages read it.
   Neither side depends on the other. Both sides are tested against the same fixed sample (`tests/data/golden_v1.*`).
3. **Keep what animation needs.** Every topology-optimisation iteration, the stiffness-matrix pattern in assembly
   order, time histories resampled to the frame rate, stress trajectories as polylines.

> Status: alpha (0.1.0.dev0). The API may still change.

## Install

```bash
pip install femotion                 # numpy + scipy only: frames (linear), topology optimisation, result files
pip install "femotion[continuum]"    # + scikit-fem, pyamg: plane and solid elasticity
pip install "femotion[mesh]"         # + gmsh: unstructured, quadrilateral, extruded and structured hexahedral meshes
pip install "femotion[tetwild]"      # + fTetWild: tetrahedra from messy surface models (Blender, downloads, scans)
pip install "femotion[opensees]"     # + OpenSeesPy: nonlinear frames
pip install "femotion[cfd]"          # + MLX: 2D lattice Boltzmann flow (Apple silicon only)
pip install "femotion[io]"           # + meshio: read / write ~30 mesh formats, export results for ParaView
pip install "femotion[sections]"     # + sectionproperties: cross-sections of any shape, fiber sections
pip install "femotion[all]"
```

Requires Python 3.12 or newer, NumPy 2.0 and SciPy 1.13 or newer (the extras state their own minimums). On Linux, gmsh
and OpenSeesPy need a few system libraries: `sudo apt install libglu1-mesa libxcursor1 libxft2 libxinerama1 libblas3
liblapack3` (Debian / Ubuntu, as on the CI).

## Quick tour

```python
import numpy as np
from femotion.frame import Model, Section, Solver

m = Model()
nodes = [m.node((x, 0, 0)) for x in np.linspace(0, 20, 41)]
for a, b in zip(nodes[:-1], nodes[1:]):
    m.beam(a, b, Section(E=2e8, A=1.0, Iy=0.5, Iz=0.5, J=1.0, mass=2.0))   # m, kN, t; E in kPa
m.fix(nodes[0])

sv = Solver(m)                       # assemble and factorise once
F = np.zeros(m.ndof); F[6 * nodes[-1] + 2] = -100.0
U = sv.solve(F)                      # each further load case is a single back-substitution
freqs, Phi = sv.modes(6)
```

Nonlinear behaviour with OpenSees, on the very same model:

```python
from femotion.frame.opensees import OpsFrame, Fiber

f = OpsFrame(m, transf="Corotational", fiber={"": Fiber.rect(0.1, 0.2, Fiber.steel_pp(3.45e5))})
lam, Us = f.pushover(F, node=nodes[-1], dof=2, target=-0.4, steps=80)   # follows the path past the peak
```

Before trusting the fast linear solver on a new model, compare it with OpenSees:

```python
from femotion.frame.check import crosscheck
crosscheck(m)    # displacements, bar forces, beam end forces and frequencies must agree to 1e-6
```

Continuum elasticity and the result file:

```python
from femotion.mesh import gmsh as G
from femotion.continuum import Elastic
from femotion.post import stress
from femotion.result import Result

mesh = G.plane(lambda occ: occ.cut([(2, occ.addRectangle(0, 0, 0, 10, 10))], [(2, occ.addDisk(0, 0, 0, 1, 1))]), size=0.2)
s = Elastic(mesh, E=1.0, nu=0.3, order=2)
s.traction(lambda x: np.isclose(x[0], 10), (1.0, 0.0))
s.fix(lambda x: np.isclose(x[0], 0), comps=(0,)); s.fix(lambda x: np.isclose(x[1], 0), comps=(1,))
s.solve()
Result(s.result_mesh(), meta={"title": "plate with a hole"}) \
    .field("u", s.nodal_displacement(), unit="m") \
    .field("vm", stress.von_mises(s.nodal_stress())) \
    .write("out", "plate")
```

Cross-sections of any shape, and results in ParaView:

```python
from sectionproperties.pre.library import rectangular_hollow_section
from femotion.frame import sections
from femotion.frame.opensees import Fiber
from femotion import io

box = rectangular_hollow_section(d=2.5, b=1.2, t=0.03, r_out=0.0, n_r=1)   # m
sec = sections.elastic(box, E=2.06e8, nu=0.3, density=7.85)               # A, Iy, Iz, torsion constant J, mass
fib = sections.fiber(box, Fiber.steel(3.45e5))                             # the same shape as an OpenSees fiber section

mesh = io.read_mesh("part.msh")                       # gmsh, Abaqus, Ansys, Nastran, VTK, …; physical groups become groups
io.export_result("out/plate.json", "out/plate.vtu")    # open in ParaView; animated results go to a .pvd collection
```

## JavaScript packages

| Package | What it does | Dependencies |
|---|---|---|
| [`@femotion/core`](js/core) | parse `fem-result/1`, read fields and polylines; display geometry (triangles, edges, boundary loops, 3D outer faces and visible edges, contours, reveal order); 2D / orthographic / perspective projections; colour maps; a 2D frame solver small enough to run every frame | none |
| [`@femotion/remotion`](js/remotion) | [Remotion](https://www.remotion.dev) components: `useFemResult`, `FieldCanvas` (per-pixel field colouring with reveal, banding, depth buffer, mesh lines), `MeshLines`, `Polylines`, `ColorBar`, `MatrixGrid` | `@femotion/core`; peers `react`, `remotion` |

The packages ship TypeScript sources (`exports` → `src/index.ts`, explicit `.ts` import extensions), so bundlers and
Node ≥ 22.18 use them directly. `cd js && npm install && npm test` runs the core tests with Node's built-in runner.

## Layout

| Module | What it does | Back-ends |
|---|---|---|
| `femotion.mesh` | unified `Mesh` (line2, tri3, tri6, quad4, tet4, hex8), structured and block-structured grids; gmsh (triangles, quadrilaterals, tetrahedra, extruded and structured hexahedra, STEP import); fTetWild for messy surface models | gmsh, fTetWild |
| `femotion.frame` | beams, bars, cables, springs: model format, fast linear solver, OpenSees back-end, cross-check, post-processing; sections and fiber sections of any shape (`frame.sections`) | own (linear), OpenSees, sectionproperties |
| `femotion.continuum` | plane stress / strain and 3D linear elasticity, linear or quadratic elements, direct or AMG | scikit-fem, pyamg |
| `femotion.topopt` | 2D SIMP topology optimisation recording every iteration | own (88-line algorithm) |
| `femotion.cfd` | 2D lattice Boltzmann flow with smoke lines, Apple GPU | own (MLX Metal kernels) |
| `femotion.dynamics` | modal superposition: step release, ground motion, moving loads | — |
| `femotion.post` | principal stresses, stress trajectories, stiffness-matrix patterns | — |
| `femotion.result` | the `fem-result/1` file | — |
| `femotion.io` | other formats: read / write meshes, export results (VTU, ParaView `.pvd` collections) | meshio |
| `femotion.models` | ready-made illustrative structures (a 500 m cable-stayed bridge) | — |
| `femotion.frame.influence` | influence lines of every section from one factorisation, lane-load and axle-train envelopes, exact section forces under distributed loads | own (linear) |
| `femotion.frame.prestress` | tendons on frame models: equivalent loads, total, primary and secondary effects | own (linear) |
| `femotion.frame.thermal` | imposed strains and curvatures (temperature, shrinkage) on frame models: secondary effects | own (linear) |
| `femotion.frame.stages` | construction stages (members, supports, links, loads, tendons added and removed) and linear creep and shrinkage step by step in time: section forces, reactions, displacements since each node was built | own (linear) |
| `femotion.codes.jtg` | Chinese highway bridge codes: traffic, crowd and temperature actions and combinations (JTG D60-2015); effective flange widths, ultimate-limit-state resistance of reinforced and prestressed concrete sections with internal and external tendons (flexure, shear, compression, tension, torsion, local compression, punching), prestress losses, crack resistance with temperature gradients, crack width, deflection and stress checks, composite flexural members, deck slabs, box girders across the bridge, creep secondary effects, strut-and-tie models, anchorage zones, diaphragms, cap beams and pile caps, hinges, elastomeric bearings, expansion joints, durability and detailing (JTG 3362-2018); alpha, see [docs/codes/](docs/codes/) | — |

### Which frame solver?

| Problem | Use | Why |
|---|---|---|
| Linear, solved many times (vehicles frame by frame, influence lines, cable-force optimisation, modes) | `frame.linear` after `crosscheck` passes | factorise once, ~0.2 ms per load case; ~20× faster than OpenSees |
| Nonlinear: yielding, plastic hinges, buckling, large displacements, slack cables, collapse after member loss, earthquake damage | `frame.opensees` | mature, widely validated |
| Linear, solved once or twice | either; OpenSees if in doubt | speed does not matter |

### Design-code checks

```python
from femotion.codes import jtg
from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces

il = InfluenceLines(sv, deck_nodes, chain_sections(deck_beams))     # every section's influence line at once
G = uniform_load_forces(sv, il.sections, "My")                         # self-weight, exact at the element ends
Q = jtg.LiveLoad(L0=16, lanes=2, mu=jtg.impact_factor(f1)).effects(il, "My")
Mmax, Mmin = jtg.basic_combination(G, Q)
sec = jtg.tee(200, 1100, 1600, 160, concrete="C40", bottom=[(10, 28, 110)], top=[(8, 20, 50)])   # mm
print(jtg.flexure(sec, Mmax.max()))      # flexure: γ0Sd = …, R = …, ratio … → OK
Q0 = jtg.LiveLoad(L0=16, lanes=2).effects(il, "My")                   # serviceability: no impact
Ms, Ml = jtg.frequent_combination(G, Q0)[0], jtg.quasi_permanent_combination(G, Q0)[0]
k = Ms.argmax()
print(jtg.crack_width(sec, Ms[k], Ml[k], env="I"))   # crack width: Wcr = … mm, limit 0.200 mm, ratio …
```

Reinforced and prestressed concrete (bonded internal tendons, pretensioned or post-tensioned, and external tendons):
ultimate limit state (flexure, shear including circular sections, compression including spiral columns, bars over the
web, biaxial bending and the effective lengths of appendix E, tension members, torsion, local compression, punching),
serviceability (crack resistance with the temperature gradient, crack width including class B members under axial
force, stiffness and deflection), stresses in service and in construction stages, composite girders (precast girder +
deck), deck slabs (wheel load distribution widths, slabs between ribs and cantilevers), box girders across the bridge
(closed transverse frames, shear flow, in-plane principal stresses), creep secondary effects and construction stages
(`femotion.frame.stages`), and the disturbed regions of chapter 8 and appendix B (strut-and-tie models, anchorage zones,
cap beams, pile caps), with inclined-section search, stirrup and bent-bar design, moment redistribution at supports,
concrete hinges, laminated elastomeric bearings and expansion joints (8.6–8.8), durability (4.5) and the numeric detailing
rules of chapter 9 (cover, anchorage, splices, spacing, ducts, end zones, columns, pile caps). Detailing never fails a
check: unmet rules go to `Check.warnings`, `Check.ok` is the verdict of the calculation. Examples: `examples/jtg_rc_girder.py` a two-span continuous girder; `examples/jtg_pc_girder.py` a
post-tensioned continuous girder from tendon profile and losses to the secondary moments of prestress and temperature
and the ultimate, serviceability, stress and anchorage-zone checks; `examples/jtg_external_girder.py` a box girder with
external tendons; `examples/jtg_composite_girder.py` a precast girder with its deck slab; `examples/jtg_staged_girder.py`
precast girders made continuous, stage by stage with creep and shrinkage of appendix C to ten years (creep secondary
moments, pre-camber); `examples/jtg_box_transverse.py` the deck slab of a box girder across the bridge;
`examples/jtg_substructure.py` a pier cap, pile caps, piles and columns; `examples/jtg_devices.py` bearings, an expansion
joint and a concrete hinge; `examples/jtg_detailing.py` durability and the detailing of a T girder. Values and formulas have been checked clause
by clause against the official texts. [docs/codes/](docs/codes/) has one file per code (in Chinese) with every clause,
formula, table and parameter the program uses, page references and the function using each, and lists what is not
covered.

## Validation

`pytest` runs closed-form and cross-solver checks: patch tests (tri3, tri6, quad4, hex8), Kirsch's plate (Kt → 3),
Timoshenko cantilevers, AMG vs direct, Euler buckling (Southwell point at 0.9 Pcr), full plastic moment, tension-only
cables, dynamic amplification 2 after sudden member loss, linear solver vs OpenSees on a 728-node cable-stayed bridge
(displacements, forces 1e-11, frequencies 2e-9), section properties against the exact torsion series of a rectangle (1e-4),
Bredt's formula and a hand-computed composite section, fiber sections of any shape reaching the plastic moment, lossless mesh
round trips through VTU, Abaqus and gmsh files, exact volumes of extruded and structured hexahedral meshes and of STEP
imports, fTetWild turning two overlapping boxes with a missing face into one solid of the union's volume, and result-file
round trips.

What is wired in, what is waiting for a use case and what was ruled out: [docs/integrations.md](docs/integrations.md).

## Development

```bash
pytest                          # Python
cd js && npm ci && npm test     # JavaScript
```

- `tests/data/golden_v1.*`: the fixed `fem-result/1` sample. The Python writer must reproduce it byte for byte and every
  reader must decode it.
- `tests/data/regression_v1.npz`: pinned results of representative computations (`tests/regression.py`), with
  tolerances per kind of computation that absorb last-digit differences between NumPy, SciPy and BLAS versions. A
  dependency release that changes a number fails here; regenerate (`python tests/regression.py`) only once you know why
  the value moved.
- The lower bounds in `pyproject.toml` are tested, not guessed: Python 3.10 with NumPy 1.24, SciPy 1.10, gmsh 4.11,
  scikit-fem 10, pyamg 5.2, OpenSeesPy 3.5, MLX 0.21, meshio 5.3, sectionproperties 3.4 and pytetwild 0.4 passes the
  whole suite, as do the newest releases on Python 3.13.
- Continuous integration (`.github/workflows/test.yml`) runs those lower bounds and the newest releases on Linux, Windows
  and Apple-silicon macOS (the only runner for the MLX flow tests) on every push, and again every Monday against fresh
  releases, so an upstream release that breaks femotion shows up before users meet it.

## Licence

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later, see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
Commercial use is allowed; if you distribute femotion, a modified version, or software built on it, or let users
interact with it over a network, you must make the corresponding source available under the same licence.
Releases up to and including commit `1be6a41` were published under Apache-2.0 and stay available under it.

Optional back-ends are installed from their own
distributions under their own licences: gmsh is GPL-2.0-or-later; OpenSees is free for education, research and
non-profit use (commercial use requires a licence from the Regents of the University of California); MLX runs on Apple
silicon only; sectionproperties meshes with Shewchuk's Triangle, which is free for non-commercial use only.
