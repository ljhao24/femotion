# Solid models (`femotion.continuum`)

Linear elasticity of plane and solid models on [scikit-fem](https://github.com/kinnala/scikit-fem), built for structures
rather than single parts: several materials, elastic supports, reinforcement and tendons that do not follow the mesh,
any number of load cases solved with one factorisation, and the force resultants a beam model can be checked against.

```python
import numpy as np
from femotion.continuum import Elastic, Isotropic, Orthotropic, Tendon, rect, export_vtu
from femotion.mesh.section import LayeredSection, sweep

deck = sweep(section, xs, z)                                   # hexahedra, cell groups "girder", "pavement"
s = Elastic(deck, materials={"girder": Isotropic(3.0e10, 0.2, 1e-5), "pavement": Isotropic(3.0e10, 0.2, 1e-5)},
            order=2, serendipity=True)                         # 20-node hexahedra
for i, (xc, yc) in enumerate(bearings):                        # bearings: springs per unit area of the plate
    fac = s.facets(rect(xc - 0.10, xc + 0.10, yc - 0.125, yc + 0.125), normal=(0, 0, -1))
    A = s.facet_integral(fac, lambda x: 1 + 0 * x[0])
    s.spring(fac, k=(kh / A, kh / A, kv / A), name=f"B{i}")

s.case("G1").body_force((0, 0, -26e3), groups=["girder"])
s.case("P").tendon(Tendon.from_pis(pis, radii).with_friction(950e3, mu=0.25, k=0.0015), anchor=(0.21, 0.21))
s.case("wheel").area_load(s.facets(normal=(0, 0, 1)), (0, 0, -70e3), rect(7.9, 8.1, 1.2, 1.8))
s.case("T+").thermal(lambda x: 25.0 * np.clip(1 - (top(x[1]) - x[2]) / 0.1, 0, 1))
s.solve()                                                      # all cases, one factorisation / AMG hierarchy

r = s.result("G1") + s.result("P") + 0.8 * s.result("T+")      # results combine linearly
r.sample(points)["stress"]                                     # stress at arbitrary points
r.section((8.24, 0, 0), (1, 0, 0))                             # {"N", "F", "M", ...} on a cut
r.spring_force("B0")
export_vtu("out/deck.vtu", s, {"G1": s.result("G1"), "sum": r})
```

Cracking — concrete softening, crack widths, bars yielding — is the OpenSees back-end `femotion.continuum.opensees`
([continuum-nonlinear.md](continuum-nonlinear.md)): the same `Mesh` and cell groups, the linear model here driving its
submodels.

## Pieces

| module | what |
|---|---|
| `materials` | `Isotropic(E, nu, alpha)`, `Orthotropic(E, nu, G, alpha, axes)` and `Orthotropic.transversely_isotropic` (fibre plates); 3D, plane stress and plane strain laws, thermal strains |
| `elastic` | `Elastic`: assembly per material group in chunks of cells (bounded memory), supports (`fix`, `spring`, `spring_point`), embedded bars (`bar`), load cases (`case`), `solve` |
| `loads` | `LoadCase`: `traction`, `pressure`, `area_load`, `body_force`, `initial_strain`, `thermal`, `point_load`, `point_forces`, `line_load`, `tendon`, `bar_prestress`, `displacement` |
| `embed` | `Locator` (points → cells and reference coordinates; interpolation and its transpose, consistent nodal forces), `Bar` (bonded or unbonded axial bar between embedded points) |
| `tendon` | `Tendon`: paths from intersection points and radii (`fillet_path`), friction `P e^{−(μθ+kx)}` from either or both ends, the forces it puts on the concrete |
| `results` | `CaseResult`: displacements and stresses (initial strains taken off), `sample`, `section`, `spring_force`, `bar_force`, `+`, `*`; `export_vtu` |
| `submodel` | `drive`: the cut boundary of a finely meshed part prescribed with the displacements of a coarser model |
| `femotion.mesh.section` | `LayeredSection` + `sweep`: structured hexahedra for members with a varying cross-section (box girders, voided slabs, end thickenings, haunches) |

### Materials and groups

Every cell needs exactly one material: `material=` (or `E`, `nu`) for all cells, `materials={group: material}` for cell
groups of the mesh, the latter overriding the former. Voigt order (xx, yy, zz, xy, yz, zx) with engineering shear
strains everywhere. An orthotropic material is given in its own axes and rotated by `axes` (rows: axes 1, 2, 3 in global
coordinates); plane stress and plane strain reduce the 3D law (the material's axis 3 should then be z).

### Supports

`fix(where, comps, value)` selects the dofs located where the predicate holds — a point, an edge, a face alike (on curved
boundaries the mid-edge nodes of quadratic cells lie on the chord) — or the dofs of given vertex ids; a nonzero `value`
is a settlement of the case "default" (other cases hold those dofs at zero; `case(name).displacement` for named cases).
Predicates come in three flavours: `fix` and `displacement` test dof locations, `traction`, `spring` and `facets` test
facet midpoints, `boundary_facets_where` tests all vertices of a facet. `spring(facets, k)` is an elastic support spread over boundary facets with a stiffness per unit
area (vector per axis or matrix): a bearing pad, a soil bed. A model on springs alone needs horizontal springs too.
`facets(where, normal=…)` picks boundary facets by position and outward normal; `facet_integral` integrates over them
(the area of a bearing plate as meshed, to turn a bearing stiffness into a stiffness per area).

### Embedded bars and tendons

A `Locator` finds any point in its cell; the shape functions of that cell carry displacements to the point and forces
back to the nodes (exactly the transpose, so equilibrium and work are preserved). On that:

- `Bar(points, EA, bonded=True)`: perfectly bonded along its length, resampled finer than the cells it crosses (rebar,
  grouted tendons, bonded plates); `bonded=False`: attached only at the given points, one force over the whole length
  (external tendons through frictionless deviators, plates between their anchors). Adds stiffness; its force is
  EA·elongation/L plus any prestress of the case.
- `Tendon`: the equivalent-load method. At every vertex of the path the change of the tendon force acts on the concrete
  (P/R of curves, dP/ds of friction, P at the anchors, spread over a bearing plate with `anchor=(a, b)`); the loads add up
  to zero; the tendon adds no stiffness.
- `line_load` / `point_forces`: forces along a polyline or at points inside the solid, e.g. anchor bolts or a dowel.

Points are found through a grid of bins holding every cell whose bounding box overlaps the bin: each point is tested
against every cell that can contain it (exact also for elongated and curved cells); `snap` moves points just outside the
mesh onto the nearest cell (anchor plates poking out of a sloping end face), `outside="skip"` drops points outside it.

### Load cases and results

`case(name)` creates a load case; its methods add to the case's load vector and chain. The vector has an external part
(tractions, body forces, point and line forces, tendon loads — their recipes are kept for sections) and an internal,
self-equilibrated one (initial strains ∫Bᵀ D ε0, bar prestress −Gᵀ P); initial strains and prestress are remembered so
that the stress is D (ε − ε0) and the bar force EA δ/L + P. `scale_load` scales the external part only. `area_load` with
a `rect` print samples the print on a fine grid of points on the facets under it, so a wheel smaller than the cells is
placed right. `solve()` factorises once (SuperLU up to 150,000 dofs, then CHOLMOD up to 800,000 if scikit-sparse or
cholespy is installed, else one AMG hierarchy) and solves every case. A `CaseResult` holds the displacements and where they came
from; `+` and scalar `*` combine results (combinations of codes), and stresses of a combination subtract the combined
initial strains, bar forces add the combined prestress.

`section(origin, normal)` gives the force and moment the positive side exerts on the negative side of a plane through
cell faces (a plane cutting through cells raises). The default `method="nodal"` is statics of the positive part — the
external loads, reactions and spring forces on its nodes, plus the share of the loads of the cells next to the plane that
falls on the nodes in it — exact to solver precision and the right comparison with beam models; embedded bars crossing
the plane count with the solid, initial strains and prestress are internal. `method="traction"` integrates σ·n over the
interior faces in the plane (stress averaged from both sides, O(h²) accurate) plus the bars crossing it, optionally over a
part of the cut (`region`), e.g. the force in a bottom slab. M is the vector moment about `ref`: for a member along +x
with z up a sagging moment is M_y < 0, the opposite sign of `femotion.frame.post.chain_force`.

The single-case use (`Elastic(...).traction(...); .fix(...); .solve(); .nodal_stress()`) works on the case "default";
`s.u`, `s.reactions()` and the like refer to it, or to the only case; with several cases use `s.result(name)`. Reading
`s.f` creates no case. `ib` and `sb` (the vector and scalar bases) carry one quadrature point per cell — they are for dof
bookkeeping and interpolation; cell volumes are `cell_volumes`.

### Compared with `femotion.frame.loads`

| continuum (`LoadCase` bound to a solid model) | frame (`LoadCase` as data, applied to a model) |
|---|---|
| `displacement` | `settlement` |
| `thermal`, `initial_strain` | `imposed` |
| `bar_prestress` (a force) | `bar_strain` (a strain) |
| `tendon` with `continuum.tendon.Tendon` (a 3D path) | `tendon` with `frame.prestress.Tendon` (eccentricities along a chain of beams) |
| results combine: `1.2 * r1 + r2`, `Elastic.combine({name: k})` | cases combine: `1.2 * case1 + case2`, `combine((k, case), …)` |

### Swept meshes

`LayeredSection(y, ny, nz, solid, names)`: the cross-section as columns (breakpoints `y`, divisions `ny`) × layers
(divisions `nz`) between interface curves z_k(y); each block solid or void, with a cell group name. `sweep(section, xs,
z)` places the same quadrilateral grid at every station x with `z(x, y)` giving the interfaces there, and joins the
stations with hexahedra: an end thickening, a haunch or a variable depth is a smooth change of the interfaces, the mesh
stays all-hexahedral and conforming. `Mesh.subset(groups)` cuts the cells of some groups out (a construction stage
before the pavement is cast).

## Verification (`tests/test_continuum_solid.py`)

| check | reference | agreement |
|---|---|---|
| orthotropic with equal constants | isotropic law | 1e-12 |
| orthotropic lamina, plane stress | Q11, Q12, Q22, Q66 of lamination theory | 1e-12 |
| plane strain thermal strain | (1 + ν) α ΔT | exact |
| off-axis tension of a rotated orthotropic block (0°, 30°, 90°) | 1/Ex of classical lamination theory | 1e-8 |
| two materials in series | σ (L1/E1 + L2/E2) | 1e-10 |
| chunked assembly | one-chunk assembly | 1e-12 |
| 20-node hexahedra, end-loaded cantilever | Timoshenko beam | 2 % |
| stiff block on area springs | q / k, bed force = load | 1e-4 |
| two load cases, one factorisation | separate solves; superposition | 1e-10 |
| area, line and point loads | resultant and moment | 1e-12 |
| free thermal expansion | no stress, u = α ΔT L | 1e-10 |
| bimetal strip (Timoshenko 1925) | curvature | 1 % |
| embedded bonded bar in a stretched prism | N = (E A + Es As) ε, by statics and by traction | 1e-9 |
| bonded bars at ±e on a cantilever | bending stiffness E I + 2 EA e² | 2 % |
| bar tensioned in a free prism | N = P (1 − EAb / (EAc + EAb)) | 1e-5 |
| block held on every face, heated | σ = −E α ΔT / (1 − 2ν) | 1e-8 |
| plane strain: 3D free strain against temperature | same stresses | 1e-10 |
| sections under temperature (free, restrained, linear gradient) and bar prestress | N = 0, −E α ΔT A, M = 0, N = 0 | statics 1e-6, traction 1e-2 |
| cantilever under self-weight cut | V = w (L − a), M = w (L − a)²/2 by statics, 8- and 20-node hexahedra | 1e-9 |
| unbonded bar crossing a cut | (E A + EA_bar) δ / L by statics and by traction | 1e-9 |
| B-matrix assembly (hex 8 / 20 nodes, tet 10, quad 8 plane strain; rotated orthotropic) | scikit-fem `BilinearForm` with C_ijkl | 1e-12 |
| sheared, elongated hexahedra | every random interior point found and mapped back | 1e-9 |
| small wheel print inside large facets | resultant; point of application | 1e-12; 1 mm |
| prescribed displacements per case | E A δ / L | 1e-9 |
| submodel driven by a coarse cantilever | beam theory and the coarse model | 1 % |
| CHOLMOD against SuperLU | same displacements | 1e-7 |
| unbonded bar over a deviator | one force from the summed elongations | 1e-9 |
| fillet path, friction from both ends, arc push | R tan(Δ/2), P e^{−(μθ+kx)}, 2P sin(Δ/2) | 1e-4 – 2e-3 |
| straight eccentric tendon | −P/A ∓ P e y / I, no reactions | 0.5 % |
| parabolic tendon (load balancing) | camber 5wL⁴/384EI + shear, w = 8Pf/L²; section N = −P, M = P f | 2 %; 0.1–0.2 % |
| cantilever cut | V = P, M = P (L − a) by statics (1e-9) and by traction (3 %); half-section force M b³/8 / I (1 %) | |
| swept box: volume, groups, subset | exact areas | 1e-12 |
| swept box cantilever | beam theory + web shear | 3 % |
| end thickening sweep | volume integrated by hand | 1e-9 |
