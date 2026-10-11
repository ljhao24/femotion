# Integrations

femotion leans on mature open-source tools and adds glue, a uniform mesh and model format, and the `fem-result/1` file.
This page lists what is wired in, and what has been evaluated but is left until a concrete need comes up, with the gap
each one would fill. Licences matter because femotion is AGPL-3.0-or-later: optional extras may be GPL or non-commercial, but nothing is
bundled; each is installed from its own distribution under its own licence.

## In use

| Tool | Module (extra) | Used for | Licence |
|---|---|---|---|
| NumPy, SciPy | everywhere (core) | arrays, sparse direct solvers, eigenvalues | BSD |
| gmsh | `femotion.mesh.gmsh` (`mesh`) | triangles, quadrilaterals (frontal or quasi-structured), tetrahedra, extruded and structured hexahedra, STEP / IGES import | GPL-2.0-or-later |
| fTetWild (pytetwild) | `femotion.mesh.tetwild` (`tetwild`) | tetrahedra from messy surface models: Blender scenes, downloads, scans | MPL-2.0 (pytetwild imports PyVista, so the extra installs it) |
| scikit-fem, pyamg | `femotion.continuum` (`continuum`) | plane and solid linear elasticity, algebraic multigrid | BSD-3, MIT |
| OpenSeesPy | `femotion.frame.opensees` (`opensees`) | nonlinear frames: plasticity, buckling, large displacements, time histories, member loss | free for education, research and non-profit use |
| sectionproperties | `femotion.frame.sections` (`sections`) | properties and fiber sections of cross-sections of any shape | MIT; its mesher Triangle is non-commercial |
| meshio | `femotion.io` (`io`) | reading and writing ~30 mesh formats, results for ParaView | MIT |
| MLX | `femotion.cfd` (`cfd`) | 2D lattice Boltzmann flow on Apple GPUs | MIT |

## Candidates

Integrate when a model needs them; each would get its own extra.

| Tool | Gap it fills | Notes |
|---|---|---|
| [pycba](https://github.com/ccaprani/pycba) | continuous beams with influence lines, vehicle libraries and load envelopes, validated against textbook cases | AGPL-3.0-or-later; overlaps `frame.linear` (influence lines are one factorisation and many back-substitutions), so only for its vehicle and envelope tools |
| [ospgrillage](https://github.com/MonashSmartStructures/ospgrillage) | grillage models of bridge decks on OpenSeesPy, with lanes and vehicle loads | MIT |
| [concreteproperties](https://github.com/robbievanleeuwen/concrete-properties) | moment–curvature and interaction diagrams of reinforced concrete sections | MIT; same author and geometry as sectionproperties |
| scikit-fem adaptive refinement | error-driven refinement of triangles and tetrahedra (`refine` on marked cells) | already installed with the `continuum` extra; the simplest way to show a mesh refining where stresses concentrate |
| [Mmg](https://www.mmgtools.org) (`mmgpy`) | metric-based, anisotropic adaptive remeshing of 2D, 3D and surface meshes, driven by a computed field | Mmg LGPL-3.0, wrapper MIT; the Python wrapper is young |
| [build123d](https://github.com/gumyr/build123d) | parametric CAD on OpenCASCADE for detailed solids (fillets, lofts, sweeps): cable saddles, anchor boxes, curved towers → STEP → `mesh.gmsh.step` | Apache-2.0 |
| [pyacvd](https://github.com/pyvista/pyacvd), [Instant Meshes](https://github.com/wjakob/instant-meshes) (`pyinstantmeshes`) | uniform triangles or clean quadrilaterals on surface models, mostly for display | MIT, BSD-3 |
| [Netgen](https://ngsolve.org) (`netgen-mesher`) | an alternative CAD mesher (curved elements) for models gmsh cannot handle | LGPL-2.1 |
| [PyVista](https://pyvista.org) | VTK filters in Python: 3D streamlines, iso-surfaces, slices of large solids, reading OpenFOAM cases | MIT; already installed by the `tetwild` extra, not used by femotion's own code |
| [ParaView](https://www.paraview.org) | interactive viewer | BSD-3; nothing to integrate — `femotion.io.export_result` writes VTU and `.pvd` series |
| [OpenFOAM](https://www.openfoam.com) | 3D flow (vortex shedding, flutter-related flows) when 2D lattice Boltzmann is not enough | GPL-3.0; external program, results through PyVista or meshio |
| [Code_Aster](https://code-aster.org), [CalculiX](http://www.calculix.de) | nonlinear continuum: plasticity, contact, fracture (gusset plates, eyebars, joints) | GPL; external programs, meshes through meshio |
| [opstool](https://github.com/yexiang92/opstool) | inspecting and plotting OpenSeesPy models while building them | GPL-3.0; debugging only |
| [Project Chrono](https://projectchrono.org) | multibody dynamics and collisions: debris after a collapse | BSD-3 |
| [SUMO](https://eclipse.dev/sumo/) | traffic simulation as realistic moving loads | EPL-2.0 |
| [PEER NGA-West2](https://ngawest2.berkeley.edu) | recorded ground motions for time histories | data, free registration |

## Considered and not planned

| Tool | Why not |
|---|---|
| TetGen (`tetgen` on PyPI) | TetGen itself is AGPL-3.0 (the wrapper's MIT licence does not cover it); gmsh and fTetWild already produce tetrahedra |
| PyMeshLab | GPL-3.0 and large; pyacvd covers surface remeshing |
| CGAL / pygalmesh | GPL, no binary wheels |
| automatic all-hexahedral meshing of arbitrary solids | no mature open-source tool (Coreform Cubit is commercial); use block assemblies (`mesh.structured.hex_block` + `weld`), `mesh.gmsh.hexes` for box-like volumes or `mesh.gmsh.extrude` for prismatic members |
