"""femotion: finite-element and simulation results made for animation.

Organised by what is being computed; every domain can have several solver back-ends,
and all of them share the same mesh type and the same result file:

    mesh/        unified Mesh type; structured grids; gmsh meshing (extra "mesh"); fTetWild (extra "tetwild")
    frame/       beams, trusses, cables, whole-bridge models: model format, fast linear solver,
                 OpenSees back-end (extra "opensees"), sections of any shape (extra "sections"),
                 post-processing, cross-check
    continuum/   plane / solid linear elasticity on scikit-fem (extra "continuum")
    topopt/      2D SIMP topology optimisation that records every iteration
    cfd/         2D lattice Boltzmann flow on Apple GPUs (extra "cfd")
    post/        principal stresses, stress trajectories, stiffness-matrix patterns
    dynamics     modal superposition (step release, ground motion, moving loads)
    result       the fem-result/1 file format (docs/result-format.md), read by the JS packages
    io           other formats through meshio: read / write meshes, export results for ParaView (extra "io")
    viz          engineering drawings and charts as plain SVG (Jupyter, .svg, HTML pages), no plotting library
    models/      ready-made illustrative structures
    codes/       design-code checks: Chinese highway bridge codes (codes.jtg: JTG D60-2015, JTG 3362-2018)

This module imports nothing heavy; import the sub-modules you need.
"""

__version__ = "0.1.0.dev0"
