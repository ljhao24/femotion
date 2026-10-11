"""Frames: beams, trusses, cables and whole-bridge models.

    model     the modelling format shared by every frame solver (Model, Section, local axes, lumped masses)
    linear    fast linear direct-stiffness solver (factorise once, solve many load cases)
    opensees  OpenSees back-end: geometric and material nonlinearity, pushover, time histories, member removal
    check     crosscheck(model): run both solvers on the same model before trusting the fast one
    sections  sections of any shape through sectionproperties: elastic properties and fiber sections (extra "sections")
    post      reactions, internal-force diagrams, mode classification on plain arrays (works with either solver)
    influence influence lines of many sections at once, lane-load and axle-train envelopes, exact distributed-load forces
    prestress tendons: equivalent loads, total / primary / secondary effects on statically indeterminate frames
    thermal   imposed strains and curvatures (temperature, shrinkage): secondary effects on statically indeterminate frames
    stages    construction stages and creep / shrinkage step by step in time (linear): members, supports, loads by stage

Linear problems solved many times go to `linear` once `check.crosscheck` passes; anything nonlinear goes to `opensees`.
"""
from .linear import Solver, pretension_influence  # noqa: F401
from .model import Model, Section, local_axes, lumped_masses  # noqa: F401
