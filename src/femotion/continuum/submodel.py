"""Submodels: a finely meshed part of a structure, driven on its cut boundary by the displacements of a coarser model of the
whole (the usual way to look at an anchorage, a bolt group or an opening without meshing the whole structure finely).

    from femotion.continuum.submodel import drive
    sub = Elastic(fine_part_mesh, materials=..., order=2, serendipity=True)
    cut = lambda X: np.isclose(X[0], 1.2) | np.isclose(X[0], 3.2) | np.isclose(X[1], 0.9) | np.isclose(X[1], 2.7)
    for name, r in global_results.items():
        drive(sub, r, cut, name)                      # boundary displacements of this case
        apply_loads_inside(sub.case(name))            # the loads of the case that act inside the part (outside="skip")
    sub.solve()

The cut must lie far enough from the detail for the coarse displacements to be right there (Saint-Venant: about one
member depth or two cell sizes of the coarse model), and every load acting inside the part must be applied to the
submodel too: body forces of its cells, tractions on its own faces, point and line loads inside it (point_forces /
line_load / tendon with outside="skip" keep only those). Displacements are interpolated from the coarse model at the
submodel's boundary nodes (femotion.continuum.results.CaseResult.sample).
"""
from __future__ import annotations


def drive(sub, parent, where, case: str):
    """prescribe on the dofs of `sub` located where `where(X)` holds the displacements of the parent result (a CaseResult
    of a model that contains the part); returns the case of `sub`"""
    return sub.case(case).displacement(where, lambda X: parent.sample(X.T)["u"].T)
