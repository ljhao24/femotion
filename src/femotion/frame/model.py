"""Frame model (the modelling format): nodes + beams + bars (cables) + spring links + supports + sections + mass rule.
No solving happens here.

Both frame solvers read it: femotion.frame.linear (fast, linear) and femotion.frame.opensees (OpenSees, linear and nonlinear).
The conventions the two must share are written once, here:
- local_axes: local x along the member; local z = the reference vector (default: global up) projected onto the plane
  normal to the member; y = z × x (the same convention as OpenSees' vecxz).
  Section Iz governs bending in the local x-y plane, Iy bending in the x-z plane (with the default reference, Iy is vertical bending).
- lumped_masses: element mass split equally between its two end nodes.

Units: m, kN, t (mass), s; E in kPa (kN/m²), loads in kN.
Model.to_mesh() turns the model into a line2 Mesh for display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Section:
    E: float  # kPa
    A: float  # m²
    Iy: float = 0.0  # about local y (bending in the x-z plane; vertical bending with the default reference vector)
    Iz: float = 0.0  # about local z (bending in the x-y plane)
    J: float = 0.0
    G: float | None = None
    mass: float = 0.0  # t/m (including superimposed dead load)
    rot_mass: float = 0.0  # t·m²/m, rotary inertia about the member axis (torsional modes)

    def g(self) -> float:
        return self.G if self.G is not None else self.E / 2.6


@dataclass
class Model:
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    beams: list = field(default_factory=list)  # (i, j, Section, ref, tag)
    trusses: list = field(default_factory=list)  # (i, j, Section, tag)
    links: list = field(default_factory=list)  # (i, j, (kx, ky, kz), (rx, ry, rz) or None, tag)
    fixed: dict = field(default_factory=dict)  # node -> list of 6 bools

    def node(self, p) -> int:
        self.X = np.vstack([self.X, np.asarray(p, float)[None, :]])
        return len(self.X) - 1

    def beam(self, i, j, sec: Section, ref=(0, 0, 1), tag=""):
        self.beams.append((i, j, sec, np.asarray(ref, float), tag))
        return len(self.beams) - 1

    def truss(self, i, j, sec: Section, tag=""):
        self.trusses.append((i, j, sec, tag))
        return len(self.trusses) - 1

    def link(self, i, j, k=(1e9, 1e9, 1e9), kr=None, tag=""):
        self.links.append((i, j, tuple(k), kr, tag))

    def fix(self, n, dofs=(1, 1, 1, 1, 1, 1)):
        old = self.fixed.get(n, [0] * 6)
        self.fixed[n] = [bool(a or b) for a, b in zip(old, dofs)]

    @property
    def ndof(self):
        return 6 * len(self.X)

    def to_mesh(self, tags=None):
        """beams and bars become line2 cells (beams first); cell_groups are keyed by tag ("beam:deck", "truss:cable", …)"""
        from ..mesh.core import Mesh

        el = [(i, j, "beam:" + tag) for (i, j, sec, ref, tag) in self.beams] + [(i, j, "truss:" + tag) for (i, j, sec, tag) in self.trusses]
        c = np.array([(i, j) for i, j, _ in el], np.int64).reshape(-1, 2)
        groups = {}
        for k, (_, _, g) in enumerate(el):
            groups.setdefault(g, []).append(k)
        return Mesh(self.X.copy(), {"line2": c}, {}, {g: {"line2": np.array(v)} for g, v in groups.items()})


def local_axes(xi, xj, ref):
    """returns (member length L, 3×3 direction cosines R): the rows of R are the local x, y, z axes in global coordinates"""
    ex = xj - xi
    L = np.linalg.norm(ex)
    ex = ex / L
    r = ref / np.linalg.norm(ref)
    if abs(np.dot(r, ex)) > 0.999:
        r = np.array([1.0, 0.0, 0.0]) if abs(ex[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ez = r - np.dot(r, ex) * ex  # local z: the reference direction (vertical by default)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    return L, np.vstack([ex, ey, ez])


def lumped_masses(m: Model, drop: set[int] | None = None, rot_mass_floor: float = 1e-3) -> np.ndarray:
    """lumped mass of every degree of freedom (ndof): element mass split to both ends; a beam's rotary inertia about its own
    axis projected onto the three global rotations; nodes without mass get 0.5 t (e.g. the anchor of a broken cable) and
    rotational masses get a floor (max translational mass × rot_mass_floor) so that the mass matrix is positive definite.
    drop: bars to leave out (broken cables)."""
    Md = np.zeros(m.ndof)
    for (i, j, sec, ref, tag) in m.beams:
        L, R = local_axes(m.X[i], m.X[j], ref)
        mm = sec.mass * L / 2
        for nd in (i, j):
            Md[6 * nd:6 * nd + 3] += mm
            Md[6 * nd + 3:6 * nd + 6] += sec.rot_mass * L / 2 * R[0] ** 2
    for e, (i, j, sec, tag) in enumerate(m.trusses):
        if drop and e in drop:
            continue
        mm = sec.mass * np.linalg.norm(m.X[j] - m.X[i]) / 2
        Md[6 * i:6 * i + 3] += mm
        Md[6 * j:6 * j + 3] += mm
    tm = Md.reshape(-1, 6)
    tm[:, :3] = np.maximum(tm[:, :3], 0.5)
    floor = rot_mass_floor * np.maximum(tm[:, :3].max(axis=1, initial=0), 1e-6)
    for a in range(3, 6):
        tm[:, a] = np.maximum(tm[:, a], floor)
    return tm.ravel()
