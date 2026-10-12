"""Prestressing tendons in solids: the path, the force along it, and the loads it puts on the concrete.

A tendon is a polyline (curves finely sampled) with a tension in every segment. Its action on the concrete is the
change of the tendon force at every point of the path: at an interior vertex P_j t_j − P_(j−1) t_(j−1) (the radial push of
a curve, P/R per unit length, and friction, dP/ds), at the anchors P t into the member. These forces are applied at their
exact positions through the shape functions of the cells they fall in (femotion.continuum.embed), so the path does not
have to follow the mesh; they add up to zero. This is the equivalent-load method: the concrete feels the tendon, the
tendon adds no stiffness (add a bonded femotion.continuum.embed.Bar for that, e.g. for loads after grouting).

Paths as on the drawings: intersection points (PI) of straight tangents, each rounded with a radius.

    from femotion.continuum.tendon import Tendon
    td = Tendon.from_pis([(0.30, y, 0.90), (3.649, y, 0.096), (12.851, y, 0.096), (16.20, y, 0.90)],
                         radii=[12.259, 12.259])
    td = td.with_friction(P_jack=950e3, mu=0.25, k=0.0015, ends="both")   # force per segment, N
    s.case("预应力").tendon(td, anchor=(0.20, 0.20))                          # anchor force spread over a 0.2 × 0.2 plate

Forces are per segment (len(points) − 1). with_friction gives P_jack·exp(−(μθ + kx)) from the jacking end(s), θ the
total angle turned and x the length from that end (the usual friction law; codes add anchor set, elastic shortening,
relaxation, creep and shrinkage on top: scale or replace `force` with their results).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def fillet_path(pis, radii, arc_step: float = 0.05) -> np.ndarray:
    """a polyline through PI points with every interior corner rounded by a circular arc of the given radius (3D; the
    arc lies in the plane of the two tangents), sampled at most arc_step apart along the arcs"""
    P = np.asarray(pis, float)
    R = np.broadcast_to(np.asarray(radii, float), (len(P) - 2,))
    # tangent lengths R tan(Δ/2) of neighbouring bends must fit on the leg between them
    T = np.zeros(len(P))
    for j in range(1, len(P) - 1):
        a, b = P[j - 1] - P[j], P[j + 1] - P[j]
        cosang = np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
        T[j] = R[j - 1] * np.tan((np.pi - np.arccos(cosang)) / 2) if R[j - 1] > 0 else 0.0
    legs = np.linalg.norm(np.diff(P, axis=0), axis=1)
    over = np.where(T[:-1] + T[1:] > legs + 1e-9)[0]
    if len(over):
        k = over[0]
        raise ValueError(f"fillet_path: the bends at PI {k} and {k + 1} need {T[k] + T[k + 1]:.3f} of a leg {legs[k]:.3f} long")
    out = [P[:1]]
    prev_end = P[0]
    for j in range(1, len(P) - 1):
        a = P[j - 1] - P[j]
        b = P[j + 1] - P[j]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        ua, ub = a / la, b / lb
        cosang = np.clip(ua @ ub, -1.0, 1.0)
        turn = np.pi - np.arccos(cosang)  # deviation angle of the tendon at this PI
        if R[j - 1] <= 0 or turn < 1e-12:
            out.append(P[j][None])
            prev_end = P[j]
            continue
        T = R[j - 1] * np.tan(turn / 2)  # tangent length
        if T > la + 1e-9 or T > lb + 1e-9:
            raise ValueError(f"radius {R[j - 1]} too large at PI {j}: tangent length {T:.3f} exceeds a leg")
        t1 = P[j] + ua * T
        t2 = P[j] + ub * T
        bis = ua + ub
        C = P[j] + bis / np.linalg.norm(bis) * (R[j - 1] / np.cos(turn / 2))  # arc centre
        n = max(2, int(np.ceil(R[j - 1] * turn / arc_step)))
        v1, v2 = t1 - C, t2 - C
        w = v2 - (v2 @ v1) / (v1 @ v1) * v1
        w = w / np.linalg.norm(w) * np.linalg.norm(v1)
        ph = np.linspace(0, turn, n + 1)
        arc = C + np.cos(ph)[:, None] * v1 + np.sin(ph)[:, None] * w
        if np.linalg.norm(t1 - prev_end) > 1e-9:
            out.append(arc)
        else:
            out.append(arc[1:])
        prev_end = t2
    if np.linalg.norm(P[-1] - prev_end) > 1e-9:
        out.append(P[-1:])
    return np.concatenate(out)


@dataclass
class Tendon:
    points: np.ndarray  # (n, dim) path from the first anchor to the last
    force: np.ndarray | float = 0.0  # tension per segment (n − 1), or one value

    def __post_init__(self):
        self.points = np.asarray(self.points, float)

    @classmethod
    def from_pis(cls, pis, radii, force=0.0, arc_step: float = 0.05) -> "Tendon":
        return cls(fillet_path(pis, radii, arc_step), force)

    @property
    def segments(self) -> int:
        return len(self.points) - 1

    def lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points, axis=0), axis=1)

    def tangents(self) -> np.ndarray:
        d = np.diff(self.points, axis=0)
        return d / np.linalg.norm(d, axis=1)[:, None]

    def length(self) -> float:
        return float(self.lengths().sum())

    def turn_angles(self) -> np.ndarray:
        """angle turned at each interior vertex (n − 2)"""
        t = self.tangents()
        return np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0))

    def forces(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.force, float), (self.segments,)).copy()

    def with_friction(self, P_jack: float, mu: float, k: float, ends: str = "both") -> "Tendon":
        """tension per segment after friction during jacking: P_jack·exp(−(μθ + kx)) at the segment midpoints, from the
        start, the end, or the larger of both (jacked at both ends)"""
        L = self.lengths()
        th = np.r_[0.0, self.turn_angles()]  # angle turned before entering segment i, from the start
        x_mid = np.cumsum(L) - L / 2
        th_start = np.cumsum(th)  # total turned up to the start of segment i (arcs sampled finely: the midpoint value)
        P_s = P_jack * np.exp(-(mu * th_start + k * x_mid))
        th_rev = np.cumsum(np.r_[0.0, self.turn_angles()[::-1]])[::-1]
        x_rev = L.sum() - x_mid
        P_e = P_jack * np.exp(-(mu * th_rev + k * x_rev))
        P = {"start": P_s, "end": P_e, "both": np.maximum(P_s, P_e)}[ends]
        return replace(self, force=P)

    def elongation(self, Ep_Ap: float) -> float:
        """elongation of the tendon under its forces, Σ P L / (Ep Ap) (compare with the jacking record)"""
        return float(np.sum(self.forces() * self.lengths()) / Ep_Ap)

    def vertex_forces(self) -> np.ndarray:
        """force on the concrete at every vertex (n, dim): P_j t_j − P_(j−1) t_(j−1); sums to zero"""
        P, t = self.forces(), self.tangents()
        F = np.zeros_like(self.points)
        F[:-1] += P[:, None] * t
        F[1:] -= P[:, None] * t
        return F


def anchor_points(center, direction, size, up=(0.0, 0.0, 1.0), n: int = 5) -> np.ndarray:
    """n × n points on a rectangular bearing plate (size = (a, b)) perpendicular to `direction` at `center`; the plate
    side a is along `up` projected onto the plate, b across. In 2D: n points on a line of length a across the tendon"""
    t = np.asarray(direction, float)
    t = t / np.linalg.norm(t)
    if len(t) == 2:
        u = np.array([-t[1], t[0]])
        g = ((np.arange(n) + 0.5) / n - 0.5) * size[0]
        return np.asarray(center, float) + g[:, None] * u
    u = np.asarray(up, float) - (np.asarray(up, float) @ t) * t
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(t, (1.0, 0.0, 0.0)) if abs(t[0]) < 0.9 else np.cross(t, (0.0, 1.0, 0.0))
    u = u / np.linalg.norm(u)
    w = np.cross(t, u)
    a, b = size
    g = (np.arange(n) + 0.5) / n - 0.5
    A, B = np.meshgrid(g * a, g * b, indexing="ij")
    return np.asarray(center, float) + A.reshape(-1, 1) * u + B.reshape(-1, 1) * w
