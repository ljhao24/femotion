"""Influence lines, moving-load envelopes and exact section forces under distributed loads (fast linear solver).

A unit load walks along a path of nodes (the deck). The stiffness is factorised once, so one batch of back-substitutions
gives the influence line of every section at once. Sections are beam ends, (beam, "i") or (beam, "j"): with nodes at the
supports and under the loads, the end of an element is where the internal forces jump, so both sides of a node are
separate sections.

    from femotion.frame.influence import InfluenceLines, chain_sections, uniform_load_forces
    secs = chain_sections(deck_beams)                    # both ends of every deck element, in order
    il = InfluenceLines(sv, deck_nodes, secs)
    Mmax, Mmin = il.lane("My", q=10.5, P=300.0)         # uniform load on the same-sign parts + one point load at the peak
    Vmax, Vmin = il.axles("Vz", [30, 120, 120, 140, 140], [3.0, 1.4, 7.0, 1.4])   # a vehicle, both directions
    Mg = uniform_load_forces(sv, secs, "My")             # self-weight (section mass × g), exact at the element ends

Components and signs are those of femotion.frame.post.chain_force: "N" (tension +), "Vy", "Vz", "T", "My" (vertical
bending, sagging + with the default reference vector), "Mz".

Shear and axial force jump by the load where a section's node is loaded. The influence line of such a section gets two
ordinates at that position (load just outside and just inside the element), so the point load is placed on the
correct side. Between nodes the line is taken as linear: refine the deck elements (the true line of a beam element is
cubic between its nodes).
"""
from __future__ import annotations

import numpy as np

from .linear import Solver, beam_k
from .model import local_axes
from .post import _COMP

_FORCE = ("N", "Vy", "Vz")


def _sign(comp: str, end: str) -> int:
    """section force = sign × element end force (as femotion.frame.post.chain_force)"""
    si, sj = (-1, 1) if comp in ("N", "T", "Vy", "Vz") else (1, -1)
    return si if end == "i" else sj


def chain_sections(beam_ids) -> list[tuple[int, str]]:
    """both ends of every element of a chain, in order: [(b0, 'i'), (b0, 'j'), (b1, 'i'), …]"""
    return [(int(b), e) for b in beam_ids for e in ("i", "j")]


def section_forces(sv: Solver, U: np.ndarray, sections, comp: str, end_forces: np.ndarray | None = None) -> np.ndarray:
    """section forces for a displacement field U under nodal loads only (end_forces: precomputed Solver.beam_end_forces)"""
    ef = sv.beam_end_forces(U) if end_forces is None else end_forces
    a, b = _COMP[comp]
    return np.array([_sign(comp, end) * ef[e][a if end == "i" else b] for e, end in sections])


def _fixed_end(p: np.ndarray, L: float) -> np.ndarray:
    """equivalent nodal loads (local, 12) of a uniform load p (local x, y, z per length) on a beam of length L"""
    f = np.zeros(12)
    f[[0, 6]] = p[0] * L / 2
    f[[1, 7]] = p[1] * L / 2
    f[[2, 8]] = p[2] * L / 2
    m_y, m_z = p[2] * L * L / 12, p[1] * L * L / 12
    f[5], f[11] = m_z, -m_z  # v–θz plane
    f[4], f[10] = -m_y, m_y  # w–θy plane: signs opposite (see femotion.frame.linear.beam_k)
    return f


def uniform_load_forces(sv: Solver, sections, comp: str, w=None, g: float = 9.81, with_trusses: bool = True):
    """section forces under uniform loads along the beams, exact at the element ends (consistent equivalent loads plus the
    fixed-end correction, no lumping error). w: (n_beams, 3) global load per length in kN/m; default self-weight,
    section mass × g downwards, plus the lumped weight of the bars (cables) when with_trusses.
    Returns the section forces, or (forces, U) with U the displacements when comp is None."""
    m = sv.m
    nb = len(m.beams)
    self_weight = w is None
    if self_weight:
        w = np.zeros((nb, 3))
        w[:, 2] = -np.array([sec.mass for (_, _, sec, _, _) in m.beams]) * g
    w = np.asarray(w, float)
    F = np.zeros(m.ndof)
    feq = np.zeros((nb, 12))
    for e, (i, j, sec, ref, tag) in enumerate(m.beams):
        L, R, T = sv._beam_cache[e]
        feq[e] = _fixed_end(R @ w[e], L)
        F[np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]] += T.T @ feq[e]
    if with_trusses and self_weight:
        for (i, j, sec, tag) in m.trusses:
            half = sec.mass * np.linalg.norm(m.X[j] - m.X[i]) / 2 * g
            F[6 * i + 2] -= half
            F[6 * j + 2] -= half
    U = sv.solve(F)
    ef = sv.beam_end_forces(U) - feq
    if comp is None:
        return ef, U
    return section_forces(sv, U, sections, comp, end_forces=ef)


class InfluenceLines:
    """influence lines of the given sections for a unit load (direction, default straight down) at each node of `path`.
    path: ordered node ids (the loaded lane); positions s are the distances along it (m)."""

    def __init__(self, sv: Solver, path, sections, direction=(0.0, 0.0, -1.0)):
        self.sv, self.sections = sv, [(int(b), str(e)) for b, e in sections]
        m = sv.m
        self.path = np.asarray(path, int)
        X = m.X[self.path]
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
        self.direction = np.asarray(direction, float)
        F = np.zeros((m.ndof, len(self.path)))
        for k, nd in enumerate(self.path):
            F[6 * nd:6 * nd + 3, k] = self.direction
        U = np.zeros_like(F)
        U[sv.free] = sv._lu.solve(F[sv.free])
        # end forces of the section beams for every load position: (n_sections, 12, n_pos)
        self._ef = {}
        for e in {b for b, _ in self.sections}:
            i, j, sec, ref, tag = m.beams[e]
            L, R, T = sv._beam_cache[e]
            dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            self._ef[e] = (beam_k(sec, L) @ T) @ U[dofs]
        self._pos = {int(nd): k for k, nd in enumerate(self.path)}

    def line(self, k: int, comp: str):
        """(s, η) of section k: positions along the path and ordinates, with two ordinates where the section's own node is
        loaded and the force jumps"""
        e, end = self.sections[k]
        i, j, sec, ref, tag = self.sv.m.beams[e]
        a, b = _COMP[comp]
        sg = _sign(comp, end)
        eta = sg * self._ef[e][a if end == "i" else b]
        s = self.s
        here, other = (i, j) if end == "i" else (j, i)
        if comp in _FORCE and here in self._pos and other in self._pos:
            kh, ko = self._pos[here], self._pos[other]
            L, R = local_axes(self.sv.m.X[i], self.sv.m.X[j], ref)
            inside = eta[kh] - sg * (R @ self.direction)[a]  # load just inside the element: end force minus the load
            if ko > kh:  # the element lies after the node along the path
                s = np.insert(s, kh + 1, s[kh]); eta = np.insert(eta, kh + 1, inside)
            else:
                s = np.insert(s, kh, s[kh]); eta = np.insert(eta, kh, inside)
        return s, eta

    def lines(self, comp: str):
        return [self.line(k, comp) for k in range(len(self.sections))]

    def lane(self, comp: str, q: float, P: float):
        """(max, min) of every section under a uniform load q (per length) on the parts of the line of one sign plus a
        single point load P at the largest ordinate of that sign"""
        mx, mn = np.zeros(len(self.sections)), np.zeros(len(self.sections))
        for k in range(len(self.sections)):
            s, eta = self.line(k, comp)
            ap, an = _signed_areas(s, eta)
            mx[k] = P * max(eta.max(), 0.0) + q * ap
            mn[k] = P * min(eta.min(), 0.0) + q * an
        return mx, mn

    def axles(self, comp: str, loads, spacings, both_ways: bool = True):
        """(max, min) of every section under a train of point loads (loads[0] leading, spacings between consecutive axles)
        moving along the whole path, in both directions; axles off the path carry nothing. Exact for the piecewise-linear
        line: an extreme always has some axle on a vertex, and every such position is tried."""
        loads = np.asarray(loads, float)
        off = np.r_[0.0, np.cumsum(spacings)]
        trains = [off, off[-1] - off] if both_ways else [off]
        mx, mn = np.zeros(len(self.sections)), np.zeros(len(self.sections))
        for k in range(len(self.sections)):
            s, eta = self.line(k, comp)
            best_hi, best_lo = 0.0, 0.0
            for o in trains:
                x0 = (s[:, None] - o[None, :]).ravel()  # lead-axle positions putting some axle on a vertex
                for side in (-1e-9, 1e-9):  # both sides of jumps
                    x = x0[:, None] + o[None, :] + side
                    v = (_interp(x, s, eta) * loads[None, :]).sum(axis=1)
                    best_hi, best_lo = max(best_hi, v.max()), min(best_lo, v.min())
            mx[k], mn[k] = best_hi, best_lo
        return mx, mn


def _interp(x, s, eta):
    """piecewise-linear line, zero off the path; at a jump the left value is used left of it, the right value right of it"""
    idx = np.searchsorted(s, x, side="right")  # first vertex beyond x
    idx = np.clip(idx, 1, len(s) - 1)
    s0, s1 = s[idx - 1], s[idx]
    e0, e1 = eta[idx - 1], eta[idx]
    t = np.where(s1 > s0, (x - s0) / np.where(s1 > s0, s1 - s0, 1.0), 0.0)
    v = e0 + t * (e1 - e0)
    return np.where((x < s[0]) | (x > s[-1]), 0.0, v)


def _signed_areas(s, eta):
    """(∫ max(η, 0) ds, ∫ min(η, 0) ds) of a piecewise-linear line, splitting segments where they cross zero"""
    ds = np.diff(s)
    a, b = eta[:-1], eta[1:]
    pos = np.zeros_like(ds)
    neg = np.zeros_like(ds)
    same = a * b >= 0
    avg = (a + b) / 2 * ds
    pos[same] = np.where(avg[same] > 0, avg[same], 0.0)
    neg[same] = np.where(avg[same] < 0, avg[same], 0.0)
    cr = ~same
    if cr.any():
        aa, bb, d = a[cr], b[cr], ds[cr]
        t = aa / (aa - bb)  # zero crossing
        left = aa * t * d / 2
        right = bb * (1 - t) * d / 2
        pos[cr] = np.maximum(left, 0) + np.maximum(right, 0)
        neg[cr] = np.minimum(left, 0) + np.minimum(right, 0)
    return pos.sum(), neg.sum()
