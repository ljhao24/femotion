"""Derived stress quantities: principal stresses and directions, von Mises. Component order as in femotion.continuum:
2D (xx, yy, xy), 3D (xx, yy, zz, xy, yz, zx); inputs may be (N, ncomp) arrays or component tuples."""
from __future__ import annotations

import numpy as np


def _split(S):
    S = np.asarray(S)
    return [S[..., i] for i in range(S.shape[-1])]


def principal_2d(S):
    """returns (σ1, σ2, θ), θ being the angle between the σ1 direction and the x axis"""
    sxx, syy, sxy = _split(S)
    c = 0.5 * (sxx + syy)
    r = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy**2)
    th = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    return c + r, c - r, th


def principal_3d(S):
    """returns (the three principal stresses, largest first (N, 3), principal directions (N, 3, 3) as column vectors)"""
    xx, yy, zz, xy, yz, zx = _split(S)
    T = np.stack([np.stack([xx, xy, zx], -1), np.stack([xy, yy, yz], -1), np.stack([zx, yz, zz], -1)], -2)
    w, V = np.linalg.eigh(T)
    return w[..., ::-1], V[..., ::-1]


def von_mises(S):
    """2D (plane stress) or 3D, chosen by the number of components"""
    S = np.asarray(S)
    if S.shape[-1] == 3:
        sxx, syy, sxy = _split(S)
        return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    xx, yy, zz, xy, yz, zx = _split(S)
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + yz**2 + zx**2))


def along(S, d):
    """normal stress along the unit direction d: dᵀσd"""
    S = np.asarray(S)
    d = np.asarray(d, float)
    if S.shape[-1] == 3:
        sxx, syy, sxy = _split(S)
        return sxx * d[0] ** 2 + syy * d[1] ** 2 + 2 * sxy * d[0] * d[1]
    xx, yy, zz, xy, yz, zx = _split(S)
    return (xx * d[0] ** 2 + yy * d[1] ** 2 + zz * d[2] ** 2
            + 2 * (xy * d[0] * d[1] + yz * d[1] * d[2] + zx * d[2] * d[0]))
