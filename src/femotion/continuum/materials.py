"""Linear elastic materials of femotion.continuum: isotropic and orthotropic, with thermal expansion.

    from femotion.continuum import Isotropic, Orthotropic
    concrete = Isotropic(E=3.0e10, nu=0.2, alpha=1e-5)
    cfrp = Orthotropic.transversely_isotropic(EL=165e9, ET=10e9, nuLT=0.3, nuTT=0.4, GLT=5e9, axis=(1, 0, 0))
    D = cfrp.D()                      # 6 × 6, global axes

Voigt order as everywhere in femotion.continuum: 3D (xx, yy, zz, xy, yz, zx), 2D (xx, yy, xy); strains are engineering
strains (γxy = 2εxy), so σ = D ε with no factors on the shear terms. A material is defined in its own axes 1, 2, 3; `axes`
holds them as the rows of a rotation matrix (axis 1 in global coordinates first). Plane problems reduce the 3D law:
plane stress with σzz = σyz = σzx = 0, plane strain with εzz = εyz = εzx = 0 (the material's axis 3 should then be the
global z axis).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))  # 3D component order
PLANE = (0, 1, 3)  # xx, yy, xy within the 3D order


def _tensor(D6: np.ndarray) -> np.ndarray:
    """6 × 6 Voigt stiffness (engineering strains) → 3 × 3 × 3 × 3 tensor with the minor symmetries"""
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT):
        for J, (k, l) in enumerate(VOIGT):
            for a, b in {(i, j), (j, i)}:
                for c, d in {(k, l), (l, k)}:
                    C[a, b, c, d] = D6[I, J]
    return C


def _voigt(C: np.ndarray) -> np.ndarray:
    return np.array([[C[i, j, k, l] for (k, l) in VOIGT] for (i, j) in VOIGT])


def _reduce(D6: np.ndarray, a6: np.ndarray, dim: int, kind: str):
    """the constitutive matrix and the thermal strain vector of a plane problem (or 3D unchanged)"""
    if dim == 3:
        return D6, a6
    p = list(PLANE)
    o = [2, 4, 5]
    if kind == "plane_stress":
        S = np.linalg.inv(D6)
        return np.linalg.inv(S[np.ix_(p, p)]), a6[p]
    if kind == "plane_strain":
        Dpp, Dpo = D6[np.ix_(p, p)], D6[np.ix_(p, o)]
        # σp = Dpp (εp − αp) − Dpo (0 − αo): an equivalent in-plane free strain αp + Dpp⁻¹ Dpo αo
        return Dpp, a6[p] + np.linalg.solve(Dpp, Dpo @ a6[o])
    raise ValueError(f"unknown kind {kind!r} (plane_stress, plane_strain, solid)")


def _reduce_strain(D6: np.ndarray, e6: np.ndarray, dim: int, kind: str) -> np.ndarray:
    """a free strain (6, …) seen by a plane problem: the in-plane part (plane stress), plus Dpp⁻¹ Dpo εo (plane strain)"""
    e6 = np.asarray(e6, float)
    if dim == 3:
        return e6
    p, o = list(PLANE), [2, 4, 5]
    if kind == "plane_stress":
        return e6[p]
    corr = np.linalg.solve(D6[np.ix_(p, p)], D6[np.ix_(p, o)])
    return e6[p] + np.einsum("ij,j...->i...", corr, e6[o])


@dataclass(frozen=True)
class Isotropic:
    """E (any consistent unit), Poisson's ratio nu, coefficient of thermal expansion alpha (per degree)"""
    E: float
    nu: float
    alpha: float = 0.0

    isotropic = True

    def lame(self, kind: str = "solid") -> tuple[float, float]:
        """Lamé constants; plane stress uses the reduced λ* = 2λμ / (λ + 2μ)"""
        lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        if kind == "plane_stress":
            lam = 2 * lam * mu / (lam + 2 * mu)
        return lam, mu

    def D6(self) -> np.ndarray:
        lam, mu = self.lame()
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[[0, 1, 2], [0, 1, 2]] += 2 * mu
        D[[3, 4, 5], [3, 4, 5]] = mu
        return D

    def alpha6(self) -> np.ndarray:
        return np.r_[np.full(3, self.alpha), np.zeros(3)]

    def D(self, dim: int = 3, kind: str = "solid") -> np.ndarray:
        """constitutive matrix (6 × 6 in 3D, 3 × 3 in 2D)"""
        return _reduce(self.D6(), self.alpha6(), dim, kind)[0]

    def thermal_strain(self, dim: int = 3, kind: str = "solid") -> np.ndarray:
        """free strain per degree, Voigt (the in-plane equivalent in plane strain)"""
        return _reduce(self.D6(), self.alpha6(), dim, kind)[1]

    def reduce_strain(self, e6, dim: int = 3, kind: str = "solid") -> np.ndarray:
        """a 3D free strain (6, …) as seen by a plane problem"""
        return _reduce_strain(self.D6(), e6, dim, kind)


@dataclass(frozen=True, eq=False)
class Orthotropic:
    """moduli E = (E1, E2, E3), Poisson's ratios nu = (ν12, ν13, ν23) (νij: contraction along j under stress along i,
    νji = νij Ej / Ei), shear moduli G = (G12, G13, G23), expansion coefficients alpha = (α1, α2, α3), material axes
    `axes` (3 × 3, rows = axes 1, 2, 3 in global coordinates; orthonormalised)"""
    E: tuple
    nu: tuple
    G: tuple
    alpha: tuple = (0.0, 0.0, 0.0)
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))

    isotropic = False

    def __post_init__(self):
        A = np.asarray(self.axes, float)
        q, r = np.linalg.qr(A.T)  # Gram-Schmidt on the rows, keeping their direction
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 2] *= -1
        object.__setattr__(self, "axes", q.T)
        if min(self.E) <= 0 or min(self.G) <= 0:
            raise ValueError("Orthotropic: moduli must be positive")
        if not np.all(np.linalg.eigvalsh(self._S_local()) > 0):
            raise ValueError("Orthotropic: the compliance is not positive definite (check the Poisson's ratios)")

    @classmethod
    def transversely_isotropic(cls, EL: float, ET: float, nuLT: float, nuTT: float, GLT: float, alphaL: float = 0.0,
                               alphaT: float = 0.0, axis=(1.0, 0.0, 0.0)) -> "Orthotropic":
        """fibre-reinforced plates and laminates: axis 1 along the fibres (`axis`, global), isotropic in the plane across
        them (GTT = ET / (2 (1 + νTT)))"""
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        t = np.cross(a, (0, 0, 1)) if abs(a[2]) < 0.9 else np.cross(a, (1, 0, 0))
        t /= np.linalg.norm(t)
        return cls(E=(EL, ET, ET), nu=(nuLT, nuLT, nuTT), G=(GLT, GLT, ET / (2 * (1 + nuTT))), alpha=(alphaL, alphaT, alphaT),
                   axes=np.array([a, t, np.cross(a, t)]))

    def _S_local(self) -> np.ndarray:
        E1, E2, E3 = self.E
        n12, n13, n23 = self.nu
        G12, G13, G23 = self.G
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
        S[0, 1] = S[1, 0] = -n12 / E1
        S[0, 2] = S[2, 0] = -n13 / E1
        S[1, 2] = S[2, 1] = -n23 / E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / G12, 1 / G23, 1 / G13  # Voigt shear order xy, yz, zx = 12, 23, 31
        return S

    def D6(self) -> np.ndarray:
        """6 × 6 stiffness in global axes"""
        C = _tensor(np.linalg.inv(self._S_local()))
        A = self.axes  # x_material = A x_global
        return _voigt(np.einsum("ai,bj,ck,dl,abcd->ijkl", A, A, A, A, C))

    def alpha6(self) -> np.ndarray:
        a = self.axes.T @ np.diag(self.alpha) @ self.axes
        return np.array([a[0, 0], a[1, 1], a[2, 2], 2 * a[0, 1], 2 * a[1, 2], 2 * a[2, 0]])

    def D(self, dim: int = 3, kind: str = "solid") -> np.ndarray:
        return _reduce(self.D6(), self.alpha6(), dim, kind)[0]

    def thermal_strain(self, dim: int = 3, kind: str = "solid") -> np.ndarray:
        return _reduce(self.D6(), self.alpha6(), dim, kind)[1]

    def reduce_strain(self, e6, dim: int = 3, kind: str = "solid") -> np.ndarray:
        return _reduce_strain(self.D6(), e6, dim, kind)


def lame(E: float, nu: float, kind: str = "solid"):
    """Lamé constants (plane stress: reduced λ)"""
    return Isotropic(E, nu).lame(kind)
