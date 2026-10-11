"""Material strengths of JTG 3362-2018 (reinforced concrete only; prestressing steel is not covered yet).

Units: MPa. Tables 3.1.3, 3.1.4, 3.1.5 (concrete), 3.2.2, 3.2.3-1, 3.2.4 (reinforcement), 5.1.4 (β), 5.1.5 (εcu),
5.2.1 (ξb); checked against the official text (docs/codes/JTG_3362-2018.md).
"""
from __future__ import annotations

from dataclasses import dataclass

#                 fck    ftk    fcd    ftd    Ec (MPa)
_CONCRETE = {
    25: (16.7, 1.78, 11.5, 1.23, 2.80e4),
    30: (20.1, 2.01, 13.8, 1.39, 3.00e4),
    35: (23.4, 2.20, 16.1, 1.52, 3.15e4),
    40: (26.8, 2.40, 18.4, 1.65, 3.25e4),
    45: (29.6, 2.51, 20.5, 1.74, 3.35e4),
    50: (32.4, 2.65, 22.4, 1.83, 3.45e4),
    55: (35.5, 2.74, 24.4, 1.89, 3.55e4),
    60: (38.5, 2.85, 26.5, 1.96, 3.60e4),
    65: (41.5, 2.93, 28.5, 2.02, 3.65e4),
    70: (44.5, 3.00, 30.5, 2.07, 3.70e4),
    75: (47.4, 3.05, 32.4, 2.10, 3.75e4),
    80: (50.2, 3.10, 34.6, 2.14, 3.80e4),
}

#                         fsk   fsd   fsd'  Es (MPa)
_REBAR = {
    "HPB300": (300, 250, 250, 2.1e5),
    "HRB400": (400, 330, 330, 2.0e5),
    "HRBF400": (400, 330, 330, 2.0e5),
    "RRB400": (400, 330, 330, 2.0e5),
    "HRB500": (500, 415, 400, 2.0e5),
}


@dataclass(frozen=True)
class Concrete:
    """concrete of strength class C25 … C80 (grade = cube strength fcu,k in MPa)"""
    grade: int
    fck: float
    ftk: float
    fcd: float
    ftd: float
    Ec: float

    @property
    def fcuk(self) -> float:
        return float(self.grade)

    @property
    def beta(self) -> float:
        """ratio of the equivalent rectangular block depth to the neutral-axis depth: 0.8 up to C50, 0.74 at C80, linear between"""
        return 0.8 - 0.06 * min(max(self.grade - 50, 0), 30) / 30

    @property
    def eps_cu(self) -> float:
        """ultimate compressive strain: 0.0033 up to C50, 0.0030 at C80, linear between"""
        return 0.0033 - 0.0003 * min(max(self.grade - 50, 0), 30) / 30


@dataclass(frozen=True)
class Rebar:
    name: str
    fsk: float
    fsd: float  # tension
    fsd_c: float  # compression (fsd')
    Es: float

    @property
    def fsv(self) -> float:
        """design strength as stirrups (shear, torsion, punching): fsd, at most 330 MPa"""
        return min(self.fsd, 330.0)

    @property
    def rho_sv_min(self) -> float:
        """minimum stirrup ratio of beams (9.3.12): 0.14 % for HPB300, 0.11 % for HRB400 and HRB500"""
        return 0.0014 if self.name == "HPB300" else 0.0011


def concrete(grade: int | str) -> Concrete:
    """concrete(30) or concrete("C30")"""
    g = int(str(grade).upper().lstrip("C"))
    if g not in _CONCRETE:
        raise ValueError(f"concrete C{g}: JTG 3362-2018 tabulates C25 … C80 in steps of 5")
    return Concrete(g, *_CONCRETE[g])


def rebar(name: str = "HRB400") -> Rebar:
    n = name.upper()
    if n not in _REBAR:
        raise ValueError(f"reinforcing steel {name}: one of {', '.join(_REBAR)}")
    return Rebar(n, *_REBAR[n])


# table 5.2.1, ordinary reinforcement: classes ≤C50 | C55, C60 | C65, C70 (C75, C80: not given, "—")
_XI_B = {"HPB300": (0.58, 0.56, 0.54), "HRB400": (0.53, 0.51, 0.49), "HRBF400": (0.53, 0.51, 0.49),
         "RRB400": (0.53, 0.51, 0.49), "HRB500": (0.49, 0.47, 0.46)}


def xi_b(c: Concrete, s: Rebar) -> float:
    """relative balanced depth ξb from table 5.2.1. The table gives no value for ordinary reinforcement in C75 and C80."""
    if c.grade >= 75:
        raise ValueError(f"table 5.2.1 gives no ξb for {s.name} in C{c.grade}")
    col = 0 if c.grade <= 50 else 1 if c.grade <= 60 else 2
    return _XI_B[s.name][col]
