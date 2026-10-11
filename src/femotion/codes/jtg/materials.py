"""Material strengths of JTG 3362-2018: concrete, reinforcing steel, prestressing steel.

Units: MPa. Tables 3.1.3, 3.1.4, 3.1.5 (concrete), 3.2.2-1, 3.2.3-1, 3.2.4 (reinforcement), 3.2.2-2, 3.2.3-2, 3.2.4
(prestressing steel), 5.1.4 (β), 5.1.5 (εcu), 5.2.1 (ξb), 6.1.4 (σcon); checked against the official text
(docs/codes/JTG_3362-2018.md).
"""
from __future__ import annotations

import dataclasses
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


# table 5.2.1, prestressing steel: ≤C50 | C55, C60 | C65, C70 | C75, C80
_XI_B_P = {"strand": (0.40, 0.38, 0.36, 0.35), "wire": (0.40, 0.38, 0.36, 0.35), "bar": (0.40, 0.38, 0.36, None)}


def xi_b(c: Concrete, s) -> float:
    """relative balanced depth ξb from table 5.2.1 for reinforcing steel (Rebar) or prestressing steel (Tendon). The
    table gives no value for ordinary reinforcement and threaded bars in C75 and C80. With several kinds of steel in the
    tension zone take the smallest (table note 1)."""
    col = 0 if c.grade <= 50 else 1 if c.grade <= 60 else 2 if c.grade <= 70 else 3
    v = (_XI_B_P[s.kind] if isinstance(s, Tendon) else _XI_B[s.name] + (None,))[col]
    if v is None:
        raise ValueError(f"table 5.2.1 gives no ξb for {s.name} in C{c.grade}")
    return v


# ── prestressing steel (tables 3.2.2-2, 3.2.3-2, 3.2.4) ──
#            kind: {fpk: fpd}, f'pd, Ep (MPa)
_TENDON = {
    "strand": ({1720: 1170, 1860: 1260, 1960: 1330}, 390, 1.95e5),
    "wire": ({1470: 1000, 1570: 1070, 1770: 1200, 1860: 1260}, 410, 2.05e5),
    "bar": ({785: 650, 930: 770, 1080: 900}, 400, 2.00e5),
}
_TENDON_NAMES = {"strand": "1×7 strand", "wire": "stress-relieved wire", "bar": "prestressing threaded bar"}


@dataclass(frozen=True)
class Tendon:
    kind: str  # "strand" (钢绞线 1×7), "wire" (消除应力钢丝), "bar" (预应力螺纹钢筋)
    fpk: float
    fpd: float
    fpd_c: float  # f'pd
    Ep: float

    @property
    def name(self) -> str:
        return f"{_TENDON_NAMES[self.kind]} fpk {self.fpk:g}"

    def with_fpd(self, fpd: float) -> "Tendon":
        """the same steel with another design strength, e.g. inside the anchorage zone of a pretensioned member (5.1.6)"""
        return dataclasses.replace(self, fpd=fpd)

    def sigma_con_max(self, external: bool = False, overstress: bool = False) -> float:
        """largest jacking stress σcon (6.1.4): wire and strand 0.75 fpk inside the concrete, 0.70 fpk external;
        threaded bars 0.85 fpk; + 0.05 fpk for overstressing or when the friction in the anchor is included"""
        if self.kind == "bar":
            k = 0.85
        else:
            k = 0.70 if external else 0.75
        return (k + (0.05 if overstress else 0.0)) * self.fpk


def tendon(kind: str = "strand", fpk: float = 1860) -> Tendon:
    """tendon("strand", 1860), tendon("wire", 1570), tendon("bar", 930); also "strand1860" style strings"""
    k = str(kind).lower()
    for name in _TENDON:
        if k.startswith(name) and k != name:
            k, fpk = name, float(k[len(name):].lstrip("-_ "))
    if k not in _TENDON:
        raise ValueError(f"prestressing steel {kind}: one of {', '.join(_TENDON)}")
    fpds, fpd_c, Ep = _TENDON[k]
    if int(fpk) not in fpds:
        raise ValueError(f"{k} fpk {fpk:g}: table 3.2.3-2 has {', '.join(str(v) for v in fpds)}")
    return Tendon(k, float(fpk), float(fpds[int(fpk)]), float(fpd_c), Ep)
