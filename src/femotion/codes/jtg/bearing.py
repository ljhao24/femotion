"""Local compression (局部承压) of JTG 3362-2018 5.7: bearing under anchorages, bearings and concentrated loads.

Units: mm, mm², MPa, forces kN. Clauses: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    Ab = jtg.bearing_base_area(a=320, b=320, cx=400, cy=400)            # figure 5.7.1, concentric and symmetric
    Fld = 1.2 * 9 * 139 * 1395 / 1e3                                     # anchorage: 1.2 × the largest jacking force
    r = jtg.local_bearing(jtg.concrete("C50"), Fld, Al=320 * 320, Aln=320 * 320 - 3.14 * 45**2, Ab=Ab,
                          mesh=(5, 113, 280, 5, 113, 280, 50), fcu=45)   # n1, As1, l1, n2, As2, l2, layer spacing
"""
from __future__ import annotations

import math

import numpy as np

from .materials import Concrete, Rebar, _CONCRETE, concrete as _concrete, rebar as _rebar
from .rc import Check


def eta_s(c: Concrete) -> float:
    """ηs (5.7.1): 1.0 up to C50, 0.76 at C80, linear between"""
    return 1.0 - 0.24 * min(max(c.grade - 50, 0), 30) / 30


def k_indirect(c: Concrete) -> float:
    """k (5.3.2): 2.0 up to C50, 1.7 at C80, linear between"""
    return 2.0 - 0.3 * min(max(c.grade - 50, 0), 30) / 30


def fcd_at(fcu: float) -> float:
    """fcd for a cube strength f'cu between the classes of table 3.1.4 (straight-line interpolation; 5.7.1: post-tensioned
    members at tensioning)"""
    gs = sorted(_CONCRETE)
    if not gs[0] <= fcu <= gs[-1]:
        raise ValueError(f"f'cu = {fcu:g} MPa outside table 3.1.4 (C{gs[0]} … C{gs[-1]})")
    return float(np.interp(fcu, gs, [_CONCRETE[g][2] for g in gs]))


def bearing_base_area(a: float, b: float, *, cx: float = math.inf, cy: float = math.inf, circle: bool = False) -> float:
    """computed base area Ab (5.7.1, figure 5.7.1) of a loaded rectangle a (along x) × b (along y), concentric and
    symmetric: extended on every side by the shorter side of the loaded area, but by no more than the distance to the
    nearer edge in that direction (cx, cy; a loaded area on an edge: 0, Ab then spans only its own width that way).
    circle=True: a loaded circle of diameter a, Ab the concentric circle extended by a (diameter 3a), or by the edge
    distance cx if less. Other layouts: pass Ab yourself."""
    if circle:
        e = min(a, cx)
        return math.pi * (a + 2 * e) ** 2 / 4
    s = min(a, b)
    return (a + 2 * min(s, cx)) * (b + 2 * min(s, cy))


def mesh_ratio(n1: int, As1: float, l1: float, n2: int, As2: float, l2: float, s: float, Acor: float) -> float:
    """ρv of a square mesh (5.7.2-3): (n1 As1 l1 + n2 As2 l2) / (Acor s)"""
    return (n1 * As1 * l1 + n2 * As2 * l2) / (Acor * s)


def spiral_ratio(Ass1: float, dcor: float, s: float) -> float:
    """ρv of a spiral (5.7.2-4): 4 Ass1 / (dcor s)"""
    return 4 * Ass1 / (dcor * s)


def local_bearing(concrete, Fld: float, *, Al: float, Aln: float | None = None, Ab: float, gamma0: float = 1.0,
                  fcu: float | None = None, mesh=None, spiral=None, Acor: float | None = None, steel="HRB400",
                  layers: int | None = None) -> Check:
    """local compression (5.7). Fld: local compressive design force (kN); for the anchorage zone of a post-tensioned
    member 1.2 × the largest jacking force (5.7.1). Al: loaded area without deducting holes, Aln: with the holes
    deducted (default Al); with a steel bearing plate both include the 45° spread through the plate; for an anchor with a
    trumpet cast in one piece with the plate, Aln = the plate less the trumpet's inner end (5.7.1). Ab: computed base
    area (bearing_base_area). fcu: cube strength at tensioning f'cu of post-tensioned members (fcd by straight-line
    interpolation in table 3.1.4); ηs, k from the concrete's class.
    Section size (5.7.1-1): γ0 Fld ≤ 1.3 ηs β fcd Aln, β = √(Ab / Al), for members with indirect reinforcement.
    Resistance (5.7.2-1), with indirect reinforcement: γ0 Fld ≤ 0.9 (ηs β fcd + k ρv βcor fsd) Aln, βcor = √(Acor / Al),
    Acor ≤ Ab. mesh = (n1, As1, l1, n2, As2, l2, s): a square mesh (Acor = l1 l2 unless Acor is given; the bar areas of
    the two directions within 50 %); spiral = (Ass1, dcor, s): a spiral (Acor = π dcor² / 4). layers: number of mesh
    layers or spiral turns, at least 4 (5.7.2 note). The resistance reported is the smaller of 5.7.1-1 and 5.7.2-1
    (details "governs"). Without indirect reinforcement the member is outside 5.7 (both clauses are for members with
    indirect reinforcement): listed as failed, with 5.7.2-1 for ρv = 0, 0.9 ηs β fcd Aln, for information."""
    c = concrete if isinstance(concrete, Concrete) else _concrete(concrete)
    st = steel if isinstance(steel, Rebar) else _rebar(steel)
    Aln = Al if Aln is None else Aln
    fcd = c.fcd if fcu is None else fcd_at(fcu)
    es = eta_s(c)
    beta = math.sqrt(Ab / Al)
    size = 1.3 * es * beta * fcd * Aln / 1e3
    d = {"fcd": fcd, "eta_s": es, "beta": beta, "Ab": Ab, "Al": Al, "Aln": Aln, "F_section_max": size}
    if mesh is None and spiral is None:
        R0 = 0.9 * es * beta * fcd * Aln / 1e3
        chk = Check("local compression (no indirect reinforcement)", gamma0 * Fld, R0, "kN", "JTG 3362-2018 5.7", d)
        chk.failed.append("5.7 covers members with indirect reinforcement (mesh or spiral): none given; the resistance "
                          "shown is 5.7.2-1 with ρv = 0")
        return chk
    if mesh is not None:
        n1, As1, l1, n2, As2, l2, s = mesh
        Acor = l1 * l2 if Acor is None else Acor
        rho_v = mesh_ratio(n1, As1, l1, n2, As2, l2, s, Acor)
        d["mesh_area_ratio"] = max(n1 * As1, n2 * As2) / min(n1 * As1, n2 * As2)
    else:
        Ass1, dcor, s = spiral
        Acor = math.pi * dcor * dcor / 4 if Acor is None else Acor
        rho_v = spiral_ratio(Ass1, dcor, s)
    notes = []
    Acor_raw = Acor
    if Acor > Ab:
        notes.append(f"Acor = {Acor:.0f} > Ab: Acor = Ab (5.7.2)")
        Acor = Ab
    beta_cor = math.sqrt(Acor / Al)
    k = k_indirect(c)
    R = 0.9 * (es * beta * fcd + k * rho_v * beta_cor * st.fsd) * Aln / 1e3
    d.update(rho_v=rho_v, beta_cor=beta_cor, k=k, Acor=Acor, fsd=st.fsd)
    d.update(mesh=None if mesh is None else tuple(mesh), spiral=None if spiral is None else tuple(spiral),
             Acor_indirect=Acor_raw)  # the inputs behind ρv and Acor, for calculation reports
    d.update(F_resistance=R, governs="5.7.1-1 (section size)" if size < R else "5.7.2-1")
    chk = Check("local compression", gamma0 * Fld, min(R, size), "kN", "JTG 3362-2018 5.7.1, 5.7.2", d, notes)
    if mesh is not None and d["mesh_area_ratio"] > 1.5:
        # the bar areas of the two directions (n1 As1, n2 As2) differ by at most 50 %: the larger ≤ 1.5 × the smaller
        chk.failed.append("mesh: bar areas of the two directions differ by more than 50 % (5.7.2)")
    if layers is not None and layers < 4:
        chk.failed.append(f"{layers} {'mesh layers' if mesh is not None else 'spiral turns'} < 4 (5.7.2 note)")
    return chk
