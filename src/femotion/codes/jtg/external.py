"""External prestressing of JTG 3362-2018: the detailing rules 9.4.21–9.4.27 that can be checked by numbers.

Units: mm. The sections themselves take external tendons through the `external` rows of jtg.rect / tee / box: flexure
5.2.5, 5.2.6 (jtg.flexure), shear 5.2.9-5 (jtg.shear, bent_external), the stresses from the prestress 6.1.6-7, 6.1.7-5,
-6 (RCSection.prestress_stress, prestress_resultant), the serviceability and stress checks of chapters 6 and 7 with
σpe,ex ≤ 0.60 fpk (7.1.5-3, jtg.service_stresses). The jacking stress limit is Tendon.sigma_con_max(external=True)
(6.1.4-2) and the friction loss prestress.friction_loss_external (table 6.2.2). Clauses: docs/codes/JTG_3362-2018.md.

    from femotion.codes import jtg
    jtg.external_detailing(wire_d=5.0, radius=3500, free_length=9000, hoop_d=20, hoop_spacing=100, crossbeam=1200)
"""
from __future__ import annotations

from .rc import Check

FREE_LENGTH_MAX = 8000.0  # free length between anchorages, deviators, positioners, seismic devices (9.4.26)
HOOP_DIAMETER_MIN = 20.0  # inner closed hoops of block deviators (9.4.25)
HOOP_SPACING_MAX = 100.0  # longitudinal spacing of the inner and outer closed hoops (9.4.25)
CROSSBEAM_MIN = 1000.0  # thickness of an anchorage crossbeam (9.4.27)


def deviation_radius_min(wire_d: float) -> float:
    """smallest radius (mm) of an external strand in a deviator (9.4.23 by 9.4.10 item 1): 4 m when the wires of the
    strand are at most 5 mm in diameter, 6 m above"""
    return 4000.0 if wire_d <= 5.0 else 6000.0


def external_detailing(*, wire_d: float | None = None, radius: float | None = None, free_length: float | None = None,
                       hoop_d: float | None = None, hoop_spacing: float | None = None,
                       crossbeam: float | None = None) -> Check:
    """the numeric detailing rules for external tendons, each checked when its value is given (all "should" rules):
    deviation radius ≥ deviation_radius_min(wire_d) (9.4.23, 9.4.10); free length ≤ 8.0 m (9.4.26); inner closed hoops of
    block deviators ≥ 20 mm in diameter, inner and outer hoops at ≤ 100 mm along the deviator (9.4.25); anchorage
    crossbeam ≥ 1000 mm thick (9.4.27). The Check's demand is the largest ratio of a value to its limit (limit / value
    for minima), 1.0 the limit; details hold each item."""
    items = []
    if radius is not None or wire_d is not None:
        if radius is None or wire_d is None:
            raise ValueError("the deviation radius check needs both radius and wire_d (9.4.10)")
        items.append(("deviation radius", radius, deviation_radius_min(wire_d), "min", "9.4.23, 9.4.10"))
    if free_length is not None:
        items.append(("free length", free_length, FREE_LENGTH_MAX, "max", "9.4.26"))
    if hoop_d is not None:
        items.append(("inner hoop diameter", hoop_d, HOOP_DIAMETER_MIN, "min", "9.4.25"))
    if hoop_spacing is not None:
        items.append(("hoop spacing", hoop_spacing, HOOP_SPACING_MAX, "max", "9.4.25"))
    if crossbeam is not None:
        items.append(("anchorage crossbeam thickness", crossbeam, CROSSBEAM_MIN, "min", "9.4.27"))
    if not items:
        raise ValueError("give at least one of wire_d and radius, free_length, hoop_d, hoop_spacing, crossbeam")
    details, failed, notes = {}, [], []
    worst = 0.0
    for name, v, lim, kind, cl in items:
        r = v / lim if kind == "max" else (lim / v if v > 0 else float("inf"))
        details[name] = {"value": v, "limit": lim, "ratio": r}
        worst = max(worst, r)
        if r > 1 + 1e-12:
            failed.append(f"{name} {v:g} mm {'>' if kind == 'max' else '<'} {lim:g} mm ({cl})")
    if free_length is not None and free_length > FREE_LENGTH_MAX:
        notes.append("free length over 8 m: fix it by calculation and allow for its effect on the structure "
                     "(commentary to 9.4.25–9.4.27)")
    if hoop_d is not None or hoop_spacing is not None:
        notes.append("block deviators need both inner closed hoops (round each deviator) and outer ones (round all of "
                     "them, 9.4.25)")
    return Check("external tendon detailing", worst, 1.0, "", "JTG 3362-2018 9.4.23–9.4.27", details, notes, failed,
                 limit_check=True, symbol="value/limit")
