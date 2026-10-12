"""A longitudinal crack along a corroding bar in a thin slab (femotion.continuum.opensees; generic illustrative values).

An 80 mm slab with one layer of transverse bars Ø8 @ 100 mm and a longitudinal bar Ø--d (default 12 mm) that corrodes,
with the cover --cover (default 25 mm) under the outer of the two (--layout outer: the corroding bar is the outer one,
the transverse bars rest on it; inner: it rests on the transverse bars). The rust grows at the bar's surface and pushes
the concrete apart until the cover splits: a crack along the bar on the soffit, or the cover delaminating along the
layer of bars, depending on what holds the slab together across the bar.

Model: a 10 mm slice across the bar (plane strain: both faces held along the bar), --width wide (default 200 mm, the
corroding bar in the middle), the bar resolved by an O-grid (femotion.mesh.inclusions) with a 0.5 mm rust ring at its
surface. The transverse bars cross the slice: smeared over their spacing (area and perimeter × slice / spacing), bonded
with the splitting law of fib Model Code 2010 (their cover is under 5 φ). The ring takes the free strain of rust_strain
(femotion.continuum.corrosion): its outer surface moves out by δ unless the concrete stops it; the ring is as stiff as
steel — rigid under the pressure (the upper bound of the rust's push), and able to carry hoop tension, which real rust
cannot (the pressure is a few per cent higher for that). Concrete: fib Model Code 2010 defaults for fcm (--fcm 33 ≈
Chinese C30, 40 ≈ C40), cracks regularised with the hoop direction around the bar as the crack direction near it and
vertical cracks elsewhere. Units N, mm, MPa. About two minutes per boundary condition.

What restrains the slab across the bar decides the crack (--sides):
- roller (default): symmetry planes at both sides, u_y = 0 over the full depth, and the transverse bars held there —
  the same bar corrodes every --width, or the slab around is much stiffer than the cover: a continuous slab;
- spring: the sides on springs of E / L per unit area (and the transverse bars on E_s A_s / L): the slab goes on for
  L = --restraint (default 1000 mm) to a rigid restraint (a web, a diaphragm), the corroding bar alone;
- open: no restraint in the plane at all (held at mid-depth only): a free strip — the extreme, an upper bound of the
  soffit crack, not a model of a slab;
- all: the three, compared.

Two stages ramp δ: to 0.02 mm (cracking of the cover, fine steps) and on to --delta (opening). Printed per boundary
condition: δ at the first crack and when the soffit cracks, the pressure of the rust on the concrete (from the
concrete's equilibrium), the widest crack on the soffit, on the top face and across the layer of bars (delamination)
against δ, and the corrosion depth x (with the mass loss) for soffit cracks of 0.05, 0.2, 0.5 and 1.0 mm where they
are reached — x from δ through the rust volume ratio n (--n, default 3), with no porous zone and with the 12.5 μm zone
of Liu & Weyers — next to the critical depths of the references (Alonso et al. 1998; Liu & Weyers 1998 and El
Maaddawy & Soudki 2007 with their own creep coefficients and without; GB/T 51355-2019, coefficients still to be checked
against the original).

    python examples/continuum_corrosion_slab.py [--sides roller|spring|open|all] [--layout outer|inner] [--d 12]
                                                [--cover 25] [--fcm 33] [--width 200] [--restraint 1000]
                                                [--delta 0.4] [--out DIR]

--out writes a ParaView series (DIR/slab_<sides>.pvd) and the fem-result/1 file DIR/slab_<sides>.json.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from femotion.continuum.corrosion import (alonso_1998, depth_for_expansion, el_maaddawy_soudki_2007, gbt51355_2019,
                                          interface_pressure, liu_weyers_1998, mass_loss, rust_strain)
from femotion.continuum.opensees import BondSlip, ConcreteDamage, ElasticIso, OpsSolid, steel
from femotion.mesh.inclusions import bar_block, hoop_normals

H, T = 80.0, 10.0  # slab depth, thickness of the slice along the bar
DT, ST, RING = 8.0, 100.0, 0.5  # transverse bars Ø8 @ 100, rust ring
SIDES = ("roller", "spring", "open")


def build(a, sides: str):
    d, cover, W = a.d, a.cover, a.width
    if a.layout == "outer":  # the corroding bar under the transverse bars
        zl, zt = cover + d / 2, cover + d + DT / 2
    else:  # resting on them
        zt, zl = cover + DT / 2, cover + DT + d / 2
    m = bar_block([0.0, T], W, H, [(W / 2, zl, d)], h=a.h, ring=RING)
    c = ConcreteDamage.from_strength(a.fcm, units="N-mm")
    cen = m.points[m.cells["hex8"]].mean(axis=1)
    near = np.hypot(cen[:, 1] - W / 2, cen[:, 2] - zl) < 2.5 * d  # crack direction: hoop near the bar, else vertical
    n = np.where(near[:, None], hoop_normals(m, [(W / 2, zl, d)]), [0.0, 1.0, 0.0])
    s = OpsSolid(m, {"concrete": c, "steel": ElasticIso(2.0e5, 0.3), "rust": ElasticIso(2.0e5, 0.3)},
                 band=("projected", n))
    k = T / ST  # the share of a transverse bar in the slice
    At = k * math.pi * DT**2 / 4
    tb = s.bar([(T / 2, 0.0, zt), (T / 2, W, zt)], area=At, material=steel(450.0, 2.0e5),
               bond=BondSlip.mc2010(a.fcm, failure="splitting"), perimeter=k * math.pi * DT, name="transverse bars")
    ends = [s.bars[tb]["nodes"][0], s.bars[tb]["nodes"][-1]]
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], T), comps=(0,))  # plane strain
    side = lambda x: np.isclose(x[1], 0) | np.isclose(x[1], W)  # noqa: E731
    zs = np.unique(m.points[np.isclose(m.points[:, 1], 0), 2])
    zm = zs[np.argmin(np.abs(zs - H / 2))]  # the node line nearest mid-depth
    if sides == "roller":
        s.fix(side, comps=(1,))
        s.fix(None, comps=(1,), nodes=ends)
        s.fix(lambda x: np.isclose(x[1], 0) & np.isclose(x[2], zm), comps=(2,))
    elif sides == "spring":
        s.face_spring(side, (0.0, c.E / a.restraint, 0.0))
        s.spring(None, (0.0, 2.0e5 * At / a.restraint, 0.0), nodes=ends)
        s.fix(lambda x: np.isclose(x[1], 0) & np.isclose(x[2], zm), comps=(2,))
    elif sides == "open":
        s.fix(lambda x: np.isclose(x[1], 0) & np.isclose(x[2], zm), comps=(1,))
        s.fix(lambda x: side(x) & np.isclose(x[2], zm), comps=(2,))
    else:
        raise ValueError(f"sides: one of {SIDES}")
    s.watch("p", interface_pressure((0.0, W / 2, zl), (1.0, 0.0, 0.0), d / 2))
    return s, c, zl


def run(a, sides: str) -> dict:
    t0 = time.time()
    s, c, zl = build(a, sides)
    W, d = a.width, a.d
    rust = lambda delta: s.initial_strain(rust_strain((0, W / 2, zl), (1, 0, 0), d / 2 - RING, RING, delta), "rust")  # noqa: E731
    d1 = 0.02
    s.static(strain=rust(d1), steps=80, record=2, name="cracking")
    s.static(strain=rust(a.delta - d1), steps=150, record=2, name="opening")

    def delta_of(i):  # free expansion reached at history index i
        h = s.history[i]
        return {"cracking": h.lam * d1, "opening": d1 + h.lam * (a.delta - d1)}.get(h.stage, 0.0)

    rec = s.recorded()
    conc = s.cells_of("concrete")
    hexes = s.mesh.cells["hex8"]
    soffit = np.intersect1d(conc, np.flatnonzero(np.isclose(s.mesh.points[hexes, 2], 0).any(axis=1)))
    first = next((i for i in rec if s.history[i].cell["cracked"][conc].any()), None)
    through = next((i for i in rec if s.history[i].cell["cracked"][soffit].any()), None)
    rows = []
    for i in rec:  # widest crack on the soffit, on the top face, and across the layer of bars beside the bar
        def ca(p, q, i=i):
            return s.cracks_along(p, q, step=i)["max_width"]
        side = max(ca((T / 2, W / 2 - y, 0.5), (T / 2, W / 2 - y, H - 0.5)) for y in (3 * d, -3 * d))
        rows.append((delta_of(i), ca((T / 2, 0, 0.5), (T / 2, W, 0.5)), ca((T / 2, 0, H - 0.5), (T / 2, W, H - 0.5)),
                     side))
    p = np.array([w["p"] for w in s.watched("p")])
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        keep = rec[::5] + [rec[-1]]
        s.export(a.out / f"slab_{sides}.pvd", steps=keep)
        s.to_result(steps=keep, meta={"title": f"rust splitting the cover of a slab ({sides})"}).write(a.out,
                                                                                                       f"slab_{sides}")
    return dict(sides=sides, s=s, c=c, rows=np.array(rows), p=p, delta=[delta_of(i) for i in range(len(s.history))],
                first=delta_of(first) if first else None, through=delta_of(through) if through else None,
                time=time.time() - t0)


def at_width(rows, w):
    """the expansion at which the soffit crack first reaches w (None if never)"""
    k = np.flatnonzero(rows[:, 1] >= w)
    if not len(k):
        return None
    j = int(k[0])
    return float(np.interp(w, rows[j - 1:j + 1, 1], rows[j - 1:j + 1, 0])) if j else float(rows[j, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sides", choices=SIDES + ("all",), default="roller")
    ap.add_argument("--layout", choices=("outer", "inner"), default="outer")
    ap.add_argument("--d", type=float, default=12.0, help="corroding bar diameter, mm")
    ap.add_argument("--cover", type=float, default=25.0, help="cover to the outer bars, mm")
    ap.add_argument("--fcm", type=float, default=33.0, help="mean cylinder strength, MPa")
    ap.add_argument("--fcu", type=float, default=30.0, help="cube strength for GB/T 51355 (C30: 30), MPa")
    ap.add_argument("--width", type=float, default=200.0, help="width of the model (spacing of the bars for roller), mm")
    ap.add_argument("--restraint", type=float, default=1000.0, help="length L of slab to a rigid restraint (spring), mm")
    ap.add_argument("--delta", type=float, default=0.4, help="final free expansion of the rust, mm")
    ap.add_argument("--n", type=float, default=3.0, help="rust volume ratio")
    ap.add_argument("--h", type=float, default=6.0, help="cell size away from the bar, mm")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cover = a.cover + (DT if a.layout == "inner" else 0.0)  # over the corroding bar
    runs = [run(a, sd) for sd in (SIDES if a.sides == "all" else (a.sides,))]
    c = runs[0]["c"]
    print(f"bar Ø{a.d:g} mm, cover {cover:g} mm over it (c/d = {cover / a.d:.2f}), transverse bars "
          f"{'over' if a.layout == 'outer' else 'under'} it, fcm {a.fcm:g} MPa, ft {c.ft:.2f} MPa, E {c.E:.0f} MPa, "
          f"model {a.width:g} mm wide")
    for r in runs:
        rows = r["rows"]
        print(f"\n── {r['sides']}: {r['s'].nc} cells, {len(r['s'].history) - 1} steps, {r['time']:.0f} s")
        print(f"first crack at δ = {1e3 * r['first']:.1f} μm; " if r["first"] else "no crack; ", end="")
        print(f"the soffit cracks at δ = {1e3 * r['through']:.1f} μm" if r["through"] else "the soffit does not crack")
        k = int(np.argmax(r["p"]))
        print(f"pressure of the rust on the concrete: up to {r['p'][k]:.1f} MPa (at δ = {1e3 * r['delta'][k]:.1f} μm), "
              f"{r['p'][-1]:.1f} MPa at the end")
        print("  δ (μm)   widest crack (mm): soffit   top face   across the bar layer")
        for dl, w, wt, wd in rows[:: max(1, len(rows) // 12)]:
            print(f"  {1e3 * dl:7.1f}                     {w:6.3f}     {wt:6.3f}     {wd:6.3f}")

    print(f"\nsoffit crack width → free expansion δ and corrosion depth x (rust volume ratio n = {a.n:g}; with a 12.5 μm "
          "porous zone in brackets)")
    print("  w (mm)" + "".join(f"   {r['sides']:>24s}" for r in runs))
    for w in (0.05, 0.2, 0.5, 1.0):
        cells = []
        for r in runs:
            dl = at_width(r["rows"], w)
            if dl is None:
                cells.append(f"— (at most {r['rows'][:, 1].max():.2f} mm)")
            else:
                x0, x1 = depth_for_expansion(dl, a.d, a.n), depth_for_expansion(dl, a.d, a.n, porous=0.0125)
                cells.append(f"δ {1e3 * dl:.0f}, x {1e3 * x0:.0f} ({1e3 * x1:.0f}) μm")
        print(f"  {w:5.2f} " + "".join(f"   {v:>24s}" for v in cells))
    x_end = depth_for_expansion(a.delta, a.d, a.n)
    print(f"  (at the end, δ = {1e3 * a.delta:g} μm: x = {1e3 * x_end:.0f} μm, {100 * mass_loss(x_end, a.d):.1f} % mass loss)")

    print("\ncritical corrosion depth for cover cracking in the references (μm):")
    print(f"  Alonso et al. 1998, accelerated tests, first visible crack: {1e3 * alonso_1998(cover, a.d):6.1f}")
    for creep, note in ((2.0, "creep coefficient 2.0 (the original)"), (0.0, "no creep")):
        lw = liu_weyers_1998(cover, a.d, c.ft, c.E, nu=0.18, creep=creep)
        print(f"  Liu & Weyers 1998, 12.5 μm porous zone, {note}: {1e3 * lw['depth']:6.1f}")
    for creep, d0 in ((2.35, 0.010), (2.35, 0.020), (0.0, 0.010)):
        em = el_maaddawy_soudki_2007(cover, a.d, c.ft, c.E, nu=0.18, delta0=d0, creep=creep)
        note = f"creep {creep:g}{' (the original)' if creep else ''}, porous zone {1e3 * d0:g} μm"
        print(f"  El Maaddawy & Soudki 2007, {note}: {1e3 * em['depth']:6.1f}")
    print(f"  GB/T 51355-2019 C.0.4, bar of a slab, fcu {a.fcu:g} (coefficients to be checked against the original): "
          f"{1e3 * gbt51355_2019(cover, a.d, a.fcu):6.1f}")


if __name__ == "__main__":
    main()
