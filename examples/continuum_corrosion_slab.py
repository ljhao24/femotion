"""A longitudinal crack along a corroding bar in a thin slab (femotion.continuum.opensees; generic illustrative values).

An 80 mm slab with one layer of transverse bars Ø8 @ 100 mm and a longitudinal bar Ø--d (default 12 mm) that corrodes,
with the cover --cover (default 25 mm) under the outer of the two (--layout outer: the corroding bar is the outer one,
the transverse bars rest on it; inner: it rests on the transverse bars). The rust grows at the bar's surface and pushes
the concrete apart until the cover splits: a crack along the bar, the kind seen on soffits parallel to the bars.

Model: a 10 mm slice across the bar (plane strain: both faces held along the bar), 300 mm wide (the corroding bar in the
middle), the bar resolved by an O-grid (femotion.mesh.inclusions) with a 0.5 mm rust ring at its surface. The transverse
bars cross the slice: smeared over their spacing (area and perimeter × slice / spacing), bonded with the law of fib Model
Code 2010. Sides: --sides open (default: the strip may widen, held at mid-depth only — little transverse restraint) or
held (no widening: the rest of the slab and the webs hold it). The ring takes the free strain of rust_strain
(femotion.continuum.corrosion): its outer surface moves out by δ unless the concrete stops it; the ring is as stiff as
steel (the rust is taken as incompressible — the upper bound of its push, see the guide). Concrete: fib Model Code 2010
defaults for fcm (--fcm 33 ≈ Chinese C30, 40 ≈ C40), cracks regularised with the hoop direction around the bar as the
crack direction near it and vertical cracks elsewhere. Units N, mm, MPa. About two minutes.

Two stages ramp δ: to 0.02 mm (cracking of the cover, fine steps) and on to --delta (opening). Printed: δ at the first
crack and when the soffit cracks, the widest crack on the soffit and on the top face against δ, and the corrosion depth
x (with the mass loss) for soffit cracks of 0.05, 0.2, 0.5 and 1.0 mm — x from δ through the rust volume ratio n (--n,
default 3), with no porous zone and with the 12.5 μm zone of Liu & Weyers (1998) — next to the critical depths of the
references (Alonso et al. 1998, Liu & Weyers 1998, El Maaddawy & Soudki 2007, GB/T 51355-2019) and the growth of the
width with the corrosion depth.

What decides the crack (defaults: about 0.05 mm on the soffit at δ = 14 μm, 1.0 mm at δ = 290 μm, x ≈ 150 μm at
n = 3): with the transverse bars under the corroding bar (--layout inner) they cross the soffit crack and stop it at
about 0.27 mm; with more rust the cracks run up to the top face (0.8 mm at δ = 385 μm) and the soffit crack closes to
0.08 mm. With the sides held (--sides held) the cover delaminates along the bar layer instead of opening at the soffit.

    python examples/continuum_corrosion_slab.py [--layout outer] [--sides open] [--d 12] [--cover 25] [--fcm 33]
                                                [--delta 0.4] [--out DIR]

--out writes a ParaView series (DIR/slab.pvd) and the fem-result/1 file DIR/slab.json.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from femotion.continuum.corrosion import (alonso_1998, depth_for_expansion, el_maaddawy_soudki_2007, gbt51355_2019,
                                          liu_weyers_1998, mass_loss, rust_strain)
from femotion.continuum.opensees import BondSlip, ConcreteDamage, ElasticIso, OpsSolid, steel
from femotion.mesh.inclusions import bar_block, hoop_normals

H, W, T = 80.0, 300.0, 10.0  # slab depth, width of the model, thickness of the slice along the bar
DT, ST, RING = 8.0, 100.0, 0.5  # transverse bars Ø8 @ 100, rust ring


def build(d: float, cover: float, fcm: float, h: float, layout: str, sides: str):
    if layout == "outer":  # the corroding bar under the transverse bars
        zl, zt = cover + d / 2, cover + d + DT / 2
    elif layout == "inner":  # resting on them
        zt, zl = cover + DT / 2, cover + DT + d / 2
    else:
        raise ValueError("layout: 'outer' or 'inner'")
    m = bar_block([0.0, T], W, H, [(W / 2, zl, d)], h=h, ring=RING)
    c = ConcreteDamage.from_strength(fcm, units="N-mm")
    cen = m.points[m.cells["hex8"]].mean(axis=1)
    near = np.hypot(cen[:, 1] - W / 2, cen[:, 2] - zl) < 2.5 * d  # crack direction: hoop near the bar, else vertical
    n = np.where(near[:, None], hoop_normals(m, [(W / 2, zl, d)]), [0.0, 1.0, 0.0])
    s = OpsSolid(m, {"concrete": c, "steel": ElasticIso(2.0e5, 0.3), "rust": ElasticIso(2.0e5, 0.3)},
                 band=("projected", n))
    k = T / ST  # the share of a transverse bar in the slice
    s.bar([(T / 2, 0.0, zt), (T / 2, W, zt)], area=k * math.pi * DT**2 / 4, material=steel(450.0, 2.0e5),
          bond=BondSlip.mc2010(fcm), perimeter=k * math.pi * DT, name="transverse bars")
    s.fix(lambda x: np.isclose(x[0], 0) | np.isclose(x[0], T), comps=(0,))  # plane strain
    side = lambda x: np.isclose(x[1], 0) | np.isclose(x[1], W)  # noqa: E731
    if sides == "held":
        s.fix(side, comps=(1, 2))
    elif sides == "open":
        zs = np.unique(m.points[np.isclose(m.points[:, 1], 0), 2])
        zm = zs[np.argmin(np.abs(zs - H / 2))]  # the node line nearest mid-depth
        s.fix(lambda x: np.isclose(x[1], 0) & np.isclose(x[2], zm), comps=(1,))
        s.fix(lambda x: side(x) & np.isclose(x[2], zm), comps=(2,))
    else:
        raise ValueError("sides: 'open' or 'held'")
    return s, c, zl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layout", choices=("outer", "inner"), default="outer")
    ap.add_argument("--sides", choices=("open", "held"), default="open")
    ap.add_argument("--d", type=float, default=12.0, help="corroding bar diameter, mm")
    ap.add_argument("--cover", type=float, default=25.0, help="cover to the outer bars, mm")
    ap.add_argument("--fcm", type=float, default=33.0, help="mean cylinder strength, MPa")
    ap.add_argument("--fcu", type=float, default=30.0, help="cube strength for GB/T 51355 (C30: 30), MPa")
    ap.add_argument("--delta", type=float, default=0.4, help="final free expansion of the rust, mm")
    ap.add_argument("--n", type=float, default=3.0, help="rust volume ratio")
    ap.add_argument("--h", type=float, default=6.0, help="cell size away from the bar, mm")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    t0 = time.time()
    s, c, zl = build(a.d, a.cover, a.fcm, a.h, a.layout, a.sides)
    r = a.d / 2
    rust = lambda delta: s.initial_strain(rust_strain((0, W / 2, zl), (1, 0, 0), r - RING, RING, delta), "rust")  # noqa: E731
    d1 = 0.02
    s.static(strain=rust(d1), steps=80, record=2, name="cracking")
    s.static(strain=rust(a.delta - d1), steps=150, record=2, name="opening")
    print(f"{s.nc} cells, {s.n_nodes} nodes, {len(s.history) - 1} steps in {time.time() - t0:.0f} s")

    cover = a.cover + (DT if a.layout == "inner" else 0.0)  # over the corroding bar
    print(f"\nbar Ø{a.d:g} mm, cover {cover:g} mm over it (c/d = {cover / a.d:.2f}), fcm {a.fcm:g} MPa, "
          f"ft {c.ft:.2f} MPa, E {c.E:.0f} MPa")

    def delta_of(i):  # free expansion reached at history index i
        h = s.history[i]
        return {"cracking": h.lam * d1, "opening": d1 + h.lam * (a.delta - d1)}.get(h.stage, 0.0)

    rec = s.recorded()
    conc = s.cells_of("concrete")
    hexes = s.mesh.cells["hex8"]
    soffit = np.intersect1d(conc, np.flatnonzero(np.isclose(s.mesh.points[hexes, 2], 0).any(axis=1)))
    first = next((i for i in rec if s.history[i].cell["cracked"][conc].any()), None)
    through = next((i for i in rec if s.history[i].cell["cracked"][soffit].any()), None)
    print(f"first crack at the bar: δ = {1e3 * delta_of(first):.1f} μm" if first else "no crack")
    print(f"the soffit cracks:      δ = {1e3 * delta_of(through):.1f} μm" if through else "the soffit does not crack")

    rows = np.array([(delta_of(i), s.cracks_along((T / 2, 0, 0.5), (T / 2, W, 0.5), step=i)["max_width"],
                      s.cracks_along((T / 2, 0, H - 0.5), (T / 2, W, H - 0.5), step=i)["max_width"]) for i in rec])
    print("\n  δ (μm)   widest crack on the soffit   on the top face (mm)")
    for dl, w, wt in rows[:: max(1, len(rows) // 16)]:
        print(f"  {1e3 * dl:7.1f}   {w:7.3f}   {wt:7.3f}")

    print(f"\nsoffit crack width → corrosion depth x (rust volume ratio n = {a.n:g}; mass loss of the bar in brackets)")
    print("  w (mm)   δ (μm)   x, no porous zone (μm)   x, 12.5 μm porous zone (μm)")
    for w in (0.05, 0.2, 0.5, 1.0):
        k = np.flatnonzero(rows[:, 1] >= w)
        if not len(k):
            print(f"  {w:5.2f}    beyond δ = {1e3 * a.delta:g} μm")
            continue
        j = int(k[0])
        dl = float(np.interp(w, rows[j - 1:j + 1, 1], rows[j - 1:j + 1, 0])) if j else rows[j, 0]
        x0, x1 = depth_for_expansion(dl, a.d, a.n), depth_for_expansion(dl, a.d, a.n, porous=0.0125)
        print(f"  {w:5.2f}   {1e3 * dl:6.1f}   {1e3 * x0:6.1f} ({100 * mass_loss(x0, a.d):.2f} %)"
              f"             {1e3 * x1:6.1f} ({100 * mass_loss(x1, a.d):.2f} %)")

    k = (rows[:, 1] >= 0.2) & (rows[:, 1] <= 1.0)
    if k.sum() > 2:  # growth of the width with the corrosion depth, x = δ / (n − 1) for x ≪ d
        g = np.polyfit(rows[k, 0] / (a.n - 1), rows[k, 1], 1)[0]
        print(f"\ngrowth between 0.2 and 1.0 mm: dw/dx = {1e-3 * g:.4f} mm per μm of corrosion depth "
              "(Alonso et al. 1994–98, accelerated tests: 0.003–0.018 mm/μm)")

    lw = liu_weyers_1998(cover, a.d, c.ft, c.E, nu=c.nu)
    em = el_maaddawy_soudki_2007(cover, a.d, c.ft, c.E, nu=c.nu, creep=0.0)
    print("\ncritical corrosion depth for cover cracking in the references (μm):")
    print(f"  Alonso et al. 1998 (accelerated tests, first visible crack):      {1e3 * alonso_1998(cover, a.d):6.1f}")
    print(f"  Liu & Weyers 1998 (thick-walled cylinder, 12.5 μm porous zone):   {1e3 * lw['depth']:6.1f}")
    print(f"  El Maaddawy & Soudki 2007 (the same, 10 μm, no creep):            {1e3 * em['depth']:6.1f}")
    print(f"  GB/T 51355-2019 C.0.4 (natural corrosion, slab bar, fcu {a.fcu:g}):     "
          f"{1e3 * gbt51355_2019(cover, a.d, a.fcu):6.1f}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        keep = rec[::5] + [rec[-1]]
        s.export(a.out / "slab.pvd", steps=keep)
        s.to_result(steps=keep, meta={"title": "rust splitting the cover of a slab"}).write(a.out, "slab")
        print(f"\nwritten: {a.out / 'slab.pvd'}, {a.out / 'slab.json'}")


if __name__ == "__main__":
    main()
