"""Smeared cracking of a reinforced concrete beam on OpenSees solids (femotion.continuum.opensees; illustrative values).

A simply supported beam 200 × 300 mm, span 2400 mm, two loads 600 mm apart (shear span a = 900 mm), 2 Ø16 bottom bars at
40 mm (HRB400-like steel, fy 450 MPa mean), concrete fcm = 38 MPa (fib Model Code 2010 defaults for ft, E, Gf, Gc).
A quarter of the beam by symmetry, hexahedra of `--h` mm (default 25), elastic pads under the bearing and the loading
plate. Units N, mm, MPa.

1) stage "self-weight" (load control);
2) stage "load" (displacement control at midspan, IMPL-EX concrete, two Newton iterations per step).

Printed: the cracking moment against the transformed section, the load–deflection curve against the uncracked and the
fully cracked beam (Branson in between), the steel stress against the cracked section, and the cracks along the soffit
with their widths against JTG 3362-2018 6.4.3 (Wcr) and Eurocode 2 7.3.4 (wk) at the same moment.

    python examples/continuum_crack_rc_beam.py [--h 25] [--steps 100] [--out DIR]

--out writes a ParaView series (DIR/beam.pvd: displacement, crack width and crack normal per cell, steel stress) and
the fem-result/1 file DIR/beam.json for animation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from femotion.continuum.opensees import ConcreteDamage, ElasticIso, OpsSolid, steel
from femotion.mesh import structured

B, D, HALF = 200.0, 300.0, 1300.0  # width, depth, half length (support at x = 100, load at x = 1000, symmetry at 1300)
AS, DEPTH, ES, FY = 2 * 201.06, 260.0, 2.0e5, 450.0


def build(h: float):
    m = structured.box(HALF, B / 2, D, int(HALF / h), max(1, int(B / 2 / h)), int(D / h))
    cen = m.centroids()
    pad = ((np.abs(cen[:, 0] - 100) < 50) & (cen[:, 2] < 50)) | ((np.abs(cen[:, 0] - 1000) < 50) & (cen[:, 2] > D - 50))
    m.cell_groups = {"concrete": {"hex8": np.flatnonzero(~pad)}, "pads": {"hex8": np.flatnonzero(pad)}}
    c = ConcreteDamage.from_strength(38.0, units="N-mm")
    s = OpsSolid(m, {"concrete": c, "pads": ElasticIso(c.E, c.nu)})
    s.bar([(20, 50, 40), (HALF, 50, 40)], area=AS / 2, material=steel(FY, ES), name="bottom bars")
    s.fix(lambda x: np.isclose(x[0], HALF), comps=(0,))  # symmetry planes
    s.fix(lambda x: np.isclose(x[1], B / 2), comps=(1,))
    # a line bearing: fixing a strip of several node lines would clamp the end (≈ 20 % stiffer in the elastic range)
    bearing = s.fix(lambda x: np.isclose(x[2], 0) & np.isclose(x[0], 100), comps=(2,))
    return s, c, bearing


def sections(c):
    n = ES / c.E
    A = B * D + n * AS
    yb = (B * D * D / 2 + n * AS * (D - DEPTH)) / A
    Ig = B * D**3 / 12 + B * D * (D / 2 - yb) ** 2 + n * AS * (yb - (D - DEPTH)) ** 2
    x = (-n * AS + np.sqrt((n * AS) ** 2 + 2 * B * n * AS * DEPTH)) / B
    Icr = B * x**3 / 3 + n * AS * (DEPTH - x) ** 2
    return n, yb, Ig, x, Icr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=25.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    s, c, bearing = build(a.h)
    print(f"concrete: fc {c.fc:.1f}, ft {c.ft:.2f}, E {c.E:.0f} MPa, Gf {c.Gf:.3f} N/mm, Gc {c.Gc:.1f} N/mm; "
          f"band h {s.h.min():.0f} mm (snap-back limit {c.h_max():.0f} mm); {s.nc} cells")
    t0 = time.time()
    s.static(s.self_weight(25e-6), steps=2, name="self-weight")  # 25 kN/m³ = 2.5e-5 N/mm³
    plate = lambda x: np.isclose(x[2], D) & (np.abs(x[0] - 1000) <= 25 + 1e-6)  # noqa: E731
    mid = s.nearest((HALF, B / 2, 0))
    st = s.push(s.traction(plate, (0, 0, -1.0)), mid, 2, -5.0, steps=a.steps, name="load")  # cell results every step
    print(f"{len(s.history) - 1} steps in {time.time() - t0:.0f} s, converged: {st.converged} {st.message}")

    u, P = s.curve(mid, 2, at=bearing, sign=1.0)
    u, P = -u, 4 * P  # the whole beam: four quarters' bearing reactions
    P -= P[s.stages[0].steps[-1]]  # the applied load only (self-weight reaction removed)
    n, yb, Ig, x, Icr = sections(c)
    i = s.first_crack()
    print(f"first crack between {P[i - 1] / 2 * 900 / 1e6:.1f} and {P[i] / 2 * 900 / 1e6:.1f} kN·m (live load), ft·W "
          f"{c.ft * Ig / yb / 1e6:.1f} kN·m: the MC2010 pre-peak law and the softening raise the apparent flexural "
          "strength, as fctm,fl = 1.31 fctm of MC2010 for h = 300 mm")
    span, sa = 2400.0, 900.0
    kb = sa * (3 * span**2 - 4 * sa**2) / (24 * c.E)  # midspan deflection per unit P/2, bending only
    Mcr = c.ft * Ig / yb
    print("    P kN   δ FE   uncracked  Branson  cracked   σs FE   σs cracked   max w FE   Wcr JTG   wk EC2   cracks")
    from femotion.codes import jtg
    from femotion.codes.jtg import sls

    sec = jtg.rect(200, 300, concrete="C30", steel="HRB400", bottom=[(2, 16, 40)])
    rho = AS / (B * min(2.5 * 40, D / 2))
    for k in s.recorded()[2::max(1, a.steps // 10)]:
        if P[k] <= 0:
            continue
        M = P[k] / 2 * sa
        Ie = Ig if M < Mcr else (Mcr / M) ** 3 * Ig + (1 - (Mcr / M) ** 3) * Icr
        r = s.bar_results(0, k)
        sfe = r["stress"][r["x"][:, 0] > 950].max()
        wj = sls.crack_width(sec, Ms=M / 1e6, Ml=M / 1e6, env="I").demand
        ss = M / (0.87 * AS * DEPTH)
        wk = (3.4 * 32 + 0.425 * 0.8 * 0.5 * 16 / rho) * max(ss - 0.4 * c.ft / rho * (1 + n * rho), 0.6 * ss) / ES
        cr = s.cracks_along((150, 50, 1), (HALF, 50, 1), step=k)
        print(f"  {P[k] / 1e3:6.1f} {u[k]:6.2f} {P[k] / 2 * kb / Ig:9.2f} {P[k] / 2 * kb / Ie:8.2f} {P[k] / 2 * kb / Icr:8.2f} "
              f"{sfe:8.1f} {n * M * (DEPTH - x) / Icr:10.1f} {cr['max_width']:10.3f} {wj:9.3f} {wk:8.3f}   "
              f"{len([q for q in cr['cracks'] if q['width'] > 0.03])}")
    cr = s.cracks_along((150, 50, 1), (HALF, 50, 1))
    print("cracks along the soffit at the last step (x mm: w mm):",
          ", ".join(f"{q['s'] + 150:.0f}: {q['width']:.3f}" for q in cr["cracks"] if q["width"] > 0.01))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        s.export(a.out / "beam.pvd")
        s.to_result(meta={"title": "RC beam, smeared cracks"}).write(a.out, "beam")
        print(f"wrote {a.out / 'beam.pvd'} and {a.out / 'beam.json'}")


if __name__ == "__main__":
    main()
