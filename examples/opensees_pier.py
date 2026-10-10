"""OpenSees back-end example: a circular reinforced-concrete bridge pier (illustrative values).

D = 2 m, 10 m tall, 40 Φ32 bars (fy 400 MPa), concrete fc 26.8 MPa; 1500 t of superstructure on top.
Fiber section (cover + hoop-confined core + bars), P-Δ.
1) self-weight → pushover to 4 % drift: peak lateral force and softening;
2) a ground motion at 1× and 3×: peak drift and residual drift (non-zero only once a plastic hinge has formed).

    python examples/opensees_pier.py [--record FILE] [--out DIR]

--record: a ground-motion file (two columns t, a or one column a; in g or m/s²). Without it a synthetic 15 s motion is
used. --out writes the 3× time history, resampled to 30 fps, as fem-result/1.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from femotion.dynamics import load_accel, resample
from femotion.frame import Model, Section
from femotion.frame.opensees import Fiber, OpsFrame, to_result

D, HP, FC, FY = 2.0, 10.0, 26.8e3, 400e3  # m, m, kPa, kPa
M_TOP = 1500.0  # t


def synthetic_motion(dt: float = 0.02, dur: float = 15.0, pga: float = 3.4) -> np.ndarray:
    """an amplitude-modulated chirp from 0.5 to 6 Hz, peak ground acceleration `pga` (m/s²)"""
    t = np.arange(0, dur, dt)
    env = (t / 2) ** 2 * np.exp(-(t - 2) / 3) * (t < 2) + np.exp(-(t - 2) / 3) * (t >= 2)
    f = 0.5 + 5.5 * t / dur
    a = env * np.sin(2 * np.pi * np.cumsum(f) * dt)
    return pga * a / np.abs(a).max()


def build():
    Ag, Ic = np.pi * D**2 / 4, np.pi * D**4 / 64
    m = Model()
    ns = [m.node((0, 0, HP * k / 5)) for k in range(6)]
    for a, b in zip(ns[:-1], ns[1:]):
        m.beam(a, b, Section(E=3.25e7, A=Ag, Iy=Ic, Iz=Ic, J=2 * Ic), ref=(1, 0, 0), tag="pier")
    m.fix(ns[0])
    M = np.zeros(m.ndof)
    for k in ns[1:]:
        M[6 * k:6 * k + 3] = Ag * 2.5 * HP / 5
        M[6 * k + 3:6 * k + 6] = 1.0
    M[6 * ns[-1]:6 * ns[-1] + 3] += M_TOP
    G = np.zeros(m.ndof)
    G[2::6] = -M[2::6] * 9.81
    fib = {"pier": Fiber.rc_circle(D=D, cover=0.06, n_bars=40, d_bar=0.032, fc=FC, fy=FY, Ec=3.25e7)}
    return m, ns, M, G, fib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    m, ns, M, G, fib = build()
    top = ns[-1]
    t = time.time()
    f = OpsFrame(m, transf="PDelta", fiber=fib, masses=M)
    f.static(G, steps=5)
    print(f"axial load ratio {M_TOP * 9.81 / (FC * np.pi * D**2 / 4):.3f}, first frequency {f.modes(1)[0][0]:.3f} Hz")
    Fl = np.zeros(m.ndof)
    Fl[6 * top] = 1.0
    lam, Us = f.pushover(Fl, node=top, dof=0, target=0.04 * HP, steps=200)
    i = int(np.argmax(lam))
    print(f"pushover: peak {lam[i]:.0f} kN at {Us[i, 6 * top] / HP:.2%} drift, {lam[-1]:.0f} kN at 4 % drift "
          f"({lam[-1] / lam[i]:.0%} of peak) [{time.time() - t:.1f} s]")
    ag, dt = load_accel(a.record) if a.record else (synthetic_motion(), 0.02)
    for scale in (1.0, 3.0):
        t = time.time()
        f = OpsFrame(m, transf="PDelta", fiber=fib, masses=M)
        f.static(G, steps=5)
        T, U = f.transient(ag * scale, dt, direction=0, zeta=0.05, t_end=15.0)
        x = U[:, 6 * top] / HP
        print(f"ground motion × {scale:.0f}: peak drift {np.abs(x).max():.2%}, residual {x[-1]:+.2%} [{time.time() - t:.1f} s]")
    if a.out:
        Uf = resample(U, dt, 30, T[-1])
        to_result(m, Uf, meta=dict(title="RC pier under ground motion × 3 (illustrative)", fps=30)).write(a.out, "opensees_pier")


if __name__ == "__main__":
    main()
