"""2D lattice Boltzmann on Apple GPUs (extra "cfd"): uniform flow, Poiseuille channel flow (profile, mass flux, wall friction),
symmetry, and the cylinder at Re = 100 against the literature."""
import numpy as np
import pytest

pytest.importorskip("mlx")
from femotion.cfd import lbm2d as lb  # noqa: E402


def test_uniform_flow_stays_uniform():
    sim = lb.LBM2D(np.zeros((20, 60), bool), u_in=0.05, nu=0.05, c_smag=0.0)
    sim.step(300, record_force=False)
    rho, ux, uy = np.array(sim.macro())
    assert np.abs(rho - 1).max() < 1e-5 and np.abs(ux - 0.05).max() < 1e-5 and np.abs(uy).max() < 1e-6


def test_channel_flow_is_poiseuille():
    """walls as solid rows (halfway bounce-back puts them at cell faces), outlet absorbing layer fixes the reference density"""
    H, NX, nu = 24, 320, 0.1
    solid = np.zeros((H + 2, NX), bool)
    solid[0] = solid[-1] = True
    sim = lb.LBM2D(solid, u_in=0.04, nu=nu, c_smag=0.0, init_u=0.0, sigma_x=lb.edge_ramp(NX, hi=60, peak=0.05))
    sim.step(12000, record_force=False)
    rho, ux = (a[1:-1] for a in np.array(sim.macro())[:2])
    flux = (rho * ux).sum(axis=0)
    assert np.ptp(flux[2:250]) < 1e-3 * flux[2:250].mean()  # mass is conserved along the channel
    y = np.arange(H) + 0.5
    u = ux[:, 160]
    assert u.max() / u.mean() == pytest.approx(1.5, rel=5e-3)
    fit = np.polyfit(y * (H - y), u, 1)
    assert np.abs(np.polyval(fit, y * (H - y)) - u).max() < 5e-3 * u.max()
    # wall friction: dp/dx = −12 μ ū / H², with p = ρ / 3 in lattice units
    r = rho.mean(axis=0)
    slope = (r[200] - r[100]) / 100
    ubar, rbar = ux[:, 100:201].mean(), r[100:201].mean()
    assert -slope == pytest.approx(36 * nu * rbar * ubar / H**2, rel=0.03)


def test_symmetric_body_has_no_lift():
    NY, NX = 80, 300
    sim = lb.LBM2D(lb.rect_mask(NX, NY, 70, 30, 90, 50), u_in=0.04, nu=0.04, c_smag=0.0, y_bc="freeslip")
    sim.step(4000)
    Fx, Fy = sim.forces()[-500:].mean(axis=0)
    assert Fx > 0 and abs(Fy) < 1e-4 * Fx


@pytest.mark.slow
def test_cylinder_at_re_100():
    """D = 40 cells, 10 % blockage, free-slip walls. Unbounded literature: St ≈ 0.164–0.166, CD ≈ 1.33–1.35 (Williamson 1996);
    blockage raises both. The reference values pin this solver's own result."""
    D, U = 40, 0.05
    NY, NX, nu = 10 * D, 30 * D, U * D / 100
    yy, xx = np.mgrid[0:NY, 0:NX] + 0.5
    solid = (xx - 8 * D) ** 2 + (yy - NY / 2 - 0.5) ** 2 <= (D / 2) ** 2  # half a cell off-centre starts the shedding
    tau = np.full(NX, 3 * nu + 0.5) + lb.edge_ramp(NX, hi=6 * D, peak=0.5)
    sim = lb.LBM2D(solid, u_in=U, nu=nu, c_smag=0.0, y_bc="freeslip", tau_x=tau, sigma_x=lb.edge_ramp(NX, hi=2 * D, peak=0.05))
    sim.step(80000)
    F = sim.forces()[40000:]
    cd = F[:, 0].mean() / (0.5 * U * U * D)
    lift = np.convolve(F[:, 1] - F[:, 1].mean(), np.ones(400) / 400, mode="valid")  # drop acoustic ripple (period ≈ 4500 steps)
    up = np.where((lift[1:] > 0) & (lift[:-1] <= 0))[0]
    st = D / (U * (up[-1] - up[0]) / (len(up) - 1))
    assert 0.164 < st < 0.18 and 1.33 < cd < 1.5
    assert st == pytest.approx(0.1678, rel=0.01) and cd == pytest.approx(1.391, rel=0.01)
