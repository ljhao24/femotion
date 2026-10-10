"""2D lattice Boltzmann (D2Q9) flow solver: BGK + Smagorinsky large-eddy model, running on Apple GPUs (hand-written Metal
kernels through MLX; extra "cfd").

Written for illustrative flow around bridge deck sections, reusable for other 2D flows: piers, cables, tower sections,
wind barriers. Illustrative only: 2D and low Reynolds number; not aerodynamic data for real bridges.

Coordinates and conventions
- field arrays have shape (NY, NX) with rows pointing down (like images); cell (i, j) is centred at (i+0.5, j+0.5).
- every physical quantity exposed to callers has y pointing up: lift up is positive, counter-clockwise vorticity positive.
- lattice units: Δx = Δt = 1, c_s² = 1/3, ν = (τ − 0.5) / 3.

Boundaries
- inlet x = 0: equilibrium at the prescribed velocity (U, 0), density taken from x = 1.
- outlet x = NX-1: unknown populations arriving from downstream extrapolated with zero gradient.
- sponge: τ increased towards the outlet (viscosity absorbs vortices); absorbing layers (after collision, relax towards
  ρ = 1, u = (U, 0) to absorb acoustic waves), placed outside the frame that is rendered. The inlet takes its density from
  downstream, so nothing else fixes the reference density: give at least an outlet layer (sigma_x) whenever the flow has a
  pressure drop (walls, obstacles), or the density drifts upwards without bound.
- top / bottom: periodic (default) or free slip (mirror reflection).
- obstacles: halfway bounce-back; forces by momentum exchange, taken from the post-collision populations every step.

Smoke lines (passive scalar)
- Dye: a scalar on the output pixel grid (may be finer than the lattice), MacCormack semi-Lagrangian advection
  (RK2 back-tracing + bilinear interpolation + extrema limiter), no diffusion; a few inlet columns are reset to a given
  profile every step, acting as smoke generators.

Rendering helpers live in femotion.cfd.render (smoothing, upscaling, grey-scale video).

Usage
    sim = LBM2D(solid, u_in=0.06, nu=0.004, c_smag=0.12, y_bc="freeslip",
                tau_x=..., sigma_x=edge_ramp(nx, lo=64, hi=192, peak=0.05), sigma_y=edge_ramp(ny, 96, 96, peak=0.05))
    sim.step(20000)                      # forces are logged every step
    m = sim.macro(); w = LBM2D.vorticity(m)
    F = sim.forces()                     # (steps, 2), physical coordinates
Validation (tests/test_cfd.py): Poiseuille channel flow (parabolic profile, constant mass flux, wall-friction pressure
gradient within 3 %); cylinder at Re = 100 (D = 40 cells, 10 % blockage, free-slip walls) gives St = 0.168, CD = 1.39,
against St ≈ 0.164–0.166, CD ≈ 1.33–1.35 unbounded (blockage raises both). About 2200 steps / s on an M1 Max
(1.7 million cells).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

# D2Q9: direction 0 rest, 1 east, 2 south (row +1, downwards in the image), 3 west, 4 north, 5 south-east, 6 south-west, 7 north-west, 8 north-east
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

_HEADER = """
constant int EX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constant int EY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constant int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
constant int MIRY[9] = {0, 1, 4, 3, 2, 8, 7, 6, 5};
constant float W9[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};

// bilinear interpolation; x, y are array index coordinates (cell centres at integers), clamped at the edges
inline float bil(const device float* a, int w, int h, float x, float y) {
    x = clamp(x, 0.0f, float(w - 1));
    y = clamp(y, 0.0f, float(h - 1));
    int x0 = min(int(x), w - 2);
    int y0 = min(int(y), h - 2);
    float fx = x - float(x0), fy = y - float(y0);
    int i = y0 * w + x0;
    return (1.0f - fy) * ((1.0f - fx) * a[i] + fx * a[i + 1])
         + fy * ((1.0f - fx) * a[i + w] + fx * a[i + w + 1]);
}
"""

# pull streaming + collision, one thread per cell. f holds the post-collision populations of the previous step.
_STEP_SRC = """
uint gid = thread_position_in_grid.x;
const int N = NX * NY;
if ((int)gid >= N) return;
int x = (int)gid % NX;
int y = (int)gid / NX;
if (solid[gid]) {
    for (int k = 0; k < 9; ++k) fout[k * N + gid] = W9[k];
    return;
}
float uin = prm[0];
float c2 = prm[1];
if (x == 0) {
    float r = 0.0f;
    for (int k = 0; k < 9; ++k) r += f[k * N + gid + 1];
    float usq = uin * uin;
    for (int k = 0; k < 9; ++k) {
        float eu = float(EX[k]) * uin;
        fout[k * N + gid] = W9[k] * r * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
    }
    return;
}
float fi[9];
for (int k = 0; k < 9; ++k) {
    int xs = min(x - EX[k], NX - 1);
    int ys = y - EY[k];
    int kk = k;
    if (ys < 0 || ys >= NY) {
        if (FREESLIP) { ys = y; kk = MIRY[k]; }
        else { ys = (ys + NY) % NY; }
    }
    int s = ys * NX + xs;
    fi[k] = solid[s] ? f[OPP[k] * N + gid] : f[kk * N + s];
}
float rho = 0.0f, jx = 0.0f, jy = 0.0f;
for (int k = 0; k < 9; ++k) {
    rho += fi[k];
    jx += float(EX[k]) * fi[k];
    jy += float(EY[k]) * fi[k];
}
float ux = jx / rho, uy = jy / rho;
float usq = ux * ux + uy * uy;
float feq[9];
float pxx = 0.0f, pyy = 0.0f, pxy = 0.0f;
for (int k = 0; k < 9; ++k) {
    float eu = float(EX[k]) * ux + float(EY[k]) * uy;
    feq[k] = W9[k] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
    float d = fi[k] - feq[k];
    pxx += float(EX[k] * EX[k]) * d;
    pyy += float(EY[k] * EY[k]) * d;
    pxy += float(EX[k] * EY[k]) * d;
}
float Q = sqrt(pxx * pxx + pyy * pyy + 2.0f * pxy * pxy);
float t0 = tau0[x];
// Hou et al. 1996：tau = (tau0 + sqrt(tau0^2 + 18*sqrt(2)*C^2*|Pi_neq|/rho)) / 2
float tau = 0.5f * (t0 + sqrt(t0 * t0 + 25.4558441f * c2 * Q / rho));
float om = 1.0f / tau;
float sg = max(sigx[x], sigy[y]);
if (sg > 0.0f) {
    // absorbing layer: after collision relax towards the target equilibrium (ρ=1, u=(U,0)), absorbing sound waves and disturbances
    float u2 = uin * uin;
    for (int k = 0; k < 9; ++k) {
        float eu = float(EX[k]) * uin;
        float ft = W9[k] * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * u2);
        float fp = fi[k] - om * (fi[k] - feq[k]);
        fout[k * N + gid] = fp - sg * (fp - ft);
    }
} else {
    for (int k = 0; k < 9; ++k) fout[k * N + gid] = fi[k] - om * (fi[k] - feq[k]);
}
"""

_MACRO_SRC = """
uint gid = thread_position_in_grid.x;
const int N = NX * NY;
if ((int)gid >= N) return;
float r = 0.0f, jx = 0.0f, jy = 0.0f;
for (int k = 0; k < 9; ++k) {
    float v = f[k * N + gid];
    r += v; jx += float(EX[k]) * v; jy += float(EY[k]) * v;
}
out[gid] = r;
out[N + gid] = jx / r;
out[2 * N + gid] = jy / r;
"""

# smoke MacCormack, step 1: plain semi-Lagrangian (RK2 back-tracing)
_DYE1_SRC = """
uint gid = thread_position_in_grid.x;
if ((int)gid >= W * H) return;
int i = (int)gid % W, j = (int)gid / W;
float dt = prm[0], s = prm[1];
if (i < (int)prm[2]) { out[gid] = prof[j]; return; }
const int NL = NXL * NYL;
const device float* ux = uv + NL;
const device float* uy = uv + 2 * NL;
float px = float(i) + 0.5f, py = float(j) + 0.5f;
float k = dt * s;
float ox = prm[3] - 0.5f, oy = prm[4] - 0.5f;
float u1 = bil(ux, NXL, NYL, px / s + ox, py / s + oy);
float v1 = bil(uy, NXL, NYL, px / s + ox, py / s + oy);
float mx_ = px - 0.5f * k * u1, my_ = py - 0.5f * k * v1;
float u2 = bil(ux, NXL, NYL, mx_ / s + ox, my_ / s + oy);
float v2 = bil(uy, NXL, NYL, mx_ / s + ox, my_ / s + oy);
out[gid] = bil(phi, W, H, px - k * u2 - 0.5f, py - k * v2 - 0.5f);
"""

# step 2: trace forward again to estimate the error, correct by half, then clamp to the four values around the back-traced point
_DYE2_SRC = """
uint gid = thread_position_in_grid.x;
if ((int)gid >= W * H) return;
int i = (int)gid % W, j = (int)gid / W;
float dt = prm[0], s = prm[1];
if (i < (int)prm[2]) { out[gid] = prof[j]; return; }
const int NL = NXL * NYL;
const device float* ux = uv + NL;
const device float* uy = uv + 2 * NL;
float px = float(i) + 0.5f, py = float(j) + 0.5f;
float k = dt * s;
float ox = prm[3] - 0.5f, oy = prm[4] - 0.5f;
float u1 = bil(ux, NXL, NYL, px / s + ox, py / s + oy);
float v1 = bil(uy, NXL, NYL, px / s + ox, py / s + oy);
// extrema around the back-traced point (same as step 1)
float mx_ = px - 0.5f * k * u1, my_ = py - 0.5f * k * v1;
float u2 = bil(ux, NXL, NYL, mx_ / s + ox, my_ / s + oy);
float v2 = bil(uy, NXL, NYL, mx_ / s + ox, my_ / s + oy);
float bx = clamp(px - k * u2 - 0.5f, 0.0f, float(W - 1));
float by = clamp(py - k * v2 - 0.5f, 0.0f, float(H - 1));
int b = min(int(by), H - 2) * W + min(int(bx), W - 2);
float a0 = phi[b], a1 = phi[b + 1], a2 = phi[b + W], a3 = phi[b + W + 1];
float lo = min(min(a0, a1), min(a2, a3)), hi = max(max(a0, a1), max(a2, a3));
// forward trace
float fx_ = px + 0.5f * k * u1, fy_ = py + 0.5f * k * v1;
float u3 = bil(ux, NXL, NYL, fx_ / s + ox, fy_ / s + oy);
float v3 = bil(uy, NXL, NYL, fx_ / s + ox, fy_ / s + oy);
float back = bil(phihat, W, H, px + k * u3 - 0.5f, py + k * v3 - 0.5f);
float r = phihat[gid] + 0.5f * (phi[gid] - back);
out[gid] = clamp(r, lo, hi);
"""

_kernels: dict[str, object] = {}


def _kernel(name: str, inputs: list[str], src: str):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=["fout" if name == "lbm_step" else "out"],
            source=src, header=_HEADER)
    return _kernels[name]


def equilibrium(rho: float, ux: float, uy: float) -> np.ndarray:
    """equilibrium populations (9,) of one cell; uy in array direction (downwards positive)."""
    eu = EX * ux + EY * uy
    return W9 * rho * (1 + 3 * eu + 4.5 * eu ** 2 - 1.5 * (ux * ux + uy * uy))


# ---------------------------------------------------------------- geometry

def polygon_mask(nx: int, ny: int, verts: np.ndarray) -> np.ndarray:
    """True where the cell centre lies inside the polygon. verts: (n, 2) in array coordinates (x right, y down, continuous). Even-odd rule."""
    xs = np.arange(nx) + 0.5
    ys = np.arange(ny) + 0.5
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros((ny, nx), bool)
    v = np.asarray(verts, float)
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        (x1, y1), (x2, y2) = a, b
        if y1 == y2:
            continue
        cond = (Y >= min(y1, y2)) & (Y < max(y1, y2))
        xint = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (X < xint)
    return inside


def rect_mask(nx: int, ny: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """True where the cell centre lies inside the rectangle [x0, x1] × [y0, y1] (array coordinates)."""
    xs = np.arange(nx) + 0.5
    ys = np.arange(ny) + 0.5
    return ((ys >= y0) & (ys <= y1))[:, None] & ((xs >= x0) & (xs <= x1))[None, :]


# ---------------------------------------------------------------- solver

def edge_ramp(n: int, lo: int = 0, hi: int = 0, peak: float = 1.0, power: float = 2.0) -> np.ndarray:
    """1D profile of length n: zero in the middle, rising as a power law to `peak` within lo cells at the low end and hi cells
    at the high end (for sponges and absorbing layers)."""
    i = np.arange(n) + 0.5
    r = np.zeros(n)
    if lo > 0:
        r = np.maximum(r, np.clip((lo - i) / lo, 0, 1))
    if hi > 0:
        r = np.maximum(r, np.clip((i - (n - hi)) / hi, 0, 1))
    return peak * r ** power


class LBM2D:
    """D2Q9 BGK + Smagorinsky.

    solid: (NY, NX) bool, True for solid. u_in: inlet velocity (lattice units). nu: molecular kinematic viscosity (lattice units).
    c_smag: Smagorinsky constant. y_bc: "periodic" or "freeslip".
    tau_x: (NX,) molecular relaxation time τ0 per column, 3ν+0.5 everywhere if omitted; the outlet sponge raises τ here.
    sigma_x, sigma_y: (NX,), (NY,) absorbing-layer strength, the larger of the two per cell (σ); after collision relax by σ
    towards the equilibrium at ρ=1, u=(U,0) (absorbs acoustic waves).
    """

    def __init__(self, solid: np.ndarray, u_in: float, nu: float, c_smag: float = 0.12,
                 y_bc: str = "periodic", tau_x: np.ndarray | None = None,
                 sigma_x: np.ndarray | None = None, sigma_y: np.ndarray | None = None,
                 init_u: float | None = None):
        solid = np.asarray(solid, bool)
        self.ny, self.nx = solid.shape
        self.N = self.nx * self.ny
        self.solid_np = solid
        self.solid = mx.array(solid.astype(np.uint8))
        self.u_in = float(u_in)
        self.nu = float(nu)
        self.tau = 3.0 * nu + 0.5
        self.c_smag = float(c_smag)
        self.freeslip = y_bc == "freeslip"
        tau0 = np.full(self.nx, self.tau) if tau_x is None else np.asarray(tau_x, float)
        self.tau0 = mx.array(tau0.astype(np.float32))
        self.sigx = mx.array((np.zeros(self.nx) if sigma_x is None else np.asarray(sigma_x)).astype(np.float32))
        self.sigy = mx.array((np.zeros(self.ny) if sigma_y is None else np.asarray(sigma_y)).astype(np.float32))
        u0 = self.u_in if init_u is None else init_u
        f = np.empty((9, self.ny, self.nx), np.float32)
        f[:] = equilibrium(1.0, u0, 0.0)[:, None, None]
        f[:, solid] = W9[:, None]
        self.f = mx.array(f)
        self.t = 0
        self._u_now = None
        self._prm = None
        self._build_links()
        self._force_buf: list[mx.array] = []
        self.force_log: list[np.ndarray] = []  # (Fx, Fy) per step, physical coordinates (y up)

    # momentum exchange: when the neighbour of fluid cell x in direction k is solid, the post-collision f_k(x) bounces back
    # and gives the body a momentum 2 e_k f_k
    def _build_links(self):
        idx, wts = [], []
        fluid = ~self.solid_np
        flat = np.arange(self.N).reshape(self.ny, self.nx)
        for k in range(1, 9):
            nb_solid = np.roll(self.solid_np, (-EY[k], -EX[k]), axis=(0, 1))
            m = fluid & nb_solid
            m[:, 0] = False
            m[:, -1] = False
            cells = flat[m]
            idx.append(k * self.N + cells)
            wts.append(np.tile([2.0 * EX[k], -2.0 * EY[k]], (cells.size, 1)))  # physical coordinates: y up
        self.n_links = int(sum(i.size for i in idx))
        self._link_idx = mx.array(np.concatenate(idx).astype(np.int32))
        self._link_w = mx.array(np.concatenate(wts).astype(np.float32))

    def force(self) -> mx.array:
        """force (Fx, Fy) that the current (post-collision) populations exert on all solids at the next streaming, physical coordinates."""
        g = mx.take(self.f.reshape(-1), self._link_idx)
        return g @ self._link_w

    def _params(self, u: float) -> mx.array:
        if self._prm is None or self._u_now != u:
            self._prm = mx.array([u, self.c_smag ** 2], dtype=mx.float32)
            self._u_now = u
        return self._prm

    def step(self, n: int = 1, record_force: bool = True, u_in: float | None = None, eval_every: int = 50):
        k = _kernel("lbm_step", ["f", "solid", "tau0", "sigx", "sigy", "prm"], _STEP_SRC)
        prm = self._params(self.u_in if u_in is None else u_in)
        for _ in range(n):
            if record_force:
                self._force_buf.append(self.force())
            self.f = k(inputs=[self.f, self.solid, self.tau0, self.sigx, self.sigy, prm],
                       template=[("NX", self.nx), ("NY", self.ny), ("FREESLIP", self.freeslip)],
                       grid=(self.N, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(9, self.ny, self.nx)], output_dtypes=[mx.float32])[0]
            self.t += 1
            if self.t % eval_every == 0:
                self._flush()
        self._flush()

    def _flush(self):
        if self._force_buf:
            F = mx.stack(self._force_buf)
            mx.eval(self.f, F)
            self.force_log.append(np.array(F))
            self._force_buf = []
        else:
            mx.eval(self.f)

    def forces(self) -> np.ndarray:
        """force at every step so far, (n, 2), physical coordinates."""
        self._flush()
        return np.concatenate(self.force_log) if self.force_log else np.zeros((0, 2))

    def macro(self) -> mx.array:
        """(3, NY, NX): density, ux, uy (uy in array direction, downwards positive)."""
        k = _kernel("lbm_macro", ["f"], _MACRO_SRC)
        return k(inputs=[self.f], template=[("NX", self.nx), ("NY", self.ny)],
                 grid=(self.N, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(3, self.ny, self.nx)], output_dtypes=[mx.float32])[0]

    @staticmethod
    def vorticity(m: mx.array) -> mx.array:
        """vorticity (physical coordinates, counter-clockwise positive) = ∂v/∂x − ∂u/∂y = −∂uy/∂x + ∂ux/∂row, central
        differences, zero at the edges."""
        ux, uy = m[1], m[2]
        dvx = (uy[1:-1, 2:] - uy[1:-1, :-2]) * 0.5
        duy = (ux[2:, 1:-1] - ux[:-2, 1:-1]) * 0.5
        w = duy - dvx
        return mx.pad(w, ((1, 1), (1, 1)))

    @staticmethod
    def speed(m: mx.array) -> mx.array:
        return mx.sqrt(m[1] * m[1] + m[2] * m[2])

    def check(self) -> bool:
        m = self.macro()
        ok = bool(mx.all(mx.isfinite(m)).item())
        return ok


# ---------------------------------------------------------------- smoke lines

class Dye:
    """passive scalar on a pixel grid. scale = pixels per lattice cell; the lattice may be wider than the frame
    (sponges outside the frame).

    profile: (H,) inlet profile; the leftmost inject_w columns are reset to it after every advection.
    offset: position (ox, oy) of the frame's top-left corner in the lattice (cells); the lattice may have absorbing layers
    outside the frame on all sides.
    """

    def __init__(self, w: int, h: int, scale: float, profile: np.ndarray, inject_w: int = 4,
                 offset: tuple[float, float] = (0.0, 0.0)):
        self.w, self.h = int(w), int(h)
        self.scale = float(scale)
        self.offset = (float(offset[0]), float(offset[1]))
        self.inject_w = int(inject_w)
        self.prof = mx.array(np.asarray(profile, np.float32))
        self.phi = mx.zeros((self.h, self.w), dtype=mx.float32)

    def advect(self, m: mx.array, dt: float):
        """m is the output of LBM2D.macro() (its velocities are used); dt is the number of lattice steps this advection spans."""
        nyl, nxl = m.shape[1], m.shape[2]
        prm = mx.array([dt, self.scale, self.inject_w, *self.offset], dtype=mx.float32)
        tmpl = [("W", self.w), ("H", self.h), ("NXL", nxl), ("NYL", nyl)]
        n = self.w * self.h
        k1 = _kernel("dye_sl", ["phi", "uv", "prm", "prof"], _DYE1_SRC)
        k2 = _kernel("dye_mc", ["phi", "phihat", "uv", "prm", "prof"], _DYE2_SRC)
        hat = k1(inputs=[self.phi, m, prm, self.prof], template=tmpl, grid=(n, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(self.h, self.w)], output_dtypes=[mx.float32])[0]
        self.phi = k2(inputs=[self.phi, hat, m, prm, self.prof], template=tmpl, grid=(n, 1, 1),
                      threadgroup=(256, 1, 1), output_shapes=[(self.h, self.w)], output_dtypes=[mx.float32])[0]
        # evaluate now: MLX is lazy, and an unevaluated advection chain keeps every step's populations alive until memory runs out
        mx.eval(self.phi)

    def frame_u8(self) -> np.ndarray:
        return np.array(mx.round(mx.clip(self.phi, 0.0, 1.0) * 255.0).astype(mx.uint8))


def smoke_profile(h: int, n_lines: int, y0: float, y1: float, sigma: float, peak: float = 1.0) -> np.ndarray:
    """n_lines equally spaced thin Gaussian bands centred from y0 to y1 (pixels, continuous coordinates); returns (h,)."""
    y = np.arange(h) + 0.5
    centers = np.linspace(y0, y1, n_lines)
    p = np.zeros(h)
    for c in centers:
        p = np.maximum(p, np.exp(-0.5 * ((y - c) / sigma) ** 2))
    return (peak * p).astype(np.float32)


# ---------------------------------------------------------------- rendering: see femotion.cfd.render
