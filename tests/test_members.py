"""JTG 3362-2018 members beyond uniaxial compression and bending (femotion.codes.jtg.members): tension members (5.4),
spiral columns (5.3.2), bars over the web (5.3.7), biaxial compression (5.3.11), effective lengths (appendix E),
appendix F, and shear of circular sections (GB 50010-2010 6.3.15).

Every function has at least one check from an independent source: a different method (elastic buckling eigenvalues,
biaxial strain compatibility, the rule set of RCSection.forces, the rigid-plastic web strip), a physical limit
(e → 0, e → ∞, symmetry), the code's own commentary (appendix E, F derivations), or a published reading (GB 50010
6.3.15 worked example). Hand calculations with the same formulas only guard the implementation."""
import math

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq

from femotion.codes import jtg
from femotion.codes.jtg import members

A = lambda d: math.pi * d * d / 4  # noqa: E731


# ── appendix E ──────────────────────────────────────────────────────────────
def _buckling_k(kth, kF, n=40):
    """effective-length factor of a column fixed at the base with a rotational spring kθ = Kθ l/EI and a lateral spring
    kF = KF l³/EI at the top: lowest eigenvalue of K − P G with Hermite beam elements (independent of appendix E)"""
    L = 1.0 / n
    ke = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L], [-12, -6 * L, 12, -6 * L],
                   [6 * L, 2 * L * L, -6 * L, 4 * L * L]]) / L ** 3
    kg = np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L], [-36, -3 * L, 36, -3 * L],
                   [3 * L, -L * L, -3 * L, 4 * L * L]]) / (30 * L)
    K = np.zeros((2 * n + 2,) * 2)
    G = np.zeros_like(K)
    for e in range(n):
        i = np.arange(2 * e, 2 * e + 4)
        K[np.ix_(i, i)] += ke
        G[np.ix_(i, i)] += kg
    K[-2, -2] += kF
    K[-1, -1] += kth
    f = np.arange(2, 2 * n + 2)
    w = eigh(K[np.ix_(f, f)], G[np.ix_(f, f)], eigvals_only=True)
    return math.pi / math.sqrt(w[w > 0].min())


def test_length_factor_classic_cases():
    """E.0.1 values; E.0.2-1 and E.0.3 at their limits approach the classic Euler cases (fixed–free 2, fixed–pinned 0.7,
    fixed–guided 1, fixed–fixed 0.5)"""
    assert [jtg.length_factor(ends=e) for e in ("fixed-fixed", "fixed-pinned", "pinned-pinned", "fixed-free")] == [0.5, 0.7, 1.0, 2.0]
    assert jtg.length_factor(k_F=0.0) == 2.0 and jtg.length_factor(k_F=math.inf) == 0.7
    assert jtg.length_factor(k_theta=math.inf, k_F=math.inf) == pytest.approx(0.5)
    assert jtg.length_factor(k_theta=math.inf, k_F=0.0) == pytest.approx(1.0, rel=0.01)
    assert jtg.length_factor(k_theta=1e-9, k_F=0.0) == pytest.approx(2.0, rel=0.015)
    with pytest.raises(ValueError):
        jtg.length_factor(ends="hinged")


def test_length_factor_against_buckling_eigenvalue():
    """appendix E fits against the exact elastic buckling of the restrained column (FE eigenvalue; commentary E-8):
    the model reproduces the commentary's characteristic equation E-17 for kθ = 0; E.0.2-1 lies within 8 % of the
    exact k, E.0.3 is never below it (conservative) and within 10 %"""
    for kF in (1.0, 5.0, 20.0):
        u = math.pi / _buckling_k(0.0, kF)
        assert math.tan(u) == pytest.approx(u - u ** 3 / kF, abs=1e-6)  # E-17
    for kth in (0.1, 1.0, 5.0, 50.0):
        for kF in (0.5, 2.0, 10.0, 50.0):
            assert jtg.length_factor(k_theta=kth, k_F=kF) == pytest.approx(_buckling_k(kth, kF), rel=0.08)
    for kF in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0):
        k, ke = jtg.length_factor(k_F=kF), _buckling_k(0.0, kF)
        assert ke - 1e-9 <= k <= 1.10 * ke


def test_effective_length_from_springs():
    """kθ = Kθ l / EI, kF = KF l³ / EI (E.0.2-2, -3); l0 = k l"""
    l, EI = 8000.0, 3.15e4 * jtg.circle(1200, 20, 25, 60).I
    Kt, KF = 2.0 * EI / l, 10.0 * EI / l ** 3
    assert jtg.effective_length(l, K_theta=Kt, K_F=KF, EI=EI) == pytest.approx(jtg.length_factor(k_theta=2.0, k_F=10.0) * l)
    assert jtg.effective_length(l, K_F=KF, EI=EI) == pytest.approx(jtg.length_factor(k_F=10.0) * l)
    assert jtg.effective_length(l, ends="fixed-free") == 2 * l and jtg.effective_length(l, k=0.8) == 0.8 * l


# ── 5.3.2 spiral columns ────────────────────────────────────────────────────
def test_spiral_column_formula():
    """5.3.2-1, -2 by hand; the spiral term against Richart's confinement law f'cc = f'c + 4.1 fl (Richart, Brandtzaeg
    and Brown 1928, the basis of k ≈ 2): 4.1 fl Acor = 2.05 fsd Aso"""
    col = jtg.circle(600, 8, 22, 50, concrete="C30")
    r = jtg.axial_spiral(col, 3000, 3000, spiral=(10, 50))
    dcor = 2 * 250 + 22
    Acor, Aso = math.pi * dcor ** 2 / 4, math.pi * dcor * A(10) / 50
    N532 = 0.9 * (13.8 * Acor + 330 * 8 * A(22) + 2.0 * 330 * Aso) / 1e3
    assert r.details["dcor"] == dcor and r.resistance == pytest.approx(N532)
    fl = 2 * 330 * A(10) / (dcor * 50)  # lateral pressure of a yielding spiral
    assert 0.9 * 2.0 * 330 * Aso == pytest.approx(0.9 * 4.1 * fl * Acor / 1.025, rel=1e-12)  # k = 2.0 vs 4.1/2 = 2.05
    assert r.ok and not r.failed


def test_spiral_column_conditions():
    """the spiral counts only with l0/i ≤ 48, Aso ≥ 0.25 A's, s ≤ 80 mm and ≤ dcor/5, and when 5.3.2-1 exceeds 5.3.1;
    at most 1.5 × 5.3.1; 9.6.2 detailing"""
    col = jtg.circle(600, 8, 22, 50, concrete="C30")
    N531 = jtg.axial(col, 3000, 3000).resistance
    assert jtg.axial_spiral(col, 3000, 3000, spiral=(10, 100)).resistance == pytest.approx(N531)  # pitch > 80
    assert jtg.axial_spiral(col, 3000, 13000, spiral=(10, 50)).resistance == pytest.approx(jtg.axial(col, 3000, 13000).resistance)  # l0/i > 48
    assert jtg.axial_spiral(col, 3000, 3000, spiral=(8, 80)).resistance == pytest.approx(N531)  # 5.3.2-1 < 5.3.1
    big = jtg.axial_spiral(col, 3000, 3000, spiral=(14, 40))
    assert big.resistance == pytest.approx(1.5 * N531) and big.details["N_5_3_2"] > 1.5 * N531
    thin = jtg.circle(600, 8, 22, 50, concrete="C30")
    r = jtg.axial_spiral(thin, 3000, 3000, spiral=(12, 30))
    assert any("40 mm" in f for f in r.failed)  # pitch < 40 mm (9.6.2)
    heavy = jtg.circle(600, 6, 36, 60, concrete="C30")
    r = jtg.axial_spiral(heavy, 3000, 3000, spiral=(8, 40))
    assert r.clause.startswith("JTG 3362-2018 5.3.2") and any("d/4" in f for f in r.failed)  # φ8 < 36/4
    with pytest.raises(ValueError):
        jtg.axial_spiral(jtg.rect(500, 500, bottom=[(3, 20, 40)], top=[(3, 20, 40)]), 1000, 3000, spiral=(10, 50))


# ── 5.3.7 bars over the web ─────────────────────────────────────────────────
def _pier(n_side=8):
    side = [(2 * A(20), z) for z in np.linspace(250, 1750, n_side)]
    return jtg.rect(1200, 2000, concrete="C35", top=[(8, 28, 60)], bottom=[(8, 28, 60)], bars=side)


def test_web_method_close_to_strain_method():
    """5.3.7 against the bar-by-bar strain method (5.1.3–5.1.5, a different method): within 2 % for a solid pier with
    side bars and a box pier, large and small eccentricity"""
    s = _pier()
    for N, M in [(1000, 3500), (2000, 4000), (3000, 3000), (8000, 3000), (15000, 2000), (30000, 300)]:
        a, b = jtg.eccentric(s, N, M, method="strain"), jtg.eccentric(s, N, M, method="web")
        assert b.resistance == pytest.approx(a.resistance, rel=0.02)
        assert b.clause.startswith("JTG 3362-2018 5.3.7")
    box = jtg.box(3000, 2000, 400, 400, 800, concrete="C40", top=[(20, 25, 70)], bottom=[(20, 25, 70)],
                  bars=[(2 * A(20), z) for z in np.linspace(300, 1700, 7)])
    for N, M in [(5000, 9000), (10000, 6000), (50000, 3000)]:
        a, b = jtg.eccentric(box, N, M, method="strain"), jtg.eccentric(box, N, M, method="web")
        assert b.resistance == pytest.approx(a.resistance, rel=0.02)


def test_web_bar_forces_are_the_rigid_plastic_strip():
    """5.3.7-3, -4 equal the force and the moment about As of a rigid-plastic steel strip spread over hsw = h0 − a's
    (fsw in compression above the neutral axis x/β, in tension below): integrated numerically here"""
    s = _pier()
    for N, M in [(3000, 3000), (2000, 4000), (8000, 3000)]:
        r = jtg.eccentric(s, N, M, method="web")
        d = r.details
        h0, hsw, fsw, Asw = d["h0"], d["hsw"], d["fsw"], d["Asw"]
        a_s = h0 - hsw
        xc = d["x"] / s.concrete.beta
        z = np.linspace(a_s, h0, 200001)
        sig = np.where(z < xc, fsw, -fsw)  # compression +
        q = Asw / hsw
        Nsw = np.trapezoid(sig * q, z)
        Msw = np.trapezoid(sig * q * (h0 - z), z)
        assert d["N_sw"] * 1e3 == pytest.approx(Nsw, rel=1e-4, abs=1e-3 * fsw * Asw)
        assert d["M_sw"] * 1e6 == pytest.approx(Msw, rel=1e-4, abs=1e-3 * fsw * Asw * hsw)


def test_web_method_hand_formula_and_conditions():
    """5.3.7-1, -2 by hand at the solution x (rectangle, large eccentricity, ξ ≤ β); needs 4 web rows and RC"""
    s = _pier()
    r = jtg.eccentric(s, 3000, 3000, method="web")
    d = r.details
    c, x, h0, hsw = s.concrete, d["x"], d["h0"], d["hsw"]
    xi, om = x / h0, hsw / h0
    As = 8 * A(28)
    Nsw = (1 + (xi - 0.8) / (0.4 * om)) * 330 * d["Asw"]
    Msw = (0.5 - ((xi - 0.8) / (0.8 * om)) ** 2) * 330 * d["Asw"] * hsw
    N = c.fcd * xi * 1200 * h0 + 330 * As - 330 * As + Nsw  # f'sd A's − σs As, σs = fsd (ξ ≤ ξb)
    e = r.details["e"] + h0 - 1000
    assert d["case"] == "large eccentricity" and r.resistance == pytest.approx(N / 1e3, rel=1e-9)
    assert N * e == pytest.approx(c.fcd * xi * (1 - 0.5 * xi) * 1200 * h0 ** 2 + 330 * As * (h0 - 60) + Msw, rel=1e-6)
    with pytest.raises(ValueError):
        jtg.eccentric(_pier(3), 3000, 3000, method="web")  # 3 web rows
    with pytest.raises(ValueError):
        jtg.eccentric(jtg.circle(1200, 24, 25, 60), 3000, 1000, method="web")


# ── appendix F and 5.3.8 ────────────────────────────────────────────────────
def test_table_F01_cells():
    """table F.0.1 transcribed cell by cell (checked against the scan p134–136); the commentary (p259) says it was
    computed from 5.3.8 with rs/r = 0.9: every one of the 846 cells is reproduced to its 4 decimals"""
    T = jtg.TABLE_F01
    assert T.shape == (47, 18)
    assert (T[0, 0], T[0, -1], T[-1, 0], T[-1, -1]) == (1.0487, 2.1758, 0.0058, 0.0822)
    assert (T[10, 8], T[30, 9], T[38, 13]) == (0.6786, 0.1908, 0.0987)  # ηe0/r 0.50, 2.00, 6.00
    g = 0.9
    for i, er in enumerate(members.F01_E):
        for j, lam in enumerate(members.F01_RHO):
            def NM(a):
                at = max(1.25 - 2 * a, 0.0)
                n = a * (1 - math.sin(2 * math.pi * a) / (2 * math.pi * a)) + (a - at) * lam  # F-4
                m = 2 / 3 * math.sin(math.pi * a) ** 3 / math.pi + lam * g * (math.sin(math.pi * a) + math.sin(math.pi * at)) / math.pi
                return n, m
            al = np.linspace(1e-6, 1, 400)
            f = [NM(a)[1] - er * NM(a)[0] for a in al]
            k = max(k for k in range(len(al) - 1) if f[k] * f[k + 1] <= 0)
            a = brentq(lambda a: NM(a)[1] - er * NM(a)[0], al[k], al[k + 1], xtol=1e-14)
            assert abs(NM(a)[0] - T[i, j]) <= 5.0001e-5, (er, lam)


def test_table_method_against_formula_and_strain():
    """method="table" (appendix F, bilinear between cells) within 2 % of the 5.3.8 formula and 3 % of the strain
    method (the coarse rows ηe0/r 2.0–2.5 cost up to 2 % by interpolation); applicability C30–C50, ρ 0.5 %–4 %, ≥ 8 bars, solid circle"""
    for D, n, d, cov, cc in [(1500, 28, 28, 75, "C35"), (1200, 24, 25, 60, "C30"), (1800, 36, 32, 90, "C40")]:
        s = jtg.circle(D, n, d, cov, concrete=cc)
        for N, M in [(20000, 500), (12000, 3000), (6000, 4000), (1500, 2500), (800, 2000)]:
            t = jtg.eccentric(s, N, M, method="table")
            assert t.resistance == pytest.approx(jtg.eccentric(s, N, M, method="code").resistance, rel=0.02)
            assert t.resistance == pytest.approx(jtg.eccentric(s, N, M, method="strain").resistance, rel=0.03)
    with pytest.raises(ValueError):
        jtg.eccentric(jtg.circle(1200, 24, 25, 60, concrete="C25"), 3000, 1000, method="table")
    with pytest.raises(ValueError):
        jtg.eccentric(jtg.circle(1500, 12, 20, 60, concrete="C30"), 3000, 1000, method="table")  # ρ < 0.5 %
    with pytest.raises(ValueError):
        jtg.circle_nu(12.0, 0.3)
    assert jtg.circle_nu(0.5, 0.3) == 0.6786 and jtg.circle_nu(0.525, 0.3) == pytest.approx((0.6786 + 0.6347) / 2)


def test_circle_formula_small_eccentricity():
    """5.3.8 at small e: the equation M = e N also has a root with N < 0 (used to fail); the root with N > 0 tends to
    the full section fcd A + fsd As as e → 0 (α → 1) and stays within 1.5 % of the strain method"""
    s = jtg.circle(1500, 28, 28, 75, concrete="C35")
    r = jtg.eccentric(s, 20000, 500)  # e0 raised to D/30 = 50 mm
    assert r.resistance == pytest.approx(jtg.eccentric(s, 20000, 500, method="strain").resistance, rel=0.015)
    Nu, a = jtg.rc._circle_code(s, 1e-3)
    assert a > 0.99 and Nu == pytest.approx(16.1 * s.area + 330 * 28 * A(28), rel=1e-3)


# ── 5.3.11 biaxial compression ──────────────────────────────────────────────
def _fiber_biaxial(b, h, bars, c, st, ev, eu, n=60):
    """biaxial strain compatibility (5.1.3–5.1.5 with the neutral axis at any angle; a different method from 5.3.11):
    rectangular block fcd over β xc normal to the axis from the most compressed corner, bars by 5.1.5-1 within ±fsd.
    u across b, v up h (centred); the load at (eu, ev). Returns Nu (N)."""
    du, dv = b / n, h / n
    U, V = np.meshgrid((np.arange(n) + 0.5) * du - b / 2, (np.arange(n) + 0.5) * dv - h / 2)
    U, V, dA = U.ravel(), V.ravel(), du * dv
    bu, bv = np.array([p[0] for p in bars]) - b / 2, h / 2 - np.array([p[1] for p in bars])
    ba = np.array([A(p[2]) for p in bars])

    def NM(th, xc):
        nu_, nv_ = math.sin(th), math.cos(th)
        dmax = b / 2 * nu_ + h / 2 * nv_
        comp = dmax - (U * nu_ + V * nv_) <= c.beta * xc
        sig = np.clip(c.eps_cu * st.Es * (1 - (dmax - (bu * nu_ + bv * nv_)) / xc), -st.fsd, st.fsd_c)
        N = c.fcd * dA * comp.sum() + (sig * ba).sum()
        return N, c.fcd * dA * U[comp].sum() + (sig * ba * bu).sum(), c.fcd * dA * V[comp].sum() + (sig * ba * bv).sum()

    def at(th):
        f = lambda xc: NM(th, xc)[2] - ev * NM(th, xc)[0]  # noqa: E731
        xs = np.geomspace(1e-2 * h, 20 * h, 80)
        fs = [f(x) for x in xs]
        k = next(k for k in range(len(xs) - 1) if fs[k] * fs[k + 1] <= 0 and NM(th, xs[k + 1])[0] > 0)
        N, Mu, _ = NM(th, brentq(f, xs[k], xs[k + 1], xtol=1e-4))
        return Mu - eu * N, N

    ths = np.linspace(0.02, math.pi / 2 - 0.02, 16)
    r = [at(t)[0] for t in ths]
    k = next(k for k in range(len(ths) - 1) if r[k] * r[k + 1] <= 0)
    return at(brentq(lambda t: at(t)[0], ths[k], ths[k + 1], xtol=1e-5))[1]


def test_biaxial_against_strain_compatibility():
    """5.3.11 (reciprocal load formula) against a biaxial strain-compatibility solution of the same rectangle, at the
    eccentricities ηe0 used by the check: within 6 % (Bresler-type formulas are known to be close for N ≥ 0.1 Nu0)"""
    b, h, (nb, nh, d, cov) = 800, 1200, (5, 7, 28, 60)
    sx, sy = jtg.rect_biaxial(b, h, layout=(nb, nh, d, cov), concrete="C40")
    us, vs = np.linspace(cov, b - cov, nb), np.linspace(cov, h - cov, nh)
    pts = sorted({(round(u, 6), round(v, 6)) for u in us for v in (cov, h - cov)}
                 | {(round(u, 6), round(v, 6)) for v in vs for u in (cov, b - cov)})
    bars = [(u, v, d) for u, v in pts]
    assert len(bars) == 2 * nb + 2 * nh - 4 and sx.bars[:, 0].sum() == pytest.approx(len(bars) * A(28))
    for N, Mx, My in [(8000, 1500, 900), (4000, 1500, 900), (2000, 1200, 700), (10000, 600, 600)]:
        r = jtg.biaxial(sx, sy, N, Mx, My)
        Nf = _fiber_biaxial(b, h, bars, sx.concrete, sx.steel, r.details["x"]["e"], r.details["y"]["e"]) / 1e3
        assert r.resistance == pytest.approx(Nf, rel=0.06)


def test_biaxial_formula_symmetry_and_limits():
    """1/Nu = 1/Nux + 1/Nuy − 1/Nu0, Nu0 = 0.9 (fcd A + f'sd A's); a square with symmetric bars gives the same for
    swapped moments; with one moment → 0 the result approaches the uniaxial Nux (within 3 %); circles use the
    resultant moment"""
    sx, sy = jtg.rect_biaxial(1000, 1000, layout=(6, 6, 25, 60), concrete="C35")
    r = jtg.biaxial(sx, sy, 6000, 1200, 500, l0x=10000, l0y=10000)
    Nu0 = 0.9 * (16.1 * 1e6 + 330 * 20 * A(25)) / 1e3
    assert r.details["Nu0"] == pytest.approx(Nu0)
    assert r.resistance == pytest.approx(1 / (1 / r.details["Nux"] + 1 / r.details["Nuy"] - 1 / Nu0))
    assert jtg.biaxial(sx, sy, 6000, 500, 1200, l0x=10000, l0y=10000).resistance == pytest.approx(r.resistance)
    u = jtg.biaxial(sx, sy, 6000, 1200, 0.0)
    assert u.resistance == pytest.approx(jtg.eccentric(sx, 6000, 1200).resistance, rel=0.03) and u.notes
    col = jtg.circle(1500, 28, 28, 75, concrete="C35")
    c = jtg.biaxial(col, None, 9000, 1800, 2400, l0x=12000, l0y=9000)
    assert c.resistance == pytest.approx(jtg.eccentric(col, 9000, 3000, 12000).resistance)
    with pytest.raises(ValueError):
        jtg.biaxial(sx, jtg.rect_biaxial(900, 1000, layout=(6, 6, 25, 60))[1], 6000, 1200, 500)
    ux, uy = jtg.rect_biaxial(1000, 1000, bars=[(60, 60, 32), (940, 60, 32), (60, 940, 25), (940, 940, 25)])
    with pytest.raises(ValueError):  # not symmetric about the horizontal axis
        jtg.biaxial(ux, uy, 6000, 1200, 500)


# ── 5.4 tension members ─────────────────────────────────────────────────────
def test_axial_tension_5_4_1():
    """Nud = fsd As + fpd Ap, all the steel (5.4.1); one side ≥ max(45 ftd/fsd, 0.2) % of the gross area (9.1.12)"""
    s = jtg.rect(400, 800, concrete="C30", top=[(4, 22, 50)], bottom=[(4, 22, 50)], bars=[(2 * A(16), 400)])
    r = jtg.tension(s, 500)
    assert r.resistance == pytest.approx(330 * (8 * A(22) + 2 * A(16)) / 1e3) and r.ok
    thin = jtg.rect(400, 800, concrete="C30", top=[(2, 16, 50)], bottom=[(4, 22, 50)])
    assert any("far face" in f for f in jtg.tension(thin, 100).failed)  # 2Φ16 = 0.126 % < 0.2 %
    pc = jtg.rect(400, 800, concrete="C50", bottom=[(4, 16, 50)], top=[(4, 16, 50)],
                  tendons=[(1112, 400, 1000, 70)], tendon="strand1860")
    assert jtg.tension(pc, 500).resistance == pytest.approx((330 * 8 * A(16) + 1260 * 1112) / 1e3)


def test_small_eccentric_tension_5_4_2():
    """5.4.2-1, -2 by hand; e0 → 0 tends to 5.4.1; for a symmetric rectangle the result is exactly the linear
    interaction 1/(1/Nud + e0/Mu) with Mu = fsd As (h0 − a') (the derivation behind 5.4.3, 5.4.4 in the commentary)"""
    s = jtg.rect(400, 800, concrete="C30", top=[(4, 22, 50)], bottom=[(4, 22, 50)])
    As, L = 4 * A(22), 700.0
    for M in (10, 50, 100, 140):
        e0 = M / 500 * 1e3
        r = jtg.tension(s, 500, M)
        e, e_ = 350 - e0, 350 + e0
        assert r.details["case"] == "small eccentricity"
        assert r.resistance == pytest.approx(min(330 * As * L / e, 330 * As * L / e_) / 1e3)
        assert r.resistance == pytest.approx(1 / (1 / (2 * 330 * As) + e0 / (330 * As * L)) / 1e3)
    assert jtg.tension(s, 500, 1e-9).resistance == pytest.approx(jtg.tension(s, 500).resistance, rel=1e-9)


def test_large_eccentric_tension_5_4_2():
    """5.4.2-3, -4 by hand (x ≥ 2a'), and e0 → ∞: Nu e0 → Mu of jtg.flexure (pure bending)"""
    s = jtg.rect(400, 800, concrete="C30", top=[(2, 16, 45)], bottom=[(5, 25, 60)])
    As, Asc, h0, a_ = 5 * A(25), 2 * A(16), 740.0, 45.0
    N, M = 400.0, 2000.0
    e = M / N * 1e3 + 400 - h0  # load to As
    # (fsd As − f'sd A's − fcd b x) e = fcd b x (h0 − x/2) + f'sd A's (h0 − a')
    q = [13.8 * 400 / 2, -13.8 * 400 * (h0 + e), (330 * As - 330 * Asc) * e - 330 * Asc * (h0 - a_)]
    x = min(v.real for v in np.roots(q) if v.real > 0)
    r = jtg.tension(s, N, M)
    assert x >= 2 * a_ and r.details["x"] == pytest.approx(x, rel=1e-6)
    assert r.resistance == pytest.approx((330 * As - 330 * Asc - 13.8 * 400 * x) / 1e3, rel=1e-6)
    sym = jtg.rect(400, 800, concrete="C30", top=[(4, 22, 50)], bottom=[(4, 22, 50)])
    Mu = jtg.flexure(sym, 1.0).resistance
    for e0 in (1e5, 1e6):
        assert jtg.tension(sym, 1.0, e0 / 1e3).resistance * e0 / 1e3 == pytest.approx(Mu, rel=0.01 if e0 < 1e6 else 1e-3)


def test_large_eccentric_tension_below_2a():
    """x < 2a': 5.2.4 with Md → Nd e' (moments about A's), or the section without its compression bars, the larger
    (as jtg.flexure); x > ξb h0 fails (5.2.2-3)"""
    s = jtg.rect(400, 800, concrete="C30", top=[(4, 22, 50)], bottom=[(4, 22, 50)])
    r = jtg.tension(s, 500, 400)
    e_ = 800 + 400 - 50
    about = 330 * 4 * A(22) * 700 / e_
    assert r.details["N_about_compression_steel"] == pytest.approx(about / 1e3)
    assert r.resistance == pytest.approx(max(about / 1e3, r.details["N_singly_reinforced"]))
    over = jtg.rect(200, 500, concrete="C25", steel="HRB400", bottom=[(6, 32, 60)])
    assert any("ξb" in f for f in jtg.tension(over, 10, 400).failed)


def test_eccentric_tension_minimum_reinforcement():
    """9.1.12: large eccentricity As/(b h0); compression bars needed by the calculation ≥ 0.2 % of the gross area"""
    s = jtg.rect(400, 800, concrete="C30", top=[(2, 16, 45)], bottom=[(5, 25, 60)])
    r = jtg.tension(s, 400, 2000)
    assert r.details["rho"] == pytest.approx(5 * A(25) / (400 * 740))
    assert any("compression bars needed" in f for f in r.failed) and r.details["N_without_compression_bars"] < 400
    assert not any("compression bars" in f for f in jtg.tension(s, 100, 200).failed)


def test_circle_tension_5_4_4():
    """5.4.4: e0 → 0 gives Nud, e0 → ∞ gives Mud (5.3.8 with N = 0, within 2 % of the strain method's pure bending);
    the linear interaction is on the safe side of the strain-compatibility interaction (commentary: 略偏于安全)"""
    c = jtg.circle(1200, 24, 25, 60, concrete="C30")
    Nud = 330 * 24 * A(25)
    assert jtg.tension(c, 1000).resistance == pytest.approx(Nud / 1e3)
    Mud = jtg.tension(c, 1000, 100).details["Mud"]
    assert Mud == pytest.approx(jtg.flexure(c, 1.0).resistance, rel=0.02)
    assert jtg.tension(c, 1.0, 1e3).resistance * 1e3 == pytest.approx(Mud, rel=2e-3)
    xb = c.xi_b()
    for e0 in (100.0, 300.0, 600.0, 1500.0):  # mm; with Nd = 1000 kN, Md = e0 kN·m
        r = jtg.tension(c, 1000, e0)
        f = lambda x: c.forces(x, xb)[1] + e0 * c.forces(x, xb)[0]  # noqa: E731
        xs = np.geomspace(1e-3, 1200, 400)
        k = next(k for k in range(len(xs) - 1) if f(xs[k]) * f(xs[k + 1]) <= 0)
        N_strain = -c.forces(brentq(f, xs[k], xs[k + 1]), xb)[0] / 1e3
        assert r.resistance <= N_strain * 1.005 and r.resistance == pytest.approx(1 / (1e3 / Nud + e0 / 1e3 / Mud))


def test_biaxial_tension_5_4_3():
    """5.4.3 by hand; with one moment zero it is 5.4.4's linear interaction; circles use the resultant e0; only for
    symmetrically reinforced rectangles"""
    sx, sy = jtg.rect_biaxial(600, 900, layout=(4, 5, 25, 60), concrete="C30")
    Nud = 330 * 14 * A(25)
    Mux, Muy = jtg.flexure(sx, 1.0).resistance, jtg.flexure(sy, 1.0).resistance
    r = jtg.biaxial_tension(sx, sy, 800, 120, 90)
    assert r.resistance == pytest.approx(1 / (1 / Nud + math.hypot(150 / Mux / 1e6, 112.5 / Muy / 1e6)) / 1e3)
    assert jtg.biaxial_tension(sx, sy, 800, 120, 0).resistance == pytest.approx(1 / (1 / Nud + 150 / Mux / 1e6) / 1e3)
    c = jtg.circle(1200, 24, 25, 60, concrete="C30")
    assert jtg.biaxial_tension(c, None, 1000, 300, 400).resistance == pytest.approx(jtg.tension(c, 1000, 500).resistance)
    unsym = jtg.rect(600, 900, top=[(3, 20, 60)], bottom=[(5, 25, 60)])
    with pytest.raises(ValueError):
        jtg.biaxial_tension(unsym, sy, 800, 120, 90)


# ── shear of circular sections ──────────────────────────────────────────────
def test_circle_shear_equivalent_rectangle():
    """GB 50010-2010 6.3.15 (b = 1.76 r, h0 = 1.6 r; worked example of 《混凝土结构设计计算算例》 ch. 8.3: r = 225 mm →
    b = 396, h0 = 360): the circle gives the same as jtg.shear on that equivalent rectangle (another code path) with
    the bars of the tension half as As, and as 5.2.9 by hand"""
    small = jtg.circle(450, 8, 20, 50, concrete="C30")
    d = jtg.shear(small, 100, (2, 10, 100)).details
    assert (d["b"], d["h0"]) == pytest.approx((396.0, 360.0))
    c = jtg.circle(1500, 28, 28, 75, concrete="C35")
    r = jtg.shear(c, 1500, (2, 12, 100))
    As = c.bars[c.bars[:, 1] > 750, 0].sum()
    eq = jtg.rect(1320, 1500, concrete="C35", bars=[(As, 1200.0)])
    v = jtg.shear(eq, 1500, (2, 12, 100))
    assert r.resistance == pytest.approx(v.resistance, rel=1e-9)
    assert r.details["V_section_max"] == pytest.approx(v.details["V_section_max"])
    P = 100 * As / (1320 * 1200)
    Vcs = 0.45e-3 * 1320 * 1200 * math.sqrt((2 + 0.6 * P) * math.sqrt(35) * 2 * A(12) / (100 * 1320) * 330)
    assert r.resistance == pytest.approx(Vcs) and "GB 50010" in r.clause


def test_circle_shear_limits_and_detailing():
    """5.2.11 upper limit, 9.6.1 hoop diameter and column spacing (not for piles); hollow circles are not covered"""
    c = jtg.circle(1500, 28, 28, 75, concrete="C35")
    assert any("5.2.11" in f for f in jtg.shear_circle(c, 6000, (4, 16, 100)).failed)
    assert any("9.6.1-4" in f for f in jtg.shear_circle(c, 1000, (2, 12, 450)).failed)
    assert not jtg.shear_circle(c, 1000, (2, 12, 450), pile=True).failed
    assert any("8 mm" in f for f in jtg.shear_circle(c, 1000, (2, 6, 100)).failed)
    with pytest.raises(NotImplementedError):
        jtg.shear(jtg.circle(2000, 30, 25, 60, hollow=1400), 1000, (2, 12, 100))
    with pytest.raises(ValueError):
        jtg.shear(c, 1000, None)
