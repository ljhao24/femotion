"""OpenSees back-end (OpenSeesPy ≥ 3.5; extra "opensees"): the same femotion.frame.Model, moved into OpenSees unchanged,
adds what the linear solver cannot do — yielding, plastic hinges, buckling, large displacements, slack cables, the dynamic
response after a member suddenly breaks, earthquake damage. Units as in femotion.frame: m, kN, t, s; E and stresses in kPa.
Results go through to_result() into a fem-result/1 file, so the display side needs no changes.

    from femotion.frame import Model, Section
    from femotion.frame.opensees import OpsFrame, Fiber
    f = OpsFrame(model, transf="Corotational", fiber={"pier": Fiber.rc_circle(D=2.0, ...)})
    f.static(F_gravity, steps=10)                                        # self-weight (held from now on)
    lam, Us = f.pushover(F_lat, node=top, dof=0, target=0.5, steps=200)   # pushover
    t, Us = f.transient(ag, dt, direction=0, zeta=0.05)                   # ground motion
    t, Us = f.release(trusses=[k], t_end=10, dt=0.01)                     # cable loss: free vibration after removal
    Us = OpsFrame(model).linear_cases([F1, F2, ...])                      # many independent linear load cases

Mapping from femotion.frame.Model
- beam → elasticBeamColumn (or, if its tag is in `fiber`, forceBeamColumn on a fiber section); local axes exactly as
  femotion.frame.model.local_axes (local z = projected reference vector, y = z × x; Iz for x-y bending, Iy for x-z bending).
- bar → Truss; bars whose tag is in cable_tags carry tension only (Elastic with Eneg = 0); initial strain eps0 > 0 is
  pretension (InitStrainMaterial; sign verified by the tests).
- link → zeroLength in global directions (OpenSees warns if the two nodes do not coincide but computes normally).
- mass: the same lumped masses as femotion.frame.linear (femotion.frame.model.lumped_masses), so frequencies and
  self-weight are directly comparable.
- nodes connected only to bars have no rotational stiffness: their rotations are restrained (as in the linear solver).

OpenSees holds one global model: only the most recently created OpsFrame is valid, and calling an older one raises.
Some OpenSees errors (for example a model without a single free dof) terminate the process; build new models with
femotion.frame.linear first. Licence: OpenSees is free for education, research and non-profit use (UC Regents).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .model import Model, Section, local_axes, lumped_masses  # noqa: F401 (re-exported)

LOG = os.path.join(tempfile.gettempdir(), "femotion_opensees.log")
_ACTIVE: list = [None]  # the OpsFrame that currently owns the global OpenSees domain


def _ops():
    import openseespy.opensees as ops  # lazy: the rest of femotion works without OpenSeesPy

    return ops


# ── fiber sections ───────────────────────────────────────────────────────
@dataclass
class Fiber:
    """fiber section (section-local coordinates y, z in m). materials: name → OpenSees uniaxialMaterial type and parameters
    (without the tag). patches: ('rect', mat, ny, nz, (y0, z0), (y1, z1)) or
    ('circ', mat, n_circ, n_rad, (yc, zc), r_in, r_out[, a0, a1]). layers: ('circ', mat, n_bars, bar_area, (yc, zc), radius)
    or ('line', mat, n_bars, bar_area, (y0, z0), (y1, z1)). fibers: single fibers (mat, y, z, area), e.g. from
    femotion.frame.sections.fiber for sections of any shape. GJ: torsional stiffness (kN·m²); fibers carry no torsion."""

    materials: dict[str, tuple]
    patches: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    GJ: float = 1e9
    fibers: list = field(default_factory=list)

    # materials
    @staticmethod
    def concrete(fc: float, Ec: float | None = None, eps0: float = 0.002, epsu: float = 0.0035, ft_ratio: float = 0.1,
                 K: float = 1.0) -> tuple:
        """Concrete04 (Popovics curve, compression negative; the initial modulus Ec is given separately so a confined core is not
        softened). fc: compressive strength (kPa, positive). K > 1 is hoop-confined core concrete (simplified Mander: strength K·fc,
        peak strain eps0·(1 + 5(K − 1)), ultimate strain about 0.004 + 0.04(K − 1)); K = 1 is cover concrete that spalls (stress
        drops to zero) beyond epsu"""
        Ec = Ec or 2 * fc / eps0
        fcc = K * fc
        e0 = eps0 * (1 + 5 * (K - 1))
        eu = epsu if K <= 1 else 0.004 + 0.04 * (K - 1)
        ft = ft_ratio * fc
        return ("Concrete04", -fcc, -e0, -eu, Ec, ft, ft / Ec)

    @staticmethod
    def steel(fy: float, E: float = 2.0e8, b: float = 0.01) -> tuple:
        """Steel02 (Giuffré-Menegotto-Pinto; b is the hardening ratio)"""
        return ("Steel02", fy, E, b, 18.0, 0.925, 0.15)

    @staticmethod
    def steel_pp(fy: float, E: float = 2.0e8) -> tuple:
        """elastic-perfectly-plastic steel (Steel01, no hardening): for plastic-moment demonstrations and checks"""
        return ("Steel01", fy, E, 0.0)

    @staticmethod
    def modulus(mat: tuple) -> float:
        """initial elastic modulus of a material tuple (Steel01 / Steel02, Concrete04, Elastic)"""
        return mat[{"Steel01": 2, "Steel02": 2, "Concrete04": 4, "Elastic": 1}[mat[0]]]

    # common sections
    @classmethod
    def rect(cls, b: float, h: float, mat: tuple, n: tuple[int, int] = (2, 40), GJ: float | None = None) -> "Fiber":
        """solid rectangle, b along local y, h along local z (h is the depth for bending in the x-z plane)"""
        E = cls.modulus(mat)
        J = b * h * min(b, h) ** 2 / 3 * (1 - 0.63 * min(b, h) / max(b, h))
        return cls({"m": mat}, [("rect", "m", n[0], n[1], (-b / 2, -h / 2), (b / 2, h / 2))], [], GJ or E / 2.6 * J)

    @classmethod
    def box(cls, b: float, h: float, tw: float, tf: float, mat: tuple, n: int = 8, GJ: float | None = None) -> "Fiber":
        """steel box / tube (truss members, steel towers): width b (local y), depth h (local z), web thickness tw, flange thickness tf"""
        hb, bb = h / 2, b / 2
        pt = [("rect", "m", n, 2, (-bb, hb - tf), (bb, hb)), ("rect", "m", n, 2, (-bb, -hb), (bb, -hb + tf)),
              ("rect", "m", 2, n, (-bb, -hb + tf), (-bb + tw, hb - tf)), ("rect", "m", 2, n, (bb - tw, -hb + tf), (bb, hb - tf))]
        E = cls.modulus(mat)
        Am = (b - tw) * (h - tf)
        J = 4 * Am**2 / (2 * (b - tw) / tf + 2 * (h - tf) / tw)
        return cls({"m": mat}, pt, [], GJ or E / 2.6 * J)

    @classmethod
    def rc_rect(cls, b: float, h: float, cover: float, nb: tuple[int, int], d_bar: float, fc: float, fy: float,
                n: tuple[int, int] = (12, 24), K: float = 1.3, Ec: float | None = None, GJ: float | None = None) -> "Fiber":
        """rectangular reinforced concrete (piers, beams): nb = (bars per row along b, bars per side along h, corners included);
        cover is measured to the bar centres. Inside the bars is hoop-confined core concrete (confinement factor K; K = 1: no core)"""
        A = np.pi * d_bar**2 / 4
        yb, zb = b / 2 - cover, h / 2 - cover
        lay = [("line", "s", nb[0], A, (-yb, zb), (yb, zb)), ("line", "s", nb[0], A, (-yb, -zb), (yb, -zb))]
        if nb[1] > 2:
            dz = 2 * zb / (nb[1] - 1)
            lay += [("line", "s", nb[1] - 2, A, (-yb, -zb + dz), (-yb, zb - dz)), ("line", "s", nb[1] - 2, A, (yb, -zb + dz), (yb, zb - dz))]
        Ec = Ec or 2 * fc / 0.002
        J = b * h * min(b, h) ** 2 / 3 * (1 - 0.63 * min(b, h) / max(b, h))
        c = cover
        pt = [("rect", "k", n[0], n[1], (-b / 2 + c, -h / 2 + c), (b / 2 - c, h / 2 - c)),
              ("rect", "c", n[0], 2, (-b / 2, h / 2 - c), (b / 2, h / 2)), ("rect", "c", n[0], 2, (-b / 2, -h / 2), (b / 2, -h / 2 + c)),
              ("rect", "c", 2, n[1], (-b / 2, -h / 2 + c), (-b / 2 + c, h / 2 - c)), ("rect", "c", 2, n[1], (b / 2 - c, -h / 2 + c), (b / 2, h / 2 - c))]
        return cls({"c": cls.concrete(fc, Ec), "k": cls.concrete(fc, Ec, K=K), "s": cls.steel(fy)}, pt, lay,
                   GJ or 0.2 * Ec / 2.4 * J)  # cracked torsional stiffness: 20 %

    @classmethod
    def rc_circle(cls, D: float, cover: float, n_bars: int, d_bar: float, fc: float, fy: float,
                  n: tuple[int, int] = (24, 10), K: float = 1.3, Ec: float | None = None, GJ: float | None = None) -> "Fiber":
        """circular reinforced concrete (round piers, piles): inside the bar circle is hoop-confined core concrete (factor K)"""
        A = np.pi * d_bar**2 / 4
        Ec = Ec or 2 * fc / 0.002
        J = np.pi * D**4 / 32
        rc = D / 2 - cover
        pt = [("circ", "k", n[0], n[1], (0.0, 0.0), 0.0, rc), ("circ", "c", n[0], 2, (0.0, 0.0), rc, D / 2)]
        return cls({"c": cls.concrete(fc, Ec), "k": cls.concrete(fc, Ec, K=K), "s": cls.steel(fy)}, pt,
                   [("circ", "s", n_bars, A, (0.0, 0.0), rc)], GJ or 0.2 * Ec / 2.4 * J)


def build_fiber_section(ops, fb: Fiber, tag: int, mat_base: int | None = None) -> int:
    """define fiber section `tag` in the current OpenSees model: its uniaxial materials get the tags mat_base + i
    (default 50000 + 100·tag); returns the tag. Shared with femotion.continuum.opensees (fiber anchor bolts)"""
    base = 50000 + 100 * tag if mat_base is None else mat_base
    mtag = {}
    for name, spec in fb.materials.items():
        mtag[name] = base + len(mtag)
        ops.uniaxialMaterial(spec[0], mtag[name], *map(float, spec[1:]))
    ops.section("Fiber", tag, "-GJ", float(fb.GJ))
    for p in fb.patches:
        if p[0] == "rect":
            _, mat, ny, nz, (y0, z0), (y1, z1) = p
            ops.patch("rect", mtag[mat], int(ny), int(nz), float(y0), float(z0), float(y1), float(z1))
        else:
            _, mat, nc, nr, (yc, zc), ri, ro, *ang = p
            a0, a1 = ang if ang else (0.0, 360.0)
            ops.patch("circ", mtag[mat], int(nc), int(nr), float(yc), float(zc), float(ri), float(ro), float(a0), float(a1))
    for ly in fb.layers:
        if ly[0] == "circ":
            _, mat, nbar, A, (yc, zc), r = ly
            ops.layer("circ", mtag[mat], int(nbar), float(A), float(yc), float(zc), float(r))
        else:
            _, mat, nbar, A, (y0, z0), (y1, z1) = ly
            ops.layer("straight", mtag[mat], int(nbar), float(A), float(y0), float(z0), float(y1), float(z1))
    for mat, y, z, A in fb.fibers:
        ops.fiber(float(y), float(z), float(A), mtag[mat])
    return tag


# ── model ────────────────────────────────────────────────────────────────
class OpsFrame:
    """femotion.frame.Model → OpenSees model. Analyses accumulate state in call order:
    static (loads held) → pushover / transient / release.

    transf: 'Linear' (small displacements; identical to femotion.frame.linear), 'PDelta', 'Corotational' (large
    displacements, buckling). fiber: {beam tag: Fiber}; those beams become nonlinear forceBeamColumn elements (n_ip
    Gauss-Lobatto points). cable_tags: tags of tension-only bars (cables, hangers); other bars carry tension and compression.
    eps0: initial strain of every bar (pretension > 0), e.g. from a cable-force optimisation.
    drop: bars not built (cables already broken). masses: lumped mass per dof (default: the shared mass rule)."""

    def __init__(self, m: Model, transf: str = "Linear", fiber: dict[str, Fiber] | None = None, n_ip: int = 5,
                 cable_tags: tuple[str, ...] = (), eps0: np.ndarray | None = None, drop: set[int] | None = None,
                 masses: np.ndarray | None = None, log: str | None = LOG):
        ops = _ops()
        self.ops, self.m = ops, m
        self.nb, self.nt = len(m.beams), len(m.trusses)
        self.drop = set(drop or ())
        self.removed: set[int] = set()
        self._pat = 0
        self._ts = 0
        ops.wipe()
        _ACTIVE[0] = self
        if log:
            ops.logFile(log, "-noEcho")
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        for n, p in enumerate(m.X):
            ops.node(n + 1, *map(float, p))
        # supports; nodes connected only to bars get their rotations restrained
        rot = np.zeros(len(m.X), bool)
        for (i, j, *_r) in m.beams:
            rot[i] = rot[j] = True
        for (i, j, k, kr, tag) in m.links:
            if kr is not None:
                for a in range(3):
                    if kr[a]:
                        rot[i] = rot[j] = True
        for n in range(len(m.X)):
            f = list(m.fixed.get(n, [False] * 6))
            if not rot[n]:
                f[3:] = [True] * 3
            if any(f):
                ops.fix(n + 1, *[int(v) for v in f])
        # masses
        self.M = lumped_masses(m, self.drop) if masses is None else np.asarray(masses, float)
        for n in range(len(m.X)):
            ops.mass(n + 1, *map(float, self.M[6 * n:6 * n + 6]))
        # beams
        fiber = fiber or {}
        sec_tag: dict[str, int] = {}
        for t, fb in fiber.items():
            sec_tag[t] = self._section(fb, len(sec_tag) + 1)
        transf_tag: dict[tuple, int] = {}
        integ: set[int] = set()
        for e, (i, j, sec, ref, tag) in enumerate(m.beams):
            L, R = local_axes(m.X[i], m.X[j], ref)
            key = tuple(np.round(R[2], 9))
            if key not in transf_tag:
                transf_tag[key] = len(transf_tag) + 1
                ops.geomTransf(transf, transf_tag[key], *map(float, R[2]))
            tt = transf_tag[key]
            if tag in sec_tag:
                it = 1000 + sec_tag[tag]
                if it not in integ:
                    ops.beamIntegration("Lobatto", it, sec_tag[tag], n_ip)
                    integ.add(it)
                ops.element("forceBeamColumn", self.beam_tag(e), i + 1, j + 1, tt, it)
            else:
                ops.element("elasticBeamColumn", self.beam_tag(e), i + 1, j + 1, sec.A, sec.E, sec.g(), sec.J, sec.Iy, sec.Iz, tt)
        # bars: one material each (initial strains differ)
        mt = 100000
        for e, (i, j, sec, tag) in enumerate(m.trusses):
            if e in self.drop:
                continue
            mt += 2
            if tag in cable_tags:
                ops.uniaxialMaterial("Elastic", mt, sec.E, 0.0, 0.0)
            else:
                ops.uniaxialMaterial("Elastic", mt, sec.E)
            mat = mt
            if eps0 is not None and eps0[e] != 0:
                ops.uniaxialMaterial("InitStrainMaterial", mt + 1, mt, float(eps0[e]))
                mat = mt + 1
            ops.element("Truss", self.truss_tag(e), i + 1, j + 1, sec.A, mat)
        mt += 2
        # spring links
        for k, (i, j, kk, kr, tag) in enumerate(m.links):
            mats, dirs = [], []
            for a in range(3):
                if kk[a]:
                    mt += 1
                    ops.uniaxialMaterial("Elastic", mt, float(kk[a]))
                    mats.append(mt); dirs.append(a + 1)
                if kr is not None and kr[a]:
                    mt += 1
                    ops.uniaxialMaterial("Elastic", mt, float(kr[a]))
                    mats.append(mt); dirs.append(a + 4)
            if mats:
                ops.element("zeroLength", self.link_tag(k), i + 1, j + 1, "-mat", *mats, "-dir", *dirs, "-orient", 1, 0, 0, 0, 1, 0)

    # element tags
    def beam_tag(self, e: int) -> int:
        return e + 1

    def truss_tag(self, e: int) -> int:
        return self.nb + e + 1

    def link_tag(self, k: int) -> int:
        return self.nb + self.nt + k + 1

    def _section(self, fb: Fiber, tag: int) -> int:
        return build_fiber_section(self.ops, fb, tag)

    def _live(self):
        if _ACTIVE[0] is not self:
            raise RuntimeError("this OpsFrame is stale: OpenSees holds a single global model and a newer OpsFrame replaced it")

    # ── reading the state ────────────────────────────────────────────────
    def U(self) -> np.ndarray:
        """current displacements of all nodes (ndof, same order as femotion.frame.linear)"""
        self._live()
        ops = self.ops
        return np.array([ops.nodeDisp(n + 1) for n in range(len(self.m.X))]).ravel()

    def truss_forces(self) -> np.ndarray:
        """axial force of every bar (tension positive; 0 for bars not built or removed)"""
        self._live()
        N = np.zeros(self.nt)
        for e in range(self.nt):
            if e not in self.drop and e not in self.removed:
                N[e] = self.ops.eleResponse(self.truss_tag(e), "axialForce")[0]
        return N

    def beam_end_forces(self) -> np.ndarray:
        """local end forces of every beam (12 components; same order and signs as femotion.frame.linear.Solver.beam_end_forces)"""
        self._live()
        out = np.zeros((self.nb, 12))
        for e in range(self.nb):
            if e not in self.removed:
                out[e] = self.ops.eleResponse(self.beam_tag(e), "localForce")
        return out

    # ── analyses ─────────────────────────────────────────────────────────
    def _pattern(self, F: np.ndarray | None, series: str = "Linear") -> int:
        ops = self.ops
        self._ts += 1
        self._pat += 1
        ops.timeSeries(series, self._ts)
        ops.pattern("Plain", self._pat, self._ts)
        if F is not None:
            Fn = np.asarray(F, float).reshape(-1, 6)
            for n in np.where(np.abs(Fn).max(axis=1) > 0)[0]:
                ops.load(int(n) + 1, *map(float, Fn[n]))
        return self._pat

    def _setup(self, tol: float = 1e-8, iters: int = 50):
        ops = self.ops
        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", tol, iters, 0)
        ops.algorithm("Newton")

    def _step(self, depth: int = 0, analyze=None, retry=None) -> bool:
        """take one step; if it does not converge try other algorithms, then halve the step (down to 1/2⁶)"""
        ops = self.ops
        if analyze() == 0:
            return True
        for alg in (("ModifiedNewton", "-initial"), ("KrylovNewton",), ("NewtonLineSearch",)):
            ops.algorithm(*alg)
            if analyze() == 0:
                ops.algorithm("Newton")
                return True
        ops.algorithm("Newton")
        if depth >= 6 or retry is None:
            return False
        retry(0.5)
        ok = self._step(depth + 1, analyze, retry) and self._step(depth + 1, analyze, retry)
        retry(2.0)
        return ok

    def static(self, F: np.ndarray, steps: int = 1, history: bool = False, hold: bool = True):
        """apply load F (ndof vector) in `steps` equal increments; hold=True keeps it applied afterwards (self-weight, the
        completed-bridge state). history=True returns (steps+1, ndof) displacements per step (row 0 before loading),
        otherwise the final displacements."""
        self._live()
        ops = self.ops
        self._pattern(F)
        self._setup()
        d = [1.0 / steps]
        ops.integrator("LoadControl", d[0])
        ops.analysis("Static")

        def retry(s):
            d[0] *= s
            ops.integrator("LoadControl", d[0])

        H = [self.U()]
        for s in range(steps):
            if not self._step(analyze=lambda: ops.analyze(1), retry=retry):
                raise RuntimeError(f"static step {s + 1}/{steps} did not converge (log: {LOG})")
            if history:
                H.append(self.U())
        if hold:
            ops.loadConst("-time", 0.0)
        return np.array(H) if history else self.U()

    def linear_cases(self, Fs) -> np.ndarray:
        """a linear model under many independent load cases (vehicles frame by frame, influence lines, influence matrices):
        the stiffness is factorised once and every case is solved separately. Returns (n_cases, ndof). Only on a freshly
        built OpsFrame (loads held by an earlier static() would leak into every case)."""
        self._live()
        ops = self.ops
        if self._pat:
            raise RuntimeError("linear_cases needs a freshly built OpsFrame (loads have already been applied)")
        self._setup()
        ops.algorithm("Linear", "-factorOnce")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        out = []
        for F in Fs:
            p = self._pattern(F, series="Constant")
            if ops.analyze(1) != 0:
                raise RuntimeError(f"linear case {len(out)} failed (log: {LOG})")
            out.append(self.U())
            ops.remove("loadPattern", p)
            ops.reset()
        self._pat = 0
        return np.array(out)

    def pushover(self, F_ref: np.ndarray, node: int, dof: int, target: float, steps: int = 100):
        """displacement-controlled pushover: load pattern F_ref × λ, driving dof (0–5) of `node` to `target`.
        Returns (λ per step, displacements (steps+1, ndof)). Follows the path past the peak (softening, post-buckling)."""
        self._live()
        ops = self.ops
        self._pattern(F_ref)
        self._setup()
        d = [target / steps]
        ops.integrator("DisplacementControl", node + 1, dof + 1, d[0])
        ops.analysis("Static")

        def retry(s):
            d[0] *= s
            ops.integrator("DisplacementControl", node + 1, dof + 1, d[0])

        lam, H = [ops.getLoadFactor(self._pat)], [self.U()]
        for s in range(steps):
            if not self._step(analyze=lambda: ops.analyze(1), retry=retry):
                print(f"pushover stopped at step {s + 1}/{steps} (no convergence; the structure may have collapsed)")
                break
            lam.append(ops.getLoadFactor(self._pat))
            H.append(self.U())
        return np.array(lam), np.array(H)

    def modes(self, n: int = 12):
        """first n frequencies (Hz) and mode shapes (ndof × n, Φᵀ M Φ = I) in the current state (applied loads, removed members)"""
        self._live()
        ops = self.ops
        lam = np.array(ops.eigen("-genBandArpack", n))
        Phi = np.zeros((self.m.ndof, n))
        for k in range(n):
            Phi[:, k] = np.array([ops.nodeEigenvector(i + 1, k + 1) for i in range(len(self.m.X))]).ravel()
            Phi[:, k] /= np.sqrt(Phi[:, k] @ (self.M * Phi[:, k]))
        return np.sqrt(np.maximum(lam, 0)) / (2 * np.pi), Phi

    def _damping(self, zeta: float, freqs: tuple[float, float] | None):
        if zeta <= 0:
            return
        if freqs is None:
            f, _ = self.modes(3)
            freqs = (f[0], f[2])
        w1, w2 = 2 * np.pi * freqs[0], 2 * np.pi * freqs[1]
        a = 2 * zeta * w1 * w2 / (w1 + w2)
        b = 2 * zeta / (w1 + w2)
        self.ops.rayleigh(a, 0.0, 0.0, b)  # mass + committed stiffness (degrades with the stiffness when nonlinear)

    def _run_transient(self, t_end: float, dt: float, sub: int):
        ops = self.ops
        self._setup(tol=1e-7)
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")
        h = [dt / sub]

        def retry(s):
            h[0] *= s

        T, H = [0.0], [self.U()]
        t = 0.0
        while t < t_end - 1e-9:
            for _ in range(sub):
                if not self._step(analyze=lambda: ops.analyze(1, h[0]), retry=retry):
                    print(f"time history stopped at t = {t:.3f} s (no convergence; the structure may have collapsed)")
                    return np.array(T), np.array(H)
            t += dt
            T.append(t)
            H.append(self.U())
        return np.array(T), np.array(H)

    def transient(self, ag: np.ndarray, dt: float, direction: int = 0, zeta: float = 0.02,
                  freqs: tuple[float, float] | None = None, sub: int = 2, t_end: float | None = None):
        """ground acceleration ag (m/s², step dt) in direction 0/1/2 = x/y/z. Rayleigh damping fitted at the two frequencies
        `freqs` (default: modes 1 and 3). Returns (times, displacements relative to the ground (nt, ndof))"""
        self._live()
        ops = self.ops
        self._damping(zeta, freqs)
        self._ts += 1
        self._pat += 1
        ops.timeSeries("Path", self._ts, "-dt", float(dt), "-values", *map(float, ag))
        ops.pattern("UniformExcitation", self._pat, direction + 1, "-accel", self._ts)
        return self._run_transient(t_end or dt * (len(ag) - 1), dt, sub)

    def release(self, trusses=(), beams=(), t_end: float = 10.0, dt: float = 0.02, zeta: float = 0.02,
                freqs: tuple[float, float] | None = None, sub: int = 2):
        """suddenly remove a set of members at t = 0 (cable loss, a broken bar) with the applied loads held, and compute the
        dynamic response that follows (nonlinear, up to collapse if it happens). Returns (times, displacements (nt, ndof));
        row 0 is the state before removal"""
        self._live()
        ops = self.ops
        self._damping(zeta, freqs)
        for e in trusses:
            ops.remove("element", self.truss_tag(e))
            self.removed.add(e)
        for e in beams:
            ops.remove("element", self.beam_tag(e))
            self.removed.add(e)
        return self._run_transient(t_end, dt, sub)


def to_result(m: Model, Us: np.ndarray, meta: dict | None = None, unit_scale: float = 1.0, unit: str = "m"):
    """displacements (ndof, or per frame (nf, ndof)) → a fem-result/1 Result: mesh = Model.to_mesh() (beams and bars as
    line2), field u = nodal translations (frames=True when per frame). Add rotations or forces with Result.field if needed."""
    from ..result import Result

    r = Result(m.to_mesh(), meta)
    U = np.asarray(Us) * unit_scale
    if U.ndim == 1:
        r.field("u", U.reshape(-1, 6)[:, :3], unit=unit, label="displacement")
    else:
        r.field("u", U.reshape(len(U), -1, 6)[:, :, :3], unit=unit, label="displacement", frames=True)
    return r


__all__ = ["OpsFrame", "Fiber", "to_result", "lumped_masses", "build_fiber_section"]
