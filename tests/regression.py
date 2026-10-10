"""Representative computations whose results are pinned in tests/data/regression_v1.npz, so that an update of NumPy, SciPy,
scikit-fem or OpenSeesPy that changes a number is caught. Only computations on meshes femotion builds itself: a gmsh update
legitimately moves mesh nodes, and those paths are covered by closed-form tests instead (Kirsch's plate, cantilevers).

Regenerate only after reviewing why a reference changed (for example a dependency fixed a bug): python tests/regression.py"""
import importlib.util
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("data") / "regression_v1.npz"

# relative tolerance per result (scaled by the largest magnitude of the reference array), wide enough for the last-digit
# differences between NumPy / SciPy / BLAS versions (checked from the lowest supported versions to the newest);
# "exact" is for integers and labels only
TOL = {"linear": 1e-9, "dynamic": 1e-7, "opensees": 1e-7, "path": 1e-5, "topopt": 1e-6, "exact": 0.0}
KIND = {
    "cs_M": "linear", "cs_U0": "linear", "cs_eps": "linear", "cs_T": "linear", "cs_U": "linear", "cs_f": "linear",
    "cs_N": "linear", "cs_B": "linear", "cs_My": "linear", "cs_react": "linear", "cs_modes": "exact",
    "ops_U": "opensees", "ops_N": "opensees", "ops_f": "opensees", "ops_buckling": "path", "ops_rc": "path",
    "hex_u": "linear", "hex_stress": "linear", "hex_sed": "linear", "weld_p": "linear", "weld_c": "exact",
    "topo_x": "topopt", "topo_comp": "topopt", "topo_energy": "topopt",
    "dyn_newmark": "dynamic", "dyn_release": "dynamic", "dyn_moving": "linear",
}


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def compute() -> dict:
    """every result whose back-end is installed (core: numpy + scipy; scikit-fem and OpenSeesPy parts when present)"""
    from femotion import dynamics, topopt
    from femotion.frame import Model, Section, Solver
    from femotion.frame import post as FP
    from femotion.mesh import structured
    from femotion.models import cable_stayed as cs

    R = {}
    # 500 m cable-stayed bridge: self-weight, cable-force optimisation, modes, forces
    br = cs.build()
    sv = Solver(br.m)
    G = sv.gravity()
    U0 = sv.solve(G)
    eps, T = cs.optimize_pretension(br, sv, U0, verbose=False)
    U = sv.solve(G + sv.pretension(eps))
    f, Phi = sv.modes(10)
    R.update(cs_M=sv.M, cs_U0=U0, cs_eps=eps, cs_T=T, cs_U=U, cs_f=f, cs_N=sv.truss_forces(U, eps), cs_B=sv.beam_end_forces(U),
             cs_My=FP.chain_force(sv.beam_end_forces(U), cs.deck_beams(br), "My"),
             cs_react=FP.reactions(sv.K, U, G + sv.pretension(eps)))
    R["cs_modes"] = np.array([f"{k}/{int(sym)}" for k, sym in (FP.classify_mode(Phi[:, i], br.deck_nodes) for i in range(10))])
    # quadratic hexahedra on a structured box; welded block assembly
    if has("skfem"):
        from femotion.continuum import Elastic
        s = Elastic(structured.box(8.0, 1, 1, 16, 2, 2), E=1000.0, nu=0.3, order=2)
        s.traction(lambda x: np.isclose(x[0], 8), (0, 0, -1.0))
        s.fix(lambda x: np.isclose(x[0], 0))
        s.solve(method="direct")
        R.update(hex_u=s.nodal_displacement(), hex_stress=s.cell_stress(), hex_sed=s.strain_energy_density())
    xs, ys = structured.graded([0, 1, 2], 0.5), structured.graded([0, 1], 0.5)
    w = structured.weld([structured.hex_block(xs, ys, 0.0, 1.0, 2),
                         structured.hex_block(xs[xs >= 1 - 1e-9], ys, 1.0, lambda x, y: 2 + 0.5 * (x - 1), 3)])
    R.update(weld_p=w.points, weld_c=w.cells["hex8"])
    # topology optimisation: 25 iterations
    nelx, nely = 60, 20
    F = np.zeros(2 * (nelx + 1) * (nely + 1))
    F[2 * topopt.node(nelx // 2, 0, nely) + 1] = -1
    a, b = topopt.node(0, nely, nely), topopt.node(nelx, nely, nely)
    r = topopt.run(nelx, nely, F, np.array([2 * a, 2 * a + 1, 2 * b, 2 * b + 1]), volfrac=0.3, maxit=25, verbose=False,
                   record_energy=True)
    R.update(topo_x=r["x"], topo_comp=np.array(r["comp"]), topo_energy=np.array(r["energy"]))
    # modal dynamics
    R["dyn_newmark"] = dynamics.newmark_modal(f[:4], np.ones(4), np.sin(np.linspace(0, 20, 500)), 0.02, 0.03, sub=2)
    R["dyn_release"] = dynamics.step_release(np.zeros(sv.m.ndof), U - U0, sv.pretension(eps), f, Phi, np.linspace(0, 5, 50))[::7]
    R["dyn_moving"] = dynamics.moving_point_loads(sv.m.ndof, br.deck_nodes, np.asarray(br.deck_x), [-100.0, 37.3, 260.0], 300.0,
                                                  ecc=1.5)
    if has("openseespy"):
        from femotion.frame.opensees import Fiber, OpsFrame
        fo = OpsFrame(br.m, eps0=eps)
        R.update(ops_U=fo.static(G), ops_N=fo.truss_forces(), ops_f=OpsFrame(br.m).modes(6)[0])
        # Euler column with an initial bow, followed past buckling (corotational)
        L, n, E, I = 10.0, 20, 2e8, 1e-5
        m = Model()
        ns = [m.node((L * k / n, 0, L / 1000 * np.sin(np.pi * k / n))) for k in range(n + 1)]
        for p, q in zip(ns[:-1], ns[1:]):
            m.beam(p, q, Section(E=E, A=0.01, Iy=I, Iz=I, J=1e-5))
        m.fix(ns[0], (1, 1, 1, 1, 0, 0))
        m.fix(ns[-1], (0, 1, 1, 0, 0, 0))
        Fb = np.zeros(m.ndof)
        Fb[6 * ns[-1]] = -1.0
        lam, _ = OpsFrame(m, transf="Corotational").pushover(Fb, node=ns[n // 2], dof=2, target=L / 10, steps=50)
        R["ops_buckling"] = lam
        # reinforced-concrete pier pushover with P-Δ
        m = Model()
        ns = [m.node((5.0 * k / 4, 0, 0)) for k in range(5)]
        for p, q in zip(ns[:-1], ns[1:]):
            m.beam(p, q, Section(E=2e8, A=0.02, Iy=6.7e-5, Iz=1.7e-5, J=1e-4), tag="pier")
        m.fix(ns[0])
        Fp = np.zeros(m.ndof)
        Fp[6 * ns[-1]] = 1.0
        fb = {"pier": Fiber.rc_circle(D=0.6, cover=0.04, n_bars=12, d_bar=0.02, fc=26.8e3, fy=4e5, Ec=3.25e7)}
        R["ops_rc"] = OpsFrame(m, fiber=fb, transf="PDelta").pushover(Fp, node=ns[-1], dof=0, target=0.2, steps=40)[0]
    return R


def differences(R: dict, ref) -> list[str]:
    """names and sizes of the results that differ from the reference beyond their tolerance"""
    bad = []
    for k in ref.files:
        if k not in R:
            continue
        a, b = ref[k], R[k]
        if a.shape != b.shape:
            bad.append(f"{k}: shape {b.shape} ≠ {a.shape}")
        elif a.dtype.kind in "US" or KIND[k] == "exact":
            if not np.array_equal(a, b):
                bad.append(f"{k}: not identical")
        else:
            err = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-300)
            if err > TOL[KIND[k]]:
                bad.append(f"{k}: relative difference {err:.1e} > {TOL[KIND[k]]:.0e}")
    return bad


if __name__ == "__main__":
    R = compute()
    assert set(R) == set(KIND), set(R) ^ set(KIND)
    np.savez_compressed(DATA, **R)
    print(f"wrote {DATA}: {len(R)} results, {DATA.stat().st_size / 1e6:.2f} MB")
