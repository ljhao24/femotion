"""The fixed fem-result/1 sample (tests/data/golden_v1.*): small, deterministic, covering every feature of the format.
Regenerate only when the format version changes: python tests/golden.py"""
from pathlib import Path

import numpy as np

from femotion.mesh import Mesh
from femotion.result import Result

DATA = Path(__file__).with_name("data")


def golden() -> Result:
    p = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    m = Mesh(p, {"tri3": [[0, 1, 4], [0, 4, 3]], "quad4": [[1, 2, 5, 4]]},
             {"left": np.array([0, 3])}, {"right": {"quad4": np.array([0])}})
    u = np.c_[p[:, 0] * 0.01, -p[:, 1] * 0.02]
    return (Result(m, meta={"title": "golden sample", "fps": 30})
            .field("u", u, unit="m", label="displacement")
            .field("vm", np.linspace(0, 5, 6), unit="MPa")
            .field("rank", np.array([0, 1, 2]), loc="cell", dtype="i32")
            .field("u_t", np.stack([u * k for k in (0.0, 0.5, 1.0)]), frames=True, unit="m")
            .polylines("flow", [np.array([[0, 0.5, 1.0], [1, 0.5, 2.0]]), np.array([[1, 0, 0.5], [1.5, 0.5, 1.5], [2, 1, 2.5]])])
            .extra("pattern", np.array([[0, -1], [-1, 1]]), "i32")
            .extra("mask", np.array([1, 0, 1, 1, 0], np.uint8), "u8"))


if __name__ == "__main__":
    golden().write(DATA, "golden_v1")
