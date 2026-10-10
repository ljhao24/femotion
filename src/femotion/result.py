"""The fem-result/1 file: mesh + fields + per-frame sequences. Solvers write it, display code reads it; neither depends
on the other. Full specification: docs/result-format.md. JavaScript reader: parseFemResult in the @femotion/core package (js/core).

Files: {name}.json (description) + {name}.bin (little-endian binary, arrays back to back, each padded to 4 bytes)
    {
      "format": "fem-result/1",
      "meta": {...anything...},
      "dim": 2,
      "points": "points",                                      // array key, (N, dim)
      "cells": [{"type": "tri3", "conn": "cells/tri3"}],       // one entry per cell block
      "groups": {"left": {"points": "groups/left"}, "deck": {"cells": {"line2": "groups/deck/line2"}}},
      "fields": [{"name": "u", "loc": "node", "key": "fields/u", "unit": "m", "frames": false, "label": "displacement"}],
      "arrays": [{"key": "points", "dtype": "f32", "offset": 0, "shape": [N, 2]}, ...]   // offset in bytes
    }
- loc = "node": one value per node (shape (N,) or (N, c)); loc = "cell": one value per cell, blocks concatenated in the
  order listed in "cells".
- frames = true: the leading dimension is the frame, (F, N, ...); meta may carry fps and t.
- extra arrays (densities of every topology-optimisation iteration, stiffness patterns, …) via extra(key, arr), looked up
  by key on the reading side; polyline sets (stress trajectories, contours) via polylines(name, lines).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mesh.core import Mesh

FORMAT = "fem-result/1"
_DT = {"f32": "<f4", "i32": "<i4", "u8": "u1"}


class Result:
    def __init__(self, mesh: Mesh, meta: dict | None = None):
        self.mesh = mesh
        self.meta = dict(meta or {})
        self.fields: list[dict] = []
        self._arrays: list[tuple[str, np.ndarray, str]] = []

    def field(self, name: str, values, loc: str = "node", unit: str | None = None, frames: bool = False,
              label: str | None = None, dtype: str = "f32", **extra) -> "Result":
        v = np.asarray(values)
        n = self.mesh.n_points if loc == "node" else sum(len(c) for c in self.mesh.cells.values())
        lead = v.shape[1] if frames else v.shape[0]
        if lead != n:
            raise ValueError(f"field {name}: leading dimension {lead} ≠ number of {'nodes' if loc == 'node' else 'cells'} {n}")
        key = f"fields/{name}"
        self.fields.append(dict(name=name, loc=loc, key=key, unit=unit, frames=frames, label=label,
                                range=[float(np.nanmin(v)), float(np.nanmax(v))], **extra))
        self._arrays.append((key, v, dtype))
        return self

    def extra(self, key: str, values, dtype: str = "f32") -> "Result":
        self._arrays.append((key, np.asarray(values), dtype))
        return self

    def polylines(self, name: str, lines: list[np.ndarray]) -> "Result":
        """a set of polylines (stress trajectories, contours, paths): each an (n, c) point list with the same column count
        (e.g. x, y, |σ|). Stored as lines/{name}/pts (all points back to back) and lines/{name}/len (points per line)"""
        lines = [np.asarray(l, float) for l in lines]
        c = lines[0].shape[1] if lines else 2
        self._arrays.append((f"lines/{name}/pts", np.concatenate(lines) if lines else np.zeros((0, c)), "f32"))
        self._arrays.append((f"lines/{name}/len", np.array([len(l) for l in lines], np.int64), "i32"))
        return self

    def write(self, outdir: str | Path, name: str, verbose: bool = True) -> Path:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        m = self.mesh
        arrays = [("points", m.points, "f32")]
        cells = []
        for k, c in m.cells.items():
            arrays.append((f"cells/{k}", c, "i32"))
            cells.append(dict(type=k, conn=f"cells/{k}"))
        groups: dict = {}
        for g, idx in m.point_groups.items():
            arrays.append((f"groups/{g}", idx, "i32"))
            groups.setdefault(g, {})["points"] = f"groups/{g}"
        for g, d in m.cell_groups.items():
            for k, idx in d.items():
                arrays.append((f"groups/{g}/{k}", idx, "i32"))
                groups.setdefault(g, {}).setdefault("cells", {})[k] = f"groups/{g}/{k}"
        arrays += self._arrays
        layout = []
        off = 0
        with open(outdir / f"{name}.bin", "wb") as fh:
            for key, a, dt in arrays:
                b = np.ascontiguousarray(a, dtype=_DT[dt]).tobytes()
                pad = (-len(b)) % 4
                fh.write(b + b"\0" * pad)
                layout.append(dict(key=key, dtype=dt, offset=off, shape=list(np.shape(a))))
                off += len(b) + pad
        desc = dict(format=FORMAT, meta=self.meta, dim=m.dim, points="points", cells=cells, groups=groups,
                    fields=self.fields, arrays=layout)
        write_json(outdir / f"{name}.json", desc)
        if verbose:
            print(f"wrote {outdir / name}.json/.bin: {m.n_points} nodes, {sum(len(c) for c in m.cells.values())} cells, "
                  f"{len(self.fields)} fields, {off / 1e6:.2f} MB")
        return outdir / f"{name}.json"


def read(path: str | Path) -> dict:
    """read {name}.json/.bin back: returns dict(mesh=Mesh, meta, fields={name: array}, arrays={key: array}, desc)"""
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    desc = json.loads(path.read_text())
    raw = path.with_suffix(".bin").read_bytes()
    arrs = {}
    for a in desc["arrays"]:
        dt = np.dtype(_DT[a["dtype"]])
        n = int(np.prod(a["shape"])) if a["shape"] else 1
        arrs[a["key"]] = np.frombuffer(raw, dt, count=n, offset=a["offset"]).reshape(a["shape"])
    pg = {g: arrs[d["points"]] for g, d in desc["groups"].items() if "points" in d}
    cg = {g: {k: arrs[v] for k, v in d["cells"].items()} for g, d in desc["groups"].items() if "cells" in d}
    mesh = Mesh(arrs["points"].astype(float), {c["type"]: arrs[c["conn"]] for c in desc["cells"]}, pg, cg)
    return dict(mesh=mesh, meta=desc["meta"], fields={f["name"]: arrs[f["key"]] for f in desc["fields"]}, arrays=arrs, desc=desc)


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    return path
