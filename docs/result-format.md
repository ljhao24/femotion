# The `fem-result/1` file format

`fem-result/1` is the only contract between computing and drawing. Solvers (Python, `femotion.result.Result`) write
it; display code (JavaScript, `@femotion/core`'s `parseFemResult`) reads it. It stores one mesh, any number of fields on the nodes or
cells of that mesh (optionally one value set per animation frame), and free-form extra arrays.

A result named `NAME` is two files side by side:

| File | Content |
|---|---|
| `NAME.json` | UTF-8 JSON description (below) |
| `NAME.bin` | all arrays, little-endian, back to back; each array padded with zero bytes to a multiple of 4 bytes |

## Description (`NAME.json`)

```json
{
  "format": "fem-result/1",
  "meta": {"title": "plate with a hole", "fps": 30},
  "dim": 2,
  "points": "points",
  "cells": [{"type": "tri6", "conn": "cells/tri6"}],
  "groups": {
    "left": {"points": "groups/left"},
    "deck": {"cells": {"line2": "groups/deck/line2"}}
  },
  "fields": [
    {"name": "u", "loc": "node", "key": "fields/u", "unit": "m", "frames": false, "label": "displacement", "range": [-0.01, 0.02]}
  ],
  "arrays": [
    {"key": "points", "dtype": "f32", "offset": 0, "shape": [120, 2]},
    {"key": "cells/tri6", "dtype": "i32", "offset": 960, "shape": [50, 6]}
  ]
}
```

| Key | Meaning |
|---|---|
| `format` | always `"fem-result/1"`; readers must reject anything else |
| `meta` | free-form object (titles, units, time step, `fps`, solver settings…) |
| `dim` | 2 or 3, the number of coordinates per point |
| `points` | key of the `(N, dim)` coordinate array |
| `cells` | blocks of same-type cells, in order; `conn` is the key of a `(M, k)` connectivity array (0-based point ids) |
| `groups` | named subsets: `points` → key of a point-id array; `cells` → {cell type: key of a cell-id array within that block} |
| `fields` | named values on the mesh (below) |
| `arrays` | layout of every array in `NAME.bin` |

### Arrays

Each entry of `arrays` gives `key`, `dtype`, byte `offset` into `NAME.bin` and `shape` (row-major, C order).

| dtype | Meaning |
|---|---|
| `f32` | IEEE-754 float32, little-endian |
| `i32` | int32, little-endian |
| `u8` | uint8 |

Offsets are multiples of 4, so every array can be viewed in place as a typed array without copying.

Key conventions: `points`, `cells/<type>`, `groups/<name>`, `groups/<name>/<type>`, `fields/<name>`,
`lines/<name>/pts` + `lines/<name>/len` (polyline sets), and any other key for extra arrays.

### Fields

| Key | Meaning |
|---|---|
| `name` | unique field name |
| `loc` | `"node"`: one value per point; `"cell"`: one value per cell, blocks concatenated in the order of `cells` |
| `key` | array key, normally `fields/<name>` |
| `frames` | `false`: shape `(N,)` or `(N, c)`; `true`: shape `(F, N)` or `(F, N, c)`, one set per frame |
| `unit`, `label` | optional, for legends |
| `range` | `[min, max]` over all values and frames (NaN ignored), for colour scales |

Further keys are allowed and passed through.

### Polyline sets

Written with `Result.polylines(name, lines)`: `lines/<name>/pts` is `(P, c)` float32 (all points of all lines back to
back, `c` columns such as x, y, |σ|) and `lines/<name>/len` is `(L,)` int32 (points per line).

## Cell types and node order

| Type | Nodes | Order |
|---|---|---|
| `line2` | 2 | end points (bars, beams, cables) |
| `tri3` | 3 | counter-clockwise |
| `tri6` | 6 | corners 0-1-2 counter-clockwise; 3 = mid-edge 01, 4 = mid-edge 12, 5 = mid-edge 20 |
| `quad4` | 4 | counter-clockwise |
| `tet4` | 4 | positive volume |
| `hex8` | 8 | VTK order: 0-1-2-3 bottom face counter-clockwise seen from above, 4-5-6-7 each directly above 0-1-2-3 |

Coordinates are right-handed with y (2D) or z (3D) pointing up.

## Versioning

The version is part of `format`. Adding optional keys to `meta`, fields or the description keeps the version; anything
that changes how existing keys are read (dtypes, layout, node order, the meaning of a key) is a new version
(`fem-result/2`). `tests/data/golden_v1.*` is a fixed sample: the Python writer must reproduce it byte for byte and
every reader must decode it.
