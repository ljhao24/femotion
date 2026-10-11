# @femotion/core

Read [femotion](../../README.md)'s `fem-result/1` files and turn finite-element results into drawable geometry. No
dependencies and no framework: use it with Canvas, SVG, WebGL, React, Remotion or plain Node.

```ts
import {parseFemResult, fieldAt, scalarOf, triangles, boundaryLoops, fit2d, projector2d, makeColormap, normalizer} from '@femotion/core';

const d = parseFemResult(JSON.parse(jsonText), binArrayBuffer);   // or: await fetchFemResult(jsonUrl, binUrl)
const vm = d.fields.vm;                                          // fields: Float32Array / Int32Array / Uint8Array views
const u = fieldAt(d, 'u_t', frame);                              // per-frame fields
const {tri, owner} = triangles(d);                               // display triangles (tri6 → 4, quad4 → 2, 3D → outer faces)
const proj = projector2d(fit2d(lo, hi, {x: 0, y: 0, w: 1920, h: 1080}));
```

| Module | Contents |
|---|---|
| `result` | `parseFemResult`, `fetchFemResult`, `fieldAt`, `scalarOf`, `polylinesOf`, `point`, `frame2dToResult`, types |
| `frame2d` | a 2D frame solver small enough to run every frame (`solve`, `addMember`, `deformedElement`), element matrices and assembly steps for teaching |
| `geometry` | `triangles`, `edges`, `boundaryLoops`, `surfaceFaces`, `visibleEdges`, `deformed`, `isolines`, `cellOrder`, `bounds` |
| `view` | `fit2d` / `projector2d`, orthographic `projector3d` / `depth3d` / `lerpView`, `perspective` / `lerpPersp` |
| `colormap` | `makeColormap`, `normalizer`, `banded`, `rgbCss` |

Tests (Node ≥ 22.18 runs TypeScript directly): `npm test`. They decode the same fixed sample
(`tests/data/golden_v1.*`) that the Python writer must reproduce byte for byte.

Licence: AGPL-3.0-or-later.
