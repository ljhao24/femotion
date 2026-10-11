# @femotion/remotion

[Remotion](https://www.remotion.dev) components for [femotion](../../README.md) results. Builds on `@femotion/core`.

```tsx
import {useFemResult, FieldCanvas, MeshLines} from '@femotion/remotion';
import {fieldAt, scalarOf, fit2d, projector2d, makeColormap, normalizer} from '@femotion/core';

const d = useFemResult('fem/plate');            // public/fem/plate.json + .bin, holds the render until loaded
if (!d) return null;
const vm = d.fields.vm;
const proj = projector2d(fit2d([0, 0], [10, 10], {x: 160, y: 90, w: 1600, h: 900}));
return (
  <>
    <FieldCanvas d={d} proj={proj} values={vm.data} norm={normalizer(vm.range)} cmap={makeColormap([[0, '#F2EADB'], [1, '#8E3A12']])} />
    <MeshLines d={d} proj={proj} width={0.6} />
  </>
);
```

| Export | Contents |
|---|---|
| `useFemResult`, `loadFemResult` | load a result from `public/` with `staticFile`, cached, `delayRender` while loading |
| `FieldCanvas` | per-pixel field colouring with reveal order, banding, 3D depth buffer and shading, mesh lines in the same canvas |
| `MeshLines`, `Polylines`, `ColorBar`, `MatrixGrid` | SVG line art: mesh and boundaries, stress trajectories and contours, colour bars, stiffness-matrix fill |

Peer dependencies: `react` ≥ 18, `remotion` ≥ 4. Licence: AGPL-3.0-or-later.
