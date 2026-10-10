// @femotion/core: read fem-result/1 files and turn them into drawable geometry. No dependencies, no framework.
//   result    parse files, read fields and polylines, build the same data from a live 2D frame
//   frame2d   a 2D frame solver small enough to run every frame, plus element matrices and assembly steps for teaching
//   geometry  display triangles, edges, boundary loops, 3D outer faces and visible edges, deformation, contours, reveal order
//   view      projections: 2D fit, 3D orthographic, perspective orbit camera (blends with orthographic)
//   colormap  colour ramps from hex stops, normalisation, banding
export * from './result.ts';
export * from './frame2d.ts';
export * from './geometry.ts';
export * from './view.ts';
export * from './colormap.ts';
