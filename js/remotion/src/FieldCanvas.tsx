// Field colouring: per-pixel rasterisation of the display triangles; node fields interpolated barycentrically, cell fields
// flat per cell. 3D (outer faces) gets a depth buffer and simple shading. Cells can be revealed in order (reveal), values
// can be banded into flat colours (bands), and mesh lines can be drawn into the same canvas (edges). Blends with
// multiply by default, which suits paper backgrounds.
import React, {useLayoutEffect, useMemo, useRef} from 'react';
import {useVideoConfig} from 'remotion';
import {banded, projAt, triangles, type FemData, type Projector, type RGB} from '@femotion/core';

export type FieldCanvasProps = {
	d: FemData;
	proj: Projector;
	/** node coordinates (default: the original ones; pass deformed coordinates to draw the deformed shape) */
	pts?: ArrayLike<number>;
	/** node values (length N) or cell values (length = total cells, with loc="cell") */
	values: ArrayLike<number>;
	loc?: 'node' | 'cell';
	/** value → 0..1 */
	norm: (v: number) => number;
	cmap: (t: number) => RGB;
	bands?: number;
	opacity?: number;
	/** draw only cells with order[cell] < upto; the fractional part fades the next cell in */
	reveal?: {order: ArrayLike<number>; upto: number};
	blend?: React.CSSProperties['mixBlendMode'];
	/** supersampling factor (smoother edges), default 2 */
	ss?: number;
	/** 3D: depth function (larger is closer); enables the depth buffer */
	depth?: (p: ArrayLike<number>) => number;
	/** 3D shading strength, 0 = none */
	shade?: number;
	/** area covered by the canvas (parent coordinates, pixels). Default: the whole frame; give the relevant region when the
	 *  canvas lives in a large world-coordinate layer */
	box?: {x: number; y: number; w: number; h: number};
	/** mesh lines drawn into the same canvas (depth-tested, so in 3D only lines on visible faces show): colour, width (px), opacity */
	edges?: {color: RGB; width?: number; opacity?: number};
};

export const FieldCanvas: React.FC<FieldCanvasProps> = ({d, proj, pts = d.points, values, loc = 'node', norm, cmap, bands = 0, opacity = 1, reveal, blend = 'multiply', ss = 2, depth, shade = 0.35, box, edges}) => {
	const vc = useVideoConfig();
	const bx = box ?? {x: 0, y: 0, w: vc.width, h: vc.height};
	const W = bx.w;
	const H = bx.h;
	const ref = useRef<HTMLCanvasElement>(null);
	const {tri, owner, emask} = useMemo(() => triangles(d), [d]);
	const q = banded(bands);

	useLayoutEffect(() => {
		const cv = ref.current;
		if (!cv) return;
		const w = W * ss;
		const h = H * ss;
		const ctx = cv.getContext('2d')!;
		const img = ctx.createImageData(w, h);
		const px = img.data;
		const zb = depth ? new Float32Array(w * h).fill(-Infinity) : null;
		const nT = tri.length / 3;
		const P = new Float32Array(d.nPoints * 2);
		const Z = depth ? new Float32Array(d.nPoints) : null;
		for (let i = 0; i < d.nPoints; i++) {
			const [x, y] = projAt(proj, pts, d.dim, i);
			P[2 * i] = (x - bx.x) * ss;
			P[2 * i + 1] = (y - bx.y) * ss;
			if (Z) Z[i] = depth!(d.dim === 2 ? [pts[2 * i], pts[2 * i + 1]] : [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]]);
		}
		const light = [0.35, -0.45, 0.82];
		const edgeOn = !!edges && (edges.opacity ?? 1) > 0.005;
		const ec = edges?.color ?? [0, 0, 0];
		const ew = ((edges?.width ?? 1) * ss) / 2; // each triangle draws half the width; neighbours make up the whole line
		const eo = edges?.opacity ?? 1;
		for (let t = 0; t < nT; t++) {
			const cell = owner[t];
			let alpha = 1;
			if (reveal) {
				const r = reveal.order[cell];
				if (r >= reveal.upto) continue;
				alpha = Math.min(1, reveal.upto - r);
			}
			const a = tri[3 * t];
			const b = tri[3 * t + 1];
			const c = tri[3 * t + 2];
			const ax = P[2 * a], ay = P[2 * a + 1], bx = P[2 * b], by = P[2 * b + 1], cx = P[2 * c], cy = P[2 * c + 1];
			const det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
			if (Math.abs(det) < 1e-9) continue;
			// mesh lines: distance from the pixel to each edge = barycentric coordinate × twice the area / edge length
			const em = edgeOn ? emask[t] : 0;
			const kA = em & 2 ? Math.abs(det) / (Math.hypot(cx - bx, cy - by) || 1) : Infinity;
			const kB = em & 4 ? Math.abs(det) / (Math.hypot(ax - cx, ay - cy) || 1) : Infinity;
			const kC = em & 1 ? Math.abs(det) / (Math.hypot(bx - ax, by - ay) || 1) : Infinity;
			let lit = 1;
			if (d.dim === 3 && shade) {
				const g = (i: number, k: number) => pts[3 * i + k];
				const u = [g(b, 0) - g(a, 0), g(b, 1) - g(a, 1), g(b, 2) - g(a, 2)];
				const v = [g(c, 0) - g(a, 0), g(c, 1) - g(a, 1), g(c, 2) - g(a, 2)];
				const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
				const len = Math.hypot(n[0], n[1], n[2]) || 1;
				lit = 1 - shade + shade * Math.abs((n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / len);
			}
			const va = loc === 'cell' ? values[cell] : values[a];
			const vb = loc === 'cell' ? values[cell] : values[b];
			const vc = loc === 'cell' ? values[cell] : values[c];
			const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
			const x1 = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
			const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
			const y1 = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
			for (let y = y0; y <= y1; y++) {
				const sy = y + 0.5;
				for (let x = x0; x <= x1; x++) {
					const sx = x + 0.5;
					const l1 = ((bx - sx) * (cy - sy) - (cx - sx) * (by - sy)) / det;
					const l2 = ((cx - sx) * (ay - sy) - (ax - sx) * (cy - sy)) / det;
					const l3 = 1 - l1 - l2;
					if (l1 < -1e-6 || l2 < -1e-6 || l3 < -1e-6) continue;
					const k = y * w + x;
					if (zb && Z) {
						const z = l1 * Z[a] + l2 * Z[b] + l3 * Z[c];
						if (z <= zb[k]) continue;
						zb[k] = z;
					}
					const col = cmap(q(norm(l1 * va + l2 * vb + l3 * vc)));
					let r = col[0] * lit;
					let g = col[1] * lit;
					let bl = col[2] * lit;
					if (em) {
						const dist = Math.min(l1 * kA, l2 * kB, l3 * kC);
						const cov = Math.max(0, Math.min(1, ew - dist + 0.5)) * eo;
						if (cov > 0) {
							r += (ec[0] - r) * cov;
							g += (ec[1] - g) * cov;
							bl += (ec[2] - bl) * cov;
						}
					}
					px[4 * k] = r;
					px[4 * k + 1] = g;
					px[4 * k + 2] = bl;
					px[4 * k + 3] = 255 * alpha;
				}
			}
		}
		ctx.putImageData(img, 0, 0);
	});

	return <canvas ref={ref} width={W * ss} height={H * ss} style={{position: 'absolute', left: bx.x, top: bx.y, width: W, height: H, opacity, mixBlendMode: blend}} />;
};
