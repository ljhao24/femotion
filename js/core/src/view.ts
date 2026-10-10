// Model coordinates → canvas coordinates. Every display component takes a Projector: p (model coordinates) → [X, Y] pixels.
// - 2D: fit2d fits the model bounding box into a box on screen (y up becomes screen down).
// - 3D: orthographic project (yaw about the vertical axis, pitch, z up as in the Python models); lerpView blends two views.
// - perspective: an orbit camera that blends continuously with the orthographic view (dolly zoom, flying into a model).
export type Projector = (p: ArrayLike<number>) => [number, number];

export type View2D = {s: number; ox: number; oy: number};
/** fit the [lo, hi] bounding box into box (pixels) keeping proportions; pad is the margin ratio */
export const fit2d = (lo: number[], hi: number[], box: {x: number; y: number; w: number; h: number}, pad = 0.06): View2D => {
	const w = hi[0] - lo[0] || 1;
	const h = hi[1] - lo[1] || 1;
	const s = Math.min((box.w * (1 - 2 * pad)) / w, (box.h * (1 - 2 * pad)) / h);
	const cx = box.x + box.w / 2;
	const cy = box.y + box.h / 2;
	return {s, ox: cx - s * (lo[0] + w / 2), oy: cy + s * (lo[1] + h / 2)};
};
export const projector2d = (v: View2D): Projector => (p) => [v.ox + v.s * p[0], v.oy - v.s * p[1]];
export const lerpView2D = (a: View2D, b: View2D, t: number): View2D => ({s: a.s + (b.s - a.s) * t, ox: a.ox + (b.ox - a.ox) * t, oy: a.oy + (b.oy - a.oy) * t});

export type View = {yaw: number; pitch: number; s: number; cx: number; cy: number};
export const project = (v: View) => {
	const ps = Math.sin((v.yaw * Math.PI) / 180);
	const pc = Math.cos((v.yaw * Math.PI) / 180);
	const ts = Math.sin((v.pitch * Math.PI) / 180);
	const tc = Math.cos((v.pitch * Math.PI) / 180);
	return (x: number, y: number, z: number): [number, number] => {
		const xs = x * pc - y * ps;
		const d = x * ps + y * pc;
		return [v.cx + v.s * xs, v.cy - v.s * (z * tc + d * ts)];
	};
};
export const projector3d = (v: View): Projector => {
	const f = project(v);
	return (p) => f(p[0], p[1], p[2] ?? 0);
};
/** depth along the 3D viewing direction (larger is closer), for painter's sorting */
export const depth3d = (v: View) => {
	const ps = Math.sin((v.yaw * Math.PI) / 180);
	const pc = Math.cos((v.yaw * Math.PI) / 180);
	const ts = Math.sin((v.pitch * Math.PI) / 180);
	const tc = Math.cos((v.pitch * Math.PI) / 180);
	return (p: ArrayLike<number>) => -(p[0] * ps + p[1] * pc) * tc + (p[2] ?? 0) * ts;
};
export const lerpView = (a: View, b: View, t: number): View => ({
	yaw: a.yaw + (b.yaw - a.yaw) * t,
	pitch: a.pitch + (b.pitch - a.pitch) * t,
	s: Math.exp(Math.log(a.s) + (Math.log(b.s) - Math.log(a.s)) * t),
	cx: a.cx + (b.cx - a.cx) * t,
	cy: a.cy + (b.cy - a.cy) * t,
});

/** projection of node i (pts: (N, dim) flattened coordinates, possibly deformed) */
export const projAt = (proj: Projector, pts: ArrayLike<number>, dim: number, i: number) =>
	proj(dim === 2 ? [pts[2 * i], pts[2 * i + 1]] : [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]]);

/** perspective camera orbiting `target` (yaw and pitch as in the orthographic View), `dist` model units away, focal length f
 *  pixels, the target landing at (cx, cy) on screen. With a large dist and f = s·dist it equals projector3d({yaw, pitch, s, …}),
 *  so orthographic and perspective views blend continuously (dolly zoom), and the camera can fly into a model */
export type PerspView = {yaw: number; pitch: number; dist: number; f: number; target: [number, number, number]; cx: number; cy: number};
export const perspective = (v: PerspView) => {
	const ps = Math.sin((v.yaw * Math.PI) / 180);
	const pc = Math.cos((v.yaw * Math.PI) / 180);
	const ts = Math.sin((v.pitch * Math.PI) / 180);
	const tc = Math.cos((v.pitch * Math.PI) / 180);
	const w = [-ps * tc, -pc * tc, ts]; // from the target towards the camera
	const r = [pc, -ps, 0];
	const u = [ps * ts, pc * ts, tc];
	const C = [v.target[0] + v.dist * w[0], v.target[1] + v.dist * w[1], v.target[2] + v.dist * w[2]];
	/** camera coordinates: right, up, distance along the view */
	const cam = (p: ArrayLike<number>): [number, number, number] => {
		const x = p[0] - C[0];
		const y = p[1] - C[1];
		const z = (p[2] ?? 0) - C[2];
		return [x * r[0] + y * r[1] + z * r[2], x * u[0] + y * u[1] + z * u[2], -(x * w[0] + y * w[1] + z * w[2])];
	};
	const proj: Projector = (p) => {
		const [xc, yc, zc] = cam(p);
		const z = Math.max(zc, 1e-3);
		return [v.cx + (v.f * xc) / z, v.cy - (v.f * yc) / z];
	};
	/** larger is closer (for the FieldCanvas depth buffer) */
	const depth = (p: ArrayLike<number>) => -cam(p)[2];
	return {cam, proj, depth, C, r, u, w};
};
/** blend two perspective views: distance and focal length logarithmically (uniform dolly), the rest linearly */
export const lerpPersp = (a: PerspView, b: PerspView, t: number): PerspView => {
	const L = (x: number, y: number) => x + (y - x) * t;
	const G = (x: number, y: number) => Math.exp(Math.log(x) + (Math.log(y) - Math.log(x)) * t);
	return {yaw: L(a.yaw, b.yaw), pitch: L(a.pitch, b.pitch), dist: G(a.dist, b.dist), f: G(a.f, b.f), target: [L(a.target[0], b.target[0]), L(a.target[1], b.target[1]), L(a.target[2], b.target[2])], cx: L(a.cx, b.cx), cy: L(a.cy, b.cy)};
};
