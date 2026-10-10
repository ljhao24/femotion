// Display geometry (pure functions, no framework): FemData broken into drawable pieces — triangles, edges, boundary loops,
// cell polygons, deformed coordinates, contour lines, cell reveal order.
import {NODES_PER_CELL, type CellType, type FemData} from './result.ts';

/** the polygon drawn for each cell (node ids around the perimeter); tri6 goes round corners and mid-edge nodes */
const RING: Record<CellType, number[]> = {line2: [0, 1], tri3: [0, 1, 2], tri6: [0, 3, 1, 4, 2, 5], quad4: [0, 1, 2, 3], tet4: [0, 1, 2], hex8: [0, 1, 2, 3]};
/** faces of 3D cells (outward normals, as femotion.mesh.core in Python; hex8 in VTK order) */
const FACES: Partial<Record<CellType, number[][]>> = {
	tet4: [
		[1, 2, 3],
		[0, 3, 2],
		[0, 1, 3],
		[0, 2, 1],
	],
	hex8: [
		[0, 3, 2, 1],
		[4, 5, 6, 7],
		[0, 1, 5, 4],
		[1, 2, 6, 5],
		[2, 3, 7, 6],
		[3, 0, 4, 7],
	],
};
const EDGES3: Partial<Record<CellType, [number, number][]>> = {
	tet4: [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]],
	hex8: [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
};

export type CellRef = {block: number; index: number; type: CellType};

/** all cells, blocks concatenated in the order of `cells` (matches loc="cell" fields) */
export const allCells = (d: FemData): CellRef[] => d.cells.flatMap((c, b) => Array.from({length: c.count}, (_, i) => ({block: b, index: i, type: c.type})));

export const cellNodes = (d: FemData, c: CellRef): number[] => {
	const blk = d.cells[c.block];
	const k = NODES_PER_CELL[blk.type];
	return Array.from(blk.conn.subarray(c.index * k, c.index * k + k));
};

export const cellPolygon = (d: FemData, c: CellRef): number[] => {
	const n = cellNodes(d, c);
	return RING[c.type].map((i) => n[i]);
};

/** display triangles (node-id triples, flattened), the cell each belongs to, and which triangle edges are real mesh lines
 *  (emask: bit0 = a–b, bit1 = b–c, bit2 = c–a; quad diagonals and the inner lines of subdivided quadratic triangles do not
 *  count), used by FieldCanvas to draw mesh lines. tri6 → 4 triangles, quad4 → 2, 3D cells → their outer faces */
export const triangles = (d: FemData): {tri: Int32Array; owner: Int32Array; emask: Uint8Array} => {
	const out: number[] = [];
	const own: number[] = [];
	const em: number[] = [];
	let base = 0;
	for (const blk of d.cells) {
		const k = NODES_PER_CELL[blk.type];
		const c = blk.conn;
		if (blk.type === 'tri3') for (let i = 0; i < blk.count; i++) out.push(c[3 * i], c[3 * i + 1], c[3 * i + 2]), own.push(base + i), em.push(7);
		if (blk.type === 'tri6')
			for (let i = 0; i < blk.count; i++) {
				const n = c.subarray(k * i, k * i + k);
				out.push(n[0], n[3], n[5], n[3], n[1], n[4], n[5], n[4], n[2], n[3], n[4], n[5]);
				own.push(base + i, base + i, base + i, base + i);
				em.push(0b101, 0b011, 0b110, 0);
			}
		if (blk.type === 'quad4')
			for (let i = 0; i < blk.count; i++) {
				const n = c.subarray(4 * i, 4 * i + 4);
				out.push(n[0], n[1], n[2], n[0], n[2], n[3]);
				own.push(base + i, base + i);
				em.push(0b011, 0b110);
			}
		if (blk.type === 'tet4' || blk.type === 'hex8')
			for (const {f, owner} of blockSurface(blk.type, c, blk.count)) {
				for (let j = 1; j + 1 < f.length; j++) {
					out.push(f[0], f[j], f[j + 1]);
					own.push(base + owner);
					// fan split: the first edge is a real edge only for j = 1, the last one only for the last triangle
					em.push((j === 1 ? 1 : 0) | 2 | (j + 2 === f.length ? 4 : 0));
				}
			}
		base += blk.count;
	}
	return {tri: Int32Array.from(out), owner: Int32Array.from(own), emask: Uint8Array.from(em)};
};

/** outer surface of one block of 3D cells: faces that occur once (outward normals), with their cell ids */
const blockSurface = (type: CellType, c: Int32Array, n: number): {f: number[]; owner: number}[] => {
	const loc = FACES[type]!;
	const k = NODES_PER_CELL[type];
	const cnt = new Map<string, number>();
	const all: {f: number[]; owner: number; key: string}[] = [];
	for (let i = 0; i < n; i++)
		for (const l of loc) {
			const f = l.map((j) => c[k * i + j]);
			const key = [...f].sort((a, b) => a - b).join(',');
			cnt.set(key, (cnt.get(key) ?? 0) + 1);
			all.push({f, owner: i, key});
		}
	return all.filter((x) => cnt.get(x.key) === 1).map(({f, owner}) => ({f, owner}));
};

/** outer faces of a 3D mesh (tetrahedra → triangles, hexahedra → quadrilaterals); owner is the global cell index */
export const surfaceFaces = (d: FemData): {f: number[]; owner: number}[] => {
	const out: {f: number[]; owner: number}[] = [];
	let base = 0;
	for (const blk of d.cells) {
		if (FACES[blk.type]) for (const x of blockSurface(blk.type, blk.conn, blk.count)) out.push({f: x.f, owner: base + x.owner});
		base += blk.count;
	}
	return out;
};

/** edges of a 3D mesh visible from the current view: edges of outer faces that face the camera (exact for convex bodies;
 *  concave parts may show a few hidden edges). depth(p): larger is closer (view.ts depth3d) */
export const visibleEdges = (d: FemData, pts: ArrayLike<number>, depth: (p: ArrayLike<number>) => number): [number, number][] => {
	const P = (i: number) => [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]];
	const seen = new Set<string>();
	const out: [number, number][] = [];
	for (const {f} of surfaceFaces(d)) {
		const a = P(f[0]);
		const b = P(f[1]);
		const c = P(f[2]);
		const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
		const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
		const nrm = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
		const len = Math.hypot(nrm[0], nrm[1], nrm[2]) || 1;
		const ctr = [0, 1, 2].map((k) => f.reduce((s, i) => s + pts[3 * i + k], 0) / f.length);
		const eps = 1e-3 * Math.max(1e-9, Math.hypot(u[0], u[1], u[2]));
		if (depth(ctr.map((x, k) => x + (eps * nrm[k]) / len)) <= depth(ctr)) continue;
		for (let i = 0; i < f.length; i++) {
			const x = f[i];
			const y = f[(i + 1) % f.length];
			const key = x < y ? `${x},${y}` : `${y},${x}`;
			if (!seen.has(key)) seen.add(key), out.push([x, y]);
		}
	}
	return out;
};

/** unique edges: [a, b, smallest reveal rank among the cells sharing it]; order defaults to cell ids */
export const edges = (d: FemData, order?: ArrayLike<number>): [number, number, number][] => {
	const map = new Map<string, [number, number, number]>();
	allCells(d).forEach((c, ci) => {
		const e3 = EDGES3[c.type];
		const ring = e3 ? [] : cellPolygon(d, c);
		const pairs: [number, number][] = [];
		if (c.type === 'line2') pairs.push([ring[0], ring[1]]);
		else if (e3) {
			const n = cellNodes(d, c);
			for (const [a, b] of e3) pairs.push([n[a], n[b]]);
		} else for (let i = 0; i < ring.length; i++) pairs.push([ring[i], ring[(i + 1) % ring.length]]);
		const r = order ? order[ci] : ci;
		for (const [a, b] of pairs) {
			const key = a < b ? `${a},${b}` : `${b},${a}`;
			const e = map.get(key);
			if (!e) map.set(key, [a, b, r]);
			else e[2] = Math.min(e[2], r);
		}
	});
	return [...map.values()];
};

/** boundary loops of a 2D mesh (outline, holes) as node-id sequences, longest first (tri6 loops include mid-edge nodes, so
 *  they draw smoothly). Independent of cell orientation */
export const boundaryLoops = (d: FemData): number[][] => {
	const cnt = new Map<string, [number, number, number]>();
	for (const c of allCells(d)) {
		if (c.type === 'line2' || EDGES3[c.type]) continue;
		const ring = cellPolygon(d, c);
		for (let i = 0; i < ring.length; i++) {
			const a = ring[i];
			const b = ring[(i + 1) % ring.length];
			const key = a < b ? `${a},${b}` : `${b},${a}`;
			const e = cnt.get(key);
			if (e) e[2]++;
			else cnt.set(key, [a, b, 1]);
		}
	}
	const adj = new Map<number, number[]>();
	for (const [a, b, n] of cnt.values()) {
		if (n !== 1) continue;
		adj.set(a, [...(adj.get(a) ?? []), b]);
		adj.set(b, [...(adj.get(b) ?? []), a]);
	}
	const seen = new Set<number>();
	const loops: number[][] = [];
	for (const start of adj.keys()) {
		if (seen.has(start)) continue;
		const loop = [start];
		seen.add(start);
		let prev = -1;
		let cur = start;
		for (;;) {
			const nxt = (adj.get(cur) ?? []).find((n) => n !== prev && !seen.has(n));
			if (nxt === undefined) break;
			loop.push(nxt);
			seen.add(nxt);
			prev = cur;
			cur = nxt;
		}
		loops.push(loop);
	}
	return loops.sort((a, b) => b.length - a.length);
};

/** deformed coordinates: points + scale·u (u is a node field; with ≥ dim components the first dim are used) */
export const deformed = (d: FemData, u: {data: ArrayLike<number>; comps: number}, scale: number): Float32Array => {
	const out = new Float32Array(d.points.length);
	for (let i = 0; i < d.nPoints; i++) for (let k = 0; k < d.dim; k++) out[i * d.dim + k] = d.points[i * d.dim + k] + scale * (k < u.comps ? u.data[i * u.comps + k] : 0);
	return out;
};

export const bounds = (points: ArrayLike<number>, dim: number) => {
	const lo = new Array(dim).fill(Infinity);
	const hi = new Array(dim).fill(-Infinity);
	for (let i = 0; i < points.length; i++) {
		const k = i % dim;
		lo[k] = Math.min(lo[k], points[i]);
		hi[k] = Math.max(hi[k], points[i]);
	}
	return {lo, hi};
};

export const centroid = (d: FemData, c: CellRef, pts: ArrayLike<number> = d.points): number[] => {
	const n = cellNodes(d, c);
	const m = c.type === 'tri6' ? n.slice(0, 3) : n;
	return Array.from({length: d.dim}, (_, k) => m.reduce((s, i) => s + pts[i * d.dim + k], 0) / m.length);
};

/** cell reveal order: a sweep by coordinate ('x' left to right, 'y' bottom to top, 'radial' outwards from a point) or
 *  random. Returns the rank of every cell */
export const cellOrder = (d: FemData, by: 'x' | 'y' | 'z' | 'radial' | 'random' = 'x', opts: {from?: number[]; seed?: number} = {}): Int32Array => {
	const cs = allCells(d);
	const key = cs.map((c, i) => {
		const p = centroid(d, c);
		if (by === 'x') return p[0];
		if (by === 'y') return p[1];
		if (by === 'z') return p[2] ?? 0;
		if (by === 'radial') return Math.hypot(...p.map((v, k) => v - (opts.from?.[k] ?? 0)));
		const s = Math.sin((i + 1) * 12.9898 + (opts.seed ?? 0) * 78.233) * 43758.5453;
		return s - Math.floor(s);
	});
	const idx = cs.map((_, i) => i).sort((a, b) => key[a] - key[b]);
	const rank = new Int32Array(cs.length);
	idx.forEach((ci, r) => (rank[ci] = r));
	return rank;
};

// ── contour lines (marching triangles) ──────────────────────────────────
/** segments where values = level on the display triangles, chained into polylines (coordinates from pts, 2D) */
export const isolines = (tri: Int32Array, pts: ArrayLike<number>, values: ArrayLike<number>, level: number, dim = 2): number[][][] => {
	const segs: [number[], number[], string, string][] = [];
	const P = (i: number) => [pts[i * dim], pts[i * dim + 1]];
	const cut = (a: number, b: number) => {
		const t = (level - values[a]) / (values[b] - values[a]);
		const pa = P(a);
		const pb = P(b);
		return {p: [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])], k: a < b ? `${a},${b}` : `${b},${a}`};
	};
	for (let i = 0; i < tri.length; i += 3) {
		const n = [tri[i], tri[i + 1], tri[i + 2]];
		const hits: {p: number[]; k: string}[] = [];
		for (let e = 0; e < 3; e++) {
			const a = n[e];
			const b = n[(e + 1) % 3];
			if (values[a] >= level !== values[b] >= level) hits.push(cut(a, b));
		}
		if (hits.length === 2) segs.push([hits[0].p, hits[1].p, hits[0].k, hits[1].k]);
	}
	// chain segments by shared end points
	const byKey = new Map<string, number[]>();
	segs.forEach((s, i) => {
		for (const k of [s[2], s[3]]) byKey.set(k, [...(byKey.get(k) ?? []), i]);
	});
	const used = new Uint8Array(segs.length);
	const lines: number[][][] = [];
	for (let i = 0; i < segs.length; i++) {
		if (used[i]) continue;
		used[i] = 1;
		const line = [segs[i][0], segs[i][1]];
		for (const [endKey, push] of [
			[segs[i][3], (p: number[]) => line.push(p)],
			[segs[i][2], (p: number[]) => line.unshift(p)],
		] as const) {
			let k = endKey;
			for (;;) {
				const j = (byKey.get(k) ?? []).find((x) => !used[x]);
				if (j === undefined) break;
				used[j] = 1;
				const s = segs[j];
				if (s[2] === k) {
					push(s[1]);
					k = s[3];
				} else {
					push(s[0]);
					k = s[2];
				}
			}
		}
		lines.push(line);
	}
	return lines;
};
