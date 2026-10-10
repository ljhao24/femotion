// Reading fem-result/1 files (written by femotion.result in Python; specification: docs/result-format.md): a mesh, fields and
// per-frame sequences. The same FemData can also be built from a model computed live in the browser (frame2dToResult),
// so every display component works with either.
import type {FemModel, FemResult as Frame2DResult} from './frame2d.ts';

export type CellType = 'line2' | 'tri3' | 'tri6' | 'quad4' | 'tet4' | 'hex8';
export const NODES_PER_CELL: Record<CellType, number> = {line2: 2, tri3: 3, tri6: 6, quad4: 4, tet4: 4, hex8: 8};

export type FieldInfo = {
	name: string;
	loc: 'node' | 'cell';
	unit?: string | null;
	label?: string | null;
	frames: boolean;
	range: [number, number];
	[k: string]: unknown;
};

export type Arr = {data: Float32Array | Int32Array | Uint8Array; shape: number[]};

export type FemData = {
	meta: Record<string, unknown>;
	dim: 2 | 3;
	/** node coordinates, (N, dim) flattened */
	points: Float32Array;
	nPoints: number;
	cells: {type: CellType; conn: Int32Array; count: number}[];
	groups: Record<string, {points?: Int32Array; cells?: Partial<Record<CellType, Int32Array>>}>;
	fields: Record<string, FieldInfo & Arr>;
	/** extra arrays (densities of every topology-optimisation iteration, stiffness patterns, …) by key */
	arrays: Record<string, Arr>;
};

/** the NAME.json description (see docs/result-format.md) */
export type ResultDescription = {
	format: string;
	meta: Record<string, unknown>;
	dim: 2 | 3;
	points: string;
	cells: {type: CellType; conn: string}[];
	groups: Record<string, {points?: string; cells?: Record<string, string>}>;
	fields: (FieldInfo & {key: string})[];
	arrays: {key: string; dtype: 'f32' | 'i32' | 'u8'; offset: number; shape: number[]}[];
};

/** pure function: description JSON + binary → FemData (no I/O; works anywhere) */
export const parseFemResult = (desc: ResultDescription, buf: ArrayBuffer): FemData => {
	if (desc.format !== 'fem-result/1') throw new Error(`unknown result format ${desc.format}`);
	const arrays: Record<string, Arr> = {};
	for (const a of desc.arrays) {
		const n = a.shape.reduce((x, y) => x * y, 1);
		const data = a.dtype === 'f32' ? new Float32Array(buf, a.offset, n) : a.dtype === 'i32' ? new Int32Array(buf, a.offset, n) : new Uint8Array(buf, a.offset, n);
		arrays[a.key] = {data, shape: a.shape};
	}
	const pts = arrays[desc.points];
	const fields: FemData['fields'] = {};
	for (const f of desc.fields) fields[f.name] = {...f, ...arrays[f.key]};
	const groups: FemData['groups'] = {};
	for (const [g, d] of Object.entries(desc.groups)) {
		groups[g] = {
			points: d.points ? (arrays[d.points].data as Int32Array) : undefined,
			cells: d.cells ? Object.fromEntries(Object.entries(d.cells).map(([k, v]) => [k, arrays[v].data as Int32Array])) : undefined,
		};
	}
	return {
		meta: desc.meta,
		dim: desc.dim,
		points: pts.data as Float32Array,
		nPoints: pts.shape[0],
		cells: desc.cells.map((c) => ({type: c.type, conn: arrays[c.conn].data as Int32Array, count: arrays[c.conn].shape[0]})),
		groups,
		fields,
		arrays,
	};
};

/** fetch NAME.json and NAME.bin from two URLs and parse them (any environment with fetch: browsers, Node ≥ 18, bundlers) */
export const fetchFemResult = async (jsonUrl: string, binUrl: string, fetchFn: typeof fetch = fetch): Promise<FemData> => {
	const desc = (await (await fetchFn(jsonUrl)).json()) as ResultDescription;
	const buf = await (await fetchFn(binUrl)).arrayBuffer();
	return parseFemResult(desc, buf);
};

// ── accessors ───────────────────────────────────────────────────────────
/** frame `frame` of a field (ignored for fields without frames); returns (N, c) flattened and the component count */
export const fieldAt = (d: FemData, name: string, frame = 0): {data: ArrayLike<number>; comps: number} => {
	const f = d.fields[name];
	if (!f) throw new Error(`no field ${name}`);
	const shape = f.frames ? f.shape.slice(1) : f.shape;
	const per = shape.reduce((a, b) => a * b, 1);
	const comps = shape.length > 1 ? shape.slice(1).reduce((a, b) => a * b, 1) : 1;
	if (!f.frames) return {data: f.data, comps};
	const k = Math.max(0, Math.min(f.shape[0] - 1, Math.round(frame)));
	return {data: f.data.subarray(k * per, (k + 1) * per), comps};
};

/** to scalars: one component, or the vector magnitude */
export const scalarOf = (v: {data: ArrayLike<number>; comps: number}, comp: number | 'mag' = 0): Float32Array => {
	const n = v.data.length / v.comps;
	const out = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		if (comp === 'mag') {
			let s = 0;
			for (let c = 0; c < v.comps; c++) s += v.data[i * v.comps + c] ** 2;
			out[i] = Math.sqrt(s);
		} else out[i] = v.data[i * v.comps + comp];
	}
	return out;
};

export const point = (d: FemData, i: number): number[] => Array.from(d.points.subarray(i * d.dim, i * d.dim + d.dim));

// ── a 2D frame computed live in the browser → the same data ──────────────
/** frame2d model and result → FemData: line2 cells, node fields u (ux, uy) and rz, cell fields N and M (mean of both ends) */
export const frame2dToResult = (m: FemModel, r?: Frame2DResult): FemData => {
	const N = m.nodes.length;
	const points = new Float32Array(N * 2);
	m.nodes.forEach((p, i) => {
		points[2 * i] = p.x;
		points[2 * i + 1] = p.y;
	});
	const conn = new Int32Array(m.elements.length * 2);
	m.elements.forEach((e, i) => {
		conn[2 * i] = e.n1;
		conn[2 * i + 1] = e.n2;
	});
	const fields: FemData['fields'] = {};
	const add = (name: string, loc: 'node' | 'cell', data: Float32Array, shape: number[]) => {
		let lo = Infinity;
		let hi = -Infinity;
		for (const v of data) {
			lo = Math.min(lo, v);
			hi = Math.max(hi, v);
		}
		fields[name] = {name, loc, frames: false, range: [lo, hi], data, shape};
	};
	if (r) {
		add('u', 'node', Float32Array.from(r.disp.flatMap((d) => [d.ux, d.uy])), [N, 2]);
		add('rz', 'node', Float32Array.from(r.disp.map((d) => d.rz)), [N]);
		add('N', 'cell', Float32Array.from(r.forces.map((f) => (f.N1 + f.N2) / 2)), [m.elements.length]);
		add('M', 'cell', Float32Array.from(r.forces.map((f) => (f.M1 + f.M2) / 2)), [m.elements.length]);
	}
	return {meta: {source: 'frame2d'}, dim: 2, points, nPoints: N, cells: [{type: 'line2', conn, count: m.elements.length}], groups: {}, fields, arrays: {}};
};

/** a polyline set stored by Result.polylines(name, lines) in Python → one point list per line (c numbers per point, the first dim are coordinates) */
export const polylinesOf = (d: FemData, name: string): number[][][] => {
	const P = d.arrays[`lines/${name}/pts`];
	const L = d.arrays[`lines/${name}/len`];
	if (!P || !L) return [];
	const c = P.shape[1] ?? 2;
	const out: number[][][] = [];
	let k = 0;
	for (const n of L.data) {
		const line: number[][] = [];
		for (let i = 0; i < n; i++, k++) line.push(Array.from(P.data.subarray(k * c, k * c + c)));
		out.push(line);
	}
	return out;
};
