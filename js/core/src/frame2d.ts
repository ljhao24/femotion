// 2D frame finite elements (direct stiffness method), Euler-Bernoulli beams, 3 dofs per node (ux, uy, rz), small enough to
// run live while rendering every frame. Coordinates: x right, y up; SI units (m, N, Pa). Nodal loads, uniform temperature
// change, temperature gradient across the section, spring supports.

export type FemNode = {x: number; y: number};

export type FemElement = {
	n1: number;
	n2: number;
	E: number; // Pa
	A: number; // m²
	I: number; // m⁴
	alpha?: number; // thermal expansion coefficient 1/°C
	dT?: number; // uniform temperature change °C (heating positive)
	dTgrad?: number; // bottom minus top temperature °C (local y up is "top")
	h?: number; // section depth m, for dTgrad
};

export type FemSupport = {
	node: number;
	ux?: boolean;
	uy?: boolean;
	rz?: boolean;
	kx?: number; // spring stiffness N/m
	ky?: number;
	kr?: number; // N·m/rad
};

export type FemLoad = {node: number; fx?: number; fy?: number; mz?: number};

export type FemModel = {
	nodes: FemNode[];
	elements: FemElement[];
	supports: FemSupport[];
	loads?: FemLoad[];
};

export type ElementForces = {
	// axial force: tension positive; bending moment: sagging (bottom in tension, ∪) positive
	N1: number;
	V1: number;
	M1: number;
	N2: number;
	V2: number;
	M2: number;
};

export type FemResult = {
	u: Float64Array; // global displacement vector
	disp: {ux: number; uy: number; rz: number}[];
	forces: ElementForces[];
	reactions: {node: number; fx: number; fy: number; mz: number}[];
};

const geom = (m: FemModel, e: FemElement) => {
	const a = m.nodes[e.n1];
	const b = m.nodes[e.n2];
	const dx = b.x - a.x;
	const dy = b.y - a.y;
	const L = Math.hypot(dx, dy);
	return {L, c: dx / L, s: dy / L};
};

const localStiffness = (e: FemElement, L: number): number[][] => {
	const {E, A, I} = e;
	const a = (E * A) / L;
	const b = (12 * E * I) / L ** 3;
	const c = (6 * E * I) / L ** 2;
	const d = (4 * E * I) / L;
	const f = (2 * E * I) / L;
	return [
		[a, 0, 0, -a, 0, 0],
		[0, b, c, 0, -b, c],
		[0, c, d, 0, -c, f],
		[-a, 0, 0, a, 0, 0],
		[0, -b, -c, 0, b, -c],
		[0, c, f, 0, -c, d],
	];
};

const transform = (c: number, s: number): number[][] => [
	[c, s, 0, 0, 0, 0],
	[-s, c, 0, 0, 0, 0],
	[0, 0, 1, 0, 0, 0],
	[0, 0, 0, c, s, 0],
	[0, 0, 0, -s, c, 0],
	[0, 0, 0, 0, 0, 1],
];

const matMul = (A: number[][], B: number[][]) =>
	A.map((row) => B[0].map((_, j) => row.reduce((sum, v, k) => sum + v * B[k][j], 0)));
const transpose = (A: number[][]) => A[0].map((_, j) => A.map((row) => row[j]));
const matVec = (A: number[][], x: number[]) => A.map((row) => row.reduce((s, v, k) => s + v * x[k], 0));

// equivalent nodal forces of temperature actions (local coordinates, acting on the nodes)
const thermalLoad = (e: FemElement): number[] => {
	const alpha = e.alpha ?? 0;
	const P = e.E * e.A * alpha * (e.dT ?? 0);
	const kappa = e.h ? (alpha * (e.dTgrad ?? 0)) / e.h : 0;
	const M = e.E * e.I * kappa;
	return [-P, 0, -M, P, 0, M];
};

const dofs = (e: FemElement) => [e.n1 * 3, e.n1 * 3 + 1, e.n1 * 3 + 2, e.n2 * 3, e.n2 * 3 + 1, e.n2 * 3 + 2];

// Gaussian elimination with partial pivoting
const solveLinear = (K: number[][], F: number[]): number[] => {
	const n = F.length;
	const A = K.map((row, i) => [...row, F[i]]);
	for (let col = 0; col < n; col++) {
		let piv = col;
		for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
		if (Math.abs(A[piv][col]) < 1e-12) throw new Error(`frame2d: singular stiffness matrix (dof ${col}); check the supports`);
		[A[col], A[piv]] = [A[piv], A[col]];
		for (let r = col + 1; r < n; r++) {
			const f = A[r][col] / A[col][col];
			if (f === 0) continue;
			for (let k = col; k <= n; k++) A[r][k] -= f * A[col][k];
		}
	}
	const x = new Array(n).fill(0);
	for (let r = n - 1; r >= 0; r--) {
		let s = A[r][n];
		for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
		x[r] = s / A[r][r];
	}
	return x;
};

export const solve = (m: FemModel): FemResult => {
	const ndof = m.nodes.length * 3;
	const K = Array.from({length: ndof}, () => new Array(ndof).fill(0));
	const F = new Array(ndof).fill(0);

	const cache = m.elements.map((e) => {
		const {L, c, s} = geom(m, e);
		const k = localStiffness(e, L);
		const T = transform(c, s);
		const Kg = matMul(matMul(transpose(T), k), T);
		const fT = thermalLoad(e);
		const fTg = matVec(transpose(T), fT);
		const idx = dofs(e);
		for (let i = 0; i < 6; i++) {
			F[idx[i]] += fTg[i];
			for (let j = 0; j < 6; j++) K[idx[i]][idx[j]] += Kg[i][j];
		}
		return {k, T, fT, idx, L};
	});

	for (const l of m.loads ?? []) {
		F[l.node * 3] += l.fx ?? 0;
		F[l.node * 3 + 1] += l.fy ?? 0;
		F[l.node * 3 + 2] += l.mz ?? 0;
	}

	const fixed = new Set<number>();
	for (const s of m.supports) {
		const b = s.node * 3;
		if (s.ux) fixed.add(b);
		if (s.uy) fixed.add(b + 1);
		if (s.rz) fixed.add(b + 2);
		if (s.kx) K[b][b] += s.kx;
		if (s.ky) K[b + 1][b + 1] += s.ky;
		if (s.kr) K[b + 2][b + 2] += s.kr;
	}

	const free = [...Array(ndof).keys()].filter((d) => !fixed.has(d));
	const Kff = free.map((i) => free.map((j) => K[i][j]));
	const Ff = free.map((i) => F[i]);
	const uf = solveLinear(Kff, Ff);
	const u = new Float64Array(ndof);
	free.forEach((d, i) => (u[d] = uf[i]));

	const forces = cache.map(({k, T, fT, idx}) => {
		const dl = matVec(T, idx.map((d) => u[d]));
		const f = matVec(k, dl).map((v, i) => v - fT[i]);
		return {N1: -f[0], V1: f[1], M1: -f[2], N2: f[3], V2: -f[4], M2: f[5]};
	});

	// support reactions = K·u − F (only nodes with restraints or springs)
	const reactions = m.supports.map((s) => {
		const r = [0, 1, 2].map((k) => {
			const row = s.node * 3 + k;
			let v = -F[row];
			for (let j = 0; j < ndof; j++) v += K[row][j] * u[j];
			return v;
		});
		return {node: s.node, fx: r[0], fy: r[1], mz: r[2]};
	});

	const disp = m.nodes.map((_, i) => ({ux: u[i * 3], uy: u[i * 3 + 1], rz: u[i * 3 + 2]}));
	return {u, disp, forces, reactions};
};

// sample an element's deflected shape: linear axially + cubic Hermite transversely (exact without span loads).
// scale magnifies the deformation; returns points in global coordinates.
export const deformedElement = (
	m: FemModel,
	r: FemResult,
	ei: number,
	scale: number,
	samples = 16,
): FemNode[] => {
	const e = m.elements[ei];
	const {L, c, s} = geom(m, e);
	const T = transform(c, s);
	const dl = matVec(T, dofs(e).map((d) => r.u[d]));
	const [u1, v1, t1, u2, v2, t2] = dl;
	const a = m.nodes[e.n1];
	const pts: FemNode[] = [];
	for (let i = 0; i <= samples; i++) {
		const xi = i / samples;
		const H1 = 1 - 3 * xi ** 2 + 2 * xi ** 3;
		const H2 = xi - 2 * xi ** 2 + xi ** 3;
		const H3 = 3 * xi ** 2 - 2 * xi ** 3;
		const H4 = -(xi ** 2) + xi ** 3;
		const ul = (1 - xi) * u1 + xi * u2;
		const vl = H1 * v1 + H2 * L * t1 + H3 * v2 + H4 * L * t2;
		const xl = xi * L + scale * ul;
		const yl = scale * vl;
		pts.push({x: a.x + c * xl - s * yl, y: a.y + s * xl + c * yl});
	}
	return pts;
};

// split a member between two nodes into n elements; returns the new node ids and elements (for smooth deflected shapes)
export const addMember = (
	m: FemModel,
	from: number,
	to: number,
	n: number,
	props: Omit<FemElement, 'n1' | 'n2'>,
): number[] => {
	const a = m.nodes[from];
	const b = m.nodes[to];
	const chain = [from];
	for (let i = 1; i < n; i++) {
		m.nodes.push({x: a.x + ((b.x - a.x) * i) / n, y: a.y + ((b.y - a.y) * i) / n});
		chain.push(m.nodes.length - 1);
	}
	chain.push(to);
	const elems: number[] = [];
	for (let i = 0; i < n; i++) {
		m.elements.push({...props, n1: chain[i], n2: chain[i + 1]});
		elems.push(m.elements.length - 1);
	}
	return elems;
};

// ── for teaching: element stiffness and the assembly process ("piece by piece into the global matrix") ──
export type ElementMatrices = {
	L: number;
	/** 6×6 stiffness in local coordinates */
	kLocal: number[][];
	/** local ← global transformation */
	T: number[][];
	/** 6×6 stiffness in global coordinates = Tᵀ k T */
	kGlobal: number[][];
	/** global dof numbers of the 6 local dofs (ux, uy, rz per node) */
	dofs: number[];
};

export const elementMatrices = (m: FemModel, ei: number): ElementMatrices => {
	const e = m.elements[ei];
	const {L, c, s} = geom(m, e);
	const kLocal = localStiffness(e, L);
	const T = transform(c, s);
	return {L, kLocal, T, kGlobal: matMul(matMul(transpose(T), kLocal), T), dofs: dofs(e)};
};

/** assemble element by element: the global matrix after every step (small demonstration models only; ndof² × elements) */
export const assemblySteps = (m: FemModel): {element: number; dofs: number[]; K: number[][]}[] => {
	const ndof = m.nodes.length * 3;
	const K = Array.from({length: ndof}, () => new Array(ndof).fill(0));
	return m.elements.map((_, ei) => {
		const {kGlobal, dofs: d} = elementMatrices(m, ei);
		for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) K[d[i]][d[j]] += kGlobal[i][j];
		return {element: ei, dofs: d, K: K.map((row) => [...row])};
	});
};
