// Display geometry and projections on the golden sample and on a small hexahedral block.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {test} from 'node:test';
import {
	boundaryLoops, bounds, cellOrder, depth3d, edges, fit2d, isolines, lerpView, parseFemResult, perspective, projector2d,
	projector3d, surfaceFaces, triangles, visibleEdges, type FemData,
} from '../src/index.ts';

const DATA = path.resolve(import.meta.dirname, '../../../tests/data');
const b = fs.readFileSync(path.join(DATA, 'golden_v1.bin'));
const g = parseFemResult(JSON.parse(fs.readFileSync(path.join(DATA, 'golden_v1.json'), 'utf8')), b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);

/** an n × n × n block of unit hexahedra (VTK order), like femotion.mesh.structured.box */
const hexBlock = (n: number): FemData => {
	const id = (i: number, j: number, k: number) => i + (n + 1) * (j + (n + 1) * k);
	const points = new Float32Array((n + 1) ** 3 * 3);
	for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) points.set([i, j, k], 3 * id(i, j, k));
	const conn: number[] = [];
	for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++)
		conn.push(id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k), id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1));
	return {meta: {}, dim: 3, points, nPoints: (n + 1) ** 3, cells: [{type: 'hex8', conn: Int32Array.from(conn), count: n ** 3}], groups: {}, fields: {}, arrays: {}};
};

test('display triangles: tri3 kept, quad4 split without its diagonal as a mesh line', () => {
	const {tri, owner, emask} = triangles(g);
	assert.equal(tri.length / 3, 4);
	assert.deepEqual(Array.from(owner), [0, 1, 2, 2]);
	const bits = (m: number) => (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1);
	assert.equal(bits(emask[2]) + bits(emask[3]), 4); // the quad's four edges, not its diagonal
});

test('edges, boundary loop, reveal order, bounds', () => {
	assert.equal(edges(g).length, 8);
	const loops = boundaryLoops(g);
	assert.equal(loops.length, 1);
	assert.deepEqual([...loops[0]].sort(), [0, 1, 2, 3, 4, 5]);
	const ord = cellOrder(g, 'x');
	assert.deepEqual([...ord].sort(), [0, 1, 2]);
	assert.equal(ord[2], 2); // the quad lies furthest right
	const bb = bounds(g.points, 2);
	assert.deepEqual([bb.lo, bb.hi], [[0, 0], [2, 1]]);
});

test('contour of a linear field is a straight line', () => {
	const {tri} = triangles(g);
	const x = Float32Array.from({length: 6}, (_, i) => g.points[2 * i]);
	const ls = isolines(tri, g.points, x, 1.5);
	assert.ok(ls.length >= 1);
	for (const l of ls) for (const p of l) assert.ok(Math.abs(p[0] - 1.5) < 1e-6);
});

test('hexahedral block: outer faces, visible edges, mesh-line marks', () => {
	const n = 2;
	const h = hexBlock(n);
	assert.equal(surfaceFaces(h).length, 6 * n * n);
	const {tri, emask} = triangles(h);
	assert.equal(tri.length / 3, 2 * 6 * n * n);
	const bits = Array.from(emask).reduce((s, m) => s + (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1), 0);
	assert.equal(bits, 6 * n * n * 4); // quad diagonals are not mesh lines
	// seen from above at an angle: three faces, 2n(n+1) edges each, minus the 3n edges they share
	const vis = visibleEdges(h, h.points, depth3d({yaw: -30, pitch: 30, s: 1, cx: 0, cy: 0}));
	assert.equal(vis.length, 3 * 2 * n * (n + 1) - 3 * n);
});

test('fit2d maps the bounding box onto the box with y pointing down', () => {
	const proj = projector2d(fit2d([-4, -2], [4, 2], {x: 0, y: 0, w: 1000, h: 500}, 0));
	const [x0, y0] = proj([-4, 2]);
	const [x1, y1] = proj([4, -2]);
	assert.ok(Math.abs(x0) < 1e-9 && Math.abs(y0) < 1e-9 && Math.abs(x1 - 1000) < 1e-9 && Math.abs(y1 - 500) < 1e-9);
});

test('a distant perspective camera with f = s·dist equals the orthographic view', () => {
	const yaw = 31, pitch = 14, sc = 60, zc = 6.6;
	const ortho = projector3d({yaw, pitch, s: sc, cx: 1000, cy: 500 + sc * zc * Math.cos((pitch * Math.PI) / 180)});
	const P = perspective({yaw, pitch, dist: 20000, f: sc * 20000, target: [0, 0, zc], cx: 1000, cy: 500});
	for (const q of [[0, 0, 0], [5.6, 1.1, 13], [-2.8, -2.8, 0], [1, -0.9, 10.6]]) {
		const [a0, a1] = ortho(q);
		const [b0, b1] = P.proj(q);
		assert.ok(Math.hypot(a0 - b0, a1 - b1) < 0.2);
	}
});

test('lerpView hits both ends', () => {
	const a = {yaw: 0, pitch: 10, s: 1, cx: 0, cy: 0};
	const c = {yaw: 90, pitch: 30, s: 4, cx: 100, cy: 50};
	assert.deepEqual(lerpView(a, c, 0), a);
	const e = lerpView(a, c, 1);
	assert.ok(Math.abs(e.s - 4) < 1e-12 && e.yaw === 90 && e.cx === 100);
});
