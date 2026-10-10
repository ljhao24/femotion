// The fixed fem-result/1 sample (../../../tests/data/golden_v1.*) that the Python writer reproduces byte for byte:
// every feature of the format must decode here exactly as written there (tests/golden.py).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {test} from 'node:test';
import {fieldAt, parseFemResult, polylinesOf, scalarOf, type FemData, type ResultDescription} from '../src/index.ts';

const DATA = path.resolve(import.meta.dirname, '../../../tests/data');
const load = (name: string): FemData => {
	const desc = JSON.parse(fs.readFileSync(path.join(DATA, `${name}.json`), 'utf8')) as ResultDescription;
	const b = fs.readFileSync(path.join(DATA, `${name}.bin`));
	return parseFemResult(desc, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
};
const close = (a: ArrayLike<number>, b: number[], tol = 1e-7) => {
	assert.equal(a.length, b.length);
	for (let i = 0; i < b.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `index ${i}: ${a[i]} ≠ ${b[i]}`);
};

const d = load('golden_v1');
const P = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]];
const U = P.flatMap(([x, y]) => [x * 0.01, -y * 0.02]);

test('meta, points, cell blocks in order', () => {
	assert.deepEqual(d.meta, {title: 'golden sample', fps: 30});
	assert.equal(d.dim, 2);
	assert.equal(d.nPoints, 6);
	close(d.points, P.flat());
	assert.deepEqual(d.cells.map((c) => [c.type, c.count, Array.from(c.conn)]), [['tri3', 2, [0, 1, 4, 0, 4, 3]], ['quad4', 1, [1, 2, 5, 4]]]);
});

test('point and cell groups', () => {
	assert.deepEqual(Array.from(d.groups.left.points!), [0, 3]);
	assert.deepEqual(Array.from(d.groups.right.cells!.quad4!), [0]);
});

test('node, cell and per-frame fields with dtypes, shapes and ranges', () => {
	const u = d.fields.u;
	assert.ok(u.data instanceof Float32Array);
	assert.deepEqual([u.loc, u.unit, u.label, u.frames, u.shape], ['node', 'm', 'displacement', false, [6, 2]]);
	close(u.data, U);
	assert.deepEqual(u.range, [-0.02, 0.02]);
	close(d.fields.vm.data, [0, 1, 2, 3, 4, 5]);
	const rank = d.fields.rank;
	assert.ok(rank.data instanceof Int32Array);
	assert.deepEqual([rank.loc, Array.from(rank.data)], ['cell', [0, 1, 2]]);
	const ut = d.fields.u_t;
	assert.deepEqual([ut.frames, ut.shape], [true, [3, 6, 2]]);
	close(fieldAt(d, 'u_t', 2).data, U);
	close(fieldAt(d, 'u_t', 1).data, U.map((v) => v * 0.5));
	assert.equal(fieldAt(d, 'u_t', 9).data[2], fieldAt(d, 'u_t', 2).data[2]); // clamped to the last frame
	close(scalarOf(fieldAt(d, 'u'), 'mag'), P.map(([x, y]) => Math.hypot(x * 0.01, y * 0.02)));
	close(scalarOf(fieldAt(d, 'u'), 1), P.map(([, y]) => -y * 0.02));
});

test('polyline sets and extra arrays', () => {
	const lines = polylinesOf(d, 'flow');
	assert.equal(lines.length, 2);
	assert.deepEqual(lines.map((l) => l.length), [2, 3]);
	close(lines[1].flat(), [1, 0, 0.5, 1.5, 0.5, 1.5, 2, 1, 2.5]);
	assert.deepEqual(polylinesOf(d, 'missing'), []);
	const pat = d.arrays.pattern;
	assert.ok(pat.data instanceof Int32Array);
	assert.deepEqual([pat.shape, Array.from(pat.data)], [[2, 2], [0, -1, -1, 1]]);
	const mask = d.arrays.mask;
	assert.ok(mask.data instanceof Uint8Array);
	assert.deepEqual(Array.from(mask.data), [1, 0, 1, 1, 0]);
});

test('unknown format versions are rejected', () => {
	const desc = JSON.parse(fs.readFileSync(path.join(DATA, 'golden_v1.json'), 'utf8'));
	assert.throws(() => parseFemResult({...desc, format: 'fem-result/2'}, new ArrayBuffer(512)), /unknown result format/);
});
