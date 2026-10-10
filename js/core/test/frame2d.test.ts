// The live 2D frame solver against textbook closed-form solutions.
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {addMember, solve, type FemModel} from '../src/index.ts';

const check = (name: string, got: number, want: number, tol = 1e-6) => {
	const ok = want === 0 ? Math.abs(got) <= tol : Math.abs(got - want) <= tol * Math.abs(want);
	assert.ok(ok, `${name}: got ${got.toExponential(5)}, want ${want.toExponential(5)}`);
};

const E = 2.06e11, A = 0.6, I = 1.2, alpha = 1.2e-5, L = 40;

test('cantilever with a tip load: deflection PL³/3EI, rotation PL²/2EI, fixed-end moment −PL', () => {
	const m: FemModel = {nodes: [{x: 0, y: 0}, {x: L, y: 0}], elements: [], supports: [{node: 0, ux: true, uy: true, rz: true}], loads: [{node: 1, fy: -1e5}]};
	addMember(m, 0, 1, 8, {E, A, I});
	const r = solve(m);
	check('tip deflection', r.disp[1].uy, -1e5 * L ** 3 / (3 * E * I));
	check('tip rotation', r.disp[1].rz, -1e5 * L ** 2 / (2 * E * I));
	check('fixed-end moment', r.forces[0].M1, -1e5 * L);
});
test('fixed-fixed beam heated by 50 °C: no displacement, axial force −EAαΔT', () => {
	const m: FemModel = {nodes: [{x: 0, y: 0}, {x: L, y: 0}], elements: [], supports: [{node: 0, ux: true, uy: true, rz: true}, {node: 1, ux: true, uy: true, rz: true}]};
	addMember(m, 0, 1, 4, {E, A, I, alpha, dT: 50});
	const r = solve(m);
	check('axial force', r.forces[2].N1, -E * A * alpha * 50);
	check('stress MPa', r.forces[2].N1 / A / 1e6, -123.6, 1e-3);
	check('mid-span displacement', Math.hypot(r.disp[3].ux, r.disp[3].uy), 0, 1e-9);
});
test('simply supported beam (pin + roller) heated: free elongation αΔTL, no axial force', () => {
	const m: FemModel = {nodes: [{x: 0, y: 0}, {x: 1000, y: 0}], elements: [], supports: [{node: 0, ux: true, uy: true}, {node: 1, uy: true}]};
	addMember(m, 0, 1, 10, {E, A, I, alpha, dT: 50});
	const r = solve(m);
	check('elongation of 1 km (m)', r.disp[1].ux, alpha * 1000 * 50);
	check('axial force', r.forces[0].N1, 0, 1e-6);
});
test('simply supported beam with a temperature gradient: mid-span deflection κL²/8', () => {
	const h = 2.5, dTg = 10;
	const m: FemModel = {nodes: [{x: 0, y: 0}, {x: L, y: 0}], elements: [], supports: [{node: 0, ux: true, uy: true}, {node: 1, uy: true}]};
	addMember(m, 0, 1, 8, {E, A, I, alpha, dTgrad: dTg, h});
	const r = solve(m);
	const kappa = alpha * dTg / h;
	// a hotter bottom lengthens the bottom fibre → the beam sags (∪), so mid-span moves down
	check('mid-span deflection', r.disp[5].uy, -kappa * L ** 2 / 8);
});
test('portal frame with fixed feet and a (nearly) rigid beam under a lateral load: sway P h³ / (24 E I_col)', () => {
	const h = 10, Ic = 0.5;
	const m: FemModel = {nodes: [{x: 0, y: 0}, {x: 0, y: h}, {x: 20, y: h}, {x: 20, y: 0}], elements: [], supports: [{node: 0, ux: true, uy: true, rz: true}, {node: 3, ux: true, uy: true, rz: true}], loads: [{node: 1, fx: 1e6}]};
	addMember(m, 0, 1, 6, {E, A: 1e4, I: Ic});
	addMember(m, 1, 2, 6, {E, A: 1e4, I: 1e6});
	addMember(m, 2, 3, 6, {E, A: 1e4, I: Ic});
	const r = solve(m);
	check('sway', r.disp[1].ux, 1e6 * h ** 3 / (24 * E * Ic), 2e-3);
});
