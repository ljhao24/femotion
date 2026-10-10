// Colour maps for fields: piecewise-linear ramps from hex stops, value normalisation, banding.
// Palettes are the application's choice; pass them as Stops.
export type Stops = [number, string][];
export type RGB = [number, number, number];

const hex = (h: string): RGB => {
	const v = parseInt(h.replace('#', ''), 16);
	return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
};

/** stops have t in 0..1; returns t → [r, g, b] (0–255) */
export const makeColormap = (stops: Stops) => {
	const s = stops.map(([t, c]) => [t, hex(c)] as [number, RGB]);
	return (t: number): RGB => {
		if (!(t > s[0][0])) return s[0][1];
		for (let i = 1; i < s.length; i++) {
			if (t <= s[i][0]) {
				const k = (t - s[i - 1][0]) / (s[i][0] - s[i - 1][0]);
				const a = s[i - 1][1];
				const b = s[i][1];
				return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
			}
		}
		return s[s.length - 1][1];
	};
};

export const rgbCss = ([r, g, b]: RGB, a = 1) => `rgba(${Math.round(r)},${Math.round(g)},${Math.round(b)},${a})`;

/** map values to 0..1; signed: 0 maps to 0.5, symmetric about zero using the larger absolute end */
export const normalizer = (range: [number, number], signed = false) => {
	if (signed) {
		const m = Math.max(Math.abs(range[0]), Math.abs(range[1])) || 1;
		return (v: number) => 0.5 + v / (2 * m);
	}
	const span = range[1] - range[0] || 1;
	return (v: number) => (v - range[0]) / span;
};

/** banding: quantise 0..1 into n levels (flat colour bands look more hand-painted than smooth gradients) */
export const banded = (n: number) => (t: number) => (n > 0 ? Math.min(n - 1, Math.floor(Math.max(0, t) * n)) / Math.max(1, n - 1) : t);
