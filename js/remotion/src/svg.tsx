// Clean line art (SVG): mesh lines, boundaries, polylines (stress trajectories, contours), colour bars, stiffness-matrix grids.
import React, {useMemo} from 'react';
import {boundaryLoops, edges, makeColormap, projAt, rgbCss, type FemData, type Projector, type Stops} from '@femotion/core';

const svgBox: React.CSSProperties = {position: 'absolute', inset: 0, overflow: 'visible'};

export type MeshLinesProps = {
	d: FemData;
	proj: Projector;
	pts?: ArrayLike<number>;
	color?: string;
	width?: number;
	/** outline and hole edges drawn heavier; null: not drawn separately */
	boundaryColor?: string | null;
	boundaryWidth?: number;
	/** cell reveal order and progress: only edges with rank < upto; the fractional part fades in */
	order?: ArrayLike<number>;
	upto?: number;
	opacity?: number;
	/** outline only */
	outlineOnly?: boolean;
	dash?: string;
};

export const MeshLines: React.FC<MeshLinesProps> = ({d, proj, pts = d.points, color = '#15294A', width = 1, boundaryColor, boundaryWidth = 2.4, order, upto = Infinity, opacity = 1, outlineOnly = false, dash}) => {
	const es = useMemo(() => edges(d, order), [d, order]);
	const loops = useMemo(() => (d.dim === 2 ? boundaryLoops(d) : []), [d]);
	const P = (i: number) => projAt(proj, pts, d.dim, i);
	const bc = boundaryColor === undefined ? color : boundaryColor;
	return (
		<svg width="100%" height="100%" style={{...svgBox, opacity}}>
			{!outlineOnly &&
				es.map(([a, b, r]) => {
					if (r >= upto) return null;
					const [x1, y1] = P(a);
					const [x2, y2] = P(b);
					return <line key={`${a}-${b}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={width} strokeLinecap="round" opacity={Math.min(1, upto - r)} strokeDasharray={dash} />;
				})}
			{bc &&
				loops.map((l, i) => (
					<polygon key={i} points={l.map((n) => P(n).join(',')).join(' ')} fill="none" stroke={bc} strokeWidth={boundaryWidth} strokeLinejoin="round" />
				))}
		</svg>
	);
};

export type PolylinesProps = {
	/** model-coordinate points of each polyline ([x, y] or [x, y, z]) */
	lines: number[][][];
	proj: Projector;
	color?: string | ((i: number) => string);
	width?: number | ((i: number) => number);
	/** 0..1: drawn fraction along each line */
	progress?: number;
	opacity?: number;
	dash?: string;
	arrow?: boolean;
};

export const Polylines: React.FC<PolylinesProps> = ({lines, proj, color = '#E8742A', width = 2, progress = 1, opacity = 1, dash, arrow = false}) => (
	<svg width="100%" height="100%" style={{...svgBox, opacity}}>
		{lines.map((l, i) => {
			const pts = l.map((p) => proj(p));
			const n = Math.max(2, Math.ceil(pts.length * progress));
			const vis = pts.slice(0, n);
			const c = typeof color === 'function' ? color(i) : color;
			const w = typeof width === 'function' ? width(i) : width;
			const end = vis[vis.length - 1];
			const prev = vis[Math.max(0, vis.length - 4)];
			const ang = Math.atan2(end[1] - prev[1], end[0] - prev[0]);
			return (
				<g key={i}>
					<polyline points={vis.map((p) => p.join(',')).join(' ')} fill="none" stroke={c} strokeWidth={w} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dash} />
					{arrow && progress > 0.05 && (
						<path d={`M ${end[0]} ${end[1]} l ${-9 * Math.cos(ang - 0.45)} ${-9 * Math.sin(ang - 0.45)} M ${end[0]} ${end[1]} l ${-9 * Math.cos(ang + 0.45)} ${-9 * Math.sin(ang + 0.45)}`} stroke={c} strokeWidth={w} strokeLinecap="round" />
					)}
				</g>
			);
		})}
	</svg>
);

export type ColorBarProps = {
	x: number;
	y: number;
	w: number;
	h: number;
	stops: Stops;
	/** values at both ends and the unit (labels) */
	lo: string;
	hi: string;
	mid?: string;
	label?: string;
	bands?: number;
	ink?: string;
	font?: string;
	vertical?: boolean;
};

export const ColorBar: React.FC<ColorBarProps> = ({x, y, w, h, stops, lo, hi, mid, label, bands = 0, ink = '#15294A', font = '"Noto Sans SC", sans-serif', vertical = false}) => {
	const cm = makeColormap(stops);
	const n = bands || 48;
	const cells = Array.from({length: n}, (_, i) => {
		const t = bands ? i / Math.max(1, n - 1) : (i + 0.5) / n;
		return vertical ? {x, y: y + h - ((i + 1) * h) / n, w, h: h / n + 0.5, c: rgbCss(cm(t))} : {x: x + (i * w) / n, y, w: w / n + 0.5, h, c: rgbCss(cm(t))};
	});
	const fs = 22;
	return (
		<svg width="100%" height="100%" style={svgBox}>
			{cells.map((c, i) => (
				<rect key={i} x={c.x} y={c.y} width={c.w} height={c.h} fill={c.c} />
			))}
			<rect x={x} y={y} width={w} height={h} fill="none" stroke={ink} strokeWidth={1.6} rx={2} />
			{label && (
				<text x={x} y={y - 12} fill={ink} fontSize={fs} fontFamily={font} fontWeight={600}>
					{label}
				</text>
			)}
			{vertical ? (
				<>
					<text x={x + w + 10} y={y + h} fill={ink} fontSize={fs - 2} fontFamily={font} dominantBaseline="middle">{lo}</text>
					<text x={x + w + 10} y={y} fill={ink} fontSize={fs - 2} fontFamily={font} dominantBaseline="middle">{hi}</text>
					{mid && <text x={x + w + 10} y={y + h / 2} fill={ink} fontSize={fs - 2} fontFamily={font} dominantBaseline="middle">{mid}</text>}
				</>
			) : (
				<>
					<text x={x} y={y + h + fs + 4} fill={ink} fontSize={fs - 2} fontFamily={font}>{lo}</text>
					<text x={x + w} y={y + h + fs + 4} fill={ink} fontSize={fs - 2} fontFamily={font} textAnchor="end">{hi}</text>
					{mid && <text x={x + w / 2} y={y + h + fs + 4} fill={ink} fontSize={fs - 2} fontFamily={font} textAnchor="middle">{mid}</text>}
				</>
			)}
		</svg>
	);
};

export type MatrixGridProps = {
	/** (G, G) flattened: -1 = always zero; otherwise the rank of the first element filling that block (femotion.post.assembly.stiffness_pattern) */
	pattern: ArrayLike<number>;
	G: number;
	x: number;
	y: number;
	size: number;
	/** number of elements assembled so far (the fractional part fades the current one in) */
	upto: number;
	ink?: string;
	fresh?: string;
	/** how many elements a freshly filled block stays highlighted */
	freshSpan?: number;
	frame?: boolean;
};

export const MatrixGrid: React.FC<MatrixGridProps> = ({pattern, G, x, y, size, upto, ink = '#15294A', fresh = '#E8742A', freshSpan = 3, frame = true}) => {
	const c = size / G;
	const rects: React.ReactNode[] = [];
	for (let i = 0; i < G; i++)
		for (let j = 0; j < G; j++) {
			const r = pattern[i * G + j];
			if (r < 0 || r >= upto) continue;
			const isFresh = upto - r <= freshSpan;
			rects.push(<rect key={i * G + j} x={x + j * c} y={y + i * c} width={c + 0.3} height={c + 0.3} fill={isFresh ? fresh : ink} opacity={Math.min(1, upto - r) * (isFresh ? 1 : 0.85)} />);
		}
	return (
		<svg width="100%" height="100%" style={svgBox}>
			{rects}
			{frame && <rect x={x} y={y} width={size} height={size} fill="none" stroke={ink} strokeWidth={2} />}
		</svg>
	);
};
