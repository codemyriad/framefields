import type { Material3D, Mesh3DData, Model3DData } from "../loaders/types.js";
import { parseHexOrRgbColor } from "./text-extruder.js";

/**
 * Procedural solids built as real meshes (one draw call each), centered on
 * the origin in the 3D world's convention: x right, y down, z away from a
 * default camera. "Top" is therefore -y.
 */
interface PrimitiveSurface {
	/** Diffuse color (hex or rgb()). Default white, so a node `color` tints it. */
	color?: string;
	material?: "lit" | "unlit" | "toon";
	shininess?: number;
	roughness?: number;
	metallic?: number;
}

export interface RoundedBoxGeometryOptions extends PrimitiveSurface {
	type: "roundedBox";
	width: number;
	height: number;
	depth: number;
	/** Corner and edge radius, clamped to half the smallest side. Default 0 (a sharp box). */
	radius?: number;
	/** Segments per rounded quarter (default 4). */
	segments?: number;
	/**
	 * Scale of the top (-y) face against the bottom along x and z: 0.9 draws
	 * a keycap-like frustum. Default 1.
	 */
	taper?: number;
}

export interface SphereGeometryOptions extends PrimitiveSurface {
	type: "sphere";
	radius: number;
	/** Segments around the equator (default 32); rings are half of it. */
	segments?: number;
}

export interface CylinderGeometryOptions extends PrimitiveSurface {
	type: "cylinder";
	/** Radius of the top (-y) cap; defaults to `radius`. */
	radiusTop?: number;
	/** Radius of the bottom (+y) cap; defaults to `radius`. */
	radiusBottom?: number;
	radius?: number;
	height: number;
	/** Segments around (default 32). */
	segments?: number;
	/** Leave the caps off (a tube). */
	openEnded?: boolean;
}

export type PrimitiveGeometryOptions =
	| RoundedBoxGeometryOptions
	| SphereGeometryOptions
	| CylinderGeometryOptions;

class MeshBuilder {
	positions: number[] = [];
	normals: number[] = [];
	uvs: number[] = [];
	indices: number[] = [];

	vertex(p: number[], n: number[], uv: [number, number]): number {
		this.positions.push(p[0]!, p[1]!, p[2]!);
		this.normals.push(n[0]!, n[1]!, n[2]!);
		this.uvs.push(uv[0], uv[1]);
		return this.positions.length / 3 - 1;
	}

	/**
	 * Joins a (cols+1) x (rows+1) vertex grid that starts at `base` into
	 * quads. Front faces wind so that cross(row step, column step) points
	 * out; `flip` reverses that for grids laid out the other way round.
	 */
	grid(base: number, cols: number, rows: number, flip = false): void {
		for (let j = 0; j < rows; j++) {
			for (let i = 0; i < cols; i++) {
				const a = base + j * (cols + 1) + i;
				const b = a + 1;
				const c = a + cols + 1;
				const d = c + 1;
				if (flip) this.indices.push(a, b, c, b, d, c);
				else this.indices.push(a, c, b, b, c, d);
			}
		}
	}

	build(name: string): Mesh3DData {
		const count = this.positions.length / 3;
		return {
			id: `${name}-mesh`,
			name,
			positions: new Float32Array(this.positions),
			normals: new Float32Array(this.normals),
			uvs: new Float32Array(this.uvs),
			indices:
				count > 65535
					? new Uint32Array(this.indices)
					: new Uint16Array(this.indices),
			materialName: `${name}-material`,
		};
	}
}

function normalize(v: number[]): number[] {
	const len = Math.hypot(v[0]!, v[1]!, v[2]!);
	return len > 1e-12 ? [v[0]! / len, v[1]! / len, v[2]! / len] : [0, 0, 0];
}

/**
 * 1D samples across a side of half-extent `e` whose ends round off with
 * radius `r`: arc-spaced through each rounded zone, nothing in between, so
 * the flat middle of a face is a single quad.
 */
function edgeSamples(e: number, r: number, segments: number): number[] {
	if (r <= 0) return [-e, e];
	const inner = e - r;
	const out: number[] = [];
	for (let i = 0; i <= segments; i++) {
		out.push(-inner - r * Math.cos(((i / segments) * Math.PI) / 2));
	}
	for (let i = inner > 1e-9 ? 0 : 1; i <= segments; i++) {
		out.push(inner + r * Math.sin(((i / segments) * Math.PI) / 2));
	}
	return out;
}

function roundedBox(o: RoundedBoxGeometryOptions, mb: MeshBuilder): void {
	const hx = o.width / 2;
	const hy = o.height / 2;
	const hz = o.depth / 2;
	const r = Math.max(0, Math.min(o.radius ?? 0, hx, hy, hz));
	const seg = Math.max(1, Math.round(o.segments ?? 4));
	const taper = o.taper ?? 1;
	const ix = hx - r;
	const iy = hy - r;
	const iz = hz - r;
	const sx = edgeSamples(hx, r, seg);
	const sy = edgeSamples(hy, r, seg);
	const sz = edgeSamples(hz, r, seg);

	// Taper scales x and z linearly from 1 at the bottom (+y) to `taper` at
	// the top (-y); normals follow through the inverse transpose of that map.
	const slope = o.height > 0 ? (1 - taper) / o.height : 0;
	const scaleAt = (y: number) => taper + slope * (y + hy);
	const place = (surface: number[], faceNormal: number[]) => {
		const inner = [
			Math.max(-ix, Math.min(ix, surface[0]!)),
			Math.max(-iy, Math.min(iy, surface[1]!)),
			Math.max(-iz, Math.min(iz, surface[2]!)),
		];
		let n =
			r > 0
				? normalize([
						surface[0]! - inner[0]!,
						surface[1]! - inner[1]!,
						surface[2]! - inner[2]!,
					])
				: faceNormal;
		if (n[0] === 0 && n[1] === 0 && n[2] === 0) n = faceNormal;
		const p = [
			inner[0]! + n[0]! * r,
			inner[1]! + n[1]! * r,
			inner[2]! + n[2]! * r,
		];
		if (taper === 1) return { p, n };
		const s = scaleAt(p[1]!);
		return {
			p: [p[0]! * s, p[1]!, p[2]! * s],
			n: normalize([
				n[0]! / s,
				n[1]! - (p[0]! * slope * n[0]!) / s - (p[2]! * slope * n[2]!) / s,
				n[2]! / s,
			]),
		};
	};

	// Each face: a fixed axis at ±extent, and two in-plane axes with samples.
	const faces: Array<{
		axis: 0 | 1 | 2;
		sign: 1 | -1;
		u: 0 | 1 | 2;
		v: 0 | 1 | 2;
	}> = [
		{ axis: 2, sign: -1, u: 0, v: 1 }, // front (-z)
		{ axis: 2, sign: 1, u: 0, v: 1 }, // back (+z)
		{ axis: 0, sign: -1, u: 2, v: 1 }, // left
		{ axis: 0, sign: 1, u: 2, v: 1 }, // right
		{ axis: 1, sign: -1, u: 0, v: 2 }, // top (-y)
		{ axis: 1, sign: 1, u: 0, v: 2 }, // bottom
	];
	const half = [hx, hy, hz];
	const samples = [sx, sy, sz];
	for (const f of faces) {
		const us = samples[f.u]!;
		const vs = samples[f.v]!;
		const faceNormal = [0, 0, 0];
		faceNormal[f.axis] = f.sign;
		const base = mb.positions.length / 3;
		for (let j = 0; j < vs.length; j++) {
			for (let i = 0; i < us.length; i++) {
				const surface = [0, 0, 0];
				surface[f.axis] = f.sign * half[f.axis]!;
				surface[f.u] = us[i]!;
				surface[f.v] = vs[j]!;
				const { p, n } = place(surface, faceNormal);
				mb.vertex(p, n, [
					(us[i]! / half[f.u]! + 1) / 2,
					(vs[j]! / half[f.v]! + 1) / 2,
				]);
			}
		}
		// cross(v axis, u axis) is +axis when (v, u) runs x→y→z→x; outward
		// needs it along the face normal, so the other faces wind the other way.
		const cyclic = (f.v + 1) % 3 === f.u;
		mb.grid(base, us.length - 1, vs.length - 1, (cyclic ? 1 : -1) !== f.sign);
	}
}

function sphere(o: SphereGeometryOptions, mb: MeshBuilder): void {
	const seg = Math.max(3, Math.round(o.segments ?? 32));
	const rings = Math.max(2, Math.round(seg / 2));
	for (let j = 0; j <= rings; j++) {
		const phi = (j / rings) * Math.PI;
		for (let i = 0; i <= seg; i++) {
			const theta = (i / seg) * Math.PI * 2;
			const n = [
				Math.sin(phi) * Math.cos(theta),
				-Math.cos(phi),
				Math.sin(phi) * Math.sin(theta),
			];
			mb.vertex([n[0]! * o.radius, n[1]! * o.radius, n[2]! * o.radius], n, [
				i / seg,
				j / rings,
			]);
		}
	}
	mb.grid(0, seg, rings);
}

function cylinder(o: CylinderGeometryOptions, mb: MeshBuilder): void {
	const seg = Math.max(3, Math.round(o.segments ?? 32));
	const rTop = o.radiusTop ?? o.radius ?? 50;
	const rBottom = o.radiusBottom ?? o.radius ?? 50;
	const hy = o.height / 2;
	// Side normals lean by the cone's slope.
	const lean = o.height > 0 ? (rBottom - rTop) / o.height : 0;
	for (let j = 0; j <= 1; j++) {
		const rr = j === 0 ? rTop : rBottom;
		const y = j === 0 ? -hy : hy;
		for (let i = 0; i <= seg; i++) {
			const theta = (i / seg) * Math.PI * 2;
			const c = Math.cos(theta);
			const s = Math.sin(theta);
			mb.vertex([c * rr, y, s * rr], normalize([c, -lean, s]), [i / seg, j]);
		}
	}
	mb.grid(0, seg, 1);
	if (o.openEnded) return;
	for (const [y, rr, ny] of [
		[-hy, rTop, -1],
		[hy, rBottom, 1],
	] as const) {
		if (rr <= 0) continue;
		const center = mb.vertex([0, y, 0], [0, ny, 0], [0.5, 0.5]);
		const ring = mb.positions.length / 3;
		for (let i = 0; i <= seg; i++) {
			const theta = (i / seg) * Math.PI * 2;
			const c = Math.cos(theta);
			const s = Math.sin(theta);
			mb.vertex([c * rr, y, s * rr], [0, ny, 0], [(c + 1) / 2, (s + 1) / 2]);
		}
		for (let i = 0; i < seg; i++) {
			// Wound to face -y on the top cap; the bottom cap turns over.
			if (ny < 0) mb.indices.push(center, ring + i, ring + i + 1);
			else mb.indices.push(center, ring + i + 1, ring + i);
		}
	}
}

/** Builds a procedural solid as a single-mesh Model3DData. */
export function generatePrimitiveGeometry(
	options: PrimitiveGeometryOptions,
): Model3DData {
	const mb = new MeshBuilder();
	if (options.type === "roundedBox") roundedBox(options, mb);
	else if (options.type === "sphere") sphere(options, mb);
	else if (options.type === "cylinder") cylinder(options, mb);
	else {
		throw new Error(
			`Unknown 3D primitive: ${(options as { type?: string }).type}`,
		);
	}

	const name = options.type;
	const mesh = mb.build(name);
	const material: Material3D = {
		name: `${name}-material`,
		diffuseColor: parseHexOrRgbColor(options.color ?? "#ffffff"),
		shading: options.material ?? "lit",
		shininess: options.shininess ?? 32,
		roughness: options.roughness,
		metallic: options.metallic,
	};

	const min: [number, number, number] = [
		Number.POSITIVE_INFINITY,
		Number.POSITIVE_INFINITY,
		Number.POSITIVE_INFINITY,
	];
	const max: [number, number, number] = [
		Number.NEGATIVE_INFINITY,
		Number.NEGATIVE_INFINITY,
		Number.NEGATIVE_INFINITY,
	];
	const pos = mesh.positions;
	for (let i = 0; i < pos.length; i += 3) {
		for (let k = 0; k < 3; k++) {
			const v = pos[i + k]!;
			if (v < min[k]!) min[k] = v;
			if (v > max[k]!) max[k] = v;
		}
	}
	const size: [number, number, number] = [
		max[0] - min[0],
		max[1] - min[1],
		max[2] - min[2],
	];
	return {
		name: `primitive-${name}`,
		meshes: [mesh],
		materials: { [material.name]: material },
		animations: [],
		bounds: {
			min,
			max,
			center: [
				(min[0] + max[0]) / 2,
				(min[1] + max[1]) / 2,
				(min[2] + max[2]) / 2,
			],
			size,
			boundingSphereRadius: Math.hypot(...size) / 2,
		},
	};
}

const primitiveCache = new Map<string, Model3DData>();

/**
 * Like generatePrimitiveGeometry, but identical options return the same
 * Model3DData, so a hundred equal keycaps upload one vertex buffer.
 */
export function getPrimitiveGeometry(
	options: PrimitiveGeometryOptions,
): Model3DData {
	const key = JSON.stringify(options);
	let model = primitiveCache.get(key);
	if (!model) {
		model = generatePrimitiveGeometry(options);
		if (primitiveCache.size > 512) primitiveCache.clear();
		primitiveCache.set(key, model);
	}
	return model;
}
