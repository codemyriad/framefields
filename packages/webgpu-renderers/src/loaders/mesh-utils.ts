import { type Vec3, Vector3Math } from "../math3d/index.js";
import type { LoadModelOptions } from "./types.js";

/**
 * Adds each triangle's (area-weighted) face normal to its three vertices, then
 * normalizes every vertex normal in place.
 */
export function accumulateVertexNormals(
	positions: ArrayLike<number>,
	indices: ArrayLike<number>,
	normals: { [i: number]: number; length: number },
): void {
	const vertex = (i: number): Vec3 => [
		positions[i * 3]!,
		positions[i * 3 + 1]!,
		positions[i * 3 + 2]!,
	];
	for (let i = 0; i < indices.length; i += 3) {
		const i0 = indices[i]!;
		const i1 = indices[i + 1]!;
		const i2 = indices[i + 2]!;
		const p0 = vertex(i0);
		const fn = Vector3Math.cross(
			Vector3Math.subtract(vertex(i1), p0),
			Vector3Math.subtract(vertex(i2), p0),
		);
		for (const v of [i0, i1, i2]) {
			normals[v * 3] += fn[0];
			normals[v * 3 + 1] += fn[1];
			normals[v * 3 + 2] += fn[2];
		}
	}

	for (let i = 0; i < normals.length; i += 3) {
		const nx = normals[i]!;
		const ny = normals[i + 1]!;
		const nz = normals[i + 2]!;
		const len = Math.hypot(nx, ny, nz) || 1.0;
		normals[i] = nx / len;
		normals[i + 1] = ny / len;
		normals[i + 2] = nz / len;
	}
}

export interface MeshBounds {
	min: Vec3;
	max: Vec3;
	center: Vec3;
	size: Vec3;
}

/** Axis-aligned bounds of xyz position arrays (all zero when empty). */
export function computeBounds(
	...positionArrays: ArrayLike<number>[]
): MeshBounds {
	let minX = Infinity,
		minY = Infinity,
		minZ = Infinity;
	let maxX = -Infinity,
		maxY = -Infinity,
		maxZ = -Infinity;
	for (const positions of positionArrays) {
		for (let i = 0; i < positions.length; i += 3) {
			const x = positions[i]!;
			const y = positions[i + 1]!;
			const z = positions[i + 2]!;
			if (x < minX) minX = x;
			if (x > maxX) maxX = x;
			if (y < minY) minY = y;
			if (y > maxY) maxY = y;
			if (z < minZ) minZ = z;
			if (z > maxZ) maxZ = z;
		}
	}
	if (!Number.isFinite(minX)) {
		minX = minY = minZ = 0;
		maxX = maxY = maxZ = 0;
	}
	return {
		min: [minX, minY, minZ],
		max: [maxX, maxY, maxZ],
		center: [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5],
		size: [maxX - minX, maxY - minY, maxZ - minZ],
	};
}

export interface ModelTransform {
	offset: Vec3;
	scale: Vec3;
}

/**
 * The placement `LoadModelOptions` asks for: recentring on the bounds
 * (`center`), fitting the largest side to `normalizeSize`, then `scale`.
 */
export function resolveModelTransform(
	bounds: MeshBounds,
	options: LoadModelOptions,
): ModelTransform {
	const maxDimension = Math.max(...bounds.size, 0.0001);
	let scaleMultiplier = 1.0;
	if (options.normalizeSize && options.normalizeSize > 0) {
		scaleMultiplier = options.normalizeSize / maxDimension;
	}

	const scale: Vec3 = [scaleMultiplier, scaleMultiplier, scaleMultiplier];
	if (typeof options.scale === "number") {
		scale[0] *= options.scale;
		scale[1] *= options.scale;
		scale[2] *= options.scale;
	} else if (Array.isArray(options.scale)) {
		scale[0] *= options.scale[0];
		scale[1] *= options.scale[1];
		scale[2] *= options.scale[2];
	}

	return { offset: options.center ? bounds.center : [0, 0, 0], scale };
}

/** `(p - offset) * scale` for every xyz position. */
export function transformPositions(
	positions: ArrayLike<number>,
	{ offset, scale }: ModelTransform,
): Float32Array {
	const out = new Float32Array(positions.length);
	for (let i = 0; i < positions.length; i += 3) {
		out[i] = (positions[i]! - offset[0]) * scale[0];
		out[i + 1] = (positions[i + 1]! - offset[1]) * scale[1];
		out[i + 2] = (positions[i + 2]! - offset[2]) * scale[2];
	}
	return out;
}

/** Uint16 indices while every vertex fits, Uint32 beyond 65535 vertices. */
export function createIndexArray(
	indices: number[],
	vertexCount: number,
): Uint16Array | Uint32Array {
	return vertexCount > 65535
		? new Uint32Array(indices)
		: new Uint16Array(indices);
}

/** The model's bounds after `transform`, as `Model3DData.bounds` reports them. */
export function transformModelBounds(
	bounds: MeshBounds,
	{ offset, scale }: ModelTransform,
): MeshBounds & { boundingSphereRadius: number } {
	const min: Vec3 = [
		(bounds.min[0] - offset[0]) * scale[0],
		(bounds.min[1] - offset[1]) * scale[1],
		(bounds.min[2] - offset[2]) * scale[2],
	];
	const max: Vec3 = [
		(bounds.max[0] - offset[0]) * scale[0],
		(bounds.max[1] - offset[1]) * scale[1],
		(bounds.max[2] - offset[2]) * scale[2],
	];
	return {
		min,
		max,
		center: [
			(min[0] + max[0]) * 0.5,
			(min[1] + max[1]) * 0.5,
			(min[2] + max[2]) * 0.5,
		],
		size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
		boundingSphereRadius:
			Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) * 0.5,
	};
}
