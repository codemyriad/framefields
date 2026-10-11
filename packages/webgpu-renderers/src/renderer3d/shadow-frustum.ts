/**
 * Fits a shadow-casting light's view-projection around the scene: tight on
 * the casters (so the map's texels go where shadows are), deep enough to
 * cover every receiver (a floor far bigger than the casters still darkens).
 */
import { type Mat4, Matrix4Math } from "../math3d/matrix4.js";
import { type Vec3, Vector3Math } from "../math3d/vector3.js";

export interface Aabb {
	min: Vec3;
	max: Vec3;
}

export interface ShadowLight {
	type: "directional" | "spot";
	/** World position (spot); directional lights only use `direction`. */
	position: Vec3;
	/** Unit direction the light travels. */
	direction: Vec3;
	/** Spot cone half-angle in degrees. */
	angle?: number;
}

export interface ShadowFrustum {
	viewProj: Mat4;
	/** World px covered by one shadow-map texel (at the casters, for spots). */
	texelWorldSize: number;
}

export function aabbCorners(box: Aabb): Vec3[] {
	const { min, max } = box;
	return [
		[min[0], min[1], min[2]],
		[max[0], min[1], min[2]],
		[min[0], max[1], min[2]],
		[max[0], max[1], min[2]],
		[min[0], min[1], max[2]],
		[max[0], min[1], max[2]],
		[min[0], max[1], max[2]],
		[max[0], max[1], max[2]],
	];
}

/** World-space bounds of a local box under a column-major model matrix. */
export function transformAabb(box: Aabb, m: ArrayLike<number>): Aabb {
	const min: Vec3 = [Infinity, Infinity, Infinity];
	const max: Vec3 = [-Infinity, -Infinity, -Infinity];
	for (const [x, y, z] of aabbCorners(box)) {
		const w = [
			m[0] * x + m[4] * y + m[8] * z + m[12],
			m[1] * x + m[5] * y + m[9] * z + m[13],
			m[2] * x + m[6] * y + m[10] * z + m[14],
		];
		for (let i = 0; i < 3; i++) {
			if (w[i] < min[i]) min[i] = w[i];
			if (w[i] > max[i]) max[i] = w[i];
		}
	}
	return { min, max };
}

function transformPoint(m: Mat4, p: Vec3): Vec3 {
	return [
		m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
		m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
		m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
	];
}

/** An up vector not parallel to `dir` (world y points down). */
function upFor(dir: Vec3): Vec3 {
	return Math.abs(dir[1]) > 0.98 ? [0, 0, 1] : [0, -1, 0];
}

export function fitShadowFrustum(
	light: ShadowLight,
	casters: Aabb[],
	receivers: Aabb[],
	mapSize: number,
): ShadowFrustum | null {
	if (casters.length === 0) return null;
	const dir = Vector3Math.normalize(light.direction);
	const casterPts = casters.flatMap(aabbCorners);
	const allPts = [...casterPts, ...receivers.flatMap(aabbCorners)];

	if (light.type === "directional") {
		const center: Vec3 = [0, 0, 0];
		for (const p of casterPts) {
			center[0] += p[0] / casterPts.length;
			center[1] += p[1] / casterPts.length;
			center[2] += p[2] / casterPts.length;
		}
		const eye = Vector3Math.subtract(
			center,
			Vector3Math.multiplyScalar(dir, 1),
		);
		const view = Matrix4Math.lookAt(eye, center, upFor(dir));
		let minX = Infinity;
		let maxX = -Infinity;
		let minY = Infinity;
		let maxY = -Infinity;
		for (const p of casterPts) {
			const v = transformPoint(view, p);
			minX = Math.min(minX, v[0]);
			maxX = Math.max(maxX, v[0]);
			minY = Math.min(minY, v[1]);
			maxY = Math.max(maxY, v[1]);
		}
		let minZ = Infinity;
		let maxZ = -Infinity;
		for (const p of allPts) {
			const z = transformPoint(view, p)[2];
			minZ = Math.min(minZ, z);
			maxZ = Math.max(maxZ, z);
		}
		// Room for blurred edges, and square texels.
		const half = Math.max(maxX - minX, maxY - minY) * 0.5 * 1.08 + 4;
		const cx = (minX + maxX) / 2;
		const cy = (minY + maxY) / 2;
		const depthPad = Math.max(10, (maxZ - minZ) * 0.02);
		const proj = Matrix4Math.orthographic(
			cx - half,
			cx + half,
			cy - half,
			cy + half,
			minZ - depthPad,
			maxZ + depthPad,
		);
		return {
			viewProj: Matrix4Math.multiply(proj, view),
			texelWorldSize: (2 * half) / mapSize,
		};
	}

	const eye = light.position;
	const view = Matrix4Math.lookAt(eye, Vector3Math.add(eye, dir), upFor(dir));
	let near = Infinity;
	let far = 0;
	let casterDist = 0;
	for (const p of allPts) {
		const z = transformPoint(view, p)[2];
		if (z > 0) {
			near = Math.min(near, z);
			far = Math.max(far, z);
		}
	}
	for (const p of casterPts) {
		casterDist = Math.max(casterDist, transformPoint(view, p)[2]);
	}
	if (!Number.isFinite(near) || far <= 0) return null;
	near = Math.max(1, near * 0.9);
	far = far * 1.05 + 1;
	const fovRad = (Math.min(170, 2 * (light.angle ?? 45) + 4) * Math.PI) / 180;
	const proj = Matrix4Math.perspective(fovRad, 1, near, far);
	return {
		viewProj: Matrix4Math.multiply(proj, view),
		texelWorldSize:
			(2 * Math.tan(fovRad / 2) * Math.max(casterDist, near)) / mapSize,
	};
}
