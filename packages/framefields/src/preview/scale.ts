import { Layer, type LayoutNode } from "../index.js";

/** The parts of a `Composition` that set its size. */
interface Sized {
	width?: unknown;
	height?: unknown;
	children?: unknown;
}

/**
 * Makes a composition draw smaller, for a lighter preview: its layers move
 * into one box, scaled to the new height, so positions, sizes and timing
 * stay as the code wrote them. The width follows the aspect ratio. A height
 * at or above the composition's own changes nothing. Returns whether it
 * changed the composition.
 */
export function scaleToHeight(target: unknown, height: number): boolean {
	const comp = target as Sized;
	if (
		typeof comp?.width !== "number" ||
		typeof comp.height !== "number" ||
		!Array.isArray(comp.children) ||
		!(height > 0) ||
		height >= comp.height
	) {
		return false;
	}
	const scale = height / comp.height;
	comp.children = [
		Layer.box({
			id: "framefields-preview-scale",
			position: "absolute",
			x: 0,
			y: 0,
			width: comp.width,
			height: comp.height,
			anchorX: 0,
			anchorY: 0,
			scale,
			overflow: "visible",
			children: comp.children as LayoutNode[],
		}),
	];
	comp.width = even(comp.width * scale);
	comp.height = even(height);
	return true;
}

/** Video sizes are even. */
function even(n: number): number {
	return Math.max(2, Math.round(n / 2) * 2);
}
