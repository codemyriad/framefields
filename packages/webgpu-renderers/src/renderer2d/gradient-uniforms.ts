import {
	gradientStopsToOklab,
	limitGradientStops,
	type ParsedGradient,
	type ParsedGradientStop,
	parseCssGradient,
} from "@framefields/core";
import { GRADIENT_UNIFORM_FLOATS } from "../shaders/gradient.js";

export interface GradientBox {
	x: number;
	y: number;
	width: number;
	height: number;
}

const GRADIENT_TYPE_CODE = { linear: 1, radial: 2, conic: 3 } as const;

/** The parsed gradient when `value` is a CSS gradient string. */
export function asGradient(value: unknown): ParsedGradient | null {
	return typeof value === "string" && value.includes("gradient(")
		? parseCssGradient(value)
		: null;
}

/**
 * Packs a gradient laid over `box` (local px) into `out` at `offset`
 * (`GRADIENT_UNIFORM_FLOATS` floats). Pass `null` to clear (solid fill).
 */
export function packGradientUniforms(
	gradient: ParsedGradient | null,
	box: GradientBox,
	out: Float32Array,
	offset: number,
): void {
	out.fill(0, offset, offset + GRADIENT_UNIFORM_FLOATS);
	if (!gradient) return;
	const stops: ParsedGradientStop[] = limitGradientStops(gradient.stops);
	const w = Math.max(box.width, 1e-3);
	const h = Math.max(box.height, 1e-3);
	const cx = box.x + gradient.center[0] * w;
	const cy = box.y + gradient.center[1] * h;
	out[offset] = GRADIENT_TYPE_CODE[gradient.type];
	out[offset + 1] = stops.length;
	out[offset + 4] = gradient.type === "linear" ? box.x + w / 2 : cx;
	out[offset + 5] = gradient.type === "linear" ? box.y + h / 2 : cy;
	if (gradient.type === "linear") {
		// CSS: 0deg points up, 90deg right; the line spans corner to corner.
		const r = (gradient.angle * Math.PI) / 180;
		const dx = Math.sin(r);
		const dy = -Math.cos(r);
		const len = Math.abs(w * dx) + Math.abs(h * dy);
		out[offset + 6] = dx / len;
		out[offset + 7] = dy / len;
	} else if (gradient.type === "radial") {
		let rx: number;
		let ry: number;
		if (gradient.radius) {
			[rx, ry] = gradient.radius;
		} else {
			// farthest-corner, as CSS sizes radial gradients by default.
			const fx = Math.max(cx - box.x, box.x + w - cx);
			const fy = Math.max(cy - box.y, box.y + h - cy);
			if (gradient.shape === "circle") {
				rx = ry = Math.hypot(fx, fy);
			} else {
				rx = fx * Math.SQRT2;
				ry = fy * Math.SQRT2;
			}
		}
		out[offset + 6] = rx;
		out[offset + 7] = ry;
	} else {
		out[offset + 6] = (gradient.from * Math.PI) / 180;
	}
	stops.forEach((s, i) => {
		out[offset + 8 + i] = s.offset;
	});
	gradientStopsToOklab(stops).forEach((lab, i) => {
		out.set(lab, offset + 16 + i * 4);
	});
}
