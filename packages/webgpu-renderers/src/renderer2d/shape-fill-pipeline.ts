import { gradientStopsToOklab } from "@framefields/core";
import { parseColor } from "../color.js";
import { GRADIENT_UNIFORM_FLOATS } from "../shaders/gradient.js";
import { shapeFillWgsl } from "../shaders/shape-fill.js";
import { BufferPool } from "./buffer-pool.js";
import { asGradient, packGradientUniforms } from "./gradient-uniforms.js";
import type { Color, DrawOpts, Rect } from "./index.js";
import { createUniformLayout, createVec2Pipeline } from "./pipeline-utils.js";
import type { TransformStack } from "./transform-stack.js";

export type ShapeFillType = "solid" | "linear" | "radial" | "none";

export interface ShapeFillConfig {
	type?: ShapeFillType;
	/** A color, or a CSS gradient string (linear/radial/conic, any stops) laid over the shape's bounds. */
	color?: Color | string;
	color2?: Color | string;
	startX?: number;
	startY?: number;
	endX?: number;
	endY?: number;
	cx?: number;
	cy?: number;
	radius?: number;
}

export class ShapeFillPipeline {
	private device: GPUDevice;
	private pipeline: GPURenderPipeline;
	private uniformLayout: GPUBindGroupLayout;
	private uniformPool: BufferPool;
	private vertexPool: BufferPool;
	private uniformData = new Float32Array(20 + GRADIENT_UNIFORM_FLOATS);

	constructor(device: GPUDevice, format: GPUTextureFormat) {
		this.device = device;
		this.uniformPool = new BufferPool(
			2048,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		);
		this.vertexPool = new BufferPool(
			8192,
			GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		);

		const module = device.createShaderModule({
			label: "shape-fill.wgsl",
			code: shapeFillWgsl,
		});

		this.uniformLayout = createUniformLayout(device);

		this.pipeline = createVec2Pipeline(device, {
			label: "ShapeFillPipeline",
			module,
			uniformLayout: this.uniformLayout,
			format,
			topology: "triangle-list",
		});
	}

	resetPools(): void {
		this.uniformPool.reset();
		this.vertexPool.reset();
	}

	draw(
		pass: GPURenderPassEncoder,
		transformStack: TransformStack,
		vertices: Float32Array,
		fill: ShapeFillConfig,
		bounds: Rect,
		surfaceWidth: number,
		surfaceHeight: number,
		opts?: DrawOpts,
	): void {
		if (vertices.length < 6 || fill.type === "none") return;

		const opacity = opts?.opacity ?? 1.0;
		transformStack.packIntoBuffer(
			this.uniformData,
			opacity,
			surfaceWidth,
			surfaceHeight,
			opts?.transform,
		);

		this.uniformData[15] = 0;
		const css = asGradient(fill.color);
		if (css) {
			this.uniformData.fill(0, 16, 20);
			packGradientUniforms(css, bounds, this.uniformData, 20);
		} else {
			const c1 = parseColor(fill.color ?? "#ffffff");
			this.uniformData[16] = c1.r;
			this.uniformData[17] = c1.g;
			this.uniformData[18] = c1.b;
			this.uniformData[19] = c1.a;
			this.packTwoColorGradient(fill, bounds, c1);
		}

		const uniformBuffer = this.uniformPool.getBuffer(
			this.device,
			this.uniformData,
		);
		const vertexBuffer = this.vertexPool.getBuffer(this.device, vertices);

		pass.setPipeline(this.pipeline);
		pass.setBindGroup(
			0,
			this.device.createBindGroup({
				layout: this.uniformLayout,
				entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
			}),
		);
		pass.setVertexBuffer(0, vertexBuffer);
		pass.draw(vertices.length / 2);
	}

	/** The `type: "linear" | "radial"` two-color fill (color → color2). */
	private packTwoColorGradient(
		fill: ShapeFillConfig,
		bounds: Rect,
		c1: Color,
	): void {
		const o = 20;
		this.uniformData.fill(0, o, o + GRADIENT_UNIFORM_FLOATS);
		if (fill.type !== "linear" && fill.type !== "radial") return;
		const c2 = parseColor(fill.color2 ?? fill.color ?? "#ffffff");
		const d = this.uniformData;
		if (fill.type === "linear") {
			const x0 = fill.startX ?? bounds.x;
			const y0 = fill.startY ?? bounds.y;
			const x1 = fill.endX ?? bounds.x + bounds.width;
			const y1 = fill.endY ?? bounds.y + bounds.height;
			const dx = x1 - x0;
			const dy = y1 - y0;
			const lenSq = Math.max(dx * dx + dy * dy, 1e-6);
			d.set(
				[1, 2, 0, 0, (x0 + x1) / 2, (y0 + y1) / 2, dx / lenSq, dy / lenSq],
				o,
			);
		} else {
			const r = fill.radius ?? Math.max(bounds.width, bounds.height) / 2;
			d.set(
				[
					2,
					2,
					0,
					0,
					fill.cx ?? bounds.x + bounds.width / 2,
					fill.cy ?? bounds.y + bounds.height / 2,
					r,
					r,
				],
				o,
			);
		}
		d[o + 8] = 0;
		d[o + 9] = 1;
		const lab = gradientStopsToOklab([
			{ offset: 0, color: [c1.r, c1.g, c1.b, c1.a] },
			{ offset: 1, color: [c2.r, c2.g, c2.b, c2.a] },
		]);
		d.set(lab[0], o + 16);
		d.set(lab[1], o + 20);
	}

	destroy(): void {
		this.uniformPool.destroy();
		this.vertexPool.destroy();
	}
}
