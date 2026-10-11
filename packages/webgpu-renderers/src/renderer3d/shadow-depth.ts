/** Shadow maps: a depth32float target rendered from a light's point of view. */
export const SHADOW_DEPTH_FORMAT: GPUTextureFormat = "depth32float";

/** Depth state for shadow-map passes: slope-scaled bias fights acne on steep faces. */
export const SHADOW_DEPTH_STENCIL: GPUDepthStencilState = {
	format: SHADOW_DEPTH_FORMAT,
	depthWriteEnabled: true,
	depthCompare: "less-equal",
	depthBias: 2,
	depthBiasSlopeScale: 2.5,
	depthBiasClamp: 0.01,
};
