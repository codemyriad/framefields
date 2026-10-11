/**
 * Screen-space track matte for 3D draws. The compositor resolves a matte of
 * any mode into a coverage texture (alpha = how much of the layer shows);
 * the draw scales its alpha by it at each pixel and discards where nothing
 * shows, so a matted layer keeps its place in the depth buffer. Unmatted
 * draws bind a 1×1 white texture (full coverage).
 *
 * The including shader declares `cam : CameraUniforms`.
 */
export function matte3dWgsl(group: number, binding: number): string {
	return `
@group(${group}) @binding(${binding}) var matteTex : texture_2d<f32>;

fn matteCoverage(fragPos : vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(matteTex));
    let p = floor(fragPos * dims / max(cam.screenParams.xy, vec2<f32>(1.0)));
    return textureLoad(matteTex, vec2<i32>(clamp(p, vec2<f32>(0.0), dims - 1.0)), 0).a;
}
`;
}
