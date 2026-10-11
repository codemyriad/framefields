/**
 * Multi-stop linear / radial / conic gradients, blended in OKLab, shared by
 * the solid (rounded rect backgrounds) and shape-fill pipelines. The including
 * shader embeds a `GradientUniforms` in its uniform block.
 */
export const gradientWgsl = `
struct GradientUniforms {
    info    : vec4<f32>, // x = type (0 none, 1 linear, 2 radial, 3 conic), y = stop count
    geom    : vec4<f32>, // linear: cx, cy, dir.x / len, dir.y / len; radial: cx, cy, rx, ry; conic: cx, cy, from (rad)
    offsets : array<vec4<f32>, 2>,
    stops   : array<vec4<f32>, 8>, // OKLab + alpha
};

fn gradientOklabToSrgb(lab: vec3<f32>) -> vec3<f32> {
    let l = pow(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z, 3.0);
    let m = pow(lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z, 3.0);
    let s = pow(lab.x - 0.0894841775 * lab.y - 1.291485548 * lab.z, 3.0);
    let lin = clamp(vec3<f32>(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
    ), vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = lin * 12.92;
    let hi = 1.055 * pow(lin, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, lin <= vec3<f32>(0.0031308));
}

/// Position along the gradient (0..1, unclamped for linear/radial) at local point p.
fn gradientT(g: GradientUniforms, p: vec2<f32>) -> f32 {
    let d = p - g.geom.xy;
    let kind = g.info.x;
    if (kind < 1.5) {
        return dot(d, g.geom.zw) + 0.5;
    }
    if (kind < 2.5) {
        return length(d / max(g.geom.zw, vec2<f32>(1e-4)));
    }
    let a = atan2(d.x, -d.y) - g.geom.z;
    return fract(a / 6.28318530718);
}

/// Straight sRGB color + alpha of the gradient at local point p.
fn evalGradient(g: GradientUniforms, p: vec2<f32>) -> vec4<f32> {
    var offs = g.offsets;
    var stops = g.stops;
    let n = i32(g.info.y);
    let t = gradientT(g, p);
    var lab = stops[0];
    if (t > offs[0].x) {
        lab = stops[n - 1];
        for (var i = 1; i < 8; i++) {
            if (i >= n) {
                break;
            }
            let b = offs[i / 4][i % 4];
            if (t <= b) {
                let a = offs[(i - 1) / 4][(i - 1) % 4];
                let f = select(1.0, (t - a) / (b - a), b - a > 1e-5);
                lab = mix(stops[i - 1], stops[i], clamp(f, 0.0, 1.0));
                break;
            }
        }
    }
    return vec4<f32>(gradientOklabToSrgb(lab.xyz), lab.w);
}
`;

/** Floats a `GradientUniforms` block takes. */
export const GRADIENT_UNIFORM_FLOATS = 48;
