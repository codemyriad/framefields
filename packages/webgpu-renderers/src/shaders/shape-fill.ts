import { gradientWgsl } from "./gradient.js";

export const shapeFillWgsl = `
${gradientWgsl}

struct Uniforms {
    transformCol0 : vec4<f32>,
    transformCol1 : vec4<f32>,
    transformCol2 : vec4<f32>,
    params        : vec4<f32>, // x = opacity, y = surfaceWidth, z = surfaceHeight, w = unused
    color         : vec4<f32>, // solid color (gradient.info.x = 0)
    gradient      : GradientUniforms,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) localPos  : vec2<f32>,
};

@vertex fn vs(@location(0) xy : vec2<f32>) -> VSOut {
    let transform = mat3x3<f32>(u.transformCol0.xyz, u.transformCol1.xyz, u.transformCol2.xyz);
    let p = transform * vec3<f32>(xy, 1.0);
    
    let surfaceSize = u.params.yz;
    let clipX = (p.x / surfaceSize.x) * 2.0 - 1.0;
    let clipY = 1.0 - (p.y / surfaceSize.y) * 2.0;

    return VSOut(vec4<f32>(clipX, clipY, 0.0, 1.0), xy);
}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
    var outColor = u.color;
    if (u.gradient.info.x > 0.5) {
        outColor = evalGradient(u.gradient, in.localPos);
    }
    
    let alpha = outColor.a * u.params.x;
    if (alpha <= 0.0) {
        discard;
    }
    
    return vec4<f32>(outColor.rgb * alpha, alpha);
}
`;
