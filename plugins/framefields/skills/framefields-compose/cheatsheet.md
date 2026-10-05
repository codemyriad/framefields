# framefields cheat sheet: cameras, paths, text, effects, limits

Practical recipes checked against framefields 2.0.2.

- **VERIFIED**: a test composition was rendered with `renderFrame` and the frame was inspected.
- **UNVERIFIED**: read from the source code, but not rendered.
- **BROKEN**: the renderer and the schema disagree; a workaround is given.

Read this before building a film that moves a camera through a large layout instead of cutting between full-screen scenes.

## The ten rules that save the most time

1. **2D camera = one big world box.** Use `anchorX: 0, anchorY: 0`. Drive `x`, `y` and `scale` with `Signal.fromArray`, not keyframes. (§1)
2. **Never zoom past 1.0 on 2D content.** Leaf nodes are rasterized at their layout size and then transformed, so zooming in blurs them. Lay out each area at the size of its closest shot. Overview shots zoom out (0.25–0.6), which stays sharp. (§1)
3. **128 keyframes per node**, summed over all its tracks. The schema enforces this. For anything per-frame, use signals. (§7)
4. **Off-screen nodes are not culled.** They cost about as much as visible ones. Give each area a `startFrame`/`durationFrames` window. (§1)
5. **Effect classes (`Blur`, `Vignette`, …) work only on media nodes** (image, svg, video) and on `comp.apply(...)`. On `box`/`flex`/`text` they are silently ignored. (§6)
6. **Camera motion blur:** make the world a `Layer.box` (not `flex`) with `motionBlurShutter`. Lower the shutter during zooms. (§6)
7. **Dots and particles:** use `Layer.box({ borderRadius })`, not `Layer.shape('circle')`. A circle shape costs about 3–4× more. (§1)
8. **Path `d` is in the shape box's local pixels.** It is not scaled to fit. Each cubic segment is cut into 24 straight steps, so keep segments short. (§3)
9. **Text without spaces never wraps** (hashes, base64). Insert `\n` yourself. Rich-text `spans` are dropped by the compositor. (§5)
10. **`followPath` with default options exceeds the keyframe cap.** Pass `autoOrient: false` and `samples` ≤ 40. It positions the node's top-left corner. (§4)

---

## 1. A 2D world with a camera

Put all content in one large absolute box. Animate the box, not its children.

- `anchorX` / `anchorY` set the pivot for `scale` and `rotation`, as a fraction of the box. The default is `0.5, 0.5` (centre). **VERIFIED**
- With `anchorX: 0, anchorY: 0`, a world point `(wx, wy)` lands on screen at `(x + wx·scale, y + wy·scale)`. To centre world point `(cx, cy)` at zoom `z`: `x = W/2 − cx·z`, `y = H/2 − cy·z`, `scale = z`.

```ts
import { Layer, LayerAnimation, Signal } from 'framefields';

type Shot = [frame: number, cx: number, cy: number, zoom: number];
const ease = (t: number) => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; // = 'power2.inOut'

// Hold before the first shot and after the last; interpolate zoom in log space.
// parallax < 1 moves a backdrop layer more slowly than the world.
export function cameraAnimation(shots: Shot[], totalFrames: number, parallax = 1, W = 1920, H = 1080, fps = 30) {
  const xs: number[] = [], ys: number[] = [], ss: number[] = [];
  for (let f = 0; f < totalFrames; f++) {
    let i = 0; while (i < shots.length - 2 && f > shots[i + 1][0]) i++;
    const [f0, x0, y0, z0] = shots[i], [f1, x1, y1, z1] = shots[Math.min(i + 1, shots.length - 1)];
    const t = f1 === f0 ? 1 : ease(Math.min(1, Math.max(0, (f - f0) / (f1 - f0))));
    const z = Math.exp(Math.log(z0) + (Math.log(z1) - Math.log(z0)) * t);
    const cx = (x0 + (x1 - x0) * t) * parallax, cy = (y0 + (y1 - y0) * t) * parallax;
    xs[f] = W / 2 - cx * z; ys[f] = H / 2 - cy * z; ss[f] = z;
  }
  return LayerAnimation.create()
    .signal('x', Signal.fromArray(xs, fps))
    .signal('y', Signal.fromArray(ys, fps))
    .signal('scale', Signal.fromArray(ss, fps));
}

const world = Layer.box({ id: 'world', position: 'absolute', x: 0, y: 0, width: 7680, height: 4320,
  anchorX: 0, anchorY: 0, background: 'transparent', motionBlurShutter: 90,
  children: [ /* absolute children in world pixels */ ] } as any)
  .animate(cameraAnimation([[0, 1700, 1300, 1], [60, 1700, 1300, 1], [120, 3840, 2160, .25], [180, 6300, 3250, 1]], 181));

comp.add(world);
comp.add(captions); // added after the world and outside it: screen space, unaffected by the camera
```

**VERIFIED**, including a 0.4× parallax backdrop, a screen-space caption on top, and motion blur.

Things to know:

- **Sharpness.**
  - Zoom 0.25 and 0.5 are sharp.
  - Zoom 3 is visibly soft. Text, images and shapes are all drawn at layout size first.
  - Give screenshots their native pixel size as their world size.
- **Signals use composition frames.** Keyframes use the node's local frames. If the world has a `startFrame`, its keyframes shift but its signals do not.
- **`box` or `flex` world.**
  - A transparent `Layer.flex` takes a fast path: its children draw directly with the parent matrix.
  - A `Layer.box` renders into a group texture first. That is one extra, cheap pass.
  - Both look the same. Only the box gets camera motion blur.
- **Clipping.** Boxes and flexes default to `overflow: 'hidden'`, clipped through the transform. Make the world rectangle large enough for everything in it.
- **Cost.** These are relative timings from one machine, so use them to compare approaches only.

  | Situation | ms per frame |
  |---|---|
  | Empty frame | 21 |
  | +600 off-screen text nodes | 198 |
  | Same 600, gated by `startFrame`/`durationFrames` | 69 |

  Per node on screen:

  | Node | ms per node per frame |
  |---|---|
  | `shape('circle')` | 2.2 |
  | `path` | 0.9 |
  | rounded `box` | 0.7 |
  | `box` | 0.5 |
  | `text` | 0.5 |
  | `image` | 0.5 |

  No node cap is enforced. Films with about 1,700 nodes render fine when most are time-gated.

## 2. 3D: `Layer.camera`, `is3D`, per-node tilt

**Default 3D camera.**

- It exists when there is a `Layer.camera`, or any node with `is3D` or `z !== 0`.
- Vertical FOV is 50°. The eye is at `(W/2, H/2, −1158)` looking at `(W/2, H/2, 0)`. Y points down; +z is farther away.
- The plane `z = 0` maps 1:1 to screen pixels.
- Props you can animate or drive with signals: `cameraX/Y/Z`, `targetX/Y/Z`, `cameraFov`, `cameraZoom`, `cameraPitch/Yaw/Roll`, `orbit*`, `focusDistance`, `fStop`.

**Dolly and multiplane parallax.** **VERIFIED**

```ts
comp.add(Layer.camera({ id: 'cam', lens: { fov: 50 } } as any).animate(LayerAnimation.create()
  .signal('cameraX', Signal.fromArray(cx, 30)).signal('targetX', Signal.fromArray(cx, 30))
  .keys('cameraY', [[0, 540]]).keys('targetY', [[0, 540]])
  .signal('cameraZ', Signal.fromArray(cz, 30)).keys('targetZ', [[0, 0]])));
comp.add(Layer.box({ id: 'far', is3D: true, z: 1500, x: -1200, y: -800, width: 4400, height: 2700,
  background: '#0c1826', material: 'unlit', children: [] } as any));      // far plane parallaxes slower
comp.add(Layer.text('Label', { id: 'near', is3D: true, z: 0, x: 780, y: 760, width: 700, height: 60,
  fontFamily: 'Inter', fontSize: 40, fill: '#fff', material: 'unlit' } as any));
```

What works and what doesn't:

- An **`is3D` text leaf stays sharp** under 3D magnification. Its glyphs are resolved per screen pixel. **VERIFIED**
- **Children of an `is3D` box are rasterized** and then projected, so they blur when magnified. **VERIFIED**
- **`transformStyle: 'preserve-3d'` is unreliable.** A child without `is3D` did not render. A child with `is3D` rendered shifted by about half its width. Avoid it, or measure the offset.
- **Per-node tilt** (no camera): set `perspective: 1200` on the box and animate `rotateX`/`rotateY`.
  - Single-axis tilts up to about 35° render.
  - (10°, −40°) made the card disappear. Keep combined tilts at or below about 30°, or check each frame.
- Use `material: 'unlit'` for flat UI.
- Do not put `is3D` nodes inside a 2D camera world. They render in a separate pass and probably ignore the world matrix. (UNVERIFIED)
- Camera depth of field (`dof`) dimmed far planes instead of blurring them. Do not rely on it. Fake depth of field with `Blur` on media backdrops.

**Choosing.** The 2D world camera is the safer default for long travel across dense layouts: sharp at zoom ≤ 1, supports flex, and captions sit naturally on top. Add 3D for `is3D` titles and far-plane media backdrops.

## 3. Path connectors that draw themselves

```ts
const d = 'M 20 300 C 400 20, 700 580, 1000 300 S 1400 100, 1480 300'; // local pixels of the shape box
Layer.shape('path', { id: 'wire', position: 'absolute', x: 100, y: 200, width: 1500, height: 600, d,
  fillType: 'none', strokeColor: '#7bc7ff', strokeWidth: 6, strokeLineCap: 'round', trimEnd: 0 } as any)
  .animate(LayerAnimation.create().fromTo('trimEnd', 0, 1, { start: 0, end: 60, ease: 'power2.inOut' }));
```

**VERIFIED**

**The shape box.**

- The box must contain the whole stroke. Anything outside it is clipped.
- Animatable props: `trimStart`, `trimEnd`, `trimOffset`, `strokeWidth`, `strokeColor`, `fillColor`, `strokeDashOffset`.
- Other options: `strokeLineCap`, `strokeLineJoin`, and `fillType: 'none'|'solid'|'linear'|'radial'` with `gradientEndColor` and `gradientAngle`.
- Each cubic segment is drawn as 24 straight steps (a quadratic as 16). Keep segments under about 300–400 px, or long curves look faceted.

**Arrowheads.**

- Stroked paths have no arrow markers.
- `Layer.shape('arrow')` is a filled block arrow pointing +x.
- For a head that rides a drawing connector, use a small filled triangle path and move it with the same eased progress. **VERIFIED**

```ts
import { parseSvgPathToCubicBeziers, buildPathLUT, samplePathChainGeometry } from 'framefields';
const segs = parseSvgPathToCubicBeziers(d), lut = buildPathLUT(segs);
const sample = (u: number) => samplePathChainGeometry(segs, lut, u * lut.totalLength); // { position, tangentAngleDeg, normal }
// e.g. 31 samples × (x, y, rotation) = 93 keys, under the 128 cap
```

- Match the ease exactly. In GSAP names, `power1` is quadratic, `power2` is cubic and `power3` is quartic.
- Hide the head with an opacity key until the wire starts drawing.

**Dashes. BROKEN as of 2.0.2.**

- The schema types `strokeDashArray` as a string, but the renderer needs `number[]`.
  - A string draws nothing.
  - An array draws correctly (marching ants via `strokeDashOffset` work), but fails `CompositorProgramSchema`.
- Workarounds:
  - Validate a copy of the spec with arrays converted to strings, and render the original.
  - Or fake dashes with several trimmed copies of the path.

## 4. Moving objects along a path

`LayerAnimation.followPath(pathLike, opts)` samples any object with `totalLength` and `getPointAt(u) → { position, tangent }` into `x`/`y` keyframes, plus `z`/`rotateZ` when used. Speed along the path is constant, with an ease. **VERIFIED**

```ts
const follower = { totalLength: lut.totalLength, getPointAt: (u: number) => {
  const g = sample(u), a = g.tangentAngleDeg * Math.PI / 180;
  return { position: [OX + g.position.x - 9, OY + g.position.y - 9, 0], tangent: [Math.cos(a), Math.sin(a), 0] };
} };
Layer.box({ id: 'packet', position: 'absolute', x: 0, y: 0, width: 18, height: 18, borderRadius: 9,
  background: '#f3bf72', startFrame: 30, durationFrames: 75 })
  .animate(LayerAnimation.create().followPath(follower, { start: 0, end: 60, autoOrient: false, samples: 30, ease: 'sine.inOut' }));
```

- The position is the node's **top-left**. Subtract half the node size.
- The defaults (`samples` up to 60 per track, `autoOrient: true` adding `rotateZ`) can reach 183 keys and fail the 128 cap.
- `start`/`end` are local frames.
- Many objects on long routes: precompute per-frame arrays and use `.signal('x', Signal.fromArray(...))`.

## 5. Text

| Want | How | Status |
|---|---|---|
| Typewriter | `.animate(LayerAnimation.create().typewriter(start, end))` animates the visible fraction 0 → 1 | VERIFIED |
| Long tokens (hashes, base64, JSON) | They do not wrap. Split them with `\n` every N characters and set `lineHeight`. Typewriter works across lines. | VERIFIED |
| Key and value in different colours | Use two text nodes on a monospace grid. JetBrains Mono advances `0.6 × fontSize` per character. Put highlight boxes behind columns at `x0 + col × 0.6 × fontSize`. | VERIFIED |
| Per-character blur-in | `animators: [TextAnimator.blurIn({ blur: 18, y: 24 })]` + `.animate(LayerAnimation.create().kineticSweep(-1, 1, 0, 40))` | VERIFIED |
| Word rise | `TextAnimator.waveRise({ unit: 'word', y: 60, rotationX: 60 })` + `kineticSweep` | VERIFIED |
| Colour sweep | `TextAnimator.colorSweep({ color })` with `kineticSweep(-1, 1, …)` goes **from** that colour **to** `fill` | VERIFIED |
| `wordPop`, `flip3D` | Same mechanism as the presets above | UNVERIFIED |
| Rich `spans` (`rich`, `color()`, `mark()`) | Dropped by the compositor; text renders in plain `fill`. Use separate text nodes. | BROKEN |
| `typewriter: { cursor, scramble }` object on the node | The scramble renders, but timing follows the node's lifetime, not the `text` track, and no cursor appeared. Control timing with the node's `startFrame`/`durationFrames`. | Partly verified |

- Animated text gets extra padding in its layer (20% of width, 50% of height). This is harmless.
- Give text boxes enough `height`. Lines that overflow are squashed or clipped.

## 6. Effects

| Effect | Where it works | Notes |
|---|---|---|
| `Blur` (Gaussian, Box, Motion, Radial, Zoom) | Media nodes: `.withEffect(new Blur({ strength }))` | Ignored on box/flex/text. For text, use `TextAnimator.blurIn`. |
| `Modulate`, `Curves`, `ColorBalance` | Media nodes, or `comp.apply(...)` | |
| `Vignette` | `comp.apply(new Vignette({ strength: 45, radius: .9, softness: .6 }))` | VERIFIED |
| `FilmGrain` | `comp.apply(new FilmGrain({ strength: 8, size: 1.4, monochrome: true, animated: true }))` | VERIFIED. 0–100 scale; 6–10 is subtle. |
| Motion blur on a moving node | `motionBlurShutter: 180` (0–360) on the node. Velocity comes from its own `x`/`y`, keyed or signal-driven. | VERIFIED on box and text |
| Motion blur on a camera pan | World must be a `Layer.box` with `motionBlurShutter`. A flex world, or children of a panning flex, do not blur. | VERIFIED |
| Motion blur during zoom | Blur is translation-only. An anchored zoom changes `x`/`y`, so it smears in one direction. | Use a shutter of 60–90, or animate `motionBlurShutter` to 0 during zooms. |
| Glow / bloom | There is no bloom class. Put a blurred copy of an image/svg underneath with `blendMode: 'screen'`, in a box larger than the art. | VERIFIED. Text `shadows` with a large radius look blocky; avoid. |
| Blend modes | `blendMode` on any node (`screen`, `multiply`, `overlay`, `lighten`, `difference`, …) | `screen` VERIFIED |

Order: grade first, then optics, then grain last. Vignette plus grain added about 33 ms per frame on a software adapter.

## 7. Limits and pitfalls

- **Keyframe cap: 128 per node**, all tracks together. Workarounds:
  - `Signal.fromArray` (no cap).
  - Fewer keys with per-key eases: `keys(prop, [[f, v, 'power2.inOut'], …])`.
  - Split the motion across nested nodes.
- **Integer frames only.** Two keys on the same frame are rejected.
- **Cost scales with time-active nodes**, on screen or not. Gate groups with `startFrame`/`durationFrames`. Keyframes inside a gated group are local to the group; signals are not.
- **Zoom > 1 blurs 2D content.** The exception is `is3D` text leaves under a 3D camera.
- **Rendering scripts don't exit by themselves.** End them with `process.exit(0)`. Otherwise a script piped into `tail` looks hung.
