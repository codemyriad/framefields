# 26 — Mac Keyboard

A 12-second product shot of a compact Mac keyboard with Touch ID, made only from 3D primitives. The
aluminium body lands, 78 keycaps rain in from the center out, someone types `Hello, framefields.`
with each key bottoming out and flashing its backlight in time with the line on screen, and then a
blue-to-pink backlight wave rolls across the board while the camera rises to top-down.

Nothing to download or generate: fonts come from the repo's `assets/fonts`.

```bash
pnpm render grid           # output/grid.png, a 16-frame contact sheet
pnpm render frames 116 290
pnpm render                # output/mac-keyboard.mp4
pnpm render:src grid       # same, on the engine's source (no package builds)
```

## The parts

| Part | Primitive |
|---|---|
| Body | `Layer3D.roundedBox`: one smooth-shaded mesh, metallic silver |
| Keycaps | `Layer3D.roundedBox` with `taper: 0.95`, so the top face is a little smaller than the base |
| Legends | `Layer3D.text`, laid flat on each cap (`rotateX: -90`) |
| ⌘ and ⌥ | Inter has no glyphs for these, so they are stroked `Layer.shape("path")` icons on a transparent `Layer3D.plane` |
| Touch ID ring | `Layer3D.cylinder`, 1.2 px tall |
| Floor | A very large, very thin `roundedBox` under a point light. `BG` matches the floor's color at the far plane, so there is no horizon line |

Every key is a `preserve-3d` box holding its cap and legends, so moving the box moves all of them
together. The cap's `tint` multiplies its color and animates as the `color` prop, which is how keys
light up. Keys of the same size share one mesh: `Layer3D` caches geometry by its options, so all
the letter keys use a single vertex buffer.

## Files

- `layout.ts`: the US layout in key units (rows, widths, legends) and `keysForChar`, which maps a
  character to the keys it needs, with shift for capitals and symbols.
- `keyboard.ts`: builds the keyboard from the layout. World axes: x right, y down, z away from the
  camera. The keyboard lies in the x/z plane, so "up" is -y.
- `timeline.ts`: the three acts and the typing schedule. Keystrokes have a fixed, human-feeling
  rhythm: about 5.5 frames per key with deterministic jitter, a bit longer on spaces and a pause
  after the comma.
- `motion.ts`: each key's full animation as one `y` track and one `color` track: rain in, press for
  every stroke that uses it, then the wave. A quick repeat (the "ll" in Hello) springs back faster
  so the second press still reads. A modifier goes down before the key it modifies and comes up
  after it.
- `film.ts`: the camera (keyed `orbitAzimuth` / `orbitElevation` / `orbitRadius`), the lights, the
  typed line and its caret, the caption, plus vignette and grain. The grain also dithers the
  floor's light pool, which would otherwise show banding in 8-bit.

## Typing in sync

On a text layer, `text` reads as a fraction of the string while it is at most 1, and as a character
count above 1. The typed line keys one count per keystroke with `"hold"` easing, so each character
appears on the exact frame its key bottoms out. The caret's `x` uses the same keys:
JetBrains Mono advances 0.6 em per character, so its position follows from the count.
