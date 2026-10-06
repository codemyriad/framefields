---
"framefields": minor
---

Preview: `exportPreview(source, { outDir, basePath })` writes the live preview as a static site that any host can serve under a path, with no server: the page, the composition's code, the browser engine, the project files it loads and the soundtrack. Several compositions can share one export. The browser engine is now built as `.js` modules, which every static host serves as JavaScript.

Preview playback options, for `startPreview` and `exportPreview`, each also settable in the page's address: `height` (`?height=360`) draws a smaller picture and the page gains a resolution menu; `playback: "realtime"` (`?playback=realtime`) draws the frame due now and skips late ones instead of buffering, so the sound never waits; `showFps` (`?fps=1`) shows the frame rate playback reaches. `window.framefieldsPreview.state()` also reports `playbackFps`, `renderMs` and the size drawn. `open` also takes a function, to open the page in a browser of the project's choosing.
