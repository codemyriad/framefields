import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { Composition, Layer } from "../index.js";
import { buildEngine, fileUrl, prebuiltEngineDir } from "./bundle.js";
import { exportPreview, type PreviewSession, startPreview } from "./index.js";
import { previewPage } from "./page.js";
import { scaleToHeight } from "./scale.js";

const entry = new URL("./fixtures/comp.ts", import.meta.url);

/** The settings a page was written with. */
function pageConfig(page: string): Record<string, unknown> {
	const json = /var CONFIG = (.*);\n/.exec(page)?.[1];
	if (!json) throw new Error("no CONFIG in the page");
	return JSON.parse(json);
}

/** The page's inline script, compiled (not run) to catch syntax errors. */
function compileScript(page: string): void {
	const script = /<script>\n([\s\S]*?)<\/script>/.exec(page)?.[1];
	if (!script) throw new Error("no inline script in the page");
	new Function(script);
}

describe("startPreview", () => {
	const sessions: PreviewSession[] = [];
	const start = async (...args: Parameters<typeof startPreview>) => {
		const s = await startPreview(...args);
		sessions.push(s);
		return s;
	};
	afterEach(async () => {
		await Promise.all(sessions.splice(0).map((s) => s.close()));
	});

	it("serves meta, the page, and the composition's browser bundle", async () => {
		const session = await start(
			{ entry },
			{ open: false, port: 0, title: "Test" },
		);

		const meta = await (await fetch(`${session.url}meta`)).json();
		expect(meta).toMatchObject({
			title: "Test",
			fps: 10,
			frameCount: 10,
			width: 64,
			audio: { state: "none" },
		});

		const page = await (await fetch(session.url)).text();
		expect(page).toContain('id="canvas"');
		expect(page).toContain(
			'<script type="module" src="/@framefields/player.js"></script>',
		);
		expect(pageConfig(page)).toEqual({
			meta: "/meta",
			audio: "/audio",
			player: "/@framefields/player.js",
			live: true,
		});

		const player = await fetch(`${session.url}@framefields/player.js`);
		expect(player.headers.get("content-type")).toContain("javascript");
		const code = await player.text();
		expect(code).toContain("startPlayer");
		expect(code).not.toContain("Could not build the preview");
	}, 60_000);

	it("writes the playback options into the page", async () => {
		const session = await start(
			{ entry },
			{
				open: false,
				port: 0,
				height: 360,
				playback: "realtime",
				showFps: true,
			},
		);
		const page = await (await fetch(session.url)).text();
		expect(pageConfig(page)).toMatchObject({
			height: 360,
			playback: "realtime",
			showFps: true,
		});
		compileScript(page);
	});

	it("serves the browser engine the project bundle imports", async () => {
		const session = await start({ entry }, { open: false, port: 0 });

		const player = await fetch(`${session.url}@framefields/engine/player.js`);
		expect(player.status).toBe(200);
		expect(player.headers.get("content-type")).toContain("javascript");
		expect(await player.text()).toContain("startPlayer");

		const shim = await fetch(`${session.url}@framefields/engine/shims/path.js`);
		expect(shim.status).toBe(200);
		expect(await shim.text()).toContain("export");

		expect(
			(await fetch(`${session.url}@framefields/engine/missing.js`)).status,
		).toBe(404);
	}, 60_000);

	it("serves project files by absolute path, with byte ranges", async () => {
		const session = await start({ entry }, { open: false, port: 0 });
		const file = fileURLToPath(entry);

		const whole = await fetch(`${session.url}${file.slice(1)}`);
		expect(whole.status).toBe(200);
		expect(await whole.text()).toContain("stand-in composition");

		const part = await fetch(`${session.url}${file.slice(1)}`, {
			headers: { range: "bytes=0-1" },
		});
		expect(part.status).toBe(206);
		expect(await part.text()).toBe("//");

		expect((await fetch(`${session.url}etc/hosts`)).status).toBe(404);

		const stat = (p: string) =>
			fetch(
				`${session.url}@framefields/stat?path=${encodeURIComponent(p)}`,
			).then((r) => r.json());
		expect(await stat(file)).toMatchObject({ exists: true });
		expect(await stat(`${file}.missing`)).toEqual({ exists: false, size: 0 });
		expect(await stat("/etc/hosts")).toEqual({ exists: false, size: 0 });
	});

	it("serves assets from the project root, not only the entry's directory", async () => {
		const project = await fs.promises.mkdtemp(
			path.join(os.tmpdir(), "gf-project-"),
		);
		try {
			const src = path.join(project, "src");
			const fonts = path.join(project, "assets", "fonts");
			await fs.promises.mkdir(src, { recursive: true });
			await fs.promises.mkdir(fonts, { recursive: true });
			// A project with no .git: the root is the nearest package.json.
			await fs.promises.writeFile(path.join(project, "package.json"), "{}");
			const comp = path.join(src, "comp.ts");
			await fs.promises.writeFile(
				comp,
				"export default function build() { return { fps: 10, durationMs: 1000, width: 64, height: 36 }; }",
			);
			const font = path.join(fonts, "Test.ttf");
			await fs.promises.writeFile(font, "font-bytes");

			const session = await start({ entry: comp }, { open: false, port: 0 });
			const res = await fetch(`${session.url}${font.slice(1)}`);
			expect(res.status).toBe(200);
			expect(await res.text()).toBe("font-bytes");
		} finally {
			await fs.promises.rm(project, { recursive: true, force: true });
		}
	});

	it("allows extra asset roots outside the project", async () => {
		const project = await fs.promises.mkdtemp(
			path.join(os.tmpdir(), "gf-project-"),
		);
		const shared = await fs.promises.mkdtemp(
			path.join(os.tmpdir(), "gf-shared-"),
		);
		try {
			const src = path.join(project, "src");
			await fs.promises.mkdir(src, { recursive: true });
			await fs.promises.writeFile(path.join(project, "package.json"), "{}");
			const comp = path.join(src, "comp.ts");
			await fs.promises.writeFile(
				comp,
				"export default function build() { return { fps: 10, durationMs: 1000, width: 64, height: 36 }; }",
			);
			const font = path.join(shared, "Shared.ttf");
			await fs.promises.writeFile(font, "shared-bytes");

			const session = await start(
				{ entry: comp },
				{ open: false, port: 0, root: [project, shared] },
			);
			const res = await fetch(`${session.url}${font.slice(1)}`);
			expect(res.status).toBe(200);
			expect(await res.text()).toBe("shared-bytes");
		} finally {
			await fs.promises.rm(project, { recursive: true, force: true });
			await fs.promises.rm(shared, { recursive: true, force: true });
		}
	});

	it("names a missing export", async () => {
		await expect(
			startPreview({ entry, export: "film" }, { open: false, port: 0 }),
		).rejects.toThrow(/no export named "film"/);
	});

	it("closes once the last tab has gone, but not on a reload", async () => {
		const lines: string[] = [];
		const session = await start(
			{ entry },
			{ open: false, port: 0, idleCloseMs: 100, log: (l) => lines.push(l) },
		);
		let isClosed = false;
		session.closed.then(() => {
			isClosed = true;
		});
		const connect = () =>
			new Promise<http.ClientRequest>((resolve) => {
				const req = http.get(`${session.url}events`, { agent: false }, () =>
					resolve(req),
				);
			});
		// A reload: the tab drops and is back well inside the grace period.
		(await connect()).destroy();
		await new Promise((r) => setTimeout(r, 30));
		const tab = await connect();
		await new Promise((r) => setTimeout(r, 200));
		expect(isClosed).toBe(false);
		// The tab closes for good.
		tab.destroy();
		expect(await session.closed).toBe("idle");
		expect(lines).toEqual([
			"[framefields preview] page opened; the server stops 0.1 s after its last tab closes",
			"[framefields preview] stopped: its tab was closed",
		]);
	});

	it("reports the page's errors once each", async () => {
		const lines: string[] = [];
		const session = await start(
			{ entry },
			{ open: false, port: 0, log: (l) => lines.push(l) },
		);
		const send = (body: unknown, type = "application/json") =>
			fetch(`${session.url}@framefields/log`, {
				method: "POST",
				headers: { "content-type": type },
				body: JSON.stringify(body),
			});
		expect((await send({ level: "error", message: "boom" })).status).toBe(204);
		await send({ level: "error", message: "boom" });
		await send({ level: "warning", message: "[framefields] font" });
		// Not JSON: a form post from another site.
		expect((await send({ message: "spoof" }, "text/plain")).status).toBe(404);
		expect(lines).toEqual([
			"[framefields preview] error in the page: boom",
			"[framefields preview] warning in the page: [framefields] font",
		]);
	});

	it("keeps notes pinned on the page as files, and reports them", async () => {
		const project = await fs.promises.mkdtemp(
			path.join(os.tmpdir(), "gf-notes-"),
		);
		try {
			const dir = path.join(project, ".framefields", "preview-notes");
			const lines: string[] = [];
			const session = await start(
				{ entry },
				{ open: false, port: 0, notesDir: dir, log: (l) => lines.push(l) },
			);
			const meta = await (await fetch(`${session.url}meta`)).json();
			expect(meta.notesDir).toBe(dir);

			const send = (body: unknown) =>
				fetch(`${session.url}@framefields/notes`, {
					method: "POST",
					headers: { "content-type": "application/json" },
					body: JSON.stringify(body),
				});
			const jpeg = `data:image/jpeg;base64,${Buffer.from("jpeg-bytes").toString("base64")}`;
			const added = await (
				await send({
					op: "add",
					note: {
						frame: 5,
						time: 0.5,
						x: 0.25,
						y: 0.5,
						text: " Bigger title ",
						image: jpeg,
					},
				})
			).json();
			expect(added).toMatchObject({
				id: 1,
				frame: 5,
				x: 0.25,
				y: 0.5,
				text: "Bigger title",
				done: false,
			});
			expect(await fs.promises.readFile(added.image, "utf8")).toBe(
				"jpeg-bytes",
			);
			expect(
				await fs.promises.readFile(
					path.join(project, ".framefields", ".gitignore"),
					"utf8",
				),
			).toBe("*\n");
			expect(lines.at(-1)).toBe(
				`[framefields preview] note 1 at 0:00.50 (frame 5), spot 25%,50%: Bigger title\n  frame with the spot marked: ${added.image} · all notes: ${dir}/notes.json`,
			);

			await send({
				op: "add",
				note: {
					frame: 2,
					time: 0.2,
					x: 0.1,
					y: 0.1,
					w: 0.5,
					h: 0.2,
					text: "Too dark",
				},
			});
			await send({ op: "update", id: 2, done: true });
			expect(lines.at(-1)).toBe("[framefields preview] note 2 marked done");

			// notes.json is the record: edits made to it show up.
			const file = JSON.parse(
				await fs.promises.readFile(path.join(dir, "notes.json"), "utf8"),
			);
			expect(file).toHaveLength(2);
			expect(file[1]).toMatchObject({ w: 0.5, h: 0.2, done: true });

			expect((await send({ op: "remove", id: 1 })).status).toBe(200);
			expect(fs.existsSync(added.image)).toBe(false);
			const list = await (
				await fetch(`${session.url}@framefields/notes`)
			).json();
			expect(list.map((n: { id: number }) => n.id)).toEqual([2]);
			expect((await send({ op: "remove", id: 9 })).status).toBe(404);
		} finally {
			await fs.promises.rm(project, { recursive: true, force: true });
		}
	});

	it("opens the page with a given function, unless a tab comes back", async () => {
		const opened: string[] = [];
		const open = (url: string) => {
			opened.push(url);
		};
		const first = await start({ entry }, { open, port: 0 });
		expect(opened).toEqual([first.url]);

		// The next preview takes over the port, and the open tab reconnects.
		const port = Number(new URL(first.url).port);
		const second = start({ entry }, { open, port });
		await first.closed;
		for (;;) {
			const ok = await new Promise<boolean>((resolve) => {
				http
					.get(`${first.url}events`, { agent: false }, () => resolve(true))
					.on("error", () => resolve(false));
			});
			if (ok) break;
			await new Promise((r) => setTimeout(r, 50));
		}
		await second;
		expect(opened).toEqual([first.url]);
	}, 10_000);

	it("a new preview of the project takes over the port", async () => {
		const first = await start({ entry }, { open: false, port: 0 });
		const port = Number(new URL(first.url).port);
		const second = await start({ entry }, { open: false, port });

		expect(await first.closed).toBe("replaced");
		expect(second.url).toBe(first.url);
		expect((await fetch(`${second.url}meta`)).status).toBe(200);
	});
});

describe("exportPreview", () => {
	const dirs: string[] = [];
	afterEach(async () => {
		await Promise.all(
			dirs
				.splice(0)
				.map((d) => fs.promises.rm(d, { recursive: true, force: true })),
		);
	});
	/** A project whose composition reads a file next to it, as films do. */
	async function project() {
		const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "gf-export-"));
		dirs.push(dir);
		await fs.promises.mkdir(path.join(dir, "src"));
		await fs.promises.mkdir(path.join(dir, "assets", "sub"), {
			recursive: true,
		});
		await fs.promises.mkdir(path.join(dir, "notes"));
		await fs.promises.writeFile(path.join(dir, "package.json"), "{}");
		await fs.promises.writeFile(
			path.join(dir, "assets", "data.json"),
			"[1,2,3]",
		);
		await fs.promises.writeFile(
			path.join(dir, "assets", "sub", "logo.svg"),
			"<svg/>",
		);
		await fs.promises.writeFile(path.join(dir, "assets", ".hidden"), "x");
		await fs.promises.writeFile(path.join(dir, "assets", "voice.wav"), "RIFF");
		await fs.promises.writeFile(path.join(dir, "notes", "todo.txt"), "x");
		const comp = path.join(dir, "src", "comp.ts");
		await fs.promises.writeFile(
			comp,
			[
				'import fs from "node:fs";',
				'const data = new URL("../assets/data.json", import.meta.url);',
				'const frames = JSON.parse(fs.readFileSync(data, "utf8")).length;',
				"export default function build() {",
				"  return { fps: 10, durationMs: frames * 1000, width: 64, height: 36 };",
				"}",
			].join("\n"),
		);
		return { dir, comp, out: path.join(dir, "site") };
	}
	const read = (file: string) => fs.promises.readFile(file, "utf8");

	it("writes the page, bundle, engine and files for a static host", async () => {
		const { dir, comp, out } = await project();
		const result = await exportPreview(
			{ entry: comp },
			{
				outDir: out,
				basePath: "films/demo",
				title: "Demo",
				audio: false,
				height: 360,
				playback: "realtime",
				showFps: true,
			},
		);
		expect(result).toEqual({
			outDir: out,
			pages: { "": "/films/demo/" },
			files: [
				"/films/demo/assets/data.json",
				"/films/demo/assets/sub/logo.svg",
				"/films/demo/assets/voice.wav",
			],
		});
		expect(await read(path.join(out, "assets", "sub", "logo.svg"))).toBe(
			"<svg/>",
		);
		expect(fs.existsSync(path.join(out, "assets", ".hidden"))).toBe(false);
		expect(fs.existsSync(path.join(out, "notes"))).toBe(false);

		const meta = JSON.parse(await read(path.join(out, "meta.json")));
		expect(meta).toMatchObject({
			title: "Demo",
			fps: 10,
			frameCount: 30,
			width: 64,
			notesDir: null,
			audio: { state: "none" },
		});

		const page = await read(path.join(out, "index.html"));
		compileScript(page);
		expect(page).toContain('<base href="/films/demo/">');
		const config = pageConfig(page);
		expect(config).toMatchObject({
			audio: "",
			live: false,
			base: "/films/demo/",
			files: {
				"/films/demo/assets/data.json": 7,
				"/films/demo/assets/sub/logo.svg": 6,
				"/films/demo/assets/voice.wav": 4,
			},
			height: 360,
			playback: "realtime",
			showFps: true,
		});
		expect(config.meta).toMatch(/^\/films\/demo\/meta\.json\?v=\w+$/);
		expect(config.player).toMatch(/^\/films\/demo\/player\.js\?v=\w+$/);

		// The bundle loads the published engine, and the composition's paths
		// point into the site, not at this machine's folders.
		const code = await read(path.join(out, "player.js"));
		expect(code).toContain('"/films/demo/framefields-engine/player.js"');
		expect(code).toContain('"file:///films/demo/src/comp.ts"');
		expect(code).not.toContain(pathToFileURL(dir).href);
		expect(code).not.toContain("/@framefields/");
		const engine = await fs.promises.readdir(
			path.join(out, "framefields-engine"),
		);
		expect(engine).toContain("player.js");
		expect(engine.some((f) => f.endsWith(".mjs"))).toBe(false);
	}, 60_000);

	it("shares one engine and one copy of the files between compositions", async () => {
		const { comp, out } = await project();
		const result = await exportPreview(
			{
				intro: { entry: comp, title: "Intro" },
				outro: { entry: pathToFileURL(comp), audio: false },
			},
			{
				outDir: out,
				audio: { file: "assets/voice.wav", peaks: [0.5, 1] },
				files: ["assets/data.json", "assets/voice.wav"],
			},
		);
		expect(result.pages).toEqual({ intro: "/intro/", outro: "/outro/" });
		expect(result.files).toEqual(["/assets/data.json", "/assets/voice.wav"]);
		expect(fs.existsSync(path.join(out, "assets", "sub"))).toBe(false);

		const intro = await read(path.join(out, "intro", "index.html"));
		expect(pageConfig(intro)).toMatchObject({
			audio: "/assets/voice.wav",
			base: "/",
		});
		expect(
			JSON.parse(await read(path.join(out, "intro", "meta.json"))),
		).toMatchObject({
			title: "Intro",
			audio: { state: "ready", durationSec: 3, peaks: [0.5, 1] },
		});
		const outro = JSON.parse(await read(path.join(out, "outro", "meta.json")));
		expect(outro).toMatchObject({ title: "outro", audio: { state: "none" } });
		expect(await read(path.join(out, "outro", "player.js"))).toContain(
			'"/framefields-engine/player.js"',
		);
	}, 60_000);

	it("refuses files outside the project, and names that clash", async () => {
		const { comp, out } = await project();
		await expect(
			exportPreview({ entry: comp }, { outDir: out, files: "../elsewhere" }),
		).rejects.toThrow(/outside the project folder/);
		await expect(
			exportPreview({ "a/b": { entry: comp } }, { outDir: out }),
		).rejects.toThrow(/can't name a preview page/);
		await expect(
			exportPreview({ assets: { entry: comp } }, { outDir: out, audio: false }),
		).rejects.toThrow(/would overlap/);
	});
});

describe("fileUrl", () => {
	it("moves a project file's URL under the site's path", () => {
		const root = { dir: "/home/me/film", url: "/previews/" };
		expect(fileUrl("/home/me/film/src/a b#1.ts", root)).toBe(
			"file:///previews/src/a%20b%231.ts",
		);
		expect(fileUrl("/home/me/other/x.ts", root)).toBe(
			"file:///home/me/other/x.ts",
		);
		expect(fileUrl("/home/me/film/x.ts")).toBe("file:///home/me/film/x.ts");
	});
});

describe("previewPage", () => {
	it("keeps its settings inside the script element", () => {
		const page = previewPage({
			meta: "meta.json",
			audio: "audio.wav",
			player: 'player.js?v="1"',
			live: false,
			base: "/films/a/",
			files: { "/films/a/x</script>.json": 3 },
		});
		expect(page).not.toContain("x</script>");
		expect(pageConfig(page).files).toEqual({ "/films/a/x</script>.json": 3 });
		expect(page).toContain('<base href="/films/a/">');
		expect(page).toContain('src="player.js?v=&quot;1&quot;"');
		compileScript(page);
	});
});

describe("scaleToHeight", () => {
	it("draws the layout smaller, as a whole", () => {
		const comp = new Composition({ width: 1920, height: 1080, fps: 30 });
		const title = Layer.box({ id: "title", x: 100, y: 50, width: 400 });
		comp.add(title);
		expect(scaleToHeight(comp, 360)).toBe(true);
		expect([comp.width, comp.height]).toEqual([640, 360]);
		expect(comp.children).toHaveLength(1);
		expect(comp.children[0]).toMatchObject({
			kind: "box",
			width: 1920,
			height: 1080,
			anchorX: 0,
			anchorY: 0,
			scale: 1 / 3,
			children: [title],
		});
	});

	it("keeps sizes even, and leaves a composition at or below the height", () => {
		const odd = new Composition({ width: 1000, height: 1000, fps: 30 });
		scaleToHeight(odd, 333);
		expect([odd.width, odd.height]).toEqual([334, 334]);

		const small = new Composition({ width: 640, height: 360, fps: 30 });
		expect(scaleToHeight(small, 360)).toBe(false);
		expect(scaleToHeight(small, 720)).toBe(false);
		expect([small.width, small.height]).toEqual([640, 360]);
		expect(scaleToHeight({ fps: 10 }, 240)).toBe(false);
	});
});

describe("browser engine", () => {
	it("loads every entry, with no Node built-ins left unshimmed", async () => {
		const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "gf-engine-"));
		try {
			await buildEngine(dir);
			for (const name of [
				"framefields",
				"effects",
				"audio",
				"signals",
				"fonts",
				"player",
			]) {
				// Importing runs the module's top-level code, as a browser would.
				await import(pathToFileURL(path.join(dir, `${name}.js`)).href);
			}
		} finally {
			await fs.promises.rm(dir, { recursive: true, force: true });
		}
	}, 120_000);
});

describe("prebuiltEngineDir", () => {
	it("finds the engine at the dist root or beside dist/preview", async () => {
		const root = await fs.promises.mkdtemp(
			path.join(os.tmpdir(), "gf-engine-"),
		);
		try {
			const engine = path.join(root, "preview-engine");
			expect(prebuiltEngineDir(root)).toBeUndefined();

			await fs.promises.mkdir(engine, { recursive: true });
			await fs.promises.writeFile(path.join(engine, "player.js"), "export {}");
			expect(prebuiltEngineDir(root)).toBe(engine);

			const nested = path.join(root, "preview");
			await fs.promises.mkdir(nested, { recursive: true });
			expect(prebuiltEngineDir(nested)).toBe(engine);
		} finally {
			await fs.promises.rm(root, { recursive: true, force: true });
		}
	});
});
