import { debugLog } from "../debug-log.js";
import { installDOMGeometryGlobals } from "./dom-matrix.js";

export async function initHeadlessWebGPU(): Promise<void> {
	const { create, globals } = await import(/* webpackIgnore: true */ "webgpu");

	for (const [key, value] of Object.entries(globals)) {
		if (key === "navigator") continue;
		try {
			(globalThis as any)[key] = value;
		} catch (e) {
			Object.defineProperty(globalThis, key, {
				value,
				configurable: true,
				enumerable: true,
				writable: true,
			});
		}
	}

	if (!(globalThis as any).navigator?.gpu) {
		const gpu = create([]);
		debugLog("[webgpu] Created headless GPU");
		if (!(globalThis as any).navigator) {
			try {
				(globalThis as any).navigator = { gpu };
			} catch (e) {
				Object.defineProperty(globalThis, "navigator", {
					value: { gpu },
					configurable: true,
					enumerable: true,
					writable: true,
				});
			}
		} else {
			try {
				(globalThis as any).navigator.gpu = gpu;
			} catch (e) {
				Object.defineProperty((globalThis as any).navigator, "gpu", {
					value: gpu,
					configurable: true,
					enumerable: true,
					writable: true,
				});
			}
		}
	}

	installDOMGeometryGlobals();

	if (typeof (globalThis as any).createImageBitmap === "undefined") {
		(globalThis as any).createImageBitmap = async (source: any) => {
			console.error(
				"createImageBitmap called in headless mode. Source:",
				source,
			);
			throw new Error(
				"createImageBitmap is not supported in headless mode. Ensure isHeadless is set to true to use the sharp-based buffer path.",
			);
		};
	}
}
