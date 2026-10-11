/**
 * Minimal DOM stand-ins for libraries (lottie) that expect a browser page when
 * running headless: a 1920×1080 placeholder element, canvas element
 * properties, and an inert IntersectionObserver.
 */

export function createHeadlessDOMElement() {
	return {
		clientWidth: 1920,
		clientHeight: 1080,
		getBoundingClientRect: () => ({
			x: 0,
			y: 0,
			left: 0,
			top: 0,
			right: 1920,
			bottom: 1080,
			width: 1920,
			height: 1080,
		}),
		style: {},
		appendChild: () => {},
		removeChild: () => {},
		addEventListener: () => {},
		removeEventListener: () => {},
		getAttribute: () => null,
	};
}

/** Gives a canvas class the element API lottie reads, where it's missing. */
export function patchCanvasPrototypeForDOM(
	proto: Record<string, unknown>,
	parentElement: unknown,
): void {
	if (typeof proto.getBoundingClientRect === "undefined") {
		proto.getBoundingClientRect = function (this: {
			width?: number;
			height?: number;
		}) {
			const width = this.width || 512;
			const height = this.height || 512;
			return {
				x: 0,
				y: 0,
				left: 0,
				top: 0,
				right: width,
				bottom: height,
				width,
				height,
			};
		};
	}

	if (!("style" in proto)) {
		Object.defineProperty(proto, "style", {
			get(this: { _style?: Record<string, unknown> }) {
				this._style ??= {};
				return this._style;
			},
			configurable: true,
		});
	}

	if (!("clientWidth" in proto)) {
		Object.defineProperty(proto, "clientWidth", {
			get(this: { width?: number }) {
				return this.width || 512;
			},
			configurable: true,
		});
	}

	if (!("clientHeight" in proto)) {
		Object.defineProperty(proto, "clientHeight", {
			get(this: { height?: number }) {
				return this.height || 512;
			},
			configurable: true,
		});
	}

	if (!("parentElement" in proto)) {
		Object.defineProperty(proto, "parentElement", {
			get() {
				return parentElement;
			},
			configurable: true,
		});
	}

	if (typeof proto.addEventListener === "undefined") {
		proto.addEventListener = () => {};
	}
	if (typeof proto.removeEventListener === "undefined") {
		proto.removeEventListener = () => {};
	}
	if (typeof proto.getAttribute === "undefined") {
		proto.getAttribute = () => null;
	}
}

export function installIntersectionObserverStub(): void {
	if (typeof globalThis.IntersectionObserver === "undefined") {
		(globalThis as unknown as Record<string, unknown>).IntersectionObserver =
			class IntersectionObserver {
				observe() {}
				unobserve() {}
				disconnect() {}
			};
	}
}
