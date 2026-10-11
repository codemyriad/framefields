/**
 * Track mattes: a node's `matte` is hoisted out into its own layer right
 * after it, so layout places it like any sibling and the renderer can draw
 * owner and matte separately and combine them.
 */
import type { LayoutNode, MatteMode } from "./schema.js";

type AnyNode = LayoutNode & Record<string, unknown>;

const GEOMETRY = ["x", "y", "width", "height"] as const;

function hoistOne(owner: AnyNode): AnyNode[] {
	const matte = owner.matte as { node: AnyNode; mode?: MatteMode } | undefined;
	const { matte: _drop, ...rest } = owner;
	const self = { ...rest } as AnyNode;
	if (Array.isArray(self.children)) {
		self.children = hoistMattes(self.children as AnyNode[]) as never;
	}
	if (!matte?.node) return [self];

	const src = matte.node;
	const id = (src.id as string | undefined) ?? `${String(owner.id)}__matte`;
	const node: AnyNode = {
		position: "absolute",
		...src,
		id,
		matteOf: String(owner.id),
		matteMode: matte.mode ?? "alpha",
	};
	// With no geometry of its own the matte covers its owner's box.
	if (GEOMETRY.every((k) => src[k] === undefined)) {
		for (const k of GEOMETRY) {
			if (typeof owner[k] === "number") node[k] = owner[k];
		}
	}
	// Timed with the owner: its keyframes start when the owner appears.
	const ownerStart =
		typeof owner.startFrame === "number" ? owner.startFrame : 0;
	node.startFrame =
		ownerStart + (typeof src.startFrame === "number" ? src.startFrame : 0);
	if (src.durationFrames === undefined && owner.durationFrames !== undefined) {
		node.durationFrames = owner.durationFrames;
	}
	if (Array.isArray(node.children)) {
		node.children = hoistMattes(node.children as AnyNode[]) as never;
	}
	self.matteId = id;
	return [self, node];
}

/** Returns the tree with every `matte` turned into a sibling matte layer. */
export function hoistMattes<T extends LayoutNode>(nodes: readonly T[]): T[] {
	return nodes.flatMap((n) => hoistOne(n as AnyNode)) as unknown as T[];
}
