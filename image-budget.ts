/**
 * Outgoing request-body image budget for Ollama Cloud.
 *
 * Ollama Cloud rejects any `/v1/chat/completions` request whose body exceeds
 * 16 MiB with `400 failed to read request body`, before any model sees it
 * (verified against the live API: 16,775,320 bytes is read, 16,798,104 bytes is
 * rejected). Pi re-sends the entire conversation, including every historical
 * inline image, on each turn, and only bounds a single image (2000x2000,
 * 4.5 MB base64) - never the total. A vision session therefore crosses the cap
 * after a handful of screenshots and then fails on every subsequent turn.
 *
 * When the serialized body exceeds the budget, this module:
 *
 * 1. De-duplicates repeated images, keeping the newest copy of each identical
 *    payload (re-reading the same screenshot across turns is common) and
 *    replacing older copies with placeholder text.
 * 2. If the body is still over budget, replaces the oldest remaining images
 *    with placeholder text, keeping the newest (a vision turn usually refers to
 *    the most recent screenshot).
 *
 * A replaced image whose source file is recoverable (a `read` tool result or an
 * `@file` attachment) gets a path-bearing placeholder so the model can re-read
 * it on demand instead of losing it for the rest of the session.
 */

/**
 * Ollama Cloud's hard request-body cap in bytes. Bodies above this are rejected
 * with HTTP 400 before parsing, so it is a ceiling, not a target.
 */
export const OLLAMA_MAX_REQUEST_BYTES = 16 * 1024 * 1024;

/** Default budget: 2 MiB below the cap, leaving room for headers and JSON escaping. */
export const DEFAULT_MAX_REQUEST_BYTES = 14 * 1024 * 1024;

/** Placeholder used when a dropped image's source path is unknown. */
export const IMAGE_OMITTED_TEXT = "[image omitted: Ollama Cloud request body limit]";

/**
 * Placeholder for a dropped image whose file path is known: the model can
 * recover the image with the `read` tool instead of losing it for the session.
 */
export function imageOmittedWithPath(path: string): string {
  return `[image omitted; re-read with read ${path}]`;
}

/** Placeholder for an older copy of an image that also appears later in the request. */
export const DUPLICATE_IMAGE_TEXT = "[duplicate image omitted: identical to a newer copy in this request]";

/**
 * Duplicate placeholder for an image whose file path is known. The path hint
 * still matters because budget trimming may later remove the newer copy.
 */
export function duplicateImageWithPath(path: string): string {
  return `[duplicate image omitted: identical to a newer copy in this request; re-read with read ${path}]`;
}

export interface ImageTrimResult {
  /** The payload to send, with repeated or over-budget images replaced by placeholder text. */
  payload: unknown;
  /** Serialized size of the original payload, in bytes. */
  beforeBytes: number;
  /** Serialized size of the returned payload, in bytes. */
  afterBytes: number;
  /** Number of distinct inline images found in the original payload. */
  imageCount: number;
  /** Older copies of an image that appears later in the payload. */
  deduplicated: number;
  /** Images replaced by the omission placeholder to fit the budget. */
  dropped: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function serializedBytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value) ?? "", "utf8");
}

function imageUrl(part: Record<string, unknown>): string | undefined {
  const imageUrlField = part.image_url;
  if (!isRecord(imageUrlField)) return undefined;
  return typeof imageUrlField.url === "string" ? imageUrlField.url : undefined;
}

/** Argument keys image-producing tools use for their source path, most specific first. */
const PATH_ARGUMENT_KEYS = ["path", "file_path", "filePath", "file", "filename", "image"] as const;

/** `<file name="/abs/path">` tags pi prepends to @file attachment text. */
const FILE_TAG_PATTERN = /<file\s+name="([^"]+)"/g;

function pathFromArguments(argumentsValue: unknown): string | undefined {
  if (!isRecord(argumentsValue)) return undefined;
  for (const key of PATH_ARGUMENT_KEYS) {
    const value = argumentsValue[key];
    if (typeof value === "string" && value.length > 0) return value;
  }
  return undefined;
}

/** Tool path from a serialized OpenAI tool call: `function.arguments` is a JSON string. */
function pathFromToolCall(call: Record<string, unknown>): string | undefined {
  const fn = call.function;
  if (!isRecord(fn) || typeof fn.arguments !== "string") return undefined;
  try {
    return pathFromArguments(JSON.parse(fn.arguments));
  } catch {
    // Malformed arguments cannot name a file; fall back to the generic placeholder.
    return undefined;
  }
}

interface ImageRef {
  /** Cloned `image_url` part to replace. */
  part: Record<string, unknown>;
  /** Source file path, when recoverable. */
  path?: string;
}

/**
 * Associate every inline image with a source path when one is recoverable:
 *
 * - Tool-result images: the serialized OpenAI `assistant.tool_calls[].function.arguments`
 *   (a JSON string), the `tool` result's `tool_call_id`, then the image-only
 *   user message pi builds from consecutive tool results
 *   ("Attached image(s) from tool result:").
 * - `@file` attachments: the `<file name="...">` tags pi prepends to the text
 *   part of the same user message.
 *
 * Images without a recoverable source fall back to the generic placeholder.
 * Inputs are the already-cloned messages, so the returned parts are safe to mutate.
 */
function collectImageRefs(messages: unknown[]): ImageRef[] {
  // tool call id -> source path
  const toolPaths = new Map<string, string>();
  const refs: ImageRef[] = [];
  // Tool call paths for the tool results seen since the last assistant tool call,
  // in order, waiting to be paired with the image message they produced.
  let pendingToolPaths: (string | undefined)[] = [];

  for (const message of messages) {
    if (!isRecord(message)) continue;

    if (message.role === "assistant") {
      const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
      for (const call of toolCalls) {
        if (!isRecord(call) || typeof call.id !== "string") continue;
        const path = pathFromToolCall(call);
        if (path) toolPaths.set(call.id, path);
      }
      // A synthetic "I have processed the tool results." bridge has no
      // tool_calls and must not reset the pairing.
      if (toolCalls.length > 0) pendingToolPaths = [];
      continue;
    }

    if (message.role === "tool") {
      const id = message.tool_call_id;
      pendingToolPaths.push(typeof id === "string" ? toolPaths.get(id) : undefined);
      continue;
    }
    if (message.role !== "user" || !Array.isArray(message.content)) continue;

    const images = message.content.filter((p): p is Record<string, unknown> => isRecord(p) && p.type === "image_url");
    if (images.length === 0) continue;

    const filePaths: string[] = [];
    for (const part of message.content) {
      if (!isRecord(part) || part.type !== "text" || typeof part.text !== "string") continue;
      for (const match of part.text.matchAll(FILE_TAG_PATTERN)) filePaths.push(match[1]);
    }

    images.forEach((part, index) => {
      refs.push({ part, path: filePaths[index] ?? pendingToolPaths[index] });
    });
    pendingToolPaths = [];
  }

  return refs;
}

/** Replace an image part with text in place, returning the resulting byte delta. */
function replaceImageWithText(ref: ImageRef, text: string): number {
  const delta = serializedBytes(ref.part) - serializedBytes({ type: "text", text });
  delete ref.part.image_url;
  ref.part.type = "text";
  ref.part.text = text;
  return delta;
}

/**
 * Trim an OpenAI-completions payload until its serialized size fits
 * `budgetBytes`. Repeated images are de-duplicated first (newest copy kept); if
 * the body is still over budget, the oldest remaining images are dropped, newest
 * kept. The input payload is never mutated: pi reuses the payload object across
 * retries.
 *
 * Returns undefined when there is nothing to change (no messages, no images, or
 * already within budget). When even de-duplicating and dropping every image
 * leaves the body over budget (oversized text), the reduced payload is still
 * returned.
 */
export function trimImagesToBudget(
  payload: unknown,
  budgetBytes: number = DEFAULT_MAX_REQUEST_BYTES,
): ImageTrimResult | undefined {
  if (!isRecord(payload) || !Array.isArray(payload.messages)) return undefined;

  const beforeBytes = serializedBytes(payload);
  if (beforeBytes <= budgetBytes) return undefined;

  // Clone the message list and each content array, cloning image parts so the
  // originals (and the caller's payload) stay untouched.
  const messages = payload.messages.map((message) => {
    if (!isRecord(message) || !Array.isArray(message.content)) return message;
    return {
      ...message,
      content: message.content.map((part) => (isRecord(part) && part.type === "image_url" ? { ...part } : part)),
    };
  });

  const refs = collectImageRefs(messages);
  if (refs.length === 0) return undefined;

  const next: Record<string, unknown> = { ...payload, messages };
  let afterBytes = beforeBytes;

  // 1. De-duplicate: keep the newest copy of each identical image payload and
  // replace every earlier copy. Swapping one part for another leaves the
  // surrounding JSON (commas, brackets) unchanged, so the byte delta is exact.
  const newestIndexByUrl = new Map<string, number>();
  refs.forEach((ref, index) => {
    const url = imageUrl(ref.part);
    if (url) newestIndexByUrl.set(url, index);
  });
  let deduplicated = 0;
  refs.forEach((ref, index) => {
    const url = imageUrl(ref.part);
    if (!url || newestIndexByUrl.get(url) === index) return;
    const text = ref.path ? duplicateImageWithPath(ref.path) : DUPLICATE_IMAGE_TEXT;
    afterBytes -= replaceImageWithText(ref, text);
    deduplicated += 1;
  });

  // 2. Drop the oldest remaining images until the body fits.
  let dropped = 0;
  for (const ref of refs) {
    if (afterBytes <= budgetBytes) break;
    if (ref.part.type !== "image_url") continue; // already replaced as a duplicate
    const text = ref.path ? imageOmittedWithPath(ref.path) : IMAGE_OMITTED_TEXT;
    afterBytes -= replaceImageWithText(ref, text);
    dropped += 1;
  }

  if (deduplicated === 0 && dropped === 0) return undefined;
  return { payload: next, beforeBytes, afterBytes, imageCount: refs.length, deduplicated, dropped };
}
