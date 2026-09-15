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
 * This module trims the oldest inline images out of the outgoing
 * OpenAI-completions payload (keeping the newest, which are the most relevant)
 * until the serialized body fits the budget.
 */

/**
 * Ollama Cloud's hard request-body cap in bytes. Bodies above this are rejected
 * with HTTP 400 before parsing, so it is a ceiling, not a target.
 */
export const OLLAMA_MAX_REQUEST_BYTES = 16 * 1024 * 1024;

/** Default budget: 2 MiB below the cap, leaving room for headers and JSON escaping. */
export const DEFAULT_MAX_REQUEST_BYTES = 14 * 1024 * 1024;

/** Text substituted for an image dropped to satisfy the body budget. */
export const IMAGE_OMITTED_TEXT = "[image omitted: Ollama Cloud request body limit]";

export interface ImageTrimResult {
  /** The payload to send, with the oldest inline images replaced by placeholder text. */
  payload: unknown;
  /** Serialized size of the original payload, in bytes. */
  beforeBytes: number;
  /** Serialized size of the returned payload, in bytes. */
  afterBytes: number;
  /** Number of inline images found in the original payload. */
  imageCount: number;
  /** Number of inline images replaced by placeholder text. */
  dropped: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function serializedBytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value) ?? "", "utf8");
}

/**
 * Drop the oldest inline images from an OpenAI-completions payload until its
 * serialized size fits `budgetBytes`. The newest images are kept, since a
 * vision turn usually refers to the most recent screenshot. The input payload
 * is never mutated: pi reuses the payload object across retries.
 *
 * Returns undefined when there is nothing to trim (no messages, no images, or
 * already within budget). When even dropping every image leaves the body over
 * budget (oversized text), the trimmed payload is still returned.
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

  const imageParts: Record<string, unknown>[] = [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (isRecord(part) && part.type === "image_url") imageParts.push(part);
    }
  }
  if (imageParts.length === 0) return undefined;

  const next: Record<string, unknown> = { ...payload, messages };
  const replacementBytes = serializedBytes({ type: "text", text: IMAGE_OMITTED_TEXT });
  let afterBytes = beforeBytes;
  let dropped = 0;

  // Oldest first. Swapping one part for another leaves the surrounding JSON
  // (commas, brackets) unchanged, so the byte delta is exact.
  for (const part of imageParts) {
    if (afterBytes <= budgetBytes) break;
    afterBytes -= serializedBytes(part) - replacementBytes;
    delete part.image_url;
    part.type = "text";
    part.text = IMAGE_OMITTED_TEXT;
    dropped += 1;
  }
  if (dropped === 0) return undefined;

  return { payload: next, beforeBytes, afterBytes, imageCount: imageParts.length, dropped };
}
