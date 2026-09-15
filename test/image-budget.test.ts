import { describe, expect, it } from "vitest";
import {
  DEFAULT_MAX_REQUEST_BYTES,
  IMAGE_OMITTED_TEXT,
  OLLAMA_MAX_REQUEST_BYTES,
  trimImagesToBudget,
} from "../image-budget.ts";

// --- Helpers ---

function imagePart(payloadBytes: number): { type: "image_url"; image_url: { url: string } } {
  return { type: "image_url", image_url: { url: `data:image/png;base64,${"A".repeat(payloadBytes)}` } };
}

/** Payload with one user message per image, each carrying a text part too. */
function payloadWithImages(imageBytes: number[]): { model: string; messages: Record<string, unknown>[] } {
  return {
    model: "gemma4:31b",
    messages: [
      { role: "system", content: "system prompt" },
      { role: "user", content: [{ type: "text", text: "describe the images" }] },
      ...imageBytes.map((bytes, index) => ({
        role: "user",
        content: [{ type: "text", text: `image ${index}` }, imagePart(bytes)],
      })),
    ],
  };
}

function bytes(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value) ?? "", "utf8");
}

function imageCount(payload: { messages: Record<string, unknown>[] }): number {
  let count = 0;
  for (const message of payload.messages) {
    if (!Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (typeof part === "object" && part !== null && (part as { type?: string }).type === "image_url") count += 1;
    }
  }
  return count;
}

function hasPlaceholder(payload: { messages: Record<string, unknown>[] }, messageIndex: number): boolean {
  const content = payload.messages[messageIndex]?.content;
  if (!Array.isArray(content)) return false;
  return content.some(
    (part) => typeof part === "object" && part !== null && (part as { text?: string }).text === IMAGE_OMITTED_TEXT,
  );
}

// ============================================================================
// trimImagesToBudget
// ============================================================================

describe("trimImagesToBudget", () => {
  const IMAGE_BYTES = 1024;

  it("exposes a default budget below Ollama's hard limit", () => {
    expect(OLLAMA_MAX_REQUEST_BYTES).toBe(16 * 1024 * 1024);
    expect(DEFAULT_MAX_REQUEST_BYTES).toBeLessThan(OLLAMA_MAX_REQUEST_BYTES);
  });

  it("returns undefined for a payload without a messages array", () => {
    expect(trimImagesToBudget({ model: "x" }, 1)).toBeUndefined();
    expect(trimImagesToBudget(null, 1)).toBeUndefined();
    expect(trimImagesToBudget([1, 2, 3], 1)).toBeUndefined();
  });

  it("returns undefined when the body is already within budget", () => {
    const payload = payloadWithImages([IMAGE_BYTES, IMAGE_BYTES]);
    expect(trimImagesToBudget(payload, bytes(payload) + 1)).toBeUndefined();
  });

  it("returns undefined when over budget but there are no images to drop", () => {
    const payload = { model: "x", messages: [{ role: "user", content: "long text".repeat(1000) }] };
    expect(trimImagesToBudget(payload, 10)).toBeUndefined();
  });

  it("drops the oldest image first and keeps the newest", () => {
    const payload = payloadWithImages([IMAGE_BYTES, IMAGE_BYTES]);
    const partBytes = bytes(imagePart(IMAGE_BYTES));
    const replacementBytes = bytes({ type: "text", text: IMAGE_OMITTED_TEXT });
    // Room for exactly one of the two images.
    const budget = bytes(payload) - partBytes + replacementBytes + 5;

    const result = trimImagesToBudget(payload, budget);
    const trimmed = result?.payload as { messages: Record<string, unknown>[] };

    expect(result).toBeDefined();
    expect(result?.dropped).toBe(1);
    expect(result?.imageCount).toBe(2);
    expect(result?.afterBytes).toBeLessThanOrEqual(budget);
    expect(hasPlaceholder(trimmed, 2)).toBe(true); // oldest image (message index 2)
    expect(hasPlaceholder(trimmed, 3)).toBe(false); // newest image kept
    expect(imageCount(trimmed)).toBe(1);
  });

  it("drops every image when the budget cannot fit any", () => {
    const payload = payloadWithImages([IMAGE_BYTES, IMAGE_BYTES, IMAGE_BYTES]);
    const result = trimImagesToBudget(payload, 1);
    const trimmed = result?.payload as { messages: Record<string, unknown>[] };

    expect(result?.dropped).toBe(3);
    expect(result?.imageCount).toBe(3);
    expect(imageCount(trimmed)).toBe(0);
    for (const index of [2, 3, 4]) expect(hasPlaceholder(trimmed, index)).toBe(true);
  });

  it("does not mutate the input payload", () => {
    const payload = payloadWithImages([IMAGE_BYTES, IMAGE_BYTES]);
    const snapshot = JSON.stringify(payload);
    const result = trimImagesToBudget(payload, 1);

    expect(result).toBeDefined();
    expect(JSON.stringify(payload)).toBe(snapshot);
    expect(imageCount(payload)).toBe(2);
  });

  it("returns a differently sized payload that fits the budget", () => {
    const payload = payloadWithImages([IMAGE_BYTES, IMAGE_BYTES, IMAGE_BYTES]);
    const result = trimImagesToBudget(payload, bytes(payload) - 1);

    expect(result).toBeDefined();
    expect(result?.beforeBytes).toBe(bytes(payload));
    expect(result?.afterBytes).toBeLessThan(result?.beforeBytes ?? 0);
    expect(result?.afterBytes).toBeLessThanOrEqual(bytes(payload) - 1);
  });

  it("handles images sharing a message content array", () => {
    const payload = {
      model: "x",
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "two images" }, imagePart(IMAGE_BYTES), imagePart(IMAGE_BYTES)],
        },
      ],
    };
    const result = trimImagesToBudget(payload, 1);
    const trimmed = result?.payload as { messages: Record<string, unknown>[] };

    expect(result?.dropped).toBe(2);
    expect(imageCount(trimmed)).toBe(0);
    const content = trimmed.messages[0].content as { type: string; text?: string }[];
    expect(content[0].type).toBe("text");
    expect(content[1].text).toBe(IMAGE_OMITTED_TEXT);
    expect(content[2].text).toBe(IMAGE_OMITTED_TEXT);
  });
});
