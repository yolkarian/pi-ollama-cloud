import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it } from "vitest";
import { getCloudApiKey, registerWebFetchTool, registerWebSearchTool } from "../web-tools.ts";

/**
 * getCloudApiKey() resolves the API key for the ollama-cloud provider through
 * pi's extension-facing model registry, with an environment fallback.
 */

const ENV_KEY = "env-fallback-key";

function createModelRegistry(apiKey: string | undefined) {
  return {
    async getApiKeyForProvider(provider: string): Promise<string | undefined> {
      expect(provider).toBe("ollama-cloud");
      return apiKey;
    },
  };
}

describe("web tool registration", () => {
  it("registers both web tools", () => {
    const registered: string[] = [];
    const pi = {
      registerTool(definition: { name: string }) {
        registered.push(definition.name);
      },
    } as unknown as ExtensionAPI;

    registerWebSearchTool(pi);
    registerWebFetchTool(pi);

    expect(registered).toEqual(["ollama_web_search", "ollama_web_fetch"]);
  });
});

describe("getCloudApiKey", () => {
  const originalEnvKey = process.env.OLLAMA_API_KEY;

  afterEach(() => {
    if (originalEnvKey === undefined) {
      delete process.env.OLLAMA_API_KEY;
    } else {
      process.env.OLLAMA_API_KEY = originalEnvKey;
    }
  });

  it("returns the provider API key when configured", async () => {
    const modelRegistry = createModelRegistry("stored-key");
    process.env.OLLAMA_API_KEY = ENV_KEY;

    const key = await getCloudApiKey(modelRegistry);
    expect(key).toBe("stored-key");
  });

  it("falls back to OLLAMA_API_KEY when the provider has no resolved key", async () => {
    const modelRegistry = createModelRegistry(undefined);
    process.env.OLLAMA_API_KEY = ENV_KEY;

    const key = await getCloudApiKey(modelRegistry);
    expect(key).toBe(ENV_KEY);
  });

  it("returns undefined when neither the provider nor environment has a key", async () => {
    const modelRegistry = createModelRegistry(undefined);
    delete process.env.OLLAMA_API_KEY;

    const key = await getCloudApiKey(modelRegistry);
    expect(key).toBeUndefined();
  });

  it("prefers the provider API key over OLLAMA_API_KEY", async () => {
    const modelRegistry = createModelRegistry("stored-key");
    process.env.OLLAMA_API_KEY = ENV_KEY;

    const key = await getCloudApiKey(modelRegistry);
    expect(key).toBe("stored-key");
  });
});
