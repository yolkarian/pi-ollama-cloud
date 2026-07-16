/**
 * Live smoke test for the Ollama Cloud web tools auth path.
 *
 * Regression coverage for the web tools' authentication path: the API key
 * may come from either auth.json or OLLAMA_API_KEY. The unit test in
 * test/web-tools.test.ts covers the resolution logic with a stub registry;
 * this script covers the roundtrip using pi's ModelRuntime-backed registry.
 *
 * Hits `${OLLAMA_BASE}/api/web_search` (defaults to
 * https://ollama.com, overridable via OLLAMA_API_BASE). Exits 0 on
 * success, 1 on any failure with a clear error message.
 */

import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { OLLAMA_BASE } from "../models.ts";
import { fetchJsonWithTimeout } from "../utils.ts";
import { getCloudApiKey } from "../web-tools.ts";

const TIMEOUT_MS = 15_000;

async function main(): Promise<void> {
  // Use the same provider-aware registry exposed to extension tool contexts.
  const runtime = await ModelRuntime.create({ modelsPath: null, allowModelNetwork: false });
  const modelRegistry = new ModelRegistry(runtime);
  modelRegistry.registerProvider("ollama-cloud", {
    name: "Ollama Cloud",
    baseUrl: `${OLLAMA_BASE}/v1`,
    apiKey: "$OLLAMA_API_KEY",
    api: "openai-completions",
    models: [
      {
        id: "smoke-test",
        name: "Smoke Test",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 1,
        maxTokens: 1,
      },
    ],
  });
  const apiKey = await getCloudApiKey(modelRegistry);

  if (!apiKey) {
    console.error(
      "FAIL: no API key resolved. Set OLLAMA_API_KEY or add an ollama-cloud entry to auth.json.",
    );
    process.exit(1);
  }
  console.log("PASS: getCloudApiKey resolved a key");

  // Hit the /api/web_search endpoint with the resolved key.
  const result = await fetchJsonWithTimeout<{ results?: Array<{ title: string }> }>(
    `${OLLAMA_BASE}/api/web_search`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: "Ollama", max_results: 1 }),
    },
    TIMEOUT_MS,
  );

  if (!result.ok) {
    console.error(`FAIL: /api/web_search returned ${result.status}: ${result.error ?? "<no body>"}`);
    process.exit(1);
  }
  console.log(`PASS: /api/web_search responded ${result.status}`);

  const data = result.data;
  if (!data || !Array.isArray(data.results) || data.results.length === 0) {
    console.error("FAIL: /api/web_search response missing results array");
    process.exit(1);
  }
  console.log(`PASS: /api/web_search returned ${data.results.length} result(s)`);
}

main().catch((err) => {
  console.error(`FAIL: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
