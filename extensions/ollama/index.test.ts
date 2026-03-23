import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createTestPluginApi } from "../../test/helpers/extensions/plugin-api.js";
import plugin, { __resetOllamaCatalogCacheForTest } from "./index.js";

const promptAndConfigureOllamaMock = vi.hoisted(() =>
  vi.fn(async () => ({
    config: {
      models: {
        providers: {
          ollama: {
            baseUrl: "http://127.0.0.1:11434",
            api: "ollama",
            models: [],
          },
        },
      },
    },
  })),
);
const ensureOllamaModelPulledMock = vi.hoisted(() => vi.fn(async () => {}));
const buildOllamaProviderMock = vi.hoisted(() => vi.fn());

vi.mock("openclaw/plugin-sdk/ollama-setup", () => ({
  promptAndConfigureOllama: promptAndConfigureOllamaMock,
  ensureOllamaModelPulled: ensureOllamaModelPulledMock,
  configureOllamaNonInteractive: vi.fn(),
  buildOllamaProvider: buildOllamaProviderMock,
}));

function registerProvider() {
  const registerProviderMock = vi.fn();

  plugin.register(
    createTestPluginApi({
      id: "ollama",
      name: "Ollama",
      source: "test",
      config: {},
      runtime: {} as never,
      registerProvider: registerProviderMock,
    }),
  );

  expect(registerProviderMock).toHaveBeenCalledTimes(1);
  return registerProviderMock.mock.calls[0]?.[0];
}

describe("ollama plugin", () => {
  it("does not preselect a default model during provider auth setup", async () => {
    const provider = registerProvider();

    const result = await provider.auth[0].run({
      config: {},
      prompter: {} as never,
    });

    expect(promptAndConfigureOllamaMock).toHaveBeenCalledWith({
      cfg: {},
      prompter: {},
    });
    expect(result.configPatch).toEqual({
      models: {
        providers: {
          ollama: {
            baseUrl: "http://127.0.0.1:11434",
            api: "ollama",
            models: [],
          },
        },
      },
    });
    expect(result.defaultModel).toBeUndefined();
  });

  it("pulls the model the user actually selected", async () => {
    const provider = registerProvider();
    const config = {
      models: {
        providers: {
          ollama: {
            baseUrl: "http://127.0.0.1:11434",
            models: [],
          },
        },
      },
    };
    const prompter = {} as never;

    await provider.onModelSelected?.({
      config,
      model: "ollama/glm-4.7-flash",
      prompter,
    });

    expect(ensureOllamaModelPulledMock).toHaveBeenCalledWith({
      config,
      model: "ollama/glm-4.7-flash",
      prompter,
    });
  });
});

describe("ollama augmentModelCatalog TTL cache", () => {
  const BASE_CTX = {
    env: process.env,
    entries: [],
    config: {
      models: { providers: { ollama: { baseUrl: "http://127.0.0.1:11434" } } },
    },
  } as const;

  beforeEach(() => {
    vi.useFakeTimers();
    __resetOllamaCatalogCacheForTest();
    buildOllamaProviderMock.mockReset();
    buildOllamaProviderMock.mockResolvedValue({
      models: [
        { id: "llama3.2", name: "Llama 3.2" },
        { id: "mistral", name: "Mistral" },
      ],
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("fetches models on first call", async () => {
    const provider = registerProvider();
    const result = await provider.augmentModelCatalog?.(BASE_CTX);

    expect(buildOllamaProviderMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ provider: "ollama", id: "llama3.2" }),
        expect.objectContaining({ provider: "ollama", id: "mistral" }),
      ]),
    );
  });

  it("returns cached result within TTL", async () => {
    const provider = registerProvider();
    await provider.augmentModelCatalog?.(BASE_CTX);
    vi.advanceTimersByTime(30_000);
    await provider.augmentModelCatalog?.(BASE_CTX);

    expect(buildOllamaProviderMock).toHaveBeenCalledTimes(1);
  });

  it("re-fetches after TTL expires", async () => {
    const provider = registerProvider();
    await provider.augmentModelCatalog?.(BASE_CTX);
    vi.advanceTimersByTime(61_000);
    await provider.augmentModelCatalog?.(BASE_CTX);

    expect(buildOllamaProviderMock).toHaveBeenCalledTimes(2);
  });

  it("re-fetches when baseUrl changes", async () => {
    const provider = registerProvider();
    await provider.augmentModelCatalog?.(BASE_CTX);
    await provider.augmentModelCatalog?.({
      ...BASE_CTX,
      config: {
        models: { providers: { ollama: { baseUrl: "http://other-host:11434" } } },
      },
    });

    expect(buildOllamaProviderMock).toHaveBeenCalledTimes(2);
  });

  it("returns stale cache on error when cache is populated", async () => {
    const provider = registerProvider();
    await provider.augmentModelCatalog?.(BASE_CTX);

    vi.advanceTimersByTime(61_000);
    buildOllamaProviderMock.mockRejectedValue(new Error("connection refused"));
    const result = await provider.augmentModelCatalog?.(BASE_CTX);

    expect(result).toEqual(expect.arrayContaining([expect.objectContaining({ id: "llama3.2" })]));
  });

  it("returns empty array on error when cache is empty", async () => {
    buildOllamaProviderMock.mockRejectedValue(new Error("connection refused"));
    const provider = registerProvider();
    const result = await provider.augmentModelCatalog?.(BASE_CTX);

    expect(result).toEqual([]);
  });
});
