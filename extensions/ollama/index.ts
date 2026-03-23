import {
  definePluginEntry,
  type OpenClawPluginApi,
  type ProviderAuthContext,
  type ProviderAuthMethodNonInteractiveContext,
  type ProviderAuthResult,
  type ProviderDiscoveryContext,
} from "openclaw/plugin-sdk/core";
import { OLLAMA_DEFAULT_BASE_URL, resolveOllamaApiBase } from "openclaw/plugin-sdk/provider-models";

const PROVIDER_ID = "ollama";
const DEFAULT_API_KEY = "ollama-local";

const OLLAMA_CATALOG_TTL_MS = 60_000;
type OllamaCacheEntry = {
  models: Array<{
    provider: string;
    id: string;
    name: string;
    contextWindow?: number;
    reasoning?: boolean;
    input?: ("text" | "image" | "document")[];
  }>;
  expiresAt: number;
  baseUrl: string;
};
let ollamaCatalogCache: OllamaCacheEntry | null = null;
export function __resetOllamaCatalogCacheForTest(): void {
  ollamaCatalogCache = null;
}

async function loadProviderSetup() {
  return await import("openclaw/plugin-sdk/ollama-setup");
}

export default definePluginEntry({
  id: "ollama",
  name: "Ollama Provider",
  description: "Bundled Ollama provider plugin",
  register(api: OpenClawPluginApi) {
    api.registerProvider({
      id: PROVIDER_ID,
      label: "Ollama",
      docsPath: "/providers/ollama",
      envVars: ["OLLAMA_API_KEY"],
      auth: [
        {
          id: "local",
          label: "Ollama",
          hint: "Cloud and local open models",
          kind: "custom",
          run: async (ctx: ProviderAuthContext): Promise<ProviderAuthResult> => {
            const providerSetup = await loadProviderSetup();
            const result = await providerSetup.promptAndConfigureOllama({
              cfg: ctx.config,
              prompter: ctx.prompter,
            });
            return {
              profiles: [
                {
                  profileId: "ollama:default",
                  credential: {
                    type: "api_key",
                    provider: PROVIDER_ID,
                    key: DEFAULT_API_KEY,
                  },
                },
              ],
              configPatch: result.config,
            };
          },
          runNonInteractive: async (ctx: ProviderAuthMethodNonInteractiveContext) => {
            const providerSetup = await loadProviderSetup();
            return await providerSetup.configureOllamaNonInteractive({
              nextConfig: ctx.config,
              opts: ctx.opts,
              runtime: ctx.runtime,
            });
          },
        },
      ],
      discovery: {
        order: "late",
        run: async (ctx: ProviderDiscoveryContext) => {
          const explicit = ctx.config.models?.providers?.ollama;
          const hasExplicitModels = Array.isArray(explicit?.models) && explicit.models.length > 0;
          const ollamaKey = ctx.resolveProviderApiKey(PROVIDER_ID).apiKey;
          if (hasExplicitModels && explicit) {
            return {
              provider: {
                ...explicit,
                baseUrl:
                  typeof explicit.baseUrl === "string" && explicit.baseUrl.trim()
                    ? resolveOllamaApiBase(explicit.baseUrl)
                    : OLLAMA_DEFAULT_BASE_URL,
                api: explicit.api ?? "ollama",
                apiKey: ollamaKey ?? explicit.apiKey ?? DEFAULT_API_KEY,
              },
            };
          }

          const providerSetup = await loadProviderSetup();
          const provider = await providerSetup.buildOllamaProvider(explicit?.baseUrl, {
            quiet: !ollamaKey && !explicit,
          });
          if (provider.models.length === 0 && !ollamaKey && !explicit?.apiKey) {
            return null;
          }
          return {
            provider: {
              ...provider,
              apiKey: ollamaKey ?? explicit?.apiKey ?? DEFAULT_API_KEY,
            },
          };
        },
      },
      wizard: {
        setup: {
          choiceId: "ollama",
          choiceLabel: "Ollama",
          choiceHint: "Cloud and local open models",
          groupId: "ollama",
          groupLabel: "Ollama",
          groupHint: "Cloud and local open models",
          methodId: "local",
        },
        modelPicker: {
          label: "Ollama (custom)",
          hint: "Detect models from a local or remote Ollama instance",
          methodId: "local",
        },
      },
      onModelSelected: async ({ config, model, prompter }) => {
        if (!model.startsWith("ollama/")) {
          return;
        }
        const providerSetup = await loadProviderSetup();
        await providerSetup.ensureOllamaModelPulled({ config, model, prompter });
      },
      augmentModelCatalog: async (ctx) => {
        const now = Date.now();
        const resolvedBaseUrl = resolveOllamaApiBase(
          ctx.config?.models?.providers?.ollama?.baseUrl,
        );

        if (ollamaCatalogCache?.baseUrl === resolvedBaseUrl && now < ollamaCatalogCache.expiresAt) {
          console.log(
            `[ollama] augmentModelCatalog: cache hit (${ollamaCatalogCache.models.length} models, expires in ${Math.round((ollamaCatalogCache.expiresAt - now) / 1000)}s)`,
          );
          return ollamaCatalogCache.models;
        }

        console.log(`[ollama] augmentModelCatalog: cache miss, fetching from ${resolvedBaseUrl}`);
        try {
          const providerSetup = await loadProviderSetup();
          const provider = await providerSetup.buildOllamaProvider(
            ctx.config?.models?.providers?.ollama?.baseUrl,
            { quiet: true },
          );
          const models = provider.models.map((m) => ({
            provider: PROVIDER_ID,
            id: m.id,
            name: m.name ?? m.id,
            contextWindow: typeof m.contextWindow === "number" ? m.contextWindow : undefined,
            reasoning: typeof m.reasoning === "boolean" ? m.reasoning : undefined,
            input: Array.isArray(m.input) ? m.input : undefined,
          }));
          console.log(
            `[ollama] augmentModelCatalog: fetched ${models.length} model(s), caching for ${OLLAMA_CATALOG_TTL_MS / 1000}s`,
          );
          ollamaCatalogCache = {
            models,
            expiresAt: now + OLLAMA_CATALOG_TTL_MS,
            baseUrl: resolvedBaseUrl,
          };
          return models;
        } catch (err) {
          const stale = ollamaCatalogCache?.models ?? [];
          console.warn(
            `[ollama] augmentModelCatalog: fetch failed (${String(err)}), returning ${stale.length} stale model(s)`,
          );
          return stale;
        }
      },
    });
  },
});
