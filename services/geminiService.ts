import { GoogleGenAI } from "@google/genai";

export type GeminiModelProfile = "flash" | "pro" | "ultra" | "balanced";

export interface GenerationOptions {
  model?: string;
  models?: string[];
  profile?: GeminiModelProfile;
  useSearch?: boolean;
  temperature?: number;
  maxOutputTokens?: number;
  topP?: number;
  topK?: number;
  systemInstruction?: string;
  timeoutMs?: number;
  retries?: number;
  operation?: string;
}

interface ModelCallContext {
  operation: string;
  model?: string;
}

export class GeminiAppError extends Error {
  code: string;
  userMessage: string;
  statusCode?: number;
  retryable: boolean;
  operation?: string;
  model?: string;

  constructor(params: {
    code: string;
    userMessage: string;
    statusCode?: number;
    retryable?: boolean;
    operation?: string;
    model?: string;
    cause?: unknown;
  }) {
    super(params.userMessage);
    this.name = "GeminiAppError";
    this.code = params.code;
    this.userMessage = params.userMessage;
    this.statusCode = params.statusCode;
    this.retryable = Boolean(params.retryable);
    this.operation = params.operation;
    this.model = params.model;
    if (params.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = params.cause;
    }
  }
}

const DEFAULT_TIMEOUT_MS = 45_000;
const DEFAULT_RETRIES = 2;

const MODEL_PROFILES: Record<GeminiModelProfile, string[]> = {
  flash: ["gemini-2.5-flash", "gemini-2.0-flash-exp", "gemini-3-flash-preview"],
  pro: ["gemini-2.5-pro", "gemini-1.5-pro"],
  ultra: ["gemini-2.5-pro", "gemini-2.5-flash"],
  balanced: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"],
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const getRuntimeEnv = (): Record<string, string | undefined> => {
  const meta = import.meta as ImportMeta & { env?: Record<string, string | undefined> };
  return meta.env ?? {};
};

const getApiKey = (): string => {
  const sessionKey = sessionStorage.getItem("geminiApiKey")?.trim();
  if (sessionKey) {
    return sessionKey;
  }

  const env = getRuntimeEnv();
  const envKey = (env.VITE_GEMINI_API_KEY ?? "").trim();
  if (envKey) {
    return envKey;
  }

  if (typeof process !== "undefined" && process.env) {
    const processKey = (process.env.API_KEY ?? process.env.GEMINI_API_KEY ?? "").trim();
    if (processKey) {
      return processKey;
    }
  }

  throw new GeminiAppError({
    code: "MISSING_API_KEY",
    userMessage:
      "Kunci API Gemini belum diatur. Silakan isi di menu Panduan & Pengaturan.",
    retryable: false,
    operation: "auth",
  });
};

const getAI = () => {
  const apiKey = getApiKey();
  return { ai: new GoogleGenAI({ apiKey }) };
};

const getStatusCode = (error: unknown): number | undefined => {
  if (!error || typeof error !== "object") {
    return undefined;
  }

  const e = error as Record<string, unknown>;
  const direct = e.statusCode ?? e.status;
  if (typeof direct === "number") {
    return direct;
  }
  if (typeof direct === "string") {
    const numeric = Number(direct);
    if (!Number.isNaN(numeric)) {
      return numeric;
    }
  }

  const code = e.code;
  if (typeof code === "number") {
    return code;
  }

  const message = (e.message ?? "").toString();
  const match = message.match(/\b([45]\d{2})\b/);
  if (match) {
    return Number(match[1]);
  }
  return undefined;
};

const isRetryableStatus = (statusCode?: number): boolean => {
  return statusCode !== undefined && [408, 429, 500, 502, 503, 504].includes(statusCode);
};

const isModelUnavailableError = (statusCode: number | undefined, message: string): boolean => {
  const lower = message.toLowerCase();
  if (statusCode !== 400 && statusCode !== 404) {
    return false;
  }
  return lower.includes("model") && (lower.includes("not found") || lower.includes("unsupported") || lower.includes("invalid"));
};

const toGeminiAppError = (error: unknown, context: ModelCallContext): GeminiAppError => {
  if (error instanceof GeminiAppError) {
    return error;
  }

  const statusCode = getStatusCode(error);
  const message = error instanceof Error ? error.message : String(error ?? "Unknown error");
  const lower = message.toLowerCase();

  if (isModelUnavailableError(statusCode, message)) {
    return new GeminiAppError({
      code: "MODEL_UNAVAILABLE",
      userMessage: "Model Gemini yang dipilih tidak tersedia pada API key ini.",
      statusCode,
      retryable: false,
      operation: context.operation,
      model: context.model,
      cause: error,
    });
  }

  if (statusCode === 403 || lower.includes("api key not valid")) {
    return new GeminiAppError({
      code: "INVALID_API_KEY",
      userMessage: "Kunci API Gemini tidak valid atau tidak memiliki izin.",
      statusCode,
      retryable: false,
      operation: context.operation,
      model: context.model,
      cause: error,
    });
  }

  if (statusCode === 429) {
    return new GeminiAppError({
      code: "QUOTA_EXCEEDED",
      userMessage: "Kuota API Gemini terlampaui. Tunggu beberapa saat atau gunakan model lain.",
      statusCode,
      retryable: true,
      operation: context.operation,
      model: context.model,
      cause: error,
    });
  }

  if (lower.includes("timed out") || lower.includes("timeout") || lower.includes("aborted")) {
    return new GeminiAppError({
      code: "REQUEST_TIMEOUT",
      userMessage: "Permintaan ke Gemini timeout. Silakan coba lagi.",
      statusCode,
      retryable: true,
      operation: context.operation,
      model: context.model,
      cause: error,
    });
  }

  if (lower.includes("safety")) {
    return new GeminiAppError({
      code: "SAFETY_BLOCK",
      userMessage: "Permintaan diblokir oleh sistem keamanan model. Ubah prompt lalu coba lagi.",
      statusCode,
      retryable: false,
      operation: context.operation,
      model: context.model,
      cause: error,
    });
  }

  return new GeminiAppError({
    code: "GEMINI_REQUEST_FAILED",
    userMessage: "Terjadi kendala saat menghubungi Gemini. Silakan coba lagi.",
    statusCode,
    retryable: isRetryableStatus(statusCode),
    operation: context.operation,
    model: context.model,
    cause: error,
  });
};

export const getUserFriendlyErrorMessage = (error: unknown): string => {
  const appError = toGeminiAppError(error, { operation: "unknown" });
  return appError.userMessage;
};

const nowMs = (): number => {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
};

const withTimeout = async <T>(factory: () => Promise<T>, timeoutMs: number, context: ModelCallContext): Promise<T> => {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      factory(),
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => {
          reject(
            new GeminiAppError({
              code: "REQUEST_TIMEOUT",
              userMessage: "Permintaan ke Gemini timeout. Silakan coba lagi.",
              retryable: true,
              operation: context.operation,
              model: context.model,
            })
          );
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
};

const buildConfig = (options: GenerationOptions): Record<string, unknown> => {
  const config: Record<string, unknown> = {};
  if (typeof options.temperature === "number") {
    config.temperature = options.temperature;
  }
  if (typeof options.maxOutputTokens === "number") {
    config.maxOutputTokens = options.maxOutputTokens;
  }
  if (typeof options.topP === "number") {
    config.topP = options.topP;
  }
  if (typeof options.topK === "number") {
    config.topK = options.topK;
  }
  if (options.systemInstruction?.trim()) {
    config.systemInstruction = options.systemInstruction.trim();
  }
  if (options.useSearch) {
    config.tools = [{ googleSearch: {} }];
  }
  return config;
};

const normalizeText = (response: any): string => {
  const directText = typeof response?.text === "string" ? response.text : "";
  if (directText.trim()) {
    return directText;
  }

  const parts = response?.candidates?.[0]?.content?.parts;
  if (Array.isArray(parts)) {
    const merged = parts
      .map((part) => (part && typeof part.text === "string" ? part.text : ""))
      .join("")
      .trim();
    if (merged) {
      return merged;
    }
  }
  return "";
};

const appendCitations = (text: string, response: any): string => {
  const chunks = response?.candidates?.[0]?.groundingMetadata?.groundingChunks;
  if (!Array.isArray(chunks) || chunks.length === 0) {
    return text;
  }

  const citations = chunks
    .map((chunk: any) => chunk?.web?.uri)
    .filter((uri: unknown): uri is string => typeof uri === "string" && uri.length > 0);

  if (citations.length === 0) {
    return text;
  }

  const unique = Array.from(new Set(citations));
  return `${text}\n\n**Sumber:**\n${unique.map((uri) => `- ${uri}`).join("\n")}`;
};

const isSafetyBlocked = (response: any): boolean => {
  const candidate = response?.candidates?.[0];
  const finishReason = String(candidate?.finishReason ?? "").toLowerCase();
  if (finishReason.includes("safety") || finishReason.includes("blocked")) {
    return true;
  }

  const blockReason = String(candidate?.safetyRatings?.[0]?.blocked ?? "").toLowerCase();
  return blockReason === "true";
};

const logGeminiEvent = (payload: Record<string, unknown>) => {
  if (payload.ok) {
    console.info("[gemini]", payload);
    return;
  }
  console.error("[gemini]", payload);
};

const resolveCandidateModels = (options: GenerationOptions): string[] => {
  if (options.model?.trim()) {
    return [options.model.trim()];
  }
  if (options.models && options.models.length > 0) {
    return Array.from(new Set(options.models.map((m) => m.trim()).filter(Boolean)));
  }
  const profile = options.profile ?? "flash";
  return MODEL_PROFILES[profile];
};

export const generateGeminiContent = async (prompt: string, options: GenerationOptions = {}): Promise<string> => {
  const operation = options.operation ?? "generate-content";
  const retries = Math.max(0, options.retries ?? DEFAULT_RETRIES);
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const candidates = resolveCandidateModels(options);

  if (candidates.length === 0) {
    throw new GeminiAppError({
      code: "NO_MODEL_CANDIDATE",
      userMessage: "Tidak ada model Gemini yang tersedia untuk dipakai.",
      retryable: false,
      operation,
    });
  }

  const { ai } = getAI();
  let lastError: GeminiAppError | null = null;

  for (const model of candidates) {
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      const startedAt = nowMs();
      try {
        const response = await withTimeout(
          () =>
            ai.models.generateContent({
              model,
              contents: prompt,
              config: buildConfig(options),
            }),
          timeoutMs,
          { operation, model }
        );

        let text = normalizeText(response);
        if (isSafetyBlocked(response)) {
          throw new GeminiAppError({
            code: "SAFETY_BLOCK",
            userMessage: "Permintaan diblokir oleh sistem keamanan model. Ubah prompt lalu coba lagi.",
            retryable: false,
            operation,
            model,
          });
        }

        if (!text) {
          throw new GeminiAppError({
            code: "EMPTY_RESPONSE",
            userMessage: "Gemini tidak mengembalikan teks pada respons ini.",
            retryable: false,
            operation,
            model,
          });
        }

        if (options.useSearch) {
          text = appendCitations(text, response);
        }

        logGeminiEvent({
          ok: true,
          operation,
          model,
          attempt: attempt + 1,
          latencyMs: Math.round(nowMs() - startedAt),
          inputTokens: response?.usageMetadata?.promptTokenCount,
          outputTokens: response?.usageMetadata?.candidatesTokenCount,
        });

        return text;
      } catch (error) {
        const appError = toGeminiAppError(error, { operation, model });
        logGeminiEvent({
          ok: false,
          operation,
          model,
          attempt: attempt + 1,
          latencyMs: Math.round(nowMs() - startedAt),
          code: appError.code,
          statusCode: appError.statusCode,
        });

        const isLastAttempt = attempt >= retries;
        if (!isLastAttempt && appError.retryable) {
          const backoffMs = (2 ** attempt) * 300 + Math.floor(Math.random() * 200);
          await sleep(backoffMs);
          continue;
        }

        if (appError.code === "MODEL_UNAVAILABLE") {
          lastError = appError;
          break;
        }

        throw appError;
      }
    }
  }

  if (lastError) {
    throw lastError;
  }

  throw new GeminiAppError({
    code: "GEMINI_REQUEST_FAILED",
    userMessage: "Permintaan ke Gemini gagal untuk semua model kandidat.",
    retryable: false,
    operation,
  });
};

const extractJson = (text: string): string => {
  const fencedMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }

  const starts = ["{", "["];
  for (let i = 0; i < text.length; i += 1) {
    if (!starts.includes(text[i])) {
      continue;
    }
    for (let j = text.length - 1; j > i; j -= 1) {
      const chunk = text.slice(i, j + 1).trim();
      if (!chunk) {
        continue;
      }
      try {
        JSON.parse(chunk);
        return chunk;
      } catch {
        // keep searching
      }
    }
  }

  throw new GeminiAppError({
    code: "INVALID_JSON_RESPONSE",
    userMessage: "AI tidak mengembalikan format JSON yang valid.",
    retryable: false,
    operation: "parse-json",
  });
};

export const generatePhotoVideoPrompt = async (
  formData: any,
  activeMode: "manual" | "ai",
  options: GenerationOptions = {}
): Promise<{ photo: string; video: string }> => {
  const promptData = { ...formData };

  if (activeMode === "ai" && promptData.imageBase64) {
    promptData.imageBase64 = `[Gambar produk telah diunggah pengguna, ukuran: ${Math.round((promptData.imageBase64.length * 3) / 4 / 1024)} KB]`;
    delete promptData.mimeType;
  }

  if (activeMode === "manual") {
    if (promptData.modelImageBase64) {
      promptData.modelImageBase64 = `[Foto referensi MODEL diunggah pengguna, ukuran: ${Math.round((promptData.modelImageBase64.length * 3) / 4 / 1024)} KB]`;
    }
    delete promptData.modelImageMimeType;

    if (promptData.productImageBase64) {
      promptData.productImageBase64 = `[Foto referensi PRODUK diunggah pengguna, ukuran: ${Math.round((promptData.productImageBase64.length * 3) / 4 / 1024)} KB]`;
    }
    delete promptData.productImageMimeType;
  }

  const prompt = `Anda adalah Creative Director dan Prompt Engineer.\nMode: ${activeMode}\nInput:\n${JSON.stringify(
    promptData,
    null,
    2
  )}\n\nKembalikan JSON valid dengan key photo_prompt dan video_prompt.`;

  const responseText = await generateGeminiContent(prompt, {
    profile: "flash",
    operation: "generate-photo-video",
    ...options,
  });

  const parsed = JSON.parse(extractJson(responseText));
  if (!parsed?.photo_prompt || !parsed?.video_prompt) {
    throw new GeminiAppError({
      code: "INVALID_JSON_SHAPE",
      userMessage: "Format output AI tidak sesuai untuk prompt foto/video.",
      retryable: false,
      operation: "generate-photo-video",
    });
  }

  return {
    photo: String(parsed.photo_prompt),
    video: String(parsed.video_prompt),
  };
};

export const generateLaunchContent = async (
  formData: any,
  mode: "product" | "ad",
  options: GenerationOptions = {}
): Promise<string> => {
  const prompt = `Anda adalah Senior Copywriter. Mode: ${mode}. Data:\n${JSON.stringify(formData, null, 2)}`;
  return generateGeminiContent(prompt, {
    profile: "flash",
    operation: "generate-launch-content",
    ...options,
  });
};

export const generateLiveScript = async (formData: any, options: GenerationOptions = {}): Promise<string> => {
  const prompt = `Anda adalah Showrunner live streaming. Data:\n${JSON.stringify(
    formData,
    null,
    2
  )}\nKembalikan JSON valid dengan key timeline (array).`;

  const responseText = await generateGeminiContent(prompt, {
    profile: "flash",
    operation: "generate-live-script",
    ...options,
  });

  const jsonString = extractJson(responseText);
  const parsed = JSON.parse(jsonString);
  if (!Array.isArray(parsed?.timeline)) {
    throw new GeminiAppError({
      code: "INVALID_JSON_SHAPE",
      userMessage: "Format timeline dari AI tidak valid.",
      retryable: false,
      operation: "generate-live-script",
    });
  }
  return jsonString;
};

export const generatePepTalk = async (
  mood: string,
  reason: string,
  options: GenerationOptions = {}
): Promise<string> => {
  const prompt = `Berikan pep talk singkat. Mood: ${mood}. Alasan: ${reason}.`;
  return generateGeminiContent(prompt, {
    profile: "flash",
    operation: "generate-pep-talk",
    ...options,
  });
};

export const generateSocialPost = async (
  formData: any,
  useSearch: boolean,
  options: GenerationOptions = {}
): Promise<string> => {
  const prompt = `Anda adalah Social Media Strategist TikTok. Data:\n${JSON.stringify(formData, null, 2)}`;
  return generateGeminiContent(prompt, {
    profile: "flash",
    useSearch,
    operation: "generate-social-post",
    ...options,
  });
};

export const analyzePerformance = async (formData: any, options: GenerationOptions = {}): Promise<string> => {
  const prompt = `Anda adalah analis konten. Data performa:\n${JSON.stringify(formData, null, 2)}`;
  return generateGeminiContent(prompt, {
    profile: "flash",
    operation: "analyze-performance",
    ...options,
  });
};

export const validateGeminiApiKey = async (
  key: string,
  options: { profile?: GeminiModelProfile; timeoutMs?: number } = {}
): Promise<{ ok: boolean; status: "valid" | "invalid"; message: string; code?: string; statusCode?: number }> => {
  const trimmed = key.trim();
  if (!trimmed) {
    return {
      ok: false,
      status: "invalid",
      message: "Kunci API kosong.",
      code: "MISSING_API_KEY",
    };
  }

  const model = MODEL_PROFILES[options.profile ?? "flash"][0];
  const ai = new GoogleGenAI({ apiKey: trimmed });
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? 20_000);

  try {
    await withTimeout(
      () =>
        ai.models.generateContent({
          model,
          contents: "health check",
        }),
      timeoutMs,
      { operation: "validate-api-key", model }
    );

    return {
      ok: true,
      status: "valid",
      message: "API key valid dan siap dipakai.",
    };
  } catch (error) {
    const appError = toGeminiAppError(error, { operation: "validate-api-key", model });
    return {
      ok: false,
      status: "invalid",
      message: appError.userMessage,
      code: appError.code,
      statusCode: appError.statusCode,
    };
  }
};
