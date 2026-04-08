import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { ctorMock, generateContentMock } = vi.hoisted(() => ({
  ctorMock: vi.fn(),
  generateContentMock: vi.fn(),
}));

vi.mock("@google/genai", () => ({
  GoogleGenAI: class GoogleGenAIMock {
    models: { generateContent: typeof generateContentMock };

    constructor(args: unknown) {
      ctorMock(args);
      this.models = {
        generateContent: generateContentMock,
      };
    }
  },
}));

import {
  GeminiAppError,
  generateGeminiContent,
  generateLaunchContent,
  generateLiveScript,
  generatePhotoVideoPrompt,
  generateSocialPost,
  getUserFriendlyErrorMessage,
  validateGeminiApiKey,
} from "../../services/geminiService";

describe("geminiService", () => {
  beforeEach(() => {
    sessionStorage.clear();
    generateContentMock.mockReset();
    ctorMock.mockReset();
    vi.stubEnv("API_KEY", "");
    vi.stubEnv("GEMINI_API_KEY", "");
    sessionStorage.setItem("geminiApiKey", "test-key");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  it("sends generation config parameters", async () => {
    generateContentMock.mockResolvedValue({ text: "ok" });

    await generateGeminiContent("prompt", {
      model: "gemini-2.5-flash",
      temperature: 0.3,
      maxOutputTokens: 123,
      topP: 0.9,
      topK: 24,
      systemInstruction: "strict-mode",
      retries: 0,
      operation: "unit-test",
    });

    expect(generateContentMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gemini-2.5-flash",
        contents: "prompt",
        config: expect.objectContaining({
          temperature: 0.3,
          maxOutputTokens: 123,
          topP: 0.9,
          topK: 24,
          systemInstruction: "strict-mode",
        }),
      })
    );
  });

  it.each([
    ["pro", "gemini-2.5-pro"],
    ["ultra", "gemini-2.5-pro"],
    ["flash", "gemini-2.5-flash"],
  ])("uses %s profile model candidates", async (profile, firstModel) => {
    generateContentMock.mockResolvedValue({ text: "ok" });

    await generateGeminiContent("prompt", {
      profile: profile as "pro" | "ultra" | "flash",
      retries: 0,
    });

    expect(generateContentMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: firstModel,
      })
    );
  });

  it("falls back to next model when first model unavailable", async () => {
    generateContentMock
      .mockRejectedValueOnce(new Error("404 model not found"))
      .mockResolvedValueOnce({ text: "fallback-ok" });

    const output = await generateLaunchContent({ productName: "X" }, "product", { retries: 0 });

    expect(output).toBe("fallback-ok");
    expect(generateContentMock).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ model: "gemini-2.5-flash" })
    );
    expect(generateContentMock).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ model: "gemini-2.0-flash-exp" })
    );
  });

  it("uses global API model when profile is api-default", async () => {
    sessionStorage.setItem("geminiApiModel", "gemini-2.5-pro");
    generateContentMock.mockResolvedValue({ text: "ok" });

    await generateLaunchContent({ productName: "X" }, "product", {
      profile: "api-default",
      retries: 0,
    });

    expect(generateContentMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gemini-2.5-pro",
      })
    );
  });

  it("uses explicit validation model when provided", async () => {
    generateContentMock.mockResolvedValue({ text: "ok" });

    const result = await validateGeminiApiKey("test-key", {
      model: "gemini-2.5-flash",
      profile: "api-default",
    });

    expect(result.ok).toBe(true);
    expect(generateContentMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gemini-2.5-flash",
      })
    );
  });

  it("uses parts fallback when response.text is empty", async () => {
    generateContentMock.mockResolvedValue({
      text: "",
      candidates: [{ content: { parts: [{ text: "A" }, { text: "B" }] } }],
    });

    const output = await generateLaunchContent({ productName: "X" }, "product", { retries: 0 });

    expect(output).toBe("AB");
  });

  it("appends citations when search is enabled", async () => {
    generateContentMock.mockResolvedValue({
      text: "hasil utama",
      candidates: [
        {
          groundingMetadata: {
            groundingChunks: [{ web: { uri: "https://a.com" } }, { web: { uri: "https://a.com" } }, { web: { uri: "https://b.com" } }],
          },
        },
      ],
    });

    const output = await generateSocialPost({ topic: "trend" }, true, { retries: 0 });

    expect(output).toContain("hasil utama");
    expect(output).toContain("**Sumber:**");
    expect(output).toContain("https://a.com");
    expect(output).toContain("https://b.com");
    expect(generateContentMock).toHaveBeenCalledWith(
      expect.objectContaining({
        config: expect.objectContaining({
          tools: [{ googleSearch: {} }],
        }),
      })
    );
  });

  it("maps safety blocked response to SAFETY_BLOCK", async () => {
    generateContentMock.mockResolvedValue({
      text: "",
      candidates: [{ finishReason: "SAFETY", content: { parts: [] } }],
    });

    await expect(generateLaunchContent({ productName: "X" }, "product", { retries: 0 })).rejects.toMatchObject({
      code: "SAFETY_BLOCK",
    });
  });

  it("maps empty response to EMPTY_RESPONSE", async () => {
    generateContentMock.mockResolvedValue({ text: "", candidates: [] });

    await expect(generateLaunchContent({ productName: "X" }, "product", { retries: 0 })).rejects.toMatchObject({
      code: "EMPTY_RESPONSE",
    });
  });

  it("returns timeout error when call exceeds timeout", async () => {
    generateContentMock.mockImplementation(() => new Promise(() => {}));

    await expect(
      generateGeminiContent("prompt", {
      model: "gemini-2.5-flash",
      timeoutMs: 20,
      retries: 0,
      })
    ).rejects.toMatchObject({ code: "REQUEST_TIMEOUT" });
  });

  it("parses photo/video JSON payload", async () => {
    generateContentMock.mockResolvedValue({
      text: "```json\n{\"photo_prompt\":\"photo text\",\"video_prompt\":\"video text\"}\n```",
    });

    const output = await generatePhotoVideoPrompt({ productName: "Sepatu" }, "manual", { retries: 0 });

    expect(output).toEqual({ photo: "photo text", video: "video text" });
  });

  it("throws invalid shape for incomplete photo/video JSON", async () => {
    generateContentMock.mockResolvedValue({
      text: "```json\n{\"photo_prompt\":\"photo only\"}\n```",
    });

    await expect(generatePhotoVideoPrompt({ productName: "Sepatu" }, "manual", { retries: 0 })).rejects.toMatchObject({
      code: "INVALID_JSON_SHAPE",
    });
  });

  it("returns live script JSON string when timeline exists", async () => {
    generateContentMock.mockResolvedValue({
      text: "```json\n{\"timeline\":[{\"time\":\"00:00-01:00\",\"title\":\"Intro\",\"script\":\"Hai\",\"icon\":\"👋\"}]}\n```",
    });

    const output = await generateLiveScript({ mode: "shopping" }, { retries: 0 });
    const parsed = JSON.parse(output);

    expect(Array.isArray(parsed.timeline)).toBe(true);
    expect(parsed.timeline[0].title).toBe("Intro");
  });

  it("returns MISSING_API_KEY when key is absent", async () => {
    sessionStorage.removeItem("geminiApiKey");

    await expect(generateLaunchContent({ productName: "X" }, "product")).rejects.toMatchObject({
      code: "MISSING_API_KEY",
    });
  });

  it("validates API key and maps invalid key errors", async () => {
    generateContentMock.mockRejectedValue(new Error("403 API key not valid"));

    const result = await validateGeminiApiKey("bad-key");

    expect(result.ok).toBe(false);
    expect(result.code).toBe("INVALID_API_KEY");
    expect(result.status).toBe("invalid");
  });

  it("maps friendly error message from unknown error", () => {
    const message = getUserFriendlyErrorMessage(new Error("429 Too Many Requests"));
    expect(message).toContain("Kuota API Gemini");
  });

  it("exposes typed GeminiAppError", () => {
    const err = new GeminiAppError({
      code: "X",
      userMessage: "Y",
      retryable: false,
    });
    expect(err.code).toBe("X");
    expect(err.userMessage).toBe("Y");
  });
});
