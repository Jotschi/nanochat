package de.jotschi.ai.processor.chat.llm;

import io.metaloom.ai.genai.llm.LargeLanguageModel;

/**
 * Model descriptor for an OpenAI-compatible endpoint - vLLM or llama.cpp in the
 * background. Pair it with {@link io.metaloom.ai.genai.llm.openai.OpenAILLMProvider}.
 */
public class VLLMModel implements LargeLanguageModel {

	public static final String MISTRAL_SMALL_24B = "mistralai/Mistral-Small-24B-Instruct-2501";

	private final String id;
	private final String url;
	private final long ctxWindowSize;

	public VLLMModel(String id, String url, long ctxWindowSize) {
		this.id = id;
		this.url = url;
		this.ctxWindowSize = ctxWindowSize;
	}

	@Override
	public String id() {
		return id;
	}

	@Override
	public String url() {
		return url;
	}

	@Override
	public long contextWindow() {
		return ctxWindowSize;
	}

	public static VLLMModel mistral24bQ8(String url) {
		return new VLLMModel(MISTRAL_SMALL_24B, url, 128_000);
	}

	@Override
	public String toString() {
		return id + " @ " + url;
	}

}
