package de.jotschi.ai.processor.chat.llm;

import io.metaloom.ai.genai.llm.LLMProviderType;
import io.metaloom.ai.genai.llm.LargeLanguageModel;

public class VLLMModel implements LargeLanguageModel {

	private String id;
	private String url;
	private long ctxWindowSize;

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

	@Override
	public LLMProviderType providerType() {
		return LLMProviderType.VLLM;
	}

	public static VLLMModel mistral24bQ8(String url) {
		return new VLLMModel("mistralai/Mistral-Small-24B-Instruct-2501", url, 128_000);
	}

}
