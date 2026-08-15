package de.jotschi.ai.processor.chat.llm;

import java.util.Random;

import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;

public class AbstractGenerator {

	protected static final int RETRY_MAX = 20;

	protected static final Random RND = new Random();

	protected LLMProvider llm;

	protected LargeLanguageModel model;

	public AbstractGenerator(LLMProvider llm, LargeLanguageModel model) {
		this.llm = llm;
		this.model = model;
	}

}
