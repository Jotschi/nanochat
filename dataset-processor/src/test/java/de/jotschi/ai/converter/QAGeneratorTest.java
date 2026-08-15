package de.jotschi.ai.converter;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import de.jotschi.ai.processor.chat.llm.Models;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QAGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QuestionAnswerResult;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.ollama.OllamaLLMProvider;
import io.vertx.core.json.JsonObject;

public class QAGeneratorTest {

	@Test
	public void testQA() throws IOException {
		List<String> lines = FileUtils.readLines(new File("dataset", "kleiner_astronaut_qa_v2.jsonl"),
				Charset.defaultCharset());
		LargeLanguageModel model = Models.OLLAMA_MISTRAL_SMALL_32_24B_Q8;
		LLMProvider ollama = new OllamaLLMProvider();
		for (int i = 0; i < 10; i++) {
			JsonObject json = new JsonObject(lines.get(i));
			String story = json.getString("story");
//			String word_1 = json.getString("word_1");
//			String word_2 = json.getString("word_2");
//			System.out.println("Input: " + word_1 + " | " + word_2);
			QAGenerator gen = new QAGenerator(ollama, model);
			QuestionAnswerResult result = gen.generateQA(story);
			System.out.println("OK: " + result.question() +  " => " + result.answer()+  " / " + result.word());
		}
	}
}
