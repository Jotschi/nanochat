package de.jotschi.ai.converter.stage1;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.StoryGenerator;
import de.jotschi.ai.processor.chat.llm.Models;
import de.jotschi.ai.processor.chat.llm.VLLMModel;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.ollama.OllamaLLMProvider;
import io.metaloom.ai.genai.llm.vllm.VLLMLLMProvider;
import io.vertx.core.json.JsonObject;

public class StoryGeneratorTest extends AbstractGeneratorTest {

	@Test
	public void testLocal() throws IOException {
		File destFile = new File("dataset", "stories_local.jsonl");
		LargeLanguageModel model = Models.OLLAMA_MISTRAL_SMALL_32_24B_Q8;
		LLMProvider ollama = new OllamaLLMProvider();
		runGenerate(destFile, ollama, model);
	}

	@Test
	public void generateParallel() throws InterruptedException {
		String url = getClusterURL();
		LargeLanguageModel model = VLLMModel.mistral24bQ8(url);
		LLMProvider vllm = new VLLMLLMProvider();
		StoryGenerator gen = new StoryGenerator(vllm, model);

		ThreadPoolExecutor exec = createExecutor(24);

		AtomicLong total = new AtomicLong();
		File destFile = new File("dataset", "stories.jsonl");
		for (int i = 0; i < 24; i++) {
			exec.execute(() -> {
				while (true) {
					JsonObject storyJson = gen.generate();
					if (storyJson != null) {
						String hash = storyJson.getString("hash");
						System.out.println("[" + total.incrementAndGet() + "] - " + hash);
						writeLocking(storyJson, destFile);
					}
				}
			});
		}

		exec.awaitTermination(10, TimeUnit.HOURS);

	}

	

	public void runGenerate(File destFile, LLMProvider llm, LargeLanguageModel model) throws IOException {

		StoryGenerator gen = new StoryGenerator(llm, model);
		while (true) {
			try {
				JsonObject storyJson = gen.generate();
				if (storyJson != null) {
					FileUtils.writeStringToFile(destFile, storyJson.encode() + "\n", Charset.defaultCharset(), true);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
//			System.out.println(story);
		}
	}
}
