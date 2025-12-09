package de.jotschi.ai.converter.stage2;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;
import de.jotschi.ai.processor.chat.llm.VLLMModel;
import de.jotschi.ai.processor.jsonl.KleinerAstronautJsonlEntry;
import de.jotschi.ai.processor.jsonl.KleinerAstronautJsonlHandler;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.vllm.VLLMLLMProvider;
import io.vertx.core.json.JsonObject;

public class StoryProcessorTest extends AbstractGeneratorTest {

//	private static LargeLanguageModel MODEL = Models.OLLAMA_PHI3_MINI;
//	private static LLMProvider ollama = new OllamaLLMProvider();

	public static File outputFile = new File("dataset", "kleiner_astronaut_qa_v4_hashed2.jsonl");
	public static final ObjectMapper mapper = new ObjectMapper();

	@Test
	public void testProcess() throws IOException, InterruptedException {
		LLMProvider llm = new VLLMLLMProvider();
		String url = getClusterURL();
		LargeLanguageModel model = VLLMModel.mistral24bQ8(url);
		KleinerAstronautJsonlHandler handler = new KleinerAstronautJsonlHandler(llm, model);
		ThreadPoolExecutor exec = createExecutor(10);
		Set<String> hashes = Collections.emptySet();
		if (outputFile.exists()) {
			hashes = loadHashes(outputFile);
		}
		File storiesFile = new File("dataset", "stories.jsonl");
		List<JsonObject> stories = FileUtils.readLines(storiesFile, Charset.defaultCharset()).stream()
				.map(line -> new JsonObject(line)).toList();
		for (JsonObject story : stories) {
			String hash = story.getString("hash");
			if (hashes.contains(hash)) {
				System.err.println("[" + hash + "] already handled.");
				continue;
			}
			exec.execute(() -> {
				try {
					processStory(handler, story);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});

		}
		exec.awaitTermination(10, TimeUnit.HOURS);

	}

	private void processStory(KleinerAstronautJsonlHandler handler, JsonObject story) throws IOException {

		long failCount = 0;

		try {
			KleinerAstronautJsonlEntry entry = mapper.readValue(story.encode(), KleinerAstronautJsonlEntry.class);
			String text = entry.getText().toLowerCase();
			List<String> foundNames = new ArrayList<>();
			for (String name : entry.getNames()) {
				if (text.contains(name.toLowerCase())) {
					foundNames.add(name);
				}
			}
			if (foundNames.isEmpty()) {
				System.err.println("[" + entry.hash() + "] failed (no names - " + failCount + ")");
				failCount++;
				return;
			}
			entry.setNames(foundNames);

			// System.out.println("[" + id + "] " + entry.getAdjective1());
			JsonObject jsonOut = handler.process(entry);
			if (jsonOut != null) {
				writeLocking(jsonOut, outputFile);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private Set<String> loadHashes(File outputFile) throws IOException {
		return new HashSet<>(FileUtils.readLines(outputFile, Charset.defaultCharset()).stream()
				.map(line -> new JsonObject(line).getString("hash")).toList());
	}

}
