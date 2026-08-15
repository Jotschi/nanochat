package de.jotschi.ai.converter.stage2;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;
import de.jotschi.ai.processor.jsonl.KleinerAstronautJsonlHandler;
import io.vertx.core.json.JsonObject;

/**
 * Stage 2: turn raw stories into request/story (+ optional question/answer)
 * records. ETL job - needs a live LLM endpoint, excluded from surefire.
 */
public class StoryProcessorTest extends AbstractGeneratorTest {

	private static final AtomicLong FAILURES = new AtomicLong();

	public static File outputFile = new File("dataset", "kleiner_astronaut_qa_v6.jsonl");
	public static final ObjectMapper mapper = new ObjectMapper();

	@Test
	public void testProcess() throws IOException, InterruptedException {
		KleinerAstronautJsonlHandler handler = new KleinerAstronautJsonlHandler(llm(), model());
		ThreadPoolExecutor exec = createExecutor(10);
		Set<String> hashes = loadHashes(outputFile);
		File storiesFolder = new File("dataset", "stories");

		AtomicLong total = new AtomicLong();
		for (File storyFile : storiesFolder.listFiles(JSONL_FILENAME_FILTER)) {
			List<JsonObject> stories = readJsonlFile(storyFile);
			for (JsonObject storyJson : stories) {
				String hash = storyJson.getString("hash");
				if (hashes.contains(hash)) {
					System.err.println("[" + hash + "] already handled.");
					continue;
				}
				exec.execute(() -> {
					try {
						total.incrementAndGet();
						if (processStory(handler, storyJson)) {
							hashes.add(hash);
						}
					} catch (IOException e) {
						e.printStackTrace();
					}
				});
				if (total.get() % 100 == 0) {
					System.err.println("[" + total.get() + " / " + FAILURES.get() + " failed]");
				}

			}
		}
		exec.awaitTermination(10, TimeUnit.HOURS);

	}

	private boolean processStory(KleinerAstronautJsonlHandler handler, JsonObject story) throws IOException {
		try {
			JsonObject jsonOut = handler.process(story);
			if (jsonOut != null) {
				writeLocking(jsonOut, outputFile);
				return true;
			} else {
				FAILURES.incrementAndGet();
				return false;
			}
		} catch (Exception e) {
			e.printStackTrace();
			FAILURES.incrementAndGet();
			return false;
		}

	}

	private Set<String> loadHashes(File outputFile) throws IOException {
		if (!outputFile.exists()) {
			return Collections.synchronizedSet(new HashSet<String>());
		}
		return Collections.synchronizedSet(new HashSet<>(FileUtils.readLines(outputFile, Charset.defaultCharset())
				.stream().map(line -> new JsonObject(line).getString("hash")).toList()));
	}

}
