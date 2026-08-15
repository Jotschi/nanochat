package de.jotschi.ai.converter.stage1;

import java.io.File;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.StoryGenerator;
import io.vertx.core.json.JsonObject;

/**
 * Stage 1: generate raw stories into {@code dataset/stories.jsonl}.
 * <p>
 * This is an ETL job, not a unit test. It runs until the JVM is killed and needs
 * a live OpenAI-compatible LLM endpoint, so it is excluded from the surefire run
 * (see the pom's {@code <excludes>}).
 */
public class StoryGeneratorTest extends AbstractGeneratorTest {

	private static final int THREADS = 24;

	@Test
	public void generateParallel() throws InterruptedException {
		StoryGenerator gen = new StoryGenerator(llm(), model());
		ThreadPoolExecutor exec = createExecutor(THREADS);

		AtomicLong total = new AtomicLong();
		File destFile = new File("dataset", "stories.jsonl");
		for (int i = 0; i < THREADS; i++) {
			exec.execute(() -> {
				while (true) {
					try {
						JsonObject storyJson = gen.generate();
						if (storyJson != null) {
							String hash = storyJson.getString("hash");
							System.out.println("[" + total.incrementAndGet() + "] - " + hash);
							writeLocking(storyJson, destFile);
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			});
		}

		exec.shutdown();
		exec.awaitTermination(10, TimeUnit.HOURS);
	}
}
