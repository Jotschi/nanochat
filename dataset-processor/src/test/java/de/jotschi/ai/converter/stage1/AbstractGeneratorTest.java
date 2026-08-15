package de.jotschi.ai.converter.stage1;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;

import de.jotschi.ai.Settings;
import de.jotschi.ai.processor.chat.llm.VLLMModel;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.openai.OpenAILLMProvider;
import io.vertx.core.json.JsonObject;

/**
 * Shared plumbing for the stage 1 and 2 ETL jobs. These are not unit tests -
 * they need a live LLM endpoint and are excluded from the surefire run.
 */
public abstract class AbstractGeneratorTest {

	public static final FilenameFilter JSONL_FILENAME_FILTER = new FilenameFilter() {
		public boolean accept(File dir, String name) {
			return name.toLowerCase().endsWith(".jsonl");
		}
	};

	protected List<JsonObject> readJsonlFile(File file) throws IOException {
		return Files.readAllLines(file.toPath(), StandardCharsets.UTF_8).stream()
				.map(String::strip)
				.filter(line -> !line.isEmpty())
				.map(JsonObject::new)
				.toList();
	}

	protected ThreadPoolExecutor createExecutor(int poolSize) {
		BlockingQueue<Runnable> workQueue = new LinkedBlockingQueue<Runnable>(256);
		ThreadFactory factory = Thread.ofVirtual().factory();
		return new ThreadPoolExecutor(poolSize, poolSize, 0L, TimeUnit.MILLISECONDS, workQueue, factory,
				new ThreadPoolExecutor.CallerRunsPolicy());
	}

	protected static synchronized void writeLocking(JsonObject json, File outputFile) {
		try {
			FileUtils.writeStringToFile(outputFile, json.encode() + "\n", StandardCharsets.UTF_8, true);
		} catch (IOException e) {
			System.err.println("Processing failed");
			e.printStackTrace();
		}
	}

	protected static String getClusterURL() {
		return Settings.clusterUrl();
	}

	protected static String getNanoChatCacheDir() {
		return Settings.nanochatCacheDir().getAbsolutePath();
	}

	/**
	 * The single LLM provider we use. The endpoint is OpenAI-compatible, so vLLM
	 * or llama.cpp can sit behind it.
	 */
	protected static LLMProvider llm() {
		return new OpenAILLMProvider();
	}

	/** Model descriptor from {@code cluster.url} / {@code llm.model} / {@code llm.context.window}. */
	protected static LargeLanguageModel model() {
		return new VLLMModel(Settings.llmModel(), Settings.clusterUrl(), Settings.llmContextWindow());
	}
}
