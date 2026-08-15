package de.jotschi.ai.converter.stage1;

import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.BeforeAll;

import io.vertx.core.json.JsonObject;

public abstract class AbstractGeneratorTest {

	public static final FilenameFilter JSONL_FILENAME_FILTER = new FilenameFilter() {
		public boolean accept(File dir, String name) {
			return name.toLowerCase().endsWith(".jsonl");
		}
	};

	protected List<JsonObject> readJsonlFile(File file) throws IOException {
		return FileUtils.readLines(file, Charset.defaultCharset()).stream().map(line -> new JsonObject(line)).toList();
	}

	private static Properties settings;

	@BeforeAll
	public static void loadSettings() throws FileNotFoundException, IOException {
		settings = new Properties();
		File settingsFile = new File("config", "settings.properties");
		if (!settingsFile.exists()) {
			fail("Settings file " + settingsFile + " not found.");
		}
		settings.load(new FileInputStream(settingsFile));
	}

	protected ThreadPoolExecutor createExecutor(int poolSize) {
		BlockingQueue<Runnable> WORK_QUEUE = new LinkedBlockingQueue<Runnable>(256);
		ThreadFactory factory = Thread.ofVirtual().factory();
		ThreadPoolExecutor exec = new ThreadPoolExecutor(poolSize, poolSize, 0L, TimeUnit.MILLISECONDS, WORK_QUEUE,
				factory, new ThreadPoolExecutor.CallerRunsPolicy());
		return exec;
	}

	protected static synchronized void writeLocking(JsonObject json, File outputFile) {
		try {

			FileUtils.writeStringToFile(outputFile, json.encode() + "\n", Charset.defaultCharset(), true);
		} catch (IOException e) {
			System.err.println("Processing failed");
			e.printStackTrace();
		}
	}

	protected static String getClusterURL() {
		return settings.getProperty("cluster.url");
	}

	protected static String getOllamaURL() {
		return settings.getProperty("ollama.host");
	}

	protected static String getNanoChatCacheDir() {
		return settings.getProperty("nanochat.cache.dir");
	}
}
