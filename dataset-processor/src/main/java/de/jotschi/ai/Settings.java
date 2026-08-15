package de.jotschi.ai;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Properties;

import de.jotschi.ai.processor.chat.llm.VLLMModel;

/**
 * Loads {@code config/settings.properties}. See
 * {@code config/settings.properties.example} for the keys.
 */
public final class Settings {

	public static final File SETTINGS_FILE = new File("config", "settings.properties");

	private static Properties props;

	private Settings() {
	}

	public static synchronized Properties load() {
		if (props == null) {
			if (!SETTINGS_FILE.exists()) {
				throw new IllegalStateException("Settings file " + SETTINGS_FILE.getAbsolutePath()
						+ " not found. Copy config/settings.properties.example and adjust it.");
			}
			Properties loaded = new Properties();
			try (InputStream in = new FileInputStream(SETTINGS_FILE)) {
				loaded.load(in);
			} catch (IOException e) {
				throw new UncheckedIOException("Could not read " + SETTINGS_FILE, e);
			}
			props = loaded;
		}
		return props;
	}

	private static String required(String key) {
		String value = load().getProperty(key);
		if (value == null || value.isBlank()) {
			throw new IllegalStateException("Missing '" + key + "' in " + SETTINGS_FILE);
		}
		return value;
	}

	/** Base URL of the OpenAI-compatible LLM endpoint (vLLM or llama.cpp). */
	public static String clusterUrl() {
		return required("cluster.url");
	}

	public static String llmModel() {
		return load().getProperty("llm.model", VLLMModel.MISTRAL_SMALL_24B);
	}

	public static long llmContextWindow() {
		return Long.parseLong(load().getProperty("llm.context.window", "128000"));
	}

	/** Where the nanochat artifacts live; must match NANOCHAT_BASE_DIR. */
	public static File nanochatCacheDir() {
		return new File(required("nanochat.cache.dir"));
	}
}
