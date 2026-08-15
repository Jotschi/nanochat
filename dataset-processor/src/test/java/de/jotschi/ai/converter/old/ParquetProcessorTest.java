package de.jotschi.ai.converter.old;

import java.io.File;

import org.junit.jupiter.api.Test;

import de.jotschi.ai.processor.chat.llm.Models;
import de.jotschi.ai.processor.parquet.KleinerAstronautParquetProcessor;
import de.jotschi.ai.processor.parquet.KleinerAstronautParquetHandler;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.ollama.OllamaLLMProvider;

/**
 * Add --add-opens=java.base/java.nio=ALL-UNNAMED to run the tests.
 */
public class ParquetProcessorTest {

	@Test
	public void testQA() {
		File datasetFolder = new File("dataset", "kleiner_astronaut");

		LargeLanguageModel model = Models.OLLAMA_MISTRAL_SMALL_32_24B_Q8;
		LLMProvider ollama = new OllamaLLMProvider();
		File datasetOut = new File("dataset", "kleiner_astronaut_qa_v3.jsonl");
		KleinerAstronautParquetHandler handler = new KleinerAstronautParquetHandler(datasetOut, ollama, model);
		KleinerAstronautParquetProcessor c = new KleinerAstronautParquetProcessor(handler);
		c.process(datasetFolder, "train");

	}
}
