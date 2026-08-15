package de.jotschi.ai.converter.old;

import java.io.File;

import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;
import de.jotschi.ai.processor.parquet.KleinerAstronautParquetHandler;
import de.jotschi.ai.processor.parquet.KleinerAstronautParquetProcessor;

/**
 * Legacy path: build the QA dataset straight from the HuggingFace
 * {@code Jotschi/kleiner-astronaut} parquet export rather than from
 * self-generated stories. Superseded by stage 1 + stage 2 and kept only for
 * reference - {@code dataset/kleiner_astronaut/} is not part of this repo.
 * <p>
 * Add {@code --add-opens=java.base/java.nio=ALL-UNNAMED} to run it. Needs a live
 * LLM endpoint, so it is excluded from the surefire run.
 */
public class ParquetProcessorTest extends AbstractGeneratorTest {

	@Test
	public void testQA() {
		File datasetFolder = new File("dataset", "kleiner_astronaut");
		File datasetOut = new File("dataset", "kleiner_astronaut_qa_v3.jsonl");
		KleinerAstronautParquetHandler handler = new KleinerAstronautParquetHandler(datasetOut, llm(), model());
		KleinerAstronautParquetProcessor c = new KleinerAstronautParquetProcessor(handler);
		c.process(datasetFolder, "train");
	}
}
