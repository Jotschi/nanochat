package de.jotschi.ai.converter.stage3;

import static org.assertj.core.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;

public class SplitDatasetTest extends AbstractGeneratorTest {

	@Test
	public void testSplit() throws IOException {
		File dest = new File(getNanoChatCacheDir(), "astronaut_basedata");
		if (!dest.exists()) {
			fail("Destination " + dest + " not found.");
		}
		File srcFile = new File("dataset", "kleiner_astronaut_qa_v6.jsonl");
		List<String> lines = FileUtils.readLines(srcFile, Charset.defaultCharset());
		int total = lines.size();
		int testSplit = (int) (0.05f * (float) total);

		int test = 0;
		int train = 0;

		File trainTarget = new File(dest, "kleiner_astronaut_qa_train.jsonl");
		if (trainTarget.exists()) {
			trainTarget.delete();
		}

		File testTarget = new File(dest, "kleiner_astronaut_qa_test.jsonl");
		if (testTarget.exists()) {
			testTarget.delete();
		}
		Collections.shuffle(lines);
		for (String line : lines) {
			File target = trainTarget;
			if (testSplit > 0) {
				target = testTarget;
				test++;
				testSplit--;
			} else {
				train++;
			}
			FileUtils.writeStringToFile(target, line + "\n", Charset.defaultCharset(), true);
		}
		System.out.println("Test: " + test);
		System.out.println("Train: " + train);
	}
}
