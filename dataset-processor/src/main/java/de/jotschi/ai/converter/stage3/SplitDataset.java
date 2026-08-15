package de.jotschi.ai.converter.stage3;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import de.jotschi.ai.Settings;
import io.vertx.core.json.JsonObject;

/**
 * Stage 3b: split the base-data records into train/test for pretraining.
 * <p>
 * Records are copied through verbatim - no chat wrapping, no reward keys. The
 * split uses the same {@link Split} predicate as {@link Dataset2Chat}, so a
 * story held out of the conversations is held out here too.
 * <p>
 * Run with:
 * {@code mvn -q compile exec:java -Dexec.mainClass=de.jotschi.ai.converter.stage3.SplitDataset}
 */
public class SplitDataset {

	public static final String BASE_DATA_DIR = "astronaut_basedata";
	public static final String TRAIN_NAME = "kleiner_astronaut_qa_train.jsonl";
	public static final String TEST_NAME = "kleiner_astronaut_qa_test.jsonl";

	private final double valFraction;

	private long train;
	private long test;
	private long skipped;

	public SplitDataset(double valFraction) {
		this.valFraction = valFraction;
	}

	public static void main(String[] args) throws IOException {
		List<File> inputs = new ArrayList<>();
		File outDir = null;
		double valFraction = Split.DEFAULT_VAL_FRACTION;

		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "--input" -> inputs.add(new File(args[++i]));
			case "--out-dir" -> outDir = new File(args[++i]);
			case "--val-fraction" -> valFraction = Double.parseDouble(args[++i]);
			default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
			}
		}
		if (inputs.isEmpty()) {
			inputs.addAll(Dataset2Chat.DEFAULT_INPUTS);
		}
		if (outDir == null) {
			outDir = new File(Settings.nanochatCacheDir(), BASE_DATA_DIR);
		}
		new SplitDataset(valFraction).run(inputs, outDir);
	}

	public void run(List<File> inputs, File outDir) throws IOException {
		List<File> present = inputs.stream().filter(File::exists).toList();
		for (File missing : inputs.stream().filter(f -> !f.exists()).toList()) {
			System.err.println("Skipping missing input: " + missing);
		}
		if (present.isEmpty()) {
			throw new IllegalStateException("None of the input files exist: " + inputs);
		}
		if (!outDir.exists() && !outDir.mkdirs()) {
			throw new IOException("Could not create output directory " + outDir);
		}

		Path trainPath = outDir.toPath().resolve(TRAIN_NAME);
		Path testPath = outDir.toPath().resolve(TEST_NAME);

		try (BufferedWriter trainOut = Files.newBufferedWriter(trainPath, StandardCharsets.UTF_8);
				BufferedWriter testOut = Files.newBufferedWriter(testPath, StandardCharsets.UTF_8)) {
			for (File input : present) {
				System.out.println("Reading " + input);
				try (Stream<String> lines = Files.lines(input.toPath(), StandardCharsets.UTF_8)) {
					for (String raw : (Iterable<String>) lines::iterator) {
						String line = raw.strip();
						if (line.isEmpty()) {
							continue;
						}
						String hash;
						try {
							hash = new JsonObject(line).getString("hash");
						} catch (Exception e) {
							skipped++;
							System.err.println("Unparseable line skipped: " + e.getMessage());
							continue;
						}
						if (hash == null) {
							skipped++;
							continue;
						}
						if (Split.isHeldOut(hash, valFraction)) {
							testOut.write(line);
							testOut.write('\n');
							test++;
						} else {
							trainOut.write(line);
							trainOut.write('\n');
							train++;
						}
					}
				}
			}
		}

		System.out.println("Train:   " + train + " -> " + trainPath);
		System.out.println("Test:    " + test + " -> " + testPath);
		System.out.println("Skipped: " + skipped);
	}

	public long train() {
		return train;
	}

	public long test() {
		return test;
	}

	public long skipped() {
		return skipped;
	}
}
