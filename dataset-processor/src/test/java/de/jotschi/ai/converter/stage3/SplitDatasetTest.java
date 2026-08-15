package de.jotschi.ai.converter.stage3;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

public class SplitDatasetTest {

	@Test
	public void testRecordsAreCopiedVerbatimIntoTheRightSplit(@TempDir Path tmp) throws IOException {
		Path input = tmp.resolve("in.jsonl");
		List<String> lines = List.of(
				record("00000000000000000000000000000000").encode(),
				record("ffffffff000000000000000000000000").encode());
		Files.write(input, lines, StandardCharsets.UTF_8);

		File outDir = tmp.resolve("basedata").toFile();
		SplitDataset splitter = new SplitDataset(0.5);
		splitter.run(List.of(input.toFile()), outDir);

		assertThat(splitter.train()).isEqualTo(1);
		assertThat(splitter.test()).isEqualTo(1);
		assertThat(splitter.skipped()).isZero();

		List<String> train = Files.readAllLines(outDir.toPath().resolve(SplitDataset.TRAIN_NAME),
				StandardCharsets.UTF_8);
		JsonObject roundTripped = new JsonObject(train.get(0));
		assertThat(roundTripped.getString("story")).contains("größer");
		assertThat(roundTripped.fieldNames()).containsExactlyInAnyOrder("hash", "request", "request_word_1", "story");
	}

	@Test
	public void testAgreesWithDataset2ChatOnWhichStoriesAreHeldOut(@TempDir Path tmp) throws IOException {
		// The regression that matters most: a story must never be in the chat
		// training set while its own text is in the pretraining validation set.
		List<String> lines = new ArrayList<>();
		for (int i = 0; i < 400; i++) {
			lines.add(record(String.format("%032x", i * 0x9E3779B9L)).encode());
		}
		Path input = tmp.resolve("in.jsonl");
		Files.write(input, lines, StandardCharsets.UTF_8);

		File chatOut = tmp.resolve("chat").toFile();
		File baseOut = tmp.resolve("base").toFile();
		new Dataset2Chat(0.25).run(List.of(input.toFile()), chatOut);
		new SplitDataset(0.25).run(List.of(input.toFile()), baseOut);

		Set<String> chatValRequests = new HashSet<>();
		for (String line : Files.readAllLines(chatOut.toPath().resolve(Dataset2Chat.VAL_NAME), StandardCharsets.UTF_8)) {
			chatValRequests.add(new JsonArray(line).getJsonObject(0).getString("content"));
		}
		Set<String> baseTestRequests = new HashSet<>();
		for (String line : Files.readAllLines(baseOut.toPath().resolve(SplitDataset.TEST_NAME),
				StandardCharsets.UTF_8)) {
			baseTestRequests.add(new JsonObject(line).getString("request"));
		}

		assertThat(chatValRequests).isNotEmpty();
		assertThat(chatValRequests).isEqualTo(baseTestRequests);
	}

	@Test
	public void testRecordWithoutHashIsSkipped(@TempDir Path tmp) throws IOException {
		Path input = tmp.resolve("in.jsonl");
		Files.write(input, List.of(new JsonObject().put("story", "Ohne Hash").encode()), StandardCharsets.UTF_8);

		SplitDataset splitter = new SplitDataset(0.5);
		splitter.run(List.of(input.toFile()), tmp.resolve("out").toFile());

		assertThat(splitter.skipped()).isEqualTo(1);
		assertThat(splitter.train()).isZero();
		assertThat(splitter.test()).isZero();
	}

	private static JsonObject record(String hash) {
		return new JsonObject()
				.put("hash", hash)
				.put("request", "Schreib ein Abenteuer " + hash)
				.put("request_word_1", "Mira")
				.put("story", "Mira flog, und das Raumschiff größer als der Mond wartete schon.");
	}
}
