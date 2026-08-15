package de.jotschi.ai.converter.stage3;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

/**
 * Offline tests for the chat converter. No LLM, no fixtures outside the repo.
 */
public class Dataset2ChatTest {

	private static final String STORY = "Mira flog mit dem Raumschiff zum Mond. Aris winkte ihr zu.";

	@Test
	public void testFourTurnWhenQuestionAndAnswerArePresent() {
		JsonObject record = new JsonObject()
				.put("hash", "75baf618ff5de97b9e057ad60e6a2690")
				.put("request", "Schreib ein Abenteuer von Mira mit Aris")
				.put("request_word_1", "Mira")
				.put("request_word_2", "Aris")
				.put("story", STORY)
				.put("question", "Wer winkte Mira zu?")
				.put("answer", "Aris winkte ihr zu.")
				.put("answer_word", "Aris");

		JsonArray chat = new Dataset2Chat(Split.DEFAULT_VAL_FRACTION).toChat(record);

		assertThat(chat).hasSize(4);
		assertThat(chat.getJsonObject(0).getString("role")).isEqualTo("user");
		assertThat(chat.getJsonObject(1).getString("role")).isEqualTo("assistant");
		assertThat(chat.getJsonObject(2).getString("role")).isEqualTo("user");
		assertThat(chat.getJsonObject(3).getString("role")).isEqualTo("assistant");

		assertThat(chat.getJsonObject(1).getString("content")).isEqualTo(STORY);
		assertThat(chat.getJsonObject(1).getString("rl_key1")).isEqualTo("Mira");
		assertThat(chat.getJsonObject(1).getString("rl_key2")).isEqualTo("Aris");
		assertThat(chat.getJsonObject(3).getString("rl_key1")).isEqualTo("Aris");
	}

	@Test
	public void testTwoTurnWhenQuestionAndAnswerAreMissing() {
		JsonObject record = new JsonObject()
				.put("hash", "75baf618ff5de97b9e057ad60e6a2690")
				.put("request", "Schreib ein Abenteuer von Mira mit Aris")
				.put("request_word_1", "Mira")
				.put("request_word_2", "Aris")
				.put("story", STORY);

		JsonArray chat = new Dataset2Chat(Split.DEFAULT_VAL_FRACTION).toChat(record);

		assertThat(chat).hasSize(2);
		assertThat(chat.getJsonObject(1).getString("rl_key1")).isEqualTo("Mira");
	}

	@Test
	public void testRewardKeyIsDroppedWhenItIsNotInTheStory() {
		// The generator was supposed to guarantee both keywords occur, but its
		// word2 check actually re-tested word1. Keys that cannot be satisfied
		// must not reach the RL stage.
		JsonObject record = new JsonObject()
				.put("hash", "75baf618ff5de97b9e057ad60e6a2690")
				.put("request", "Schreib ein Abenteuer von Mira mit einer Hexe")
				.put("request_word_1", "Mira")
				.put("request_word_2", "Hexe")
				.put("story", STORY);

		JsonArray chat = new Dataset2Chat(Split.DEFAULT_VAL_FRACTION).toChat(record);

		assertThat(chat.getJsonObject(1).getString("rl_key1")).isEqualTo("Mira");
		assertThat(chat.getJsonObject(1).containsKey("rl_key2")).as("'Hexe' is not in the story").isFalse();
	}

	@Test
	public void testRecordWithoutRequestOrStoryIsRejected() {
		Dataset2Chat converter = new Dataset2Chat(Split.DEFAULT_VAL_FRACTION);
		assertThat(converter.toChat(new JsonObject().put("story", STORY))).isNull();
		assertThat(converter.toChat(new JsonObject().put("request", "Schreib"))).isNull();
	}

	@Test
	public void testRunSplitsByStoryHashAndWritesUtf8(@TempDir Path tmp) throws IOException {
		Path input = tmp.resolve("in.jsonl");
		// Two hashes chosen so one lands in each split at 50%.
		List<String> lines = List.of(
				record("00000000000000000000000000000000", "Mira").encode(),
				record("ffffffff000000000000000000000000", "Mira").encode());
		Files.write(input, lines, StandardCharsets.UTF_8);

		File outDir = tmp.resolve("out").toFile();
		Dataset2Chat converter = new Dataset2Chat(0.5);
		converter.run(List.of(input.toFile()), outDir);

		assertThat(converter.written()).isEqualTo(2);
		assertThat(converter.skipped()).isZero();

		List<String> train = Files.readAllLines(outDir.toPath().resolve(Dataset2Chat.TRAIN_NAME), StandardCharsets.UTF_8);
		List<String> val = Files.readAllLines(outDir.toPath().resolve(Dataset2Chat.VAL_NAME), StandardCharsets.UTF_8);
		assertThat(train).hasSize(1);
		assertThat(val).hasSize(1);

		// Umlauts must survive the round trip regardless of the platform charset.
		assertThat(new JsonArray(train.get(0)).getJsonObject(1).getString("content")).contains("Raumschiff größer");
	}

	@Test
	public void testMissingInputsAreSkippedNotFatal(@TempDir Path tmp) throws IOException {
		Path input = tmp.resolve("in.jsonl");
		Files.write(input, List.of(record("75baf618ff5de97b9e057ad60e6a2690", "Mira").encode()),
				StandardCharsets.UTF_8);

		Dataset2Chat converter = new Dataset2Chat(Split.DEFAULT_VAL_FRACTION);
		converter.run(List.of(tmp.resolve("nope.jsonl").toFile(), input.toFile()), tmp.resolve("out").toFile());

		assertThat(converter.written()).isEqualTo(1);
	}

	private static JsonObject record(String hash, String word) {
		return new JsonObject()
				.put("hash", hash)
				.put("request", "Schreib ein Abenteuer von " + word)
				.put("request_word_1", word)
				.put("story", word + " flog, und das Raumschiff größer als der Mond wartete schon.");
	}
}
