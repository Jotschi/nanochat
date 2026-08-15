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
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

/**
 * Stage 3a: turn the base-data records into nanochat chat conversations.
 * <p>
 * Emits one JSON <em>array</em> per line. A record that carries a question and
 * an answer becomes a four-turn conversation:
 *
 * <pre>
 * user      request
 * assistant story      rl_key1 = request_word_1, rl_key2 = request_word_2
 * user      question
 * assistant answer     rl_key1 = answer_word
 * </pre>
 *
 * A record without them (the v6 schema) falls back to the first two turns.
 * <p>
 * Run with:
 * {@code mvn -q compile exec:java -Dexec.mainClass=de.jotschi.ai.converter.stage3.Dataset2Chat}
 */
public class Dataset2Chat {

	public static final String TRAIN_NAME = "kleiner_astronaut_conversations_train.jsonl";
	public static final String VAL_NAME = "kleiner_astronaut_conversations_val.jsonl";

	/** Inputs used when none are given on the command line. */
	public static final List<File> DEFAULT_INPUTS = List.of(
			new File("dataset/stories_done", "kleiner_astronaut_qa_v5_combined.jsonl"),
			new File("dataset", "kleiner_astronaut_qa_v6.jsonl"));

	private final double valFraction;

	private long written;
	private long fourTurn;
	private long twoTurn;
	private long skipped;
	private long heldOut;

	public Dataset2Chat(double valFraction) {
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
			inputs.addAll(DEFAULT_INPUTS);
		}
		if (outDir == null) {
			outDir = Settings.nanochatCacheDir();
		}
		new Dataset2Chat(valFraction).run(inputs, outDir);
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

		Path train = outDir.toPath().resolve(TRAIN_NAME);
		Path val = outDir.toPath().resolve(VAL_NAME);

		// One writer per split, opened once - the previous version reopened the
		// file for every single line.
		try (BufferedWriter trainOut = Files.newBufferedWriter(train, StandardCharsets.UTF_8);
				BufferedWriter valOut = Files.newBufferedWriter(val, StandardCharsets.UTF_8)) {
			for (File input : present) {
				System.out.println("Reading " + input);
				try (Stream<String> lines = Files.lines(input.toPath(), StandardCharsets.UTF_8)) {
					for (String line : (Iterable<String>) lines::iterator) {
						convertLine(line, trainOut, valOut);
					}
				}
			}
		}

		System.out.println("Written:   " + written + " (4-turn: " + fourTurn + ", 2-turn: " + twoTurn + ")");
		System.out.println("Held out:  " + heldOut);
		System.out.println("Skipped:   " + skipped);
		System.out.println("Train:     " + train);
		System.out.println("Val:       " + val);
	}

	private void convertLine(String rawLine, BufferedWriter trainOut, BufferedWriter valOut) throws IOException {
		String line = rawLine.strip();
		if (line.isEmpty()) {
			return;
		}
		JsonObject json;
		try {
			json = new JsonObject(line);
		} catch (Exception e) {
			skipped++;
			System.err.println("Unparseable line skipped: " + e.getMessage());
			return;
		}
		JsonArray conversation = toChat(json);
		if (conversation == null) {
			skipped++;
			return;
		}
		boolean isVal = Split.isHeldOut(json.getString("hash"), valFraction);
		BufferedWriter out = isVal ? valOut : trainOut;
		out.write(conversation.encode());
		out.write('\n');
		written++;
		if (isVal) {
			heldOut++;
		}
	}

	/**
	 * @return the conversation, or null when the record cannot be used
	 */
	public JsonArray toChat(JsonObject json) {
		String request = json.getString("request");
		String story = json.getString("story");
		if (request == null || request.isBlank() || story == null || story.isBlank()) {
			return null;
		}

		JsonArray chat = new JsonArray();
		chat.add(message("user", request));
		JsonObject storyTurn = message("assistant", story);
		putRewardKey(storyTurn, "rl_key1", json.getString("request_word_1"), story);
		putRewardKey(storyTurn, "rl_key2", json.getString("request_word_2"), story);
		chat.add(storyTurn);

		// The question/answer pair only exists in the v5 schema. When it is
		// there we get the reading-comprehension turns that are the whole point
		// of the model; when it is not, the two-turn conversation still teaches
		// request -> story.
		String question = json.getString("question");
		String answer = json.getString("answer");
		if (question != null && !question.isBlank() && answer != null && !answer.isBlank()) {
			chat.add(message("user", question));
			JsonObject answerTurn = message("assistant", answer);
			putRewardKey(answerTurn, "rl_key1", json.getString("answer_word"), answer);
			chat.add(answerTurn);
			fourTurn++;
		} else {
			twoTurn++;
		}
		return chat;
	}

	/**
	 * Attach a reward key, but only when it actually occurs in the text it is
	 * meant to describe. The generator was supposed to guarantee this and did
	 * not: {@code word2Needle} was derived from {@code word1}, so the second
	 * keyword was never verified. An unverifiable key would make the RL reward
	 * unreachable, so drop it rather than ship it.
	 */
	private void putRewardKey(JsonObject turn, String keyName, String word, String text) {
		if (word == null || word.isBlank()) {
			return;
		}
		if (!Words.contains(text, word)) {
			return;
		}
		turn.put(keyName, word);
	}

	private JsonObject message(String role, String content) {
		return new JsonObject().put("role", role).put("content", content);
	}

	public long written() {
		return written;
	}

	public long fourTurn() {
		return fourTurn;
	}

	public long twoTurn() {
		return twoTurn;
	}

	public long skipped() {
		return skipped;
	}

	public long heldOut() {
		return heldOut;
	}
}
