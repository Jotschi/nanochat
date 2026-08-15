package de.jotschi.ai.converter.stage3;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;
import io.metaloom.ai.genai.utils.TextUtils;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

public class Dataset2ChatTest extends AbstractGeneratorTest {


	public static long SKIPPED_STORIES = 0;
	public static long TRIMMED_STORIES = 0;
	public static long TOTAL = 0;
	
	private File INPUT_DATASET;
	private File NANOCHAT_DIR;
	private File OUTPUT_TRAIN;
	private File OUTPUT_VAL;

	@BeforeAll
	public void setupFolders() {
		this.INPUT_DATASET = new File("dataset", "kleiner_astronaut_qa_v5_combined.jsonl");
//	public static final File OUTPUT_TRAIN = new File("dataset", "kleiner_astronaut_conversations_v3_train.jsonl");
//	public static final File OUTPUT_VAL = new File("dataset", "kleiner_astronaut_conversations_v3_val.jsonl");
		
		this.NANOCHAT_DIR = new File(getNanoChatCacheDir());
		
		this.OUTPUT_TRAIN = new File(NANOCHAT_DIR, "kleiner_astronaut_conversations_v5_train.jsonl");
		this.OUTPUT_VAL = new File(NANOCHAT_DIR, "kleiner_astronaut_conversations_v5_val.jsonl");
		
	}
	
	@Test
	public void testConvert() throws IOException {
		if (OUTPUT_TRAIN.exists()) {
			OUTPUT_TRAIN.delete();
		}

		if (OUTPUT_VAL.exists()) {
			OUTPUT_VAL.delete();
		}

		List<String> lines = FileUtils.readLines(INPUT_DATASET, Charset.defaultCharset());
		int total = lines.size();
		int val_size = (int) (total * 0.05f);

		for (String line : lines) {
			File destFile = OUTPUT_TRAIN;
			if (val_size >= 0) {
				destFile = OUTPUT_VAL;
				val_size--;
			}
			line = line.trim();
			line = TextUtils.trimNonAsciiFromEnd(line);
			if (line.isEmpty()) {
				continue;
			}
			try {
				JsonObject json = new JsonObject(line);
				List<JsonArray> list = toChat(json);
				if (!list.isEmpty()) {
					for (JsonArray conv : list) {
						FileUtils.writeStringToFile(destFile, conv.encode() + "\n", Charset.defaultCharset(), true);
						TOTAL++;
					}
				}
			} catch (Exception e) {
				System.out.println(line);
				e.printStackTrace();
			}
		}
		System.out.println("Total skipped: " + SKIPPED_STORIES);
		System.out.println("Total trimmed: " + TRIMMED_STORIES);
		System.out.println("Total written: " + TOTAL);
	}

	private List<JsonArray> toChat(JsonObject json) {
		String request = json.getString("request");
		if (request == null) {
			return Collections.emptyList();
		}
		String request_word_1 = json.getString("request_word_1");
		String request_word_2 = json.getString("request_word_2");
		String answer_word = json.getString("answer_word");

		String story = json.getString("story");
//		if (TextUtils.count('*', story) > 0) {
//			return Collections.emptyList();
//		}
//		if (story.length() > 17000) {
//			String trimmedStory = TextUtils.softClamp(story, 1700, '.', '!', '?', '\n');
//			if (hasWord(trimmedStory, request_word_1) && hasWord(trimmedStory, request_word_2)
//					&& hasWord(trimmedStory, answer_word)) {
//				story = trimmedStory;
//				TRIMMED_STORIES++;
//			} else {
//				SKIPPED_STORIES++;
//				return Collections.emptyList();
//			}
//		}
//		if (story.contains("\":")) {
//			return Collections.emptyList();
//		}
		String answer = json.getString("answer");
		String question = json.getString("question");

		List<JsonArray> chats = new ArrayList<>();
		JsonArray chat = new JsonArray();
		chat.add(message("user", request));
		chat.add(message("assistant", story).put("rl_key1", request_word_1).put("rl_key2", request_word_2));
		chat.add(message("user", question));
		chat.add(message("assistant", answer).put("rl_key1", answer_word));
		chats.add(chat);

//		JsonArray chat2 = new JsonArray();
//		chat2.add(message("user", request));
//		chat2.add(message("assistant", story).put("rl_key1", request_word_1).put("rl_key2", request_word_2));
//		chats.add(chat2);

		return chats;
	}

	private boolean hasWord(String text, String word) {
		String wordNeedle = word.toLowerCase();
		wordNeedle = wordNeedle.substring(0, wordNeedle.length() - 2);
		return text.toLowerCase().contains(wordNeedle);
	}

	private JsonObject message(String role, String msg) {
		return new JsonObject().put("role", role).put("content", msg);
	}
}
