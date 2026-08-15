package de.jotschi.ai.processor.jsonl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageResult;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QAGenerator;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.utils.TextUtils;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

public class KleinerAstronautJsonlHandler {

	private static final int STORY_MAX_LEN = 300;

	private QAGenerator qaGenerator;

	private AnfrageGenerator anfrageGenerator;

	public KleinerAstronautJsonlHandler(LLMProvider llm, LargeLanguageModel model) {
		this.anfrageGenerator = new AnfrageGenerator(llm, model);
		this.qaGenerator = new QAGenerator(llm, model);
	}

	public JsonObject process(JsonObject entry) {
		try {
			String hash = entry.getString("hash");
			String text = entry.getString("text");
			if (text == null) {
				text = entry.getString("story");
			}

			// Trim the len down
			text = TextUtils.softClamp(text, STORY_MAX_LEN, '?', '.', '!', '\n');

			if (!TextUtils.isAscii(text)) {
				System.err.println("[" + hash + "] failed - malformed content 'non ascii'");
				return null;
			}

			if (TextUtils.count('*', text) > 0) {
				System.err.println("[" + hash + "] failed - (malformed content '*')");
				return null;
			}

			if (TextUtils.isEnglish(text)) {
				System.err.println("[" + hash + "] failed - (text is english)");
				return null;
			}

			List<String> storyWords = loadWords(entry, text);
			if (storyWords.isEmpty()) {
				System.err.println("[" + hash + "] failed - (lacking words)");
				return null;
			}

			List<String> names = loadNames(entry, text);
			if (names.isEmpty()) {
				System.err.println("[" + hash + "] failed - (Names did not match with story)");
				return null;
			}

			List<String> qaWordPool = new ArrayList<>();
			qaWordPool.addAll(names);
			qaWordPool.addAll(storyWords);
			if (qaWordPool.isEmpty()) {
				System.err.println("[" + hash + "] failed - (Empty word pool)");
				return null;
			}
			AnfrageResult result = anfrageGenerator.generateTriggerQuestion(text, qaWordPool);
			if (result == null) {
				System.err.println("[" + hash + "] failed - (Trigger not generated)");
				return null;
			}
//			QuestionAnswerResult qa = qaGenerator.generateQA(text);
//			if (qa != null) {
			JsonObject jsonOut = new JsonObject();
			jsonOut.put("hash", hash);
			jsonOut.put("request", result.anfrage());
			jsonOut.put("request_word_1", result.word1());
			jsonOut.put("request_word_2", result.word2());
			jsonOut.put("story", text);
//				jsonOut.put("story_adj_1", entry.getAdjective1());
//				jsonOut.put("story_adj_2", entry.getAdjective2());
//				jsonOut.put("story_topic", entry.getTopic());
//				jsonOut.put("story_verb", entry.getVerb());
//				jsonOut.put("story_word_1", word1);
//				jsonOut.put("story_word_2", word2);
//				jsonOut.put("story_target_len", entry.getLen());
			// jsonOut.put("story_names", entry.getNames());
//				jsonOut.put("story_start", entry.getStart());
//				jsonOut.put("question", qa.question());
//				jsonOut.put("question_typ", qa.typ());
//				jsonOut.put("answer", qa.answer());
//				jsonOut.put("answer_word", qa.word());
			return jsonOut;
//			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;

	}

	private List<String> loadWords(JsonObject entry, String text) {
		String word1 = entry.getString("story_word_1");
		if (word1 == null) {
			word1 = entry.getString("word");
		}

		String word2 = entry.getString("story_word_2");
		if (word2 == null) {
			word2 = entry.getString("spaceWord");
		}

		// Poor mans declension handling
		String word1Needle = word1.toLowerCase();
		word1Needle = word1Needle.substring(0, word1Needle.length() - 2);
		String word2Needle = word1.toLowerCase();
		word2Needle = word2Needle.substring(0, word2Needle.length() - 2);

		// Only accept stories that are consistent with the words
		boolean hasWord1 = text.toLowerCase().contains(word1Needle);
		boolean hasWord2 = text.toLowerCase().contains(word2Needle);

		if (!hasWord1 || !hasWord2) {
			return Collections.emptyList();
		}

		return List.of(word1, word2);

	}

	private List<String> loadNames(JsonObject entry, String story) {
		JsonArray names = entry.getJsonArray("story_names");
		if (names == null) {
			names = entry.getJsonArray("names");
		}

		story = story.toLowerCase();
		List<String> foundNames = new ArrayList<>();
		for (int i = 0; i < names.size(); i++) {
			String name = names.getString(i);
			if (story.contains(name.toLowerCase())) {
				foundNames.add(name);
			}
		}

		return foundNames;
	}

}
