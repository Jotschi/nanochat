package de.jotschi.ai.processor.jsonl;

import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageResult;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QAGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QuestionAnswerResult;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.utils.TextUtils;
import io.vertx.core.json.JsonObject;

public class KleinerAstronautJsonlHandler {

	private QAGenerator qaGenerator;

	private AnfrageGenerator anfrageGenerator;

	public KleinerAstronautJsonlHandler(LLMProvider llm, LargeLanguageModel model) {
		this.anfrageGenerator = new AnfrageGenerator(llm, model);
		this.qaGenerator = new QAGenerator(llm, model);
	}

	public JsonObject process(KleinerAstronautJsonlEntry entry) {
		try {
			String text = entry.getText();

			if (TextUtils.count('*', text) > 0) {
				System.err.println("Skipping story " + entry.hash() + " - malformed content '*'");
				return null;
			}

			String word1 = entry.getWord();
			String word2 = entry.getSpaceWord();

			// Poor mans declension handling
			String word1Needle = word1.toLowerCase();
			word1Needle = word1Needle.substring(0, word1Needle.length() - 2);
			String word2Needle = word1.toLowerCase();
			word2Needle = word2Needle.substring(0, word2Needle.length() - 2);

			// Only accept stories that are consistent with the words
			boolean hasWord1 = text.toLowerCase().contains(word1Needle);
			boolean hasWord2 = text.toLowerCase().contains(word2Needle);

			if (!hasWord1 || !hasWord2) {
				System.err.println(
						"Skipping story " + entry.hash() + " - lacking words: " + word1Needle + " / " + word2Needle);
				return null;
			}

			AnfrageResult result = anfrageGenerator.generateTriggerQuestion(text, word1, word2);
			QuestionAnswerResult qa = qaGenerator.generateQA(text);
			if (qa != null) {
				JsonObject jsonOut = new JsonObject();
				jsonOut.put("hash", entry.hash());
				jsonOut.put("request", result.anfrage());
				jsonOut.put("request_word_1", result.word1());
				jsonOut.put("request_word_2", result.word2());
				jsonOut.put("story", text);
				jsonOut.put("story_adj_1", entry.getAdjective1());
				jsonOut.put("story_adj_2", entry.getAdjective2());
				jsonOut.put("story_topic", entry.getTopic());
				jsonOut.put("story_verb", entry.getVerb());
				jsonOut.put("story_word_1", word1);
				jsonOut.put("story_word_2", word2);
				jsonOut.put("story_target_len", entry.getLen());
				jsonOut.put("story_names", entry.getNames());
				jsonOut.put("story_start", entry.getStart());
				jsonOut.put("question", qa.question());
				jsonOut.put("question_typ", qa.typ());
				jsonOut.put("answer", qa.answer());
				jsonOut.put("answer_word", qa.word());
				return jsonOut;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;

	}

}
