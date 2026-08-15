package de.jotschi.ai.processor.parquet;

import de.jotschi.ai.converter.stage3.Words;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import org.apache.commons.io.FileUtils;

import de.jotschi.ai.processor.DatasetEntryHandler;
import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageResult;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QAGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QuestionAnswerResult;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.utils.TextUtils;
import io.vertx.core.json.JsonObject;

public class KleinerAstronautParquetHandler implements DatasetEntryHandler<KleinerAstronautParquetEntry> {

	private final File outputFile;

	private QAGenerator qaGenerator;

	private AnfrageGenerator anfrageGenerator;

	public KleinerAstronautParquetHandler(File outputFile, LLMProvider ollama, LargeLanguageModel model) {
		this.outputFile = outputFile;
		this.anfrageGenerator = new AnfrageGenerator(ollama, model);
		this.qaGenerator = new QAGenerator(ollama, model);
	}

	@Override
	public void process(KleinerAstronautParquetEntry entry) {
//		if (entry.id() <= 6870) {
//			return;
//		}
		try {
			String text = entry.text();

			if (TextUtils.count('*', text) > 0) {
				System.err.println("Skipping story " + entry.hash() + " - malformed content '*'");
				return;
			}

			String word1 = entry.word1();
			String word2 = entry.word2();

			// Only accept stories that are consistent with the words.
			// Words.contains does the declension stemming with the length/number
			// guards. The previous version derived the second needle from word1,
			// so word2 was never actually checked against the story.
			boolean hasWord1 = Words.contains(text, word1);
			boolean hasWord2 = Words.contains(text, word2);

			if (!hasWord1 || !hasWord2) {
				System.err.println(
						"Skipping story " + entry.hash() + " - lacking words: " + word1 + " / " + word2);
				return;
			}

			AnfrageResult result = anfrageGenerator.generateTriggerQuestion(text, List.of(word1, word2));
			QuestionAnswerResult qa = qaGenerator.generateQA(text);
			if (qa != null) {
				JsonObject jsonOut = new JsonObject();
				jsonOut.put("hash", entry.hash());
				jsonOut.put("request", result.anfrage());
				jsonOut.put("request_word_1", result.word1());
				jsonOut.put("request_word_2", result.word2());
				jsonOut.put("story", text);
				jsonOut.put("story_adj_1", entry.adj1());
				jsonOut.put("story_adj_2", entry.adj2());
				jsonOut.put("story_topic", entry.topic());
				jsonOut.put("story_verb", entry.verb());
				jsonOut.put("story_word_1", word1);
				jsonOut.put("story_word_2", word2);
				jsonOut.put("question", qa.question());
				jsonOut.put("question_typ", qa.typ());
				jsonOut.put("answer", qa.answer());
				jsonOut.put("answer_word", qa.word());
				try {
					FileUtils.writeStringToFile(outputFile, jsonOut.encode() + "\n", Charset.defaultCharset(), true);
				} catch (IOException e) {
					System.err.println("Processing failed");
					e.printStackTrace();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
