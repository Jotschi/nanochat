package de.jotschi.ai.processor.chat.llm.anfrage;

import java.util.Collections;
import java.util.List;

import de.jotschi.ai.processor.chat.llm.AbstractGenerator;
import io.metaloom.ai.genai.llm.LLMContext;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.prompt.Prompt;
import io.metaloom.ai.genai.llm.prompt.impl.PromptImpl;
import io.metaloom.ai.genai.utils.TextUtils;

public class AnfrageGenerator extends AbstractGenerator {

	private static final int ANFRAGE_OUTPUT_MAX_LEN = 150;

	private static final List<String> ANFRAGE_LIST = List.of("Schreibe mir", "Erfinde eine", "Schreib eine",
			"Schreib mir", "Schreib ein", "Erzähle mir", "Kannst du mir", "Bitte erzähle mir", "Eine Geschichte",
			"Ein Abendteuer", "Es war einmal", "Ich möchte gerne", "Lies mir", "Kannst du");

	// Du möchtest eine kurze Kindergeschichte hören.

	public static final String GENERATE_ANFRAGE_PROMPT_TEMPLATE = """
			Du bist ein 8-jähriges Kind.

			Schreibe eine einzige Anfrage, die zu der zu folgenden Text passt.

			Wichtig:

			Die Anfrage muss mit '${anfang}' beginnen und '${word1}'/'${word2}' beinhalten.

			Maximal 1 Satz.

			Maximal 100 Zeichen.

			Benutze sehr einfache Sprache.

			Schreibe sehr kurz.

			Erlaubte Anfänge für Anfragen:
			"Schreibe mir", "Erfinde eine", "Schreib eine", "Schreib mir",
			"Schreib ein", "Erzähle mir", "Kannst du mir", "Bitte erzähle mir",
			"Eine Geschichte", "Ein Abenteuer", "Es war einmal",
			"Ich möchte gerne", "Lies mir..vor", "Kannst du"

			Vervollständige diesen Anfang:
			'${anfang}..'

			Text:
			${text}
			""";

	public AnfrageGenerator(LLMProvider llm, LargeLanguageModel model) {
		super(llm, model);
	}

	public AnfrageResult generateTriggerQuestion(String story, List<String> words) {
		String word1 = pickRandomAndRemove(words);
		String word2 = pickRandomAndRemove(words);

		// Pick another word if it is not in the story
		if (!hasWord(story, word1) && !words.isEmpty()) {
			word1 = pickRandomAndRemove(words);
		}

		// Pick another word if it is not in the story
		if (!hasWord(story, word2) && !words.isEmpty()) {
			word2 = pickRandomAndRemove(words);
		}

		if (!hasWord(story, word1) || !hasWord(story, word2)) {
			return null;
		}

		for (int i = 0; i < RETRY_MAX; i++) {
			String randomAnfang = ANFRAGE_LIST.get(RND.nextInt(ANFRAGE_LIST.size()));

			Prompt prompt = new PromptImpl(GENERATE_ANFRAGE_PROMPT_TEMPLATE);
			prompt.set("text", TextUtils.quote(story));
			prompt.set("word1", word1);
			prompt.set("word2", word2);
			prompt.set("anfang", randomAnfang);

			LLMContext ctx = LLMContext.ctx(prompt, model);
			ctx.setTemperature(1);
			try {
				String anfrage = llm.generate(ctx);

				// Retry on invalid JSON
				if (anfrage == null || anfrage.isEmpty() || anfrage.contains("..")) {
					System.err.println("Retry.. " + i + " Text invalid/incomplete: " + anfrage);
					continue;
				}
				if (TextUtils.isEnglish(anfrage)) {
					System.err.println("Retry.. " + i + " Text is english: " + anfrage);
					continue;
				}
				if (!TextUtils.isAscii(anfrage)) {
					System.err.println("Retry.. " + i + " Text is non ascii: " + anfrage);
					continue;
				}

				// Retry on invalid case - Quality gate - anfrage
				anfrage = anfrage.replace("\"", "").replace("'", "");
				if (!anfrage.startsWith(randomAnfang) || anfrage.contains(":")
						|| anfrage.toLowerCase().contains("anfrage")
						|| anfrage.toLowerCase().trim().equalsIgnoreCase(randomAnfang.toLowerCase().trim())
						|| anfrage.toLowerCase().contains("anweisung") || anfrage.length() > ANFRAGE_OUTPUT_MAX_LEN) {
					int len = anfrage.length();
					// System.out.println(prompt.input());
					System.err.println("Retry.. " + i + " " + anfrage + " / " + randomAnfang + ", len: " + len
							+ ", limit: " + ANFRAGE_OUTPUT_MAX_LEN);
					continue;
				}

				// Poor mans declension handling
				if (!hasWord(anfrage, word1)) {
					System.err.println("Retry.. " + i + " " + anfrage + " - lacking word: " + word1);
					continue;
				}

				if (!hasWord(anfrage, word2)) {
					System.err.println("Retry.. " + i + " " + anfrage + " - lacking word: " + word2);
					continue;
				}

				return new AnfrageResult(anfrage, word1, word2);
			} catch (Exception e) {
				System.out.println("Retry.. " + i + " " + e.getMessage());
				e.printStackTrace();
				// NOOP
			}

		}
		return null;
	}

	private boolean hasWord(String text, String word) {
		if (word == null || text == null) {
			return false;
		}
		String needle = word.toLowerCase();
		// Poormans declination handling
		needle = needle.substring(0, needle.length() - 2);
		return text.toLowerCase().contains(" " + needle);
	}

	private String pickRandomAndRemove(List<String> words) {
		Collections.shuffle(words);
		return words.removeFirst();
	}
}
