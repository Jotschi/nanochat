package de.jotschi.ai.processor.chat.llm.anfrage;

import java.util.List;

import de.jotschi.ai.processor.chat.llm.AbstractGenerator;
import io.metaloom.ai.genai.llm.LLMContext;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.prompt.Prompt;
import io.metaloom.ai.genai.llm.prompt.impl.PromptImpl;
import io.metaloom.ai.genai.utils.TextUtils;
import io.vertx.core.json.JsonObject;

public class AnfrageGenerator extends AbstractGenerator {

	private static final int ANFRAGE_OUTPUT_MAX_LEN = 150;

	private static final int STORY_MAX_LEN = 150;

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

	public AnfrageResult generateTriggerQuestion(String story, String word1, String word2) {
		for (int i = 0; i < RETRY_MAX; i++) {
			String randomAnfang = ANFRAGE_LIST.get(RND.nextInt(ANFRAGE_LIST.size()));

			Prompt prompt = new PromptImpl(GENERATE_ANFRAGE_PROMPT_TEMPLATE);
			story = TextUtils.softClamp(story, STORY_MAX_LEN, '?', '.', '!', '\n');
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
				String word1Needle = word1.toLowerCase();
				word1Needle = word1Needle.substring(0, word1Needle.length() - 2);
				String word2Needle = word1.toLowerCase();
				word2Needle = word2Needle.substring(0, word2Needle.length() - 2);

				boolean hasWord1 = anfrage.toLowerCase().contains(word1Needle);
				boolean hasWord2 = anfrage.toLowerCase().contains(word2Needle);

				if (!hasWord1 || !hasWord2) {
					System.err.println("Retry.. " + i + " " + anfrage + " - lacking word " + word1Needle + " / " + word2Needle);
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
}
