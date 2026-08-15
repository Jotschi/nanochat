package de.jotschi.ai.converter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import de.jotschi.ai.processor.chat.llm.AbstractGenerator;
import io.metaloom.ai.genai.llm.LLMContext;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.prompt.Prompt;
import io.metaloom.ai.genai.llm.prompt.impl.PromptImpl;
import io.metaloom.utils.hash.HashUtils;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

public class StoryGenerator extends AbstractGenerator {

	public final static String PROMPT_TEMPLATE = """
			Du bist ein Author von Kindergeschichten vom kleinen Astronauten.
			Schreibe eine kreative Weltraum-Abenteuergeschichte, in der das Thema „${topic}“ vorkommt.

			Seed: ${seed}

			Regeln:
			${rules}

			Inspiration:
			'${adjective1} ${word} ${verb} ${adjective2} ${spaceWord}'""";

	private static final int MAX_STORY_LEN = 1000;

	public StoryGenerator(LLMProvider llm, LargeLanguageModel model) {
		super(llm, model);
	}

	public JsonObject generate() {

		for (int i = 0; i < RETRY_MAX; i++) {
			StorySeed seed = StorySeed.seed();

			Prompt prompt = new PromptImpl(PROMPT_TEMPLATE);
			prompt.set("verb", seed.verb());
			prompt.set("word", seed.word());
			prompt.set("topic", seed.topic());
			prompt.set("spaceWord", seed.spaceWord());
			prompt.set("adjective1", seed.adjective1());
			prompt.set("adjective2", seed.adjective2());
			prompt.set("seed", seed.randomStr());
			prompt.set("rules", randomRules(seed));

//		System.out.println(prompt.input());
			LLMContext ctx = LLMContext.ctx(prompt, model);
			ctx.setSeed(RND.nextInt());
			ctx.setTemperature(1);
			try {
				String story = llm.generate(ctx);
				// Quality Gate
				if (passQualityGate(story, seed.len())) {
					System.out.println("OK [len: " + story.length() + "] - " + seed.len());
					JsonObject json = new JsonObject();
					json.put("hash", HashUtils.computeMD5(story).toString());
					json.put("text", story);
					json.put("verb", seed.verb());
					json.put("word", seed.word());
					json.put("topic", seed.topic());
					json.put("spaceWord", seed.spaceWord());
					json.put("adjective1", seed.adjective1());
					json.put("adjective2", seed.adjective2());
					json.put("names", new JsonArray(seed.names(story)));
					return json;
				}
			} catch (Exception e) {
				System.err.println("Retry [" + i + "] - error");
				e.printStackTrace();
			}
		}
		System.err.println("Failed after " + RETRY_MAX);
		return null;
	}

	private String randomRules(StorySeed seed) {

		StringBuilder b = new StringBuilder();
		List<String> rules = new ArrayList<>();
		rules.add("Keine Einleitung, kein Titel, keine Überschrift.");
		rules.add("Gib ausschließlich die Geschichte aus.");
		rules.add("Nenne keinen Autor und erwähne dich selbst nicht.");
		rules.add("Maximal " + seed.len() + " Sätze.");
		rules.add("Beginne mit einem ungewöhnlich kreativen, kindgerechten Einstieg.");
		rules.add("Ersetze schwierige Wörter durch leicht verständliche Umschreibungen.");
		rules.add("Die Geschichte soll eine klare Handlung haben und kindgerecht spannend sein.");
		rules.add("Sei kreativ mit den Namen - hier ein paar Beispiele (" + String.join(",", seed.names()) + ")");
		rules.add("Sei kreativ mit dem Anfang - hier ein paar Beispiele für den Anfang ("
				+ String.join(",", seed.beginnings()) + ")");

		Collections.shuffle(rules);
		for (String r : rules) {
			b.append(" * " + r + "\n");
		}

		return b.toString();

	}

	private boolean passQualityGate(String story, int len) {
		if (story == null || story.isEmpty()) {
			System.err.println("Retry - quality gate failed (null)");
			return false;
		}
//		if (story.length() > MAX_STORY_LEN) {
//			System.err.println("Retry - quality gate failed (len) - " + len);
//			return false;
//		}
		story = story.toLowerCase();
		Set<String> flags = Set.of("generated", "_helper", "minäkello", "Gesamtlänge:", "Hinweis:", "Inspiration:",
				"Handlung:", "quadratic", "mtl:", "_text", "shift_", "admin", "mtkxqh", "(Ende", "written", "_max_",
				"*Ende", "mtmax", "Hauptcharaktere", "quadratic_equation", "story", "mtzE", "children", "Author",
				"little", "Note:", "prompt", "Instruction", "starbringer", "count", "Titel:", "mtkzrZ",
				"ranktenmutigorbit", "Ende der Geschichte", "Autor:", "_content", "*", "..");
		for (String flag : flags) {
			if (story.contains(flag.toLowerCase())) {
				System.err.println("Retry - quality gate failed (flag - '" + flag + "')");
				return false;
			}
		}

		return true;
	}
}
