package de.jotschi.ai.converter.stage1;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import io.metaloom.ai.genai.llm.LLMContext;
import io.metaloom.ai.genai.llm.LLMProvider;
import io.metaloom.ai.genai.llm.LargeLanguageModel;
import io.metaloom.ai.genai.llm.prompt.Prompt;
import io.metaloom.ai.genai.llm.prompt.impl.PromptImpl;
import io.metaloom.ai.genai.utils.TextUtils;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;

public class StoryBeginningListGeneratorTest extends AbstractGeneratorTest {

	String promptStr = """
			Erstelle eine Liste von 10 Satzanfängen (Nur die ersten drei Wörter).
			Lass dir Anfänge für eine Weltraum Kindergeschichte einfallen.

			Sei kreativ. Verwende auch ungewöhnliche Anfänge.

			Gib die Liste als JSON aus:
			{
				"anfänge": ["..."];
			}
			""";

	@Test
	public void testGenerateStoryBeginnings() throws IOException {

		File destFile = new File("dataset", "beginnings.lst");
		LargeLanguageModel llmModel = model();
		LLMProvider provider = llm();
		Prompt prompt = new PromptImpl(promptStr);

		while (true) {
			try {
				LLMContext ctx = LLMContext.ctx(prompt, llmModel);
				ctx.setTemperature(1);
				ctx.setSeed(new Random().nextInt());
				JsonObject json = provider.generateJson(ctx);
				JsonArray array = json.getJsonArray("anfänge");
				System.out.println(array.encodePrettily());
				for (int i = 0; i < array.size(); i++) {
					String val = array.getString(i);
					val = TextUtils.trimToWords(val, 3);
					FileUtils.writeStringToFile(destFile, val + "\n", Charset.defaultCharset(), true);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}
}
