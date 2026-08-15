package de.jotschi.ai.converter;

import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QAGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.qa.QuestionAnswerResult;

/**
 * Smoke check for the question/answer generator - the half that produces the
 * third and fourth turns of a conversation. Needs a live LLM endpoint, so it is
 * excluded from the surefire run.
 */
public class QAGeneratorTest extends AbstractGeneratorTest {

	@Test
	public void testQA() {
		QAGenerator gen = new QAGenerator(llm(), model());
		for (int i = 0; i < 5; i++) {
			QuestionAnswerResult result = gen.generateQA(AnfrageGeneratorTest.STORY_1);
			if (result == null) {
				System.err.println("No QA generated after the retry budget was exhausted");
				continue;
			}
			System.out.println("OK: [" + result.typ() + "] " + result.question() + " => " + result.answer()
					+ " / " + result.word());
		}
	}
}
