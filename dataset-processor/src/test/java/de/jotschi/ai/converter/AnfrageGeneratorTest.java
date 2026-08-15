package de.jotschi.ai.converter;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import de.jotschi.ai.converter.stage1.AbstractGeneratorTest;
import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageGenerator;
import de.jotschi.ai.processor.chat.llm.anfrage.AnfrageResult;

/**
 * Smoke check for the request ("Anfrage") generator. Needs a live LLM endpoint,
 * so it is excluded from the surefire run.
 */
public class AnfrageGeneratorTest extends AbstractGeneratorTest {

	public static final String STORY_1 = """
			Es war einmal ein kleiner Astronaut namens Max, der sich auf einer kleinen Welt mit vielen Sternen und Planeten bewegte. Eines Tages erzählte ihm sein Großvater von einem ungewöhnlichen Phänomen, das man Kometen nennt – wie große Sternechen aus Stein, die durch den Himmel ziehen.\n\nEin Abendabend kam, und Max hatte endlich genug davon, zu schauen, was über ihm passierte. Er wollte die Nachtessen mit seinem Großvater verbringen, aber in seinen Gedanken war er immer noch an diesen Kometen.\n\n\"Großvater\", sagte Max, \"kannst du mir sagen, wie man ein Komet sehen kann?\" Der alte Mann lachte und nahm Max auf den Schoß. \"Aber bitte schau nicht nur draußen!\" sprach er. Er führte Max in sein Zimmer, das mit vielen Stern-Bildern dekoriert war, die ihn an seine Abenteuer im Weltall erinnerten.\n\nMax und sein Großvater kuschelten sich zusammen auf den Bettwangen und begannen ein Märchen zu erzählen. In ihrem Märchen lebte eine Komet namens Clara, die mit ihren Freunden – Sternbögen, Mondlichtern und anderen Weltraumkreaturen – durchs All reiste.\n Cookies machten in kleinen Galaxien. Dabei gärtnerten sie kreativ neue Planeten, und Robert, ein fiktiver Held, der sich selbst als Kreativ-Robert bezeichnete, half ihnen dabei. Clara war die heldenhafteste Komet, die es je gab, denn sie hatte das unglaubliche Talent, aus kleinen Steinchen große Schönheit in den Sternenwäldern zu schaffen.\n\n\"Kometen\", sagte Großvater mit einer weichen Stimme, \"sind wie kleine Helden der Nacht, die unter dem riesigen Himmel ihr Abenteuer erleben.\" Max nickte an und fühlte sich sehr wohl auf seiner kleinen Welt in diesem abendlichen Moment voller Wärme und Fantasie.\n\nAls das Mondlicht hereinbrach, schlug es ein Ende zu dieser Geschichte. \"Wir müssen noch eine Nachtessen haben\", sagte Großvater leise, aber mit einer freudigen Stimmung in der Luft. Max starrte aus dem Fenster, wie die Sterne den Himmel erhellten und dachte: \"Ich wünschte ich könnte einem Kometen begegnen\".\n\n\"Kommen wir doch gemeinsam nachts aufs Dach\", rief Großvater. \"Und sagen uns der Mond, dass du als kleiner Astronaut mit deinem Kometen-Freund Clara in den Sternenwäldern zu ihm kommt!\"\n\nMax' Herz schlug vor Freude. Und so schauten sie gemeinsam nach oben und sahen, wie das weiße Licht des Mondes sich im Himmel spiegelte – eine wunderbare Nacht für Max, in der er seine eigenen Kometen-Abenteuer ausfechten konnte.
			""";

	@Test
	public void testSingleAnfrage() {
		AnfrageGenerator gen = new AnfrageGenerator(llm(), model());
		AnfrageResult result = gen.generateTriggerQuestion(STORY_1, new ArrayList<>(List.of("Max", "Hexe")));
		System.out.println("OK: " + result.anfrage() + " | " + result.word1() + " | " + result.word2());
	}
}
