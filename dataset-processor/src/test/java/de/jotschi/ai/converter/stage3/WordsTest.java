package de.jotschi.ai.converter.stage3;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

public class WordsTest {

	@Test
	public void testMatchesGermanInflection() {
		assertThat(Words.contains("Er stieg in das Raumschiffes ein.", "Raumschiff")).isTrue();
		assertThat(Words.contains("Die Schatzkarten lagen dort.", "Schatzkarte")).isTrue();
	}

	@Test
	public void testShortWordsAreNotStemmed() {
		// "Ax" would stem to "" and then match anything at all. This is the bug
		// that made every short reward key trivially satisfied.
		assertThat(Words.contains("Eine Geschichte ohne den Namen.", "Ax")).isFalse();
		assertThat(Words.contains("Ax fliegt zum Mond.", "Ax")).isTrue();
	}

	@Test
	public void testNumbersAreNotStemmed() {
		assertThat(Words.contains("Es waren 1234 Sterne.", "1234")).isTrue();
		assertThat(Words.contains("Es waren 99 Sterne.", "123456")).isFalse();
	}

	@Test
	public void testMissingWordDoesNotMatch() {
		assertThat(Words.contains("Mira fliegt zum Mond.", "Raumschiff")).isFalse();
	}

	@Test
	public void testNullsAreSafe() {
		assertThat(Words.contains(null, "Mira")).isFalse();
		assertThat(Words.contains("Mira fliegt.", null)).isFalse();
		assertThat(Words.stem(null)).isNull();
	}
}
