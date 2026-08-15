package de.jotschi.ai.converter.stage3;

import io.metaloom.ai.genai.utils.TextUtils;

/**
 * Keyword matching for the RL reward keys.
 * <p>
 * German inflects, so "Raumschiff" in the request may appear as "Raumschiffes"
 * in the story. The generator's quality gate handles this by chopping the last
 * two characters off the needle - but only for words longer than 5 characters
 * and only when the word is not a number. Both guards matter:
 * <ul>
 * <li>without the length guard a 2-character word stems to the empty string,
 * which {@code contains("")} accepts for <em>any</em> text;</li>
 * <li>without the number guard "42" would stem to the empty string too.</li>
 * </ul>
 * {@link TextUtils#hasWord} drops both guards, and so did the Python side
 * ({@code key[:-2]} applied unconditionally), which is why short reward keys
 * always matched. Use this class instead.
 */
public final class Words {

	private static final int MIN_STEM_LEN = 5;

	private Words() {
	}

	/** Lower-cased matching needle, shortened by two characters when it is safe. */
	public static String stem(String word) {
		if (word == null) {
			return null;
		}
		String lower = word.toLowerCase();
		if (lower.length() > MIN_STEM_LEN && !TextUtils.isNumber(lower)) {
			return lower.substring(0, lower.length() - 2);
		}
		return lower;
	}

	/** True when {@code text} contains {@code word}, tolerating German inflection. */
	public static boolean contains(String text, String word) {
		String needle = stem(word);
		if (text == null || needle == null || needle.isEmpty()) {
			return false;
		}
		return text.toLowerCase().contains(needle);
	}
}
