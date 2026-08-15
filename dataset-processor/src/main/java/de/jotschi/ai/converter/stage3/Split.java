package de.jotschi.ai.converter.stage3;

/**
 * Deterministic, story-level train/validation split.
 * <p>
 * The bucket is derived from the story's own MD5 hash, so:
 * <ul>
 * <li>it is reproducible - no shuffle, no seed to remember;</li>
 * <li>every consumer agrees without sharing state, which means the chat
 * conversations and the pretraining base data hold out <em>the same</em>
 * stories;</li>
 * <li>it is stable as the corpus grows - adding stories never moves an existing
 * one across the boundary.</li>
 * </ul>
 * The previous implementations got this wrong in two different ways: the chat
 * converter took the unshuffled first 5% of the file, and the base-data splitter
 * shuffled with an unseeded Random. Neither could be reproduced, and a story
 * could land in chat-train while its own text was in base-val.
 */
public final class Split {

	public static final double DEFAULT_VAL_FRACTION = 0.05;

	private static final long BUCKETS = 10_000L;

	private Split() {
	}

	/**
	 * @param hash          hex-encoded story hash (MD5), at least 8 characters
	 * @param valFraction   fraction held out, e.g. 0.05 for 5%
	 * @return true when this story belongs to the validation split
	 */
	public static boolean isHeldOut(String hash, double valFraction) {
		if (hash == null || hash.length() < 8) {
			throw new IllegalArgumentException("Need at least an 8 character hex hash, got: " + hash);
		}
		if (valFraction < 0 || valFraction >= 1) {
			throw new IllegalArgumentException("valFraction must be in [0,1), got: " + valFraction);
		}
		long bucket = Long.parseUnsignedLong(hash.substring(0, 8), 16) % BUCKETS;
		return bucket < Math.round(valFraction * BUCKETS);
	}

	public static boolean isHeldOut(String hash) {
		return isHeldOut(hash, DEFAULT_VAL_FRACTION);
	}
}
