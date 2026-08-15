package de.jotschi.ai.converter.stage3;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

import org.junit.jupiter.api.Test;

public class SplitTest {

	@Test
	public void testStable() {
		String hash = "75baf618ff5de97b9e057ad60e6a2690";
		boolean first = Split.isHeldOut(hash);
		for (int i = 0; i < 100; i++) {
			assertThat(Split.isHeldOut(hash)).as("split must not vary between calls").isEqualTo(first);
		}
	}

	@Test
	public void testFractionIsApproximatelyHonoured() throws NoSuchAlgorithmException {
		int total = 20_000;
		int heldOut = 0;
		for (int i = 0; i < total; i++) {
			if (Split.isHeldOut(md5("story-" + i), 0.05)) {
				heldOut++;
			}
		}
		double ratio = (double) heldOut / total;
		assertThat(ratio).as("5%% split over %d hashes", total).isBetween(0.04, 0.06);
	}

	@Test
	public void testZeroFractionHoldsOutNothing() throws NoSuchAlgorithmException {
		for (int i = 0; i < 500; i++) {
			assertThat(Split.isHeldOut(md5("story-" + i), 0.0)).isFalse();
		}
	}

	@Test
	public void testGrowingTheCorpusDoesNotMoveExistingStories() throws NoSuchAlgorithmException {
		// The whole point of hashing rather than shuffling: adding stories must
		// never reclassify one that is already assigned.
		String[] existing = { md5("a"), md5("b"), md5("c") };
		boolean[] before = new boolean[existing.length];
		for (int i = 0; i < existing.length; i++) {
			before[i] = Split.isHeldOut(existing[i]);
		}
		for (int i = 0; i < 5_000; i++) {
			Split.isHeldOut(md5("filler-" + i));
		}
		for (int i = 0; i < existing.length; i++) {
			assertThat(Split.isHeldOut(existing[i])).isEqualTo(before[i]);
		}
	}

	@Test
	public void testRejectsBadInput() {
		assertThatThrownBy(() -> Split.isHeldOut(null)).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> Split.isHeldOut("abc")).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> Split.isHeldOut("75baf618", 1.0)).isInstanceOf(IllegalArgumentException.class);
	}

	private static String md5(String text) throws NoSuchAlgorithmException {
		MessageDigest digest = MessageDigest.getInstance("MD5");
		return HexFormat.of().formatHex(digest.digest(text.getBytes()));
	}
}
