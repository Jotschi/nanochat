package de.jotschi.ai.converter;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.IOUtils;

import io.metaloom.ai.genai.utils.TextUtils;

public class StorySeed {

	private static List<String> ALL_ADJECTIVES;
	private static List<String> ALL_NOMEN_NO_SPACE;
	private static List<String> ALL_NOMEN_SPACE;
	private static List<String> ALL_TOPICS;
	private static List<String> ALL_VERBS;
	private static List<String> ALL_BEGINNINGS;
	private static List<String> ALL_NAMES;

	static {
		ALL_ADJECTIVES = IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/adjectives.lst"),
				Charset.defaultCharset());
		ALL_NOMEN_NO_SPACE = IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/nomen_no_space.lst"),
				Charset.defaultCharset());
		ALL_NOMEN_SPACE = IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/nomen_space.lst"),
				Charset.defaultCharset());
		ALL_TOPICS = IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/topics.lst"),
				Charset.defaultCharset());
		ALL_VERBS = IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/verbs.lst"),
				Charset.defaultCharset());
		ALL_BEGINNINGS = filterText(IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/beginnings.lst"),
				Charset.defaultCharset()));
		ALL_NAMES = IOUtils.readLines(StorySeed.class.getResourceAsStream("/story/names.lst"),
				Charset.defaultCharset());
	}

	private int len;
	private String randomStr;
	private List<String> names;
	private List<String> beginnings;
	private String verb;
	private String topic;
	private String word;
	private String spaceWord;
	private String adjective1;
	private String adjective2;
	private static final Random RND = new Random();

	public StorySeed() {
		this.len = RND.nextInt(7, 12);
		this.randomStr = TextUtils.randomBase64Str(64);
		this.names = randomNames(4);
		this.beginnings = randomBeginnings(4);
		this.verb = ALL_VERBS.get(RND.nextInt(ALL_VERBS.size()));
		this.topic = ALL_TOPICS.get(RND.nextInt(ALL_TOPICS.size()));
		this.word = ALL_NOMEN_NO_SPACE.get(RND.nextInt(ALL_NOMEN_NO_SPACE.size()));
		this.spaceWord = ALL_NOMEN_SPACE.get(RND.nextInt(ALL_NOMEN_SPACE.size()));
		this.adjective1 = ALL_ADJECTIVES.get(RND.nextInt(ALL_ADJECTIVES.size()));
		this.adjective2 = ALL_ADJECTIVES.get(RND.nextInt(ALL_ADJECTIVES.size()));
	}

	private static List<String> filterText(List<String> lines) {
		return lines.stream().filter(t -> TextUtils.isAscii(t)).toList();
	}

	private List<String> randomBeginnings(int len) {
		List<String> list = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			String name = ALL_BEGINNINGS.get(RND.nextInt(ALL_BEGINNINGS.size()));
			list.add(name);
		}
		return list;
	}

	private List<String> randomNames(int len) {
		List<String> list = new ArrayList<>();
		for (int i = 0; i < len; i++) {
			String name = ALL_NAMES.get(RND.nextInt(ALL_NAMES.size()));
			list.add(name);
		}
		return list;
	}

	public static StorySeed seed() {
		return new StorySeed();
	}

	public String verb() {
		return verb;
	}

	public String randomStr() {
		return randomStr;
	}

	public String word() {
		return word;
	}

	public String topic() {
		return topic;
	}

	public String spaceWord() {
		return spaceWord;
	}

	public String adjective1() {
		return adjective1;
	}

	public String adjective2() {
		return adjective2;
	}

	public List<String> names() {
		return names;
	}

	public List<String> beginnings() {
		return beginnings;
	}

	public int len() {
		return len;
	}

	/**
	 * Return the names that match up with the story.
	 * 
	 * @param story
	 * @return
	 */
	public List<String> names(String story) {
		return names.stream().filter(n -> story.toLowerCase().contains(n.toLowerCase())).toList();
	}
}
