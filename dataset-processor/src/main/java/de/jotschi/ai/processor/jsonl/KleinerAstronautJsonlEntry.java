package de.jotschi.ai.processor.jsonl;

import java.util.List;

import de.jotschi.ai.processor.DatasetEntry;

public class KleinerAstronautJsonlEntry implements DatasetEntry {

	private String hash;
	private String text;
	private String story;
	private String verb;
	private String word;
	private String topic;
	private String spaceWord;
	private String adjective1;
	private String adjective2;
	private String len;
	private String start;
	private List<String> names;

	public KleinerAstronautJsonlEntry() {
	}

	public String getHash() {
		return hash;
	}

	@Override
	public String hash() {
		return hash;
	}

	public void setHash(String hash) {
		this.hash = hash;
	}

	public String getStory() {
		return story;
	}

	public void setStory(String story) {
		this.story = story;
	}

	public String getText() {
		return text;
	}

	public void setText(String text) {
		this.text = text;
	}

	public String getVerb() {
		return verb;
	}

	public void setVerb(String verb) {
		this.verb = verb;
	}

	public String getWord() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}

	public String getTopic() {
		return topic;
	}

	public void setTopic(String topic) {
		this.topic = topic;
	}

	public String getSpaceWord() {
		return spaceWord;
	}

	public void setSpaceWord(String spaceWord) {
		this.spaceWord = spaceWord;
	}

	public String getAdjective1() {
		return adjective1;
	}

	public void setAdjective1(String adjective1) {
		this.adjective1 = adjective1;
	}

	public String getAdjective2() {
		return adjective2;
	}

	public void setAdjective2(String adjective2) {
		this.adjective2 = adjective2;
	}

	public String getLen() {
		return len;
	}

	public void setLen(String len) {
		this.len = len;
	}

	public String getStart() {
		return start;
	}

	public void setStart(String start) {
		this.start = start;
	}

	public List<String> getNames() {
		return names;
	}

	public void setNames(List<String> names) {
		this.names = names;
	}

}
