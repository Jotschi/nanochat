package de.jotschi.ai.processor.parquet;

import de.jotschi.ai.processor.DatasetEntry;

public class KleinerAstronautParquetEntry implements DatasetEntry {

	private long id;
	private String topic;
	private String adj1;
	private String adj2;
	private String verb;
	private String word1;
	private String word2;
	private String text;

	public KleinerAstronautParquetEntry(long id, String topic, String adj1, String adj2, String verb, String word1,
			String word2, String text) {
		this.id = id;
		this.topic = topic;
		this.adj1 = adj1;
		this.adj2 = adj2;
		this.verb = verb;
		this.word1 = word1;
		this.word2 = word2;
		this.text = text;
	}

	@Override
	public String hash() {
		return null;
	}

	public String topic() {
		return topic;
	}

	public String adj1() {
		return adj1;
	}

	public String adj2() {
		return adj2;
	}

	public String text() {
		return text;
	}

	public String verb() {
		return verb;
	}

	public String word1() {
		return word1;
	}

	public String word2() {
		return word2;
	}

}
