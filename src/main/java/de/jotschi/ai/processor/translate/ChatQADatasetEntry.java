package de.jotschi.ai.processor.translate;

import de.jotschi.ai.processor.DatasetEntry;

public class ChatQADatasetEntry implements DatasetEntry {

	private final String messages;
	private final String source;
	private final String hash;

	public ChatQADatasetEntry(String hash, String messages, String source) {
		this.hash = hash;
		this.messages = messages;
		this.source = source;
	}

	public String source() {
		return source;
	}

	public String messages() {
		return messages;
	}

	@Override
	public String toString() {
		return "[" + source() + " => title: " + messages() + "]";
	}

	@Override
	public String hash() {
		return hash;
	}
}
