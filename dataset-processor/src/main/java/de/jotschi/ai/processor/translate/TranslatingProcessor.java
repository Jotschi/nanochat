package de.jotschi.ai.processor.translate;

import java.io.File;
import java.util.List;

import org.apache.arrow.vector.table.Row;
import org.apache.arrow.vector.util.JsonStringHashMap;

import de.jotschi.ai.processor.AbstractProcessor;

public class TranslatingProcessor extends AbstractProcessor<ChatQADatasetEntry> {

	@Override
	public void process(File datasetFolder, String filter) {

	}

	@Override
	protected ChatQADatasetEntry toDatasetEntry(long id, Row row) {
		List<?> msgs = row.getList("messages");
		for (Object msg : msgs) {
			if (msg instanceof JsonStringHashMap jsonMap) {
				Object role = jsonMap.get("role");
				Object content = jsonMap.get("content");
				System.out.println("Role:" + role);
				System.out.println("Content:" + content);
			} else {
				System.out.println(msg.getClass().getName());
			}
		}
		String source = row.getVarCharObj("source");
		return new ChatQADatasetEntry("" + row.getRowNumber(), "", source);
	}

	@Override
	protected List<String> fields() {
		return List.of("messages", "source");
	}

}
