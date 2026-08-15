package de.jotschi.ai.processor;

import java.io.File;

public interface Processor<T extends DatasetEntry> {

	void process(File datasetFolder, String filter);

}
