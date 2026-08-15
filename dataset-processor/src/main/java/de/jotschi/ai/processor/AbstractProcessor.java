package de.jotschi.ai.processor;

import java.io.File;
import java.io.FilenameFilter;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import org.apache.arrow.dataset.file.FileFormat;
import org.apache.arrow.dataset.file.FileSystemDatasetFactory;
import org.apache.arrow.dataset.jni.NativeMemoryPool;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.dataset.source.Dataset;
import org.apache.arrow.dataset.source.DatasetFactory;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Row;
import org.apache.arrow.vector.table.Table;

import me.tongfei.progressbar.ProgressBar;

public abstract class AbstractProcessor<T extends DatasetEntry> implements Processor<T> {

	public void process(File datasetFolder, String filter, DatasetEntryHandler<T> processor) {
		ProgressBar pb = new ProgressBar("Scan", 60_000);

		FilenameFilter filenameFilter = (dir, name) -> name.endsWith(".parquet");
		for (File file : datasetFolder.listFiles(filenameFilter)) {
			if (file.getName().contains(filter)) {
				scanFile(pb, file, processor);
			} else {
				System.out.println("Skipping " + file);
			}
		}
	}

	private void scanFile(ProgressBar pb, File file, DatasetEntryHandler<T> processor) {
		AtomicLong id = new AtomicLong(0);
		scanFile(file, row -> {
			T entry = toDatasetEntry(id.getAndIncrement(), row);
			processor.process(entry);
			if (pb != null) {
				pb.step();
			}
		}, fields());
	}

	protected abstract List<String> fields();

	protected abstract T toDatasetEntry(long id, Row row);

	private void scanFile(File file, Consumer<Row> rowConsumer, List<String> fields) {
		String uri = file.toURI().toString();
		String[] fieldsArray = fields.toArray(String[]::new);
		ScanOptions options = new ScanOptions(/* batchSize */ 32768, Optional.of(fieldsArray));
		try (BufferAllocator allocator = new RootAllocator();
				DatasetFactory datasetFactory = new FileSystemDatasetFactory(allocator, NativeMemoryPool.getDefault(),
						FileFormat.PARQUET, uri)) {

			Dataset dataset = datasetFactory.finish();
			Scanner scanner = dataset.newScan(options);
			try (ArrowReader reader = scanner.scanBatches()) {
				while (reader.loadNextBatch()) {
					try (VectorSchemaRoot root = reader.getVectorSchemaRoot()) {
						try (Table t = new Table(root)) {
							t.forEach(rowConsumer);
						}
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
