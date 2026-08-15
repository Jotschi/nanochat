package de.jotschi.ai.processor;

@FunctionalInterface
public interface DatasetEntryHandler<T extends DatasetEntry> {

    void process(T entry);
}
