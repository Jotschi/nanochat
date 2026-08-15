package de.jotschi.ai.converter.stage1;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import io.metaloom.utils.hash.HashUtils;
import io.metaloom.utils.hash.MD5;
import io.vertx.core.json.JsonObject;

public class JsonlHasherTest {

	@Test
	public void testHash() throws IOException {
		File destFile = new File("dataset", "kleiner_astronaut_qa_v4_hashed.jsonl");
		List<String> lines = FileUtils.readLines(new File("dataset", "kleiner_astronaut_qa_v4.jsonl"),
				Charset.defaultCharset());
		for (String line : lines) {
			JsonObject json = new JsonObject(line);
			String text = json.getString("story");
			MD5 hash = HashUtils.computeMD5(text);
			json.put("hash", hash.toString());
			json.remove("id");
			FileUtils.writeStringToFile(destFile, json.encode() + "\n", Charset.defaultCharset(), true);
		}
	}
}
