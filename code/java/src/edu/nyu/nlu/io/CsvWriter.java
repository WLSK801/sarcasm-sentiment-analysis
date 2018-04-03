package edu.nyu.nlu.io;

import com.opencsv.CSVWriter;

import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.io.IOException;

public class CsvWriter {

    public void write(List<String[]> data, String[] headers, String path) throws IOException {
        if (data == null || headers == null || path == null) {
            throw new IllegalArgumentException("arguments cannnot be null.");
        }
        else if (data.size() <= 0) {
            throw new IllegalArgumentException("data is empty.");
        }
        else if (data.get(0).length != headers.length) {
            throw new IllegalArgumentException("heads length must match columns number.");
        }
        try (Writer writer = Files.newBufferedWriter(Paths.get(path));

                CSVWriter csvWriter = new CSVWriter(writer, CSVWriter.DEFAULT_SEPARATOR, CSVWriter.DEFAULT_QUOTE_CHARACTER,
                        CSVWriter.DEFAULT_ESCAPE_CHARACTER, CSVWriter.DEFAULT_LINE_END);) {
            csvWriter.writeNext(headers);

            for (String[] row : data) {
                if (row.length != headers.length) {
                    throw new IllegalArgumentException("heads length must match columns number.");
                }
                csvWriter.writeNext(row);
            }
        }
    }
}