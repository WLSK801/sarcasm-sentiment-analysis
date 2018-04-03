package edu.nyu.nlu.io;

import com.opencsv.CSVReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class CsvReader {
    
    public List<String[]> read(String path) throws IOException {
        List<String[]> data = new ArrayList<>();
        try (
                Reader reader = Files.newBufferedReader(Paths.get(path));
                CSVReader csvReader = new CSVReader(reader);
            ) {
                // Reading Records One by One in a String array
                String[] nextRecord;
                while ((nextRecord = csvReader.readNext()) != null) {
                   data.add(nextRecord);
                }
            }
        return data;
    }
}