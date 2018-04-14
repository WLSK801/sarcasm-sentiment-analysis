package edu.nyu.nlu.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.nyu.nlu.find.FindDifferent;
import edu.nyu.nlu.io.CsvReader;
import edu.nyu.nlu.io.CsvWriter;

public class TestDocument {

    public static void main(String[] args) {
        FindDifferent find = new FindDifferent();
        //String text = "poor, white people in the usa are particular good at this.";
        CsvReader reader = new CsvReader();
        List<String[]> test = new ArrayList<>();
        try {
            test = reader.read("./data/test.csv");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        List<String[]> newTest = new ArrayList<>(test.size());
        for (String[] row : test) {
            List<String> pairs = find.findWithLabel(0, row[0]);
            StringBuilder sb = new StringBuilder();
            for (String pair : pairs) {
                sb.append(pair).append("\t");
            }
            String[] newRow = new String[]{row[0], row[1], sb.toString()};
            newTest.add(newRow);
        }
        CsvWriter writer = new CsvWriter();
        try {
            writer.write(newTest, new String[]{"text", "label", "pairs_diff"}, "./data/test_return_pair.csv");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

}
