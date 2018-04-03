package edu.nyu.nlu.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.nyu.nlu.find.FindDifferent;
import edu.nyu.nlu.io.CsvReader;
import edu.nyu.nlu.io.CsvWriter;

public class TestSentence {

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
            String val = find.find(2, row[0]);
            String[] newRow = new String[]{row[0], row[1], val};
            newTest.add(newRow);
        }
        CsvWriter writer = new CsvWriter();
        try {
            writer.write(newTest, new String[]{"text", "label", "number of structures"}, "./data/test_return.csv");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

}
