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
        String text = "poor, white people in the usa are particular good at this.";
        String text2 = "Their computer-animated faces are very expressive.";
        //String text2 = "Their computer-animated faces are very expressive.";
        for (String ele :find.findWithLabel(2, text)) {
            System.out.println(ele);
        }
    }

}