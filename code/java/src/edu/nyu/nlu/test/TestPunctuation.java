package edu.nyu.nlu.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.nyu.nlu.find.FindDifferent;
import edu.nyu.nlu.find.Kinsman;
import edu.nyu.nlu.find.PunctuationFind;
import edu.nyu.nlu.io.CsvReader;
import edu.nyu.nlu.io.CsvWriter;

public class TestPunctuation {

    public static void main(String[] args) {
        PunctuationFind find = new PunctuationFind();
        String text = "poor, white people in the usa are particular good at this.";
        String text2 = "Their computer-animated faces are very expressive.";
        //String text2 = "Their computer-animated faces are very expressive.";
        System.out.println(find.findPunctuation(".", text2));
        System.out.println(find.getKinsmanSentiment(",", text, Kinsman.AUNT));
    }

}