package edu.nyu.nlu.exmaple;

import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.*;
import java.util.*;

import org.ejml.simple.SimpleMatrix;

public class TestSentiment {
    public static String text = "Joe Smith was born in California. "
            + "In 2017, he went to Paris, France in the summer. " + "His flight left at 3:00pm on July 10th, 2017. "
            + "After eating some escargot for the first time, Joe said, \"That was delicious!\" "
            + "He sent a postcard to his sister Jane Smith. "
            + "After hearing about Joe's trip, Jane decided she might go to France one day.";

    public static void main(String[] args) {
        // set up pipeline properties
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,parse,sentiment");
        // set a property for an annotator, in this case the coref annotator is
        // being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        // build pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // create a document object
        CoreDocument document = new CoreDocument(text);
        // annnotate the document
        pipeline.annotate(document);

        // 10th token of the document
        CoreLabel token = document.tokens().get(10);
        System.out.println("Example: token");
        System.out.println(token);
        System.out.println();

        // text of the first sentence
        String sentenceText = document.sentences().get(0).text();
        System.out.println("Example: sentence");
        System.out.println(sentenceText);
        System.out.println();

        // second sentence
        CoreSentence sentence = document.sentences().get(1);

        // dependency parse for the second sentence
        SemanticGraph dependencyParse = sentence.dependencyParse();
        System.out.println("Example: dependency parse");
        System.out.println(dependencyParse);
        System.out.println();
        
        // get first root
        System.out.println(dependencyParse.getFirstRoot());
        
        // get sentiment 
        Tree tree = sentence.sentimentTree();
        int sentiment = RNNCoreAnnotations.getPredictedClass(tree.firstChild().firstChild());
        System.out.println(tree);
        System.out.println(tree.firstChild().firstChild());
        System.out.println(sentiment);
        SimpleMatrix sentiment_new = RNNCoreAnnotations.getPredictions(tree); 
        System.out.println(sentiment_new);
    }

}
