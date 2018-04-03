package edu.nyu.nlu.find;

import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;

public class FindDifferent {

    private Properties props;

    private StanfordCoreNLP pipeline;

    public FindDifferent() {
        // set up pipeline properties
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,parse,sentiment");
        // set a property for an annotator, in this case the coref annotator is
        // being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        // build pipeline
        pipeline = new StanfordCoreNLP(props);

    }

    public String find(int different, String input) {
        if (different > 5 || different < 0) {
            throw new IllegalArgumentException("");
        }
        Tree tree = processTree(input);
        List<Tree> leaveList = tree.getLeaves();
        if (leaveList.size() == 0)
            return "";
        int count = recursiveFind(tree, different);
        return count + "";
    }

    private Tree processTree(String input) {
        // create a document object
        CoreDocument document = new CoreDocument(input);
        // annnotate the document
        pipeline.annotate(document);
        CoreSentence sentence = document.sentences().get(0);
        // get sentiment
        Tree tree = sentence.sentimentTree();
        return tree;
    }

    private int recursiveFind(Tree treeNode, int diff) {
        if (treeNode.isLeaf())
            return 0;
        List<Tree> children = treeNode.getChildrenAsList();
        int max = 0;
        int min = 100;
        for (Tree child : children) {
            int sentiment = RNNCoreAnnotations.getPredictedClass(child);
            if (sentiment < min)
                min = sentiment;
            if (sentiment > max)
                max = sentiment;
        }
        int count = 0;
        if (max - min >= diff)
            count = 1;
        for (Tree child : children) {
            count = count + recursiveFind(child, diff);
        }
        return count;
    }
}
