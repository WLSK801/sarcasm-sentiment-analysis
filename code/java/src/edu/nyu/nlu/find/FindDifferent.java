package edu.nyu.nlu.find;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.ejml.simple.SimpleMatrix;

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
    
    public List<String> findWithLabel(int different, String input) {
        if (different > 5 || different < 0) {
            throw new IllegalArgumentException("");
        }
        Tree tree = processTree(input);
        List<String> pairs = new ArrayList<>();
        List<Tree> leaveList = tree.getLeaves();
        if (leaveList.size() == 0)
            return pairs;
        recursiveFindWithLabel(tree, different, pairs);
        return pairs;
    }
    
    public List<String[]> checkLabelDistribution(String input) {
        Tree tree = processTree(input);
        List<String[]> pairs = new ArrayList<>();
        List<Tree> leaveList = tree.getLeaves();
        if (leaveList.size() == 0)
            return pairs;
        recursiveCheckhLabel(tree, pairs);
        return pairs;
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
    private void recursiveFindWithLabel(Tree treeNode, int diff, List<String> paitInfor) {
        if (treeNode.isLeaf())
            return;
        List<Tree> children = treeNode.getChildrenAsList();
        if (children.size() <= 0) return;
        int[] sentimentArray = new int[children.size()];
        int index = 0;
        for (Tree child : children) {
            int sentiment = RNNCoreAnnotations.getPredictedClass(child);
            sentimentArray[index++] = sentiment;
        }
        for (int i = 0; i < sentimentArray.length; i++) {
            for (int j = i + 1; j < sentimentArray.length; j++) {
                int currentDifferent = Math.abs(sentimentArray[i] - sentimentArray[j]);
                if (currentDifferent >= diff) {
                    String pair =  children.get(i).label().toString() + "," + 
                 children.get(j).label().toString() + "," + currentDifferent;
                    paitInfor.add(pair);
                }
            }
        }
        
        for (Tree child : children) {
            recursiveFindWithLabel(child, diff, paitInfor);
        }
        return;
    }
    private void recursiveCheckhLabel(Tree treeNode, List<String[]> paitInfor) {
        if (treeNode.isLeaf())
            return;
        List<Tree> children = treeNode.getChildrenAsList();
        if (children.size() <= 0) return;
        SimpleMatrix sentiment_new = RNNCoreAnnotations.getPredictions(treeNode);
        if (sentiment_new == null) {
            return;
        }
        double[] distribution = sentiment_new.getMatrix().getData();
        int maxKey = 0;
        double maxValue = Double.MIN_VALUE;
        double secondMax = Double.MIN_VALUE;
        for (int i = 0; i < distribution.length; i++) {
            if (distribution[i] > maxValue) {
                maxKey = i;
                secondMax = maxValue;
                maxValue = distribution[i];
            }
            
        }
        
        boolean check = false;
        double current = maxValue;
        for (int i = maxKey + 1; i < distribution.length; i++) {
            if (distribution[i] > current) check = true;
            current = distribution[i];
        }
        current = maxValue;
        for (int i = maxKey - 1; i >= 0; i--) {
            if (distribution[i] > current) check = true;
            current = distribution[i];
        }
        paitInfor.add(new String[]{"" + (check ? 1 : 0), String.format("%.2f", maxValue - secondMax)});
        for (Tree child : children) {
            recursiveCheckhLabel(child, paitInfor);
        }
        return;
        
        
    }
}
