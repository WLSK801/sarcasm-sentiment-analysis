package edu.nyu.nlu.find;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;

public class PunctuationFind {
    
    private StanfordCoreNLP pipeline;

    public PunctuationFind() {
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
    
    public String getKinsmanSentiment(String pun, String text, Kinsman kinsman) {
        String result = "";
        Tree tree = processTree(text);
        List<Tree> leaveList = tree.getLeaves();
        if (leaveList.size() == 0)
            return null;
        List<Tree> punRef = new ArrayList<>();
        find(tree, pun, punRef);
        for (Tree leaf : punRef) {
            if (kinsman == Kinsman.PARENT) {
                Tree current = leaf.parent(tree).parent(tree);
                if (current != null) {
                    result = result + RNNCoreAnnotations.getPredictedClass(current) + ",";
                }
            }
            if (kinsman == Kinsman.AUNT) {
                if (leaf.parent(tree).parent(tree) == null ||
                        leaf.parent(tree).parent(tree).siblings(tree).size() == 0)
                    continue;
                List<Tree> aunts = leaf.parent(tree).parent(tree).siblings(tree);
                for (Tree aunt : aunts) {
                    result += RNNCoreAnnotations.getPredictedClass(aunt) + ",";
                }
            }
            if (kinsman == Kinsman.COUSION) {
                if (leaf.parent(tree).parent(tree) == null ||
                        leaf.parent(tree).parent(tree).siblings(tree).size() == 0)
                    continue;
                List<Tree> aunts = leaf.parent(tree).parent(tree).siblings(tree);
                for (Tree aunt : aunts) {
                    for (Tree cousion : aunt.getChildrenAsList()) {
                        if (!cousion.isLeaf())
                            result += RNNCoreAnnotations.getPredictedClass(cousion)+ ",";
                    }
                    
                }
            }
        }
        return result;
    }
    
    public List<Tree> findPunctuation(String pun, String text) {
        Tree tree = processTree(text);
        List<Tree> leaveList = tree.getLeaves();
        if (leaveList.size() == 0)
            return null;
        List<Tree> punRef = new ArrayList<>();
        find(tree, pun, punRef);
        return punRef;
    }
    private void find(Tree treeNode, String pun, List<Tree> punList) {
        if (treeNode.isLeaf()) {
            if (pun.equals(treeNode.label().value())) {
                punList.add(treeNode);
            }
            else {
                return;
            }
        }
        List<Tree> children = treeNode.getChildrenAsList();
        for (Tree child : children) {
            find(child, pun, punList);
        }
        return;
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
}
