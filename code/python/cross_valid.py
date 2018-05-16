__author__      = 'Yu Cao'

import pandas as pd
import numpy as np
import collections
import pickle
from string import punctuation as Punct
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def vocabBuild(countList, freqThreshold=20):
    vocab = collections.Counter()
    for item in countList: 
        if item is not None:
            vocab.update(item.keys())

    vocab = vocab.most_common()
    for c, item in enumerate(vocab):
        if item[1] < freqThreshold: break
    
    return [ x[0] for x in vocab[:c] ]

def embed(vocab, df):
    def process(count):
        if count is None: return None
        vector = np.empty(lenVocab)
        vector.fill(-1.)
        for key in count.keys(): 
            if key in vocab:
                 vector[vocab.index(key)] = count[key]
        
        return vector

    lenVocab = len(vocab)
    df['vector'] = pd.Series(
        [ process(x) for x in df['count'] ],
        index=df.index
    )



def removeNone(vectorDF, labelDF):
    tlist = [(v, l) for (v, l) in zip(vectorDF, labelDF) 
        if v is not None
    ]
    return np.array([v for (v, l) in tlist]), np.array([l for (v, l) in tlist])

def getBowVectorizer(df, smallerDim=100):
    vectorizer = CountVectorizer()
    dimReducer = PCA(n_components=smallerDim)
    _, corpus = removeNone(df['vector'], df['text'])
    vectorizer.fit(corpus)
    dimReducer.fit(vectorizer.transform(corpus).toarray())
    return vectorizer, dimReducer

def getDimReducer(df, smallerDim=100):
    dimReducer = PCA(n_components=smallerDim)
    X, _ = removeNone(df['vector'], df['label'])
    return dimReducer.fit(X)

def bowToArray(df, bowVectorizer, dimReducer):
    _, corpus = removeNone(df['vector'], df['text'])
    X_bow = bowVectorizer.transform(corpus)
    if dimReducer is not None:
        return dimReducer.transform(X_bow.toarray())
    else:
        return X_bow.toarray()

def getPredictorTarget(df, bowVectorizer=None, dimReducer=None):
    if bowVectorizer is None:
        X, y = removeNone(df['vector'], df['label'])
        if dimReducer is not None:
            X = dimReducer.transform(X)
    else:
        X = bowToArray(df, bowVectorizer, dimReducer)
        _, y = removeNone(df['vector'], df['label'])
    
    return X, y

def modelTrain(df, bowVectorizer=None, dimReducer=None):
    modelList = [
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]
    X, y = getPredictorTarget(df, bowVectorizer, dimReducer)
    for model in modelList:
        model.fit(X, y)
    return modelList

def modelTrainComb(df, bowVec, dimRed1, dimRed2):
    modelList = [
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]
    X1, y = getPredictorTarget(df, bowVec, dimRed1)
    X2, _ = getPredictorTarget(df, None, None)
    X = np.concatenate((X1, X2), axis=1)
    for model in modelList:
        model.fit(X, y)
    return modelList        

def evalModels(strategy, df, modelList, bowVectorizer=None, dimReducer=None):
    X, y = getPredictorTarget(df, bowVectorizer, dimReducer)
    for model in modelList:
        y_pred = model.predict(X)
        modelName = str(model)[:str(model).find('(')]
        with open('crossvalid\\' + modelName + '-' + strategy + '.txt', 'a') as fout:
            print('%.4f\t%.4f\t%.4f\t%.4f' % (
                precision_score(y, y_pred),
                recall_score(y, y_pred),
                accuracy_score(y, y_pred),
                f1_score(y, y_pred)
            ),
                file=fout
            )

def evalRandGuess(df):
    _, y = removeNone(df['count'], df['label'])
    y_pred = np.random.randint(2, size=len(y))
    for i in range(len(y_pred)):
        if y_pred[i] == 0: y_pred[i] = -1
    
    with open('crossvalid\\randGuess.txt', 'a') as fout:
        print('%.4f\t%.4f\t%.4f\t%.4f' % (
                precision_score(y, y_pred),
                recall_score(y, y_pred),
                accuracy_score(y, y_pred),
                f1_score(y, y_pred)
            ),
            file=fout
        )

def evalCombine(df, modelList, bowVec, dimRed1, dimRed2):
    X1, y = getPredictorTarget(df, bowVec, dimRed1)
    X2, _ = getPredictorTarget(df, None, None)
    X = np.concatenate((X1, X2), axis=1)
    for model in modelList:
        y_pred = model.predict(X)
        modelName = str(model)[:str(model).find('(')]
        with open('crossvalid\\%s-combine.txt' % modelName, 'a') as fout:
            print('%.4f\t%.4f\t%.4f\t%.4f' % (
                precision_score(y, y_pred),
                recall_score(y, y_pred),
                accuracy_score(y, y_pred),
                f1_score(y, y_pred)
            ),
                file=fout
            )



def loadData(filename):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, encoding='utf-8')

def getfnameList(n):
    fnameList = []
    for i in range(n):
        fnameList.append(('train%02d.dat' % i, 'test%02d.dat' % i))
    return fnameList

def crossValidation(n):
    fnameList = getfnameList(n)
    for i in range(n):
        trainSet = loadData('crossvalid\\' + fnameList[i][0])
        testSet = loadData('crossvalid\\' + fnameList[i][1])

        vocab = vocabBuild(trainSet['count'], 20)
        embed(vocab, trainSet)
        embed(vocab, testSet)

        modelList = modelTrain(trainSet)
        evalModels('sc', testSet, modelList)

        dim_reducer = getDimReducer(trainSet)
        modelList = modelTrain(trainSet, dimReducer=dim_reducer)
        evalModels('sc-pca', testSet, modelList, dimReducer=dim_reducer)

        bowVectorizer, dimReducer = getBowVectorizer(trainSet)
        modelList = modelTrain(trainSet, bowVectorizer, dimReducer)
        evalModels('bow-pca', testSet, modelList, bowVectorizer, dimReducer)

        evalRandGuess(testSet)

        modelList = modelTrainComb(trainSet, bowVectorizer, dimReducer, dim_reducer)
        evalCombine(testSet, modelList, bowVectorizer, dimReducer, dim_reducer)

        

'''
main
'''
crossValidation(5)
