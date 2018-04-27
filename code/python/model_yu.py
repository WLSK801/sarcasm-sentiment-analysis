__author__      = 'Yu Cao'

import pandas as pd
import numpy as np
import collections
from string import punctuation as Punct
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def loadCorpus(corpus_path):
    df = pd.read_csv(corpus_path)
    df = df.fillna('')
    df['count'] = pd.Series(
        [ rawToCount(x) for x in df['pairs_diff'] ], 
        index = df.index
    )
    return shuffle(df)

def isPunct(s):
    return s in Punct or s in ["``", "''"]

def rawToCount(pairs_diff):
    pairs_diff = pairs_diff.replace('@', '')
    tripleList = [x.split(',') for x in pairs_diff.split('\t')]
    tripleList = [x for x in tripleList
        if len(x) == 3 and not isPunct(x[0]) and not isPunct(x[1])
    ]
    if len(tripleList) < 3: return None

    count = {}
    for triple in tripleList[:-1]:
        key = '_'.join(triple[:2])
        if (not key in count) or (count[key] < float(triple[2])):
            count[key] = float(triple[2])
        
    return count

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
    return [v for (v, l) in tlist], [l for (v, l) in tlist]

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
        GaussianNB(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]
    X, y = getPredictorTarget(df, bowVectorizer, dimReducer)
    for model in modelList:
        model.fit(X, y)
    return modelList

def evalModels(df, modelList, bowVectorizer=None, dimReducer=None):
    X, y = getPredictorTarget(df, bowVectorizer, dimReducer)
    for model in modelList:
        y_pred = model.predict(X)
        print(
            '\n*** %s ***\n' % str(model)[:str(model).find('(')],
            'Prec ' + str(precision_score(y, y_pred)),
            'Reca ' + str(recall_score(y, y_pred)),
            'F1 ' + str(f1_score(y, y_pred)),
        )


'''
main
'''
trainSet = loadCorpus('train.csv')
testSet = loadCorpus('test.csv')

vocab = vocabBuild(trainSet['count'])
embed(vocab, trainSet)
embed(vocab, testSet)

modelList = modelTrain(trainSet, dimReducer=None)
evalModels(testSet, modelList, dimReducer=None)

dim_reducer = getDimReducer(trainSet)
modelList = modelTrain(trainSet, bowVectorizer=None, dimReducer=dim_reducer)
evalModels(testSet, modelList, bowVectorizer=None, dimReducer=dim_reducer)

bowVectorizer, dimReducer = getBowVectorizer(trainSet)
modelList = modelTrain(trainSet, bowVectorizer, dimReducer)
evalModels(testSet, modelList, bowVectorizer, dimReducer)
