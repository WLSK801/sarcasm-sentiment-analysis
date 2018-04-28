__author__      = 'Yu Cao'

import pandas as pd
import numpy as np
from scipy import stats
import collections
from string import punctuation as Punct


def loadCorpus(corpus_path):
    df = pd.read_csv(corpus_path)
    df = df.fillna('')
    df['count'] = pd.Series(
        [ rawToCount(x) for x in df['pairs_diff'] ], 
        index = df.index
    )
    return df

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

def compareMeanContrast(df, key):
    summary = collections.defaultdict(list)
    for c, cnt in enumerate(df['count']):
        if cnt is not None and key in cnt:
            summary[df['label'][c]].append(cnt[key])
    return np.mean(summary[1]), np.mean(summary[-1]), stats.ttest_ind(summary[1],summary[-1])
            
def keyExam(innerVocab, cmp_f):
    cnt0, cnt1 = 0, 0
    for key in innerVocab.keys():
        pos, neg, ttest = innerVocab[key]
        if cmp_f(pos, neg):
            cnt0 += 1
            if ttest[1] < 0.05:
                cnt1 += 1
                print(key, '{:.4f} {:.4f} p = {:.4f}'.format(pos, neg, ttest[1]))
    return cnt0, cnt1
'''
main
'''
trainSet = loadCorpus('train.csv')
vocab = vocabBuild(trainSet['count'])

innerVocab = {}
for key in vocab:
    innerVocab[key] = compareMeanContrast(trainSet, key)
    
print('For the 5 most frequent syntactic category pairs, compare the mean sentiment contrasts of sarcastic and non-sarcastic examples.\n')
for key in vocab[:5]:
    pos, neg, ttest = innerVocab[key]
    print(key, '{:.4f} {:.4f} p = {:.4f}'.format(pos, neg, ttest[1]))

print('\nShow syntactic category pairs for which the mean contrast of sarcastic examples is (significantly) greater than that of non-sarcastic examples.\n')
print(keyExam(innerVocab, lambda x, y: x > y))

print('\nShow syntactic category pairs for which the mean contrast of non-sarcastic examples is (significantly) greater than that of sarcastic examples.\n')
print(keyExam(innerVocab, lambda x, y: x <= y))
