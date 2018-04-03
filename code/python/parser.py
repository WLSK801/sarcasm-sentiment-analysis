import re
import random
import matplotlib.pyplot as plt
import collections
import numpy as np
import os
import nltk
import json
import csv
import pandas as pd

# data path
dirr = "<data>"
# output path
O_dirr = "<data>"

def load_sarc_responses(train_file, test_file, comment_file, lower=True):
  '''loads SARC data from csv files
  Args:
    train_file: csv file with train sequences
    test_file: csv file with train sequences
    comment_file: json file with details about all comments
    lower: boolean; if True, converts comments to lowercase
  Returns:
    train_sequences, train_labels, test_sequences, test_labels
    train_sequences: {'ancestors': list of ancestors for all sequences,
                      'responses': list of responses for all sequences}
    train_labels: list of labels for responses for all sequences.
  '''

  with open(comment_file, 'r') as f:
    comments = json.load(f)

  train_docs = {'ancestors': [], 'responses': []}
  train_labels = []
  with open(train_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
      ancestors = row[0].split(' ')
      responses = row[1].split(' ')
      labels = row[2].split(' ')
      if lower:
        train_docs['ancestors'].append([comments[r]['text'].lower() for r in ancestors])
        train_docs['responses'].append([comments[r]['text'].lower() for r in responses])
      else:
        train_docs['ancestors'].append([comments[r]['text'] for r in ancestors])
        train_docs['responses'].append([comments[r]['text'] for r in responses])
      train_labels.append(labels)

  test_docs = {'ancestors': [], 'responses': []}
  test_labels = []
  with open(test_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
      ancestors = row[0].split(' ')
      responses = row[1].split(' ')
      labels = row[2].split(' ')
      if lower:
        test_docs['ancestors'].append([comments[r]['text'].lower() for r in ancestors])
        test_docs['responses'].append([comments[r]['text'].lower() for r in responses])
      else:
        test_docs['ancestors'].append([comments[r]['text'] for r in ancestors])
        test_docs['responses'].append([comments[r]['text'] for r in responses])
      test_labels.append(labels)

  return train_docs, test_docs, train_labels, test_labels
comments_fn = "comments.json"
train_fn = "train-balanced.csv"
test_fn = "test-balanced.csv"
train_seqs, test_seqs, train_labels, test_labels =\
load_sarc_responses(os.path.join(dirr, train_fn), os.path.join(dirr, test_fn), os.path.join(dirr, comments_fn))
# Only use responses for this method. Ignore ancestors.
train_resp = train_seqs['responses']
test_resp = test_seqs['responses']

# Split into first and second responses and their labels.
# {0: list_of_first_responses, 1: list_of_second_responses}
train_docs = {i: [l[i] for l in train_resp] for i in range(2)}
test_docs = {i: [l[i] for l in test_resp] for i in range(2)}
train_labels = {i: [2*int(l[i])-1 for l in train_labels] for i in range(2)}
test_labels = {i: [2*int(l[i])-1 for l in test_labels] for i in range(2)}

train_text = train_docs[0] + train_docs[1]
train_lab = train_labels[0] + train_labels[1]
test_text = test_docs[0] + test_docs[1]
test_lab = test_labels[0] + test_labels[1]

train_df = [("data", train_text), ("label", train_lab)]
test_df = [("data", test_text), ("label", test_lab)]
train_df = pd.DataFrame.from_items(train_df)
test_df = pd.DataFrame.from_items(test_df)
train_df.to_csv(O_dirr + "train.csv", encoding='utf-8', index=False)
test_df.to_csv(O_dirr + "test.csv", encoding='utf-8', index=False)