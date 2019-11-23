

import gzip
import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict # Dictionaries with default values
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import ast

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

fulldata = []
for l in readGz("train_Category.json.gz"):
  fulldata.append(l)

tenk_reviews = [d['review_text'] for d in fulldata[:10000]]

ngramCount = defaultdict(int)
totalngrams = 0
ngram = 2
punct = string.punctuation
ngrams = []
for t in tenk_reviews:
  t = t.lower() #remove capitalize
  t = [c for c in t if not (c in punct)]   #remove punctuation
  t = ''.join(t)  # convert back to string
  words = t.strip().split()
  for i,w in enumerate(words):   #Obtain bigrams list
    if (i<len(words)-(ngram-1)):
      ngrams.append(' '.join(words[i:i+(ngram)]))
    else:
      continue

for b in ngrams:  #calculate bigrams count and total bigrams
  totalngrams += 1
  ngramCount[b] += 1

#################################################################################

# Question 1

print("Total number of unique bigrams: "+ str(len(ngramCount)))

counts = [(ngramCount[b], b) for b in ngramCount]
counts.sort(reverse=True)

print("Top 5 occuring bigrams with their counts in the corpus are: ")
print(counts[:5])

#################################################################################

# Question 2

bigrams = [w[1] for w in counts[:1000]]
bigrams[:10]
bigramId = dict(zip(bigrams, range(len(bigrams))))
bigramSet = set(bigrams)

def feature(t):
  ngrams = []
  feat = [0] * len(bigramSet)
  t = t.lower() #remove capitalize
  t = [c for c in t if not (c in punct)]   #remove punctuation
  t = ''.join(t)  # convert back to string
  words = t.strip().split()  # tokenizes
  for i,w in enumerate(words):   #Obtain bigrams list
    if (i<len(words)-1):
      ngrams.append(' '.join(words[i:i+2]))
    else:
      continue
  for b in ngrams:
    if not (b in bigramSet): continue
    feat[bigramId[b]] += 1
  feat.append(1)
  return feat

Y = [d['rating'] for d in fulldata[:10000]]
X = [feature(t) for t in tenk_reviews]
clf = linear_model.Ridge(1.0, fit_intercept=False) # MSE + 1.0 l2
clf.fit(X, Y)
theta = clf.coef_
predictions = clf.predict(X)

#weights = list(zip(theta, bigrams + ['constant_feat']))
#weights.sort()

def MSE(predictions, labels):
  differences = [(x - int(y)) ** 2 for x, y in zip(predictions, labels)]
  return sum(differences) / len(differences)

print("MSE of the predictor: " + str(MSE(predictions,Y)))


#################################################################################

# Question 3

