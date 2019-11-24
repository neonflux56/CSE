

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
import math



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
#Repeat the above experiment using both unigrams and bigrams,

#Define top 500 uniwords
unigramCount = defaultdict(int)
totalunigrams = 0
punct = string.punctuation
for t in tenk_reviews:
  t = t.lower() #remove capitalize
  t = [c for c in t if not (c in punct)]   #remove punctuation
  t = ''.join(t)  # convert back to string
  words = t.strip().split()
  for u in words:
    totalunigrams += 1
    unigramCount[u] += 1
unigram_counts = [(unigramCount[b], b) for b in unigramCount]
unigram_counts.sort(reverse=True)
unigrams = [w[1] for w in unigram_counts[:500]]
unigramId = dict(zip(unigrams, range(len(unigrams))))
unigramSet = set(unigrams)

#Define top 500 biwords
bigrams = [w[1] for w in counts[:500]]
bigramId = dict(zip(bigrams, range(len(bigrams))))
bigramSet = set(bigrams)

#Run regression
def feature(t):
  bigramsoft = []
  unigramsoft = []
  feat = [0] * (len(bigramSet)+len(unigramSet))
  t = t.lower() #remove capitalize
  t = [c for c in t if not (c in punct)]   #remove punctuation
  t = ''.join(t)  # convert back to string
  words = t.strip().split()  # tokenizes

  for w in words:   #Add 500 features for unigrams
    if not (w in unigramSet): continue
    feat[unigramId[w]] += 1

  for i,w in enumerate(words):   #Obtain bigrams list
    if (i<len(words)-1): bigramsoft.append(' '.join(words[i:i+2]))
    else: continue
  for b in bigramsoft:    #Add 500 features for bigrams
    if not (b in bigramSet): continue
    feat[bigramId[b]] += 1

  feat.append(1)   #Add offset
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

# Question 4

req_words = ['stories','magician','psychic','writing','wonder']
N = len(tenk_reviews)

#IDF
idf_req_words1 = []
for w in req_words:
  df = 0
  for t in tenk_reviews:
    t = t.lower()  # remove capitalize
    t = [c for c in t if not (c in punct)]  # remove punctuation
    t = ''.join(t)  # convert back to string
    words = t.strip().split()
    for w1 in words:
      if w==w1 :
        df += 1
        break
      else:continue
    continue
  idf_req_words1.append(math.log10( N / df ))

#TFIDF
def tfidf(term,doc,Doc_corpus):
  df = 0
  tf = 0
  for t1 in Doc_corpus:
    t = t1.lower()  # remove capitalize
    t = [c for c in t if not (c in punct)]  # remove punctuation
    t = ''.join(t)  # convert back to string
    words1 = t.strip().split()
    for w1 in words1:
      if w==w1 :
        df += 1
        break
      else:continue
    continue
  idf = (math.log10( len(Doc_corpus)*1.0 / df ))
  doc = doc.lower()
  doc = [c for c in doc if not (c in punct)]   #remove punctuation
  doc = ''.join(doc)  # convert back to string
  words = doc.strip().split()
  for w1 in words:
    if term==w1: tf+= 1
    else: continue
  return (tf*idf)


tfidf_req_words = []
for i,w in enumerate(req_words):
  tfidf_req_words.append(tfidf(w,tenk_reviews[0],tenk_reviews))

print("Inverse Doc frequency of the words are: " + str(idf_req_words1))
print("Tf-idf of the words with respect to first review are: " + str(tfidf_req_words))

#################################################################################

# Question 5


# Top 1000 unigrams
unigrams = [w[1] for w in unigram_counts[:1000]]
unigramId = dict(zip(unigrams, range(len(unigrams))))
unigramSet = set(unigrams)


#Run regression
def feature(text):
  unigramsoft = []
  feat = [0] * len(unigramSet)
  t = text.lower() #remove capitalize
  t = [c for c in t if not (c in punct)]   #remove punctuation
  t = ''.join(t)  # convert back to string
  words = t.strip().split()
  for w in words:   #Add 1000 tfidf features for unigrams
    if not (w in unigramSet): continue
    feat[unigramId[w]] = tfidf(w,text,tenk_reviews[:500])
  feat.append(1)   #Add offset
  return feat


Y = [d['rating'] for d in fulldata[:500]]
X = [feature(text) for text in tenk_reviews[:500]]
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

# Question 6
