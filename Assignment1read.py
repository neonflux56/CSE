import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import matplotlib.pyplot as plt
import csv

import os
os.chdir('C:\\Users\\asg007\\PycharmProjects\\CSE')


def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path,'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')


# Setup train and validation and discard the rating column and add read/not read value 1 or 0
fulldata = []
for l in readCSV("train_Interactions.csv.gz"):
  fulldata.append(l)
valid = fulldata[190000:]
train = fulldata[:190000]


def getreq(val):
  n = []
  n.append(val[0])
  n.append(val[1])
  n.append(1)
  return n
valid = [getreq(v) for v in valid]
train = [getreq(v) for v in train]


#Generate new validation set with negative labels
books = []
usersperbook = defaultdict(list)
booksperuser = defaultdict(list)
for user,book,_ in readCSV("train_Interactions.csv.gz"):
    books.append(book)
    usersperbook[book].append(user)
    booksperuser[user].append(book)

def negbook(val):
  neg = []
  while len(neg)== 0:
    b = random.choice(list(usersperbook.keys()))
    if b not in booksperuser[val]:
      neg.append(val)
      neg.append(b)
      neg.append(0)
  return neg

negvalid = [negbook(v[0]) for v in valid]
negtrain = [negbook(v[0]) for v in train]

len(negvalid)
len(negtrain)

new_train = train + negtrain
new_valid = valid + negvalid

len(new_valid)

random.shuffle(new_train)
random.shuffle(new_valid)



# Jaccard Similarity
Set_usersperbook = defaultdict(set)
Set_booksperuser = defaultdict(set)
for user,book,_ in new_train:
  Set_usersperbook[book].add(user)
  Set_booksperuser[user].add(book)

def Jaccard(s1,s2) :
  numer = len(s1.intersection(s2))
  denom = len(s1.union(s2))
  return numer/denom

def maxsimilarityval(u,b) :
  users = Set_usersperbook[b]
  similarities = []
  for b1 in Set_booksperuser[u]:
    if b==b1: continue
    sim = Jaccard(users,Set_usersperbook[b1])
    similarities.append(sim)
    similarities.sort(reverse=True)
  return sum(similarities[:2])/2

def minsimilarityval(u,b) :
  users = Set_usersperbook[b]
  similarities = []
  for b1 in Set_booksperuser[u]:
    if b==b1: continue
    sim = Jaccard(users,Set_usersperbook[b1])
    similarities.append(sim)
    similarities.sort(reverse=False)
  return (similarities[0])


# Popularity
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in new_train:
  bookCount[book] += 1
  totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return2 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return2.add(i)
  if count > (totalRead/1.4849999999999999) : break

return2 = list(return2)


#Generate Feature vector

ytrain = [v[2] for v in new_train]
yvalid = [v[2] for v in new_valid]

def feature1(u,b):
    feat = [1]
    feat.append(maxsimilarityval(u,b))
    feat.append(minsimilarityval(u, b))
    if (b in return2):
        feat.append(1)
    else:
        feat.append(0)
    return feat

Xtrain = [feature1(u,b) for u,b,_ in new_train]
Xvalid = [feature1(u,b) for u,b,_ in new_valid]


Xtest = []
for i,l in enumerate(open("pairs_Read.txt")):
    if l.startswith("userID"):
        continue
    u, b = l.strip().split('-')
    Xtest.append(feature1(u, b))


# Logistic

validAccuracy = []
C = np.arange(1,25,1)
for c in C:
    weights = []
    for i in ytrain:
        if i == False:
            weights.append(1)
        else:
            weights.append(1)
    mod = linear_model.LogisticRegression(C=c)
    mod.fit(Xtrain, ytrain, sample_weight=weights)


    # On Validation

    pred = mod.predict(Xvalid)
    TP_ = np.logical_and(pred, yvalid)
    FP_ = np.logical_and(pred, np.logical_not(yvalid))
    TN_ = np.logical_and(np.logical_not(pred), np.logical_not(yvalid))
    FN_ = np.logical_and(np.logical_not(pred), yvalid)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Accuracy = (TP + TN)*1.0 / (TP + FP + TN + FN)
    Precision = TP *1.0 / (TP + FP)
    Recall = TP*1.0 / (TP + FN)
    validAccuracy.append(Accuracy)
    print("Positives:" + str(sum(pred)) + " Accuracy:" + str(Accuracy) + " Precision:" + str(Precision) + " Recall:" + str(Recall) + " FP:" + str(FP) + " FN:" + str(FN))

plt.plot([str(c) for c in C],validAccuracy)


#select best model

mod = linear_model.LogisticRegression(C=15)
mod.fit(Xtrain, ytrain, sample_weight=weights)

# On Test and write

predtest = mod.predict(Xtest)
for i,l in enumerate(open("pairs_Read.txt")):
  if l.startswith("userID"):
    with open('Read_Predictions_updated.csv', 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(["userID-bookID", "prediction"])
    continue
  u,b = l.strip().split('-')
  if b not in books:
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
    continue
  with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u + "-" + b, predtest[i-1]))


print(len(predtest))

print(sum(predtest))