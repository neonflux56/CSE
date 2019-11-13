import gzip
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
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

usersperbook = defaultdict(list)
booksperuser = defaultdict(list)
for user,book,_ in readCSV("train_Interactions.csv.gz"):
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
new_valid = valid + negvalid
random.shuffle(new_valid)

actual = [v[2] for v in new_valid]

#baseline model on trainbooks to rank popularity, then applying on new_valid
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in train:
  bookCount[book] += 1
  totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalRead/2: break

pred = []
for u,b,_ in new_valid:
  if b in return1:
    pred.append(1)
  else:
    pred.append(0)

correct = []
for i,j in enumerate(actual) :
  correct.append( j == pred[i])

print("Accuracy of baseline model on new validation set: " + str(sum(correct)/len(correct)) )


#####################################################################

# Question2

X = np.arange(1,4.0,0.1)
Accuracy = []

for x in X:
# new baseline model on train books to rank popularity, then applying on new_valid
  bookCount = defaultdict(int)
  totalRead = 0

  for user,book,_ in train:
    bookCount[book] += 1
    totalRead += 1

  mostPopular = [(bookCount[x], x) for x in bookCount]
  mostPopular.sort()
  mostPopular.reverse()

  return1 = set()
  count = 0
  for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > (totalRead/x) : break

  pred = []
  for u,b,_ in new_valid:
    if b in return1:
      pred.append(1)
    else:
      pred.append(0)

  correct = []
  for i,j in enumerate(actual) :
    correct.append( j == pred[i])
  Accuracy.append(sum(correct)/len(correct))

plt.plot(X,Accuracy)
thresholdval = X[Accuracy.index(max(Accuracy))]
print("Baseline Threshold denominator value: " + str(thresholdval) + " gives Accuracy :" + str(max(Accuracy)))


#####################################################################

# Question3


Set_usersperbook = defaultdict(set)
Set_booksperuser = defaultdict(set)
for user,book,_ in train:
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



Y = np.arange(0.01,0.011,0.0001)
Accuracy1 = []
for y in Y:

  pred1 = []
  for u,b,_ in new_valid:
    if maxsimilarityval(u,b) > y:
      pred1.append(1)
    else:
      pred1.append(0)

  correct1 = []
  for i, j in enumerate(actual):
    correct1.append(j == pred1[i])
  Accuracy1.append(sum(correct1) / len(correct1))

plt.plot(Y, Accuracy1)
maxsimthreshold = Y[Accuracy1.index(max(Accuracy1))]

print("Max similarity threshold: " + str(maxsimthreshold) + " gives Accuracy :" + str(max(Accuracy1)))


#####################################################################

# Question4

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in train:
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
  if count > (totalRead/thresholdval) : break

pred2 = []
for u,b,_ in new_valid:
  if ((maxsimilarityval(u,b) > maxsimthreshold) and (b in return2)):
    pred2.append(1)
  else:
    pred2.append(0)


correct2 = []
for i, j in enumerate(actual):
  correct2.append(j == pred2[i])

Accuracy2 = (sum(correct2) / len(correct2))

print("positives :" + str(sum(pred2)))
print("Using both Max similarity threshold: " + str(maxsimthreshold) +"\n and Popularity model threshold denom value: " + str(thresholdval) + "\n gives Accuracy :" + str(Accuracy2))


#####################################################################

# QUESTION 5

#The column name 'prediction' contains the predicted values for each user-book pair.

pred3 = []
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    with open('Read_Predictions_updated.csv', 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(["userID-bookID", "prediction"])
    continue
  u,b = l.strip().split('-')
  if b in Set_booksperuser[u]:
    pred3.append(1)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
    continue
  if ((maxsimilarityval(u,b) > maxsimthreshold) and (b in return2)):
    pred3.append(1)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
  else:
    pred3.append(0)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,0))

print(sum(pred3))





#############################################################################










