import gzip
from collections import defaultdict
import random
import numpy as np
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



Y = np.arange(0.1,0.2,0.01)
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
  if count > (totalRead/1.4849999999999999) : break

return2 = list(return2)
pred2 = []
for u,b,_ in new_valid:
  if ((maxsimilarityval(u,b) > 0.0067) and (b in return2)):
    pred2.append(1)
    continue
  else:
    pred2.append(0)


correct2 = []
for i, j in enumerate(actual):
  correct2.append(j == pred2[i])

Accuracy2 = (sum(correct2) / len(correct2))

print("positives :" + str(sum(pred2)) + "Accuracy : "+ str(sum(correct2) / len(correct2)))
print("Using both Max similarity threshold: " + str(0.01) +"\n and Popularity model threshold denom value: " + str(1.3) + "\n gives Accuracy :" + str(Accuracy2))

#####################################################################



X = np.arange(1.4,1.5,0.01)
Updated_Accuracy = []
Y = np.arange(0.0068, 0.0069, 0.00001)

for x in X:
  for y in Y:

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
      if (b in return1) and (maxsimilarityval(u,b) > y):
        pred.append(1)
      else:
        pred.append(0)

    correct = []
    for i,j in enumerate(actual) :
      correct.append( j == pred[i])
    Updated_Accuracy.append(sum(correct)*1.0/len(correct) )

    print("Accuracy : " + str(sum(correct)*1.0/len(correct)) + " x : " + str(x) + " y : " + str(y))

max(Updated_Accuracy)








#####################################################################

# QUESTION 5

#The column name 'prediction' contains the predicted values for each user-book pair.

return2 = list(return2)
pred3 = []
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    with open('Read_Predictions_updated.csv', 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(["userID-bookID", "prediction"])
    continue
  u,b = l.strip().split('-')
  if b not in books:
    pred3.append(1)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
    continue
  if b in Set_booksperuser[u]:
    pred3.append(1)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
    continue
  if ((maxsimilarityval(u,b) > 0.006698) and (b in return2)):
    pred3.append(1)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
    continue
  if b in return2[:int(len(return2)/4)]:
    pred3.append(1)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,1))
    continue
  else:
    pred3.append(0)
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,0))



print(len(pred3))

print(sum(pred3))





#############################################################################










