

import gzip
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
#import os
#os.chdir('C:\\Users\\asg007\\PycharmProjects\\CSE')

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


def maxsimilarityval1(u,b) :
  users = Set_usersperbook[b]
  similarities = []
  for b1 in Set_booksperuser[u]:
    if b==b1: continue
    sim = Jaccard(users,Set_usersperbook[b1])
    similarities.append(sim)
    similarities.sort(reverse=True)
  return similarities[0]


def minsimilarityval(u,b) :
  users = Set_usersperbook[b]
  similarities = []
  for b1 in Set_booksperuser[u]:
    if b==b1: continue
    sim = Jaccard(users,Set_usersperbook[b1])
    similarities.append(sim)
    similarities.sort(reverse=True)
  return sum([i==0 for i in similarities])


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

def ifcond(u,b) :
  if b not in books:
    return 1
  if b in Set_booksperuser[u]:
    return 1
  if b in return2[:int(len(return2)/2)]:
    return 1
  if (maxsimilarityval(u,b) > 0.0067) and (b in return2) and (maxsimilarityval1(u,b) > 0.00685) and (minsimilarityval(u,b)<4):
    return 1
  else:
    return 0

for u,b,_ in new_valid:
    pred2.append(ifcond(u,b))



correct2 = []
for i, j in enumerate(actual):
  correct2.append(j == pred2[i])

Accuracy2 = (sum(correct2) / len(correct2))


print("Using both Max similarity threshold: " + str(0.01) +"\n and Popularity model threshold denom value: " + str(1.3) + "\n gives Accuracy :" + str(Accuracy2))

TP_ = np.logical_and(pred2, actual)
FP_ = np.logical_and(pred2, np.logical_not(actual))
TN_ = np.logical_and(np.logical_not(pred2), np.logical_not(actual))
FN_ = np.logical_and(np.logical_not(pred2), actual)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
Accuracy = (TP + TN) * 1.0 / (TP + FP + TN + FN)
Precision = TP * 1.0 / (TP + FP)
Recall = TP * 1.0 / (TP + FN)
print("Positives:" + str(sum(pred2)) + " Accuracy:" + str(Accuracy) + " Precision:" + str(Precision) + " Recall:" + str(
  Recall) + " FP:" + str(FP) + " FN:" + str(FN))

#####################################################################

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


def maxsimilarityval1(u,b) :
  users = Set_usersperbook[b]
  similarities = []
  for b1 in Set_booksperuser[u]:
    if b==b1: continue
    sim = Jaccard(users,Set_usersperbook[b1])
    similarities.append(sim)
    similarities.sort(reverse=True)
  return similarities[0]



print(sum(maxsimilarityval(u,b) > 0.02 for u,b,_ in new_valid))

def minsimilarityval(u,b) :
  users = Set_usersperbook[b]
  similarities = []
  for b1 in Set_booksperuser[u]:
    if b==b1: continue
    sim = Jaccard(users,Set_usersperbook[b1])
    similarities.append(sim)
    similarities.sort(reverse=True)
  return sum([i==0.0 for i in similarities])


Updated_Accuracy = []
A = np.arange(0.0075,0.0076, 0.0001)
X = np.arange(2.815,2.82,0.001)
#C = np.arange(0.0072, 0.0075, 0.0001)
#D = np.arange(7,8,1)

for a in A:
  for x in X:

    bookCount = defaultdict(int)
    totalRead = 0

    for user, book, _ in train:
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
      if count > (totalRead / x): break
    return2 = list(return2)


    def ifcond(u, b):
      if b not in books:
        return 0
      if b in Set_booksperuser[u]:
        return 1
      if minsimilarityval(u,b) > 80:
        return 0
      if maxsimilarityval(u, b) > a and (b in return2):
        return 1
      else:
        return maxsimilarityval(u,b)


    userbooklist = defaultdict(list)
    usersimilaritylist = defaultdict(list)
    utest = defaultdict(list)
    for u, b, _ in new_valid:
      userbooklist[u].append(b)
      usersimilaritylist[u].append(ifcond(u, b))
      utest[u] = list(zip(usersimilaritylist[u], userbooklist[u]))
      utest[u].sort()


    def prediction(u, b):
      for i, j in enumerate(utest[u]):
        if j[1] != b:
          continue
        else:
          if i < int(len(utest[u]) / 2):
            return 0
          else:
            return 1


    pred2 = []
    for u, b, _ in new_valid:
      pred2.append(prediction(u, b))

    correct = []
    for i, j in enumerate(actual):
      correct.append(j == pred2[i])

    TP_ = np.logical_and(pred2, actual)
    FP_ = np.logical_and(pred2, np.logical_not(actual))
    TN_ = np.logical_and(np.logical_not(pred2), np.logical_not(actual))
    FN_ = np.logical_and(np.logical_not(pred2), actual)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Accuracy = (TP + TN) * 1.0 / (TP + FP + TN + FN)
    Precision = TP * 1.0 / (TP + FP)
    Recall = TP * 1.0 / (TP + FN)
    Updated_Accuracy.append(Accuracy)
    print("Positives:" + str(sum(pred2)) + " Accuracy:" + str(Accuracy) + " Precision:" + str(
      Precision) + " Recall:" + str(Recall) + " FP:" + str(FP) + " FN:" + str(
      FN) + " A:" + str(a) + " X:" + str(x)  )#+ " D:" + str(d))

print(max(Updated_Accuracy))


#####################################################################


############# validation

bookCount = defaultdict(int)
totalRead = 0

for user, book, _ in train:
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
  if count > (totalRead /2.816): break
return2 = list(return2)


def ifcond(u, b):
  if b not in books:
    return 0
  if b in Set_booksperuser[u]:
    return 1
  if minsimilarityval(u, b) > 80:
    return 0
  if maxsimilarityval(u, b) > a and (b in return2):
    return 1
  else:
    return maxsimilarityval(u, b)



userbooklist = defaultdict(list)
usersimilaritylist = defaultdict(list)
utest = defaultdict(list)
for u,b,_ in new_valid:
  userbooklist[u].append(b)
  usersimilaritylist[u].append(ifcond(u,b))
  utest[u] = list(zip(usersimilaritylist[u],userbooklist[u]))
  utest[u].sort()

def prediction(u,b):
  for i,j in enumerate(utest[u]):
    if j[1] != b:
      continue
    else:
      if i<int(len(utest[u])/2):
        return 0
      else:
        return 1

pred2=[]
for u,b,_ in new_valid:
  pred2.append(prediction(u,b))



correct = []
for i, j in enumerate(actual):
  correct.append(j == pred2[i])

TP_ = np.logical_and(pred2, actual)
FP_ = np.logical_and(pred2, np.logical_not(actual))
TN_ = np.logical_and(np.logical_not(pred2), np.logical_not(actual))
FN_ = np.logical_and(np.logical_not(pred2), actual)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
Accuracy = (TP + TN) * 1.0 / (TP + FP + TN + FN)
Precision = TP * 1.0 / (TP + FP)
Recall = TP * 1.0 / (TP + FN)
print("Positives:" + str(sum(pred2)) + " Accuracy:" + str(Accuracy) + " Precision:" + str(
          Precision) + " Recall:" + str(Recall) + " FP:" + str(FP) + " FN:" + str(FN) )#+ " A:" + str(a) + " X:" + str(x)  )#+ " D:" + str(d))


####TEST

userbooklist = defaultdict(list)
usersimilaritylist = defaultdict(list)
utest = defaultdict(list)
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    #header
    continue
  u,b = l.strip().split('-')
  userbooklist[u].append(b)
  usersimilaritylist[u].append(ifcond(u,b))
  utest[u] = list(zip(usersimilaritylist[u],userbooklist[u]))
  utest[u].sort()

predictions = open("predictions_Read.txt", 'w')
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,b = l.strip().split('-')
  predictions.write(u + '-' + b + ',' + str(prediction(u, b)) + '\n')

predictions.close()

#####################################










userlist = defaultdict(list)
usercount = defaultdict(int)
pred2 = []
for u, b, _ in new_valid:
  usercount[u] =+ 1
  if u not in userlist.keys():
    pred2.append(ifcond(u, b))
    userlist[u] = (ifcond(u,b))
    continue
  if u in userlist.keys() and usercount[u]%2 != 0:
    pred2.append(ifcond(u, b))
    userlist[u] = (ifcond(u,b))
  else:
    pred2.append(int(not(bool(userlist[u]))))




correct = []
for i, j in enumerate(actual):
  correct.append(j == pred2[i])

len(correct)
sum(correct)

TP_ = np.logical_and(pred2, actual)
FP_ = np.logical_and(pred2, np.logical_not(actual))
TN_ = np.logical_and(np.logical_not(pred2), np.logical_not(actual))
FN_ = np.logical_and(np.logical_not(pred2), actual)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
Accuracy = (TP + TN) * 1.0 / (TP + FP + TN + FN)
Precision = TP * 1.0 / (TP + FP)
Recall = TP * 1.0 / (TP + FN)
print("Positives:" + str(sum(pred2)) + " Accuracy:" + str(Accuracy) + " Precision:" + str(
          Precision) + " Recall:" + str(Recall) + " FP:" + str(FP) + " FN:" + str(FN) )#+ " A:" + str(a) + " X:" + str(x)  )#+ " D:" + str(d))




usercount1 = defaultdict(int)
pred3 = []
userlist1 = defaultdict(list)
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    with open('Read_Predictions_updated.csv', 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(["userID-bookID", "prediction"])
    continue
  u,b = l.strip().split('-')
  usercount1[u] =+ 1
  if u not in userlist1.keys():
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u + "-" + b, ifcond(u,b)))
    userlist1[u] = (ifcond(u,b))
    pred3.append(ifcond(u, b))
    continue
  if u in userlist1.keys() and usercount1[u]%2 !=0 :
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u + "-" + b, ifcond(u, b)))
    userlist1[u] = (ifcond(u, b))
    pred3.append(ifcond(u, b))
  else:
    with open('Read_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u + "-" + b, int(not(bool(userlist1[u])))))
    pred3.append(int(not(bool(userlist1[u]))))

print(len(pred3))
print(sum(pred3))





#####################################################################

# QUESTION 5

#The column name 'prediction' contains the predicted values for each user-book pair.

bookCount = defaultdict(int)
totalRead = 0

for user, book, _ in train:
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
  if count > (totalRead / 1.585): break
return2 = list(return2)

def ifcond(u, b):
  if b not in books:
    return 0
  if b in Set_booksperuser[u]:
    return 1
  if b in return2[:int(len(return2) / 10)]:
    return 1
  if (maxsimilarityval(u, b) > 0.02):  # and (b in return1):# and (minsimilarityval(u, b) < d):
    return 1
  if (maxsimilarityval(u, b) > 0.007) and (b in return2):  # and (minsimilarityval(u, b) < d):
    return 1
  else:
    return 0


for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u, b = l.strip().split('-')





#############################################################################










