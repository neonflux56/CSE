import gzip
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

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

userperbook = defaultdict(list)
bookperuser = defaultdict(list)
for user,book,_ in readCSV("train_Interactions.csv.gz"):
  userperbook[book].append(user)
  bookperuser[user].append(book)

def negbook(val):
  neg = []
  while len(neg)== 0:
    b = random.choice(list(userperbook.keys()))
    if b not in bookperuser[val]:
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

print("Accuracy : " + str(sum(correct)/len(correct)) )


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
print("Accuracy : " + str(max(Accuracy)))


#####################################################################

# Question3







#Create sets

#usersPerBook = defaultdict(set)
#booksPerUser = defaultdict(set)
#
#for t in train:
#  u,i = t[0].t[1]
#  usersPerBook[i].add(u)
#  booksPerUser[u].add(i)
#
#def Jaccard(s1,s2):
#  num = len(s1.intersection(s2))
#  denom = len(s1.union(s2))
#  return num/denom














