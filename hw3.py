import gzip
from collections import defaultdict
import random

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)

for user,book,r in readCSV("train_Interactions.csv.gz"):
  r = int(r)
  allRatings.append(r)
  userRatings[user].append(r)

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,b = l.strip().split('-')
  if u in userAverage:
    predictions.write(u + '-' + b + ',' + str(userAverage[u]) + '\n')
  else:
    predictions.write(u + '-' + b + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
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

predictions = open("predictions_Read.txt", 'w')
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,b = l.strip().split('-')
  if b in return1:
    predictions.write(u + '-' + b + ",1\n")
  else:
    predictions.write(u + '-' + b + ",0\n")

predictions.close()


# Setup train and validation and discard the rating column
train = []
for l in readCSV("train_Interactions.csv.gz"):
  train.append(l)
valid = train[190000:]
train = train[:190000]
def getreq(val):
  n = []
  n.append(val[0])
  n.append(val[1])
  return n
valid = [getreq(v) for v in valid]
train = [getreq(v) for v in train]


#Generate new validation set with negative labels

books = []
ubdict = defaultdict(list)
for user,book,_ in readCSV("train_Interactions.csv.gz"):
  books.append(book)
  ubdict[user].append(book)

def negbook(val):
  neg = []
  while len(neg)== 0:
    b = random.choice(books)
    if b not in ubdict[val]:
      neg.append(val)
      neg.append(b)
  return neg
negvalid = [negbook(v[0]) for v in valid]
new_valid = valid + negvalid
random.shuffle(new_valid)






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














