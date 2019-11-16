
import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import random
import csv
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

# Setup train and validation
fulldata = []
for l in readCSV("train_Interactions.csv.gz"):
  fulldata.append(l)
valid = fulldata[190000:]
train = fulldata[:190000]


allRatings = []
usersperbook = defaultdict(list)
booksperuser = defaultdict(list)
for user,book,r in train:
  r = int(r)
  allRatings.append(r)
  usersperbook[book].append(user)
  booksperuser[user].append(book)


N = len(train)
nUsers = len(booksperuser)
nBooks = len(usersperbook)
users = list(booksperuser.keys())
books = list(usersperbook.keys())

globaltrainAverage = sum(allRatings) / len(allRatings)

alpha = globaltrainAverage
userBiases = defaultdict(float)
bookBiases = defaultdict(float)


def prediction(user, book):
    if book in bookBiases.keys():
        return alpha + userBiases[user] + bookBiases[book]
    else:
        return globaltrainAverage

def unpack(theta):
    global alpha
    global userBiases
    global bookBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    bookBiases = dict(zip(books, theta[1+nUsers:]))


def MSE(predictions, labels):
    differences = [(x-int(y))**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d[0], d[1]) for d in train]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost) + " for lambda: " + str(lamb))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in bookBiases:
        cost += lamb*bookBiases[i]**2
    return cost

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(train)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dbookBiases = defaultdict(float)
    for d in train:
        u,i = d[0], d[1]
        pred = prediction(u, i)
        diff = pred - int(d[2])
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dbookBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in bookBiases:
        dbookBiases[i] += 2*lamb*bookBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dbookBiases[i] for i in books]
    return np.array(dtheta)

labels = [d[2] for d in train]

#Gradient descent
scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nBooks),derivative, args = (labels, 1))

#MSE Train
mse_train = MSE([prediction(d[0], d[1]) for d in train] , labels)

#MSE of validation set
predvalid = []
for u,b,r in valid:
    predvalid.append(prediction(u, b))

actualvalid = [int(v[2]) for v in valid]
msevalid = MSE(predvalid,actualvalid)
print("MSE for validation set : " + str(msevalid))


###############################################################


#Question 10
#Lowest and highest biases

def get_key(val,dict):
    for x,y in dict.items():
        if y == val:
            return x

print("Book with the maximum bias value : " + get_key(max(bookBiases.values()),bookBiases))
print("Book with the minimum bias value : " + get_key(min(bookBiases.values()),bookBiases))
print("User with the maximum bias value : " + get_key(max(userBiases.values()),userBiases))
print("User with the minimum bias value : " + get_key(min(userBiases.values()),userBiases))

###############################################################

#Question 11

C = [0.000005,0.000007,0.000009,0.00001,0.000012,0.000014,0.000016]
msevalidc = []
for c in C:
    alpha = globaltrainAverage
    userBiases = defaultdict(float)
    bookBiases = defaultdict(float)
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nBooks),derivative, args = (labels, c))
    msevalidc.append(MSE([prediction(d[0], d[1]) for d in valid], [int(d[2]) for d in valid]))

plt.plot([str(v) for v in C],msevalidc)
# The best is C = 0.00001 with lowest MSE on validation


scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nBooks),derivative, args = (labels, C[msevalidc.index(min(msevalidc))]))
print("MSE for validation set: " + str(MSE(predvalid,actualvalid)))

# submit it to file
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    with open('Rating_Predictions_updated.csv', 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(["userID-bookID", "prediction"])
    continue
  u,b = l.strip().split('-')
  with open('Rating_Predictions_updated.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow((u+"-"+b,prediction(u,b)))



###############################################################


