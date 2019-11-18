
import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

from Assignment1Rating_Approach1 import bi

os.chdir('C:\\Users\\asg007\\PycharmProjects\\CSE')

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
userGamma = {}
bookGamma = {}
g=2
for u in booksperuser:
    userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(g)]
for i in usersperbook:
    bookGamma[i] = [random.random() * 0.1 - 0.05 for k in range(g)]



def prediction(user, book):
    if book in bookBiases.keys():
        return alpha + userBiases[user] + bookBiases[book] + inner(userGamma[user], bookGamma[book])
    else:
        return globaltrainAverage

def unpack(theta):
    global alpha
    global userBiases
    global bookBiases
    global bookGamma
    global userGamma
    index = 0
    alpha = theta[index]
    index += 1
    userBiases = dict(zip(users, theta[index:nUsers+index]))
    index += nUsers
    bookBiases = dict(zip(books, theta[index:nBooks+index]))
    index += nBooks
    for u in users:
        userGamma[u] = theta[index:index+g]
        index += g
    for i in books:
        bookGamma[i] = theta[index:index+g]
        index += g

def inner(x, y):
    return sum([a*b for a,b in zip(x,y)])

def MSE(predictions, labels):
    differences = [(x-int(y))**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d[0], d[1]) for d in train]
    cost = MSE(predictions, labels)
    #print("MSE = " + str(cost) + " for lambda: " + str(lamb))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
        for k in range(g):
            cost += lamb*userGamma[u][k]**2
    for i in bookBiases:
        cost += lamb*bookBiases[i]**2
        for k in range(g):
            cost += lamb*bookGamma[i][k]**2
    return cost

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(train)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dbookBiases = defaultdict(float)
    dUserGamma = {}
    dbookGamma = {}
    for u in booksperuser:
        dUserGamma[u] = [0.0 for k in range(g)]
    for i in usersperbook:
        dbookGamma[i] = [0.0 for k in range(g)]
    for d in train:
        u,i = d[0], d[1]
        pred = prediction(u, i)
        diff = pred - int(d[2])
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dbookBiases[i] += 2/N*diff
        for k in range(g):
            dUserGamma[u][k] += 2/N*bookGamma[i][k]*diff
            dbookGamma[i][k] += 2/N*userGamma[u][k]*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
        for k in range(g):
            dUserGamma[u][k] += 2*lamb*userGamma[u][k]
    for i in bookBiases:
        dbookBiases[i] += 2*lamb*bookBiases[i]
        for k in range(g):
            dbookGamma[i][k] += 2*lamb*bookGamma[i][k]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dbookBiases[i] for i in books]
    for u in users:
        dtheta += dUserGamma[u]
    for i in books:
        dtheta += dbookGamma[i]
    return np.array(dtheta)



labels = [d[2] for d in train]


C = [0.0001,0.00009]
#G = [2,3,4]
g = 2
msevalidc = []
for c in C:
    #for g in G:
    alpha = globaltrainAverage
    userBiases = defaultdict(float)
    bookBiases = defaultdict(float)
    userGamma = {}
    bookGamma = {}
    for u in booksperuser:
        userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(g)]
    for i in usersperbook:
        bookGamma[i] = [random.random() * 0.1 - 0.05 for k in range(g)]
    scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nBooks) + [random.random() * 0.1 - 0.05 for k in range(g*(nUsers+nBooks))],derivative, args = (labels, c))
    mse_valid = MSE([prediction(d[0], d[1]) for d in valid], [int(d[2]) for d in valid])
    msevalidc.append(mse_valid)
    print("MSE of validation set = " + str(mse_valid) + " for k value = " + str(g) + " and c value = " + str(c) )
print("Minimum MSE = " + str( min(msevalidc)))
plt.plot([str(v) for v in C],msevalidc)

# for g in G:

g=2
alpha = globaltrainAverage
userBiases = defaultdict(float)
bookBiases = defaultdict(float)
userGamma = {}
bookGamma = {}
for u in booksperuser:
    userGamma[u] = [random.random() * 0.1 - 0.05 for k in range(g)]
for i in usersperbook:
    bookGamma[i] = [random.random() * 0.1 - 0.05 for k in range(g)]
scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0] * (nUsers + nBooks) + [random.random() * 0.1 - 0.05 for k in range(g * (nUsers + nBooks))], derivative,args=(labels, 0.000016))
mse_valid = MSE([prediction(d[0], d[1]) for d in valid], [int(d[2]) for d in valid])
print("MSE of validation set = " + str(mse_valid) + " for k value = " + str(g) + " and c value = " + str(0.000009))




#MSE Train
mse_train = MSE([prediction(d[0], d[1]) for d in train] , labels)
print("MSE for train set : " + str(mse_train))

#MSE of validation set
predvalid = []
for u,b,r in valid:
    predvalid.append(prediction(u, b))
actualvalid = [int(v[2]) for v in valid]
msevalid = MSE(predvalid,actualvalid)
print("MSE for validation set : " + str(msevalid))



predictions = open("predictions_Rating.txt", 'w')
pred = []
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split('-')
    pred.append(prediction(u,b))
    _ = predictions.write(u + '-' + b + ',' + str(prediction(u,b)) + '\n')
print(sum(pred))
predictions.close()

