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

import os
os.chdir('C:\\Users\\asg007\\PycharmProjects\\CSE')


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

trainRatings = [r[2] for r in ratingsTrain]
globalAverage = sum(trainRatings) * 1.0 / len(trainRatings)

betaU = {}
betaI = {}
for u in ratingsPerUser:
  betaU[u] = 0

for b in ratingsPerItem:
  betaI[b] = 0

alpha = globalAverage

def iterate(lamb):
  newAlpha = 0
  for u,b,r in ratingsTrain:
    newAlpha += r - (betaU[u] + betaI[b])
  alpha = newAlpha / len(ratingsTrain)
  for u in ratingsPerUser:
    newBetaU = 0
    for b,r in ratingsPerUser[u]:
      newBetaU += r - (alpha + betaI[b])
    betaU[u] = newBetaU / (lamb + len(ratingsPerUser[u]))
  for b in ratingsPerItem:
    newBetaI = 0
    for u,r in ratingsPerItem[b]:
      newBetaI += r - (alpha + betaU[u])
    betaI[b] = newBetaI / (lamb + len(ratingsPerItem[b]))
  mse = 0
  for u,b,r in ratingsTrain:
    prediction = alpha + betaU[u] + betaI[b]
    mse += (r - prediction)**2
  regularizer = 0
  for u in betaU:
    regularizer += betaU[u]**2
  for b in betaI:
    regularizer += betaI[b]**2
  mse /= len(ratingsTrain)
  return mse, mse + lamb*regularizer


#Better lambda

L = np.arange(3,4,0.1)
validMSElist = []
iterations = 1
for l in L:
    newMSE, newObjective = iterate(l)
    while iterations < 10 or objective - newObjective > 0.0001:
      mse, objective = newMSE, newObjective
      newMSE, newObjective = iterate(l)
      iterations += 1
      #print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
      #print("MSE after " + str(iterations) + " iterations = " + str(newMSE))
    validMSE = 0
    for u,b,r in ratingsValid:
      bu = 0
      bi = 0
      if u in betaU:
        bu = betaU[u]
      if b in betaI:
        bi = betaI[b]
      prediction = alpha + bu + bi
      validMSE += (r - prediction)**2
    validMSE /= len(ratingsValid)
    validMSElist.append(validMSE)
    print("Validation MSE = " + str(validMSE))
plt.plot(L,validMSElist)

#Using better lambda

newMSE, newObjective = iterate(3)
while iterations < 10 or objective - newObjective > 0.0000001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(3)
    iterations += 1
    # print("Objective after " + str(iterations) + " iterations = " + str(newObjective))
    # print("MSE after " + str(iterations) + " iterations = " + str(newMSE))
validMSE = 0
for u, b, r in ratingsValid:
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    prediction = alpha + bu + bi
    validMSE += (r - prediction) ** 2
validMSE /= len(ratingsValid)
validMSElist.append(validMSE)
print("Validation MSE = " + str(validMSE))

predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split('-')
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if b in betaI:
        bi = betaI[b]
    _ = predictions.write(u + '-' + b + ',' + str(alpha + bu + bi) + '\n')

predictions.close()