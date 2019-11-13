
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



# Setup train and validation
fulldata = []
for l in readCSV("train_Interactions.csv.gz"):
  fulldata.append(l)
valid = fulldata[190000:]
train = fulldata[:190000]


#Get all data

allRatings = []
usersperbook = defaultdict(list)
booksperuser = defaultdict(list)
userRatings = defaultdict(list)
bookRatings = defaultdict(list)
for user,book,r in train:
  r = int(r)
  allRatings.append(r)
  userRatings[user].append(r)
  bookRatings[book].append(r)
  usersperbook[book].append(user)
  booksperuser[user].append(book)

globaltrainAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
  userAverage[u] = sum(userRatings[u]) / len(userRatings[u])



# Using gradient descent to rate predictions

lamda = 1
alpha_ = globaltrainAverage
beta_u_val = [((sum(v)/len(v)) ) for v in userRatings.values()]
beta_i_val = [((sum(v)/len(v)) ) for v in bookRatings.values()]
beta_u = dict(zip(booksperuser.keys(),beta_u_val))
beta_i = dict(zip(usersperbook.keys(),beta_i_val))


def MSE(x,y):
    return sum( [(j - y[i])**2 for i,j in enumerate(x) ]) / len(x)

msetrain = []
while(True) :
    predtrain = []
    for u, b, r in train[:10000]:
        old_alpha_ = alpha_
        old_beta_u = beta_u[u]
        old_beta_i = beta_i[b]
        alpha_ = sum([i - (old_beta_u + old_beta_i) for i in allRatings[:10000]]) / len(train)
        beta_u[u] = sum([i - (old_alpha_ + old_beta_i) for i in userRatings[u]]) / (lamda + len(userRatings[u]))
        beta_i[b] = sum([i - (old_alpha_ + old_beta_u) for i in bookRatings[b]]) / (lamda + len(bookRatings[b]))
        predtrain.append(alpha_ + beta_u[u] + beta_i[b])
    msetrain.append(MSE(predtrain, allRatings[:10000]))
    if( msetrain[-1] < 4 ) : break

#Use model for validation set prediction

predvalid = []
for u,b,r in valid:
    if b == 'b21479253':
        predvalid.append(globaltrainAverage)
    else:
        predvalid.append(alpha_ + beta_u[u] + beta_i[b])

actualvalid = [int(v[2]) for v in valid]
msevalid = MSE(predvalid,actualvalid)
print("MSE for validation set : " + str(msevalid))

#################################################################

#Question 10
#Lowest and highest biases

def get_key(val,dict):
    for x,y in dict.items():
        if y == val:
            print(x)


print(get_key(max(beta_i.values()),beta_i))

print(get_key(min(beta_i.values()),beta_i))

print(get_key(max(beta_u.values()),beta_u))

print(get_key(min(beta_i.values()),beta_u))





#################################################################

#Question 11

# Tune lamda

C = [0.01,0.1,1,10,100]
msevalidc = []
for c in C:
    alpha_ = globaltrainAverage
    beta_u_val = [0]*len(booksperuser.keys())
    beta_i_val = [0]*len(usersperbook.keys())
    beta_u = dict(zip(booksperuser.keys(),beta_u_val))
    beta_i = dict(zip(usersperbook.keys(),beta_i_val))
    msetrain = []
    while (True):
        predtrain = []
        for u, b, r in train[:100000]:
            old_alpha_ = alpha_
            old_beta_u = beta_u[u]
            old_beta_i = beta_i[b]
            alpha_ = sum([i - (old_beta_u + old_beta_i) for i in allRatings[:100000]]) / len(train)
            beta_u[u] = sum([i - (old_alpha_ + old_beta_i) for i in userRatings[u]]) / (c + len(userRatings[u]))
            beta_i[b] = sum([i - (old_alpha_ + old_beta_u) for i in bookRatings[b]]) / (c + len(bookRatings[b]))
            predtrain.append(alpha_ + beta_u[u] + beta_i[b])
        msetrain.append(MSE(predtrain, allRatings[:100000]))
        if (msetrain[-1] - msetrain[-2] < 0.05): break

    predvalid = []
    for u, b, r in valid:
        if b == 'b21479253':
            predvalid.append(globaltrainAverage)
        else:
            predvalid.append(alpha_ + beta_u[u] + beta_i[b])

    actualvalid = [int(v[2]) for v in valid]
    msevalidc.append( MSE(predvalid, actualvalid) )

plt.plot(C,msevalidc)

#Submit to kaggle

#best lamda
lamda = 1
alpha_ = globaltrainAverage
beta_u_val = [0]*len(booksperuser.keys())
beta_i_val = [0]*len(usersperbook.keys())
beta_u = dict(zip(booksperuser.keys(),beta_u_val))
beta_i = dict(zip(usersperbook.keys(),beta_i_val))

msetrain = []
while(True) :
    predtrain = []
    for u, b, r in train[:100000]:
        old_alpha_ = alpha_
        old_beta_u = beta_u[u]
        old_beta_i = beta_i[b]
        alpha_ = sum([i - (old_beta_u + old_beta_i) for i in allRatings[:100000]]) / len(train)
        beta_u[u] = sum([i - (old_alpha_ + old_beta_i) for i in userRatings[u]]) / (lamda + len(userRatings[u]))
        beta_i[b] = sum([i - (old_alpha_ + old_beta_u) for i in bookRatings[b]]) / (lamda + len(bookRatings[b]))
        predtrain.append(alpha_ + beta_u[u] + beta_i[b])
    msetrain.append(MSE(predtrain, allRatings[:100000]))
    if( msetrain[-1] < 1.2 ) : break

#Use model for pairs_rating prediction

#The column name 'prediction' contains the predicted rating values for each user-book pair.


for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    with open('Rating_Predictions_updated.csv', 'w') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(["userID-bookID", "prediction"])
    continue
  u,b = l.strip().split('-')
  if b == 'b21479253':
      with open('Rating_Predictions_updated.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow((u + "-" + b, globaltrainAverage))
  else:
      with open('Rating_Predictions_updated.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow((u+"-"+b,alpha_ + beta_u[u] + beta_i[b]))
