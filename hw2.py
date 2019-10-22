import random
import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict


#########################################

# From https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
f = open("5year.arff", 'r')

while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)



#########################################

#Question 1

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

mod = linear_model.LogisticRegression(C=1.0)
mod.fit(X,y)
pred = mod.predict(X)
correct = pred == y

#accuracy
sum(correct)*1.0 / len(correct)


#BER
TP_ = numpy.logical_and(pred, y)
FP_ = numpy.logical_and(pred, numpy.logical_not(y))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(y))
FN_ = numpy.logical_and(numpy.logical_not(pred), y)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)

1 - 0.5 * (TP*1.0 / (TP + FN) + TN*1.0 / (TN + FP))



#########################################


#Question 2

#shuffle
Xy = list(zip(X,y))
random.shuffle(Xy)

X = [d[0] for d in Xy]
y = [d[1] for d in Xy]


#split into train valid test 50,25,25
N = len(Xy)
Ntrain = N*0.5
Ntrain = int(Ntrain)
NValid = N*0.25
NValid = Ntrain + NValid
NValid = int(NValid)

Xtrain = X[:Ntrain]
Xvalid = X[Ntrain:NValid]
Xtest = X[NValid:]

ytrain = y[:Ntrain]
yvalid = y[Ntrain:NValid]
ytest = y[NValid:]

mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(Xtrain,ytrain)
predtrain = mod.predict(Xtrain)
predval = mod.predict(Xvalid)
pred = mod.predict(Xtest)


#Accuracy and BER of training set
TP_ = numpy.logical_and(predtrain, ytrain)
FP_ = numpy.logical_and(predtrain, numpy.logical_not(ytrain))
TN_ = numpy.logical_and(numpy.logical_not(predtrain), numpy.logical_not(ytrain))
FN_ = numpy.logical_and(numpy.logical_not(predtrain), ytrain)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
print("The accuracy and BER of training set is : " + str((TP + TN)*1.0 / (TP + FP + TN + FN)) + " , " + str(1 - 0.5 * (TP*1.0 / (TP + FN) + TN*1.0 / (TN + FP))))



#Accuracy and BER of validation set
TP_ = numpy.logical_and(predval, yvalid)
FP_ = numpy.logical_and(predval, numpy.logical_not(yvalid))
TN_ = numpy.logical_and(numpy.logical_not(predval), numpy.logical_not(yvalid))
FN_ = numpy.logical_and(numpy.logical_not(predval), yvalid)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
print("The accuracy and BER of validation set is : " + str((TP + TN)*1.0 / (TP + FP + TN + FN)) + " , " + str(1 - 0.5 * (TP*1.0 / (TP + FN) + TN*1.0 / (TN + FP))))


#Accuracy and BER of test set
TP_ = numpy.logical_and(pred, ytest)
FP_ = numpy.logical_and(pred, numpy.logical_not(ytest))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(ytest))
FN_ = numpy.logical_and(numpy.logical_not(pred), ytest)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)
print("The accuracy and BER of test set is : " + str((TP + TN)*1.0 / (TP + FP + TN + FN)) + " , " + str(1 - 0.5 * (TP*1.0 / (TP + FN) + TN*1.0 / (TN + FP))))



#########################################



#Question 3 regularization


Cindex = [-4,-3,-2,-1,0,1,2,3,4]

Acctrain = []
BERtrain = []
Accval = []
BERval = []
Acctest = []
BERtest = []

for ci in Cindex:
    mod = linear_model.LogisticRegression(C=10**ci, class_weight='balanced')
    mod.fit(Xtrain, ytrain)
    predtrain = mod.predict(Xtrain)
    predval = mod.predict(Xvalid)
    pred = mod.predict(Xtest)

    # Accuracy and BER of training set
    TP_ = numpy.logical_and(predtrain, ytrain)
    FP_ = numpy.logical_and(predtrain, numpy.logical_not(ytrain))
    TN_ = numpy.logical_and(numpy.logical_not(predtrain), numpy.logical_not(ytrain))
    FN_ = numpy.logical_and(numpy.logical_not(predtrain), ytrain)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Acctrain.append((TP + TN) * 1.0 / (TP + FP + TN + FN))
    BERtrain.append(1 - 0.5 * (TP * 1.0 / (TP + FN) + TN * 1.0 / (TN + FP)))

    # Accuracy and BER of validation set
    TP_ = numpy.logical_and(predval, yvalid)
    FP_ = numpy.logical_and(predval, numpy.logical_not(yvalid))
    TN_ = numpy.logical_and(numpy.logical_not(predval), numpy.logical_not(yvalid))
    FN_ = numpy.logical_and(numpy.logical_not(predval), yvalid)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Accval.append((TP + TN) * 1.0 / (TP + FP + TN + FN))
    BERval.append(1 - 0.5 * (TP * 1.0 / (TP + FN) + TN * 1.0 / (TN + FP)))

    # Accuracy and BER of test set
    TP_ = numpy.logical_and(pred, ytest)
    FP_ = numpy.logical_and(pred, numpy.logical_not(ytest))
    TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(ytest))
    FN_ = numpy.logical_and(numpy.logical_not(pred), ytest)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Acctest.append((TP + TN) * 1.0 / (TP + FP + TN + FN))
    BERtest.append(1 - 0.5 * (TP * 1.0 / (TP + FN) + TN * 1.0 / (TN + FP)))

#plot
plt.close()
print(" Distribution of Accuracy and BER for all three sets of data at different C values is as follows:")
plt.xlabel("C value: power of 10 ")
plt.ylabel("BER")
plt.title("Regularization pipeline")

plt.plot(Cindex, Acctrain, '--g', label='Accuracytrain')
plt.plot(Cindex, Accval, '--b', label='Accuracyval')
plt.plot(Cindex, Acctest, '--r', label='Accuracytest')

plt.plot(Cindex, BERtrain, '-g', label='BERtrain')
plt.plot(Cindex, BERval, '-b', label='BERval')
plt.plot(Cindex, BERtest, '-r', label='BERtest')

plt.legend()


#small C values will increase the regularization strenght which implies the creation of simple models that tend to underfit the data.
# By using bigger C values, the model can increase it's complexity and adjust better to the data
#We go with the last c value since the difference between test and train ber is smallest, meaning the model is more generalised to unseen data.
# That means the highest complexity model, ie, low regularization strength or lamda value




#########################################

# Question 4 Sample weights


#shuffle

X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

Xy = list(zip(X,y))
random.shuffle(Xy)

X = [d[0] for d in Xy]
y = [d[1] for d in Xy]


#split into train test
N = len(Xy)
Ntrain = N*0.8
Ntrain = int(Ntrain)

Xtrain = X[:Ntrain]
Xtest = X[Ntrain:]

ytrain = y[:Ntrain]
ytest = y[Ntrain:]

#equally weighted

weights = [1.0] * len(ytrain)
mod = linear_model.LogisticRegression(C=1, solver='lbfgs')
mod.fit(Xtrain, ytrain, sample_weight=weights)
pred = mod.predict(Xtest)

TP_ = numpy.logical_and(pred, ytest)
FP_ = numpy.logical_and(pred, numpy.logical_not(ytest))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(ytest))
FN_ = numpy.logical_and(numpy.logical_not(pred), ytest)
TP = sum(TP_)
FP = sum(FP_)
TN = sum(TN_)
FN = sum(FN_)

Precision = TP *1.0 / (TP + FP)
Precision
Recall = TP*1.0 / (TP + FN)
Recall
F1 = 2 * (Precision*Recall) /(Precision + Recall)
F1
F10 = (101) * (Precision*Recall) /((100 * Precision) + Recall)
F10


#Better weights

weights = []
for i in ytrain:
    if i == False:
        weights.append(1)
    else:
        weights.append(18)
mod = linear_model.LogisticRegression(C=1, solver='lbfgs')
mod.fit(Xtrain, ytrain, sample_weight=weights)
pred = mod.predict(Xtest)

TP_ = numpy.logical_and(pred, ytest)
FP_ = numpy.logical_and(pred, numpy.logical_not(ytest))
TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(ytest))
FN_ = numpy.logical_and(numpy.logical_not(pred), ytest)
TP = sum(TP_)
TP
FP = sum(FP_)
FP
TN = sum(TN_)
TN
FN = sum(FN_)
FN

Precision = TP *1.0 / (TP + FP)
Precision
Recall = TP*1.0 / (TP + FN)
Recall
F1 = 2 * (Precision*Recall) /(Precision + Recall)
F1
F10 = (101) * (Precision*Recall) /((100 * Precision) + Recall)
F10


######################################################################################################

# Question 7

#Consider the 64 values in the dataset removing the 1 that we added earlier
random.shuffle(dataset)
X = [d[1:-1] for d in dataset]
y = [d[-1] for d in dataset]
N = len(X)
Ntrain = N*0.5
Ntrain = int(Ntrain)
NValid = N*0.25
NValid = Ntrain + NValid
NValid = int(NValid)
Xtrain = X[:Ntrain]
Xvalid = X[Ntrain:NValid]
Xtest = X[NValid:]
ytrain = y[:Ntrain]
yvalid = y[Ntrain:NValid]
ytest = y[NValid:]

pca = PCA(n_components=64)
pca.fit(Xtrain)

#first PCA component
pca.components_[0]

#training a model

Xpca_train = numpy.matmul(Xtrain, pca.components_.T)
Xpca_valid = numpy.matmul(Xvalid, pca.components_.T)
Xpca_test = numpy.matmul(Xtest, pca.components_.T)

n = numpy.arange(5, 35, 5).tolist()
Bpca_val = []
Bpca_test = []

for i in n:
    Xpca_train = [x[:60] for x in Xpca_train]
    Xpca_valid = [x[:60] for x in Xpca_valid]
    Xpca_test = [x[:60] for x in Xpca_test]

    mod = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
    mod.fit(Xpca_train, ytrain)
    predval = mod.predict(Xpca_valid)
    pred = mod.predict(Xpca_test)

    #  BER of validation set
    TP_ = numpy.logical_and(predval, yvalid)
    FP_ = numpy.logical_and(predval, numpy.logical_not(yvalid))
    TN_ = numpy.logical_and(numpy.logical_not(predval), numpy.logical_not(yvalid))
    FN_ = numpy.logical_and(numpy.logical_not(predval), yvalid)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Bpca_val.append(1 - 0.5 * (TP * 1.0 / (TP + FN) + TN * 1.0 / (TN + FP)))

    #  BER of test set
    TP_ = numpy.logical_and(pred, ytest)
    FP_ = numpy.logical_and(pred, numpy.logical_not(ytest))
    TN_ = numpy.logical_and(numpy.logical_not(pred), numpy.logical_not(ytest))
    FN_ = numpy.logical_and(numpy.logical_not(pred), ytest)
    TP = sum(TP_)
    FP = sum(FP_)
    TN = sum(TN_)
    FN = sum(FN_)
    Bpca_test.append(1 - 0.5 * (TP * 1.0 / (TP + FN) + TN * 1.0 / (TN + FP)))

    Bpca_val
    Bpca_test