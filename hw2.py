import random
import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt

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

# Question 4 Sample weights



