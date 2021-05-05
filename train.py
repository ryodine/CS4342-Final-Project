import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

data = np.load("data.npy")
labels = np.load("labels.npy")
l = len(data)
train_index = np.arange(1,6*len(data)//10)
test_index = np.arange(6*len(data)//10,8*len(data)//10)
validate_index = np.arange(8*len(data)//10.8,len(data))

def onehot(y):
    yy = np.zeros((len(y), 5))
    yy[np.arange(y.shape[0]), y.astype(int)] = 1
    return yy

def norm(X):
    return preprocessing.normalize(X, norm='l2')

X = data[train_index]
X = norm(X.reshape(X.shape[0],np.prod(X.shape[1:])))
y = onehot(labels[train_index])

Xt = data[test_index]
Xt = norm(Xt.reshape(Xt.shape[0],np.prod(Xt.shape[1:])))
yt = onehot(labels[test_index])
print(yt.shape)


def classify(name, clf):
    print(name + "\n")
    print(clf.score(X, labels[train_index]))
    print(clf.score(Xt, labels[test_index]))


print("pct classes", np.sum(yt,axis=0)/np.sum(yt))

#classify("random forest", RandomForestClassifier().fit(X,y))
#classify("svm", SVC().fit(X, y))
classify("softmax", LogisticRegression(max_iter=10000, multi_class="ovr").fit(X,labels[train_index]))