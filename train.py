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
    return y

def norm(X):
    return preprocessing.normalize(X, norm='l2')

X = data[train_index]
X = norm(X.reshape(X.shape[0],X.shape[1]*X.shape[2]))
y = onehot(labels[train_index])

#clf = RandomForestClassifier().fit(X,y)
clf = SVC(class_weight='balanced').fit(X, y)
#clf = LogisticRegression(max_iter=10000).fit(X,y)

Xt = data[test_index]
Xt = norm(Xt.reshape(Xt.shape[0],Xt.shape[1]*Xt.shape[2]))
yt = onehot(labels[test_index])

print(clf.score(X, y))
print(clf.score(Xt, yt))
