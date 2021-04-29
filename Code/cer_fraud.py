import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

import pickle

dataraw = pd.read_csv('certificate_fraud.csv')

dr=dataraw.dropna()
dataraw.info()
dataraw
x=dr.iloc[:,:-1]
y=dr.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=.2,shuffle=True)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

preds=clf.predict(X_test)

data=[1,1,1,0,0,0]

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))