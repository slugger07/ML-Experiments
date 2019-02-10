import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('creditcard.csv')
features = ['V%d' % number for number in range(1, 28)] + ['Amount']

target = 'Class'
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

LR = LogisticRegression()
neigh = KNeighborsClassifier(n_neighbors=3)
svmclf = svm.SVC(gamma='scale')
DTclf = tree.DecisionTreeClassifier()
RFclf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

LR.fit(X_train, y_train)
neigh.fit(X_train, y_train)
svmclf.fit(X_train, y_train)
DTclf.fit(X_train, y_train)
RFclf.fit(X_train, y_train)

y_predLR = LR.predict(X_test)
y_predneigh = neigh.predict(X_test)
y_predsvm = svmclf.predict(X_test)
y_DTclf = DTclf.predict(X_test)
y_RFclf =  RFclf.predict(X_test)

print("Accuracy Scores")
print("LR : ",accuracy_score(y_test, y_predLR))
print("KNN : ",accuracy_score(y_test, y_predneigh))
print("SVM : ", accuracy_score(y_test, y_predsvm))
print("DecisionTrees : ", accuracy_score(y_test,y_DTclf))
print("Random Forest : ", accuracy_score(y_test,y_RFclf))
