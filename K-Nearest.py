from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from sklearn import datasets

iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target
print(iris_data)

x_train, x_test, y_train, y_test=(train_test_split(iris_data, iris_labels, test_size=0.20))
classifier=KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print("Confusion matrix is as follow: \n",cm)
print("Accuracy matrix")
print(classification_report(y_test, y_pred))
print("Correct Prediction", accuracy_score(y_test, y_pred))
print("Wrong Prediction", (1-accuracy_score(y_test, y_pred)))
