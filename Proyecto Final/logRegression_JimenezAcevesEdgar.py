import matplotlib.pyplot as plt

import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def sigmoid(x):
    return 1/(1+np.exp(-x))
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

class LogisticRegression():
    def __init__(self,lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            dw = (1/n_samples) * np.dot(X.T, (predictions-y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X,self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred


print("Logistic Regression")
X = np.ndarray(shape=(101,16), dtype=float, order='F')
y = np.ndarray(shape=(101), dtype=float, order='F')
cont = 0

with open("./DataSetZoo.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        X[cont][0] = row[1]
        X[cont][1] = row[2]
        X[cont][2] = row[3]
        X[cont][3] = row[4]
        X[cont][4] = row[5]
        X[cont][5] = row[6]
        X[cont][6] = row[7]
        X[cont][7] = row[8]
        X[cont][8] = row[9]
        X[cont][9] = row[10]
        X[cont][10] = row[11]
        X[cont][11] = row[12]
        X[cont][12] = row[13]
        X[cont][13] = row[14]
        X[cont][14] = row[15]
        X[cont][15] = row[16]
        y[cont] = row[17]
        cont = cont +1
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1234)
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_pred,y_test)
    print("Accuracy:")
    print(acc)
    precision = precision_score(y_test, y_pred,average='macro', zero_division=1)
    print("Precision:")
    print(precision)
    sensitivity = recall_score(y_test, y_pred,average='macro', zero_division=1)
    print("Sensitivity:")
    print(sensitivity)
    #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #specificity = tn / (tn + fp)
    print("Specifity:")
    print("No se puede calcular debido a las multiples clasificaciones")
    f1 = f1_score(y_test, y_pred,average='macro', zero_division=1)
    print("F1 score:")
    print(f1)
print("Programa terminado, presiona enter para terminar")
input()
