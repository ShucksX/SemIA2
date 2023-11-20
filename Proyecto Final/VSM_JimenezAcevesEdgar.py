import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import numpy as np
import csv


def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

class VSM:

    def __init__(self, learning_rate=0.0001, lambda_param=0.01,n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self,X,y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w)- self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param *self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X , self.w) - self.b
        return np.sign(approx)
print("Maquina Vector Soporte")
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
    
    clf = VSM()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_pred, y_test)
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

