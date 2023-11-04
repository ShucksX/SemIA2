import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import numpy as np
import csv

def unit_step_func(x):
    return np.where(x > 0, 1, 0)
    
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #inicializar parametros
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                #Actualizar
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


print("Cual dataset vas a usar? (1 a 3)")
dataset = int(float(input()))
if(dataset == 1):
    X = np.ndarray(shape=(63,1), dtype=float, order='F')
    y = np.ndarray(shape=(63), dtype=float, order='F')
    cont = 0
    with open("./Dataset1.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            X[cont][0] = row[0]
            y[cont] = row[1]
            cont = cont +1
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1234)
        p = Perceptron(learning_rate=0.01, n_iters=1000)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)
        acc = accuracy(predictions,y_test)
        print("Precision:")
        print(acc)

elif(dataset == 2):
    X2 = np.ndarray(shape=(4898,11), dtype=float, order='F')
    y2 = np.ndarray(shape=(4898), dtype=float, order='F')
    cont = 0
    with open("./Dataset2.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            X2[cont][0] = row[0]
            X2[cont][1] = row[1]
            X2[cont][2] = row[2]
            X2[cont][3] = row[3]
            X2[cont][3] = row[3]
            X2[cont][4] = row[4]
            X2[cont][5] = row[5]
            X2[cont][6] = row[6]
            X2[cont][7] = row[7]
            X2[cont][8] = row[8]
            X2[cont][9] = row[9]
            X2[cont][10] = row[10]
            y2[cont] = row[11]
            cont = cont +1
        X_train, X_test, y_train, y_test = train_test_split(X2,y2, test_size = 0.2, random_state=1234)
        p = Perceptron(learning_rate=0.01, n_iters=1000)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)
        acc = accuracy(predictions,y_test)
        print("Precision:")
        print(acc)

elif(dataset == 3):
    X3 = np.ndarray(shape=(768,8), dtype=float, order='F')
    y3 = np.ndarray(shape=(768), dtype=float, order='F')
    cont = 0
    with open("./Dataset3.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            X3[cont][0] = row[0]
            X3[cont][1] = row[1]
            X3[cont][2] = row[2]
            X3[cont][3] = row[3]
            X3[cont][3] = row[3]
            X3[cont][4] = row[4]
            X3[cont][5] = row[5]
            X3[cont][6] = row[6]
            X3[cont][7] = row[7]
            y3[cont] = row[8]
            cont = cont +1
        X_train, X_test, y_train, y_test = train_test_split(X3,y3, test_size = 0.2, random_state=1234)
        p = Perceptron(learning_rate=0.01, n_iters=1000)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)
        acc = accuracy(predictions,y_test)
        print("Precision:")
        print(acc)

else:
    print("Opcion invalida")
print("Programa terminado, presiona enter para terminar")
input()

