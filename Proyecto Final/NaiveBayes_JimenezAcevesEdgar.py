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

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype =np.float64)
        self._var = np.zeros((n_classes, n_features), dtype =np.float64)
        self._priors = np.zeros(n_classes, dtype =np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors= []
        #Calcular la probabilidad posterior
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        #Devolver la class con el posterior mas grande
        return self._classes[np.argmax(posteriors)]


    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) /(2*var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator / denominator

print("Naive Bayes")
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
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
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

