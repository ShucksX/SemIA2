import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score



import numpy as np
import csv

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        #Calcular distancias
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #Obtener la k mas cercana
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #Mayoria
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
        

print("K nearest neighbors")
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
    clf = KNN(k=5)
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

