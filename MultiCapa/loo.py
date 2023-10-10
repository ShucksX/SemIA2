import matplotlib.pyplot as plt
import numdifftools as nd

import numpy as np
import random
import csv

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut


def evaluate(y):
  setosa = np.array([0,0,1])
  versicolor = np.array([0,1,0])
  virginica =  np.array([1,0,0])
  if((y== setosa).all()):
    print("Setosa")
  elif((y==versicolor).all()):
    print("Versicolor")
  elif((y==virginica).all()):
    print("Virginica")
  else:
    print(y)



#1/Cuantas vecens se hace el entrenamiento
print("Cuantas particiones gustas hacer?")
partic = int(float(input()))
#Cantidad de capas ocultas
print("Cuantas capas oculta tendra el perceptron?")
capas = int(float(input()))
#Cantidad de neuronas por capa
neuronas = np.zeros(shape =  (capas))


X = np.ndarray(shape=(150,4), dtype=float, order='F')
y = np.ndarray(shape=(150,3), dtype=float, order='F')
cont = 0

with open("./irisbin.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
      X[cont][0] = row[0]
      X[cont][1] = row[1]
      X[cont][2] = row[2]
      X[cont][3] = row[3]
      if(row[4] == "1"):
        y[cont][0] = 1
      else:
        y[cont][0] = 0
      if(row[5] == "1"):
        y[cont][1] = 1
      else:
        y[cont][1] = 0
      if(row[6] == "1"):
        y[cont][2] = 1
      else:
        y[cont][2] = 0
      cont += 1
print("Entradas")
print(X)
print("Salidas esperados")
print(y)

# Create an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(capas,), activation='relu', max_iter=partic, random_state=42)

# Initialize Leave-One-Out cross-validation
loo = LeaveOneOut()

# Initialize lists to store results
predictions = []
true_labels = []

# Perform Leave-One-Out cross-validation
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the MLP on the training data
    mlp.fit(X_train, y_train)

    # Predict on the test data
    y_pred = mlp.predict(X_test)

    # Store the prediction and true label
    predictions.append(y_pred[0])
    true_labels.append(y_test[0])
    cont = 0
    for row in y_pred:
      print("Caracteristicas;")
      print(X_test[cont])
      print("Resultado esperado")
      evaluate(y_test[cont])
      print("Resultado del perceptron:")
      evaluate(row)
      print("---------")
      cont += 1
    
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Precision: {accuracy:.2f}")
