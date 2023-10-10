import numpy as np
from sklearn.model_selection import LeavePOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numdifftools as nd

import numpy as np
import random
import csv


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

# Number of data points to leave out
k = 2

# Initialize an MLP classifier
classifier = MLPClassifier(hidden_layer_sizes=(capas,10), max_iter=partic, random_state=42)

# Create a LeavePOut iterator
leave_p_out = LeavePOut(p=k)

# List to store the model's performance scores
scores = []

# Perform leave-k-out cross-validation
cont = 0
for train_index, test_index in leave_p_out.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the MLP classifier on the training data
    classifier.fit(X_train, y_train)

    # Predict on the test data
    y_pred = classifier.predict(X_test)

    # Calculate accuracy for this iteration
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)

# Calculate the average performance score
average_accuracy = np.mean(scores)
print(y_pred)
print(X_test)
print("Average Accuracy:", average_accuracy)
