import matplotlib.pyplot as plt
import numdifftools as nd

import numpy as np
import random
import csv

def sigmoide(suma):
  return 1/(1+np.exp(-suma))
def sigmoide_derivative(suma):
  return suma*(1-suma)


#1/Cuantas vecens se hace el entrenamiento
print("Cuantas particiones gustas hacer?")
partic = int(float(input()))
#Cantidad de capas ocultas
print("Cuantas capas oculta tendra el perceptron?")
capas = int(float(input()))
#Cantidad de neuronas por capa
neuronas = np.zeros(shape =  (capas))
neucont = 0;

while (neucont < capas):
  print("Cuantas neuronas tendra la capa")
  print(neucont)
  neuronas[neucont] = (int(float(input())))
  neucont = neucont +1
#print (neuronas)
#Definir variables aleatoriamente
a = 0.1
pesos = np.zeros(shape = (capas+1,int(neuronas.max()),int(neuronas.max())))
contcapa = 0
contneurona = 0
contpeso = 0
while (contcapa < capas +1):
  limneu = 0
  if (contcapa == capas):
    limneu = 1
  else:
    limneu = neuronas[contcapa]
  while(contneurona < limneu):
    neuant = 0;
    if (contcapa-1 < 0):
      neuant = 2;
    else:
      neuant = neuronas[contcapa-1]
    while (contpeso < neuant):
      pesos[contcapa,contneurona,contpeso] = random.random()
      contpeso = contpeso+1
    contpeso = 0
    contneurona = contneurona + 1
  contneurona = 0
  contcapa = contcapa + 1
print("Factor de aprendizaje")
print(a)
print("Pesos iniciales")
print (pesos)

#Inicio del perceptron
print("ARCHIVO: concentlite.csv")
contpart = 0
y = 0
while (contpart < partic):
  print("Particion:")
  print(contpart+1)
  row_count = 0
  #Contar lineas
  with open("./concentlite.csv", 'r') as file:
    csvreader = csv.reader(file)
    row_count = sum(1 for row in csvreader)
    print(row_count)
  #Perceptron
  grupo1 =np.zeros(shape =(2,row_count))
  grupo2 =np.zeros(shape =(2,row_count))
  salidas =np.zeros(shape =(capas,int(neuronas.max())))
  deltas =np.zeros(shape =(capas+1,int(neuronas.max())))

  contcol =0
  with open("./concentlite.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
      yd = int(float(row[2]))
      #Salidas de la capa de entrada                      
      x0 = float(row[0])
      x1 = float(row[1])
      #Obtener salida
      contcapa = 0
      contneurona = 0
      contpeso = 0
      
      while (contcapa < capas +1):
        limneu = 0
        if (contcapa == capas):
          limneu = 1
        else:
          limneu = neuronas[contcapa]
        while(contneurona < limneu):
          neuant = 0;
          if (contcapa-1 < 0):
              suma = 0
              suma = suma + (x0*pesos[contcapa,contneurona,0])
              suma = suma + (x1*pesos[contcapa,contneurona,1])
              #Guardar salida de la neurona en la capa actual
              salidas[contcapa,contneurona] = sigmoide(suma)
              contpeso = contpeso+1
          else:
            neuant = neuronas[contcapa-1]
            suma = 0
            while (contpeso < neuant):
              suma = suma + (salidas[contcapa-1,contpeso]*pesos[contcapa,contneurona,contpeso])
              contpeso = contpeso+1
            if(contcapa == capas):
              y = sigmoide(suma)
            else:
              salidas[contcapa,contneurona] = sigmoide (suma)
          contpeso = 0
          contneurona = contneurona + 1
        contneurona = 0
        contcapa = contcapa + 1
      #Retropropagacion
      perdida = yd-y
      print(perdida)
      d_salida =perdida * (1/(abs(y)+1)**2)
      #Getting data for plotting
      if(yd < 0):
        grupo1[0,contcol] = x0
        grupo1[1,contcol] = x1
      else:
        grupo2[0,contcol] = x0
        grupo2[1,contcol] = x1
      contcol = contcol +1
      #print("Salida")
      #print(deltas)
        
    print("Pesos tras entrenamiento")
    print(pesos)
    plt.plot(grupo1[0],grupo1[1], 'ro')
    plt.plot(grupo2[0],grupo2[1], 'bo')
    plt.show()


  #Contar lineas
  print ("PRUEBA")
  with open("./concentlite.csv", 'r') as file:
    csvreader = csv.reader(file)
    row_count = sum(1 for row in csvreader)
    print(row_count)
  #PRUEBA
  grupo1 =np.zeros(shape =(2,row_count))
  grupo2 =np.zeros(shape =(2,row_count))
  salidas =np.zeros(shape =(capas,int(neuronas.max())))
  contcol =0
  with open("./concentlite.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
      yd = int(float(row[2]))
      #Salidas de la capa de entrada                      
      x0 = float(row[0])
      x1 = float(row[1])
      #Obtener salida
      contcapa = 0
      contneurona = 0
      contpeso = 0
      
      while (contcapa < capas +1):
        limneu = 0
        if (contcapa == capas):
          limneu = 1
        else:
          limneu = neuronas[contcapa]
        while(contneurona < limneu):
          neuant = 0;
          if (contcapa-1 < 0):
              suma = 0
              suma = suma + (x0*pesos[contcapa,contneurona,0])
              suma = suma + (x1*pesos[contcapa,contneurona,1])
              #Guardar salida de la neurona en la capa actual
              salidas[contcapa,contneurona] = sigmoide(suma)
              contpeso = contpeso+1
          else:
            neuant = neuronas[contcapa-1]
            suma = 0
            while (contpeso < neuant):
              suma = suma + (salidas[contcapa-1,contpeso]*pesos[contcapa,contneurona,contpeso])
              contpeso = contpeso+1
            if(contcapa == capas):
              y = sigmoide(suma)
              #print(y)
              #Getting data for plotting
              if(y < 0):
                grupo1[0,contcol] = x0
                grupo1[1,contcol] = x1
              else:
                grupo2[0,contcol] = x0
                grupo2[1,contcol] = x1
            else:
              salidas[contcapa,contneurona] = sigmoide (suma)
          contpeso = 0
          contneurona = contneurona + 1
        contneurona = 0
        contcapa = contcapa + 1
        
      contcol = contcol +1
    contpart = contpart +1
    plt.plot(grupo1[0],grupo1[1], 'ro')
    plt.plot(grupo2[0],grupo2[1], 'bo')
    plt.show()


