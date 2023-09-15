import matplotlib.pyplot as plt
import random
import csv

print("¿Uno de cada cuantos datos seran para prueba?")
numprueba = int(float(input()))
print("Cuantas particiones gustas hacer?")
partic = int(float(input()))
print("ARCHIVO: spheres2d50.csv")
contpart = 0
while (contpart < partic):
  print("Particion:")
  print(contpart+1)
  cont = 0

  w0 = random.random()
  w1 = random.random()
  w2 = random.random()
  a = random.random()

  xcoord = []
  ycoord = []
  zcoord = []
  yd = []

  with open("./spheres2d50.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
      if(cont < numprueba-1):
        cont = cont + 1
        suma = int(float(row[0]))*w0+int(float(row[1]))*w1 + int(float(row[2]))*w2
        y = 0
        if (suma < 0):
            y= -1
        else:
            y = 1
        error = int(float(row[3]))-y
        if (error!=0):
            w0 = w0 +(a*error*int(float(row[0])))
            w1 = w1 +(a*error*int(float(row[1])))
            w2 = w1 +(a*error*int(float(row[2])))
      else:
        xcoord.append(int(float(row[0])))
        ycoord.append(int(float(row[1])))
        zcoord.append(int(float(row[2])))
        yd.append(int(float(row[3])))
        cont = 0
  print ("Pesos tras entrenamiento:")
  print(w0)
  print (w1)
  print (w2)
  print(a)

  print("Prueba")
  contprueba = 0
  contcorrec = 0
  while (contprueba < len(xcoord)):
    contprueba = contprueba + 1
    suma = xcoord[contprueba]*w0+ycoord[contprueba]*w1 + zcoord[contprueba]*w2
    y = 0
    if (suma < 0):
      y= -1
    else:
      y = 1
    error = yd[contprueba]-y
    if (error!=0):
      contcorrec = contcorrec + 1
    contprueba = contprueba + 1
  print("Predicciones correctas:")
  print(contcorrec)
  print("de")
  print(len(xcoord))
  contpart = contpart +1




