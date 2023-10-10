import math
import random
import numdifftools as nd
import numpy as np

def funcion(x):
    aux = -((x[0]**2)+(3*(x[1]**2)))
    return 10-math.exp(aux)

def gradiente_desc(inicio,aprendizaje,iteraciones):
    vector = inicio
    ap = aprendizaje
    for bea in range(iteraciones):
        print("Valores de x")
        print(vector[0])
        print(vector[1])
        print("---------------")
        grad2 =nd.Gradient(funcion)(vector)
        print("Gradiente")
        print(grad2)
        diff1= -ap* grad2[0]
        diff2= -ap* grad2[1]
        
        vector[0] = vector[0]+diff1
        vector[1] = vector[1]+diff2
    return vector

print ("Gradiente descendiente")
x = [random.uniform(-1,10),random.uniform(-1,1)]
#print(x)
#print(funcion(x))

grad = gradiente_desc(x,0.2,30)
print(grad[0])
print(grad[1])

