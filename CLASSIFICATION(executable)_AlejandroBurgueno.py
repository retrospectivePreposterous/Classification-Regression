# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 20:10:05 2021

@author: Alejandro
"""

print("................................................")
print("Por favor, espere mientras se carga el modelo...")

# Importación de librerías generales
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn import svm

# CARGADO Y PROGRAMA
# ----------------------------------------------------------
# Lectura del dataset
iris = pd.read_csv('iris.csv')

# Se aislan las variables de entrada 'X' de la variable predictiva 'y'
X = iris.drop('variety', axis=1)     # Variables 'X', todas menos la etiqueta 'variedad'
y = iris['variety']                  # Variable 'y', la etiqueta 'variedad'

# Se convierten las variables a numpy para posteriormente poder aplicar reshape y adaptar su dimensión
X = np.nan_to_num(X)
y = np.nan_to_num(y)

# Para entrenar los modelos de clasificación es preciso codificar las strings de etiquetas a números
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Finalmente, para evitar overfitting se dividen el conjunto de datos en conjunto de entrenamiento y de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_test = X_test.reshape(-X_test.shape[0],X_test.shape[1])    #Reshape de X_test para poder introducirlo

# Se entrena el modelo
clf = svm.SVC(kernel="poly", gamma='auto', C=20)
clf.fit(X_train, y_train)

# INTERFAZ
# ----------------------------------------------------------

loop = 1

print("¡El modelo está listo!")
print("................................................")
print("")

print("LAS MEDIDAS SON EN (cm)")


while loop == 1:
    
    sl = input("Introduce LONGITUD de SÉPALO (4.3 a 7.9)\n")
    sw = input("Introduce ANCHO de SÉPALO (2.0 a 4.4)\n")
    pl = input("Introduce LONGITUD de PÉTALO (1.0 a 6.9)\n")
    pw = input("Introduce ANCHO de PÉTALO (0.1 a 2.5)\n")
    print("")
    
    prediction = clf.predict([[sl,sw,pl,pw]])
    
    if prediction == [0]:
        print("¡Es una flor de variedad SETOSA!")
    elif prediction == [1]:
        print("¡Es una flor de variedad VERSICOLOR!")
    elif prediction == [2]:
        print("¡Es una flor de variedad VIRGINICA!")
    
    print("")
    print("¿Seguir prediciendo (Si 1 / No 0)?")
    loop = int(input())
