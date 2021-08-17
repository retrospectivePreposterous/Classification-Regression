# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:02:54 2021

@author: Alejandro
"""

print("................................................")
print("Por favor, espere mientras se carga el modelo...")

# Importación de librerías generales
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# CARGADO Y PROGRAMA
# ----------------------------------------------------------
# Lectura del dataset
air_q = pd.read_csv('AirQualityUCI.csv')

# Como el tiempo no es relevante en este método, se eliminan los valores anómalos sin atender a su remplazamiento en las 24 horas
for i in air_q:
  air_q = air_q[air_q[i] != -200]

# Se mantiene la variable CO(GT), eliminando el resto que mantienen una fuerte correlación directa
variables_to_drop = ['Date', 'Time', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
air_q = air_q.drop(variables_to_drop, axis=1)

# Se aislan las variables de entrada 'X' de la variable predictiva 'y'
X = air_q.drop('AH', axis=1)     # Variables 'X', todas menos la etiqueta 'AH'
y = air_q['AH']                  # Variable 'y', la etiqueta 'AH'

# Se convierten las variables a numpy para posteriormente poder aplicar reshape y adaptar su dimensión
X = np.nan_to_num(X)
y = np.nan_to_num(y)

# Finalmente, para evitar overfitting se dividen el conjunto de datos en conjunto de entrenamiento y de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_test = X_test.reshape(-X_test.shape[0],X_test.shape[1])    #Reshape de X_test para poder introducirlo

# Se entrena el modelo de Regresion de Random Forest
regr = RandomForestRegressor(max_depth=20, n_estimators = 900, random_state=0)
regr.fit(X_train, y_train)

# INTERFAZ
# ----------------------------------------------------------

loop = 1

print("¡El modelo está listo!")
print("................................................")
print("")

print("PREDICE LA HUMEDAD ABSOLUTA EN FUNCIÓN DEL C0, PT08.S3, Temperatura Y Humedad Relativa")


while loop == 1:
    
    co = input("Introduce la cantidad de CO en mg/m3 (0.3 a 8.1)\n")
    pt = input("Introduce la cantidad de PT08.S3 - Óxido de Tungsteno (461 a 1935)\n")
    t = input("Introduce la Temperatura ºC (6.3 a 30.0)\n")
    rh = input("Introduce la Humedad Relativa % (14.9 a 83.0)\n")
    print("")
    
    prediction = regr.predict([[co,pt,t,rh]])
    
    print("La Humedad Absoluta es igual a " + str(round(prediction[0],2)) )
    
    print("")
    print("¿Seguir prediciendo (Si 1 / No 0)?")
    loop = int(input())