import pandas as pd

import numpy as np

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=';')

data_set = np.array(df, dtype=np.float)

target = data_set[:, -1] # variable cible

data = data_set[:, :-1]

#Début de la standarisation des variables

data = data - data.mean(0)

data = data / data.std(0)

target = target - np.mean(target)

target = target / np.std(target)

#Fin de la standarisation des variables

# Début de la programmation de l'estimateur des moindres carrés "beta"

dataTxtarget = np.dot(data.T, target) #produit matriciel en la transposée de data et target

datainv = np.linalg.inv(np.dot(data.T, data))

beta = np.dot(datainv, dataTxtarget)

#Fin de la programmmation de l'estimateur des moindres carrés "beta"

int(np.sum((np.dot(data, beta) - target)**2)) #Calcul des résidus