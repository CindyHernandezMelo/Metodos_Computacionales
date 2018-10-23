import numpy as np
from numpy import linalg as LA

link = 'http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat'

#Leer datos en formato float64 y str
data_numbers = np.genfromtxt(link,delimiter =',')
data_text    = np.genfromtxt(link,delimiter =',', dtype = 'str')

#Combinar datos float64 y str
data = np.empty(data_numbers.shape)
data = data_numbers.copy()

#Binarizar los datos
for i in range(data.shape[0]):
    if data_text[i,1] == data_text[568,1]:
        data[i,1] = 1
    else:
        data[i,1] = 0
        
data2 = np.zeros(data.shape)

for i in range(data.shape[1]):
    data2[:,i]=(data[:,i]-np.mean(data[:,i]))/(np.var(data[:,i])**0.5)

covarianza_calculada = np.zeros([data.shape[1],data.shape[1]])

for i in range(np.shape(covarianza_calculada)[0]):
    for j in range(np.shape(covarianza_calculada)[1]):
        covarianza_calculada[i,j] = np.sum(data2[:,i]*data2[:,j])/(data.shape[0]-1)
        
valores_propios ,vectores_propios = LA.eigh(covarianza_calculada)

for i in range(len(valores_propios)):
    print('Valor Propio')
    print(valores_propios[i])
    print('Vector Propio')
    print(vectores_propios[:,i])
    
    
