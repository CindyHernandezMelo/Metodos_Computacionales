import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

link = 'http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat'

#Leer datos en formato float64 y str
data_numbers = np.genfromtxt(link,delimiter =',')
data_text    = np.genfromtxt(link,delimiter =',', dtype = 'str')

#Combinar datos float64 y str
data = np.empty(data_numbers.shape)
data = data_numbers.copy()

#Binarizar los datos
for i in range(data.shape[0]):
    if data_text[i,1] == 'B':
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
    
print('Los componentes mas importantes se encuentran en los dos ultimos datos')
nuevos_vectores = np.dot(data2,vectores_propios[:,-2:])
    
plt.figure()
plt.scatter(nuevos_vectores[:,0][np.where(data[:,1] == 0)], nuevos_vectores[:,1][np.where(data[:,1] == 0)], label = 'Benigno', alpha=0.4, c='g')
plt.scatter(nuevos_vectores[:,0][np.where(data[:,1] == 1)], nuevos_vectores[:,1][np.where(data[:,1] == 1)], label = 'Maligno', alpha=0.2, c='r')
plt.legend()
plt.savefig('HernandezCindy_PCA.pdf')

print('El metodo de PCA si sirve porque se pueden diferenciar claramente ambos clusters. El unico inconveniente es que ambos clusters parecen tener unos Outliers en donde se interceptan')