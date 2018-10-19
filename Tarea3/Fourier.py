import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
from scipy.fftpack import fft, fftfreq

signal = np.genfromtxt('signal.dat')
incompletos = np.genfromtxt('incompletos.dat')

plt.figure()
plt.plot(signal[:,0], signal[:,-1], label = 'Signal.dat')
plt.legend()
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Datos de Signal.dat')
plt.savefig('HernandezCindy_signal.pdf')

#Transformada de Fourier
signal_n = np.shape(signal)[0]
signal_fr = fftfreq(signal_n, signal[1,0]-signal[0,0])

signal_fft = np.zeros(signal_n)+0*1j

for i in range(signal_n):  
    signal_fft[i] = np.sum(signal[:,-1]*np.exp(-2*np.pi*1j*i*np.arange(1,signal_n+1)/signal_n))

plt.figure()
plt.plot(signal_fr, abs(signal_fft), label ='Signal.dat')
plt.legend()
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.title('Transformada de Fourier de Signal.dat')

