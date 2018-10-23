import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from scipy import interpolate

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
plt.savefig('HernandezCindy_TF.pdf')

#Diseno del Filtro
signal_idxpasabajas1000 = np.ones(signal_n) 
signal_idxpasabajas1000 = np.where(abs(signal_fr)<1000, signal_idxpasabajas1000, 0)

signal_pasabajas1000 = np.fft.ifft(signal_fft*signal_idxpasabajas1000)

plt.figure()
plt.plot(signal[:,0],signal_pasabajas1000, label ='Signal.dat')
plt.legend()
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Datos de Signal.dat filtrados')
plt.savefig('HernandezCindy_filtrada.pdf')

# Incompletos.dat
print('')
print('No se puede realizar la transformada de Fourier con la interpolacion')
print('porque no todos los datos estan tomados con el mismo dt')

f_cuadra =  interpolate.interp1d(incompletos[:,0], incompletos[:,-1], kind =  'quadratic')
f_cubica =  interpolate.interp1d(incompletos[:,0], incompletos[:,-1], kind = 'cubic')

incompletos_t = np.linspace(incompletos[:,0][0],incompletos[:,0][-1],512)

incompletos_cuadra = f_cuadra(incompletos_t)
incompletos_cubica = f_cubica(incompletos_t)

incompletos_cuadra_fft = fft(incompletos_cuadra)
incompletos_cubica_fft = fft(incompletos_cubica)

plt.figure()
plt.subplot(311)
plt.plot(signal_fr, abs(signal_fft), label ='Signal.dat')
plt.title('Transformada de Fourier')
plt.legend()
plt.subplot(312)
plt.plot(signal_fr, abs(incompletos_cuadra_fft), label = 'Interpolacion Cuadratica')
plt.legend()
plt.subplot(313)
plt.plot(signal_fr, abs(incompletos_cubica_fft), label = 'Interpolacion Cubica')
plt.legend()
print('')
print('Los componentes que no se encuentran en las frecuencias principales')
print('aumentan su magnitud')

signal_idxpasabajas500 = np.ones(signal_n) 
signal_idxpasabajas500 = np.where(abs(signal_fr)<500, signal_idxpasabajas500, 0)

incompletos_cuadra_pasabajas1000 = np.fft.ifft(incompletos_cuadra_fft*signal_idxpasabajas1000)
incompletos_cubica_pasabajas1000 = np.fft.ifft(incompletos_cubica_fft*signal_idxpasabajas1000)

signal_pasabajas500 = np.fft.ifft(signal_fft*signal_idxpasabajas1000)
incompletos_cuadra_pasabajas500 = np.fft.ifft(incompletos_cuadra_fft*signal_idxpasabajas500)
incompletos_cubica_pasabajas500 = np.fft.ifft(incompletos_cubica_fft*signal_idxpasabajas500)

plt.figure()
plt.subplot(121)
plt.plot(signal[:,0],signal_pasabajas1000, label ='Signal.dat')
plt.plot(signal[:,0],incompletos_cuadra_pasabajas1000, label ='Cuadratica')
plt.plot(signal[:,0],incompletos_cubica_pasabajas1000, label ='Cubica')
plt.title('Filtro Pasa Bajas de 1000Hz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()

plt.subplot(122)
plt.plot(signal[:,0],signal_pasabajas500, label ='Signal.dat')
plt.plot(signal[:,0],incompletos_cuadra_pasabajas500, label ='Cuadratica')
plt.plot(signal[:,0],incompletos_cubica_pasabajas500, label ='Cubica')
plt.title('Filtro Pasa Bajas de 500Hz')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.savefig('HernandezCindy_2Filtros.pdf')


