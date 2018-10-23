import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import fftpack
from scipy.fftpack import fft, fftfreq

# con ayuda de https://www.scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html 

arbol = plt.imread('Arboles.png')
arbol_fft = fftpack.fft2(arbol)
plt.figure()
plt.imshow(np.abs(arbol_fft), norm=LogNorm(vmin=1))
plt.colorbar()
plt.savefig('HernandezCindy_FT2D.pdf')

r, c = arbol_fft.shape
a_filter = 20
arbol_fft[:,a_filter:r-a_filter] = 0
arbol_fft[a_filter:r-a_filter,:] = 0

arbol_fft[:,a_filter:r-a_filter] = 0
plt.figure()
plt.imshow(np.abs(arbol_fft), norm=LogNorm(vmin=1))
plt.colorbar()
plt.savefig('HernandezCindy_FT2D_filtrada.pdf')

arbol_filtrada = fftpack.ifft2(arbol_fft)

plt.figure()
plt.imshow(abs(arbol_filtrada), plt.cm.gray)
plt.savefig('HernandezCindy_Imagen_filtrada.pdf')