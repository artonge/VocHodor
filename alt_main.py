from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.io import wavfile
import numpy as np
import math
import wave
import matplotlib.pyplot as plt


# Return the closest power of 2 from the given n
# n -> INT
# Return INT
def closestPowerOf2(n):
    if n < 2: return 0
    power = 2
    while n > power: power *= 2
    return power/2


# fe = 48 000Hz = la frequence dechantillionnage du signal
# x = le signal
fe, x_tmp  = wavfile.read('photographe.wav') # Signal de la parole
x = x_tmp[:, 0]

Tvoulu = 0.05                                #  50ms     |  FLOAT  |  Duree theorique d'une fenetre
Nvoulu = Tvoulu*fe                           #  2400     |  INT    |  Nombre theorique d'echantillon dans une fenetre
N  = closestPowerOf2(Nvoulu)                 #  2048     |  INT    |  On aroundi a la puissance de 2 la plus proche pour la fft
T  = float(N) / float(fe)                    #  42.66ms  |  FLOAT  |  On decoupe le signal en fenetres de duree T = 85.3ms
dt = T / 2.0                                 #  21.33ms  |  FLOAT  |  Duree du d'avancement dans le signal lors de la decoupe
dn = N / 2                                   #  1024     |  INT    |  Pas d'avancement en echantillons
xLen   = len(x)                              #  76776    |  INT    |  Nombre d'echantillions dans le signal
nbFenetre = xLen / dn                        #  74       |  INT    |  Nombre de fenetres
windowOverflow = xLen - nbFenetre * dn + dn  #  2024     |  INT    |  Nombre d'echantillons restant a la fin
teta = 512                                   #  512      |  INT    |  Quefrence seuil empirique

print 'Tvoulu -> ', Tvoulu
print 'Nvoulu -> ', Nvoulu
print 'fe -> ', fe
print 'N  -> ', N
print 'T  -> ', T
print 'dt -> ', dt
print 'dn -> ', dn
print 'len -> ', xLen
print 'nbFenetre -> ', nbFenetre
print 'windowOverflow -> ', windowOverflow
print 'teta -> ', teta


# Fonction de hann
# Fenetre progressive
# q INT -> le numero d'echantillon dans la decoupe centree en 0    |    q E [-N/2, N/2]
# Return FLOAT
hann_cache = {}
def hann(q):
    if q not in hann_cache:
        hann_cache[q] = (1.0 + math.cos(2.0 * math.pi * q / N)) / 2.0
    return hann_cache[q]


# Fonction de decoupe
# m INT -> le numero de la decoupe
# q INT -> le numero d'echantillon dans la decoupe centree en 0    |    q E [-N/2, N/2]
# Return np.array: INT
def u(m, q): return x[m * dn + q] * hann(q)


# Fonction de reconstruction d'un echantillon a partir de la FFT
# m INT -> le numero de la decoupe
# q INT -> le numero d'echantillon dans la decoupe centree en 0    |    q E [-N/2, N/2]
# Return np.array: INT
def recon_u(m, q): return FFT_X(m)[q]


# Fonction de reconstruction d'un echantillon a partir des decoupes
# n INT -> le numero de l'echantillon
# Return np.array: INT
def recon_x(n):
    m = n / dn
    q = n - m * dn
    return  recon_u(m, q) + recon_u(m+1, q-dn)


# TF a court terme du signal x
# m INT -> le numero de la decoupe
# Return np.array: [FLOATj]
FFT_X_cache = {}
def FFT_X(m):
    if m not in FFT_X_cache:
        _u = np.ndarray(shape=(N))
        for q in range(-N/2, N/2): _u[q] = u(m, q-dn)
        FFT_X_cache[m] = _u
    return FFT_X_cache[m]


# TF a court terme du signal x
# m INT -> le numero de la decoupe
# k INT -> indice voulu dans la decoupe
# Return np.array: FLOATj
def X(m, k): return FFT_X(m)[k]




# TEST RECONSTRUCTION DE X DEPUIS LE DECOUPAGE PUIS LA FFT
_x = np.ndarray(shape=(xLen), dtype=np.complex128)
for n in range(0, xLen-windowOverflow): _x[n] = recon_x(n)
_x = np.around(_x.real, 0).astype('int16')
wavfile.write("test.wav", fe, _x)
print np.average(x - _x)
t = np.arange(0, x.shape[0], 1)
plt.subplot(211)
plt.plot(t, x)
plt.subplot(212)
plt.plot(t, _x)
plt.show()


