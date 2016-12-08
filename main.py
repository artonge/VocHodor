from scipy.fftpack import fft
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
(fe, x) = wavfile.read('photographe.wav') # Signal de la parole
e = wavfile.read('chasse.wav') # Signal du son

Tvoulu = 0.05  #  50ms              | 20ms(Duree d'un phoneme) < Tvoulu < 125ms(Frequence la plus basse de la voix d'un homme)
Nvoulu = Tvoulu*fe   #  7200        | Nombre d'echantillon dans Tvoulu
N  = closestPowerOf2(Nvoulu) # 2048 | Nombre d'ecahntillons dans une fenetre. On en garde seulement une puissance de 2 pour la fft
T  = float(N) / float(fe) #  85.3ms | Duree d'une fenetre. On decoupe le signal en bouts de duree T = 85.3ms
dt = T / 2.0  #  42.65ms            | Pas d'avancement dans le signal lors de la decoupe
dn = N / 2    #  1024               | Pas d'avancement en echantillons


print 'Tvoulu -> ', Tvoulu
print 'Nvoulu -> ', Nvoulu
print 'fe -> ', fe
print 'N  -> ', N
print 'T  -> ', T
print 'dt -> ', dt
print 'dn -> ', dn


# Fonction de hann
# Fenetre progressive
# q INT -> qeme echantillon dans un decoupage    |    q E [-N/2, N/2]
# Return FLOAT
hann_cache = {}
def hann(q):
    if q not in hann_cache:
        hann_cache[q] = (1.0 + math.cos(2.0 * math.pi * q / N)) / 2.0
    return hann_cache[q]


# Fonction de decoupe
# m INT -> le numero de la decoupe
# q INT -> le numero d'echantillon dans la decoupe
# Return [INT INT]
def u(m, q): return x[m * dn + q] * hann(q)


# Fonction de reconstruction d'un echantillon
# n INT -> le numero de l'echantillon
# Return [INT INT]
def recon_x(n):
    m = n / dn
    q = n - m * dn
    return u(m, q) + u(m+1, q-dn)


# TF du signal x
# m INT -> le numero de la decoupe
# k INT -> indice voulu fans la Transform
# Return [INT INT]
def X(m, k):
    _u = []
    for q in range(0, N): _u.append(u(m, q-(N/2)))
    return fft(_u)[k]


# TF du signal x
# m INT -> le numero de la decoupe
# k INT -> indice voulu fans la Transform
# Return [[INT INT]]
def FFT_X(m):
    print m
    _u = []
    for q in range(0, N): _u.append(u(m, q-(N/2)))
    return fft(_u)


m = 20
k = 1
X(m, k);



































# TEST RECONSTRUCTION DE X -- OK -- CA MARCHE BUENO
# _x = []
# for i in range(0, len(x)): _x.append(recon_x(i))
# _x = np.array(_x)
# signalLength = x.shape[0]
# t = np.arange(0, signalLength, 1)
# plt.figure(1)
# plt.subplot(211)
# plt.plot(t, x)
# plt.subplot(212)
# plt.plot(t, _x)
# plt.show()
#################################################
#################################################
# TEST FFT DES BOUT DE X -- OK -- CA MARCHE BUENO
# _X = 0
# for i in range(0, len(x)/N):
#     if i == 0: _X = FFT_X(i)
#     else: _X = np.append(_X, FFT_X(i))
# 
# signalLength = _X.shape[0]
# t = np.arange(0, signalLength, 1)
# plt.figure(1)
# plt.plot(t, _X)
# plt.show()
