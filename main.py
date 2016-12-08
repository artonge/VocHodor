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
N  = closestPowerOf2(Nvoulu) # 2048 | On en garde seulement une puissance de 2 pour la fft
T  = float(N) / float(fe) #  85.3ms | On decoupe le signal en bouts de duree T = 85.3ms
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
# q -> qeme echantillon dans un decoupage    |    q E [-N/2, N/2]
# Return FLOAT
def hann(q): return (1.0 + math.cos(2.0 * math.pi * q / N)) / 2.0


# Fonction de decoupe
# m -> le numero de la decoupe
# q -> le numero d'echantillon dans la decoupe
# Return INT
def u(m, q):
    if m * dn + q > len(x)-1: print m, dn, q
    return x[m * dn + q] * hann(q)

# Fonction de reconstruction d'un echantillon
# n -> le numero de l'echantillon
# Return INT
def recon_x(n):
    m = n / dn
    q = n - m * dn
    return u(m, q) + u(m+1, q-dn)


# TF du signal x
# m ->
# k -> indice voulu fans la Transform
def X(m, k):
    _u = []
    for q in range(0, N): _u.append(u(m, q-(N/2)))
    return fft(_u)[k]


def FFT_X(m):
    _u = []
    for q in range(0, N): _u.append(u(m, q-(N/2)))
    return fft(_u)


m = 20
k = 1
X(m, k);

_X = 0
for i in range(0, len(x)):
    if _X == 0: X = FFT_X(i/dn)
    else: _X += FFT_X(i/dn)
# _X = np.array(_X)
# signalLength = _X.shape[0]
# t = np.arange(0, signalLength, 1)
# plt.figure(1)
# plt.plot(t, _X)
# plt.show()

































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
