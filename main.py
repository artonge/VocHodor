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
fe, x  = wavfile.read('photographe.wav') # Signal de la parole
fee, e = wavfile.read('chasse.wav') # Signal du son


Tvoulu = 0.05                                #  50ms     |  FLOAT  |  Duree theorique d'une fenetre
Nvoulu = Tvoulu*fe                           #  2400     |  INT    |  Nombre theorique d'echantillon dans une fenetre
N  = closestPowerOf2(Nvoulu)                 #  2048     |  INT    |  On aroundi a la puissance de 2 la plus proche pour la fft
T  = float(N) / float(fe)                    #  42.66ms  |  FLOAT  |  On decoupe le signal en fenetres de duree T = 85.3ms
dt = T / 2.0                                 #  21.33ms  |  FLOAT  |  Duree du d'avancement dans le signal lors de la decoupe
dn = N / 2                                   #  1024     |  INT    |  Pas d'avancement en echantillons
xLen   = len(x)                              #  76776    |  INT    |  Nombre d'echantillions dans le signal
nbFenetre = xLen / dn                        #  74       |  INT    |  Nombre de fenetres
windowOverflow = xLen - nbFenetre * dn + dn  #  2024     |  INT    |  Nombre d'echantillons restant a la fin
teta = 40                                    #  40       |  INT    |  Quefrence seuil empirique

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


#################################
#################################

hann = np.ndarray(shape=(N))
for q in range(N): hann[q] = (1.0 + math.cos(2.0 * math.pi * (q - N/2) / N)) / 2.0

#################################
#################################

u = np.ndarray(shape=(nbFenetre, N, 2))
for m in range(nbFenetre):
    for q in range(N): u[m, q] = x[m*dn + q - N/2] * hann[q]

#################################
#################################

X = np.ndarray(shape=(nbFenetre, N, 2), dtype=np.complex128)
for m in range(nbFenetre):
    for q in range(N): X[m] = fft2(u[m])

#################################
#################################

c = np.ndarray(shape=(nbFenetre, N, 2))
for m in range(nbFenetre):
    c[m] = ifft2(np.log(np.absolute(X[m]))).real

#################################
#################################

for m in range(nbFenetre):
    for j in range(N):
        if (j < 50 or N - 50 < j): c[m, j] = [0, 0]

#################################
#################################

C = np.ndarray(shape=(nbFenetre, N, 2), dtype=np.complex128)
for m in range(nbFenetre): C[m] = fft2(c[m])

#################################
#################################

H = np.ndarray(shape=(nbFenetre, N, 2), dtype=np.complex128)
for m in range(nbFenetre):
    for j in range(N): H[m, j] = np.exp(C[m, j])

#################################
#################################

p = np.ndarray(shape=(nbFenetre, N, 2))
for m in range(nbFenetre):
    for q in range(N): p[m, q] = e[m*dn + q - N/2] * hann[q]

#################################
#################################

E = np.ndarray(shape=(nbFenetre, N, 2), dtype=np.complex128)
for m in range(nbFenetre): E[m] = fft2(p[m])

#################################
#################################

V = np.ndarray(shape=(nbFenetre, N, 2), dtype=np.complex128)
for m in range(nbFenetre): V[m] = E[m] * H[m]

#################################
#################################

a = np.ndarray(shape=(nbFenetre, N, 2))
for m in range(nbFenetre): a[m] = ifft2(V[m]).real

#################################
#################################

v = np.ndarray(shape=(xLen, 2))
for m in range(nbFenetre-1):
    for q in range(N): v[m*dn + q-N/2] = a[m][q-N/2] + a[m+1][q-N/2-dn]

#################################
#################################

# # RESULTAT TP
v = np.around(v, 0).astype('int16')
wavfile.write("test.wav", fe, v)
t = np.arange(0, x.shape[0], 1)
plt.subplot(311)
plt.plot(t, x)
plt.subplot(312)
plt.plot(t, v)
plt.show()

#################################
#################################

# # DEBUG = REPLACE BY WANTED ARRAY
DEBUG = H # u, X, c, C, p, E, V, a, v...
plt.subplot(313)
plt.plot(t, DEBUG[30])
