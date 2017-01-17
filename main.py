from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.io import wavfile
import numpy as np
import math
import wave
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


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
tetaSeuil  = 4                                    #  4        |  INT    |  Quefrence seuil empirique

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
print 'tetaSeuil -> ', tetaSeuil


#################################
#################################
# FONCTION DE HANN
#################################
#################################

w = np.ndarray(shape=(N))
for q in range(N): w[q] = (1.0 + math.cos(2.0 * math.pi * (q - N/2) / N)) / 2.0


def hann(signal):
    chunks = np.ndarray(shape=(nbFenetre, N, 2))
    for m in range(nbFenetre):
        for q in range(N): chunks[m, q] = signal[m*dn + q - N/2] * w[q]

    return chunks

#################################
#################################
# CALCULE DE LA FFT A COURT TERM
#################################
#################################

def fftCourtTerm(chunk) :
    return fft2(chunk)

#################################
#################################
# CALCULE DU CEPSTRE
#################################
#################################

def cepstreCourtTerm(chunk) :
    chunk = np.absolute(chunk)
    chunk = np.log(chunk)
    chunk = ifft2(chunk)
    return chunk

#################################
#################################
# EXTRACTION REPONSE FREQUENTIELLE DU CONDUIT VOCAL
#################################
#################################

def repFreqConduitVocal(chunk):

    fft = fftCourtTerm(chunk)
    cepstre = cepstreCourtTerm(fft)

    # Filtrage de la Quefrence
    # On garde juste la reponse frequentielle de la glotte
    for j in range(N):
        if (j < tetaSeuil or N - tetaSeuil < j):
            cepstre[j] = [0, 0]

    # On repasse dans le domaine frequentielle
    fftCepstre = fft2(cepstre)
    fftCepstre = np.exp(fftCepstre)
    return fftCepstre

#################################
#################################
#################################
#################################

start = time.time()

resultH = Pool(6).map_async(repFreqConduitVocal, hann(x))
resultE = Pool(2).map_async(fftCourtTerm, hann(e))

H = resultH.get()
E = resultE.get()

print time.time() - start

#################################
#################################
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
# t = np.arange(0, x.shape[0], 1)
# plt.subplot(311)
# plt.plot(t, x)
# plt.subplot(312)
# plt.plot(t, v)
# plt.show()

#################################
#################################

# # DEBUG = REPLACE BY WANTED ARRAY
# print H[30], len(H[30])
# DEBUG = H # u, X, c, C, H, p, E, V, a, v...
# # t = np.arange(0, DEBUG[30], 1)
# plt.plot(DEBUG[30])
# plt.show()
