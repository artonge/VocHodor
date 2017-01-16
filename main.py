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
teta = 40                                   #  512      |  INT    |  Quefrence seuil empirique

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
# q INT -> le numero d'echantillon dans la decoupe centree en 0    |    q E [0, N]
# Return FLOAT
hann_cache = {}
def hann(q):
    if q not in hann_cache:
        hann_cache[q] = (1.0 + math.cos(2.0 * math.pi * (q - N/2) / N)) / 2.0
    return hann_cache[q]


# Fonction de decoupe
# m INT -> le numero de la decoupe
# q INT -> le numero d'echantillon dans la decoupe centree en 0    |    q E [-N/2, N/2]
# Return np.array: [INT INT]
def u(m, q): return x[m * dn + (q - N/2)] * hann(q)


# Fonction de reconstruction d'un echantillon a partir de la FFT
# m INT -> le numero de la decoupe
# k INT -> le numero d'echantillon dans la decoupe centree en 0    |    q E [-N/2, N/2]
# Return np.array: [INT INT]
recon_u_cache = {}
def recon_u(m, k):
    if m not in recon_u_cache: recon_u_cache[m] = ifft2(FFT_X(m))
    return recon_u_cache[m][k - N/2]


# Fonction de reconstruction d'un echantillon a partir des decoupes
# n INT -> le numero de l'echantillon [0 .. xLen]
# Return np.array: [INT INT]
def recon_x(n):
    m = n / dn
    k = n - m * dn
    return  recon_u(m, k) + recon_u(m+1, k-dn)


# TF a court terme du signal x
# m INT -> le numero de la decoupe [0 .. nbWindow]
# Return np.array: [[FLOATj FLOATj], ...]
FFT_X_cache = {}
def FFT_X(m):
    if m not in FFT_X_cache:
        _u = np.ndarray(shape=(N,2))
        for k in range(0, N): _u[k] = u(m, k-dn)
        FFT_X_cache[m] = fft2(_u)
    return FFT_X_cache[m]


# TF a court terme du signal x
# m INT -> le numero de la decoupe [0 .. nbWindow]
# k INT -> indice voulu dans la decoupe [0 .. N]
# Return np.array: [FLOATj FLOATj]
def X(m, k): return FFT_X(m)[k]


# Cepstre a court terme  du signal x
# m INT -> le numero de la decoupe [0 .. nbWindow]
# j INT -> le numero de la quefrence [0 .. N]
# Return np.array: [FLOAT FLOAT]
c_cache = {}
def c(m, j):
    if m not in c_cache:
        _X = FFT_X(m)
        moduleX = np.absolute(_X)
        logModuleX = np.log(moduleX)
        c_cache[m] = ifft2(logModuleX).real
    return c_cache[m][j]


# Cepstre prime ==> cepstre sans les hautes frequences
# Pour recuperer seulement les frequence du conduit vocal
# Si j > teta alors on retourne 0
# m INT -> le numero de la decoupe [0 .. nbWindow]
# j INT -> le numero de la quefrence [0 .. N]
# Return np.array: [FLOAT FLOAT]
def c_prime(m, j):
    if j < teta or N - teta < j: return [0, 0]
    else: return c(m, j).real


# TF de cepstre prime
# m INT -> le numero de la decoupe [0 .. nbWindow]
# Return np.array: [[FLOATj FLOATj], ...]
FFT_C_prime_cache = {}
def FFT_C_prime(m):
    if m not in FFT_C_prime_cache:
        _c = np.ndarray(shape=(N,2), dtype=np.complex128)
        for j in range(N): _c[j] = c_prime(m, j)
        FFT_C_prime_cache[m] = fft2(_c)
    return FFT_C_prime_cache[m]


# TF de cepstre prime
# m INT -> le numero de la decoupe [0 .. nbWindow]
# k INT -> indice voulu dans la decoupe [0 .. N]
# Return np.array: [FLOATj FLOATj]
def C_prime(m, k): return FFT_C_prime(m)[k]


# TF du conduit vocal
# m INT -> le numero de la decoupe [0 .. nbWindow]
# k INT -> indice voulu dans la decoupe [0 .. N]
# Return np.array: [FLOATj FLOATj]
def H(m, k): return np.exp(FFT_C_prime(m))[k]


# Fonction de decoupe
# m INT -> le numero de la decoupe [0 .. nbWindow]
# q INT -> le numero d'echantillon dans la decoupe centree en 0 [0 .. N]
# Return np.array: [INT INT]
def p(m, q): return e[m * dn + q - N/2] * hann(q)


# TF a court terme du signal e
# m INT -> le numero de la decoupe [0 .. nbWindow]
# Return np.array: [[FLOATj FLOATj], ...]
FFT_E_cache = {}
def FFT_E(m):
    if m not in FFT_E_cache:
        _u = np.ndarray(shape=(N,2))
        for q in range(N): _u[q] = p(m, q)
        FFT_E_cache[m] = fft2(_u)
    return FFT_E_cache[m]


# TF a court terme du signal e
# m INT -> le numero de la decoupe [0 .. nbWindow]
# k INT -> indice voulu dans la decoupe [0 .. N]
# Return np.array: [FLOATj FLOATj]
def E(m, k): return FFT_E(m)[k]


# TF de la nouvelle voix
# m INT -> le numero de la decoupe [0 .. nbWindow]
# k INT -> indice voulu dans la decoupe [0 .. N]
# Return np.array: [FLOATj FLOATj]
def V(m, k): return E(m, k) * H(m, k)


# Nouvelle voix
# m INT -> le numero de la decoupe [0 .. nbWindow]
# q INT -> le numero de la quefrence [0 .. N]
# Return np.array: [FLOAT FLOAT]
a_cache = {}
def a(m, q):
    if m not in a_cache:
        _V = np.ndarray(shape=(N,2), dtype=np.complex128)
        for k in range(0, N): _V[k] = V(m, k)
        a_cache[m] = ifft2(_V).real
    return a_cache[m][q]


# Reconstruction de la nouvelle voix
# n INT -> le numero de l'echantillon [0 .. xLen]
# Return np.array: [FLOAT FLOAT]
def recon_v(n):
    m = n / dn
    q = n - m * dn + N/2
    return a(m, q) + a(m+1, q-dn)



# TEST RECONSTRUCTION DE X DEPUIS LE DECOUPAGE PUIS LA FFT
# _x = np.ndarray(shape=(xLen,2), dtype=np.complex128)
# for n in range(0, xLen-windowOverflow): _x[n] = recon_x(n)
# _x = np.around(_x.real, 0).astype('int16')
# wavfile.write("test.wav", fe, _x)
# print np.average(x - _x)
# t = np.arange(0, x.shape[0], 1)
# plt.subplot(211)
# plt.plot(t, x[:, 1])
# plt.subplot(212)
# plt.plot(t, _x[:, 1])
# plt.show()



# AFFICHAGE CEPSTRE
# _c = np.ndarray(shape=(xLen,2))
# _c_prime = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     j = n - m * dn
#     _c[n] = c(m, j)
#     _c_prime[n] = c_prime(m, j)
# 
# _c = np.around(_c.real, 0).astype('int16')
# wavfile.write("test.wav", fe, _x)
# print np.average(x - _x)
# t = np.arange(0, x.shape[0], 1)
# plt.subplot(211)
# # plt.plot(t, ifft2(np.log(np.absolute(fft2(x)))))
# plt.plot(t[0:N], c_cache[30])
# plt.subplot(212)
# # plt.plot(t, ifft2(np.log(np.absolute(fft2(x)))))
# plt.plot(t[0:N], c_cache[30])
# plt.show()



# AFFICHAGE H
# _H = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     k = n - m * dn
#     _H[n] = H(m, k)
# # AFFICHAGE E
# _E = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     k = n - m * dn
#     _E[n] = E(m, k)
# # AFFICHAGE X
# _X = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     k = n - m * dn
#     _X[n] = X(m, k)
# # AFFICHAGE V
# _V = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     k = n - m * dn
#     _V[n] = V(m, k)
# AFFICHAGE C
# _C = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     k = n - m * dn
#     _C[n] = C_prime(m, k)
# # AFFICHAGE C
# _c = np.ndarray(shape=(xLen,2))
# for n in range(0, xLen-windowOverflow):
#     m = n / dn
#     k = n - m * dn
#     _c[n] = c(m, k)



# t = np.arange(0, FFT_E(30).shape[0], 1)
# plt.subplot(211)
# plt.plot(t, FFT_C_prime(30))
# plt.subplot(212)
# plt.plot(t, np.exp(FFT_C_prime(30)))
# plt.show()




# t = np.arange(0, x.shape[0], 1)
# plt.subplot(412)
# plt.plot(t, _E)
# plt.subplot(413)
# plt.plot(t, _X)
# plt.subplot(414)
# plt.plot(t, _V)
# plt.subplot(311)
# plt.plot(t, _C)
# plt.subplot(312)
# plt.plot(t, _H)
# plt.plot(t[0:N], c_cache[30])
# plt.subplot(313)
# plt.plot(t, _c)
# plt.plot(t[0:N], c_cache[30])
# plt.show()



# # TEST RESULTAT TP
_v = np.ndarray(shape=(xLen,2), dtype=np.complex128)
for n in range(0, xLen-windowOverflow): _v[n] = recon_v(n)
_v = np.around(_v.real, 0).astype('int16')
wavfile.write("test.wav", fe, _v)
# t = np.arange(0, x.shape[0], 1)
# plt.subplot(211)
# plt.plot(t, x)
# plt.subplot(212)
# plt.plot(t, _v)
# plt.show()
