# cython: language_level=3


import numpy as np
cimport numpy as np

import cython
DTYPE = np.float
ctypedef np.float_t DTYPE_t

from libc.math cimport sqrt, exp, abs, sin, cos, M_PI

cdef extern from "gsl/gsl_sf_bessel.h":
    double gsl_sf_bessel_J1(double x)
         
def real_j1(double x):
     return gsl_sf_bessel_J1 (x)

def real_close2zero(double x):
    if abs(x) < 1e-6:
        return True
    else:
        return False

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mod2(np.ndarray[DTYPE_t, ndim=1] vec):
    cdef double v1 = vec[0]
    cdef double v2 = vec[1]
    return sqrt(v1*v1 + v2*v2)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef norm2(np.ndarray[DTYPE_t, ndim=1] vec):
    cdef double v1 = vec[0]
    cdef double v2 = vec[1]
    return v1*v1 + v2*v2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dot2(np.ndarray[DTYPE_t, ndim=1] vec1, np.ndarray[DTYPE_t, ndim=1] vec2):
    cdef double v1 = vec1[0]
    cdef double v2 = vec1[1]

    cdef double w1 = vec2[0]
    cdef double w2 = vec2[1]

    return v1*w1 + v2*w2


def real_exp(double r):
    return exp(r)

def phase_shift(complex r, double phi):
    cdef double c = cos(phi)
    cdef double s = sin(phi)

    return r*(c + 1.0j*s)

@cython.boundscheck(False)
@cython.wraparound(False)
def map_array(np.ndarray[DTYPE_t, ndim=1] T, func):
    cdef unsigned int N = T.shape[0], n = 0
    cdef np.ndarray R = np.zeros(N, dtype=np.complex)
    cdef complex[:] rR = R
    cdef double[:] rT = T
    for n in xrange(N):
        rR[n] = func(rT[n])
    return R

@cython.boundscheck(False)
@cython.wraparound(False)
def k_shift(np.ndarray[DTYPE_t] b1, np.ndarray[DTYPE_t, ndim=1] b2, double k, int l, double t):
    cdef np.ndarray K = np.zeros(3, dtype=DTYPE)
    cdef double t2 = (k/(2.0*M_PI) + l)
    K[0] = t2*b1[0] + t*b2[0]
    K[1] = t2*b1[1] + t*b2[1]
    return K

