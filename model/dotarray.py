#!/usr/bin/env python3

'''
    This file is part of MagDots.

    (C) Ivan Lisenkov
    Oakland Univerity,
    Michigan, USA

    2015

    MagDots: Magnetization dynamics of nanoelement arrays

    How to cite.
    If you are using this program or/and produce scientific publications based on it,
    we kindly ask you to cite it as:

    "
    Ivan Lisenkov, Vasyl Tyberkevych, Sergey Nikitov, Andrei Slavin
    Theoretical formalism for collective spin-wave edge excitations in arrays of dipolarly interacting magnetic nanodots
    arXiv:1511.08483
    http://arxiv.org/abs/1511.08483
    "

    MagDots is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MagDots is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MagDots.  If not, see <http://www.gnu.org/licenses/>.
'''


import itertools
import numpy as np
import scipy.constants as const
import scipy.linalg as LA
import scipy.special as sf
import scipy.optimize as OPT

from functools import wraps
from multiprocessing import Pool
from numpy.fft import fft, fftshift

from .dot import Dot
from .lattice import Lattice
from ..util.util import norm, mod, mat_cross, complex_quad, make_block_diag
from ..util_pyx.util import map_array, k_shift

mu0 = const.codata.value('mag. constant')
gamma = const.codata.value('electron gyromag. ratio')

class ArrayUnstable(Exception):
    pass

class ArrayModesHaveImaginaryFrequencies(ArrayUnstable):
    pass

class ArrayModesHaveNegativeNorms(ArrayUnstable):
    pass

def magn_norm(m, J):
    mc = np.conj(m)
    A = -np.imag((J.dot(m)).dot(mc))
    return A


class Wave:
    self_vectors = True
    check_stability = True

    @staticmethod
    def __sort(w, v):
        w = -np.imag(w)
        idx = np.argsort(w)
        w = w[idx]
        v = v[:, idx]
        return w,v

    def omega(self, k):
        #if norm(k) < 1e-6:
            #Fk = self.F0
        #else:
        Fk = self.Fk(k)

        Omega = self.B*gamma  + gamma*mu0*self._dot.Ms*Fk

        J = self.J

        T = J.dot(Omega)
        W, V = LA.eig(T)

        if self.check_stability:
            for w in W:
                if abs(np.imag(w)) < abs(np.real(w)) and abs(w)/(gamma*mu0)>1e-8:
                    raise ArrayModesHaveImaginaryFrequencies()
            for i, w in enumerate(W):
                m = V[:, i]
                A = magn_norm(m, J)
                if np.imag(w)<0 and A<0 and abs(w)/(gamma*mu0)>1e-8:
                    raise ArrayModesHaveNegativeNorms()

        if self.self_vectors:
            W, V = self.__sort(W, V)
            return W, V
        else:
            return -np.imag(W)

    @property
    def B(self):
        try:
            B = self._B
        except AttributeError:
            B = self._B = self._calculate_B()
        return B

    @property
    def J(self):
        try:
            mu = self._J
        except AttributeError:
            mu = self._J = self._get_J()
        return mu

class BulkWave(Wave):
    K_MAX = 40

    def __init__(self, dot, lattice, mu, K_MAX = None, Bbias = None):
        if K_MAX:
            self.K_MAX = K_MAX

        self._dot = dot
        self._la = lattice
        self._mu = mu

        if Bbias is None:
            self._Bbias = np.zeros_like(mu)
        else:
            assert(Bbias.shape == mu.shape)
            self._Bbias = Bbias


    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value
        self._J = self._get_J()

    @property
    def Bbias(self):
        return self._Bbias

    @Bbias.setter
    def Bbias(self, Bbias):
        assert(Bbias.shape == self.mu.shape)
        self._Bbias = Bbias
        self._B = self._calculate_B()

    def Fk(self, k):
        return self._Fk(k) + self._dot.K

    @property
    def F0(self):
        try:
            F0 = self.__F0_value
        except AttributeError:
            G = np.zeros(3)
            F0 = self.__F0_value = self.Fk(G)
        return F0

    def _get_J(self):
        return mat_cross(self._mu)

    def _calculate_B(self):
        dot = self._dot
        mu = self.mu
        k0 = np.zeros(3, dtype=float)
        F0 = self.F0
        Bbias = self._Bbias

        N = len(mu)//3

        def equilibrium_eq(mu_B):
            val = np.zeros(4*N)
            l_mu = mu_B[0:3*N]
            l_B = mu_B[3*N:4*N]
            F0_mu = np.real(F0.dot(l_mu))
            B_mu = np.zeros(3*N)

            for n in range(N):
                B_mu[n*3:(n+1)*3] = l_B[n]*l_mu[n*3:(n+1)*3]

            val[0:3*N] = B_mu - Bbias + F0_mu

            for n in range(N):
                val[3*N+n] = norm(l_mu[3*n:3*(n+1)]) - 1.0

            return val

        mu_B0 = np.zeros(4*N)
        mu_B0[0:3*N] = mu
        mu_B0[3*N:4*N] = -0

        sol = OPT.root(equilibrium_eq, mu_B0, method='lm')
        if not sol.success:
            raise RuntimeError("Equlibrium condition cannot be found")

        self.mu = sol.x[0:3*N]

        B = make_block_diag([sol.x[3*N + n ]*np.identity(3) for n in range(N)])

        return mu0*dot.Ms*B

    #@profile
    def _Fk(self, k):
        dot = self._dot
        Nfuncs = dot.get_elements()
        F = np.zeros(len(Nfuncs), dtype=complex)
        for n in range(-self.K_MAX, self.K_MAX):
            for m in range(-self.K_MAX, self.K_MAX):
                K = n*self._la.b1 + m*self._la.b2 + k
                for i,f in enumerate(Nfuncs):
                    F[i] += f(K)
        Fk = dot.elements_to_matrix(F)
        return 1.0/self._la.S0 * Fk


class EdgeWave(BulkWave):
    N_MAX = 32
    L_MAX = 32
    def __init__(self, dot, lattice, mu, B_comp = False, B_comp_middle = False, K_MAX = None,
            N_MAX=None, L_MAX=None, Bbias = None):
        super().__init__(dot, lattice, mu = None, K_MAX = K_MAX, Bbias = Bbias)
        if N_MAX:
            self.N_MAX = N_MAX
        if L_MAX:
            self.L_MAX = L_MAX
        self._B_comp = B_comp
        self._B_comp_middle = B_comp_middle
        self._mu = self.__constr_magn_lattice(mu, self.N_MAX)

    @staticmethod
    def __constr_magn_lattice(ma_la, N_MAX):
        unrolled = [mu for i,mu in zip(range(0, N_MAX+1), ma_la)]
        return unrolled


    def _calculate_B(self):
        N_MAX = self.N_MAX
        dot = self._dot
        mu_lat = self.mu
        if not self._B_comp:
            B00 = np.zeros(mu_lat[0].shape, dtype=complex)
            Eks = np.array([ -mu0*dot.Ms*Ek for Ek in self._Ek(0)])
            for Ek, mu in zip(Eks[N_MAX:2*N_MAX+1], mu_lat):
                B00 += Ek.dot(mu)
            B0 = [np.copy(B00)]
            for n in range(1,N_MAX+1):
                B00 = np.zeros(mu_lat[0].shape, dtype=complex)
                for Ek, mu in zip(Eks[N_MAX-n:2*N_MAX+1-n], mu_lat):
                    B00 += Ek.dot(mu)
                B0.append(np.copy(B00))
            B0a = np.array(B0)
            B0_flat = []
            for Bmu, mu in zip(B0, mu_lat):
                B = np.identity(len(mu), dtype=complex)
                for i in range(0, len(mu)//3):
                    local_mu = mu[3*i:3*(i+1)]
                    local_Bmu = Bmu[3*i:3*(i+1)]
                    nonzero_dir = np.argmax(np.abs(local_mu))     # We assume that magnetization of each dot
                    local_Beff = local_Bmu[nonzero_dir]/local_mu[nonzero_dir]           # is along one axis
                    B[3*i:3*(i+1), 3*i:3*(i+1)] = np.identity(3)*(local_Beff + self._Bbias)
                B0_flat.append(B)
            B0 = make_block_diag(B0_flat)
        else:
            B_inf = super()._calculate_B()
            B0 = make_block_diag(tuple(itertools.repeat(B_inf,self.N_MAX+1)))
        return B0

    def _get_J(self):
        Js = [mat_cross(mu) for mu in self.mu]
        return make_block_diag(Js)

    #@profile
    def Fk(self, k):
        Eks = self._Ek(k)
        Eks = _make_toeplitz_from_blocks(Eks)
        return Eks

    def _Ek(self, k):
        dot = self._dot
        N_funcs = dot.get_elements()

        E_comps = [self._Ek_component(k, func) for func in N_funcs]

        K = dot.K

        Eks = [dot.elements_to_matrix(Ek) for Ek in zip(*E_comps)]

        Eks[self.N_MAX] += K

        return Eks

    def _Ek_component(self, k, n_func):

        b1 = self._la.b1
        b2 = self._la.b2
        #@profile
        def func(l, t):
            K = k_shift(b1, b2, k, l, t)# (k/(2*np.pi) + l)*b1 + t*b2
            return n_func(K)

        Ek33 = self._calculate_sum(func)

        Ek33 /= self._la.S0
        return Ek33

    #@profile
    def _calculate_sum(self, func):
        N_MAX = self.N_MAX

        N = 8*1024
        NK = 512
        tick = 1.0/NK

        T = np.linspace(-tick*N, tick*N, num = N, endpoint=False)

        L_MAX = self.L_MAX
        S = []

        L_MAX_FFT = 2

        for l in range(-L_MAX_FFT, L_MAX_FFT+1):
            S_fft = self._calculate_fft(lambda t: func(l,t), T, tick, N, NK)
            S_fft[N_MAX] = complex_quad(lambda t: func(l,t), -np.inf, np.inf, limit=1000, epsabs=1e-5)[0]
            S.append(S_fft)

        for l in range(-L_MAX - L_MAX_FFT, -L_MAX_FFT):
            S_fft = np.zeros(2*N_MAX+1, dtype=complex)
            S_fft[N_MAX] = complex_quad(lambda t: func(l,t), -np.inf, np.inf, limit=1000, epsabs=1e-5)[0]
            S.append(S_fft)

        for l in range(L_MAX_FFT+1, L_MAX+1):
            S_fft = np.zeros(2*N_MAX+1, dtype=complex)
            S_fft[N_MAX] = complex_quad(lambda t: func(l,t), -np.inf, np.inf, limit=1000, epsabs=1e-5)[0]
            S.append(S_fft)

        N = np.zeros(2*N_MAX+1, dtype=complex)
        for s in S:
            N += s
        return N

    #@profile
    def _calculate_fft(self, func, T, tick, N, NK):
        N_MAX = self.N_MAX

        F = map_array(T, func)
        FFT = fft(F)*tick*2

        W = np.fft.fftfreq(N, d=tick)
        FFT = np.exp(-1.0j*np.pi*W*tick*(N))*FFT
        FFT = np.fft.fftshift(FFT)

        s_fft = FFT[N//2-2*N_MAX*N//NK:N//2+2*(N_MAX+1)*N//NK:2*N//NK]

        return s_fft

    def _calculate_integral(self, func, n):
        def integrant_cos(alpha):
            return func(alpha)*np.cos(-2*n*np.pi*alpha)

        I = complex_quad(integrant_cos, -np.inf, np.inf, limit=1000, epsabs=1e-5)[0]
        return I

def _make_toeplitz_from_blocks(B):
    assert(len(B)%2 == 1)
    t = int(len(B)/2 + 1)
    T = np.vstack([np.hstack(B[i:i+t]) for i in range(t-1, -1, -1)])
    return T
