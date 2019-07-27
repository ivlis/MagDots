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

import numpy as np
import scipy.integrate as intgr
import scipy.linalg as LA
import scipy.constants as const

from functools import wraps

from multiprocessing import Pool

mu0 = const.codata.value('mag. constant')
gamma = const.codata.value('electron gyromag. ratio')


def norm(vec):
    return vec.dot(vec)

def mod(vec):
    return np.sqrt(norm(vec))

def make_block_diag(B):
    return LA.block_diag(*B)

def set_limit_value(arg, arg_value, limit):
    def decorator(fn):
        #@wraps(fn)                   # this makes program slower
        def wrapper(*args, **kwargs):
            if arg in kwargs:
                value = kwargs[arg]
            else:
                value = args[arg]
            if arg_value == value:
                return limit
            else:
                return fn(*args, **kwargs)
        return wrapper
    return decorator

def mat_cross(a):
    L = len(a)
    M = np.zeros((L,L), dtype=float)
    for l in range(0,L//3):
        la = a[3*l:3*(l+1)]
        M[3*l:3*(l+1), 3*l:3*(l+1)] = np.array([
                [    0, -la[2],  la[1]],
                [ la[2],     0, -la[0]],
                [-la[1],  la[0],     0]
                ])
    return M

def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = intgr.quad(real_func, a, b, **kwargs)
    imag_integral = intgr.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1.0j*imag_integral[0], real_integral[1:], imag_integral[1:])


def rotate(v, a):
    M = np.array([[ np.cos(a), -np.sin(a), 0],
                  [ np.sin(a),  np.cos(a), 0],
                  [         0,          0, 1]])

    return M.dot(v)


def Rx(a):
    Rx = np.array([[1,         0,          0]
                  [ 0, np.cos(a), -np.sin(a)],
                  [ 0, np.sin(a),  np.cos(a)]])
    return Rx


def Ry(a):
    Ry = np.array([[ np.cos(a), 0, np.sin(a)],
                   [         0, 1,         0],
                   [-np.sin(a), 0, np.cos(a)]])
    return Ry


def Rz(a):
    Rz = np.array([[ np.cos(a), -np.sin(a), 0],
                  [ np.sin(a),  np.cos(a), 0],
                  [         0,          0, 1]])
    return Rz


def parallel_calculation(wave, K):

    B = wave.B # Precalculate B

    with Pool() as pool:
        W_V = pool.map(wave.omega, K)

    W, V = zip(*W_V)
    W = np.array(W)
    V = np.array(V)

    return W,V


class BulkData:
    def __init__(self, W,  V, K, K_mod, J, points, B):
        self.W = W
        self.V = V
        self.K_mod = K_mod
        self.K = K
        self.J = J
        self.B = B
        self.points = points

    @classmethod
    def load_from_file(cls, filename):
        data = np.load(filename)

        K = data['K']
        K_mod = data['K_mod']
        W = data['W']
        V = data['V']
        J = data['J']
        B = data['B']
        points = data['points']

        return cls(W = W, K = K, V = V, K_mod = K_mod, J = J, points = points, B = B)


    def save_to_file(self, filename):

        np.savez(filename, K = self.K, K_mod = self.K_mod, W = self.W, points = self.points, V =
                self.V, J = self.J, B = self.B)


class EdgeData:
    def __init__(self, W,  V, K, J, points, B):
        self.W = W
        self.V = V
        self.K = K
        self.J = J
        self.B = B
        self.points = points

    @classmethod
    def load_from_file(cls, filename):
        data = np.load(filename)

        K = data['K']
        W = data['W']
        V = data['V']
        J = data['J']
        B = data['B']
        points = data['points']

        return cls(W = W, K = K, V = V, J = J, points = points, B = B)


    def save_to_file(self, filename):

        np.savez(filename, K = self.K, W = self.W, points = self.points, V =
                self.V, J = self.J, B = self.B)




def calculate_bulk(bulk, K, K_mod, points, parallel=True):


    #Perform calculations
    if parallel:
        W, V  = parallel_calculation(bulk, K)
    else:
        W_V = list(map(bulk.omega, K))
        W, V = zip(*W_V)
        W = np.array(W)
        V = np.array(V)


    K = np.array(K)

    J = bulk.J
    B = bulk.B

    return  BulkData(W=W, V=V, K=K, K_mod = K_mod, J = J, points = points, B = B)


def calculate_edge(edge, K, points):


    #Perform calculations
    W, V  = parallel_calculation(edge, K)

    J = edge.J
    B = edge.B

    return  EdgeData(W=W, V=V, K=K, J = J, points = points, B = B)
