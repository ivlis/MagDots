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

from ..util.util import mod

def WignerSeitzCell(la, segments):
    G, X, M = la.GXM
    vertices = (G,X,M,G)
    return ContourIterator(vertices, segments)


class ContourIterator:
    def __init__(self, vertices, segments):
        self._vertices = vertices
        self._segments = segments

    def __iter__(self):
        v = self._vertices
        acc = 0.0
        for A, B in zip(v, v[1:]):
            for k, acc in self._advance(A, B, acc):
                yield k, acc
        G = v[-1]
        yield G, acc + mod(k-G)

    def _advance(self, P1, P2, acc):
        p = self._segments
        step = (P2 - P1)/(p-1)
        m = mod(step)
        for i in range(0, p):
            yield P1 + step*i, acc
            acc += m


class Lattice:
    def __init__(self, a1, a2):
        ez = np.array([0,0,1])
        S0 = np.abs(np.cross(a1, a2).dot(ez))
        self._b1 = -2.0*np.pi/S0*np.cross(ez, a2)
        self._b2 =  2.0*np.pi/S0*np.cross(ez, a1)
        self._a1 = a1
        self._a2 = a2
        self._S0 = S0

    def _calcGXM(self):

        b1 = self._b1

        b21 = self._b2
        b22 = self._b1 + self._b2

        
        if mod(b1-b21) < mod(b1-b22):
            b2 = b21
        else:
            b2 = b22
        
        X1 = b1/2.0
        X2 = b2/2.0
        
        A1 = b1[0]
        B1 = b1[1]

        A2 = b2[0]
        B2 = b2[1]

        C1 = -A1*X1[0] - B1*X1[1]
        C2 = -A2*X2[0] - B2*X2[1]

        Mx = (B1*C2 - B2*C1) / (A1*B2 - A2*B1)
        My = (C1*A2 - C2*A1) / (A1*B2 - A2*B1)

        self._M = np.array([Mx, My, 0.0])
        self._X = X1
        self._G = np.array([0,0,0])

    @property
    def GXM(self):
        try:
            G = self._G
        except AttributeError:
            self._calcGXM()
            G = self._G
        return (G, self._X, self._M)

    @property
    def b1(self):
        return self._b1

    @property
    def b2(self):
        return self._b2

    @property
    def S0(self):
        return self._S0

    @property
    def a1(self):
        return self._a1

    @property
    def a2(self):
        return self._a2

