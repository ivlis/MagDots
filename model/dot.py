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



import functools
import numpy as np
import scipy.constants as const
mu0 = const.codata.value('mag. constant')
import scipy.special as sf

from ..util.util import set_limit_value, make_block_diag
from ..util_pyx.util import mod2, norm2, dot2, real_exp, real_j1, real_close2zero, phase_shift

def construct_Nk(dot, fn):
    elements = dot.get_elements()
    vals = [fn(e) for e in elements]
    Nk = dot.elements_to_matrix(vals) + dot.K
    return Nk

class Dot:
    """Implement demag tensor calculation for single dot"""
    def __init__(self, R, h, Ms, anisotropy=None):
        self._R = R
        self._h = h
        self._Ms = Ms
        self._S = np.pi*R*R

        if anisotropy is None:
            self._na = None
        else:
            (na, Ha) = anisotropy
            self._na = na
            self._Ha = Ha

    @property
    def number_of_elements(self):
        return 1

    def get_elements(self):
        return [self.N11, self.N22, self.N12, self.N33]

    @staticmethod
    def elements_to_matrix(e):
        Nk = np.zeros((3,3), dtype=complex)
        Nk[0][0] = e[0]
        Nk[1][1] = e[1]
        Nk[1][0] = e[2]
        Nk[0][1] = e[2]
        Nk[2][2] = e[3]
        return Nk

    @property
    def R(self):
        return self._R

    @property
    def h(self):
        return self._h

    @property
    def S(self):
        return self._S

    @property
    def Ms(self):
        return self._Ms

    @property
    def K(self):
        na = self._na
        if na is None:
            return np.zeros((3,3), dtype=complex)
        else:
            Ha = self._Ha
            return -Ha*np.outer(na,na)/(2.0*mu0*self._Ms)

    def _fkh(self, kmod):
        if real_close2zero(kmod):
            return 0.0
        else:
            kh = kmod*self._h
            return 1.0 - (1.0 - real_exp(-kh))/kh

    def _sigma(self, k):
        kr = mod2(k)*self.R
        if real_close2zero(kr):
            return self.S
        else:
            return self.S * 2.0*real_j1(kr)/kr


    def _nij_helper(fn):
        @functools.wraps(fn)
        def wrapper(self, k):
            modk = mod2(k)
            kr = modk*self.R
            fkh = self._fkh(modk)
            sigma2S = self._sigma(k)**2/self.S
            return sigma2S * fn(self, k, kr, fkh)
        return wrapper

    @_nij_helper
    def N11(self, k, kr, fkh):
        if real_close2zero(kr):
            return 0
        else:
            return k[0]*k[0]/norm2(k)*fkh

    @_nij_helper
    def N12(self, k, kr, fkh):
        if real_close2zero(kr):
            return 0
        else:
            return k[0]*k[1]/norm2(k)*fkh

    @_nij_helper
    def N22(self, k, kr, fkh):
        if real_close2zero(kr):
            return 0
        else:
            return k[1]*k[1]/norm2(k)*fkh

    @_nij_helper
    def N33(self, k, kr, fkh):
        return 1.0 - fkh


class DoubleDot(Dot):
    def __init__(self, R, h, Ms, d, anisotropies=None):
        super().__init__(R=R, h=h, Ms=Ms)
        self._d = d
        self._anisotropies = anisotropies

    @staticmethod
    def _shift(fun, d):
        """Adds phase shift factor to function.

        :fun: function to  altered. Should have signature as fun(k)
        :d: shift vector
        :returns: fun(k)*exp(ikr)

        """
        #@profile
        def shifted(k):
            kd = dot2(k,d)
            f = fun(k)
            res = phase_shift(f, kd)
            return res
        return shifted

    @property
    def number_of_elements(self):
        return 2

    def get_elements(self):
        """@todo: Docstring for get_elements.
        :returns: @todo

        """
        single_dot = super().get_elements()
        d = self._d
        elements = single_dot
        elements.extend([self._shift(f, -d) for f in single_dot])
        elements.extend([self._shift(f, d) for f in single_dot])
        return elements

    @classmethod
    def elements_to_matrix(cls, e):
        Nk = np.zeros((6,6), dtype=complex)
        Nk11 = super().elements_to_matrix(e[0:4])
        Nk12 = super().elements_to_matrix(e[4:8])
        Nk21 = super().elements_to_matrix(e[8:12])
        Nk[0:3,0:3] = Nk11
        Nk[3:6,3:6] = Nk11
        Nk[0:3,3:6] = Nk12
        Nk[3:6,0:3] = Nk21
        return Nk

    @property
    def K(self):
        ans = self._anisotropies
        if ans is None:
            K = np.zeros((6,6), dtype=complex)
        else:
            K = [-Ha*np.outer(na,na)/(2.0*mu0*self._Ms)  for (na, Ha) in ans]
            K = make_block_diag(K)
        return K


class InfiniteDot(Dot):
    def __init__(self, R, Ms):
        self._R = R
        self._Ms = Ms
        self._S = np.pi*R*R

    @set_limit_value(1,0.0,0.0)
    def _fkh(self, kmod):
        return 1.0
