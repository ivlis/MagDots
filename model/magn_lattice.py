#!/usr/bin/env python3
# encoding: utf-8

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

class FM_state:
    def __init__(self, mu):
        self._mu = mu

    def __iter__(self):
        return self

    def __next__(self):
        return self._mu

class AFM_state:
    def __init__(self, mu):
        self._mu = mu

    def __iter__(self):
        self.i = 1
        return self

    def __next__(self):

        if self.i == 1:
            self.i = -1
        else:
            self.i = 1

        return self.i*self._mu


