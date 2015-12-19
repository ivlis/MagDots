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



import sys
sys.path.append("../../..")

import numpy as np
import scipy.constants as const

from matplotlib import pyplot as plt
from multiprocessing import Pool

mu0 = const.codata.value('mag. constant')
gamma = const.codata.value('electron gyromag. ratio')

from magdots.model.dot import Dot, DoubleDot
from magdots.model.dotarray import BulkWave, EdgeWave
from magdots.model.lattice import Lattice
from magdots.model.magn_lattice import FM_state

from magdots.util.util import calculate_edge

def main():

    np.set_printoptions(precision=3)

    R = 1.0
    a = 2.2*R
    Ms = 1.0
    h = 0.25*R

    # CAFM
    a1 = np.array([2.0*a, 0.0*a, 0.0])
    a2 = np.array([1.0*a, 1.0*a, 0.0])

    d = np.array([1.0*a, 0.0*a, 0.0])

    Ha = 2.0 * mu0 * Ms
    na = np.array([0,0,1])

    dot = DoubleDot(R=R, h=h, d=d, Ms=Ms, anisotropies=[(na, Ha), (na,Ha)])

    la = Lattice(a1=a1, a2=a2)

    mu = np.array([0.0,0.0,1.0, 0.0, 0.0,-1.0])
    magn_lattice = FM_state(mu)

    L_MAX = 30 
    N_MAX = 41

    edge = EdgeWave(dot, la, magn_lattice, L_MAX=L_MAX, N_MAX=N_MAX, B_comp=False)

    edge.self_vectors = True

    points = 4*2 # number of points for calculaion

    K = np.linspace(0, np.pi, num=points)

    result = calculate_edge(edge, K, points)

    result.save_to_file("edge_CAFM.npz")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(result.K, result.W/(mu0*gamma*Ms), linewidth = 1.2, color='k')

    plt.show()

if __name__ == "__main__":
    main()



