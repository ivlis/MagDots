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

mu0 = const.codata.value('mag. constant')
gamma = const.codata.value('electron gyromag. ratio')

from magdots.model.dot import Dot, DoubleDot
from magdots.model.lattice import Lattice, ContourIterator 
from magdots.model.dotarray import BulkWave

from magdots.util.util import calculate_bulk

def main():

    np.set_printoptions(precision=3)

    # Setting array's physical parameters

    R = 1.0
    a = 4.1*R
    a1 = np.array([1.0*a, 0.0*a, 0.0])
    a2 = np.array([0.0*a, 1.0*a, 0.0])
    Ms = 1.0
    h = 5.0*R

    # Setting array's magnetic parameters

    mu = np.array([0.0,0.0,1.0])


    dot = Dot(R=R, h=h, Ms=Ms)
    la = Lattice(a1=a1, a2=a2)
    bulk = BulkWave(dot = dot, lattice = la, mu=mu,  K_MAX=30)


    # Setting points on the Brillouin zone to calculate
    points = 30
    G,X,M = la.GXM
    vertices = [G, X, M, G]

    K_Kmod = list(ContourIterator(vertices, points))
    K, K_mod = zip(*K_Kmod)

    #Perform calculations
    result = calculate_bulk(bulk, K, K_mod, points)

    result.save_to_file("bulk_FM.npz")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(result.K_mod, result.W/(gamma * mu0 *Ms), linewidth = 1.2, color='k')

    ax.set_xticks([result.K_mod[n*points] for n in range(0, len(vertices))])
    ax.set_xticklabels(["$\Gamma$", "$X$", "$M$","$\Gamma$"])
    ax.set_xlim(0, result.K_mod[-1])



    plt.show()


if __name__ == "__main__":
    main()

