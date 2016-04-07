#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import scipy.constants as const
mu0 = const.codata.value('mag. constant')

from nose2.tools import params

from ..model.dot import Dot, DoubleDot
from ..model.dotarray import EdgeWave
from ..model.lattice import Lattice
from ..model.magn_lattice import FM_state


R = 1.0
p = dict(R = R,
              h = 30.0*R,
              Ms = 1.0
              )
cylindricalDot = Dot(**p)
p = dict(R = R,
              h = 30.0*R,
              Ms = 1.0,
              d = np.array([3.0*R,0,0])
              )
doubleDot = DoubleDot(**p)

a1 = np.array([3.0, 0.0, 0.0])
a2 = np.array([0.0, 3.0, 0.0])
square_la = Lattice(a1, a2)


a1 = np.array([3.0, 0.0, 0.0])
a2 = np.array([0.0, 6.0, 0.0])
rectangular_la = Lattice(a1, a2)

a1 = np.array([3.0, 0.0, 0.0])
a2 = 3.0*np.array([0.5, np.sqrt(3.0)/2.0, 0.0])
triangular_la = Lattice(a1, a2)

matrix = [(dot, la) for dot in [cylindricalDot, doubleDot]
                    for la in [square_la, rectangular_la, triangular_la]]

@params(*matrix)
def test_E00_trace(dot, la):
    mu = np.array([0.0,0.0,1.0])
    magn_lattice = FM_state(mu)

    L_MAX = 60
    N_MAX = 60

    edge = EdgeWave(dot, la, magn_lattice, L_MAX = L_MAX, N_MAX = N_MAX)

    Ek0 = edge._Ek(0)
    cell_dim = dot.number_of_elements
    Ek00 = Ek0[N_MAX]
    Ek02 = Ek0[N_MAX+2]
    Ek0m2 = Ek0[N_MAX-2]
    npt.assert_allclose(np.trace(Ek00), 1*cell_dim, atol=1e-1)
    npt.assert_allclose(np.trace(Ek02), 0, atol=1e-2)
    npt.assert_allclose(Ek02, np.conj(Ek0m2).T, atol=1e-3)

# @params(*matrix)
# def test_Fk(dot, la):
#     mu = None
#     bulk = BulkWave(dot, la, mu, K_MAX = 100)
#     k = np.array([0.3, 0.6, 0])
#     Fk = bulk.Fk(k)
#     Fminusk = bulk.Fk(-k)
#     cell_dim = dot.number_of_elements
#     npt.assert_allclose(np.trace(Fk), 1*cell_dim, atol=1e-2)
#     npt.assert_allclose(Fk, np.conj(Fk).T, atol=1e-2)
#     npt.assert_allclose(Fk, Fminusk.T, atol=1e-2)
