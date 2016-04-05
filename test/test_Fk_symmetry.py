#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import scipy.constants as const
mu0 = const.codata.value('mag. constant')

from nose2.tools import params

from ..model.dot import Dot, DoubleDot
from ..model.dotarray import BulkWave
from ..model.lattice import Lattice


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
def test_F0_trace(dot, la):
    mu = None
    bulk = BulkWave(dot, la, mu, K_MAX = 100)
    F0 = bulk.F0
    cell_dim = dot.number_of_elements
    npt.assert_allclose(np.trace(F0), 1*cell_dim, atol=1e-2)
    npt.assert_allclose(np.imag(F0), np.zeros((3*cell_dim, 3*cell_dim)), atol=1e-3)

@params(*matrix)
def test_Fk(dot, la):
    mu = None
    bulk = BulkWave(dot, la, mu, K_MAX = 100)
    k = np.array([0.3, 0.6, 0])
    Fk = bulk.Fk(k)
    Fminusk = bulk.Fk(-k)
    cell_dim = dot.number_of_elements
    npt.assert_allclose(np.trace(Fk), 1*cell_dim, atol=1e-2)
    npt.assert_allclose(Fk, np.conj(Fk).T, atol=1e-2)
    npt.assert_allclose(Fk, Fminusk.T, atol=1e-2)
