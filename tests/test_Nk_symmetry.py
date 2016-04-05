#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

import scipy.constants as const
mu0 = const.codata.value('mag. constant')

from nose2.tools import params

from ..model.dot import Dot, DoubleDot


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

@params(cylindricalDot, doubleDot)
def test_minus_k(dot):
    k = np.array([0.1,0.2,0])
    elements = dot.get_elements()
    vals = [e(k) for e in elements]
    Nk = dot.elements_to_matrix(vals)
    vals = [e(-k) for e in elements]
    Nminusk = dot.elements_to_matrix(vals)
    Nkconj = np.conj(Nk).T
    npt.assert_allclose(Nk, Nminusk.T, atol=1e-7)
    npt.assert_allclose(Nk, Nkconj, atol=1e-7)
