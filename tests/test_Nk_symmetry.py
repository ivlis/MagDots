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
    KX = np.linspace(-10,10, num=20)
    KY = np.linspace(-10,10, num=20)
    for kx in KX:
        for ky in KY:
            k = np.array([kx, ky, 0])
            Nk = dot.Nk(lambda e: e(k))
            Nminusk = dot.Nk(lambda e: e(-k))
            Nkconj = np.conj(Nk).T
            npt.assert_allclose(Nk, Nminusk.T, atol=1e-7)
            npt.assert_allclose(Nk, Nkconj, atol=1e-7)
