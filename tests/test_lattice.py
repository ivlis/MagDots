#!/usr/bin/env python3

import numpy as np
import numpy.testing as npt
import unittest

from ..model.lattice import Lattice

class TestLattice(unittest.TestCase):
    def test_properties(self):
        a1 = np.array([2.0, 0.0, 0.0])
        a2 = np.array([0.0, 0.5, 0.0])
        la = Lattice(a1, a2)
        npt.assert_allclose(a1, la.a1, atol=1e-7)
        npt.assert_allclose(a2, la.a2, atol=1e-7)

    def test_rectangular(self):
        a1 = np.array([2.0, 0.0, 0.0])
        a2 = np.array([0.0, 0.5, 0.0])
        la = Lattice(a1, a2)
        b1 = la.b1
        b2 = la.b2
        S0 = la.S0
        self.assertTrue(np.allclose(b1, 2.0*np.pi*np.array([0.5, 0.0, 0.0])))
        self.assertTrue(np.allclose(b2, 2.0*np.pi*np.array([0.0, 2.0, 0.0])))
        self.assertEqual(S0, a1[0]*a2[1])

    def test_triangular(self):
        a1 = np.array([1.0, 0.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3.0)/2.0, 0.0])
        la = Lattice(a1, a2)
        b1 = la.b1
        b2 = la.b2
        S0 = la.S0
        self.assertTrue(S0 - 0.5*np.sqrt(3) < 1e-8)
        self.assertTrue(np.allclose(b1, 2.0*np.pi/S0*np.array([np.sqrt(3.0)/2.0, -0.5, 0.0])))
        self.assertTrue(np.allclose(b2, 2.0*np.pi/S0*np.array([0.0, 1.0, 0.0])))

    def test_arbitary(self):
        a1 = np.array([1.4, 8.2, 0.0])
        a2 = np.array([-2.2, 2.5, 0.0])
        la = Lattice(a1, a2)
        b1 = la.b1
        b2 = la.b2
        npt.assert_allclose([a1.dot(b2), a2.dot(b1)], np.zeros(2), atol=1e-7)
        npt.assert_allclose([a1.dot(b1), a1.dot(b1)], [2*np.pi, 2*np.pi], atol=1e-7)

    def test_GXM(self):
        a1 = np.array([1.0, 0.0, 0.0])
        a2 = np.array([0.0, 1.0, 0.0])
        la = Lattice(a1, a2)
        (G, X, M) = la.GXM
        X_should_be = la.b1/2
        M_should_be = (la.b1 + la.b2)/2
        npt.assert_allclose(X,X_should_be)
        npt.assert_allclose(M,M_should_be)
