#!/usr/bin/env python3

import unittest
import copy
import numpy as np
import scipy.special as sf

import scipy.constants as const
mu0 = const.codata.value('mag. constant')

from ..model.dot import Dot, DoubleDot

def mod(k):
    return np.sqrt(k.dot(k))

def norm(k):
    return k.dot(k)

class TestInit(unittest.TestCase):
    def test_cylinder(self):
        R = 1.0
        h = 30.0*R
        Ms = 1.0
        dot = Dot(R, h, Ms)
        self.assertEqual(dot.h, h)
        self.assertEqual(dot.R, R)
        self.assertEqual(dot.Ms, Ms)
    def test_cylinder_with_anisotropy(self):
        R = 1.0
        h = 30.0*R
        Ms = 1.0
        Ha = 1.0
        na = np.array([1,0,0])
        dot = Dot(R, h, Ms, anisotropy=(na,Ha))
        K_should_be = -Ha*np.outer(na,na)/(2.0*mu0*Ms)
        self.assertTrue(np.allclose(dot.K, K_should_be))


class TestElements(unittest.TestCase):
    def setUp(self):
        R = 1.0
        params = dict(R = R,
                      h = 30.0*R,
                      Ms = 1.0,
                      )
        self.params = params
    def test_single_dot(self):
        dot = Dot(**self.params)
        e = [1,2,3,4]
        Nk = dot.elements_to_matrix(e)
        Nk_should_be = np.array(
                [[1, 3, 0],
                 [3, 2, 0],
                 [0, 0, 4]]
                )
        self.assertTrue(np.allclose(Nk, Nk_should_be))
    def test_double_dot(self):
        d = np.array([1.0,0.0,0.0])
        dot = DoubleDot(**self.params, d = d)
        e = list(range(1,4*3+1))
        Nk = dot.elements_to_matrix(e)
        Nk_should_be = np.array(
                [
                 [ 1,  3,  0,  5,  7,  0],
                 [ 3,  2,  0,  7,  6,  0],
                 [ 0,  0,  4,  0,  0,  8],
                 [ 9, 11,  0,  1,  3,  0],
                 [11, 10,  0,  3,  2,  0],
                 [ 0,  0, 12,  0,  0,  4],

                 ]
                )
        self.assertTrue(np.allclose(Nk, Nk_should_be))

class TestNij(unittest.TestCase):
    def setUp(self):
        R = 1.0
        h = 30.0*R
        Ms = 1.0
        self.dot = Dot(R, h, Ms)
        self.k = np.array([1.1,2.3])
        self.h = h
        self.R = R
    def test_N11(self):
        dot = self.dot
        self.assertEqual(dot.N11(np.array([0.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(dot.N11(self.k), 0.0833650226208)
    def test_N22(self):
        dot = self.dot
        self.assertEqual(dot.N22(np.array([0.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(dot.N22(self.k), 0.364463611293)
    def test_N12(self):
        dot = self.dot
        self.assertEqual(dot.N12(np.array([0.0, 0.0, 0.0])), 0.0)
        self.assertAlmostEqual(dot.N12(self.k), 0.174308683662)
    def test_N33(self):
        dot = self.dot
        self.assertEqual(dot.N33(np.array([0.0, 0.0, 0.0])), np.pi)
        self.assertAlmostEqual(dot.N33(self.k), 0.0059326607591)
    def test_fkh(self):
        k = np.array([0.5, 0, 0])
        kh = mod(k)*self.h
        self.assertEqual(self.dot._fkh(0.0), 0.0)
        self.assertEqual(self.dot._fkh(mod(k)), 1.0-(1.0-np.exp(-kh))/kh)
    def test_Nk(self):
        R = self.R
        k = np.array([0.5, 0, 0])
        kr = mod(k)*R
        fkh = self.dot._fkh(mod(k))
        N11 = (2.0*np.pi*R*R*sf.j1(kr)/kr)**2/(np.pi*R*R)*k[0]*k[0]/norm(k)*fkh
        N33 = (2.0*np.pi*R*R*sf.j1(kr)/kr)**2/(np.pi*R*R)*(1.0-fkh)
        self.assertAlmostEqual(self.dot.N11(k), N11)
        self.assertAlmostEqual(self.dot.N33(k), N33)
        self.assertAlmostEqual(self.dot.N22(k), 0.0)
