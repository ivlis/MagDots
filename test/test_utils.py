#!/usr/bin/env python3

import unittest

from ..util import *

class TestSetLimitValue(unittest.TestCase):
    def test_arg(self):
        @set_limit_value(0,0,0)
        def fn(a):
            return 1
        self.assertEqual(fn(0), 0)
        self.assertEqual(fn(1), 1)
    def test_kwarg(self):
        @set_limit_value("y", 1, 1)
        def fn(y=0):
            return 0
        self.assertEqual(fn(y=0), 0)
        self.assertEqual(fn(y=1), 1)

class TestNorm(unittest.TestCase):
    def test_norm(self):
        v = np.array([3.0,4.0])
        self.assertEqual(norm(v), 25.0)
    def test_mod(self):
        v = np.array([3.0,4.0])
        self.assertEqual(mod(v), 5.0)
