#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

from ..model.dotarray import _make_toeplitz_from_blocks

class TestToeplitz(unittest.TestCase):
    def test_toeplitz(self):
        B = [np.array([[i,i], [i,i]]) for i in range(-3, 4)]
        T = _make_toeplitz_from_blocks(B)
        T_should_be = np.array(
            [[ 0,  0,  1,  1,  2,  2,  3,  3],
             [ 0,  0,  1,  1,  2,  2,  3,  3],
             [-1, -1,  0,  0,  1,  1,  2,  2],
             [-1, -1,  0,  0,  1,  1,  2,  2],
             [-2, -2, -1, -1,  0,  0,  1,  1],
             [-2, -2, -1, -1,  0,  0,  1,  1],
             [-3, -3, -2, -2, -1, -1,  0,  0],
             [-3, -3, -2, -2, -1, -1,  0,  0]]
        )
        npt.assert_allclose(T, T_should_be)
