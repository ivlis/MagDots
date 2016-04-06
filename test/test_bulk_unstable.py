import sys
sys.path.append("../..")

import unittest
import numpy as np
import scipy.constants as const

import numpy.testing as npt

mu0 = const.codata.value('mag. constant')
gamma = const.codata.value('electron gyromag. ratio')

from ..model.dot import Dot, DoubleDot
from ..model.lattice import Lattice, ContourIterator
from ..model.dotarray import BulkWave, magn_norm, ArrayUnstable
from ..util.util import calculate_bulk, BulkData

def _calc_norms(V, J):
    a, b_ = J.shape
    norms = [magn_norm(m, J) for m in np.reshape(np.swapaxes(V,1,2), (-1,a))]
    return norms

class TestBulkFM(unittest.TestCase):
    def test_bulk_negative_norm(self):

        # Setting array's physical parameters

        R = 1.0
        a = 4.1*R
        a1 = np.array([1.0*a, 0.0*a, 0.0])
        a2 = np.array([0.0*a, 1.0*a, 0.0])
        Ms = 1.0/mu0 # mu0 Ms = 1T
        h = 2.1*R

        # Setting array's magnetic parameters

        mu = np.array([0.0,0.0,1.0])

        dot = Dot(R=R, h=h, Ms=Ms)
        la = Lattice(a1=a1, a2=a2)
        bulk = BulkWave(dot = dot, lattice = la, mu=mu,  K_MAX=30)

        # Setting points on the Brillouin zone to calculate
        points = 4
        G,X,M = la.GXM
        vertices = [G, X, M, G]

        K_Kmod = list(ContourIterator(vertices, points))
        K, K_mod = zip(*K_Kmod)

        #Perform calculations
        with self.assertRaises(ArrayUnstable):    
            result = calculate_bulk(bulk, K, K_mod, points, parallel=False)
