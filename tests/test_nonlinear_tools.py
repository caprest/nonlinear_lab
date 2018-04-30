from nonlinearlab.ccm.nonlinear_tools import *

import unittest
import numpy as np


class Tools(unittest.TestCase):
    def test_l1(self):
        a = [0, 3]
        b = [4, 0]
        self.assertEqual(7, dist_l1(a, b))

    def test_l2(self):
        a = [0, 3]
        b = [4, 0]
        self.assertEqual(5, dist_l2(a, b))

    def test_l2_2(self):
        a = [0, 0, 0]
        b = [0, 0, 4]
        self.assertEqual(4,dist_l2(a,b))
    def test_emb_1(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        emb = embedding(a, 5)
        ans = np.array(
            [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10]])
        diff = np.sum(np.abs(emb - ans))
        self.assertEqual(0, diff)


if __name__ == '__main__':
    unittest.main()
