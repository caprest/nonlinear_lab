import unittest

from nonlinearlab.causality.ccm import *
from nonlinearlab.ccm.sample_generator import generate_logistic
from nonlinearlab.ccm.statistic_tools import *


class TestCCM(unittest.TestCase):
    def test_calic_all_dist(self):
        emb = np.array([[0,0,0],[0,0,4],[2,0,0],[0,3,0],[0,0,1]])
        dist_arr, dist_idx = calic_all_dist(emb)
        ans_arr =np.array([
            [0,4,2,3,1],
            [4,0,np.sqrt(20),5,3],
            [2,np.sqrt(20),0,np.sqrt(13),np.sqrt(5)],
            [3,5,np.sqrt(13),0,np.sqrt(10)],
            [1,3,np.sqrt(5),np.sqrt(10),0]
        ])
        ans_idx = np.array([
            [0,4,2,3,1],
            [1,4,0,2,3],
            [2,0,4,3,1],
            [3,0,4,2,1],
            [4,0,2,1,3]
        ],dtype=int)
        try:
            self.assertEqual(0,np.sum(np.abs(ans_arr-dist_arr)))
        except AssertionError:
            print(dist_arr)
            print(ans_arr)
            raise
        try:
            self.assertEqual(0,np.sum(np.abs(ans_idx-dist_idx)))
        except AssertionError:
            print(dist_idx)
            print(ans_idx)
            raise

    def test_k_nearest(self):
        dist_arr =np.array([
            [0,4,2,3,1],
            [4,0,np.sqrt(20),5,3],
            [2,np.sqrt(20),0,np.sqrt(13),np.sqrt(5)],
            [3,5,np.sqrt(13),0,np.sqrt(10)],
            [1,3,np.sqrt(5),np.sqrt(10),0]
        ])
        dist_idx = np.array([
            [0,4,2,3,1],
            [1,4,0,2,3],
            [2,0,4,3,1],
            [3,0,4,2,1],
            [4,0,2,1,3]
        ],dtype=int)
        k_dist,k_idx = k_nearest(dist_arr,dist_idx, L =4, k= 3)
        ans_dist = np.array([
            [2,3,4],
            [4,np.sqrt(20),5],
            [2,np.sqrt(13),np.sqrt(20)],
            [3,np.sqrt(13),5]
        ])
        ans_idx = np.array([
            [2,3,1],
            [0,2,3],
            [0,3,1],
            [0,2,1]
        ])
        try:
            self.assertEqual(0,np.sum(np.abs(ans_dist-k_dist)))
        except AssertionError:
            print(ans_dist)
            print(k_dist)

        try:
            self.assertEqual(0,np.sum(np.abs(ans_idx-k_idx)))
        except AssertionError:
            print(ans_idx)
            print(k_idx)

    def test_x_map(self):
        x = np.array([1,2,3,4,5,6,7])
        k_dist = np.array([
            [1,2,3],
            [1,2,4],
            [0,1,3],
            [1,1,5]
        ])
        k_idx = np.array([
            [2,3,1],
            [0,2,3],
            [0,3,1],
            [0,2,1]
        ])
        u1 = [np.exp(-1),np.exp(-2),np.exp(-3)]

    def test_plot(self):
        x, y = generate_logistic(length=2000)
        x = x[1000:1999]
        y = y[1000:1999]
        convergence_plot(x, y, length=300)





if __name__ == '__main__':
    unittest.main()

