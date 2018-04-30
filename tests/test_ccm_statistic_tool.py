import unittest
import numpy as np
from nonlinearlab.ccm.statistic_tools import *


class Stats(unittest.TestCase):

    def corr1(self):
        a = [1,1,1,1,1]
        b = [1,1,1,1,1]
        cor = correlation(a,b)
        self.assertEqual(1, cor)
    def corr2(self):
        a = [50,60,70,80,90]
        b = [40,70,90,60,100]
        cor = correlation(a,b)
        self.assertEqual(220/np.sqrt(200*456), cor)
    def corr3(self):
        a = [1,1,4]
        b = [1,1,7]
        cor = correlation(a,b)
        self.assertEqual((2+2+8)/np.sqrt(6*(4+4+16)))

if __name__ == '__main__':
    unittest.main()