__author__  = 'Fabian Gittins'
__date__    = '30/09/2023'

import unittest
from muller import muller

class Test(unittest.TestCase):
    def test_quadratic(self):
        f = lambda x: x**2 - 612
        x = (10, 20, 30)

        root = muller(f, x)
        
        self.assertAlmostEqual(root, 612**(1/2))

if __name__ == '__main__':
    unittest.main()
    