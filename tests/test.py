import unittest
from math import pi, sin, exp

from muller import muller


class Test(unittest.TestCase):
    def test_quadratic(self):
        def f(x):
            return x**2 - 612

        x = (10, 20, 30)

        root = muller(f, x)

        self.assertAlmostEqual(root, 612**(1/2), delta=1e-5)

        x = (-10, -20, -30)

        root = muller(f, x)

        self.assertAlmostEqual(root, -612**(1/2), delta=1e-5)

    def test_sine(self):
        def f(x):
            return sin(x)

        x = (1, 2, 3)

        root = muller(f, x)

        self.assertAlmostEqual(root, pi, delta=1e-5)

        x = (2, 4, 6)

        root = muller(f, x)

        self.assertAlmostEqual(root, 2*pi, delta=1e-5)

    def test_expsine(self):
        def f(x):
            return exp(-x)*sin(x)

        x = (-2, -3, -4)

        root = muller(f, x)

        self.assertAlmostEqual(root, -pi, delta=1e-5)

        x = (-1, 0, 1/2)

        root = muller(f, x)

        self.assertAlmostEqual(root, 0, delta=1e-5)

        x = (-1, 0, 1)

        root = muller(f, x)

        self.assertAlmostEqual(root, pi, delta=1e-5)


if __name__ == '__main__':
    unittest.main()
