"""Test suite for `muller` root finder."""

from math import exp, pi, sin
from unittest import TestCase

from muller import muller


class TestMuller(TestCase):
    def test_quadratic(self):
        def f(x):
            return x**2 - 612

        x = (10, 20, 30)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)

    def test_sine(self):
        def f(x):
            return sin(x)

        x = (1, 2, 3)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, pi, delta=1e-5)

        x = (2, 4, 6)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, 2 * pi, delta=1e-5)

    def test_expsine(self):
        def f(x):
            return exp(-x) * sin(x)

        x = (-2, -3, -4)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, -pi, delta=1e-5)

        x = (-1, 0, 1 / 2)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, 0, delta=1e-5)

        x = (-1, 0, 1)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, pi, delta=1e-5)
