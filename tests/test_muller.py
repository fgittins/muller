"""Test suite for `muller` root finder."""

from math import exp, pi, sin
from unittest import TestCase

from muller import muller


def f(x, p=612):
    return x**2 - p


def g(x):
    return exp(-x) * sin(x)


class TestMuller(TestCase):
    def test_quadratic(self):
        x = (10, 20, 30)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)

    def test_sin(self):
        x = (1, 2, 3)

        res = muller(sin, x)

        self.assertAlmostEqual(res.root, pi, delta=1e-5)

        x = (2, 4, 6)

        res = muller(sin, x)

        self.assertAlmostEqual(res.root, 2 * pi, delta=1e-5)

    def test_expsin(self):
        x = (-2, -3, -4)

        res = muller(g, x)

        self.assertAlmostEqual(res.root, -pi, delta=1e-5)

        x = (-1, 0, 1 / 2)

        res = muller(g, x)

        self.assertAlmostEqual(res.root, 0, delta=1e-5)

        x = (-1, 0, 1)

        res = muller(g, x)

        self.assertAlmostEqual(res.root, pi, delta=1e-5)

    def test_args(self):
        x = (10, 20, 30)
        p = 612

        res = muller(f, x, args=(p,))

        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)

        res = muller(f, x, args=(p,))

        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)
