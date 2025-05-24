"""Test suite for `muller` root finder."""

from math import exp, pi, sin
from unittest import TestCase

from muller import muller

Scalar = float | complex


def f(x: Scalar, n: int = 2, p: int = 612) -> Scalar:
    return x**n - p


def g(x: float) -> float:
    return exp(-x) * sin(x)


class TestMuller(TestCase):
    def test_quadratic(self) -> None:
        x = (10, 20, 30)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)

        res = muller(f, x)

        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)

    def test_sin(self) -> None:
        x = (1, 2, 3)

        res = muller(sin, x)

        self.assertAlmostEqual(res.root, pi, delta=1e-5)

        x = (2, 4, 6)

        res = muller(sin, x)

        self.assertAlmostEqual(res.root, 2 * pi, delta=1e-5)

    def test_expsin(self) -> None:
        x = (-2, -3, -4)

        res = muller(g, x)

        self.assertAlmostEqual(res.root, -pi, delta=1e-5)

        x = (-1, 0, 1 / 2)

        res = muller(g, x)

        self.assertAlmostEqual(res.root, 0, delta=1e-5)

        x = (-1, 0, 1)

        res = muller(g, x)

        self.assertAlmostEqual(res.root, pi, delta=1e-5)

    def test_args(self) -> None:
        x = (10, 20, 30)
        n, p = 2, 612

        res = muller(f, x, args=(n, p))

        self.assertAlmostEqual(res.root, 612 ** (1 / 2), delta=1e-5)

        x = (-10, -20, -30)

        res = muller(f, x, args=(n, p))

        self.assertAlmostEqual(res.root, -(612 ** (1 / 2)), delta=1e-5)

    def test_complex(self) -> None:
        x = (-1, (-1 + 1j) / 2, 1j)
        n, p = 3, 1

        res = muller(f, x, args=(n, p))

        self.assertAlmostEqual(
            res.root, (-1 + 3 ** (1 / 2) * 1j) / 2, delta=1e-5
        )

        x = (-1, -(1 + 1j) / 2, -1j)

        res = muller(f, x, args=(n, p))

        self.assertAlmostEqual(
            res.root, -(1 + 3 ** (1 / 2) * 1j) / 2, delta=1e-5
        )
