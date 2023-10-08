# muller
Python implementation of Muller's method for root finding.

### Usage
`muller` has no dependencies.

Here is a quick example to show how it is used. First, import the root finder
```
from muller import muller
```
We will look for a root of
$$
f(x) = e^{-x} \sin(x).
$$
This has zeros at $x = n \pi$ with $n$ an integer.

We will look for the root at $x = 0$. We need three initial guesses for the root and will use $x = -1, 0, 1$. So,
```
from math import exp, sin

def f(x): return exp(-x)*sin(x)

xguesses = (-1, 0, 1)
```
To search for the root, we type
```
root = muller(f, xguesses)
```

### Testing
To test, run
```
python -m unittest tests.test
```
in the root directory.
