from dataclasses import dataclass
from typing import Any, Callable, Tuple, Union

Scalar = Union[float, complex]


@dataclass
class Results:
    """Represents root-finding result.

    Attributes
    ----------
    root : Scalar
        Estimated root.
    iterations : int
        complex of iterations needed to find root.
    converged : bool
        True if routine converged.
    flag : str
        Description of cause of termination.
    """

    root: Scalar
    iterations: int
    converged: bool
    flag: str


def muller(
    f: Callable[..., Scalar],
    x: Tuple[Scalar, Scalar, Scalar],
    xtol: float = 1e-5,
    ftol: float = 1e-5,
    maxiter: int = 50,
    args: Union[None, Tuple[Any, ...]] = None,
) -> Results:
    """Muller's method for root finding of scalar function.

    Implementation is based on description of Muller's method in Sec. 9.5.2 of
    Ref. [1].

    Parameters
    ----------
    f : Callable
        Function to find root of.
    x : (3,) Tuple
        Three initial guesses.
    xtol : float, optional
        Absolute error in `x` between iterations that is acceptable for
        convergence.
    ftol : float, optional
        Minimum absolute value of function `f` that is acceptable for
        convergence.
    maxiter : int, optional
        Maximum complex of iterations.
    args : tuple, optional
        Additional arguments to pass to `f`.

    Returns
    -------
    res : Results
        Contains results of routine.

    References
    ----------
    [1] Press et al. (2007), "Numerical recipes. The Art of Scientific
        Computing, 3rd Edition" (Cambridge University Press, Cambridge, UK;
        http://numerical.recipes/book.html).
    """
    if xtol < 0:
        raise ValueError(f"xtol is negative (xtol = {xtol} < 0)")
    if ftol < 0:
        raise ValueError(f"ftol is negative (ftol = {ftol} < 0)")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    if args is not None:

        def call_f(x: Scalar) -> Scalar:
            return f(x, *args)
    else:

        def call_f(x: Scalar) -> Scalar:
            return f(x)

    ximinus2, ximinus1, xi = x
    yiminus2, yiminus1, yi = call_f(ximinus2), call_f(ximinus1), call_f(xi)

    converged = False
    flag = "Routine did not converge"

    xiplus1 = ximinus2
    i = 0
    while i < maxiter:
        q: Scalar = (xi - ximinus1) / (ximinus1 - ximinus2)
        A = q * yi - q * (1 + q) * yiminus1 + q**2 * yiminus2
        B = (2 * q + 1) * yi - (1 + q) ** 2 * yiminus1 + q**2 * yiminus2
        C = (1 + q) * yi

        denomplus: Scalar = B + (B**2 - 4 * A * C) ** (1 / 2)
        denomminus: Scalar = B - (B**2 - 4 * A * C) ** (1 / 2)

        if abs(denomplus) >= abs(denomminus):
            xiplus1 = xi - (xi - ximinus1) * 2 * C / denomplus
        else:
            xiplus1 = xi - (xi - ximinus1) * 2 * C / denomminus

        yiplus1 = call_f(xiplus1)

        if ftol >= abs(yiplus1):
            flag = f"Routine has reached desired tolerance in function value ({abs(yiplus1)} <= {ftol})"
            converged = True
            break
        if xtol >= abs(xiplus1 - xi):
            flag = f"Routine has reached desired tolerance in root value ({abs(xiplus1 - xi)} <= {xtol})"
            converged = True
            break

        ximinus2, ximinus1, xi = ximinus1, xi, xiplus1
        yiminus2, yiminus1, yi = yiminus1, yi, yiplus1

        i += 1

    return Results(xiplus1, i + 1, converged, flag)
