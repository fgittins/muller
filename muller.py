__author__  = 'Fabian Gittins'
__date__    = '30/09/2023'

def muller(f, x, tol=1e-5, maxiter=50):
    """
    Muller's method for root finding of scalar function.

    Parameters
    ----------
    f : callable
        Function to find root of
    x : (3,) array_like
        Three initial guesses
    tol : float, optional
        Absolute tolerance for termination
    maxiter : float, optional
        Maximum number of iterations

    Returns
    -------
    root : float
        Root of function
    """
    if len(x) != 3:
        raise ValueError('x must have three components')

    x0, x1, x2 = x
    y0 = f(x0)
    y1 = f(x1)
    y2 = f(x2)

    converged = False

    for i in range(maxiter):
        divdiff01   = (y1 - y0)/(x1 - x0)
        divdiff02   = (y2 - y0)/(x2 - x0)
        divdiff12   = (y2 - y1)/(x2 - x1)
        divdiff012  = (divdiff12 - divdiff01)/(x2 - x0)
        w           = divdiff01 + divdiff02 - divdiff12

        delta       = (w**2 - 4*y2*divdiff012)**(1/2)
        denomplus   = w + delta
        denomminus  = w - delta

        if abs(denomplus) > abs(denomminus):
            x3 = x2 - 2*y2/denomplus
        else:
            x3 = x2 - 2*y2/denomminus

        y3 = f(x3)

        y0, y1, y2 = y1, y2, y3
        x0, x1, x2 = x1, x2, x3

        if abs(y3) < tol:
            converged = True
            break

    if converged:
        print('Method converged after {} iterations'.format(i))
    else:
        raise ValueError('Method did not converge within {} iterations'
                         .format(maxiter))
    return x3
