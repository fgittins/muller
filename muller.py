__author__  = 'Fabian Gittins'
__date__    = '05/10/2023'

def muller(f, x, tol=1e-5, maxiter=50, verbose=False):
    """Muller's method for root finding of scalar function.

    Implementation is based on description of Muller's method in Sec. 9.5.2 of 
    Ref. [1].

    Parameters
    ----------
    f : callable
        Function to find root of
    x : (3,) array_like
        Three initial guesses
    tol : float, optional
        Absolute tolerance for termination
    maxiter : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Prints final number of iterations

    Returns
    -------
    root : float
        Root of function

    References
    ----------
    [1] Press et al. (2007), "Numerical recipes. The Art of Scientific 
        Computing, 3rd Edition" (Cambridge University Press, Cambridge, UK; 
        http://numerical.recipes/book.html).
    """
    if len(x) != 3:
        raise ValueError('x must have three components')
    if tol <= 0:
        raise ValueError('tol is too small (tol = {} <= 0)'.format(tol))
    if type(maxiter) != int:
        raise ValueError('maxiter must be integer')
    if maxiter < 1:
        raise ValueError('maxiter must be greater than 0')
    if type(verbose) != bool:
        raise ValueError('verbose must be boolean')

    ximinus2, ximinus1, xi = x
    yiminus2, yiminus1, yi = f(ximinus2), f(ximinus1), f(xi)

    converged = False

    for i in range(maxiter):
        q           = (xi - ximinus1)/(ximinus1 - ximinus2)
        A           = q*yi - q*(1 + q)*yiminus1 + q**2*yiminus2
        B           = (2*q + 1)*yi - (1 + q)**2*yiminus1 + q**2*yiminus2
        C           = (1 + q)*yi

        denomplus   = B + (B**2 - 4*A*C)**(1/2)
        denomminus  = B - (B**2 - 4*A*C)**(1/2)

        if abs(denomplus) >= abs(denomminus):
            xiplus1 = xi - (xi - ximinus1)*2*C/denomplus
        else:
            xiplus1 = xi - (xi - ximinus1)*2*C/denomminus

        yiplus1 = f(xiplus1)

        if abs(yiplus1) <= tol:
            converged = True
            break

        ximinus2, ximinus1, xi = ximinus1, xi, xiplus1
        yiminus2, yiminus1, yi = yiminus1, yi, yiplus1

    if converged and verbose:
        print('Method converged after {} iterations'.format(i))
    elif not converged:
        raise ValueError('Method did not converge within {} iterations'
                         .format(maxiter))
    return xiplus1
