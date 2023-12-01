from typing import Callable, Tuple, Any
import numpy as np
import scipy.optimize as sco

def hm(fun: Callable, x0: np.ndarray, args: Tuple[Any] = None, method=None,
       jac=None, hess=None, hessp=None, bounds=None, constraints=(),
       tol=1e-6, callback=None, options=None, alpha_fun=None, beta_fun=None,
       theta_fun=None) -> sco.OptimizeResult:
    """
    HM algorithm implementation.

    Parameters:
    - fun: callable
        The objective function to be minimized.
    - x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    - **kwargs: Any
        Same arguments as in scipy.optimize.minimize

    Returns:
    - res: OptimizeResult
        Calculated minimum point
    """
    if len(x0.shape) != 1:
        raise ValueError("x0 should have shape (n,)")
    if jac in None:
        raise ValueError("A function for Jacoby matrix computation must be provied")

    K_MAX = 1e+6

    k = 0
    xk = x0

    gradk = jac(xk)
    dk = - gradk
    alphak = alpha_fun(fun, xk, gradk, dk, 1.0)

    while not ((np.linalg.norm(gradk) < tol) and (k >= K_MAX)):
        xk = xk + alphak * dk
        gradk_new = jac(xk)
        bk = beta_fun(gradk, gradk_new)
        thetak = theta_fun(fun, bk, xk, dk, gradk_new)
        dk_new = - gradk_new + bk * dk - thetak * gradk_new
        gradk = gradk_new
        dk = dk_new 

    return xk
