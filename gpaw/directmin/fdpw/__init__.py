"""
Direct optimization for FD and PW modes
"""

from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.fdpw.sd_outer import SteepestDescent, FRcg, LBFGS, \
    LBFGS_P, LSR1P
from gpaw.directmin.fdpw.ls_outer import UnitStepLength


def sd_outer(method, wfs, dim):
    """
    Initialize search direction "p" that is
    class like conjugate gradient or Quasi-Newton methods
    """

    if isinstance(method, basestring):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'SD':
            return SteepestDescent(wfs, dim)
        elif name == 'FRcg':
            return FRcg(wfs, dim)
        elif name == 'LBFGS':
            return LBFGS(wfs, dim, **kwargs)
        elif name == 'LBFGS_P':
            return LBFGS_P(wfs, dim, **kwargs)
        elif name == 'LSR1P':
            return LSR1P(wfs, dim, **kwargs)
        else:
            raise ValueError('Check keyword for search direction!')
    else:
        raise ValueError('Check keyword for search direction!')


def ls_outer(method, objective_function):
    """
    Initialize line search  to find optimal step "alpha"
    along search directions "p'
    x <- x + alpha * p.
    """
    if isinstance(method, basestring):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'UnitStep':
            return UnitStepLength(objective_function, **kwargs)
        else:
            raise ValueError('Check keyword for line search!')
    else:
        raise ValueError('Check keyword for line search!')
