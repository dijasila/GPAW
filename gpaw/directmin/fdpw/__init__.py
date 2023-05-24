"""
Direct optimization for FD and PW modes
"""

from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.fdpw.sd_outer import SteepestDescent, FRcg, HZcg, \
    PRcg, PRpcg, QuickMin, LBFGS, LBFGS_P, LSR1P, PFRcg
#from gpaw.directmin.fdpw.ls_outer import UnitStepLength, \
#    StrongWolfeConditions, Parabola, TwoStepParabola, \
#    TwoStepParabolaAwc, TwoStepParabolaCubicAwc, \
#    TwoStepParabolaCubicDescent
from gpaw.directmin.ls_etdm import MaxStep, StrongWolfeConditions, Parabola

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
        elif name == 'PFRcg':
            return PFRcg(wfs, dim)
        elif name == 'PRcg':
            return PRcg(wfs, dim)
        elif name == 'PRpcg':
            return PRpcg(wfs, dim)
        elif name == 'HZcg':
            return HZcg(wfs, dim)
        elif name == 'QuickMin':
            return QuickMin(wfs, dim)
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


def ls_outer(method, objective_function, searchdir_algo):
    """
    Initialize line search  to find optimal step "alpha"
    along search directions "p'
    x <- x + alpha * p.
    """
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name').replace('-', '').lower()
        if name == 'swcawc':
            # for swc-awc we need to know
            # what search. dir. algo is used
            if 'searchdirtype' not in kwargs:
                kwargs['searchdirtype'] = searchdir_algo.type

        ls_algo = {'maxstep': MaxStep,
                   'parabola': Parabola,
                   'swcawc': StrongWolfeConditions
                   }[name](objective_function, **kwargs)

        return ls_algo
    else:
        raise ValueError('Check keyword for line search!')
