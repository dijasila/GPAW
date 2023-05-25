"""
Direct optimization for FD and PW modes
"""

from gpaw.xc import xc_string_to_dict
from gpaw.directmin.sd_etdm import SteepestDescent, FRcg, LBFGS, LBFGS_P, \
    LSR1P, ModeFollowing
from gpaw.directmin.ls_etdm import MaxStep, StrongWolfeConditions

def sd_outer(method, etdm=None, pd=None):
    """
    Initialize search direction "p" that is
    class like conjugate gradient or Quasi-Newton methods
    """

    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        names = kwargs.pop('name').replace('-', '').lower().split('_')
        concave_step_length = 0.1
        if 'concave_step_length' in kwargs.keys():
            concave_step_length = kwargs.pop('concave_step_length')

        searchdir = {'sd': SteepestDescent,
                     'frcg': FRcg,
                     'lbfgs': LBFGS,
                     'lbfgsp': LBFGS_P,
                     'lsr1p': LSR1P
                     }[names[0]](**kwargs)

        if len(names) == 2:
            if names[1] == 'gmf':
                pd['gmf'] = True
                searchdir = ModeFollowing(partial_diagonalizer(pd, etdm),
                                          searchdir, concave_step_length)

        return searchdir
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
                   'swcawc': StrongWolfeConditions
                   }[name](objective_function, **kwargs)

        return ls_algo
    else:
        raise ValueError('Check keyword for line search!')


def partial_diagonalizer(method, domom):
    from gpaw.directmin.derivatives import Davidson
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'Davidson':
            return Davidson(domom, **kwargs)
        else:
            raise ValueError('Check keyword for partial diagonalizer!')