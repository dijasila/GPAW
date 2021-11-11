"""
Exponential Transformation Direct Minimization
"""

from gpaw.xc import xc_string_to_dict
from gpaw.directmin.sd_etdm import SteepestDescent, FRcg, QuickMin, LBFGS, \
    LBFGS_P, LSR1P, ModeFollowing
from gpaw.directmin.ls_etdm import MaxStep, StrongWolfeConditions, Parabola

def search_direction(method, etdm=None, pd=None):
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        names = kwargs.pop('name').replace('-', '').lower().split('_')

        searchdir = {'sd': SteepestDescent,
                     'frcg': FRcg,
                     'quickmin': QuickMin,
                     'lbfgs': LBFGS,
                     'lbfgsp': LBFGS_P,
                     'lsr1p': LSR1P
                     }[names[0]](**kwargs)

        if len(names) == 2:
            if names[1] == 'mmf':
                searchdir = ModeFollowing(partial_diagonalizer(pd, etdm),
                    searchdir)

        return searchdir
    else:
        raise ValueError('Check keyword for search direction!')


def line_search_algorithm(method, objective_function, searchdir_algo):
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
