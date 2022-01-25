"""
Exponential Transformation Direct Minimization
"""

from gpaw.xc import xc_string_to_dict
from gpaw.directmin.sd_etdm import SteepestDescent, FRcg, QuickMin, LBFGS, \
    LBFGS_P, LSR1P
from gpaw.directmin.ls_etdm import MaxStep, StrongWolfeConditions, Parabola


def search_direction(method):
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')

        searchdir = {'sd': SteepestDescent,
                     'fr-cg': FRcg,
                     'quick-min': QuickMin,
                     'l-bfgs': LBFGS,
                     'l-bfgs-p': LBFGS_P,
                     'l-sr1p': LSR1P
                     }[name](**kwargs)

        return searchdir
    else:
        raise ValueError('Check keyword for search direction!')


def line_search_algorithm(method, objective_function, searchdir_algo):
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'swc-awc':
            # for swc-awc we need to know
            # what search. dir. algo is used
            if 'searchdirtype' not in kwargs:
                kwargs['searchdirtype'] = searchdir_algo.type

        ls_algo = {'max-step': MaxStep,
                   'parabola': Parabola,
                   'swc-awc': StrongWolfeConditions
                   }[name](objective_function, **kwargs)

        return ls_algo
    else:
        raise ValueError('Check keyword for line search!')
