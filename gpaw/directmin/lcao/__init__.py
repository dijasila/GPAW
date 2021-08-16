"""
Exponential Transformation Direct Minimization
"""

from gpaw.xc import xc_string_to_dict
from gpaw.directmin.lcao.sd_lcao import SteepestDescent, FRcg, QuickMin,\
    LBFGS, LBFGS_P, LSR1P
from gpaw.directmin.lcao.ls_lcao import UnitStepLength, \
    StrongWolfeConditions, Parabola


def search_direction(method, wfs):
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')

        searchdir = {'SD': SteepestDescent,
                     'FRcg': FRcg,
                     # 'HZcg': HZgc,
                     'QuickMin': QuickMin,
                     'LBFGS': LBFGS,
                     'LBFGS_P': LBFGS_P,
                     'LSR1P': LSR1P
                     }[name](wfs, **kwargs)

        return searchdir
    else:
        raise ValueError('Check keyword for search direction!')


def line_search_algorithm(method, objective_function):
    if isinstance(method, str):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')

        ls_algo = {'UnitStep': UnitStepLength,
                   'Parabola': Parabola,
                   'SwcAwc': StrongWolfeConditions
                   }[name](objective_function, **kwargs)
        
        return ls_algo
    else:
        raise ValueError('Check keyword for line search!')
