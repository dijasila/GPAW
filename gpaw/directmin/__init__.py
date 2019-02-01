from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.sd_lcao import SteepestDescent, FRcg, HZcg, \
    QuickMin, LBFGS, LBFGS_P
from gpaw.directmin.ls_lcao import UnitStepLength, \
    StrongWolfeConditions, Parabola


def search_direction(method, wfs):
    if isinstance(method, basestring):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'SD':
            return SteepestDescent(wfs)
        elif name == 'FRcg':
            return FRcg(wfs)
        elif name == 'HZcg':
            return HZcg(wfs)
        elif name == 'QuickMin':
            return QuickMin(wfs)
        elif name == 'LBFGS':
            return LBFGS(wfs, **kwargs)
        elif name == 'LBFGS_P':
            return LBFGS_P(wfs, **kwargs)
        else:
            raise ValueError('Check keyword for search direction!')
    else:
        raise ValueError('Check keyword for search direction!')


def line_search_algorithm(method, objective_function):
    if isinstance(method, basestring):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'UnitStep':
            return UnitStepLength(objective_function)
        elif name == 'Parabola':
            return Parabola(objective_function)
        elif name == 'SwcAwc':
            return StrongWolfeConditions(objective_function,
                                         **kwargs
                                         )
        else:
            raise ValueError('Check keyword for line search!')
    else:
        raise ValueError('Check keyword for line search!')
