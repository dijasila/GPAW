from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.fd.sd_outer import SteepestDescent, FRcg, HZcg, \
    PRcg, PRpcg, QuickMin, LBFGS
from gpaw.directmin.fd.ls_outer import UnitStepLength, \
    StrongWolfeConditions, Parabola, TwoStepParabola, \
    TwoStepParabolaAwc, TwoStepParabolaCubicAwc, \
    TwoStepParabolaCubicDescent


def sd_outer(method, wfs, dim):
    if isinstance(method, basestring):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'SD':
            return SteepestDescent(wfs, dim)
        elif name == 'FRcg':
            return FRcg(wfs, dim)
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
        else:
            raise ValueError('Check keyword for search direction!')
    else:
        raise ValueError('Check keyword for search direction!')


def ls_outer(method, objective_function):
    if isinstance(method, basestring):
        method = xc_string_to_dict(method)

    if isinstance(method, dict):
        kwargs = method.copy()
        name = kwargs.pop('name')
        if name == 'UnitStep':
            return UnitStepLength(objective_function)
        elif name == 'Parabola':
            return Parabola(objective_function)
        elif name == 'TSP':
            return TwoStepParabola(objective_function)
        elif name == 'TSPAWC':
            return TwoStepParabolaAwc(objective_function)
        elif name == 'TSPCAWC':
            return TwoStepParabolaCubicAwc(objective_function)
        elif name == 'TSPCD':
            return TwoStepParabolaCubicDescent(objective_function)
        elif name == 'SwcAwc':
            return StrongWolfeConditions(objective_function,
                                         **kwargs
                                         )
        else:
            raise ValueError('Check keyword for line search!')
    else:
        raise ValueError('Check keyword for line search!')
