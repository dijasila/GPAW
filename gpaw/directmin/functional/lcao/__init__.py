"""
files in functional should describe the derivatives of the KS functional
"""

from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.functional.lcao.ks import KSLCAO
from gpaw.directmin.functional.lcao.pz import PZSICLcao


def get_functional(func):

    if isinstance(func, basestring):
        func = xc_string_to_dict(func)

    if isinstance(func, dict):
        kwargs = func.copy()
        name = kwargs.pop('name').replace('-', '').lower()
        functional = {'ks': KSLCAO,
                      'pzsic': PZSICLcao}[name](**kwargs)
        return functional
    else:
        raise TypeError('Check functional parameter.')
