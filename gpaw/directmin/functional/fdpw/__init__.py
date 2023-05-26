from gpaw.xc import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.functional.fdpw.ks import KSFDPW
from gpaw.directmin.functional.fdpw.pz import PZSICFDPW
from gpaw.directmin.functional.fdpw.dftpzxt import DFTPZSICXT


def get_functional(func, *args):

    if isinstance(func, KSFDPW) or isinstance(func, PZSICFDPW) \
            or isinstance(func, DFTPZSICXT):
        return func
    elif isinstance(func, basestring):
        func = xc_string_to_dict(func)

    if isinstance(func, dict):
        kwargs = func.copy()
        name = kwargs.pop('name').replace('-', '').lower()
        functional = {'ks': KSFDPW,
                      'pzsic': PZSICFDPW,
                      'pzsicxt': DFTPZSICXT}[name](*args, **kwargs)
        return functional
    else:
        raise TypeError('Check functional parameter.')