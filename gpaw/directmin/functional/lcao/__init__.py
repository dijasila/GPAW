from gpaw.xc.__init__ import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.functional.lcao.ks import KSLCAO


def get_functional(odd, wfs, dens, ham):

    if isinstance(odd, basestring):
        odd = xc_string_to_dict(odd)

    if isinstance(odd, dict):
        kwargs = odd.copy()
        name = kwargs.pop('name')
        if name == 'ks':
            return KSLCAO(wfs, dens, ham, **kwargs)
        else:
            raise NotImplementedError('Check name of the '
                                      'functional (ks)')
    else:
        raise NotImplementedError('Check functional parameter')
