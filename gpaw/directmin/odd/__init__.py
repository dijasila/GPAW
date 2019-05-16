from gpaw.xc.__init__ import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.odd.pz import PzCorrections
from gpaw.directmin.odd.spz import SPzCorrections
from gpaw.directmin.odd.spz_2 import SPzCorrectionsLcao2

def odd_corrections(odd, wfs, dens, ham):

    if isinstance(odd, basestring):
        odd = xc_string_to_dict(odd)

    if isinstance(odd, dict):
        kwargs = odd.copy()
        name = kwargs.pop('name')
        if name == 'PZ_SIC':
            return PzCorrections(wfs, dens, ham, **kwargs)
        elif name == 'SPZ_SIC':
            return SPzCorrections(wfs, dens, ham, **kwargs)
        elif name == 'SPZ_SIC2':
            return SPzCorrectionsLcao2(wfs, dens, ham, **kwargs)
        elif name == 'Zero':
            return None
        else:
            raise NotImplementedError('Check name of the '
                                      'ODD corrections')
    else:
        raise NotImplementedError('Check ODD Corrections parameter')
