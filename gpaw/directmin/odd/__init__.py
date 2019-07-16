from gpaw.xc.__init__ import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.odd.pz import PzCorrectionsLcao
from gpaw.directmin.odd.zero import ZeroCorrectionsLcao


def odd_corrections(odd, wfs, dens, ham):

    if isinstance(odd, basestring):
        odd = xc_string_to_dict(odd)

    if isinstance(odd, dict):
        kwargs = odd.copy()
        name = kwargs.pop('name')
        if name == 'Zero':
            return ZeroCorrectionsLcao(wfs, dens, ham, **kwargs)
        elif name == 'PZ_SIC':
            return PzCorrectionsLcao(wfs, dens, ham, **kwargs)
        else:
            raise NotImplementedError('Check name of the '
                                      'ODD corrections')
    else:
        raise NotImplementedError('Check ODD Corrections parameter')
