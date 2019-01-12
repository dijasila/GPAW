from gpaw.xc.__init__ import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.odd.pz import PzCorrectionsLcao


def odd_corrections(odd, wfs):

    if isinstance(odd, basestring):
        odd = xc_string_to_dict(odd)

    if isinstance(odd, dict):
        kwargs = odd.copy()
        name = kwargs.pop('name')
        if name is 'PZ_SIC':
            return PzCorrectionsLcao(wfs, **kwargs)
        elif name is 'Zero':
            return ZeroCorrLcao(wfs, **kwargs)
        else:
            raise NotImplementedError('Check name of the '
                                      'ODD corrections')
    else:
        raise NotImplementedError('Check ODD Corrections parameter')
