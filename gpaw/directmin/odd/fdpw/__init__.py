from gpaw.xc.__init__ import xc_string_to_dict
from ase.utils import basestring
from gpaw.directmin.odd.fdpw.pz import PzCorrections
from gpaw.directmin.odd.fdpw.zero import ZeroCorrections
from gpaw.directmin.odd.fdpw.dftpzxt import DftPzSicXT


def odd_corrections(odd, wfs, dens, ham):

    if isinstance(odd, basestring):
        odd = xc_string_to_dict(odd)

    if isinstance(odd, dict):
        kwargs = odd.copy()
        name = kwargs.pop('name')
        if name == 'PZ-SIC':
            return PzCorrections(wfs, dens, ham, **kwargs)
        elif name == 'PZ-SIC-XT':
            return DftPzSicXT(wfs, dens, ham, **kwargs)
        elif name == 'ks':
            return ZeroCorrections(wfs, dens, ham, **kwargs)
        else:
            raise NotImplementedError('Check name of the '
                                      'ODD corrections')
    else:
        raise NotImplementedError('Check ODD Corrections parameter')
