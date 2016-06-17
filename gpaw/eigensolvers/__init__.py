"""Module defining  ``Eigensolver`` classes."""

from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.eigensolvers.direct import DirectPW
from gpaw.lcao.eigensolver import DirectLCAO


def get_eigensolver(name, mode, convergence=None):
    """Create eigensolver object."""
    if name is None:
        if mode.name == 'lcao':
            name = 'lcao'
        else:
            name = 'dav'
    if isinstance(name, str):
        eigensolver = {'rmm-diis': RMM_DIIS,
                       'cg': CG,
                       'dav': Davidson,
                       'lcao': DirectLCAO,
                       'direct': DirectPW
                       }[name]()
    else:
        eigensolver = name

    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence['eigenstates']

    assert isinstance(eigensolver, DirectLCAO) == (mode.name == 'lcao')

    return eigensolver
