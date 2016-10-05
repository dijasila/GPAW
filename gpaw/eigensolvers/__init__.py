"""Module defining  ``Eigensolver`` classes."""

from ase.utils import basestring

from gpaw.eigensolvers.rmmdiis import RMMDIIS
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.lcao.eigensolver import DirectLCAO


def get_eigensolver(eigensolver, mode, convergence=None):
    """Create eigensolver object."""
    if eigensolver is None:
        if mode.name == 'lcao':
            eigensolver = 'lcao'
        else:
            eigensolver = 'dav'
            
    if isinstance(eigensolver, basestring):
        eigensolver = {'name': eigensolver}
        
    if isinstance(eigensolver, dict):
        name = eigensolver.pop('name')
        eigensolver = {'rmm-diis': RMMDIIS,
                       'cg': CG,
                       'dav': RMMDIIS,#Davidson,
                       'lcao': DirectLCAO
                       }[name](**eigensolver)
    
    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence['eigenstates']

    assert isinstance(eigensolver, DirectLCAO) == (mode.name == 'lcao')

    return eigensolver
