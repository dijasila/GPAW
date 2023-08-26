from gpaw import GPAW
#from ase import Atoms
#from ase.build import bulk
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor

from gpaw.hybrids.libexexex import call_libexexex, verify
from ase.io import read
#from gpaw import setup_paths

import numpy as np

from gpaw.basis_data import Basis, BasisFunction
from gpaw.lcao.generate_ngto_augmented import get_ngto
import os

libexexex_dir = '/home/kuisma/libexexex/'
def test_He_one_gaussian():
    basis = Basis('He', 'asd', False,
                      EquidistantRadialGridDescriptor(1/640, 6400))
    basis.bf_j = [BasisFunction(None, l, rcut, 
                                get_ngto(basis.rgd, l, alpha, rcut)[:basis.rgd.ceil(rcut)], 
                                label) for (label, l, alpha, rcut) in
                  [ ('1s', 0, 0.2976, 7.787708969805111),
                    ('2p', 1, 1.2750, 4.124098921940588) ]] 
    kpts = [1,1,1]
    basis.write_xml()
    atoms = read(libexexex_dir + 'examples/He__Lapack__KS_real__NC1/gaussian__no_minimal_basis/original_FHI_aims/geometry.in')
    calc = GPAW(h=0.15, mode='lcao', xc='PBE', setups='ae',
                basis={'He': basis}, kpts=kpts, occupations={'width':0},
                symmetry={'point_group': False, 'time_reversal': True}, nbands=3)
    atoms.calc = calc

    atoms.get_potential_energy()
    call_libexexex(atoms, kpts, calc, xc='PBE0')
    os.system(f'cp {libexexex_dir}/examples/He__Lapack__KS_real__NC1/gaussian__no_minimal_basis/ef_pbc_lists.0000.nml .')
    failed = verify('.', libexexex_dir +'examples/He__Lapack__KS_real__NC1/gaussian__no_minimal_basis')
    assert set(failed) == set(['cutcb_rcut', 'outer_radius', 'atom_radius', 'ks_eigenvector'])
    
