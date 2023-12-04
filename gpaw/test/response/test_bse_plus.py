import matplotlib.pyplot as plr
import numpy as np
from ase.build import bulk
import pytest
from gpaw import GPAW, FermiDirac, PW
from gpaw.mpi import world
from gpaw.response.bse import BSE

@pytest.mark.response
def test_bse_plus(in_tmp_dir, scalapack):
    calc = GPAW(mode='pw',
                kpts={'size': (2,2,2), 'gamma': True},
                occupations=FermiDirac(0.01),
                nbands=8,
                symmetry='off',
                convergence={'bands': -4, 'density':1e-7, 'eigenstates':1e-10})

    a = 5.431

    atoms = bulk('Si', 'diamond', a=a)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si.gpw', 'all')

    bse = BSE('Si.gpw',
            ecut=20, spinors=False,
            valence_bands=range(4),
            conduction_bands=range(4,8),
            eshift=0,
            mode = 'BSE',
            nbands=8,
            q_c=[0.0, 0.0, 0.0],
            write_h=False,
            write_v=False)
    
    
    chi_irr_bse = bse.get_chi_GG(eta=0.2,optical = True, hybrid = True,
                                 w_w=np.array([-3,0,6]))
    ref = [(-0.12319305784052169-0.005900101520066767j),
           (-3.519763508433997e-10-0.035759806607911705j),
           (3.031540803184087e-05-0.0004896329314266086j)]
    if world.rank == 0:
        for i, r in enumerate(ref):
            assert np.allclose(chi_irr_bse[i,i,i+1], r)
