import matplotlib.pyplot as plt
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
                convergence={'bands': -4})

    a = 5.431

    atoms = bulk('Si', 'diamond', a=a)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si.gpw', 'all')

    bse = BSE('Si.gpw',
            ecut=20, spinors=False,
            valence_bands=range(2,4),
            conduction_bands=range(4,6),
            eshift=0,
            mode = 'BSE',
            nbands=8,
            write_h=False,
            write_v=False)
    
    
    chi_irr_bse = bse.get_vchi(eta=0.2,optical = True, q_c = [0,0,0], hybrid = True,
                    w_w=np.array([-3,0,6]))


    assert np.allclose(np.array(chi_irr_bse)[0,:3,:3], 
   np.array([[ 2.76303362+1.41542559j, -0.06519706-0.01796091j,  0.01792886-0.06572987j]
 [-0.01769456+0.06515327j, -0.01168023+0.01055983j, -0.00986236-0.01076832j]
 [ 0.06568665+0.01766283j,  0.00986239+0.0107683j,  -0.01166505+0.01061083j]]))

 #   chi_irr_BSE_correct = np.load('chi_irr_BSE_correct.npy') 
    
 #   assert np.allclose(chi_irr_BSE_new.real, chi_irr_BSE_correct.real, atol=1e-5, rtol=1e-4)
 #   assert np.allclose(chi_irr_BSE_new.imag, chi_irr_BSE_correct.imag, atol=1e-5, rtol=1e-4)
