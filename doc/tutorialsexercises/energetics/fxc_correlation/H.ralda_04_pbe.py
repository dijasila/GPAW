from ase import Atoms
from ase.parallel import paropen
from gpaw import GPAW
from gpaw import PW

resultfile = paropen('H.ralda.DFT_corr_energies.txt', 'a')

H = Atoms('H', [(0, 0, 0)])
H.center(vacuum=2.0)
calc = GPAW(mode=PW(400, force_complex_dtype=True),
            parallel={'domain': 1},
            hund=True,
            txt='H.ralda_01_lda.output.txt',
            xc='PBE')

H.calc = calc
E_pbe = H.get_potential_energy()
E_c_pbe = -calc.get_xc_difference('GGA_X_PBE')

resultfile.write(f'PBE correlation: {E_c_pbe} eV')
resultfile.write('\n')

calc.diagonalize_full_hamiltonian()
calc.write('H.ralda.pbe_wfcs.gpw', mode='all')
