from ase import Atoms
from ase.parallel import paropen
from gpaw import GPAW
from gpaw import PW

resultfile = paropen('H.ralda.DFT_corr_energies.txt', 'w')
resultfile.write('DFT Correlation energies for H atom\n')

H = Atoms('H', [(0, 0, 0)])
H.center(vacuum=2.0)
calc = GPAW(mode=PW(400, force_complex_dtype=True),
            parallel={'domain': 1},
            hund=True,
            txt='H.ralda_01_lda.output.txt',
            xc='LDA')

H.calc = calc
E_lda = H.get_potential_energy()
E_c_lda = -calc.get_xc_difference('LDA_X')

resultfile.write(f'LDA correlation: {E_c_lda} eV')
resultfile.write('\n')

calc.diagonalize_full_hamiltonian()
calc.write('H.ralda.lda_wfcs.gpw', mode='all')
