from ase import Atoms
from ase.parallel import paropen
from gpaw import GPAW, FermiDirac
from gpaw.mixer import MixerSum
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw import PW

# CO

CO = Atoms('CO', [(0, 0, 0), (0, 0, 1.1283)])
CO.set_pbc(True)
CO.center(vacuum=3.0)
calc = GPAW(mode=PW(600, force_complex_dtype=True),
            parallel={'domain': 1},
            xc='PBE',
            txt='CO.ralda_01_CO_pbe.txt',
            convergence={'density': 1.e-6})

CO.calc = calc
E0_pbe = CO.get_potential_energy()

E0_hf = nsc_energy(calc, 'EXX')

calc.diagonalize_full_hamiltonian()
calc.write('CO.ralda.pbe_wfcs_CO.gpw', mode='all')

# C

C = Atoms('C')
C.set_pbc(True)
C.set_cell(CO.cell)
C.center()
calc = GPAW(mode=PW(600, force_complex_dtype=True),
            parallel={'domain': 1},
            xc='PBE',
            mixer=MixerSum(beta=0.1, nmaxold=5, weight=50.0),
            hund=True,
            occupations=FermiDirac(0.01, fixmagmom=True),
            txt='CO.ralda_01_C_pbe.txt',
            convergence={'density': 1.e-6})

C.calc = calc
E1_pbe = C.get_potential_energy()

E1_hf = nsc_energy(calc, 'EXX')

f = paropen('CO.ralda.PBE_HF_C.dat', 'w')
print(E1_pbe, E1_hf, file=f)
f.close()

calc.diagonalize_full_hamiltonian()
calc.write('CO.ralda.pbe_wfcs_C.gpw', mode='all')

# O

O = Atoms('O')
O.set_pbc(True)
O.set_cell(CO.cell)
O.center()
calc = GPAW(mode=PW(600, force_complex_dtype=True),
            parallel={'domain': 1},
            xc='PBE',
            mixer=MixerSum(beta=0.1, nmaxold=5, weight=50.0),
            hund=True,
            txt='CO.ralda_01_O_pbe.txt',
            convergence={'density': 1.e-6})

O.calc = calc
E2_pbe = O.get_potential_energy()

E2_hf = nsc_energy(calc, 'EXX')

calc.diagonalize_full_hamiltonian()
calc.write('CO.ralda.pbe_wfcs_O.gpw', mode='all')

f = paropen('CO.ralda.PBE_HF_CO.dat', 'w')
print('PBE: ', E0_pbe - E1_pbe - E2_pbe, file=f)
print('HF: ', E0_hf - E1_hf - E2_hf, file=f)
f.close()
