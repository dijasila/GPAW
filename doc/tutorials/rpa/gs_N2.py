from ase.optimize import BFGS
from ase.structure import molecule
from ase.parallel import paropen
from gpaw import GPAW, PW
from gpaw.xc.exx import EXX

# N -------------------------------------------

N = molecule('N')
N.pbc = True
N.cell = (6., 6., 7.)
N.center()
calc = GPAW(mode=PW(600),
            dtype=complex,
            nbands=8,
            maxiter=300,
            xc='PBE',
            hund=True,
            txt='N.txt',
            convergence={'density': 1.e-6})

N.calc = calc
E1_pbe = N.get_potential_energy()

exx = EXX(calc, ecut=1200, txt='N_exx_1200.txt')
exx.calculate()
E1_hf = exx.get_total_energy()

calc.diagonalize_full_hamiltonian(nbands=4800)
calc.write('N.gpw', mode='all')

# N2 ------------------------------------------

N2 = molecule('N2')
N2.pbc = True
N2.cell = (6., 6., 7.)
N2.center()
calc = GPAW(mode=PW(600),
            dtype=complex,
            maxiter=300,
            xc='PBE',
            txt='N2.txt',
            convergence={'density': 1.e-6})

N2.calc = calc
dyn = BFGS(N2)
dyn.run(fmax=0.05)
E2_pbe = N2.get_potential_energy()

exx = EXX(calc, ecut=1200, txt='N2_exx_1200.txt')
exx.calculate()
E2_hf = exx.get_total_energy()

f = paropen('PBE_HF.dat', 'w')
print >> f, 'PBE: ', E2_pbe - 2 * E1_pbe
print >> f, 'HF: ', E2_hf - 2 * E1_hf
f.close()

calc.diagonalize_full_hamiltonian(nbands=4800)
calc.write('N2.gpw', mode='all')
