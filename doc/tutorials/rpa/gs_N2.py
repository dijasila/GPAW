from ase import *
from ase.structure import molecule
from ase.parallel import paropen
from gpaw import *
from gpaw.wavefunctions.pw import PW
from gpaw.xc.hybridk import HybridXC

# N -------------------------------------------

N = molecule('N')
N.set_pbc(True)
N.set_cell((6., 6., 7.))
N.center()
calc = GPAW(mode=PW(500),
            dtype=complex,
            nbands=8,
            maxiter=300,
            xc='PBE',
            hund=True,
            txt='N.txt',
            convergence={'density': 1.e-6})

N.set_calculator(calc)
N.set_calculator(calc)
E1_pbe = N.get_potential_energy()
E1_hf = E1_pbe + calc.get_xc_difference(HybridXC('EXX'))

calc.diagonalize_full_hamiltonian(nbands=4800)
calc.write('N.gpw', mode='all')

# N2 ------------------------------------------

N2 = molecule('N2')
N2.set_pbc(True)
N2.set_cell((6., 6., 7.))
N2.center()
calc = GPAW(mode=PW(500),
            dtype=complex,
            maxiter=300,
            xc='PBE',
            txt='N2.txt',
            convergence={'density': 1.e-6})

N2.set_calculator(calc)
dyn = optimize.BFGS(N2)
dyn.run(fmax=0.05)
E2_pbe = N2.get_potential_energy()
E2_hf = E2_pbe + calc.get_xc_difference(HybridXC('EXX'))
f = paropen('PBE_HF.dat', 'w')
print >> f, 'PBE: ', E2_pbe - 2*E1_pbe
print >> f, 'HF: ', E2_hf - 2*E1_hf
f.close()

calc.diagonalize_full_hamiltonian(nbands=4800)
calc.write('N2.gpw', mode='all')
