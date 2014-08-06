from gpaw import GPAW, PW 
from ase import Atoms
from ase.optimize import QuasiNewton
from gpaw.lrtddft import LrTDDFT

molecule = Atoms('Na2', positions=((0.0, 0.0, 0.0), (2.5, 0.0, 0.0)))

molecule.center(vacuum=6.0)

calc = GPAW(mode='lcao', xc='PBE', basis='dzp')

molecule.set_calculator(calc)

# Find the theoretical bond length:
relax = QuasiNewton(molecule, logfile='qn.log')
relax.run(fmax=0.05)

lr = LrTDDFT(calc, xc='LDA', istart=0, jend=10, nspins=2)
lr.write('Omega_Na2.gz')
