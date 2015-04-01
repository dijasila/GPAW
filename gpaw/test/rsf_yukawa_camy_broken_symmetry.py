from ase import Atoms
from gpaw import GPAW, KohnShamConvergenceError
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import equal
from gpaw.eigensolvers import RMM_DIIS
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('HYB_GGA_XC_LCY_PBE') is not None

tio2 = Atoms('TiO2', [(0, 0, 0), (0.66, 0.66, 1.34), (0.66, 0.66, -1.34)])
tio2.center(vacuum=4)
tio2.translate([0.01, 0.02, 0.03])

c = {'energy': 0.001, 'eigenstates': 3, 'density': 3}

# gen('H', xcname='PBEsol') - generator comes with local corrections

# Atomization energies are from M. Seth, T. Ziegler, JCTC 8, 901-907
# dx.doi.org/10.1021/ct300006h
# The LCY-PBE value is the same as the PBE0 value, so this might be an
# error.

tio2.calc = GPAW(txt='TiO2-CAMY-B3LYP-BS.txt', xc=HybridXC('CAMY_B3LYP'),
            eigensolver=RMM_DIIS(), maxiter=42,
            convergence=c, occupations=FermiDirac(width=0.0, fixmagmom=True))
tio2.set_initial_magnetic_moments([2.0, -1.0, -1.0])
try:
    e_tio2 = tio2.get_potential_energy()
except KohnShamConvergenceError:
    pass
# dissoziation energy
print(tio2.calc.scf.converged)
assert tio2.calc.scf.converged
