from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import equal
from gpaw.eigensolvers import RMM_DIIS
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('HYB_GGA_XC_LCY_PBE') is not None

tio2 = Atoms('TiO2', [(0, 0, 0), (0.66, 0.66, 1.34), (0.66, 0.66, -1.34)])
tio2.center(vacuum=4)
ti = Atoms('Ti', [(0, 0, 0)])
ti.center(vacuum=4)
o = Atoms('O', [(0, 0, 0)])
o.center(vacuum=4)

c = {'energy': 0.001, 'eigenstates': 3, 'density': 3}

# gen('H', xcname='PBEsol') - generator comes with local corrections

# Atomization energies are from M. Seth, T. Ziegler, JCTC 8, 901-907
# dx.doi.org/10.1021/ct300006h
# The LCY-PBE value is the same as the PBE0 value, so this might be an
# error.

for xc, dE in [('LCY_BLYP', 143.3)
#               ('LCY_PBE', 149.2),
#               ('CAMY_B3LYP', 147.1)
                ]:
    if not newlibxc:
        print('Skipped')
        continue
    ti.calc = GPAW(txt='Ti-' + xc + '.txt', xc=HybridXC(xc), convergence=c,
            occupations=FermiDirac(width=0.0, fixmagmom=True),
            eigensolver=RMM_DIIS())
    ti.set_initial_magnetic_moments([2.0])
    e_ti = ti.get_potential_energy()
    o.calc = GPAW(txt='O-' + xc + '.txt', xc=HybridXC(xc), convergence=c,
            eigensolver=RMM_DIIS(),
            occupations=FermiDirac(width=0.0, fixmagmom=True))
    o.set_initial_magnetic_moments([2.0])
    e_o = o.get_potential_energy()
    tio2.calc = GPAW(txt='TiO2-' + xc + '.txt', xc=HybridXC(xc),
            eigensolver=RMM_DIIS(), 
            convergence=c, occupations=FermiDirac(width=0.0, fixmagmom=True))
    tio2.set_initial_magnetic_moments([4.0,-2.0,-2.0])
    e_tio2 = tio2.get_potential_energy()
    # dissoziation energy
    e_diss = (e_ti + 2 * e_o - e_tio2)/2.0
    print(xc, e_diss, dE * kcal / mol)
    equal(e_diss, dE*kcal/mol, 0.5)
