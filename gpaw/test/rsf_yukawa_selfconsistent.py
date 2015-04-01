"""Test selfconsistent RSF calculation with Yukawa potential. No EXX local corr."""
from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen
from gpaw.eigensolvers import RMM_DIIS
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('HYB_GGA_XC_LCY_PBE') is not None

work_atoms = [
    Atoms('TiO2', [(0, 0, 0), (0.66, 0.66, 1.34), (0.66, 0.66, -1.34)]),
    Atoms('Ti', [(0, 0, 0)]),
    Atoms('O', [(0, 0, 0)])]

for work_atom in work_atoms:
    work_atom.center(vacuum=4)
    work_atom.translate([0.01, 0.02, 0.03])
c = {'energy': 0.001, 'eigenstates': 3, 'density': 3}

# Atomization energies are from M. Seth, T. Ziegler, JCTC 8, 901-907
# dx.doi.org/10.1021/ct300006h
# The LCY-PBE value is the same as the PBE0 value, so this might be an
# error.

calculator = GPAW(convergence=c, eigensolver=RMM_DIIS(),
        occupations=FermiDirac(width=0.0, fixmagmom=True))
for xc, dE, ediff in [('LCY_BLYP', 143.3, 0.35),
               ('LCY_PBE', 149.2, 0.45),
               ('CAMY_B3LYP', 147.1, 0.4)
                ]:
    if not newlibxc:
        print('Skipped')
        continue
    energies = {}
    calculator.set(xc=HybridXC(xc))
    for work_atom in work_atoms:
        name = work_atom.get_chemical_formula()
        calculator.set(txt=name + '-' + xc + '.txt')
        work_atom.calc = calculator
        if name == 'O2Ti':
            work_atom.set_initial_magnetic_moments([2.0, -1.0, -1.0])
            if xc == 'CAMY_B3LYP':
                work_atom.center(vacuum=4)
        else:
            work_atom.set_initial_magnetic_moments([2.0])
        energies[name] = work_atom.get_potential_energy()
    # dissoziation energy
    e_diss = (energies['Ti'] + 2 * energies['O'] - energies['O2Ti']) / 2.0
    print(xc, e_diss, dE * kcal / mol)
    equal(e_diss, dE * kcal / mol, ediff)
