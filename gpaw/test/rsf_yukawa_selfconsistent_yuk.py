"""Test selfconsistent RSF calculation with Yukawa potential including vc."""
from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW
from gpaw.xc.hybrid import HybridXC
from gpaw.poisson import PoissonSolver
from gpaw.xc import XC
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen
from gpaw.eigensolvers import RMM_DIIS
from gpaw.cluster import Cluster
import _gpaw

newlibxc = _gpaw.lxcXCFuncNum('HYB_GGA_XC_LCY_PBE') is not None
h = 0.25

work_atoms = []
for work_atom in [
        Atoms('TiO2', [(0, 0, 0), (0.66, 0.66, 1.34), (0.66, 0.66, -1.34)]),
        Atoms('Ti', [(0, 0, 0)]),
        Atoms('O', [(0, 0, 0)])]:
    work_atom = Cluster(work_atom)
    work_atom.minimal_box(4, h=h)
    work_atom.translate([0.01, 0.02, 0.03])
    work_atoms.append(work_atom)
c = {'energy': 0.001, 'eigenstates': 3, 'density': 3}

# Atomization energies are from M. Seth, T. Ziegler, JCTC 8, 901-907
# dx.doi.org/10.1021/ct300006h
# The LCY-PBE value is the same as the PBE0 value, so this might be an
# error.

calculator = GPAW(convergence=c, eigensolver=RMM_DIIS(),
        poissonsolver=PoissonSolver(use_charge_center=True),
        occupations=FermiDirac(width=0.0, fixmagmom=True), h=h)
for xc, dE, ediff, yuk_gamma in [('LCY_BLYP', 143.3, 0.4, 0.75),
#               ('LCY_PBE', 149.2, 0.4, 0.75),
#               ('CAMY_B3LYP', 147.1, 0.35, 0.34)
                ]:
    if not newlibxc:
        print('Skipped')
        continue
    for atom in ['Ti', 'O']:
        gen(atom, xcname='PBE', scalarrel=False, exx=True,
                yukawa_gamma=yuk_gamma, gpernode=149)  # magic ...
    energies = {}
    calculator.set(xc=HybridXC(xc))
    for work_atom in work_atoms:
        name = work_atom.get_chemical_formula()
        calculator.set(txt=name + '-' + xc + '.txt')
        work_atom.calc = calculator
        if name == 'O2Ti':  #  Speed up a little
            work_atom.set_initial_magnetic_moments([2.0, -1.0, -1.0])
            calculator.set(xc=XC('PBE'))
            work_atom.get_potential_energy()
            calculator.set(xc=HybridXC(xc))
        else:
            work_atom.set_initial_magnetic_moments([2.0])
        energies[name] = work_atom.get_potential_energy()
    # dissoziation energy
    e_diss = (energies['Ti'] + 2 * energies['O'] - energies['O2Ti']) / 2.0
    print(xc, e_diss, dE * kcal / mol)
    equal(e_diss, dE * kcal / mol, ediff)
