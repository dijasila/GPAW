"""Test selfconsistent RSF calculation with Yukawa potential."""
from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW
from gpaw.xc.hybrid import HybridXC
from gpaw.xc import XC
from gpaw.poisson import PoissonSolver
from gpaw.occupations import FermiDirac
from gpaw.cluster import Cluster
from gpaw.test import equal
from gpaw.eigensolvers import RMMDIIS

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
c = {'energy': 0.01, 'eigenstates': 3, 'density': 3}

# Atomization energies are from M. Seth, T. Ziegler, JCTC 8, 901-907
# dx.doi.org/10.1021/ct300006h
# The LCY-PBE value is the same as the PBE0 value, so this might be an
# error.

calculator = GPAW(convergence=c, eigensolver=RMMDIIS(),
                  occupations=FermiDirac(width=0.0, fixmagmom=True),
                  poissonsolver=PoissonSolver(use_charge_center=True), h=h)
for xc, dE, ediff in [  # ('LCY-BLYP', 143.3, 0.3),
                        # ('CAMY-B3LYP', 147.1, 0.25),
                        ('LCY-PBE', 149.2, 0.7)]:
    energies = {}
    calculator.set(xc=HybridXC(xc))
    for work_atom in work_atoms:
        name = work_atom.get_chemical_formula()
        calculator.set(txt=name + '-' + xc + '.txt')
        work_atom.calc = calculator
        if name == 'O2Ti':   # Speed up a litte, help CAMY
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
