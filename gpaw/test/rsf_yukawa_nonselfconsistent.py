"""Test non-selfconsistent RSF calculation with Yukawa potential."""
from ase import Atoms
from ase.units import kcal, mol
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import equal

h = 0.25
tio2 = Cluster(Atoms('TiO2', positions=[(0, 0, 0), (0.66, 0.66, 1.34),
    (0.66, 0.66, -1.34)]))
tio2.minimal_box(4, h=h)
tio2.translate([0.01, 0.02, 0.03])
ti = Cluster(Atoms('Ti', [(0, 0, 0)]))
ti.minimal_box(4, h=h)
ti.translate([0.01, 0.02, 0.03])
o = Cluster(Atoms('O', [(0, 0, 0)]))
o.minimal_box(4, h=h)
o.translate([0.01, 0.02, 0.03])

c = {'energy': 0.001, 'eigenstates': 4, 'density': 3}

# Dissoziation energies from M. Seth, T. Ziegler, JCTC 8, 901-907
# dx.doi.org/10.1021/ct300006h
# The LCY-PBE value is the same as the PBE0 value, so this might be an
# error.
xc = 'PBE'
ti.calc = GPAW(txt='Ti-' + xc + '.txt', xc=xc, convergence=c,
            occupations=FermiDirac(width=0.0, fixmagmom=True), h=h)
ti.set_initial_magnetic_moments([2.0])
e_ti = ti.get_potential_energy()
o.calc = GPAW(txt='O-' + xc + '.txt', xc=xc, convergence=c,
            occupations=FermiDirac(width=0.0, fixmagmom=True), h=h)
o.set_initial_magnetic_moments([2.0])
e_o = o.get_potential_energy()
tio2.calc = GPAW(txt='TiO2-' + xc + '.txt', xc=xc, convergence=c,
        occupations=FermiDirac(width=0.0, fixmagmom=True), h=h)
tio2.set_initial_magnetic_moments([1.0, -0.5, -0.5])
e_tio2 = tio2.get_potential_energy()

for xc, dE, ediff in [('CAMY_B3LYP', 147.1, 0.15),
                    ('LCY_BLYP', 143.3, 0.25),
                    ('LCY_PBE', 149.2, 0.35)]:
    de_ti = e_ti + ti.calc.get_xc_difference(HybridXC(xc))
    de_o = e_o + o.calc.get_xc_difference(HybridXC(xc))
    de_tio2 = e_tio2 + tio2.calc.get_xc_difference(HybridXC(xc))
    # dissoziation energy
    e_diss = (de_ti + 2 * de_o - de_tio2) / 2.0
    print(xc, e_diss, dE * kcal / mol)
    equal(e_diss, dE * kcal / mol, ediff)
