from ase import Atoms, io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.structure import molecule

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning
from gpaw.analyse.vdwradii import vdWradii
from gpaw.test import equal

h = 0.4
s = Cluster(molecule('Na2'))
s.minimal_box(3., h=h)

                       
cc = GPAW(h=h, xc='PBE')
c = vdWTkatchenko09prl(HirshfeldPartitioning(cc),
                       vdWradii(s.get_chemical_symbols(), 'PBE'))
s.set_calculator(c)
E = s.get_potential_energy()
F_ac = s.get_forces()
s.write('H2.traj')

s_out = io.read('H2.traj')
##print s_out.get_potential_energy(), E
##print s_out.get_forces()
equal(s_out.get_potential_energy(), E)
for fi, fo in zip(F_ac, s_out.get_forces()):
    equal(fi, fo)
