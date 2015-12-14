from ase import Atoms, io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.structure import molecule
from ase.parallel import barrier

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning
from gpaw.analyse.vdwradii import vdWradii
from gpaw.test import equal

h = 0.4
s = Cluster(molecule('Na2'))
s.minimal_box(3., h=h)

out_traj = 'Na2.traj'
out_txt = 'Na2.txt'
                       
cc = GPAW(h=h, xc='PBE', txt=out_txt)

# this is needed to initialize txt output
cc.initialize(s)

c = vdWTkatchenko09prl(HirshfeldPartitioning(cc),
                       vdWradii(s.get_chemical_symbols(), 'PBE'))
s.set_calculator(c)
E = s.get_potential_energy()
F_ac = s.get_forces()
s.write(out_traj)

barrier()

# test I/O, accuracy due to text output
accuracy = 1.e-5
for fname in [out_traj, out_txt]:
    s_out = io.read(fname)
    ##print s_out.get_potential_energy(), E
    ##print s_out.get_forces()
    equal(s_out.get_potential_energy(), E, accuracy)
    for fi, fo in zip(F_ac, s_out.get_forces()):
        equal(fi, fo, accuracy)
