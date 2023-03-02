from ase import optimize
from ase.build import molecule
from ase.vibrations.infrared import Infrared
from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster

h = 0.20

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(4, h=h)

# relax the molecule
calc = GPAW(xc='PBE', h=h, occupations=FermiDirac(width=0.1))
atoms.calc = calc

dyn = optimize.FIRE(atoms)
dyn.run(fmax=0.01)
atoms.write('relaxed.traj')

# finite displacement for vibrations
atoms.calc.set(symmetry={'point_group': False})
ir = Infrared(atoms, name='ir')
ir.run()
