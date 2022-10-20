from ase.vibrations.raman import StaticRamanCalculator
from gpaw import GPAW, FermiDirac
from gpaw.cluster import Cluster
from gpaw.external import static_polarizability


h = 0.2
xc = 'PBE'
atoms = Cluster('relaxed.traj')
atoms.minimal_box(4., h=h)

atoms.calc = GPAW(xc=xc, h=h, 
                  occupations=FermiDirac(width=0.1),
                  symmetry={'point_group': False})
atoms.get_potential_energy()


class Polarizability:
    def __call__(self, atoms):
        return static_polarizability(atoms)

name = exname = 'static_raman'
ram = StaticRamanCalculator(
    atoms, Polarizability, name=name)
ram.run()

ram.summary()
