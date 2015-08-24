from ase.structure import molecule
from gpaw import GPAW, Mixer
from gpaw.xc.libvdwxc import LibVDWXC

system = molecule('H2O')
system.center(vacuum=1.5)
system.pbc = 1

calc = GPAW(eigensolver='rmm-diis',
            mixer=Mixer(0.3, 5, 10.),
            xc=LibVDWXC())

system.set_calculator(calc)
system.get_potential_energy()
