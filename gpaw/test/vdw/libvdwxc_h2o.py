from ase.build import molecule
from gpaw import GPAW, Mixer
from gpaw.xc.libvdwxc import vdw_df

system = molecule('H2O')
system.center(vacuum=1.5)
system.pbc = 1

for mode in ['lcao', 'fd', 'pw']:
    calc = GPAW(mode=mode,
                basis='szp(dzp)',
                mixer=Mixer(0.3, 5, 10.),
                xc=vdw_df())
    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 6)

    system.set_calculator(calc)
    system.get_potential_energy()
