from ase.build import molecule
from gpaw import GPAW, Mixer
from gpaw.xc.libvdwxc import vdw_df

system = molecule('H2O')
system.center(vacuum=1.5)
system.pbc = 1

def calculate(mode):
    kwargs = dict(mode=mode,
                  basis='szp(dzp)',
                  xc=vdw_df(),
                  mixer=Mixer(0.3, 5, 10.))
    calc = GPAW(**kwargs)
    def stopcalc():
        calc.scf.converged = True
    calc.attach(stopcalc, 6)

    system.set_calculator(calc)
    system.get_potential_energy()
    return calc

for mode in ['fd', 'pw', 'lcao']:
    calc = calculate(mode)

E1 = calc.get_potential_energy()
calc.write('dump.libvdwxc.gpw')
calc2 = GPAW('dump.libvdwxc.gpw', txt='restart.txt')
system2 = calc.get_atoms()

# Verify same energy after restart
E2 = system2.get_potential_energy()
assert abs(E2 - E1) < 1e-14  # Should be exact

# Trigger recaclulation of essentially same system
system2.positions[0, 0] += 1e-13
print('reconverge')
E3 = system2.get_potential_energy()
err2 = abs(E3 - E2)
assert err2 < 5e-6  # Around SCF precision
