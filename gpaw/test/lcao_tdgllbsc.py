from ase import Atoms
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.test import equal

# silane
atoms = Atoms('SiH4',[( 0.0000,  0.0000,  0.0000),
                      ( 0.8544,  0.8544,  0.8544),
                      (-0.8544, -0.8544,  0.8544),
                      ( 0.8544, -0.8544, -0.8544),
                      (-0.8544,  0.8544, -0.8544)])
atoms.center(vacuum=4.0)

calc = GPAW(nbands  = 5,
            h       = 0.40,
            mode    = 'lcao',
            xc      = 'GLLBSC',
            basis   = 'dzp',
            dtype   = complex)

atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

calc2 = LCAOTDDFT('gs.gpw')
calc2.kick_strength = [0.0, 0.0, 0.0] # hack for zero kicks

# make sure that the dipole moment stays at zero
calc2.attach(equal, 1, calc2.density.finegd.calculate_dipole_moment(calc2.density.rhot_g), 0.0, 1.0e-6)

calc2.propagate(10.0, 50, 'dm.dat')
