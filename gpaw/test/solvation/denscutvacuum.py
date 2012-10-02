from ase.structure import molecule
from gpaw import GPAW
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.contributions import (
    MODE_ELECTRON_DENSITY_CUTOFF,
    MODE_SURFACE_TENSION
    )

h = .3

water = molecule('H2O')
water.center(3.0)
water.calc = GPAW(xc='PBE', h=h)
E = water.get_potential_energy()
F = water.get_forces()

water.calc = SolvationGPAW(
    xc='PBE', h=h,
    solvation={
        'el':{
            'mode':MODE_ELECTRON_DENSITY_CUTOFF,
            'cutoff':5.e-4,
            'exponent':2.,
            'epsilon_r':1.0
            },
        'cav':{
            'mode':MODE_SURFACE_TENSION,
            'surface_tension': .0
            }
        }
    )
Esol = water.get_potential_energy()
Fsol = water.get_forces()

assert Esol == E
assert (F == Fsol).all()
assert Esol == water.get_electrostatic_energy()
assert water.get_repulsion_energy() == .0
assert water.get_dispersion_energy() == .0
assert water.get_cavity_formation_energy() == .0
assert water.get_thermal_motion_energy() == .0
