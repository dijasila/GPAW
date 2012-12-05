from ase.structure import molecule
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.contributions import (
    MODE_ELECTRON_DENSITY_CUTOFF,
    MODE_SURFACE_TENSION,
    MODE_AREA_VOLUME_VDW
    )

h = .3

water = molecule('H2O')
water.center(3.0)
from gpaw import GPAW
water.calc = GPAW(xc='PBE', h=h)
E = water.get_potential_energy()
F = water.get_forces()

water.calc = SolvationGPAW(
    xc='PBE', h=h,
    solvation={
        'el':{
            # parameters from Andreusi et al. J Chem Phys 136, 064102 (2012)
            'rho_min':0.0001,
            'rho_max':0.005,
            'mode':MODE_ELECTRON_DENSITY_CUTOFF
            },
        'cav':{
            'mode':MODE_SURFACE_TENSION,
            'surface_tension': .0
            },
        'dis':{
            'mode':MODE_AREA_VOLUME_VDW,
            'surface_tension': .0,
            'pressure': .0
            }
        }
    )
Esol = water.get_potential_energy()
#Fsol = water.get_forces()

assert Esol ==  E
#assert (F == Fsol).all()
assert Esol == water.calc.get_electrostatic_energy()
assert water.calc.get_repulsion_energy() == .0
assert water.calc.get_dispersion_energy() == .0
assert water.calc.get_cavity_formation_energy() == .0
assert water.calc.get_thermal_motion_energy() == .0
