from ase.structure import molecule
from ase.data.vdw import vdw_radii
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.contributions import (
    MODE_RADII_CUTOFF, MODE_SURFACE_TENSION
    )
from ase.calculators.test import numeric_forces
from ase.parallel import parprint
import numpy


class ReferenceSolvationGPAW(SolvationGPAW):
    def get_forces(self, atoms):
        forces = numeric_forces(atoms)
        return forces


atoms = molecule('H2O')
atoms.set_initial_magnetic_moments(None)
atoms.center(3.)
atoms.rattle(stdev=.2, seed=42)

DFs = []
params = {
    'xc':'PBE',
    'h':.2,
    'solvation':{
        'el':{
            'mode':MODE_RADII_CUTOFF,
            'radii':[vdw_radii[a.number] for a in atoms],
            'exponent':3.4,
            'epsilon_r':80.
            }
        }
    }
cav = {
    'cav':{
        'mode':MODE_SURFACE_TENSION,
        #scale to get non-vanishing forces
        'surface_tension': 0.0045 * 10.
        }
    }

atoms.calc = SolvationGPAW(**params)
atoms.get_potential_energy()
Fel = atoms.get_forces()
atoms.calc.set(solvation=cav)
atoms.get_potential_energy()
Felcav = atoms.get_forces()

if False:
    atoms.calc = ReferenceSolvationGPAW(**params)
    atoms.get_potential_energy()
    Felref = atoms.get_forces()
    atoms.calc.set(solvation=cav)
    atoms.get_potential_energy()
    Felcavref = atoms.get_forces()
else:
    Felref = [[ 1.29276937,  1.48687732, -4.93777752],
              [-0.87351208, -0.67926378,  3.05644592],
              [-0.54559522, -0.7702362,   1.8756392 ]]
    Felcavref = [[ 1.32413255,  1.49849259, -5.05619571],
                 [-0.88826466, -0.78665117,  3.1117918 ],
                 [-0.56220416, -0.67447946,  1.93870498]]

def cmp_forces(F1, F2, tol=.05):
    parprint(F1)
    parprint(F2)
    parprint(F1 - F2)
    parprint(numpy.abs(F1 - F2).max())
    assert numpy.allclose(Fel, Felref, rtol=.0, atol=tol)

parprint('el')
cmp_forces(Felref, Fel)
parprint('cav')
cmp_forces(Felcavref - Felref, Felcav - Fel)
parprint('total')
cmp_forces(Felcavref, Felcav)
parprint('Acav = ', atoms.calc.get_cavity_surface_area())
