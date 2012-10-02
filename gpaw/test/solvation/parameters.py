from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.contributions import (
    MODE_ELECTRON_DENSITY_CUTOFF, MODE_RADII_CUTOFF,
    MODE_SURFACE_TENSION, CONTRIBUTIONS
    )

calc0 = SolvationGPAW()
assert calc0.input_parameters['solvation'] == {
    'el' : {'mode':None},
    'rep': {'mode':None},
    'dis': {'mode':None},
    'cav': {'mode':None},
    'tm' : {'mode':None}
    }

params = {
    'el' : {'mode':MODE_RADII_CUTOFF, 'epsilon_r':80.},
    'dis': {},
    'cav': {'mode':MODE_SURFACE_TENSION, 'surface_tension':2.3},
    'tm' : {'mode':None}
    }

def update(d1, d2):
    d1 = d1.copy()
    d1.update(d2)
    return d1

expected = {
    'el' : update(
        CONTRIBUTIONS['el'][MODE_RADII_CUTOFF].default_parameters,
        params['el']
        ),
    'rep': {'mode':None},
    'dis': {'mode':None},
    'cav': update(
        CONTRIBUTIONS['cav'][MODE_SURFACE_TENSION].default_parameters,
        params['cav']
        ),
    'tm' : {'mode':None}
    }

calc1 = SolvationGPAW(solvation=params)
assert calc1.input_parameters['solvation'] == expected

calc0.set(solvation=params)
assert calc0.input_parameters['solvation'] == expected

el_change = {'mode':MODE_ELECTRON_DENSITY_CUTOFF, 'cutoff':1e-5}
calc0.set(solvation={'el':el_change})
assert calc0.input_parameters['solvation'] == update(
    expected,
    {'el': update(
        CONTRIBUTIONS['el'][MODE_ELECTRON_DENSITY_CUTOFF].default_parameters,
        el_change
        )}
    )
