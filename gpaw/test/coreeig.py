#!/usr/bin/env python
from ase import *
from gpaw import GPAW, restart, extra_parameters
from gpaw.test import equal

usenewxc = extra_parameters.get('usenewxc')
extra_parameters['usenewxc'] = True

from gpaw.utilities.kspot import CoreEigenvalues
try:
    a = 7.0
    calc = GPAW(h=0.1)
    system = Atoms('Ne', calculator=calc)
    system.center(vacuum=a / 2)
    e0 = system.get_potential_energy()
    niter0 = calc.get_number_of_iterations()
    calc.write('Ne.gpw')

    del calc, system

    atoms, calc = restart('Ne.gpw')
    calc.restore_state()
    e_j = CoreEigenvalues(calc).get_core_eigenvalues(0)
    assert abs(e_j[0] - (-30.344066)) * 27.21 < 0.1 # Error smaller than 0.1 eV

    energy_tolerance = 0.0004
    equal(e0, -0.0107707223, energy_tolerance)
except:
    extra_parameters['usenewxc'] = usenewxc
    raise
else:
    extra_parameters['usenewxc'] = usenewxc
