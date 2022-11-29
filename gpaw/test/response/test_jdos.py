# General modules
import pytest
import numpy as np
from itertools import product

# Script modules
from ase.units import Hartree

from gpaw import GPAW
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.jdos import JDOSCalculator

import matplotlib.pyplot as plt


def test_iron_jdos(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    wd = FrequencyDescriptor.from_array_or_dict(np.linspace(-10.0, 10.0, 321))
    eta = 0.2

    spincomponent_s = ['00', '+-']

    # ---------- Script ---------- #

    # Get the ground state calculator from the fixture
    calc = GPAW(gpw_files['fe_pw_wfs'], parallel=dict(domain=1))
    nbands = calc.parameters.convergence['bands']

    # Set up the JDOSCalculator
    gs = ResponseGroundStateAdapter(calc)
    jdos_calc = JDOSCalculator(gs)

    for q_c, spincomponent in product(q_qc, spincomponent_s):
        jdos_w = jdos_calc.calculate(q_c, wd,
                                     eta=eta,
                                     spincomponent=spincomponent,
                                     nbands=nbands)
        plt.subplot()
        plt.plot(wd.omega_w * Hartree, jdos_w)
        plt.title(f'{q_c} {spincomponent}')
        plt.show()
