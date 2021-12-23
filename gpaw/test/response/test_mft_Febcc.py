# Import modules
from gpaw import GPAW, PW, FermiDirac
from gpaw.test import equal
from ase.build import bulk
import numpy as np
from My_classes.Exchange_calculator import IsotropicExchangeCalculator, \
    compute_magnon_energy_simple, compute_magnon_energy_FM
