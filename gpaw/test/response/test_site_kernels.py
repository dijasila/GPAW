"""Test if site-kernels give the right number of elements and overall scale."""


# Import modules
from gpaw import GPAW, PW
from ase.units import Bohr
from My_classes.Calculator_classes import StaticChiKSFactory
from My_functions.Calc_site_kernels import calc_K_mixed_shapes
from gpaw.response.susceptibility import get_pw_coordinates
from gpaw.test import equal
from ase.build import bulk
import numpy as np
