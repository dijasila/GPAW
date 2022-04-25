import numpy as np
from ase import Atoms
import ase.units as units
from gpaw import GPAW, PW
from gpaw.hyperfine import hyperfine_parameters

h = Atoms('H', magmoms=[1])
h.center(vacuum=3)
h.calc = GPAW(mode=PW(400), txt=None)
e = h.get_potential_energy()
A = hyperfine_parameters(h.calc)[0] * 5.586
a = np.trace(A) / 3
frequency = a * units._e / units._hplanck  # Hz
wavelength = units._c / frequency  # meters
print(f'{wavelength * 100:.1f} cm')
assert abs(wavelength - 0.232) < 0.0005
