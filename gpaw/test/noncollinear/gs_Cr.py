import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, MixerDif

d = 2.66
atoms = Atoms('Cr3', positions=[(0, 0, 0), (d, 0, 0), (2 * d, 0, 0)],
              cell=[[d * 3 / 2, -d * np.sqrt(3) / 2, 0],
                    [d * 3 / 2, d * np.sqrt(3) / 2, 0],
                    [0, 0, 6]],
              pbc=True)
magmoms = [[3, 3, 0], [3, -1, 0], [-4, 0, 1.0]]

calc = GPAW(mode=PW(400),
            symmetry='off',
            # mixer=MixerDif(),
            experimental={'magmoms': magmoms},
            kpts=(4, 4, 1))
atoms.set_calculator(calc)
atoms.get_potential_energy()
m_v, m_av = calc.density.estimate_magnetic_moments()
v0 = m_av[0]
v1 = m_av[1]
v2 = m_av[2]
a01 = np.dot(v0, v1) / np.dot(v0, v0)**0.5 / np.dot(v1, v1)**0.5
a12 = np.dot(v1, v2) / np.dot(v1, v1)**0.5 / np.dot(v2, v2)**0.5
a20 = np.dot(v2, v0) / np.dot(v2, v2)**0.5 / np.dot(v0, v0)**0.5
b01 = np.arccos(a01) * 180 / np.pi
b12 = np.arccos(a12) * 180 / np.pi
b20 = np.arccos(a20) * 180 / np.pi
print(b01, b12, b20)
calc.write('Cr3.gpw')

calc = GPAW('Cr3.gpw', txt=None)
m_av = calc.density.estimate_magnetic_moments()
print(m_av)
