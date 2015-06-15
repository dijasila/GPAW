import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.df import DielectricFunction
from matplotlib import rc

width = 8.6 / 2.54
rc('figure', figsize=(width, width), dpi=800)

NBN = 1
NGr = 1
a = 2.5
c = 3.22

GR = Atoms(symbols='C2', positions=[(0.5*a,-np.sqrt(3)/6*a,0.0),(0.5*a, +np.sqrt(3)/6*a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
GR.set_pbc((True,True,True))

GR2 = GR.copy()
cell = GR2.get_cell()
uv = cell[0]-cell[1]
uv = uv/np.sqrt(np.sum(uv**2.0))
dist = np.array([0.5*a,-np.sqrt(3)/6*a]) - np.array([0.5*a, +np.sqrt(3)/6*a])
dist = np.sqrt(np.sum(dist**2.0))
GR2.translate(uv*dist)

BN = Atoms(symbols='BN', positions=[(0.5*a,-np.sqrt(3)/6*a,0.0),(0.5*a, +np.sqrt(3)/6*a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
BN.set_pbc((True,True,True))

NB = Atoms(symbols='NB', positions=[(0.5*a,-np.sqrt(3)/6*a,0.0),(0.5*a, +np.sqrt(3)/6*a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
NB.set_pbc((True,True,True))

NB.translate([0, 0, (NGr + 1.0 * (1 - NGr % 2)) * c] + uv * dist * (NGr % 2))
BN.translate([0, 0, (NGr + 1.0 * (NGr % 2)) * c] + uv * dist * (NGr % 2))

BNNB = BN + NB
BNNB.set_pbc((True, True, True))
old_cell = BN.get_cell()
old_cell[2, 2] = 2 * c
BNNB.set_cell(old_cell)
BNNB.center()

atoms = BNNB

calc = GPAW(h=0.18,
            mode=PW(400),
            kpts={'density': 2.0, 'gamma': True},
            occupations=FermiDirac(0.01),
            )

atoms.set_calculator(calc)
#atoms.get_potential_energy()
#calc.write('gs.gpw', 'all')

from gpaw.bztools import tesselate_brillouin_zone
ik_kc = tesselate_brillouin_zone('gs.gpw', 4)

#responseGS = GPAW('gs.gpw',
#                  fixdensity=True, kpts=ik_kc,
#                  parallel={'band': 1})

#responseGS.diagonalize_full_hamiltonian(nbands=20, expert=True)
#responseGS.write('gsresponse.gpw', 'all')

df = DielectricFunction('gsresponse.gpw')
df1_w, df2_w = df.get_dielectric_function(q_c=[0, 0, 0])

from matplotlib import pyplot as plt
from ase.units import Hartree

omega_w = df.chi0.omega_w * Hartree
plt.plot(omega_w, df2_w.real, label='Re')
plt.plot(omega_w, df2_w.imag, label='Im')
plt.xlabel('Frequency (eV)')
plt.legend()

plt.savefig('/home/morten/BN_new_code.pdf', bbox_inches='tight')
