# web-page: tas2_eps.png
from ase import Atoms
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.df import DielectricFunction
from gpaw.mpi import world

# 1) Ground-state.

a = 3
atom = Atoms('Na',
             cell=[a, 0, 0],
             pbc=[1, 1, 1])
atom.center(vacuum=1 * a, axis=(1, 2))
atom.center()
atoms = atom.repeat((2, 1, 1))

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.01),
            kpts={'density': 5})

atoms.calc = calc
atoms.get_potential_energy()
calc.write('na-gs.gpw')

# 2) Unoccupied bands

from gpaw.bztools import find_high_symmetry_monkhorst_pack as fhsmp
kpts = fhsmp(atoms=atoms, density=20)

responseGS = GPAW('na-gs.gpw').fixed_density(
    kpts=kpts,
    parallel={'band': 1},
    nbands=60,
    convergence={'bands': 50})

responseGS.get_potential_energy()
responseGS.write('na-gsresponse.gpw', 'all')

# 3) Dielectric function

df = DielectricFunction(
    'na-gsresponse.gpw',
    eta=25e-3,
    rate='eta',
    frequencies={'type': 'nonlinear', 'domega0': 0.01},
    integrationmode='tetrahedron integration')

df1tetra_w, df2tetra_w = df.get_dielectric_function(direction='x',
                                                    filename='df_tetra.csv')

df = DielectricFunction(
    'na-gsresponse.gpw',
    eta=25e-3,
    rate='eta',
    frequencies={'type': 'nonlinear', 'domega0': 0.01})
df1_w, df2_w = df.get_dielectric_function(direction='x',
                                          filename='df_point.csv')
omega_w = df.get_frequencies()

# convolve with gaussian to smooth the curve
sigma = 0.05
df2_wreal_result = gaussian_filter1d(df2_w.real, sigma)
df2_wimag_result = gaussian_filter1d(df2_w.imag, sigma)

if world.rank == 0:
    plt.figure(figsize=(6, 6))
    plt.plot(omega_w, df2tetra_w.real, 'blue', label='tetra Re')
    plt.plot(omega_w, df2tetra_w.imag, 'red', label='tetra Im')
    plt.plot(omega_w, df2_w.real, 'green', label='Re')
    plt.plot(omega_w, df2_w.imag, 'orange', label='Im')
    plt.plot(omega_w, df2_wreal_result, 'pink', label='Inter Re')
    plt.plot(omega_w, df2_wimag_result, 'yellow', label='Inter Im')
    plt.xlabel('Frequency (eV)')
    plt.ylabel('$\\varepsilon$')
    plt.xlim(0, 10)
    plt.ylim(-20, 20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('na_eps.png', dpi=600)
# plt.show()