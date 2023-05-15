# web-page: tas2_eps.png
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
import matplotlib.pyplot as plt

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.df import DielectricFunction
from gpaw.mpi import world

# 1) Ground-state.

from ase.io import read
a = 4.23 / 2.0
atoms = Atoms('Na', scaled_positions=[[0, 0, 0]], cell=(a, a, a), pbc=True)

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.01),
            kpts={'density': 5})

atoms.calc = calc
atoms.get_potential_energy()
calc.write('na-gs.gpw')

# 2) Unoccupied bands

def get_kpts_size(atoms, kptdensity):
    """Try to get a reasonable Monkhorst-Pack grid which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, offset = k2so(atoms=atoms, density=kptdensity)

    # XXX Should fix kpts2sizeandoffsets
    for i in range(3):
        if not atoms.pbc[i]:
            size[i] = 1
            offset[i] = 0

    for i in range(len(size)):
        if size[i] % 6 != 0 and size[i] != 1:  # works for hexagonal cells XXX
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts
kpts = get_kpts_size(atoms, kptdensity=10)

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

df1tetra_w, df2tetra_w = df.get_dielectric_function(direction='x')

df = DielectricFunction(
    'na-gsresponse.gpw',
    eta=25e-3,
    rate='eta',
    frequencies={'type': 'nonlinear', 'domega0': 0.01})
df1_w, df2_w = df.get_dielectric_function(direction='x')
omega_w = df.get_frequencies()

if world.rank == 0:
    plt.figure(figsize=(6, 6))
    plt.plot(omega_w, df2tetra_w.real, label='tetra Re')
    plt.plot(omega_w, df2tetra_w.imag, label='tetra Im')
    plt.plot(omega_w, df2_w.real, label='Re')
    plt.plot(omega_w, df2_w.imag, label='Im')
    plt.xlabel('Frequency (eV)')
    plt.ylabel('$\\varepsilon$')
    plt.xlim(0, 10)
    plt.ylim(-20, 20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('na_eps.png', dpi=600)
# plt.show()