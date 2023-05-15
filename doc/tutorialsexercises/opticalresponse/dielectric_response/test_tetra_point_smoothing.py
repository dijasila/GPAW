import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.bztools import find_high_symmetry_monkhorst_pack
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


a = 2.5
c = 3.22
filename='gs.gpw'
density=30

def test_gs(filename=filename):
    # Graphene:
    atoms = Atoms(
        symbols='C2', positions=[(0.5 * a, -np.sqrt(3) / 6 * a, 0.0),
                                 (0.5 * a, np.sqrt(3) / 6 * a, 0.0)],
        cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
              (0.5 * a, 0.5 * 3**0.5 * a, 0),
              (0.0, 0.0, c * 2.0)],
        pbc=[True, True, False])
    # (Note: the cell length in z direction is actually arbitrary)

    atoms.center(axis=2)

    calc = GPAW(h=0.18,
                mode=PW(400),
                kpts={'density': 10.0, 'gamma': True},
                occupations=FermiDirac(0.1))

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(filename)


def response_function(gs_gpw=filename, density=density,
                      filename='gsresponse.gpw'):
    kpts = find_high_symmetry_monkhorst_pack(gs_gpw, density=density)
    responseGS = GPAW(gs_gpw).fixed_density(
        kpts=kpts,
        parallel={'band': 1},
        nbands=30,
        occupations=FermiDirac(0.001),
        convergence={'bands': 20})

    responseGS.write(filename, 'all')


def test_multi_kpd_response(kpd=[10, 20]):
    for k in kpd:
        response_function(gs_gpw=filename, density=k,
                          filename=f'gsresponse-{k}.gpw')


def test_df_csv():
    df_tetra = np.loadtxt('df_tetra.csv', delimiter=',')
    df_point = np.loadtxt('df_point.csv', delimiter=',')

    # convolve with gaussian to smooth the curve
    sigma = 7
    df2_wimag_result = gaussian_filter1d(df_point[:, 4], sigma)

    plt.figure(figsize=(6, 6))
    plt.plot(df_tetra[:, 0], df_tetra[:, 4] * 2, label='Img Tetrahedron')
    plt.plot(df_point[:, 0], df_point[:, 4] * 2, label='Img Point sampling')
    plt.plot(df_point[:, 0], df2_wimag_result * 2, 'magenta', label='Inter Im')

    plt.xlabel('Frequency (eV)')
    plt.ylabel('$\\mathrm{Im}\\varepsilon$')
    plt.xlim(0, 6)
    plt.ylim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphene_eps.png', dpi=600)

    from gpaw.test import findpeak
    w1, I1 = findpeak(df_point[200:, 0], df_point[200:, 4])
    w2, I2 = findpeak(df_tetra[200:, 0], df_tetra[200:, 4])
    w3, I3 = findpeak(df_point[200:, 0], df2_wimag_result[200:])
    print(w1, I1)
    print(w2, I2)
    print(w3, I3)

