from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.bztools import find_high_symmetry_monkhorst_pack
from gpaw.response.df import DielectricFunction
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

a = 2.5
c = 3.22

def test_gs():
    start_time = time.time()
    gs_file = 'gs.gpw'
    response_file = 'gsresponse.gpw'
    
    # Graphene:
    atoms = Atoms(
        symbols='C2', positions=[(0.5 * a, -np.sqrt(3) / 6 * a, 0.0),
                                 (0.5 * a, np.sqrt(3) / 6 * a, 0.0)],
        cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
              (0.5 * a, 0.5 * 3**0.5 * a, 0),
              (0.0, 0.0, 3.2)],
        pbc=[True, True, False])

    atoms.center(axis=2)

    calc = GPAW(mode=PW(400),
                kpts={'density': 10.0, 'gamma': True},
                occupations=FermiDirac(0.1))

    atoms.calc = calc
    atoms.get_potential_energy()
    gs_time = time.time() - start_time
    calc.write(gs_file)

    start_time = time.time()
    density = 15
    kpts = find_high_symmetry_monkhorst_pack(gs_file, density=density)
    responseGS = GPAW(gs_file).fixed_density(
        kpts=kpts,
        parallel={'band': 1},
        nbands=20,
        occupations=FermiDirac(0.001),
        convergence={'bands': 10})
    responseGS.write(response_file, 'all')
    response_time = time.time() - start_time
    
    start_time = time.time()
    df = DielectricFunction(response_file,
                            eta=25e-3,
                            rate='eta',
                            frequencies={'type': 'nonlinear',
                                         'domega0': 0.1},
                            integrationmode='tetrahedron integration')
    df1_tetra, df2_tetra = df.get_dielectric_function(q_c=[0, 0, 0],
                                                      filename='df_tetra.csv')
    dielectric1_time = time.time() - start_time
    start_time = time.time()

    df = DielectricFunction(response_file,
                            frequencies={'type': 'nonlinear',
                                         'domega0': 0.1},
                            eta=25e-3,
                            rate='eta')
    df1_point, df2_point = df.get_dielectric_function(q_c=[0, 0, 0],
                                                      filename='df_point.csv')

    dielectric2_time = time.time() - start_time
    
    print([gs_time, response_time, dielectric1_time, dielectric2_time])
   
    omega = df.get_frequencies()
    df2_tetra = np.imag(df2_tetra)
    df2_point = np.imag(df2_point)

    # Convolve with Gaussian to smoothen the curve
    slicer = [(freq >= 1.4) and (freq <= 20) for freq in omega] #Do not use frequencies near the w=0 singularity
    sigma = 1.9
    df2_gauss = gaussian_filter1d(df2_point[slicer], sigma)
    
    rms_diff_tetra_point = np.sqrt(np.sum((df2_tetra[slicer]-df2_point[slicer])**2)/np.sum(slicer))
    rms_diff_tetra_gauss = np.sqrt(np.sum((df2_tetra[slicer]-df2_gauss)**2)/np.sum(slicer))
    print(rms_diff_tetra_point)
    print(rms_diff_tetra_gauss)

    # Plot the figures
    plt.figure(figsize=(6, 6))
    plt.plot(omega[slicer], df2_tetra[slicer], label='Tetrahedron')
    plt.plot(omega[slicer], df2_point[slicer], label='Point sampling')
    plt.plot(omega[slicer], df2_gauss, label='Gaussian convolution')

    plt.xlabel('Frequency (eV)')
    plt.ylabel('$\\mathrm{Im}\\varepsilon$')
    plt.xlim(0, 6)
    plt.ylim(0, 20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphene_eps.png', dpi=300)

    assert rms_diff_tetra_point < 1.45
    assert rms_diff_tetra_point < 1.10
    assert rms_diff_tetra_point * 0.8 > rms_diff_tetra_gauss

    from gpaw.test import findpeak
    freq1, amp1 = findpeak(omega[slicer], df2_tetra[slicer])
    freq2, amp2 = findpeak(omega[slicer], df2_gauss)
    print([freq1, freq2, I1, I2])

test_gs()
