# Creates: tcm_1.12.png, tcm_2.48.png, table_1.12.txt, table_2.48.txt
import numpy as np
from matplotlib import pyplot as plt

from gpaw import GPAW
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.lcaotddft.tcm import TCMPlotter

# Load the objects
calc = GPAW('unocc.gpw', txt=None)
ksd = KohnShamDecomposition(calc, 'ksd.ulm')
dmat = DensityMatrix(calc)
fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')

plt.figure(figsize=(8, 8))

def do(w):
    # Select the frequency and the density matrix
    rho_uMM = fdm.FReDrho_wuMM[w]
    freq = fdm.freq_w[w]
    frequency = freq.freq * au_to_eV
    print('Frequency: %.2f eV' % frequency)
    print('Folding: %s' % freq.folding)

    # Transform the LCAO density matrix to KS basis
    rho_up = ksd.transform(rho_uMM)

    # Photoabsorption decomposition
    dmrho_vp = ksd.get_dipole_moment_contributions(rho_up)
    weight_p = 2 * freq.freq / np.pi * dmrho_vp[0].imag / au_to_eV * 1e5
    print('Total absorption: %.2f eV^-1' % np.sum(weight_p))

    # Print contributions as a table
    table = ksd.get_contributions_table(weight_p, minweight=0.1)
    print(table)
    with open('table_%.2f.txt' % frequency, 'w') as f:
        f.write('Frequency: %.2f eV\n' % frequency)
        f.write('Folding: %s\n' % freq.folding)
        f.write('Total absorption: %.2f eV^-1\n' % np.sum(weight_p))
        f.write(table)

    # Plot the decomposition as a TCM
    de = 0.01
    energy_o = np.arange(-3, 0.1 + 1e-6, de)
    energy_u = np.arange(-0.1, 3 + 1e-6, de)
    plt.clf()
    plotter = TCMPlotter(ksd, energy_o, energy_u, sigma=0.1)
    plotter.plot_TCM(weight_p)
    plotter.plot_DOS(fill={'color': '0.8'}, line={'color': 'k'})
    plotter.plot_TCM_diagonal(freq.freq * au_to_eV, color='k')
    plotter.set_title('Photoabsorption TCM of Na8 at %.2f eV' % frequency)

    # Check that TCM integrates to correct absorption
    tcm_ou = ksd.get_TCM(weight_p, ksd.get_eig_n()[0],
                         energy_o, energy_u, sigma=0.1)
    print('TCM absorption: %.2f eV^-1' % (np.sum(tcm_ou) * de**2))

    # Save the plot
    plt.savefig('tcm_%.2f.png' % frequency)


do(0)
do(1)
