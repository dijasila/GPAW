# Import the required modules: General
import numpy as np
import matplotlib.pyplot as plt
from ase.build import mx2
from gpaw import GPAW, FermiDirac
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.nlopt.basic import is_file
from gpaw.nlopt.shg import get_shg

# P1
# Make the structure and add the vaccum around the layer
atoms = mx2(formula='MoS2', a=3.16, thickness=3.17, vacuum=5.0)
atoms.center(vacuum=15, axis=2)

# GPAW parameters
nk = 40
params_gs = dict(
    mode='lcao',
    symmetry={'point_group': False, 'time_reversal': True},
    nbands='nao',
    convergence={'bands': -10},
    parallel={'domain': 1},
    occupations=FermiDirac(width=0.05),
    kpts={'size': (nk, nk, 1), 'gamma': True},
    xc='PBE',
    txt='gs.txt')

# Start a ground state calculation, if that has not been done earlier
resetc = True
gs_name = 'gs.gpw'
if is_file_exist(gs_name) or resetc:
    calc = GPAW(**params_gs)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.calc.write(gs_name, mode='all')

# P2
# The momentum matrix are calculated if not available
mml_name = 'mml.npz'
if is_file_exist(mml_name) or resetc:
    make_nlodata(gs_name=gs_name, out_name=mml_name)

# P3
# Shift parameters
eta = 0.05  # Broadening in eV
w_ls = np.linspace(0, 6, 500)  # in eV
pol = 'yyy'

# LG calculation
shg_name1 = 'shg_' + pol + '_lg.npy'
if is_file_exist(shg_name1) or resetc:
    get_shg(
        freqs=w_ls, eta=eta, pol=pol, gauge='lg',
        out_name=shg_name1, mml_name=mml_name)

# VG calculation
shg_name2 = 'shg_' + pol + '_vg.npy'
if is_file_exist(shg_name2) or resetc:
    get_shg(
        freqs=w_ls, eta=eta, pol=pol, gauge='vg',
        out_name=shg_name2, mml_name=mml_name)

# P4
# Plot and save both spectra
cell = atoms.get_cell()
cellsize = atoms.get_cell_lengths_and_angles()
mult = cellsize[2] * 1e-10  # make the sheet sus.
legls = []
res_name = [shg_name1, shg_name2]
plt.figure(figsize=(6.0, 4.0), dpi=300)
for ii, name in enumerate(res_name):
    # Load the data
    shg = np.load(name)
    w_l = shg[0]
    plt.plot(np.real(w_l), np.real(mult * shg[1] * 1e18), '-')
    plt.plot(np.real(w_l), np.imag(mult * shg[1] * 1e18), '--')
    legls.append('{}: Re'.format(name))
    legls.append('{}: Im'.format(name))
    plt.xlabel(r'$\hbar\omega$ (eV)')
    plt.ylabel(r'$\chi_{}$ (nm$^2$/V)'.format(pol))
    plt.legend(legls, ncol=2)
plt.tight_layout()
plt.savefig('shg.png', dpi=300)
