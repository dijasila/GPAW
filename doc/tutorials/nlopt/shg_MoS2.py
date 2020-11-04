import numpy as np
from ase.build import mx2
from gpaw import GPAW, FermiDirac
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.nlopt.shg import get_shg

# Make the structure and add the vaccum around the layer
atoms = mx2(formula='MoS2', a=3.16, thickness=3.17, vacuum=5.0)
atoms.center(vacuum=15, axis=2)

# GPAW parameters:
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

# Ground state calculation:
gs_name = 'gs.gpw'
calc = GPAW(**params_gs)
atoms.calc = calc
atoms.get_potential_energy()
atoms.calc.write(gs_name, mode='all')

# Calculate momentum matrix:
mml_name = 'mml.npz'
make_nlodata(gs_name=gs_name, out_name=mml_name)

# Shift parameters:
eta = 0.05  # Broadening in eV
w_ls = np.linspace(0, 6, 500)  # in eV
pol = 'yyy'

# LG calculation
shg_name1 = 'shg_' + pol + '_lg.npy'
get_shg(
    freqs=w_ls, eta=eta, pol=pol, gauge='lg',
    out_name=shg_name1, mml_name=mml_name)

# VG calculation
shg_name2 = 'shg_' + pol + '_vg.npy'
get_shg(
    freqs=w_ls, eta=eta, pol=pol, gauge='vg',
    out_name=shg_name2, mml_name=mml_name)
