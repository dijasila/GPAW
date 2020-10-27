
# Import the required modules: General
import numpy as np


# Import the required modules: GPAW/ASE
from gpaw import GPAW, PW, FermiDirac
from ase.build import mx2
from gpaw.nlopt.shg import get_shg
from gpaw.nlopt.nlobas import make_nlodata
from gpaw.nlopt.basic import is_file_exist

# Test the nlores functions


def test_nlores(in_tmp_dir):

    # Make a test structure and add the vaccum around the layer
    atoms = mx2(formula='MoS2', a=3.16, thickness=3.17, vacuum=5.0)
    atoms.center(vacuum=15, axis=2)

    # GPAW parameters
    nk = 4
    params_gs = dict(
        mode=PW(600),
        symmetry={'point_group': False, 'time_reversal': True},
        nbands='300%',
        convergence={'bands': -10},
        parallel={'domain': 1},
        occupations=FermiDirac(width=0.05),
        kpts={'size': (nk, nk, 1), 'gamma': True},
        xc='PBE',
        txt='gs.txt')

    # SHG calculation params
    eta = 0.05
    w_ls = np.linspace(0.001, 6, 2)  # in eV

    # Start a ground state calculation
    if is_file_exist('gs.gpw'):
        calc = GPAW(**params_gs)
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.calc.write('gs.gpw', mode='all')

    # Compute momentum matrix
    if is_file_exist('mml.npz'):
        make_nlodata()

    # Compute responses
    pols = ['yyy', 'xxy', 'xxx']
    shg_vg = {}
    shg_lg = {}
    for pol in pols:

        res = get_shg(
            freqs=w_ls,
            eta=eta,
            pol=pol,
            out_name='shg_{}_vg.npy'.format(pol))
        shg_vg[pol] = res[1]

        res = get_shg(
            freqs=w_ls,
            eta=eta,
            pol=pol,
            out_name='shg_{}_lg.npy'.format(pol))
        shg_lg[pol] = res[1]

    # Now check the results
    assert np.any(np.abs((shg_vg['yyy'] + shg_vg['xxy'])
                         / (shg_vg['yyy'] - shg_vg['xxy'])) < 1e-3), \
        'Symmtery of VG calculation is not satisfied'
    assert np.any(np.abs(shg_vg['xxx'] / shg_vg['yyy']) < 1e-3), \
        'Symmtery of VG calculation is not satisfied'
    assert np.any(np.abs((shg_lg['yyy'] + shg_lg['xxy'])
                         / (shg_lg['yyy'] - shg_lg['xxy'])) < 1e-1), \
        'Symmtery of LG calculation is not satisfied'
    assert np.any(np.abs(shg_lg['xxx'] / shg_lg['yyy']) < 1e-1), \
        'Symmtery of LG calculation is not satisfied'
