import pytest
import os
import numpy as np
from gpaw import GPAW
import gpaw.wannier90 as w90
from gpaw.wannier.w90 import read_wout_all
from pathlib import Path

@pytest.mark.wannier
@pytest.mark.serial
@pytest.mark.parametrize('mode', ['symm', 'nosymm'])
def test_wannier90(gpw_files, mode, in_tmp_dir):
    o_ai = [[], [0, 1, 2, 3]]
    bands = range(4)
    calc = GPAW(gpw_files[f'gaas_pw_{mode}_wfs'])
    seed = 'GaAs'
    if mode == 'symm':
        assert calc.wfs.kd.nbzkpts > calc.wfs.kd.nibzkpts
    else:
        assert calc.wfs.kd.nbzkpts == calc.wfs.kd.nibzkpts

    w90.write_input(calc, orbitals_ai=o_ai,
                    bands=bands,
                    seed=seed,
                    num_iter=1000,
                    plot=False)
    try:
        os.system('wannier90.x -pp ' + seed)
    except FileNotFoundError:
        return  # no wannier90.x executable
    w90.write_projections(calc, orbitals_ai=o_ai, seed=seed)
    w90.write_eigenvalues(calc, seed=seed)
    w90.write_overlaps(calc, seed=seed)
    os.system('wannier90.x ' + seed)
    with (Path('GaAs.wout')).open() as fd:
        w = read_wout_all(fd)
    centers = np.sum(np.array(w['centers']), axis=0)
    centers_correct = np.array([5.68, 5.68, 5.68])
    assert np.allclose(centers, centers_correct)
    spreads = np.sum(np.array(w['spreads']))
    assert spreads == pytest.approx(4.4999, abs=0.002)
