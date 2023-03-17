import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response.ibz2bz import IBZ2BZMaps
@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('wfs',['fancy_si_pw',
                                'al_pw',
                                'fe_pw'])
def test_ibz2bz(in_tmp_dir, gpw_files, wfs):
    calc = GPAW(gpw_files[wfs+'_wfs'])
    calc_nosym = GPAW(gpw_files[wfs+'_nosym_wfs'])
    wfs = calc.wfs
    nbands = wfs.bd.nbands
    nbzk = wfs.kd.nbzkpts
    dtype = wfs.dtype
    ibz2bz = IBZ2BZMaps.from_calculator(calc)
    r_cR = wfs.gd.get_grid_point_coordinates()
    
    for s in range(wfs.nspins):
        for K in range(nbzk):
            assert np.array_equal(wfs.kd.bzk_kc[K], calc_nosym.wfs.kd.bzk_kc[K])
            #assert np.fix(ibz2bz[K].map_kpoint())
            ut_nR = calc.wfs.gd.empty(nbands, dtype)
            ik = wfs.kd.bz2ibz_k[K]
            kpt = wfs.kpt_qs[ik][s]
            psit_nG = kpt.psit_nG
            for n in range(nbands):
                eps = kpt.eps_n[n]
                #ut_R = ibz2bz[K].map_pseudo_wave_to_BZ(
                #    wfs.pd.ifft(psit_nG[n], ik), r_cR)

                print('test')
