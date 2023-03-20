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
    wfs_nosym = calc_nosym.wfs
    nbands = wfs.bd.nbands
    nbzk = wfs.kd.nbzkpts
    dtype = wfs.dtype
    ibz2bz = IBZ2BZMaps.from_calculator(calc)
    r_cR = wfs.gd.get_grid_point_coordinates()
    ibz2bz_nosym = IBZ2BZMaps.from_calculator(calc_nosym)


    def find_degenerate_subspace(eps, i, nbands):
        j = i
        dim = 1
        while j < nbands-1 and abs(eps[j]- eps[j+1]) < 1e-6:
            dim += 1
            j +=1
        return dim
    
    for s in range(wfs.nspins):
        for K in range(nbzk):
            assert np.array_equal(wfs.kd.bzk_kc[K], wfs_nosym.kd.bzk_kc[K])
            #ut_nR = calc.wfs.gd.empty(nbands, dtype)
            ik = wfs.kd.bz2ibz_k[K]
            kpt = wfs.kpt_qs[ik][s]
            psit_nG = kpt.psit_nG
            psit_nG_nosym = wfs_nosym.kpt_qs[K][s].psit_nG
            n = 0
            eps_n = kpt.eps_n
            while n < nbands:
                dim = find_degenerate_subspace(eps_n, n, nbands)
                if dim > 1:
                    n += dim
                    continue
                eps = eps_n[n]
                ut_R_IBZ = wfs.pd.ifft(psit_nG[n], ik)
                ut_R = ibz2bz[K].map_pseudo_wave_to_BZ(ut_R_IBZ, r_cR)
                ut_R_nosym = wfs_nosym.pd.ifft(psit_nG_nosym[n], K)
                ut_R = ibz2bz_nosym[K].map_pseudo_wave_to_BZ(ut_R_nosym, r_cR)
                assert(np.allclose(ut_R, ut_R_nosym))
                n += dim
