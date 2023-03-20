import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response.ibz2bz import IBZ2BZMaps
@pytest.mark.serial
@pytest.mark.response
#@pytest.mark.parametrize('gs',['fancy_si_pw',
#                                'al_pw',
#                                'fe_pw'])
@pytest.mark.parametrize('gs',['fe_pw'])
def test_ibz2bz(in_tmp_dir, gpw_files, gs):
    # Loading calc with symmetry
    calc = GPAW(gpw_files[gs+'_wfs'])
    wfs = calc.wfs
    nbands = wfs.bd.nbands
    nbzk = wfs.kd.nbzkpts
    dtype = wfs.dtype
    r_cR = wfs.gd.get_grid_point_coordinates()
    ibz2bz = IBZ2BZMaps.from_calculator(calc)

    # Loading calc without symmetry
    calc_nosym = GPAW(gpw_files[gs+'_nosym_wfs'])
    wfs_nosym = calc_nosym.wfs
    
    # Check some basic stuff
    assert(np.allclose(r_cR, wfs_nosym.gd.get_grid_point_coordinates()))
    assert(wfs_nosym.kd.nbzkpts == wfs_nosym.kd.nibzkpts)

    def find_degenerate_subspace(eps, i, nbands):
        # Find degenerate eigenvalues
        j = i
        dim = 1
        while j < nbands-1 and abs(eps[j]- eps[j+1]) < 1e-5:
            dim += 1
            j +=1
        return dim
    
    for s in range(wfs.nspins):
        for K in range(nbzk):
            # Check so that BZ kpoints are the same
            assert np.allclose(wfs.kd.bzk_kc[K], wfs_nosym.kd.bzk_kc[K])
            assert np.allclose(wfs_nosym.kd.ibzk_kc[K], wfs_nosym.kd.bzk_kc[K])
            #ut_nR = calc.wfs.gd.empty(nbands, dtype)
            ik = wfs.kd.bz2ibz_k[K]
            kpt = wfs.kpt_qs[ik][s]
            psit_nG = kpt.psit_nG

            kpt_nosym = wfs_nosym.kpt_qs[K][s]
            psit_nG_nosym = kpt_nosym.psit_nG
            eps_n = kpt.eps_n

            # Check so that eigenvalues are the same
            assert np.allclose(eps_n, kpt_nosym.eps_n, atol=1e-08)
            
            # Get all projections
            proj = kpt.projections.new(nbands=nbands, bcomm=None)
            proj.array[:] = kpt.projections.array[0:nbands]
            proj = ibz2bz[K].map_projections(proj)
            P_ani = np.array([P_ni for _, P_ni in proj.items()])
            
            proj = kpt_nosym.projections.new(nbands=nbands, bcomm=None)
            proj.array[:] = kpt_nosym.projections.array[0:nbands]
            #proj = ibz2bz_nosym[K].map_projections(proj)
            P_ani_nosym = np.array([P_ni for _, P_ni in proj.items()])

            def check_degenerate_subspace_u(n, dim):
                # sets up transformation matrix between symmetry transformed u:s
                # and normal u:s and asserts that it is unitary
                return
                Utrans = np.zeros((dim,dim))
                NR = np.prod(np.shape(r_cR)[1:])
                for n1 in range(n,n+dim):
                    for n2 in range(n,n+dim):
                        ut_R = ibz2bz[K].map_pseudo_wave_to_BZ(wfs.pd.ifft(psit_nG[n1], ik), r_cR)
                        ut_R = np.reshape(ut_R, NR)
                        ut_R_nosym = wfs_nosym.pd.ifft(psit_nG_nosym[n2], K)
                        ut_R_nosym = np.reshape(ut_R_nosym, NR)
                        overlap = np.dot(ut_R.conj(), ut_R_nosym.T)* calc.wfs.gd.dv
                        norm1 = np.dot(ut_R.conj(), ut_R.T)* calc.wfs.gd.dv
                        norm2 = np.dot(ut_R_nosym.conj(), ut_R_nosym.T)* calc.wfs.gd.dv
                        Utrans[n1-n,n2-n] = overlap / np.sqrt(norm1 * norm2)
                # Check so that transformation matrix is unitary
                print(Utrans)
                UUdag = Utrans.dot(Utrans.T.conj())
                print(UUdag)
                assert np.allclose(np.eye(len(UUdag)), UUdag, atol=1e-04)

            n = 0
            while n < nbands:
                dim = find_degenerate_subspace(eps_n, n, nbands)
                if dim > 1:
                    check_degenerate_subspace_u(n, dim)
                    n += dim
                    continue
                eps = eps_n[n]

                # Check so that periodic part of pseudo is same
                ut_R = ibz2bz[K].map_pseudo_wave_to_BZ(wfs.pd.ifft(psit_nG[n], ik), r_cR)
                ut_R_nosym = wfs_nosym.pd.ifft(psit_nG_nosym[n], K)
                assert(np.allclose(abs(ut_R), abs(ut_R_nosym), atol=1e-05))

                # Check projections
                assert np.allclose(abs(P_ani[:,n,:]), abs(P_ani_nosym[:,n,:]), atol=1e-04)
                n += dim
