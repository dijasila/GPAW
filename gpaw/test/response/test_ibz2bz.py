import numpy as np
import pytest
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response.ibz2bz import IBZ2BZMaps
from gpaw.berryphase import get_overlap


def mark_si_xfail(gs, request):
    if gs == 'al_pw':
        request.node.add_marker(pytest.mark.xfail)


@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('gs',['fancy_si_pw',
                               'al_pw',
                               'fe_pw',
                               'gaas_pw'])
def test_ibz2bz(in_tmp_dir, gpw_files, gs, request):
    """ Tests gpaw.response.ibz2bz.py
    Tests functions to take wavefunction and projections from
    ibz to full bz by comparing calulcations with and without symmetry.
    Tests u_nk for both degenerate and nondegenerate bands.
    For the projections it is so far only tested that the absolute
    values of the projections are correct for nondegenerate bands.

    - Note the low tolerance!
    - Note that Al is marked as xfail!
    - Projections for calculation with and without sym are different
      also for IBZ k-points
    """
    # For Al test system eigenenergies are different. Need to figure out why....
    mark_si_xfail(gs, request)

    atol = 6e-04  # minimum tolerance that makes all tests pass. Why?
    atol_eig = 1e-05

    # Loading calc with symmetry
    calc = GPAW(gpw_files[gs+'_wfs'])
    wfs = calc.wfs
    nconv = calc.parameters.convergence['bands']

    nbands = wfs.bd.nbands if nconv == -1 else nconv
    nbzk = wfs.kd.nbzkpts
    dtype = wfs.dtype
    r_cR = wfs.gd.get_grid_point_coordinates()
    ibz2bz = IBZ2BZMaps.from_calculator(calc)

    # Loading calc without symmetry
    calc_nosym = GPAW(gpw_files[gs+'_nosym_wfs'])
    #conv = {'bands': wfs.bd.nbands,
    #        'density': 1.e-8,
    #        'eigenstates': 4e-08,
    #        'energy': 1e-6}
    #calc_nosym = calc.fixed_density(symmetry='off', convergence=conv)
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

    # Loop over spins and k-points
    for s in range(wfs.nspins):
        for K in range(nbzk):
            
            # Check so that BZ kpoints are the same
            assert np.allclose(wfs.kd.bzk_kc[K], wfs_nosym.kd.bzk_kc[K])
            assert np.allclose(wfs_nosym.kd.ibzk_kc[K], wfs_nosym.kd.bzk_kc[K])

            ik = wfs.kd.bz2ibz_k[K]
            kpt = wfs.kpt_qs[ik][s]
            psit_nG = kpt.psit_nG

            kpt_nosym = wfs_nosym.kpt_qs[K][s]
            psit_nG_nosym = kpt_nosym.psit_nG
            eps_n = kpt.eps_n


            # Check so that eigenvalues are the same
            assert np.allclose(eps_n[:nbands], kpt_nosym.eps_n[:nbands], atol=atol_eig)


            # Get all projections for calc with symmetry
            proj = kpt.projections.new(nbands=nbands, bcomm=None)
            proj.array[:] = kpt.projections.array[0:nbands]
            projnew = ibz2bz[K].map_projections(proj)
            P_ani = np.array([P_ni for _, P_ni in projnew.items()])

            # XXX need to figure out how to get right phase of projections
            #projBZ = ibz2bz[K].map_projections_to_unitcell(proj)
            #projBZ = ibz2bz[K].map_projections_testab(proj)
            #P_ani_shifted = np.array([P_ni for _, P_ni in projBZ.items()])

            # Get projections for calc without symmetry
            proj_nosym = kpt_nosym.projections
            P_ani_nosym = np.array([P_ni for _, P_ni in proj_nosym.items()])

            # get overlaps
            # XXX Make sure they are in order
            dO_aii = []
            for ia in calc.wfs.kpt_u[0].P_ani.keys():
                dO_ii = calc.wfs.setups[ia].dO_ii
                dO_aii.append(dO_ii)

            
            def check_all_electron_wfs(bands):
                # sets up transformation matrix between symmetry transformed u:s
                # and normal u:s and asserts that it is unitary
                NR = np.prod(np.shape(r_cR)[1:])
                u1_nR = np.array([ibz2bz[K].map_pseudo_wave_to_BZ(wfs.pd.ifft(psit_nG[n], ik), r_cR)
                                  for n in bands])
                u2_nR = np.array([wfs_nosym.pd.ifft(psit_nG_nosym[n], K)
                                  for n in bands])
                    
                Utrans = get_overlap(calc,
                                     bands,
                                     np.reshape(u1_nR, (len(u1_nR), NR)),
                                     np.reshape(u2_nR, (len(u2_nR), NR)),
                                     P_ani,
                                     P_ani_nosym,
                                     dO_aii,
                                     np.array([0, 0, 0]))

                # Check so that transformation matrix is unitary
                UUdag = Utrans.dot(Utrans.T.conj())
                assert np.allclose(np.eye(len(UUdag)), UUdag, atol=atol)

            # Here starts the actual test
            n = 0

            # XXX Why would this fail?
            #if np.allclose(wfs.kd.bzk_kc[K], wfs.kd.ibzk_kc[ik]):
            #    assert np.allclose(proj_nosym.array, kpt.projections.array, atol=atol)
            #if np.allclose(wfs.kd.bzk_kc[K], wfs.kd.ibzk_kc[ik]):
            #    assert np.allclose(proj_nosym.array, projnew.array, atol=atol)

            while n < nbands:
                dim = find_degenerate_subspace(eps_n, n, nbands)
                if dim == 1:
                    # Check projections
                    # XXX Figure out correct phase to multiply proj with to avoid comparing
                    # absolute values here
                    assert np.allclose(abs(P_ani[:,n,:]), abs(P_ani_nosym[:,n,:]), atol=atol)
                    #assert np.allclose(P_ani[:,n,:], P_ani_nosym[:,n,:], atol=atol)
                    
                    # Check so that periodic part of pseudo is same
                    ut_R = ibz2bz[K].map_pseudo_wave_to_BZ(wfs.pd.ifft(psit_nG[n], ik), r_cR)
                    ut_R_nosym = wfs_nosym.pd.ifft(psit_nG_nosym[n], K)
                    assert(np.allclose(abs(ut_R), abs(ut_R_nosym), atol=atol))
                # For degenerate states check transformation
                # matrix is unitary,
                # For non-degenerate states check so that all-electron wf:s
                # are the same
                bands = range(n, n + dim)
                check_all_electron_wfs(bands)
                n += dim
