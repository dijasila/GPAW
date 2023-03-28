import numpy as np
import pytest
from gpaw import GPAW
from gpaw.response.ibz2bz import IBZ2BZMaps
from gpaw.berryphase import get_overlap


def mark_xfail(gs, request):
    if gs in ['al_pw', 'fe_pw', 'co_pw']:
        request.node.add_marker(pytest.mark.xfail)


def find_degenerate_subspace(eps_n, n_start, nbands, atol_eig):
    # Find degenerate eigenvalues
    n = n_start
    dim = 1
    while n < nbands - 1 and abs(eps_n[n] - eps_n[n + 1]) < atol_eig:
        dim += 1
        n += 1
    return dim

def compare_P_ani(P_ani, P_ani_nosym, n, atol):
    # compares so that projections at given k and band index n
    # differ by at most a global phase
    phase = 0
    for a in range(len(P_ani)):
        P_ni = P_ani[a]
        for j in range(len(P_ni[n, :])):
            # Only compare elements with finite value
            if abs(P_ni[n, j]) > 0.0001:
                newphase = P_ni[n, j] / P_ani_nosym[a, n, j]
                if phase != 0.0:
                    assert np.allclose(newphase, phase, atol=atol)
                phase = newphase
    assert np.allclose(abs(phase), 1.0, atol=atol)

def get_overlaps_from_setups(wfs):
    dO_aii = []
    for ia in wfs.kpt_u[0].P_ani.keys():
        dO_ii = wfs.setups[ia].dO_ii
        dO_aii.append(dO_ii)
    return dO_aii

def check_all_electron_wfs(calc, bands, NR, u1_nR, u2_nR, P_ani, P_ani_nosym, dO_aii, atol):
    """sets up transformation matrix between symmetry
       transformed u:s and normal u:s in degenerate subspace
       and asserts that it is unitary
    
    Parameters
    ---------
    calc: GPAW calculator object
    bands: int list
         band indexes in degenerate subspace
    NR: int 
        Total number of real-space grid points
    u1_nR: np.array
    u2_nR: np.array
        Periodic part of pseudo wave function for two calculations
    P_ani: np.array
    P_nosym: np.array
        Projections for two calculations
    dO_aii: list of np.arrays
       see get_overlaps_from_setups
    atol: float
       absolute tolerance when comparing arrays
    """

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

    # Check so that Utrans transforms pseudo wf:s
    # Row/column definition of indexes in einsum comes from
    # definition of Utrans
    u_transformed = np.einsum('ji,jklm->iklm', Utrans, u1_nR)
    assert np.allclose(u_transformed, u2_nR)

@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('gs', ['fancy_si_pw',
                                'al_pw',
                                'fe_pw',
                                'gaas_pw',
                                'co_pw'])
@pytest.mark.parametrize('only_ibz_kpts', [True, False])
# paramaterize for different gs calcs.
# XXX while testing also  parametrize for restricting the test
# to IBZ k-points. When all x-fails have been resolved we can remove
# this parametrization.
def test_ibz2bz(in_tmp_dir, gpw_files, gs, only_ibz_kpts, request):
    """ Tests gpaw.response.ibz2bz.py
    Tests functionalities to take wavefunction and projections from
    ibz to full bz by comparing calculations with and without symmetry.

    - Note that Al,  Fe and Co are  marked as xfail!
    - For Al eigenvalues are different. Probably due to k-mesh.
      See "Note your k-mesh ..." in gs output files.
    - For Fe and Co projections between two calculations differ by more than a
      phase. This is the case for both IBZ-k and symmetry related k, so it is
      a convention of the gs calculation?
    """
    
    # Al, Fe and Co fails. Need to figure out why (see above)
    mark_xfail(gs, request)

    # can set individual tolerance for eigenvalues
    atol = 1e-05
    atol_eig = 1e-05

    # Loading calc with symmetry
    calc = GPAW(gpw_files[gs + '_wfs'])
    wfs = calc.wfs
    nconv = calc.parameters.convergence['bands']

    # setting basic stuff
    nbands = wfs.bd.nbands if nconv == -1 else nconv
    nbzk = wfs.kd.nbzkpts
    r_cR = wfs.gd.get_grid_point_coordinates()
    ibz2bz = IBZ2BZMaps.from_calculator(calc)

    # Loading calc without symmetry
    calc_nosym = GPAW(gpw_files[gs + '_nosym_wfs'])
    wfs_nosym = calc_nosym.wfs
    
    # Check some basic stuff
    assert np.allclose(r_cR, wfs_nosym.gd.get_grid_point_coordinates())
    assert wfs_nosym.kd.nbzkpts == wfs_nosym.kd.nibzkpts
    assert nconv == calc_nosym.parameters.convergence['bands']
    
    # Loop over spins and k-points
    for s in range(wfs.nspins):
        for K in range(nbzk):
            ik = wfs.kd.bz2ibz_k[K]  # IBZ k-point

            # if only_ibz_kpts fixture is true only test ibz k-points
            if only_ibz_kpts and not np.allclose(wfs.kd.bzk_kc[K],
                                              wfs.kd.ibzk_kc[ik]):
                continue

            # Check so that BZ kpoints are the same
            assert np.allclose(wfs.kd.bzk_kc[K], wfs_nosym.kd.bzk_kc[K])
            assert np.allclose(wfs_nosym.kd.ibzk_kc[K], wfs_nosym.kd.bzk_kc[K])

            # Get data for calc with symmetry
            kpt = wfs.kpt_qs[ik][s]
            psit_nG = kpt.psit_nG
            eps_n = kpt.eps_n

            # Get data for calc without symmetry
            kpt_nosym = wfs_nosym.kpt_qs[K][s]
            psit_nG_nosym = kpt_nosym.psit_nG

            # Check so that eigenvalues are the same
            assert np.allclose(eps_n[:nbands],
                               kpt_nosym.eps_n[:nbands],
                               atol=atol_eig)

            # Get all projections for calc with symmetry
            proj = kpt.projections.new(nbands=nbands, bcomm=None)
            proj.array[:] = kpt.projections.array[0:nbands]
            projnew = ibz2bz[K].map_projections(proj)
            P_ani = np.array([P_ni for _, P_ni in projnew.items()])

            # Get projections for calc without symmetry
            proj_nosym = kpt_nosym.projections
            P_ani_nosym = np.array([P_ni for _, P_ni in proj_nosym.items()])

            # get overlaps
            dO_aii = get_overlaps_from_setups(wfs)
            assert np.allclose(np.array(dO_aii),
                               np.array(get_overlaps_from_setups(wfs_nosym)))

            # Here starts the actual test
            # Loop over all bands
            n = 0
            while n < nbands:
                dim = find_degenerate_subspace(eps_n, n, nbands, atol_eig)
                if dim == 1:
                    # First check so that projections differ by at most
                    # a global phase
                    compare_P_ani(P_ani, P_ani_nosym, n, atol)

                    # Check so that periodic part of pseudo is same,
                    # up to a phase
                    ut_R = ibz2bz[K].map_pseudo_wave_to_BZ(
                        wfs.pd.ifft(psit_nG[n], ik), r_cR)
                    ut_R_nosym = wfs_nosym.pd.ifft(psit_nG_nosym[n], K)
                    assert np.allclose(abs(ut_R), abs(ut_R_nosym), atol=atol)

                # For degenerate states check transformation
                # matrix is unitary,
                # For non-degenerate states check so that all-electron wf:s
                # are the same up to phase
                bands = range(n, n + dim)
                NR = np.prod(np.shape(r_cR)[1:])
                u1_nR = np.array([ibz2bz[K].map_pseudo_wave_to_BZ(
                    wfs.pd.ifft(psit_nG[n], ik), r_cR) for n in bands])
                u2_nR = np.array([wfs_nosym.pd.ifft(
                    psit_nG_nosym[n], K) for n in bands])

                check_all_electron_wfs(calc, bands, NR, u1_nR, u2_nR, P_ani, P_ani_nosym, dO_aii, atol)
                n += dim
