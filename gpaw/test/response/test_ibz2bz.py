import numpy as np
import pytest
from gpaw import GPAW
from gpaw.response.ibz2bz import IBZ2BZMaps
import gpaw.mpi as mpi


def mark_xfail(gs, request):
    if gs in ['al_pw', 'fe_pw', 'co_pw']:
        request.node.add_marker(pytest.mark.xfail)


@pytest.mark.serial
@pytest.mark.response
@pytest.mark.parametrize('gs', ['fancy_si_pw',
                                'al_pw',
                                'fe_pw',
                                'gaas_pw'])
@pytest.mark.parametrize('only_ibz_kpts', [True, False])
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

    XXX Todo: Add Co from test_mft as test system and make fixture.
    . When xfails are figured out, remove only_ibz_kpts parametrization
    """
    
    # Al, Fe and Co fails. Need to figure out why (see above)
    mark_xfail(gs, request)

    # can set individual tolerance for eigenvalues
    atol = 1e-05
    atol_eig = 1e-05

    # Loading calc with symmetry
    calc = GPAW(gpw_files[gs + '_wfs'],
                communicator=mpi.serial_comm)
    wfs = calc.wfs
    nconv = calc.parameters.convergence['bands']

    # setting basic stuff
    nbands = wfs.bd.nbands if nconv == -1 else nconv
    nbzk = wfs.kd.nbzkpts
    r_cR = wfs.gd.get_grid_point_coordinates()
    ibz2bz = IBZ2BZMaps.from_calculator(calc)

    # Loading calc without symmetry
    calc_nosym = GPAW(gpw_files[gs + '_nosym_wfs'],
                      communicator=mpi.serial_comm)
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

            # Get projections, wfs and energies at BZ k-point K using
            # ibz2bz for calculations with symmetry. Also get
            # overlaps and energies and make some basic checks
            proj_sym, proj_nosym, ut_nR_sym, ut_nR_nosym, dO_aii, eps_n = \
                get_data_from_wfs(wfs, wfs_nosym, nbands,
                                  ibz2bz, K, ik, s, atol, atol_eig)

            # Here starts the actual test
            # Loop over all bands
            n = 0
            while n < nbands:
                dim = find_degenerate_subspace(eps_n, n, nbands, atol_eig)
                if dim == 1:
                    # First check so that projections differ by at most
                    # a global phase
                    compare_projections(proj_sym, proj_nosym, n, atol)

                    # Check so that periodic part of pseudo is same,
                    # up to a phase
                    assert np.allclose(abs(ut_nR_sym[n]),
                                       abs(ut_nR_nosym[n]), atol=atol)

                # For degenerate states check transformation
                # matrix is unitary,
                # For non-degenerate states check so that all-electron wf:s
                # are the same up to phase
                bands = range(n, n + dim)

                check_all_electron_wfs(bands, ut_nR_sym[bands],
                                       ut_nR_nosym[bands],
                                       proj_sym, proj_nosym, dO_aii,
                                       wfs.gd.dv, wfs.gd.cell_cv,
                                       calc.spos_ac, atol)
                n += dim


def get_overlap(bands, u1_nR, u2_nR, P1_ani, P2_ani, dO_aii, dv, cell_cv):
    """ Computes overlap of all-electron wavefunctions
    Similar to gpaw.berryphase.get_overlap but adapted
    to work with projector objects rather than arrays.
    XXX Eventually berryphase.get_overlap should be replaced
    by this function

    Parameters
    ----------
    bands:  integer list
            bands to calculate overlap for
    u1_nR:  np.array
            flattened u_nR array
    u2_nR:  np.array
    P1_ani: GPAW Projections object
    P2_ani: GPAW Projections object
    dO_aii: dict
            overlaps from setups
    dv:     float
            calc.wfs.gd.dv
    cell_cv: np.array
             calc.wfs.gd.cell_cv
    """
    M_nn = np.dot(u1_nR.conj(), u2_nR.T) * dv

    for ia, _ in P1_ani.items():
        P1_ni = P1_ani[ia][bands]
        P2_ni = P2_ani[ia][bands]
        dO_ii = dO_aii[ia]
        M_nn += P1_ni.conj() @ (dO_ii) @ (P2_ni.T)

    return M_nn


def equal_dicts(dict_1, dict_2, atol):
    """ Checks so that two dicts with np.arrays are
    equal"""
    
    for k in dict_1:
        try:
            np.allclose(dict_1[k], dict_2[k], atol=atol)
        except AssertionError:
            return False
        return True


def find_degenerate_subspace(eps_n, n_start, nbands, atol_eig):
    # Find degenerate eigenvalues
    n = n_start
    dim = 1
    while n < nbands - 1 and abs(eps_n[n] - eps_n[n + 1]) < atol_eig:
        dim += 1
        n += 1
    return dim


def compare_projections(proj_sym, proj_nosym, n, atol):
    # compares so that projections at given k and band index n
    # differ by at most a global phase
    phase = 0
    for a, P_ni in proj_sym.items():
        P_ni = proj_sym[a]
        for j in range(len(P_ni[n, :])):
            # Only compare elements with finite value
            if abs(P_ni[n, j]) > 0.0001:
                newphase = P_ni[n, j] / proj_nosym[a][n, j]
                if phase != 0.0:
                    assert np.allclose(newphase, phase, atol=atol)
                phase = newphase
    assert np.allclose(abs(phase), 1.0, atol=atol)


def get_overlaps_from_setups(wfs):
    dO_aii = {}
    for ia in wfs.kpt_u[0].P_ani.keys():
        dO_aii[ia] = wfs.setups[ia].dO_ii
    return dO_aii


def check_all_electron_wfs(bands, u1_nR, u2_nR,
                           proj_sym, proj_nosym, dO_aii,
                           dv, cell_cv, spos_ac, atol):
    """sets up transformation matrix between symmetry
       transformed u:s and normal u:s in degenerate subspace
       and asserts that it is unitary
    
    Parameters
    ---------
    bands: int list
         band indexes in degenerate subspace
    u1_nR: np.array
    u2_nR: np.array
        Periodic part of pseudo wave function for two calculations
    proj_sym: Projections object
    proj_nosym: Projections object
        Projections for two calculations
    dO_aii: list of np.arrays
       see get_overlaps_from_setups
    dv:     float
            calc.wfs.gd.dv
    cell_cv: np.array
             calc.wfs.gd.cell_cv
    atol: float
       absolute tolerance when comparing arrays
    """
    NR = np.prod(np.shape(u1_nR)[1:])
    Utrans = get_overlap(bands,
                         np.reshape(u1_nR, (len(u1_nR), NR)),
                         np.reshape(u2_nR, (len(u2_nR), NR)),
                         proj_sym,
                         proj_nosym,
                         dO_aii,
                         dv,
                         cell_cv)

    # Check so that transformation matrix is unitary
    UUdag = Utrans @ Utrans.T.conj()
    assert np.allclose(np.eye(len(UUdag)), UUdag, atol=atol)

    # Check so that Utrans transforms pseudo wf:s
    # Row/column definition of indexes in einsum comes from
    # definition of Utrans
    u_transformed = np.einsum('ji,jklm->iklm', Utrans, u1_nR)
    assert np.allclose(u_transformed, u2_nR)


def get_data_from_wfs(wfs_sym, wfs_nosym, nbands, ibz2bz,
                      K, ik, s, atol, atol_eig):
    """Gets pseudo wfs, projections, energies and overlaps
    from wfs object for calculation with and without symmetry.
    For the calculation with symmetry the quantities are
    transformed from the IBZ (ik) to BZ (K) using ibz2bzMaps
    """
    r_cR = wfs_nosym.gd.get_grid_point_coordinates()

    # Get data for calc with symmetry
    kpt = wfs_sym.kpt_qs[ik][s]
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
    proj_sym = ibz2bz[K].map_projections(proj)
            
    # Get projections for calc without symmetry
    proj_nosym = kpt_nosym.projections

    # Get pseudo wfs for both calculations
    ut_nR_sym = np.array([ibz2bz[K].map_pseudo_wave_to_BZ(
        wfs_sym.pd.ifft(psit_nG[n], ik), r_cR) for n in range(nbands)])
    ut_nR_nosym = np.array([wfs_nosym.pd.ifft(
        psit_nG_nosym[n], K) for n in range(nbands)])
            
    # get overlaps
    dO_aii = get_overlaps_from_setups(wfs_sym)
    assert equal_dicts(dO_aii,
                       get_overlaps_from_setups(wfs_nosym),
                       atol)
    return proj_sym, proj_nosym, ut_nR_sym, ut_nR_nosym, dO_aii, eps_n
