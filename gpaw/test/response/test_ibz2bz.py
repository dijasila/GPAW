import numpy as np
import pytest
from gpaw import GPAW
from gpaw.response.ibz2bz import IBZ2BZMaps
import gpaw.mpi as mpi


def mark_xfail(gs, request):
    if gs in ['al_pw', 'fe_pw']:
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

    - Note that Al,  Fe are  marked as xfail!
    - For Al eigenvalues are different. Probably due to k-mesh.
      See "Note your k-mesh ..." in gs output files.
    - For Fe projections between two calculations differ by more than a
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
    ibz2bz = IBZ2BZMaps.from_calculator(calc)

    # Loading calc without symmetry
    calc_nosym = GPAW(gpw_files[gs + '_nosym_wfs'],
                      communicator=mpi.serial_comm)
    wfs_nosym = calc_nosym.wfs
    
    # Check some basic stuff
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

            r_vR = wfs_nosym.gd.get_grid_point_coordinates()
            assert np.allclose(r_vR, wfs_nosym.gd.get_grid_point_coordinates())
            
            # Get data for calc without symmetry at BZ kpt K
            eps_n_nosym, ut_nR_nosym, proj_nosym, dO_aii_nosym = \
                get_ibz_data_from_wfs(wfs_nosym, nbands, K, s)

            # Get data for calc with symmetry at ibz kpt ik
            eps_n, ut_nR, proj, dO_aii = get_ibz_data_from_wfs(wfs,
                                                               nbands,
                                                               ik, s)
    
            # Map projections and u:s from ik to K
            proj_sym = ibz2bz[K].map_projections(proj)
            ut_nR_sym = np.array([ibz2bz[K].map_pseudo_wave_to_BZ(
                ut_nR[n], r_vR) for n in range(nbands)])

            # Check so that eigenvalues are the same
            assert np.allclose(eps_n[:nbands],
                               eps_n_nosym[:nbands],
                               atol=atol_eig)

            # Check so that overlaps are the same for both calculations
            assert equal_dicts(dO_aii,
                               dO_aii_nosym,
                               atol)

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

                check_all_electron_wfs(bands, ut_nR_sym,
                                       ut_nR_nosym,
                                       proj_sym, proj_nosym, dO_aii,
                                       wfs.gd.dv, atol)
                n += dim


def get_overlap(bands, u1_nR, u2_nR, proj1, proj2, dO_aii, dv):
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
            u_nR array
    u2_nR:  np.array
            u_nR array
    proj1: GPAW Projections object
    proj2: GPAW Projections object
    dO_aii: dict
            overlaps from setups
    dv:     float
            calc.wfs.gd.dv
    """
    NR = np.prod(np.shape(u1_nR)[1:])
    u1_nR = np.reshape(u1_nR, (len(u1_nR), NR))
    u2_nR = np.reshape(u2_nR, (len(u2_nR), NR))
    M_nn = (u1_nR[bands].conj() @ u2_nR[bands].T) * dv

    for a in proj1.map:
        P1_ni = proj1[a][bands]
        P2_ni = proj2[a][bands]
        dO_ii = dO_aii[a]
        M_nn += P1_ni.conj() @ (dO_ii) @ (P2_ni.T)

    return M_nn


def equal_dicts(dict_1, dict_2, atol):
    """ Checks so that two dicts with np.arrays are
    equal"""
    assert len(dict_1.keys()) == len(dict_2.keys())
    for key in dict_1:
        # Make sure the dictionaries contain the same set of keys
        if key not in dict_2:
            return False
        # Make sure that the arrays are identical
        if not np.allclose(dict_1[key], dict_2[key], atol=atol):
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
    phase = None
    newphase = None
    for a, P_ni in proj_sym.items():
        for j in range(len(P_ni[n, :])):
            # Only compare elements with finite value
            if abs(P_ni[n, j]) > atol:
                newphase = P_ni[n, j] / proj_nosym[a][n, j]
                if phase is not None:
                    assert np.allclose(newphase, phase, atol=atol)
                phase = newphase
    assert np.allclose(abs(phase), 1.0, atol=atol)


def get_overlaps_from_setups(wfs):
    dO_aii = {}
    for a in wfs.kpt_u[0].projections.map:
        dO_aii[a] = wfs.setups[a].dO_ii
    return dO_aii


def check_all_electron_wfs(bands, u1_nR, u2_nR,
                           proj_sym, proj_nosym, dO_aii,
                           dv, atol):
    """sets up transformation matrix between symmetry
       transformed u:s and normal u:s in degenerate subspace
       and asserts that it is unitary. It also checks that
       the pseudo wf:s transform according to the same
       transformation.
    
       Let |psi^1_i> denote the all electron wavefunctions
       from the calculation with symmetry and |psi^2_i>
       the corresponding wavefunctions from the calculation
       without symmetry.
       If the set {|psi^1_i>} span the same subspace as the set
       {|psi^2_i>} they fulfill:
       |psi^2_i> = |Psi^1_k> <Psi^1_k | Psi^2_i> == Utrans_ki |Psi^1_k>
       where summation over repeated indexes is assumed and U is
       a unitary transformation.
    
    Parameters
    ---------
    bands: list of ints
         band indexes in degenerate subspace
    u1_nR: np.array
    u2_nR: np.array
        Periodic part of pseudo wave function for two calculations
    proj_sym: Projections object
    proj_nosym: Projections object
        Projections for two calculations
    dO_aii: dict with np.arrays
       see get_overlaps_from_setups
    dv:     float
            calc.wfs.gd.dv
    atol: float
       absolute tolerance when comparing arrays
    """
    Utrans = get_overlap(bands,
                         u1_nR,
                         u2_nR,
                         proj_sym,
                         proj_nosym,
                         dO_aii,
                         dv)

    # Check so that transformation matrix is unitary
    UUdag = Utrans @ Utrans.T.conj()
    assert np.allclose(np.eye(len(UUdag)), UUdag, atol=atol)

    # Check so that Utrans transforms pseudo wf:s
    # Row/column definition of indexes in einsum comes from
    # definition of Utrans
    u_transformed = np.einsum('ji,jklm->iklm', Utrans, u1_nR[bands])
    assert np.allclose(u_transformed, u2_nR[bands])


def get_ibz_data_from_wfs(wfs, nbands, ik, s):
    """ gets data at ibz k-point ik
    """
    # get energies and wfs
    kpt = wfs.kpt_qs[ik][s]
    psit_nG = kpt.psit_nG
    eps_n = kpt.eps_n
    
    # Get periodic part of pseudo wfs
    ut_nR = np.array([wfs.pd.ifft(
        psit_nG[n], ik) for n in range(nbands)])
    
    # Get projections
    proj = kpt.projections.new(nbands=nbands, bcomm=None)
    proj.array[:] = kpt.projections.array[:nbands]
    
    # get overlaps
    dO_aii = get_overlaps_from_setups(wfs)
    return eps_n, ut_nR, proj, dO_aii
