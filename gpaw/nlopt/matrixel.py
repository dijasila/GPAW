from os import path
from typing import Optional

from ase.parallel import parprint
from ase.units import Ha
from ase.utils.timing import Timer
import numpy as np

from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.ase_interface import GPAW
from gpaw.nlopt.adapters import GSInfo
from gpaw.nlopt.basic import NLOData
from gpaw.typing import ArrayND
from gpaw.utilities.progressbar import ProgressBar


def get_mml(gs: GSInfo,
            spin: int,
            ni: int,
            nf: int,
            timer: Optional[Timer] = None) -> ArrayND:
    """Compute the momentum matrix elements.

    Input:
        gs_name         Ground state file name
        spin            Which spin channel (for spin-polarized systems 0 or 1)
        ni, nf          First and last band to compute the mml (0 to nb)
        timer           Timer to keep track of time
    Output:
        p_kvnn2         A big array in master
    """

    # Start the timer
    if timer is None:
        timer = Timer()
    parprint('Calculating momentum matrix elements...')

    # Specify desired range and number of bands in calculation
    bands = slice(ni, nf)
    nb = nf - ni

    # Spin input
    assert spin < gs.ns, 'Wrong spin input'

    # Real and reciprocal space parameters
    nk = np.shape(gs.ibzk_kc)[0]

    # Parallelisation and memory estimate
    comm = gs.comm
    rank = comm.rank
    size = comm.size
    nkcore = int(np.ceil(nk / size))  # Number of k-points pr. core
    est_mem = 2 * 3 * nk * nb**2 * 16 / 2**20
    parprint(f'At least {est_mem:.2f} MB of memory is required on master.')

    # Allocate the matrix elements
    p_kvnn = np.zeros((nkcore, 3, nb, nb), dtype=complex)

    # Initial call to print 0% progress
    if rank == 0:
        pb = ProgressBar()

    # Calculate matrix elements in loop over k-points
    for ik in range(nkcore):
        k_ind = rank + size * ik
        if k_ind >= nk:
            break
        wfs = gs.get_wfs(k_ind, spin)

        with timer('Contribution from pseudo wave functions'):
            G_plus_k_Gv, u_nG = gs.get_plane_wave_coefficients(
                wfs, bands=bands, spin=spin)
            p_vnn = np.einsum('Gv,mG,nG->vmn',
                              G_plus_k_Gv, u_nG.conj(), u_nG) * gs.ucvol

        with timer('Contribution from PAW corrections'):
            P_ani = gs.get_wave_function_projections(
                wfs, bands=bands, spin=spin)
            for P_ni, nabla_iiv in zip(P_ani.values(), gs.nabla_aiiv):
                p_vnn -= 1j * np.einsum('mi,nj,ijv->vmn',
                                        P_ni.conj(), P_ni, nabla_iiv)

        p_kvnn[ik] = p_vnn

        # Print the progress
        if rank == 0:
            pb.update(ik / nkcore)

    if rank == 0:
        pb.finish()

    # Gather all data to the master
    with timer('Gather the data to master'):
        if rank == 0:
            recv_buf = np.empty((size, nkcore, 3, nb, nb),
                                dtype=complex)
        else:
            recv_buf = None
        gs.comm.gather(p_kvnn, 0, recv_buf)
        if rank == 0:
            assert recv_buf is not None
            p_kvnn2 = np.zeros((nk, 3, nb, nb), dtype=complex)
            for ii in range(size):
                k_inds = range(ii, nk, size)
                p_kvnn2[k_inds] = recv_buf[ii, :len(k_inds)]

    # Print the timing
    if rank == 0:
        timer.write()

    if rank == 0:
        return p_kvnn2
    else:
        return np.array([], dtype=complex)


def make_nlodata(gs_name: str,
                 comm: MPIComm,
                 spin: str = 'all',
                 ni: Optional[int] = None,
                 nf: Optional[int] = None) -> NLOData:

    """Get all required NLO data and store it in a file.

    Writes NLO data to file: w_sk, f_skn, E_skn, p_skvnn.

    Parameters:

    gs_name:
        Ground state filename.
    comm:
        Communicator for parallelisation.
    spin:
        Spin channels to include ('all', 's0' , 's1').
    ni:
        First band to compute the mml.
    nf:
        Last band to compute the mml (relative to number of bands
        for nf <= 0).

    """

    assert path.exists(gs_name), \
        f'The gs file: {gs_name} does not exist!'
    calc = GPAW(gs_name, txt=None, communicator=serial_comm)

    assert not calc.symmetry.point_group, \
        'Point group symmtery should be off.'

    gs: GSInfo
    if calc.calculation.state.density.collinear:
        from gpaw.nlopt.adapters import CollinearGSInfo
        gs = CollinearGSInfo(calc, comm)
    else:
        from gpaw.nlopt.adapters import NoncollinearGSInfo
        gs = NoncollinearGSInfo(calc, comm)

    # Parse spin input
    ns = gs.ns
    if spin == 'all':
        spins = list(range(ns))
    elif spin == 's0':
        spins = [0]
    elif spin == 's1':
        spins = [1]
        assert spins[0] < ns, 'Wrong spin input'
    else:
        raise NotImplementedError

    # Parse band input
    nb_full = gs.nb_full
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb_full
    nf = nb_full + nf if (nf <= 0) else nf

    return _make_nlodata(gs=gs, spins=spins, ni=ni, nf=nf)


def _make_nlodata(gs: GSInfo,
                  spins: list,
                  ni: int,
                  nf: int) -> NLOData:

    # Start the timer
    timer = Timer()

    # Get the energy and fermi levels (data is only in master)
    with timer('Get energies and fermi levels'):
        ibzwfs = gs.ibzwfs

        # Get the data
        E_skn, f_skn = ibzwfs.get_all_eigs_and_occs()
        # Energy is returned in Ha. For now we will change
        # it to eV avoid altering the module too much.
        E_skn *= Ha

        w_sk = np.array([ibzwfs.ibz.weight_k for _ in range(gs.ndens)])
        w_sk *= gs.bzvol * ibzwfs.spin_degeneracy

    # Compute the momentum matrix elements
    with timer('Compute the momentum matrix elements'):
        p_skvnn = []
        for spin in spins:
            p_kvnn = get_mml(gs=gs, ni=ni, nf=nf,
                             spin=spin, timer=timer)
            p_skvnn.append(p_kvnn)
        if not gs.collinear:
            p_skvnn = [p_skvnn[0] + p_skvnn[1]]

    # Save the output to the file
    return NLOData(w_sk=w_sk,
                   f_skn=f_skn[:, :, ni:nf],
                   E_skn=E_skn[:, :, ni:nf],
                   p_skvnn=np.array(p_skvnn, complex),
                   comm=gs.comm)


def get_rml(E_n, p_vnn, pol_v, Etol=1e-6):
    """
    Compute the position matrix elements

    Input:
        E_n             Energies
        p_vnn           Momentum matrix elements
        pol_v           Tensor element
        Etol            Tol. in energy to consider degeneracy
    Output:
        r_vnn, D_vnn    Position and velocity difference matrix el.
    """

    # Useful variables
    nb = len(E_n)
    r_vnn = np.zeros((3, nb, nb), complex)
    D_vnn = np.zeros((3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    # Loop over components
    for v1 in set(pol_v):
        r_vnn[v1] = p_vnn[v1] / (1j * E_nn)
        r_vnn[v1, zeroind] = 0
        p_n = np.diag(p_vnn[v1])
        D_vnn[v1] = np.tile(p_n[:, None], (1, nb)) - \
            np.tile(p_n[None, :], (nb, 1))

    return r_vnn, D_vnn


def get_derivative(E_n, r_vnn, D_vnn, pol_v, Etol=1e-6):
    """
    Compute the generalized derivative of position matrix elements

    Input:
        E_n             Energies
        r_vnn           Momentum matrix elements
        D_vnn           Velocity difference
        pol_v           Tensor element
        Etol            Tol. in energy to consider degeneracy
    Output:
        rd_vvnn         Generilized derivative of position
    """

    # Useful variables
    nb = len(E_n)
    rd_vvnn = np.zeros((3, 3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    for v1 in set(pol_v):
        for v2 in set(pol_v):
            tmp = (r_vnn[v1] * np.transpose(D_vnn[v2])
                   + r_vnn[v2] * np.transpose(D_vnn[v1])
                   + 1j * np.dot(r_vnn[v1], r_vnn[v2] * E_nn)
                   - 1j * np.dot(r_vnn[v2] * E_nn, r_vnn[v1])) / E_nn
            tmp[zeroind] = 0
            rd_vvnn[v1, v2] = tmp

    return rd_vvnn
