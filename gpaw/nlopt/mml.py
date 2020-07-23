# Import the required modules: General
import numpy as np
import sys
from math import pi

# Import the required modules: GPAW/ASE
from gpaw import GPAW
from gpaw.mpi import world, broadcast, serial_comm
from gpaw.nlopt.output import print_progressbar, parprint
from gpaw.fd_operators import Gradient
from gpaw.response.pair import PairDensity
from gpaw.gaunt import gaunt
from gpaw.spinorbit import get_radial_potential
from ase.units import Bohr, _hbar, _e, _me, alpha
from ase.utils.timing import Timer


# Calculate and save the momentum matrix elements


def get_dipole_transitions(atoms, momname=None, basename=None):
    """
    Finds the nabla matrix elements:
    <psi_n|nabla|psi_m> = <u_n|nabla|u_m> + ik<u_n|u_m>
                            where psi_n = u_n(r)*exp(ikr).

    Input:
        atoms           Relevant ASE atoms object
        momname         Suffix for the dipole transition file
        basename        Suffix used for the gs.gpw file
    Output:
        dip_vknm.npy    Array with dipole matrix elements
    """

    timer = Timer()
    parprint('Calculating momentum matrix elements...')

    # Load the ground state calculations
    if basename is None:
        calc_name = 'gs.gpw'
    else:
        calc_name = 'gs_{}.gpw'.format(basename)
    calc = GPAW(calc_name, txt=None)
    calc.initialize_positions(atoms)

    # Useful variables
    bzk_kc = calc.get_ibz_k_points()
    nb = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]

    if calc.parameters['mode'] == 'lcao':
        # Distribute the k points and data betweeon cores
        wfs = {}
        parprint('Distributing wavefunctions...')
        sys.stdout.flush()
        for k_c in range(nk):
            # Collects the wavefunctions and projections to master
            with timer('Collects the wavefunctions and projections'):
                wf = np.array(
                    [calc.wfs.get_wave_function_array(
                        ii, k_c, 0,
                        realspace=True,
                        periodic=True) for ii in range(nb)],
                    dtype=complex)
                P_nI = calc.wfs.collect_projections(k_c, 0)

            # Distributes the information to rank k % size.
            with timer('k-info distribution'):
                if world.rank == 0:
                    if k_c % world.size == world.rank:
                        wfs[k_c] = wf, P_nI
                    else:
                        world.send(P_nI, dest=k_c % world.size, tag=nk + k_c)
                        world.send(wf, dest=k_c % world.size, tag=k_c)
                else:
                    if k_c % world.size == world.rank:
                        nproj = sum(setup.ni for setup in calc.wfs.setups)
                        if not calc.wfs.collinear:
                            nproj *= 2
                        P_nI = np.empty(
                            (calc.wfs.bd.nbands, nproj), calc.wfs.dtype)
                        shape = () if calc.wfs.collinear else(2,)
                        wf = np.tile(
                            calc.wfs.empty(
                                shape,
                                global_array=True,
                                realspace=True),
                            (nb, 1, 1, 1))
                        world.receive(P_nI, src=0, tag=nk + k_c)
                        world.receive(wf, src=0, tag=k_c)
                        wfs[k_c] = wf, P_nI

        # Compute the matrix elements
        nkcore = int(np.ceil(nk / world.size))
        dip_vknm = np.zeros((3, nkcore, nb, nb), dtype=complex)
        parprint('Evaluating dipole transition matrix elements...')
        sys.stdout.flush()

        overlap_knm = np.zeros((nk, nb, nb), dtype=complex)
        # Initial call to print 0% progress
        count = 0
        ncount = len(wfs)
        print_progressbar(
            count,
            len(wfs),
            prefix='Progress:',
            suffix='Complete',
            length=50)

        with timer('Compute the gradient'):
            nabla_v = [
                Gradient(
                    calc.wfs.gd, v, 1.0, 4,
                    complex).apply for v in range(3)]
        phases = np.ones((3, 2), dtype=complex)
        grad_nv = calc.wfs.gd.zeros((nb, 3), complex)
        for k_c, (wf, P_nI) in wfs.items():
            # Calculate <phit|nabla|phit> for the pseudo wavefunction
            with timer('Derivative calculation'):
                for v in range(3):
                    for i in range(nb):
                        nabla_v[v](wf[i], grad_nv[i, v], phases)

                dip_vknm[:, count] = np.transpose(
                    calc.wfs.gd.integrate(wf, grad_nv), (2, 0, 1))
                overlap_knm[k_c] = [
                    calc.wfs.gd.integrate(
                        wf[i], wf) for i in range(nb)]
                k_v = np.dot(
                    calc.wfs.kd.ibzk_kc[k_c],
                    calc.wfs.gd.icell_cv) * 2 * pi
                dip_vknm[:, count] += 1j * k_v[:, None, None] * \
                    overlap_knm[None, k_c, :, :]

            # The PAW corrections are added
            with timer('PAW correction calculation'):
                I1 = 0
                # np.einsum is slow but very memory efficient.
                for setup in calc.wfs.setups:
                    I2 = I1 + setup.ni
                    P_ni = P_nI[:, I1:I2]
                    dip_vknm[:, count, :, :] += np.einsum('ni,ijv,mj->vnm',
                                                          P_ni.conj(),
                                                          setup.nabla_iiv,
                                                          P_ni)
                    I1 = I2

            # Print the progress
            count += 1
            print_progressbar(
                count,
                ncount,
                prefix='Progress:',
                suffix='Complete',
                length=50)

        # Gather all data to the master
        with timer('Gather data to master'):
            recvbuf = None
            if world.rank == 0:
                recvbuf = np.empty((world.size, 3, nkcore, nb, nb),
                                   dtype=complex)
            world.gather(dip_vknm, 0, recvbuf)
            if world.rank == 0:
                dip_vknm2 = np.zeros((3, nk, nb, nb), dtype=complex)
                for rank in range(world.size):
                    kind = range(rank, nk, world.size)
                    dip_vknm2[:, kind] = recvbuf[rank, :, :len(kind)]
                dip_vknm = dip_vknm2
        # world.sum(dip_vknm)
    else:
        # Distributes k points
        with timer('k-info distribution'):
            kdata = {}
            for k in range(nk):
                if world.rank == 0:
                    if k % world.size == world.rank:
                        k_ind = np.array(k)
                        kdata[k] = k_ind
                    else:
                        k_ind = np.array(k)
                        world.send(k_ind, dest=k % world.size, tag=k)
                else:
                    if k % world.size == world.rank:
                        k_ind = np.array(k, dtype=int)
                        world.receive(k_ind, src=0, tag=k)
                        kdata[k] = k_ind

        # Initial call to print 0% progress
        count = 0
        print_progressbar(
            count,
            len(kdata),
            prefix='Progress:',
            suffix='Complete',
            length=50)
        kd = calc.wfs.kd

        nkcore = int(np.ceil(nk / world.size))
        dip_vknm = np.zeros((3, nkcore, nb, nb), dtype=complex)
        # Loop over k points
        for k in kdata.keys():
            pair = PairDensity(calc_name, world=serial_comm, txt=None)
            ik = kd.ibz2bz_k[k]
            kpt = pair.get_k_point(0, ik, 0, nb, load_wfs=True)

            # Compute all matrix elements
            with timer('Compute matrix elemnts'):
                for nni in range(nb):
                    tmp = pair.optical_pair_velocity(nni, [], kpt, kpt)
                    dip_vknm[:, count, nni, :] = np.transpose(1j * tmp)

            # Print progress
            count += 1
            print_progressbar(
                count,
                len(kdata),
                prefix='Progress:',
                suffix='Complete',
                length=50)

        # Gather all data to the master
        with timer('Gather data to master'):
            recvbuf = None
            if world.rank == 0:
                recvbuf = np.empty((world.size, 3, nkcore, nb, nb),
                                   dtype=complex)
            world.gather(dip_vknm, 0, recvbuf)
            if world.rank == 0:
                dip_vknm2 = np.zeros((3, nk, nb, nb), dtype=complex)
                for rank in range(world.size):
                    kind = range(rank, nk, world.size)
                    dip_vknm2[:, kind] = recvbuf[rank, :, :len(kind)]
                dip_vknm = dip_vknm2

    # Save the data
    if world.rank == 0:
        if momname is None:
            np.save('dip_vknm.npy', dip_vknm)
        else:
            np.save('dip_vknm_{}.npy'.format(momname), dip_vknm)
        timer.write()

# Set the required variables from calc


def set_variables(calc):
    """
    Set the required variables from calc

    Input:
        calc        GPAW calculator
    Output:
        nb, nk, mu, kbT, bz_vol, w_k, kd
    """

    # Only the master read the gs
    if world.rank == 0:
        # Get the required variables
        nb = calc.get_number_of_bands()
        mu = calc.get_fermi_level()
        occupations = calc.parameters['occupations']
        kbT = occupations['width']
        w_k = calc.get_k_point_weights()
        nk = len(w_k)
        kd = calc.wfs.kd
        bz_vol = np.linalg.det(2 * pi * calc.wfs.gd.icell_cv)
    else:
        # In other cores set them to none
        nb = None
        nk = None
        w_k = None
        mu = None
        kbT = None
        kd = None
        bz_vol = None

    # Broadcast them to all cores
    nb = broadcast(nb, root=0)
    nk = broadcast(nk, root=0)
    mu = broadcast(mu, root=0)
    kbT = broadcast(kbT, root=0)
    w_k = broadcast(w_k, root=0)
    bz_vol = broadcast(bz_vol, root=0)
    kd = broadcast(kd, root=0)

    # Return data
    return nb, nk, mu, kbT, bz_vol, w_k, kd


# Load the mometum and gs file in master


def load_gsmoms(basename, momname):
    """
    Load the calcualtions in the master

    Input:
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        calc,           The GPAW calculator
        moms            The mometum matrix elements dimension (nk,3,nb,nb)
    """

    if world.rank == 0:
        # Load the ground state calculation
        if basename is None:
            calc = GPAW('gs.gpw', txt=None, communicator=serial_comm)
        else:
            calc = GPAW(
                'gs_{}.gpw'.format(basename),
                txt=None,
                communicator=serial_comm)

        # Load the momentum matrix elements
        if momname is None:
            moms = np.load('dip_vknm.npy')  # [:,k,:,:]dim, k
        else:
            moms = np.load('dip_vknm_{}.npy'.format(
                momname))  # [:,k,:,:]dim, k
        # Make it momentum
        moms = -1j * moms
        moms = np.swapaxes(moms, 0, 1)
    else:
        calc = None
        moms = None

    # Return gs calc and moms (only in master)
    return calc, moms


# Distribute the data among the cores


def distribute_data(alldata, datashape):
    """
    Distribute the data among the cores

    Input:
        alldata         A list of all data (the first index should be k)
        datashape       A list of sizes for alldata elements
    Output:
        k_info          A  dictionary of data with key of k index
    """

    # Check the datashape
    nk = datashape[0][0]
    if world.rank == 0:
        for cdata, cshape in zip(alldata, datashape):
            assert cdata.shape == cshape or cdata[0] == nk, 'Wrong datashape.'

    # Distribute the data of k-points between cores
    k_info = {}

    # Loop over k points
    for kk in range(nk):
        if world.rank == 0:
            if kk % world.size == world.rank:
                k_info[kk] = [cdata[kk] for cdata in alldata]
            else:
                for ii, cdata in enumerate(alldata):
                    data_k = np.array(cdata[kk], dtype=complex)
                    world.send(
                        data_k, dest=kk %
                        world.size, tag=ii * nk + kk)
        else:
            if kk % world.size == world.rank:
                dataset = []
                for ii, cshape in enumerate(datashape):
                    data_k = np.empty(cshape[1:], dtype=complex)
                    world.receive(data_k, src=0, tag=ii * nk + kk)
                    dataset.append(data_k)
                k_info[kk] = dataset

    # Return the dictionary
    return k_info


# Get the generalized derivative of momentum


def calc_derivative(mom, E_n, ni, nf):
    """
    Calculate the (generlized) derivative of the momentum (using sum rules)

    Input:
    mom             Momentum matrix elements at k point
    E_n             Energies of the bands at k point
        ni, nf      The first and last band indices
    Output:
        mom_der     Derivative of the momentum
    """

    # Useful variables
    nb = len(E_n)
    mom_der = np.zeros((3, 3, nb, nb), dtype=np.complex)

    # Loop over Coordinates
    for aa in range(3):
        for bb in range(3):
            # Loop over bands
            for nni in range(nb):
                for mmi in range(nb):
                    tmp = 0
                    # Make the summation
                    for lli in range(nb):
                        Eml = E_n[mmi] - E_n[lli]
                        Eln = E_n[lli] - E_n[nni]
                        if mmi != lli and np.abs(Eml) > 1e-10:
                            pml = mom[aa, mmi, lli]
                            pln = mom[bb, lli, nni]
                            tmp += pml * pln / Eml
                        if nni != lli and np.abs(Eln) > 1e-10:
                            pml = mom[bb, mmi, lli]
                            pln = mom[aa, lli, nni]
                            tmp -= pml * pln / Eln

                    # Save the generelized derivative of momentum
                    mom_der[aa, bb, mmi, nni] = (mmi == nni) * (aa == bb) \
                        + tmp * (_hbar / (Bohr * 1e-10))**2 / (_e * _me)

    # Return the generlized derivative
    return mom_der


# Get the Fermi energy


def fermi_level(calc, eps_skn=None, nelectrons=None):
    """Get Fermi energy from calculation.

    Input:
        calc        GPAW calculator
        eps_skn     shape=(ns, nk, nb) eigenvalues
        nelectrons  number of electrons (taken from calc if None)

    Output:
        out         fermi level
    """

    from gpaw.occupations import occupation_numbers
    from ase.units import Ha
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    eps_skn.sort(axis=-1)
    occ = calc.occupations.todict()
    weight_k = calc.get_k_point_weights()
    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


# Function for integrating over a triangle with delta integrand


def triangle_delta(F, E, omega, itype=1):
    r"""
    Calculate the contribution of each triangle.
    S_\Delta(\omega) = \int_\Delta F(\mathbf{k}) \delta(\omega-E(\mathbf{k}))
                                                  \mathrm{d}\mathbf{k}

    Input:
        F       The value of function F at 3 edges.
        E       The energy at 3 edges.
        omega   The real frequency array.
        itype   Type of method: 1=average, 2=linear
    Output:
        Sdel   The integral value over the triangle.
    """
    # Sort with respect to energies
    sind = np.argsort(E)
    sE = E[sind]
    sF = F[sind]

    # Depending on the frequency make the output
    Sdel = np.zeros(len(omega), dtype=np.complex)
    low_ind = np.logical_and(omega >= sE[0], omega < sE[1])
    high_ind = np.logical_and(omega >= sE[1], omega < sE[2])
    if itype == 1:
        Fval = (sF[0] + sF[1] + sF[2]) / 3
        if np.any(low_ind):
            Sdel[low_ind] = 2 * ((omega[low_ind] - sE[0]) /
                                 ((sE[2] - sE[0]) * (sE[1] - sE[0]))) * Fval
        if np.any(high_ind):
            Sdel[high_ind] = 2 * ((sE[2] - omega[high_ind]) /
                                  ((sE[2] - sE[1]) * (sE[2] - sE[0]))) * Fval
    elif itype == 2:
        if np.any(low_ind):
            Sdel[low_ind] = 2 * ((omega[low_ind] - sE[0])
                                 / ((sE[2] - sE[0]) * (sE[1] - sE[0]))) \
                              * (sF[0] + (omega[low_ind] - sE[0]) / 2
                                 * ((sF[1] - sF[0]) / (sE[1] - sE[0])
                                 + (sF[2] - sF[0]) / (sE[2] - sE[0])))
        if np.any(high_ind):
            Sdel[high_ind] = 2 * ((sE[2] - omega[high_ind])
                                  / ((sE[2] - sE[1]) * (sE[2] - sE[0]))) \
                               * (sF[2] + (omega[high_ind] - sE[2]) / 2
                                  * ((sF[2] - sF[1]) / (sE[2] - sE[1])
                                  + (sF[2] - sF[0]) / (sE[2] - sE[0])))
    else:
        parprint('Integration type ' + itype + ' not implemented.')
        raise NotImplementedError

    # Return the output
    return Sdel


# Integration over 2 triangles


def triangle_int(F, E, omega, itype=1, tri1=[0, 1, 3], tri2=[0, 2, 3]):
    r"""
    Calculate the contribution of each rectangular.

    Input:
        F           The value of function F at 3 edges.
        E           The energy at 3 edges.
        omega       The real frequency array.
        itype       Type of method: 1=average, 2=linear
        tri1        Edge of traingle 1
        tri2        Edge of traingle 2
    Output:
        intval      The integral value over the triangle.
    """

    # Use the triangle method for integration
    intval = triangle_delta(F[tri1], E[tri1], omega, itype=itype)
    intval += triangle_delta(F[tri2], E[tri2], omega, itype=itype)

    # Return the integral
    return intval


# Find the neighbors and distribute the data among them


def get_neighbors(moms, E_nk, f_nk, kd, nb, qind=[
                  [1, 0, 0], [0, 1, 0], [1, 1, 0]]):
    """
    Get the neigbor points and their data for trangular method.

    Input:
        moms        Momentum matrix elements (only in the master)
        E_nk        Energies (only in the master)
        f_nk        Fermi values (only in the master)
        kd          k descriptor object
        nb          Number of bands
        qind        Edges of the triangles
    Output:
        k_info      k point data dictionary
        dA          The integral element area
    """

    # Useful variables
    nk = len(kd.ibz2bz_k)
    N_c = kd.N_c

    # Find the neighbors
    assert N_c[2] == 1, 'Triangular method is only implemented for 2D systems.'
    q_vecs = []
    for ii, qq in enumerate(qind):
        q_vecs.append([qq[0] / N_c[0], qq[1] / N_c[1], qq[2] / N_c[2]])
    nkt = len(kd.bzk_kc)
    neighbors = np.zeros((4, nkt), dtype=np.int32)
    neighbors[0] = np.arange(nkt)
    for ind, qpt in enumerate(q_vecs):
        neighbors[ind + 1] = np.array(kd.find_k_plus_q(qpt))

    # Depending on the tsym set variables
    tsym = kd.symmetry.time_reversal
    if tsym is False:
        # nei_ind = kd.bz2ibz_k[neighbors]
        # nei_ind = nei_ind[:, kd.ibz2bz_k]
        nei_ind = neighbors
        nk2 = nk
        psigns = np.ones((4, nk), dtype=float)
        dA = 1.0 / (N_c[0] * N_c[1])
    else:
        p1 = int((N_c[0] / 2 - 1) * N_c[1])
        p2 = int((N_c[0] - 1) * N_c[1])
        nk2 = p2 - p1
        nei_ind = neighbors[:, p1:p2]
        nei_ind = [kd.bz2ibz_k[nei_ind[ii]] for ii in range(len(q_vecs) + 1)]
        psigns = -2 * kd.time_reversal_k + 1
        dA = 2.0 / (N_c[0] * N_c[1])
        psigns = psigns[neighbors[:, p1:p2]]
    psigns = np.expand_dims(psigns, axis=(2, 3, 4))

    # if world.rank == 0:
    #     allE = [E_nk[nei_ind[0]]]
    #     allf = [f_nk[nei_ind[0]]]
    #     allP = [moms[nei_ind[0]].real * psigns[0]+1j * moms[nei_ind[0]].imag]
    #     for it in range(1, 4):
    #         p_nn = np.zeros((nk, 3, nb, nb), complex)
    #         for ik in range(nk2):
                
    #             k1 = nei_ind[0][ik]
    #             k2 = nei_ind[it][ik]
    #             print('Compute overlap for {}-{}.'.format(k1, k2))
    #             sys.stdout.flush()
    #             M_nn = get_overlap(calc, k1, k1)
    #             Mi_nn = (np.abs(M_nn)**2>0.5)*1.0
    #             if np.any(np.sum(Mi_nn, axis=0) != np.ones((nb))):
    #                 print('Overlap matrix is not correct for {}-{}.'.format(k1, k2))
    #             E_nk[k2] = np.dot(Mi_nn, E_nk[k2])
    #             f_nk[k2] = np.dot(Mi_nn, f_nk[k2])
    #             for v in range(3):
    #                 p_nn[k2, v] = np.dot(Mi_nn, moms[k2, v])
    #         allE.append(E_nk[k2])
    #         allf.append(f_nk[k2])
    #         allP.append(p_nn[nei_ind[it]].real * psigns[it]+1j * p_nn[nei_ind[it]].imag)
    # else:
    #     zz = np.zeros((nk2))
    #     allE = [zz, zz, zz, zz]
    #     allP = [zz, zz, zz, zz]
    #     allf = [zz, zz, zz, zz]

    # Only to avoid error
    if world.rank != 0:
        moms = np.zeros((nk))
        E_nk = np.zeros((nk))
        f_nk = np.zeros((nk))

    # Distribute the data among the cores
    k_info = distribute_data(
        [moms[nei_ind[0]].real * psigns[0] +
            1j * moms[nei_ind[0]].imag,
            moms[nei_ind[1]].real * psigns[1] +
            1j * moms[nei_ind[1]].imag,
            moms[nei_ind[2]].real * psigns[2] +
            1j * moms[nei_ind[2]].imag,
            moms[nei_ind[3]].real * psigns[3] +
            1j * moms[nei_ind[3]].imag,
            E_nk[nei_ind[0]], E_nk[nei_ind[1]],
            E_nk[nei_ind[2]], E_nk[nei_ind[3]],
            f_nk[nei_ind[0]], f_nk[nei_ind[1]],
            f_nk[nei_ind[2]], f_nk[nei_ind[3]]],
        [(nk2, 3, nb, nb),
         (nk2, 3, nb, nb),
         (nk2, 3, nb, nb),
         (nk2, 3, nb, nb),
         (nk2, nb), (nk2, nb), (nk2, nb), (nk2, nb),
         (nk2, nb), (nk2, nb), (nk2, nb), (nk2, nb)])
    # k_info = distribute_data(
    #     [allP[:], allE[:], allf[:]],
    #     [(nk2, 3, nb, nb),
    #      (nk2, 3, nb, nb),
    #      (nk2, 3, nb, nb),
    #      (nk2, 3, nb, nb),
    #      (nk2, nb), (nk2, nb), (nk2, nb), (nk2, nb),
    #      (nk2, nb), (nk2, nb), (nk2, nb), (nk2, nb)])

    # Return the k data
    return k_info, dA

# Get spin-orbit correction to momentum

def get_soc_momentum(dVL_avii, Pt_asni, ni, nf):
    """
    Get spin-orbit correction to momentum

    Input:
        dVL_avii        Derivative of the KS potential
        Pt_asni         PAW corrections
        ni, nf          The first and last band indices
    Output:
        p_vmm           Correction to the momentum
    """

    # Initialize variables
    nb = nf-ni
    Na = len(dVL_avii)
    p_vmm = np.zeros((3, 2*nb, 2*nb), complex)

    # Loop over atoms
    for ai in range(Na):
        Pt_sni = Pt_asni[ai][ni:nf]
        Ni = len(Pt_sni[0])
        P_sni = np.zeros((2, 2 * nb, Ni), complex)
        dVL_vii = dVL_avii[ai]
        P_sni[0, ::2] = Pt_sni
        P_sni[1, 1::2] = Pt_sni

        # The sigma cross rhat 
        p_vssii = np.zeros((3, 2, 2, Ni, Ni), complex)
        p_vssii[0, 0, 0] = - dVL_vii[1]
        p_vssii[0, 0, 1] = + 1.0j * dVL_vii[2]
        p_vssii[0, 1, 0] = - 1.0j * dVL_vii[2]
        p_vssii[0, 1, 1] = + dVL_vii[1]
        p_vssii[1, 0, 0] = + dVL_vii[0]
        p_vssii[1, 0, 1] = - dVL_vii[2]
        p_vssii[1, 1, 0] = - dVL_vii[2]
        p_vssii[1, 1, 1] = - dVL_vii[0]
        p_vssii[2, 0, 0] = 0
        p_vssii[2, 0, 1] = dVL_vii[1] + 1.0j * dVL_vii[0]
        p_vssii[2, 1, 0] = dVL_vii[1] - 1.0j * dVL_vii[0]
        p_vssii[2, 1, 1] = 0

        # Loop over spins and directions
        for v in range(3):
            for s1 in range(2):
                for s2 in range(2):
                    p_ii = p_vssii[v, s1, s2]
                    P1_mi = P_sni[s1]
                    P2_mi = P_sni[s2]
                    p_vmm[v] += np.dot(np.dot(P1_mi.conj(), p_ii), P2_mi.T)

    # Return output
    return p_vmm

# Get SOC part of calculations

def get_soc_paw(calc):
    """
    Input:
        calc        GPAW calculator (only in master)
    Output
        dVL_avii    Derivative of KS potential (in all cores)
        Pt_kasni    PAW coefficients (in all cores)
    """

    if world.rank == 0:
        # Compute the PAW correction
        G_LLL = gaunt(lmax=2)
        Na = len(calc.atoms)
        dVL_avii = []
        for ai in range(Na):
            a = calc.wfs.setups[ai]

            xc = calc.hamiltonian.xc
            D_sp = calc.density.D_asp[ai]

            v_g = get_radial_potential(a, xc, D_sp)
            rgd = a.xc_correction.rgd
            r_g = rgd.r_g.copy()
            r_g[0] = 1.0e-12
            v_g2 = v_g*r_g

            Ng = len(v_g)
            phi_jg = a.data.phi_jg

            dVL_vii = np.zeros((3, a.ni, a.ni), complex)
            N1 = 0
            for j1, l1 in enumerate(a.l_j):
                Nm1 = 2 * l1 + 1
                N2 = 0
                for j2, l2 in enumerate(a.l_j):
                    Nm2 = 2 * l2 + 1
                    # if (l1 == l2) or (l1 == l2 + 1) or (l1 == l2 - 1):
                    f_g = phi_jg[j1][:Ng] * v_g * phi_jg[j2][:Ng]
                    cc = a.xc_correction.rgd.integrate(f_g) / (4 * np.pi)
                    
                    for v in range(3):
                        Lv = 1 + (v + 2) % 3
                        # dVL_vii[v, N1:N1 + Nm1, N2:N2 + Nm2] = cc * G_LLL[Lv, N1:N1 + Nm1, N2:N2 + Nm2]
                        dVL_vii[v, N1:N1 + Nm1, N2:N2 + Nm2] = cc * G_LLL[Lv, l2**2:l2**2 + Nm2, l1**2:l1**2 + Nm1].T
                    N2 += Nm2
                N1 += Nm1
            tmp = dVL_vii * alpha**2 / (4.0) * (4 * np.pi / 3)**0.5
            dVL_avii.append(tmp)

        Pt_kasni = []
        nk = len(calc.wfs.kpt_u)
        for k_c in range(nk):
            Pt_kasni.append(calc.wfs.kpt_u[k_c].P_ani)

    else:
        Pt_kasni = None
        dVL_avii = None
    
    dVL_avii = broadcast(dVL_avii, 0)
    Pt_kasni = broadcast(Pt_kasni, 0)

    # Return the output
    return dVL_avii, Pt_kasni

    

def get_overlap(calc, ik, jk, bands=None):

    from gpaw.utilities.blas import gemmdot

    dO_aii = []
    for ia in calc.wfs.kpt_u[0].P_ani.keys():
        dO_ii = calc.wfs.setups[ia].dO_ii
        dO_aii.append(dO_ii)

    cell_cv = calc.wfs.gd.cell_cv
    icell_cv = (2 * np.pi) * np.linalg.inv(cell_cv).T
    r_g = calc.wfs.gd.get_grid_point_coordinates()
    Ng = np.prod(np.shape(r_g)[1:])
    Na = len(calc.atoms)
    if bands is None:
        Nn = calc.get_number_of_bands()
        bands = list(range(Nn))
    else:
        Nn = len(bands)

    # G_c = np.array([0, 0, 0])
    # G_c[direction] = 1
    # G_v = np.dot(G_c, icell_cv)
    # kpts_kc = calc.get_bz_k_points()
    kpts_kc = calc.wfs.kd.ibzk_kc
    kpts_kv = np.dot(kpts_kc, icell_cv)
    # if Nloc > 1:
    #     b_c = kpts_kc[kpts_kq[0][1]] - kpts_kc[kpts_kq[0][0]]
    #     b_v = np.dot(b_c, icell_cv)
    # else:
    #     b_v = G_v
    b_c = kpts_kc[ik] - kpts_kc[jk]
    bG_v = np.dot(b_c, icell_cv)

    P1_ani = {}
    P2_ani = {}
    for ia in range(Na):
        P0_ni = calc.wfs.kpt_u[ik].P_ani[ia]
        P1_ani[ia] = P0_ni
        P0_ni = calc.wfs.kpt_u[jk].P_ani[ia]
        P2_ani[ia] = P0_ni

    
    u1_nR = np.array([calc.wfs.get_wave_function_array(ib, ik, 0)
                      for ib in bands])
    u1_nR[:] *= np.exp(-1.0j * gemmdot(kpts_kv[ik], r_g, beta=0.0))
    u2_nR = np.array([calc.wfs.get_wave_function_array(ib, jk, 0)
                      for ib in bands])
    u2_nR[:] *= np.exp(-1.0j * gemmdot(kpts_kv[jk], r_g, beta=0.0))

    u1_nR  = np.reshape(u1_nR, (Nn, Ng))
    u2_nR  = np.reshape(u2_nR, (Nn, Ng))
    M_nn = np.dot(u1_nR.conj(), u2_nR.T) * calc.wfs.gd.dv
    r_av = np.dot(calc.spos_ac, cell_cv)

    for ia in range(len(P1_ani)):
        P1_ni = P1_ani[ia][bands]
        P2_ni = P2_ani[ia][bands]
        phase = np.exp(-1.0j * np.dot(bG_v, r_av[ia]))
        dO_ii = dO_aii[ia]
        M_nn += P1_ni.conj().dot(dO_ii).dot(P2_ni.T) * phase

    return M_nn


# Trace bands in a k path

def trace_bands_path(calc, kp_kc, bmax=2, ni=None, nf=None):

    # Useful variables
    nb = calc.get_number_of_bands()
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    E_nk = calc.band_structure().todict()['energies'][0]
    E_kn = E_nk[kp_kc, ni:nf]
    Ed_kn = np.diff(E_kn, axis=0)
    Bmax = np.amax(np.abs(Ed_kn))
    nk = len(kp_kc)
    nb2 = nf-ni

    def get_deg(E_n, E_th=1e-5):
        Ed_n = np.diff(E_n)
        dg_ii = np.where(Ed_n<1e-5)[0]
        dg_ls = []
        for dg_i in dg_ii:
            if dg_ls != [] and dg_ls[-1][-1] == dg_i:
                dg_ls[-1].append(dg_i+1)
            else:
                dg_ls.append([dg_i, dg_i+1])
        return dg_ls

    # Initialize step
    ik = 0
    E_n0 = E_kn[0]

    # Loop over k points along the path
    u_knn = np.zeros((nk, nb2, nb2))
    u_knn[0] = np.eye(nb2)
    for jk in range(1, nk):
        warn_flag = False
        warn_k = []
        warn_b = []
        E_n1 = E_kn[jk]

        # Goup the bands together
        E_nm = np.tile(E_n1[:, None], (1, nb2)) - \
                       np.tile(E_n0[None, :], (nb2, 1))
        cind = np.abs(E_nm)>bmax*Bmax
        E_nm[cind] = 0
        E_nm[~cind] = 1
        E_nm = E_nm.astype(int)
        nblist = []
        ib = 0
        while ib < nb2:
            for nrepi in range(2, nb2-ib+1):
                tmp = np.ones((nrepi, nrepi), int) - E_nm[ib:ib+nrepi, ib:ib+nrepi]
                if np.all(tmp[:, -1] == tmp[-1, :]) and np.sum(tmp[-1, :-1]) == nrepi-1:
                    nblist.append(np.arange(ib, ib+nrepi-1))
                    ib += nrepi-1
                    break
            if nrepi == nb2-ib or nb2-ib == 1:
                nblist.append(np.arange(ib, nb2))
                break
        # Check if all bands are included
        assert [item for sublist in nblist for item in sublist] == list(range(nb2)), 'Error in seperating the bands.'

        # Find the degenerate sets
        dg_ls0 = get_deg(E_n0)
        dg_ls1 = get_deg(E_n1)

        if jk == 3:
            aa = 1
        # print(jk, nblist)
        # Now comput the overlap for each group of bands
        u_nn = np.zeros((nb2, nb2))
        for bands in nblist:
            nbl = len(bands)
            # If it is a seperate band, it is easy
            if nbl == 1:
                u_nn[bands[0], bands[0]] = 1
                continue
            # Compute the overlap
            M_nn = get_overlap(calc, kp_kc[jk], kp_kc[ik], bands=bands)
            M_nn = np.abs(M_nn)**2
            M_nn[M_nn<0.01] = 0
            # M_nn = np.dot(M_nn, np.conj(M_nn.T))
            M_nn2 = M_nn.copy()
            M_nn2[M_nn >= 0.6] = 1
            M_nn2[M_nn < 0.6] = 0
            if np.all(np.sum(M_nn2, axis=0) == np.ones(nbl)) and np.all(np.sum(M_nn2, axis=1) == np.ones(nbl)):
                u_nn[bands[0]:bands[-1]+1, bands[0]:bands[-1]+1] = M_nn2
            else:
                # parprint('There is an issue with band sorting for {}, {}. Try to fix it.'.format(jk, bands))
                M_nn3 = M_nn2.copy()
                notok = np.where(np.sum(M_nn2, axis=1) != np.ones(nbl))[0]
                notpos = np.where(np.sum(M_nn2, axis=0) != np.ones(nbl))[0]
                nnok = len(notok)
                M_pp = M_nn[np.ix_(notok, notpos)]
                M_pp[M_pp < 0.1] = 0
                M_pp2 = M_pp.copy()
                M_pp2[np.where(np.amax(M_pp, axis=1)[:, None] == M_pp)] = 1
                M_pp3 = M_pp2.copy()
                M_pp3[M_pp3 < 1] = 0
                if np.all(np.sum(M_pp3, axis=0) == np.ones(nnok)) and np.all(np.sum(M_pp3, axis=1) == np.ones(nnok)):
                    M_nn3[np.ix_(notok, notpos)] = M_pp3
                else:
                    notok2 = np.where(np.sum(M_pp3, axis=0) != np.ones(nnok))[0]
                    # nopos2 = np.where(np.sum(M_pp3, axis=1) != np.ones(nnok))[0]
                    nnok2 = len(notok2)
                    M_ss = M_pp[np.ix_(notok2, notok2)]
                    M_ss2 = M_ss.copy()
                    M_ss2[np.where(np.amax(M_ss, axis=0) == M_ss)] = 1
                    M_ss2[M_ss2 < 1] = 0
                    if np.all(np.sum(M_ss2, axis=0) == np.ones(nnok2)) and np.all(np.sum(M_ss2, axis=1) == np.ones(nnok2)):
                        M_pp3[np.ix_(notok2, notok2)] = M_ss2
                    else:
                        warn_flag = True
                        warn_k.append(jk)
                        warn_b.append(bands[notok])
                        # parprint('There is an issue with band sorting for {}, {}. Not worked.'.format(jk, bands[notok]))
                        M_pp3[np.ix_(notok2, notok2)] = np.eye(nnok2)
                    if np.all(np.sum(M_pp3, axis=0) == np.ones(nnok)) and np.all(np.sum(M_pp3, axis=1) == np.ones(nnok)):
                        M_nn3[np.ix_(notok, notpos)] = M_pp3
                    else:
                        M_nn3[np.ix_(notok, notpos)] = np.eye(nnok)
                if np.all(np.sum(M_nn3, axis=0) == np.ones(nbl)) and np.all(np.sum(M_nn3, axis=1) == np.ones(nbl)):
                    u_nn[bands[0]:bands[-1]+1, bands[0]:bands[-1]+1] = M_nn3
                else:
                    warn_flag = True
                    warn_k.append(jk)
                    warn_b.append(bands[notok])
                    # parprint('There is an issue with band sorting for {}, {}. Not worked.'.format(jk, bands[notok]))
                    u_nn[bands[0]:bands[-1]+1, bands[0]:bands[-1]+1] = np.eye((nbl))
                    # plt.matshow(M_nn3)
                    # plt.tight_layout()
                    # plt.savefig('mat.png', dpi=300)
                    # aa = 1
        
        if warn_flag == True:
            for ik, ib in zip(warn_k, warn_b):
                # parprint('There is an issue with band sorting for {}, {}.'.format(ik, ib))
                pass
        # u_knn[jk] = np.dot(u_knn[jk-1], u_nn)
        u_knn[jk] = np.dot(u_nn, u_knn[jk-1])
        E_n0 = E_n1
        ik = jk


    # Emin = np.amin(E_nk, axis=0)
    # Emax = np.amax(E_nk, axis=0)
    # Emm = Emax-Emin
    # Ed_kn = np.diff(E_nk, axis=1)
    # E_knm = np.tile(E_nk[:, :, None], (1, 1, nb)) - \
    #                 np.tile(E_nk[:, None, :], (1, nb, 1))
    
    
    

    # Emin = np.amin(E_kn, axis=0)
    # Emax = np.amax(E_kn, axis=0)
    # Emm = Emax-Emin
    # u_knn = follow_bands(calc, kp_kc, bmax=2, ni=0, nf=None)
    # u_knn = follow_bands(calc, kp_kc, bmax=2, ni=0, nf=4)
    # E_kn = E_nk[kp_kc, 0:nb]
    # Enew_kn = np.array([np.dot(E_kn[ik].T, u_knn[ik]) for ik in range(0, len(kp_kc))])
    # plt.plot(np.arange(0, len(kp_kc)), Enew_kn)
    # # plt.ylim([-10, 10])
    # # plt.tight_layout()
    # # plt.savefig('mat.png', dpi=300)
    # # plt.close()
    # plt.plot(np.arange(0, len(kp_kc)), E_kn, '--')
    # # plt.ylim([-38.8, -38.2])
    # plt.ylim([-10, 10])
    # plt.tight_layout()
    # plt.savefig('mat2.png', dpi=300)
    # plt.close()
    # for ib in range(nb):
    #     E_k = E_nk[:, ib]
    #     Ed_k = 

    # Return the output
    return u_knn

# Trace bands in whole BZ, with respect to Gamma-point

def trace_bands(gs_name='gs', out_name='rot'):

    # if world.rank == 0:
    timer = Timer()
    parprint(
        'Make the matrices for tracing bands in BZ (in {:d} cores).'.format(
            world.size))

    # Load the calculations
    with timer('Load the calculations'):
        calc = GPAW(gs_name+'.gpw', txt=None, parallel={'kpt':1,'band':1}, communicator=serial_comm)
    kd = calc.wfs.kd
    nb = calc.get_number_of_bands()
    w_k = calc.get_k_point_weights()
    nk = len(w_k)

    # Make the k-path list
    with timer('Make the k point lists'):
        N_c = kd.N_c
        kp0 = np.arange(0, nk, N_c[1])
        # kp0[0] = kp0[1]
        # kp0[1] = 0
        kp0[-1] += -int(N_c[1]/2-1)
        kp_jkc = []
        for k0 in kp0:
            kp_jkc.append(np.arange(k0, k0+int(N_c[1]/2+1)))
            if k0 != 0 and k0 != nk-int(N_c[1]/2+1):
                kp_jkc.append(np.arange(k0, k0-int(N_c[1]/2), -1))
        parprint('There are {} k paths in the BZ.'.format(len(kp_jkc)))

    with timer('Fix the first path'):
        u_knn = np.zeros((nk, nb, nb))
        if world.rank == 0:
            u_knn0 = trace_bands_path(calc, kp0, bmax=2, ni=0, nf=None)
            u_knn[kp0] = u_knn0
        else:
            u_knn0 = None
        u_knn0 = broadcast(u_knn0)

    with timer('Fix the rest'):
        # Initial call to print 0% progress
        count = 0
        ncount = np.ceil(len(kp_jkc)/world.size)
        # ncount = len(kp_jkc)
        print_progressbar(
            count, ncount,
            prefix='Progress:',
            suffix='Complete',
            length=50)
        for jk, kp_kc in enumerate(kp_jkc):
            if jk % world.size == world.rank:
                u_knn1 = trace_bands_path(calc, kp_kc, bmax=2, ni=0, nf=None)
                k0i = np.where(kp0 == kp_kc[0])[0]
                # u_knn[kp_kc[0]] = u_knn0[k0i]
                for ik in range(1, len(kp_kc)):
                    u_knn[kp_kc[ik]] = np.dot(u_knn0[k0i], u_knn1[ik])
            
                # Print the progress
                if world.rank == 0:
                    count += 1
                    print_progressbar(
                        count,
                        ncount,
                        prefix='Progress:',
                        suffix='Complete',
                        length=50)

    # Sum over all nodes
    with timer('Gather data from cores'):
        world.sum(u_knn)

    # Save the data
    if world.rank == 0:
        # Save it to the file
        np.save(out_name+'.npy', u_knn)

        # Print the timing
        timer.write()