
# Import the required modules: General
import numpy as np
from math import pi, floor
import matplotlib.pyplot as plt


# Import the required modules: GPAW/ASE
from gpaw import GPAW, PW, FermiDirac
from ase.units import Bohr, _hbar, _e, _me, _eps0, alpha
from gpaw.spinorbit import get_spinorbit_eigenvalues, get_radial_potential
from ase.utils.timing import Timer
from gpaw.mpi import world, broadcast, serial_comm
from gpaw.berryphase import parallel_transport
from gpaw.fd_operators import Gradient
from ase.dft.dos import linear_tetrahedron_integration
from gpaw.response.pair import PairDensity

# Import the required modules: nlo
from gpaw.nlopt.output import print_progressbar, plot_spectrum, plot_polar
from gpaw.nlopt.output import is_file_exist, parprint, plot_kfunction
from gpaw.nlopt.mml import *
from gpaw.nlopt.symmetry import get_tensor_elements

# Check the sum rule


def check_sumrule(
        addsoc=False,
        socscale=1.0,
        Etol=1e-3, ftol=1e-4,
        ni=None, nf=None,
        momname=None,
        basename=None):
    """
    Check the sum rule sum fnm*(pnm0*pmn1+pnm1*pmn0)/Emn

    Input:
        addsoc          Add spin-orbit coupling (default off)
        socscale        SOC scale (default 1.0)
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        ni, nf          First and last bands in the calculations (def. 0 to nb)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        sumrule         It should be zero or one for complete basis set
    """

    timer = Timer()
    parprint('Calculating the sum rule (using {:d} cores).'.format(world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    nb2 = nf - ni
    gspin = 2.0
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            nv = calc.get_number_of_electrons()
            E_nk = calc.band_structure().todict()['energies'][0]
            f_nk = np.zeros((nk, nb), dtype=np.float64)
            for k_c in range(nk):
                f_nk[k_c] = calc.get_occupation_numbers(
                    kpt=k_c, spin=0) / w_k[k_c] / 2.0
        else:
            nv = None
            E_nk = None
            f_nk = None

    # Get eigenvalues and wavefunctions with spin orbit coupling (along the
    # z-direction)
    if addsoc:
        gspin = 1.0
        nb2 = 2 * (nf - ni)
        with timer('Spin-orbit coupling calculation'):
            if world.rank == 0:
                # Get the SOC
                (e_mk, wfs_knm) = get_spinorbit_eigenvalues(
                    calc, return_wfs=True, bands=np.arange(ni, nf),
                    scale=socscale)

                # Make the data ready
                e_mk = e_mk.T
                wfs_knm = np.array(wfs_knm)

                # Update the Fermi level
                mu = fermi_level(calc,
                                 e_mk[np.newaxis],
                                 2 * calc.get_number_of_electrons())
            else:
                e_mk = None
                wfs_knm = None
                mu = None
            mu = broadcast(mu, 0)

            # Distribute the SOC
            k_info2 = distribute_data(
                [e_mk, wfs_knm], [(nk, nb2), (nk, nb2, nb2)])
            
            # Get SOC from PAW
            dVL_avii, Pt_kasni = get_soc_paw(calc)
            # phi_km, S_km = parallel_transport(calc, direction=0, spinors=True, scale=1.0e-6)

    # Distribute the k points between cores
    nv = broadcast(nv, 0)
    with timer('k-info distribution'):
        k_info = distribute_data(
            [moms, E_nk, f_nk], [
                (nk, 3, nb, nb), (nk, nb), (nk, nb)])

    # Initial call to print 0% progress
    count = 0
    ncount = len(k_info)
    print_progressbar(count, ncount)

    # Loop over the k points
    sumrule = np.zeros(9, complex)
    sumrule2 = np.zeros(9, complex)
    sumrule0 = np.zeros(9, complex)
    for k_c, data_k in k_info.items():
        mom, E_n, f_n = tuple(data_k)
        mom = mom[:, ni:nf, ni:nf]
        mom0 = mom.copy()
        f_n0 = f_n.copy()
        f_n0 = f_n0.real

        # Make the position matrix elements and Delta
        with timer('Position matrix elements calculation'):
            pos0 = np.zeros((3, nb, nb), complex)
            E_nm0 = np.tile(E_n[:, None], (1, nb)) - \
                np.tile(E_n[None, :], (nb, 1))
            zeroind = np.abs(E_nm0) < Etol
            E_nm0[zeroind] = 1
            for aa in range(3):
                pos0[aa] = mom0[aa] / (1j * E_nm0)
                pos0[aa, zeroind] = 0
            # pos = pos0

        # Need SOC or not
        if addsoc:
            E_n, wfs_nm = tuple(k_info2.get(k_c))
            # tmp = (E_n - mu) / kbT
            # tmp = np.clip(tmp, -100, 100)
            # f_n = 1 / (np.exp(tmp) + 1.0)
            f_n = np.zeros(nb2, float)
            f_n[::2] = f_n0
            f_n[1::2] = f_n0

            # Make the new momentum
            with timer('New momentum calculation'):
                mom2 = np.zeros((3, nb2, nb2), dtype=complex)
                mom2s = np.zeros((3, nb2, nb2), dtype=complex)
                for pp in range(3):
                    mom2[pp] = np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                                mom0[pp]),
                                        np.transpose(wfs_nm[:, 0::2]))
                    mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                                mom0[pp]),
                                        np.transpose(wfs_nm[:, 1::2]))
            
                    # Get the soc correction to momentum
                    p_vmm = socscale * \
                            get_soc_momentum(dVL_avii, Pt_kasni[k_c], ni, nf)
                    mom2[pp] -= 1* np.dot(np.dot(np.conj(wfs_nm), p_vmm[pp]),
                                        np.transpose(wfs_nm))
                mom = mom2
        else:
            E_n = E_n[ni:nf]
            f_n = f_n[ni:nf]

        # Make the position operator from another method
        if addsoc:
            pos2 = np.zeros((3, nb2, nb2), complex)
            for pp in range(3):
                pos2[pp] = np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                            pos0[pp]),
                                    np.transpose(wfs_nm[:, 0::2]))
                pos2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                            pos0[pp]),
                                    np.transpose(wfs_nm[:, 1::2]))

            E_nm2 = np.tile(E_n[:, None], (1, nb2)) - \
                np.tile(E_n[None, :], (nb2, 1))
            pos = pos2
            mom3 = np.array([1j*pos2[v]*E_nm2 for v in range(3)]) 

            # print(mom2[1, 25, 26])
            # print(mom3[1, 25, 26])
            # print(mom2[1, 25, 26]-50*mom2s[1, 25, 26])
            # print(mom2s[1, 25, 26])
            # print((mom3[1, 25, 26]-mom2[1, 25, 26])/mom2s[1, 25, 26])
           
            
        # aa = np.dot(np.conj(wfs_nm), wfs_nm.T)
        # parprint(np.allclose(aa, np.eye(nb2), rtol=1e-08, atol=1e-12))

        # Loop over bands
        E_n = E_n.real
        f_n = f_n.real
        f_n0 = f_n0.real
        with timer('Sum over bands'):
            for nni in range(nb2):
                for mmi in range(nb2):
                    Emn = E_n[mmi] - E_n[nni]
                    fnm = f_n[nni] - f_n[mmi]
                    if np.abs(fnm) < ftol or np.abs(Emn) < Etol:
                        continue
                    for pol in range(9):
                        aa, bb = pol % 3, floor(pol / 3)
                        sumrule[pol] += fnm * \
                            (mom[aa, nni, mmi] * mom[bb, mmi, nni]) / \
                            Emn * w_k[k_c] / 1.0
                        sumrule2[pol] += fnm * \
                            (mom3[aa, nni, mmi] * pos[bb, mmi, nni]) \
                            * w_k[k_c] / 1.0

            for nni in range(nb):
                for mmi in range(nb):
                    fnm = f_n0[nni] - f_n0[mmi]
                    if np.abs(fnm) < ftol:
                        continue
                    for pol in range(9):
                        aa, bb = pol % 3, floor(pol / 3)
                        sumrule0[pol] += fnm * \
                            (mom0[aa, nni, mmi] * pos0[bb, mmi, nni]) \
                            * w_k[k_c] / 1.0

        # Print the progress
        count += 1
        print_progressbar(count, ncount)

    # Sum over all nodes
    world.sum(sumrule)
    world.sum(sumrule2)
    world.sum(sumrule0)

    # Make the output and print it
    sum_rule = gspin * (_hbar / (Bohr * 1e-10))**2 / (_e * _me) / nv * sumrule
    sum_rule = np.reshape(sum_rule, (3, 3), order='C')
    parprint('The sum rule matrix is:')
    parprint(np.array_str(sum_rule, precision=6, suppress_small=True))

    sum_rule2 = gspin * (_hbar / (Bohr * 1e-10))**2 / (_e * _me) / nv * (sumrule2*1j)
    sum_rule2 = np.reshape(sum_rule2, (3, 3), order='C')
    parprint('The sum rule matrix is:')
    parprint(np.array_str(sum_rule2, precision=6, suppress_small=True))

    sum_rule0 = 2 * (_hbar / (Bohr * 1e-10))**2 / (_e * _me) / nv * (sumrule0*1j)
    sum_rule0 = np.reshape(sum_rule0, (3, 3), order='C')
    parprint('The sum rule matrix is:')
    parprint(np.array_str(sum_rule0, precision=6, suppress_small=True))

    # Print the timing
    if world.rank == 0:
        timer.write()

    # Return the sum_rule
    return sum_rule


# Check the sum rule nr 2


def check_sumrule2(
        bands=[0],
        ni=None, nf=None,
        momname=None,
        basename=None,
        figname=None):
    """
    Check the sum rule 2

    Input:
        ni, nf          First and last bands in the calculations (def. 0 to nb)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        sumrule         It should be zero or one for complete basis set
    """

    timer = Timer()
    parprint('Calculating the sum rule 2 (using {:d} cores).'.format(world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    Etol = 1.0e-3
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    nb2 = nf - ni
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            E_nk = calc.band_structure().todict()['energies'][0]
        else:
            E_nk = None
    
    # Only do the calculations in master
    if world.rank == 0:
        # Loop over the k points
        sumrule = np.zeros((3, 3, nk, len(bands)), dtype=complex)

        mom = moms[:, :, ni:nf, ni:nf]
        mom = np.swapaxes(mom, 0, 1)
        p_nn = np.einsum('...ii->...i', mom)
        p_nn = p_nn.real
        E_kn = E_nk[:, ni:nf].real

        # Useful variables
        nk = len(kd.ibz2bz_k)
        N_c = kd.N_c

        # Find the neighbors
        qind=[[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        Nq = len(qind)+1
        assert N_c[2] == 1, 'Triangular method is only implemented for 2D systems.'
        q_vecs = []
        for ii, qq in enumerate(qind):
            q_vecs.append([qq[0] / N_c[0], qq[1] / N_c[1], qq[2] / N_c[2]])
        nkt = len(kd.bzk_kc)
        neighbors = np.zeros((Nq, nkt), dtype=np.int32)
        neighbors[0] = np.arange(nkt)
        for ind, qpt in enumerate(q_vecs):
            neighbors[ind + 1] = np.array(kd.find_k_plus_q(qpt))

        # Depending on the tsym set variables
        tsym = kd.symmetry.time_reversal
        if tsym is False:
            nei_ind = kd.bz2ibz_k[neighbors]
            nei_ind = nei_ind[:, kd.ibz2bz_k]
            p1 = 0
            p2 = nk
            psigns = np.ones((nk), dtype=int)
        else:
            # nei_ind = neighbors[:, kd.ibz2bz_k]
            # nei_ind = [kd.bz2ibz_k[nei_ind[ii]] for ii in range(Nq)]
            nei_ind = kd.bz2ibz_k[neighbors]
            nei_ind = nei_ind[:, kd.ibz2bz_k]
            psigns = -2 * kd.time_reversal_k + 1
            psigns = psigns[neighbors[:, kd.ibz2bz_k]]
            
        # Compute the drivatives in ab directions
        dp_vvkn = np.zeros((2, 3, nk, len(bands)))
        dE_vkn = np.zeros((2, nk, len(bands)))
        for nni, nn in enumerate(bands):
            for v1 in range(2):
                vv1 = v1*2+1
                dE_vkn[v1, :, nni] = (E_kn[nei_ind[vv1], nn]-E_kn[nei_ind[vv1+1], nn]) / (2.0/N_c[v1])
                for v2 in range(3):
                    dp_vvkn[v1, v2, :, nni] = (psigns[vv1] * p_nn[v2, nei_ind[vv1], nn]
                                              - psigns[vv1+1] * p_nn[v2, nei_ind[vv1+1], nn]) / (2.0/N_c[v1])

        # Transform to xy
        cell = calc.atoms.cell[:2, :2]
        icell = 2*pi*calc.wfs.gd.icell_cv[:2, :2]
        dp_vvkn2 = np.einsum('ij,jlkn->ilkn', icell, dp_vvkn)
        dE_vkn2 = np.einsum('ij,jkn->ikn', icell, dE_vkn)
        scale = (_me*(Bohr*1e-10)**2*_e/_hbar**2)

        # Loop over bands
        with timer('Sum over bands'):
            scale = (_hbar / (Bohr * 1e-10))**2 / (_e * _me)
            for pol in range(9):
                aa, bb = pol % 3, floor(pol / 3)
                sumrule[aa, bb] = (aa == bb)*1
                for nni, nn in enumerate(bands):
                    for mmi in range(nf-ni):
                        Enm = E_kn[:, nn] - E_kn[:, mmi]
                        if np.all(np.abs(Enm)) < Etol:
                            continue
                        usind = np.where(np.abs(Enm)>Etol)[0]
                        sumrule[aa, bb, usind, nni] += scale * (mom[aa, usind, nn, mmi] * mom[bb, usind, mmi, nn] 
                                                    + mom[bb, usind, nn, mmi] * mom[aa, usind, mmi, nn]).real / Enm[usind]

    else:
        sumrule = None
    sumrule = broadcast(sumrule, 0)

    # plot_kfunction(kd, sumrule[1, 1, :, 0], figname='xx_band12_sum.png', tsymm='even', dtype='re', clim=(-3,3))
    # plot_kfunction(kd, dp_vvkn2[1, 1, :, 0], figname='xx_band12_dp.png', tsymm='even', dtype='re', clim=(-3,3))

    # Print the timing
    if world.rank == 0:
        timer.write()

    # Return the sum_rule
    return sumrule

# Calculate the linear response


def calculate_df(
        freqs=1.0,
        eta=0.05,
        pol='xx',
        eshift=0.0,
        addsoc=False,
        socscale=1.0,
        intmethod='no',
        Etol=1e-3, ftol=1e-4,
        ni=None, nf=None,
        blist=None,
        outname=None,
        momname=None,
        basename=None):
    """
    Calculate the RPA linear response  (nonmagnetic semiconductors)

    Input:
        freqs           Excitation frequency array (a numpy array or list)
        eta             Broadening (a single number or an array)
        pol             Tensor element
        eshift          scissors shift (default 0)
        addsoc          Add spin-orbit coupling (default off)
        socscale        SOC scale (default 1.0)
        intmethod       Integral method (defaul no)
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        ni, nf          First and last bands in the calculations (0 to nb)
        blist           List of bands in the sum
        outname         Output filename (default is df.npy)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        df.npy          Numpy array containing the spectrum and frequencies
    """

    timer = Timer()
    parprint(
        'Calculating the linear spectrum (using {:d} cores).'.format(
            world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = freqs + 1j * eta
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    polstr = pol
    pol = ['xyz'.index(ii) for ii in polstr]
    gspin = 2.0
    nb2 = nf - ni
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            E_nk = calc.band_structure().todict()['energies'][0]
            f_nk = np.zeros((nk, nb), dtype=np.float64)
            for k_c in range(nk):
                f_nk[k_c] = calc.get_occupation_numbers(
                    kpt=k_c, spin=0) / w_k[k_c] / 2.0
        else:
            E_nk = None
            f_nk = None

    # Get eigenvalues and wavefunctions with spin orbit coupling (along the
    # z-direction)
    if addsoc:
        gspin = 1.0
        nb2 = 2 * (nf - ni)
        with timer('Spin-orbit coupling calculation'):
            if world.rank == 0:
                # Get the SOC
                (e_mk, wfs_knm) = get_spinorbit_eigenvalues(
                    calc, return_wfs=True, bands=np.arange(ni, nf),
                    scale=socscale)

                # Make the data ready
                e_mk = e_mk.T
                wfs_knm = np.array(wfs_knm)

                # Update the Fermi level
                mu = fermi_level(calc,
                                 e_mk[np.newaxis],
                                 2 * calc.get_number_of_electrons())
            else:
                e_mk = None
                wfs_knm = None
                mu = None
            mu = broadcast(mu, 0)

            # Distribute the SOC
            k_info2 = distribute_data(
                [e_mk, wfs_knm], [(nk, nb2), (nk, nb2, nb2)])
            
            # Get SOC from PAW
            dVL_avii, Pt_kasni = get_soc_paw(calc)

    parprint(
        'Evaluating linear response for {}-polarization ({:.1f} meV).'.format(
            polstr,
            1000 *
            eta))

    # Choose the integration method
    if blist is None:
        blist = list(range(nb2))
    else:
        assert max(
            blist) < nb2, 'Maximum of blist should be smaller than nb.'

    if intmethod == 'no':
        # Initialize variables
        suml = np.zeros((nw), dtype=complex)

        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info = distribute_data(
                [moms, E_nk, f_nk], [(nk, 3, nb, nb), (nk, nb), (nk, nb)])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over the k points
        for k_c, data_k in k_info.items():
            mom, E_n, f_n = tuple(data_k)
            mom = mom[:, ni:nf, ni:nf]

            # Need SOC or not
            if addsoc:
                E_n, wfs_nm = tuple(k_info2.get(k_c))
                tmp = (E_n - mu) / kbT
                tmp = np.clip(tmp, -100, 100)
                f_n = 1 / (np.exp(tmp) + 1.0)

                # Make the new momentum
                with timer('New momentum calculation'):
                    mom2 = np.zeros((3, nb2, nb2), dtype=complex)
                    mom2s = np.zeros((3, nb2, nb2), dtype=complex)
                    for pp in range(3):
                        mom2[pp] += 1*np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 0::2]))
                        mom2[pp] += 1*np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 1::2]))
                
                        # Get the soc correction to momentum
                        p_vmm = socscale * \
                                get_soc_momentum(dVL_avii, Pt_kasni[k_c],
                                                 ni, nf)
                        mom2s[pp] += np.dot(np.dot(np.conj(wfs_nm), p_vmm[pp]),
                                           np.transpose(wfs_nm))
                    mom = mom2
            else:
                E_n = E_n[ni:nf]
                f_n = f_n[ni:nf]

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    # Loop over second band
                    for mmi in blist:
                        # Get Emn
                        fnm = f_n[nni] - f_n[mmi]
                        Emn = E_n[mmi] - E_n[nni]
                        
                        if np.abs(fnm) < ftol or np.abs(Emn) < Etol:
                            continue
                        pnm = mom[pol[0], nni, mmi] * mom[pol[1], mmi, nni]
                        suml += fnm * np.real(pnm) / (
                            Emn * (w_lc - Emn - fnm * eshift)) \
                            * w_k[k_c]  # *2/2 real/TRS

            # Print the progress
            count += 1
            print_progressbar(count, ncount)

    elif intmethod == 'tri':
        # Useful variable
        w_l = w_lc.real
        # Extend the frequency list for Hilbert transform
        w_l = np.hstack((w_l, w_l[-1] + w_l))
        nw = 2 * nw
        nb2 = nf - ni

        # Initialize variables
        suml = np.zeros((nw), dtype=complex)
        assert not addsoc, 'Triangular method is only implemented without SOC.'

        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info, dA = get_neighbors(moms, E_nk, f_nk, kd, nb, qind=[
                                       [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over the k points
        for k_c, data_k in k_info.items():
            # print(data_k)
            mom0, mom1, mom2, mom3, E_n0, E_n1, E_n2, E_n3, \
                f_n0, f_n1, f_n2, f_n3 = tuple(data_k)
            mom = np.array([mom0[:, ni:nf, ni:nf], mom1[:, ni:nf, ni:nf],
                            mom2[:, ni:nf, ni:nf], mom3[:, ni:nf, ni:nf]])
            E_n = np.array([E_n0[ni:nf], E_n1[ni:nf],
                            E_n2[ni:nf], E_n3[ni:nf]])
            f_n = np.array([f_n0[ni:nf], f_n1[ni:nf],
                            f_n2[ni:nf], f_n3[ni:nf]])

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # Do not do unimportant calculations
                        if np.all(np.abs(f_n[:, nni] - f_n[:, mmi])) < ftol:
                            continue

                        # Comput the value of integrand on triangles
                        pnmn = np.real(mom[:, pol[0], nni, mmi]
                                * mom[:, pol[1], mmi, nni])
                        # Factor 2 is due for area of rectangles 
                        Fval = -1j * pnmn * (f_n[:, nni] - f_n[:, mmi]) \
                            / (E_n[:, mmi] - E_n[:, nni]) * dA / 2.0
                        Eval = E_n[:, mmi] - E_n[:, nni]

                        # Use the triangle method for integration
                        suml += triangle_int(Fval, Eval, w_l, itype=1)

            # Print the progress
            count += 1
            print_progressbar(count, ncount)

    # elif intmethod == 'tri2':
    #     w_l = w_lc.real
    #     # Extend the frequency list for Hilbert transform
    #     w_l = np.hstack((w_l, w_l[-1] + w_l))
    #     nw = 2 * nw
    #     nb2 = nf - ni
    #     suml = np.zeros((nw), dtype=complex)
    #     bz2ibz = calc.get_bz_to_ibz_map()

    #     # Loop over bands
    #     with timer('Sum over bands'):
    #         for nni in blist:
    #             for mmi in blist:
    #                 fnm = f_nk[:, nni]-f_nk[:, mmi]
    #                 Emn = E_nk[:, mmi]-E_nk[:, nni]
    #                 zeroind = np.abs(Emn) < Etol
    #                 Emn[zeroind] = 1
    #                 fnm[zeroind] = 0
    #                 pnmn = np.real(moms[:, pol[0], nni, mmi]
    #                                * moms[:, pol[1], mmi, nni])
    #                 weight = fnm*pnmn/Emn
    #                 Emn[zeroind] = 0
    #                 suml += linear_tetrahedron_integration(calc.atoms.cell, 
    #                                                np.reshape(Emn[bz2ibz], (kd.N_c[0], kd.N_c[1])),
    #                                                w_l,
    #                                                weight)

    else:
        parprint('Integration mode ' + intmethod + ' not implemented.')
        raise NotImplementedError

    # Sum over all nodes
    world.sum(suml)

    # Make the output in SI unit
    dim_sigma = gspin * 1j * _e**2 * _hbar / (_me**2 * (2 * pi)**3)
    dim_chi = 1j * _hbar / (_eps0 * _e)
    dim_sum = (_hbar / (Bohr * 1e-10))**2 / \
        (_e**2 * (Bohr * 1e-10)**3) * bz_vol
    dim_SI = dim_sigma * dim_chi * dim_sum

    if intmethod == 'no':
        chi1 = dim_SI * suml / w_lc
    elif intmethod == 'tri':
        # Make the real part of the chi and broaden the spectra
        with timer('Hilbert transform'):
            suml2 = suml.imag / w_l
            suml_b = np.zeros(len(w_l), dtype=complex)
            for ind, omega in enumerate(w_lc):
                suml_b[ind] = np.trapz(
                    suml2 * w_l / (w_l**2 - omega**2), w_l)
            chi1 = dim_SI * 2 * suml_b
            chi1 = chi1[:int(nw / 2)]
            # chi1 = dim_SI * suml
    df = np.vstack((freqs, chi1))

    if world.rank == 0:
        # Save the output
        if outname is None:
            np.save('df.npy', df)
        else:
            np.save('{}.npy'.format(outname), df)
            # np.savetxt('{}.txt'.format(outname), np.transpose(df))

        # Print the timing
        timer.write()

    # Return df
    return df

# Calculate the SHG response (for nonmagnetic semiconductors) in the
# velocity gauge, reqularized version


def calculate_shg_rvg(
        freqs=1.0,
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        addsoc=False,
        socscale=1.0,
        intmethod='no',
        summethod='loop',
        Etol=1e-3, ftol=1e-4,
        ni=None, nf=None,
        blist=None,
        outname=None,
        momname=None,
        basename=None):
    """
    Calculate RPA SHG spectrum in velocity gauge (nonmagnetic semiconductors)

    Input:
        freqs           Excitation frequency array (a numpy array or list)
        eta             Broadening (a single number or an array)
        pol             Tensor element
        addsoc          Add spin-orbit coupling (default off)
        socscale        SOC scale (default 1.0)
        intmethod       Integral method (defaul no)
        summethod       Fast summation flag (default loop)
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        ni, nf          First and last bands in the calculations (0 to nb)
        blist           List of bands in the sum
        outname         Output filename (default is shg.npy)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        shg.npy          Numpy array containing the spectrum and frequencies
    """

    timer = Timer()
    parprint(
        'Calculating SHG spectrum in velocity gauge (in {:d} cores).'.format(
            world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = freqs + 1j * eta
    w_lc = np.hstack((-w_lc[-1::-1], w_lc))
    nw = 2 * nw
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    polstr = pol
    pol = ['xyz'.index(ii) for ii in polstr]
    gspin = 2.0
    nb2 = nf - ni
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            E_nk = calc.band_structure().todict()['energies'][0]
            f_nk = np.zeros((nk, nb), dtype=np.float64)
            for k_c in range(nk):
                f_nk[k_c] = calc.get_occupation_numbers(
                    kpt=k_c, spin=0) / w_k[k_c] / 2.0
        else:
            E_nk = None
            f_nk = None

    # Get eigenvalues and wavefunctions with spin orbit coupling (along the
    # z-direction)
    if addsoc:
        gspin = 1.0
        nb2 = 2 * (nf - ni)
        with timer('Spin-orbit coupling calculation'):
            if world.rank == 0:
                # Get the SOC
                (e_mk, wfs_knm) = get_spinorbit_eigenvalues(
                    calc, return_wfs=True, bands=np.arange(ni, nf),
                    scale=socscale)

                # Make the data ready
                e_mk = e_mk.T
                wfs_knm = np.array(wfs_knm)

                # Update the Fermi level
                mu = fermi_level(calc,
                                 e_mk[np.newaxis],
                                 2 * calc.get_number_of_electrons())
            else:
                e_mk = None
                wfs_knm = None
                mu = None

            # Update the Fermi level
            mu = broadcast(mu, 0)

            # Distribute the SOC
            k_info2 = distribute_data(
                [e_mk, wfs_knm], [(nk, nb2), (nk, nb2, nb2)])

            # Get SOC from PAW
            dVL_avii, Pt_kasni = get_soc_paw(calc)

    # Initialize the outputs
    sum2 = np.zeros((nw), dtype=np.complex)
    sum2b_D = np.zeros((nw), dtype=np.complex)
    sum2b_d = np.zeros((nw), dtype=np.complex)
    if blist is None:
        blist = list(range(nb2))
    else:
        assert max(
            blist) < nb2, 'Maximum of blist should be smaller than nb.'

    # Depending on the integration method
    parprint(
        'Evaluating SHG response for {}-polarization ({:.1f} meV).'.format(
            polstr,
            1000 *
            eta))
    if intmethod == 'no':
        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info = distribute_data(
                [moms, E_nk, f_nk], [(nk, 3, nb, nb), (nk, nb), (nk, nb)])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over k points
        for k_c, data_k in k_info.items():
            mom, E_n, f_n = tuple(data_k)
            mom = mom[:, ni:nf, ni:nf]
            f_n0 = f_n.copy()

            # Add SOC or not
            if addsoc:
                E_n, wfs_nm = tuple(k_info2.get(k_c))
                E_n = E_n.real
                tmp = (E_n - mu) / kbT
                tmp = np.clip(tmp, -100, 100)
                f_n = 1 / (np.exp(tmp) + 1.0)
                f_n = np.zeros(nb2, complex)
                f_n[::2] = f_n0
                f_n[1::2] = f_n0

                # Make the new momentum
                with timer('New momentum calculation'):
                    mom2 = np.zeros((3, nb2, nb2), dtype=complex)
                    for pp in range(3):
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 0::2]))
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 1::2]))

                        # Get the soc correction to momentum
                        p_vmm = socscale * get_soc_momentum(dVL_avii, Pt_kasni[k_c], ni, nf)
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm), p_vmm[pp]),
                                           np.transpose(wfs_nm))
                    mom = mom2
            else:
                E_n = E_n[ni:nf].real
                f_n = f_n[ni:nf].real

            # Loop over bands
            with timer('Sum over bands'):
                # Shift the bands
                # E_nk += eshift/2.0*np.sign(E_n)
                # Loop over bands
                for nni in blist:
                    for mmi in blist:
                        # Remove the non important term using time-reversal
                        # symmtery
                        if mmi <= nni:
                            continue

                        # Useful variables
                        fnm = f_n[nni] - f_n[mmi]
                        Emn = E_n[mmi] - E_n[nni] + fnm * eshift
                        

                        # Comute the 2-band term
                        if np.abs(Emn) > Etol and np.abs(fnm) > ftol:
                            pnml = (mom[pol[0], nni, mmi]
                                    * (mom[pol[1], mmi, nni]
                                       * (mom[pol[2], mmi, mmi]
                                          - mom[pol[2], nni, nni])
                                       + mom[pol[2], mmi, nni]
                                       * (mom[pol[1], mmi, mmi]
                                          - mom[pol[1], nni, nni]))
                                    * w_k[k_c] / 2)
                            sum2b_d += 1j * fnm * np.imag(pnml) * \
                                (1 / (Emn**4 * (w_lc - Emn)) -
                                 16 / (Emn**4 * (2 * w_lc - Emn)))

                        if summethod == 'fast':
                            # Required parameters
                            fnl = f_n[nni] - f_n[blist]
                            fml = f_n[mmi] - f_n[blist]

                            # Comute the 3-band term
                            if np.abs(Emn) > Etol:
                                if np.any(np.abs(fnl)) > ftol:
                                    Eln = E_n[blist] - E_n[nni] + fnl * eshift
                                    pnml = (mom[pol[0], nni, mmi]
                                            * (mom[pol[1], mmi, blist]
                                               * mom[pol[2], blist, nni]
                                               + mom[pol[2], mmi, blist]
                                               * mom[pol[1], blist, nni]))
                                    pnml = 1j * np.imag(pnml)
                                    Fval = (8 / (Emn**3)
                                            * np.sum(pnml * fnl
                                                     / (Emn - 2 * Eln)))
                                    sum2b_D += (Fval * w_k[k_c]
                                                / 2 / (w_lc - Emn / 2))
                                if np.any(np.abs(fml)) > ftol:
                                    Eml = E_n[mmi] - E_n[blist] - fml * eshift
                                    pnml = (mom[pol[0], nni, mmi]
                                            * (mom[pol[1], mmi, blist]
                                               * mom[pol[2], blist, nni]
                                               + mom[pol[2], mmi, blist]
                                               * mom[pol[1], blist, nni]))
                                    pnml = 1j * np.imag(pnml)
                                    Fval = (8 / (Emn**3)
                                            * np.sum(pnml * fml
                                                     / (Emn - 2 * Eml)))
                                    sum2b_D += Fval * \
                                        w_k[k_c] / 2 / (w_lc - Emn / 2)
                                if np.any(np.abs(fnm)) > ftol:
                                    Eln = E_n[blist] - E_n[nni] + fnl * eshift
                                    pnml = (mom[pol[0], nni, blist]
                                            * (mom[pol[1], blist, mmi]
                                               * mom[pol[2], mmi, nni]
                                               + mom[pol[2], blist, mmi]
                                               * mom[pol[1], mmi, nni]))
                                    pnml = 1j * np.imag(pnml)
                                    Fval = (fnm / (Emn**3)
                                            * np.sum(pnml
                                                     / (2 * Emn - Eln)))
                                    sum2b_D += Fval * \
                                        w_k[k_c] / 2 / (w_lc - Emn)
                                    Eml = E_n[mmi] - E_n[blist] - fml * eshift
                                    pnml = (mom[pol[0], blist, mmi]
                                            * (mom[pol[1], mmi, nni]
                                               * mom[pol[2], nni, blist]
                                               + mom[pol[2], mmi, nni]
                                               * mom[pol[1], nni, blist]))
                                    pnml = 1j * np.imag(pnml)
                                    Fval = (-fnm / (Emn**3)
                                            * np.sum(pnml
                                                     / (2 * Emn - Eml)))
                                    sum2b_D += Fval * \
                                        w_k[k_c] / 2 / (w_lc - Emn)

                        elif summethod == 'loop':
                            # Loop over the last band index
                            for lli in blist:
                                fnl = f_n[nni] - f_n[lli]
                                fml = f_n[mmi] - f_n[lli]

                                # Do not do zero calculations
                                if np.abs(fnl) < ftol and np.abs(fml) < ftol:
                                    continue

                                # Compute the susceptibility with 1/w form
                                Eln = E_n[lli] - E_n[nni] + fnl * eshift
                                Eml = E_n[mmi] - E_n[lli] - fml * eshift
                                pnml = (mom[pol[0], nni, mmi]
                                        * (mom[pol[1], mmi, lli]
                                           * mom[pol[2], lli, nni]
                                           + mom[pol[2], mmi, lli]
                                           * mom[pol[1], lli, nni]))
                                pnml = 1j * np.imag(pnml) * w_k[k_c] / 2

                                # Compute the divergence-free terms
                                if np.abs(Emn) > Etol and np.abs(
                                        Eml) > Etol and np.abs(Eln) > Etol:
                                    ftermD = (16 / (Emn**3 * (2 * w_lc - Emn))
                                              * (fnl / (Emn - 2 * Eln)
                                                 + fml / (Emn - 2 * Eml))) \
                                        + fnl / (Eln**3 * (2 * Eln - Emn)
                                                 * (w_lc - Eln)) \
                                        + fml / (Eml**3 * (2 * Eml - Emn)
                                                 * (w_lc - Eml))
                                    sum2b_D += pnml * ftermD
                        else:
                            parprint(
                                'Method ' + summethod + ' not implemented.')
                            raise NotImplementedError

            # Print the progress
            count += 1
            print_progressbar(count, ncount)
    elif intmethod == 'tri':
        # Useful variable
        itype = 1
        # tri1=[0, 1, 2]
        # tri2=[1, 2, 3]
        tri1=[0, 1, 3]
        tri2=[0, 2, 3]
        w_l = w_lc.real

        # Initialize variables
        # moms = moms_new
        sum2b = np.zeros((nw), dtype=float)
        assert not addsoc, 'Triangular method is only implemented without SOC.'

        # Distribute the k points between cores
        # if world.rank == 0:
        #     for vv in range(3):
        #         for ik in range(nk):
        #             moms[ik, vv] = (np.conj(moms[ik, vv].T)+moms[ik, vv])/2
        with timer('k-info distribution'):
            k_info, dA = get_neighbors(moms, E_nk, f_nk, kd, nb, qind=[
                                       [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over the k points
        for k_c, data_k in k_info.items():
            # print(data_k)
            mom0, mom1, mom2, mom3, E_n0, E_n1, E_n2, E_n3, \
                f_n0, f_n1, f_n2, f_n3 = tuple(data_k)
            mom = np.array([mom0[:, ni:nf, ni:nf], mom1[:, ni:nf, ni:nf],
                            mom2[:, ni:nf, ni:nf], mom3[:, ni:nf, ni:nf]])
            E_n = np.array([E_n0[ni:nf].real, E_n1[ni:nf].real,
                            E_n2[ni:nf].real, E_n3[ni:nf].real])
            f_n = np.array([f_n0[ni:nf].real, f_n1[ni:nf].real,
                            f_n2[ni:nf].real, f_n3[ni:nf].real])

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                r_nm = np.zeros((4, 3, nb2, nb2), dtype=complex)
                D_nm = np.zeros((4, 3, nb2, nb2), dtype=complex)
                E_nm = np.zeros((4, nb2, nb2), dtype=float)
                for ii in range(4):
                    tmp = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp) < Etol
                    tmp[zeroind] = 1
                    E_nm[ii] = tmp
                    for aa in set(pol):
                        r_nm[ii, aa] = mom[ii, aa] / (1j * E_nm[ii])
                        r_nm[ii, aa, zeroind] = 0
                        p_nn = np.diag(mom[ii, aa])
                        D_nm[ii, aa] = (np.tile(p_nn[:, None], (1, nb2))
                                        - np.tile(p_nn[None, :], (nb2, 1)))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((4, 3, 3, nb2, nb2), dtype=complex)
                for ii in range(4):
                    tmp1 = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp1) < Etol
                    for aa in set(pol):
                        for bb in set(pol):
                            tmp = (r_nm[ii, aa] * np.transpose(D_nm[ii, bb])
                                   + r_nm[ii, bb] * np.transpose(D_nm[ii, aa])
                                   + 1j * np.dot(r_nm[ii, aa],
                                                 r_nm[ii, bb] * E_nm[ii])
                                   - 1j * np.dot(r_nm[ii, bb] * E_nm[ii],
                                                 r_nm[ii, aa])) / E_nm[ii]
                            tmp[zeroind] = 0
                            rnm_der[ii, aa, bb] = tmp
                    E_nm[ii, zeroind] = 0

            # Loop over bands
            with timer('Sum over bands'):
                # Shift the bands
                # E_nk += eshift/2.0*np.sign(E_n)
                # Loop over bands
                for nni in blist:
                    for mmi in blist:
                        # Remove the non important term using time-reversal
                        # symmtery
                        if mmi <= nni:
                            continue

                        # Useful variables
                        Emn = E_n[:, mmi] - E_n[:, nni]
                        fnm = f_n[:, nni] - f_n[:, mmi]

                        # Comute the 2-band term
                        if np.any(np.abs(fnm) > ftol) and np.all(np.abs(Emn) > Etol):
                            pnml = (mom[:, pol[0], nni, mmi]
                                    * (mom[:, pol[1], mmi, nni]
                                       * (mom[:, pol[2], mmi, mmi]
                                          - mom[:, pol[2], nni, nni])
                                       + mom[:, pol[2], mmi, nni]
                                       * (mom[:, pol[1], mmi, mmi]
                                          - mom[:, pol[1], nni, nni]))
                                    / 2)
                            Fval += fnm * np.imag(pnml) / (Emn**4)
                            # Use the triangle method for integration
                            tmp1 = (triangle_int(Fval, Emn,
                                                w_l, itype=itype, 
                                                tri1=tri1, tri2=tri2)) * dA / 2
                            tmp2 = -16 * (triangle_int(Fval, Emn,
                                                2 * w_l, itype=itype, 
                                                tri1=tri1, tri2=tri2)) * dA / 2
                            sum2b_d += 1j * (tmp1 + tmp2)

                        # Loop over the last band index
                        for lli in blist:
                            fnl = f_n[:, nni] - f_n[:, lli]
                            fml = f_n[:, mmi] - f_n[:, lli]

                            # Do not do zero calculations
                            if np.all(np.abs(fnl) < ftol) and np.all(np.abs(fml) < ftol):
                                continue

                            # Compute the susceptibility with 1/w form
                            Eln = E_n[:, lli] - E_n[:, nni]
                            Eml = E_n[:, mmi] - E_n[:, lli]
                            pnml = (mom[:, pol[0], nni, mmi]
                                    * (mom[:, pol[1], mmi, lli]
                                        * mom[:, pol[2], lli, nni]
                                        + mom[:, pol[2], mmi, lli]
                                        * mom[:, pol[1], lli, nni]))
                            pnml = np.imag(pnml) / 2

                            # Compute the divergence-free terms
                            ftermD = 0
                            if np.all(np.abs(Emn) > Etol):
                                Fval = pnml / Emn**3 * (fnl / (Emn - 2 * Eln) + fml / (Emn - 2 * Eml))
                                ftermD += 16 * (triangle_int(Fval, Emn,
                                                2 * w_l, itype=itype, 
                                                tri1=tri1, tri2=tri2))
                            if np.all(np.abs(Eln) > Etol):
                                Fval = fnl * pnml / (Eln**3 * (2 * Eln - Emn))
                                ftermD += (triangle_int(Fval, Eln,
                                           w_l, itype=itype, 
                                           tri1=tri1, tri2=tri2))  
                            if np.all(np.abs(Eml) > Etol):
                                Fval = fml * pnml / (Eml**3 * (2 * Eml - Emn))
                                ftermD += (triangle_int(Fval, Eml,
                                           w_l, itype=itype, 
                                           tri1=tri1, tri2=tri2))
                                
                                sum2b_D += 1j * ftermD * dA / 2
            # Print the progress
            count += 1
            print_progressbar(count, ncount)
    else:
        parprint('Integration method ' + intmethod + ' not implemented.')
        raise NotImplementedError

    # Sum over all nodes
    world.sum(sum2)
    world.sum(sum2b_D)
    world.sum(sum2b_d)

    # Make the output in SI unit
    dim_ee = gspin * _e**3 * _hbar**2 / (_me**3 * (2.0 * pi)**3)
    dim_chi = 1j * _hbar / (_eps0 * 2.0 * _e)  # 2 beacuse of frequecny
    dim_sum = (_hbar / (Bohr * 1e-10))**3 / \
        (_e**4 * (Bohr * 1e-10)**3) * bz_vol
    dim_SI = dim_chi * dim_ee * dim_sum
    if world.rank == 0:
        if intmethod == 'no':
            chi2 = dim_SI * sum2 / (w_lc**3)
            chi2b = dim_SI * (sum2b_D + sum2b_d)  # 1j is for im(PnmPmlPln)
        elif intmethod == 'tri':
            # Make the real part of the chi and broaden the spectra
            with timer('Hilbert transform'):
                sum2b_d2 = np.zeros(len(w_l), complex)
                sum2b_D2 = np.zeros(len(w_l), complex)
                for ind, omega in enumerate(w_lc):
                    sum2b_d2[ind] = np.trapz(
                        sum2b_d * w_l / (omega ** 2 - w_l ** 2), w_l)
                    sum2b_D2[ind] = np.trapz(
                        sum2b_D * w_l / (omega ** 2 - w_l ** 2), w_l)
                chi2b = dim_SI * (sum2b_D2 + sum2b_d2)
                chi2 = chi2b

        nw = int(nw / 2)
        chi2 = chi2[nw:] + chi2[nw - 1::-1]
        chi2b = chi2b[int(nw):] + chi2b[nw - 1::-1]
        chi2 = chi2b
        shg = np.vstack((freqs, chi2, chi2b))

        # Save the data
        # Save it to the file
        if outname is None:
            np.save('shg.npy', shg)
        else:
            np.save('{}.npy'.format(outname), shg)

        # Print the timing
        timer.write()

    else:
        shg = None

    # Return shg
    return shg

# Calculate the SHG response (for nonmagnetic semiconductors) in the
# length gauge, reqularized version


def calculate_shg_rlg(
        freqs=1.0,
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        addsoc=False,
        socscale=1.0,
        intmethod='no',
        dermethod='sum',
        Etol=1e-3, ftol=1e-4,
        ni=None, nf=None,
        blist=None,
        outname=None,
        momname=None,
        basename=None):
    """
    Calculate SHG spectrum in length gauge (nonmagnetic semiconductors)

    Input:
        freqs           Excitation frequency array (a numpy array or list)
        eta             Broadening (a single number or an array)
        pol             Tensor element
        eshift          scissors shift (default 0)
        addsoc          Add spin-orbit coupling (default off)
        socscale        SOC scale (default 1.0)
        intmethod       Integral method (defaul no)
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        ni, nf          First and last bands in the calculations (0 to nb)
        blist           List of bands in the sum
        outname         Output filename (default is shg.npy)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        shg.npy          Numpy array containing spectrum and frequencies
    """

    timer = Timer()
    parprint(
        'Calculating SHG spectrum in length gauge (in {:d} cores).'.format(
            world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = freqs + 1j * eta
    w_lc = np.hstack((-w_lc[-1::-1], w_lc))
    nw = 2 * nw
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    polstr = pol
    pol = ['xyz'.index(ii) for ii in polstr]
    gspin = 2.0
    nb2 = nf - ni
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            E_nk = calc.band_structure().todict()['energies'][0]
            f_nk = np.zeros((nk, nb), dtype=np.float64)
            for k_c in range(nk):
                f_nk[k_c] = calc.get_occupation_numbers(
                    kpt=k_c, spin=0) / w_k[k_c] / 2.0
        else:
            E_nk = None
            f_nk = None
    
    # Find the neighboring points
    # nei_ind, psigns, q_vecs = find_neighbors(kd, qind=[[1, 0, 0], [0, 1, 0]])

    # if world.rank == 0:
    #     # u_knn = np.load('rot_{}.npy'.format(basename))
    #     u_knn_0 = np.load('rot_{}_0.npy'.format(basename))
    #     u_knn_1 = np.load('rot_{}_1.npy'.format(basename))
    # else:
    #     u_knn = None
    #     u_knn_0 = None
    #     u_knn_1 = None       
    # # u_knn = broadcast(u_knn)
    # u_knn_0 = broadcast(u_knn_0)
    # u_knn_1 = broadcast(u_knn_1)

    # Get eigenvalues and wavefunctions with spin orbit coupling (along the
    # z-direction)
    if addsoc:
        gspin = 1.0
        nb2 = 2 * (nf - ni)
        with timer('Spin-orbit coupling calculation'):
            if world.rank == 0:
                # Get the SOC
                (e_mk, wfs_knm) = get_spinorbit_eigenvalues(
                    calc, return_wfs=True, bands=np.arange(ni, nf),
                    scale=socscale)

                # Make the data ready
                e_mk = e_mk.T
                wfs_knm = np.array(wfs_knm)

                # Update the Fermi level
                mu = fermi_level(calc,
                                 e_mk[np.newaxis],
                                 2 * calc.get_number_of_electrons())
            else:
                e_mk = None
                wfs_knm = None
                mu = None

            # Update the Fermi level
            mu = broadcast(mu, 0)

            # Distribute the SOC
            k_info2 = distribute_data(
                [e_mk, wfs_knm], [(nk, nb2), (nk, nb2, nb2)])

            # Get SOC from PAW
            dVL_avii, Pt_kasni = get_soc_paw(calc)

    # Initialize the outputs
    sum2b_A = np.zeros((nw), dtype=np.complex)
    sum2b_B = np.zeros((nw), dtype=np.complex)
    sum2b_C = np.zeros((nw), dtype=np.complex)
    if blist is None:
        blist = list(range(nb2))
    else:
        assert max(
            blist) < nb2, 'Maximum of blist should be smaller than nb.'
    blist = np.array(blist, int)

    # Depending on the integration method
    parprint(
        'Evaluating SHG response for {}-polarization ({:.1f} meV).'.format(
            polstr,
            1000 *
            eta))
    if intmethod == 'no':

        u_knn = None
        # if world.rank == 0:
        #     u_knn = np.load('rot_{}.npy'.format(basename))
        # else:
        #     u_knn = None
        # u_knn = broadcast(u_knn)
        # if world.rank == 0:
        #     for ik in range(nk):
        #         u_nn = np.dot(u_knn[ik].T, np.arange(0, nb)).astype(int)
        #         E_nk[ik] = E_nk[ik][u_nn]
        #         f_nk[ik] = f_nk[ik][u_nn]
        #         for v in range(3):
        #             moms[ik, v] = moms[ik, v][np.ix_(u_nn, u_nn)]

        # with timer('Change the order of bands'):
        #     if world.rank == 0:
        #         # u_knn = np.load('rot_{}.npy'.format(basename))
        #         moms_new = np.zeros((nk, 3, nb, nb), complex)
        #         for ik in range(nk):
        #             u_nn = np.dot(u_knn[ik].T, np.arange(0, nb)).astype(int)
        #             E_nk[ik] = E_nk[ik][u_nn]
        #             f_nk[ik] = f_nk[ik][u_nn]
        #             for v in range(3):
        #                 moms[ik, v] = moms[ik, v][np.ix_(u_nn, u_nn)]
                    # E_nk[ik] = np.dot(E_nk[ik].T, u_knn[ik])
                    # f_nk[ik] = np.dot(f_nk[ik].T, u_knn[ik])
                    # for v in range(3):
                    #     moms[ik, v] = np.dot(u_knn[ik].T, np.dot(moms[ik, v], u_knn[ik]))
        # from mpl_toolkits.mplot3d import Axes3D
        # from matplotlib import cm
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(np.arange(0, kd.N_c[0]), np.arange(0, kd.N_c[1]), E_nk[kd.bz2ibz_k, 14].reshape(kd.N_c[0], kd.N_c[1]),
        #                 cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # # plt.tight_layout()
        # fig.savefig('mat.png', dpi=300)
        # plt.close()
        # if world.rank == 0:
        #     N_c = kd.N_c
        #     kp0 = np.arange(0, nk, N_c[1])
        #     kp0[-1] += -int(N_c[1]/2-1)
        #     kp0 -= 2
        #     kp0 = np.delete(kp0, [0, -1])
        #     # kp0 = np.arange(21, 59)+2*38
        #     Enew_kn = np.array([E_nk[ik][np.dot(u_knn_0[ik].T, np.arange(0, nb)).astype(int)] for ik in kp0])
        #     # Enew_kn = np.array([np.dot(u_knn[ik].T, E_nk[ik]) for ik in kp0])
        #     plt.plot(np.arange(0, len(kp0)), Enew_kn[:, 8:18])
        #     plt.tight_layout()
        #     plt.savefig('mat.png', dpi=300)
        #     plt.close()
        #     plt.plot(np.arange(0, len(kp0)), E_nk[kp0, 8:18], '--')
        #     # plt.ylim([0, 5])
        #     plt.tight_layout()
        #     plt.savefig('mat2.png', dpi=300)
        #     plt.close()

        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info = distribute_data(
                [moms, E_nk, f_nk], [(nk, 3, nb, nb), (nk, nb), (nk, nb)])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)
        sum_d = np.zeros((3, 3), complex)

        # Loop over k points
        for k_c, data_k in k_info.items():
            mom, E_n, f_n = tuple(data_k)
            mom = mom[:, ni:nf, ni:nf]
            f_n0 = f_n[ni:nf].copy()

            # Add SOC or not
            if addsoc:
                E_n, wfs_nm = tuple(k_info2.get(k_c))
                E_n = E_n.real
                tmp = (E_n - mu) / kbT
                tmp = np.clip(tmp, -100, 100)
                f_n = 1 / (np.exp(tmp) + 1.0)
                f_n = np.zeros(nb2, complex)
                f_n[::2] = f_n0
                f_n[1::2] = f_n0

                # Make the new momentum
                with timer('New momentum calculation'):
                    mom2 = np.zeros((3, nb2, nb2), dtype=complex)
                    for pp in range(3):
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 0::2]))
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 1::2]))

                        # Get the soc correction to momentum
                        p_vmm = socscale * \
                                get_soc_momentum(dVL_avii, Pt_kasni[k_c],
                                                 ni, nf)
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm), p_vmm[pp]),
                                           np.transpose(wfs_nm))
                    mom = mom2
            else:
                E_n = E_n[ni:nf].real
                f_n = f_n[ni:nf].real

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                r_nm = np.zeros((3, nb2, nb2), complex)
                D_nm = np.zeros((3, nb2, nb2), complex)
                E_nm = np.tile(E_n[:, None], (1, nb2)) - \
                    np.tile(E_n[None, :], (nb2, 1))
                zeroind = np.abs(E_nm) < Etol
                E_nm[zeroind] = 1
                # np.fill_diagonal(E_nm, 1.0)
                for aa in set(pol):
                    r_nm[aa] = mom[aa] / (1j * E_nm)
                    r_nm[aa, zeroind] = 0
                    # np.fill_diagonal(r_nm[aa], 0.0)
                    p_nn = np.diag(mom[aa])
                    D_nm[aa] = np.tile(p_nn[:, None], (1, nb2)) - \
                        np.tile(p_nn[None, :], (nb2, 1))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((3, 3, nb2, nb2), complex)
                if dermethod == 'sum':
                    for aa in set(pol):
                        for bb in set(pol):
                            tmp = (r_nm[aa] * np.transpose(D_nm[bb])
                                + r_nm[bb] * np.transpose(D_nm[aa])
                                + 1j * np.dot(r_nm[aa], r_nm[bb] * E_nm)
                                - 1j * np.dot(r_nm[bb] * E_nm, r_nm[aa])) / E_nm
                            tmp[zeroind] = 0
                            rnm_der[aa, bb] = tmp
                            # np.fill_diagonal(rnm_der[aa, bb], 0.0)
                elif dermethod == 'log':
                    rd_vvnn = get_derivative(calc, nei_ind[:, k_c], q_vecs, ni+blist, ovth=0.5, u_knn=u_knn, timer=timer, psigns=psigns[:, k_c])
                    scale = (_me*(Bohr*1e-10)**2*_e/_hbar**2)
                    for v1 in range(3):
                        for v2 in range(3):
                            rnm_der[v1, v2] = r_nm[v2]*(rd_vvnn[v1, v2]*(-1)*scale-1*D_nm[v1]/E_nm)
                else:
                    parprint('Derivative mode ' + dermethod + ' not implemented.')
                    raise NotImplementedError

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # for v1 in range(3):
                        #     for v2 in range(3):
                        #         sum_d[v1, v2] += (f_n[nni] - f_n[mmi])*(rd_vvnn[v1, v2, nni, mmi]*(-1)*scale*mom[v2, nni, mmi]).real*w_k[k_c]
                        # Remove the non important term using time-reversal
                        if mmi <= nni:
                            continue
                        fnm = f_n[nni] - f_n[mmi]
                        Emn = E_nm[mmi, nni] + fnm * eshift

                        # Two band part
                        if np.abs(fnm) > ftol:
                            tmp = 2 * np.imag(
                                r_nm[pol[0], nni, mmi]
                                * (rnm_der[pol[1], pol[2], mmi, nni]
                                   + rnm_der[pol[2], pol[1], mmi, nni])) \
                                / (Emn * (2 * w_lc - Emn))
                            tmp += np.imag(
                                r_nm[pol[1], mmi, nni]
                                * rnm_der[pol[2], pol[0], nni, mmi]
                                + r_nm[pol[2], mmi, nni]
                                * rnm_der[pol[1], pol[0], nni, mmi]) \
                                / (Emn * (w_lc - Emn))
                            tmp += np.imag(
                                r_nm[pol[0], nni, mmi]
                                * (r_nm[pol[1], mmi, nni]
                                   * D_nm[pol[2], mmi, nni]
                                   + r_nm[pol[2], mmi, nni]
                                   * D_nm[pol[1], mmi, nni])) \
                                * (1 / (w_lc - Emn)
                                   - 4 / (2 * w_lc - Emn)) / Emn**2
                            tmp -= np.imag(
                                r_nm[pol[1], mmi, nni]
                                * rnm_der[pol[0], pol[2], nni, mmi]
                                + r_nm[pol[2], mmi, nni]
                                * rnm_der[pol[0], pol[1], nni, mmi]) \
                                / (2 * Emn * (w_lc - Emn))
                            sum2b_B += 1j * fnm * tmp * w_k[k_c] / 2  # 1j imag

                            # Three band term
                            # for lli in blist:
                            #     Eml = E_nm[mmi, lli]
                            #     Eln = E_nm[lli, nni]
                            #     rnml = np.real(
                            #         r_nm[pol[0], nni, mmi]
                            #         * (r_nm[pol[1], mmi, lli]
                            #            * r_nm[pol[2], lli, nni]
                            #            + r_nm[pol[2], mmi, lli]
                            #            * r_nm[pol[1], lli, nni])) \
                            #         / (Eln - Eml)
                            #     sum2b_A += 2 * fnm / \
                            #         (2 * w_lc - Emn) * rnml * w_k[k_c] / 2
                            #     rnml = np.real(
                            #         r_nm[pol[0], lli, mmi]
                            #         * (r_nm[pol[1], mmi, nni]
                            #            * r_nm[pol[2], nni, lli]
                            #            + r_nm[pol[2], mmi, nni]
                            #            * r_nm[pol[1], nni, lli])) \
                            #         / (Emn + Eln) \
                            #         - np.real(
                            #         r_nm[pol[0], nni, lli]
                            #         * (r_nm[pol[1], lli, mmi]
                            #            * r_nm[pol[2], mmi, nni]
                            #            + r_nm[pol[2], lli, mmi]
                            #            * r_nm[pol[1], mmi, nni])) \
                            #         / (Emn + Eml)
                            #     sum2b_A += fnm / (w_lc - Emn) * \
                            #         rnml * w_k[k_c] / 2
                        for lli in blist:
                            fnl = f_n[nni] - f_n[lli]
                            fml = f_n[mmi] - f_n[lli]
                            Eml = E_nm[mmi, lli] - fml * eshift
                            Eln = E_nm[lli, nni] + fnl * eshift
                            # Do not do zero calculations
                            if (np.abs(fnm) < ftol and np.abs(fnl) < ftol
                                    and np.abs(fml) < ftol):
                                continue
                            if np.abs(Eln - Eml) < Etol:
                                continue

                            rnml = np.real(
                                r_nm[pol[0], nni, mmi]
                                * (r_nm[pol[1], mmi, lli]
                                   * r_nm[pol[2], lli, nni]
                                   + r_nm[pol[2], mmi, lli]
                                   * r_nm[pol[1], lli, nni])) \
                                * w_k[k_c] / (2 * (Eln - Eml))
                            if np.abs(fnm) > ftol:
                                sum2b_A += 2 * fnm / (2 * w_lc - Emn) * rnml
                            if np.abs(fnl) > ftol:
                                sum2b_A += -fnl / (w_lc - Eln) * rnml
                            if np.abs(fml) > ftol:
                                sum2b_A += fml / (w_lc - Eml) * rnml

                # Print the progress
                count += 1
                print_progressbar(count, ncount)

    elif intmethod == 'tri':
        # Useful variable
        itype = 1
        tri1=[0, 1, 3]
        tri2=[0, 2, 3]
        w_l = w_lc.real
        nb2 = nf - ni

        # with timer('Change the order of bands'):
        #     if world.rank == 0:
        #         u_knn = np.load('rot_{}.npy'.format(basename))
        #         moms_new = np.zeros((nk, 3, nb, nb), complex)
        #         for ik in range(nk):
        #             E_nk[ik] = np.dot(E_nk[ik].T, u_knn[ik])
        #             f_nk[ik] = np.dot(f_nk[ik].T, u_knn[ik])
        #             for v in range(3):
        #                 moms[ik, v] = np.dot(u_knn[ik].T, np.dot(moms[ik, v], u_knn[ik]))

        # Initialize variables
        assert not addsoc, 'Triangular method is only implemented without SOC.'

        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info, dA = get_neighbors(moms, E_nk, f_nk, kd, nb, qind=[
                                       [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over the k points
        for k_c, data_k in k_info.items():
            # Get 4-points data (corners of a rectangle)
            mom0, mom1, mom2, mom3, E_n0, E_n1, E_n2, E_n3, \
                f_n0, f_n1, f_n2, f_n3 = tuple(data_k)
            mom = np.array([mom0[:, ni:nf, ni:nf], mom1[:, ni:nf, ni:nf],
                            mom2[:, ni:nf, ni:nf], mom3[:, ni:nf, ni:nf]])
            E_n = np.array([E_n0[ni:nf], E_n1[ni:nf],
                            E_n2[ni:nf], E_n3[ni:nf]])
            f_n = np.array([f_n0[ni:nf], f_n1[ni:nf],
                            f_n2[ni:nf], f_n3[ni:nf]])

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                r_nm = np.zeros((4, 3, nb2, nb2), dtype=np.complex)
                D_nm = np.zeros((4, 3, nb2, nb2), dtype=np.complex)
                E_nm = np.zeros((4, nb2, nb2), dtype=np.complex)
                for ii in range(4):
                    tmp = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp) < Etol
                    tmp[zeroind] = 1
                    E_nm[ii] = tmp
                    for aa in set(pol):
                        r_nm[ii, aa] = mom[ii, aa] / (1j * E_nm[ii])
                        r_nm[ii, aa, zeroind] = 0
                        p_nn = np.diag(mom[ii, aa])
                        D_nm[ii, aa] = (np.tile(p_nn[:, None], (1, nb2))
                                        - np.tile(p_nn[None, :], (nb2, 1)))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((4, 3, 3, nb2, nb2), dtype=np.complex)
                for ii in range(4):
                    tmp = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp) < Etol
                    for aa in set(pol):
                        for bb in set(pol):
                            tmp = (r_nm[ii, aa] * np.transpose(D_nm[ii, bb])
                                   + r_nm[ii, bb] * np.transpose(D_nm[ii, aa])
                                   + 1j * np.dot(r_nm[ii, aa],
                                                 r_nm[ii, bb] * E_nm[ii])
                                   - 1j * np.dot(r_nm[ii, bb] * E_nm[ii],
                                                 r_nm[ii, aa])) / E_nm[ii]
                            tmp[zeroind] = 0
                            rnm_der[ii, aa, bb] = tmp

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # Remove non important term using time-reversal
                        # symmtery
                        if mmi <= nni:
                            continue
                        fnm = f_n[:, nni] - f_n[:, mmi]
                        Emn = E_nm[:, mmi, nni] + fnm * eshift

                        # Two band part
                        if np.any(np.abs(fnm)) > ftol:
                            # Term 1
                            Fval = 2 * fnm * np.imag(
                                r_nm[:, pol[0], nni, mmi]
                                * (rnm_der[:, pol[1], pol[2], mmi, nni]
                                   + rnm_der[:, pol[2], pol[1], mmi, nni])) \
                                / Emn
                            val = triangle_int(Fval, Emn, 2 * w_l, itype=itype, tri1=tri1, tri2=tri2)
                            # Term 2
                            Fval = fnm * np.imag(
                                r_nm[:, pol[1], mmi, nni]
                                * rnm_der[:, pol[2], pol[0], nni, mmi]
                                + r_nm[:, pol[2], mmi, nni]
                                * rnm_der[:, pol[1], pol[0], nni, mmi]) \
                                / Emn
                            val += triangle_int(Fval, Emn, w_l, itype=itype, tri1=tri1, tri2=tri2)
                            # Term 3
                            Fval = fnm * np.imag(
                                r_nm[:, pol[0], nni, mmi]
                                * (r_nm[:, pol[1], mmi, nni]
                                   * D_nm[:, pol[2], mmi, nni]
                                   + r_nm[:, pol[2], mmi, nni]
                                   * D_nm[:, pol[1], mmi, nni])) \
                                / Emn**2
                            val += triangle_int(Fval, Emn, w_l, itype=itype, tri1=tri1, tri2=tri2)
                            val += -4 * \
                                triangle_int(Fval, Emn, 2 * w_l, itype=itype, tri1=tri1, tri2=tri2)
                            # Term 4
                            Fval = fnm * np.imag(
                                r_nm[:, pol[1], mmi, nni]
                                * rnm_der[:, pol[0], pol[2], nni, mmi]
                                + r_nm[:, pol[2], mmi, nni]
                                * rnm_der[:, pol[0], pol[1], nni, mmi]) \
                                / (2 * Emn)
                            val += -1 * \
                                triangle_int(Fval, Emn, w_l, itype=itype, tri1=tri1, tri2=tri2)
                            sum2b_B += 1j * val * dA / 4.0  # 1j for imag

                            # Three band term
                            # for lli in blist:
                            #     Eml = E_nm[:, mmi, lli]
                            #     Eln = E_nm[:, lli, nni]
                            #     rnml = np.real(
                            #         r_nm[:, pol[0], nni, mmi]
                            #         * (r_nm[:, pol[1], mmi, lli]
                            #            * r_nm[:, pol[2], lli, nni]
                            #            + r_nm[:, pol[2], mmi, lli]
                            #            * r_nm[:, pol[1], lli, nni])) \
                            #         / (Eln - Eml)
                            #     Fval = 2 * fnm * rnml * dA / 4.0
                            #     sum2b_A += triangle_int(Fval, Emn,
                            #                             2 * w_l, itype=itype, tri1=tri1, tri2=tri2)

                            #     rnml = np.real(
                            #         r_nm[:, pol[0], lli, mmi]
                            #         * (r_nm[:, pol[1], mmi, nni]
                            #            * r_nm[:, pol[2], nni, lli]
                            #            + r_nm[:, pol[2], mmi, nni]
                            #            * r_nm[:, pol[1], nni, lli])) \
                            #         / (Emn + Eln) \
                            #         - np.real(
                            #         r_nm[:, pol[0], nni, lli]
                            #         * (r_nm[:, pol[1], lli, mmi]
                            #            * r_nm[:, pol[2], mmi, nni]
                            #            + r_nm[:, pol[2], lli, mmi]
                            #            * r_nm[:, pol[1], mmi, nni])) \
                            #         / (Emn + Eml)
                            #     Fval = fnm * rnml * dA / 4.0
                            #     sum2b_A += triangle_int(Fval, Emn,
                            #                             w_l, itype=itype, tri1=tri1, tri2=tri2)
                        for lli in blist:
                            fnl = f_n[:, nni] - f_n[:, lli]
                            fml = f_n[:, mmi] - f_n[:, lli]
                            Eml = E_nm[:, mmi, lli] - fml * eshift
                            Eln = E_nm[:, lli, nni] + fnl * eshift
                            
                            # Do not do zero calculations
                            if (np.all(np.abs(fnm)) < ftol
                                    and np.all(np.abs(fnl)) < ftol
                                    and np.all(np.abs(fml)) < ftol):
                                continue
                            if np.any(np.abs(Eln - Eml) < Etol):
                                continue

                            rnml = np.real(
                                r_nm[:, pol[0], nni, mmi]
                                * (r_nm[:, pol[1], mmi, lli]
                                   * r_nm[:, pol[2], lli, nni]
                                   + r_nm[:, pol[2], mmi, lli]
                                   * r_nm[:, pol[1], lli, nni])) / (Eln - Eml)
                            if np.any(np.abs(fnm)) > ftol:
                                Fval = 2 * fnm * rnml * dA / 4.0
                                sum2b_A += triangle_int(Fval, Emn,
                                                        2 * w_l, itype=itype, tri1=tri1, tri2=tri2)
                            if np.any(np.abs(fnl)) > ftol:
                                Fval = -1 * fnl * rnml * dA / 4.0
                                sum2b_A += triangle_int(Fval, Eln,
                                                        w_l, itype=itype, tri1=tri1, tri2=tri2)
                            if np.any(np.abs(fml)) > ftol:
                                Fval = fml * rnml * dA / 4.0
                                sum2b_A += triangle_int(Fval, Eml,
                                                        w_l, itype=itype, tri1=tri1, tri2=tri2)

            # Print the progress
            count += 1
            print_progressbar(count, ncount)
    else:
        parprint('Integration mode ' + intmethod + ' not implemented.')
        raise NotImplementedError

    # Sum over all nodes
    # par.comm.Barrier()
    with timer('Gather data from cores'):
        world.sum(sum2b_A)
        world.sum(sum2b_B)
        world.sum(sum2b_C)
        # world.sum(sum_d)

    # Make the output in SI unit
    dim_ee = gspin * _e**3 / (_eps0 * (2.0 * pi)**3)
    dim_sum = (_hbar / (Bohr * 1e-10))**3 / \
        (_e**5 * (Bohr * 1e-10)**3) * (_hbar / _me)**3 * bz_vol
    dim_ee_SI = dim_ee * dim_sum
    dim_ie_SI = 1j * dim_ee * dim_sum
    chi2b = np.zeros((5, nw), complex)

    # nv = calc.get_number_of_electrons()
    # sum_rule = gspin * (_hbar / (Bohr * 1e-10))**2 / (_e * _me) / nv * sum_d
    # sum_rule = np.reshape(sum_rule, (3, 3), order='C')
    # parprint('The sum rule matrix is:')
    # parprint(np.array_str(sum_rule, precision=6, suppress_small=True))

    # Save the data
    if world.rank == 0:
        if intmethod == 'no':
            chi2b[1, :] = dim_ee_SI * sum2b_A
            chi2b[2, :] = dim_ie_SI * sum2b_B
            # chi2b[3, :] = dim_ie_SI * sum2b_C / w_lc
            # chi2b[4, :] = dim_ee_SI*sum2b_D/w_lc

        elif intmethod == 'tri':
            # Make the real part of the chi and broaden the spectra
            with timer('Hilbert transform'):
                suml_A = np.zeros(len(w_l), complex)
                suml_B = np.zeros(len(w_l), complex)
                for ind, omega in enumerate(w_lc):
                    suml_A[ind] = np.trapz(
                        sum2b_A * w_l / (omega ** 2 - w_l ** 2), w_l)
                    suml_B[ind] = np.trapz(
                        sum2b_B * w_l / (omega ** 2 - w_l ** 2), w_l)
                chi2b[1, :] = dim_ee_SI * suml_A
                chi2b[2, :] = dim_ie_SI * suml_B

        chi2b[0, :] = chi2b[1, :] + chi2b[2, :] + chi2b[3, :] + chi2b[4, :]
        chi2 = chi2b[0, :]
        nw = int(nw / 2)
        chi2 = chi2[nw:] + chi2[nw - 1::-1]
        chi2b = chi2b[:, int(nw):] + chi2b[:, nw - 1::-1]

        # A multi-col output
        shg = np.vstack((freqs, chi2, chi2b))

        # Save it to the file
        if outname is None:
            np.save('shg.npy', shg)
        else:
            np.save('{}.npy'.format(outname), shg)

        # Print the timing
        timer.write()
    else:
        shg = None

    # Return SHG respnse
    return shg

# Get the polarized SHG output


def get_shg_polarized(
        atoms,
        prename='shg_',
        postname='',
        wind=[1],
        theta=0.0,
        phi=0.0,
        pte=[1.0],
        ptm=[0.0],
        E0=[1.0],
        outname=None,
        outbasis='xyz'):

    if world.rank == 0:
        # Check the input arguments
        pte = np.array(pte)
        ptm = np.array(ptm)
        E0 = np.array(E0)
        assert np.all(
            np.abs(pte) ** 2 + np.abs(ptm) ** 2) == 1, \
            '|pte|**2+|ptm|**2 should be one.'
        assert len(pte) == len(ptm), 'Size of pte and ptm should be the same.'

        # Useful variables
        costh = np.cos(theta)
        sinth = np.sin(theta)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        nw = len(wind)
        npsi = len(pte)

        # Transfer matrix between (x y z)/(atm ate k) unit vectors basis
        if theta == 0:
            transmat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            transmat = [[cosphi * costh, sinphi * costh, -sinth],
                        [-sinphi, cosphi, 0],
                        [sinth * cosphi, sinth * sinphi, costh]]
        transmat = np.array(transmat)

        # Get the symmtery
        tensordict = get_tensor_elements(atoms)
        # No calculation is done when the structure has inversion center
        if len(tensordict.keys()) == 1:
            parprint('It has inversion center, no calculation is required.')
            return 0

        # Load the data from the disk and make chi
        chi = {}
        for pol, relation in tensordict.items():
            if pol == 'zero':
                if relation != '':
                    for zpol in relation.split('='):
                        chi[zpol] = 0.0
            else:
                filename = prename + pol + postname + '.npy'
                chidata = np.load(filename)
                for zpol in relation.split('='):
                    if zpol[0] == '-':
                        chi[zpol[1:]] = -chidata[1]
                    else:
                        chi[zpol] = chidata[1]

        # Check the E0
        if len(E0) == 1:
            E0 = E0 * np.ones((nw))

        # in xyz coordinate
        Einc = np.zeros((3, npsi), dtype=complex)
        for ii in range(3):
            Einc[ii] = (pte * transmat[0][ii] + ptm * transmat[1][ii])

        # Loop over components
        chipol = np.zeros((3, npsi, nw), dtype=complex)
        for ind, wi in enumerate(wind):
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        pol = 'xyz'[ii] + 'xyz'[jj] + 'xyz'[kk]
                        if np.any(chi[pol] != 0.0):
                            chipol[ii, :, ind] += chi[pol][wi] * \
                                Einc[jj, :] * Einc[kk, :] * E0[ind]**2

        # Change the output basis if needed, and return
        if outbasis == 'xyz':
            chipol_new = chipol
        elif outbasis == 'pol':
            chipol_new = np.zeros((3, npsi, nw), dtype=complex)
            for ind, wi in enumerate(wind):
                chipol[:, :, ind] = np.dot(transmat.T, chipol[:, :, ind])
                chipol_new[0, :, ind] = chipol[0, :, ind] * \
                    pte + chipol[1, :, ind] * ptm
                chipol_new[1, :, ind] = -chipol[0, :, ind] * \
                    ptm + chipol[1, :, ind] * pte
            
        else:
            parprint('Output basis should be either "xyz" or "pol"')
            raise NotImplementedError

        # Save it to the file
        if outname is None:
            np.save('pol.npy', chipol_new)
        else:
            np.save('{}.npy'.format(outname), chipol_new)
        return chipol_new
    else:
        return None


# Calculate the shift current (for nonmagnetic semiconductors)


def calculate_shift_current(
        freqs=1.0,
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        addsoc=False,
        socscale=1.0,
        intmethod='no',
        dermethod='sum',
        Etol=1e-3, ftol=1e-6,
        ni=None, nf=None,
        blist=None,
        outname=None,
        momname=None,
        basename=None):
    """
    Calculate the shift current (nonmagnetic semiconductors)

    Input:
        freqs           Excitation frequency array (a numpy array or list)
        eta             Broadening (a single number or an array)
        pol             Tensor element
        addsoc          Add spin-orbit coupling (default off)
        socscale        SOC scale (default 1.0)
        intmethod       Integral method (defaul no)
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        ni, nf          First and last bands in the calculations (0 to nb)
        blist           List of bands in the sum
        outname         Output filename (default is shg.npy)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        shg.npy          Numpy array containing spectrum and frequencies
    """

    timer = Timer()
    parprint(
        'Calculating shift current (in {:d} cores).'.format(
            world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    freqs = np.array(freqs)
    nw = len(freqs)
    w_l = freqs
    w_lc = freqs + 1j * eta
    # w_lc = np.hstack((-w_lc[-1::-1], w_lc))
    # nw = 2 * nw
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    polstr = pol
    pol = ['xyz'.index(ii) for ii in polstr]
    gspin = 2.0
    nb2 = nf - ni
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            E_nk = calc.band_structure().todict()['energies'][0]
            f_nk = np.zeros((nk, nb), dtype=np.float64)
            for k_c in range(nk):
                f_nk[k_c] = calc.get_occupation_numbers(
                    kpt=k_c, spin=0) / w_k[k_c] / 2.0
        else:
            E_nk = None
            f_nk = None

    # Get eigenvalues and wavefunctions with spin orbit coupling (along the
    # z-direction)
    if addsoc:
        gspin = 1.0
        nb2 = 2 * (nf - ni)
        with timer('Spin-orbit coupling calculation'):
            if world.rank == 0:
                # Get the SOC
                (e_mk, wfs_knm) = get_spinorbit_eigenvalues(
                    calc, return_wfs=True, bands=np.arange(ni, nf),
                    scale=socscale)

                # Make the data ready
                e_mk = e_mk.T
                wfs_knm = np.array(wfs_knm)

                # Update the Fermi level
                mu = fermi_level(calc,
                                 e_mk[np.newaxis],
                                 2 * calc.get_number_of_electrons())
            else:
                e_mk = None
                wfs_knm = None
                mu = None

            # Update the Fermi level
            mu = broadcast(mu, 0)

            # Distribute the SOC
            k_info2 = distribute_data(
                [e_mk, wfs_knm], [(nk, nb2), (nk, nb2, nb2)])
    if blist is None:
        blist = list(range(nb2))
    else:
        assert max(
            blist) < nb2, 'Maximum of blist should be smaller than nb.'
    blist = np.array(blist)

    # Depending on the integration method
    parprint(
        'Evaluating shift current for {}-polarization ({:.1f} meV).'.format(
            polstr,
            1000 *
            eta))
    if intmethod == 'no':
        # Initialize the outputs
        w_l = w_lc.real
        sum2b = np.zeros((nw), complex)
        
        u_knn = None
        # if world.rank == 0:
        #     u_knn = np.load('rot_{}.npy'.format(basename))
        # u_knn = broadcast(u_knn)
        # if world.rank == 0:
        #     for ik in range(nk):
        #         u_nn = np.dot(u_knn[ik].T, np.arange(0, nb)).astype(int)
        #         E_nk[ik] = E_nk[ik][u_nn]
        #         f_nk[ik] = f_nk[ik][u_nn]
        #         for v in range(3):
        #             moms[ik, v] = moms[ik, v][np.ix_(u_nn, u_nn)]

        #     N_c = kd.N_c
        #     kp0 = np.arange(N_c[1], nk, N_c[1])
        #     kp0[-1] += -int(N_c[1]/2-1)
        #     kp0 += 1
        #     kp_kc = kp0
        #     E_kn = calc.band_structure().todict()['energies'][0]
        #     Enew_kn = np.array([np.dot(u_knn[ik].T, E_kn[ik]) for ik in kp_kc])
            
        #     plt.plot(np.arange(0, len(kp_kc)), Enew_kn[:, 8:18])
        #     plt.tight_layout()
        #     plt.savefig('mat.png', dpi=300)
        #     plt.close()
        #     plt.plot(np.arange(0, len(kp_kc)), E_kn[kp_kc, 8:18], '--')
        #     plt.tight_layout()
        #     plt.savefig('mat2.png', dpi=300)
        #     plt.close()

        # from mpl_toolkits.mplot3d import Axes3D
        # from matplotlib import cm
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(np.arange(0, kd.N_c[0]), np.arange(0, kd.N_c[1]), E_nk[kd.bz2ibz_k, 12].reshape(kd.N_c[0], kd.N_c[1]),
        #                 cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # fig.savefig('mat.png', dpi=300)
        # plt.close()
        
        # Find the neighboring points
        nei_ind, psigns, q_vecs = find_neighbors(kd, qind=[[1, 0, 0], [0, 1, 0]])
        nei_ind0, psigns, q_vecs0 = find_neighbors(kd, qind=[[1, 0, 0], [-1, 0, 0]])
        nei_ind1, psigns, q_vecs1 = find_neighbors(kd, qind=[[0, 1, 0], [0, -1, 0]])
        # if world.rank == 0:
        #     u_knn0 = np.load('rot_{}_0.npy'.format(basename))
        #     u_knn1 = np.load('rot_{}_1.npy'.format(basename))
        # else:
        #     u_knn0 = None
        #     u_knn1 = None            
        # u_knn0 = broadcast(u_knn0)
        # u_knn1 = broadcast(u_knn1)
        # get_derivative_new(calc, blist, ovth=0.9, out_name='der.npy', timer=None)

        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info = distribute_data(
                [moms, E_nk, f_nk], [(nk, 3, nb, nb), (nk, nb), (nk, nb)])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over k points
        for k_c, data_k in k_info.items():
            mom, E_n, f_n = tuple(data_k)
            mom = mom[:, ni:nf, ni:nf]

            # Add SOC or not
            if addsoc:
                E_n, wfs_nm = tuple(k_info2.get(k_c))
                E_n = E_n.real
                tmp = (E_n - mu) / kbT
                tmp = np.clip(tmp, -100, 100)
                f_n = 1 / (np.exp(tmp) + 1.0)

                # Make the new momentum
                with timer('New momentum calculation'):
                    mom2 = np.zeros((3, nb2, nb2), dtype=complex)
                    for pp in range(3):
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 0::2]))
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 1::2]))
                    mom = mom2
            else:
                E_n = E_n[ni:nf].real
                f_n = f_n[ni:nf].real

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                # print(nb2)
                r_nm = np.zeros((3, nb2, nb2), complex)
                D_nm = np.zeros((3, nb2, nb2), complex)
                E_nm = np.tile(E_n[:, None], (1, nb2)) - \
                    np.tile(E_n[None, :], (nb2, 1))
                zeroind = np.abs(E_nm) < Etol
                E_nm[zeroind] = 1
                # np.fill_diagonal(E_nm, 1.0)
                for aa in pol:
                    r_nm[aa] = mom[aa] / (1j * E_nm)
                    r_nm[aa, zeroind] = 0
                    # np.fill_diagonal(r_nm[aa], 0.0)
                    p_nn = np.diag(mom[aa])
                    D_nm[aa] = np.tile(p_nn[:, None], (1, nb2)) - \
                        np.tile(p_nn[None, :], (nb2, 1))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((3, 3, nb2, nb2), complex)
                if dermethod == 'sum':
                    for aa in pol:
                        for bb in pol:
                            tmp = (r_nm[aa] * np.transpose(D_nm[bb])
                                + r_nm[bb] * np.transpose(D_nm[aa])
                                + 1j * np.dot(r_nm[aa], r_nm[bb] * E_nm)
                                - 1j * np.dot(r_nm[bb] * E_nm, r_nm[aa])) / E_nm
                            tmp[zeroind] = 0
                            rnm_der[aa, bb] = tmp
                            # np.fill_diagonal(rnm_der[aa, bb], 0.0)
                elif dermethod == 'log':
                    # rd_vvnn = np.zeros((3, 3, nb2, nb2), complex)
                    # rd_vvnn[0] = get_derivative_full(calc, nei_ind0[:, k_c], q_vecs0, ni+blist, ovth=0.5, timer=timer)
                    # rd_vvnn[1] = get_derivative_full(calc, nei_ind1[:, k_c], q_vecs1, ni+blist, ovth=0.5, timer=timer)
                    # icell_cv = (2 * np.pi) * np.linalg.inv(calc.wfs.gd.cell_cv).T
                    # rd_vvnn = np.einsum('ij,jknm->iknm', np.linalg.inv(icell_cv), rd_vvnn)
                    # parprint(np.linalg.inv(icell_cv))
                    # get_derivative_new(calc, blist, timer=timer)

                    rd_vvnn2 = get_derivative(calc, nei_ind[:, k_c], q_vecs, ni+blist, ovth=0.5, u_knn=u_knn, timer=timer, psigns=psigns[:, k_c])
                    rd_vvnn = np.zeros((3, 3, nb2, nb2), complex)
                    scale = (_me*(Bohr*1e-10)**2*_e/_hbar**2)#/Bohr
                    for v1 in range(3):
                        for v2 in range(3):
                            rd_vvnn[v1, v2][np.ix_(ni+blist, ni+blist)] = rd_vvnn2[v1, v2]
                            rnm_der[v1, v2] = r_nm[v2]*(rd_vvnn[v1, v2]*(-1)*scale-D_nm[v1]/E_nm)
                else:
                    parprint('Derivative mode ' + dermethod + ' not implemented.')
                    raise NotImplementedError

                # plt.matshow(np.abs(rnm_der[0, 0]), vmin=0, vmax=0.1)
                # plt.tight_layout()
                # plt.savefig('mat1.png', dpi=300)
                # plt.close()
                # plt.matshow(np.abs(rnm_der2[0, 0]), vmin=0, vmax=0.1)
                # plt.tight_layout()
                # plt.savefig('mat2.png', dpi=300)
                # plt.close()
                
            # if np.any(np.abs(r_nm)>10):
            #     aa=1
            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # Remove the non important term
                        if mmi <= nni:
                            continue
                        fnm = f_n[nni] - f_n[mmi]
                        Emn = E_n[mmi] - E_n[nni]

                        # Two band part
                        if np.abs(fnm) > ftol:
                            tmp = np.imag(
                                r_nm[pol[1], mmi, nni]
                                * rnm_der[pol[0], pol[2], nni, mmi]
                                + r_nm[pol[2], mmi, nni]
                                * rnm_der[pol[0], pol[1], nni, mmi]) \
                                * (eta / (pi * ((w_l - Emn) ** 2 + eta ** 2)) - eta / (pi * ((w_l + Emn) ** 2 + eta ** 2)))
                            sum2b += fnm * tmp * w_k[k_c] 

                # Print the progress
                count += 1
                print_progressbar(count, ncount)

    elif intmethod == 'tri':
        # Useful variable
        itype = 1
        # tri1=[0, 1, 2]
        # tri2=[1, 2, 3]
        tri1=[0, 1, 3]
        tri2=[0, 2, 3]
        w_l = w_lc.real
        # w_l = np.hstack((w_l, w_l[-1]+w_l))
        # nw = 2*nw
        # with timer('Change the order of bands'):
        #     if world.rank == 0:
        #         u_knn = np.load('rot_{}.npy'.format(basename))
        #         moms_new = np.zeros((nk, 3, nb, nb), complex)
        #         for ik in range(nk):
        #             E_nk[ik] = np.dot(E_nk[ik].T, u_knn[ik])
        #             f_nk[ik] = np.dot(f_nk[ik].T, u_knn[ik])
        #             for v in range(3):
        #                 moms[ik, v] = np.dot(u_knn[ik].T, np.dot(moms[ik, v], u_knn[ik]))

        
        # N_c = kd.N_c
        # kp0 = np.arange(0, nk, N_c[1])
        # kp0[-1] += -int(N_c[1]/2-1)
        # kp_jkc = []
        # for k0 in kp0:
        #     kp_jkc.append(np.arange(k0, k0+int(N_c[1]/2+1)))
        #     if k0 != 0 and k0 != nk-int(N_c[1]/2+1):
        #         kp_jkc.append(np.arange(k0, k0-int(N_c[1]/2), -1))
        # parprint('There are {} k paths in the BZ.'.format(len(kp_jkc)))

        # bi = 5
        # bf = 20
        # sel_kp = kp_jkc[1]
        # sel_kp = kp0
        # plt.plot(np.arange(0, len(sel_kp)), E_nk[sel_kp, :])
        # # plt.ylim([-2, 0])
        # plt.tight_layout()
        # plt.savefig('mat.png', dpi=300)
        # plt.subplot(1, 2, 1)
        # plt.plot(np.arange(0, len(sel_kp)), np.abs(moms[sel_kp, 0, 11, bi:bf])**2+np.abs(moms[sel_kp, 1, 11, bi:bf])**2+np.abs(moms[sel_kp, 2, 11, bi:bf])**2)
        # # plt.yscale('log')
        # # plt.ylim([0, 0.15])
        # plt.subplot(1, 2, 2)
        # plt.plot(np.arange(0, len(sel_kp)), np.abs(moms_new[sel_kp, 0, 11, bi:bf])**2+np.abs(moms_new[sel_kp, 1, 11, bi:bf])**2+np.abs(moms_new[sel_kp, 2, 11, bi:bf])**2)
        # # plt.ylim([0, 0.15])
        # plt.tight_layout()
        # plt.savefig('mat2.png', dpi=300)

        # Initialize variables
        # moms = moms_new
        sum2b = np.zeros((nw), dtype=float)
        assert not addsoc, 'Triangular method is only implemented without SOC.'

        # Distribute the k points between cores
        # if world.rank == 0:
        #     for vv in range(3):
        #         for ik in range(nk):
        #             moms[ik, vv] = (np.conj(moms[ik, vv].T)+moms[ik, vv])/2
        with timer('k-info distribution'):
            k_info, dA = get_neighbors(moms, E_nk, f_nk, kd, nb, qind=[
                                       [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over the k points
        for k_c, data_k in k_info.items():
            # print(data_k)
            mom0, mom1, mom2, mom3, E_n0, E_n1, E_n2, E_n3, \
                f_n0, f_n1, f_n2, f_n3 = tuple(data_k)
            mom = np.array([mom0[:, ni:nf, ni:nf], mom1[:, ni:nf, ni:nf],
                            mom2[:, ni:nf, ni:nf], mom3[:, ni:nf, ni:nf]])
            E_n = np.array([E_n0[ni:nf].real, E_n1[ni:nf].real,
                            E_n2[ni:nf].real, E_n3[ni:nf].real])
            f_n = np.array([f_n0[ni:nf].real, f_n1[ni:nf].real,
                            f_n2[ni:nf].real, f_n3[ni:nf].real])

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                r_nm = np.zeros((4, 3, nb2, nb2), dtype=complex)
                D_nm = np.zeros((4, 3, nb2, nb2), dtype=complex)
                E_nm = np.zeros((4, nb2, nb2), dtype=float)
                for ii in range(4):
                    tmp = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp) < Etol
                    tmp[zeroind] = 1
                    E_nm[ii] = tmp
                    for aa in set(pol):
                        r_nm[ii, aa] = mom[ii, aa] / (1j * E_nm[ii])
                        r_nm[ii, aa, zeroind] = 0
                        p_nn = np.diag(mom[ii, aa])
                        D_nm[ii, aa] = (np.tile(p_nn[:, None], (1, nb2))
                                        - np.tile(p_nn[None, :], (nb2, 1)))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((4, 3, 3, nb2, nb2), dtype=complex)
                for ii in range(4):
                    tmp1 = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp1) < Etol
                    for aa in set(pol):
                        for bb in set(pol):
                            tmp = (r_nm[ii, aa] * np.transpose(D_nm[ii, bb])
                                   + r_nm[ii, bb] * np.transpose(D_nm[ii, aa])
                                   + 1j * np.dot(r_nm[ii, aa],
                                                 r_nm[ii, bb] * E_nm[ii])
                                   - 1j * np.dot(r_nm[ii, bb] * E_nm[ii],
                                                 r_nm[ii, aa])) / E_nm[ii]
                            tmp[zeroind] = 0
                            rnm_der[ii, aa, bb] = tmp
                    E_nm[ii, zeroind] = 0

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # Remove the non important term
                        if mmi <= nni:
                            continue
                        fnm = f_n[:, nni] - f_n[:, mmi]
                        Emn = E_nm[:, mmi, nni]

                        # Two band part
                        if np.all(np.abs(fnm)) > ftol:
                            Fval = fnm * np.imag(
                                r_nm[:, pol[1], mmi, nni]
                                * rnm_der[:, pol[0], pol[2], nni, mmi]
                                + r_nm[:, pol[2], mmi, nni]
                                * rnm_der[:, pol[0], pol[1], nni, mmi])

                            # Use the triangle method for integration
                            tmp = (triangle_int(Fval, Emn,
                                                w_l, itype=itype, 
                                                tri1=tri1, tri2=tri2)) * dA / 2
                            tmp2 = (triangle_int(Fval, -Emn,
                                                w_l, itype=itype, 
                                                tri1=tri1, tri2=tri2)) * dA / 2
                            sum2b += tmp.real-tmp2.real

                # Print the progress
                count += 1
                print_progressbar(count, ncount)
    else:
        parprint('Integration mode ' + intmethod + ' not implemented.')
        raise NotImplementedError

    # Sum over all nodes
    world.sum(sum2b)

    # Make the output in SI unit
    dim_init = -1j * gspin * _e**3 / (2 * _hbar * (2.0 * pi)**3)
    dim_sum = (_hbar / (Bohr * 1e-10))**3 / \
        (_e**4 * (Bohr * 1e-10)**3) * (_hbar / _me)**3 * bz_vol
    dim_SI = 1j * dim_init * dim_sum # 1j due to imag in loop
    if intmethod == 'no':
        sigma2 = dim_SI * sum2b
        sigma2b = sigma2
    elif intmethod == 'tri':
        # Make the real part of the chi and broaden the spectra
        with timer('Convolution with Lorentzian'):
            sum2b_conv = np.zeros(len(w_l), dtype=np.complex)
            for ind, omega in enumerate(w_l):
                sum2b_conv[ind] = np.trapz(
                    sum2b * eta / (pi * ((w_l - omega) ** 2
                                         + eta ** 2)), w_l)
            sigma2 = dim_SI * sum2b_conv
            sigma2b = dim_SI * sum2b
    # A multi-col output 
    # nw = int(nw / 2)
    # sigma2 = sigma2[nw:] - sigma2[nw - 1::-1]
    # sigma2b = sigma2b[nw:] - sigma2b[nw - 1::-1]
    shift = np.vstack((freqs, sigma2, sigma2b))

    # Save the data
    if world.rank == 0:
        # Save it to the file
        if outname is None:
            np.save('shift.npy', shift)
        else:
            np.save('{}.npy'.format(outname), shift)

        # Print the timing
        timer.write()

    # Return shift current
    return shift


# Calculate the EOP response (for nonmagnetic semiconductors) in the
# length gauge, reqularized version


def calculate_eop_rlg(
        freqs=1.0,
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        addsoc=False,
        socscale=1.0,
        intmethod='no',
        dermethod='sum',
        Etol=1e-3, ftol=1e-4,
        ni=None, nf=None,
        blist=None,
        outname=None,
        momname=None,
        basename=None):
    """
    Calculate EOP spectrum in length gauge (nonmagnetic semiconductors)

    Input:
        freqs           Excitation frequency array (a numpy array or list)
        eta             Broadening (a single number or an array)
        pol             Tensor element
        eshift          scissors shift (default 0)
        addsoc          Add spin-orbit coupling (default off)
        socscale        SOC scale (default 1.0)
        intmethod       Integral method (defaul no)
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        ni, nf          First and last bands in the calculations (0 to nb)
        blist           List of bands in the sum
        outname         Output filename (default is shg.npy)
        momname         Suffix for the momentum file (default gs.gpw)
        basename        Suffix for the gs file (default gs.gpw)
    Output:
        eop.npy          Numpy array containing spectrum and frequencies
    """

    timer = Timer()
    parprint(
        'Calculating EOP spectrum in length gauge (in {:d} cores).'.format(
            world.size))

    # Load the required data only in the master
    with timer('Load the calculations'):
        calc, moms = load_gsmoms(basename, momname)

    # Useful variables
    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = freqs + 1j * eta
    # w_lc = np.hstack((-w_lc[-1::-1], w_lc))
    # nw = 2 * nw
    nb, nk, mu, kbT, bz_vol, w_k, kd = set_variables(calc)
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb
    nf = nb+nf if (nf < 0) else nf
    assert nf <= nb, 'nf should be less than the number of bands ({})'.format(
        nb)
    polstr = pol
    pol = ['xyz'.index(ii) for ii in polstr]
    gspin = 2.0
    nb2 = nf - ni
    parprint('First/last bands: {}/{}, Total: {}, k-points: {}/{}'.format(
        ni, nf - 1, nb, len(kd.bzk_kc[:, 0]), nk))

    # x1 = 0.01/1.01
    # x2 = 1.0-x1
    # w_lc = 1.01*w_lc

    # x1 = 0.5
    # x2 = 0.5
    # w_lc = 2.0*w_lc
    w1 = freqs + 1j * eta
    w2 = -0.98*w1
    w_lc =  w1+w2
    x1 = w1/w_lc
    x2 = w2/w_lc
    

    # Get the energy and fermi levels
    with timer('Get energies and fermi levels'):
        # Get the energy, fermi
        if world.rank == 0:
            E_nk = calc.band_structure().todict()['energies'][0]
            f_nk = np.zeros((nk, nb), dtype=np.float64)
            for k_c in range(nk):
                f_nk[k_c] = calc.get_occupation_numbers(
                    kpt=k_c, spin=0) / w_k[k_c] / 2.0
        else:
            E_nk = None
            f_nk = None
    
    # Find the neighboring points
    nei_ind, psigns, q_vecs = find_neighbors(kd, qind=[[1, 0, 0], [0, 1, 0]])

    # Get eigenvalues and wavefunctions with spin orbit coupling
    if addsoc:
        gspin = 1.0
        nb2 = 2 * (nf - ni)
        with timer('Spin-orbit coupling calculation'):
            if world.rank == 0:
                # Get the SOC
                (e_mk, wfs_knm) = get_spinorbit_eigenvalues(
                    calc, return_wfs=True, bands=np.arange(ni, nf),
                    scale=socscale)

                # Make the data ready
                e_mk = e_mk.T
                wfs_knm = np.array(wfs_knm)

                # Update the Fermi level
                mu = fermi_level(calc,
                                 e_mk[np.newaxis],
                                 2 * calc.get_number_of_electrons())
            else:
                e_mk = None
                wfs_knm = None
                mu = None

            # Update the Fermi level
            mu = broadcast(mu, 0)

            # Distribute the SOC
            k_info2 = distribute_data(
                [e_mk, wfs_knm], [(nk, nb2), (nk, nb2, nb2)])

            # Get SOC from PAW
            dVL_avii, Pt_kasni = get_soc_paw(calc)

    # Initialize the outputs
    sum2b_A = np.zeros((nw), dtype=np.complex)
    sum2b_B = np.zeros((nw), dtype=np.complex)
    sum2b_C = np.zeros((nw), dtype=np.complex)
    if blist is None:
        blist = list(range(nb2))
    else:
        assert max(
            blist) < nb2, 'Maximum of blist should be smaller than nb.'
    blist = np.array(blist, int)

    # Depending on the integration method
    parprint(
        'Evaluating EOP response for {}-polarization ({:.1f} meV).'.format(
            polstr,
            1000 *
            eta))
    if intmethod == 'no':
        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info = distribute_data(
                [moms, E_nk, f_nk], [(nk, 3, nb, nb), (nk, nb), (nk, nb)])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)
        sum_d = np.zeros((3, 3), complex)

        # Loop over k points
        for k_c, data_k in k_info.items():
            mom, E_n, f_n = tuple(data_k)
            mom = mom[:, ni:nf, ni:nf]
            f_n0 = f_n[ni:nf].copy()

            # Add SOC or not
            if addsoc:
                E_n, wfs_nm = tuple(k_info2.get(k_c))
                E_n = E_n.real
                tmp = (E_n - mu) / kbT
                tmp = np.clip(tmp, -100, 100)
                f_n = 1 / (np.exp(tmp) + 1.0)
                f_n = np.zeros(nb2, complex)
                f_n[::2] = f_n0
                f_n[1::2] = f_n0

                # Make the new momentum
                with timer('New momentum calculation'):
                    mom2 = np.zeros((3, nb2, nb2), dtype=complex)
                    for pp in range(3):
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 0::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 0::2]))
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm[:, 1::2]),
                                                  mom[pp]),
                                           np.transpose(wfs_nm[:, 1::2]))

                        # Get the soc correction to momentum
                        p_vmm = socscale * \
                                get_soc_momentum(dVL_avii, Pt_kasni[k_c],
                                                 ni, nf)
                        mom2[pp] += np.dot(np.dot(np.conj(wfs_nm), p_vmm[pp]),
                                           np.transpose(wfs_nm))
                    mom = mom2
            else:
                E_n = E_n[ni:nf].real
                f_n = f_n[ni:nf].real

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                r_nm = np.zeros((3, nb2, nb2), complex)
                D_nm = np.zeros((3, nb2, nb2), complex)
                E_nm = np.tile(E_n[:, None], (1, nb2)) - \
                    np.tile(E_n[None, :], (nb2, 1))
                zeroind = np.abs(E_nm) < Etol
                E_nm[zeroind] = 1
                # np.fill_diagonal(E_nm, 1.0)
                for aa in set(pol):
                    r_nm[aa] = mom[aa] / (1j * E_nm)
                    r_nm[aa, zeroind] = 0
                    # np.fill_diagonal(r_nm[aa], 0.0)
                    p_nn = np.diag(mom[aa])
                    D_nm[aa] = np.tile(p_nn[:, None], (1, nb2)) - \
                        np.tile(p_nn[None, :], (nb2, 1))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((3, 3, nb2, nb2), complex)
                if dermethod == 'sum':
                    for aa in set(pol):
                        for bb in set(pol):
                            tmp = (r_nm[aa] * np.transpose(D_nm[bb])
                                + r_nm[bb] * np.transpose(D_nm[aa])
                                + 1j * np.dot(r_nm[aa], r_nm[bb] * E_nm)
                                - 1j * np.dot(r_nm[bb] * E_nm, r_nm[aa])) / E_nm
                            tmp[zeroind] = 0
                            rnm_der[aa, bb] = tmp
                            # np.fill_diagonal(rnm_der[aa, bb], 0.0)
                elif dermethod == 'log':
                    rd_vvnn = get_derivative(calc, nei_ind[:, k_c], q_vecs, ni+blist, ovth=0.5, timer=timer, psigns=psigns[:, k_c])
                    scale = (_me*(Bohr*1e-10)**2*_e/_hbar**2)
                    for v1 in range(3):
                        for v2 in range(3):
                            rnm_der[v1, v2] = r_nm[v2]*(rd_vvnn[v1, v2]*(-1)*scale-D_nm[v1]/E_nm)
                else:
                    parprint('Derivative mode ' + dermethod + ' not implemented.')
                    raise NotImplementedError

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # Remove the non important term using time-reversal
                        if mmi == nni:
                            continue
                        fnm = f_n[nni] - f_n[mmi]
                        Emn = E_nm[mmi, nni] + fnm * eshift

                        # Two band part
                        if np.abs(fnm) > ftol:
                            iterm = np.imag(r_nm[pol[0], nni, mmi]*(rnm_der[pol[2], pol[1], mmi, nni]/x2+rnm_der[pol[1], pol[2], mmi, nni]/x1))/(Emn*(w_lc - Emn))

                            iterm += np.imag(r_nm[pol[1], mmi, nni]*rnm_der[pol[2], pol[0], nni, mmi]*x1/x2)/(Emn*(x1*w_lc - Emn))
                            iterm += np.imag(r_nm[pol[2], mmi, nni]*rnm_der[pol[1], pol[0], nni, mmi]*x2/x1)/(Emn*(x2*w_lc - Emn))

                            iterm -= np.imag(r_nm[pol[0], nni, mmi]*(r_nm[pol[1], mmi, nni]*D_nm[pol[2], mmi, nni]/x2**2+r_nm[pol[2], mmi, nni]*D_nm[pol[1], mmi, nni]/x1**2)) \
                                / (w_lc - Emn) / Emn**2

                            iterm += np.imag(r_nm[pol[1], mmi, nni]*r_nm[pol[0], nni, mmi]*D_nm[pol[2], mmi, nni])*x1**2/(x2**2*(x1*w_lc-Emn)*Emn**2)
                            iterm += np.imag(r_nm[pol[2], mmi, nni]*r_nm[pol[0], nni, mmi]*D_nm[pol[1], mmi, nni])*x2**2/(x1**2*(x2*w_lc-Emn)*Emn**2)   

                            iterm -= np.imag(r_nm[pol[1], mmi, nni]*rnm_der[pol[0], pol[2], nni, mmi])/((x1*w_lc-Emn)*w_lc)
                            iterm -= np.imag(r_nm[pol[2], mmi, nni]*rnm_der[pol[0], pol[1], nni, mmi])/((x2*w_lc-Emn)*w_lc)

                            iterm -= np.imag(r_nm[pol[1], nni, mmi]*r_nm[pol[2], mmi, nni]*D_nm[pol[0], mmi, nni])/((x2*w_lc-Emn)*w_lc**2)
                            iterm -= np.imag(r_nm[pol[2], nni, mmi]*r_nm[pol[1], mmi, nni]*D_nm[pol[0], mmi, nni])/((x1*w_lc-Emn)*w_lc**2)

                            sum2b_B += 1j *fnm * iterm * w_k[k_c] / 2  # 1j imag

                        for lli in blist:
                            fnl = f_n[nni] - f_n[lli]
                            fml = f_n[mmi] - f_n[lli]
                            Eml = E_nm[mmi, lli] - fml * eshift
                            Eln = E_nm[lli, nni] + fnl * eshift
                            E12 = x1*Eln-x2*Eml
                            # E12 = 1.0 if np.abs(E12)<Etol else E12
                            E21 = x2*Eln-x1*Eml
                            # E21 = 1.0 if np.abs(E21)<Etol else E21

                            # Do not do zero calculations
                            if (np.abs(fnm) < ftol and np.abs(fnl) < ftol
                                    and np.abs(fml) < ftol):
                                continue

                            rnml1 = np.real(r_nm[pol[0], nni, mmi]*r_nm[pol[1], mmi, lli]*r_nm[pol[2], lli, nni]) 
                            rnml2 = np.real(r_nm[pol[0], nni, mmi]*r_nm[pol[2], mmi, lli]*r_nm[pol[1], lli, nni])
                            if np.abs(fnm) > ftol:
                                sum2b_A += fnm / (w_lc - Emn) * (rnml1 / E12 + rnml1 / E21) * w_k[k_c] / 2
                            if np.abs(fnl) > ftol:
                                sum2b_A += -x2 * fnl * rnml1 / E12 / (x2*w_lc - Eln) * w_k[k_c] / 2
                                sum2b_A += -x1 * fnl * rnml2 / E21 / (x1*w_lc - Eln) * w_k[k_c] / 2
                            if np.abs(fml) > ftol:
                                sum2b_A += x1 * fml / (x1*w_lc - Eml) * rnml1 / E12 * w_k[k_c] / 2
                                sum2b_A += x2 * fml / (x2*w_lc - Eml) * rnml2 / E21 * w_k[k_c] / 2

                # Print the progress
                count += 1
                print_progressbar(count, ncount)
    elif intmethod == 'tri':
        # Useful variable
        itype = 1
        tri1=[0, 1, 3]
        tri2=[0, 2, 3]
        w_l = w_lc.real
        nb2 = nf - ni

        # Initialize variables
        assert not addsoc, 'Triangular method is only implemented without SOC.'

        # Distribute the k points between cores
        with timer('k-info distribution'):
            k_info, dA = get_neighbors(moms, E_nk, f_nk, kd, nb, qind=[
                                       [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Initial call to print 0% progress
        count = 0
        ncount = len(k_info)
        print_progressbar(count, ncount)

        # Loop over the k points
        for k_c, data_k in k_info.items():
            # Get 4-points data (corners of a rectangle)
            mom0, mom1, mom2, mom3, E_n0, E_n1, E_n2, E_n3, \
                f_n0, f_n1, f_n2, f_n3 = tuple(data_k)
            mom = np.array([mom0[:, ni:nf, ni:nf], mom1[:, ni:nf, ni:nf],
                            mom2[:, ni:nf, ni:nf], mom3[:, ni:nf, ni:nf]])
            E_n = np.array([E_n0[ni:nf], E_n1[ni:nf],
                            E_n2[ni:nf], E_n3[ni:nf]])
            f_n = np.array([f_n0[ni:nf], f_n1[ni:nf],
                            f_n2[ni:nf], f_n3[ni:nf]])

            # Make the position matrix elements and Delta
            with timer('Position matrix elements calculation'):
                r_nm = np.zeros((4, 3, nb2, nb2), dtype=np.complex)
                D_nm = np.zeros((4, 3, nb2, nb2), dtype=np.complex)
                E_nm = np.zeros((4, nb2, nb2), dtype=np.complex)
                for ii in range(4):
                    tmp = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp) < Etol
                    tmp[zeroind] = 1
                    E_nm[ii] = tmp
                    for aa in set(pol):
                        r_nm[ii, aa] = mom[ii, aa] / (1j * E_nm[ii])
                        r_nm[ii, aa, zeroind] = 0
                        p_nn = np.diag(mom[ii, aa])
                        D_nm[ii, aa] = (np.tile(p_nn[:, None], (1, nb2))
                                        - np.tile(p_nn[None, :], (nb2, 1)))

                # Make the generalized derivative of rnm
                rnm_der = np.zeros((4, 3, 3, nb2, nb2), dtype=np.complex)
                for ii in range(4):
                    tmp = (np.tile(E_n[ii, :, None], (1, nb2))
                           - np.tile(E_n[ii, None, :], (nb2, 1)))
                    zeroind = np.abs(tmp) < Etol
                    for aa in set(pol):
                        for bb in set(pol):
                            tmp = (r_nm[ii, aa] * np.transpose(D_nm[ii, bb])
                                   + r_nm[ii, bb] * np.transpose(D_nm[ii, aa])
                                   + 1j * np.dot(r_nm[ii, aa],
                                                 r_nm[ii, bb] * E_nm[ii])
                                   - 1j * np.dot(r_nm[ii, bb] * E_nm[ii],
                                                 r_nm[ii, aa])) / E_nm[ii]
                            tmp[zeroind] = 0
                            rnm_der[ii, aa, bb] = tmp

            # Loop over bands
            with timer('Sum over bands'):
                for nni in blist:
                    for mmi in blist:
                        # Remove non important term using time-reversal
                        # symmtery
                        if mmi <= nni:
                            continue
                        fnm = f_n[:, nni] - f_n[:, mmi]
                        Emn = E_nm[:, mmi, nni] + fnm * eshift

                        # Two band part
                        if np.any(np.abs(fnm)) > ftol:

                            iterm = fnm * np.imag(r_nm[:, pol[0], nni, mmi]*(rnm_der[:, pol[2], pol[1], mmi, nni]/x2+rnm_der[:, pol[1], pol[2], mmi, nni]/x1))/(Emn)
                            val = triangle_int(iterm, Emn, w_l, itype=itype, tri1=tri1, tri2=tri2)

                            iterm = fnm * np.imag(r_nm[:, pol[1], mmi, nni]*rnm_der[:, pol[2], pol[0], nni, mmi]*x1/x2)/(Emn)
                            val += triangle_int(iterm, Emn, x1*w_l, itype=itype, tri1=tri1, tri2=tri2)
                            iterm = fnm * np.imag(r_nm[:, pol[2], mmi, nni]*rnm_der[:, pol[1], pol[0], nni, mmi]*x2/x1)/(Emn)
                            val += triangle_int(iterm, Emn, x2*w_l, itype=itype, tri1=tri1, tri2=tri2)

                            iterm = -fnm * np.imag(r_nm[:, pol[0], nni, mmi]*(r_nm[:, pol[1], mmi, nni]*D_nm[:, pol[2], mmi, nni]/x2**2+r_nm[:, pol[2], mmi, nni]*D_nm[:, pol[1], mmi, nni]/x1**2))/Emn**2
                            val += triangle_int(iterm, Emn, w_l, itype=itype, tri1=tri1, tri2=tri2)

                            iterm = fnm * np.imag(r_nm[:, pol[1], mmi, nni]*r_nm[:, pol[0], nni, mmi]*D_nm[:, pol[2], mmi, nni])*x1**2/(x2**2*Emn**2)
                            val += triangle_int(iterm, Emn, x1*w_l, itype=itype, tri1=tri1, tri2=tri2)
                            iterm = fnm * np.imag(r_nm[:, pol[2], mmi, nni]*r_nm[:, pol[0], nni, mmi]*D_nm[:, pol[1], mmi, nni])*x2**2/(x1**2*Emn**2) 
                            val += triangle_int(iterm, Emn, x2*w_l, itype=itype, tri1=tri1, tri2=tri2)

                            iterm = -fnm * np.imag(r_nm[:, pol[1], mmi, nni]*rnm_der[:, pol[0], pol[2], nni, mmi])
                            val += triangle_int(iterm, Emn, x1*w_l, itype=itype, tri1=tri1, tri2=tri2)/(w_l)
                            iterm = -fnm * np.imag(r_nm[:, pol[2], mmi, nni]*rnm_der[:, pol[0], pol[1], nni, mmi])
                            val += triangle_int(iterm, Emn, x2*w_l, itype=itype, tri1=tri1, tri2=tri2)/(w_l)

                            iterm = -fnm * np.imag(r_nm[:, pol[1], nni, mmi]*r_nm[:, pol[2], mmi, nni]*D_nm[:, pol[0], mmi, nni])
                            val += triangle_int(iterm, Emn, x2*w_l, itype=itype, tri1=tri1, tri2=tri2)/(w_l**2)
                            iterm = -fnm * np.imag(r_nm[:, pol[2], nni, mmi]*r_nm[:, pol[1], mmi, nni]*D_nm[:, pol[0], mmi, nni])
                            val += triangle_int(iterm, Emn, x1*w_l, itype=itype, tri1=tri1, tri2=tri2)/(w_l**2)

                            sum2b_B += 1j * val * dA / 4.0  # 1j for imag

                        for lli in blist:
                            fnl = f_n[:, nni] - f_n[:, lli]
                            fml = f_n[:, mmi] - f_n[:, lli]
                            Eml = E_nm[:, mmi, lli] - fml * eshift
                            Eln = E_nm[:, lli, nni] + fnl * eshift
                            E12 = x1*Eln-x2*Eml
                            E21 = x2*Eln-x1*Eml
                            
                            # Do not do zero calculations
                            if (np.all(np.abs(fnm)) < ftol
                                    and np.all(np.abs(fnl)) < ftol
                                    and np.all(np.abs(fml)) < ftol):
                                continue

                            rnml1 = np.real(r_nm[:, pol[0], nni, mmi]*r_nm[:, pol[1], mmi, lli]*r_nm[:, pol[2], lli, nni]) 
                            rnml2 = np.real(r_nm[:, pol[0], nni, mmi]*r_nm[:, pol[2], mmi, lli]*r_nm[:, pol[1], lli, nni])
                            if np.any(np.abs(fnm) > ftol):
                                Fval = fnm * (rnml1 / E12 + rnml1 / E21) * dA / 4.0
                                sum2b_A += triangle_int(Fval, Emn, w_l, itype=itype, tri1=tri1, tri2=tri2)
                            if np.any(np.abs(fnl) > ftol):
                                Fval = -x2 * fnl * rnml1 / E12 * dA / 4.0
                                sum2b_A += triangle_int(Fval, Eln, x2*w_l, itype=itype, tri1=tri1, tri2=tri2)
                                Fval = -x1 * fnl * rnml2 / E21 * dA / 4.0
                                sum2b_A += triangle_int(Fval, Eln, x1*w_l, itype=itype, tri1=tri1, tri2=tri2)
                            if np.any(np.abs(fml) > ftol):
                                Fval = x1 * fml * rnml1 / E12 * dA / 4.0
                                sum2b_A += triangle_int(Fval, Eml, x1*w_l, itype=itype, tri1=tri1, tri2=tri2)
                                Fval = x2 * fml * rnml2 / E21 * dA / 4.0
                                sum2b_A += triangle_int(Fval, Eml, x2*w_l, itype=itype, tri1=tri1, tri2=tri2)


            # Print the progress
            count += 1
            print_progressbar(count, ncount)
    else:
        parprint('Integration mode ' + intmethod + ' not implemented.')
        raise NotImplementedError

    # Sum over all nodes
    # par.comm.Barrier()
    with timer('Gather data from cores'):
        world.sum(sum2b_A)
        world.sum(sum2b_B)

    # Make the output in SI unit
    dim_ee = gspin * _e**3 / (_eps0 * (2.0 * pi)**3)
    dim_sum = (_hbar / (Bohr * 1e-10))**3 / \
        (_e**5 * (Bohr * 1e-10)**3) * (_hbar / _me)**3 * bz_vol
    dim_ee_SI = dim_ee * dim_sum
    dim_ie_SI = 1j * dim_ee * dim_sum
    chi2b = np.zeros((5, nw), complex)

    # Save the data
    if world.rank == 0:
        if intmethod == 'no':
            chi2b[1, :] = dim_ee_SI * sum2b_A
            chi2b[2, :] = dim_ie_SI * sum2b_B
            # chi2b[3, :] = dim_ie_SI * sum2b_C / w_lc
            # chi2b[4, :] = dim_ee_SI*sum2b_D/w_lc
        elif intmethod == 'tri':
            # Make the real part of the chi and broaden the spectra
            with timer('Hilbert transform'):
                suml_A = np.zeros(len(w_l), complex)
                suml_B = np.zeros(len(w_l), complex)
                for ind, omega in enumerate(w_lc):
                    suml_A[ind] = np.trapz(
                        sum2b_A * w_l / (omega ** 2 - w_l ** 2), w_l)
                    suml_B[ind] = np.trapz(
                        sum2b_B * w_l / (omega ** 2 - w_l ** 2), w_l)
                chi2b[1, :] = dim_ee_SI * suml_A
                chi2b[2, :] = dim_ie_SI * suml_B

        chi2b[0, :] = chi2b[1, :] + chi2b[2, :] + chi2b[3, :] + chi2b[4, :]
        chi2 = chi2b[0, :]
        # nw = int(nw / 2)
        # chi2 = chi2[nw:] + chi2[nw - 1::-1]
        # chi2b = chi2b[:, int(nw):] + chi2b[:, nw - 1::-1]

        # A multi-col output
        eop = np.vstack((freqs, chi2, chi2b))

        # Save it to the file
        if outname is None:
            np.save('shg.npy', eop)
        else:
            np.save('{}.npy'.format(outname), eop)

        # Print the timing
        timer.write()
    else:
        eop = None

    # Return EOP respnse
    return eop

# Test the script


if __name__ == '__main__':
    from ase.io import read

    # Load the input structure and add the vaccum around the layer
    atoms_name = 'structure.json'
    atoms = read(atoms_name)
    atoms.center(vacuum=15, axis=2)
    cell = atoms.get_cell()
    cellsize = atoms.get_cell_lengths_and_angles()

    # Base name for gs and other files
    nk = 60
    basename = 'nk{}'.format(nk)
    reset_calc = False
    addsoc = False
    plot_sigma = False

    # GPAW parameters
    params_gs = dict(
        mode=PW(600),
        symmetry={'point_group': False, 'time_reversal': True},
        nbands='400%',
        convergence={'bands': -10},
        parallel={'domain': 1},
        occupations=FermiDirac(width=0.05),
        kpts={'size': (nk, nk, 1), 'gamma': True},
        xc='PBE',
        txt='gs_{}_paw.txt'.format(basename))
    # params_gs = dict(
    #     mode='lcao',
    #     symmetry={'point_group': False, 'time_reversal': True},
    #     nbands = 'nao',
    #     convergence={'bands': 'all'},
    #     basis='dzp',
    #     parallel={'domain': 1},
    #     occupations=FermiDirac(width=0.05),
    #     kpts={'size': (nk, nk, 1), 'gamma': True},
    #     xc='PBE')

    # Calculation params
    eta = 0.05  # Broadening in eV
    tensordict = get_tensor_elements(atoms)
    pols = tensordict.keys()  # or simply give the list of pols = ['yyy']
    w_ls = np.linspace(0.001, 6, 500)  # in eV
    if plot_sigma:
        mult1 = -1j * _eps0 * w_ls * _e / _hbar * \
            cellsize[2] * 1e-10 / (2 * np.pi)
        mult2 = 2 * mult1
    else:
        mult1 = 1
        mult2 = 1

    # Start a ground state calculation, if that has not been done earlier
    gs_name = 'gs_' + basename + '.gpw'
    if is_file_exist(gs_name) or reset_calc:
        calc = GPAW(**params_gs)
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.calc.write(gs_name, mode='all')

    # The momentum matrix are calculated if not available
    # reset_calc = True
    dip_name = 'dip_vknm_' + basename + '.npy'
    if is_file_exist(dip_name) or reset_calc:
        get_dipole_transitions(atoms, momname=basename, basename=basename)

    # Calculate SHG spectrum and plot it
    # reset_calc = False
    for pol in pols:
        if pol == 'zero':
            continue
        # blist = list(range(0, 45))
        blist = None
        shg_name1 = 'shg_' + pol + '_' + basename + '_vg_' + str(addsoc)
        if is_file_exist(shg_name1 + '.npy') or reset_calc:
            calculate_shg_rvg(
                freqs=w_ls,
                eta=eta,
                pol=pol,
                addsoc=addsoc,
                socscale=1.0,
                ni=0,
                nf=None,
                blist=blist,
                intmethod='no',
                outname=shg_name1,
                momname=basename,
                basename=basename)
        shg_name2 = 'shg_' + pol + '_' + basename + '_lg_' + str(addsoc)
        if is_file_exist(shg_name2 + '.npy') or reset_calc:
            calculate_shg_rlg(
                freqs=w_ls,
                eta=eta,
                pol=pol,
                eshift=1.0,
                addsoc=addsoc,
                socscale=0.0,
                ni=0,
                nf=None,
                blist=blist,
                intmethod='no',
                outname=shg_name2,
                momname=basename,
                basename=basename)
        plot_spectrum(figname=('shg_re_' + pol + '_' + basename + '_'
                               + str(addsoc) + '.png'),
                      resname=[shg_name1, shg_name2],
                      dtype='re',
                      wlim=(w_ls[0], w_ls[-1]),
                      ylim=None,
                      mult=mult2,
                      pind=[1])
        plot_spectrum(figname=('shg_im_' + pol + '_' + basename + '_'
                               + str(addsoc) + '.png'),
                      resname=[shg_name1, shg_name2],
                      dtype='im',
                      wlim=(w_ls[0], w_ls[-1]),
                      ylim=None,
                      mult=mult2,
                      pind=[1])

        # Calculate shift current and plot it
        shift_name1 = 'shift_' + pol + '_' + basename + '_' + str(addsoc)
        if is_file_exist(shift_name1 + '.npy') or reset_calc:
            calculate_shift_current(
                freqs=w_ls,
                eta=eta,
                pol=pol,
                addsoc=addsoc,
                socscale=1.0,
                ni=0,
                nf=None,
                intmethod='no',
                outname=shift_name1,
                momname=basename,
                basename=basename)
        shift_name2 = 'shift_' + pol + '_' + \
            basename + '_' + str(addsoc) + '_tri'
        # if is_file_exist(shift_name1 + '.npy') or reset_calc:
        #     calculate_shift_current(
        #         freqs=w_ls,
        #         eta=eta,
        #         pol=pol,
        #         addsoc=addsoc,
        #         socscale=1.0,
        #         ni=0,
        #         nf=None,
        #         intmethod='tri',
        #         outname=shift_name2,
        #         momname=basename,
        #         basename=basename)
        plot_spectrum(figname=('shift_re_' + pol + '_' + basename + '_'
                               + str(addsoc) + '.png'),
                      resname=[shift_name1],
                      dtype='re',
                      wlim=(w_ls[0], w_ls[-1]),
                      ylim=None,
                      pind=[1])

    # Plot all on one graph
    shg_names = ['shg_' + pol + '_' + basename + '_lg_' + str(addsoc)
                 for pol in pols if pol != 'zero']
    legends = [pol for pol in pols if pol != 'zero']
    plot_spectrum(
        figname='shg_all_' + basename + '_' + str(addsoc) + '.png',
        resname=shg_names,
        dtype='abs',
        mult=1.0,
        wlim=(w_ls[0], w_ls[-1]),
        ylim=None,
        leg=legends,
        pind=[1])
    shift_names = ['shift_' + pol + '_' + basename + '_' + str(addsoc)
                   for pol in pols if pol != 'zero']
    legends = [pol for pol in pols if pol != 'zero']
    plot_spectrum(
        figname='shift_all_' + basename + '_' + str(addsoc) + '.png',
        resname=shift_names,
        dtype='re',
        wlim=(w_ls[0], w_ls[-1]),
        ylim=None,
        leg=legends,
        pind=[1])

    # Calculate linear spectrum and plot it
    df_name1 = 'df_' + basename + '_' + str(addsoc)
    if is_file_exist(df_name1 + '.npy') or reset_calc:
        calculate_df(
            freqs=w_ls,
            eta=eta,
            pol='xx',
            addsoc=addsoc,
            socscale=1.0,
            ni=0,
            nf=None,
            intmethod='no',
            outname=df_name1,
            momname=basename,
            basename=basename)
    # df_name2 = 'df_' + basename + '_' + str(addsoc) + '_tri'
    # if is_file_exist(df_name2 + '.npy') or reset_calc:
    #     calculate_df(
    #         freqs=w_ls,
    #         eta=eta,
    #         pol='zz',
    #         addsoc=addsoc,
    #         socscale=1.0,
    #         ni=0,
    #         nf=None,
    #         intmethod='no',
    #         outname=df_name2,
    #         momname=basename,
    #         basename=basename)
    plot_spectrum(
        figname='df_im_' + basename + '_' + str(addsoc) + '.png',
        resname=[df_name1],
        dtype='comp',
        wlim=(w_ls[0], w_ls[-1]),
        ylim=None,
        pind=[1])

    # Check the sum rule
    check_sumrule(ni=0, nf=None, momname=basename, basename=basename)

    # Get polarized
    selw = 0.66
    psi = np.linspace(0, 2 * pi, 201)
    chipol = get_shg_polarized(
        atoms,
        prename='shg_',
        postname='_' + basename + '_lg_' + str(addsoc),
        wind=[np.argmin(np.abs(w_ls - selw))],
        theta=0,
        phi=0,
        pte=np.sin(psi),
        ptm=np.cos(psi),
        E0=[1.0],
        outname=None,
        outbasis='pol')
    # Plot the polar graph
    if world.rank == 0:
        fig = plt.figure(figsize=(4.0, 4.0), dpi=300)
        plot_polar(
            psi, [np.abs(chipol[0]), np.abs(chipol[1])],
            fig=fig,
            figname='shg_polarized_' + basename + '_' + str(addsoc) + '.png',
            leg=['Par.', 'Perp.'],
            title=r'At $\hbar\omega=${:.2} (eV)'.format(
                w_ls[np.argmin(np.abs(w_ls - selw))]))
