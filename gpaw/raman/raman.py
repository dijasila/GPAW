"""
Calculates Raman matrices and intensities.

Momentum matrices are not symmetric. The first index is the bra, the other the
ket.

i -> j -> m -> n
i, n are valence; j, m are conduction, also i=n in the end
see https://doi.org/10.1038/s41467-020-16529-6
"""

import numpy as np
from ase.units import invcm


def lorentzian(w, gamma):
    l = 0.5 * gamma / (np.pi * (w**2 + 0.25 * gamma**2))
    return l


def gaussian(w, sigma):
    g = 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-w**2 / (2 * sigma**2))
    return g


def calculate_raman(calc, w_ph, w_in, d_i, d_o, resonant_only=False,
                    gamma_l=0.1, momname='mom_skvnm.npy',
                    elphname='gsqklnn.npy', gridspacing=1.0):
    """
    Calculates the first order Raman tensor contribution for a given
    polarization.

    Parameters
    ----------
    calc: GPAW
        Converged ground state calculation
    phonon: Array, str
        Array of phonon frequencies in eV, or name of file with them
    w_in: float
        Laser energy in eV
    d_i: int
        Incoming polarization
    d_o: int
        Outgoing polarization
    gamma_l: float
        Line broadening in eV
    momname: str
        Name of momentum file
    elphname: str
        Name of electron-phonon file
    gridspacing: float
        grid spacing in cm^-1
    """
    # those won't make sense here
    assert calc.wfs.gd.comm.size == 1
    assert calc.wfs.bd.comm.size == 1
    kd = calc.wfs.kd

    print("Calculating Raman spectrum: Laser frequency = {} eV".format(w_in))

    # Phonon frequencies
    if isinstance(w_ph, str):
        w_ph = np.load(w_ph)
    assert max(w_ph) < 1.  # else not eV units
    nmodes = len(w_ph)

    # Set grid
    w_max = np.round(np.max(w_ph) / invcm + 50, -1)  # max of grid in rcm
    ngrid = int(w_max / gridspacing)
    w = np.linspace(0., w_max, num=ngrid) * invcm  # in eV

    # Load files
    mom_skvnm = np.load(momname, mmap_mode='c')
    g_sqklnn = np.load(elphname, mmap_mode='c')  # [s,q=0,k,l,n,m]

    # Define a few more variables
    opt = 'optimal'  # mode for np.einsum. not sure what is fastest
    nspins = g_sqklnn.shape[0]
    nk = g_sqklnn.shape[2]
    assert nmodes == g_sqklnn.shape[3]
    assert mom_skvnm.shape[0] == nspins
    assert mom_skvnm.shape[1] == nk
    assert mom_skvnm.shape[-1] == g_sqklnn.shape[-1]

    # valence is ket in ij and bra in mn
    # ab is in and out polarization
    # l is the phonon mode and w is the raman shift

    # XXX: The below can probably be made better by lambdas a lot
    def _term1(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv):
        term1_l = np.zeros((elph_lnn.shape[0]), dtype=complex)
        t1_ij = f_vc * mom_dnn[0, nc:, :nv].T / (w_in - E_vc)
        for l in range(nmodes):
            t1_xx = elph_lnn[l]
            t1_mn = (f_vc * mom_dnn[1, :nv, nc:] / (w_in - w_ph[l] - E_vc)).T
            term1_l[l] += np.einsum('sj,jm,ms', t1_ij, t1_xx[nc:, nc:], t1_mn,
                                    optimize=opt)
            term1_l[l] -= np.einsum('is,ni,sn', t1_ij, t1_xx[:nv, :nv], t1_mn,
                                    optimize=opt)
        return term1_l

    def _term2(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv):
        term2_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t2_ij = f_vc * mom_dnn[0, nc:, :nv].T / (w_in - E_vc)
        t2_xx = mom_dnn[1]  # XXX might need to T or conj
        for l in range(nmodes):
            for wi in range(ngrid):
                t2_mn = f_vc.T * elph_lnn[l][nc:, :nv] / (w[wi] - E_vc.T)
                term2_lw[l, wi] += np.einsum('sj,jm,ms', t2_ij,
                                             t2_xx[nc:, nc:], t2_mn,
                                             optimize=opt)
                term2_lw[l, wi] -= np.einsum('is,ni,sn', t2_ij,
                                             t2_xx[:nv, :nv], t2_mn,
                                             optimize=opt)
        return term2_lw

    def _term3(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv):
        term3_lw = np.zeros((nmodes, ngrid), dtype=complex)
        for l in range(nmodes):
            t3_xx = elph_lnn[l]
            for wi in range(ngrid):
                t3_ij = f_vc * mom_dnn[1, nc:, :nv].T / (-w_in + w[wi] - E_vc)
                t3_mn = (f_vc * mom_dnn[0, :nv, nc:] / (-w_in - w_ph[l] + w[wi]
                                                        - E_vc)).T
                term3_lw[l, wi] += np.einsum('sj,jm,ms', t3_ij,
                                             t3_xx[nc:, nc:], t3_mn,
                                             optimize=opt)
                term3_lw[l, wi] -= np.einsum('is,ni,sn', t3_ij,
                                             t3_xx[:nv, :nv], t3_mn,
                                             optimize=opt)
        return term3_lw

    def _term4(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv):
        term4_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t4_xx = mom_dnn[0]  # XXX might need to T or conj
        for l in range(nmodes):
            for wi in range(ngrid):
                t4_ij = f_vc * mom_dnn[1, nc:, :nv].T / (-w_in + w[wi] - E_vc)
                t4_mn = (f_vc * elph_lnn[l, nc:, :nv].T / (w[wi] - E_vc)).T
                term4_lw[l, wi] += np.einsum('sj,jm,ms', t4_ij,
                                             t4_xx[nc:, nc:], t4_mn,
                                             optimize=opt)
                term4_lw[l, wi] -= np.einsum('is,ni,sn', t4_ij,
                                             t4_xx[:nv, :nv], t4_mn,
                                             optimize=opt)
        return term4_lw

    def _term5(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv):
        term5_l = np.zeros((nmodes), dtype=complex)
        t5_xx = mom_dnn[0]  # XXX might need to T or conj
        for l in range(nmodes):
            t5_ij = f_vc * elph_lnn[l, :nv, nc:] / (-w_ph[l] - E_vc)
            t5_mn = (f_vc * mom_dnn[1, :nv, nc:] / (w_in - w_ph[l] - E_vc)).T
            term5_l[l] += np.einsum('sj,jm,ms', t5_ij, t5_xx[nc:, nc:], t5_mn,
                                    optimize=opt)
            term5_l[l] -= np.einsum('is,ni,sn', t5_ij, t5_xx[:nv, :nv], t5_mn,
                                    optimize=opt)
        return term5_l

    def _term6(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv):
        term6_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t6_xx = mom_dnn[1]  # XXX might need to T or conj
        for l in range(nmodes):
            t6_ij = f_vc * elph_lnn[l, :nv, nc:] / (-w_ph[l] - E_vc)
            for wi in range(ngrid):
                t6_mn = (f_vc * mom_dnn[0, :nv, nc:] / (-w_in - w_ph[l] + w[wi]
                                                        - E_vc)).T
                term6_lw[l, wi] += np.einsum('sj,jm,ms', t6_ij,
                                             t6_xx[nc:, nc:], t6_mn,
                                             optimize=opt)
                term6_lw[l, wi] -= np.einsum('is,ni,sn', t6_ij,
                                             t6_xx[:nv, :nv], t6_mn,
                                             optimize=opt)
        return term6_lw

    # -------------------------------------------------------------------------

    print("Evaluating Raman sum")
    raman_lw = np.zeros((nmodes, ngrid), dtype=complex)

    # Loop over kpoints - this is parallelised
    for kpt in calc.wfs.kpt_u:
        print("Rank {}: s={}, k={}".format(kd.comm.rank, kpt.s, kpt.k))

        # Check if we need to add timer-add time reversed kpoint
        if (calc.symmetry.time_reversal and not
            np.allclose(kd.ibzk_kc[kpt.k], [0., 0., 0.])):
            add_time_reversed = True
        else:
            add_time_reversed = False

        # Limit sums to relevant bands, partially occupied bands are a pain.
        # So, in principal, partially occupied bands can be initial and
        # final states, but we need to restrict to a positive deltaE if we
        # allow this.
        f_n = kpt.f_n / kpt.weight
        assert np.isclose(max(f_n), 1.0, atol=0.1)
        vs = np.where(f_n >= 0.1)[0]
        cs = np.where(f_n < 0.9)[0]
        nv = max(vs) + 1  # VBM+1 index
        nc = min(cs)  # CBM index

        # Precalculate f * (1-f) term
        f_vc = np.outer(kpt.f_n[vs], 1. - kpt.f_n[cs])
        # Precalculate E-E term
        E_vc = np.empty((len(vs), len(cs)), dtype=complex)
        for n in vs:
            E_vc[n] = kpt.eps_n[cs] - kpt.eps_n[n] + 1j * gamma_l
            # set weights for negative energy transitions zero
            neg = np.where(E_vc[n] <= 0.)[0]
            f_vc[n, neg] = 0.

        # Obtain appropriate part of mom and g arrays
        mom_dnn = np.ascontiguousarray(mom_skvnm[kpt.s, kpt.k, [d_i, d_o]])
        assert mom_dnn.shape[0] == 2
        g_lnn = np.ascontiguousarray(g_sqklnn[kpt.s, 0, kpt.k])

        # Raman contribution of this k-point
        this_lw = np.zeros((nmodes, ngrid), dtype=complex)

        # Resonant term
        term1_l = _term1(f_vc, E_vc, mom_dnn, g_lnn, nc, nv)
        # print("Term1: ", np.max(np.abs(term1_l)))
        this_lw += term1_l[:, None]

        if not resonant_only:
            term2_lw = _term2(f_vc, E_vc, mom_dnn, g_lnn, nc, nv)
            # print("Term2: ", np.max(np.abs(term2_lw)))
            this_lw += term2_lw

            term3_lw = _term3(f_vc, E_vc, mom_dnn, g_lnn, nc, nv)
            # print("Term3: ", np.max(np.abs(term3_lw)))
            this_lw += term3_lw

            term4_lw = _term4(f_vc, E_vc, mom_dnn, g_lnn, nc, nv)
            # print("Term4: ", np.max(np.abs(term4_lw)))
            this_lw += term4_lw

            term5_l = _term5(f_vc, E_vc, mom_dnn, g_lnn, nc, nv)
            # print("Term5: ", np.max(np.abs(term5_l)))
            this_lw += term5_l[:, None]

            term6_lw = _term6(f_vc, E_vc, mom_dnn, g_lnn, nc, nv)
            # print("Term6: ", np.max(np.abs(term6_lw)))
            this_lw += term6_lw

        # At the moment we only allow no symmetry, so all k-points same
        # weight, or time_reversal only. In the later case r -> 2*Re(r)
        # because gdd-> (gdd)^* for k-> -k
        if add_time_reversed:
            raman_lw += 2. * this_lw.real
        else:
            raman_lw += this_lw

    # Collect parallel contributions
    kd.comm.sum(raman_lw)

    if kd.comm.rank == 0:
        print('Raman intensities per mode')
        print('--------------------------')
        ff = '  Phonon {} with energy = {:4.2f} rcm: {:.4f}'
        for l in range(nmodes):
            print(ff.format(l, w_ph[l] / invcm, abs(np.sum(raman_lw[l]))))

        # Save Raman intensities to disk
        raman = np.vstack((w / invcm, raman_lw))
        np.save("Rlab_{}{}.npy".format('xyz'[d_i], 'xyz'[d_o]), raman)


def calculate_raman_tensor(calc, w_ph, w_in, resonant_only=False,
                           gamma_l=0.1, momname='mom_skvnm.npy',
                           elphname='gsqklnn.npy'):
    for i in range(3):
        for j in range(3):
            calculate_raman(calc, w_ph, w_in, d_i=i, d_o=j,
                            resonant_only=resonant_only, gamma_l=gamma_l,
                            momname=momname, elphname=elphname)


def calculate_raman_intensity(d_i, d_o, ramanname=None, T=300):
    """
    Calculates Raman intensities from Raman tensor.

    Parameters
    ----------
    d_i: int
        Incoming polarization
    d_o: int
        Outgoing polarization
    """
    # KtoeV = 8.617278E-5
    cm = 1. / 8065.544  # cm^-1 to eV
    w_ph = np.load("vib_frequencies.npy")  # in ev?

    # Load raman matrix elements R_lab
    xyz = 'xyz'
    if ramanname is None:
        tmp = np.load("Rlab_{}{}.npy".format(xyz[d_i], xyz[d_o]))
    else:
        tmp = np.load("Rlab_{}{}_{}.npy".format(xyz[d_i], xyz[d_o], ramanname))
    w = tmp[0].real
    raman_lw = tmp[1:]

    intensity = np.zeros_like(w)
    for l in range(len(raman_lw)):
        # occ = 1. / (np.exp(w_ph[l] / (KtoeV * T)) - 1.) + 1.
        delta = gaussian(w=w - w_ph[l] / cm, sigma=5.)
        # print(occ, np.max(delta), w_ph[l], np.max(np.abs(raman_lw[l])**2))
        # intensity += occ / w_ph[l] * np.abs(raman_lw[l])**2 * delta
        # ignore phonon occupation numbers for now
        # not sure, if the above is correct or not, but the below yields nicer
        # looking results
        intensity += np.abs(raman_lw[l])**2 * delta

    raman = np.vstack((w, intensity))
    if ramanname is None:
        np.save("RI_{}{}.npy".format(xyz[d_i], xyz[d_o]), raman)
    else:
        np.save("RI_{}{}_{}.npy".format(xyz[d_i], xyz[d_o], ramanname), raman)


def plot_raman(figname="Raman.png", relative=True, w_min=None, w_max=None,
               ramanname=None, yscale="linear"):
    """Plots a given Raman spectrum.

    Parameters
    ----------
    figname: str
        Filename for figure

    """
    from scipy import signal
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    if ramanname is None:
        legend = False
        RI_name = ["RI.npy"]
    elif type(ramanname) == list:
        legend = True
        RI_name = ["RI_{}.npy".format(name) for name in ramanname]
    else:
        legend = False
        RI_name = ["RI_{}.npy".format(ramanname)]

    ylabel = "Intensity (arb. units)"
    cm = plt.get_cmap('inferno')
    cNorm = colors.Normalize(vmin=0, vmax=len(RI_name))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    peaks = None
    for i, name in enumerate(RI_name):
        RI = np.real(np.load(name))
        if w_min is None:
            w_min = np.min(RI[0])
        if w_max is None:
            w_max = np.max(RI[0])
        r = RI[1][np.logical_and(RI[0] >= w_min, RI[0] <= w_max)]
        w = RI[0][np.logical_and(RI[0] >= w_min, RI[0] <= w_max)]
        cval = scalarMap.to_rgba(i)
        if relative:
            ylabel = "I/I_max"
            r /= np.max(r)
        if peaks is None:
            peaks = signal.find_peaks(r[np.logical_and(w >= w_min, w <= w_max)
                                        ])[0]
            locations = np.take(w[np.logical_and(w >= w_min, w <= w_max)],
                                peaks)
            intensities = np.take(r[np.logical_and(w >= w_min, w <= w_max)],
                                  peaks)
        if legend:
            plt.plot(w, r, color=cval, label=ramanname[i])
        else:
            plt.plot(w, r, color=cval)
    for i, loc in enumerate(locations):
        if intensities[i] / np.max(intensities) > 0.05:
            plt.axvline(x=loc, color="grey", linestyle="--")
    plt.yscale(yscale)
    plt.minorticks_on()
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Raman intensity")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel(ylabel)
    if not relative:
        plt.yticks([])
    plt.savefig(figname, dpi=300)
    plt.clf()
