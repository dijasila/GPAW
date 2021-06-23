"""
Calculates Raman matrices and intensities.

Momentum matrices are not symmetric. The first index is the bra, the other the
ket.

i -> j -> m -> n
i, n are valence; j, m are conduction, also i=n in the end
see https://doi.org/10.1038/s41467-020-16529-6
"""

import numpy as np
from ase.phonons import Phonons
from ase.units import invcm


def lorentzian(w, gamma):
    l = 0.5 * gamma / (np.pi * (w**2 + 0.25 * gamma**2))
    return l


def gaussian(w, sigma):
    g = 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-w**2 / (2 * sigma**2))
    return g


def calculate_raman(atoms, calc, w_in, d_i, d_o, resonant_only=False,
                    gamma_l=0.1, phononname='phonon', momname='mom_skvnm.npy',
                    elphname='gsqklnn.npy'):
    """
    Calculates the first order Raman tensor contribution for a given
    polarization.

    Parameters
    ----------
    atoms: Atoms
        Primitive cell geometry
    calc: GPAW
        Converged ground state calculation
    w_in: float
        Laser energy in eV
    d_i: int
        Incoming polarization
    d_o: int
        Outgoing polarization
    gamma_l: float
        Line broadening in eV
    phononname: str
        Name of phonon cache
    momname: str
        Name of momentum file
    elphname: str
        Name of electron-phonon file
    """

    print("Calculating Raman spectrum: Laser frequency = {} eV".format(w_in))

    ph = Phonons(atoms=atoms, name=phononname, supercell=(1, 1, 1))
    ph.read()
    w_ph = np.array(ph.band_structure([[0, 0, 0]])[0])  # in eV
    if w_ph.dtype == "complex128":
        w_ph = w_ph.real
    assert max(w_ph) < 1.  # else not eV units
    w_max = int(np.round(np.max(w_ph) / invcm + 100, -1))  # in rcm
    # NOTE: Should make grid-spacing an input variable
    ngrid = w_max + 1
    w_cm = np.linspace(0., w_max, num=ngrid)  # Defined in cm^-1
    w = w_cm * invcm  # eV (Raman shift?)

    # Exclude 3 accustic phonons + anything imaginary (<10cm^-1)
    l_min = max(np.where(w_ph / invcm < 30.)[0].size, 3)
    w_ph = w_ph[l_min:]
    ieta = complex(0, gamma_l)
    nmodes = len(w_ph)

    # Load files
    mom_sk = np.load(momname)  # [:,k,:,:]dim, k
    elph_sk = np.load(elphname)[:, 0, :, l_min:]  # [s,q=0,k,l,n,m]

    nspins = elph_sk.shape[0]
    nk = elph_sk.shape[1]

    # ab is in and out polarization
    # l is the phonon mode and w is the raman shift
    raman_lw = np.zeros((len(w_ph), len(w)), dtype=complex)

    print("Evaluating Raman sum")
    opt = 'optimal'  # mode for np.einsum. not sure what is fastest

    # note: valence is ket in ij and bra in mn
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
            for wi in range(len(w)):
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
            for wi in range(len(w)):
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
            for wi in range(len(w)):
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
            for wi in range(len(w)):
                t6_mn = (f_vc * mom_dnn[0, :nv, nc:] / (-w_in - w_ph[l] + w[wi]
                                                        - E_vc)).T
                term6_lw[l, wi] += np.einsum('sj,jm,ms', t6_ij,
                                             t6_xx[nc:, nc:], t6_mn,
                                             optimize=opt)
                term6_lw[l, wi] -= np.einsum('is,ni,sn', t6_ij,
                                             t6_xx[:nv, :nv], t6_mn,
                                             optimize=opt)
        return term6_lw

    for s in range(nspins):
        E_kn = calc.band_structure().todict()["energies"][s]
        for k in range(nk):
            print("For k = {}".format(k))

            if (calc.symmetry.time_reversal and not
                np.allclose(calc.wfs.kd.ibzk_kc[k], [0., 0., 0.])):
                add_time_reversed = True
            else:
                add_time_reversed = False

            this_lw = np.zeros((len(w_ph), len(w)), dtype=complex)

            weight = calc.wfs.collect_auxiliary("weight", k, s)
            f_n = calc.wfs.collect_occupations(k, s)
            f_n = f_n / weight
            elph_lnn = elph_sk[s, k]
            E_n = E_kn[k]

            # limit sums to relevant bands, partially occupied bands are a pain
            # So, in principal, partially occupied bands can be initial and
            # final states, but we need to restrict to a positive deltaE if we
            # allow this.
            vs = np.where(f_n >= 0.1)[0]
            cs = np.where(f_n < 0.9)[0]
            nv = max(vs) + 1  # VBM+1 index
            nc = min(cs)  # CBM index

            # give momentum matrix for d_i and d_o directions
            mom_dnn = np.ascontiguousarray(mom_sk[s, k, [d_i, d_o]])
            assert mom_dnn.shape[0] == 2

            # f * (1-f) term
            f_vc = np.outer(f_n[vs], 1. - f_n[cs])
            # E-E term
            E_vc = np.empty((len(vs), len(cs)), dtype=complex)
            for n in vs:
                E_vc[n] = E_n[cs] - E_n[n] + ieta
                # set weights for negative energy transitions zero
                neg = np.where(E_vc[n] <= 0.)[0]
                f_vc[n, neg] = 0.

            # Term 1
            term1_l = _term1(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv)
            print("Term1: ", np.max(np.abs(term1_l)))
            this_lw += term1_l[:, None]

            if not resonant_only:
                term2_lw = _term2(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv)
                print("Term2: ", np.max(np.abs(term2_lw)))
                this_lw += term2_lw

                term3_lw = _term3(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv)
                print("Term3: ", np.max(np.abs(term3_lw)))
                this_lw += term3_lw

                term4_lw = _term4(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv)
                print("Term4: ", np.max(np.abs(term4_lw)))
                this_lw += term4_lw

                term5_l = _term5(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv)
                print("Term5: ", np.max(np.abs(term5_l)))
                this_lw += term5_l[:, None]

                term6_lw = _term6(f_vc, E_vc, mom_dnn, elph_lnn, nc, nv)
                print("Term6: ", np.max(np.abs(term6_lw)))
                this_lw += term6_lw

            # At the moment we only allow no symmetry, so all k-points same
            # weight, or time_reversal only. In the later case r -> 2*Re(r)
            # because gdd-> (gdd)^* for k-> -k
            if add_time_reversed:
                raman_lw += 2. * this_lw.real
            else:
                raman_lw += this_lw

    for l in range(nmodes):
        print("Phonon {} with energy = {}: {}".format(l, w_ph[l] / invcm,
              np.max(np.abs(raman_lw[l]))))

    raman = np.vstack((w_cm, raman_lw))
    np.save("vib_frequencies.npy", w_ph)
    xyz = 'xyz'
    np.save("Rlab_{}{}.npy".format(xyz[d_i], xyz[d_o]), raman)


def calculate_raman_tensor(atoms, calc, w_in, d_i, d_o, resonant_only=False,
                    gamma_l=0.1, phononname='phonon', momname='mom_skvnm.npy',
                    elphname='gsqklnn.npy'):
    for i in range(3):
        for j in range(3):
            calculate_raman(atoms, calc, w_in, d_i=i, d_o=j,
                            resonant_only=resonant_only, gamma_l=gamma_l, phononname=phononname, momname=momname,
                            elphname=elphname)


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
