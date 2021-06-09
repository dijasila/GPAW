"""
Calculates Raman matrices and intensities.

Momentum matrices are not symmetric. The first index is the bra, the other the
ket.
"""

import numpy as np
from ase.phonons import Phonons


def L(w, gamma=10 / 8065.544):
    # Lorentzian
    lor = 0.5 * gamma / (np.pi * ((w.real)**2 + 0.25 * gamma**2))
    return lor


def gaussian(w, sigma):
    g = 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-w**2 / (2 * sigma**2))
    return g

def _term1(f_vc, E_vc, mom_dnn, elph_lnn, vs, cs):
    term1_l = np.zeros((elph_lnn.shape[0]), dtype=complex)
    print(mom_dnn[0, cs][:,vs])
    print(mom_dnn[0, vs][:,cs])

    t1_ij = (f_vc * mom_vnn[d_i, vs, cs] / (w_in - E_vc))
    for l in range(nmodes):
        t1_xx = elph_lnn[l]
        t1_mn = fvc.T * mom_vnn[d_o][ cs, vs] / (w_in - w_ph[l] - Enm.T)
        print(t_1_mn)
        term1_l[l] += np.einsum('sj,jm,ms', t1_ij, t1_xx[nc:, nc:], t1_mn,
                                optimize=opt)
        term1_l[l] -= np.einsum('is,ni,sn', t1_ij, t1_xx[:nv, :nv], t1_mn,
                                optimize=opt)
    return term1_l

def calculate_raman(atoms, calc, w_in, d_i, d_o, resonant_only=False,
                    ramanname=None, momname=None, basename=None, gamma_l=0.1):
    """
    Calculates the first order Raman spectrum

    Input:
        atoms           ASE atoms object used for the phonon calculation
        resonant_only   If False, use all Fermi terms
        ramanname       Suffix for the raman.npy file
        momname         Suffix for the momentumfile
        basename        Suffix for the gs.gpw and gqklnn.npy files
        w_in, gamma_l   Laser energy, broadening factor for the electron
                        energies (in eV)
        d_i, d_o        Laser polarization in, out (0, 1, 2 for x, y, z)
    Output:
        RI.npy          Numpy array containing the raman spectre
    """

    print("Calculating Raman spectrum: Laser frequency = {} eV".format(w_in))

    cm = 1. / 8065.544  # cm^-1 to eV

    ph = Phonons(atoms=atoms, name='elph', supercell=(1, 1, 1))
    ph.read()
    w_ph = np.array(ph.band_structure([[0, 0, 0]])[0]) # in eV
    if w_ph.dtype == "complex128":
        w_ph = w_ph.real
    assert max(w_ph) < 1. # else not eV units
    w_max = int(np.round(np.max(w_ph) / cm + 100, -1)) # in rcm
    # NOTE: Should make grid-spacing an input variable
    ngrid = w_max + 1
    w_cm = np.linspace(0., w_max, num=ngrid)  # Defined in cm^-1
    w = w_cm * cm  # eV (Raman shift?)

    # Exclude 3 accustic phonons + anything imaginary (<10cm^-1)
    l_min = max(np.where(w_ph / cm < 30.)[0].size, 3)
    w_ph = w_ph[l_min:]
    ieta = complex(0, gamma_l)
    nmodes = len(w_ph)

    # Load files
    if momname is None:
        mom_sk = np.load("mom_skvnm.npy")  # [:,k,:,:]dim, k
    else:
        mom_sk = np.load("mom_skvnm_{}.npy".format(momname))
    if basename is None:
        elph_sk = np.load("gsqklnn.npy")[:, 0, :, l_min:]  # [s,q=0,k,l,n,m]
    else:
        elph_sk = np.load("gsqklnn_{}.npy".format(basename))[:, 0, :, l_min:]

    nspins = elph_sk.shape[0]
    nk = elph_sk.shape[1]

    # ab is in and out polarization
    # l is the phonon mode and w is the raman shift
    raman_lw = np.zeros((len(w_ph), len(w)), dtype=complex)

    print("Evaluating Raman sum")
    opt = 'optimal'  # mode for np.einsum. not sure what is fastest



    def _term2(f_n, E_n, mom_vnn, elph_lnn, nc, nv):
        term2_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t2_ij = (f_n[:nv, None] * (1. - f_n[None, nc:]) *
                 mom_vnn[d_i, :nv, nc:] /
                 (w_in - (E_n[None, nc:] - E_n[:nv, None]) + ieta))
        t2_xx = mom_vnn[d_o]
        for l in range(nmodes):
            t2_wmn = (f_n[None, None, :nv] * (1. - f_n[None, nc:, None]) *
                      elph_lnn[l][None, nc:, :nv] /
                      (w[:, None, None] - (E_n[None, None, :nv] -
                                           E_n[None, nc:, None]) + ieta))
            term2_lw[l] += np.einsum('sj,jm,wms->w', t2_ij, t2_xx[nc:, nc:],
                                     t2_wmn, optimize=opt)
            term2_lw[l] -= np.einsum('is,ni,wsn->w', t2_ij, t2_xx[:nv, :nv],
                                     t2_wmn, optimize=opt)
        return term2_lw

    def _term3(f_n, E_n, mom_vnn, elph_lnn, nc, nv):
        term3_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t3_wij = (f_n[None, :nv, None] * (1. - f_n[None, None, nc:]) *
                  mom_vnn[d_o][None, :nv, nc:] /
                  (-w_in + w[:, None, None] - (E_n[None, None, nc:] -
                                               E_n[None, :nv, None]) + ieta))
        for l in range(nmodes):
            t3_xx = elph_lnn[l]
            t3_wmn = (f_n[None, None, :nv] * (1. - f_n[None, nc:, None]) *
                      mom_vnn[d_i][None, nc:, :nv] /
                      (-w_in - w_ph[l] + w[:, None, None] -
                       (E_n[None, None, :nv] - E_n[None, nc:, None]) + ieta))
            term3_lw[l] += np.einsum('wsj,jm,wms->w', t3_wij, t3_xx[nc:, nc:],
                                     t3_wmn, optimize=opt)
            term3_lw[l] -= np.einsum('wis,ni,wsn->w', t3_wij, t3_xx[:nv, :nv],
                                     t3_wmn, optimize=opt)
        return term3_lw

    def _term4(f_n, E_n, mom_vnn, elph_lnn, nc, nv):
        term4_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t4_wij = (f_n[None, :nv, None] * (1. - f_n[None, None, nc:]) *
                  mom_vnn[d_o][None, :nv, nc:] /
                  (-w_in + w[:, None, None] - (E_n[None, None, nc:] -
                                               E_n[None, :nv, None]) + ieta))
        t4_xx = mom_vnn[d_i]
        for l in range(nmodes):
            t4_wmn = (f_n[None, None, :nv] * (1. - f_n[None, nc:, None]) *
                      elph_lnn[l][None, nc:, :nv] /
                      (w[:, None, None] - (E_n[None, None, :nv] -
                                           E_n[None, nc:, None]) + ieta))
            term4_lw[l] += np.einsum('wsj,jm,wms->w', t4_wij, t4_xx[nc:, nc:],
                                     t4_wmn, optimize=opt)
            term4_lw[l] -= np.einsum('wis,ni,wsn->w', t4_wij, t4_xx[:nv, :nv],
                                     t4_wmn, optimize=opt)
        return term4_lw

    def _term5(f_n, E_n, mom_vnn, elph_lnn, nc, nv):
        term5_l = np.zeros((nmodes), dtype=complex)
        t5_xx = mom_vnn[d_i]
        for l in range(nmodes):
            t5_ij = (f_n[:nv, None] * (1. - f_n[None, nc:]) *
                     elph_lnn[l, :nv, nc:] /
                     (-w_ph[l] - (E_n[None, nc:] - E_n[:nv, None]) + ieta))
            t5_mn = (f_n[None, :nv] * (1. - f_n[nc:, None]) *
                     mom_vnn[d_o, nc:, :nv] /
                     (w_in - w_ph[l] - (E_n[None, :nv] - E_n[nc:, None]) +
                      ieta))
            term5_l[l] += np.einsum('sj,jm,ms', t5_ij, t5_xx[nc:, nc:], t5_mn,
                                    optimize=opt)
            term5_l[l] -= np.einsum('is,ni,sn', t5_ij, t5_xx[:nv, :nv], t5_mn,
                                    optimize=opt)
        return term5_l

    def _term6(f_n, E_n, mom_vnn, elph_lnn, nc, nv):
        term6_lw = np.zeros((nmodes, ngrid), dtype=complex)
        t6_xx = mom_vnn[d_o]
        for l in range(nmodes):
            t6_ij = (f_n[:nv, None] * (1. - f_n[None, nc:]) *
                     elph_lnn[l, :nv, nc:] /
                     (-w_ph[l] - (E_n[None, nc:] - E_n[:nv, None]) + ieta))
            t6_wmn = (f_n[None, None, :nv] * (1. - f_n[None, nc:, None]) *
                      mom_vnn[d_i][None, nc:, :nv] /
                      (-w_in - w_ph[l] + w[:, None, None] -
                       (E_n[None, None, :nv] - E_n[None, nc:, None]) + ieta))
            term6_lw[l] += np.einsum('sj,jm,wms->w', t6_ij, t6_xx[nc:, nc:],
                                     t6_wmn, optimize=opt)
            term6_lw[l] -= np.einsum('is,ni,wsn->w', t6_ij, t6_xx[:nv, :nv],
                                     t6_wmn, optimize=opt)
        return term6_lw

    for s in range(nspins):
        E_kn = calc.band_structure().todict()["energies"][s]
        for k in range(nk):
            print("For k = {}".format(k))

            # XXX: Need to rethink weights and stuff, as symmetry not allowed
            # at the moment.
            weight = calc.wfs.collect_auxiliary("weight", k, s)
            f_n = calc.wfs.collect_occupations(k, s)
            f_n = f_n / weight
            elph_lnn = weight * elph_sk[s, k]
            # give momentum matrix for d_i and d_o directions
            mom_dnn = mom_sk[s, k+1, [d_i, d_o]]
            assert mom_dnn.shape[0] == 2
            E_n = E_kn[k]

            # limit sums to relevant bands, partially occupied bands are a pain
            # So, in principal, partially occupied bands can be initial and final
            # states, but we need to restrict to a positive deltaE if we allow
            # this.
            vs = np.where(f_n >= 0.1)[0]
            cs = np.where(f_n < 0.9)[0]

            # for TESTING need different shapes for v and c
            print(cs, vs)
            cs = cs[:-1]
            print(cs)


            # f * (1-f) term
            f_vc = np.outer(f_n[vs], 1. - f_n[cs])
            # E-E term
            E_vc = np.empty((len(vs), len(cs)), dtype=complex)
            for n in vs:
                E_vc[n] = E_n[cs] - E_n[n] + ieta
                # set weights for negative energy transitions zero
                neg = np.where(E_vc[n] <= 0.)[0]
                f_vc[n, neg] = 0.
            # i -> j -> m -> n
            # i, n are valence; j, m are conduction, also i=n in the end
            # see https://doi.org/10.1038/s41467-020-16529-6

            # Term 1
            term1_l = _term1(f_vc, E_vc, mom_dnn, elph_lnn, vs, cs)
            print("Term1: ", np.max(np.abs(term1_l)))
            raman_lw += term1_l[:, None] * weight

            if not resonant_only:
                term2_lw = _term2(f_n, E_n, mom_vnn, elph_lnn, nc, nv)
                print("Term2: ", np.max(np.abs(term2_lw)))
                raman_lw += term2_lw

                term3_lw = _term3(f_n, E_n, mom_vnn, elph_lnn, nc, nv)
                print("Term3: ", np.max(np.abs(term3_lw)))
                raman_lw += term3_lw

                term4_lw = _term4(f_n, E_n, mom_vnn, elph_lnn, nc, nv)
                print("Term4: ", np.max(np.abs(term4_lw)))
                raman_lw += term4_lw

                term5_l = _term5(f_n, E_n, mom_vnn, elph_lnn, nc, nv)
                print("Term5: ", np.max(np.abs(term5_l)))
                raman_lw += term5_l[:, None]

                term6_lw = _term6(f_n, E_n, mom_vnn, elph_lnn, nc, nv)
                print("Term6: ", np.max(np.abs(term6_lw)))
                raman_lw += term6_lw

    for l in range(nmodes):
        print("Phonon {} with energy = {}: {}".format(l, w_ph[l] / cm,
              np.max(np.abs(raman_lw[l]))))

    raman = np.vstack((w_cm, raman_lw))
    np.save("vib_frequencies.npy", w_ph)
    xyz = 'xyz'
    if ramanname is None:
        np.save("Rlab_{}{}.npy".format(xyz[d_i], xyz[d_o]), raman)
    else:
        np.save("Rlab_{}{}_{}.npy".format(xyz[d_i], xyz[d_o], ramanname),
                raman)


def calculate_raman_tensor(atoms, calc, resonant_only=True, ramanname=None,
                           momname=None, basename=None, w_in=2.54066,
                           gamma_l=0.2):
    for i in range(3):
        for j in range(3):
            calculate_raman(atoms, calc, resonant_only, ramanname,
                            momname, basename, w_in, gamma_l, d_i=i, d_o=j)


def calculate_raman_intensity(d_i, d_o, ramanname=None, T=300):
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
    """
        Plots a given Raman spectrum

        Input:
            yscale: Linear or logarithmic yscale
            figname: Name of the generated figure
            relative: Scale to the highest peak
            w_min, w_max: The plotting range wrt the Raman shift
            ramanname: Suffix used for the file containing the Raman spectrum

        Output:
            ramanname: image containing the Raman spectrum.

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
