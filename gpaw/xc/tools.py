import numpy as np
from ase.units import Hartree

from gpaw.xc import XC
from gpaw.utilities import unpack


def vxc(gs, xc=None, coredensity=True):
    """Calculate XC-contribution to eigenvalues."""

    ham = gs.hamiltonian
    dens = gs.density

    if xc is None:
        xc = ham.xc
    elif isinstance(xc, str):
        xc = XC(xc)

    if dens.nt_sg is None:
        dens.interpolate_pseudo_density()

    thisisatest = not True

    if xc.orbital_dependent:
        gs.get_xc_difference(xc)

    # Calculate XC-potential:
    vxct_sg = ham.finegd.zeros(gs.nspins)
    xc.calculate(dens.finegd, dens.nt_sg, vxct_sg)
    vxct_sG = ham.restrict_and_collect(vxct_sg)
    if thisisatest:
        vxct_sG[:] = 1

    # ... and PAW corrections:
    dvxc_asii = {}
    for a, D_sp in dens.D_asp.items():
        dvxc_sp = np.zeros_like(D_sp)
        xc.calculate_paw_correction(gs.setups[a], D_sp, dvxc_sp, a=a,
                                    addcoredensity=coredensity)
        dvxc_asii[a] = [unpack(dvxc_p) for dvxc_p in dvxc_sp]
        if thisisatest:
            dvxc_asii[a] = [gs.setups[a].dO_ii]

    vxc_un = np.empty((gs.kd.mynk * gs.nspins, gs.bd.mynbands))
    for u, vxc_n in enumerate(vxc_un):
        kpt = gs.kpt_u[u]
        vxct_G = vxct_sG[kpt.s]
        for n in range(gs.bd.mynbands):
            psit_G = gs.get_wave_function_array(u, n)
            vxc_n[n] = gs.gd.integrate((psit_G * psit_G.conj()).real,
                                       vxct_G, global_integral=False)

        for a, dvxc_sii in dvxc_asii.items():
            P_ni = kpt.P_ani[a]
            vxc_n += (np.dot(P_ni, dvxc_sii[kpt.s]) *
                      P_ni.conj()).sum(1).real

    gs.gd.comm.sum(vxc_un)
    vxc_skn = gs.kd.collect(vxc_un, broadcast=True)

    if xc.orbital_dependent:
        vxc_skn += xc.exx_skn

    return gs.bd.collect(vxc_skn.T.copy(), broadcast=True).T * Hartree
