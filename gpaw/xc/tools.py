import numpy as np

from gpaw.xc import XC
from gpaw.utilities import unpack


def vxc(paw, xc=None):
    "Calculate XC-contribution to eigenvalues."
    
    ham = paw.hamiltonian
    dens = paw.density
    wfs = paw.wfs

    if xc is None:
        xc = ham.xc
    else:
        xc = XC(xc)

    if dens.nt_sg is None:
        dens.interpolate()

    thisisatest = not True
    
    # Calculate XC-potential:
    vxct_sg = ham.finegd.zeros(wfs.nspins)
    xc.calculate(dens.finegd, dens.nt_sg, vxct_sg)
    vxct_sG = ham.gd.empty(wfs.nspins)
    ham.restrict(vxct_sg, vxct_sG)
    if thisisatest:
        vxct_G[:] = 1
        
    # ... and PAW corrections:
    dvxc_asii = {}
    for a, D_sp in dens.D_asp.items():
        dvxc_sp = np.zeros_like(D_sp)
        wfs.setups[a].xc_correction.calculate(xc, D_sp, dvxc_sp)
        dvxc_asii[a] = [unpack(dvxc_p) for dvxc_p in dvxc_sp]
        if thisisatest:
            dvxc_asii[a] = [wfs.setups[a].dO_ii]

    vxc_un = np.empty((wfs.kd.mynks, wfs.bd.mynbands))
    for vxc_n, kpt in zip(vxc_un, wfs.kpt_u):
        for n, psit_G in enumerate(kpt.psit_nG):
            vxc_n[n] = wfs.gd.integrate((psit_G * psit_G.conj()).real,
                                        vxct_sG[kpt.s],
                                        global_integral=False)

        for a, dvxc_sii in dvxc_asii.items():
            P_ni = kpt.P_ani[a]
            vxc_n += (np.dot(P_ni, dvxc_sii[kpt.s]) *
                      P_ni.conj()).sum(1).real

    wfs.gd.comm.sum(vxc_un)

    return vxc_un
