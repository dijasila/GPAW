import numpy as np
import ase.units as units

from gpaw.utilities import unpack
from gpaw.wavefunctions.pw import PWWaveFunctions


def stress(calc):
    wfs = calc.wfs
    
    if not isinstance(wfs, PWWaveFunctions):
        raise NotImplementedError('Calculation of stress tensor is only ' +
                                  'implemented for plane-wave mode.')

    dens = calc.density
    ham = calc.hamiltonian

    p = calc.wfs.get_kinetic_stress(calc.wfs.symmetry).real
    print p
    p += ham.xc.stress_tensor_contribution(dens.nt_sg)
    print p
    # tilde-n_c contribution to dsigma/depsilon is missing!

    p -= np.eye(3) * (ham.epot / 3)
    p += dens.ghat.stress_tensor_contribution(ham.vHt_q, dens.Q_aL)
    print p

    p -= np.eye(3) * ham.ebar
    p += ham.vbar.stress_tensor_contribution(dens.nt_sQ[0])
    print p

    p += dens.nct.stress_tensor_contribution(ham.vt_Q)
    print p

    s = 0.0
    s0 = 0.0
    for kpt in wfs.kpt_u:
        a_ani = {}
        for a, P_ni in kpt.P_ani.items():
            Pf_ni = P_ni * kpt.f_n[:, None]
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            dS_ii = ham.setups[a].dO_ii
            a_ni = (np.dot(Pf_ni, dH_ii) -
                    np.dot(Pf_ni * kpt.eps_n[:, None], dS_ii))
            s0 += np.vdot(P_ni, a_ni)
            a_ani[a] = a_ni.conj()
        s += wfs.pt.stress_tensor_contribution(kpt.psit_nG, a_ani,
                                               q=kpt.q)
    p += 2 * (s - 0.5 * s0 * np.eye(3)).real
    print p

    vol = calc.atoms.get_volume() / units.Bohr**3
    sigma_vv = 0.5 / vol * (p + p.T)

    calc.text('Stress tensor:')
    for sigma_v in sigma_vv:
        calc.text('%10.6f %10.6f %10.6f' %
                  tuple(units.Hartree / units.Bohr**3 * sigma_v))

    return sigma_vv
