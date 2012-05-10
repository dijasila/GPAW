import numpy as np
import ase.units as units

from gpaw.utilities import unpack
from gpaw.wavefunctions.pw import ft, PWWaveFunctions


def stress(calc):
    wfs = calc.wfs
    
    if not isinstance(wfs, PWWaveFunctions):
        raise NotImplementedError('Calculation of stress tensor is only ' +
                                  'implemented for plane-wave mode.')

    dens = calc.density
    ham = calc.hamiltonian

    p = calc.wfs.get_kinetic_stress().trace().real

    p += ham.xc.stress_tensor_contribution(dens.nt_sg)
    # tilde-n_c contribution to dsigma/depsilon is missing!

    p -= ham.epot
    p_aL = dens.ghat.stress_tensor_contribution(ham.vHt_q)
    for a, Q_L in dens.Q_aL.items():
        p += np.dot(Q_L, p_aL[a])

    p_a = ham.vbar.stress_tensor_contribution(dens.nt_sQ[0])
    p += sum(p_a.values())[0] - 3 * ham.ebar

    p_a = dens.nct.stress_tensor_contribution(ham.vt_Q)
    p += sum(p_a.values())[0]

    s = 0.0
    s0 = 0.0
    for kpt in wfs.kpt_u:
        p_ani = wfs.pt.stress_tensor_contribution(kpt.psit_nG, q=kpt.q)
        for a, p_ni in p_ani.items():
            P_ni = kpt.P_ani[a]
            Pf_ni = P_ni * kpt.f_n[:, None]
            dH_ii = unpack(ham.dH_asp[a][kpt.s])
            dS_ii = ham.setups[a].dO_ii
            a_ni = (np.dot(Pf_ni, dH_ii) -
                    np.dot(Pf_ni * kpt.eps_n[:, None], dS_ii))
            s += np.vdot(p_ni, a_ni)
            s0 += np.vdot(P_ni, a_ni)

    p += 2 * (s - 1.5 * s0).real

    vol = calc.atoms.get_volume() / units.Bohr**3
    sigma_vv = np.eye(3) * p / 3 / vol

    calc.text('Stress tensor:')
    for sigma_v in sigma_vv:
        calc.text('%10.6f %10.6f %10.6f' %
                  tuple(units.Hartree / units.Bohr**3 * sigma_v))

    return sigma_vv
