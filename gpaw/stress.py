import numpy as np
import ase.units as units

from gpaw.wavefunctions.pw import ft, PWWaveFunctions


def stress(calc):
    wfs = calc.wfs
    
    if not isinstance(wfs, PWWaveFunctions):
        raise NotImplementedError('Calculation of stress tensor is only ' +
                                  'implemented for plane-wave mode.')

    dens = calc.density
    ham = calc.hamiltonian

    p = calc.wfs.get_kinetic_stress().trace().real

    p += 3 * calc.hamiltonian.stress[0, 0]

    p -= ham.epot
    p_a = dens.ghat.stress_tensor_contribution(ham.vHt_q)
    for a, Q_L in dens.Q_aL.items():
        p += Q_L[0] * p_a[a][0, 0]

    p_a = ham.vbar.stress_tensor_contribution(dens.nt_sQ[0])
    p += sum(p_a.values())[0, 0] - 3 * ham.ebar

    #p_a = dens.nct.stress_tensor_contribution(ham.vt_Q)
    #p += sum(p_a.values())[0, 0]

    s = 0.0
    s0 = 0.0
    for kpt in wfs.kpt_u:
        p_ani = wfs.pt.stress_tensor_contribution(kpt.psit_nG, q=kpt.q)
        #print p_ani
        #print kpt.P_ani
        s += (p_ani[0][0,0] * kpt.P_ani[0][0,0] * (
                ham.dH_asp[0][0,0]
                - ham.setups[0].dO_ii[0,0] * kpt.eps_n[0]
                ) *
              kpt.f_n[0])
        s0 += (kpt.P_ani[0][0,0] * kpt.P_ani[0][0,0] * (
                ham.dH_asp[0][0,0]
                - ham.setups[0].dO_ii[0,0] * kpt.eps_n[0]
                ) *
              kpt.f_n[0])

    #print 2*(s-1.5*s0) / vol * units.Hartree / units.Bohr**3
    p +=2*(s-1.5*s0).real

    vol = calc.atoms.get_volume() / units.Bohr**3
    sigma_vv = np.eye(3) * p / 3 / vol

    calc.text('Stress tensor:')
    for sigma_v in sigma_vv:
        calc.text('%10.6f %10.6f %10.6f' %
                  tuple(units.Hartree / units.Bohr**3 * sigma_v))

    return sigma_vv
