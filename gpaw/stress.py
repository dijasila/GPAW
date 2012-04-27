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

    ppot_a = dens.ghat.stress_tensor_contribution(ham.vHt_q)
    p -= ham.epot
    for a, Q_L in dens.Q_aL.items():
        p += Q_L[0] * ppot_a[a][0, 0]

    pbar_a = ham.vbar.stress_tensor_contribution(dens.nt_sQ[0])
    p += sum(pbar_a.values())[0, 0] - 3 * ham.ebar

    vol = calc.atoms.get_volume() / units.Bohr**3
    sigma_vv = np.eye(3) * p / 3 / vol

    calc.text('Stress tensor:')
    for sigma_v in sigma_vv:
        calc.text('%10.6f %10.6f %10.6f' %
                  tuple(units.Hartree / units.Bohr**3 * sigma_v))

    return sigma_vv
