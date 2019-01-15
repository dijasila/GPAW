import numpy as np
from numpy.lib.scimath import sqrt as csqrt
import ase.units as units
from ase.units import Bohr, Hartree


def phononbuildingblock(atoms, Z_avv, freqs, modes):
    # Simple function for calculating phonon building block
    me = 1822.888
    m_a = atoms.get_masses() * me

    phbb = {'Z_avv': Z_avv,
            'freqs': freqs,
            'modes': modes,
            'masses': m_a,
            'cell': atoms.cell}

    return phbb


def dopedsemiconductor(path, effectivemass, doping, temperature):
    # Chi0 for zero temperature
    def chi0T0(q, w, me, mup):
        kf = np.sqrt(2 * me * mup)
        N = kf**2 / (2 * np.pi)
        vf = kf / me
        z = q / (2 * kf)
        u = w / (q * vf)
        G = N / (me * z * (q**2) * (vf**2))
        chi0T0_qwm = np.zeros((len(q), len(w[0]), len(mup[0, 0])), complex)
        chi0T0_qwm[:, :, :] = 2 * z

        # Condition 1
        mask1 = (np.absolute(z - u.real) >= 1)
        chi0T0_qwm -= (mask1 * ((z - u.real) / np.absolute(z - u.real))
                       * csqrt((z - u)**2 - 1))

        # Condition 2
        mask2 = (np.absolute(z + u.real) >= 1)
        chi0T0_qwm -= (mask2 * ((z + u.real) / np.absolute(z + u.real))
                       * csqrt((z + u)**2 - 1))

        # Condition 3
        mask3 = (np.absolute(z - u.real) < 1)
        chi0T0_qwm += 1j * mask3 * csqrt(1 - (z - u)**2)

        # Condition 4
        mask4 = (np.absolute(z + u.real) < 1)
        chi0T0_qwm -= 1j * mask4 * csqrt(1 - (z + u)**2)

        chi0T0_qwm *= G
        return - q**2 * chi0T0_qwm

    # Sum Argument
    def arg(q, w, me, T, mu, mup):
        argument = (chi0T0(q, w, me, mup) /
                    (4 * T * (np.cosh((mu - mup) / (2 * T)))**2))
        return argument

    # Polarizability
    def P(q_q, w_w, me, T, mu, mupmax, N):
        mup_m = np.linspace(10**(-5), mupmax, N)
        q_qwm = q_q[:, None, None]
        w_qwm = w_w[None, :, None]
        mup_qwm = mup_m[None, None, :]
        return np.trapz(arg(q_qwm, w_qwm, me, T, mu, mup_qwm), mup_qwm, axis=2)

    # Polarizability in the relaxation time approximation
    def Pgamma(q_q, w_w, me=None, efermi=None, T=0.0,
               mupmax=None, N=1000, gamma=1e-2):
        assert efermi is not None, print('You have to set a fermi energy!')
        assert me is not None, \
            print('You have to set an effective electron mass!')

        gamma = gamma / Hartree
        efermi = efermi / Hartree
        T = T / Hartree

        if T / efermi > 1e-7:
            # Temperature dependent chemical potential
            mu = T * np.log(np.exp(efermi / T) - 1)
        else:
            mu = efermi
            T = 0

        if mupmax is None:
            mupmax = 20 * T + mu * (mu > 0)

        a = 1j * gamma / w_w
        iw_w = w_w + 1j * gamma
        P0 = P(q_q, np.array([0j]), me, T, mu, mupmax, N)
        P1 = P(q_q, iw_w, me, T, mu, mupmax, N)

        return ((1 + a) * P1 / (1 + a * P1 / P0))

    # Reading old file
    block = np.load(path)
    chiM_qw = block['chiM_qw']
    q_q = block['q_abs']
    omega_w = block['omega_w']

    V_q = 2 * np.pi / q_q
    chi0Mnew_qw = Pgamma(q_q, omega_w, me=effectivemass,
                         efermi=doping, T=temperature)
    chi0Mnew_qw += chiM_qw / (1 + V_q[:, None] * chiM_qw)
    dopedchiM_qw = chi0Mnew_qw / (1 - chi0Mnew_qw * V_q[:, None])

    data = {'isotropic_q': True,
            'q_abs': q_q,
            'omega_w': omega_w,
            'chiM_qw': dopedchiM_qw,
            'chiD_qw': block['chiD_qw'],
            'z': block['z'],
            'drhoM_qz': block['drhoM_qz'],
            'drhoD_qz': block['drhoD_qz']}

    return data
    

def GrapheneBB(path, doping):
    Ef = doping / Hartree
    c = 137.0
    vf = c / 300
    kf = Ef / vf
    tau = 1 / (1e-3 / Hartree)

    # Auxiliary functions
    def F(x):
        return x * ((x**2 - 1)**0.5) - np.arccosh(x)

    def C(x):
        return x * ((1 - x**2)**0.5) - np.arccos(x)

    prefactor = vf**(-2)

    # Real part of the Polarizability
    def P(q_q, w_w):
        q_qw = q_q[:, None]
        w_qw = w_w[None, :]
        a = -2 * vf * kf / np.pi
        b = 1 / (4 * np.pi) * \
            (vf * q_qw)**2 / (w_qw**2 - (vf * q_qw)**2)**0.5
        b1 = 1 / (4 * np.pi) * \
            (vf * q_qw)**2 / ((vf * q_qw)**2 - w_qw**2)**0.5
        F1 = F((2 * vf * kf + w_qw) / (vf * q_qw))
        F2 = F((w_qw - 2 * vf * kf) / (vf * q_qw))
        F3 = F((2 * vf * kf - w_qw) / (vf * q_qw))
        C1 = C((2 * vf * kf + w_qw) / (vf * q_qw))
        C2 = C((2 * vf * kf - w_qw) / (vf * q_qw))

        Pol_qw = np.zeros((len(q_q), len(w_w)), complex)
        Pol_qw[:, :] = a

        # Region I
        mask1 = (np.real(w_qw) >= vf * q_qw) * \
                (vf * q_qw + np.real(w_qw) <= 2 * vf * kf)
        Pol_qw += mask1 * b * (F1 - F3)

        # Region II
        mask2 = (np.real(w_qw) >= vf * q_qw) * \
                (vf * q_qw + np.real(w_qw) >= 2 * vf * kf) * \
                (np.real(w_qw) - vf * q_qw <= 2 * vf * kf)
        Pol_qw += mask2 * b * (F1 + 1j * C2)

        # Region III
        mask3 = np.real(w_qw) - vf * q_qw >= 2 * vf * kf
        Pol_qw += mask3 * b * (F1 - F2 - 1j) * np.pi

        # Region IV
        mask4 = (vf * q_qw >= np.real(w_qw)) * \
                (vf * q_qw + np.real(w_qw) <= 2 * vf * kf)
        Pol_qw += mask4 * 1j * b1 * (F3 - F1)

        # Region V
        mask5 = (vf * q_qw >= np.real(w_qw)) * \
                (vf * q_qw + np.real(w_qw) >= 2 * vf * kf) * \
                (vf * q_qw - 2 * vf * kf <= np.real(w_qw))
        Pol_qw += mask5 * b1 * (C2 - 1j * F1)

        # Region VI
        mask6 = vf * q_qw - 2 * vf * kf >= np.real(w_qw)
        Pol_qw += mask6 * b1 * (C1 + C2)

        Pol_qw *= prefactor

        assert np.allclose(mask1 + mask2 + mask3 + mask4 + mask5 + mask6,
                           1)
        return Pol_qw

    # Relaxation time approximation for the Polarizability
    def Pgamma(q_q, w_w):
        a = 1j * w_w * tau
        iw_w = w_w + 1j / tau
        P0 = P(q_q, np.array([0j]))
        P1 = P(q_q, iw_w)
        return ((1 - a) * P0 * P1 / (P1 - a * P0))

    block = np.load(path)
    z = block['z']
    q_q = block['q_abs']
    omega_w = block['omega_w']
    nw = len(omega_w)
    nq = len(q_q)
    chi0M_qw = np.zeros([nq, nw], dtype=complex)
    chiM_qw = np.zeros([nq, nw], dtype=complex)

    # for iq, q in enumerate(q_q):
    #     print(iq, len(q_q))
    #     for iw, w in enumerate(omega_w):
    V_q = 2 * np.pi / q_q
    chi0M_qw = Pgamma(q_q, omega_w) + (-1.3 / Bohr) * q_q[:, None]**2
    chiM_qw = chi0M_qw / (1 - chi0M_qw * V_q[:, None])

    # Renormalize monopole density
    drhoM_qz = block['drhoM_qz']

    data = {'isotropic_q': True,
            'q_abs': q_q,
            'omega_w': omega_w,
            'chiM_qw': chiM_qw,
            'chiD_qw': block['chiD_qw'],
            'z': z,
            'drhoM_qz': drhoM_qz,
            'drhoD_qz': block['drhoD_qz']}

    return data


def phonon_polarizability(bb, Z_avv, freqs, modes, m_a, cell_cv):
    Hartree = units.Hartree

    # Make new bb
    bb = dict(np.load(bb))
    chiM_qw = bb['chiM_qw']
    chiD_qw = bb['chiD_qw']
    q_abs = bb['q_abs']
    omega_w = bb['omega_w']
    
    # Get phonons at q=0
    Z_vx = Z_avv.swapaxes(0, 1).reshape((3, -1))
    f2_w, D_xw = (freqs[0] / Hartree)**2, modes[0]  # Pick out q=0

    alpha_wvv = np.zeros((len(omega_w), 3, 3), dtype=complex)
    m_x = np.repeat(m_a, 3)**0.5
    gamma = 0.1e-3 / Hartree
    for f2, D_x in zip(f2_w, D_xw.T):
        if f2 < (1e-3 / Hartree)**2:
            continue
        DM_x = D_x / m_x
        Z_v = np.dot(Z_vx, DM_x)

        alpha_wvv += (np.outer(Z_v, Z_v)[np.newaxis] /
                      ((f2 - omega_w**2) -
                       1j * gamma * omega_w)[:, np.newaxis, np.newaxis])

    vol = abs(np.linalg.det(cell_cv)) / units.Bohr**3
    L = np.abs(cell_cv[2, 2] / Bohr)
    alpha_wvv *= 1 / vol * L

    Vm_qw = 2 * np.pi / q_abs[:, None]
    Vd_qw = 2 * np.pi
    newbb = dict(bb)
    chi0Mnew_qw = -(q_abs[:, None])**2 * alpha_wvv[:, 0, 0][np.newaxis]
    chi0Dnew_qw = -(np.ones(len(q_abs))[:, None] *
                    alpha_wvv[:, 2, 2][np.newaxis])
    chi0Mnew_qw += chiM_qw / (1 + Vm_qw * chiM_qw)
    chi0Dnew_qw += chiD_qw / (1 + Vd_qw * chiD_qw)
    chiMnew_qw = chi0Mnew_qw / (1 - Vm_qw * chi0Mnew_qw)
    chiDnew_qw = chi0Dnew_qw / (1 - Vd_qw * chi0Dnew_qw)
    rhoMnew_qz = bb['drhoM_qz']
    rhoDnew_qz = bb['drhoD_qz']
    
    newbb['chiM_qw'] = chiMnew_qw
    newbb['chiD_qw'] = chiDnew_qw
    newbb['omega_w'] = omega_w
    newbb['q_abs'] = q_abs
    newbb['drhoM_qz'] = rhoMnew_qz
    newbb['drhoD_qz'] = rhoDnew_qz
    newbb['isotropic_q'] = True
    
    return newbb
