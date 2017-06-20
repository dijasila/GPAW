from __future__ import print_function
import numpy as np
from ase.units import Ha, alpha, Bohr
from gpaw.xc import XC


s = np.array([[0.0]])
p = np.zeros((3, 3), complex)  # y, z, x
p[0, 1] = -1.0j
p[1, 0] = 1.0j
d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
d[0, 3] = -1.0j
d[3, 0] = 1.0j
d[1, 2] = -3**0.5 * 1.0j
d[2, 1] = 3**0.5 * 1.0j
d[1, 4] = -1.0j
d[4, 1] = 1.0j
Lx_lmm = [s, p, d]

p = np.zeros((3, 3), complex)  # y, z, x
p[1, 2] = -1.0j
p[2, 1] = 1.0j
d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
d[0, 1] = 1.0j
d[1, 0] = -1.0j
d[2, 3] = -3**0.5 * 1.0j
d[3, 2] = 3**0.5 * 1.0j
d[3, 4] = -1.0j
d[4, 3] = 1.0j
Ly_lmm = [s, p, d]

p = np.zeros((3, 3), complex)  # y, z, x
p[0, 2] = 1.0j
p[2, 0] = -1.0j
d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
d[0, 4] = 2.0j
d[4, 0] = -2.0j
d[1, 3] = 1.0j
d[3, 1] = -1.0j
Lz_lmm = [s, p, d]


def get_radial_potential(calc, a, ai):
    """Calculates dV/dr / r for the effective potential.
    Below, f_g denotes dV/dr = minus the radial force"""

    rgd = a.xc_correction.rgd
    r_g = rgd.r_g
    r_g[0] = 1.0e-12
    dr_g = rgd.dr_g
    Ns = calc.wfs.nspins

    D_sp = calc.density.D_asp[ai]
    B_pq = a.xc_correction.B_pqL[:, :, 0]
    n_qg = a.xc_correction.n_qg
    D_sq = np.dot(D_sp, B_pq)
    n_sg = np.dot(D_sq, n_qg) / (4 * np.pi)**0.5
    n_sg[:] += a.xc_correction.nc_g / Ns

    # Coulomb force from nucleus
    fc_g = a.Z / r_g**2

    # Hartree force
    rho_g = 4 * np.pi * r_g**2 * dr_g * np.sum(n_sg, axis=0)
    fh_g = -np.array([np.sum(rho_g[:ig]) for ig in range(len(r_g))]) / r_g**2

    # xc force
    xc = XC(calc.get_xc_functional())
    v_sg = np.zeros_like(n_sg)
    xc.calculate_spherical(a.xc_correction.rgd, n_sg, v_sg)
    fxc_sg = np.array([a.xc_correction.rgd.derivative(v_sg[s])
                       for s in range(Ns)])
    fxc_g = np.sum(fxc_sg, axis=0) / Ns

    f_g = fc_g + fh_g + fxc_g

    return f_g / r_g


def get_spinorbit_eigenvalues(calc, bands=None, gw_kn=None, return_spin=False,
                              return_wfs=False, scale=1.0,
                              theta=0.0, phi=0.0):

    if bands is None:
        bands = np.arange(calc.get_number_of_bands())

    Na = len(calc.atoms)
    Nk = len(calc.get_ibz_k_points())
    Ns = calc.wfs.nspins
    Nn = len(bands)
    if Ns == 1:
        if gw_kn is None:
            e_kn = [calc.get_eigenvalues(kpt=k)[bands] for k in range(Nk)]
        else:
            assert Nk == len(gw_kn)
            assert Nn == len(gw_kn.T)
            e_kn = gw_kn
        e_skn = np.array([e_kn, e_kn])
    else:
        e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)[bands]
                           for k in range(Nk)] for s in range(2)])

    # <phi_i|dV_adr / r * L_v|phi_j>
    dVL_avii = []
    for ai in range(Na):
        a = calc.wfs.setups[ai]
        v_g = get_radial_potential(calc, a, ai)
        Ng = len(v_g)
        phi_jg = a.data.phi_jg

        # dVL_svii = np.zeros((Ns, 3, a.ni, a.ni), complex)
        dVL_vii = np.zeros((3, a.ni, a.ni), complex)
        N1 = 0
        for j1, l1 in enumerate(a.l_j):
            Nm = 2 * l1 + 1
            N2 = 0
            for j2, l2 in enumerate(a.l_j):
                if l1 == l2:
                    f_g = phi_jg[j1][:Ng] * v_g * phi_jg[j2][:Ng]
                    I = a.xc_correction.rgd.integrate(f_g) / (4 * np.pi)
                    dVL_vii[0, N1:N1 + Nm, N2:N2 + Nm] = I * Lx_lmm[l1]
                    dVL_vii[1, N1:N1 + Nm, N2:N2 + Nm] = I * Ly_lmm[l1]
                    dVL_vii[2, N1:N1 + Nm, N2:N2 + Nm] = I * Lz_lmm[l1]
                else:
                    pass
                N2 += 2 * l2 + 1
            N1 += Nm
        dVL_avii.append(dVL_vii)

    e_km = []
    if return_wfs:
        v_knm = []
    if return_spin:
        v_knm = []
        s_kvm = []

    # Hamiltonian with SO in KS basis
    # The even indices in H_mm are spin up along \hat n defined by \theta, phi
    # Basis change matrix for constructing Pauli matrices in \theta,\phi basis:
    #     \sigma_i^n = C^\dag\sigma_i C
    C_ss = np.array([[np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                      -np.sin(theta / 2) * np.exp(-1.0j * phi / 2)],
                     [np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                      np.cos(theta / 2) * np.exp(1.0j * phi / 2)]])
    sx_ss = np.array([[0, 1], [1, 0]], complex)
    sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
    sz_ss = np.array([[1, 0], [0, -1]], complex)
    sx_ss = C_ss.T.conj().dot(sx_ss).dot(C_ss)
    sy_ss = C_ss.T.conj().dot(sy_ss).dot(C_ss)
    sz_ss = C_ss.T.conj().dot(sz_ss).dot(C_ss)

    for k in range(Nk):
        # Evaluate H in a basis of S_z eigenstates
        H_mm = np.zeros((2 * Nn, 2 * Nn), complex)
        i1 = np.arange(0, 2 * Nn, 2)
        i2 = np.arange(1, 2 * Nn, 2)
        H_mm[i1, i1] = e_skn[0, k]
        H_mm[i2, i2] = e_skn[1, k]
        for ai in range(Na):
            Pt_sni = [calc.wfs.kpt_u[k + s * Nk].P_ani[ai][bands]
                      for s in range(Ns)]
            Ni = len(Pt_sni[0][0])
            P_sni = np.zeros((2, 2 * Nn, Ni), complex)
            dVL_vii = dVL_avii[ai] * scale * alpha**2 / 4.0 * Ha
            if Ns == 1:
                P_sni[0, ::2] = Pt_sni[0]
                P_sni[1, 1::2] = Pt_sni[0]
            else:
                P_sni[0, ::2] = Pt_sni[0]
                P_sni[1, 1::2] = Pt_sni[1]
            H_ssii = np.zeros((2, 2, Ni, Ni), complex)
            H_ssii[0, 0] = dVL_vii[2]
            H_ssii[0, 1] = dVL_vii[0] - 1.0j * dVL_vii[1]
            H_ssii[1, 0] = dVL_vii[0] + 1.0j * dVL_vii[1]
            H_ssii[1, 1] = -dVL_vii[2]

            # Tranform to theta, phi basis
            H_ssii = np.tensordot(C_ss, H_ssii, ([0, 1]))
            H_ssii = np.tensordot(C_ss.T.conj(), H_ssii, ([1, 1]))
            for s1 in range(2):
                for s2 in range(2):
                    H_ii = H_ssii[s1, s2]
                    P1_mi = P_sni[s1]
                    P2_mi = P_sni[s2]
                    H_mm += np.dot(np.dot(P1_mi.conj(), H_ii), P2_mi.T)

        e_m, v_snm = np.linalg.eigh(H_mm)
        e_km.append(e_m)
        if return_wfs or return_spin:
            v_knm.append(v_snm)
        if return_spin:
            sx_m = []
            sy_m = []
            sz_m = []
            for m in range(2 * Nn):
                v_sn = np.array([v_snm[::2, m], v_snm[1::2, m]])
                sx_m.append(np.trace(v_sn.T.conj().dot(sx_ss).dot(v_sn)))
                sy_m.append(np.trace(v_sn.T.conj().dot(sy_ss).dot(v_sn)))
                sz_m.append(np.trace(v_sn.T.conj().dot(sz_ss).dot(v_sn)))
            s_kvm.append([sx_m, sy_m, sz_m])

    if return_spin:
        if return_wfs:
            return np.array(e_km).T, np.array(s_kvm).real, v_knm
        else:
            return np.array(e_km).T, np.array(s_kvm).real
    else:
        if return_wfs:
            return np.array(e_km).T, v_knm
        else:
            return np.array(e_km).T


def set_calculator(calc, e_km, v_knm=None, width=None):
    from gpaw.occupations import FermiDirac
    from ase.units import Hartree

    if width is None:
        width = calc.occupations.width * Hartree
    calc.wfs.bd.nbands *= 2
    # calc.wfs.nspins = 1
    for kpt in calc.wfs.kpt_u:
        kpt.eps_n = e_km[kpt.k] / Hartree
        kpt.f_n = np.zeros_like(kpt.eps_n)
        kpt.weight /= 2
    ef = calc.occupations.fermilevel
    calc.occupations = FermiDirac(width)
    calc.occupations.nvalence = calc.wfs.setups.nvalence - calc.density.charge
    calc.occupations.fermilevel = ef
    calc.occupations.calculate_occupation_numbers(calc.wfs)
    for kpt in calc.wfs.kpt_u:
        kpt.f_n *= 2
        kpt.weight *= 2


def get_anisotropy(calc, theta=0.0, phi=0.0, nbands=None, width=None):
    """Calculates the sum of occupied spinorbit eigenvalues. Returns the result
    relative to the sum of eigenvalues without spinorbit coupling"""

    e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                       for k in range(len(calc.get_ibz_k_points()))]
                      for s in range(2)])
    e_kn = np.reshape(np.swapaxes(e_skn, 0, 1), (len(e_skn[0]),
                                                 2 * len(e_skn[0, 0])))
    e_kn = np.sort(e_kn, 1)
    if nbands is None:
        nbands = len(e_skn[0, 0])
    f_skn = np.array([[calc.get_occupation_numbers(kpt=k, spin=s)
                       for k in range(len(calc.get_ibz_k_points()))]
                      for s in range(2)])
    f_kn = np.reshape(np.swapaxes(f_skn, 0, 1), (len(f_skn[0]),
                                                 2 * len(f_skn[0, 0])))
    f_kn = np.sort(f_kn, 1)[:, ::-1]
    E = np.sum(e_kn * f_kn)

    e_mk = get_spinorbit_eigenvalues(calc, theta=theta, phi=phi,
                                     bands=range(nbands))
    set_calculator(calc, e_mk.T, width=width)
    f_km = np.array([calc.get_occupation_numbers(kpt=k)
                     for k in range(len(calc.get_ibz_k_points()))])
    E_so = np.sum(e_mk.T * f_km)
    return E_so - E


def get_magnetic_moments(calc, theta=0.0, phi=0.0, nbands=None):
    """Calculates the magnetic moments inside all PAW spheres"""

    from gpaw.wannier90 import get_spinorbit_projections
    from gpaw.utilities import unpack

    if nbands is None:
        nbands = calc.get_number_of_bands()
    Nk = len(calc.get_ibz_k_points())

    C_ss = np.array([[np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
                      -np.sin(theta / 2) * np.exp(-1.0j * phi / 2)],
                     [np.sin(theta / 2) * np.exp(1.0j * phi / 2),
                      np.cos(theta / 2) * np.exp(1.0j * phi / 2)]])

    m_av = []
    e_mk, v_knm = get_spinorbit_eigenvalues(calc,
                                            theta=theta,
                                            phi=phi,
                                            return_wfs=True,
                                            bands=range(nbands))

    for a in range(len(calc.atoms)):
        N0_p = calc.density.setups[a].N0_p
        N0_ij = unpack(N0_p)

        Dx_ij = np.zeros_like(N0_ij, complex)
        Dy_ij = np.zeros_like(N0_ij, complex)
        Dz_ij = np.zeros_like(N0_ij, complex)
        tot = 0.0
        for ik in range(Nk):
            P_ami = get_spinorbit_projections(calc, ik, v_knm[ik])

            P_smi = np.array([P_ami[a][:, ::2], P_ami[a][:, 1::2]])
            P_smi = np.dot(C_ss, np.swapaxes(P_smi, 0, 1))

            P0_mi = P_smi[0]
            P1_mi = P_smi[1]

            f0_n = calc.get_occupation_numbers(kpt=ik, spin=0)
            f1_n = calc.get_occupation_numbers(kpt=ik, spin=1)
            f_n = np.zeros(2 * len(f0_n))
            f_n[::2] = f0_n
            f_n[1::2] = f1_n
            f_nn = np.diagflat(f_n[:len(P0_mi)])
            tot += f_nn[0, 0]

            Dx_ij += P0_mi.conj().T.dot(f_nn).dot(P1_mi)
            Dx_ij += P1_mi.conj().T.dot(f_nn).dot(P0_mi)
            Dy_ij -= 1.0j * P0_mi.conj().T.dot(f_nn).dot(P1_mi)
            Dy_ij += 1.0j * P1_mi.conj().T.dot(f_nn).dot(P0_mi)
            Dz_ij += P0_mi.conj().T.dot(f_nn).dot(P0_mi)
            Dz_ij -= P1_mi.conj().T.dot(f_nn).dot(P1_mi)

        mx = np.sum(N0_ij * Dx_ij).real
        my = np.sum(N0_ij * Dy_ij).real
        mz = np.sum(N0_ij * Dz_ij).real

        m_av.append([mx, my, mz])

    return m_av


def get_parity_eigenvalues(calc, ik=0, spin_orbit=False, bands=None, Nv=None,
                           inversion_center=[0, 0, 0], deg_tol=1.0e-6,
                           tol=1.0e-6):
    """Calculates parity eigenvalues at time-reversal invariant k-points.
    Only works in plane wave mode.
    """

    kpt_c = calc.get_ibz_k_points()[ik]
    if Nv is None:
        Nv = int(calc.get_number_of_electrons() / 2)

    if bands is None:
        bands = range(calc.get_number_of_bands())

    # Find degenerate subspaces
    eig_n = calc.get_eigenvalues(kpt=ik)[bands]
    e_in = []
    used_n = []
    for n1, e1 in enumerate(eig_n):
        if n1 not in used_n:
            n_n = []
            for n2, e2 in enumerate(eig_n):
                if np.abs(e1 - e2) < deg_tol:
                    n_n.append(n2)
                    used_n.append(n2)
            e_in.append(n_n)

    print()
    print(' Inversion center at: %s' % inversion_center)
    print(' Calculating inversion eigenvalues at k = %s' % kpt_c)
    print()

    center_v = np.array(inversion_center) / Bohr
    G_Gv = calc.wfs.pd.get_reciprocal_vectors(q=ik, add_q=True)

    psit_nG = np.array([calc.wfs.kpt_u[ik].psit_nG[n]
                        for n in bands])
    if spin_orbit:
        e_nk, v_knm = get_spinorbit_eigenvalues(calc, return_wfs=True,
                                                bands=bands)
        psit0_mG = np.dot(v_knm[ik][::2].T, psit_nG)
        psit1_mG = np.dot(v_knm[ik][1::2].T, psit_nG)
    for n in range(len(bands)):
        psit_nG[n] /= (np.sum(np.abs(psit_nG[n])**2))**0.5
    if spin_orbit:
        for n in range(2 * len(bands)):
            A = np.sum(np.abs(psit0_mG[n])**2) + np.sum(np.abs(psit1_mG[n])**2)
            psit0_mG[n] /= A**0.5
            psit1_mG[n] /= A**0.5

    P_GG = np.ones((len(G_Gv), len(G_Gv)), float)
    for iG, G_v in enumerate(G_Gv):
        P_GG[iG] -= ((G_Gv[:] + G_v).round(6)).any(axis=1)
    assert (P_GG == P_GG.T).all()

    phase_G = np.exp(-2.0j * np.dot(G_Gv, center_v))

    p_n = []
    print('n   P_n')
    for n_n in e_in:
        if spin_orbit:
            # The dimension of parity matrix is doubled with spinorbit
            m_m = [2 * n_n[0] + i for i in range(2 * len(n_n))]
            Ppsit0_mG = np.dot(P_GG, psit0_mG[m_m].T).T
            Ppsit0_mG[:] *= phase_G
            Ppsit1_mG = np.dot(P_GG, psit1_mG[m_m].T).T
            Ppsit1_mG[:] *= phase_G
            P_nn = np.dot(psit0_mG[m_m].conj(), np.array(Ppsit0_mG).T)
            P_nn += np.dot(psit1_mG[m_m].conj(), np.array(Ppsit1_mG).T)
        else:
            Ppsit_nG = np.dot(P_GG, psit_nG[n_n].T).T
            Ppsit_nG[:] *= phase_G
            P_nn = np.dot(psit_nG[n_n].conj(), np.array(Ppsit_nG).T)
        P_eig = np.linalg.eigh(P_nn)[0]
        if np.allclose(np.abs(P_eig), 1, tol):
            P_n = np.sign(P_eig).astype(int).tolist()
            if spin_orbit:
                # Only include one of the degenerate pair of eigenvalues
                Pm = np.sign(P_eig).tolist().count(-1)
                Pp = np.sign(P_eig).tolist().count(1)
                P_n = Pm // 2 * [-1] + Pp // 2 * [1]
            print('%s: %s' % (str(n_n)[1:-1], str(P_n)[1:-1]))
            p_n += P_n
        else:
            print('  %s are not parity eigenstates' % n_n)
            print('     P_n: %s' % P_eig)
            print('     e_n: %s' % eig_n[n_n])
            p_n += [0 for n in n_n]

    return np.ravel(p_n)
