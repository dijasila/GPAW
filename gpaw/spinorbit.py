import numpy as np
from ase.units import Ha, alpha#, Bohr
from gpaw.xc import XC

s = np.array([[0.0]])
p = np.zeros((3, 3), complex) # y, z, x
p[0, 1] = -1.0j
p[1, 0] = 1.0j
d = np.zeros((5, 5), complex) # xy, yz, z^2, xz, x^2-y^2
d[0, 3] = -1.0j
d[3, 0] = 1.0j
d[1, 2] = -3**0.5 * 1.0j
d[2, 1] = 3**0.5 * 1.0j
d[1, 4] = -1.0j
d[4, 1] = 1.0j
Lx_lmm = [s, p, d]

p = np.zeros((3, 3), complex) # y, z, x
p[1, 2] = -1.0j
p[2, 1] = 1.0j
d = np.zeros((5, 5), complex) # xy, yz, z^2, xz, x^2-y^2
d[0, 1] = 1.0j
d[1, 0] = -1.0j
d[2, 3] = -3**0.5 * 1.0j
d[3, 2] = 3**0.5 * 1.0j
d[3, 4] = -1.0j
d[4, 3] = 1.0j
Ly_lmm = [s, p, d]

p = np.zeros((3, 3), complex) # y, z, x
p[0, 2] = 1.0j
p[2, 0] = -1.0j
d = np.zeros((5, 5), complex) # xy, yz, z^2, xz, x^2-y^2
d[0, 4] = 2.0j
d[4, 0] = -2.0j
d[1, 3] = 1.0j
d[3, 1] = -1.0j
Lz_lmm = [s, p, d]

def get_radial_potential(calc, a, ai):
    """Calculates dV/dr / r fro the effective potential.
    Below f_g denotes dV/dr - minus the radial force"""

    rgd = a.xc_correction.rgd
    r_g = rgd.r_g
    #r_c = r_g[-1] / 2.0
    #r0 = np.argsort(np.abs(r_g - r_c))[0]
    #print r_c, r_g[r0]
    r_g[0] = 1.0e-12
    dr_g = rgd.dr_g
    Ns = calc.wfs.nspins

    D_sp = calc.density.D_asp[ai]
    B_pq = a.xc_correction.B_pqL[:, :, 0]
    n_qg = a.xc_correction.n_qg
    D_sq = np.dot(D_sp, B_pq)
    n_sg = np.dot(D_sq, n_qg) / (4 * np.pi)**0.5
    #import pylab as pl
    #pl.plot(r_g, n_sg[0])
    n_sg[:] += a.xc_correction.nc_g / Ns
    #pl.plot(r_g, n_sg[0])
    #pl.axis([None, 3.0, None, None])
    #pl.show()
    #n_sg = np.array([a.xc_correction.nc_g / Ns for s in range(Ns)])
    #n_sg[0,3*len(r_g)/4:] = 0.0
    #print a.Z, rgd.integrate(n_sg[0])#, r_g[-1] * Bohr

    # Coulomb force from nucleus
    fc_g = a.Z / r_g**2
    
    # Hartree force
    rho_g = 4 * np.pi * r_g**2 * dr_g * np.sum(n_sg, axis=0)
    fh_g = -np.array([np.sum(rho_g[:ig]) for ig in range(len(r_g))]) / r_g**2
    #print np.sum(rho_g)
    #fh_g -= 15.0/r_g**2
    #print fh_g[-20:]
    #vh_g = rgd.poisson(np.sum(n_sg, axis=0)) / r_g
    #fh_g = rgd.derivative(vh_g)
    #print fh_g[-20:]
    #print fc_g[-20:]
    #print
    # xc force
    xc = XC(calc.get_xc_functional())
    v_sg = np.zeros_like(n_sg)
    xc.calculate_spherical(a.xc_correction.rgd, n_sg, v_sg)
    fxc_sg = np.array([a.xc_correction.rgd.derivative(v_sg[s])
                       for s in range(Ns)])
    #fxc_sg = np.array([a.xc_correction.rgd.derivative(np.sum(v_sg, axis=0))
    #                   for s in range(Ns)])
    
    f_sg = np.tile(fc_g, (Ns, 1)) + np.tile(fh_g, (Ns, 1)) + fxc_sg
    #f_sg[0, r0:] = 0.0
    #print f_sg
    return f_sg[:] / r_g

def get_spinorbit_eigenvalues(calc, bands=None, return_spin=False,
                              return_wfs=False):
    
    if bands is None:
        bands = range(calc.get_number_of_bands())

    Na = len(calc.atoms)
    Nk = len(calc.get_ibz_k_points())
    Ns = calc.wfs.nspins
    Nn = len(bands)
    if Ns == 1:
        e_kn = [calc.get_eigenvalues(kpt=k)[bands] for k in range(Nk)]
        e_skn = np.array([e_kn, e_kn])
    else:
        e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)[bands]
                           for k in range(Nk)] for s in range(2)])

    # <phi_i|dV_adr / r * L_v|phi_j>
    dVL_asvii = []
    for ai in range(Na):
        a = calc.wfs.setups[ai]
        v_sg = get_radial_potential(calc, a, ai)
        Ng = len(v_sg[0])
        phi_jg = a.data.phi_jg

        dVL_svii = np.zeros((Ns, 3, a.ni, a.ni), complex)
        N1 = 0
        for j1, l1 in enumerate(a.l_j):
            Nm = 2 * l1 + 1
            N2 = 0
            for j2, l2 in enumerate(a.l_j):
                if l1 == l2:
                    f_sg = phi_jg[j1][:Ng] * v_sg[:] * phi_jg[j2][:Ng]
                    r_g = a.xc_correction.rgd.r_g
                    dr_g = a.xc_correction.rgd.dr_g
                    #I_s = np.sum(a.xc_correction.rgd.integrate(f_sg[:])*r_g**2*dr_g)
                    #I_s = np.sum(a.xc_correction.rgd.integrate(f_sg)*r_g**2*dr_g)
                    I_s = a.xc_correction.rgd.integrate(f_sg) / (4 * np.pi)
                    for s in range(Ns):
                        dVL_svii[s, 0, N1:N1+Nm, N2:N2+Nm] = Lx_lmm[l1] * I_s[s]
                        dVL_svii[s, 1, N1:N1+Nm, N2:N2+Nm] = Ly_lmm[l1] * I_s[s]
                        dVL_svii[s, 2, N1:N1+Nm, N2:N2+Nm] = Lz_lmm[l1] * I_s[s]
                else:
                    pass
                N2 += 2 * l2 + 1
            N1 += Nm
        dVL_asvii.append(dVL_svii)
    
    e_km = []
    if return_spin:
        #s_x = np.array([[0, 1.0], [1.0, 0]])
        #s_y = np.array([[0, -1.0j], [1.0j, 0]])
        #s_z = np.array([[1.0, 0], [0, -1.0]])
        s_km = []
    if return_wfs:
        v_knm = []
    # Hamiltonian with SO in KS basis
    for k in range(Nk):
        H_mm = np.zeros((2 * Nn, 2 * Nn), complex)
        H_mm[range(2*Nn)[::2], range(2*Nn)[::2]] = e_skn[0, k, :]
        H_mm[range(2*Nn)[1::2], range(2*Nn)[1::2]] = e_skn[1, k, :]
        for ai in range(Na):
            P_sni = [calc.wfs.kpt_u[k + s * Nk].P_ani[ai][bands]
                     for s in range(Ns)]
            dVL_svii = dVL_asvii[ai]
            if Ns == 1:
                P_ni = P_sni[0]
                Hso_nvn = np.dot(np.dot(P_ni.conj(), dVL_svii[0]), P_ni.T)
                Hso_nvn *= alpha**2 / 4.0 * Ha
                H_mm[::2, ::2] += Hso_nvn[:, 2, :]
                H_mm[1::2, 1::2] -= Hso_nvn[:, 2, :]
                H_mm[::2, 1::2] += Hso_nvn[:, 0, :] - 1.0j * Hso_nvn[:, 1, :]
                H_mm[1::2, ::2] += Hso_nvn[:, 0, :] + 1.0j * Hso_nvn[:, 1, :]
            else:
                P_ni = P_sni[0]
                Hso0_nvn = np.dot(np.dot(P_ni.conj(), dVL_svii[0]), P_ni.T)
                Hso0_nvn *= alpha**2 / 4.0 * Ha
                P_ni = P_sni[1]
                Hso1_nvn = np.dot(np.dot(P_ni.conj(), dVL_svii[1]), P_ni.T)
                Hso1_nvn *= alpha**2 / 4.0 * Ha
                H_mm[::2, ::2] += Hso0_nvn[:, 2, :]
                H_mm[1::2, 1::2] -= Hso1_nvn[:, 2, :]
                H_mm[::2, 1::2] += Hso1_nvn[:, 0, :] - 1.0j * Hso1_nvn[:, 1, :]
                H_mm[1::2, ::2] += Hso0_nvn[:, 0, :] + 1.0j * Hso0_nvn[:, 1, :]
                #H_mm = 0.5 * (H_mm + H_mm.T.conj())
        e_m, v_snm = np.linalg.eigh(H_mm)
        e_km.append(e_m)
        if return_wfs:
            v_knm.append(v_snm)
        if return_spin:
            s_m = np.sum(np.abs(v_snm[::2, :])**2, axis=0)
            s_m -= np.sum(np.abs(v_snm[1::2, :])**2, axis=0)
            s_km.append(s_m)
    
    if return_spin:
        if return_wfs:
            return np.array(e_km).T, np.array(s_km).T, v_knm
        else:
            return np.array(e_km).T, np.array(s_km).T
    else:
        if return_wfs:
            return np.array(e_km).T, v_knm
        else:
            return np.array(e_km).T
    

def get_spinorbit_correction(calc, bands=None,
                             return_spin=False, return_wfs=False):
    
    if bands is None:
        bands = range(calc.get_number_of_bands())

    Na = len(calc.atoms)
    Nk = len(calc.get_ibz_k_points())
    Ns = calc.get_number_of_spins()
    if Ns == 1:
        e_kn = [calc.get_eigenvalues(kpt=k)[bands] for k in range(Nk)]
        e_skn = np.array([e_kn, e_kn])
    else:
        e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)[bands]
                           for k in range(Nk)] for s in range(2)])
    e_snk = np.swapaxes(e_skn, 1, 2)
    dVL_avii = []
    a_Z = []
    for ai in range(Na):
        a = calc.wfs.setups[ai]
        a_Z.append(a.Z)
        r_g = a.rgd.r_g
        r_g[0] = 1.0e-6

        dVL_vii = np.zeros((3, a.ni, a.ni), complex)
        N1 = 0
        for j1, l1 in enumerate(a.l_j):
            N1m = 2 * l1 + 1
            N2 = 0
            for j2, l2 in enumerate(a.l_j):
                N2m = 2 * l2 + 1
                if l1 == l2:
                    f_g = a.data.phi_jg[j1].conj() * a.data.phi_jg[j2] / r_g**3
                    phi2 = a.rgd.integrate(f_g) / (4 * np.pi)
                    dVL_vii[0, N1:N1+N1m, N2:N2+N2m] = Lx_lmm[l1] * phi2
                    dVL_vii[1, N1:N1+N1m, N2:N2+N2m] = Ly_lmm[l1] * phi2
                    dVL_vii[2, N1:N1+N1m, N2:N2+N2m] = Lz_lmm[l1] * phi2
                else:
                    pass
                N2 += N2m
            N1 += N1m
        dVL_avii.append(dVL_vii)
    
    for ni, n in enumerate(bands):
        for k in range(Nk):
            if Ns == 1:
                P_ani = calc.wfs.kpt_u[k].P_ani
                dE_ss = np.zeros((2, 2), complex)
                for ai in range(Na):
                    P_ni = P_ani[ai]
                    dVL_vii = dVL_avii[ai]
                    dEa_v = np.dot(np.dot(P_ni[n].conj(), dVL_vii), P_ni[n])
                    dE_ss[0, 0] -= a_Z[ai] * dEa_v[2]
                    dE_ss[0, 1] -= a_Z[ai] * (dEa_v[0] - 1.0j * dEa_v[1])
                dE_ss[1, 0] = dE_ss[0, 1].conj()
                dE_ss[1, 1] = -dE_ss[0, 0]
                de_s, v_ss = np.linalg.eigh(dE_ss)
            else:
                P0_ani = calc.wfs.kpt_u[k].P_ani
                P1_ani = calc.wfs.kpt_u[Nk + k].P_ani
                dE0 = 0.0
                dE1 = 0.0
                for ai in range(Na):
                    P0_ni = P0_ani[ai]
                    P1_ni = P1_ani[ai]
                    dVL_vii = dVL_avii[ai]
                    dE0 -= a_Z[ai] * np.dot(np.dot(P0_ni[n].conj(),
                                                   dVL_vii[2]), P0_ni[n])
                    dE1 -= a_Z[ai] * np.dot(np.dot(P1_ni[n].conj(),
                                                   dVL_vii[2]), P1_ni[n])
                de_s = np.array([dE0, -dE1])
            e_snk[:, ni, k] += de_s.real * alpha**2 / 4.0 * Ha

    e_nk = np.reshape(np.swapaxes(e_snk, 0, 1), (2 * len(bands), Nk))
    
    return e_nk

def get_parity_eigenvalues(calc, spin_orbit=False, ik=0, tol=1.0e-3):
    
    Nn = calc.get_number_of_bands()
    vol = np.abs(np.linalg.det(calc.wfs.gd.cell_cv))
    psit_nG = np.array([calc.wfs.kpt_u[ik].psit_nG[n]
                        for n in range(Nn)])

    G_Gv = calc.wfs.pd.get_reciprocal_vectors(q=ik)

    if spin_orbit:
        e_nk, v_knm = get_spinorbit_eigenvalues(calc, return_wfs=True)
        psit_mG = np.dot(v_knm[ik][::2, ::2].T, psit_nG)
        psit_nG = psit_mG

    P_GG = np.zeros((len(G_Gv), len(G_Gv)), float)
    for i1, G1 in enumerate(G_Gv):
        for i2, G2 in enumerate(G_Gv):
            if np.dot(G1 + G2, G1 + G2) < 1.0e-6:
                P_GG[i1, i2] = 1
    ps = []
    for n in range(calc.get_number_of_bands()):
        psit_G = psit_nG[n]
        Ppsit_G = np.dot(P_GG, psit_G)
        if np.sum(np.abs(Ppsit_G - psit_G)) / vol < tol:
            p = 1
            ps.append(p)
        elif np.sum(np.abs(Ppsit_G + psit_G)) / vol < tol:
            p = -1
            ps.append(p)
        else:
            print('n =', n, 'is not a parity eigenvalue',
                  np.sum(np.abs(Ppsit_G + psit_G)) /
                  vol, np.sum(np.abs(Ppsit_G - psit_G)) / vol)

    return ps
