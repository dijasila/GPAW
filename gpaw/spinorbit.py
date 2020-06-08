import warnings
from typing import Union, List, TYPE_CHECKING, Dict, Tuple, Any
from pathlib import Path

import numpy as np
from ase.units import Ha, alpha, Bohr

from gpaw.mpi import send, receive
from gpaw.utilities.ibz2bz import construct_symmetry_operators
from gpaw.setup import Setup

if TYPE_CHECKING:
    from gpaw import GPAW

_L_vlmm: List[List[np.ndarray]] = []  # see get_L_vlmm() below

# typehints:
ArrayND = Any
Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND


def soc_eigenstates(calc: Union[GPAW, str, Path],
                    bands: List[int] = None,
                    myeig_skn: Array3D = None,
                    scale: float = 1.0,
                    theta: float = 0.0,
                    phi: float = 0.0,
                    return_wfs: bool = False) -> Dict[str, ArrayND]:
    """Calculate SOC eigenstates.

    Parameters:
        calc: Calculator
            GPAW calculator or path to gpw-file.
        bands: list of ints
            List of band indices for which to calculate soc.
        myeig_skn: (ns, nk, nb)-shaped ndarray
            Use these eigenvalues instead of those from calc.get_eigenvalues().
        return_wfs: bool
            Flag for returning wave functions.
        scale: float
            Scale the spinorbit coupling by this amount.
        theta: float
            Angle in radians.
        phi: float
            Angle in radians.

    Returns a dict containing the keys: 'e_km', 's_kvm' and possibly 'v_kmm'.
    """

    from gpaw import GPAW

    if not isinstance(calc, GPAW):
        calc = GPAW(calc, txt=None)

    if bands is None:
        bands = np.arange(calc.get_number_of_bands())

    # <phi_i|dV_adr / r * L_v|phi_j>
    dVL_avii = {a: soc(calc.wfs.setups[a],
                       calc.hamiltonian.xc,
                       D_sp) * scale * Ha
                for a, D_sp in calc.density.D_asp.items()}

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
    s_vss = [C_ss.T.conj().dot(sx_ss).dot(C_ss),
             C_ss.T.conj().dot(sy_ss).dot(C_ss),
             C_ss.T.conj().dot(sz_ss).dot(C_ss)]

    kd = calc.wfs.kd

    e_km = np.zeros((kd.nbzkpts, 2 * len(bands)))
    s_kvm = np.zeros((kd.nbzkpts, 3, 2 * len(bands)))
    if return_wfs:
        v_kmm = np.zeros((kd.nbzkpts, 2 * len(bands), 2 * len(bands)), complex)

    transform = SymmetryTransformer(calc).trasnsform_projections

    for i, (e_sn, P_snI) in distribute_ibz_k_points(calc, bands).items():
        for K, k in enumerate(kd.bz2ibz_k):
            if k != i:
                continue
            if myeig_skn is not None:
                e_sn = myeig_skn[:, i]
            tP_snI = transform(K, P_snI)
            e_m, s_vm, v_mm = _soc_eig(e_sn, tP_snI, dVL_avii, s_vss, C_ss)
            e_km[K] = e_m
            s_kvm[K] = s_vm
            if return_wfs:
                v_kmm[K] = v_mm

    calc.world.sum(e_km)
    calc.world.sum(s_kvm)

    results = {'e_km': e_km,
               's_kvm': s_kvm}

    if return_wfs:
        calc.world.sum(v_kmm)
        results['v_kmm'] = v_kmm

    return results


def soc(a: Setup, xc, D_sp: Array2D) -> Array3D:
    """<phi_i|dV_adr / r * L_v|phi_j>"""
    v_g = get_radial_potential(a, xc, D_sp)
    Ng = len(v_g)
    phi_jg = a.data.phi_jg

    Lx_lmm, Ly_lmm, Lz_lmm = get_L_vlmm()

    dVL_vii = np.zeros((3, a.ni, a.ni), complex)
    N1 = 0
    for j1, l1 in enumerate(a.l_j):
        Nm = 2 * l1 + 1
        N2 = 0
        for j2, l2 in enumerate(a.l_j):
            if l1 == l2:
                f_g = phi_jg[j1][:Ng] * v_g * phi_jg[j2][:Ng]
                c = a.xc_correction.rgd.integrate(f_g) / (4 * np.pi)
                dVL_vii[0, N1:N1 + Nm, N2:N2 + Nm] = c * Lx_lmm[l1]
                dVL_vii[1, N1:N1 + Nm, N2:N2 + Nm] = c * Ly_lmm[l1]
                dVL_vii[2, N1:N1 + Nm, N2:N2 + Nm] = c * Lz_lmm[l1]
            else:
                pass
            N2 += 2 * l2 + 1
        N1 += Nm
    return dVL_vii * alpha**2 / 4.0


def get_radial_potential(a: Setup, xc, D_sp: Array2D) -> Array1D:
    """Calculates (dV/dr)/r for the effective potential.
    Below, f_g denotes dV/dr = minus the radial force"""

    rgd = a.xc_correction.rgd
    r_g = rgd.r_g.copy()
    r_g[0] = 1.0e-12
    dr_g = rgd.dr_g

    B_pq = a.xc_correction.B_pqL[:, :, 0]
    n_qg = a.xc_correction.n_qg
    D_sq = np.dot(D_sp, B_pq)
    n_sg = np.dot(D_sq, n_qg) / (4 * np.pi)**0.5
    Ns = len(D_sp)
    if Ns == 4:
        Ns = 1
    n_sg[:Ns] += a.xc_correction.nc_g / Ns

    # Coulomb force from nucleus
    fc_g = a.Z / r_g**2

    # Hartree force
    rho_g = 4 * np.pi * r_g**2 * dr_g * np.sum(n_sg, axis=0)
    fh_g = -np.array([np.sum(rho_g[:ig]) for ig in range(len(r_g))]) / r_g**2

    f_g = fc_g + fh_g

    # xc force
    if xc.name != 'GLLBSC':
        v_sg = np.zeros_like(n_sg)
        xc.calculate_spherical(a.xc_correction.rgd, n_sg, v_sg)
        fxc_g = np.mean([a.xc_correction.rgd.derivative(v_g) for v_g in v_sg],
                        axis=0)
        f_g += fxc_g

    return f_g / r_g


class SymmetryTransformer:
    """Transforms PAW projections from IBZ to BZ k-point."""
    def __init__(self, calc):
        self.wfs = calc.wfs
        self.spos_ac = calc.spos_ac
        self.I_a = [0]
        I = 0
        for setup in calc.wfs.setups:
            I += setup.R_sii.shape[2]
            self.I_a.append(I)

    def trasnsform_projections(self,
                               bz_index: int,
                               P_snI: Array3D) -> Array3D:
        """Transform to bz_index."""
        a_a, U_aii, time_rev = construct_symmetry_operators(
            self.wfs, self.spos_ac, bz_index)
        tP_snI = np.empty_like(P_snI, dtype=complex)
        I_a = self.I_a
        a = 0
        for b, U_ii in zip(a_a, U_aii):
            P_sni = np.dot(P_snI[:, :, I_a[b]:I_a[b + 1]], U_ii)
            if time_rev:
                P_sni = P_sni.conj()
            tP_snI[:, :, I_a[a]:I_a[a + 1]] = P_sni
            a += 1
        return tP_snI


KPT = Tuple[Array2D, Array3D]  # eigs, projections


def distribute_ibz_k_points(calc: GPAW,
                            bands: List[int]) -> Dict[int, KPT]:
    """Distrubite eigenvalues and projections across all CPU's.

    Returns dict mapping IBZ index to a (eig_sn, P_snI) tuple.
    """
    assert calc.density.collinear
    world = calc.world
    mykpts = {}
    for i in range(calc.wfs.kd.nibzkpts):
        eps0_n = calc.get_eigenvalues(i, 0)
        P0_nI = calc.wfs.collect_projections(i, 0)
        if calc.wfs.nspins == 1:
            eps1_n = eps0_n
            P1_nI = P0_nI
        else:
            eps1_n = calc.get_eigenvalues(i, 1)
            P1_nI = calc.wfs.collect_projections(i, 1)
        r = i % world.size
        if world.rank == 0:
            data = (np.array([eps0_n[bands], eps1_n[bands]]),
                    np.array([P0_nI[bands], P1_nI[bands]]))
            if r != 0:
                send(data, r, world)
        elif world.rank == r:
            data = receive(0, world)
        if r == world.rank:
            mykpts[i] = data
    return mykpts


def _soc_eig(e_sn: Array2D,
             P_snI: Array3D,
             dVL_avii: Dict[int, Array3D],
             s_vss: Array3D,
             C_ss: Array2D) -> Tuple[Array1D, Array2D, Array2D]:
    """Evaluate H in a basis of S_z eigenstates."""
    Nn = len(e_sn[0])
    H_mm = np.zeros((2 * Nn, 2 * Nn), complex)
    i1 = np.arange(0, 2 * Nn, 2)
    i2 = np.arange(1, 2 * Nn, 2)
    H_mm[i1, i1] = e_sn[0]
    H_mm[i2, i2] = e_sn[1]
    I1 = 0
    for a, dVL_vii in dVL_avii.items():
        Ni = dVL_vii.shape[2]
        I2 = I1 + Ni
        Pt_sni = P_snI[:, :, I1:I2]
        P_sni = np.zeros((2, 2 * Nn, Ni), complex)
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
        I1 = I2

    e_m, v_snm = np.linalg.eigh(H_mm)

    sx_m = []
    sy_m = []
    sz_m = []
    for m in range(2 * Nn):
        v_sn = np.array([v_snm[::2, m], v_snm[1::2, m]])
        sx_m.append(np.trace(v_sn.T.conj().dot(s_vss[0]).dot(v_sn)))
        sy_m.append(np.trace(v_sn.T.conj().dot(s_vss[1]).dot(v_sn)))
        sz_m.append(np.trace(v_sn.T.conj().dot(s_vss[2]).dot(v_sn)))

    return e_m, np.array([sx_m, sy_m, sz_m]).real, v_snm


def get_spinorbit_eigenvalues(calc, bands=None, gw_kn=None,
                              return_spin=False,
                              return_wfs=False,
                              scale=1.0,
                              theta=0.0, phi=0.0):
    warnings.warn('Please use soc_eigenstates() instead.')

    results = soc_eigenstates(calc, bands, gw_kn, scale, theta, phi,
                              return_wfs)
    values = [results['e_km'].T]
    if return_spin:
        values.append(results['s_kvm'])
    if return_wfs:
        values.append(results['v_kmm'])
    return tuple(values) if len(values) > 1 else values[0]


def set_calculator(calc, e_km, v_knm=None, width=None):
    from gpaw.occupations import FermiDirac
    from ase.units import Hartree

    noncollinear = not calc.density.collinear

    if width is None:
        width = calc.occupations.width * Hartree
    if not noncollinear:
        calc.wfs.bd.nbands *= 2
    # calc.wfs.nspins = 1
    for kpt in calc.wfs.kpt_u:
        kpt.eps_n = e_km[kpt.k] / Hartree
        kpt.f_n = np.zeros_like(kpt.eps_n)
        if not noncollinear:
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

    Ns = calc.wfs.nspins
    noncollinear = not calc.density.collinear

    e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                       for k in range(len(calc.get_ibz_k_points()))]
                      for s in range(Ns)])
    e_kn = np.reshape(np.swapaxes(e_skn, 0, 1), (len(e_skn[0]),
                                                 Ns * len(e_skn[0, 0])))
    e_kn = np.sort(e_kn, 1)
    if nbands is None:
        nbands = len(e_skn[0, 0])
    f_skn = np.array([[calc.get_occupation_numbers(kpt=k, spin=s)
                       for k in range(len(calc.get_ibz_k_points()))]
                      for s in range(Ns)])
    f_kn = np.reshape(np.swapaxes(f_skn, 0, 1), (len(f_skn[0]),
                                                 Ns * len(f_skn[0, 0])))
    f_kn = np.sort(f_kn, 1)[:, ::-1]
    if noncollinear:
        f_kn *= 2

    E = np.sum(e_kn * f_kn)

    from gpaw.occupations import occupation_numbers
    e_km = soc_eigenstates(calc, theta=theta, phi=phi,
                           bands=range(nbands))['e_km']
    if width is None:
        width = calc.occupations.width * Ha
    if width == 0.0:
        width = 1.e-6
    weight_k = calc.get_k_point_weights() / 2
    ne = calc.wfs.setups.nvalence - calc.density.charge
    f_km = occupation_numbers({'name': 'fermi-dirac', 'width': width},
                              np.array([e_km]),
                              weight_k=weight_k,
                              nelectrons=ne)[0][0]
    E_so = np.sum(e_km * f_km)
    return E_so - E


def get_spinorbit_projections(calc, ik, v_nm):
    # For spinors the number of projectors and bands are doubled
    Na = len(calc.atoms)
    Nk = len(calc.get_ibz_k_points())
    Ns = calc.wfs.nspins

    v0_mn = v_nm[::2].T
    v1_mn = v_nm[1::2].T

    P_ani = {}
    for ia in range(Na):
        P0_ni = calc.wfs.kpt_u[ik].P_ani[ia]
        P1_ni = calc.wfs.kpt_u[(Ns - 1) * Nk + ik].P_ani[ia]

        P0_mi = np.dot(v0_mn, P0_ni)
        P1_mi = np.dot(v1_mn, P1_ni)
        P_mi = np.zeros((len(P0_mi), 2 * len(P0_mi[0])), complex)
        P_mi[:, ::2] = P0_mi
        P_mi[:, 1::2] = P1_mi
        P_ani[ia] = P_mi

    return P_ani


def get_spinorbit_wavefunctions(calc, ik, v_nm):
    # For spinors the number of bands is doubled and a spin dimension is added
    Ns = calc.wfs.nspins
    Nn = calc.wfs.bd.nbands

    v0_mn = v_nm[::2].T
    v1_mn = v_nm[1::2].T

    u0_nG = np.array([calc.wfs.get_wave_function_array(n, ik, 0)
                      for n in range(Nn)])
    u1_nG = np.array([calc.wfs.get_wave_function_array(n, ik, (Ns - 1))
                      for n in range(Nn)])
    u0_mG = np.swapaxes(np.dot(v0_mn, np.swapaxes(u0_nG, 0, 2)), 1, 2)
    u1_mG = np.swapaxes(np.dot(v1_mn, np.swapaxes(u1_nG, 0, 2)), 1, 2)
    u_mG = np.zeros((len(u0_mG),
                     2,
                     len(u0_mG[0]),
                     len(u0_mG[0, 0]),
                     len(u0_mG[0, 0, 0])), complex)
    u_mG[:, 0] = u0_mG
    u_mG[:, 1] = u1_mG

    return u_mG


def get_magnetic_moments(calc, theta=0.0, phi=0.0, nbands=None, width=None):
    """Calculates the magnetic moments inside all PAW spheres"""

    from gpaw.utilities import unpack

    if nbands is None:
        nbands = calc.get_number_of_bands()
    Nk = len(calc.get_ibz_k_points())

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

    states = soc_eigenstates(calc,
                             theta=theta,
                             phi=phi,
                             return_wfs=True,
                             bands=range(nbands))
    e_km = states['e_km']
    v_knm = states['v_kmm']

    from gpaw.occupations import occupation_numbers
    if width is None:
        width = calc.occupations.width * Ha
    if width == 0.0:
        width = 1.e-6
    weight_k = calc.get_k_point_weights() / 2
    ne = calc.wfs.setups.nvalence - calc.density.charge
    f_km = occupation_numbers({'name': 'fermi-dirac', 'width': width},
                              e_km[np.newaxis],
                              weight_k=weight_k,
                              nelectrons=ne)[0][0]

    m_v = np.zeros(3, complex)
    for ik in range(Nk):
        ut0_nG = np.array([calc.wfs.get_wave_function_array(n, ik, 0)
                           for n in range(nbands)])
        ut1_nG = np.array([calc.wfs.get_wave_function_array(n, ik, 1)
                           for n in range(nbands)])
        mocc = np.where(f_km[ik] * Nk - 1.0e-6 < 0.0)[0][0]
        for m in range(mocc + 1):
            f = f_km[ik, m]
            ut0_G = np.dot(v_knm[ik][::2, m], np.swapaxes(ut0_nG, 0, 2))
            ut1_G = np.dot(v_knm[ik][1::2, m], np.swapaxes(ut1_nG, 0, 2))
            ut_sG = np.array([ut0_G, ut1_G])

            mx_G = np.zeros(np.shape(ut0_G), complex)
            my_G = np.zeros(np.shape(ut0_G), complex)
            mz_G = np.zeros(np.shape(ut0_G), complex)
            for s1 in range(2):
                for s2 in range(2):
                    mx_G += ut_sG[s1].conj() * sx_ss[s1, s2] * ut_sG[s2]
                    my_G += ut_sG[s1].conj() * sy_ss[s1, s2] * ut_sG[s2]
                    mz_G += ut_sG[s1].conj() * sz_ss[s1, s2] * ut_sG[s2]
            m_v += calc.wfs.gd.integrate(np.array([mx_G, my_G, mz_G])) * f

    m_av = []
    for a in range(len(calc.atoms)):
        N0_p = calc.density.setups[a].N0_p.copy()
        N0_ij = unpack(N0_p)
        Dx_ij = np.zeros_like(N0_ij, complex)
        Dy_ij = np.zeros_like(N0_ij, complex)
        Dz_ij = np.zeros_like(N0_ij, complex)
        Delta_p = calc.density.setups[a].Delta_pL[:, 0].copy()
        Delta_ij = unpack(Delta_p)
        for ik in range(Nk):
            P_ami = get_spinorbit_projections(calc, ik, v_knm[ik])
            P_smi = np.array([P_ami[a][:, ::2], P_ami[a][:, 1::2]])
            P_smi = np.dot(C_ss, np.swapaxes(P_smi, 0, 1))

            P0_mi = P_smi[0]
            P1_mi = P_smi[1]
            f_mm = np.diag(f_km[ik])

            Dx_ij += P0_mi.conj().T.dot(f_mm).dot(P1_mi)
            Dx_ij += P1_mi.conj().T.dot(f_mm).dot(P0_mi)
            Dy_ij -= 1.0j * P0_mi.conj().T.dot(f_mm).dot(P1_mi)
            Dy_ij += 1.0j * P1_mi.conj().T.dot(f_mm).dot(P0_mi)
            Dz_ij += P0_mi.conj().T.dot(f_mm).dot(P0_mi)
            Dz_ij -= P1_mi.conj().T.dot(f_mm).dot(P1_mi)

        mx = np.sum(N0_ij * Dx_ij).real
        my = np.sum(N0_ij * Dy_ij).real
        mz = np.sum(N0_ij * Dz_ij).real

        m_av.append([mx, my, mz])
        m_v[0] += np.sum(Delta_ij * Dx_ij) * (4 * np.pi)**0.5
        m_v[1] += np.sum(Delta_ij * Dy_ij) * (4 * np.pi)**0.5
        m_v[2] += np.sum(Delta_ij * Dz_ij) * (4 * np.pi)**0.5

    return m_v.real, m_av


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
        v_knm = soc_eigenstates(calc,
                                return_wfs=True,
                                bands=bands)['v_kmm']
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


def get_L_vlmm():
    if len(_L_vlmm) == 3:
        return _L_vlmm

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
    _L_vlmm.append([s, p, d])

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
    _L_vlmm.append([s, p, d])

    p = np.zeros((3, 3), complex)  # y, z, x
    p[0, 2] = 1.0j
    p[2, 0] = -1.0j
    d = np.zeros((5, 5), complex)  # xy, yz, z^2, xz, x^2-y^2
    d[0, 4] = 2.0j
    d[4, 0] = -2.0j
    d[1, 3] = 1.0j
    d[3, 1] = -1.0j
    _L_vlmm.append([s, p, d])

    return _L_vlmm
