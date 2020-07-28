import warnings
from typing import Union, List, TYPE_CHECKING, Dict, Any, Optional
from pathlib import Path

import numpy as np
from ase.units import Ha, alpha, Bohr

from gpaw.projections import Projections
from gpaw.kpoint import KPoint
from gpaw.setup import Setup
from gpaw.utilities.partition import AtomPartition

if TYPE_CHECKING:
    from gpaw import GPAW

_L_vlmm: List[List[np.ndarray]] = []  # see get_L_vlmm() below

# typehints:
ArrayND = Any
Array1D = ArrayND
Array2D = ArrayND
Array3D = ArrayND


class KPointCollection:
    def __init__(self, kd, kpts: List[KPoint]):
        self.kd = kd
        self.kpts = {kpt.k: kpt for kpt in kpts}

        # Initialize ranks:
        self.ranks = np.zeros(kd.nbzkpts, int)
        for kpt in kpts:
            self.ranks[kpt.k] = kd.comm.rank
        kd.comm.sum(self.ranks)

    def eigenvalues(self) -> Optional[Array2D]:
        return self._collect('eps_n', Ha)

    def spin_projections(self) -> Optional[Array3D]:
        return self._collect('spin_projection_vn')

    def _collect(self, attr: str, scale: float = 1.0) -> Optional[ArrayND]:
        if self.kpts[0].projections.atom_partition.comm.rank != 0:
            return
        if self.kpts[0].projections.bcomm.rank != 0:
            return

        comm = self.kd.comm
        if comm.rank == 0:
            kpt = next(iter(self.kpts.values()))
            shape = getattr(kpt, attr).shape
            x_kn = np.empty((self.kd.nbzkpts,) + shape)
            for k, rank in enumerate(self.ranks):
                if rank == comm.rank:
                    x_kn[k] = getattr(self.kpts[k], attr) * scale
                else:
                    comm.receive(x_kn[k], rank)
            return x_kn

        for k, rank in enumerate(self.ranks):
            if rank == self.kd.comm.rank:
                comm.send(getattr(self.kpts[k], attr) * scale, 0)


def soc_eigenstates(calc: Union['GPAW', str, Path],
                    n1: int = 0,
                    n2: int = 0,
                    eigenvalues: Array3D = None,
                    scale: float = 1.0,
                    theta: float = 0.0,
                    phi: float = 0.0,
                    return_wfs: bool = False,
                    occupations: Dict = None
                    ) -> KPointCollection:
    """Calculate SOC eigenstates.

    Parameters:
        calc: Calculator
            GPAW calculator or path to gpw-file.
        bands: list of ints
            List of band indices for which to calculate soc.
        eigenvalues: (ns, nk, nb)-shaped ndarray [units: eV]
            Use these eigenvalues instead of those from calc.get_eigenvalues().
        return_wfs: bool
            Flag for returning wave functions.
        occupations: dict
            Example: {'name': 'fermi-dirac', 'width': 0.01} (width in eV).
            This will calculate the Fermi-level also.
        scale: float
            Scale the spinorbit coupling by this amount.
        theta: float
            Angle in radians.
        phi: float
            Angle in radians.

    Returns a dict containing:

    ======================  =========================
    key                     value
    ======================  =========================
    eigenvalues             (nk,ne)-shaped ndarray
    spin_projections        (nk,3,ne)-shaped ndarray
    eigenstates (optional)  (nk,ne,ne)-shaped ndarray
    fermi_level (optional)  float
    ======================  =========================

    Units for eigenvalues and fermi_level are eV.
    """

    from gpaw import GPAW

    if not isinstance(calc, GPAW):
        calc = GPAW(calc, txt=None)

    if n2 == 0:
        n2 = calc.get_number_of_bands()

    kd = calc.wfs.kd
    bd = calc.wfs.bd
    gd = calc.wfs.gd
    spos_ac = calc.spos_ac
    setups = calc.wfs.setups
    atom_partition = calc.density.atom_partition

    if eigenvalues is not None:
        assert eigenvalues.shape == (2, kd.nibzkpts, n2 - n1)

    # <phi_i|dV_adr / r * L_v|phi_j>
    dVL_avii = {a: soc(calc.wfs.setups[a],
                       calc.hamiltonian.xc,
                       D_sp) * scale
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

    mykpts = []
    for kpt_s in calc.wfs.kpt_qs:
        kpt1 = kpt_s[0]
        P1_nI = kpt1.projections.collect()
        eps1_n = bd.collect(kpt1.eps_n)

        if len(kpt_s) == 2:
            kpt2 = kpt_s[1]
            P2_nI = kpt2.projections.collect()
            eps2_n = bd.collect(kpt2.eps_n, broadcast=True)
        else:
            P2_nI = P1_nI
            eps2_n = eps1_n

        if bd.comm.rank != 0:
            continue

        ibz_index = kpt1.k

        if gd.comm.rank == 0:
            P1_nI = P1_nI[n1:n2]
            P2_nI = P2_nI[n1:n2]
        else:
            P1_nI = P2_nI = np.zeros((n2 - n1, 0), complex)

        if eigenvalues is None:
            eps1_n = eps1_n[n1:n2]
            eps2_n = eps2_n[n1:n2]
        else:
            eps1_n, eps2_n = eigenvalues[:, ibz_index]

        ibzkpt = KPoint(1.0, None, 0, 0)

        ibzkpt.eps_n = np.empty((n2 - n1) * 2)
        ibzkpt.eps_n[::2] = eps1_n
        ibzkpt.eps_n[1::2] = eps2_n

        ibzkpt.projections = Projections(
            nbands=2 * (n2 - n1),
            nproj_a=kpt1.projections.nproj_a,
            atom_partition=AtomPartition(gd.comm,
                                         np.zeros(len(spos_ac), int)),
            collinear=False)
        ibzkpt.projections.array[:] = 0.0
        ibzkpt.projections.array[::2, 0] = P1_nI
        ibzkpt.projections.array[1::2, 1] = P2_nI

        for bz_index in np.nonzero(kd.bz2ibz_k == ibz_index)[0]:
            kpt = ibzkpt.transform(kd, setups, spos_ac, bz_index)
            kpt.projections = kpt.projections.redist(atom_partition)
            add_soc(kpt, dVL_avii, s_vss, C_ss)
            kpt.q = bz_index
            mykpts.append(kpt)

    return KPointCollection(kd, mykpts)


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


def add_soc(kpt: KPoint,
            dVL_avii: Dict[int, Array3D],
            s_vss: Array3D,
            C_ss: Array2D) -> None:
    """Evaluate H in a basis of S_z eigenstates."""
    H_mm = np.diag(kpt.eps_n.astype(complex))
    for a, dVL_vii in dVL_avii.items():
        ni = dVL_vii.shape[1]
        H_ssii = np.zeros((2, 2, ni, ni), complex)
        H_ssii[0, 0] = dVL_vii[2]
        H_ssii[0, 1] = dVL_vii[0] - 1.0j * dVL_vii[1]
        H_ssii[1, 0] = dVL_vii[0] + 1.0j * dVL_vii[1]
        H_ssii[1, 1] = -dVL_vii[2]

        # Tranform to theta, phi basis
        H_ssii = np.tensordot(C_ss, H_ssii, ([0, 1]))
        H_ssii = np.tensordot(C_ss.T.conj(), H_ssii, ([1, 1]))

        P_msi = kpt.projections[a]
        for s1 in range(2):
            for s2 in range(2):
                H_ii = H_ssii[s1, s2]
                P1_mi = P_msi[:, s1]
                P2_mi = P_msi[:, s2]
                H_mm += np.dot(np.dot(P1_mi.conj(), H_ii), P2_mi.T)

    kpt.eps_n, v_snm = np.linalg.eigh(H_mm)

    P_msI = kpt.projections.array
    m, s, I = P_msI.shape
    P_msI[:] = v_snm.T.dot(P_msI.reshape((m, s * I))).reshape((m, s, I))

    sx_m = []
    sy_m = []
    sz_m = []
    for v_sn in v_snm.T:
        v_sn = np.array([v_sn[::2], v_sn[1::2]])
        sx_m.append(np.trace(v_sn.T.conj().dot(s_vss[0]).dot(v_sn)))
        sy_m.append(np.trace(v_sn.T.conj().dot(s_vss[1]).dot(v_sn)))
        sz_m.append(np.trace(v_sn.T.conj().dot(s_vss[2]).dot(v_sn)))

    kpt.spin_projection_vn = np.array([sx_m, sy_m, sz_m]).real
    kpt.v_snm = v_snm


def get_spinorbit_eigenvalues(calc, bands=None, gw_kn=None,
                              return_spin=False,
                              return_wfs=False,
                              scale=1.0,
                              theta=0.0, phi=0.0):
    warnings.warn('Please use soc_eigenstates() instead.')

    results = soc_eigenstates(calc, bands, gw_kn, scale, theta, phi,
                              return_wfs)
    values = [results['eigenstates'].T]
    if return_spin:
        values.append(results['spin_projections'])
    if return_wfs:
        values.append(results['eigenstates'])
    return tuple(values) if len(values) > 1 else values[0]


def set_calculator(calc, e_km, v_knm=None, width=None):
    raise DeprecationWarning(
        "Please use ef = soc_eigenstates(..., occupations=...)['fermi_level'] "
        'instead.')


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
                           bands=range(nbands))['eigenvalues']
    if width is None:
        assert calc.wfs.occupations.name == 'fermi-dirac'
        width = calc.wfs.occupations.width * Ha
    if width == 0.0:
        width = 1.e-6
    weight_k = np.ones(len(e_km)) / (2 * len(e_km))
    ne = calc.wfs.setups.nvalence - calc.density.charge
    f_km = occupation_numbers({'name': 'fermi-dirac', 'width': width},
                              e_km[np.newaxis],
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
    assert len(calc.get_bz_k_points()) == len(calc.get_ibz_k_points())

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

    raise RuntimeError(
        'This function has no tests.  It is very likely that it no longer '
        'works correctly after merging !677.')

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
    e_km = states['eigenvalues']
    v_knm = states['eigenstates']

    from gpaw.occupations import occupation_numbers
    if width is None:
        assert calc.wfs.occupations.name == 'fermi-dirac'
        width = calc.wfs.occupations.width * Ha
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

    assert len(calc.get_bz_k_points()) == len(calc.get_ibz_k_points())

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
                                bands=bands)['eigenstates']
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
