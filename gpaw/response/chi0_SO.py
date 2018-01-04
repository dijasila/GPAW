import numpy as np

import sys

from gpaw.response.chi0 import Chi0
from gpaw.spinorbit import get_spinorbit_eigenvalues, set_calculator
import gpaw.mpi as mpi
from gpaw.mpi import serial_comm
from gpaw import GPAW
from gpaw.response.df import DielectricFunction

from ase.units import Hartree


class Chi0_SO(Chi0):

    def __init__(self, calc, *args, **kwargs):
        socalc = GPAW(calc, txt=None,
                      communicator=serial_comm)
        e_mk, v_knm = get_spinorbit_eigenvalues(socalc, return_wfs=True)
        set_calculator(socalc, e_mk.T)
        self.socalc = socalc

        kwargs.pop('omegamax')
        omegamax = self.find_maximum_SO_frequency(socalc) * Hartree

        Chi0.__init__(self, calc, omegamax=omegamax, *args, **kwargs)

        self.pair.fermi_level = socalc.occupations.fermilevel

        self.e_mk = e_mk
        self.v_knm = v_knm

    def find_maximum_SO_frequency(self, calc):
        """Determine the maximum electron-hole pair transition energy."""
        self.epsmin = 10000.0
        self.epsmax = -10000.0
        nbands = calc.wfs.bd.nbands
        for kpt in calc.wfs.kpt_u:
            self.epsmin = min(self.epsmin, kpt.eps_n[0])
            self.epsmax = max(self.epsmax, kpt.eps_n[nbands - 1])

        return self.epsmax - self.epsmin

    def calculate(self, *args, **kwargs):
        kwargs['spin'] = 0
        return Chi0.calculate(self, *args, **kwargs)

    def get_eigenvalues(self, k_v, *args, **kwargs):
        kd = self.calc.wfs.kd
        pd = kwargs['pd']

        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        q_c = pd.kd.bzk_kc[0]
        K1 = self.pair.find_kpoint(k_c)
        K2 = self.pair.find_kpoint(k_c + q_c)
        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]

        e1_m = self.e_mk[:, ik1] / Hartree
        e2_m = self.e_mk[:, ik2] / Hartree
        deps_nm = np.subtract(e1_m[:, np.newaxis], e2_m)

        return deps_nm.reshape(-1)

    def get_matrix_element(self, k_v, *args, **kwargs):
        nspins = self.calc.wfs.nspins
        kd = self.calc.wfs.kd
        pd = kwargs['pd']
        kwargs['n1'] = 0
        kwargs['n2'] = self.nbands
        kwargs['m1'] = 0
        kwargs['m2'] = self.nbands

        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        q_c = pd.kd.bzk_kc[0]
        optical_limit = np.allclose(q_c, 0.0)
        K1 = self.pair.find_kpoint(k_c)
        K2 = self.pair.find_kpoint(k_c + q_c)
        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]

        v1_nm = self.v_knm[ik1]
        v2_nm = self.v_knm[ik2]

        nb = self.nbands
        nG = pd.ngmax + 2 * optical_limit

        n_nmG = np.zeros((nb * 2, nb * 2, nG),
                         dtype=complex)
        n_nmG[:nb, :nb] = Chi0.get_matrix_element(self, k_v, 0,
                                                  SO=True, **kwargs)

        if nspins == 2:
            n_nmG[nb:, nb:] = Chi0.get_matrix_element(self, k_v, 1,
                                                      SO=True, **kwargs)
        else:
            n_nmG[nb:, nb:] = n_nmG[:nb, :nb]

        f1_n = self.socalc.wfs.kpt_u[ik1].f_n
        f2_n = self.socalc.wfs.kpt_u[ik2].f_n

        df_nm = (f1_n[:, np.newaxis] - f2_n)
        df_nm[df_nm <= 1e-20] = 0.0

        n_nmG = np.dot(np.dot(v1_nm, n_nmG).T, v2_nm.conj().T).T
        n_nmG *= df_nm[..., np.newaxis]**0.5
        if self.calc.wfs.nspins == 1:
            n_nmG /= 2**0.5

        return np.array(n_nmG).reshape(-1, nG)

    def get_intraband_response(self, k_v, *args, **kwargs):
        nspins = self.calc.wfs.nspins
        kd = self.calc.wfs.kd
        pd = kwargs['pd']

        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self.pair.find_kpoint(k_c)
        ik1 = kd.bz2ibz_k[K1]
        v1_nm = self.v_knm[ik1]

        n1, n2 = v1_nm.shape
        nb = self.nbands
        
        vel_nmv = np.zeros((n1, n2, 3), dtype=complex)
        m_m = np.arange(0, self.nbands)
        kpt1 = self.pair.get_k_point(0, k_c, 0, self.nbands)
        for n in range(self.nbands):
            vel_nmv[n, :nb] = self.pair.optical_pair_velocity(n, m_m,
                                                              kpt1, kpt1)

        if nspins == 2:
            kpt1 = self.pair.get_k_point(1, k_c, 0, self.nbands)
            for n in range(self.nbands):
                vel_nmv[nb + n, nb:] = self.pair.optical_pair_velocity(n, m_m,
                                                                       kpt1,
                                                                       kpt1)
        else:
            vel_nmv[nb:, nb:] = vel_nmv[:nb, :nb]

        # vel_nmv = np.dot(np.dot(v1_nm, vel_nmv).T, v1_nm.conj().T).T

        if self.calc.wfs.nspins == 1:
            vel_nmv /= 2**0.5
        vel_nv = np.diagonal(vel_nmv).T

        return vel_nv

    def get_intraband_eigenvalue(self, k_v, *args, **kwargs):
        pd = kwargs['pd']
        kd = self.calc.wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self.pair.find_kpoint(k_c)
        ik1 = kd.bz2ibz_k[K1]
        # kpt1 = self.pair.get_k_point(0, k_c, 0, self.nbands)

        # return np.concatenate([kpt1.eps_n, kpt1.eps_n])
        return self.e_mk[:, ik1] / Hartree


class DielectricFunctionSO(DielectricFunction):

    def __init__(self, calc, name=None, frequencies=None, domega0=0.1,
                 omega2=10.0, omegamax=None, ecut=50, hilbert=True,
                 nbands=None, eta=0.2, ftol=1e-6, threshold=1,
                 intraband=True, nblocks=1, world=mpi.world, txt=sys.stdout,
                 gate_voltage=None, truncation=None, disable_point_group=False,
                 disable_time_reversal=False,
                 integrationmode=None, pbc=None, rate=0.0,
                 omegacutlower=None, omegacutupper=None, eshift=0.0):
        """Creates a DielectricFunction object.

        calc: str
            The groundstate calculation file that the linear response
            calculation is based on.
        name: str
            If defined, save the density-density response function to::

                name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())

            where q_c is the reduced momentum and N_c is the number of
            kpoints along each direction.
        frequencies: np.ndarray
            Specification of frequency grid. If not set the non-linear
            frequency grid is used.
        domega0: float
            Frequency grid spacing for non-linear frequency grid at omega = 0.
        omega2: float
            Frequency at which the non-linear frequency grid has doubled
            the spacing.
        omegamax: float
            The upper frequency bound for the non-linear frequency grid.
        ecut: float
            Plane-wave cut-off.
        hilbert: bool
            Use hilbert transform.
        nbands: int
            Number of bands from calc.
        eta: float
            Broadening parameter.
        ftol: float
            Threshold for including close to equally occupied orbitals,
            f_ik - f_jk > ftol.
        threshold: float
            Threshold for matrix elements in optical response perturbation
            theory.
        intraband: bool
            Include intraband transitions.
        world: comm
            mpi communicator.
        nblocks: int
            Split matrices in nblocks blocks and distribute them G-vectors or
            frequencies over processes.
        txt: str
            Output file.
        gate_voltage: float
            Shift Fermi level of ground state calculation by the
            specified amount.
        truncation: str
            'wigner-seitz' for Wigner Seitz truncated Coulomb.
            '2D, 1D or 0d for standard analytical truncation schemes.
            Non-periodic directions are determined from k-point grid
        eshift: float
            Shift unoccupied bands
        """

        self.chi0 = Chi0_SO(calc, frequencies, domega0=domega0,
                            omega2=omega2, omegamax=omegamax,
                            ecut=ecut, hilbert=hilbert, nbands=nbands,
                            eta=eta, ftol=ftol, threshold=threshold,
                            intraband=intraband, world=world, nblocks=nblocks,
                            txt=txt, gate_voltage=gate_voltage,
                            disable_point_group=disable_point_group,
                            disable_time_reversal=disable_time_reversal,
                            integrationmode=integrationmode,
                            pbc=pbc, rate=rate, eshift=eshift)

        self.name = name

        self.omega_w = self.chi0.omega_w
        if omegacutlower is not None:
            inds_w = np.logical_and(self.omega_w > omegacutlower / Hartree,
                                    self.omega_w < omegacutupper / Hartree)
            self.omega_w = self.omega_w[inds_w]

        nw = len(self.omega_w)

        world = self.chi0.world
        self.mynw = (nw + world.size - 1) // world.size
        self.w1 = min(self.mynw * world.rank, nw)
        self.w2 = min(self.w1 + self.mynw, nw)
        self.truncation = truncation
