"""This module defines a density class."""

from __future__ import print_function, division
from math import pi, sqrt

import numpy as np
from ase.units import Bohr
from ase.utils.timing import timer
from distutils.version import LooseVersion

from gpaw import debug
from gpaw.atom_centered_functions import AtomCenteredFunctions as ACF
from gpaw.matrix import AtomBlockMatrix
from gpaw.mixer import get_mixer_from_keywords, MixerWrapper
from gpaw.transformers import Transformer
from gpaw.utilities.blas import gemm
from gpaw.utilities.debug import frozen
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.pw import PWDescriptor


class NullBackgroundCharge:
    charge = 0.0

    def set_grid_descriptor(self, gd):
        pass

    def add_charge_to(self, rhot_r):
        pass

    def add_fourier_space_charge_to(self, pd, rhot_q):
        pass


@frozen
class AtomBlockDensityMatrix(AtomBlockMatrix):
    def __init__(self, setups, spinpolarized=False, collinear=True,
                 comm=None, rank_a=None):
        self.setups = setups
        self.spinpolarized = spinpolarized
        self.collinear = collinear
        self.rank_a = rank_a

        self.D_asii = {}

        AtomBlockMatrix.__init__(self, self.D_asii, 1 + spinpolarized, comm,
                                 [setup.ni for setup in setups])

    def initialize(self, charge, magmom_a, hund):
        f_asi = {}
        for a, M in enumerate(magmom_a):
            if self.rank_a[a] != self.comm.rank:
                continue
            f_si = self.setups[a].calculate_initial_occupation_numbers(
                abs(M), hund, charge=charge / len(self.setups),
                nspins=1 + self.spinpolarized)
            if M < 0:
                f_si = f_si[::-1]
            f_asi[a] = f_si
            self.D_asii[a] = self.setups[a].initialize_density_matrix(f_si)
        return f_asi

    def calculate_multipole_moments(self):
        """Calculate multipole moments of compensation charges.

        Returns the total compensation charge in units of electron
        charge, so the number will be negative because of the
        dominating contribution from the nuclear charge."""

        comp_charge = 0.0
        Q_aL = {}
        for a, D_sii in self.D_asii.items():
            setup = self.setups[a]
            Q_L = np.einsum('sij,ijL->L', D_sii, setup.Delta_iiL)
            Q_L[0] += setup.Delta0
            comp_charge += Q_L[0]
            Q_aL[a] = Q_L
        return self.comm.sum(comp_charge) * sqrt(4 * pi), Q_aL

    def update(self, wfs):
        for a, rank in enumerate(self.rank_a):
            if rank == self.comm.rank:
                if a not in self.D_asii:
                    ni = self.size_a[a]
                    self.D_asii[a] = np.empty((self.nspins, ni, ni))
            else:
                assert a not in self.D_asii

        for D_sii in self.D_asii.values():
            D_sii[:] = 0.0

        for kpt in wfs.mykpts:
            self.update1(kpt, wfs)

        for D_sii in self.D_asii.values():
            wfs.kptband_comm.sum(D_sii)

        self.symmetrize(wfs.kd.symmetry)

    def _update(self, D_sii, P_ni, f_n, spin):
        if self.collinear:
            D_ii = np.dot(P_ni.T.conj() * f_n, P_ni).real
            D_sii[spin] += D_ii
        else:
            n, i = P_ni.shape
            P_nsi = P_ni.reshape((n // 2, 2, i))
            D_ssii = np.einsum('isn,ntj->stij', P_nsi.T.conj() * f_n, P_nsi)
            D_sii[0] += D_ssii[0, 0].real + D_ssii[1, 1].real
            D_sii[1] += 2 * D_ssii[0, 1].real  # ???
            D_sii[2] += 2 * D_ssii[0, 1].imag  # ???
            D_sii[3] += D_ssii[0, 0].real - D_ssii[1, 1].real

    def update1(self, kpt, wfs):
        if kpt.rho_MM is None:
            P_In = kpt.P.matrix
            for a, I1, I2 in P_In.indices:
                P_ni = P_In.array[I1:I2].T
                self._update(self.D_asii[a], P_ni, kpt.f_n, P_In.spin)
        else:
            for a, P_qMi in wfs.P_aqMi.items():
                P_Mi = P_qMi[kpt.q]
                rhoP_Mi = np.zeros_like(P_Mi)
                D_ii = np.zeros(self.D_asii[a][kpt.s].shape, kpt.rho_MM.dtype)
                gemm(1.0, P_Mi, kpt.rho_MM, 0.0, rhoP_Mi)
                gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
                self.D_asii[a][kpt.s] += D_ii.real

        if hasattr(kpt, 'c_on'):
            1 / 0
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn = ne * np.outer(c_n.conj(), c_n)
                D_sii = 42
                D_sii[kpt.s] += np.dot(P_ni.T.conj(), np.dot(d_nn, P_ni)).real

    def symmetrize(self, symmetry):
        if len(symmetry.op_scc) == 0:
            return

        all_D_asii = self.broadcast()

        for s in range(1 + self.spinpolarized):
            D_aii = [D_sii[s] for D_sii in all_D_asii]
            for a, D_sii in self.D_asii.items():
                setup = self.setups[a]
                D_sii[s] = setup.symmetrize(a, D_aii, symmetry.a_sa)

    def estimate_magnetic_moments(self):
        magmom_a = np.zeros(self.rank_a.shape)
        if self.spinpolarized:
            for a, D_sii in self.D_asii.items():
                magmom_a[a] = np.einsum('ij,ij', D_sii[0] - D_sii[1],
                                        self.setups[a].N0_ii)
            self.comm.sum(magmom_a)
        return magmom_a


class Density:
    """Density object.

     ``gd``          Grid descriptor for coarse grids.
     ``finegd``      Grid descriptor for fine grids.
     ``interpolate`` Function for interpolating the electron density.
     ``mixer``       ``DensityMixer`` object.

     Soft and smooth pseudo functions on uniform 3D grids:

     ``nt_sR``  Electron density on the coarse grid.
     ``nt_sr``  Electron density on the fine grid.
     ``nt_r``   Electron density on the fine grid.
     ``rhot_r`` Charge density on the fine grid.
     ``nct_R``  Core electron-density on the coarse grid.
    """

    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 background_charge=None):
        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.spinpolarized = nspins == 2
        self.collinear = True
        self.charge = float(charge)
        self.redistributor = redistributor

        # This can contain e.g. a Jellium background charge
        if background_charge is None:
            background_charge = NullBackgroundCharge()
        background_charge.set_grid_descriptor(self.finegd)
        self.background_charge = background_charge
        self.log = None

        self.charge_eps = 1e-7

        self.D_II = None

        self.nct_a = None
        self.ghat_aL = None

        self.Q_aL = None

        self.nct_R = None
        self.rhot_r = None

        self.nt_sR = None
        self.nt_sr = None
        self.nt_r = None

        self.fixed = False
        self.hund = False
        self.mixer = None
        self.magmom_a = None

        # XXX at least one test will fail because None has no 'reset()'
        # So we need DummyMixer I guess
        self.set_mixer(None)

        self.timer = nulltimer
        self.error = None
        self.setups = None

    @property
    def nt_sG(self):
        return self.nt_sR

    @property
    def nt_sg(self):
        return self.nt_sr

    def __str__(self):
        s = 'Densities:\n'
        s += '  Coarse grid: {}*{}*{} grid\n'.format(*self.gd.N_c)
        s += '  Fine grid: {}*{}*{} grid\n'.format(*self.finegd.N_c)
        s += '  Total Charge: {:.6f}'.format(self.charge)
        if self.fixed:
            s += '\n  Fixed'
        return s

    def summary(self, atoms, magmom, log):
        if not self.spinpolarized:
            return
        try:
            # XXX This doesn't always work, HGH, SIC, ...
            sc = self.get_spin_contamination(atoms, int(magmom < 0))
            log('Spin contamination: {sc:.6f} electrons'.format(sc=sc))
        except (TypeError, AttributeError):
            pass

    def initialize(self, setups, timer, magmom_a, hund):
        self.timer = timer
        self.setups = setups
        self.hund = hund
        self.magmom_a = magmom_a
        self.D_II = AtomBlockDensityMatrix(self.setups, self.spinpolarized,
                                           self.collinear, comm=self.gd.comm)

    def reset(self):
        self.Q_aL = None
        self.nct_R = None
        self.rhot_r = None
        self.nt_sR = None
        self.nt_sr = None
        self.nt_r = None

    def set_positions(self, spos_ac, rank_a):
        self.D_II.rank_a = rank_a
        self.nct_a.set_positions(spos_ac)
        self.ghat_aL.set_positions(spos_ac)
        #self.nt_sR = None
        #self.Q_aL = None

    def update_pseudo_core_density(self):
        if self.nct_R is None:
            self.nct_R = self.gd.zeros()
            self.nct_a.add_to(self.nct_R, 1 / (1 + self.spinpolarized),
                              force_real_space=True)

    def calculate_pseudo_density(self, wfs):
        """Calculate nt_sR from scratch.

        nt_sR will be equal to nct_R plus the contribution from
        wfs.add_to_density().
        """
        self.update_pseudo_core_density()
        self.nt_sR[:] = 0.0
        wfs.calculate_density_contribution(self.nt_sR)
        self.nt_sR += self.nct_R

    @timer('Density')
    def update(self, wfs):
        self.calculate_pseudo_density(wfs)
        self.D_II.update(wfs)
        comp_charge, self.Q_aL = self.D_II.calculate_multipole_moments()

        if wfs.mode == 'lcao':
            self.normalize(comp_charge)

        self.mix(comp_charge)

    @timer('Mix')
    def mix(self, comp_charge):
        assert isinstance(self.mixer, MixerWrapper), self.mixer
        self.error = self.mixer.mix(self.nt_sR, self.D_II.D_asii)
        self.interpolate_pseudo_density()

    def initialize_from_atomic_densities(self, basis_functions):
        """Initialize D_asp, nt_sR and Q_aL from atomic densities.

        nt_sR is initialized from atomic orbitals, and will
        be constructed with the specified magnetic moments and
        obeying Hund's rules if ``hund`` is true."""

        # XXX does this work with blacs?  What should be distributed?
        # Apparently this doesn't use blacs at all, so it's serial
        # with respect to the blacs distribution.  That means it works
        # but is not particularly efficient (not that this is a time
        # consuming step)

        self.log('Density initialized from atomic densities')

        c = self.charge - self.background_charge.charge
        f_asi = self.D_II.initialize(c, self.magmom_a, self.hund)

        self.nt_sR = self.gd.empty(1 + self.spinpolarized)
        self.update_pseudo_core_density()
        self.nt_sR[:] = self.nct_R
        basis_functions.add_to_density(self.nt_sR, f_asi)
        comp_charge, self.Q_aL = self.D_II.calculate_multipole_moments()
        self.normalize(comp_charge)
        self.mixer.reset()
        self.mix(comp_charge)

    @timer('Density initialized from wave functions')
    def initialize_from_wavefunctions(self, wfs):
        """Initialize D_asp, nt_sR and Q_aL from wave functions."""
        self.log('Density initialized from wave functions')
        self.nt_sR = self.gd.empty(self.nspins)
        self.calculate_pseudo_density(wfs)
        self.D_II.update(wfs)
        comp_charge, self.Q_aL = self.D_II.calculate_multipole_moments()
        self.normalize(comp_charge)
        self.mixer.reset()
        self.mix(comp_charge)

    def normalize(self, comp_charge):
        total_charge = (self.charge + comp_charge -
                        self.background_charge.charge)
        pseudo_charge = self.gd.integrate(self.nt_sR).sum()
        if pseudo_charge != 0:
            x = -total_charge / pseudo_charge
            self.nt_sR *= x
        else:
            # Use homogeneous background:
            volume = self.gd.get_size_of_global_array().prod() * self.gd.dv
            self.nt_sR[:] = -total_charge / volume

    def set_mixer(self, mixer):
        if mixer is None:
            mixer = {}
        if isinstance(mixer, dict):
            mixer = get_mixer_from_keywords(self.gd.pbc_c.any(),
                                            1 + self.spinpolarized,
                                            **mixer)
        if not hasattr(mixer, 'mix'):
            raise ValueError('Not a mixer: %s' % mixer)
        self.mixer = MixerWrapper(mixer, 1 + self.spinpolarized, self.gd)

    def estimate_magnetic_moments(self):
        return self.D_II.estimate_magnetic_moments()

    def get_correction(self, a, spin):
        """Integrated atomic density correction.

        Get the integrated correction to the pseuso density relative to
        the all-electron density.
        """
        setup = self.setups[a]
        return sqrt(4 * pi) * (
            np.dot(self.D_asp[a][spin], setup.Delta_pL[:, 0]) +
            setup.Delta0 / self.nspins)

    def estimate_memory(self, mem):
        nspins = self.nspins
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()

        arrays = mem.subnode('Arrays')
        for name, size in [('nt_sR', nbytes * nspins),
                           ('nt_sr', nfinebytes * nspins),
                           ('nt_r', nfinebytes),
                           ('rhot_r', nfinebytes),
                           ('nct_R', nbytes)]:
            arrays.subnode(name, size)

        lfs = mem.subnode('Localized functions')
        for name, obj in [('nct', self.nct_a),
                          ('ghat', self.ghat_aL)]:
            obj.estimate_memory(lfs.subnode(name))
        self.mixer.estimate_memory(mem.subnode('Mixer'), self.gd)

        # TODO
        # The implementation of interpolator memory use is not very
        # accurate; 20 MiB vs 13 MiB estimated in one example, probably
        # worse for parallel calculations.

    def get_spin_contamination(self, atoms, majority_spin=0):
        """Calculate the spin contamination.

        Spin contamination is defined as the integral over the
        spin density difference, where it is negative (i.e. the
        minority spin density is larger than the majority spin density.
        """

        if majority_spin == 0:
            smaj = 0
            smin = 1
        else:
            smaj = 1
            smin = 0
        nt_sr, gd = self.get_all_electron_density(atoms)
        dt_sr = nt_sr[smin] - nt_sr[smaj]
        dt_sr = np.where(dt_sr > 0, dt_sr, 0.0)
        return gd.integrate(dt_sr)

    def write(self, writer):
        D_sP = self.D_II.pack()
        writer.write(density=self.gd.collect(self.nt_sR) / Bohr**3,
                     atomic_density_matrices=D_sP)

    def read(self, reader):
        self.nt_sR = self.gd.empty(self.nspins)
        self.gd.distribute(reader.density.density, self.nt_sR)
        self.nt_sR *= reader.bohr**3

        # Read atomic density matrices
        D_sP = reader.density.atomic_density_matrices
        self.D_II.unpack(D_sP)

    def initialize_from_other_density(self, dens, kptband_comm):
        """Redistribute pseudo density and atomic density matrices.

        Collect dens.nt_sR and dens.D_asp to world master and distribute."""

        self.nt_sR = redistribute_array(dens.nt_sR, dens.gd, self.gd,
                                        self.nspins, kptband_comm)

        self.D_II.redistribute(dens)


@frozen
class RealSpaceDensity(Density):
    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 stencil=3,
                 background_charge=None):
        Density.__init__(self, gd, finegd, nspins, charge, redistributor,
                         background_charge=background_charge)
        self.stencil = stencil
        self.interpolator = None

    def initialize(self, setups, timer, magmom_a, hund):
        Density.initialize(self, setups, timer, magmom_a, hund)

        # Interpolation function for the density:
        self.interpolator = Transformer(self.redistributor.aux_gd,
                                        self.finegd, self.stencil)

        spline_aj = []
        for setup in setups:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        self.nct_a = ACF(self.gd, spline_aj,
                         integral=[setup.Nct for setup in setups],
                         cut=True)
        self.ghat_aL = ACF(self.finegd, [setup.ghat_l for setup in setups],
                           integral=sqrt(4 * pi))

    def interpolate_pseudo_density(self):
        """Interpolate pseudo density to fine grid."""

        comp_charge, Q_aL = self.D_II.calculate_multipole_moments()
        self.nt_sr = self.finegd.empty(1 + self.spinpolarized)

        for a, b in zip(self.nt_sR, self.nt_sr):
            a = self.redistributor.distribute(a)
            self.interpolator.apply(a, b)

        self.nt_r = self.nt_sr.sum(0)

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if not self.gd.pbc_c.all():
            # With zero-boundary conditions in one or more directions,
            # this is not the case.
            A = self.background_charge.charge - self.charge - comp_charge
            if abs(A) > 1.0e-14:
                B = self.finegd.integrate(self.nt_r)
                self.nt_sr *= A / B
                self.nt_r *= A / B

        self.rhot_r = self.nt_r.copy(0)

        self.ghat_aL.add_to(self.rhot_r, Q_aL)
        self.background_charge.add_charge_to(self.rhot_r)

        if debug:
            charge = self.finegd.integrate(self.rhot_r) + self.charge
            if abs(charge) > self.charge_eps:
                raise RuntimeError('Charge not conserved: excess={}'
                                   .format(charge))

    def get_pseudo_core_kinetic_energy_density_lfc(self):
        return ACF(self.gd,
                   [[setup.tauct] for setup in self.setups],
                   cut=True)

    def calculate_dipole_moment(self):
        return self.finegd.calculate_dipole_moment(self.rhot_r)


@frozen
class ReciprocalSpaceDensity(Density):
    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 background_charge=None):
        assert gd.comm.size == 1
        serial_finegd = finegd.new_descriptor(comm=gd.comm)

        from gpaw.utilities.grid import GridRedistributor
        noredist = GridRedistributor(redistributor.comm,
                                     redistributor.broadcast_comm, gd, gd)
        Density.__init__(self, gd, serial_finegd, nspins, charge,
                         redistributor=noredist,
                         background_charge=background_charge)

        self.pd2 = PWDescriptor(None, gd)
        self.pd3 = PWDescriptor(None, serial_finegd)

        self.G3_G = self.pd2.map(self.pd3)

        self.xc_redistributor = GridRedistributor(redistributor.comm,
                                                  redistributor.comm,
                                                  serial_finegd, finegd)

        self.rhot_q = None
        self.nt_Q = None

    def initialize(self, setups, timer, magmom_av, hund):
        Density.initialize(self, setups, timer, magmom_av, hund)

        spline_aj = []
        for setup in setups:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        self.nct_a = ACF(self.pd2, spline_aj)

        self.ghat_aL = ACF(self.pd3, [setup.ghat_l for setup in setups],
                           blocksize=256, comm=self.xc_redistributor.comm)

    def interpolate_pseudo_density(self):
        """Interpolate pseudo density to fine grid."""

        if self.nt_sr is None:
            self.nt_sr = self.finegd.empty(1 + self.spinpolarized)
            self.nt_Q = self.pd2.empty()

        self.nt_Q[:] = 0.0
        for nt_R, nt_r in zip(self.nt_sR, self.nt_sr):
            nt_r[:], nts_Q = self.pd2.interpolate(nt_R, self.pd3)
            self.nt_Q += nts_Q

        comp_charge, Q_aL = self.D_II.calculate_multipole_moments()

        self.rhot_q = self.pd3.zeros()
        self.rhot_q[self.G3_G] = self.nt_Q * 8
        self.ghat_aL.add_to(self.rhot_q, Q_aL)
        self.background_charge.add_fourier_space_charge_to(self.pd3,
                                                           self.rhot_q)
        self.rhot_q[0] = 0.0

    def interpolate(self, in_xR, out_xR=None):
        """Interpolate array(s)."""
        if out_xR is None:
            out_xR = self.finegd.empty(in_xR.shape[:-3])

        a_xR = in_xR.reshape((-1,) + in_xR.shape[-3:])
        b_xR = out_xR.reshape((-1,) + out_xR.shape[-3:])

        for in_R, out_R in zip(a_xR, b_xR):
            out_R[:] = self.pd2.interpolate(in_R, self.pd3)[0]

        return out_xR

    distribute_and_interpolate = interpolate

    def get_pseudo_core_kinetic_energy_density_lfc(self):
        return ACF(self.pd,
                   [[setup.tauct] for setup in self.setups],
                   cut=True)

    def calculate_dipole_moment(self):
        if LooseVersion(np.__version__) < '1.6.0':
            raise NotImplementedError
        pd = self.pd3
        N_c = pd.tmp_Q.shape

        m0_q, m1_q, m2_q = [i_G == 0
                            for i_G in np.unravel_index(pd.Q_qG[0], N_c)]
        rhot_q = self.rhot_q.imag
        rhot_cs = [rhot_q[m1_q & m2_q],
                   rhot_q[m0_q & m2_q],
                   rhot_q[m0_q & m1_q]]
        d_c = [np.dot(rhot_s[1:], 1.0 / np.arange(1, len(rhot_s)))
               for rhot_s in rhot_cs]
        return -np.dot(d_c, pd.gd.cell_cv) / pi * pd.gd.dv


def redistribute_array(nt_sR, gd1, gd2, nspins, kptband_comm):
    nt_sR = gd1.collect(nt_sR)
    new_nt_sR = gd2.empty(nspins)
    if kptband_comm.rank == 0:
        gd2.distribute(nt_sR, new_nt_sR)
    kptband_comm.broadcast(new_nt_sR, 0)
    return new_nt_sR
