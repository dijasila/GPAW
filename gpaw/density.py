"""This module defines a density class."""

from __future__ import print_function, division
from math import pi, sqrt

import numpy as np
from ase.units import Bohr
from distutils.version import LooseVersion

from gpaw import debug
from gpaw.atom_centered_functions import AtomCenteredFunctions as ACF
from gpaw.debug import frozen
from gpaw.mixer import get_mixer_from_keywords, MixerWrapper
from gpaw.transformers import Transformer
from gpaw.utilities import unpack_atomic_matrices, pack_atomic_matrices
from gpaw.utilities.partition import AtomPartition
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.pw import PWDescriptor


class NullBackgroundCharge:
    charge = 0.0

    def set_grid_descriptor(self, gd):
        pass

    def add_charge_to(self, rhot_g):
        pass

    def add_fourier_space_charge_to(self, pd, rhot_q):
        pass


class UniformGridDensity:
    def __init__(self, gd, spinpolarized, collinear):
        self.gd = gd
        self.spinpolarized = spinpolarized
        self.collinear = collinear

        shape = (2, 2) if not collinear else (1 + spinpolarized,)
        self.array = gd.empty(shape)

    def arrays(self):
        for a in self.array:
            if self.collinear:
                yield a
            else:
                for b in a:
                    yield b

    def get_electron_density(self):
        if self.collinear:
            return self.array.sum(0)
        return ...

    def initialize_with_pseudo_core_density(self, nct_R):
        self.array[:] = nct_R

    def add_from_basis_set(self, basis, f_asi):
        basis.add_to_density(self.array, f_asi)

    def normalize(self, total_charge):
        """Normalize pseudo density."""
        pseudo_charge = self.gd.integrate(self.array).sum()
        x = -total_charge / pseudo_charge
        self.array *= x

    def interpolate(self, interpolator, redistributor):
        out = UniformGridDensity(interpolator.gdout, self.spinpolarized,
                                 self.collinear)
        for a, b in zip(self.arrays(), out.arrays()):
            c = redistributor.distribute(a)
            interpolator.apply(c, b)

            # With periodic boundary conditions, the interpolation will
            # conserve the number of electrons.
            if not self.gd.pbc_c.all():
                # With zero-boundary conditions in one or more directions,
                # this is not the case.
                C = self.gd.integrate(c)
                if abs(C) > 1.0e-14:
                    B = out.gd.integrate(b)
                    b *= C / B

        return out


class AtomBlockDensityMatrix:
    def __init__(self, setups, spinpolarized=False, collinear=True,
                 comm=None, rank_a=None):
        self.setups = setups
        self.spinpolarized = spinpolarized
        self.comm = comm
        self.rank_a = rank_a

        self.D_asii = {}

    def initialize(self, charge, magmom_a, hund):
        f_asi = {}
        for a, M in enumerate(magmom_a):
            f_si = self.setups[a].calculate_initial_occupation_numbers(
                abs(M), hund, charge=charge, nspins=1 + self.spinpolarized)
            if M < 0:
                f_si = f_si[::-1]
            self.D_asii[a] = self.setups[a].initialize_density_matrix(f_si)
            f_asi[a] = f_si
        return f_asi

    def set_ranks(self, rank_a):
        self.rank_a = rank_a

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
        for D_sii in self.D_asii.values():
            D_sii[:] = 0.0

        for kpt in wfs.kpt_u:
            self.update1(kpt)

        for D_sii in self.D_asii.values():
            wfs.kptband_comm.sum(D_sii)

        self.symmetrize(wfs.kd.symmetry)

    def update1(self, kpt):
        P_In = kpt.P_In
        if kpt.rho_MM is None:
            for a, I1, I2 in P_In.indices:
                P_in = P_In.array[I1:I2]
                self.D_asii[a][P_In.spin] += np.dot(P_in.conj() * kpt.f_n,
                                                    P_in.T).real
        else:
            P_Mi = self.P_aqMi[a][kpt.q]
            rhoP_Mi = np.zeros_like(P_Mi)
            D_ii = np.zeros(D_sii[kpt.s].shape, kpt.rho_MM.dtype)
            gemm(1.0, P_Mi, kpt.rho_MM, 0.0, rhoP_Mi)
            gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
            D_sii[kpt.s] += D_ii.real

        if hasattr(kpt, 'c_on'):
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn = ne * np.outer(c_n.conj(), c_n)
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

    def broadcast(self):
        D_asii = []
        for a, setup in enumerate(self.setups):
            D_sii = self.D_asii.get(a)
            if D_sii is None:
                ni = setup.ni
                D_sii = np.empty((self.nspins, ni, ni))
            self.comm.broadcast(D_sii, self.rank_a[a])
            D_asii.append(D_sii)
        return D_asii

    def estimate_magnetic_moments(self):
        magmom_a = np.zeros(self.rank_a.shape)
        if self.spinpolarized:
            for a, D_sii in self.D_asii.items():
                magmom_a[a] = np.einsum('ij,ij',
                                        D_sii[0] - D_sii[1],
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

     ``nt_sG``  Electron density on the coarse grid.
     ``nt_sg``  Electron density on the fine grid.
     ``nt_g``   Electron density on the fine grid.
     ``rhot_g`` Charge density on the fine grid.
     ``nct_G``  Core electron-density on the coarse grid.
    """

    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 background_charge=None):
        self.gd = gd
        self.finegd = finegd
        self.spinpolarized = nspins == 2
        self.collinear = True
        self.charge = float(charge)
        self.redistributor = redistributor

        # This can contain e.g. a Jellium background charge
        if background_charge is None:
            background_charge = NullBackgroundCharge()
        background_charge.set_grid_descriptor(self.finegd)
        self.background_charge = background_charge

        self.charge_eps = 1e-7

        self.D_II = None

        self.Q_aL = None

        self.nct_a = None
        self.ghat_aL = None

        self.nct_R = None
        self.rhot_r = None

        self.nt = None
        self.finent = None

        self.fixed = False
        self.hund = False
        self.mixer = None
        self.magmom_a = None
        self.log = None

        # XXX at least one test will fail because None has no 'reset()'
        # So we need DummyMixer I guess
        self.set_mixer(None)

        self.timer = nulltimer
        self.error = None
        self.setups = None

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

    def resettttttttttttttttt(self):
        # TODO: reset other parameters?
        self.nt = None

    def set_positions(self, spos_ac, rank_a):
        self.D_II.set_ranks(rank_a)
        self.nct_a.set_positions(spos_ac)
        self.ghat_aL.set_positions(spos_ac)
        self.mixer.reset()

        self.nct_R = self.gd.zeros()
        self.nct_a.add_to(self.nct_R, 1 / (1 + self.spinpolarized))

        self.nt = None
        self.Q_aL = None

    def calculate_pseudo_density(self, wfs):
        """Calculate nt_sG from scratch.

        nt_sG will be equal to nct_G plus the contribution from
        wfs.add_to_density().
        """
        self.nt.initialize_with_pseudo_core_density(self.nct_R)
        wfs.calculate_density_contribution(self.nt)

    def update(self, wfs):
        self.timer.start('Density')
        with self.timer('Pseudo density'):
            self.calculate_pseudo_density(wfs)
        with self.timer('Atomic density matrices'):
            self.D_II.update(wfs)
        with self.timer('Multipole moments'):
            comp_charge, self.Q_aL = self.D_II.calculate_multipole_moments()

        if wfs.mode == 'lcao':
            with self.timer('Normalize'):
                self.normalize(comp_charge)

        with self.timer('Mix'):
            self.mix(comp_charge)
        self.timer.stop('Density')

    def mix(self, comp_charge):
        assert isinstance(self.mixer, MixerWrapper), self.mixer
        self.error = self.mixer.mix(self.nt.array, self.D_II.D_asii)
        assert self.error is not None, self.mixer

        self.interpolate_pseudo_density()
        comp_charge, self.Q_aL = self.D_II.calculate_multipole_moments()
        self.calculate_pseudo_charge()

    def initialize_from_atomic_densities(self, basis_functions):
        """Initialize D_asp, nt_sG and Q_aL from atomic densities.

        nt_sG is initialized from atomic orbitals, and will
        be constructed with the specified magnetic moments and
        obeying Hund's rules if ``hund`` is true."""

        # XXX does this work with blacs?  What should be distributed?
        # Apparently this doesn't use blacs at all, so it's serial
        # with respect to the blacs distribution.  That means it works
        # but is not particularly efficient (not that this is a time
        # consuming step)

        self.log('Density initialized from atomic densities')

        c = (self.charge - self.background_charge.charge) / len(self.setups)
        f_asi = self.D_II.initialize(c, self.magmom_a, self.hund)

        self.nt = UniformGridDensity(self.gd, self.spinpolarized,
                                     self.collinear)
        self.nt.initialize_with_pseudo_core_density(self.nct_R)
        self.nt.add_from_basis_set(basis_functions, f_asi)
        self.calculate_normalized_charges_and_mix()

    def initialize_from_wavefunctions(self, wfs):
        """Initialize D_asp, nt_sG and Q_aL from wave functions."""
        self.log('Density initialized from wave functions')
        self.timer.start('Density initialized from wave functions')
        self.nt_sG = self.gd.empty(self.nspins)
        self.calculate_pseudo_density(wfs)
        D_asp = self.setups.empty_atomic_matrix(self.nspins,
                                                wfs.atom_partition)
        wfs.calculate_atomic_density_matrices(D_asp)
        self.D_asp = D_asp
        self.calculate_normalized_charges_and_mix()
        self.timer.stop('Density initialized from wave functions')

    def initialize_directly_from_arrays(self, nt_sG, D_asp):
        """Set D_asp and nt_sG directly."""
        self.nt_sG = nt_sG
        self.D_asp = D_asp
        D_asp.check_consistency()
        # No calculate multipole moments?  Tests will fail because of
        # improperly initialized mixer

    def calculate_normalized_charges_and_mix(self):
        comp_charge, self.Q_aL = self.D_II.calculate_multipole_moments()
        total_charge = (self.charge + comp_charge -
                        self.background_charge.charge)
        self.nt.normalize(total_charge)
        self.mix(comp_charge)

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
        for name, size in [('nt_sG', nbytes * nspins),
                           ('nt_sg', nfinebytes * nspins),
                           ('nt_g', nfinebytes),
                           ('rhot_g', nfinebytes),
                           ('nct_G', nbytes)]:
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
        nt_sg, gd = self.get_all_electron_density(atoms)
        dt_sg = nt_sg[smin] - nt_sg[smaj]
        dt_sg = np.where(dt_sg > 0, dt_sg, 0.0)
        return gd.integrate(dt_sg)

    def write(self, writer):
        writer.write(density=self.gd.collect(self.nt_sG) / Bohr**3,
                     atomic_density_matrices=pack_atomic_matrices(self.D_asp))

    def read(self, reader):
        nt_sG = self.gd.empty(self.nspins)
        self.gd.distribute(reader.density.density, nt_sG)
        nt_sG *= reader.bohr**3

        # Read atomic density matrices
        natoms = len(self.setups)
        atom_partition = AtomPartition(self.gd.comm, np.zeros(natoms, int),
                                       'density-gd')
        D_asp = self.setups.empty_atomic_matrix(self.nspins, atom_partition)
        self.atom_partition = atom_partition  # XXXXXX
        spos_ac = np.zeros((natoms, 3))  # XXXX
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)

        D_sP = reader.density.atomic_density_matrices
        if self.gd.comm.rank == 0:
            D_asp.update(unpack_atomic_matrices(D_sP, self.setups))
            D_asp.check_consistency()

        self.initialize_directly_from_arrays(nt_sG, D_asp)

    def initialize_from_other_density(self, dens, kptband_comm):
        """Redistribute pseudo density and atomic density matrices.

        Collect dens.nt_sG and dens.D_asp to world master and distribute."""

        new_nt_sG = redistribute_array(dens.nt_sG, dens.gd, self.gd,
                                       self.nspins, kptband_comm)

        self.atom_partition, self.atomdist, D_asp = \
            redistribute_atomic_matrices(dens.D_asp, self.gd, self.nspins,
                                         self.setups, self.redistributor,
                                         kptband_comm)

        self.initialize_directly_from_arrays(new_nt_sG, D_asp)


@frozen
class RealSpaceDensity(Density):
    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 stencil=3,
                 background_charge=None):
        Density.__init__(self, gd, finegd, nspins, charge, redistributor,
                         background_charge=background_charge)
        self.stencil = stencil

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
        self.finent = self.nt.interpolate(self.interpolator,
                                          self.redistributor)

    def calculate_pseudo_charge(self):
        self.rhot_r = self.finent.get_electron_density()
        self.ghat_aL.add_to(self.rhot_r, self.Q_aL)
        self.background_charge.add_charge_to(self.rhot_r)

        if debug:
            charge = self.finegd.integrate(self.rhot_r) + self.charge
            if abs(charge) > self.charge_eps:
                raise RuntimeError('Charge not conserved: excess={}'
                                   .format(charge))

    def get_pseudo_core_kinetic_energy_density_lfc(self):
        return ACF(self.gd,
                   [[setup.tauct] for setup in self.setups],
                   forces=True, cut=True)

    def calculate_dipole_moment(self):
        return self.finegd.calculate_dipole_moment(self.rhot_g)


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

    def interpolate_pseudo_density(self, comp_charge=None):
        """Interpolate pseudo density to fine grid."""
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()

        if self.nt_sg is None:
            self.nt_sg = self.finegd.empty(self.nspins)
            self.nt_sQ = self.pd2.empty(self.nspins)

        for nt_G, nt_Q, nt_g in zip(self.nt_sG, self.nt_sQ, self.nt_sg):
            nt_g[:], nt_Q[:] = self.pd2.interpolate(nt_G, self.pd3)

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

    def calculate_pseudo_charge(self):
        self.nt_Q = self.nt_sQ.sum(axis=0)
        self.rhot_q = self.pd3.zeros()
        self.rhot_q[self.G3_G] = self.nt_Q * 8
        self.ghat.add(self.rhot_q, self.Q_aL)
        self.background_charge.add_fourier_space_charge_to(self.pd3,
                                                           self.rhot_q)
        self.rhot_q[0] = 0.0

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


def redistribute_array(nt_sG, gd1, gd2, nspins, kptband_comm):
    nt_sG = gd1.collect(nt_sG)
    new_nt_sG = gd2.empty(nspins)
    if kptband_comm.rank == 0:
        gd2.distribute(nt_sG, new_nt_sG)
    kptband_comm.broadcast(new_nt_sG, 0)
    return new_nt_sG


def redistribute_atomic_matrices(D_asp, gd2, nspins, setups, redistributor,
                                 kptband_comm):
    D_sP = pack_atomic_matrices(D_asp)
    natoms = len(setups)
    atom_partition = AtomPartition(gd2.comm, np.zeros(natoms, int),
                                   'density-gd')
    D_asp = setups.empty_atomic_matrix(nspins, atom_partition)
    spos_ac = np.zeros((natoms, 3))  # XXXX
    atomdist = redistributor.get_atom_distributions(spos_ac)

    if gd2.comm.rank == 0:
        if kptband_comm.rank > 0:
            nP = sum(setup.ni * (setup.ni + 1) // 2
                     for setup in setups)
            D_sP = np.empty((nspins, nP))
        kptband_comm.broadcast(D_sP, 0)
        D_asp.update(unpack_atomic_matrices(D_sP, setups))
        D_asp.check_consistency()
    return atom_partition, atomdist, D_asp
