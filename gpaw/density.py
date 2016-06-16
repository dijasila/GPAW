# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a density class."""

from math import pi, sqrt

import numpy as np

from gpaw import debug
from gpaw.mixer import get_mixer_from_keywords, MixerWrapper, DummyMixer
from gpaw.transformers import Transformer
from gpaw.lfc import LFC, BasisFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from gpaw.utilities import unpack2
from gpaw.utilities.partition import AtomPartition
from gpaw.utilities.timing import nulltimer
from gpaw.io import read_atomic_matrices
from gpaw.mpi import SerialCommunicator
from gpaw.arraydict import ArrayDict


class NullBackgroundCharge:
    charge = 0.0
    def set_grid_descriptor(self, gd): pass
    def add_charge_to(self, rhot_g): pass


class Density(object):
    """Density object.

    Attributes:
     =============== =====================================================
     ``gd``          Grid descriptor for coarse grids.
     ``finegd``      Grid descriptor for fine grids.
     ``interpolate`` Function for interpolating the electron density.
     ``mixer``       ``DensityMixer`` object.
     =============== =====================================================

    Soft and smooth pseudo functions on uniform 3D grids:
     ========== =========================================
     ``nt_sG``  Electron density on the coarse grid.
     ``nt_sg``  Electron density on the fine grid.
     ``nt_g``   Electron density on the fine grid.
     ``rhot_g`` Charge density on the fine grid.
     ``nct_G``  Core electron-density on the coarse grid.
     ========== =========================================
    """

    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 collinear=True, background_charge=None):
        """Create the Density object."""

        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.charge = float(charge)
        self.redistributor = redistributor
        self.atomdist = None

        self.collinear = collinear
        self.ncomp = 1 if collinear else 2
        self.ns = self.nspins * self.ncomp**2

        # This can contain e.g. a Jellium background charge
        if background_charge is None:
            background_charge = NullBackgroundCharge()
        background_charge.set_grid_descriptor(self.finegd)
        self.background_charge = background_charge

        self.charge_eps = 1e-7

        self.D_asp = None
        self.Q_aL = None

        self.nct_G = None
        self.nt_sG = None
        self.rhot_g = None
        self.nt_sg = None
        self.nt_g = None

        self.atom_partition = None

        # XXX at least one test will fail because None has no 'reset()'
        # So we need DummyMixer I guess
        self.set_mixer(DummyMixer())
        self.timer = nulltimer
        self.density_error = None

    def initialize(self, setups, timer, magmom_av, hund):
        self.timer = timer
        self.setups = setups
        self.hund = hund
        self.magmom_av = magmom_av

    def reset(self):
        # TODO: reset other parameters?
        self.nt_sG = None

    def set_positions_without_ruining_everything(self, spos_ac,
                                                 atom_partition):
        rank_a = atom_partition.rank_a
        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        if (self.atom_partition is not None and
            self.D_asp is None and (rank_a == self.gd.comm.rank).any()):
            self.D_asp = self.setups.empty_atomic_matrix(self.ns,
                                                         self.atom_partition)

        if (self.atom_partition is not None and self.D_asp is not None
            and not isinstance(self.gd.comm, SerialCommunicator)):
            self.timer.start('Redistribute')
            self.D_asp.redistribute(atom_partition)
            self.timer.stop('Redistribute')

        self.atom_partition = atom_partition
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)

    def set_positions(self, spos_ac, atom_partition):
        self.set_positions_without_ruining_everything(spos_ac, atom_partition)
        self.nct.set_positions(spos_ac)
        self.ghat.set_positions(spos_ac)
        self.mixer.reset()

        self.nt_sg = None
        self.nt_g = None
        self.rhot_g = None
        self.Q_aL = None

    def calculate_pseudo_density(self, wfs):
        """Calculate nt_sG from scratch.

        nt_sG will be equal to nct_G plus the contribution from
        wfs.add_to_density().
        """
        wfs.calculate_density_contribution(self.nt_sG)
        self.nt_sG[:self.nspins] += self.nct_G

    @property
    def D_asp(self):
        if self._D_asp is not None:
            assert isinstance(self._D_asp, ArrayDict), type(self._D_asp)
            self._D_asp.check_consistency()
        return self._D_asp

    @D_asp.setter
    def D_asp(self, value):
        if isinstance(value, dict):
            tmp = self.setups.empty_atomic_matrix(self.ns, self.atom_partition)
            tmp.update(value)
            value = tmp
        assert isinstance(value, ArrayDict) or value is None, type(value)
        if value is not None:
            value.check_consistency()
        self._D_asp = value

    def update(self, wfs):
        self.timer.start('Density')
        self.timer.start('Pseudo density')
        self.calculate_pseudo_density(wfs)
        self.timer.stop('Pseudo density')
        self.timer.start('Atomic density matrices')
        wfs.calculate_atomic_density_matrices(self.D_asp)
        self.timer.stop('Atomic density matrices')
        self.timer.start('Multipole moments')
        comp_charge = self.calculate_multipole_moments()
        self.timer.stop('Multipole moments')

        if isinstance(wfs, LCAOWaveFunctions):
            self.timer.start('Normalize')
            self.normalize(comp_charge)
            self.timer.stop('Normalize')

        self.timer.start('Mix')
        self.mix(comp_charge)
        self.timer.stop('Mix')
        self.timer.stop('Density')

    def normalize(self, comp_charge=None):
        """Normalize pseudo density."""
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()

        pseudo_charge = self.gd.integrate(self.nt_sG[:self.nspins]).sum()

        if (pseudo_charge + self.charge + comp_charge
            - self.background_charge.charge != 0):
            if pseudo_charge != 0:
                x = (self.background_charge.charge - self.charge
                     - comp_charge) / pseudo_charge
                self.nt_sG *= x
            else:
                # Use homogeneous background:
                volume = self.gd.get_size_of_global_array().prod() * self.gd.dv
                total_charge = (self.charge + comp_charge
                                - self.background_charge.charge)
                self.nt_sG[:self.nspins] = -total_charge / volume

    def mix(self, comp_charge):
        assert isinstance(self.mixer, MixerWrapper), self.mixer
        self.density_error = self.mixer.mix(self.nt_sG, self.D_asp)
        assert self.density_error is not None, self.mixer

        comp_charge = None
        self.interpolate_pseudo_density(comp_charge)
        self.calculate_pseudo_charge()

    def calculate_multipole_moments(self):
        """Calculate multipole moments of compensation charges.

        Returns the total compensation charge in units of electron
        charge, so the number will be negative because of the
        dominating contribution from the nuclear charge."""

        comp_charge = 0.0
        Ddist_asp = self.atomdist.to_aux(self.D_asp)
        def shape(a):
            return self.setups[a].Delta_pL.shape[1],
        self.Q_aL = ArrayDict(Ddist_asp.partition, shape)
        for a, D_sp in Ddist_asp.items():
            Q_L = self.Q_aL[a] = np.dot(D_sp[:self.nspins].sum(0),
                                        self.setups[a].Delta_pL)
            Q_L[0] += self.setups[a].Delta0
            comp_charge += Q_L[0]
        return Ddist_asp.partition.comm.sum(comp_charge) * sqrt(4 * pi)

    def get_initial_occupations(self, a):
        # distribute charge on all atoms
        # XXX interaction with background charge may be finicky
        c = (self.charge - self.background_charge.charge) / len(self.setups)
        M_v = self.magmom_av[a]
        M = (M_v**2).sum()**0.5
        f_si = self.setups[a].calculate_initial_occupation_numbers(
            M, self.hund, charge=c, nspins=self.nspins * self.ncomp)

        if self.collinear:
            if M_v[2] < 0:
                f_si = f_si[::-1].copy()
        else:
            f_i = f_si.sum(axis=0)
            fm_i = f_si[0] - f_si[1]
            f_si = np.zeros((4, len(f_i)))
            f_si[0] = f_i
            if M > 0:
                f_si[1:4] = np.outer(M_v / M, fm_i)
        return f_si

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

        self.D_asp = self.setups.empty_atomic_matrix(self.ns,
                                                     self.atom_partition)
        f_asi = {}
        for a in basis_functions.atom_indices:
            f_asi[a] = self.get_initial_occupations(a)

        # D_asp does not have the same distribution as the basis functions,
        # so we have to loop over atoms separately.
        for a in self.D_asp:
            f_si = f_asi.get(a)
            if f_si is None:
                f_si = self.get_initial_occupations(a)
            self.D_asp[a][:] = self.setups[a].initialize_density_matrix(f_si)

        self.nt_sG = self.gd.zeros(self.ns)
        basis_functions.add_to_density(self.nt_sG, f_asi)
        self.nt_sG[:self.nspins] += self.nct_G
        self.calculate_normalized_charges_and_mix()

    def initialize_from_wavefunctions(self, wfs):
        """Initialize D_asp, nt_sG and Q_aL from wave functions."""
        self.timer.start("Density initialize from wavefunctions")
        self.nt_sG = self.gd.empty(self.ns)
        self.calculate_pseudo_density(wfs)
        D_asp = self.setups.empty_atomic_matrix(self.ns,
                                                wfs.atom_partition)
        wfs.calculate_atomic_density_matrices(D_asp)
        self.D_asp = D_asp
        self.calculate_normalized_charges_and_mix()
        self.timer.stop("Density initialize from wavefunctions")

    def initialize_directly_from_arrays(self, nt_sG, D_asp):
        """Set D_asp and nt_sG directly."""
        self.nt_sG = nt_sG
        self.D_asp = D_asp
        D_asp.check_consistency()
        # No calculate multipole moments?  Tests will fail because of
        # improperly initialized mixer

    def calculate_normalized_charges_and_mix(self):
        comp_charge = self.calculate_multipole_moments()
        self.normalize(comp_charge)
        self.mix(comp_charge)

    def set_mixer(self, mixer):
        if mixer is None:
            mixer = {}
        if isinstance(mixer, dict):
            mixer = get_mixer_from_keywords(self.gd.pbc_c.any(), self.nspins,
                                            **mixer)
        if not hasattr(mixer, 'mix'):
            raise ValueError('Not a mixer: %s' % mixer)
        self.mixer = MixerWrapper(mixer, self.nspins, self.gd)

    def estimate_magnetic_moments(self):
        magmom_av = np.zeros_like(self.magmom_av)
        if self.nspins == 2 or not self.collinear:
            for a, D_sp in self.D_asp.items():
                if self.collinear:
                    magmom_av[a, 2] = np.dot(D_sp[0] - D_sp[1],
                                             self.setups[a].N0_p)
                else:
                    magmom_av[a] = np.dot(D_sp[1:4], self.setups[a].N0_p)
            self.gd.comm.sum(magmom_av)
        return magmom_av

    def get_correction(self, a, spin):
        """Integrated atomic density correction.

        Get the integrated correction to the pseuso density relative to
        the all-electron density.
        """
        setup = self.setups[a]
        return sqrt(4 * pi) * (
            np.dot(self.D_asp[a][spin], setup.Delta_pL[:, 0])
            + setup.Delta0 / self.nspins)

    def get_all_electron_density(self, atoms=None, gridrefinement=2,
                                 spos_ac=None, skip_core=False):
        """Return real all-electron density array.

           Usage: Either get_all_electron_density(atoms) or
                         get_all_electron_density(spos_ac=spos_ac)

           skip_core=True theoretically returns the
                          all-electron valence density (use with
                          care; will not in general integrate
                          to valence)
        """
        if spos_ac is None:
            spos_ac = atoms.get_scaled_positions() % 1.0

        # Refinement of coarse grid, for representation of the AE-density
        # XXXXXXXXXXXX think about distribution depending on gridrefinement!
        if gridrefinement == 1:
            gd = self.redistributor.aux_gd
            n_sg = self.nt_sG.copy()
            # This will get the density with the same distribution
            # as finegd:
            n_sg = self.redistributor.distribute(n_sg)
        elif gridrefinement == 2:
            gd = self.finegd
            if self.nt_sg is None:
                self.interpolate_pseudo_density()
            n_sg = self.nt_sg.copy()
        elif gridrefinement == 4:
            # Extra fine grid
            gd = self.finegd.refine()

            # Interpolation function for the density:
            interpolator = Transformer(self.finegd, gd, 3) # XXX grids!

            # Transfer the pseudo-density to the fine grid:
            n_sg = gd.empty(self.nspins)
            if self.nt_sg is None:
                self.interpolate_pseudo_density()
            for s in range(self.nspins):
                interpolator.apply(self.nt_sg[s], n_sg[s])
        else:
            raise NotImplementedError

        # Add corrections to pseudo-density to get the AE-density
        splines = {}
        phi_aj = []
        phit_aj = []
        nc_a = []
        nct_a = []
        for a, id in enumerate(self.setups.id_a):
            if id in splines:
                phi_j, phit_j, nc, nct = splines[id]
            else:
                # Load splines:
                phi_j, phit_j, nc, nct = self.setups[a].get_partial_waves()[:4]
                splines[id] = (phi_j, phit_j, nc, nct)
            phi_aj.append(phi_j)
            phit_aj.append(phit_j)
            nc_a.append([nc])
            nct_a.append([nct])

        # Create localized functions from splines
        phi = BasisFunctions(gd, phi_aj)
        phit = BasisFunctions(gd, phit_aj)
        nc = LFC(gd, nc_a)
        nct = LFC(gd, nct_a)
        phi.set_positions(spos_ac)
        phit.set_positions(spos_ac)
        nc.set_positions(spos_ac)
        nct.set_positions(spos_ac)

        I_sa = np.zeros((self.nspins, len(spos_ac)))
        a_W = np.empty(len(phi.M_W), np.intc)
        W = 0
        for a in phi.atom_indices:
            nw = len(phi.sphere_a[a].M_w)
            a_W[W:W + nw] = a
            W += nw

        x_W = phi.create_displacement_arrays()[0]
        #D_asp = self.atomic_matrix_distributor.distribute(self.D_asp)
        #D_asp = self.atomdist.to_aux(self.D_asp)
        D_asp = self.D_asp  # XXX really?

        rho_MM = np.zeros((phi.Mmax, phi.Mmax))
        for s, I_a in enumerate(I_sa):
            M1 = 0
            for a, setup in enumerate(self.setups):
                ni = setup.ni
                D_sp = D_asp.get(a)
                if D_sp is None:
                    D_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                else:
                    I_a[a] = ((setup.Nct) / self.nspins -
                              sqrt(4 * pi) *
                              np.dot(D_sp[s], setup.Delta_pL[:, 0]))

                    if not skip_core:
                        I_a[a] -= setup.Nc / self.nspins

                if gd.comm.size > 1:
                    gd.comm.broadcast(D_sp, D_asp.partition.rank_a[a])
                M2 = M1 + ni
                rho_MM[M1:M2, M1:M2] = unpack2(D_sp[s])
                M1 = M2

            assert np.all(n_sg[s].shape == phi.gd.n_c)
            phi.lfc.ae_valence_density_correction(rho_MM, n_sg[s], a_W, I_a,
                                                  x_W)
            phit.lfc.ae_valence_density_correction(-rho_MM, n_sg[s], a_W, I_a,
                                                   x_W)

        a_W = np.empty(len(nc.M_W), np.intc)
        W = 0
        for a in nc.atom_indices:
            nw = len(nc.sphere_a[a].M_w)
            a_W[W:W + nw] = a
            W += nw
        scale = 1.0 / self.nspins

        for s, I_a in enumerate(I_sa):

            if not skip_core:
                nc.lfc.ae_core_density_correction(scale, n_sg[s], a_W, I_a)

            nct.lfc.ae_core_density_correction(-scale, n_sg[s], a_W, I_a)
            gd.comm.sum(I_a)
            N_c = gd.N_c
            g_ac = np.around(N_c * spos_ac).astype(int) % N_c - gd.beg_c

            if not skip_core:

                for I, g_c in zip(I_a, g_ac):
                    if (g_c >= 0).all() and (g_c < gd.n_c).all():
                        n_sg[s][tuple(g_c)] -= I / gd.dv

        return n_sg, gd

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
        for name, obj in [('nct', self.nct),
                          ('ghat', self.ghat)]:
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

    def read(self, reader, parallel, kptband_comm):
        if reader['version'] > 0.3:
            self.density_error = reader['DensityError']

        if not reader.has_array('PseudoElectronDensity'):
            return

        hdf5 = hasattr(reader, 'hdf5')
        nt_sG = self.gd.empty(self.nspins)
        if hdf5:
            # Read pseudoelectron density on the coarse grid
            # and broadcast on kpt_comm and band_comm:
            indices = [slice(0, self.nspins)] + self.gd.get_slice()
            do_read = (kptband_comm.rank == 0)
            reader.get('PseudoElectronDensity', out=nt_sG, parallel=parallel,
                       read=do_read, *indices)  # XXX read=?
            kptband_comm.broadcast(nt_sG, 0)
        else:
            for s in range(self.nspins):
                self.gd.distribute(reader.get('PseudoElectronDensity', s),
                                   nt_sG[s])

        # Read atomic density matrices
        natoms = len(self.setups)
        atom_partition = AtomPartition(self.gd.comm, np.zeros(natoms, int),
                                       'density-gd')
        D_asp = self.setups.empty_atomic_matrix(self.ns, atom_partition)
        self.atom_partition = atom_partition  # XXXXXX
        spos_ac = np.zeros((natoms, 3)) # XXXX
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)

        all_D_sp = reader.get('AtomicDensityMatrices', broadcast=True)
        if self.gd.comm.rank == 0:
            D_asp.update(read_atomic_matrices(all_D_sp, self.setups))
            D_asp.check_consistency()

        self.initialize_directly_from_arrays(nt_sG, D_asp)


class RealSpaceDensity(Density):
    def __init__(self, gd, finegd, nspins, charge, redistributor,
                 collinear=True, stencil=3,
                 background_charge=None):
        Density.__init__(self, gd, finegd, nspins, charge, redistributor,
                         collinear=collinear,
                         background_charge=background_charge)
        self.stencil = stencil

    def initialize(self, setups, timer, magmom_av, hund):
        Density.initialize(self, setups, timer, magmom_av, hund)

        # Interpolation function for the density:
        self.interpolator = Transformer(self.redistributor.aux_gd,
                                        self.finegd, self.stencil)

        spline_aj = []
        for setup in setups:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        self.nct = LFC(self.gd, spline_aj,
                       integral=[setup.Nct for setup in setups],
                       forces=True, cut=True)
        self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups],
                        integral=sqrt(4 * pi), forces=True)

    def set_positions(self, spos_ac, rank_a=None):
        Density.set_positions(self, spos_ac, rank_a)
        self.nct_G = self.gd.zeros()
        self.nct.add(self.nct_G, 1.0 / self.nspins)

    def interpolate_pseudo_density(self, comp_charge=None):
        """Interpolate pseudo density to fine grid."""
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()

        self.nt_sg = self.distribute_and_interpolate(self.nt_sG, self.nt_sg)

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if not self.gd.pbc_c.all():
            # With zero-boundary conditions in one or more directions,
            # this is not the case.
            pseudo_charge = (self.background_charge.charge - self.charge
                             - comp_charge)
            if abs(pseudo_charge) > 1.0e-14:
                x = (pseudo_charge /
                     self.finegd.integrate(self.nt_sg[:self.nspins]).sum())
                self.nt_sg *= x

    def interpolate(self, in_xR, out_xR=None):
        """Interpolate array(s)."""

        # ndim will be 3 in finite-difference mode and 1 when working
        # with the AtomPAW class (spherical atoms and 1d grids)
        ndim = self.gd.ndim

        if out_xR is None:
            out_xR = self.finegd.empty(in_xR.shape[:-ndim])

        a_xR = in_xR.reshape((-1,) + in_xR.shape[-ndim:])
        b_xR = out_xR.reshape((-1,) + out_xR.shape[-ndim:])

        for in_R, out_R in zip(a_xR, b_xR):
            self.interpolator.apply(in_R, out_R)

        return out_xR

    def distribute_and_interpolate(self, in_xR, out_xR=None):
        in_xR = self.redistributor.distribute(in_xR)
        return self.interpolate(in_xR, out_xR)

    def calculate_pseudo_charge(self):
        self.nt_g = self.nt_sg[:self.nspins].sum(axis=0)
        self.rhot_g = self.nt_g.copy()
        self.ghat.add(self.rhot_g, self.Q_aL)
        self.background_charge.add_charge_to(self.rhot_g)

        if debug:
            charge = self.finegd.integrate(self.rhot_g) + self.charge
            if abs(charge) > self.charge_eps:
                raise RuntimeError('Charge not conserved: excess=%.9f' %
                                   charge)

    def get_pseudo_core_kinetic_energy_density_lfc(self):
        return LFC(self.gd,
                   [[setup.tauct] for setup in self.setups],
                   forces=True, cut=True)

    def calculate_dipole_moment(self):
        return self.finegd.calculate_dipole_moment(self.rhot_g)
