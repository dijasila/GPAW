# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a density class."""

from math import pi, sqrt

import numpy as np

from gpaw import debug
from gpaw.mixer import Mixer, MixerSum
from gpaw.transformers import Transformer
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions import LCAOWaveFunctions


class Density:
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
    
    def __init__(self, gd, finegd, nspins, charge):
        """Create the Density object."""

        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.charge = charge

        self.charge_eps = 1e-7
        self.D_asp = None

        self.nct_G = None
        self.nt_sG = None
        self.nt_g = None
        self.rhot_g = None
        self.nt_sg = None
        self.nt_g = None

    def initialize(self, setups, stencil, timer, magmom_a, hund):
        self.timer = timer
        self.setups = setups
        self.hund = hund
        self.magmom_a = magmom_a
        
        # Interpolation function for the density:
        self.interpolate = Transformer(self.gd, self.finegd, stencil).apply

        self.nct = LFC(self.gd, [[setup.nct] for setup in setups],
                       integral=[setup.Nct for setup in setups], forces=True)
        self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups],
                        integral=sqrt(4 * pi), forces=True)

    def set_positions(self, spos_ac):
        self.nct.set_positions(spos_ac)
        self.ghat.set_positions(spos_ac)
        self.mixer.reset()

        self.nct_G = self.gd.zeros()
        self.nct.add(self.nct_G)
        self.nt_sG = None
        self.nt_sg = None
        self.nt_g = None
        self.rhot_g = None
        self.D_asp = {}
        self.Q_aL = {}
        for a in self.nct.my_atom_indices:
            ni = self.setups[a].ni
            self.D_asp[a] = np.empty((self.nspins, ni * (ni + 1) // 2))
            lmax = self.setups[a].lmax
            self.Q_aL[a] = np.empty((lmax + 1)**2)

    def update(self, wfs, basis_functions=None):
        if self.nt_sG is None:
            self.nt_sG = self.gd.zeros(self.nspins)
        else:
            self.nt_sG[:] = 0.0

        if basis_functions:
            self.initialize_from_atomic_densities(basis_functions)
        else:
            wfs.add_to_density(self.nt_sG, self.D_asp)

        self.nt_sG += self.nct_G

        comp_charge = self.calculate_multipole_moments()
        
        if basis_functions or isinstance(wfs, LCAOWaveFunctions):
            pseudo_charge = self.gd.integrate(self.nt_sG).sum()
            if pseudo_charge != 0:
                x = -(self.charge + comp_charge) / pseudo_charge
                self.nt_sG *= x

        if not self.mixer.mix_rho:
            self.mixer.mix(self)
            comp_charge = self.calculate_multipole_moments()
            
        if self.nt_sg is None:
            self.nt_sg = self.finegd.empty(self.nspins)

        for s in range(self.nspins):
            self.interpolate(self.nt_sG[s], self.nt_sg[s])

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if not self.gd.domain.pbc_c.all():
            # With zero-boundary conditions in one or more directions,
            # this is not the case.
            pseudo_charge = -(self.charge + comp_charge)
            if pseudo_charge != 0:
                x = pseudo_charge / self.finegd.integrate(self.nt_sg).sum()
                self.nt_sg *= x

        self.nt_g = self.nt_sg.sum(axis=0)
        self.rhot_g = self.nt_g.copy()
        self.ghat.add(self.rhot_g, self.Q_aL)
        if debug:
            charge = self.finegd.integrate(self.rhot_g) + self.charge
            if abs(charge) > self.charge_eps:
                raise RuntimeError('Charge not conserved: excess=%.9f' %
                                   charge)

        if self.mixer.mix_rho:
            self.mixer.mix(self)

    def calculate_multipole_moments(self):
        comp_charge = 0.0
        for a, D_sp in self.D_asp.items():
            Q_L = self.Q_aL[a]
            Q_L[:] = np.dot(D_sp.sum(0), self.setups[a].Delta_pL)
            Q_L[0] += self.setups[a].Delta0
            comp_charge += Q_L[0]
        return self.gd.comm.sum(comp_charge) * sqrt(4 * pi)

    def initialize_from_atomic_densities(self, basis_functions):
        """Initialize density from atomic densities.

        The density is initialized from atomic orbitals, and will
        be constructed with the specified magnetic moments and
        obeying Hund's rules if ``hund`` is true."""
        
        f_sM = np.empty((self.nspins, basis_functions.Mmax))
        self.D_asp = {}
        M1 = 0
        for a in basis_functions.atom_indices:
            f_si = self.setups[a].calculate_initial_occupation_numbers(
                    self.magmom_a[a], self.hund)

            if 1:
                assert self.gd.comm.size == 1
                self.D_asp[a] = self.setups[a].initialize_density_matrix(f_si)

            M2 = M1 + f_si.shape[1]
            f_sM[:, M1:M2] = f_si
            M1 = M2

        basis_functions.add_to_density(self.nt_sG, f_sM)

    def set_mixer(self, mixer, fixmom, width):
        if mixer is not None:
            if self.nspins == 2 and (not fixmom or width != 0):
                if isinstance(mixer, Mixer):
                    raise RuntimeError("""Cannot use Mixer in spin-polarized
                    calculations with fixed moment or finite Fermi-width""")
            self.mixer = mixer
        else:
            if self.nspins == 2 and (not fixmom or width != 0):
                self.mixer = MixerSum()#mix, self.gd)
            else:
                self.mixer = Mixer()#mix, self.gd, self.nspins)

        if self.nspins == 1 and isinstance(mixer, MixerSum):
            raise RuntimeError('Cannot use MixerSum with nspins==1')

        self.mixer.initialize(self)
        
    def calculate_magnetic_moments(self):
        if self.nspins == 1:
            assert not self.magmom_a.any()
        else:
            self.magmom_a.fill(0.0)
            for a, D_sp in self.D_asp.items():
                self.magmom_a[a] = np.dot(D_sp[0] - D_sp[1],
                                          self.setups[a].N0_p)
            self.gd.comm.sum(self.magmom_a)
        return self.magmom_a

    def get_density_array(self):
        XXX
        # XXX why not replace with get_spin_density and get_total_density?
        """Return pseudo-density array."""
        if self.nspins == 2:
            return self.nt_sG
        else:
            return self.nt_sG[0]
    
    def get_all_electron_density(self, gridrefinement=2, collect=True):
        """Return real all-electron density array."""

        # Refinement of coarse grid, for representation of the AE-density
        if gridrefinement == 1:
            gd = self.gd
            n_sg = self.nt_sG.copy()
        elif gridrefinement == 2:
            gd = self.finegd
            n_sg = self.nt_sg.copy()
        elif gridrefinement == 4:
            # Extra fine grid
            gd = self.finegd.refine()
            
            # Interpolation function for the density:
            interpolator = Transformer(self.finegd, gd, 3)

            # Transfer the pseudo-density to the fine grid:
            n_sg = gd.empty(self.nspins)
            for s in range(self.nspins):
                interpolator.apply(self.nt_sg[s], n_sg[s])
        else:
            raise NotImplementedError

        # Add corrections to pseudo-density to get the AE-density
        splines = {}
        for nucleus in self.nuclei:
            nucleus.add_density_correction(n_sg, self.nspins, gd, splines)

        if collect:
            n_sg = gd.collect(n_sg)

        # Return AE-(spin)-density
        if self.nspins == 2 or n_sg is None:
            return n_sg
        else:
            return n_sg[0]

    def initialize_kinetic(self):
        """Initial pseudo electron kinetic density."""
        """flag to use local variable in tpss.c"""

        self.taut_sG = self.gd.zeros(self.nspins)
        self.taut_sg = self.finegd.zeros(self.nspins)

    def update_kinetic(self, kpt_u):
        """Calculate pseudo electron kinetic density.
        The pseudo electron-density ``taut_sG`` is calculated from the
        wave functions, the occupation numbers"""

        ## Add contribution from all k-points:
        for kpt in kpt_u:
            kpt.add_to_kinetic_density(self.taut_sG[kpt.s])
        self.band_comm.sum(self.taut_sG)
        self.kpt_comm.sum(self.taut_sG)
        """Add the pseudo core kinetic array """
        for nucleus in self.nuclei:
            nucleus.add_smooth_core_kinetic_energy_density(self.taut_sG,
                                                           self.nspins,
                                                           self.gd)

        """Transfer the density from the coarse to the fine grid."""
        for s in range(self.nspins):
            self.interpolate(self.taut_sG[s], self.taut_sg[s])

        return 
