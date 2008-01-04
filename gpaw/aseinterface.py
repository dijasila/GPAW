# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

import os
import weakref

import numpy as npy
import ase
from ase.parallel import register_parallel_cleanup_function
from ase.units import Bohr, Hartree

from gpaw.paw import PAW

register_parallel_cleanup_function()


class Calculator(PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation.
    """

    def __init__(self, filename=None, **kwargs):
        # Set units to ASE units:
        self.a0 = Bohr
        self.Ha = Hartree

        PAW.__init__(self, filename, **kwargs)

        self.text('ase: ', os.path.dirname(ase.__file__))
        self.text('numpy:', os.path.dirname(npy.__file__))
        self.text('units: Bohr and Hartree')

    def convert_units(self, parameters):
        if parameters.get('h') is not None:
            parameters['h'] /= self.a0
        if parameters.get('width') is not None:
            parameters['width'] /= self.Ha
        if parameters.get('external') is not None:
            parameters['external'] = parameters['external'] / self.Ha
        if ('convergence' in parameters and
            'energy' in  parameters['convergence']):
            parameters['convergence']['energy'] /= self.Ha
        
    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        
        if atoms is None:
            atoms = self.atoms_from_file

        self.calculate(atoms)

        if force_consistent:
            # Free energy:
            return self.Ha * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return self.Ha * (self.Etot + 0.5 * self.S)

    def get_forces(self, atoms):
        """Return the forces for the current state of the ListOfAtoms."""
        if self.F_ac is None:
            if hasattr(self, 'nuclei') and not self.nuclei[0].ready:
                self.converged = False
        self.calculate(atoms)
        self.calculate_forces()
        return self.F_ac * (self.Ha / self.a0)
      
    def get_stress(self):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    """
    def _SetListOfAtoms(self, atoms):
        ""Make a weak reference to the ListOfAtoms.""
        self.lastcount = -1
        self.atoms = weakref.proxy(atoms)
        self.extra_list_of_atoms_stuff = (atoms.GetTags(),
                                          atoms.GetMagneticMoments())
        self.plot_atoms()
    """
    
    def get_number_of_bands(self):
        """Return the number of bands."""
        return self.nbands 
  
    def get_xc_functional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def get_bz_k_points(self):
        """Return the k-points."""
        return self.bzk_kc
 
    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        return self.nspins == 2
    
    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.ibzk_kc

    def get_ibz_k_point_weights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.weight_k

    def get_pseudo_valence_density(self):
        """Return pseudo-density array."""
        return self.density.get_density_array() / self.a0**3

    def get_electron_density(self, gridrefinement=2):
        """Return reconstructed all-electron density array."""
        return self.density.get_all_electron_density(gridrefinement)\
               / self.a0**3

    def GetWignerSeitzDensities(self, spin):
        if not hasattr(self, 'wignerseitz'):
            from gpaw.analyse.wignerseitz import WignerSeitz
            self.wignerseitz = WignerSeitz(self.gd, self.nuclei)
        
        return self.wignerseitz.expand_density(self.density.nt_sG[spin],
                                               spin, self.nspins)

    def GetWignerSeitzLDOS(self, a, spin, npts=201, width=None):
        if width is None:
            width = self.GetElectronicTemperature()
        if width == 0:
            width = 0.1

        from gpaw.utilities.dos import raw_wignerseitz_LDOS, fold_ldos
        energies, weights = raw_wignerseitz_LDOS(self, a, spin)
        return fold_ldos(energies, weights, npts, width)        
    
    def GetOrbitalLDOS(self, a, spin, angular, npts=201, width=None):
        if width is None:
            width = self.GetElectronicTemperature()
        if width == 0.0:
            width = 0.1

        from gpaw.utilities.dos import raw_orbital_LDOS, fold_ldos
        energies, weights = raw_orbital_LDOS(self, a, spin, angular)
        return fold_ldos(energies, weights, npts, width)

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0):
        """Return pseudo-wave-function array."""
        return self.get_wave_function_array(band, kpt, spin) / self.a0**1.5

    def get_eigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        result = PAW.get_eigenvalues(self, kpt, spin)
        if result is not None:
            return result * self.Ha

    def GetWannierLocalizationMatrix(self, nbands, dirG, kpoint,
                                     nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        c = dirG.index(1)
        kpts = self.GetBZKPoints()
        G = kpts[nextkpoint, c] - kpts[kpoint, c] + G_I[c]

        return self.get_wannier_integrals(c, spin, kpoint, nextkpoint, G)

    def get_magnetic_moment(self):
        """Return the magnetic moment."""
        return self.occupation.magmom

    def get_fermi_level(self):
        """Return the Fermi-level."""
        e = self.occupation.get_fermi_level()
        if e is None:
            # Zero temperature calculation - return vacuum level:
            e = 0.0
        return e * self.Ha

    def get_grid_spacings(self):
        return self.a0 * self.gd.h_c

    def get_number_of_grid_points(self):
        return self.gd.N_c

    def GetEnsembleCoefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.GetPotentialEnergy()
        E0 = self.get_xc_difference('XC-9-1.0')
        coefs = (E + E0,
                 self.get_xc_difference('XC-0-1.0') - E0,
                 self.get_xc_difference('XC-1-1.0') - E0,
                 self.get_xc_difference('XC-2-1.0') - E0)
        self.text('BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs)
        return npy.array(coefs)

    def get_electronic_temperature(self):
        return self.kT * self.Ha
