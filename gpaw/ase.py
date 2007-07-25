# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

from ASE.Units import units, Convert
import ASE

from gpaw.paw import PAW
from gpaw.mpi import parallel
from gpaw.mpi.paw import MPIPAW, get_parallel_environment


def gpaw(filename=None, **kwargs):
    if parallel or get_parallel_environment() is None:
        return ASEPAW(filename=filename, **kwargs)
    else:
        return MPIPAW(get_parallel_environment(), filename, **kwargs)

        
class ASEPAW(PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation.
    """

    def __init__(self, **kwargs):
        # Set units to ASE units:
        lengthunit = units.GetLengthUnit()
        energyunit = units.GetEnergyUnit()
        self.a0 = Convert(1, 'Bohr', lengthunit)
        self.Ha = Convert(1, 'Hartree', energyunit)

        # Convert from ASE units:
        if 'h' in kwargs:
            kwargs['h'] /= self.a0
        if 'width' in kwargs:
            kwargs['width'] /= self.Ha
        if 'external' in kwargs:
            kwargs['external'] = kwargs['external'] / self.Ha
        
        PAW.__init__(self, **kwargs)

        self.text('ASE: ', os.path.dirname(ASE.__file__))
        self.text('units:', lengthunit, 'and', energyunit)

    def GetPotentialEnergy(self, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        
        self.update()

        if force_consistent:
            # Free energy:
            return self.Ha * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return self.Ha * (self.Etot + 0.5 * self.occupation.S)

    def GetCartesianForces(self):
        """Return the forces for the current state of the ListOfAtoms."""
        self.update(forces=True)
        return self.F_ac * (self.Ha / self.a0)
      
    def GetStress(self):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def _SetListOfAtoms(self, atoms):
        """Make a weak reference to the ListOfAtoms."""
        self.lastcount = -1
        self.set_atoms(atoms)
        self.plot_atoms()

    def GetNumberOfBands(self):
        """Return the number of bands."""
        return self.nbands 
  
    def GetXCFunctional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def GetBZKPoints(self):
        """Return the k-points."""
        return self.k_kc
 
    def GetSpinPolarized(self):
        """Is it a spin-polarized calculation?"""
        return self.paw.nspins == 2
    
    def GetIBZKPoints(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.ibzk_kc

    # Alternative name:
    GetKPoints = GetIBZKPoints
 
    def GetIBZKPointWeights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.weights_k

    def GetDensityArray(self):
        """Return pseudo-density array."""
        return self.density.get_density_array() / self.a0**3

    def GetWaveFunctionArray(self, band=0, kpt=0, spin=0):
        """Return pseudo-wave-function array."""
        c =  1.0 / self.a0**1.5
        return self.get_wave_function_array(band, kpt, spin) * c

    def GetEigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        return self.get_eigenvalues(kpt, spin) * self.Ha

    def GetWannierLocalizationMatrix(self, G_I, kpoint, nextkpoint, spin,
                                     dirG, **args):
        """Calculate integrals for maximally localized Wannier functions."""

        c = dirG.index(1)
        return self.get_wannier_integrals(c, spin, kpoint, nextkpoint, G_I)

    def GetMagneticMoment(self):
        """Return the magnetic moment."""
        return self.magmom

    def GetFermiLevel(self):
        """Return the Fermi-level."""
        return self.occ.get_fermi_level()

    def GetElectronicStates(self):
        """Return electronic-state object."""
        from ASE.Utilities.ElectronicStates import ElectronicStates
        self.write('tmp27.nc')
        return ElectronicStates('tmp27.nc')
    
