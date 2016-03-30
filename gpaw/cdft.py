import functools

from ase.calculators.calculator import Calculator
from ase.data import covalent_radii
from ase.units import Bohr, Hartree
from ase.utils import convert_string_to_fd
import numpy as np
from scipy.optimize import minimize

from gpaw.external import ExternalPotential
            
        
class CDFT(Calculator):
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, calc, regions, charges, potentials=None, txt='-',
                 tolerance=0.01):
        """Constrained DFT calculator.
        
        calc: GPAW instance
            DFT calculator object to be constrained.
        regions: list of list of int
            Atom indices of atoms in the different regions.
        charges: list of float
            constrained charges in the different regions.
        potentials: list of float
            Initial values for potential.
        txt: None or str or file descriptor
            Log file.  Default id '-' meaning standard out.  Use None for
            no output.
        tolerance: float
            Maximum accepted error in constrained charges.
        """
        Calculator.__init__(self)
        self.calc = calc
        self.charge_i = np.array(charges, dtype=float)
        self.tolerance = tolerance
        
        if potentials is None:
            self.v_i = 0.1 * np.sign(self.charge_i)
        else:
            self.v_i = np.array(potentials) / Hartree

        self.log = convert_string_to_fd(txt)
        
        self.ext = CDFTPotential(regions, txt=self.log)
        calc.set(external=self.ext)
            
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms)
        
        if 'positions' in system_changes:
            self.ext.set_positions_and_atomic_numbers(
                self.atoms.positions / Bohr, self.atoms.numbers)
            
        self.atoms.calc = self.calc
        
        p = functools.partial(print, file=self.log)
        
        def f(v_i):
            self.ext.set_levels(v_i)
            e = self.atoms.get_potential_energy() / Hartree
            e += np.dot(v_i, self.charge_i)
            dens = self.calc.density
            dn_i = (-dens.finegd.integrate(self.ext.w_ig, dens.rhot_g) -
                    self.charge_i)
            p('Potentials:', v_i * Hartree)
            p('Energy: {0:.6f} eV'.format(e * Hartree))
            p('Errors:', dn_i)
            return -e, dn_i

        m = minimize(f, self.v_i, jac=True,
                     options={'gtol': self.tolerance, 'norm': np.inf})
        assert m.success, m
        p(m.message)
        p('Iterations:', m.nfev)
        self.v_i = m.x
        self.dn_i = m.jac
        self.results['energy'] = -m.fun * Hartree
        self.results['forces'] = self.atoms.get_forces()


def gaussians(gd, positions, numbers):
    r_Rv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
    radii = covalent_radii[numbers]
    cutoffs = radii + 3.0
    sigmas = radii * min(covalent_radii) + 0.5
    result_R = gd.zeros()
    for pos, Z, rc, sigma in zip(positions, numbers, cutoffs, sigmas):
        d2_R = ((r_Rv - pos)**2).sum(3)
        a_R = Z / (sigma * (2 * np.pi)**0.5) * np.exp(-d2_R / (2 * sigma**2))
        a_R[d2_R > rc**2] = 0.0
        result_R += a_R
    return result_R
    
    
class CDFTPotential(ExternalPotential):
    def __init__(self, regions, txt='-'):
        self.indices_i = regions
        self.log = convert_string_to_fd(txt)
        self.v_i = None
        self.pos_av = None
        self.Z_a = None
        self.w_ig = None
        
    def __str__(self):
        return 'CDFTPotential({})'.format(self.indices_i)
        
    def set_levels(self, v_i):
        self.v_i = np.array(v_i, dtype=float)
        self.vext_g = None
        
    def set_positions_and_atomic_numbers(self, pos_av, Z_a):
        self.pos_av = pos_av
        self.Z_a = Z_a
        self.w_ig = None
        self.vext_g = None
        
    def initialize_partitioning(self, gd):
        self.w_ig = gd.empty(len(self.indices_i))
        ntot_g = gd.zeros()
        missing = list(range(len(self.Z_a)))

        print('Electrons:', file=self.log)
        for i, indices in enumerate(self.indices_i):
            n_g = gaussians(gd, self.pos_av[indices], self.Z_a[indices])
            print('atoms {0}: {1:.3f}'.format(indices, gd.integrate(n_g)),
                  file=self.log)
            ntot_g += n_g
            self.w_ig[i] = n_g
            for a in indices:
                missing.remove(a)
        
        ntot_g += gaussians(gd, self.pos_av[missing], self.Z_a[missing])
        ntot_g[ntot_g == 0] = 1.0
        self.w_ig /= ntot_g

        volume_i = gd.integrate(self.w_ig)
        print('Volumes:', file=self.log)
        for indices, volume in zip(self.indices_i, volume_i):
            print('atoms {0}: {1:.3f} Ang^3'.format(indices, volume * Bohr**3),
                  file=self.log)

    def calculate_potential(self, gd):
        if self.w_ig is None:
            self.initialize_partitioning(gd)
        self.vext_g = np.einsum('i,ijkl', self.v_i, self.w_ig)
