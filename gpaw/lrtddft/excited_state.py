"""Excited state as calculator object."""

import numpy as np

from ase.units import Bohr, Hartree
from ase.calculators.general import Calculator
from ase.calculators.test import numeric_forces
from gpaw import GPAW
from gpaw.density import RealSpaceDensity
from gpaw.output import initialize_text_stream
from gpaw.mpi import rank, world
from gpaw.transformers import Transformer
from gpaw.utilities.blas import axpy
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class FiniteDifferenceCalculator(Calculator):
    def __init__(self, lrtddft, d=0.001, txt=None, parallel=None):
        """Finite difference calculator for LrTDDFT.

        parallel: Can be used to parallelize the numerical force 
        calculation over images
        """
        if lrtddft is not None:
            self.lrtddft = lrtddft
            self.calculator = self.lrtddft.calculator
            self.timer = self.calculator.timer
            self.set_atoms(self.calculator.get_atoms())

            if txt is None:
                self.txt = self.lrtddft.txt
            else:
                rank = world.rank
                self.txt, firsttime = initialize_text_stream(txt, rank)
                                                              
        self.d = d
        self.parallel = parallel

    def calculate(self, atoms):
        redo = self.calculator.calculate(atoms)
        E0 = self.calculator.get_potential_energy()
        lr = self.lrtddft
        if redo:
            if hasattr(self, 'density'):
                del(self.density)
            self.lrtddft.forced_update()
        self.lrtddft.diagonalize()
        return E0

    def set(self, **kwargs):
        self.calculator.set(**kwargs)

class ExcitedState(FiniteDifferenceCalculator,GPAW):
    def __init__(self, lrtddft, index, d=0.001, txt=None,
                 parallel=None, name=None):
        """ExcitedState object.

        parallel: Can be used to parallelize the numerical force calculation 
        over images.
        """
        FiniteDifferenceCalculator.__init__(self, lrtddft, d, txt, parallel)

        if type(index) == type(1):
            self.index = UnconstraintIndex(index)
        else:
            self.index = index
        self.name = name

        self.energy = None
        self.forces = None

        print >> self.txt, 'ExcitedState', self.index,
        if name:
            print >> self.txt, ('name=' + name),
        print >> self.txt
 
    def calculation_required(self, atoms, quantities):
        if len(quantities) == 0:
            return False
        if atoms != self.atoms:
            return True
        for quantity in ['energy', 'forces']:
            if quantity in quantities:
                quantities.remove(quantity)
                if self.__dict__[quantity] is None:
                    return True
        return len(quantities) > 0

    def get_potential_energy(self, atoms=None):
        """Evaluate potential energy for the given excitation."""
        if atoms is None:
            atoms = self.atoms
        if self.calculation_required(atoms, ['energy']):
            if atoms is None or atoms == self.atoms:
                self.energy = self.calculate(atoms)
                return self.energy
            else:
                return self.calculate(atoms)
        else:
            return self.energy

    def calculate(self, atoms):
        E0 = FiniteDifferenceCalculator.calculate(self, atoms)
        index = self.index.apply(self.lrtddft)
        return E0 + self.lrtddft[index].energy * Hartree

    def get_forces(self, atoms):
        """Get finite-difference forces"""
        if self.calculation_required(atoms, ['forces']):
            atoms.set_calculator(self)
            name = self.name
            if name:
                name += 'force'
            self.forces = numeric_forces(atoms, d=self.d, 
                                         parallel=self.parallel,
                                         name=name)
            if self.txt:
                print >> self.txt, 'Excited state forces in eV/Ang:'
                symbols = self.atoms.get_chemical_symbols()
                for a, symbol in enumerate(symbols):
                    print >> self.txt, ('%3d %-2s %10.5f %10.5f %10.5f' %
                                        ((a, symbol) + tuple(self.forces[a])))
        return self.forces

    def get_stress(self, atoms):
        """Return the stress for the current state of the Atoms."""
        raise NotImplementedError

    def initialize_density(self, method='dipole'):
        if hasattr(self, 'density') and self.density.method == method:
            return

        gsdensity = self.calculator.density
        lr = self.lrtddft
        self.density = ExcitedStateDensity(
            gsdensity.gd, gsdensity.finegd, lr.kss.npspins,
            gsdensity.charge,
            method=method)
        index = self.index.apply(self.lrtddft)
        energy = self.get_potential_energy()
        self.density.initialize(self.lrtddft, index)
        self.density.update(self.calculator.wfs)

    def get_pseudo_density(self, *args, **kwargs):
        """Return pseudo-density array."""
        method = kwargs.pop('method', 'dipole')
        self.initialize_density(method)
        return GPAW.get_pseudo_density(self, *args, **kwargs)

    def get_all_electron_density(self, *args, **kwargs):
        """Return all electron density array."""
        method = kwargs.pop('method', 'dipole')
        self.initialize_density(method)
        return GPAW.get_all_electron_density(self, *args, **kwargs)

class UnconstraintIndex:
    def __init__(self, index):
        assert(type(index) == type(1))
        self.index = index
    def apply(self, *argv):
        return self.index

class MinimalOSIndex:
    """
    Constraint on minimal oscillator strength.

    Searches for the first excitation that has a larger
    oscillator strength than the given minimum.

    direction:
        None: averaged (default)
        0, 1, 2: x, y, z
    """
    def __init__(self, fmin=0.02, direction=None):
        self.fmin = fmin
        self.direction = direction

    def apply(self, lrtddft):
        index = None
        i = 0
        fmax = 0.
        idir = 0
        if self.direction is not None:
            idir = 1 + self.direction
        while i < len(lrtddft):
            ex = lrtddft[i]
            f = ex.get_oscillator_strength()[idir]
            fmax = max(f, fmax)
            if f > self.fmin:
                return i
            i += 1
        error = 'The intensity constraint |f| > ' + str(self.fmin) + ' '
        error += 'can not be satisfied (max(f) = ' + str(fmax) + ').'
        raise RuntimeError(error)
        
class MaximalOSIndex:
    """
    Select maximal oscillator strength.

    Searches for the excitation with maximal oscillator strength
    in a given energy range.

    energy_range:
        None: take all (default)
        [Emin, Emax]: take only transition in this energy range
        Emax: the same as [0, Emax]
    direction:
        None: averaged (default)
        0, 1, 2: x, y, z
    """
    def __init__(self, energy_range=None, direction=None):
        if energy_range is None:
            energy_range = np.array([0.0, 1.e32])
        elif isinstance(energy_range, (int, long, float)):
            energy_range = np.array([0.0, energy_range]) / Hartree
        self.energy_range = energy_range

        self.direction = direction

    def apply(self, lrtddft):
        index = None
        fmax = 0.
        idir = 0
        if self.direction is not None:
            idir = 1 + self.direction
        emin, emax = self.energy_range
        for i, ex in enumerate(lrtddft):
            f = ex.get_oscillator_strength()[idir]
            e = ex.get_energy()
            if e >= emin and e < emax and f > fmax:
                fmax = f
                index = i
        if index is None:
            raise RuntimeError('No transition in the energy range ' +
                               '[%g,%g]' % self.energy_range)
        return index

class ExcitedStateDensity(RealSpaceDensity):
    """Approximate excited state density object."""
    def __init__(self, *args, **kwargs):
        self.method = kwargs.pop('method', 'dipole')
        RealSpaceDensity.__init__(self, *args, **kwargs)

    def initialize(self, lrtddft, index):
        self.lrtddft = lrtddft
        self.index = index

        calc = lrtddft.calculator
        self.gsdensity = calc.density
        self.gd = self.gsdensity.gd
        self.nbands =  calc.wfs.bd.nbands
        self.D_asp = {}
        for a, D_sp in self.gsdensity.D_asp.items():
            self.D_asp[a] = 1. *  D_sp
        
        # obtain weights
        ex = lrtddft[index]
        energy = ex.energy
        wocc_sn = np.zeros((self.nspins, self.nbands))
        wunocc_sn = np.zeros((self.nspins, self.nbands))
        for f, k in zip(ex.f, ex.kss):
            # XXX why not k.fij * k.energy / energy ???
            if self.method == 'dipole':
                erat = k.energy / ex.energy
            elif self.method == 'orthogonal':
                erat = 1.
            else:
                raise NotImplementedError(
                    'method should be either "dipole" or "orthogonal"')
            wocc_sn[k.pspin, k.i] += erat * f**2
            wunocc_sn[k.pspin, k.j] += erat * f**2
        self.wocc_sn = wocc_sn
        self.wunocc_sn = wunocc_sn

        RealSpaceDensity.initialize(
            self, calc.wfs.setups, calc.timer, None, False)

        spos_ac = calc.get_atoms().get_scaled_positions() % 1.0
        self.set_positions(spos_ac, calc.wfs.rank_a)

    def update(self, wfs):
        self.timer.start('Density')
        self.timer.start('Pseudo density')
        self.calculate_pseudo_density(wfs)
        self.timer.stop('Pseudo density')
        self.timer.start('Atomic density matrices')
        f_un = []
        for kpt in wfs.kpt_u:
            f_n = kpt.f_n - self.wocc_sn[kpt.s] + self.wunocc_sn[kpt.s]
            if self.nspins > self.gsdensity.nspins:
                f_n = kpt.f_n - self.wocc_sn[1] + self.wunocc_sn[1]
            f_un.append(f_n)
        wfs.calculate_atomic_density_matrices_with_occupation(self.D_asp,
                                                              f_un)
        self.timer.stop('Atomic density matrices')
        self.timer.start('Multipole moments')
        comp_charge = self.calculate_multipole_moments()
        self.timer.stop('Multipole moments')
        
        if isinstance(wfs, LCAOWaveFunctions):
            self.timer.start('Normalize')
            self.normalize(comp_charge)
            self.timer.stop('Normalize')

        self.timer.stop('Density')

    def calculate_pseudo_density(self, wfs):
        """Calculate nt_sG from scratch.

        nt_sG will be equal to nct_G plus the contribution from
        wfs.add_to_density().
        """
        nvspins = wfs.kd.nspins
        npspins = self.nspins
        self.nt_sG = self.gd.zeros(npspins)

        for s in range(npspins):
            for kpt in wfs.kpt_u:
                if s == kpt.s or npspins > nvspins:
                    f_n = kpt.f_n / (1. + int(npspins > nvspins))
                    for f, psit_G in zip((f_n - self.wocc_sn[s] +
                                          self.wunocc_sn[s]),
                                         kpt.psit_nG):
                        axpy(f, psit_G**2, self.nt_sG[s])
        self.nt_sG[:self.nspins] += self.nct_G
