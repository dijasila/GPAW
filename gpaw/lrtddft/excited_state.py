"""Excited state as calculator object."""

import numpy as np

from ase.units import Bohr, Hartree
from ase.calculators.general import Calculator
from ase.calculators.test import numeric_forces
from gpaw import GPAW
from gpaw.output import initialize_text_stream
from gpaw.mpi import rank
from gpaw.transformers import Transformer
from gpaw.utilities.blas import axpy


class FiniteDifferenceCalculator(Calculator):
    def __init__(self, lrtddft, d=0.001, txt=None, parallel=None):
        """Finite difference calculator for LrTDDFT.

        parallel: Can be used to parallelize the numerical force 
        calculation over images
        """
        if lrtddft is not None:
            self.lrtddft = lrtddft
            self.calculator = self.lrtddft.calculator
            self.set_atoms(self.calculator.get_atoms())

            if txt is None:
                self.txt = self.lrtddft.txt
            else:
                rank = self.calculator.wfs.world.rank
                self.txt, firsttime = initialize_text_stream(txt, rank)
                                                              
        self.d = d
        self.parallel = parallel

    def calculate(self, atoms):
        redo = self.calculator.calculate(atoms)
        E0 = self.calculator.get_potential_energy()
        lr = self.lrtddft
        if redo:
            self.lrtddft.forced_update()
        self.lrtddft.diagonalize()
        return E0

    def set(self, **kwargs):
        self.calculator.set(**kwargs)

class ExcitedState(FiniteDifferenceCalculator):
    def __init__(self, lrtddft, index, d=0.001, txt=None,
                 parallel=None):
        """ExcitedState object.

        parallel: Can be used to parallelize the numerical force calculation over
        images.
        """
        FiniteDifferenceCalculator.__init__(self, lrtddft, d, txt, parallel)

        if type(index) == type(1):
            self.index = UnconstraintIndex(index)
        else:
            self.index = index

        self.energy = None
        self.forces = None
        
        print >> self.txt, 'ExcitedState', self.index
 
    def get_potential_energy(self, atoms=None):
        """Evaluate potential energy for the given excitation."""
        if atoms is None:
            atoms = self.atoms
            self.energy = self.calculate(atoms)
        if (self.energy is None) or atoms != self.atoms:  
            energy = self.calculate(atoms)
            return energy
        else:
            return self.energy

    def calculate(self, atoms):
        E0 = FiniteDifferenceCalculator.calculate(self, atoms)
        index = self.index.apply(self.lrtddft)
        return E0 + self.lrtddft[index].energy * Hartree

    def get_forces(self, atoms):
        """Get finite-difference forces"""
        if (self.forces is None) or atoms != self.atoms:
            atoms.set_calculator(self)
            self.forces = numeric_forces(atoms, d=self.d, 
                                         parallel=self.parallel)
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

    def get_pseudo_density(self, spin=None, gridrefinement=1,
                           pad=True, broadcast=True):
        """Return pseudo-density array.

        If *spin* is not given, then the total density is returned.
        Otherwise, the spin up or down density is returned (spin=0 or
        1)."""

        calc = self.calculator
        gd = calc.density.gd # XXX get from a better place
        npspins = self.lrtddft.kss.npspins
        nvspins = self.lrtddft.kss.nvspins
        nt_sG = gd.zeros(npspins)
        nbands = calc.wfs.bd.nbands

        # obtain weights
        ex = self.lrtddft[self.index.apply(self.lrtddft)]
        wocc_sn = np.zeros((npspins, nbands))
        wunocc_sn = np.zeros((npspins, nbands))
        energy = self.get_potential_energy() / Hartree
        for f, k in zip(ex.f, ex.kss):
            # XXX why not k.fij * k.energy / energy ???
            erat = k.energy / energy
            wocc_sn[k.pspin, k.i] += erat * f**2
            wunocc_sn[k.pspin, k.j] += erat * f**2

        # sum up
        for s in range(npspins):
            for kpt in calc.wfs.kpt_u:
                if s == kpt.s or npspins > nvspins:
                    f_n = kpt.f_n / (1. + int(npspins > nvspins))
                    for fo, fu, psit_G in zip(f_n - wocc_sn[s],
                                              wunocc_sn[s],
                                              kpt.psit_nG):
                        axpy(fo, psit_G**2, nt_sG[s])
                        axpy(fu, psit_G**2, nt_sG[s])
                         
        if gridrefinement == 1:
            pass
        elif gridrefinement == 2:
            finegd = self.calculator.density.finegd
            nt_sg = finegd.empty(npspins)
            interpolator = Transformer(gd, finegd, calc.density.stencil)
            for nt_G, nt_g in zip(nt_sG, nt_sg):
                interpolator.apply(nt_G, nt_g)
            nt_sG = nt_sg
        else:
            raise NotImplementedError

        if spin is None:
            if npspins == 1:
                nt_G = nt_sG[0]
            else:
                nt_G = nt_sG.sum(axis=0)
        else:
            if npspins == 1:
                nt_G = 0.5 * nt_sG[0]
            else:
                nt_G = nt_sG[spin]

        if pad:
            nt_G = gd.zero_pad(nt_G)

        return nt_G / Bohr**3

class UnconstraintIndex:
    def __init__(self, index):
        assert(type(index) == type(1))
        self.index = index
    def apply(self, *argv):
        return self.index

class MinimalOSIndex:
    """
    Constraint on minimal oscillator strength.

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
        while i < len(lrtddft):
            ex = lrtddft[i]
            idir = 0
            if self.direction is not None:
                idir = 1 + self.direction
            f = ex.get_oscillator_strength()[idir]
            fmax = max(f, fmax)
            if f > self.fmin:
                return i
            i += 1
        error = 'The intensity constraint |f| > ' + str(self.fmin) + ' '
        error += 'can not be satisfied (max(f) = ' + str(fmax) + ').'
        raise RuntimeError(error)
        
