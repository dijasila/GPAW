import numpy as np

import gpaw.mpi
from gpaw import restart

import ase.io.ulm as ulm

from ase import Atoms

from scipy.ndimage.interpolation import shift

import pointgroups

"""
This module contains the SymmetryCalculator class and interfaces
to GPAW.

PGS refers to Point Group Symmetry.

The code was originally written for the paper
S. Kaappa, S. Malola, H. Hakkinen;  arXiv:1808.01868 [physics.atm-clus]
https://arxiv.org/abs/1808.01868
"""


class SymmetryCalculator:
    """
    General calculator to assign symmetry representations of a specific
    point group to real-space wave functions.

    Parameters
    ----------
    filename : string
        GPAW restart file that containss the wave functions
    statelist : array
        List of states to be analysed
    pointgroup : string
        The name of the point group for which the analysis is done
    mpi : communicator, optional
        Communicator for parallel calculations
    overlapfile : string, optional
        output file for the overlap integrals
    symmetryfile : string, optional
        Output file for symmetry representation weights for different bands

    """

    def __init__(self,
                 filename,
                 statelist,
                 pointgroup,
                 mpi=None,
                 overlapfile='overlaps.txt',
                 symmetryfile='symmetries.txt'):
        self.filename = filename
        self.pointgroup = pointgroups.list_of_pointgroups[pointgroup]()
        self.statelist = statelist
        self.n = len(statelist)
        self.overlapfile = overlapfile
        self.symmetryfile = symmetryfile

        self.h = np.ones(3, dtype=float)
        self.initial_rotations = []
        self.cutarea = None
        self.shiftvector = np.zeros(3)

        self.overlaps = np.zeros((self.n, len(self.pointgroup.operations)))
        self.fullnorms = np.ndarray([self.n, 1])
        self.cutnorms = np.ndarray([self.n, 1])
        self.norms_calculated = False

        self.mpi = mpi

    def initialize(self):
        self.load_data()
        return

    def load_data(self):
        return

    def get_energy(self, band):
        """
        Energies are only used while writing the output.
        """
        return 0.

    def get_wf(self, band):
        """
        Return a wave function of a given index.

        A single wave function should be of type np.ndarray([gx,gy,gz])
        where g are numbers of grid points along each direction.
        """
        return np.ones([1, 1, 1])

    def set_gridspacing(self, h):
        """ For normalization and correct length units,
        the grid spacing of the wave function data should be defined.

        Parameters
        ----------
        h : array_like
            Grid spacing as a 3-element array consisting of
            the grid spacings to each direction x,y,z."""
        self.h = h
        return self.h

    def set_cutarea(self, array):
        """
        Parameters
        ----------
        array : array_like
            should be of the same shape as the wave functions
            consisting of ones and zeros corresponding to
            the to-be-analyzed volume. """

        self.cutarea = np.array(array)
        return self.cutarea

    def set_shiftvector(self, vector):
        """
        Parameters
        ----------
        vector : array_like
            Should be 3-element array in length units (eg. angstroms).
        """

        self.shiftvector = np.array(vector)
        return self.shiftvector

    def set_initialrotations(self, Rx, Ry, Rz):
        """Before the actual symmetry analysis, the wave functions
        are rotated so that
        the main axis becomes parallel to z-axis and
        one of the secondary axis becomes parallel to x-axis.

        Parameters
        ----------
        Rx : float
            Rotation angle around x-axis in degrees
        Ry : float
            Rotation angle around y-axis in degrees
        Rz : float
            Rotation angle around z-axis in degrees

        """

        self.initial_rotations = [self.pointgroup.rotate(angle=Rx, axis='x'),
                                  self.pointgroup.rotate(angle=Ry, axis='y'),
                                  self.pointgroup.rotate(angle=Rz, axis='z')]
        return self.initial_rotations

    def calculate(self, analyze=True):
        # Single processor:
        if self.mpi is None:

            # Go through each state:
            for index, state in enumerate(self.statelist):

                wf = self.get_wf(state)
                norm0 = self.calculate_fullnorm(index, wf)

                if self.cutarea is not None:
                    wf *= self.cutarea

                # Apply initial transformations:
                wf = self.do_initial_transformations(wf)

                # Get norm:
                norm1 = self.calculate_cutnorm(index, wf)

                # Go through each operation:
                for j, operation in enumerate(self.pointgroup.operation_names):

                    newwf = self.pointgroup.operations[j][1](wf)

                    # Rotations may throw some of the density outside the box:
                    norm2 = np.multiply(np.conj(newwf), newwf).sum()

                    integral = np.multiply(np.conj(wf), newwf).sum()
                    norm = np.sqrt(norm1 * norm2)
                    self.overlaps[index][j] = integral/norm

            # Format the output file:
            self.format_file()

            # Print values to output file:
            self.print_out()
            self.norms_calculated = True

        # Processors in parallel:
        else:
            mpisize = self.mpi.size

            # Divide the states for processors:
            for k in range(self.n // mpisize + int(self.n % mpisize != 0)):

                # Go through each state:
                for index, state in enumerate(
                        self.statelist[k*mpisize:(k+1)*mpisize]):

                    # Only a single processor goes through a state:
                    if index % mpisize == self.mpi.rank:

                        wf = self.get_wf(state)

                        norm0 = self.calculate_fullnorm(k*mpisize+index, wf)

                        if self.cutarea is not None:
                            wf *= self.cutarea

                        # Apply initial transformations:
                        wf = self.do_initial_transformations(wf)

                        # Get norm:
                        norm1 = self.calculate_cutnorm(k*mpisize+index, wf)

                        # Go through each operation:
                        for j, operation in enumerate(
                                self.pointgroup.operation_names):

                            newwf = self.pointgroup.operations[j][1](wf)
                            norm2 = np.multiply(np.conj(newwf), newwf).sum()
                            integral = np.multiply(np.conj(wf), newwf).sum()
                            norm = np.sqrt(norm1 * norm2)
                            self.overlaps[k*mpisize+index][j] = integral/norm

                # Distribute the newly calculated values for other processors
                for rank in range(len(
                        self.statelist[k*mpisize:(k+1)*mpisize])):
                    self.mpi.world.broadcast(
                        self.overlaps[k*mpisize+rank], rank)
                    self.mpi.world.broadcast(
                        self.cutnorms[k*mpisize+rank], rank)
                    self.mpi.world.broadcast(
                        self.fullnorms[k*mpisize+rank], rank)

            # Format the output file:
            if self.mpi.rank == 0:
                self.format_file()

            # Print values to output file:
            if self.mpi.rank == 0:
                self.print_out()
            self.norms_calculated = True

        if analyze:
            self.analyze()

    def do_initial_transformations(self, data):
        """
        Shift the wave function to the center and rotate for a proper
        orientation for analysis
        """
        data = shift(data, self.shiftvector)
        newdata = data
        operations = self.initial_rotations
        for operation in operations:
            newdata = operation(newdata)
        return newdata

    def calculate_fullnorm(self, stateindex, wf):
        norm0 = np.multiply(np.conj(wf), wf).sum()
        self.fullnorms[stateindex][0] = norm0
        return norm0

    def calculate_cutnorm(self, stateindex, wf):
        norm1 = np.multiply(np.conj(wf), wf).sum()
        self.cutnorms[stateindex][0] = norm1
        return norm1

    def print_out(self):
        f = open(self.overlapfile, 'a')
        try:
            for i in range(self.n):
                f.write('\n%12d' % self.statelist[i])
                f.write('%12.06f' % self.get_energy(self.statelist[i]))
                for overlap in self.overlaps[i]:
                    f.write('%12.08f' % overlap)
        finally:
            f.close()

    def format_file(self):
        f = open(self.overlapfile, 'w')
        try:
            f.write('#%11s%12s' % ('band', 'energy'))
            for name in self.pointgroup.operation_names:
                f.write('%12s' % name)
        finally:
            f.close()

    def analyze(self, gridspacing=None):
        """
        The symmetry representations are
        resolved from the overlap matrix.

        Parameters
        ----------
        gridspacing : array_like
            Grid spacings along the x, y and z axes
        """

        tofile = self.symmetryfile
        if self.mpi is not None:
            if self.mpi.rank != 0:
                return 0
        reduced_tables = []

        # Format the analysis file:
        f = open(tofile, 'w')
        try:
            f.write('%s %s\n'%('# INPUTFILE ', self.filename))
            f.write('%s %s\n'%('# POINTGROUP', str(self.pointgroup)))
            f.write('# REPRESENTATIONS ')
            for s in self.pointgroup.symmetries:
                f.write('%6s' % s)
            f.write('\n# \n#')
            f.write('%11s%12s%12s' % ('band', 'energy', 'full_norm'))
            for s in self.pointgroup.symmetries:
                f.write('%12s' % s)
            f.write('%16s\n' % 'best_symmetry')
        finally:
            f.close()

        # Go through each state:
        for index, state in enumerate(self.statelist):
            reduced_table = []

            # Sum the overlaps belonging to same symmetry operation
            # ie. reduce the overlaps table:
            start = 0
            for j in self.pointgroup.nof_operations:
                reduced_table.append(sum(
                    self.overlaps[index][start:(start+j)])/j)
                start += j

            # Solve the linear equation for the
            # wavefunction symmetry coefficients:
            coefficients = np.linalg.solve(
                np.array(self.pointgroup.get_normalized_table()).transpose(),
                reduced_table)

            # If the norms are not calculated, lets do it:
            if not self.norms_calculated:
                wf = self.get_wf(state)
                self.calculate_fullnorm(index, wf)
                if self.cutarea is not None:
                    wf *= self.cutarea
                self.calculate_cutnorm(index, wf)

            gridspacing = self.h

            # Normalize:

            # shrink to cut area and
            # remove the effect of grid spacing in the norms:
            coefficients *= self.cutnorms[index] * np.prod(gridspacing)

            # remove the effect of grid spacing in the norms:
            fullnorm = self.fullnorms[index] * np.prod(gridspacing)

            # Look for the best candidate for the symmetry:
            symmetry = self.pointgroup.symmetries[np.argmax(coefficients)]

            # Write to file:
            f = open(tofile, 'a')
            try:
                f.write("%12d%12.06f%12.06f" % (state,
                                                self.get_energy(state),
                                                fullnorm))
                for c in coefficients:
                    f.write('%12.06f' % c)

                f.write('%16s\n' % symmetry)

            finally:
                f.close()

        self.norms_calculated = True

        return 1

    def read(self, fromfile):
        """
        Read the overlaps file if it is already done.
        """

        # Read only to master processor
        # since the analysis is only done using it:
        if self.mpi is not None:
            if self.mpi.rank != 0:
                return 0

        overlaps = []
        statelist = []
        f = open(fromfile, 'r')
        try:
            for i, line in enumerate(f):
                if line.split()[0].startswith('#'):
                    continue
                elements = line.split()
                statelist.append(int(elements[0]))
                numbers = [float(k) for k in elements[2:]]
                overlaps.append(numbers)
        finally:
            f.close()
        self.statelist = statelist
        self.overlaps = np.array(overlaps).astype(float)
        return statelist, self.overlaps


class GPAWSymmetryCalculator(SymmetryCalculator):
    """
    Data extractor for GPAW restart files.
    
    Does not work as such due to parallel read.
    """

    def initialize(self):
        self.load_data()

    def load_data(self):
        self.atoms, self.calc = restart(self.filename,
                                        parallel={'domain': gpaw.mpi.size})
        self.energies = self.calc.get_eigenvalues(broadcast=True)
        return 1

    def get_energy(self, band):
        """Energies are only used for output."""
        return self.energies[band]

    def get_wf(self, band):
        return self.calc.get_pseudo_wave_function(band, pad=False)


class GPAWULMSymmetryCalculator(SymmetryCalculator):
    """
    Data extractor for ulm-type GPAW output files.

    Reads the atoms, energies and wave functions from
    GPAW restart file for a single processor.
    """

    def initialize(self):
        self.load_data()

    def get_atoms(self):
        reader = ulm.open(self.filename)
        symbols = reader.atoms.numbers
        cell = reader.atoms.cell
        positions = reader.atoms.positions
        return Atoms(symbols, positions=positions, cell=cell)

    def load_data(self):
        self.atoms = self.get_atoms()
        return 1

    def get_wf(self, n, k=0, s=0):
        reader = ulm.open(self.filename)
        wf = reader.wave_functions.proxy("values", k, s)[n]
        return wf

    def get_energy(self, band):
        reader = ulm.open(self.filename)
        return reader.wave_functions.eigenvalues.ravel()[band]
