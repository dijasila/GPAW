from typing import Sequence, Any, Dict, List, Union

from ase.units import Bohr
import numpy as np
from numpy.linalg import inv, det, solve
from scipy.ndimage import map_coordinates

from .group import PointGroup

Array1D = Any
Array2D = Any
Array3D = Any
Axis = Union[str, Sequence[float], None]


class SymmetryChecker:
    def __init__(self,
                 group: PointGroup,
                 center: Sequence[float],
                 radius: float = 2.0,
                 x: Axis = None,
                 y: Axis = None,
                 z: Axis = None,
                 grid_spacing: float = 0.2):
        self.group = group
        self.normalized_table = group.get_normalized_table()
        self.points = sphere(radius, grid_spacing)
        self.center = center
        self.grid_spacing = grid_spacing
        self.rotation = rotation_matrix([x, y, z])

    def check_function(self,
                       function: Array3D,
                       grid_vectors: Array2D) -> Dict[str, Any]:
        dv = abs(det(grid_vectors))
        norm1 = (function**2).sum() * dv
        M = inv(grid_vectors.dot(self.rotation)).T

        overlaps: List[float] = []
        for op in self.group.operations.values():
            pts = (self.points.dot(op.T) + self.center).dot(M.T)
            values = map_coordinates(function, pts.T, mode='wrap')
            if not overlaps:
                values1 = values
            overlaps.append(values.dot(values1) * self.grid_spacing**3)

        reduced_overlaps = []
        i1 = 0
        for n in self.group.nops:
            i2 = i1 + n
            reduced_overlaps.append(sum(overlaps[i1:i2]) / n)
            i1 = i2

        characters = solve(self.normalized_table.T, reduced_overlaps)
        best = self.group.symmetries[characters.argmax()]

        return {'symmetry': best,
                'norm': norm1,
                'overlaps': overlaps,
                'characters': {symmetry: value
                               for symmetry, value
                               in zip(self.group.symmetries, characters)}}

    def check_band(self, calc, band, spin=0):
        wfs = calc.get_pseudo_wave_function(band, spin=spin, pad=True)
        return self.check_function(wfs, calc.wfs.gd.h_cv * Bohr)


def sphere(radius: float, grid_spacing: float) -> Array2D:
    npts = int(radius / grid_spacing) + 1
    x = np.linspace(-npts, npts, 2 * npts + 1) * grid_spacing
    points = np.array(np.meshgrid(x, x, x, indexing='ij')).reshape((3, -1)).T
    points = points[(points**2).sum(1) <= radius**2]
    return points


def rotation_matrix(axes: Sequence[Axis]) -> Array3D:
    if all(axis is None for axis in axes):
        return np.eye(3)

    j = -1
    for i, axis in enumerate(axes):
        if axis is None:
            assert j == -1
            j = i
    assert j != -1

    axes = [normalize(axis) for axis in axes]
    axes[j] = np.cross(axes[j - 2], axes[j - 1])

    return np.array(axes)


def normalize(vector):
    if isinstance(vector, str):
        return {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[vector]
    return np.array(vector) / np.linalg.nor(vector)


'''
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

        # Format the analysis file:
        f = open(tofile, 'w')
        try:
            f.write('%s %s\n' % ('# INPUTFILE ', self.filename))
            f.write('%s %s\n' % ('# POINTGROUP', str(self.pointgroup)))
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

'''
