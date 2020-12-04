"""Point Group Symmetry.

The code was originally written for the paper:

    S. Kaappa, S. Malola, H. Hakkinen

    J. Phys. Chem. A; vol. 122, 43, pp. 8576-8584 (2018)

"""

from .group import PointGroup
from .check import SymmetryChecker

__all__ = ['PointGroup', 'SymmetryChecker', 'point_group_names']

point_group_names = ['C1', 'Cs', 'Ci', 'C2', 'D3', 'D5',
                     'C2v', 'C3v', 'C4v', 'C2h', 'D2d', 'D3h', 'D5h',
                     'Ico', 'Ih', 'Oh', 'Td', 'Th']
