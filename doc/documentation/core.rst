====================
Core data structures
====================

Uniform grids
=============

>>> from gpaw.core import UniformGrid
>>> grid = UniformGrid(cell=[a, a, a], size=(n, n, n), pbc=(True, True, True))
>>> u1 = grid.zeros()
>>> u1.data[:] = 1.0



aos = AO(

Conventions: grid, pws, aos?
