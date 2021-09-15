====================
Core data structures
====================

Uniform grids
=============

>>> from gpaw.core import UniformGrid
>>> a = 4.0
>>> n = 20
>>> grid = UniformGrid(cell=[a, a, a], size=(n, n, n), pbc=(True, True, True))
>>> u1 = grid.empty()
>>> u1.data[:] = 1.0


Plane waves
===========

>>> from gpaw.core import PlaneWaves
>>> pws = PlaneWaves(ecut=100, grid=grid)
>>> p1 = pws.empty()
>>> u1.fft(out=p1)
PlaneWaveExpansions(pw=PlaneWaves(ecut=100, grid=20*20*20), shape=())
>>> p1.data[0]
(8000+0j)


Atom-centered functions
=======================

aos = AO()



Conventions: grid, pws, aos?


API
===

Uniform grids
-------------

.. automodule:: gpaw.core.uniform_grid
