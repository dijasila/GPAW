====================
Core data structures
====================

.. modeule gpaw.core

Uniform grids
=============

A uniform grid can be created with the :class:`UniformGrid` class:

>>> import numpy as np
>>> from gpaw.core import UniformGrid
>>> a = 4.0
>>> n = 20
>>> grid = UniformGrid(cell=a * np.eye(3),
...                    size=(n, n, n))

Given a :class:`UniformGrid` object, one can create
:class:`UniformGridFunctions` objects like this

>>> u1 = grid.empty()
>>> u1.data.shape
(20, 20, 20)
>>> u1.data[:] = 1.0
>>> grid.zeros((3, 2)).data.shape
(3, 2, 20, 20, 20)


Plane waves
===========

A set of plane-waves are characterized by a cutoff energy and a uniform
grid
>>> from gpaw.core import PlaneWaves
>>> pw = PlaneWaves(ecut=100, cell=grid.cell)
>>> p1 = pw.empty()
>>> u1.fft(out=p1)
PlaneWaveExpansions(pw=PlaneWaves(ecut=100, grid=20*20*20), shape=())
>>> G = pw.reciprocal_vectors()
>>> G.shape
(1536, 3)
>>> G[0]
array([0., 0., 0.])
>>> p1.data[0]
(8000+0j)
>>> p1.ifft(out=u1)
>>> u1.data[0, 0, 0]
1.0


Distributed arrays
==================

...


Block boundary conditions
=========================

...


Matrix elements
===============

>>> def T(psi):
...     out = psi.empty_like()
...     out.data[:] = psi.pw.ekin * psit.data
...     return out
>>> H = psi.matrix_elements(psit, function=T)

Same as:

>>> Tpsi = T(psi)
>>> psi.matrix_elements(Tpsi, symmetric=True)

but faster.


Atom-centered functions
=======================

alpha = 4.0
rcut = 2.0
l = 0
gauss = (l, rcut, labmda r: np.exp(-alpha * r**2))
grid = UniformGrid(cell=[4.0, 1.0, 1.0], size=[40, 10, 10])
pos = [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
acf = grid.atom_centered_functions([[gauss], [gauss]], pos)
coefs = acf.empty()
coefs[0] = [(4 * pi)**0.5]
coefs[1] = [2 * (4 * pi)**0.5]
f = grid.zeros()
acf.add(f, coefs)
x, y = f.xy(..., 5, 5)
plt.plot(x, y)


API
===

Uniform grids
-------------

.. autoclass:: gpaw.core.UniformGrid
    :undoc-members:
.. autoclass:: gpaw.core.PlaneWaves
    :undoc-members:
.. autoclass:: gpaw.core.atom_centered_functions.AtomCenteredFunctions
    :undoc-members:
.. autoclass:: gpaw.core.layout.Layout
    :undoc-members:
.. autoclass:: gpaw.core.uniform_grid.UniformGridFunctions
    :undoc-members:
.. autoclass:: gpaw.core.arrays.DistributedArrays
    :undoc-members:
.. autoclass:: gpaw.core.plane_waves.PlaneWaveExpansions
    :undoc-members:
.. autoclass:: gpaw.core.plane_waves.Empty
    :undoc-members:
.. autoclass:: gpaw.core.plane_waves.PWMapping
    :undoc-members:


Matrix object
=============

.. module:: gpaw.core.matrix
.. autoclass:: Matrix
   :undoc-members:

A simple example that we can run with MPI on 4 cores::

    from gpaw.matrix import Matrix
    from gpaw.mpi import world
    a = Matrix(5, 5, dist=(world, 2, 2, 2))
    a.array[:] = world.rank
    print(world.rank, a.array.shape)

Here, we have created a 5x5 :class:`Matrix` of floats distributed on a 2x2
BLACS grid with a block size of 2 and we then print the shapes of the ndarrays,
which looks like this (in random order)::

    1 (2, 3)
    2 (3, 2)
    3 (2, 2)
    0 (3, 3)

Let's create a new matrix ``b`` and :meth:`redistribute <Matrix.redist>` from
``a`` to ``b``::

    b = a.new(dist=(None, 1, 1, None))
    a.redist(b)
    if world.rank == 0:
        print(b.array)

This will output::

    [[ 0.  0.  2.  2.  0.]
     [ 0.  0.  2.  2.  0.]
     [ 1.  1.  3.  3.  1.]
     [ 1.  1.  3.  3.  1.]
     [ 0.  0.  2.  2.  0.]]

Matrix-matrix multiplication
works like this::

    from gpaw.matrix import matrix_matrix_multiply as mmm
    c = mmm(1.0, a, 'N', a, 'T')
    mmm(1.0, a, 'N', a, 'T', 1.0, c, symmetric=True)
