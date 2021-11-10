====================
Core data structures
====================

.. module:: gpaw.core

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

>>> func_R = grid.empty()
>>> func_R.data.shape
(20, 20, 20)
>>> func_R.data[:] = 1.0
>>> grid.zeros((3, 2)).data.shape
(3, 2, 20, 20, 20)


Plane waves
===========

A set of plane-waves are characterized by a cutoff energy and a uniform
grid
>>> from gpaw.core import PlaneWaves
>>> pw = PlaneWaves(ecut=100, cell=grid.cell)
>>> func_G = pw.empty()
>>> func_R.fft(out=func_G)
PlaneWaveExpansions(pw=PlaneWaves(ecut=100, grid=20*20*20), shape=())
>>> G = pw.reciprocal_vectors()
>>> G.shape
(1536, 3)
>>> G[0]
array([0., 0., 0.])
>>> func_G.data[0]
(1+0j)
>>> func_G.ifft(out=func_R)
>>> func_R.data[0, 0, 0]
1.0


Distributed arrays
==================

...


Block boundary conditions
=========================

...


Matrix elements
===============

>>> def T(psit_nG):
...     out = psit_nG.empty_like()
...     out.data[:] = psit_nG.desc.ekin_G * psit_nG.data
...     return out
>>> H_nn = psit_nG.matrix_elements(psit_nG, function=T)

Same as:

>>> Tpsit_nG = T(psit_nG)
>>> psit_nG.matrix_elements(Tpsit_nG, symmetric=True)

but faster.


Atom-centered functions
=======================

.. literalinclude: acf_example.py

.. figure:: acf_example.png


Examples
========

.. math::

   D_{\sigma,i_1,i_2}^a
   P_in^a
   \tilde\psi_{n\mathbf{R}}
   \tilde\psi_{n\mathbf{G}}
   \tilde p_i^a(\mathbf{r}-\mathbf{R}^a)



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
    a.data[:] = world.rank
    print(world.rank, a.data.shape)

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

    c = a.multiply(a, opb='T')
