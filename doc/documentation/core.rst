==============================
Introduction to GPAW internals
==============================

This guide will contain graphs showing the relationship between objects
that build up a DFT calculation engine.

.. hint::

   Here is a simple graph showing the relations between the classes ``A``,
   ``B`` and ``C``:

   .. image:: abc.svg

   Here, the object of type ``A`` has an attribute ``a`` of type ``int`` and an
   attribute ``b`` of type ``B`` or ``C``, where ``C`` inherits from ``B``.
   Specialized subclasses are colored light green.

.. contents::


DFT components
==============

The components needed for a DFT calculation are created by a "builder" that
can be made with the :func:`~gpaw.new.builder.builder` function, an ASE
:class:`ase.Atoms` object and some input parameters:

>>> from ase import Atoms
>>> atoms = Atoms('Li', cell=[2, 2, 2], pbc=True)
>>> from gpaw.new.builder import builder
>>> params = {'mode': 'pw', 'kpts': (5, 5, 5), 'txt': None}
>>> b = builder(atoms, params)

.. image:: builder.svg

Normally, you will not need to create a DFT-component builder yourself.  It will happen automatically when you create a DFT-calculation object like this:

>>> from gpaw.new.calculation import DFTCalculation
>>> calculation = DFTCalculation.from_parameters(atoms, params)

or when you create an ASE-calculator interface:

>>> from gpaw.new.ase_interface import GPAW
>>> atoms.calc = GPAW(**params)

Full picture
============

.. image:: code.svg


DFT-calculation object
======================

.. module:: gpaw.core

An instance of the :class:`gpaw.new.calculation.DFTCalculation` class has
the following attributes:

.. list-table::

  * - ``density``
    - :class:`gpaw.new.density.Density`
  * - ``ibzwfs``
    - :class:`gpaw.new.ibzwfs.IBZWaveFunctions`
  * - ``potential``
    - :class:`gpaw.new.potential.Potential`
  * - ``scf_loop``
    - :class:`gpaw.new.scf.SCFLoop`
  * - ``pot_calc``
    - :class:`gpaw.new.pot_calc.PotentialCalculator`


Naming convention for arrays
============================

Commonly used indices:

 =======  ====================================================================
 index    description
 =======  ====================================================================
 ``a``    Atom number
 ``c``    Unit cell axis-index (0, 1, 2)
 ``v``    *xyz*-index (0, 1, 2)
 ``K``    BZ **k**-point index
 ``k``    IBZ **k**-point index
 ``q``    IBZ **k**-point index (local, i.e. it starts at 0 on each processor)
 ``s``    Spin index (`\sigma`)
 ``s``    Symmetry index
 ``u``    Combined spin and **k**-point index (local)
 ``R``    Three indices into the coarse 3D grid
 ``r``    Three indices into the fine 3D grid
 ``G``    Index of plane-wave coefficient (wave function expansion, ``ecut``)
 ``g``    Index of plane-wave coefficient (densities, ``2 * ecut``)
 ``h``    Index of plane-wave coefficient (compensation charges, ``8 * ecut``)
 ``X``    ``R`` or ``G``
 ``x``    ``r``, ``g`` or ``h``
 ``x``    Zero or more extra dimensions
 ``M``    LCAO orbital index (`\mu`)
 ``n``    Band number
 ``n``    Principal quantum number
 ``l``    Angular momentum quantum number (s, p, d, ...)
 ``m``    Magnetic quantum number (0, 1, ..., 2*`ell` - 1)
 ``L``    ``l`` and ``m`` (``L = l**2 + m``)
 ``j``    Valence orbital number (``n`` and ``l``)
 ``i``    Valence orbital number (``n``, ``l`` and ``m``)
 ``q``    ``j1`` and ``j2`` pair
 ``p``    ``i1`` and ``i2`` pair
 ``r``    CPU-rank
 =======  ====================================================================

Examples:

.. list-table::

  * - ``density.D_asii``
    - `D_{\sigma,i_1,i_2}^a`
    - :class:`~atom_arrays.AtomArrays`
  * - ``density.nt_sR``
    - `\tilde{n}_\sigma(\mathbf{r})`
    - :class:`~uniform_grid.UniformGridFunctions`
  * - ``ibzwfs.wfs_qs[q][s].P_ain``
    - `P_{\sigma \mathbf{k} in}^a`
    - :class:`~atom_arrays.AtomArrays`
  * - ``ibzwfs.wfs_qs[q][s].psit_nX``
    - `\tilde{\psi}_{\sigma \mathbf{k} n}(\mathbf{r})`
    - :class:`~uniform_grid.UniformGridFunctions` |
      :class:`~plane_waves.PlaneWaveExpansions`
  * - ``ibzwfs.wfs_qs[q][s].pt_aX``
    - `\tilde{p}_{\sigma \mathbf{k} i}^a(\mathbf{r}-\mathbf{R}^a)`
    - :class:`~atom_centered_functions.AtomCenteredFunctions`


Domain descriptors
==================

GPAW has two different container types for storing one or more functions
in a unit cell (wave functions, electron densities, ...):

* :class:`~plane_waves.PlaneWaveExpansions`
* :class:`UniformGridFunctions`

.. image:: da.svg

Uniform grids
-------------

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
-----------

A set of plane-waves are characterized by a cutoff energy and a uniform
grid:

>>> from gpaw.core import PlaneWaves
>>> pw = PlaneWaves(ecut=100, cell=grid.cell)
>>> func_G = pw.empty()
>>> func_R.fft(out=func_G)
PlaneWaveExpansions(pw=PlaneWaves(ecut=100, cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], pbc=[True, True, True], comm=0/1, dtype=float64), shape=())
>>> G = pw.reciprocal_vectors()
>>> G.shape
(1536, 3)
>>> G[0]
array([0., 0., 0.])
>>> func_G.data[0]
(1+0j)
>>> func_G.ifft(out=func_R)
UniformGridFunctions(grid=UniformGrid(size=[20, 20, 20], cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], pbc=[True, True, True], comm=0/1, dtype=float64), shape=())
>>> round(func_R.data[0, 0, 0], 15)
1.0

Atoms-arrays
============

.. image:: aa.svg


Block boundary conditions
=========================

...


Matrix elements
===============

>>> psit_nG = pw.zeros(5)
>>> def T(psit_nG):
...     """Kinetic energy operator."""
...     out = psit_nG.new()
...     out.data[:] = psit_nG.desc.ekin_G * psit_nG.data
...     return out
>>> H_nn = psit_nG.matrix_elements(psit_nG, function=T)

Same as:

>>> Tpsit_nG = T(psit_nG)
>>> psit_nG.matrix_elements(Tpsit_nG, symmetric=True)
Matrix(float64: 5x5)

but faster.


Atom-centered functions
=======================

.. image:: acf.svg

.. literalinclude:: acf_example.py

.. figure:: acf_example.png


API
===

Core
----

.. autoclass:: gpaw.core.UniformGrid
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.PlaneWaves
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.atom_centered_functions.AtomCenteredFunctions
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.uniform_grid.UniformGridFunctions
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.arrays.DistributedArrays
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.atom_arrays.AtomArrays
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.plane_waves.PlaneWaveExpansions
    :members:
    :undoc-members:
.. autoclass:: gpaw.core.plane_waves.Empty
    :members:
    :undoc-members:


DFT
---

.. autoclass:: gpaw.new.calculation.DFTCalculation
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.density.Density
    :members:
    :undoc-members:
.. autofunction:: gpaw.new.builder.builder
.. autoclass:: gpaw.new.ibzwfs.IBZWaveFunctions
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.potential.Potential
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.pot_calc.PotentialCalculator
    :members:
    :undoc-members:
.. autoclass:: gpaw.new.scf.SCFLoop
    :members:
    :undoc-members:


Matrix object
=============

.. module:: gpaw.core.matrix
.. autoclass:: Matrix
   :members:
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

Let's create a new matrix ``b`` and :meth:`redistribute <Matrix.redist>`
from
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
