.. _gpu:

GPU
===

Ground-state calculations on a GPU is an experimental feature at the moment:

* only PW-mode
* it has only been implemented in the new GPAW
* only parallelization over **k**-points
* no forces

See :git:`gpaw/test/gpu/pw_test.py` for an example.


The gpaw.gpu module
===================

.. module:: gpaw.gpu

.. data:: cupy

   :mod:`cupy` module (or :mod:`gpaw.gpu.cpupy` if :mod:`cupy` is not available)

.. data:: cupyx

   ``cupyx`` module (or :mod:`gpaw.gpu.cpupyx` if ``cupyx`` is not available)

.. autodata:: cupy_is_fake
.. autodata:: is_hip
.. autofunction:: as_xp
.. autofunction:: cupy_eigh


Fake cupy library
=================

.. module:: gpaw.gpu.cpupy
.. module:: gpaw.gpu.cpupyx

The implementation uses cupy_.  In the code, we don't do ``import cupy as cp``.
Instead we use ``from gpaw.gpu import cupy as cp``.  This allows us to use a
fake ``cupy`` implementation so that we can run GPAW's ``cupy`` code without
having a physical GPU.  To enable the fake ``cupy`` module, do::

  GPAW_CPUPY=1 python ...

This allows users without a GPU to find out if their code interferes with the
GPU implementation, simply by running the tests.

.. _cupy: https://cupy.dev/


CuPy enabled container objects
==============================

The following objects:

* :class:`~gpaw.core.uniform_grid.UniformGridFunctions`
* :class:`~gpaw.core.plane_waves.PlaneWaveExpansions`
* :class:`~gpaw.core.atom_arrays.AtomArrays`
* :class:`~gpaw.core.matrix.Matrix`

can have their data (``.data`` attribute) stored in a :class:`cupy.ndarray`
array instead of, as normal, a :class:`numpy.ndarray` array.  In additions,
these objects now have an ``xp`` attribute that can be :mod:`numpy` or
:mod:`cupy`.

Also, the :class:`~gpaw.core.atom_centered_functions.AtomCenteredFunctions`
object can do its operations on the GPU.


GPU-aware MPI
=============

Use a GPU-aware MPI implementation and set the :envvar:`GPAW_GPU` when compiling
GPAW's C-extension.

.. envvar:: GPAW_GPU

   Add support for passing :class:`cupy.ndarray` objects to MPI
