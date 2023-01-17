.. _gpu:

GPU
===

It is possible to do PW-mode groundstate calculations with the wave functions
on the GPU.  It has only been implemented in the new GPAW.  See
:git:`gpaw/test/gpu/pw_test.py` for an example.


Fake cupy library
=================

The implementation uses cupy_.  In the code, we don't do ``import cupy as cp``.
Instead we use ``from gpaw.gpu import cupy as cp``.  This allows us to use a
fake ``cupy`` implementation so that we can run GPAW's ``cupy`` code without
having a physical GPU.  To enable the fake ``cupy`` module, do::

  GPAW_CPUPY=1 python ...

This allows users without a GPU to find out if their code interferes with the
GPU implementation, simply by running the tests.

.. _cupy:: https://cupy.dev/
