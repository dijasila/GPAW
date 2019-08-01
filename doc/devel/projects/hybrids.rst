Hybrid functionals
==================

:Who:
    Jens Jørgen

Currently we have two implementations of exact exchange:

1) :git:`~gpaw/xc/hybrid.py`: Can handle Gamma-point only
   calculations self-consistently (for molecules and large cells).

2) :git:`~gpaw/xc/exx.py`: Can handle k-points, but not
   self-consitently.

Things to work on:

* Implement forces.
* Self-consistent k-point calculations.

See :ref:`exx`.
