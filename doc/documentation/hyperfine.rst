.. module:: gpaw.hyperfine
.. _hyperfine:

Isotropic and anisotropic hyperfine coupling paramters
======================================================

.. contents::


Python API and CLI
------------------

Use the :func:`hyperfine_parameters` function or the CLI tool::

    $ python3 -m gpaw.hyperfine <gpw-file>

.. autofunction:: hyperfine_parameters

For details, see :doi:`Peter E. Blöchl <10.1103/PhysRevB.62.6158>` and
:doi:`Oleg V. Yazyev *et al.* <10.1103/PhysRevB.71.115110>`.


G-factors
---------

Here is a list of g-factors (from Wikipedia_):

.. csv-table::
    :file: g-factors.csv

.. _Wikipedia: https://en.wikipedia.org/wiki/Gyromagnetic_ratio


Hydrogen 21 cm line
-------------------

Here is how to calculate the famous hydrogen spectral line of 21 cm:

.. literalinclude:: hyperfine_21.py
    :end-before: assert

The output will be ``23.2 cm``.
It's slightly off because the LDA spin-density at the position of the hydrogen
nucleus is a bit too low (should be `1/\pi` in atomic units).
