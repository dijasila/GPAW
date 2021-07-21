.. _introduction_to_paw:

===================
Introduction to PAW
===================

A simple example
================

We look at the `2\sigma`\ * orbital of a CO molecule: |ts|

.. |ts| image:: 2sigma.png

The main quantity in the PAW method is the pseudo wave-function (blue
crosses) defined in all of the simulation box:

.. math::

  \tilde{\psi}(\mathbf{r}) =  \tilde{\psi}(ih, jh, kh),

where `h` is the grid spacing and `(i, j, k)` are the indices of the
grid points.

.. figure:: co_wavefunctions.png

In order to get the all-electron wave function, we add and subtract
one-center expansions of the all-electron (thick lines) and pseudo
wave-functions (thin lines):

.. math::

  \tilde{\psi}^a(\mathbf{r}) =  \sum_i C_i^a \tilde{\phi}_i^a(\mathbf{r})

.. math::

  \psi^a(\mathbf{r}) =  \sum_i C_i^a \phi_i^a(\mathbf{r}),

where `a` is C or O and `\phi_i` and `\tilde{\phi}_i` are atom
centered basis functions formed as radial functions on logarithmic
radial grid multiplied by spherical harmonics.

The expansion coefficients are given as:

.. math::

  C_i^a = \int d\mathbf{r} \tilde{p}^a_i(\mathbf{r} - \mathbf{R}^a)
  \tilde{\psi}(\mathbf{r}).


Approximations
==============

* Frozen core orbitals.
* Truncated angular momentum expansion of compensation charges.
* Finite number of basis functions and projector functions.


More information on PAW
=======================

You can find additional information on the
:ref:`literature <literature_reports_presentations_and_theses>` page, or
by reading the :download:`paw note <paw_note.pdf>`.

.. _paw_papers:


Articles on the PAW formalism
-----------------------------

The original article introducing the PAW formalism:
   | P. E. Blöchl
   | :doi:`Projector augmented-wave method <10.1103/PhysRevB.50.17953>`
   | Physical Review B, Vol. **50**, 17953, 1994

A different formulation of PAW by Kresse and Joubert designed to make the transition from USPP to PAW easy.
  | G. Kresse and D. Joubert
  | :doi:`From ultrasoft pseudopotentials to the projector augmented-wave method <10.1103/PhysRevB.59.1758>`
  | Physical Review B, Vol. **59**, 1758, 1999

A second, more pedagogical, article on PAW by Blöchl and co-workers.
  | P. E. Blöchl, C. J. Först, and J. Schimpl
  | :doi:`Projector Augmented Wave Method: ab-initio molecular dynamics with full wave functions <10.1007/BF02712785>`
  | Bulletin of Materials Science, Vol. **26**, 33, 2003


Script
======

.. literalinclude:: co_wavefunctions.py

