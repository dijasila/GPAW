.. _magnetism:

===============
Magnetism in 2D
===============

This exercise investigates magnetic order in two dimensions. While a collinear
spin-density functional theory calculation might reveal that it is
energetically favorable for a given 2D material to order magnetically, two 
dimensional magnetic order at finite temperatures also requires presence of
magnetic anisotropy, usually arising from the spin-orbit coupling.

This exercise will teach you how to extract magnetic exchange and anisotropy
parameters for a localized spin model based on first principles calculations.
It will also touch upon the Mermin-Wagner theorem and show why anisotropy is
crucial in order to sustain magnetic order in two dimensions.

In the first part of the project, you will calculate the Curie temperature of
a |CrI3| monolayer. Afterwards, you will investigate the magnetic order in
|VI2|, which has antiferromagnetic coupling and noncollinear order. Finally,
you will finish the project by performing a search for new magnetic 2D materials 
with high critical temperatures based on a database of hypothetical monolayers.


Part 1: Critical temperature of |CrI3|
======================================

Following the instructions in the Jupyter notebook
:download:`magnetism1.ipynb`
you will in this first part of the project set up a monolayer of |CrI3| (using 
the input file
:download:`CrI3.xyz`)
and calculate its critical temperature.

The procedure will be as follows:

1) Set up the atomic structure and optimize the geometry of |CrI3|
2) Calculate the Heisenberg exchange parameters based on a total energy mapping 
   analysis
3) Show that the magnetic ground state is thermodynamically unstable when
   anisotropy is neglected (The Mermin-Wagner theorem)
4) Calculate the magnetic anisotropy and estimate the critical temperature


Part 2: Noncollinear magnetism in |VI2|
=======================================

:download:`magnetism2.ipynb`, :download:`VI2.xyz`

If the magnetic atoms form a hexagonal lattice and the exchange coupling is
anti-ferromagnetic, the ground state will have a non-collinear structure. In
the notebook ``magnetism2.ipynb`` you will

* Relax the atomic positions of the material

* Compare a collinear anti-ferromagnetic structure with the ferromagnetic state

* Obtain the non-collinear ground state

* Calculate the magnetic anisotropy and discuss whether or not the material
  will exhibit magnetic order at low temperature


Part 3: Find new magnetic monolayers with high critical temperatures
====================================================================

:download:`magnetism3.ipynb`

In this last part you will search the database and pick one material you
expect to have a large critical temperature. The total energy mapping analysis
is carried out to obtain exchange coupling parameters and a first principles
estimate of the critical temperature. The guidelines for the analysis is found
in the notebook ``magnetism3.ipynb``.

.. |CrI3| replace:: CrI\ :sub:`3`

.. |VI2| replace:: VI\ :sub:`2`
