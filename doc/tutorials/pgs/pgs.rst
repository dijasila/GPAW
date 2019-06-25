.. _pgs:

====================================
Point group symmetry representations
====================================

In the chemist's point of view, group theory is a long-known approach to
assign symmetry representations to molecular vibrations and wave
<<<<<<< HEAD
functions [1]_[2]_. For larger but still symmetric molecules (eg.
nanoclusters [3]_), assignment of the representations by hand for each
=======
functions [1]_. For larger but still symmetric molecules (eg.
nanoclusters [2]_), assignment of the representations by hand for each
>>>>>>> 93e31804fac404f73c30a402f152f9c4405b0ab8
electron band becomes tedious. This tool is an automatic routine to
resolve the representations.

In this implementation, the wave functions are operated with rotations
and mirroring with cubic interpolation onto the same grid, and the
overlap of the resulting function with the original one is calculated.
The representations are then given as weights that are the coefficients 
of the linear combination of the overlaps in the basis of the character
<<<<<<< HEAD
table rows ie. the irreducible representations.
=======
table rows.
>>>>>>> 93e31804fac404f73c30a402f152f9c4405b0ab8

Prior to symmetry analysis, you should have the restart file that
includes the wave functions, and knowledge of

* The point group to consider
* The bands you want to analyze
* The main axis and the secondary axis of the molecule, corresponding
  to the point group
* The atom indices whose center-of-mass is shifted to the center of the unit
<<<<<<< HEAD
  cell ie. to the crossing of the main and secondary axes.
=======
* cell ie. to the crossing of the main and secondary axes.
>>>>>>> 93e31804fac404f73c30a402f152f9c4405b0ab8
* The atom indices around which you want to perform the analysis 
  (optional)

Example: The water molecule
---------------------------

To resolve the symmetry representations of occupied states of the water 
molecule in C2v point group, the following script can be used:

.. literalinclude:: h2o-C2v.py

The output of the representations looks like this:

.. literalinclude:: symmetries-h2o.txt

The bands have very distinct representations as expected.


.. [1] K. C. Molloy. Group Theory for Chemists: Fundamental Theory and
                     Applications. Woodhead Publishing 2011

<<<<<<< HEAD
.. [2] J. F. Cornwell. Group Theory in Physics: An Introduction. Elsevier Sci-
ence and Technology (1997)

.. [3] Kaappa, Malola, Häkkinen; Point Group Symmetry Analysis of the
=======
.. [2] Kaappa, Malola, Häkkinen; Point Group Symmetry Analysis of the
>>>>>>> 93e31804fac404f73c30a402f152f9c4405b0ab8
       Electronic Structure of Bare and Protected Metal Nanocrystals 
       https://arxiv.org/abs/1808.01868 (2018)

