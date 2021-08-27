.. module:: gpaw.point_groups

.. _point groups:

====================================
Point group symmetry representations
====================================

In the chemist's point of view, group theory is a long-known approach to
assign symmetry representations to molecular vibrations and wave
functions [1]_ [2]_. For larger but still symmetric molecules (eg.
nanoclusters [3]_), assignment of the representations by hand for each
electron band becomes tedious. This tool is an automatic routine to
resolve the representations.

In this implementation, the wave functions are operated with rotations
and mirroring with cubic interpolation onto the same grid, and the
overlap of the resulting function with the original one is calculated.
The representations are then given as weights that are the coefficients
of the linear combination of the overlaps in the basis of the character
table rows ie. the irreducible representations.

Prior to symmetry analysis, you should have the restart file that
includes the wave functions, and knowledge of

* The point group to consider
* The bands you want to analyze
* The main axis and the secondary axis of the molecule, corresponding
  to the point group
* The atom indices whose center-of-mass is shifted to the center of the unit
  cell ie. to the crossing of the main and secondary axes.
* The atom indices around which you want to perform the analysis
  (optional)


Example: The water molecule
---------------------------

To resolve the symmetry representations of occupied states of the water
molecule in C2v point group, the :download:`h2o.py` script can be used:

.. literalinclude:: h2o.py
    :start-after: creates:
    :end-before: Analyze

In order to analyze the symmetry, you need to create a
:class:`SymmetryChecker` object and call its
:meth:`~SymmetryChecker.check_band` method like this:

.. literalinclude:: h2o.py
    :start-after: Analyze
    :end-before: Write

You can also write the wave functions to a file:

.. literalinclude:: h2o.py
    :start-after: Write
    :end-before: Create

and do the analysis later:

>>> from gpaw import GPAW
>>> from gpaw.point_groups import SymmetryChecker
>>> calc = GPAW('h2o.gpw')
>>> center = calc.atoms.positions[0]  # oxygen atom
>>> checker = SymmetryChecker('C2v', center, radius=2.0)
>>> checker.check_calculation(calc, n1=0, n2=4)

This will produce the following output:

.. literalinclude:: h2o-symmetries.txt

The bands have very distinct representations as expected.

.. note::

    There is also a simple command-line interface::

        $ python3 -m gpaw.point_groups C2v h2o.gpw -c O -b 0:4


.. autoclass:: SymmetryChecker
    :members:
.. autoclass:: PointGroup
    :members:


.. [1] K. C. Molloy. Group Theory for Chemists: Fundamental Theory and
                     Applications. Woodhead Publishing 2011

.. [2] J. F. Cornwell. Group Theory in Physics: An Introduction. Elsevier
       Science and Technology (1997)

.. [3] Kaappa, Malola, Häkkinen; Point Group Symmetry Analysis of the
       Electronic Structure of Bare and Protected Metal Nanocrystals
       J. Phys. Chem. A; vol. 122, 43, pp. 8576-8584 (2018)
