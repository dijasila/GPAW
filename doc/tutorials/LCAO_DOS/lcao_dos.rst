.. _bandstructures:

=========================================
Plotting projected density of states using the LCAO
=========================================

In this tutorial we demonstrate how to plot projected density of states from an LCAO calculation.

First, a standard ground state LCAO calculation is performed and the results
are saved to a *.gpw* file. 
The projections are made on a basis of atomic orbitals, for each of the species in a system (here Ga-s, Ga-p, As-s and As-p states).

.. literalinclude:: lcao_dos.py
    :lines: 1-129

Next, :mod:`basis.get_l_numbers()` module is used for obtaining information about the spherical harmonics in the setups of each species. 
The resulting figure shows DOS projection to all states (color lines) and the total DOS (black lines).

.. figure:: dos_GaAs.png
   :width: 600 px



The full script: :download:`lcao_dos.py`.
