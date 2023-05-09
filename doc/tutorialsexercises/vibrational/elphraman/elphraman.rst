.. _elphraman:

=======================================
Raman spectroscopy for extended systems
=======================================

This tutorial shows how to calculate the Raman spectrum of diamond using the
electron-phonon coupling based Raman methods as described in :ref:`raman`.


Phonons
=======

The phonon frequencies enter the Raman calculation by defining the line positions in the final spectrum. Their associated mode vectors are necessary to project the
electron-phonon matrix from Cartesian coordinates in the modes.

It is crucial to use tightly relaxed structures and well-converged parameters to
obtain accurate phonon frequencies. In this case we choose the separate out the
phonon from the potential calculation for illustrative purposes:
(:git:`~doc/tutorialsexercises/vibrational/elphraman/phonon.py`)

.. literalinclude:: phonon.py

As exercise, check the dependence of the phonon frequencies with the calculation
mode, supercell size and convergence parameters.

This given set of parameters yield this result::

    i    cm^-1
  ------------
    0    -0.03
    1    -0.00
    2    -0.00
    3  1304.46
    4  1304.61
    5  1304.73

Effective Potential
===================

At the heart of the electron-phonon coupling is the calculation of the gradient
of the effective potential, which is done using finite displacements just like the
phonons. Those two calculations can run simultaneous, of the required set of
parameters coincide.

The electron-phonon matrix is quite sensitive to self-interaction of a displaced
atom with its periodic images, so a sufficiently large supercell needs to be used.
The ``(2x2x2)`` supercell used in this example is a bit too small. When using a supercell for the calculation you have to consider, that the atoms object needs
to contain the primitive cell, not the supercell, while the parameters for the
calculator object need to be good for the supercell, not the primitive cell.
(:git:`~doc/tutorialsexercises/vibrational/elphraman/elph.py`)

.. literalinclude:: elph.py

This calculation merely dumped the effective potential at various displacements
onto the harddrive. We now need to calculate the actual derivative and the matrix
elements with the Kohn-Sham wave-functions.

For this we first need to complete a ground-state calculation for the supercell
with all bands converged. This calculation needs to be done in LCAO mode with
parallelisation over domains disabled. (:git:`~doc/tutorialsexercises/vibrational/elphraman/supercell_matrix.py`)

.. literalinclude:: supercell_matrix.py

The ``calculate_supercell_matrix()`` method will then compute the gradients and
calculate the matrix elements. The results are saved in ``pckl`` files in a
basis of LCAO orbitals and supercell indices.


Momentum matrix
===============

The Raman tensor does not only depend on the electron-phonon matrix, but also
on the transition probability between electronic states, which in this case is
expressed in the momentum matrix elements.
Those can be extracted directly from a converged LCAO calculation of the primitive
cell. (:git:`~doc/tutorialsexercises/vibrational/elphraman/momentum_matrix.py`)

.. literalinclude:: momentum_matrix.py

It it convienient to save the calculation, as we will need it for the next step.

Phonon mode projected electron-phonon matrix
============================================

With all the above calculations finished we can extract the electron-phonon
matrix in the Bloch basis of the primitive cell projected onto the phonon modes:
(:git:`~doc/tutorialsexercises/vibrational/elphraman/elph_matrix.py`)

.. literalinclude:: elph_matrix.py

This will save the electron-phonon matrix as a numpy file to the disk.
The optional ``load_gx_as_needed`` tag prevents from all supercell pickle files
being read at once. Instead they are loaded and processed one-by-one. This can
save lots of memory for larger systems with hundreds of atoms, where the
supercell matrix can be over 100GiB large.


Note: This part has not been tested properly for parallel runs and should be done
in serial mode only.

Raman spectrum
==============

With all ingredients provided we can now commence with the computation of the
Raman tensor. This can take some time, if calculation of the non-resonant terms
is included. (:git:`~doc/tutorialsexercises/vibrational/elphraman/raman_intensities.py`)

.. literalinclude:: raman_intensities.py

This part consists of three steps. First we calculate the Raman tensor entry for
a given set of polarisations, which are saved in files of the form ``Rlab_zy.npy``.
From those the intensities, summed over all phonon modes, are calculated on a grid
and saved in files of the form ``RI_zy.npy``. Lastly, the spectrum is plotted.

Note: This part has not been tested properly for parallel runs and should be done
in serial mode only.

.. image:: Raman_all.png
    :scale: 30
    :align: center
