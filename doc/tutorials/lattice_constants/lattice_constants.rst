.. _lattice_constants:

=========================
Finding lattice constants
=========================

.. seealso::

   * `ASE EOS tutorial
     <https://wiki.fysik.dtu.dk/ase/tutorials/eos/eos.html>`_
   * `ASE Finding lattice constants tutorial
     <https://wiki.fysik.dtu.dk/ase/tutorials/lattice_constant.html>`_

   * `ASE equation of state module
     <https://wiki.fysik.dtu.dk/ase/ase/utils.html#equation-of-state>`_


BCC iron
========

Let's try to converge the lattice constant with respect to number of
plane-waves:

.. literalinclude:: iron.py
    :lines: 1-19

.. image:: Fe_conv_ecut.png

Using a plane-wave cutoff energy of 600 eV, we now check convergence
with respect to number of **k**-points and with respect to the
smearing width used for occupations numbers:

.. literalinclude:: iron.py
    :lines: 21-

.. image:: Fe_conv_k.png

(see also :download:`analysis script <iron.agts.py>`).
