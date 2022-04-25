.. _elph:

========================
Electron-phonon coupling
========================

Electron-phonon coupling is implemented for the LCAO mode.


Introduction
============

The electron-phonon interaction can be defined as

.. math::

    H_{el-ph} = \sum_{l,ij} g_{ij}^l c^{*}_i c_j ( a_l^{*} + a_l  ) .

The phonon modes `l` are coupled to the electronic states `i`, `j` via the
electron-phonon coupling matrix

.. math::

    g_{ij}^l = \sqrt{  \frac{\hbar}{2 M \omega_l}} \langle i \vert \nabla_u V_{eff} \cdot \mathbf e_l \vert j \rangle .

`\omega_l` and `\mathbf e_l` are the frequency and mass-scaled polarization
vector of the `l` th phonon. `M` is an effective mass and nabla_u denotes the
gradient wrt atomic displacements.

The implementation supports calculations of the el-ph coupling in both finite and
periodic systems, i.e. expressed in a basis of molecular orbitals or Bloch states.

The implementation is based on finite-difference calculations of the the atomic
gradients of the effective potential expressed on a real-space grid. The el-ph
couplings are obtained from LCAO representations of the atomic gradients of the
effective potential and the electronic states.

The current implementation supports spin-paired and spin-polarized computations.

A short example is given below. Another worked out example can be found in the
tutorial for Raman calculations :ref:`here <elphraman>`.

Example
=======

In a typical application one would compute the phonon modes separately as those
need very different convergence settings. (:git:`~doc/tutorialsexercises/vibrational/elph/phonon.py`)

.. literalinclude:: phonon.py

The corresponding calculation of the effective potential changes can be done
simultaneously. (:git:`~doc/tutorialsexercises/vibrational/elph/elph.py`)

.. literalinclude:: elph.py

The last line in the above script constructs the electron-phonon matrix in terms
of LCAO orbitals (and cell repetitions) and saves it as ASE JSON cache to the
`supercell` directory.

After both calculations are finished the final electron-phonon matrix can be constructed
with a 'simple' script. (:git:`~doc/tutorialsexercises/vibrational/elph/construct_matrix.py`)

.. literalinclude:: construct_matrix.py


Code
====

.. autoclass:: gpaw.elph.electronphonon.ElectronPhononCoupling
    :members:
