.. _elph:

========================
Electron-phonon coupling
========================

At the heart of the electron-phonon coupling is the calculation of the gradient of the effective potential, which is done using finite displacements just like the phonons. Those two calculations can run simultaneous, if the required set of parameters coincide. The electron-phonon matrix is quite sensitive to self-interaction of a displaced atom with its periodic images, so a sufficiently large supercell needs to be used. The (3x3x3) supercell used in this example is a bit too small. When using a supercell for the calculation you have to consider, that the atoms object needs to contain the primitive cell, not the supercell, while the parameters for the calculator object need to be good for the supercell, not the primitive cell. (:git:`~doc/tutorialsexercises/vibrational/elph/effective_potential.py`)

.. literalinclude:: effective_potential.py

This script executes `2*3*N` displacements and saves the change in total energy and effective potential into a file cache in the directory `elph`.
The phonon/effective potential calculation can take quite some time, but can be distributed over several images. The :meth:`~gpaw.elph.DisplacementRunner` class is based on ASEs ``ase.phonon.Displacement`` class, which allows to select atoms to be displaced using the ``set_atoms`` function.

In the second step we map the gradient of the effective potential to the LCAO orbitals of the supercell (:git:`~doc/tutorialsexercises/vibrational/elph/supercell.py`).
For this we first need to create the supercell

.. literalinclude:: supercell.py
    :start-at: atoms
    :end-at: atoms_N

and run GPAW to obtain the wave function. This step currently currently only
works for k-point parallelization, so you need to include
``parallel={'domain': 1, 'band': 1}`` in your script.

.. literalinclude:: supercell.py
    :start-at: sc = Supercell(atoms, supercell=(3, 3, 3))
    :end-at: sc.calculate_supercell_matrix(calc, fd_name='elph')

invokes the calculation of the supercell matrix, which is stored in the `supercell` file cache.

After both calculations are finished the final electron-phonon matrix can be constructed. (:git:`~doc/tutorialsexercises/vibrational/elph/gmatrix.py`)

.. literalinclude:: gmatrix.py

For this we need to perform another ground state calculation
(:git:`~doc/tutorialsexercises/vibrational/elph/scf.py`) to obtain the wave
function used to project the electron-phonon coupling matrix into. The
electron-phonon matrix is computed for a list of **q** values, which need to
be commensurate with the **k**-point mesh chosen. The
``ElectronPhononMatrix`` class doesn't like parallelization so much and
should be done separately from the rest.


----------
Exercise
----------

The electron-phonon matrix can be used to calculate acoustic and optical deformation potentials without much effort. The optical deformation potential is identical to the bare electron-phonon coupling matrix Ref. [#Li2017]_:

.. math::

    D_{ODP} =  \langle m \vert \nabla_u V_{eff} \cdot \mathbf e_l \vert n \rangle .


For the silicon VBM at Gamma, for the LO phonons at Gamma, we expect a deformation potential of about `\vert M \vert \approx 3.6` eV/Angstrom.

Converge the deformation potential of Si with respect to the supercell size, k-points, grid spacing and SCF convergence parameters.

----------
References
----------

.. [#Li2017] Z. Li, P. Graziosi, and N. Neophytou, "Deformation potential extraction and computationally efficient mobility calculations in silicon from first principles", Physical Review B 104, 195201 (2021)

