.. _elphtheory:

===============================
Electron-Phonon Coupling Theory
===============================

Phonons can interact with the electrons in a variety of ways. For example, when an electron moves through the crystal, it can scatter off of a phonon, thereby transferring some of its energy to the lattice. Conversely, when a phonon vibrates, it can create an oscillating electric field that can interact with the electrons, inducing a change in their energies and momenta.
The coupling between electrons and lattice vibrations is responsible for a range of interesting and important phenomena, from electrical and thermal conductivity to superconductivity.

The first order electron-phonon coupling matrix `g_{mn}^\nu(\mathbf{k}, \mathbf{q})` couples the electronic states `m(\mathbf{k}+ \mathbf{q}),n(\mathbf{k})` via phonons `\nu` at wave vectors `\mathbf{q}` and frequencies `\omega_\nu`:[#Giustino2017]_

.. math::

    g_{mn}^\nu(\mathbf{k}, \mathbf{q}) = \sqrt{  \frac{\hbar}{2 m_0 \omega_\nu}} M_{mn}^\nu(\mathbf{k}, \mathbf{q}) .

with

.. math::

    M_{mn}^\nu(\mathbf{k}, \mathbf{q}) = \langle \psi_{m \mathbf{k}+ \mathbf{q}}  \vert \nabla_u V^{KS} \cdot \mathbf{e}_\nu \vert \psi_{n\mathbf{k}} \rangle.

Here `m_0` is the sum of the masses of all the atoms in the unit cell and `\nabla_u` denotes the gradient with respect to atomic displacements. For the three translational modes at `\vert \mathbf{q} \vert = 0` the matrix elements `g_{mn}^\nu = 0`, as a consequence of the acoustic sum rule.

--------------
Implementation
--------------

Within the PAW framework to Kohn-Sham potential can be split into a local part `V(\mathbf{r})` represented on a regular grid and a nonlocal part `\Delta H^a_{i_1 i_2}`:

.. math::

    V^{KS} = V + \Delta H^a_{i_1 i_2}.

In GPAW `\nabla_u V^{KS}(\mathbf{r})` is determined using the finite difference method in a supercell. The potential at the displaced coordinates is computed by the :meth:`~gpaw.elph.DisplacementRunner` class, which is based on ASEs ``ase.phonon.Displacement`` class. The central difference derivative, as evaluted in the :meth:`~gpaw.elph.Supercell` class, consists of four contributions:

.. math::

    \langle \psi_{i}  \vert \nabla_u V^{KS} \vert \psi_{j} \rangle =
    \langle \tilde \psi_{i}  \vert \nabla_u V(\mathbf{r}) \vert \tilde \psi_{j} \rangle +
    \sum_{a, ij} \langle  \tilde \psi_{i} \vert \tilde p^a_i \rangle (\nabla_u \Delta H^a_{i_1 i_2}) \langle \tilde p^a_j \vert \tilde \psi_{j} \rangle +
    \sum_{a, ij} \langle  \tilde \psi_{i} \vert \nabla_u  \tilde p^a_i \rangle \Delta H^a_{i_1 i_2} \langle \tilde p^a_j \vert \tilde \psi_{j} \rangle +
    \sum_{a, ij} \langle  \tilde \psi_{i} \vert \tilde p^a_i \rangle \Delta H^a_{i_1 i_2} \langle \sum_{a, ij} \langle  \tilde \psi_{i} \vert \nabla_u  \tilde p^a_i \rangle \Delta H^a_{i_1 i_2} \langle \tilde p^a_j \vert \tilde \psi_{j} \rangle \tilde p^a_j \vert \tilde \psi_{j} \rangle


Here we do not project the derivatives onto electronics states actually, not rather onto LCAO orbitals `\Psi_{NM}`, where `N` denotes the cell index and `M` the orbital index. We use a The Fourier transform from the `\mathbf{k}`-space Bloch  to the real space representation so that we can can later to compute `M_{mn}^\nu` for arbitrary `\mathbf{q}`:

.. math::

    \mathbf{g}_{\substack{N M\\	N^\prime M^\prime}}^{sc} =  FFT\left[ \langle \Psi_{NM}(\mathbf{k}) \vert \nabla_u V^{KS} \vert \Psi_{N^\prime M^\prime}(\mathbf{k}) \rangle \right].

Finally, the electron-phonon coupling matrix is obtained by projecting the supercell matrix into the primitive unit cell bands `m, n` and phonon modes `\nu` in :meth:`~gpaw.elph.ElectronPhononMatrix`:

.. math::

    M_{mn}^\nu(\mathbf{k}, \mathbf{q}) = \sum_{\substack{N M\\	N^\prime M^\prime}} C_{mM}^{\star} C_{nM^\prime} \mathbf{g}_{\mu}^{sc} \cdot \mathbf{u}_{q \nu} e^{2\pi i [(\mathbf{k}+\mathbf{q})\cdot \mathbf{R_N} - \mathbf{k}\cdot \mathbf{R_N^\prime}]},

where `C_{nM}` are the LCAO coefficients and `\mathbf{u}_{q \nu}` are the mass-scaled phonon displacement vectors.


Checkout :ref:`elph` for an example and exerice and :ref:`raman` and :ref:`elphraman` for an application.

----------
References
----------

.. [#Giustino2017] F. Giustino, "Electron-phonon interactions from first principles", Reviews of Modern Physics 89, 015003 (2017).


Code
====

.. autoclass:: gpaw.elph.DisplacementRunner
    :members:

.. autoclass:: gpaw.elph.Supercell
    :members:

.. autoclass:: gpaw.elph.ElectronPhononMatrix
    :members: 
