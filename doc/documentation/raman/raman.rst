.. _raman:

========================
Raman spectroscopy
========================

GPAW implements Raman spectroscopy for zone-center phonons of extended systems
using the electron-phonon coupling (see :ref:`elph`) within the LCAO mode.

The implementation is based upon Ref. [#Taghizadeh2020]_ .


Example
=======

In a typical application one would compute the phonon modes electron-phonon
components separately as those need very different convergence settings. An
example can found under :ref:`elph`.

The Raman code offers wrappers for the electron-phonon part of the calculations:
:meth:`~gpaw.raman.elph.run_elph`
:meth:`~gpaw.raman.elph.calculate_supercell_matrix`
:meth:`~gpaw.raman.elph.get_elph_matrix`

where the last function is currently not parallelised and needs to be run in a
separate serial run.

Using the wrappers an electron-phonon calculation could look like this
(:git:`~doc/documentation/raman/elph.py`):

.. literalinclude:: elph.py

The last steps, which usually are not computationally intensive, have not been
parallelised yet, and need to be executed in a serial run.

The previously calculated supercell matrix needs to be converted into the
electron-phonon matrix for the Bloch states, which can be like this
(:git:`~doc/documentation/raman/elph_matrix.py`):

.. literalinclude:: elph_matrix.py

The electron-phonon matrix is saved as ``gsqklnn.npy`` and is our first
ingredient for the Raman calculation.

The optional ``load_gx_as_needed`` tag prevents from all supercell pickle files
being read at once. Instead they are loaded and processed one-by-one. This can
save lots of memory for larger systems with hundreds of atoms, where the
supercell matrix can be over 100GiB large.

Our second ingredient is the transition dipole moments at the ground-state
structure, which are computed like this
(:git:`~doc/documentation/raman/dipole_transitions.py`):

.. literalinclude:: dipole_transitions.py

This script save the transition dipole moments as ``dip_svknm.npy``. The last
ingredient, the pickle file for the phonons should have been calculated in a
separate calculation as well. Now the Raman intensities can be computed
(:git:`~doc/documentation/raman/raman_intensities.py`):

.. literalinclude:: raman_intensities.py

The :meth:`calculate_raman` function computes the Raman tensor for each mode
for the given incident and outgoing directions. The results are saved as
``Rlab_??.npy`` files. The optional ``resonant_only`` tag can be used to
deactivate the calculation of the last 5 terms in Ref. [#Taghizadeh2020]_ Eq.10.
This might be necessary for very large unit cells.

The :meth:`calculate_raman_intensity` function computes the Raman intensity on
a frequency grid using the ``Rlab`` files and and saves them into ``RI_??.npy``
files.

Lastly, we can plot the ``RI`` files with :meth:`plot_raman`. As the Raman
intensities are saved as ``npy`` files the users can of course use their own
routines instead for plotting.

.. image:: Raman_xx_532nm.png
   :scale: 30
   :align: center 

----------
References
----------

.. [#Taghizadeh2020] A. Taghizadeh, U. Leffers, T.G. Pedersen, K.S. Thygesen,
                   "A library of ab initio Raman spectra for automated
                   identification of 2D materials",
                   *Nature Communications* **11**, 3011 (2020).

----
Code
----

.. autofunction:: gpaw.raman.elph.run_elph
.. autofunction:: gpaw.raman.elph.calculate_supercell_matrix
.. autofunction:: gpaw.raman.elph.get_elph_matrix
.. autofunction:: gpaw.raman.dipoletransition.get_dipole_transitions
.. autofunction:: gpaw.raman.raman.calculate_raman
.. autofunction:: gpaw.raman.raman.calculate_raman_intensity
.. autofunction:: gpaw.raman.raman.plot_raman
