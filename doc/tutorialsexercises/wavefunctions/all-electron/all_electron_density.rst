.. _all electron density:

================================
Getting the all-electron density
================================

The variational quantity of the PAW formalism is the pseudo-density
`\tilde{n}`. This is also the density returned by the
:meth:`~gpaw.calculator.GPAW.get_pseudo_density` method of the GPAW
calculator. Sometimes it is desirable to work with the true all-electron
density.  The PAW formalism offers a recipe for reconstructing the all-electron
density from the pseudo-density, and in GPAW, this can be reached by
the method :meth:`~gpaw.calculator.GPAW.get_all_electron_density` of the
:class:`~gpaw.calculator.GPAW` class:

.. method:: get_all_electron_density(spin=None, gridrefinement=2, pad=True)

    Return reconstructed all-electron density array.


The :meth:`~gpaw.calculator.GPAW.get_all_electron_density` method is used in
the same way as you would normally use the
:meth:`~gpaw.calculator.GPAW.get_pseudo_density` method, i.e.:

.. literalinclude:: C6H6.py
   :end-before: literalinclude division line

would give you the pseudo-density in ``nt`` and the all-electron
density in ``n_ae``.

As the all-electron density has more structure than the
pseudo-density, it is necessary to refine the density grid used to
represent the pseudo-density. This can be done using the
``gridrefinement`` keyword of the ``get_all_electron_density`` method for ``n_ae_fine``:

.. literalinclude:: C6H6.py
   :start-after: literalinclude division line

Current only the values 1, 2, and 4 are supported (2 is default).

The all-electron density will always integrate to the total number of
electrons of the considered system (independent of the grid
resolution), while the pseudo density will integrate to some more or
less arbitrary number. This fact is illustrated in the following
example.

.. seealso::

    :ref:`bader analysis`


-------------
Example: NaCl
-------------

As an example of application, consider the three systems Na, Cl, and
NaCl. The pseudo- and all-electron densities of these three systems
can be calculated with the script :download:`NaCl.py`:

.. literalinclude:: NaCl.py

The result for the integrated pseudo- and all-electron densities of
the three systems is:

.. csv-table::
  :file: all_electron.csv
  :header: formula, ñ, n

From which we see that the all-electron densities integrate to the
total number of electrons in the system, as expected.
