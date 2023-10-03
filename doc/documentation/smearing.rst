.. _smearing:

Occupation number smearing
==========================

.. module:: gpaw.occupations

.. seealso:: :ref:`manual_occ`

Convergence with respect to number of k-point for bulk Cu energy with
different smearing methods:

.. literalinclude:: cu_calc.py

.. figure:: cu.png

(made with :download:`cu_plot.py`).  See also figure 3 in
:doi:`Blöchl et. al <10.1103/PhysRevB.49.16223>`.


.. autofunction:: create_occ_calc
.. autofunction:: fermi_dirac
.. autofunction:: marzari_vanderbilt
.. autofunction:: methfessel_paxton
.. autoclass:: OccupationNumberCalculator
   :members:
.. autoclass:: FixedOccupationNumbers
.. autoclass:: ParallelLayout
.. autofunction:: occupation_numbers


Tetrahedron method
------------------

.. module:: gpaw.tetrahedron
.. autoclass:: TetrahedronMethod
