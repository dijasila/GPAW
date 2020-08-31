.. _smearing:

Occupation number smearing
==========================

.. module:: gpaw.occupations

.. seealso:: :ref:`manual_occ`

Bulk Cu with diferent smearing methods:

.. figure:: cu.png

(made with :download:`cu.py`).

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
