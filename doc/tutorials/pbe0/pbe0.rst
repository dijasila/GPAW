.. _pbe0_tut:

==================================
PBE0 calculations for bulk silicon
==================================

This tutorial will do non-selfconsistent PBE0 based on self-consistent PBE.

.. seealso::

   * :ref:`bandstructures` tutorial.
   * :ref:`band exercise` exercice.


PBE and PBE0 band gaps
======================

The band structure can be calculated like this:

.. literalinclude:: gaps.py

.. csv-table:: gaps.csv
    :header: **k**-points,
             `\Delta E_{PBE}(\Gamma, \Gamma)`,
             `\Delta E_{PBE}(\Gamma, X)`,
             `\Delta E_{PBE0}(\Gamma, \Gamma)`,
             `\Delta E_{PBE0}(\Gamma, X)`
             

Lattice constant and bulk modulus
=================================

.. literalinclude:: eos.py

.. image:: a.png
