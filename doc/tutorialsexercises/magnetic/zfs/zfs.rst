.. _zfs:

====================
Zero-field splitting
====================

.. warning:: **Work in progress**

This tutorial calculates the zero-field splitting for the NV center in diamond
and bi-radicals.

.. module:: gpaw.zero_field_splitting
.. autofunction:: zfs
.. autofunction:: convert_tensor


Examples
========

Diamond NV- center
------------------

For a NV center in a cubic supercell, the D and E values are presented below
with and without relaxaing the cell. The experimental value is around 2880
MHz.

.. csv-table::
    :file: zfs_nv.csv
    :header: atoms, relaxed, D (MHz), E(MHz)

:download:`diamond_nv_minus.py`.


Bi-radical
----------

:download:`biradical.py`.
:download:`plot.py`.
