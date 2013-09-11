=====================
Testing PAW data sets
=====================

.. toctree::
    
    
Comparing to non-relativstic FHI-aims calculations
==================================================

By comparing non-relativistic calculations using FHI-aims and GPAW, the only difference between the two codes should be the finite basis set for FHI-aims and the PAW approximation for GPAW.

We do calculations for bulk FCC and bulk oxides in the rocksalt structure.

In the figures below the colors indicate the absolute errors compared to FHI-aims numbers. The number of valence electrons explicitly included in the PAW data sets are shown for each element.  As an example, there are two versions of the PAW data sets for Fe:  one with 8 electrons and another with 16 electrons.

The oxide formation energy in the rocksalt structure for element A is defined as:
    
.. math:: F = E^{RS}(AO) - E^{FCC}(A) - E^{FCC}(O).
    
.. figure:: dfe.png
   :width: 100%

The compression energy is defined as:
    
.. math:: C = E^{X}(0.9 a_r^X) - E^{X}(a_r^X),

where `a_r^X` is the equilibrium lattice constant calculated with relativistic corrections.

.. figure:: dce.png
   :width: 100%
   
.. figure:: a.png
   :width: 100%


All the numbers can be found :ref:`below <bulk>`.


Convergence of the energy with respect to number of plane-waves
===============================================================

For convergence of relative energies (shown below for a cut-off of 350 eV) we use the difference between FCC and an isolated atom.  For absolute energy convergence, see the table :ref:`below <conv>`.

.. figure:: conv.png
   :width: 100%


Eggbox errors
=============

Eggbox errors:
    
.. figure:: egg.png
   :width: 100%


Test of LCAO basis sets
=======================

The three figure below compare LCAO with the dzp basis sets to converged plane-wave calculations.

.. figure:: dfelcao.png
   :width: 100%

.. figure:: dcelcao.png
   :width: 100%

.. figure:: alcao.png
   :width: 100%


All the numbers    
===============

.. _bulk:

Non-relativistic calculations compared to FHI-aims:
    
.. csv-table::
    :header: , `F`, `\\Delta F`, `\\Delta C^{FCC}`, `\\Delta C^{RSXXX}`, `a^{FCC}`, `\\Delta a^{FCC}`, `a^{RSXXX}`, `\\Delta a^{RSXXX}`
    :file: bulk.csv

Energy difference convergence (FCC minus isolated atom, relative to calculation at 600 ev):

.. csv-table::
    :header: , 300, 350, 400, 450, 500, 550
    :file: relconv.csv

.. _conv:

Absolute energy convergence (FCC, relative to calculation at 1000 ev):
        
.. csv-table::
    :header: , 300, 350, 400, 450, 500, 550, 600
    :file: absconv.csv

Eggbox errors in meV for diferent grid-spacings:

.. csv-table::
    :header: , 0.18, 0.18, 0.18 
    :file: egg.csv

LCAO calculations with dzp basis set:
    
.. csv-table::
    :header: , `\\Delta F`, `\\Delta C^{FCC}`, `\\Delta C^{RS}`, `\\Delta a^{FCC}`, `\\Delta a^{RS}`, `\\Delta a^{FCC}`, `\\Delta a^{RS}`
    :file: lcao.csv
