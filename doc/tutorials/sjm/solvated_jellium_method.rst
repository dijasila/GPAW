.. module:: gpaw.solvation.sjm
.. _solvated_jellium_method:

=============================
Solvated Jellium Method (SJM)
=============================

Theoretical Background
======================

The Solvated Jellium method (SJM) is simple method for the simulation 
of electrochemical interfaces in DFT. A full description of the model
can be found in [#SJM18]_. It can be used like the standard GPAW calculator,
meaning stable intermediates and reaction barriers can be calculated at
defined electrode potential via e.g. the Nudged Elastic Band method (NEB)
[#NEB00]_.

The basis of the model is keeping control of the electrode potential
by charging the electrodes interface, while keeping the periodic 
unit cell charge neutral. This is done by adding a JelliumSlab in
the region above the electrode surface. Doing so both electrons/holes
in the SCF cycle and spatially constant counter charge are introduced,
therefore keeping the net charge at 0.

Additionally, an implicit solvent [#HW14]_ is introduced above the slab,
which screens the electric field created by dipole consisting of electrode
and counter charge. 

The electrode potential is then defined as the Fermi Level (`\mu`) referenced
to the electrostatic potential deep in the solvent, where the whole 
charge on the electrode has been screened.

.. math:: \Phi_e = \Phi_w - \mu.

The energy used in the analysis of electrode reactions is the Grand Potential
Energy 

.. math:: \Omega = E_{tot} + \Phi_e N_e

.. autoclass:: gpaw.solvation.sjm.SJM

Usage Example: A simple Au(111) slab
====================================

As a usage example, given here is the calculation of a simple Au slab
at a potential of -1 V versus SHE. Keep in mind that the absolute 
potential has to be provided, where the value of the SHE potential on 
an absolute scale is approx. 4.4V. Additionally, the parameters used in 
the example are very bad, so no physically reasonable results should
be expected.

.. literalinclude:: Au111.py

Usage Example: Running a constant potential NEB calculation
===========================================================

A complete script for performing an NEB calculation can be downloaded here:

.. literalinclude:: run_SJM_NEB.py


.. Note:: In this example the keyword 'H2O_layer = True' in the 'SJM_Power12Potential'  
    class has been used. This keyword frees the interfacebetween the electrode 
    and a water layer from the implicit solvent. It is needed since the rather 
    high distance between the two subsystems would lead to partial solvation 
    of the interface region, therefore screening the electric field in the 
    most interesting area.



References
==========

.. [#SJM18] G. Kastlunger, P. Lindgren, A. A. Peterson, 
            coming soon, http://dx.doi.org/10.1021/acs.jpcc.8b02465.
.. [#NEB00] G. Henkelman and H. Jonsson,
            `Improved Tangent Estimate in the NEB method for Finding Minimum Energy Paths and Saddle Points <http://dx.doi.org/10.1063/1.1323224>`_,
            *J. Chem. Phys.* **113**, 9978 (2000)
.. [#HW14] A. Held and M. Walter,
           `Simplified continuum solvent model with a smooth cavity based on volumetric data <http://dx.doi.org/10.1063/1.4900838>`_,
           *J. Chem. Phys.* **141**, 174108 (2014).
