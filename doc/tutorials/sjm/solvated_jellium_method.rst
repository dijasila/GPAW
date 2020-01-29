.. module:: gpaw.solvation.sjm
.. _solvated_jellium_method:

=============================
Solvated Jellium Method (SJM)
=============================

Theoretical Background
======================

The Solvated Jellium method (SJM) is a simple method for the simulation
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
therefore keeping the unit cell charge neutral.

Additionally, an implicit solvent [#HW14]_ is introduced above the slab,
which screens the electric field created by dipole consisting of electrode
and counter charge.

The electrode potential is then defined as the Fermi Level (`\mu`) referenced
to the electrostatic potential deep in the solvent, where the whole
charge on the electrode has been screened and no electric field is present.

.. math:: \Phi_e = \Phi_w - \mu.

The energy used in the analysis of electrode reactions is the Grand Potential
Energy

.. math:: \Omega = E_{tot} + \Phi_e N_e

Usage Example: A simple Au(111) slab
====================================

As a usage example, given here is the calculation of a simple Au slab
at a potential of -1 V versus SHE. Keep in mind that the absolute
potential has to be provided, where the value of the SHE potential on
an absolute scale is approx. 4.4V.

.. literalinclude:: Au111.py

The output in 'Au_pot_3.4.txt' is extended by the grand canonical energy
and contains the new part::

 ----------------------------------------------------------
 Grand Potential Energy (E_tot + E_solv - mu*ne):
 Extrpol:    -9.004478833351179
 Free:    -9.025229571371211
 -----------------------------------------------------------

These energies are written e.g. into trajectory files if
:literal:`write_grandcanonical_energy = True`.


Since we set :literal:`verbose = True`, the code produced three
files:

elstat_potential.out:
 Electrostatic potential averaged over xy and referenced to the systems
 Fermi Level. The outer parts should correspond to the respective work
 functions.

cavity.out:
 The shape of the implicit solvent cavity averaged over xy.

background_charge.out:
 The shape of the jellium background charge averaged over x and y.

.. Note:: Alternatively, :literal:`verbose = 'cube'` corresponds to :literal:`True`
          plus creation of a cube file including the dielectric function
          (cavity) on the 3-D grid.

Structure optimization
======================

Any kind of constant potential structure optimization can be performed by
applying ase's built-in optimizers. Two options are given for the potential
equilibration during structure optimization:

potential_equilibration_mode = 'seq':
 Sequential mode. This is the default optimization mode, which fully
 equilibrates the potential after each ionic step, if the current
 potential differs from the target by more
 than :literal:`dpot`. This mode is generally slower (up to 1.5 time CPU hours
 compared to constant charge GPAW), but very reliable.

potential_equilibration_mode = 'sim':
 Simultaneous mode. In this mode :literal:`ne` is constantly optimized together
 with the geometry. It is generally reliable (with few exceptions) and does not
 lead to a significant performance penalty compared to constant charge
 relaxations. However, after convergence it is adviced to check the final
 potential, since there is no guarantee for it to be within :literal:`dpot`.
 During the optimization the
 potential can oscillate, which mostly calms down close to convergence. One
 can, however, control the oscillation via :literal:`max_pot_deviation`. This
 keyword automatically triggers a tight and complete potential equilibration
 to the target, if the current potential is outside the given threshold.

Usage Example: Running a constant potential NEB calculation
===========================================================

A complete automatized script for performing a NEB calculation can be downloaded here:
:download:`run_SJM_NEB.py<run_SJM_NEB.py>`. It can, of course, be substituted
by a simpler, more manual script as can be found in
:ref:`the NEB tutorial<neb>`


.. Note:: In this example the keyword 'H2O_layer = True' in the 'SJM_Power12Potential'
    class has been used. This keyword frees the interface between the electrode
    and a water layer from the implicit solvent. It is needed since the rather
    high distance between the two subsystems would lead to partial solvation
    of the interface region, therefore screening the electric field in the
    most interesting area.
.. Note:: For paralle NEB runs :literal:`potential_equilibration_mode = 'sim'`
          should be used for efficiency, since not every image triggers
          a potential equilibration step in sequential mode. Such a run would
          lead to unnecessary idle time on some cores.
.. Note:: For CI-NEBs (with :literal:`climb = True`), we advice to either set
         :literal:`max_pot_deviation` to a tighter value (e.g. 0.05) in
         simultaneous mode or use the sequential mode.

.. autoclass:: gpaw.solvation.sjm.SJM


References
==========

.. [#SJM18] G. Kastlunger, P. Lindgren, A. A. Peterson,
            `Controlled-Potential Simulation of Elementary Electrochemical Reactions: Proton Discharge on Metal Surfaces <http://dx.doi.org/10.1021/acs.jpcc.8b02465>`_,
            *J. Phys. Chem. C* **122** (24), 12771 (2018)
.. [#NEB00] G. Henkelman and H. Jonsson,
            `Improved Tangent Estimate in the NEB method for Finding Minimum Energy Paths and Saddle Points <http://dx.doi.org/10.1063/1.1323224>`_,
            *J. Chem. Phys.* **113**, 9978 (2000)
.. [#HW14] A. Held and M. Walter,
           `Simplified continuum solvent model with a smooth cavity based on volumetric data <http://dx.doi.org/10.1063/1.4900838>`_,
           *J. Chem. Phys.* **141**, 174108 (2014).
