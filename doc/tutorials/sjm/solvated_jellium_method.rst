.. module:: gpaw.solvation.sjm
.. _solvated_jellium_method:

FIXME/ap: Move the module documentation to the theory.

FIXME/ap: Most of the background should be remove from here and just linked to the theory page.

FIXME/ap: Update the script so that the radii part is gone, but make a note that this can be adjusted if the solvent penetrates the slab.

FIXME/ap: Be sure to note that ghost atoms can be added.

FIXME/ap: Show a plot of the dielectric versus position as a check that should be done at the start of calculations.

FIXME/ap: Discuss the old sequential versus simultaneous modes.

FIXME/ap: Show a simple NEB script, and have a link to the more elaborate one.

=============================
Solvated Jellium Method (SJM)
=============================

Theoretical Background
======================

The Solvated Jellium method (SJM) is a simple method for the simulation
of electrochemical interfaces in DFT. A full description of the model
can be found in [#SJM18]_. The method allows you to change the simulated
electrode potential (manifested as the work function) by varying the
number of electrons in the simulation; calculations can be run in either
constant-charge or constant-potential mode.
It can be used like the standard GPAW calculator,
meaning stable intermediates and reaction barriers can be found, at
defined electrode potentials, by using standard methods such as
QuasiNewton or the Nudged Elastic Band (NEB) [#NEB00]_ method.

The basis of the model is keeping control of the electrode potential
by adding or subtracting fractions of an electron to the system (that is,
charging the electrode's interface), while keeping the periodic
unit cell charge neutral. This is done by adding a region of jellium in
the vacuum above the electrode surface. In so doing, both electrons/holes
are added to the SCF cycle and a spatially constant counter charge are
introduced, therefore keeping the unit cell charge-neutral.

Additionally, the jellium region is immersed in an implicit solvent [#HW14]_,
which screens the electric field created by the dipole consisting of electrode
and counter charge.

The electrode potential is then defined as the Fermi Level (`\mu`) referenced
to the electrostatic potential deep in the solvent, where the whole
charge on the electrode has been screened and no electric field is present.

.. math:: \Phi_\mathrm{e} = \Phi_\mathrm{w} - \mu.

The energy used in the analysis of electrode reactions is the grand-potential
energy

.. math:: \Omega \equiv E_\mathrm{tot} + \Phi_\mathrm{e} N_\mathrm{e}

Whereas :math:`E_\mathrm{tot}` is consistent with the forces in traditional
electronic structure calculations, the grand-potential energy :math:`\Omega`
is consistent with the forces in electronically grand-canonical (that is,
constant-potential) simulations. This means that relaxations that follow forces
will find local minima in :math:`\Omega`, and generally methods that rely
on consistent force and energy information (such as BFGSLineSearch or NEB)
will work fine as long as :math:`\Omega` is employed. Thus, this calculator
returns :math:`\Omega` by default, rather than :math:`E_\mathrm{tot}`.

Usage Example: A simple Au(111) slab
====================================

FIXME/ap: Simplify this script a bit. I.e., we can remove jelliumregion.
Do we need so many imports from solvation?

As a simple example, we'll examine the calculation of a simple Au slab
at a potential of -1 V versus SHE. Keep in mind that the absolute
potential has to be provided, where the value of the SHE potential on
an absolute scale is approximately 4.4 V.

.. literalinclude:: Au111.py

If you examine the output in 'Au.txt', you'll see that the code
varies the number of excess electrons until the target potential
is approximately 3.4 V. This process usually takes a few steps
on the first image, but often takeso

FIXME/ap: pick up here! And change the output below to have right
mu!

The output in 'Au.txt' is extended by the grand canonical energy
and contains the new part::

    Legendre-transformed energies (Omega = E - N mu)
      (grand potential energies)
      N (excess electrons):   +0.100000
      mu (workfunction, eV):   +3.564009
    --------------------------
    Free energy:     -8.997787
    Extrapolated:    -8.976676

These energies are written e.g. into trajectory files if
:literal:`write_grandcanonical_energy = True` (default).

After converging the constant potential scf loop we can write
some additional information in a xy format for plotting. By
default three files will be written in the created `sjm_traces` directory:

elstat_potential.out:
 Electrostatic potential averaged over xy and referenced to the systems
 Fermi Level. The outer parts should correspond to the respective work
 functions and the potential at the right boundary should correspond to
 the potential set in the input.

cavity.out:
 The shape of the implicit solvent cavity averaged over xy. Multiplying
 the cavity function with epsinf corresponds to the xy-averaged dielectric
 constant distribution in the unit cell.

background_charge_XX.out:
 The shape of the jellium background charge averaged over xy and
 normalized.

.. Note:: Alternatively, :literal:`calc.write_sjm_traces(style='cube')`
          can be used to write cube files of the three described
          quantities on the 3-D grid.

Structure optimization
======================

Any kind of constant potential structure optimization can be performed by
applying ase's built-in optimizers. Here, it is wort noting that if
:literal:`target_potential` is set to None (default), a constant charge optimization
is performed. In such a case :literal:`excess_electrons` will define the charge.

The :literal:`always_adjust` keyword can be used to speed up an optimization.
If it is set to False (default), the potential is only equilibrated if
the resulting potential is outside of the the set :literal:`tol`. This is slower,
but very robust. Setting :literal:`always_adjust=True` will make the code equilibrate
the potential at every step. In this case :literal:`tol` can be loosened speeding
up the optimization substantially.

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
.. Note:: For paralle NEB runs :literal:`always_adjust = True`
          should be used for efficiency, since not every image triggers
          a potential equilibration step in sequential mode. Such a run would
          lead to unnecessary idle time on some cores.
.. Note:: For CI-NEBs (with :literal:`climb = True`), we advice to either set
         :literal:`tol` to a tighter value (e.g. 0.05) if
         :literal:`always_adjust=True` or set the keyword to False

.. autoclass:: gpaw.solvation.sjm.SJM


References
==========

.. [#SJM18] G. Kastlunger, P. Lindgren, A. A. Peterson,
            :doi:`Controlled-Potential Simulation of Elementary Electrochemical Reactions: Proton Discharge on Metal Surfaces <10.1021/acs.jpcc.8b02465>`,
            *J. Phys. Chem. C* **122** (24), 12771 (2018)
.. [#NEB00] G. Henkelman and H. Jonsson,
            :doi:`Improved Tangent Estimate in the NEB method for Finding Minimum Energy Paths and Saddle Points <10.1063/1.1323224>`,
            *J. Chem. Phys.* **113**, 9978 (2000)
.. [#HW14] A. Held and M. Walter,
           :doi:`Simplified continuum solvent model with a smooth cavity based on volumetric data <10.1063/1.4900838>`,
           *J. Chem. Phys.* **141**, 174108 (2014).
