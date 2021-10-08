.. _solvated_jellium_method:

=============================
Solvated Jellium Method (SJM)
=============================

Please familiarize yourself with the theory of the SJM approach before starting this tutorial, by reading the :ref:`SJM documentation page<sjm>`.

Usage Example: A simple Au(111) slab
====================================

As a simple example, we'll examine the calculation of a Au slab with a water overlayer calculated at a potential of -1.0 V versus SHE.
To speed the computation, we'll use a tiny slab with a single water molecule over the top.

To use the solvated jellium approach, instead of importing GPAW as our calculator object, we import :class:`~gpaw.solvation.sjm.SJM`.
All of the SJM-specific parameters are fed to the calculator as a single dictionary called :literal:`sj`.
In this script, the :literal:`sj` dictionary contains only a single parameter: the target potential; the rest of the SJM parameters are kept at their default values.
Keep in mind that the *absolute* potential has to be provided, where the value of the SHE potential on an absolute scale is approximately 4.4 V.

Since SJM utilizes an implicit solvent to screen the field, we will need to specify some solvent parameters as well.
The SJM calculator is a subclass of the :class:`~gpaw.solvation.calculator.SolvationGPAW` calculator, so any solvation parameters can be passed straight through to SolvationGPAW.
Some reasonable values to simulate room-temperature water are below.
In practice, since the purpose of the implicit solvent is just to screen the field, the net results---such as binding energies and barrier heights---will be relatively insensitive to the choice of implicit solvent parameters, so long as they are kept consistent.

.. literalinclude:: Au111.py

If you examine the output in 'Au.txt', you'll see that the code varies the number of excess electrons until the target potential is within a tolerance of 3.4 V; the default tolerance is 0.01 V.
This process usually takes a few steps on the first image, but is faster on subsequent images of a trajectory, since the changes are less dramatic and the potential--charge slope is retained from previous steps.
You should see that in net the routine removed about 0.1 electrons from the simulation, as compared to a charge-neutral simulation, in order to achieve the desired work function.

You'll notice that the output in 'Au.txt' contains additional information, as compared to a charge-neutral simulation::

    Legendre-transformed energies (Omega = E - N mu)
      (grand potential energies)
      N (excess electrons):   -0.111796
      mu (workfunction, eV):   +3.394631
    --------------------------
    Free energy:    -21.542749
    Extrapolated:   -21.520483

These Legendre-transformed energies, `\Omega = E - N \mu`, are written into the any ASE trajectories created, and are the quantity returned by :literal:`calc.get_potential_energy()`.
As discussed in :ref:`grand-potential-energy`, these are the energies are consistent with the forces in the grand-canonical scheme, and are thuse compatible with method such as saddle-point searches and energy optimizations.
If you'd rather have it write out the traditional canonical energies, you can use the :literal:`write_grandcanonical_energy` keyword.

In the last line of the script above, we wrote out some additional information in the form of 'sjm traces'.
These are traces of the electrostatic potential, the background charge, and the solvent cavity across the `z` axis (that is, `xy`-averaged). 
It's a good idea to take a look at these, and perhaps make a plot for each, when running a simulation with a new system so you can be sure it is behaving as you expect.
The three files that are created are:

potential.txt:
 Electrostatic potential averaged over `xy` and referenced to the system's Fermi Level.
 The outer parts should correspond to the respective work functions and the potential at the top boundary should correspond to the potential set in the input.

cavity.txt:
 The shape of the implicit solvent cavity averaged over `xy`.
 Multiplying the cavity function with epsinf corresponds to the `xy`-averaged dielectric constant distribution in the unit cell.

background_charge.txt:
 The shape of the jellium background charge averaged over `xy` and normalized.

.. Note:: Alternatively, :literal:`calc.write_sjm_traces(style='cube')`
          can be used to write cube files on the 3-D grid, instead of traces.


FIXME/ap: PICKUP editing here.

FIXME/ap: Move the module documentation to the theory.

FIXME/ap: Most of the background should be remove from here and just linked to the theory page.

FIXME/ap: Update the script so that the radii part is gone, but make a note that this can be adjusted if the solvent penetrates the slab.

FIXME/ap: Be sure to note that ghost atoms can be added.

FIXME/ap: Show a plot of the dielectric versus position as a check that should be done at the start of calculations.

FIXME/ap: Discuss the old sequential versus simultaneous modes.

FIXME/ap: Show a simple NEB script, and have a link to the more elaborate one.

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
