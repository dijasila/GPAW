.. module:: gpaw.solvation.sjm
.. _sjm:

======================================================
Solvated Jellium (constant-potential electrochemistry)
======================================================

Overview
========

The Solvated Jellium method (:class:`~gpaw.solvation.sjm.SJM`) is a simple method for the simulation of electrochemical interfaces in DFT.
A full description of the approach can be found in [Kastlunger2018]_.
The method allows you to control the simulated electrode potential (manifested as the topside work function) by varying the number of electrons in the simulation; calculations can be run in either constant-charge or constant-potential mode.
The :class:`~gpaw.solvation.sjm.SJM` calculator can be used just like the standard GPAW calculator; it returns energy and forces, but can do so at a fixed potential.
(Please see the note below on the Legendre-transform of the energy.)
The potential is controlled by a simple iterative technique; in practice if you are running a trajectory (such as a relaxation or nudged elastic band) the first image will take longer than a conventional calculation as the potential equilibrates, but the computational cost is much less on subsequent images; practically, we estimate the extra cost to be <50% compared to a traditional DFT calculation of a full trajectory.
For a practical guide on the use of the method, please see the :ref:`solvated_jellium_method` tutorial.


Theoretical background
======================

The philosophy of the solvated jellium method is to construct the simplest model that captures the physics of interest, without introducing spurious effects.
The solvated jellium approach consists of two components: jellium and an implicit solvent.
A schematic is shown below:

.. image:: overview.png
           :width: 600 px
           :align: center

In this figure, the jellium is shown by the hashed marks, while the implicit solvent is shown in the blue shaded region. Note that an explicit solvent (the water molecules) is also conventionally used in this approach, as the major purpose of the implicit solvent is not to simulate the solvation of individual species but rather to screen the net field.
A more detailed discussion of both of these components follows.


The jellium slab: charging
--------------------------

In a periodic system we cannot have a net charge; therefore, any additional, fractional electrons that are added to the system must be compensated by an equal amount of counter charge.
In GPAW, this is conveniently accomplished with a :class:`~gpaw.jellium.JelliumSlab`.
This adds a smeared-out background charge in a fixed region of the simulation cell; in the figure above it is shown as the dashed region to the right of the atoms.
The :class:`~gpaw.jellium.JelliumSlab` also increases the number of electrons in the simulation by an amount equal to the total charge of the slab.
When you run a simulation, you should see that these excess electrons localize on the `+z` side (the right side in these figures) of the metal atoms that simulate the electrode surface, and not on the `-z` (left) side, which simulates the bulk.
This is accomplished by only putting the jellium region on one side of the simulation, and employing a dipole correction (included by default when you run SJM) to electrostatically decouple the two sides of the cell.

The figure below shows the difference in two simulations, one run at 4.4 V and one run at 4.3 V.
The orange curve shows where the potential drops off, and the blue curve shows where the electrons localize.

.. image:: traces.png
           :width: 600 px
           :align: center

The jellium region is conventionally thought of as a region of smeared-out positive charge, accompanied by a positive number of electrons.
However, the signs can readily be reversed, making the jellium region a smeared-out negative region accompanied by a reduction in the total number of electrons.
In this way, the same tool can be used to perturb the electrons in either a positive or negative direction, and thus vary the potential in either direction in order to find its target.
Note also that the jellium region does not overlap any atoms, separating this from approaches that employ a homogeneous background charge throughout the unit cell (in which spurious interactions can occur).
This is important to not distort the electronic structure of the atoms and molecules being simulated.
Additionally, note that the jellium is enclosed in a regular slab geometry in the figure above, but this need not be the case; it can, for example, follow the cavity of the implicit solvent if this is preferred (by using the :code:`jelliumregion` keyword as described in the :class:`~gpaw.solvation.sjm.SJM` documentation).


The solvation: screening
------------------------

By itself, the excess electrons and the jellium counter charge would set up an artificially high potential field in the region of the reaction.
To screen this large field, an implicit solvent is added to the simulation in the region above the explicit solvent, completely surrounding the jellium counter charge.
For this purpose, the solvated jellium method employs the implicit solvation model of Held and Walter [Held2014]_, which changes the dielectric constant of the vacuum region.
(You can learn more about the solvation method in the :ref:`continuum_solvent_model` tutorial.)

Here, the primary purpose of the implicit solvent is *not* to solvate the species reacting at the surface; explicit solvent (shown by the water molecules above) is typically employed in SJ simulations for this purpose.
The implicit solvent is located above the explicit solvent (and therefore may provide some solvent stabilization to the explicit solvent molecules).
This can be seen in the figure above, where the implicit solvent is shown as the blue shaded region.
In this figure, the small amount of solvent that is apparent at a `z` coordinate corresponding to the water layer is just the result of the implicit solvent penetrating slightly into the cavity at the center of a hexagonal ice-like water structure.
It is important that the implicit solvent not be present in the region of the reaction, as this would be "double"-solvating those parts.
If this occurs, "ghost" atoms can be added to exclude the solvent from specific regions.

Generalized Poisson equation
----------------------------

In net, the SJ method is manifested as two changes to the generalized Poisson equation,

.. math:: \nabla \Big(\epsilon(\br) \nabla \Phi(\br)\Big) = -4\pi \Big[ \rho_\mathrm{explicit}(\br) + \rho_\mathrm{jellium} (\br) \Big],

where `\epsilon(\br)` accounts for the solvation; that is, the dielectric constant is spatially variant, and the spatially-resolved charge density is modified by the presence of the `\rho_\mathrm{jellium}(\br)` term, which contains the smeared-out counter charge in a region away from all of the atoms (and electronic density) of the system.
`\rho_\mathrm{explicit} (\br)` contains the standard charge density of the system; that is, due to the electrons and nuclei.
Since the changes to the Poisson equation are relatively simple, it can be solved without relying on linearization.

The electrode potential
-----------------------

The electrode potential (`\phi_\mathrm{e}`) is then defined as the Fermi-level energy (`\mu`) referenced to a point deep in the solvent (`\Phi_\mathrm{w}`), where the whole charge on the electrode has been screened and no electric field is present.
(This is equivalently the topside work function of the slab.)
This is divided by the unit electronic charge `e` to convert from energy (typically in eV) to potential (typically in V) dimensions. 

.. math:: \phi_\mathrm{e} = \frac{\Phi_\mathrm{w} - \mu}{e} .

Note that this gives the potential with respect to vacuum; if you would like your potential on a reference electrode scale, such as SHE, please see the :ref:`solvated_jellium_method` tutorial.

.. _grand-potential-energy:

Legendre-transformed energy
---------------------------

The energy used in the analysis of electrode reactions is the grand-potential
energy

.. math:: \Omega \equiv E_\mathrm{tot} + \Phi_\mathrm{e} N_\mathrm{e} .

Whereas :math:`E_\mathrm{tot}` is consistent with the forces in traditional
electronic structure calculations, the grand-potential energy :math:`\Omega`
is consistent with the forces in electronically grand-canonical (that is,
constant-potential) simulations. This means that relaxations that follow forces
will find local minima in :math:`\Omega`, and generally methods that rely
on consistent force and energy information (such as BFGSLineSearch or NEB)
will work fine as long as :math:`\Omega` is employed. Thus, this calculator
returns :math:`\Omega` by default, rather than :math:`E_\mathrm{tot}`.

Potential control
=================

The below figure shows both the localization of excess electrons and the local change in potential, when the total number of electrons in an example simulation are changed.

.. image:: delta-ne-phi.png
           :width: 600 px
           :align: center

As mentioned above, the excess electrons localize only on the top side of the slab, which is meant to represent the electrode surface, and not on the bottom side, which is mean to represent the bulk.
The potential drop is seen to localize in the Stern layer where the reaction takes place.
Over reasonable deviations, the relationship between the number of excess electrons and the potential :math:`\phi` is approximately linear:

.. image:: charge-potential.png
           :width: 600 px
           :align: center

Due to the simple relationship between the excess electrons and the potential, reaching a desired potential is typically a fast process.
If you are running a trajectory---for example, a relaxation, a molecular dynamics simulation, or a saddle-point search---the first image will often take a few repetitions (that is, sequential constant-electron calculations) until the desired potential is reached.
Atoms typically move relatively little from image-to-image in a trajectory; therefore, subsequent images are often already at the target potential and no equilibration steps are necessary; when equilibration steps are required, the slope (of potential vs. number of electrons) is recalled from the last adjustment, and it often only takes a single equilibration step.
Typically, over the course of a full trajectory, the added computational cost of working in the constant-potential ensemble is minimal, generally <50% greater computational time compared to a constant-charge calculation.
As described in the  :ref:`solvated_jellium_method` tutorial, this can sometimes be further improved by simultaneously optimizing the potential with the atomic positions.


References
==========

.. [Kastlunger2018] G. Kastlunger, P. Lindgren, A. A. Peterson,
                    :doi:`Controlled-Potential Simulation of Elementary Electrochemical Reactions: Proton Discharge on Metal Surfaces <10.1021/acs.jpcc.8b02465>`,
                    *J. Phys. Chem. C* **122** (24), 12771 (2018)
.. [Held2014] A. Held and M. Walter,
           :doi:`Simplified continuum solvent model with a smooth cavity based on volumetric data <10.1063/1.4900838>`,
           *J. Chem. Phys.* **141**, 174108 (2014).

Class documentation
===================

.. autoclass:: gpaw.solvation.sjm.SJM

.. autoclass:: gpaw.solvation.sjm.SJMPower12Potential
