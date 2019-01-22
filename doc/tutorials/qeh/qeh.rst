.. _qeh tutorial:
.. module:: gpaw.qeh

=====================================================
The Quantum Electrostatic Heterostructure (QEH) model
=====================================================

Brief overview
==============

In this tuturial we show how to calculate the linear response of a general van
der Waals Heterostructure (vdWH) by means of the quantum electrostatic
heterostructure (QEH) model. This method allows to overcome the computational
limitation of the standard ab-initio approaches by combining quantum accuracy
at the monolayer level and macroscopic electrostatic coupling between the
layers. More specifically, the first step consists in calculating the response
function of each of the layers composing the vdWHs in the isolated condition and
encoding it into a so called dielectric building block. Then, in the next step
the dielectric building blocks are coupled together through a macroscopic Dyson
equation. The validity of such an approach strongly relies on the absence of
hybridization among the layers, condition which is usually satisfied by vdWHs.

A thourough description of the QEH model can be found in [#qeh_theory]_:

.. [#qeh_theory] K. Andersen, S. Latini and K.S. Thygesen
    Dielectric genome of van der Waals heterostructures
    *Nano Letters* **15** (7), 4616-4621 (2015)


Constructing a dielectric building block
========================================

First, we need a ground state calculation for each of the layers in the vdWH.
We will consider a MoS2/WSe2 bilayer. In the following script
we show how to get the ground state gpw-file for the MoS2 layer:

.. literalinclude:: gs_MoS2.py

The gpw-file for WSe2 can be obtained in a similar way.
The gpw-files stores all the necessary eigenvalues and eigenfunctions for the
response calculation.

Next the building blocks are created by using the *BuildingBlock* class.
Essentially, a Dyson equation for the isolated layer is solved to obtain the
the full response function `\chi(q,\omega)`. Starting from `\chi(q,\omega)`
the monopole and dipole component of the response function and of the induced
densities are condensed into the dielectric building block. Here is how to get
the MoS2 and building block:

.. literalinclude:: bb_MoS2.py

The same goes for the WSe2 building block.
Once the building blocks have been created, they need to be interpolated to the
same kpoint and frequency grid. This is done as follows:

.. literalinclude:: interpolate_bb.py

Specifically, this interpolates the WSe2 building block to the MoS2 grid.

Finally the building blocks are ready to be combined electrostatically.


Interlayer excitons in MoS2/WSe2
================================

As shown in [#interlayer]_ the MoS2_WSe2 can host excitonic excitations where
the electron and the hole are spatially separated, with the electron sitting
in the MoS2 layer and the hole in the WSe2 one.

Because of the finite separation distance we expect that the electron-hole
screened interaction between the two particles will not diverge when the
in-plane separation goes to 0. To illustrate this we show how to calculate
the screened electron-hole interaction using the QEH model and the building
blocks created above:


.. literalinclude:: interlayer.py
    :end-before: get_exciton_binding_energies

Here is the generated plot:

.. image:: W_r.svg

As expected the interaction does not diverge!

If one is also interested in the interlayer exciton binding energy, it can be
readily calculated by appending the following lines in the script above:

.. literalinclude:: interlayer.py
    :start-after: show

We find an interlayer exciton binding energy of `\sim 0.3` eV!

Command line interface for the QEH code and getting the default dielectric BBs
==============================================================================
The QEH code includes a simple command line interface (CLI) that makes playing
around with different heterostructures easy. In order to use the CLI easily it is
recommended to bind an alias to the qeh module
(e. g. ``alias qeh="python3 -m gpaw.qeh"``). For example, plasmons in a
doped graphene boron-nitride heterostructure can be calculated and plottet by::

  >> qeh graphene+doping=0.5 3BN graphene+doping=0.5 --plasmons --plot

(Note that the first time you call this command you will most likely not have
downloaded the pre-calculated dielectric building blocks. Simply follow the
instructions shown in the terminal if this is the case, and run the command
above again.)
  
To view the full documentation for the QEH CLI use::

  >> qeh -h

Default layer thicknesses
=========================
Working with the QEH heterostructure class gives the user maximum freedom to
choose all input parameters (distances between layers etc.), which on the other
hand is tedious. To more easily set up heterostructures without having to specify
all standard parameters for layers use the ``make_heterostructure`` function
from the QEH module. For example, to make a similar calculation of graphene hBN
heterostructure plasmons to the example in the description of command line
interface simply do::

  from gpaw.qeh import make_heterostructure, plot_plasmons

  layers = ['graphene+doping=0.5', '3BN', 'graphene+doping=0.5']
  het = make_heterostructure(layers)
  output = het.get_plasmon_eigenmodes()
  plot_plasmons(output)

The QEH module then uses default values for layer thicknesses calculated from the interlayer distances of each layer its corresponding bulk phases as found in the inorganic crystal structure database (ICSD).

``make_heterostructure`` takes additional arguments for user specified layer thicknesses and momentum and frequency grids. To use these simply do::

  from gpaw.qeh import make_heterostructure, plot_plasmons

  layers = ['graphene+doping=0.5', '3BN']
  omegamin = 0.001  # Do not set to zero
  omegamax = 1.5
  nomega = 1000
  qmin = 0.001  # Do not set to zero
  qmax = 0.01
  nq = 100
  het = make_heterostructure(layers,
                             frequencies=[omegamin, omegamax, nomega],
                             momenta=[qmin, qmax, nq],
                             thicknesses=[3.3, 3.3, 3.3, 3.3])
  output = het.get_plasmon_eigenmodes()
  plot_plasmons(output)

Doped semiconductors
====================
With the QEH module it is possible to calculate the response of doped semi
conductors. This is done by specifying a doping level and an effective mass::

  from gpaw.qeh import make_heterostructure, plot_plasmons

  layers = ['H-MoS2+doping=0.1,em=0.43', 'BN',
            'H-MoS2+doping=0.1,em=0.43']
  het = make_heterostructure(layers)
  output = het.get_plasmon_eigenmodes()
  plot_plasmons(output)

The effect of doping is included as a free electron gas with the specified effective mass. If no effective mass is specified the QEH module falls back to default values for the effective mass. Currently the QEH module knows effective masses for all transition metal dichalcogenides calculated in the Computational 2D materials Database project (C2DB `<http://c2db.fysik.dtu.dk/>`_).

Phononic contribution to the dielectric response
================================================
Polar crystals will have an additional contribution to their dielectric response
originating from coupling of electric field to phonons (in addition to the
electronic response). This effect can be included by adding ´´+phonons´´
similar to how doping is included for semiconductors. For example to calculate
coupling between graphene plasmons and phonons in boron nitride do::

  from gpaw.qeh import make_heterostructure, plot_plasmons
  layers = ['graphene+doping=0.1', 'BN+phonons']
  het = make_heterostructure(layers)
  output = het.get_plasmon_eigenmodes()
  plot_plasmons(output)

Which is equivalent to calling the CLI with::

  >> qeh graphene+doping=0.1 BN+phonons --plasmons --plot

  
.. [#interlayer] S. Latini, K.T. Winther, T. Olsen and K.S. Thygesen
   Interlayer Excitons and Band Alignment in MoS2/hBN/WSe2
   van der Waals Heterostructures
   *Nano Letters*, 2017, 17 (2), pp 938–945
