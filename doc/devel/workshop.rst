.. _workshop:

====================================================
Electronic structure calculations with the GPAW code
====================================================

Users and developers meeting Technical University of Denmark, May 21-23, 2013
=============================================================================

The meeting provides an overview of the present status of the open
source electronic structure program GPAW. The latest developments will
be illustrated through invited and contributed talks from GPAW
developers/users and other scientists from computational condensed
matter community. Topics include many-body methods for electronic
transport and excitations, new exchange-correlation functionals,
coupled electron-ion dynamics and applications within catalysis and
energy materials research. The meeting comprises two days with oral
presentations and poster session relevant to anyone interested in
GPAW, followed by one day with hands-on coding activities for
developers.

`Sponsored by psi-k <http://www.psi-k.org/>`__.


Program
=======

All talks are 25 minutes plus 5 minutes for questions.


Tuesday (May 21)
----------------

.. list-table::
 :widths: 1 3 7

 * - 10:00
   - Jussi Enkovaara
   - Overview of GPAW
 * - 10:30
   - Jens Jørgen Mortensen
   - Short history of GPAW and latest news about PAW setups
 * - 11:00
   - Ask Hjorth Larsen
   - LCAO in GPAW
 * - 11:30
   - Samuli Hakala
   - Multi-GPU Accelerated Large Scale Electronic Structure Calculations
 * - 12:00
   - 
   - *Lunch*
 * - 13:30
   - Jess Wellendorff
   - Exchange-correlation functionals with error estimation
 * - 14:00
   - Bjørk Hammer
   - Meta-GGA versus vdW (optB88) for adsorption systems
 * - 14:30
   - Hannes Jónsson
   - Orbital density dependent functionals
 * - 15:00
   -
   - *Coffee*
 * - 15:30
   - Per Hyldgaard
   - Electron response in the Rutgers-Chalmers van der Waals density
     Functionals
 * - 16:00
   - Thomas Olsen
   - Extending the random phase approximation with renormalized adiabatic
     exchange kernels
 * - 16:30
   - Mikael Kuisma
   - Implementation of Spin-Polarized GLLB-SC potential for GPAW
 * - 18:00
   -
   - *Grill*


Wednesday (May 22)
------------------

.. list-table::
 :widths: 1 3 7

 * - 9:00
   - Jun Yan
   - Implementations and applications based on linear density response function
 * - 9:30
   - Arto Sakko, Tuomas Rossi
   - Combining TDDFT and classical electrodynamics simulations for plasmonics
 * - 10:00
   - Kristian Thygesen
   - Plasmonics with GPAW
 * - 10:30
   -
   - *Coffee*
 * - 11:00
   - Angel Rubio
   - Time-dependent density functional theory for non-linear phenomena
     in solids and nanostructures: fundamentals and applications
 * - 11:30
   - Michael Walter
   - Extensions to GPAW: From polarizable environments to excited state
     properties
 * - 12:00
   - 
   - *Lunch*
 * - 13:00
   - Olga Lopez-Acevedo
   - Multiscale simulations with ASE
 * - 13:30
   - Ivano E. Castelli
   - Computational screening of materials for water splitting
 * - 14:00
   -
   - *Coffee*
 * - 14:15
   - Martti Puska
   - Non-adiabatic electron-ion dynamics 
 * - 14:45
   - Ari Ojanperä
   - Applications of Ehrenfest dynamics: from excited state evolution of
     protected gold clusters to stopping of high-energy ions in graphene
 * - 15:15
   - Lauri Lehtovaara
   - Au40(SR)24 Cluster as a Chiral Dimer of 8-Electron Superatoms:
     Structure and Optical Properties


Thursday (May 23)
-----------------

Activities for GPAW developers (we start at 9:00):

* Coordination of code development and discussions about the future:
  Quick tour of ongoing projects --- what's the current status?
  
* Introduction to Sphinx and reStructuredText

* Introduction to testing of GPAW

* Hands on: Write new documentation/tutorials and how to make sure
  they stay up to date

* *Lunch*

* Status of unmerged branches:

  * rpa-gpu-expt
  * cuda
  * lcaotddft
  * lrtddft_indexed
  * aep1
  * libxc1.2.0

* Questions open for discussion:

  * When do we drop support for Python 2.4 and 2.5?
  * Strategy for porting GPAW to Python 3?
  * Switch from SVN to Bazaar and Launchpad?

* Hands on: Write new documentation/tutorials --- continued

* Presentations of today's work
