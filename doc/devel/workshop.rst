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


Location
========

Day 1 and 2 of the workshop will take place at DTU_ in building 306,
auditorium 37.  Day 3 will be in building 307, room 127.  For
directions, see this map_.

.. _DTU: http://www.dtu.dk/english
.. _map: http://www.dtu.dk/english/about_dtu/dtu%20directory/map_of_lyngby.aspx


Program
=======

All talks are 25 minutes plus 5 minutes for questions (except three
talks that are 15+5).


Tuesday (May 21)
----------------

.. list-table::
 :widths: 1 3 7 2

 * - 10:00
   - Kristian Thygesen
   - Welcome
   -
 * - 10:00
   - Jussi Enkovaara
   - `Overview of GPAW
     <http://dcwww.camp.dtu.dk/~jensj/gpaw2013/a2.pdf>`__
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00098_854x480.mkv>`__
 * - 10:30
   - Jens Jørgen Mortensen
   - `Short history of GPAW and latest news about PAW setups
     <http://dcwww.camp.dtu.dk/~jensj/gpaw2013/a3.pdf>`__
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00099_854x480.mkv>`__
 * - 11:00
   - Ask Hjorth Larsen
   - LCAO in GPAW
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00100_854x480.mkv>`__
 * - 11:30
   - Samuli Hakala
   - Multi-GPU Accelerated Large Scale Electronic Structure Calculations
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00101_854x480.mkv>`__
 * - 12:00
   - 
   - 
   - *Lunch*
 * - 13:30
   - Henrik H. Kristoffersen
   - Simple methods for photo reaction modeling
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00103_854x480.mkv>`__
 * - 13:50
   - Lasse B. Vilhelmsen
   - Automated Two Step Structure Prediction within GPAW
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00104_854x480.mkv>`__
 * - 14:10
   - Bjørk Hammer
   - Challenges with the currently (correctly) implemented NEB-method. Should
     ASE revert to the original more robust NEB-formulation with springs?
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00105_854x480.mkv>`__
 * - 14:30
   - Jess Wellendorff
   - Exchange-correlation functionals with error estimation
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00106_854x480.mkv>`__
 * - 15:00
   - Hannes Jónsson
   - Orbital density dependent functionals
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00107_854x480.mkv>`__
 * - 15:30
   -
   - 
   - *Coffee*
 * - 16:00
   - Per Hyldgaard
   - Electron response in the Rutgers-Chalmers van der Waals density
     Functionals
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00108_854x480.mkv>`__
 * - 16:30
   - Thomas Olsen
   - Extending the random phase approximation with renormalized adiabatic
     exchange kernels
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00109_854x480.mkv>`__
 * - 17:00
   - Mikael Kuisma
   - Spin-Polarized GLLB-SC potential and efficient real time
     LCAO-TDDFT for large systems
   - `video
     <https://wiki.fysik.dtu.dk/gpaw-files/workshop13/00110_854x480.mkv>`__
 * - 18:00
   - 
   -
   - *Barbecue*


Wednesday (May 22)
------------------

.. list-table::
 :widths: 1 3 7

 * - 9:00
   - Jun Yan
   - Plasmon, exciton and RPA correlation energy: implementations and
     applications based on linear density response function
 * - 9:30
   - Arto Sakko, Tuomas Rossi
   - Combining TDDFT and classical electrodynamics simulations for plasmonics
 * - 10:00
   - Falco Hüser
   - The GW approximation in GPAW
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
   - Elvar Örn Jónsson
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
  Quick tour of :ref:`projects` --- what's the current status?
  
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

* Approximately 14:00: Presentations of today's work and wrap up

