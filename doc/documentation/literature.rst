.. _literature:

----------
Literature
----------


Links to guides and manual pages
--------------------------------

* The GPAW calculator :ref:`manual`

* The :ref:`devel` pages

* The :ref:`guide for developers <developersguide>`

* The code :ref:`overview`

* The :ref:`features_and_algorithms` used in the code


.. _special_topics:

Specialized information
-----------------------

Here is a list of specific advanced topics and functionalities of the
GPAW calculator:

.. toctree::
   :maxdepth: 2
   
   toc-special


.. _literature_reports_presentations_and_theses:

Reports, presentations, and theses using gpaw
---------------------------------------------

* A short note on the basics of PAW: `paw note`_

* A master thesis on the inclusion of non-local exact exchange in the
  PAW formalism, and the implementation in gpaw: `exact exchange`_

* A master thesis on the inclusion of a localized basis in the PAW
  formalism, plus implementation and test results in GPAW: `lcao`_

* A master thesis on the inclusion of localized basis sets in the PAW
  formalism, focusing on basis set generation and force calculations:
  `localized basis sets`_

* A course report on a project involving the optimization of the
  setups (equivalent of pseudopotentials) in gpaw: `setup
  optimization`_

* Slides from a talk about PAW: `introduction to PAW slides`_

* Slides from a talk about GPAW development: `gpaw for developers`_

* Slides from a mini symposium during early development stage: `early gpaw`_

.. _paw note: ../paw_note.pdf
.. _exact exchange: ../_static/rostgaard_master.pdf
.. _lcao: ../_static/marco_master.pdf
.. _localized basis sets: ../_static/askhl_master.pdf
.. _setup optimization: ../_static/askhl_10302_report.pdf
.. _introduction to PAW slides: ../_static/mortensen_paw.pdf
.. _gpaw for developers: ../_static/mortensen_gpaw-dev.pdf
.. _early gpaw: ../_static/mortensen_mini2003talk.pdf


.. _paw_papers:

Articles on the PAW formalism
-----------------------------

The original article introducing the PAW formalism:
   | P. E. Blöchl
   | `Projector augmented-wave method`__
   | Physical Review B, Vol. **50**, 17953, 1994

   __ http://dx.doi.org/10.1103/PhysRevB.50.17953

A different formulation of PAW by Kresse and Joubert designed to make the transistion from USPP to PAW easy.
  | G. Kresse and D. Joubert
  | `From ultrasoft pseudopotentials to the projector augmented-wave method`__
  | Physical Review B, Vol. **59**, 1758, 1999

  __ http://dx.doi.org/10.1103/PhysRevB.59.1758

A second, more pedagogical, article on PAW by Blöchl and co-workers.
  | P. E. Blöchl, C. J. Först, and J. Schimpl
  | `Projector Augmented Wave Method: ab-initio molecular dynamics with full wave functions`__
  | Bulletin of Materials Science, Vol. **26**, 33, 2003

  __ http://www.ias.ac.in/matersci/



.. _gpaw_publications:

Publications using the gpaw code
--------------------------------

.. If the first author is A. Einstein, then remember to use
   \A. Einstein so that we don't start an enumerated list (A, B, C,
   ...).

1) The first article introducing the gpaw project:
   
   \J. J. Mortensen, L. B. Hansen, and K. W. Jacobsen

   `Real-space grid implementation of the projector augmented wave method`__
   
   Physical Review B, Vol. **71**, 035109 (2005)

   __ http://dx.doi.org/10.1103/PhysRevB.71.035109

   .. 21 January 2005

#) A description of a statistical approach to the exchange-correlation
   energy in DFT:

   \J. J. Mortensen, K. Kaasbjerg, S. L. Frederiksen, J. K. Nørskov,
   J. P. Sethna, and K. W. Jacobsen

   `Bayesian Error Estimation in Density-Functional Theory`__
  
   Physical Review Letters, Vol. **95**, 216401 (2005)

   __ http://dx.doi.org/10.1103/PhysRevLett.95.216401

   .. 15 November 2005


#) First article related to ligand protected gold clusters:

   \J. Akola, M. Walter, R. L. Whetten, H. Häkkinen, and H. Grönbeck

   `On the structure of Thiolate-Protected Au25`__
  
   Journal of the American Chemical Society, Vol. **130**, 3756-3757 (2008)

   __ http://dx.doi.org/10.1021/ja800594p

   .. 6 March 2008


#) The article describing the time-dependent DFT implementations in
   gpaw:

   \M. Walter, H. Häkkinen, L. Lehtovaara, M. Puska, J. Enkovaara,
   C. Rostgaard, and J. J. Mortensen

   `Time-dependent density-functional theory in the projector
   augmented-wave method`__

   Journal of Chemical Physics, Vol. **128**, 244101 (2008)

   __ http://dx.doi.org/10.1063/1.2943138

   .. 23 June 2008


#) Second article related to ligand protected gold clusters:
   
   \M. Walter, J. Akola, O. Lopez-Acevedo, P.D. Jadzinsky, G. Calero,
   C.J. Ackerson, R.L. Whetten, H. Grönbeck, and H. Häkkinen

   `A unified view of ligand-protected gold clusters as superatom complexes`__
   
   Proceedings of the National Academy of Sciences, Vol. **105**,
   9157-9162 (2008) 
 
   __ http://www.pnas.org/cgi/content/abstract/0801001105v1

   .. 1 July 2008

#) Description of the delta SCF method implemented in GPAW for
   determination of excited-state energy surfaces:

   Jeppe Gavnholt, Thomas Olsen, Mads Engelund, and Jakob Schiotz

   `Delta self-consistent field method to obtain potential energy
   surfaces of excited molecules on surfaces`__

   Physical Review B, Vol. **78**, 075441 (2008)

   __ http://dx.doi.org/10.1103/PhysRevB.78.075441

   .. 27 August 2008


#) GPAW applied to the study of graphene edges:

   Pekka Koskinen, Sami Malola, and Hannu Häkkinen

   `Self-passivating edge reconstructions of graphene`__

   Physical Review Letters, Vol. **101**, 115502 (2008)

   __ http://dx.doi.org/10.1103/PhysRevLett.101.115502

   .. 10 September 2008


#) Application of delta SCF method, for making predictions on
   hot-electron assisted chemistry:

   Thomas Olsen, Jeppe Gavnholt, and Jakob Schiotz
  
   `Hot-electron-mediated desorption rates calculated from
   excited-state potential energy surfaces`__

   Physical Review B, Vol. **79**, 035403 (2009)

   __ http://dx.doi.org/10.1103/PhysRevB.79.035403

   .. 6 January 2009 


#) A DFT study of a large thiolate protected gold cluster with 144 Au
   atoms and 60 thiolates:

   Olga Lopez-Acevedo, Jaakko Akola, Robert L. Whetten, Henrik
   Grönbeck, and Hannu Häkkinen

   `Structure and Bonding in the Ubiquitous Icosahedral Metallic Gold
   Cluster Au144(SR)60`__

   The Journal of Physical Chemistry C, in press (2009)

   __ http://dx.doi.org/10.1021/jp8115098

   .. 16 January 2009


#) A study of gold cluster stability on a rutile TiO\ :sub:`2`
   surface, and CO adsorbed on such clusters:

   Georg K. H. Madsen and Bjørk Hammer

   `Effect of subsurface Ti-interstitials on the bonding of small gold
   clusters on rutile TiO_2 (110)`__

   Journal of Chemical Physics, **130**, 044704 (2009)

   __ http://dx.doi.org/10.1063/1.3055419 

   .. 26 January 2009

#) Interpretation of STM images with DFT calculations:

   \F. Yin, J. Akola, P. Koskinen, M. Manninen, and R. E. Palmer

   `Bright Beaches of Nanoscale Potassium Islands on Graphite in STM
   Imaging`__
  
   Physical Review Letters, Vol. **102**, 106102 (2009)

   __ http://dx.doi.org/10.1103/PhysRevLett.102.106102

#) X. Lin, N. Nilius, H.-J. Freund, M. Walter, P. Frondelius,
   K. Honkala, and H. Häkkinen

   `Quantum well states in two-dimensional gold clusters on MgO thin films`__  

   Physical Review Letters, Vol. **102**, 206801 (2009)

   __ http://dx.doi.org/10.1103/PhysRevLett.102.206801

   .. 5 June 2009

#) The effect of frustrated rotations in HEFatS is calculated using
   the delta SCF method

   Thomas Olsen

   `Inelastic scattering in a local polaron model with quadratic
   coupling to bosons`__

   Physical Review B, Vol. **79**, 235414 (2009)

   __ http://dx.doi.org/10.1103/PhysRevB.79.235414

   .. 12 June 2009 

#) Olga Lopez-Acevedo, Jyri Rintala, Suvi Virtanen, Cristina Femoni,
   Cristina Tiozzo, Henrik Grönbeck, Mika Pettersson and Hannu Häkkinen

   `Characterization of Iron-Carbonyl-Protected Gold Clusters`__

   Journal of the American Chemical Society, (2009)

   __ http://dx.doi.org/10.1021/ja905182g

   .. 14 August 2009

#) Jiří Klimeš, David R. Bowler, and Angelos Michaelides

   `Chemical accuracy for the van der Waals density functional`__

   __ http://arxiv.org/abs/0910.0438

   .. 2 October 2009
