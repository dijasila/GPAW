.. _lrtddft:

=====================
Linear response TDDFT
=====================

Ground state
============

The linear response TDDFT calculation needs a converged ground state
calculation with a set of unoccupied states.
Note, that the eigensolver 'rmm-diis' should not be used
for the calculation of unoccupied states, better use 'dav' or 'cg':

.. literalinclude:: Be_gs_8bands.py

Calculating the Omega Matrix
============================

The next step is to calculate the Omega Matrix from the ground state orbitals:

.. literalinclude:: Be_8bands_lrtddft.py

alternatively one can also restrict the number of transitions by their energy:

.. literalinclude:: Be_8bands_lrtddft_dE.py

Note, that parallelization over spin does not work here. As a workaround,
domain decomposition only (``parallel={'domain': world.size}``, 
see :ref:`manual_parsize_domain`) 
has to be used for spin polarised 
calculations in parallel.

Extracting the spectrum
=======================

The dipole spectrum can be evaluated from the Omega matrix and written to
a file:

.. literalinclude:: Be_spectrum.py


The spectrum may be also extracted and plotted in energy or
wavelength directly::

  import matplotlib.pyplot as plt
  from gpaw.lrtddft import LrTDDFT
  from gpaw.lrtddft.spectrum import get_folded_spectrum

  lr = LrTDDFT.read('lr.dat.gz')

  plt.subplot(121)
  # spectrum in energy
  x, y = get_folded_spectrum(lr, width=0.05)
  plt.plot(x, y[:, 0])
  plt.xlabel('energy [eV]')
  plt.ylabel('folded oscillator strength')

  plt.subplot(122)
  # spectrum in wavelengths
  x, y = get_folded_spectrum(lr, energyunit='nm', width=10)
  plt.plot(x, y[:, 0])
  plt.xlabel('wavelength [nm]')

  plt.show()


Testing convergence
===================

You can test the convergence of the Kohn-Sham transition basis size by restricting
the basis in the diagonalisation step, e.g.::

  from gpaw.lrtddft import LrTDDFT 

  lr = LrTDDFT.read('lr.dat.gz')
  lr.diagonalize(restrict={'energy_range':6.})

This can be automated by using the check_convergence function::

  from gpaw.lrtddft import LrTDDFT
  from gpaw.lrtddft.convergence import check_convergence

  lr = LrTDDFT.read('lr.dat.gz')
  check_convergence(lr,
                    'linear_response',
                    'my plot title',
                     dE=.2,
		     emax=6.)

which will create a directory 'linear_response'. In this directory there will be a
file 'conv.gpl' for gnuplot that compares the spectra varying the basis size.

Analysing the transitions
=========================

The single transitions (or a list of transitions) can be analysed as follows 
(output printed)::

  from gpaw.lrtddft import LrTDDFT

  lr = LrTDDFT.read('lr.dat.gz')
  lr.diagonalize()

  # analyse transition 1
  lr.analyse(1)

  # analyse transition 0-10
  lr.analyse(range(11))

Relaxation in the excited state
===============================

Despite that we do not have analytical gradients in linear response TDDFT,
we may use finite differences for relaxation in the excited state.
This example shows how to relax the sodium dimer
in the B excited state:

.. literalinclude:: Na2_relax_excited.py

The example runs on a single core. If started on 8 cores, it will split
``world`` into 4 independent parts (2 cores each) that can calculate
4 displacements in parallel at the same time.


Quick reference
===============

Parameters for LrTDDFT:

================  ==============  ===================  ========================================
keyword           type            default value        description
================  ==============  ===================  ========================================
``calculator``    ``GPAW``                             Calculator object of ground state
                                                       calculation
``nspins``        ``int``         1                    number of excited state spins, i.e.
                                                       singlet-triplet transitions are 
                                                       calculated with ``nspins=2``. Effective
                                                       only if ground state is spin-compensated
``xc``            ``string``      xc of calculator     Exchange-correlation for LrTDDFT, can 
                                                       differ from ground state value
``restrict``	  ``dict``        {}		       Restrictions ``eps``, ``istart``, ``jend``
                                                       and ``energy_range`` collected as dict.		       
``eps``           ``float``       0.001                Minimal occupation difference for a transition
``istart``        ``int``         0                    first occupied state to consider
``jend``          ``int``         number of bands      last unoccupied state to consider
``energy_range``  ``float``       None                 Energy range to consider in the involved
                                                       Kohn-Sham orbitals (replaces [istart,jend])
================  ==============  ===================  ========================================
