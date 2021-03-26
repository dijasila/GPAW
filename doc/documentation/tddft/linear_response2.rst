.. _lrtddft2:

=========================================
Linear response TDDFT 2 - indexed version
=========================================

Ground state
============

The linear response TDDFT calculation needs a converged ground state
calculation with a set of unoccupied states. It is safer to use 'dav' or 'cg'
eigensolver instead of the default 'rmm-diis' eigensolver to converge
unoccupied states. However, 'dav' and 'cg' are often too expensive for large
systems. In this case, you should use 'rmm-diis' with tens or hundreds of
extra states in addition to the unoccupied states you wish to converge.

We demonstrate the code usage for
:download:`(R)-methyloxirane molecule <lrtddft2/r-methyloxirane.xyz>`.

First, we calculate the ground state:

.. literalinclude:: lrtddft2/gs.py
    :start-after: Start

Text output: :download:`gs.out <lrtddft2/gs.out>`.

Then, we converge unoccupied states:

.. literalinclude:: lrtddft2/unocc.py
    :start-after: Start

Text output: :download:`unocc.out <lrtddft2/unocc.out>`.

Let's have a look on the states in the output file:

.. literalinclude:: lrtddft2/unocc.out
    :start-at: Band  Eigenvalues  Occupancy
    :end-before: Fermi level

We see that the Kohn-Sham eigenvalue difference between HOMO and
the highest converged unoccupied state is about 8.2 eV.
Thus, all Kohn-Sham single-particle transitions up to this energy
difference can be calculated from these unoccupied states.
If more is needed, then more unoccupied states would need to be converged.


Calculating response matrix and spectrum
========================================

The next step is to calculate the response matrix with ``LrTDDFT2``.

A very important convergence parameter is the number of Kohn-Sham
single-particle transitions used to calculate the response matrix.
This can be set through state indices (see bottom of the page),
or as demonstrated here, through an energy cutoff parameter ``max_energy_diff``.
This parameter defines the maximum energy difference of
the Kohn-Sham transitions included in the calculation.

Note! If the used gpw file does not contain enough unoccupied states so
that **all** single-particle transitions defined by ``max_energy_diff``
can be included, then the calculation does not usually make sense.
Thus, check carefully the states in the unoccupied states calculation
(see the example above).

Note also! The ``max_energy_diff`` parameter does **not** mean that
the TDDFT excitations would be converged up to this energy.
Typically, the ``max_energy_diff`` needs to be much larger than the smallest
excitation energy of interest to obtained well converged results.
Checking the convergence with respect to the number of states
included in the calculation is crucial.

In this script, we set ``max_energy_diff`` to 7 eV.
We also show how to parallelize calculation
over Kohn-Sham electron-hole (eh) pairs with ``LrCommunicators``
(8 tasks are used for each ``GPAW`` calculator):

.. literalinclude:: lrtddft2/lr2.py
    :start-after: Start

Text output (:download:`lr2_with_07.00eV.out <lrtddft2/lr2_with_07.00eV.out>`)
shows the number of Kohn-Sham transitions within the set 7 eV limit:

.. literalinclude:: lrtddft2/lr2_with_07.00eV.out

The TDDFT excitations are
(:download:`transitions_with_07.00eV.dat <lrtddft2/transitions_with_07.00eV.dat>`):

.. literalinclude:: lrtddft2/transitions_with_07.00eV.dat


Note: Unfortunately, spin is not implemented yet. For now, use 'lrtddft'.


Restarting and recalculating
============================

The calculation can be restarted with the same scipt.
As an example, here we increase the energy cutoff to 8 eV.
The matrix elements calculated earlier up to 7 eV are reused,
and only the missing matrix elements are calculated:

.. literalinclude:: lrtddft2/lr2_restart.py
    :start-after: Start

Text output (:download:`lr2_with_08.00eV.out <lrtddft2/lr2_with_08.00eV.out>`)
shows the number of Kohn-Sham transitions within the set limit:

.. literalinclude:: lrtddft2/lr2_with_08.00eV.out

The TDDFT excitations are
(:download:`transitions_with_08.00eV.dat <lrtddft2/transitions_with_08.00eV.dat>`):

.. literalinclude:: lrtddft2/transitions_with_08.00eV.dat

It's important to note that also the first excitations change in comparison
to the earlier calculation with the 7 eV cutoff energy.
As stated earlier, the results must be converged with respect to
the cutoff energy, and typically the cutoff energy needs to be
much larger than the smallest excitation energy of interest.

Here we plot the photoabsorption and rotatory strength spectra from
the data files (
:download:`spectrum_with_07.00eV.dat <lrtddft2/spectrum_with_07.00eV.dat>` and
:download:`spectrum_with_08.00eV.dat <lrtddft2/spectrum_with_08.00eV.dat>`):

.. image:: lrtddft2/abs_spec.png

.. image:: lrtddft2/rot_spec.png

We note that the cutoff energy (and the number of unoccupied bands)
should be increased more to converge the spectra properly.


Analyzing spectrum
==================

Once the response matrix has been calculated, the same input script can be used
for calculating the spectrum and analyzing the transitions.
But, as all the expensive calculation is done already, it's sufficient
to run the script in serial.
Here is an example script for analysis without parallelization settings:

.. literalinclude:: lrtddft2/lr2_analyze.py
    :start-after: Start

The script produces the same spectra and transitions as above.
In addition, it demonstrates how to analyze transition contributions.
An example for the first TDDFT excitation (``index=0``), corresponding
to the first excitation in
:download:`transitions_with_08.00eV.dat <lrtddft2/transitions_with_08.00eV.dat>`
(:download:`tc_000_with_08.00eV.txt <lrtddft2/tc_000_with_08.00eV.txt>`):

.. literalinclude:: lrtddft2/tc_000_with_08.00eV.txt

Note that while this excitation is at about 5.9 eV, it has non-negligible
contributions from Kohn-Sham single-particle transitions above this energy.
This is generally the case, and it is basically the reason why
the ``max_energy_diff`` parameter has to be typically much higher
than the highest excitation energies of interest.


Quick reference
===============

Parameters for LrCommunicators:

================  =================  ===================  ========================================
keyword           type               default value        description
================  =================  ===================  ========================================
``world``          ``Communicator``  None                 parent communicator 
                                                          (usually gpaw.mpi.world)
``dd_size``        `int``            None                 Number of domains for domain 
                                                          decomposition
``eh_size``        `int``            None                 Number of groups for parallelization
                                                          over e-h -pairs
================  =================  ===================  ========================================

Note: world.size = dd_size x eh_size


Parameters for LrTDDFT2:

====================  ==================  ===================  ========================================
keyword               type                default value        description
====================  ==================  ===================  ========================================
``basefilename``      ``string``                               Prefix for all files created by LRTDDFT2
                                                               (e.g. ``dir/lr``)
``gs_calc``           ``GPAW``                                 Ground-state calculator, which has been 
                                                               loaded from a file with 
                                                               communicator=lr_communicators
                                                               calculation
``fxc``               ``string``          None                 Exchange-correlation kernel
``min_occ``           ``int``             None                 Index of the first occupied state 
                                                               (inclusive)
``max_occ``           ``int``             None                 Index of the last occupied state 
                                                               (inclusive)
``min_unocc``         ``int``             None                 Index of the first unoccupied state 
                                                               (inclusive)
``max_unocc``         ``int``             None                 Index of the last unoccupied state 
                                                               (inclusive)
``max_energy_diff``   ``float``           None                 Maximum Kohn-Sham eigenenergy difference
``recalculate``       ``string``          None                 What should be recalculated. Usually 
                                                               nothing.
``lr_communicators``  ``LrCommuncators``  None                 
``txt``               ``string``          None                 Output
====================  ==================  ===================  ========================================
