.. _lcaotddft:

===================================================
Time-propagation TDDFT with LCAO: Theory and usage
===================================================

This page documents the use of time-propagation TDDFT in :ref:`LCAO
mode <lcao>`. The implementation is described in Ref. [#Kuisma2015]_.

Real-time propagation of LCAO functions
=======================================

In the real-time LCAO-TDDFT approach, the time-dependent wave functions are
represented using the
localized basis functions `\tilde{\phi}_{\mu}(\mathbf r)` as

.. math::

  \tilde{\psi}_n(\mathbf{r},t) = \sum_{\mu} \tilde{\phi}_{\mu}(\mathbf{r}-\mathbf{R}^\mu) c_{\mu n}(t) .

The time-dependent Kohn--Sham equation in the PAW formalism can be written as

.. math::

  \left[ \widehat T^\dagger \left( -i \frac{{\rm d}}{{\rm d}t} + \hat H_{\rm KS}(t) \right) \widehat T \right]  \tilde{\Psi(\mathbf{r},t)} = 0.

From this, the following matrix equation can be derived for the
LCAO wave function coefficients:

.. math::
  {\rm i}\mathbf{S} \frac{{\rm d}\mathbf{C}(t)}{{\rm d}t} = \mathbf{H}(t) \mathbf{C}(t).

In the current implementation, `\mathbf C`, `\mathbf S` and
`\mathbf H` are dense matrices which are distributed using
ScaLAPACK/BLACS.  Currently, a semi-implicit Crank--Nicolson method (SICN) is
used to propagate the wave functions. For wave functions at time `t`, one
propagates the system forward using `\mathbf H(t)` and solving the
linear equation

.. math::

  \left( \mathbf{S} + {\rm i} \mathbf{H}(t) {\rm d}t / 2 \right) \mathbf{C}'(t+{\rm d}t) = \left( \mathbf{S} - {\rm i} \mathbf{H}(t) {\rm d}t / 2 \right) \mathbf{C}(t).

Using the predicted wave functions `C'(t+\mathrm dt)`, the
Hamiltonian `H'(t+\mathrm dt)` is calculated and the Hamiltonian at
middle of the time step is estimated as

.. math::

   \mathbf{H}(t+{\rm d}t/2) = (\mathbf{H}(t) + \mathbf{H}'(t+{\rm d}t)) / 2

With the improved Hamiltonian, the wave functions are again propagated
from `t` to `t+\mathrm dt` by solving

.. math::

  \left( \mathbf{S} + {\rm i} \mathbf{H}(t+{\rm d}t/2) {\rm d}t / 2 \right) \mathbf{C}(t+{\rm d}t) = \left( \mathbf{S} - {\rm i} \mathbf{H}(t+{\rm d}t/2) {\rm d}t / 2 \right) \mathbf{C}(t).

This procedure is repeated using 500--2000 time steps of 5--40 as to
calculate the time evolution of the electrons.

.. _example:

Example usage
=============

First do a standard ground-state calculation with the ``GPAW`` calculator:

.. literalinclude:: lcaotddft.py
   :lines: 3-21

Some important points are:

* The grid spacing is only used to calculate the Hamiltonian matrix and
  therefore a coarser grid than usual can be used.
* Completely unoccupied bands should be left out of the calculation,
  since they are not needed.
* The density convergence criterion should be a few orders of magnitude
  more accurate than in usual ground-state calculations.
* The convergence tolerance of the Poisson solver should be at least ``1e-16``,
  but ``1e-20`` does not hurt (note that this is the **quadratic** error).
* One should use multipole-corrected Poisson solvers or
  other advanced Poisson solvers in any TDDFT run
  in order to guarantee the convergence of the potential with respect to
  the vacuum size.
  See the documentation on :ref:`advancedpoisson`.

Next the calculation proceeds as in the grid mode with ``TDDFT`` object.
We kick the system in the z direction and propagate 3000 steps of 10 as:

.. literalinclude:: lcaotddft.py
   :lines: 23-35

After the time propagation, the spectrum can be calculated:

.. literalinclude:: lcaotddft.py
   :lines: 37-39

The previous example as a complete script can be downloaded here:
:download:`lcaotddft.py`.


General notes about basis sets
==============================

In time-propagation LCAO-TDDFT, it is much more important to think
about the basis sets compared to ground-state LCAO calculations.  It
is required that the basis set can represent both the occupied
(holes) and relevant unoccupied states (electrons) adequately.  Custom
basis sets for the time propagation should be generated according to
one's need, and then benchmarked.

**Irrespective of the basis sets you choose, ALWAYS, ALWAYS, benchmark LCAO
results with respect to the FD time-propagation code** on the largest system
possible. For example, one can create a prototype system which consists of
similar atomic species with similar roles as in the parent system, but small
enough to calculate with grid propagation mode.
See Figs. 4 and 5 of Ref. [#Kuisma2015]_ for example benchmarks.

After these remarks, we describe two packages of basis sets that can be used
as a starting point for choosing a suitable basis set for your needs.
Namely, :ref:`pvalence basis sets` and :ref:`coopt basis sets`.


.. _pvalence basis sets:

p-valence basis sets
--------------------

The so-called p-valence basis sets are constructed for transition metals by
replacing the p-type polarization function of the default basis sets with a
bound unoccupied p-type orbital and its split-valence complement. Such basis
sets correspond to the ones used in Ref. [#Kuisma2015]_. These basis sets
significantly improve density of states of unoccupied states.

The p-valence basis sets can be easily obtained for the appropriate elements
with the :command:`gpaw install-data` tool using the following options::

    $ gpaw install-data {<directory>} --basis --version=pvalence

See :ref:`installation of paw datasets` for more information on basis set
installation. It is again reminded that these basis sets are not thoroughly
tested and **it is essential to benchmark the performance of the basis sets
for your application**.


.. _coopt basis sets:

Completeness-optimized basis sets
---------------------------------

A systematic approach for improving the basis sets can be obtained with the
so-called completeness-optimization approach. This approach is used in Ref.
[#Rossi2015]_ to generate basis set series for TDDFT calculations of copper,
silver, and gold clusters.

For further details of the basis sets, as well as their construction and
performance, see [#Rossi2015]_. For convenience, these basis sets can be easily
obtained with::

    $ gpaw install-data {<directory>} --basis --version=coopt

See :ref:`installation of paw datasets` for basis set installation. Finally,
it is again emphasized that when using the basis sets, **it is essential to
benchmark their suitability for your application**.


Parallelization
===============

LCAO-TDDFT is parallelized using ScaLAPACK. It runs without ScaLAPACK,
but in this case only a single core is used for linear alrebra.

* Use ``parallel={'sl_default':(N, M, 64)}``;  See :ref:`manual_parallel`.
* It is necessary that N*M equals the total number of cores used
  by the calculator, and ``max(N,M)*64 < nbands``, where ``64`` is the used
  block size. The block size can be changed to, e.g., 16 if necessary.
* Apart from parallelization of linear algrebra, normal domain and
  band parallelizations can be used. As in ground-state LCAO calculations,
  use band parallelization to reduce memory consumption.


.. TODO

    Timing
    ======
    
    Add ``ParallelTimer`` example


Advanced analysis tools
=======================

In :ref:`example` it was demonstrated how to calculate photoabsorption
spectrum from the time-dependent dipole moment data collected with
``DipoleMomentWriter`` observer.
The code is not limited to this analysis but any (also user-written)
analysis tools can be embedded in the general time-propagation framework.

Here we describe some analysis tools available in GPAW.
We use a finite sodium atom chain as an example system.
First, let's do the ground-state calculation:

.. literalinclude:: lcaotddft_Na8/gs.py

Note the recommended use of :ref:`advancedpoisson` and
:ref:`pvalence basis sets`.


Recording the wave functions and replaying the time propagation
---------------------------------------------------------------

We can record the time-dependent wave functions during the propagation
with ``WaveFunctionWriter()`` observer:

.. literalinclude:: lcaotddft_Na8/td.py

.. tip::

   The time propagation can be continued in the same manner
   from the restart file:

   .. literalinclude:: lcaotddft_Na8/tdc.py

The created ``wfw.ulm`` file contains the time-dependent wave functions
`C_{\mu n}(t)` that define the state of the system at each time.
We can use the file to replay the time propagation:

.. literalinclude:: lcaotddft_Na8/td_replay.py

The ``update`` keyword in ``propagator`` has following options:

==============  ===============================
``update``      variables updated during replay
==============  ===============================
``'all'``       Hamiltonian and density
``'density'``   density
``'none'``      nothing
==============  ===============================


Kohn--Sham decomposition of the density matrix
----------------------------------------------

Kohn--Sham decomposition is an illustrative way of analyzing electronic
excitations in Kohn--Sham electron-hole basis.
See Ref. [#Rossi2017]_ for the description of the GPAW implementation
and demonstration.

Here we demonstrate how to construct the photoabsorption decomposition
at a specific frequency in Kohn--Sham electon-hole basis.

First, let's calculate and :download:`plot <lcaotddft_Na8/spec_plot.py>`
the spectrum:

.. literalinclude:: lcaotddft_Na8/spectrum.py

.. image:: lcaotddft_Na8/spec.png
   :scale: 70%

The two main resonances are analyzed in the following.

Frequency-space density matrix
""""""""""""""""""""""""""""""

We generate the density matrix for the frequencies of interest:

.. literalinclude:: lcaotddft_Na8/td_fdm_replay.py

.. tip::

   Instead of replaying the propagation, one can do the same analysis
   on-the-fly by attaching the analysis tools to
   the usual time-propagation calculation.

Transform the density matrix to Kohn--Sham electron-hole basis
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

First, we construct the Kohn--Sham electron-hole basis.
For that we need to calculate unoccupied Kohn--Sham states,
which is conveniently done by restarting from the earlier
ground-state file:

.. literalinclude:: lcaotddft_Na8/ksd_init.py

Next, we can use the created objects to transform the LCAO density matrix
to the Kohn--Sham electron-hole basis and visualize the photoabsorption
decomposition as a transition contribution map (TCM):

.. literalinclude:: lcaotddft_Na8/tcm_plot.py

Note that the sum over the decomposition (the printed total absorption)
equals to the value of the photoabsorption spectrum at the particular
frequency in question.

We obtain the following transition contributions for the resonances
(presented both as tables and TCMs):

.. literalinclude:: lcaotddft_Na8/table_1.12.txt
   :language: none

.. image:: lcaotddft_Na8/tcm_1.12.png
   :scale: 70%

.. literalinclude:: lcaotddft_Na8/table_2.48.txt
   :language: none

.. image:: lcaotddft_Na8/tcm_2.48.png
   :scale: 70%


Induced density
---------------

The density matrix gives access to any other quantities.
For instance, the induced density can be conveniently obtained
from the density matrix:

.. literalinclude:: lcaotddft_Na8/fdm_ind.py

The resulting cube files can be visualized, for example, with
:download:`this script <lcaotddft_Na8/ind_plot.py>` producing
the figures:

.. image:: lcaotddft_Na8/ind_1.12.png
   :scale: 70%

.. image:: lcaotddft_Na8/ind_2.48.png
   :scale: 70%


Advanced tutorials
==================

Plasmon resonance of silver cluster
-----------------------------------

One should think about what type of transitions of interest are present,
and make sure that the basis set can represent such Kohn-Sham electron and
hole wave functions. The first transitions in silver clusters will be
`5s \rightarrow 5p` like. We require 5p orbitals in the basis set, and thus,
we must generate a custom basis set.

Here is how to generate a double-zeta basis set with 5p orbital in valence
for silver for GLLB-SC potential. Note that the extra 5p valence state
effectively improves on the ordinary polarization function, so this basis set
is **better** than the default double-zeta polarized one.
We will use the 11-electron Ag setup, since the semi-core p states included
in the default setup are not relevant here.

.. literalinclude:: lcaotddft_basis.py

We calculate the icosahedral Ag55 cluster: :download:`ag55.xyz`

This code uses ScaLAPACK parallelization with 48 cores.

.. literalinclude:: lcaotddft_ag55.py

Code runs for approximately two hours wall-clock time.
The resulting spectrum shows already emerging plasmonic excitation
around 4 eV.
For more details, see [#Kuisma2015]_.

.. image:: fig1.png

.. TODO

   Large organic molecule
   ----------------------
   
   On large organic molecules, on large conjugated systems, there will `\pi \rightarrow \pi^*`,
   `\sigma \rightarrow \sigma^*`. These states consist of only
   the valence orbitals of carbon, and they are likely by quite similar few eV's
   below and above the fermi lavel. These are thus a reason to believe that these
   states are well described with hydrogen 1s and carbon 2s and 2p valence orbitals
   around the fermi level.
   
   Here, we will calculate a small and a large organic molecule with lcao-tddft.
   

References
==========

.. [#Kuisma2015]
   M. Kuisma, A. Sakko, T. P. Rossi, A. H. Larsen, J. Enkovaara,
   L. Lehtovaara, and T. T. Rantala,
   Localized surface plasmon resonance in silver nanoparticles:
   Atomistic first-principles time-dependent density functional theory
   calculations,
   *Phys. Rev. B* **69**, 245419 (2004).
   `doi:10.1103/PhysRevB.91.115431 <https://doi.org/10.1103/PhysRevB.91.115431>`_

.. [#Rossi2015]
   T. P. Rossi, S. Lehtola, A. Sakko, M. J. Puska, and R. M. Nieminen,
   Nanoplasmonics simulations at the basis set limit
   through completeness-optimized, local numerical basis sets,
   *J. Chem. Phys.* **142**, 094114 (2015).
   `doi:10.1063/1.4913739 <https://doi.org/10.1063/1.4913739>`_

.. [#Rossi2017]
   T. P. Rossi, M. Kuisma, M. J. Puska, R. M. Nieminen, and P. Erhart,
   Kohn--Sham Decomposition in Real-Time Time-Dependent
   Density-Functional Theory:
   An Efficient Tool for Analyzing Plasmonic Excitations,
   *J. Chem. Theory Comput.* **13**, 4779 (2017).
   `doi:10.1021/acs.jctc.7b00589 <https://doi.org/10.1021/acs.jctc.7b00589>`_
