.. _do-gmf:

==================================================================================
Excited-State Calculations with Direct Optimization and Generalized Mode Following
==================================================================================

The main challenge of variational density functional calculations of excited electronic
states is that excited states often correspond to saddle points of the energy as a
function of the electronic degrees of freedom (the orbital variations), but standard
self-consistent field (SCF) algorithms are designed for minimizations, as the ground state
is always a minimum of the energy. Alternatively, direct optimization (DO), which
converges more robustly than SCF algorithms in the vicinity of electronic degenracies
commonly observed in excited states, can be used in GPAW. In DO, excited states are
directly found as stationary solutions on the electronic energy surface. To do so, any
optimization method can, in principle, be used, as long as it is able to find saddle
points. To avoid converging to a minimum (variational collapse), the
:ref:`maximum overlap method <mom>` can be used in conjunction with an algorithm that
can converge to saddle points of unspecified order, but variational collapse can still
occur in challenging cases. Alternatively, the generalized mode following (GMF) method
can be used together with the DO approach (DO-GMF) in GPAW. This method estimates the
saddle point order of the target excited state solution prior to the optimization and
then specifically targets a stationary solution with this saddle point order. This
approach converges more robustly than both DO with quasi-Newton optimization methods and
SCF algorithms, and it inherently eliminates variational collapse by construction.

--------------------------
Generalized mode following
--------------------------

~~~~~~~~~~~~~~
Implementation
~~~~~~~~~~~~~~

The implementation of the DO-GMF method is presented in [#dogmfgpaw1]_ (LCAO approach).

GMF is a generalization of the minimum mode following method traditionally used to
optimize 1\textsuperscript{st}-order saddle points on the nuclear potential energy
surface. The method recasts the challenging saddle point search as a minimization by
inverting the projection of the gradient on the lowest eigenmode of the Hessian. It is
generalized to target an ``n^{th}``-order saddle point by inverting the
projections on the lowest ``n`` eigenmodes, ``v``, of the Hessian yielding the modified gradient

.. math::
    g^{\mathrm{\,mod}} = g - 2\sum_{i = 1}^{n}v_{i}v_{i}^{\mathrm{T}}g

if the energy surface is concave along all target eigenvectors or

.. math::
    g^{\mathrm{\,mod}} = -\sum_{i = 1 \\ \lambda_{i} \geq 0}^{n}v_{i}v_{i}^{\mathrm{T}}g

if any target eigenvalue, \lambda, is non-negative. Notice that only the non-concave
target eigenvectors are followed, if any exist, to increase stability of the method.

The target eigenvalues and eigenvectors of the electronic Hessian matrix are obtained
by using a finite difference generalized Davidson method, whose implementation is
presented in [#gendavidson]_.

~~~~~~~~~~~~~~~~~
How to use DO-GMF
~~~~~~~~~~~~~~~~~

To provide initial guess orbitals for the excited state DO-GMF calculation, a ground
state calculation is typically performed first. Then, a DO-GMF calculation can be
requested as follows::

  from gpaw.directmin.etdm import ETDM

  calc.set(eigensolver=ETDM(
           partial_diagonalizer={'name': 'Davidson', 'logfile': None},
           linesearch_algo={'name': 'max-step'},
           searchdir_algo={'name': 'LBFGS-P_GMF'},
           need_init_orbs=False),
           occupations={'name': 'mom', 'numbers': f,
                        'use_fixed_occupations': True})

where the log file can be specified and ``f`` contains the occupation numbers of the
excited state (see examples below). Line search algorithms cannot be applied for saddle
point searches in this implementation. Any search direction algorithm can be used by
appending the name keyword with ``_GMF``.

A helper function can be used to create the list of excited-state occupation numbers::

  from gpaw.directmin.tools import excite
  f = excite(calc, i, a, spin=(si, sa))

which will promote an electron from occupied orbital ``i`` in spin
channel ``si`` to unoccupied orbital ``a`` in spin channel ``sa``
(the index of HOMO and LUMO is 0). For example,
``excite(calc, -1, 2, spin=(0, 1))`` will remove an electron from
the HOMO-1 in spin channel 0 and add an electron to LUMO+2 in spin
channel 1.

.. _h2oexample:

---------------------------------------------------
Example I: Excitation energy Rydberg state of water
---------------------------------------------------
In this example, the excitation energies of the singlet and
triplet states of water corresponding to excitation
from the HOMO-1 non-bonding (`n`) to the LUMO `3s` Rydberg
orbitals are calculated.
In order to calculate the energy of the open-shell singlet state,
first a calculation
of the mixed-spin state obtained for excitation within the same
spin channel is performed, and, then, the spin-purification
formula [#spinpur]_ is used: `E_s=2E_m-E_t`, where `E_m` and `E_t` are
the energies of the mixed-spin and triplet states, respectively.
The calculations use the Finite Difference mode to obtain an accurate
representation of the diffuse Rydberg orbital [#momgpaw1]_.

.. literalinclude:: mom_h2o.py

..  _coexample:

----------------------------------------------------------------
Example II: Geometry relaxation excited-state of carbon monoxide
----------------------------------------------------------------
In this example, the bond length of the carbon monoxide molecule
in the lowest singlet `\Pi(\sigma\rightarrow \pi^*)` excited state
is optimized using two types of calculations, each based on a
different approximation to the potential energy curve of an open-shell
excited singlet state.
The first is a spin-polarized calculation of the mixed-spin state
as defined in :ref:`h2oexample`. The second is a spin-paired calculation
where the occupation numbers of the open-shell orbitals are set
to 1 [#levi2018]_. Both calculations use LCAO basis and the
:ref:`direct optimization <directopt>` (DO) method.

In order to obtain the correct angular momentum
of the excited state, the electron is excited into a complex
`\pi^*_{+1}` or `\pi^*_{-1}` orbital, where +1 or −1 is the
eigenvalue of the z-component angular momentum operator. The
use of complex orbitals provides an excited-state density
with the uniaxial symmetry consistent with the symmetry of the
molecule [#momgpaw1]_.

.. literalinclude:: domom_co.py

The electronic configuration of the `\Pi(\sigma\rightarrow \pi^*)`
state includes two unequally occupied, degenerate `\pi^*` orbitals.
Because of this, convergence to this excited state is more
difficult when using SCF eigensolvers with density mixing
instead of DO, unless symmetry constraints on the density
are enforced during the calculation. Convergence of such
excited-state calculations with an SCF eigensolver can be
improved by using a Gaussian smearing of the holes and excited
electrons [#levi2018]_.
Gaussian smearing is implemented in MOM and can be used
by specifying a ``width`` in eV for the Gaussian smearing
function::

  mom.prepare_mom_calculation(..., width=0.01, ...)

For difficult cases, the ``width`` can be increased at regular
intervals by specifying a ``width_increment=...``.
*Note*, however, that too extended smearing can lead to
discontinuities in the potentials and forces close to
crossings between electronic states [#momgpaw2]_, so
this feature should be used with caution and only
at geometries far from state crossings.

----------
References
----------

.. [#momgpaw1] A. V. Ivanov, G. Levi, H. Jónsson
               :doi:`Method for Calculating Excited Electronic States Using Density Functionals and Direct Orbital Optimization with Real Space Grid or Plane-Wave Basis Set <10.1021/acs.jctc.1c00157>`,
               *J. Chem. Theory Comput.*, (2021).

.. [#momgpaw2] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Density Functional Calculations of Excited States via Direct Optimization <10.1021/acs.jctc.0c00597>`,
               *J. Chem. Theory Comput.*, **16** 6968–6982 (2020).

.. [#momgpaw3] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Calculations of Excited States Via Direct Optimization of Orbitals in DFT <10.1039/D0FD00064G>`,
               *Faraday Discuss.*, **224** 448-466 (2020).

.. [#imom]     G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill
               :doi:`Simple Models for Difficult Electronic Excitations <10.1021/acs.jctc.7b00994>`,
               *J. Chem. Theory Comput.*, **14** 1501-1509 (2018).

.. [#dongmom]  X. Dong, A. D. Mahler, E. M. Kempfer-Robertson, L. M. Thompson
               :doi:`Global Elucidation of Self-Consistent Field Solution Space Using Basin Hopping <10.1021/acs.jctc.0c00488>`,
               *J. Chem. Theory Comput.*, **16** 5635−5644 (2020).

.. [#spinpur]  T. Ziegler, A. Rauk, E. J. Baerends
               :doi:`On the calculation of multiplet energies by the hartree-fock-slater method <10.1007/BF00551551>`
               *Theoret. Chim. Acta*, **43** 261–271 (1977).

.. [#levi2018] G. Levi, M. Pápai, N. E. Henriksen, A. O. Dohn, K. B. Møller
               :doi:`Solution structure and ultrafast vibrational relaxation of the PtPOP complex revealed by ∆SCF-QM/MM Direct Dynamics simulations <10.1021/acs.jpcc.8b00301>`,
               *J. Phys. Chem. C*, **122** 7100-7119 (2018).