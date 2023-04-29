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

-----------------------------------------------------------
Example I: Charge transfer excited state of N-phenylpyrrole
-----------------------------------------------------------

In this example, variational collapse of a charge transfer state of N-phenylpyrrole is
avoided by using the DO-GMF method and specifically targeting an excited state as a
saddle point on the electronic energy surface. The excited state is accessible by a single
excitation from the HOMO to the LUMO in one spin channel with respect to the
ground state orbitals. No spin purification is applied. After a ground state calculation,
the excited state is directly targeted as a ``6^{th}``-order saddle point on the
electronic energy surface. While an unconstrained optimization of this excited state with
DO-MOM leads to variational collapse to a lower-energy saddle point with pronounced mixing
between the HOMO and LUMO and a low dipole moment of only -3.396 D, DO-GMF does not
suffer from variational collapse and converges to a higher-energy ``6^{th}``-order saddle
point with a dipole moment of -10.227 D. This solution shows much less mixing between the
HOMO and LUMO involved in the excitation.

.. literalinclude:: tPP.py

The target saddle point order is set by using the ``sp_order`` keyword of the
``partial_diagonalizer``. It is recommended to deactivate updates of the reference by
setting the ``update_ref_orbs_counter`` keyword to a large value (e.g. 1000). The
unitary invariant representation should be used (if the density functional is orbital
density independent) because the redundant rotations among the occupied orbitals
introduce many degenerate eigenvectors of the electronic Hessian with zero curvature,
which can lead to convergence problems of the generalized Davidson method. The keyword
``use_fixed_occupations`` is set to True to deactivate the use of the maximum overlap
method since variational collapse is impossible with the DO-GMF method.

----------
References
----------

.. [#dogmfgpaw1] Y. L. A. Schmerwitz, G. Levi, H. Jónsson
               :doi:`Calculations of Excited Electronic States by Converging on Saddle Points Using Generalized Mode Following <10.48550/ARXIV.2302.05912>`, (2023).

.. [#gendavidson] M. Crouzeix, B. Philippe, M. Sadkane
               :doi:`The Davidson Method <10.1137/0915004>`, *SIAM J. Sci. Comput.*, (1994).
