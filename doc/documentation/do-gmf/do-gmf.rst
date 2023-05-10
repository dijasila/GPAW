.. _do-gmf:

==================================================================================
Excited State Calculations with Direct Optimization and Generalized Mode Following
==================================================================================
The direct optimization generalized mode following (DO-GMF) method can be used to perform
variational calculations of excited electronic states, where, contrary to
:ref:`linear response TDDFT <lrtddft>`, the orbitals are optimized for the excited state.

The main challenge of variational density functional calculations of excited states
is that excited states often correspond to saddle points on the surface describing
the variation of the energy as a function of the electronic degrees of freedom (the orbital
variations). :ref:`Standard self-consistent field (SCF) algorithms  <manual_eigensolver>`
typically perform well in ground state calculations, as the latter is a minimum of the energy,
but face convergence issues in excited state calculations. As an alternative,
:ref:`direct optimization (DO) <directopt>` approaches can be used, which have been found to
converge more robustly than the standard eigensolvers for excited states, especially in the
vicinity of electronic degeneracies. One option is to use quasi-Newton algorithms that
can converge to saddle points of arbitrary order in conjunction with the
:ref:`maximum overlap method (MOM) <mom>` to try to reduce the risk of converging to a
minimum or lower energy saddle point (variational collapse). This is the DO-MOM method illustrated
:ref:`here <directopt>`. However, DO-MOM can still be affected by variational collapse
in challenging cases. GPAW also implements an alternative DO approach using a
generalized mode following (GMF) method. DO-GMF targets a stationary solution with
a specific saddle point order and is more robust than both DO-MOM and the standard
SCF algorithms, while being inherently free from variational collapse.

--------------------------
Generalized mode following
--------------------------

~~~~~~~~~~~~~~
Implementation
~~~~~~~~~~~~~~

The implementation of the DO-GMF method is presented in [#dogmfgpaw1]_. For the moment,
the method can be used only in the LCAO mode.

GMF is a generalization of the minimum mode following method traditionally used to
optimize first-order saddle points on the potential energy surface for atomic rearrangements.
The method recasts the challenging saddle point search as a minimization by
inverting the projection of the gradient on the lowest eigenmode of the Hessian. It is
generalized to target an `n`-th-order saddle point on the electronic energy surface by
inverting the projections on the eigenmodes, `v_i`, of the electronic Hessian corresponding
to the `n` lowest eigenvalues, yielding the modified gradient

.. math::
    g^{\mathrm{\,mod}} = g - 2\sum_{i = 1}^{n}v_{i}v_{i}^{\mathrm{T}}g

if the energy surface is concave along all target eigenvectors, or

.. math::
    g^{\mathrm{\,mod}} = -\sum_{i = 1, \lambda_{i} \geq 0}^{n}v_{i}v_{i}^{\mathrm{T}}g

if any target eigenvalue, \lambda, is non-negative. Notice that in the latter case
only the target eigenvectors along which the energy surface is convex are followed
to increase stability of the method. The target eigenvalues and eigenvectors of the
electronic Hessian matrix are obtained by using a finite difference generalized Davidson
method [#gendavidson]_. This method can also be used for stability analysis of an
electronic solution (see :ref:`stabanalysisexample` below).

~~~~~~~~~~~~~~~~~
How to use DO-GMF
~~~~~~~~~~~~~~~~~

To provide initial guess orbitals for an excited state DO-GMF calculation, a ground
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

where a log file for the partial Hessian diagonalization can be specified and ``f`` contains
the occupation numbers of the excited state (see :ref:`ethyleneexample` and :ref:`tPPexample`).
Line search algorithms cannot be applied for saddle point searches, so a maximum step length is
used. Any of the search direction algorithms implemented in GPAW (see :ref:`directmin`) can be
used by appending ``_gmf`` to the ``name`` keyword of the ETDM search direction algorithms
(e.g. use ``l-bfgs-p_gmf`` to use the ``l-bfgs-p`` search direction with GMF).

A helper function can be used to create the list of excited state occupation numbers::

  from gpaw.directmin.tools import excite
  f = excite(calc, i, a, spin=(si, sa))

which will promote an electron from occupied orbital ``i`` in spin
channel ``si`` to unoccupied orbital ``a`` in spin channel ``sa``
(the index of HOMO and LUMO is 0). For example,
``excite(calc, -1, 2, spin=(0, 1))`` will remove an electron from
the HOMO-1 in spin channel 0 and add an electron to LUMO+2 in spin
channel 1.

.. _ethyleneexample:

-------------------------------------------
Example I: Doubly excited state of ethylene
-------------------------------------------
In this example, the lowest doubly excited state of ethylene is obtained with the DO-GMF method.
First, a ground state optimization is performed and then, the excited state is targeted as a
second-order saddle point on the electronic energy surface. The target saddle point order of 2
is automatically deduced correctly from the occupation numbers corresponding to an excitation from
the HOMO to the LUMO in both spin channels simultaneosly with respect to the ground state
orbitals.

.. literalinclude:: ethylene.py

It is recommended to deactivate updates of the reference orbitals by setting the
``update_ref_orbs_counter`` keyword to a large value (e.g. 1000). The unitary invariant
representation should be used (if the density functional is orbital density independent) because
the redundant rotations among the occupied orbitals introduce many degenerate eigenvectors of the
electronic Hessian with zero curvature, which can lead to convergence problems of the generalized
Davidson method. The keyword ``use_fixed_occupations`` is set to ``True`` to deactivate the use of
the maximum overlap method since variational collapse is impossible with the DO-GMF method.

.. _tPPexample:

------------------------------------------------------------
Example II: Charge transfer excited state of N-phenylpyrrole
------------------------------------------------------------
In this example, variational collapse of a charge transfer state of N-phenylpyrrole is
avoided by using the DO-GMF method and specifically targeting an excited state as a
saddle point on the electronic energy surface. The excited state is accessible by a single
excitation from the HOMO to the LUMO in one spin channel with respect to the
ground state orbitals. No spin purification is applied. After a ground state calculation,
the excited state is directly targeted as a sixth-order saddle point on the
electronic energy surface. While an unconstrained optimization of this excited state with
DO-MOM leads to variational collapse to a lower-energy saddle point with pronounced mixing
between the HOMO and LUMO and a low dipole moment of only -3.396 D, DO-GMF does not
suffer from variational collapse and converges to a higher-energy sixth-order saddle
point with a dipole moment of -10.227 D. This solution shows much less mixing between the
HOMO and LUMO involved in the excitation.

.. literalinclude:: tPP.py

The target saddle point order cannot be deduced from the occupation numbers alone in this case
since the target excited state solution is a sixth-order saddle point on the electronic energy
surface, but the occupation numbers with respect to the ground state orbitals suggest a target
saddle point order of 1. This discrepancy exists because the charge transfer excitation leads to a
large energetic rearrangement of the orbitals. One way to take this energetic rearrangement into
account is to perform a constrained optimization freezing the hole and excited electron and
minimizing all other electronic degrees of freedom. The occupation numbers after constrained
optimization suggest a target saddle point order of 7, and the full electronic Hessian has seven
negative eigenvalues, one of which is close to zero, pointing towards a target saddle point order
of 6. At this point, trial calculations targeting a sixth-order and a seventh-order saddle point,
respectively, can be started in parallel and the correct target saddle point order deduced from
the obtained solutions. The target saddle point order is set by using the ``sp_order`` keyword of
the ``partial_diagonalizer``.

.. _stabanalysisexample:

-----------------------------------------------------------------------------------
Example III: Stability analysis and breaking instability of ground state dihydrogen
-----------------------------------------------------------------------------------
In this example, the generalized Davidson method is used for stability analysis of the
ground state of the dihydrogen molecule. The molecule is stretched beyond the
Coulson-Fischer point, at which both a ground state solution with conserved symmetry and
two lower-energy degenerate ground state solutions with broken spin symmetry exist. First,
a spin-polarized direct minimization is performed starting from the GPAW initial guess
for the orbitals. Stability analysis confirms that the obtained solution is a first-order
saddle point on the electronic energy surface, meaning that the symmetry-conserving
solution is obtained. Second, the electronic structure is displaced along the eigenvector
of the electronic Hessian corresponding to its lowest, negative eigenvalue, and thereby,
the instability is broken. This displaced electronic structure is reoptimized yielding a
lower-energy solution with broken spin symmetry. Stability analysis is applied to this
solution to confirm that it is a minimum on the electronic energy surface.

.. literalinclude:: H2_instability.py

----------
References
----------

.. [#dogmfgpaw1] Y. L. A. Schmerwitz, G. Levi, H. Jónsson
               :doi:`Calculations of Excited Electronic States by Converging on Saddle Points Using Generalized Mode Following <10.48550/ARXIV.2302.05912>`, (2023).

.. [#gendavidson] M. Crouzeix, B. Philippe, M. Sadkane
               :doi:`The Davidson Method <10.1137/0915004>`, *SIAM J. Sci. Comput.*, (1994).
