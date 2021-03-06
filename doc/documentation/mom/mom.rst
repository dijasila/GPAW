.. _mom:

======================
Maximum Overlap Method
======================

The maximum overlap method (MOM) can be used to perform
variational calculations of excited states. It is an
alternative to the linear expansion :ref:`dscf` for obtaining
excited states within a time-independent DFT framework. Since
MOM calculations are variational, atomic forces are readily
available from routines for ground-state calculations and can
be used to perform geometry optimization and molecular dynamics.

Excited-state solutions of the SCF equations are obtained
for non-Aufbau orbital occupations and can correspond
to saddle points of the energy as a function of the electronic
degrees of freedom (the orbital variations) [#momgpaw1]_ [#momgpaw2]_
[#momgpaw3]_.
Hence, excited-state calculations can be affected by variational
collapse to lower-energy solutions. MOM is used to choose
a non-Aufbau distribution of the occupation numbers consistent
with the choice of initial guess for an excited state during
optimization of the wave function, thereby avoiding variational
collapse.

--------------
Implementation
--------------
The GPAW implementation of MOM is presented in [#momgpaw1]_
(real space grid and plane waves approaches) and [#momgpaw2]_
(LCAO approach).

The orbitals `\{|\psi_{i}\rangle\}` used as initial guess for an
excited-state calculation are taken as the reference orbitals
for MOM (this approach is also known as initial maximum overlap
method, see [#imom]_). The implementation in GPAW supports the
use of fractional occupation numbers. Let `\{|\psi_{n}\rangle\}_{s}`
be a subspace of `N` initial guess orbitals with occupation
number `f_{s}` and `\{|\psi_{m}^{(k)}\rangle\}` the orbitals
determined at iteration `k` of the wave-function optimization.
An occupation number of `f_{s}` is given to the first `N`
orbitals with the biggest numerical weights, evaluated as
[#Dongmom]_:

.. math::
   :label: eq:mommaxoverlap

    P_{m}^{(k)} = \max_{n}\left( |O_{nm}^{(k)}| \right)

where `O_{nm}^{(k)} = \langle\psi_n | \psi_{m}^{(k)}\rangle`.
Alternatively, the numerical weights can be evaluated as
the following projections onto the manifold `\{|\psi_{n}\rangle\}_{s}`
[#imom]_:

.. math::
   :label: eq:momprojections

    P_{m}^{(k)} = \left(\sum_{n=1}^{N}  |O_{nm}^{(k)}|^{2} \right)^{1/2}

In :ref:`plane-waves<manual_mode>` or :ref:`finite-difference <manual_stencils>`
modes, the elements of the overlap matrix are calculated from:

.. math::
    O_{nm}^{(k)} = \langle\tilde{\psi}_n | \tilde{\psi}_{m}^{(k)}\rangle +
    \sum_{a, i_1, i_2} \langle\tilde{\psi}_n | \tilde{p}_{i_1}^{a}\rangle \left( \langle\phi_{i_1}^{a} | \phi_{i_2}^{a}\rangle -
    \langle\tilde{\phi}_{i_1}^{a} | \tilde{\phi}_{i_2}^{a}\rangle \right) \langle\tilde{p}_{i_2}^{a} | \tilde{\psi}_{m}^{(k)}\rangle

where `|\tilde{\psi}_{n}\rangle` and `|\tilde{\psi}_{m}^{(k)}\rangle`
are the pseudo orbitals, `|\tilde{p}_{i_1}^{a}\rangle`, `|\phi_{i_1}^{a}\rangle`
and `|\tilde{\phi}_{i_1}^{a}\rangle` are projector functions, partial
waves and pseudo partial waves localized on atom `a`, respectively.
In :ref:`LCAO <lcao>`, the overlaps `O_{nm}^{(k)}` are calculated as:

.. math::
    O_{nm}^{(k)} = \sum_{\mu\nu} c^*_{\mu n}S_{\mu\nu}c^{(k)}_{\nu m}, \qquad
    S_{\mu\nu} = \langle\Phi_{\mu} | \Phi_{\nu}\rangle +
    \sum_{a, i_1, i_2} \langle\Phi_{\mu} | \tilde{p}_{i_1}^{a}\rangle \left( \langle\phi_{i_1}^{a} | \phi_{i_2}^{a}\rangle -
    \langle\tilde{\phi}_{i_1}^{a} | \tilde{\phi}_{i_2}^{a}\rangle \right) \langle\tilde{p}_{i_2}^{a} | \Phi_{\nu}\rangle

where `c^*_{\mu n}` and `c^{(k)}_{\nu m}` are the expansion
coefficients for the initial guess orbitals and orbitals at
iteration `k`, while `|\Phi_{\nu}\rangle` are the basis functions.

--------------
Notes on usage
--------------
Tipically, one first performs a ground-state calculation.
To prepare the calculator for an excited-state calculation,
the function ``mom.mom_calculation`` can be used::

  from gpaw import mom

  mom.mom_calculation(calc, atoms, f)

where ``f`` contains the occupation numbers of the excited state
(see examples below).

The default is to use eq. :any:`eq:mommaxoverlap` to compute
the numerical weights used to assign the occupation numbers.
This was found to be more stable in the presence of many diffuse
virtual orbitals [#Dongmom]_. In order to use eq. :any:`eq:momprojections`,
instead, corresponding to the original MOM approach [#imom]_,
one has to specify::

  mom.mom_calculation(..., use_projections=True, ...)


For such cases, it is possible to use a Gaussian smearing
of the holes and excited electrons in the MOM calculation
to improve convergence. This is done by specifying

.. autofunction:: gpaw.mom.mom_calculation

----------------------------------------
Example I: Molecular Rydberg excitations
----------------------------------------

.. literalinclude:: mom_h2o.py

---------------------------------------------
Example II: Excited-state geometry relaxation
---------------------------------------------

.. literalinclude:: mom_co.py

----------
References
----------

.. [#momgpaw1] A. V. Ivanov, G. Levi, H. Jónsson
               `Direct Optimization Method for Variational Excited-State Density Functional Calculations Using Real Space Grid or Plane Waves <https://arxiv.org/abs/2102.06542>`_
               (2020).

.. [#momgpaw2] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Density Functional Calculations of Excited States via Direct Optimization <10.1021/acs.jctc.0c00597>`,
               *J. Chem. Theory Comput.*, **16** 6968–6982 (2020).

.. [#momgpaw3] G. Levi, A. V. Ivanov, H. Jónsson
               :doi:`Variational Calculations of Excited States Via Direct Optimization of Orbitals in DFT <10.1039/D0FD00064G>`,
               *Faraday Discuss.*, **224** 448-466 (2020).

.. [#imom]     G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill
               :doi:`Simple Models for Difficult Electronic Excitations <10.1021/acs.jctc.7b00994>`,
               *J. Chem. Theory Comput.*, **14** 1501-1509 (2018).

.. [#Dongmom]  X. Dong, A. D. Mahler, E. M. Kempfer-Robertson, L. M. Thompson
               :doi:`Global Elucidation of Self-Consistent Field Solution Space Using Basin Hopping <10.1021/acs.jctc.0c00488>`,
               *J. Chem. Theory Comput.*, **16** 5635−5644 (2020).