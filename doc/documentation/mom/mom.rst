.. _mom:

======================
Maximum Overlap Method
======================

The maximum overlap method (MOM) is used to perform
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
collapse to lower-energy solutions. MOM is a simple strategy to
choose non-Aufbau occupation numbers consistent
with the initial guess for an excited state during
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

Typically, one first performs a ground-state calculation.
To prepare the calculator for an excited-state calculation,
the function ``mom.prepare_mom_calculation`` can be used::

  from gpaw import mom

  mom.prepare_mom_calculation(calc, atoms, f)

where ``f`` contains the occupation numbers of the excited state
(see examples below). Alternatively, the MOM calculation can be
initialized by setting ``calc.set(occupations={'name': 'mom', 'numbers': f_sn}``.

The default is to use eq. :any:`eq:mommaxoverlap` to compute
the numerical weights used to assign the occupation numbers.
This was found to be more stable in the presence of diffuse
virtual orbitals [#Dongmom]_. In order to use eq. :any:`eq:momprojections`,
instead, corresponding to the original MOM approach [#imom]_,
one has to specify::

  mom.prepare_mom_calculation(..., use_projections=True, ...)

SCF algorithms based on diagonalization of the Hamiltonian
matrix tend to fail when degenerate or nearly degenerate
orbitals are unequally occupied, a situation that is common
in excited-state calculations.
For such cases, it is possible to use a Gaussian smearing
of the holes and excited electrons in the MOM calculation
to improve convergence. This is done by specifying a ``width``
in eV (e.g. ``width=0.01``) for the Gaussian smearing function.
For difficult cases, the ``width`` can be increased at regular
intervals to force convergence by specifying a ``width_increment=...``.
*Note*, however, that too extended smearing can lead to
discontinuities in the potentials and forces close to
crossings between electronic states [#momgpaw2]_, so this feature
should only be used at geometries far from state crossings.

.. autofunction:: gpaw.mom.prepare_mom_calculation


.. _example_1:

-----------------------------------------------------
Example I: Excitation energy molecular Rydberg states
-----------------------------------------------------
In this example, the excitation energies of the singlet and
triplet states of water corresponding to excitation
from the HOMO-1 non-bonding (`n`) to the LUMO `3s` Rydberg
orbitals are calculated. The `n` and `3s` orbitals have the
same symmetry (a1), therefore, variational collapse can
potentially affect a calculation without MOM. In order to calculate
the energy of the open-shell singlet state, first a calculation
of the mixed-spin state obtained for excitation within the same
spin channel is performed, and, then, the spin-purification
formula is used: `E_s=2E_m-E_t`, where `E_m` and `E_t` are
the energies of the mixed-spin and triplet states, respectively.

.. literalinclude:: mom_h2o.py


---------------------------------------------
Example II: Excited-state geometry relaxation
---------------------------------------------
Here, the bond length of the carbon monoxide molecule
is optimized in the singlet excited state obtained by
promotion of an electron from the HOMO `\sigma` orbital
to the LUMO `\pi^*_x` or `\pi^*_y` orbital. A practical
choice for geometry optimization and dynamics in an
open-shell singlet excited state is to employ a spin-paired
approach where the occupation numbers of the open-shell
orbitals are set to 1 [#levi2018]_. This approach delivers
pure singlet states while avoiding an additional calculation of the
corresponding triplet state needed to employ the spin-purification
formula (see :ref:`example_1`). Since the `\pi^*_x` and
`\pi^*_y` orbitals of carbon monoxide are degenerate,
diagonalization-based SCF algorithms fail to converge
to the `\sigma\rightarrow\pi^*` excited state unless
symmetry constraints on the density are used. Here,
Gaussian smearing of the excited electron is used to
force equal fractional occupations of the two `\pi^*`
orbitals to avoid convergence issues.

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

.. [#levi2018] G. Levi, M. Pápai, N. E. Henriksen, A. O. Dohn, K. B. Møller
               :doi:`Solution structure and ultrafast vibrational relaxation of the PtPOP complex revealed by ∆SCF-QM/MM Direct Dynamics simulations <10.1021/acs.jpcc.8b00301>`,
               *J. Phys. Chem. C*, **122** 7100-7119 (2018).