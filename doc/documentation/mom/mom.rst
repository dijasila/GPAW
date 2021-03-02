.. _mom:

======================
Maximum Overlap Method
======================

The maximum overlap method (MOM) can be used to convergence to
excited-state solutions of the SCF equations, corresponding to
non-Aufbau orbital occupations. Thus, it is an alternative to
the linear expansion :ref:`dscf` approach for variational calculations
of excited states.

Excited-state solutions often correspond to saddle points of
the energy as a function of the electronic degrees of freedom
(i.e. the orbital variations), and excited-state calculations
are susceptible to variational collapse to lower-energy
solutions of the same symmetry. MOM

--------------
Implementation
--------------
The GPAW implementation of MOM is presented in

.. math::
    P_{m}^{(k)} = \left(\sum_{n=1}^{N}  |O_{nm}^{(k)}|^{2} \right)^{1/2}

.. math::
    P_{m}^{(k)} = \max_{n}\left( |O_{nm}^{(k)}| \right)

.. math::
    O_{nm}^{(k)} = \langle\psi_n | \psi_{m}^{(k)}\rangle = \langle\tilde{\psi}_n | \tilde{\psi}_{m}^{(k)}\rangle +
    \sum_{a, i_1, i_2} \langle\tilde{\phi}_n | \tilde{p}_{i_1}^{a}\rangle \left( \langle\phi_{i_1}^{a} | \phi_{i_2}^{a}\rangle -
    \langle\tilde{\phi}_{i_1}^{a} | \tilde{\phi}_{i_2}^{a}\rangle \right) \langle\tilde{p}_{i_2}^{a} | \tilde{\psi}_{m}^{(k)}\rangle

.. math::
    O_{nm}^{(k)} = \sum_{\mu\nu} c^*_{\mu n}S_{\mu\nu}c^{(k)}_{\nu m}, \qquad
    S_{\mu\nu} = \langle\Phi_{\mu} | \Phi_{\nu}\rangle +
    \sum_{a, i_1, i_2} \langle\Phi_{\mu} | \tilde{p}_{i_1}^{a}\rangle \left( \langle\phi_{i_1}^{a} | \phi_{i_2}^{a}\rangle -
    \langle\tilde{\phi}_{i_1}^{a} | \tilde{\phi}_{i_2}^{a}\rangle \right) \langle\tilde{p}_{i_2}^{a} | \Phi_{\nu}\rangle

------------------------------------------------------
Example of application to Rydberg excitations in water
------------------------------------------------------

.. literalinclude:: mom_h2o.py