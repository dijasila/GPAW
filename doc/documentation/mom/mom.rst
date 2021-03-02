.. _mom:

======================
Maximum Overlap Method
======================

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

