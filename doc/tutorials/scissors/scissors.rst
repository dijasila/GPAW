.. _scissors operator:

=================
Scissors operator
=================

.. warning:: **Work in progress**

.. module:: gpaw.lcao.scissors
.. autoclass:: Scissors

:ref:`lcao`

.. math::

 \sum_\nu H_{\mu\nu} C_{\nu n} = \sum_{\nu} S_{\mu\nu} C_{\nu n} \epsilon_n

.. math::

    \Delta H_{\mu\nu} = \sum_i \left(
        \Delta_{i,\text{occ}}
        \sum_{n,n'}^{\text{occ}}
        \sum_{\mu',\nu'\in\Omega_i}
        C_{n\mu}^{-1}
        C_{\mu'n}
        S_{\mu'\nu'}
        C_{\nu'n'}
        C_{n'\nu}^{-1} \right.
        \\
        \left. + \Delta_{i,\text{unocc}}
        \sum_{n,n'}^{\text{unocc}}
        \sum_{\mu',\nu'\in\Omega_i}
        C_{n\mu}^{-1}
        C_{\mu'n}
        S_{\mu'\nu'}
        C_{\nu'n'}
        C_{n'\nu}^{-1}
        \right)


Example
=======

:download:`mos2ws2.py`.
