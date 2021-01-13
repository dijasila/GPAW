.. _scissors operator:

===============================
Scissors operator for LCAO mode
===============================

.. warning:: **Work in progress**

.. module:: gpaw.lcao.scissors
.. autoclass:: Scissors

:ref:`lcao`

.. math::

 \sum_\nu H_{\mu\nu} C_{\nu n} = \sum_{\nu} S_{\mu\nu} C_{\nu n} \epsilon_n

`\Delta H = \sum_i(\Delta H^{i,\text{occ}}+\Delta H^{i,\text{unocc}})`

.. math::

    \Delta H_{\mu\nu}^{i,\text{occ}} =
        \Delta_{i,\text{occ}}
        \sum_{n,n'}^{\text{occ}}
        \sum_{\mu',\nu'\in\Omega_i}
        C_{n\mu}^{-1}
        C_{\mu'n}
        S_{\mu'\nu'}
        C_{\nu'n'}
        C_{n'\nu}^{-1},

.. math::

    \Delta H_{\mu\nu}^{i,\text{unocc}} =
        \Delta_{i,\text{unocc}}
        \sum_{n,n'}^{\text{unocc}}
        \sum_{\mu',\nu'\in\Omega_i}
        C_{n\mu}^{-1}
        C_{\mu'n}
        S_{\mu'\nu'}
        C_{\nu'n'}
        C_{n'\nu}^{-1}.


Example
=======

:download:`mos2ws2.py`.
