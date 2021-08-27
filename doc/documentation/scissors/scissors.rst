.. _scissors operator:

===============================
Scissors operator for LCAO mode
===============================

.. warning:: **Work in progress**

.. module:: gpaw.lcao.scissors
.. autoclass:: Scissors

In :ref:`lcao` we solve the following generalized eigenvalue problem:

.. math::

 \sum_\nu (H + \Delta H)_{\mu\nu} C_{\nu n}
 = \sum_{\nu} S_{\mu\nu} C_{\nu n} \epsilon_n,

where `\Delta H` is a scissors operator.

Space is divided into regions `\Omega_i` and for each region we define desired
shifts of the occupied and unoccupied bands: `\Delta_{i,\text{occ}}` and
`\Delta_{i,\text{unocc}}`.  The scissors operator is given as:

.. math::

    \Delta H = \sum_i(\Delta H^{i,\text{occ}}+\Delta H^{i,\text{unocc}}),

where

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
