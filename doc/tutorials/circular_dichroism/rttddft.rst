.. _circular_dichroism_rtddft:

=======================================
Circular dichroism with real-time TDDFT
=======================================

In this tutorial, we calculate the rotatory strength spectrum of
:download:`(R)-methyloxirane molecule <r-methyloxirane.xyz>`.

The equations underlying the implementation are described in
Ref. [#Makkonen2021]_.


LCAO mode
---------

In this example, we use :ref:`real-time TDDFT LCAO mode <lcaotddft>`.
We recall that the LCAO calculations are
:ref:`sensitive to the used basis sets <note basis sets>`,
and we also demonstrate the construction of augmented basis sets.

We augment the default dzp basis sets with
numerical Gaussian-type orbitals (NGTOs):

.. literalinclude:: lcao/basis.py

Here, the Gaussian parameters correspond to diffuse functions
in aug-cc-pvdz basis sets as obtained from
`Basis Set Exchange <https://www.basissetexchange.org/>`_.

Let's do the ground-state calculation (note the ``basis`` keyword):

.. literalinclude:: lcao/gs.py

Then, we carry the time propagation:

.. literalinclude:: lcao/td.py

Finally, we calculate the spectrum (takes a few seconds):

.. literalinclude:: lcao/spec.py

The resulting spectrum:

.. image:: lcao/spectra.png


.. [#Makkonen2021]
   | E. Makkonen, T. P. Rossi, A. H. Larsen, O. Lopez-Acevedo, P. Rinke,  M. Kuisma, and X. Chen,
   | :doi:`Real-time time-dependent density functional theory implementation of electronic circular dichroism applied to nanoscale metal-organic clusters <10.1063/5.0038904>`
   | J. Chem. Phys. **154**, 114102 (2021)
