.. _sic:

===================================================
Perdew-Zunger Self-Interaction Corrections (PZ-SIC)
===================================================
The self-interaction corrected density functional with PZ-SIC has the following form:

.. math:: E^{PZ-SIC}[\{n_i\}] = E^{DFA}[n] - \beta \sum_{i=1}^{N} \left(E_H[n_i] + E^{DFA}_{xc}[n_i]\right)

here `E^{DFA}` - density functional approximation,
`E_H` - Hartree energy and 
`E^{DFA}_{xc}` - exchange correlation part of density functional approximation. 
`n` is the total density and `n_i = |\psi_i (\mathbf{r})|^{2}` - orbital density. 
`\beta` is a scaling factor.

The SIC functional is not a unitary invariant functional and is dependent on orbital densities. 
Therefore, the :ref:`fully variational approach <directmin>` is used to find optimal 
orbitals which provide the ground state energy.

Example
--------
The current implementation uses :ref:`LCAO aprroximation <lcao>` 
with direct minimisation described :ref:`here <directmin>` .
Since the functional is not a unitary invariant functional,
it is necessary to use complex orbitals to find the lowest energy state.
Here is an example:

.. literalinclude:: sic_example.py

The orbital-density dependent potentials are evaluated on a coarse grid in order 
to increase calculation speed. To evaluate these potentials on a fine grid, use:

.. code-block:: python

  odd_parameters={'name': 'PZ_SIC',
                  'scaling_factor': (0.5, 0.5),
                  'sic_coarse_grid': False}


**Important:** Firstly, be aware that unoccupied orbitals
are not affected by SIC directly in LCAO mode.
Secondly, the implementation relies on SciPy library.
In order to obtain a good performance,
please be sure that your SciPy library uses Math Kernel Library (MKL).
Otherwise you can use

.. code-block:: python

  eigensolver=DirectMinLCAO(matrix_exp='egdecomp', ..),

See :ref:`direct minimisation implementation <directmin>` for details.