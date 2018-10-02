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
`\beta` is a screening factor. 

The SIC functional is not a unitary invariant functional and is dependent on orbital densities. 
Therefore, the :ref:`fully variational approach <directmin>` is used to find optimal 
orbitals which provide the ground state energy.

Example
--------
The current implementation uses :ref:`LCAO aprroximation <lcao>` 
and the algorithm described :ref:`here <directmin>` . 
Since the functional is not a unitary invariant functional,
it is necessary to use complex orbitals to find the lowest energy state.
Here is an example:

.. literalinclude:: sic_example.py

The orbital-density dependent potentials are evaluated on a coarse grid in order 
to increase calculation speed. To evaluate these potentials on a fine grid, use::

  >>> opt = ODD(..., sic_coarse_grid=False)

