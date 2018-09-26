.. _sic:

===================================================
Perdew-Zunger Self-Interaction Corrections (PZ-SIC)
===================================================
Self-interaction corrected density functional with PZ-SIC has the following form:

.. math:: E^{PZ-SIC}[\{n_i\}] = E^{DFA}[n] - \beta \sum_{i=1}^{N} \left(E_H[n_i] + E^{DFA}_{xc}[n_i]\right)

here `E^{DFA}` - density functional approximation,
`E_H` - Hartree energy and 
`E^{DFA}_{xc}` - exchange correclation part of density functional approximation. 
`n` is the total density and `n_i = |\psi_i (\mathbf{r})|^{2}` - orbital density. 
`\beta` is screening factor. 

The SIC functional is not unitary invariant and dependends on orbital densities. Therefore, :ref:`fully variational approach <directmin>` is used to find optimal orbitals which provides the ground state energy.

Example.
--------
Current implementation uses :ref:`LCAO aprroximation <lcao>` and algrotihm described :ref:`here <directmin>` . 
Since functional is not unitary invariant it's nessaccary to use complex orbitals to find the lowest energy state.
Here is example:

.. literalinclude:: sic_example.py
