.. _directmin:

================================
Direct Minimization Methods
================================

An alternative to self-consistent field algorithms is employing direct minimisation methods 
which avoid using density mixing and diagonalisation of the Kohn-Sham hamiltonian. 

LCAO mode.
----------

Exponential Transformation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the localized basis set approach, the orbitals are expanded in a finite basis set: 

.. math:: \phi_{i} ({\bf r}) = \sum_{\mu=1..M} O_{\mu i} \chi_{\mu}({\bf r}), \quad i = 1 .. M

and the energy needs to be minimised with respect 
to the expansion coefficients subject to orthonormality constraints,

.. math:: E_0 = \min_{O^{\dagger}SO = I} E\left(O\right)

For example, if we have some orthonormal reference orbitals with coefficient matrix `C`, 
then one set of the coefficients can be transformed into another 
set by using unitary transformation:

.. math:: O = C \exp(A)

where `A` is a skew-hermitian matrix. If the reference orbitals are fixed, 
then the energy is a function of the skew-hermitian matrix and the constraint conditions are automatically satisfied:
							
.. math:: E\left(O\right) = E\left(C e^A \right) = E\left(A\right) 

Furthermore, the skew-hermitian matrices form a linear space and, 
therefore, conventional unconstrained minimisation algorithms 
can be applied to minimise energy with respect to skew-hermitian matrices.

Implementation
~~~~~~~~~~~~~~~

The iteratives are:

.. math:: A^{(k+1)} = A^{(k)} + \gamma^{(k)} Q^{(k)}

Here `Q` is the search direction and `\gamma` is step length.
The search direction is calculated according L-BFGS algorithm with preconditioning, 
and the step length satisfies the Strong Wolfe Conditions
and/or approximate Wolfe Conditions.
The last two conditions are important as they guarantee stability
and fast convergence of the LBFGS algorithm.

.. The preconditioning can be calculated for this problem as:

Example
~~~~~~~~
First of all, it is necessary to ensure that the number of bands used in calculations is equal to the number of atomic orbitals.
Secondly, as the inexact line search method is used 
in order to find an optimal step length during the minimisation,
it is important to reduce error in the energy due to inaccuracies in the Poisson solver.

Here is example:

.. literalinclude:: directmin_ch4.py

Not only can direct minimisation be applied to Kohn-Sham functionals
but also to :ref:`self-interaction corrected functionals <sic>`.

Performance. G2 molecular set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of energy and gradient evaluations in direct minimisation
and the number of iterations in the scf algorithm employing default density-mixing are shown below. 
Figure (a) shows several examples of molecules. 
Figure (b) shows the results of the direct minimisation for molecules for which scf 
with default density-mixing fails to converge.  

|scf_vs_dm|

.. |scf_vs_dm| image:: scf_vs_dm.png
