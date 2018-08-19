.. _directmin:

================================
Direct minimization methods
================================

Alternative to self-consistent field method is direct minimisation methods.

LCAO mode.
----------

Exponential transformation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the localized basis set approach, the orbitals are expanded in a finite basis set: 

.. math:: \phi_{i} ({\bf r}) = \sum_{\mu=1..M} O_{\mu i} \chi_{\mu}({\bf r}), \quad i = 1 .. M

and the energy needs to be minimised with respect to the expansion coefficients subject to orthonormality constraints,

.. math:: E_0 = \min_{O^{\dagger}SO = I} E\left(O\right)

Say that we have some orthonormal  reference orbitals with coefficient matrix `C`, then one set of coefficients can be transformed into another set by using unitary transformation(2,3):

.. math:: O = C \exp(A)

where `A` is a skew-hermitian matrix. If the reference orbitals are fixed, then the energy is a function of the skew-hermitian matrix and constraint conditions are automatically satisfied:
							
.. math:: E\left(O\right) = E\left(C e^A \right) = E\left(A\right) 

Furthermore, the skew-hermitian matrices form a linear space and therefore, conventional unconstraint minimisation algorithms can be applied to minimise energy with respect to skew-hermitian matrices.

Implementation.
~~~~~~~~~~~~~~~

Iteratives are:

.. math:: A^{(k+1)} = A^{(k)} + \gamma^{(k)} Q^{(k)}

Here `Q` is the search direction, `\gamma` is step length.
The search direction is calculated according L-BFGS algorithm, 
and step length satisfies the Strong Wolfe Conditions
and/or approximate Wolfe Conditions.
The last two conditions are important as they guarantee stability
and fast convergence of the LBFGS algorithm.

.. The preconditioning can be calculated for this problem as:

Example.
~~~~~~~~
First of all, it's very important to include all bands in calculations
that is total number of bands equals number of atomic orbitals
Secondly, as the inexact line search method is used in order to find optimal step length during the minimisation,
it is important to get rid off noise in energy due to inaccuracy in Poisson solver.

Here is example:

.. literalinclude:: directmin_h2o.py
