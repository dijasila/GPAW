.. _directmin:

================================
Direct Minimization Methods
================================

An alternative to self-consistent field algorithms is employing
direct minimisation methods which avoid using density mixing and
diagonalisation of the Kohn-Sham hamiltonian.

LCAO mode.
----------

Exponential Transformation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the localized basis set approach,
the orbitals are expanded onto a finite basis set:

.. math:: \phi_{i} ({\bf r}) = \sum_{\mu=1..M} O_{\mu i} \chi_{\mu}({\bf r}), \quad i = 1 .. M

and the energy needs to be minimised with respect
to the expansion coefficients subject to orthonormality constraints,

.. math:: E_0 = \min_{O^{\dagger}SO = I} E\left(O\right)

If we have some orthonormal reference orbitals with known
coefficient matrix (c.m.) `C`, then *any* c.m. 'O'
can be obtained from the reference c.m. `C`
by some unitary transformation:

.. math:: O = C U

where U is a unitary matrix. Thus, objectives are to find the unitary
matrix which transforms the reference c.m. into an optimal c.m.,
which minimises the energy of electronic system.
A unitary matrix can paramtrised as
the exponential of a skew-hermitain matrix `A`.

.. math:: U = \exp(A)

This parametrisation is convenient since the orthonormality
constraints are satisfied:

.. math:: UU^{\dagger} = \exp(A)\exp(A^{\dagger}) = \exp(A)\exp(-A) =I

If the reference c.m. are fixed,
then the energy is a function of `A`.

.. math:: F\left(A\right) = E\left(C e^A \right)

Furthermore, the skew-hermitian matrices form a linear space and,
therefore, conventional unconstrained minimisation algorithms
can be applied to minimise energy
with respect to `A`.

Example
~~~~~~~~
First of all, it is necessary to ensure that the number of bands used
in calculations is equal to the number of atomic orbitals.
Secondly, one need to use 'dummy' mixer which does not mix.
Here is example of how to run calculations:

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

Implementation
~~~~~~~~~~~~~~~

The iteratives are:

.. math:: A^{(k+1)} = A^{(k)} + \gamma^{(k)} Q^{(k)}

Here `Q` is the search direction and `\gamma` is step length.
The search direction is calculated according L-BFGS algorithm
with preconditioning, and the step length satisfies
the Strong Wolfe Conditions and/or approximate Wolfe Conditions.
The last two conditions are important as they guarantee stability
and fast convergence of the LBFGS algorithm.

Three different algorithms are available to
calculate matrix exponential:

1. The scaling and squaring algorithm which is based on the equation:

   .. math:: \exp(A) = \exp(A/2^{m})^{2^{m}}

   Since :math:`A/2^{m}` has a small norm then :math:`\exp(A/2^{m})`
   can be effectively estimated using a Pade approximant
   of order :math:`[q/q]`. Here q and m are positive integers.
   The algorithm of Al-Moly and Higham [#AlMoly]_ is used here
   as implemented in SciPy library.

2. Using eigendecompostion of matrix :math:`iA`.
   Let :math:`\Omega` be a diagonal real-valued matrix with elements
   corresponding to the eigenvalues of the matrix :math:`iA`,
   and let :math:`U` be a matrix, columns of which are
   the eigenvectors of :math:`iA`.
   Then the matrix exponential of :math:`A` is:

   .. math:: \exp(A) = U \exp(-i\Omega) U^{\dagger}.

3. For a unitary invariant functionals the matrix `A`
   can be parametrised as [#Hutter]_:

   .. math::

     A = \begin{pmatrix}
     0 & A_{ov} \\
     -A_{ov}^{\dagger} & 0
     \end{pmatrix},

   where :math:`A_{ov}` is :math:`N \times (M-N)`,
   where :math:`N` is a number of occupied states and
   :math:`M` is a number of basis functions.
   `0` here is :math:`N \times N` zero matrix. In this case
   the matrix exponentail can be calculated as [#Hutter]_:

   .. math::

     \exp(A) = \begin{pmatrix}
     \cos(P) & P^{-1/2} \sin(P^{1/2}) A_{ov}\\
     -A_{ov}^{\dagger} P^{-1/2} \sin(P^{1/2}) & I_{M-N} +
      A_{ov}^{\dagger}\cos(P^{1/2} - I_N) P^{-1} A_{ov} )
     \end{pmatrix},

   where :math:`P = A_{ov}A_{ov}^{\dagger}`.

References
----------

.. [#AlMoly] A. H. Al-Moly, and N. J. Higham,
           *SIAM J. Matrix Anal. Appl.*, **31(3)**, 970–989, (2009).

.. [#Hutter] J. Hutter, M. Parrinello, and S. Vogel,
             *J. Chem. Phys.* **101**, 3862 (1994)