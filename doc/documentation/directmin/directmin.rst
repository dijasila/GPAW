.. _directmin:

================================
Direct Minimization Methods
================================

An alternative to self-consistent field algorithms is employing
direct minimization methods, which avoid density mixing and
diagonalization of the Kohn-Sham Hamiltonian matrix.

LCAO mode
----------


Exponential Transformation Direct Minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orbitals are expanded into a finite basis set:

.. math:: \phi_{i} ({\bf r}) = \sum_{\mu=1..M} O_{\mu i} \chi_{\mu}({\bf r}), \quad i = 1 .. M

and the energy needs to be minimised with respect
to the expansion coefficients subject to orthonormality constraints:

.. math:: E_0 = \min_{O^{\dagger}SO = I} E\left(O\right)

If we have some orthonormal reference orbitals with known
coefficient matrix (c.m.) `C`, then *any* c.m. `O`
can be obtained from the reference c.m. `C`
by some unitary transformation:

.. math:: O = C U

where U is a unitary matrix. Thus, the objective is to find the unitary
matrix which transforms the reference c.m. into an optimal c.m.,
minimizing the energy of the electronic system.
A unitary matrix can parametrised as
the exponential of a skew-hermitian matrix `A`:

.. math:: U = \exp(A)

This parametrisation is advantageous since the orthonormality
constraints are automatically satisfied:

.. math:: UU^{\dagger} = \exp(A)\exp(A^{\dagger}) = \exp(A)\exp(-A) =I

If the reference c.m. is fixed,
then the energy is a function of `A`:

.. math:: F\left(A\right) = E\left(C e^A \right)

The skew-hermitian matrices form a linear space and,
therefore, conventional unconstrained minimization algorithms
can be applied to minimise the energy
with respect to `A`.

Example
~~~~~~~~
Firstly, it is necessary to ensure that the number of bands used
in calculations is equal to the number of atomic orbitals.
Secondly, one needs to use a mixer which does not mix the density.
Here is example of how to run the calculations:

.. literalinclude:: h2o.py

As one can see, it is possible to specify the amount of memory used in L-BFGS algorithm.
The large the memory, the more efficient algorithm is. Default value is 3.

**Important:** The exponential matrix is calculated here using
the SciPy function *expm*. In order to obtain a good performance,
please be sure that your SciPy library is optimized
(e.g. use the Math Kernel Library (MKL)).
Otherwise see `Implementation Details`_.

Since fot this example, the occupied orbitals have the same
occupation numbers a more efficient algorithm can be used:

.. code-block:: python

    calc = GPAW(eigensolver=DirectMinLCAO(matrix_exp='egdecomp2',
                                          representation='u_invar'),
                ...)

.. _Performance:

Performance
~~~~~~~~~~~~~

G2 molecular set
`````````````````

The number of energy and gradient evaluations in direct minimization
using L-BFGS algorithm (memory=3) with preconditioning
and the number of iterations in the SCF algorithm employing default
density-mixing are shown below.
Figure (a) shows several examples of molecules.
Figure (b) shows the results of the direct minimization for molecules
for which SCF with default density-mixing fails to converge.
SCF fails to converge for 5 molecules,
while direct minimization demonstrates a stable performance. Note,
by choosing different mixing parameters one can improve the convergence
of SCF methods.

.. image:: g2.png

32-576 Water molecules
```````````````````````
In this example, the ground state of 32, 64, 128, 256, 384 and 576
water molecules is calculated respectively. The geometries are taken
from `here <https://wiki.fysik.dtu.dk/gpaw/devel/benchmarks.html>`_
The GPAW parameters used in this test include the PBE functional,
the DZP basis set, grid spacing h = 0.2,
8-core domain decomposition, and
the ’eingensolver’ convergence criterion 1.0e-10.
The ratio of the elapsed times spent by the default eigensolver and
the direct minimization methods as a function of
water molecules is shown below.
In the picture, ‘ss’ refers to the scaling and squaring method [#AlMoly]_ for
the calculation of the matrix exponential, while
‘uinv’ refers to the method for the calculation of
the matrix exponential taking into account the unitary invariance
of the KS functional [#Hutter]_ (see `Implementation Details`_).
As can be seen, direct minimization converges faster
by a factor of 1.5 for 32 molecules and a factor of 2 for 512 molecules
using the 'uinv' method. 'ss' outperforms the SCF by a factor of 1.2.

.. image:: water.png
  :width: 70%
  :align: center

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The implementation follows the ref. [#Ivanov2021]_ The iteratives are:

.. math:: A^{(k+1)} = A^{(k)} + \gamma^{(k)} Q^{(k)}

Here `Q` is the search direction and `\gamma` is step length.
The search direction is calculated according to the L-BFGS algorithm
with preconditioning, and the step length satisfies
the Strong Wolfe Conditions [#Nocedal]_ and/or
approximate Wolfe Conditions [#Hager]_.
The last two conditions are important as they guarantee stability
and fast convergence of the L-BFGS algorithm [#Nocedal]_. Apart from L-BFGS
algorithm, one can use limited-memory symmetric rank one (L-SR1, default memory is 20)
quasi-Newton algorithm which also demonstrates good performance and is
especially recommended for the calculation of the excited states [#Levi2020]_
(see also :ref:`Maximum Overlap Method and Direct Optimization <mom>` ).
There is also an option to use a conjugate gradient algortihm but it is less efficient.


Here are three algorithms which can be used to calculate the
matrix exponential:

1. The scaling and squaring algorithm which is based on the equation:

   .. math:: \exp(A) = \exp(A/2^{m})^{2^{m}}

   Since :math:`A/2^{m}` has a small norm, then :math:`\exp(A/2^{m})`
   can be effectively estimated using a Pade approximant
   of order :math:`[q/q]`. Here q and m are positive integers.
   The algorithm of Al-Moly and Higham [#AlMoly]_
   from the SciPy library is used here.

2. Using eigendecompostion of the matrix :math:`iA`.
   Let :math:`\Omega` be a diagonal real-valued matrix with elements
   corresponding to the eigenvalues of the matrix :math:`iA`,
   and let :math:`U` be a matrix, the columns of which are
   the eigenvectors of :math:`iA`.
   Then the matrix exponential of :math:`A` is:

   .. math:: \exp(A) = U \exp(-i\Omega) U^{\dagger}.

3. For a unitary invariant functional, the matrix `A`
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
   the matrix exponential can be calculated as [#Hutter]_:

   .. math::

     \exp(A) = \begin{pmatrix}
     \cos(P) & P^{-1/2} \sin(P^{1/2}) A_{ov}\\
     -A_{ov}^{\dagger} P^{-1/2} \sin(P^{1/2}) & I_{M-N} +
      A_{ov}^{\dagger}\cos(P^{1/2} - I_N) P^{-1} A_{ov} )
     \end{pmatrix},

   where :math:`P = A_{ov}A_{ov}^{\dagger}`.

The first method is the default choice. If one would like to use
the second algorithm, then do the following:

.. code-block:: python

    from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
    calc = GPAW(eigensolver=DirectMinLCAO(matrix_exp='egdecomp'),
                ...)

To use the third method, firstly please ensure that your functional
is unitary invariant and then do the following:

.. code-block:: python

    from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
    calc = GPAW(eigensolver=DirectMinLCAO(matrix_exp='egdecomp2',
                                          representation='u_invar'),
                ...)

The last option is the most efficient but it is valid only for a unitary invariant functionals
(e.g. when occupied orbitals for a given spin state have the same occupation numbers)

For all three algorithms, the unitary invariant representation can be chosen.

ScaLAPCK and the parallelization over bands are currently not supported.
It is also not recommended to use the direct minimization for metals
because the occupation numbers are not found variationally
but rather fixed during calculations.

References
~~~~~~~~~~

.. [#Ivanov2021] A. V. Ivanov, E.Ö. Jónsson, T. Vegge, and H. Jónsson,
         *Comput. Phys. Commun.*, **267**, 108047 (2021).

.. [#Levi2020] G. Levi, A. V. Ivanov, and H. Jónsson, J.
           *Chem. Theory Comput.*, **16**, 6968, (2020).

.. [#Hutter] J. Hutter, M. Parrinello, and S. Vogel,
             *J. Chem. Phys.* **101**, 3862 (1994)

.. [#Nocedal] J. Nocedal and S. J. Wright, *Numerical Optimization*,
              2nd ed. (Springer, New York, NY, USA, 2006).

.. [#Hager] W. W. Hager and H. Zhang,
            *SIAM Journal on Optimization* **16**, 170 (2006).

.. [#AlMoly] A. H. Al-Moly, and N. J. Higham,
           *SIAM J. Matrix Anal. Appl.*, **31(3)**, 970–989, (2009).
