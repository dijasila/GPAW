.. _bse:

.. default-role:: math

==========================================
Bethe Salpeter Equation (BSE) for excitons
==========================================

Introduction
============
The BSE object calculates optical properties of extended systems including the electron-hole interaction (excitonic effects). 


The four point Bethe-Salpeter equation
======================================

Please refer to :ref:`df_theory` for the documentation on the density response function  `\chi`. 
Most of the derivations in this page follow reference  \ [#Review]_.

The following diagrams  \ [#Review]_ representing the four point Bethe-Salpeter equation: 

.. image:: Feynman.png
	   :height: 200 px

It can be written as: 

.. math::
   :label: chi_4point

   \chi(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3, \mathbf{r}_4; \omega)
   = \chi^{0}(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3, \mathbf{r}_4; \omega)
   + \int d \mathbf{r}_5 d \mathbf{r}_6 d \mathbf{r}_7 d \mathbf{r}_8
     \chi^{0}(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_5, \mathbf{r}_6; \omega)
       K( \mathbf{r}_5, \mathbf{r}_6, \mathbf{r}_7, \mathbf{r}_8; \omega)
          \chi(\mathbf{r}_7, \mathbf{r}_8, \mathbf{r}_3, \mathbf{r}_4; \omega)

where 

.. math::

   K = V - W
   = \frac{1}{| \mathbf{r}_5 -  \mathbf{r}_7|} 
     \delta_{ \mathbf{r}_5, \mathbf{r}_6}  \delta_{ \mathbf{r}_7, \mathbf{r}_8}
     -  \frac{\epsilon^{-1}( \mathbf{r}_5,  \mathbf{r}_6; \omega )}
      {| \mathbf{r}_5 -  \mathbf{r}_6|} 
     \delta_{ \mathbf{r}_5, \mathbf{r}_7}  \delta_{ \mathbf{r}_6, \mathbf{r}_8}
   
The density response function `\chi`, defined as  `\chi(\mathrm{r}, \mathrm{r}^{\prime}) = \delta n(\mathrm{r}) / \delta V_{ext}(\mathrm{r}^{\prime})`, has a form of 

.. math::
   :label: chi_2point

   \chi(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3, \mathbf{r}_4; \omega)
   = \chi(\mathbf{r}_1, \mathbf{r}_3; \omega)  \delta_{ \mathbf{r}_1, \mathbf{r}_2}
      \delta_{ \mathbf{r}_3, \mathbf{r}_4}

The above equation also applies for the non interacting density response function  `\chi^0`. As a result, the four point Bethe-Salpeter equation :eq:`chi_4point`  can be reduced to:

.. math::
   :label: chi_reduced

   \chi(\mathbf{r}, \mathbf{r}^{\prime}; \omega)
   = \chi^0(\mathbf{r}, \mathbf{r}^{\prime}; \omega)
     + \int d \mathbf{r}_5 d \mathbf{r}_7 
      \chi^0(\mathbf{r}, \mathbf{r}_5; \omega)
      \frac{1}{| \mathbf{r}_5 -  \mathbf{r}_7|}  
       \chi(\mathbf{r}_7, \mathbf{r}^{\prime}; \omega)
     + \int d \mathbf{r}_5 d \mathbf{r}_6 
      \chi^0(\mathbf{r}, \mathbf{r}_5,  \mathbf{r}_6; \omega)
        \frac{\epsilon^{-1}( \mathbf{r}_5,  \mathbf{r}_6; \omega )}
      {| \mathbf{r}_5 -  \mathbf{r}_6|} 
      \chi(\mathbf{r}_5, \mathbf{r}_6, \mathbf{r}^{\prime}; \omega)      

Transform using electron-hole pair basis
========================================
Since for each excitation, only a limited number of electron-hole pairs will contribute , the above equation can be effectively transformed to electron-hole pair space. Supposed that the eigenfunctions `\psi_{n}` of the effective Kohn-Sham hamiltonian form an orthonormal and complete basis set, any four point function  `S` can then be transformed as 

.. math::
   :label: S

   S(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3, \mathbf{r}_4; \omega)
   = \sum_{n_1 n_2 n_3 n_4} \psi^{\ast}_{n_{1}}(\mathbf{r}_1)
    \psi_{n_{2}}(\mathbf{r}_2)  \psi_{n_{3}}(\mathbf{r}_3) 
    \psi^{\ast}_{n_{4}}(\mathbf{r}_4) 
    S_{\begin{array}{l} n_1 n_2 \\ n_3 n_4  \end{array}} (\omega)

The non interacting density response function  `\chi^0`

.. math::
   :label: chi_0
   
    \chi^0(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3, \mathbf{r}_4; \omega)
    = \sum_{n n^{\prime}} \frac{f_n - f_{n^{\prime}}}{\epsilon_n - \epsilon_{n^{\prime}}-\omega} \psi^{\ast}_n(\mathbf{r}_1)
    \psi_{n^{\prime}}(\mathbf{r}_2)  \psi_n(\mathbf{r}_3) 
    \psi^{\ast}_{n^{\prime}}(\mathbf{r}_4) 

is then diagonal in the electron-hole basis with 

.. math::
   :label: chi_0_eh

    \chi^0_{\begin{array}{l} n_1 n_2 \\ n_3 n_4  \end{array}} (\omega)
    =   \frac{f_{n_2} - f_{n_1}}{\epsilon_{n_2} - \epsilon_{n_1}-\omega} \delta_{n_1, n_3} \delta_{n_2, n_4} 

Substitute Eq. :eq:`S` and :eq:`chi_0` into Eq. :eq:`chi_reduced` and by using Eq. :eq:`chi_2point` ,the four point Bethe-Salpeter equation in electron-hole pair space becomes

.. math::
   :label: chi_eh

    \chi_{\begin{array}{l} n_1 n_2 \\ n_3 n_4  \end{array}} (\omega)
    = \chi^0_{n_1 n_2} (\omega) \left[ \delta_{n_1 n_3} \delta_{n_2 n_4} + \sum_{n_5 n_6} 
     K_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}} (\omega)
     \chi_{\begin{array}{l} n_5 n_6 \\ n_3 n_4  \end{array}} (\omega) \right] 

with  `K = V - W` and 

.. math::

    V_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}} 
    = \int d \mathbf{r} d \mathbf{r}^{\prime}
    \psi_{n_1}(\mathbf{r}) \psi_{n_2}^{\ast}(\mathbf{r}) \frac{1}{|  \mathbf{r}-\mathbf{r}^{\prime} |}
     \psi^{\ast}_{n_5}(\mathbf{r}^{\prime}) \psi_{n_6}(\mathbf{r}^{\prime}) 

.. math::

    W_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}} (\omega)
    = \int d \mathbf{r} d \mathbf{r}^{\prime}
    \psi_{n_1}(\mathbf{r}) \psi_{n_2}^{\ast}(\mathbf{r}^{\prime}) \frac{\epsilon^{-1}( \mathbf{r},  \mathbf{r}^{\prime}; \omega )}{|  \mathbf{r}-\mathbf{r}^{\prime} |}
     \psi^{\ast}_{n_5}(\mathbf{r}) \psi_{n_6}(\mathbf{r}^{\prime})


Bethe-Salpeter equation as an effective two-particle Hamiltonian
================================================================

In order to solve Eq. :eq:`chi_eh`, one has to invert a matrix for each frequency. 
This problem can be reformulated as an effective eigenvalue problem. Rewrite Eq. :eq:`chi_eh`
as 

.. math::

   \sum_{n_5 n_6} \left[ \delta_{n_1 n_5} \delta_{n_2 n_6}  - 
   \chi^0_{n_1 n_2}(\omega) K_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}} (\omega)
    \right]
     \chi_{\begin{array}{l} n_5 n_6 \\ n_3 n_4  \end{array}} (\omega)
   =  \chi^0_{n_1 n_2}(\omega)

Insert Eq. :eq:`chi_0_eh` into the above equation, one gets

.. math::
   :label: chi_rewrite

   \sum_{n_5 n_6} \left[  (\epsilon_{n_2} - \epsilon_{n_1}-\omega)
    \delta_{n_1 n_5} \delta_{n_2 n_6}
   - (f_{n_2} - f_{n_1}) K_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}} (\omega)
   \right]
   \chi_{\begin{array}{l} n_5 n_6 \\ n_3 n_4  \end{array}} (\omega)
   = f_{n_2} - f_{n_1}    

By using a static interaction kernel `K(\omega=0)`, an effective frequency-indendepnt 
two particle Hamiltonian is defined as: 

.. math::

   \mathcal{H}_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}} 
   \equiv  (\epsilon_{n_2} - \epsilon_{n_1}) \delta_{n_1 n_5} \delta_{n_2 n_6}
   - (f_{n_2} - f_{n_1}) K_{\begin{array}{l} n_1 n_2 \\ n_5 n_6  \end{array}}

Inserting the above effective Hamiltonian into Eq. :eq:`chi_rewrite`, one can then write 

.. math::

   \chi_{\begin{array}{l} n_1 n_2 \\ n_3 n_4  \end{array}} = 
   \left[ \mathcal{H} - I \omega \right]^{-1}_{\begin{array}{l} n_1 n_2 \\ n_3 n_4  \end{array}}
   (f_{n_2} - f_{n_1})

where `I` is an identity matrix that has the same size as `\mathcal{H}`. 

The spectral representation of the inverse two-particle Hamiltonian is 

.. math::

   \left[ \mathcal{H} - I \omega \right]^{-1}_{\begin{array}{l} n_1 n_2 \\ n_3 n_4  \end{array}}
   = \sum_{\lambda \lambda^{\prime}} 
   \frac{A^{n_1 n_2}_{\lambda} A^{n_3 n_4}_{\lambda^{\prime}} N^{-1}_{\lambda \lambda^{\prime}}}{E_{\lambda} - \omega}

with the eigenvalues `E_{\lambda}` and eigenvectors `A_{\lambda}` given by 

.. math::

   \mathcal{H} A_{\lambda} = E_{\lambda} A_{\lambda} 

and the overlap matrix `N_{\lambda \lambda^{\prime} }` defined by

.. math::

    N_{\lambda \lambda^{\prime}} \equiv 
    \sum_{n_1 n_2} [A_{\lambda}^{n_1 n_2}]^{\ast} A_{\lambda^{\prime}}^{n_1 n_2}

If the Hamiltonian `\mathcal{H}` is Hermitian, the eigenvectors `A_{\lambda}` are then orthogonal and 

.. math::

	N_{\lambda \lambda^{\prime}} = \delta_{\lambda \lambda^{\prime}}


.. [#Review] G. Onida, L. Reining and A. Rubio,
            Electronic excitations: density-functional versus many-body Green's-function approaches,
            *Rev. Mod. Phys.* **74**, 601 (2002)

