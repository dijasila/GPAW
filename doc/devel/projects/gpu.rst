===
GPU
===

Aalto Effort
==========

(Samuli Hakala, Ville Havu, Jussi Enkovaara (CSC) )
We have been implementing the most performance critical C-kernels
in the finite-difference mode to utilize GPUs. Implementation is done
using CUDA and cuBLAS libraries when possible, Python interface to CUDA,
PyCUDA is also utilized. Code supports using multiple GPUs with MPI. 
First tests indicate promising speedups compared
to CPU-only code, and we are hoping to test larger systems (where
the benefits are expected to be larger) soon. Currently, we are extending the
CUDA implementation to real-time TDDFT.

Code is not in full production level yet.

Stanford/SUNCAT Effort
======================

(Lin Li, Jun Yan, Christopher O'Grady)

We believe that GPAW has two areas where significant improvement would
be very helpful: ease-of-convergence and performance.

We think the first of those is going to be significantly improved (for
small systems that are largely of interest to SUNCAT) by the addition
of the planewave-basis mode.  To complement the GPU grid-mode work of
Samuli Hakala et. al. at Aalto University we are seeing if it is possible to
significantly improve the planewave performance (and
performance-per-dollar) of GPAW using GPUs.  We are also using GPUs to
see if the Random Phase Approximation performance can be improved.
