# Simple test case for new BLACS/ScaLAPACK
# We compare the results here to parallel/bandpar3.py
# with analogous parameters.
#
# Note that GPAW does not do transpose for calls
# to LAPACK involving operations on symmetric matrices. 
# Hence, UPLO = 'U' in Fortran equals UPLO = 'L' in
# NumPy C-style. HEre, A and B arrays here have elements
# in lower half, while their counterparts in the new 
# BLACS/ScaLAPACK implementation will have elements in
# the upper half of the distributed matrix.
import numpy as np
from gpaw.utilities.lapack import diagonalize, inverse_cholesky

N = 16;
A = np.empty((N,N))
A[:,0:4] = 0.0*np.eye(N,4,0)
A[:,0:4] = A[:,0:4]+ 0.1*np.eye(N,4,-1) # -1
A[:,4:8] = 1.0*np.eye(N,4,-4)
A[:,4:8] = A[:,4:8] + 0.1*np.eye(N,4,-5) # -1
A[:,8:12] = 2.0*np.eye(N,4,-8)
A[:,8:12] = A[:,8:12] + 0.1*np.eye(N,4,-9) # -1 
A[:,12:16] = 3.0*np.eye(N,4,-12)
A[:,12:16] = A[:,12:16]+ 0.1*np.eye(N,4,-13) # -1
print "Hamiltonian =", A

B = np.empty((N,N))
B[:,0:4] = 1.0*np.eye(N,4,0)
B[:,0:4] = B[:,0:4]+ 0.2*np.eye(N,4,-1) # -1
B[:,4:8] = 1.0*np.eye(N,4,-4)
B[:,4:8] = B[:,4:8] + 0.2*np.eye(N,4,-5) # -1
B[:,8:12] = 1.0*np.eye(N,4,-8)
B[:,8:12] = B[:,8:12] + 0.2*np.eye(N,4,-9) # -1 
B[:,12:16] = 1.0*np.eye(N,4,-12)
B[:,12:16] = B[:,12:16] + 0.2*np.eye(N,4,-13) # -1
print "Overlap = ", B

w = np.empty(N)

info = diagonalize(A, w)

if info != 0:
    print "WARNING: diagonalize info=", info
print "lambda", w
print "eigenvectors", A


info = diagonalize(A, w, B)

if info != 0:
    print "WARNING: general diagonalize info = print", info
print "lambda", w
print "eigenvectors", A

info = inverse_cholesky(B)

if info != 0:
    print "WARNING: general diagonalize info = print", info
print "overlap", B
