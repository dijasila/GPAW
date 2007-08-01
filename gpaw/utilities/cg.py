from Numeric import reshape, vdot

def CG(A, X, B, maxiter=20, tolerance=1.0e-10, verbose=False):
    """Solve X*A=B using conjugate gradient method.

    ``X`` and ``B`` are ``ndarrays```of shape ``(m, nx, ny, nz)``
    coresponding to matrices of size ``m*n`` (``m=nx*ny*nz``) and
    ``A`` is a callable representing an ``n*n`` matrix (``A(X)=X*A``).
    On return we have ``A(X)==B`` within ``tolerance``."""

    m = len(X)
    shape = (m, 1, 1, 1)
    R = B - A(X)
    P = R.copy()
    c1 = reshape([vdot(r, r) for r in R], shape)
    for i in range(maxiter):
        error = sum(c1.flat)
        if verbose:
            print 'CG-%03d: %e' % (i, error)
        if error < tolerance:
            return i, error
        Q = A(P)
        alpha = c1 / reshape([vdot(p, q) for p, q in zip(P, Q)], shape)
        X += alpha * P
        R -= alpha * Q
        c0 = c1
        c1 = reshape([vdot(r, r) for r in R], shape)
        beta = c1 / c0
        P *= beta
        P += R
        
    raise ArithmeticError('Did not converge!')
