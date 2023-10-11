import numpy as np

def sampling_branches(w_dist, ps='2l', ϖ=1, d=[1e-5, 0.1]):
    if len(w_dist) == 1:  # the value of ps is irrelevant in the case of a single pole
        assert d[0] >= 0
        assert ps == '2l'
        w_grid = np.array([w_dist + 1j*d[0], w_dist + 1j*ϖ], dtype=complex)
        return w_grid

    if ps == '1l':  # only one branch
        assert ϖ >= 0
        w_grid = np.array(w_dist + 1j*ϖ, dtype=complex)
        return w_grid

    if ps == '2l':  # two branches
        assert d[0] >= 0
        assert d[1] >= 0
        assert ϖ > d[0] and ϖ > d[1]
        w_grid = np.concatenate((np.array([w_dist[0] + 1j*d[0]]), w_dist[1:] + 1j*d[1], w_dist + 1j*ϖ))
        return w_grid

    raise ValueError(f'Unknown ps: {ps}.')

def frequency_distribution(npol, wrange, alpha=1):

    if npol == 1:
        w_grid = np.array([0.])
        return w_grid

    assert(wrange[0].real >= 0)
    if not wrange[1].real > wrange[0].real:
        raise ValueError('Frequency range inverted.')

    if alpha == 0:  # homogeneous distribution
        w_grid = np.linspace(wrange[0], wrange[1], 2*npol)
        return w_grid

   
    ws = wrange[1] - wrange[0]
    wgrid = semi_homogenous_partition(npol)**alpha * ws + wrange[0]
    #partition = np.ones(npol)
    #for i in range(1, npol+1):
    #    partition[i] = pivot_slice(i, 1/3)  # semi-homogeneous partition
    #wgrid = wrange[0] + ws*partition**alpha
    return wgrid

def mpa_frequency_sampling_new(npol, w0, d, ps='2l', alpha=1):
    w0 = np.array(w0)
    grid_p = frequency_distribution(npol, w0.real, alpha)
    grid_w = sampling_branches(grid_p, ps=ps,ϖ = w0[0].imag, d=d) 
    return grid_w


"""

def pivot_slice(wrange, poles_remaining, pivot=1/2):
    
    assert(0 < pivot < 1)

    

    pivot_slice = np.array([0, pivot, 1])
    #pivot_slice(1, pivot) = 0
    #pivot_slice(2, pivot) = 1
    #pivot_slice(3, pivot) = pivot
    if(npol % 2 == 0):
        pivot_slice(npol, pivot) = (pivot_slice(npol-1, pivot) + pivot_slice(npol-3, pivot))/2
    else:
        pivot_slice(npol, pivot) = (pivot_slice(npol-2, pivot) + pivot_slice(npol-3, pivot))/2
    pivot_slice(2, pivot) = 1


    assert(npol > 1)
    lp = int(np.log(npol - 1) / np.log(2))
    r = int((npol - 1) % (2**lp))
   """    


def mpa_frequency_sampling(npol, w0, d, ps='2l', alpha=1):
    """
    This function creates a frequency grid in the complex plane.
    The grid can have 1 or 2 branches with points non homogeneously
    distributed along the real frequency axis.
    See Fig. 1 and Eq. (18) of Ref. [1], and Eq. (11) of Ref. [2]
    [1] DA. Leon et al, PRB 104, 115157 (2021)
    [2] DA. Leon et al, PRB 107, 155130 (2023)

    Parameters
    ----------
    npol : numper of poles (half the number of frequency points)
    w0 : array of two complex numbers defining the sampling range
    d : array of two real numbers defining the damping range
    ps : string of length 2 defining a sampling with 1 or 2 lines
    alpha : exponent of the distribution of points along the real axis
    ______________________________________________________________________
                  Example: double parallel sampling with 9 poles
    ----------------------------------------------------------------------
                            complex frequency plane
    ----------------------------------------------------------------------
    |(w0[0].real, w0[0].imag) .. . . . . . .   . (w0[1].real, w0[1].imag)|
    |                                                                    |
    |     (w0[0].real, d[0])  .. . . . . . .   . (w0[1].real, d[1])      |
    ______________________________________________________________________
    """
 
    w0 = np.array(w0)
    assert (w0.real >= 0).all()
    assert (w0.imag >= 0).all()
    assert w0[1].real >= w0[0].real # positive interval

    if npol == 1:
        w_grid = np.array(w0, dtype=complex)
        return w_grid

    if ps == '1l':  # DALV: We could use a match-case function
        if alpha == 0:
            return np.linspace(w0[0], w0[1], 2 * npol)
        raise ValueError("If ps = '1l', only alpha = 0 is implemented")

    if ps == '2l':
        assert (w0.imag > d).all() # two branches
        if alpha == 0: # homogeneous distribution
            w_grid = np.concatenate((np.linspace(complex(np.real(w0[0]),
                                     d[1]), complex(np.real(w0[1]), d[1]),
                                    npol), np.linspace(w0[0], w0[1],
                                                       npol)))
            w_grid[0] = complex(np.real(w0[0]), d[0])
            return w_grid
        # inhomogeneous distribution in powers of 1/2^alpha
        ws = w0[1] - w0[0] # complex length
        w_grid = np.ones(2 * npol, dtype=complex) # number of freqs = 2 npol
        w_grid[0] = complex(np.real(w0[0]), d[0])
        w_grid[npol - 1] = complex(np.real(w0[1]), d[1])
        w_grid[npol] = w0[0]
        w_grid[2 * npol - 1] = w0[1]
        #lp = int(np.log2(npol - 1))
        lp = int(np.log(npol - 1) / np.log(2))

        r = int((npol - 1) % (2**lp))

        if r > 0:
            for i in range(1, 2 * r):
                w_grid[npol + i] = w0[0] + ws * (i / 2.**(lp + 1)
                                                 )**alpha
                w_grid[i] = complex(np.real(w_grid[npol + i]), d[1])
            for i in range(2 * r, npol):
                w_grid[npol + i] = w0[0] + ws * ((i - r) / 2.**(lp)
                                                 )**alpha
                w_grid[i] = complex(np.real(w_grid[npol + i]), d[1])
        else:
            w_grid[npol + 1] = w0[0] + ws / (2.**(lp + 1))**alpha
            w_grid[1] = complex(np.real(w_grid[npol + 1]), d[1])
            for i in range(2 * r + 2, npol - 1):
                w_grid[npol + i] = w0[0] + ws * ((i - 1 - r) / 2.**(lp)
                                                 )**alpha
                w_grid[i] = complex(np.real(w_grid[npol + i]), d[1])

        # Round up to the next power of 2. This will determine the minimum grid spacing
        dw = 1 / 2**np.ceil(np.log2(npol))
        print('dw', dw)
        dw_n = np.zeros(npol)

        data = { 1: (0,0,0), 

                 2: (1,0,0),

                 3: (0,2,0),
                 4: (2,2,0),

                 5: (2,1,1), # XXX
                 6: (2,3,0), 
                 7: (4,2,0),
                 8: (6,1,0),

                 9: (2,5,1), # XXX
                 10:(2,7,0),
                 11:(4,6,0),
                 12:(6,5,0),
                 13:(8,4,0),
                 14:(10,3,0),
                 15:(12,2,0),
                 16:(14,1,0),   # (npol - floorpow(npol) - 1)*2 

                 17:(2,13,1)
                 }

        lp = int(np.log(npol - 1) / np.log(2))
        if npol != 2**lp + 1: 
            ndw1 = (npol-(2**lp+1))*2
            ndw3 = 0
        else:
            ndw1 = 2
            ndw3 = 1
        ndw2 = npol - 1 - ndw1 - ndw3

        #ndw1, ndw2, ndw3 = data[npol]
        ndw2, ndw3 = ndw2 + ndw1, ndw1 + ndw2 + ndw3
        dw_n[1:ndw1+1] = 1
        dw_n[ndw1+1:ndw2+1] = 2
        dw_n[ndw2+1:ndw3+1] = 4
        print('dw_n', dw_n)
        w_grid2 = np.cumsum(dw_n) * dw
        print('w_grid reference', w_grid, 'new grid', w_grid2)
        assert np.allclose(w_grid[:npol].real, w_grid2.real)
        return w_grid
    raise ValueError("Only '1l' or '2l' values are implemented")

def semi_homogenous_partition(npol):
    """
    Returns a semi-homogenous partition with n-poles between 0 and 1
    according to 
       DA Leon, C Cardoso, T Chiarotti, D Varsano, E Molinari, A Ferretti
       Physical Review B 104 (11), 115157
    """
    small_cases = { 1: np.array([0.0]),
                    2: np.array([0.0, 1.0]),
                    3: np.array([0.0, 0.5, 1.0])}
    if npol < 4:
        return small_cases[npol]
    # Calculate the grid spacing
    # Round up to the next power of 2. This will determine the minimum grid spacing
    dw = 1 / 2**np.ceil(np.log2(npol))
    dw_n = np.zeros(npol)

    # Get the previous power of two, by searching down, 
    # e.g. lp(4) = 2, lp(7)=4, lp(8)=4, lp(9)=8 
    lp = 2**int(np.log2(npol - 1))

    # There are usually 2 kinds of intervals in a semi homogenous grid, they are always in
    # order such that smallest intervals are closer to zero.
    # The interval sizes are dw, 2*dw.
    # There is an exception to this rule when npol == power of two + 1,
    # because then there would be only one type of interval with the rule below.
    # To keep the grid inhomogenous, one adds the third interval 4 * dw.
    if npol == lp + 1: 
        np1 = 2
        np3 = 1
    else:
        np1 = (npol - (lp + 1))*2
        np3 = 0
    # The number of intervals is always one less, than the number of points in the grid
    # Therefore, We can deduce np2 from np1 and np3.
    np2 = npol - 1 - np1 - np3

    # Build up the intervals onto an array
    dw_n = np.repeat([0, 1, 2, 4], [1, np1, np2, np3])

    # Sum over the intervals to build the point grid
    w_grid = np.cumsum(dw_n) * dw
    return w_grid

#grid_w = mpa_frequency_sampling_new(4, [1j, 1.+1j], [0.01, 0.1], ps='2l', alpha=1)
#print(grid_w)
#asd

#for n in range(4, 20):
#    wgrid = mpa_frequency_sampling(n, [0.0+1j, 1.0+1j], [0.1,0.1], ps='2l', alpha=1)
#    print('DATA', n, wgrid)
#    wgrid2 = semi_homogenous_partition(n)
#    print('FROM PARTITION', wgrid2)
#
#    assert np.allclose(wgrid[:n].real, wgrid2)
#
#
