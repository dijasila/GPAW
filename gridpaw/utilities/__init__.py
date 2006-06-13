# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Utility functions and classes."""

from math import sqrt

import Numeric as num

import _gridpaw
from gridpaw import debug


# Error function:
erf = _gridpaw.erf


# Factorials:
fac = [1, 1, 2, 6, 24, 120, 720, 5040, 40320,
       362880, 3628800, 39916800, 479001600]


def contiguous(array, typecode):
    """Convert a sequence to a contiguous Numeric array."""
    array = num.asarray(array, typecode)
    if array.iscontiguous():
        return array
    else:
        return num.array(array)


def is_contiguous(array, typecode=None):
    """Check for contiguity and type."""
    if typecode is None:
        return array.iscontiguous()
    else:
        return array.iscontiguous() and array.typecode() == typecode


# Radial-grid Hartree solver:
#
#                       l
#             __  __   r
#     1      \   4||    <   * ^    ^
#   ------ =  )  ---- ---- Y (r)Y (r'),
#    _ _     /__ 2l+1  l+1  lm   lm
#   |r-r'|    lm      r
#                      >
# where
#
#   r = min(r, r')
#    <
#
# and
#
#   r = max(r, r')
#    >
#
if debug:
    def hartree(l, nrdr, beta, N, vr):
        """Calculates radial Coulomb integral.

        The following integral is calculated::
        
                                      ^
                             n (r')Y (r')
                  ^    / _    l     lm
          v (r)Y (r) = |dr --------------,
           l    lm     /       _   _
                              |r - r'|

        where input and output arrays `nrdr` and `vr`::

                  dr
          n (r) r --  and  v (r) r,
           l      dg        l

        are defined on radial grids as::

              beta g
          r = ------,  g = 0, 1, ..., N - 1.
              N - g

        """
        assert is_contiguous(nrdr, num.Float)
        assert is_contiguous(vr, num.Float)
        assert nrdr.shape == vr.shape and len(vr.shape) == 1
        return _gridpaw.hartree(l, nrdr, beta, N, vr)
else:
    hartree = _gridpaw.hartree


def unpack(M):
    assert is_contiguous(M, num.Float)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = num.zeros((n, n), num.Float)
    _gridpaw.unpack(M, M2)
    return M2

    
def unpack2(M):
    assert is_contiguous(M, num.Float)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = num.zeros((n, n), num.Float)
    _gridpaw.unpack(M, M2)
    M2 *= 0.5
    M2.flat[0::n + 1] *= 2
    return M2

    
def pack(M2):
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = 2 * M2[r, c]
            assert abs(M2[r, c] - M2[c, r]) < 1e-10 # ?????
            p += 1
    assert p == len(M)
    return M


def check_unit_cell(cell):
    """Check that the unit cell (3*3 matrix) is orthorhombic (diagonal)."""
    c = cell.copy()
    # Zero the diagonal:
    c.flat[::4] = 0.0
    if num.sometrue(c.flat):
        raise RuntimeError('Unit cell not orthorhombic')
    

class DownTheDrain:
    """Definition of a stream that throws away all output."""
    
    def write(self, string):
        pass
    
    def flush(self):
        pass


"""
class OutputFilter:
    def __init__(self, out, threshold, level=500):
        self.threshold = threshold
        self.verbosity = verbosity

    def write(self, string):
        if kfdce

"""

def warning(msg):
    r"""Put string in a box.

    >>> print Warning('Watch your step!')
     /\/\/\/\/\/\/\/\/\/\/\
     \                    /
     /  WARNING:          \
     \  Watch your step!  /
     /                    \
     \/\/\/\/\/\/\/\/\/\/\/
    """
    
    n = len(msg)
    if n % 2 == 1:
        n += 1
        msg += ' '
    bar = (n / 2 + 3) * '/\\'
    space = (n / 2 + 2) * '  '
    format = ' %s\n \\%s/\n /  WARNING:%s\\\n \\  %s  /\n /%s\\\n %s/'
    return format % (bar, space, space[10:], msg, space, bar[1:])


def center(atoms):
    """Method for centering atoms in input ListOfAtoms"""
    pos = atoms.GetCartesianPositions()
    cntr = 0.5 * (num.minimum.reduce(pos) + num.maximum.reduce(pos))
    cell = num.diagonal(atoms.GetUnitCell())
    atoms.SetCartesianPositions(pos - cntr + 0.5 * cell)


# Function used by test-suite:
def equal(a, b, e=0):
    assert abs(a - b) <= e, '%f != %f (error: %f > %f)' % (a, b, abs(a - b), e)
