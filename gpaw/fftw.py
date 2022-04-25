"""
Python wrapper for FFTW3 library
================================

.. autoclass:: FFTPlans

"""
from __future__ import annotations

import numpy as np
from scipy.fft import fftn, ifftn, irfftn, rfftn

import _gpaw
from gpaw.typing import Array3D, DTypeLike, IntVector

ESTIMATE = 64
MEASURE = 0
PATIENT = 32
EXHAUSTIVE = 8


def have_fftw() -> bool:
    """Did we compile with FFTW?"""
    return hasattr(_gpaw, 'FFTWPlan')


def check_fft_size(n: int, factors=[2, 3, 5, 7]) -> bool:
    """Check if n is an efficient fft size.

    Efficient means that n can be factored into small primes (2, 3, 5, 7).

    >>> check_fft_size(17)
    False
    >>> check_fft_size(18)
    True
    """

    if n == 1:
        return True
    for x in factors:
        if n % x == 0:
            return check_fft_size(n // x, factors)
    return False


def get_efficient_fft_size(N: int, n=1, factors=[2, 3, 5, 7]) -> int:
    """Return smallest efficient fft size.

    Must be greater than or equal to N and divisible by n.

    >>> get_efficient_fft_size(17)
    18
    """
    N = -(-N // n) * n
    while not check_fft_size(N, factors):
        N += n
    return N


def empty(shape, dtype=float):
    """numpy.empty() equivalent with 16 byte alignment."""
    assert dtype == complex
    N = np.prod(shape)
    a = np.empty(2 * N + 1)
    offset = (a.ctypes.data % 16) // 8
    a = a[offset:2 * N + offset].view(complex)
    a.shape = shape
    return a


def create_plans(size_c: IntVector,
                 dtype: DTypeLike,
                 flags: int = MEASURE) -> FFTPlans:
    """Create plan-objects for FFT and inverse FFT."""
    if have_fftw():
        return FFTWPlans(size_c, dtype, flags)
    return NumpyFFTPlans(size_c, dtype)


class FFTPlans:
    def __init__(self,
                 size_c: IntVector,
                 dtype: DTypeLike):
        if dtype == float:
            rsize_c = (size_c[0], size_c[1], size_c[2] // 2 + 1)
            self.tmp_Q = empty(rsize_c, complex)
            self.tmp_R = self.tmp_Q.view(float)[:, :, :size_c[2]]
        else:
            self.tmp_Q = empty(size_c, complex)
            self.tmp_R = self.tmp_Q

    def fft(self) -> None:
        """Do FFT from ``tmp_R`` to ``tmp_Q``.

        >>> plans = create_plans([4, 1, 1], float)
        >>> plans.tmp_R[:, 0, 0] = [1, 0, 1, 0]
        >>> plans.fft()
        >>> plans.tmp_Q[:, 0, 0]
        array([2.+0.j, 0.+0.j, 2.+0.j, 0.+0.j])
        """
        raise NotImplementedError

    def ifft(self) -> None:
        """Do inverse FFT from ``tmp_Q`` to ``tmp_R``.

        >>> plans = create_plans([4, 1, 1], complex)
        >>> plans.tmp_Q[:, 0, 0] = [0, 1j, 0, 0]
        >>> plans.ifft()
        >>> plans.tmp_R[:, 0, 0]
        array([ 0.+1.j, -1.+0.j,  0.-1.j,  1.+0.j])
        """
        raise NotImplementedError


class FFTWPlans(FFTPlans):
    """FFTW3 3d transforms."""
    def __init__(self, size_c, dtype, flags=MEASURE):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')
        super().__init__(size_c, dtype)
        self._fftplan = _gpaw.FFTWPlan(self.tmp_R, self.tmp_Q, -1, flags)
        self._ifftplan = _gpaw.FFTWPlan(self.tmp_Q, self.tmp_R, 1, flags)

    def fft(self):
        _gpaw.FFTWExecute(self._fftplan)

    def ifft(self):
        _gpaw.FFTWExecute(self._ifftplan)

    def __del__(self):
        _gpaw.FFTWDestroy(self._fftplan)
        _gpaw.FFTWDestroy(self._ifftplan)


class NumpyFFTPlans(FFTPlans):
    """Numpy fallback."""
    def fft(self):
        if self.tmp_R.dtype == float:
            self.tmp_Q[:] = rfftn(self.tmp_R, overwrite_x=True)
        else:
            self.tmp_Q[:] = fftn(self.tmp_R, overwrite_x=True)

    def ifft(self):
        if self.tmp_R.dtype == float:
            self.tmp_R[:] = irfftn(self.tmp_Q, self.tmp_R.shape,
                                   norm='forward', overwrite_x=True)
        else:
            self.tmp_R[:] = ifftn(self.tmp_Q, self.tmp_R.shape,
                                  norm='forward', overwrite_x=True)


# The rest of this file will be removed in the future ...

def check_fftw_inputs(in_R, out_R):
    for arr in in_R, out_R:
        # Note: Arrays not necessarily contiguous due to 16-byte alignment
        assert arr.ndim == 3  # We can perhaps relax this requirement
        assert arr.dtype == float or arr.dtype == complex

    if in_R.dtype == out_R.dtype == complex:
        assert in_R.shape == out_R.shape
    else:
        # One real and one complex:
        R, C = (in_R, out_R) if in_R.dtype == float else (out_R, in_R)
        assert C.dtype == complex
        assert R.shape[:2] == C.shape[:2]
        assert C.shape[2] == 1 + R.shape[2] // 2


class FFTPlan:
    """FFT 3d transform."""
    def __init__(self,
                 in_R: Array3D,
                 out_R: Array3D,
                 sign: int,
                 flags: int = MEASURE):
        check_fftw_inputs(in_R, out_R)
        self.in_R = in_R
        self.out_R = out_R
        self.sign = sign
        self.flags = flags

    def execute(self) -> None:
        raise NotImplementedError


class FFTWPlan(FFTPlan):
    """FFTW3 3d transform."""
    def __init__(self, in_R, out_R, sign, flags=MEASURE):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')
        self._ptr = _gpaw.FFTWPlan(in_R, out_R, sign, flags)
        FFTPlan.__init__(self, in_R, out_R, sign, flags)

    def execute(self):
        _gpaw.FFTWExecute(self._ptr)

    def __del__(self):
        if getattr(self, '_ptr', None) and _gpaw is not None:
            _gpaw.FFTWDestroy(self._ptr)
        self._ptr = None


class NumpyFFTPlan(FFTPlan):
    """Numpy fallback."""
    def execute(self):
        if self.in_R.dtype == float:
            self.out_R[:] = np.fft.rfftn(self.in_R)
        elif self.out_R.dtype == float:
            self.out_R[:] = np.fft.irfftn(self.in_R, self.out_R.shape)
            self.out_R *= self.out_R.size
        elif self.sign == 1:
            self.out_R[:] = np.fft.ifftn(self.in_R, self.out_R.shape)
            self.out_R *= self.out_R.size
        else:
            self.out_R[:] = np.fft.fftn(self.in_R)


def create_plan(in_R: Array3D,
                out_R: Array3D,
                sign: int,
                flags: int = MEASURE) -> FFTPlan:
    if have_fftw():
        return FFTWPlan(in_R, out_R, sign, flags)
    return NumpyFFTPlan(in_R, out_R, sign, flags)
