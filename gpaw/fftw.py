"""Python wrapper for FFTW3 library."""

import os

import numpy as np

import _gpaw


ESTIMATE = 64
MEASURE = 0
PATIENT = 32
EXHAUSTIVE = 8


if os.environ.get('GPAW_FFTWSO'):
    import warnings
    warnings.warn('GPAW_FFTWSO is set to "{}"; ignoring.  '
                  'Please use customize.py to link FFTW instead.'
                  .format(os.environ['GPAW_FFTWSO']))


def have_fftw():
    return hasattr(_gpaw, 'FFTWPlan')


def check_fft_size(n):
    """Check if n is an efficient fft size.

    Efficient means that n can be factored into small primes (2, 3, 5, 7)."""

    if n == 1:
        return True
    for x in [2, 3, 5, 7]:
        if n % x == 0:
            return check_fft_size(n // x)
    return False


def get_efficient_fft_size(N, n=1):
    """Return smallest efficient fft size.

    Must be greater than or equal to N and divisible by n.
    """
    N = -(-N // n) * n
    while not check_fft_size(N):
        N += n
    return N


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


class FFTWPlan:
    """FFTW3 3d transform."""
    def __init__(self, in_R, out_R, sign, flags=MEASURE):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')

        check_fftw_inputs(in_R, out_R)

        self._ptr = _gpaw.FFTWPlan(in_R, out_R, sign, flags)
        self.in_R = in_R
        self.out_R = out_R
        self.sign = sign
        self.flags = flags

    def execute(self):
        _gpaw.FFTWExecute(self._ptr)

    def __del__(self):
        if getattr(self, '_ptr', None):
            _gpaw.FFTWDestroy(self._ptr)
        self._ptr = None


class NumpyFFTPlan:
    """Numpy fallback."""
    def __init__(self, in_R, out_R, sign, flags=None):
        check_fftw_inputs(in_R, out_R)
        self.in_R = in_R
        self.out_R = out_R
        self.sign = sign

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


def empty(shape, dtype=float):
    """numpy.empty() equivalent with 16 byte alignment."""
    assert dtype == complex
    N = np.prod(shape)
    a = np.empty(2 * N + 1)
    offset = (a.ctypes.data % 16) // 8
    a = a[offset:2 * N + offset].view(complex)
    a.shape = shape
    return a


if have_fftw():
    FFTPlan = FFTWPlan
else:
    FFTPlan = NumpyFFTPlan
