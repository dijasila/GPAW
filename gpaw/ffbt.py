from math import factorial as fac
import numpy as np

from gpaw import debug
from gpaw.spline import Spline


def generate_bessel_coefficients(lmax):
    """Generate spherical Bessel function expansion coefficients.

    The coefficients c_lm (see expansion below) can be generated from the
    recurrence relation [see also master thesis of Marco Vanin (2008)]

                        2l+1
    j   (x) + j   (x) = ‾‾‾‾ j (x)    for l≥1
     l-1       l+1       x    l

    which implies that

    c     = (2l+1) c      for l≥1 and m∊{0,1}
     l+1m           lm

    while

    c     = (2l+1) c   - c          for l≥1 and m≥2
     l+1m           lm    l-1m-2

    With c_00 = i, c_10 = i and c_11=-1, it can be proven that the expansion
    has the closed-form solution:

           1+m     (2l-m)!
    c   = i    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
     lm        2^(l-m)(l-m)!m!
    """
    c_lm = []
    for l in range(lmax + 1):
        c_m = []
        for m in range(l + 1):
            c_m.append((1.0j)**(1 + m) * fac(2 * l - m)
                       / (2**(l - m) * fac(l - m) * fac(m)))
        c_lm.append(c_m)
    return c_lm


c_lm = generate_bessel_coefficients(lmax=6)


def spherical_bessel(l, x_g):
    r"""Calculate the spherical Bessel function j_l(x).

    Evaluates the spherical Bessel function via the expansion [see master
    thesis of Marco Vanin (2008)]

                      l
                      __
                1     \
    j_l(x) = ‾‾‾‾‾‾‾  /  Re{ c_lm x^m e^(-ix) }
             x^(l+1)  ‾‾
                      m=0

    for non-negative real-valued x.
    """
    assert x_g.dtype == float and np.all(x_g >= 0)
    jl_g = np.zeros_like(x_g)

    # Mask out x = 0
    x0_g = x_g < 1e-10
    # j_0(x=0) = 1, j_l(x=0)=0 for l>0
    if l == 0:
        jl_g[x0_g] = 1.

    # Evaluate j_l(x) using the coefficients
    xpos_g = x_g[~x0_g]
    for m in range(l + 1):
        jl_g[~x0_g] += (c_lm[l][m] * xpos_g**m * np.exp(-1.j * xpos_g)).real
    jl_g[~x0_g] /= xpos_g**(l + 1)

    return jl_g


def ffbt(l, f_g, r_g, k_q):
    r"""Fast Fourier-Bessel transform.

    The following integral is calculated using l+1 FFTs::

                    oo
                   /
              l+1 |  2           l
      g(k) = k    | r dr j (kr) r f (r)
                  |       l
                 /
                  0

    Assuming that f(r)=0 ∀ r≥rc for some rc>0, the integral can be rewritten as
    follows using the coefficient expansion of j_l(x) given above:

           l
           __      /  rc                       \
           \   m   |  /      m+1       -ikr    |
    g(k) = /  k  Re<  | c   r    f(r) e     dr >
           ‾‾      |  /  lm                    |
           m=0     \  0                        /

    With a uniform radial grid,

    r(g) = g rc / N    for g=0,1,...,Q-1

    the inner integral is evaluated using an FFT,

    Q-1
    __
    \   ⎧     m+1      ⎫|        -2πiqg/Q
    /   |c   r    f(r) ||       e         Δr
    ‾‾  ⎩ lm           ⎭|
    g=0                  r=r(g)

    where Δr=rc/N is the input real-space grid spacing and Q≥2N to leave some
    zero padding of f(r).

    We evaluate the integral for q=0,1,...,Q-1, but return only the first Q/2
    frequencies, corresponding to the "positive" frequencies:

            2π
    k(q) = ‾‾‾‾ q    for q=0,1,...,Q/2-1
           Q Δr
    """

    dr = r_g[1]
    Nq = len(k_q)

    if debug:
        # We assume a uniform real-space grid from r=0 to r=rc-Δr
        assert r_g[0] == 0.
        assert np.allclose(r_g[1:] - r_g[:-1], dr)
        # We assume a uniform reciprocal-space grid, with a specific grid
        # spacing in relation to Δr, starting from k=0 and including all the
        # "positive frequencies"
        assert k_q[0] == 0.
        assert Nq >= len(r_g)
        dk = np.pi / (Nq * dr)
        assert np.allclose(k_q[1:] - k_q[:-1], dk)

    # Perform the transform
    g_q = np.zeros(Nq)
    for m in range(l + 1):
        g_q += (k_q**m *
                # FFT with Q=2Nq k-points (why?)
                np.fft.fft(c_lm[l][m] * r_g**(m + 1) * f_g, 2 * Nq)
                # Take the real part of the Q/2 positive frequency points
                [:Nq].real) * dr
    return g_q


def rescaled_bessel_limit(l):
    """Get the x->0 limit of a rescaled spherical bessel function.

    Calculates the closed form solution to the limit

         j_l(x)   2^l l!
    lim  ‾‾‾‾‾‾ = ‾‾‾‾‾‾‾
    x->0  x^l     (2l+1)!
    """
    return 2**l * fac(l) / fac(2 * l + 1)


class FourierBesselTransformer:
    def __init__(self, rcmax, ng):
        """Construct the transformer with ffbt-compliant grids."""
        self.ng = ng
        self.rcmax = rcmax
        self.dr = rcmax / self.ng
        self.r_g = np.arange(self.ng) * self.dr
        # Positive frequency grid
        Nq = 2 * self.ng
        self.dk = np.pi / (Nq * self.dr)
        self.k_q = np.arange(Nq) * self.dk
        # Number of positive and negative frequency grid points
        self.Q = 2 * Nq

    def transform(self, spline):
        """Fourier-Bessel transform a given radial function f(r).

        Calculates
                 rc
                 /
        f(k) = k | r^2 dr (kr)^l j_l(kr) f(r)
                 /
                 0
        """
        assert spline.get_cutoff() <= self.rcmax, (spline.get_cutoff(),
                                                   self.rcmax)
        l = spline.get_angular_momentum_number()
        f_g = spline.map(self.r_g)
        f_q = ffbt(l, f_g, self.r_g, self.k_q)
        return f_q

    def rescaled_transform(self, spline):
        """Perform a rescaled Fourier-Bessel transform of f(r).

        Calculates

                  rc
        ˷         /         r^l
        f(k) = 4π | r^2 dr  ‾‾‾ j_l(kr) f(r)
                  /         k^l
                  0
        """
        # Calculate f(k) and rescale for finite k
        l = spline.get_angular_momentum_number()
        f_q = self.transform(spline)
        f_q[1:] *= 4 * np.pi / self.k_q[1:]**(2 * l + 1)

        # Calculate k=0 contribution
        f_q[0] = self.calculate_rescaled_average(spline)

        return f_q

    def calculate_rescaled_average(self, spline):
        """Calculate the rescaled transform for k=0.

        Calculates

                    rc
        ˷           /                     j_l(kr)
        f(k=0) = 4π | r^2 dr r^(2l)  lim  ‾‾‾‾‾‾‾ f(r)
                    /               kr->0 (kr)^l
                    0
        """
        l = spline.get_angular_momentum_number()
        f_g = spline.map(self.r_g)
        prefactor = 4 * np.pi * rescaled_bessel_limit(l) * self.dr
        return prefactor * np.dot(self.r_g**(2 * l + 2), f_g)


def rescaled_fbt(spline, N=2**10):
    """Calculate rescaled Fourier-Bessel transform in spline representation."""
    # Fourier transform the spline, sampling it on a uniform grid
    rcut = 50.0  # Why not spline.get_cutoff() * 2 or similar?
    assert spline.get_cutoff() <= rcut
    transformer = FourierBesselTransformer(rcut, N)
    f_q = transformer.rescaled_transform(spline)

    # Return spline representation of the transform
    l = spline.get_angular_momentum_number()
    kmax = transformer.k_q[-1]
    return Spline(l, kmax, f_q)
