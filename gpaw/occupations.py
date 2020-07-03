"""Smearing functions and occupation number calculators."""

import warnings
from math import pi, nan, inf
from typing import List, Tuple, Optional, NamedTuple, Any, Callable

import numpy as np
from scipy.special import erf
from ase.units import Ha

from gpaw.band_descriptor import BandDescriptor
from gpaw.mpi import serial_comm, broadcast_float

MPICommunicator = Any  # typehint


class ParallelLayout(NamedTuple):
    """Collection of parallel stuff."""
    bd: Optional[BandDescriptor]
    kpt_comm: MPICommunicator
    domain_comm: MPICommunicator


def fermi_dirac(eig, fermi_level, width):
    x = (eig - fermi_level) / width
    x = np.clip(x, -100, 100)
    y = np.exp(x)
    z = y + 1.0
    f = 1.0 / z
    dfde = (f - f**2) / width
    y *= x
    y /= z
    y -= np.log(z)
    e_entropy = y * width
    return f, dfde, e_entropy


def marzari_vanderbilt(eig, fermi_level, width):
    x = (eig - fermi_level) / width
    expterm = np.exp(-(x + (1 / np.sqrt(2)))**2)
    f = expterm / np.sqrt(2 * np.pi) + 0.5 * (1 - erf(1. / np.sqrt(2) + x))
    dfde = expterm * (2 + np.sqrt(2) * x) / np.sqrt(np.pi) / width
    s = expterm * (1 + np.sqrt(2) * x) / (2 * np.sqrt(np.pi))
    e_entropy = -s * width
    return f, dfde, e_entropy


def methfessel_paxton(eig, fermi_level, width, order=0):
    x = (eig - fermi_level) / width
    f = 0.5 * (1 - erf(x))
    for i in range(order):
        f += (coff_function(i + 1) *
              hermite_poly(2 * i + 1, x) * np.exp(-x**2))
    dfde = 1 / np.sqrt(pi) * np.exp(-x**2)
    for i in range(order):
        dfde += (coff_function(i + 1) *
                 hermite_poly(2 * i + 2, x) * np.exp(-x**2))
    dfde *= 1.0 / width
    e_entropy = (0.5 * coff_function(order) *
                 hermite_poly(2 * order, x) * np.exp(-x**2))
    e_entropy = -e_entropy * width
    return f, dfde, e_entropy


def coff_function(n):
    return (-1)**n / (np.product(np.arange(1, n + 1)) *
                      4**n * np.sqrt(np.pi))


def hermite_poly(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return (2 * x * hermite_poly(n - 1, x) -
                2 * (n - 1) * hermite_poly(n - 2, x))


def create_occupation_number_object(width=None,
                                    name=None,
                                    fixmagmom=False,
                                    order=None):
    if width == 0.0:
        return ZeroWidth(fixmagmom=fixmagmom)
    if name == 'methfessel-paxton':
        return MethfesselPaxton(width, order=order, fixmagmom=fixmagmom)
    assert order is None
    if name == 'fermi-dirac':
        return FermiDirac(width, fixmagmom=fixmagmom)
    if name == 'marzari-vanderbilt':
        return MarzariVanderbilt(width, fixmagmom=fixmagmom)
    if name == 'orbital-free':
        return ThomasFermiOccupations()
    raise ValueError('Unknown occupation number object name: ' + name)


class OccupationNumbers:
    """Base class for all occupation number objects."""
    name = 'unknown'

    def __init__(self,
                 fixmagmom: bool = False):
        self.fixmagmom = fixmagmom
        self.fixed_magmom_value: Optional[float] = None

    def todict(self):
        return {'fixmagmom': self.fixmagmom}

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: List[List[float]],
                  weights: List[float],
                  parallel: ParallelLayout = None,
                  fermi_levels_guess: List[float] = None
                  ) -> Tuple[List[np.ndarray],
                             List[float],
                             float]:
        """Calculate occupation numbers from eigenvalues in eV.

        occ: dict
            Example: {'name': 'fermi-dirac', 'width': 0.05} (width in eV).
        eps_skn: ndarray, shape=(nspins, nibzkpts, nbands)
            Eigenvalues.
        weight_k: ndarray, shape=(nibzkpts,)
            Weights of k-points in IBZ (must sum to 1).
        nelectrons: int or float
            Number of electrons.

        Returns a tuple containing:

        * f_skn (sums to nelectrons)
        * fermi-level
        * magnetic moment
        * entropy as -S*T
        """

        eig_qn = [np.asarray(eig_n) for eig_n in eigenvalues]
        weight_q = np.asarray(weights)

        if parallel is None:
            parallel = ParallelLayout(None, serial_comm, serial_comm)

        if fermi_levels_guess is None:
            fermi_levels_guess = [nan] * (1 + int(self.fixmagmom))

        if self.fixmagmom:
            return self._fixed_magmom(
                nelectrons, eig_qn, weight_q, parallel, fermi_levels_guess)
        else:
            return self._free_magmom(
                nelectrons, eig_qn, weight_q, parallel, fermi_levels_guess[0])

    def _free_magmom(self, nelectrons, eig_qn, weight_q,
                     parallel, fermi_level_guess):
        domain_comm = parallel.domain_comm

        f_qn = np.empty((len(weight_q), len(eig_qn[0])))

        result = np.empty(2)

        if domain_comm.rank == 0:
            # Let the master domain do the work and broadcast results:
            result[:] = self._calculate(
                nelectrons, eig_qn, weight_q, f_qn,
                parallel, fermi_level_guess)

        domain_comm.broadcast(result, 0)

        for f_n in f_qn:
            domain_comm.broadcast(f_n, 0)

        fermi_level, e_entropy = result
        return f_qn, [fermi_level], e_entropy

    def _fixed_magmom(self,
                      nelectrons: float,
                      eig_qn: List[List[float]],
                      weight_q: List[float],
                      parallel,
                      fermi_levels_guess: List[float]
                      ) -> Tuple[List[np.ndarray],
                                 List[float],
                                 float]:
        magmom = self.fixed_magmom_value
        assert magmom is not None
        f1_qn, fermi_levels1, e_entropy1 = self._free_magmom(
            (nelectrons + magmom) / 2,
            eig_qn[::2],
            weight_q[::2],
            parallel,
            fermi_levels_guess[0])

        f2_qn, fermi_levels2, e_entropy2 = self._free_magmom(
            (nelectrons - magmom) / 2,
            eig_qn[1::2],
            weight_q[1::2],
            parallel,
            fermi_levels_guess[1])

        f_qn = []
        for f1_n, f2_n in zip(f1_qn, f2_qn):
            f_qn += [f1_n, f2_n]

        return (f_qn,
                fermi_levels1 + fermi_levels2,
                e_entropy1 + e_entropy2)


class SmoothDistribution(OccupationNumbers):
    """Base class for Fermi-Dirac and other smooth distributions."""
    def __init__(self, width, **kwargs):
        """Smooth distribution.

        Find the Fermi level by integrating in energy until
        the number of electrons is correct.

        width: float
            Width of distribution in eV.
        fixmagmom: bool
            Fix spin moment calculations.  A separate Fermi level for
            spin up and down electrons is found: self.fermilevel +
            self.split and self.fermilevel - self.split.
        """

        self.width = width / Ha
        OccupationNumbers.__init__(self, **kwargs)

    def todict(self):
        dct = OccupationNumbers.todict(self)
        dct['width'] = self.width * Ha
        return dct

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   parallel,
                   fermi_level_guess):

        if np.isnan(fermi_level_guess) or self.width == 0.0:
            zero = ZeroWidth()
            fermi_level_guess, _ = zero._calculate(
                nelectrons, eig_qn, weight_q, f_qn, parallel)
            if self.width == 0.0 or np.isinf(fermi_level_guess):
                return fermi_level_guess, 0.0

        x = fermi_level_guess

        data = np.empty(3)

        def func(x, data=data):
            data[:] = 0.0
            for eig_n, weight, f_n in zip(eig_qn, weight_q, f_qn):
                f_n[:], dfde_n, e_entropy_n = self.distribution(eig_n, x)
                data += [weight * x_n.sum()
                         for x_n in [f_n, dfde_n, e_entropy_n]]
            if parallel.bd is not None:
                parallel.bd.comm.sum(data)
            parallel.kpt_comm.sum(data)
            f, dfde = data[:2]
            df = f - nelectrons
            return df, dfde

        fermi_level, niter = findroot(func, x)

        e_entropy = data[2]

        return fermi_level, e_entropy


class FermiDirac(SmoothDistribution):
    name = 'fermi-dirac'
    extrapolate_factor = -0.5

    def distribution(self, eig_n, fermi_level):
        return fermi_dirac(eig_n, fermi_level, self.width)

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'fermi-dirac'
        return dct

    def __str__(self):
        return f'  Fermi-Dirac: width={self.width * Ha:.4f} eV\n'


class MarzariVanderbilt(SmoothDistribution):
    name = 'marzari-vanderbilt'
    # According to Nicola Marzari, one should not extrapolate M-V energies
    # https://lists.quantum-espresso.org/pipermail/users/2005-October/003170.html
    extrapolate_factor = 0.0

    def distribution(self, eig_n, fermi_level):
        return marzari_vanderbilt(eig_n, fermi_level, self.width)

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'marzari-vanderbilt'
        return dct

    def __str__(self):
        s = '  Marzari-Vanderbilt: width={0:.4f} eV\n'.format(
            self.width * Ha)
        return SmoothDistribution.__str__(self) + s


class MethfesselPaxton(SmoothDistribution):
    name = 'methfessel_paxton'

    def __init__(self, width, order=0):
        SmoothDistribution.__init__(self, width)
        self.order = order
        self.extrapolate_factor = -1.0 / (self.order + 2)

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'methfessel-paxton'
        dct['order'] = self.order
        return dct

    def __str__(self):
        s = '  Methfessel-Paxton: width={0:.4f} eV, order={1}\n'.format(
            self.width * Ha, self.order)
        return SmoothDistribution.__str__(self) + s

    def distribution(self, eig_n, fermi_level):
        return methfessel_paxton(eig_n, fermi_level, self.width, self.order)


def findroot(func: Callable[[float], Tuple[float, float]],
             x: float,
             tol: float = 1e-10) -> Tuple[float, int]:
    """Function used for locating Fermi level.

    The function should return a (value, derivative) tuple:

    >>> x, _ = findroot(lambda x: (x, 1.0), 1.0)
    >>> assert abs(x) < 1e-10
    """
    xmin = -np.inf
    xmax = np.inf

    # Try 10 step using the gradient:
    niter = 0
    while True:
        f, dfdx = func(x)
        if abs(f) < tol:
            return x, niter
        if f < 0.0 and x > xmin:
            xmin = x
        elif f > 0.0 and x < xmax:
            xmax = x
        dx = -f / max(dfdx, 1e-18)
        if niter == 10 or abs(dx) > 0.01 or not (xmin < x + dx < xmax):
            break  # try bisection
        x += dx
        niter += 1

    # Bracket the solution:
    if not np.isfinite(xmin):
        xmin = x
        fmin = f
        step = 0.01
        while fmin > tol:
            xmin -= step
            fmin = func(xmin)[0]
            step *= 2

    if not np.isfinite(xmax):
        xmax = x
        fmax = f
        step = 0.01
        while fmax < 0:
            xmax += step
            fmax = func(xmax)[0]
            step *= 2

    # Bisect:
    while True:
        x = (xmin + xmax) / 2
        f = func(x)[0]
        if abs(f) < tol:
            return x, niter
        if f > 0:
            xmax = x
        else:
            xmin = x
        niter += 1
        assert niter < 1000


class ZeroWidth(OccupationNumbers):
    name = 'zero-width'
    extrapolate_factor = 0.0
    width = 0.0

    def distribution(self, eig_n, fermi_level):
        f_n = np.zeros_like(eig_n)
        f_n[eig_n < fermi_level] = 1.0
        f_n[eig_n == fermi_level] = 0.5
        return f_n, np.zeros_like(eig_n), np.zeros_like(eig_n)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   parallel,
                   fermi_level_guess=nan):
        eig_kn, weight_k, nkpts_r = self._collect_eigelvalues(
            eig_qn, weight_q, parallel)

        if eig_kn is not None:
            f_kn = np.zeros_like(eig_kn)
            f_m = f_kn.ravel()
            w_kn = np.empty_like(eig_kn)
            w_kn[:] = weight_k[:, np.newaxis]
            eig_m = eig_kn.ravel()
            w_m = w_kn.ravel()
            m_i = eig_m.argsort()
            w_i = w_m[m_i]
            sum_i = np.add.accumulate(w_i)
            filled_i = (sum_i <= nelectrons)
            i = sum(filled_i)
            f_m[m_i[:i]] = 1.0
            if i == len(m_i):
                fermi_level = inf
            else:
                extra = nelectrons - (sum_i[i - 1] if i > 0 else 0.0)
                if extra > 0:
                    assert extra <= w_i[i]
                    f_m[m_i[i]] = extra / w_i[i]
                    fermi_level = eig_m[m_i[i]]
                else:
                    fermi_level = (eig_m[m_i[i]] + eig_m[m_i[i - 1]]) / 2
        else:
            f_kn = None
            fermi_level = nan

        self._distribute_occupation_numbers(f_kn, f_qn, nkpts_r, parallel)

        if parallel.kpt_comm.rank == 0 and parallel.bd is not None:
            fermi_level = broadcast_float(fermi_level, parallel.bd.comm)
        fermi_level = broadcast_float(fermi_level, parallel.kpt_comm)

        e_entropy = 0.0
        return fermi_level, e_entropy

    def _collect_eigelvalues(self, eig_qn, weight_q, parallel):
        kpt_comm = parallel.kpt_comm
        bd = parallel.bd

        nkpts_r = np.zeros(kpt_comm.size, int)
        nkpts_r[kpt_comm.rank] = len(weight_q)
        kpt_comm.sum(nkpts_r)
        weight_k = np.zeros(nkpts_r.sum())
        k1 = nkpts_r[:kpt_comm.rank].sum()
        k2 = k1 + len(weight_q)
        weight_k[k1:k2] = weight_q
        kpt_comm.sum(weight_k, 0)

        eig_kn = None
        k = 0
        for rank, nkpts in enumerate(nkpts_r):
            for q in range(nkpts):
                if rank == kpt_comm.rank:
                    eig_n = eig_qn[q]
                    if bd is not None:
                        eig_n = bd.collect(eig_n)
                if bd is None or bd.comm.rank == 0:
                    if kpt_comm.rank == 0:
                        if k == 0:
                            eig_kn = np.empty((nkpts_r.sum(), len(eig_n)))
                        if rank == 0:
                            eig_kn[k] = eig_n
                        else:
                            kpt_comm.receive(eig_kn[k], rank)
                    elif rank == kpt_comm.rank:
                        kpt_comm.send(eig_n, 0)
                k += 1
        return eig_kn, weight_k, nkpts_r

    def _distribute_occupation_numbers(self, f_kn, f_qn, nkpts_r, parallel):
        kpt_comm = parallel.kpt_comm
        bd = parallel.bd
        k = 0
        for rank, nkpts in enumerate(nkpts_r):
            for q in range(nkpts):
                if kpt_comm.rank == 0:
                    if rank == 0:
                        if bd is None or bd.comm.size == 1:
                            f_qn[q] = f_kn[k]
                        else:
                            bd.distribute(None if f_kn is None else f_kn[k],
                                          f_qn[q])
                    elif f_kn is not None:
                        kpt_comm.send(f_kn[k], rank)
                elif rank == kpt_comm.rank:
                    if bd is None or bd.comm.size == 1:
                        kpt_comm.receive(f_qn[q], 0)
                    else:
                        if bd.comm.rank == 0:
                            f_n = bd.empty(global_array=True)
                            kpt_comm.receive(f_n, 0)
                        else:
                            f_n = None
                        bd.distribute(f_n, f_qn[q])
                k += 1


class FixedOccupations(OccupationNumbers):
    extrapolate_factor = 0.0

    def __init__(self, f_sn):
        OccupationNumbers.__init__(self)
        self.f_sn = np.array(f_sn)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   parallel,
                   fermi_level_guess=nan):
        for q, f_n in enumerate(f_qn):
            s = q % len(self.f_sn)
            parallel.bd.distribute(self.f_sn[s], f_n)

        return inf, 0.0


class ThomasFermiOccupations(OccupationNumbers):
    extrapolate_factor = 0.0

    def todict(self):
        return {'name': 'orbital-free'}

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   parallel,
                   fermi_level_guess=nan):
        assert len(f_qn) == 1
        f_qn[0][:] = [nelectrons]
        return inf, 0.0


def occupation_numbers(occ, eig_skn, weight_k, nelectrons):
    """Calculate occupation numbers from eigenvalues in eV.

    occ: dict
        Example: {'name': 'fermi-dirac', 'width': 0.05} (width in eV).
    eps_skn: ndarray, shape=(nspins, nibzkpts, nbands)
        Eigenvalues.
    weight_k: ndarray, shape=(nibzkpts,)
        Weights of k-points in IBZ (must sum to 1).
    nelectrons: int or float
        Number of electrons.

    Returns a tuple containing:

    * f_skn (sums to nelectrons)
    * fermi-level
    * magnetic moment
    * entropy as -S*T
    """

    warnings.warn('...')
    occ = create_occupation_number_object(**occ)
    f_kn, (fermi_level,), e_entropy = occ.calculate(
        nelectrons * len(eig_skn) / 2,
        [eig_n for eig_kn in eig_skn for eig_n in eig_kn],
        list(weight_k) * len(eig_skn))

    f_kn *= np.array(weight_k)[:, np.newaxis]

    if len(eig_skn) == 1:
        f_skn = np.array([f_kn]) * 2
        e_entropy *= 2
        magmom = 0.0
    else:
        f_skn = np.array([f_kn[::2], f_kn[1::2]])
        f1, f2 = f_skn.sum(axis=(1, 2))
        magmom = f1 - f2

    return f_skn, fermi_level / Ha, magmom, e_entropy
