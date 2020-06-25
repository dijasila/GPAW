# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Occupation number objects."""

from math import pi, nan
from typing import List, Tuple

import numpy as np
from ase.units import Ha

from gpaw.utilities import erf
from gpaw.mpi import serial_comm, broadcast_float


def create_occupation_number_object(name=None,
                                    magmom=None,
                                    bd=None,
                                    kpt_comm=None,
                                    domain_comm=None,
                                    **kwargs):
    if name == 'fermi-dirac':
        return FermiDirac(**kwargs)
    if name == 'methfessel-paxton':
        return MethfesselPaxton(**kwargs)
    if name == 'marzari-vanderbilt':
        return MarzariVanderbilt(**kwargs)
    if name == 'orbital-free':
        return TFOccupations()
    raise ValueError('Unknown occupation number object name: ' + name)


def occupation_numbersxxx(occ, eps_skn, weight_k, nelectrons):
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

    from types import SimpleNamespace
    from gpaw.grid_descriptor import GridDescriptor

    occ = create_occupation_number_object(**occ)

    eps_skn = np.asarray(eps_skn) / Ha
    weight_k = np.asarray(weight_k)
    nspins, nkpts, nbands = eps_skn.shape
    f_skn = np.empty_like(eps_skn)

    wfs = SimpleNamespace(kpt_u=[],
                          nvalence=nelectrons,
                          nspins=nspins,
                          kptband_comm=serial_comm,
                          world=serial_comm,
                          gd=GridDescriptor([4, 4, 4], [1.0, 1.0, 1.0],
                                            comm=serial_comm),
                          bd=SimpleNamespace(nbands=nbands,
                                             collect=lambda x: x,
                                             comm=serial_comm),
                          kd=SimpleNamespace(mynk=nkpts,
                                             comm=serial_comm,
                                             nspins=nspins,
                                             nibzkpts=nkpts,
                                             weight_k=weight_k,
                                             collect=lambda x, broadcast: x))

    for s in range(nspins):
        for k in range(nkpts):
            kpt = SimpleNamespace(s=s,
                                  weight=weight_k[k] * 2 / nspins,
                                  eps_n=eps_skn[s, k],
                                  f_n=f_skn[s, k])
            wfs.kpt_u.append(kpt)

    occ.calculate(wfs)

    return f_skn, occ.fermilevel, occ.magmom, occ.e_entropy


def findroot(func, x, tol=1e-10):
    """Function used for locating Fermi level."""
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


class OccupationNumbers:
    """Base class for all occupation number objects."""
    def __init__(self,
                 fixmagmom: bool = False,
                 bd=None,
                 kpt_comm=None,
                 domain_comm=None):
        self.fixmagmom = fixmagmom
        if fixmagmom:
            assert magmom is None
        self.magmom = magmom
        self.bd = bd
        self.kpt_comm = kpt_comm or serial_comm
        self.domain_comm = domain_comm or serial_comm

    def todict(self):

    def __call__(self,
                 nelectrons: float,
                 eigenvalues: List[List[float]],
                 weights: List[float],
                 fermi_levels_guess: List[float] = [nan, nan]
                 ) -> Tuple[List[np.ndarray],
                            List[float],
                            float]:
        """Calculate everything.

        The following is calculated:

        * occupation numbers
        * entropy
        * Fermi level
        """

        if self.fixed_magmom is not None:
            return self._fixed_magmom(
                nelectrons, eigenvalues, weights, fermi_levels_guess)

        f_kn = [np.empty_like(eig_n) for eig_n in eigenvalues]

        result = np.empty(2)

        if self.domain_comm.rank == 0:
            # Let the master domain do the work and broadcast results:
            result[:] = self._calculate(
                nelectrons, eigenvalues, weights, f_kn, fermi_levels_guess[0])

        self.domain_comm.broadcast(result, 0)
        for f_n in f_kn:
            self.domain_comm.broadcast(f_n, 0)
        fermi_level, e_entropy = result
        return f_kn, [fermi_level], e_entropy

    def _fixed_magmom(self,
                      nelectrons: float,
                      eigenvalues: List[List[float]],
                      weights: List[float],
                      fermi_levels_guess: List[float]
                      ) -> Tuple[List[np.ndarray],
                                 List[float],
                                 float]:
        f1_qn, (fermi_level1,), e_entropy1 = self(
            (nelectrons + self.magmom) / 2,
            eigenvalues[::2],
            weights[::2],
            [fermi_levels_guess[0]])
        f2_qn, (fermi_level2,), e_entropy2 = self(
            (nelectrons - self.magmom) / 2,
            eigenvalues[1::2],
            weights[1::2],
            [fermi_levels_guess[1]])
        f_qn = []
        for f1_n, f2_n in zip(f1_qn, f2_qn):
            f_qn += [f1_n, f2_n]
        e_entropy = e_entropy1 + e_entropy2
        return f_qn, [fermi_level1, fermi_level2], e_entropy

    def extrapolate_energy_to_zero_width(self, e_free):
        return e_free


class ZeroWidth(OccupationNumbers):
    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess=nan):
        eig_kn, weight_k = self._collect_eigelvalues(eig_qn, weight_q)

        if eig_kn is not None:
            w_kn = np.empty_like(eig_kn)
            w_kn[:] = weight_k[:, np.newaxis]
            eig_m = eig_kn.ravel()
            w_m = w_kn.ravel()
            m_i = eig_m.argsort()
            w_i = w_m[m_i]
            f_i = np.add.accumulate(w_i) - 0.5 * w_i
            i = np.nonzero(f_i >= nelectrons)[0][0]
            f_kn = np.zeros_like(eig_kn)
            f_m = f_kn.ravel()
            f_m[m_i[:i]] = 1.0
            extra = nelectrons - f_kn.sum(axis=1).dot(weight_k)
            if extra > 0.0:
                assert extra < w_i[i]
                f_m[m_i[i]] = extra / w_i[i]
                fermi_level = eig_m[m_i[i]]
            else:
                fermi_level = (eig_m[m_i[i]] + eig_m[m_i[i - 1]]) / 2
        else:
            fermi_level = nan

        self._distribute_occupation_numbers(f_kn, f_qn)

        if self.kpt_comm.rank == 0 and self.bd is not None:
            fermi_level = broadcast_float(fermi_level, self.bd.comm)
        fermi_level = broadcast_float(fermi_level, self.kpt_comm)

        e_entropy = 0.0
        return fermi_level, e_entropy

    def _collect_eigelvalues(self, eig_qn, weight_q):
        kpt_comm = self.kpt_comm
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
                eig_n = eig_qn[q]
                if self.bd is not None:
                    eig_n = self.bd.collect(eig_n)
                if self.bd is None or self.bd.comm.rank == 0:
                    if kpt_comm.rank == 0:
                        if k == 0:
                            eig_kn = np.empty((nkpts_r.sum(), len(eig_n)))
                        if rank == 0:
                            eig_kn[k] = eig_n
                        else:
                            kpt_comm.receive(eig_kn[k], rank)
                    else:
                        kpt_comm.send(eig_n, 0)
                k += 1
        return eig_kn, weight_k, nkpts_r

    def _distribute_occupation_numbers(self, f_kn, f_qn, nkpts_r):
        kpt_comm = self.kpt_comm
        k = 0
        for rank, nkpts in enumerate(nkpts_r):
            for q in range(nkpts):
                if kpt_comm.rank == 0:
                    if self.bd.comm.rank == 0:
                        if rank == 0:
                            f_qn[q] = f_kn[k]
                        else:
                            kpt_comm.send(f_kn[k], rank)
                else:
                    if self.bd is None or self.bd.comm.size == 1:
                        kpt_comm.receive(f_qn[q], 0)
                    else:
                        if self.bd.comm.rank == 0:
                            f_n = self.bd.empty(global_array=True)
                            kpt_comm.receive(f_n, 0)
                        else:
                            f_n = None
                        self.bd.distribute(f_n, f_qn[q])
                k += 1


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
                   fermi_level_guess):
        if np.isnan(fermi_level_guess):
            zero = ZeroWidth(self.bd, self.kpt_comm, self.domain_comm)
            fermi_level_guess = zero._calculate(
                nelectrons, eig_qn, weight_q, f_qn)

        x = fermi_level_guess

        data = np.empty(3)

        def f(x, data=data):
            data[:] = 0.0
            for eig_n, weight, f_n in zip(eig_qn, weight_q, f_qn):
                data += weight * self.distribution(eig_n, x, f_n)
            if self.bd is not None:
                self.bd.comm.sum(data)
            self.kpt_comm.sum(data)
            nelectronsx, dnde = data[:2]
            dn = nelectronsx - nelectrons
            return dn, dnde

        fermilevel, niter = findroot(f, x)

        e_entropy = data[2]

        return fermilevel, e_entropy


class FermiDirac(SmoothDistribution):
    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'fermi-dirac'
        return dct

    def __str__(self):
        return f'  Fermi-Dirac: width={self.width * Ha:.4f} eV\n'

    def distribution(self, eig_n, fermi_level, f_n):
        x = (eig_n - fermi_level) / self.width
        #x = x.clip(-100, 100)
        y = np.exp(x)
        z = y + 1.0
        f_n[:] = 1.0 / z
        n = f_n.sum()
        dnde = (n - (f_n**2).sum()) / self.width
        y *= x
        y /= z
        y -= np.log(z)
        e_entropy = y.sum() * self.width
        return np.array([n, dnde, e_entropy])

    def extrapolate_energy_to_zero_width(self, E):
        return E - 0.5 * self.e_entropy


class MethfesselPaxton(SmoothDistribution):
    def __init__(self, width, order=0):
        SmoothDistribution.__init__(self, width)
        self.order = order

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'methfessel-paxton'
        dct['order'] = self.order
        return dct

    def __str__(self):
        s = '  Methfessel-Paxton: width={0:.4f} eV, order={1}\n'.format(
            self.width * Ha, self.order)
        return SmoothDistribution.__str__(self) + s

    def distribution(self, kpt, fermilevel):
        x = (kpt.eps_n - fermilevel) / self.width
        x = x.clip(-100, 100)

        z = 0.5 * (1 - erf(x))
        for i in range(self.order):
            z += (self.coff_function(i + 1) *
                  self.hermite_poly(2 * i + 1, x) * np.exp(-x**2))
        kpt.f_n[:] = kpt.weight * z
        n = kpt.f_n.sum()

        dnde = 1 / np.sqrt(pi) * np.exp(-x**2)
        for i in range(self.order):
            dnde += (self.coff_function(i + 1) *
                     self.hermite_poly(2 * i + 2, x) * np.exp(-x**2))
        dnde = dnde.sum()
        dnde *= kpt.weight / self.width
        e_entropy = (0.5 * self.coff_function(self.order) *
                     self.hermite_poly(2 * self.order, x) * np.exp(-x**2))
        e_entropy = -kpt.weight * e_entropy.sum() * self.width

        sign = 1 - kpt.s * 2
        return np.array([n, dnde, n * sign, e_entropy])

    def coff_function(self, n):
        return (-1)**n / (np.product(np.arange(1, n + 1)) *
                          4**n * np.sqrt(np.pi))

    def hermite_poly(self, n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            return (2 * x * self.hermite_poly(n - 1, x) -
                    2 * (n - 1) * self.hermite_poly(n - 2, x))

    def extrapolate_energy_to_zero_width(self, E):
        return E - self.e_entropy / (self.order + 2)


class MarzariVanderbilt(SmoothDistribution):
    def __init__(self, width, fixmagmom=False):
        SmoothDistribution.__init__(self, width, fixmagmom)

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'marzari-vanderbilt'
        return dct

    def __str__(self):
        s = '  Marzari-Vanderbilt: width={0:.4f} eV\n'.format(
            self.width * Ha)
        return SmoothDistribution.__str__(self) + s

    def distribution(self, eig_n, fermi_level, f_n):
        x = (eig_n - fermi_level) / self.width
        x = x.clip(-100, 100)

        expterm = np.exp(-(x + (1 / np.sqrt(2)))**2)

        z = expterm / np.sqrt(2 * np.pi) + 0.5 * (1 - erf(1. / np.sqrt(2) + x))
        f_n[:] = z
        n = f_n.sum()

        dnde = expterm * (2 + np.sqrt(2) * x) / np.sqrt(np.pi)
        dnde = dnde.sum() / self.width

        s = expterm * (1 + np.sqrt(2) * x) / (2 * np.sqrt(np.pi))

        e_entropy = -s.sum() * self.width

        return np.array([n, dnde, e_entropy])

    def extrapolate_energy_to_zero_width(self, E):
        # According to Nicola Marzari, one should not extrapolate M-V energies
        # https://lists.quantum-espresso.org/pipermail/users/
        #         2005-October/003170.html
        return E


class FixedOccupations(ZeroWidth):
    def __init__(self, occupation):
        self.occupation = np.array(occupation)
        ZeroWidth.__init__(self, True)

    def spin_paired(self, wfs):
        return self.fixed_moment(wfs)

    def fixed_moment(self, wfs):
        for kpt in wfs.kpt_u:
            wfs.bd.distribute(self.occupation[kpt.s], kpt.f_n)


class HMMMMMTFOccupations(FermiDirac):
    def __init__(self):
        FermiDirac.__init__(self, width=0.0, fixmagmom=False)

    def todict(self):
        return {'name': 'orbital-free'}

    def occupy(self, f_n, eps_n, ne, weight=1):
        """Fill in occupation numbers.

        In TF mode only one band. Is guaranteed to work only
        for spin-paired case.

        return HOMO and LUMO energies."""
        # Same as occupy in FermiDirac expect one band: weight = ne
        return FermiDirac.occupy(self, f_n, eps_n, 1, ne * weight)
