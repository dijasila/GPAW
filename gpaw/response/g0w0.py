import functools
import os
import pickle
import warnings
from math import pi
from pathlib import Path

import gpaw.mpi as mpi
import numpy as np
from ase.parallel import paropen
from ase.units import Ha
from ase.utils import opencew, pickleload
from ase.utils.timing import timer
from gpaw import GPAW, debug
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.chi0 import Chi0Calculator
from gpaw.pw.descriptor import (PWDescriptor,
                                count_reciprocal_vectors)
from gpaw.response.fxckernel_calc import calculate_kernel
from gpaw.response.hilbert import GWHilbertTransforms
from gpaw.response.pair import NoCalculatorPairDensity
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.utilities.progressbar import ProgressBar
from gpaw.xc.exx import EXX, select_kpts
from gpaw.xc.fxc import XCFlags
from gpaw.xc.tools import vxc
from gpaw.response.context import calc_and_context
from gpaw.response.screened_interaction import WCalculator

from ase.utils.filecache import MultiFileJSONCache as FileCache


class Sigma:
    def __init__(self, iq, q_c, fxc, esknshape, **inputs):
        self.iq = iq
        self.q_c = q_c
        self.fxc = fxc
        self._buf = np.zeros((2, * esknshape))
        # self-energies and derivatives:
        self.sigma_eskn, self.dsigma_eskn = self._buf

        self.inputs = inputs

    def sum(self, comm):
        comm.sum(self._buf)

    def __iadd__(self, other):
        self._buf += other._buf
        return self

    @classmethod
    def fromdict(cls, dct):
        instance = cls(dct['iq'], dct['q_c'], dct['fxc'], dct['sigma_eskn'].shape, **dct['inputs'])
        instance.sigma_eskn[:] = dct['sigma_eskn']
        instance.dsigma_eskn[:] = dct['dsigma_eskn']
        return instance

    def todict(self):
        return {'iq': self.iq,
                'q_c': self.q_c,
                'fxc': self.fxc,
                'sigma_eskn': self.sigma_eskn,
                'dsigma_eskn': self.dsigma_eskn,
                'inputs': self.inputs }

class G0W0Outputs:
    def __init__(self, fd, shape, ecut_e, sigma_eskn, dsigma_eskn,
                 eps_skn, vxc_skn, exx_skn, f_skn):
        self.extrapolate(fd, shape, ecut_e, sigma_eskn, dsigma_eskn)
        self.Z_skn = 1 / (1 - self.dsigma_skn)

        # G0W0 single-step.
        # If we want GW0 again, we need to grab the expressions
        # from e.g. e73917fca5b9dc06c899f00b26a7c46e7d6fa749
        # or earlier and use qp correctly.
        self.qp_skn = eps_skn + self.Z_skn * (
            -vxc_skn + exx_skn + self.sigma_skn)

        self.sigma_eskn = sigma_eskn
        self.dsigma_eskn = dsigma_eskn

        self.eps_skn = eps_skn
        self.vxc_skn = vxc_skn
        self.exx_skn = exx_skn
        self.f_skn = f_skn

    def extrapolate(self, fd, shape, ecut_e, sigma_eskn, dsigma_eskn):
        if len(ecut_e) == 1:
            self.sigma_skn = sigma_eskn[0]
            self.dsigma_skn = dsigma_eskn[0]
            self.sigr2_skn = None
            self.dsigr2_skn = None
            return

        from scipy.stats import linregress

        # Do linear fit of selfenergy vs. inverse of number of plane waves
        # to extrapolate to infinite number of plane waves

        print('', file=fd)
        print('Extrapolating selfenergy to infinite energy cutoff:',
              file=fd)
        print('  Performing linear fit to %d points' % len(ecut_e),
              file=fd)
        self.sigr2_skn = np.zeros(shape)
        self.dsigr2_skn = np.zeros(shape)
        self.sigma_skn = np.zeros(shape)
        self.dsigma_skn = np.zeros(shape)
        invN_i = ecut_e**(-3. / 2)
        for m in range(np.product(shape)):
            s, k, n = np.unravel_index(m, shape)

            slope, intercept, r_value, p_value, std_err = \
                linregress(invN_i, sigma_eskn[:, s, k, n])

            self.sigr2_skn[s, k, n] = r_value**2
            self.sigma_skn[s, k, n] = intercept

            slope, intercept, r_value, p_value, std_err = \
                linregress(invN_i, dsigma_eskn[:, s, k, n])

            self.dsigr2_skn[s, k, n] = r_value**2
            self.dsigma_skn[s, k, n] = intercept

        if np.any(self.sigr2_skn < 0.9) or np.any(self.dsigr2_skn < 0.9):
            print('  Warning: Bad quality of linear fit for some (n,k). ',
                  file=fd)
            print('           Higher cutoff might be necesarry.', file=fd)

        print('  Minimum R^2 = %1.4f. (R^2 Should be close to 1)' %
              min(np.min(self.sigr2_skn), np.min(self.dsigr2_skn)),
              file=fd)

    def get_results_eV(self):
        results = {
            'f': self.f_skn,
            'eps': self.eps_skn * Ha,
            'vxc': self.vxc_skn * Ha,
            'exx': self.exx_skn * Ha,
            'sigma': self.sigma_skn * Ha,
            'dsigma': self.dsigma_skn,
            'Z': self.Z_skn,
            'qp': self.qp_skn * Ha}

        results.update(
            sigma_eskn=self.sigma_eskn * Ha,
            dsigma_eskn=self.dsigma_eskn)

        if self.sigr2_skn is not None:
            assert self.dsigr2_skn is not None
            results['sigr2_skn'] = self.sigr2_skn
            results['dsigr2_skn'] = self.dsigr2_skn

        return results


class QSymmetryOp:
    def __init__(self, symno, U_cc, sign):
        self.symno = symno
        self.U_cc = U_cc
        self.sign = sign

    def apply(self, q_c):
        return self.sign * (self.U_cc @ q_c)

    def check_q_Q_symmetry(self, Q_c, q_c):
        d_c = self.apply(q_c) - Q_c
        assert np.allclose(d_c.round(), d_c)

    def get_shift0(self, q_c, Q_c):
        shift0_c = q_c - self.apply(Q_c)
        assert np.allclose(shift0_c.round(), shift0_c)
        return shift0_c.round().astype(int)

    def get_M_vv(self, cell_cv):
        # We'll be inverting these cells a lot.
        # Should have an object with the cell and its inverse which does this.
        return cell_cv.T @ self.U_cc.T @ np.linalg.inv(cell_cv).T

    @classmethod
    def get_symops(cls, qd, iq, q_c):
        # Loop over all k-points in the BZ and find those that are
        # related to the current IBZ k-point by symmetry
        Q1 = qd.ibz2bz_k[iq]
        done = set()
        for Q2 in qd.bz2bz_ks[Q1]:
            if Q2 >= 0 and Q2 not in done:
                time_reversal = qd.time_reversal_k[Q2]
                symno = qd.sym_k[Q2]
                Q_c = qd.bzk_kc[Q2]

                symop = cls(
                    symno=symno,
                    U_cc=qd.symmetry.op_scc[symno],
                    sign=1 - 2 * time_reversal)

                symop.check_q_Q_symmetry(Q_c, q_c)
                # Q_c, symop = QSymmetryOp.from_qd(qd, Q2, q_c)
                yield Q_c, symop
                done.add(Q2)


gw_logo = """\
  ___  _ _ _
 |   || | | |
 | | || | | |
 |__ ||_____|
 |___|
"""


def get_max_nblocks(world, calc, ecut):
    nblocks = world.size
    if not isinstance(calc, (str, Path)):
        raise Exception('Using a calulator is not implemented at '
                        'the moment, load from file!')
        # nblocks_calc = calc
    else:
        nblocks_calc = GPAW(calc)
    ngmax = []
    for q_c in nblocks_calc.wfs.kd.bzk_kc:
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(np.min(ecut) / Ha,
                          nblocks_calc.wfs.gd, complex, qd)
        ngmax.append(pd.ngmax)
    nG = np.min(ngmax)

    while nblocks > nG**0.5 + 1 or world.size % nblocks != 0:
        nblocks -= 1

    mynG = (nG + nblocks - 1) // nblocks
    assert mynG * (nblocks - 1) < nG
    return nblocks


def get_frequencies(frequencies, domega0, omega2):
    if domega0 is not None or omega2 is not None:
        assert frequencies is None
        frequencies = {'type': 'nonlinear',
                       'domega0': 0.025 if domega0 is None else domega0,
                       'omega2': 10.0 if omega2 is None else omega2}
        warnings.warn(f'Please use frequencies={frequencies}')
    elif frequencies is None:
        frequencies = {'type': 'nonlinear',
                       'domega0': 0.025,
                       'omega2': 10.0}
    else:
        assert frequencies['type'] == 'nonlinear'
    return frequencies


def choose_ecut_things(ecut, ecut_extrapolation):
    if ecut_extrapolation is True:
        pct = 0.8
        necuts = 3
        ecut_e = ecut * (1 + (1. / pct - 1) * np.arange(necuts)[::-1] /
                         (necuts - 1))**(-2 / 3)
    elif isinstance(ecut_extrapolation, (list, np.ndarray)):
        ecut_e = np.array(np.sort(ecut_extrapolation))
        ecut = ecut_e[-1]
    else:
        ecut_e = np.array([ecut])
    return ecut, ecut_e


class G0W0Calculator:
    def __init__(self, filename='gw', *,
                 wcalc,
                 restartfile=None,
                 kpts, bands, nbands=None,
                 do_GW_too=False,
                 eta,
                 ecut_e,
                 frequencies=None,
                 savepckl=True):

        """G0W0 calculator, initialized through G0W0 object.

        The G0W0 calculator is used is used to calculate the quasi
        particle energies through the G0W0 approximation for a number
        of states.

        Parameters
        ----------
        filename: str
            Base filename of output files.
        wcalc: WCalculator object
            Defines the calculator for computing the screened interaction
        restartfile: str
            File that stores data necessary to restart a calculation.
        kpts: list
            List of indices of the IBZ k-points to calculate the quasi particle
            energies for.
        bands:
            Range of band indices, like (n1, n2), to calculate the quasi
            particle energies for. Bands n where n1<=n<n2 will be
            calculated.  Note that the second band index is not included.
        frequencies:
            Input parameters for frequency_grid.
            Can be array of frequencies to evaluate the response function at
            or dictionary of parameters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut_e: array(float)
            Plane wave cut-off energies in eV. Defined with choose_ecut_things
        nbands: int
            Number of bands to use in the calculation. If None, the number will
            be determined from :ecut: to yield a number close to the number of
            plane waves used.
        do_GW_too: bool
            When carrying out a calculation including vertex corrections, it
            is possible to get the standard GW results at the same time
            (almost for free).
        savepckl: bool
            Save output to a pckl file.
        """

        self.wcalc = wcalc
        self.fd = self.wcalc.fd
        self.timer = self.wcalc.timer

        # Note: self.wcalc.wd should be our only representation
        # of the frequencies.
        # We should therefore get rid of self.frequencies.
        # It is currently only used by the restart code,
        # so should be easy to remove after some further adaptation.
        self.frequencies = frequencies

        self.ecut_e = ecut_e / Ha

        if self.wcalc.ppa and self.wcalc.pair.nblocks > 1:
            raise ValueError(
                'PPA is currently not compatible with block parallelisation.')

        self.world = self.wcalc.context.world
        self.fd = self.fd

        print(gw_logo, file=self.fd)

        self.do_GW_too = do_GW_too

        if not self.wcalc.fxc_mode == 'GW':
            assert self.wcalc.xckernel.xc != 'RPA'

        if self.do_GW_too:
            assert self.wcalc.xckernel.xc != 'RPA'
            assert self.wcalc.fxc_mode != 'GW'
            if restartfile is not None:
                raise RuntimeError('Restart function does not currently work '
                                   'with do_GW_too=True.')

        self.filename = filename
        self.restartfile = restartfile
        self.savepckl = savepckl
        self.eta = eta / Ha

        self.qcache = FileCache('qcache_' + self.filename)

        self.kpts = kpts
        self.bands = bands

        b1, b2 = self.bands
        self.shape = (self.wcalc.gs.nspins, len(self.kpts), b2 - b1)

        self.nbands = nbands

        if self.wcalc.gs.nspins != 1 and self.wcalc.fxc_mode != 'GW':
            raise RuntimeError('Including a xc kernel does currently not '
                               'work for spinpolarized systems.')

        self.pair_distribution = self.wcalc.pair.distribute_k_points_and_bands(
            b1, b2, self.wcalc.gs.kd.ibz2bz_k[self.kpts])

        self.print_parameters(kpts, b1, b2)
        self.fd.flush()
        self.hilbert_transform = None  # initialized when we create Chi0

        if self.wcalc.ppa:
            print('Using Godby-Needs plasmon-pole approximation:',
                  file=self.fd)
            print('  Fitting energy: i*E0, E0 = %.3f Hartee' % self.wcalc.E0,
                  file=self.fd)
        else:
            print('Using full frequency integration', file=self.fd)

    def print_parameters(self, kpts, b1, b2):
        p = functools.partial(print, file=self.fd)
        p()
        p('Quasi particle states:')
        if kpts is None:
            p('All k-points in IBZ')
        else:
            kptstxt = ', '.join(['{0:d}'.format(k) for k in self.kpts])
            p('k-points (IBZ indices): [' + kptstxt + ']')
        p('Band range: ({0:d}, {1:d})'.format(b1, b2))
        p()
        p('Computational parameters:')
        if len(self.ecut_e) == 1:
            p('Plane wave cut-off: {0:g} eV'.format(self.wcalc.chi0calc.ecut
                                                    * Ha))
        else:
            assert len(self.ecut_e) > 1
            p('Extrapolating to infinite plane wave cut-off using points at:')
            for ec in self.ecut_e:
                p('  %.3f eV' % (ec * Ha))
        p('Number of bands: {0:d}'.format(self.nbands))
        p('Coulomb cutoff:', self.wcalc.truncation)
        p('Broadening: {0:g} eV'.format(self.eta * Ha))
        p()
        p('fxc mode:', self.wcalc.fxc_mode)
        p('Kernel:', self.wcalc.xckernel.xc)
        p('Do GW too:', self.do_GW_too)
        p()

    def get_eps_and_occs(self):
        eps_skn = np.empty(self.shape)  # KS-eigenvalues
        f_skn = np.empty(self.shape)  # occupation numbers

        nspins = self.wcalc.gs.nspins
        b1, b2 = self.bands
        for i, k in enumerate(self.kpts):
            for s in range(nspins):
                u = s + k * nspins
                kpt = self.wcalc.gs.kpt_u[u]
                eps_skn[s, i] = kpt.eps_n[b1:b2]
                f_skn[s, i] = kpt.f_n[b1:b2] / kpt.weight

        return eps_skn, f_skn

    @timer('G0W0')
    def calculate(self):
        """Starts the G0W0 calculation.

        Returns a dict with the results with the following key/value pairs:

        ===========  =============================================
        key          value
        ===========  =============================================
        ``f``        Occupation numbers
        ``eps``      Kohn-Sham eigenvalues in eV
        ``vxc``      Exchange-correlation
                     contributions in eV
        ``exx``      Exact exchange contributions in eV
        ``sigma``    Self-energy contributions in eV
        ``dsigma``   Self-energy derivatives
        ``sigma_e``  Self-energy contributions in eV
                     used for ecut extrapolation
        ``Z``        Renormalization factors
        ``qp``       Quasi particle (QP) energies in eV
        ``iqp``      GW0/GW: QP energies for each iteration in eV
        ===========  =============================================

        All the values are ``ndarray``'s of shape
        (spins, IBZ k-points, bands)."""

        loaded = False
        if self.restartfile is not None:
            loaded = self.load_restart_file()
            if not loaded:
                self.last_q = -1
                self.previous_sigma = 0.
                self.previous_dsigma = 0.

            else:
                print('Reading ' + str(self.last_q + 1) +
                      ' q-point(s) from the previous calculation: ' +
                      self.restartfile + '.sigma.pckl', file=self.fd)
        else:
            self.last_q = -1
            self.previous_sigma = 0.
            self.previous_dsigma = 0.

        self.fd.flush()

        # Loop over q in the IBZ:
        print('Summing all q:', file=self.fd)

        self.calculate_all_q_points()
        return self.postprocess(loaded)

    def postprocess(self, loaded):
        # Integrate over all q-points, and accumulate the quasiparticle shifts
        for iq, q_c in enumerate(self.wcalc.qd.ibzk_kc):
            #if self.qcomm.rank == 0:
            key = str(iq)

            print(self.qcache[key],'Key is', type(self.qcache[key]))
            sigmas_contrib = {fxc_mode:Sigma.fromdict(sigma) for fxc_mode, sigma in self.qcache[key].items()}

            if iq == 0:
                sigmas = sigmas_contrib
            else:
                for fxc_mode in self.fxc_modes:
                    sigmas[fxc_mode] += sigmas_contrib[fxc_mode]

        all_results = {}
        for fxc_mode, sigma in sigmas.items():
            all_results[fxc_mode] = self.postprocess_single(fxc_mode, sigma,
                                                            loaded)
        self.all_results = all_results
        self.print_results(self.all_results)

        # After we have written the results restartfile is obsolete
        if self.restartfile is not None:
            if self.world.rank == 0:
                if os.path.isfile(self.restartfile + '.sigma.pckl'):
                    os.remove(self.restartfile + '.sigma.pckl')

        return self.results  # XXX ugly discrepancy

    def postprocess_single(self, fxc_name, sigma, loaded):
        sigma.sum(self.world)  # (Not so pretty that we finalize the sum here)

        if self.restartfile is not None and loaded:
            assert not self.do_GW_too
            sigma.sigma_eskn += self.previous_sigma
            sigma.dsigma_eskn += self.previous_dsigma

        output = self.calculate_g0w0_outputs(sigma)
        result = output.get_results_eV()

        if self.savepckl:
            with paropen(f'{self.filename}_results_{fxc_name}.pckl',
                         'wb') as fd:
                pickle.dump(result, fd, 2)

        return result

    @property
    def results_GW(self):
        if self.do_GW_too:
            return self.all_results['GW']

    @property
    def results(self):
        return self.all_results[self.wcalc.fxc_mode]

    def calculate_q(self, ie, k, kpt1, kpt2, pd0, Wdict,
                    *, symop, sigmas, blocks1d, Q_aGii):
        """Calculates the contribution to the self-energy and its derivative
        for a given set of k-points, kpt1 and kpt2."""

        N_c = pd0.gd.N_c
        i_cG = symop.apply(np.unravel_index(pd0.Q_qG[0], N_c))

        q_c = self.wcalc.gs.kd.bzk_kc[kpt2.K] - self.wcalc.gs.kd.bzk_kc[kpt1.K]

        shift0_c = symop.get_shift0(q_c, pd0.kd.bzk_kc[0])
        shift_c = kpt1.shift_c - kpt2.shift_c - shift0_c

        I_G = np.ravel_multi_index(i_cG + shift_c[:, None], N_c, 'wrap')

        G_Gv = pd0.get_reciprocal_vectors()

        pos_av = np.dot(self.wcalc.pair.spos_ac, pd0.gd.cell_cv)
        M_vv = symop.get_M_vv(pd0.gd.cell_cv)

        myQ_aGii = []
        for a, Q_Gii in enumerate(Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                            np.dot(M_vv, pos_av[a]))))
            U_ii = self.wcalc.gs.setups[a].R_sii[symop.symno]
            Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                           U_ii.T).transpose(1, 0, 2)
            if symop.sign == -1:
                Q_Gii = Q_Gii.conj()
            myQ_aGii.append(Q_Gii)

        if debug:
            self.check(ie, i_cG, shift0_c, N_c, q_c, myQ_aGii)

        if self.wcalc.ppa:
            calculate_sigma = self.calculate_sigma_ppa
        else:
            calculate_sigma = self.calculate_sigma

        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            eps1 = kpt1.eps_n[n]
            C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                      for Qa_Gii, P1_ni in zip(myQ_aGii, kpt1.P_ani)]
            n_mG = self.wcalc.pair.calculate_pair_densities(
                ut1cc_R, C1_aGi, kpt2, pd0, I_G)
            if symop.sign == 1:
                n_mG = n_mG.conj()

            f_m = kpt2.f_n
            deps_m = eps1 - kpt2.eps_n

            nn = kpt1.n1 + n - self.bands[0]

            assert set(Wdict) == set(sigmas)
            for fxc_mode in self.fxc_modes:
                sigma = sigmas[fxc_mode]
                W = Wdict[fxc_mode]
                sigma_contrib, dsigma_contrib = calculate_sigma(
                    n_mG, deps_m, f_m, W, blocks1d)
                sigma.sigma_eskn[ie, kpt1.s, k, nn] += sigma_contrib
                sigma.dsigma_eskn[ie, kpt1.s, k, nn] += dsigma_contrib

    def check(self, ie, i_cG, shift0_c, N_c, q_c, Q_aGii):
        I0_G = np.ravel_multi_index(i_cG - shift0_c[:, None], N_c, 'wrap')
        qd1 = KPointDescriptor([q_c])
        pd1 = PWDescriptor(self.ecut_e[ie], self.wcalc.gs.gd, complex, qd1)
        G_I = np.empty(N_c.prod(), int)
        G_I[:] = -1
        I1_G = pd1.Q_qG[0]
        G_I[I1_G] = np.arange(len(I0_G))
        G_G = G_I[I0_G]
        assert len(I0_G) == len(I1_G)
        assert (G_G >= 0).all()
        for a, Q_Gii in enumerate(
                self.wcalc.pair.initialize_paw_corrections(pd1)):
            e = abs(Q_aGii[a] - Q_Gii[G_G]).max()
            assert e < 1e-12

    @timer('Sigma')
    def calculate_sigma(self, n_mG, deps_m, f_m, C_swGG, blocks1d):
        """Calculates a contribution to the self-energy and its derivative for
        a given (k, k-q)-pair from its corresponding pair-density and
        energy."""
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)

        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        w_m = self.wcalc.wd.get_floor_index(o_m, safe=False)
        m_inb = np.where(w_m < len(self.wcalc.wd) - 1)[0]
        o1_m = np.empty(len(o_m))
        o2_m = np.empty(len(o_m))
        o1_m[m_inb] = self.wcalc.wd.omega_w[w_m[m_inb]]
        o2_m[m_inb] = self.wcalc.wd.omega_w[w_m[m_inb] + 1]

        x = 1.0 / (self.wcalc.qd.nbzkpts * 2 * pi * self.wcalc.gs.volume)
        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):

            if w >= len(self.wcalc.wd.omega_w) - 1:
                continue

            C1_GG = C_swGG[s][w]
            C2_GG = C_swGG[s][w + 1]
            p = x * sgn
            myn_G = n_G[blocks1d.myslice]

            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma

    def calculate_all_q_points(self):
        """XXX UPDATE THIS. No longer a generator. Calculates the screened potential for each q-point in the 1st BZ.
        Since many q-points are related by symmetry, the actual calculation is
        only done for q-points in the IBZ and the rest are obtained by symmetry
        transformations. Results are returned as a generator to that it is not
        necessary to store a huge matrix for each q-point in the memory."""
        # The decorator $timer('W') doesn't work for generators, do we will
        # have to manually start and stop the timer here:
        pb = ProgressBar(self.fd)

        self.timer.start('W')
        print('\nCalculating screened Coulomb potential', file=self.fd)
        if self.wcalc.truncation is not None:
            print('Using %s truncated Coloumb potential'
                  % self.wcalc.truncation, file=self.fd)

        chi0calc = self.wcalc.chi0calc

        if self.wcalc.truncation == 'wigner-seitz':
            wstc = WignerSeitzTruncatedCoulomb(
                self.wcalc.gs.gd.cell_cv,
                self.wcalc.gs.kd.N_c,
                chi0calc.fd)
        else:
            wstc = None

        self.hilbert_transform = GWHilbertTransforms(
            self.wcalc.wd.omega_w, self.eta)
        print(self.wcalc.wd, file=self.fd)

        # Find maximum size of chi-0 matrices:
        nGmax = max(count_reciprocal_vectors(chi0calc.ecut,
                                             self.wcalc.gs.gd, q_c)
                    for q_c in self.wcalc.qd.ibzk_kc)
        nw = len(self.wcalc.wd)

        size = self.wcalc.blockcomm.size

        mynGmax = (nGmax + size - 1) // size
        mynw = (nw + size - 1) // size

        # some memory sizes...
        if self.world.rank == 0:
            siz = (nw * mynGmax * nGmax +
                   max(mynw * nGmax, nw * mynGmax) * nGmax) * 16
            sizA = (nw * nGmax * nGmax + nw * nGmax * nGmax) * 16
            print('  memory estimate for chi0: local=%.2f MB, global=%.2f MB'
                  % (siz / 1024**2, sizA / 1024**2), file=self.fd)
            self.fd.flush()

        # Need to pause the timer in between iterations
        self.timer.stop('W')
        
        for iq, q_c in enumerate(self.wcalc.qd.ibzk_kc):
            with self.qcache.lock(str(iq)) as qhandle:
                if qhandle is None: 
                    continue

                result = self.calculate_q_point(iq, q_c, pb, wstc, chi0calc)

                if self.world.rank == 0: #  XXX: Replace with self.qcomm
                    qhandle.save(result)
        pb.finish()

    def calculate_q_point(self, iq, q_c, pb, wstc, chi0calc):
        # Reset calculation
        sigmashape = (len(self.ecut_e), *self.shape)
        sigmas = {fxc_mode: Sigma(iq, q_c, fxc_mode, sigmashape)
                  for fxc_mode in self.fxc_modes}

        if len(self.ecut_e) > 1:
            chi0bands = chi0calc.create_chi0(q_c, extend_head=False)
        else:
            chi0bands = None

        m1 = chi0calc.nocc1
        for ie, ecut in enumerate(self.ecut_e):
            self.timer.start('W')

            # First time calculation
            if ecut == chi0calc.ecut:
                # Nothing to cut away:
                m2 = self.nbands
            else:
                m2 = int(self.wcalc.gs.volume * ecut**1.5
                         * 2**0.5 / 3 / pi**2)
                if m2 > self.nbands:
                    raise ValueError(f'Trying to extrapolate ecut to'
                                     f'larger number of bands ({m2})'
                                     f' than there are bands '
                                     f'({self.nbands}).')
            pdi, Wdict, blocks1d, Q_aGii = self.calculate_w(
                chi0calc, q_c, chi0bands,
                m1, m2, ecut, wstc, iq)
            m1 = m2

            self.timer.stop('W')

            for nQ, (bzq_c, symop) in enumerate(QSymmetryOp.get_symops(
                    self.wcalc.qd, iq, q_c)):

                for progress, kpt1, kpt2 in self.pair_distribution.kpt_pairs_by_q(
                        bzq_c, 0, m2):
                    pb.update((nQ + progress) / self.wcalc.qd.mynk)
     
                    k1 = self.wcalc.gs.kd.bz2ibz_k[kpt1.K]
                    i = self.kpts.index(k1)
     
                    self.calculate_q(ie, i, kpt1, kpt2, pdi, Wdict,
                                     symop=symop,
                                     sigmas=sigmas,
                                     blocks1d=blocks1d,
                                     Q_aGii=Q_aGii)

        return sigmas

    @property
    def fxc_modes(self):
        modes = [self.wcalc.fxc_mode]
        if self.do_GW_too:
            modes.append('GW')
        return modes

    @timer('WW')
    def calculate_w(self, chi0calc, q_c, chi0bands,
                    m1, m2, ecut, wstc,
                    iq):
        """Calculates the screened potential for a specified q-point."""

        chi0 = chi0calc.create_chi0(q_c, extend_head=False)
        chi0calc.fd = self.fd
        chi0calc.print_chi(chi0.pd)
        chi0calc.update_chi0(chi0, m1, m2, range(self.wcalc.gs.nspins))

        if len(self.ecut_e) > 1:
            # Add chi from previous cutoff with remaining bands
            chi0.chi0_wGG += chi0bands.chi0_wGG
            chi0bands.chi0_wGG[:] = chi0.chi0_wGG.copy()
            if chi0.optical_limit:
                chi0.chi0_wxvG += chi0bands.chi0_wxvG
                chi0bands.chi0_wxvG[:] = chi0.chi0_wxvG.copy()
                chi0.chi0_wvv += chi0bands.chi0_wvv
                chi0bands.chi0_wvv[:] = chi0.chi0_wvv.copy()

        Wdict = {}

        for fxc_mode in self.fxc_modes:
            pdi, blocks1d, W_wGG = self.wcalc.dyson_and_W_old(
                wstc, iq, q_c, chi0calc, chi0, ecut, Q_aGii=chi0calc.Q_aGii,
                fxc_mode=fxc_mode)

            if self.wcalc.ppa:
                W_xwGG = W_wGG  # (ppa API is nonsense)
            else:
                with self.timer('Hilbert'):
                    W_xwGG = self.hilbert_transform(W_wGG)

            Wdict[fxc_mode] = W_xwGG

        return pdi, Wdict, blocks1d, chi0calc.Q_aGii

    @timer('Kohn-Sham XC-contribution')
    def calculate_ks_xc_contribution(self):
        name = self.filename + '.vxc.npy'
        fd, vxc_skn = self.read_contribution(name)
        if vxc_skn is None:
            print('Calculating Kohn-Sham XC contribution', file=self.fd)
            self.fd.flush()
            vxc_skn = vxc(self.wcalc.gs, self.wcalc.gs.hamiltonian.xc) / Ha
            n1, n2 = self.bands
            vxc_skn = vxc_skn[:, self.kpts, n1:n2]
            np.save(fd, vxc_skn)
            fd.close()
        return vxc_skn

    @timer('EXX')
    def calculate_exact_exchange(self):
        name = self.filename + '.exx.npy'
        fd, exx_skn = self.read_contribution(name)
        if exx_skn is None:
            print('Calculating EXX contribution', file=self.fd)
            self.fd.flush()
            exx = EXX(self.wcalc.gs, kpts=self.kpts, bands=self.bands,
                      txt=self.filename + '.exx.txt', timer=self.timer)
            exx.calculate()
            exx_skn = exx.get_eigenvalue_contributions() / Ha
            np.save(fd, exx_skn)
            fd.close()
        return exx_skn

    def read_contribution(self, filename):
        fd = opencew(filename)  # create, exclusive, write
        if fd is not None:
            # File was not there: nothing to read
            return fd, None

        try:
            with open(filename, 'rb') as fd:
                x_skn = np.load(fd)
        except IOError:
            print('Removing broken file:', filename, file=self.fd)
        else:
            print('Read:', filename, file=self.fd)
            if x_skn.shape == self.shape:
                return None, x_skn
            print('Removing bad file (wrong shape of array):', filename,
                  file=self.fd)

        if self.world.rank == 0:
            os.remove(filename)

        return opencew(filename), None

    def print_results(self, results):
        description = ['f:      Occupation numbers',
                       'eps:     KS-eigenvalues [eV]',
                       'vxc:     KS vxc [eV]',
                       'exx:     Exact exchange [eV]',
                       'sigma:   Self-energies [eV]',
                       'dsigma:  Self-energy derivatives',
                       'Z:       Renormalization factors',
                       'qp:      QP-energies [eV]']

        print('\nResults:', file=self.fd)
        for line in description:
            print(line, file=self.fd)

        b1, b2 = self.bands
        names = [line.split(':', 1)[0] for line in description]
        ibzk_kc = self.wcalc.gs.kd.ibzk_kc
        for s in range(self.wcalc.gs.nspins):
            for i, ik in enumerate(self.kpts):
                print('\nk-point ' +
                      '{0} ({1}): ({2:.3f}, {3:.3f}, {4:.3f})'.format(
                          i, ik, *ibzk_kc[ik]) + '                ' +
                      self.wcalc.fxc_mode, file=self.fd)
                print('band' +
                      ''.join('{0:>8}'.format(name) for name in names),
                      file=self.fd)

                def actually_print_results(resultset):
                    for n in range(b2 - b1):
                        print('{0:4}'.format(n + b1) +
                              ''.join('{0:8.3f}'
                                      .format(resultset[name][s, i, n])
                                      for name in names),
                              file=self.fd)

                for fxc_mode in results:
                    print(fxc_mode.rjust(69), file=self.fd)
                    actually_print_results(results[fxc_mode])

        self.timer.write(self.fd)

    @timer('PPA-Sigma')
    def calculate_sigma_ppa(self, n_mG, deps_m, f_m, W, *unused):
        W_GG, omegat_GG = W

        sigma = 0.0
        dsigma = 0.0

        for m in range(len(n_mG)):
            deps_GG = deps_m[m]
            sign_GG = 2 * f_m[m] - 1
            x1_GG = 1 / (deps_GG + omegat_GG - 1j * self.eta)
            x2_GG = 1 / (deps_GG - omegat_GG + 1j * self.eta)
            x3_GG = 1 / (deps_GG + omegat_GG - 1j * self.eta * sign_GG)
            x4_GG = 1 / (deps_GG - omegat_GG - 1j * self.eta * sign_GG)
            x_GG = W_GG * (sign_GG * (x1_GG - x2_GG) + x3_GG + x4_GG)
            dx_GG = W_GG * (sign_GG * (x1_GG**2 - x2_GG**2) +
                            x3_GG**2 + x4_GG**2)
            nW_G = np.dot(n_mG[m], x_GG)
            sigma += np.vdot(n_mG[m], nW_G).real
            nW_G = np.dot(n_mG[m], dx_GG)
            dsigma -= np.vdot(n_mG[m], nW_G).real

        x = 1 / (self.wcalc.qd.nbzkpts * 2 * pi * self.wcalc.gs.volume)
        return x * sigma, x * dsigma

    def save_restart_file(self, nQ):
        sigma = self.sigmas[self.wcalc.fxc_mode]
        sigma_eskn_write = sigma.sigma_eskn.copy()
        dsigma_eskn_write = sigma.dsigma_eskn.copy()
        self.world.sum(sigma_eskn_write)
        self.world.sum(dsigma_eskn_write)
        data = {'last_q': nQ,
                'sigma_eskn': sigma_eskn_write + self.previous_sigma,
                'dsigma_eskn': dsigma_eskn_write + self.previous_dsigma,
                'kpts': self.kpts,
                'bands': self.bands,
                'nbands': self.nbands,
                'ecut_e': self.ecut_e,
                'frequencies': self.frequencies,
                'integrate_gamma': self.wcalc.integrate_gamma}

        if self.world.rank == 0:
            with open(self.restartfile + '.sigma.pckl', 'wb') as fd:
                pickle.dump(data, fd, 2)

    def load_restart_file(self):
        try:
            with open(self.restartfile + '.sigma.pckl', 'rb') as fd:
                data = pickleload(fd)
        except IOError:
            return False
        else:
            if (data['kpts'] == self.kpts and
                data['bands'] == self.bands and
                data['nbands'] == self.nbands and
                (data['ecut_e'] == self.ecut_e).all and
                data['frequencies']['type'] == self.frequencies['type'] and
                data['frequencies']['domega0'] ==
                self.frequencies['domega0'] and
                data['frequencies']['omega2'] == self.frequencies['omega2'] and
                data['integrate_gamma'] == self.wcalc.integrate_gamma):
                self.last_q = data['last_q']
                self.previous_sigma = data['sigma_eskn']
                self.previous_dsigma = data['dsigma_eskn']
                return True
            else:
                raise ValueError(
                    'Restart file not compatible with parameters used in '
                    'current calculation. Check kpts, bands, nbands, ecut, '
                    'domega0, omega2, integrate_gamma.')

    def calculate_g0w0_outputs(self, sigma):
        eps_skn, f_skn = self.get_eps_and_occs()
        kwargs = dict(
            fd=self.fd,
            shape=self.shape,
            ecut_e=self.ecut_e,
            eps_skn=eps_skn,
            vxc_skn=self.calculate_ks_xc_contribution(),
            exx_skn=self.calculate_exact_exchange(),
            f_skn=f_skn)

        return G0W0Outputs(sigma_eskn=sigma.sigma_eskn,
                           dsigma_eskn=sigma.dsigma_eskn,
                           **kwargs)


def choose_bands(bands, relbands, nvalence, nocc):
    if bands is not None and relbands is not None:
        raise ValueError('Use bands or relbands!')

    if relbands is not None:
        bands = [nvalence // 2 + b for b in relbands]

    if bands is None:
        bands = [0, nocc]

    return bands


class G0W0Kernel:
    def __init__(self, xc, **kwargs):
        self.xc = xc
        self.xcflags = XCFlags(xc)
        self._kwargs = kwargs

    def calculate(self, nG, iq, G2G):
        return calculate_kernel(
            xcflags=self.xcflags,
            nG=nG, iq=iq, cut_G=G2G, **self._kwargs)


class G0W0(G0W0Calculator):
    def __init__(self, calc, filename='gw',
                 ecut=150.0,
                 ecut_extrapolation=False,
                 xc='RPA',
                 ppa=False,
                 E0=Ha,
                 Eg=None,
                 eta=0.1,
                 nbands=None,
                 bands=None,
                 relbands=None,
                 frequencies=None,
                 domega0=None,  # deprecated
                 omega2=None,  # deprecated
                 nblocks=1,
                 nblocksmax=False,
                 kpts=None,
                 world=mpi.world,
                 timer=None,
                 fxc_mode='GW',
                 truncation=None,
                 integrate_gamma=0,
                 q0_correction=False,
                 **kwargs):
        """G0W0 calculator wrapper.

        The G0W0 calculator is used is used to calculate the quasi
        particle energies through the G0W0 approximation for a number
        of states.

        Parameters
        ----------
        calc:
            GPAW calculator object or filename of saved calculator object.
        filename: str
            Base filename of output files.
        restartfile: str
            File that stores data necessary to restart a calculation.
        kpts: list
            List of indices of the IBZ k-points to calculate the quasi particle
            energies for.
        bands:
            Range of band indices, like (n1, n2), to calculate the quasi
            particle energies for. Bands n where n1<=n<n2 will be
            calculated.  Note that the second band index is not included.
        relbands:
            Same as *bands* except that the numbers are relative to the
            number of occupied bands.
            E.g. (-1, 1) will use HOMO+LUMO.
        frequencies:
            Input parameters for frequency_grid.
            Can be array of frequencies to evaluate the response function at
            or dictionary of parameters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut: float
            Plane wave cut-off energy in eV.
        ecut_extrapolation: bool or list
            If set to True an automatic extrapolation of the selfenergy to
            infinite cutoff will be performed based on three points
            for the cutoff energy.
            If an array is given, the extrapolation will be performed based on
            the cutoff energies given in the array.
        nbands: int
            Number of bands to use in the calculation. If None, the number will
            be determined from :ecut: to yield a number close to the number of
            plane waves used.
        ppa: bool
            Sets whether the Godby-Needs plasmon-pole approximation for the
            dielectric function should be used.
        xc: str
            Kernel to use when including vertex corrections.
        fxc_mode: str
            Where to include the vertex corrections; polarizability and/or
            self-energy. 'GWP': Polarizability only, 'GWS': Self-energy only,
            'GWG': Both.
        do_GW_too: bool
            When carrying out a calculation including vertex corrections, it
            is possible to get the standard GW results at the same time
            (almost for free).
        Eg: float
            Gap to apply in the 'JGMs' (simplified jellium-with-gap) kernel.
            If None the DFT gap is used.
        truncation: str
            Coulomb truncation scheme. Can be either wigner-seitz,
            2D, 1D, or 0D
        integrate_gamma: int
            Method to integrate the Coulomb interaction. 1 is a numerical
            integration at all q-points with G=[0,0,0] - this breaks the
            symmetry slightly. 0 is analytical integration at q=[0,0,0] only -
            this conserves the symmetry. integrate_gamma=2 is the same as 1,
            but the average is only carried out in the non-periodic directions.
        E0: float
            Energy (in eV) used for fitting in the plasmon-pole approximation.
        q0_correction: bool
            Analytic correction to the q=0 contribution applicable to 2D
            systems.
        nblocks: int
            Number of blocks chi0 should be distributed in so each core
            does not have to store the entire matrix. This is to reduce
            memory requirement. nblocks must be less than or equal to the
            number of processors.
        nblocksmax: bool
            Cuts chi0 into as many blocks as possible to reduce memory
            requirements as much as possible.
        savepckl: bool
            Save output to a pckl file.
        """

        frequencies = get_frequencies(frequencies, domega0, omega2)

        gpwfile = calc
        calc, context = calc_and_context(gpwfile, filename + '.txt',
                                         world, timer)
        gs = calc.gs_adapter()

        # Check if nblocks is compatible, adjust if not
        if nblocksmax:
            nblocks = get_max_nblocks(context.world, gpwfile, ecut)

        pair = NoCalculatorPairDensity(gs, nblocks=nblocks, context=context)

        kpts = list(select_kpts(kpts, gs.kd))

        if nbands is None:
            nbands = int(gs.volume * (ecut / Ha)**1.5 * 2**0.5 / 3 / pi**2)
        else:
            if ecut_extrapolation:
                raise RuntimeError(
                    'nbands cannot be supplied with ecut-extrapolation.')

        ecut, ecut_e = choose_ecut_things(ecut, ecut_extrapolation)

        if ppa:
            # use small imaginary frequency to avoid dividing by zero:
            frequencies = [1e-10j, 1j * E0]

            parameters = {'eta': 0,
                          'hilbert': False,
                          'timeordered': False}
        else:
            # frequencies = self.frequencies
            parameters = {'eta': eta,
                          'hilbert': True,
                          'timeordered': True}

        from gpaw.response.chi0 import new_frequency_descriptor
        chi_context = context.with_txt(filename + '.w.txt')
        wd = new_frequency_descriptor(
            gs, nbands, frequencies, fd=chi_context.fd)

        chi0calc = Chi0Calculator(
            wd=wd, pair=pair,
            nbands=nbands,
            ecut=ecut,
            intraband=False,
            context=chi_context,
            **parameters)

        bands = choose_bands(bands, relbands, gs.nvalence, chi0calc.nocc2)

        if Eg is None and xc == 'JGMsx':
            Eg = gs.get_band_gap()

        if Eg is not None:
            Eg /= Ha

        xckernel = G0W0Kernel(xc=xc, ecut=ecut / Ha,
                              gs=gs,
                              ns=gs.nspins,
                              wd=wd,
                              Eg=Eg,
                              timer=context.timer,
                              fd=context.fd)

        wcalc = WCalculator(chi0calc, ppa, xckernel,
                            context, E0,
                            fxc_mode, truncation,
                            integrate_gamma,
                            q0_correction)

        super().__init__(filename=filename,
                         wcalc=wcalc,
                         ecut_e=ecut_e,
                         eta=eta,
                         nbands=nbands,
                         bands=bands,
                         frequencies=frequencies,
                         kpts=kpts,
                         **kwargs)
