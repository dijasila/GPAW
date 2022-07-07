import functools
import os
import pickle
import warnings
from math import pi

import numpy as np
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import paropen
from ase.units import Ha
from ase.utils import opencew, pickleload
from ase.utils.timing import timer

import gpaw.mpi as mpi
from gpaw import debug
from gpaw.calculator import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.fxckernel_calc import calculate_kernel
from gpaw.response.kernels import get_coulomb_kernel, get_integrated_kernel
from gpaw.response.pair import NoCalculatorPairDensity
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.utilities.progressbar import ProgressBar
from gpaw.pw.descriptor import (PWDescriptor, PWMapping,
                                count_reciprocal_vectors)
from gpaw.xc.exx import EXX, select_kpts
from gpaw.xc.fxc import XCFlags
from gpaw.xc.tools import vxc
from gpaw.response.context import calc_and_context
from gpaw.response.temp import DielectricFunctionCalculator
from gpaw.response.q0_correction import Q0Correction
from gpaw.response.hilbert import GWHilbertTransforms
from gpaw.response.gamma_int import GammaIntegrator


class Sigma:
    def __init__(self, esknshape):
        self._buf = np.zeros((2, * esknshape))
        # self-energies and derivatives:
        self.sigma_eskn, self.dsigma_eskn = self._buf

    def sum(self, comm):
        comm.sum(self._buf)


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
    if isinstance(calc, GPAW):
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


def get_qdescriptor(kd, atoms):
    # Find q-vectors and weights in the IBZ:
    assert -1 not in kd.bz2bz_ks
    offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
    bzq_qc = monkhorst_pack(kd.N_c) + offset_c
    qd = KPointDescriptor(bzq_qc)
    qd.set_symmetry(atoms, kd.symmetry)
    return qd


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


class G0W0:
    def __init__(self, calc, filename='gw', *,
                 restartfile=None,
                 kpts=None, bands=None, relbands=None, nbands=None, ppa=False,
                 xc='RPA', fxc_mode='GW', do_GW_too=False,
                 Eg=None,
                 truncation=None, integrate_gamma=0,
                 ecut=150.0, eta=0.1, E0=1.0 * Ha,
                 frequencies=None,
                 domega0=None,  # deprecated
                 omega2=None,  # deprecated
                 q0_correction=False,
                 nblocks=1, savepckl=True,
                 world=mpi.world, ecut_extrapolation=False,
                 timer=None,
                 nblocksmax=False):

        """G0W0 calculator.

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
        self.frequencies = get_frequencies(frequencies, domega0, omega2)

        if ppa and (nblocks > 1 or nblocksmax):
            raise ValueError(
                'PPA is currently not compatible with block parallellisation.')

        ecut, ecut_e = choose_ecut_things(ecut, ecut_extrapolation)
        self.ecut_e = ecut_e / Ha

        # Check if nblocks is compatible, adjust if not
        if nblocksmax:
            nblocks = get_max_nblocks(world, calc, ecut)

        calc, self.context = calc_and_context(calc, filename + '.txt',
                                              world, timer)
        self.gs = calc.gs_adapter()

        self.pair = NoCalculatorPairDensity(self.gs, nblocks=nblocks,
                                            context=self.context)

        self.timer = self.context.timer
        self.fd = self.context.fd
        self.ecut = ecut / Ha
        self.blockcomm = self.pair.blockcomm
        self.world = self.context.world
        self.vol = self.pair.vol

        print(gw_logo, file=self.fd)

        self.xc = xc
        self.fxc_mode = fxc_mode
        self.do_GW_too = do_GW_too

        if not self.fxc_mode == 'GW':
            assert self.xc != 'RPA'

        if self.do_GW_too:
            assert self.xc != 'RPA'
            assert self.fxc_mode != 'GW'
            if restartfile is not None:
                raise RuntimeError('Restart function does not currently work '
                                   'with do_GW_too=True.')

        if Eg is None and self.xc == 'JGMsx':
            Eg = self.gs.get_band_gap()

        if Eg is not None:
            Eg /= Ha

        self.Eg = Eg

        self.xcflags = XCFlags(self.xc)

        self.filename = filename
        self.restartfile = restartfile
        self.savepckl = savepckl
        self.ppa = ppa
        self.truncation = truncation
        self.integrate_gamma = integrate_gamma
        self.eta = eta / Ha
        self.E0 = E0 / Ha

        self.wd = None

        self.blocks1d = None
        self.blockdist = None

        self.kpts = list(select_kpts(kpts, self.gs.kd))
        self.bands = bands = self.choose_bands(bands, relbands)

        b1, b2 = bands
        self.shape = (self.gs.nspins, len(self.kpts), b2 - b1)

        if nbands is None:
            nbands = int(self.vol * self.ecut**1.5 * 2**0.5 / 3 / pi**2)
        else:
            if ecut_extrapolation:
                raise RuntimeError(
                    'nbands cannot be supplied with ecut-extrapolation.')

        self.nbands = nbands
        self.nspins = self.gs.nspins

        if self.nspins != 1 and self.fxc_mode != 'GW':
            raise RuntimeError('Including a xc kernel does currently not '
                               'work for spinpolarized systems.')

        self.pair_distribution = self.pair.distribute_k_points_and_bands(
            b1, b2, self.kd.ibz2bz_k[self.kpts])

        self.qd = get_qdescriptor(self.kd, self.gs.atoms)

        if q0_correction:
            assert self.truncation == '2D'
            self.q0_corrector = Q0Correction(
                cell_cv=self.gd.cell_cv, bzk_kc=self.kd.bzk_kc,
                N_c=self.qd.N_c)
        else:
            self.q0_corrector = None

        self.print_parameters(kpts, b1, b2, ecut_extrapolation)
        self.fd.flush()
        self.hilbert_transform = None  # initialized when we create Chi0

    @property
    def kd(self):
        return self.gs.kd

    @property
    def gd(self):
        return self.gs.gd

    def choose_bands(self, bands, relbands):
        if bands is not None and relbands is not None:
            raise ValueError('Use bands or relbands!')

        if relbands is not None:
            bands = [self.gs.nvalence // 2 + b for b in relbands]

        if bands is None:
            bands = [0, self.nocc2]

        return bands

    def print_parameters(self, kpts, b1, b2, ecut_extrapolation):
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
        if not ecut_extrapolation:
            p('Plane wave cut-off: {0:g} eV'.format(self.ecut * Ha))
        else:
            p('Extrapolating to infinite plane wave cut-off using points at:')
            p('  [%.3f, %.3f, %.3f] eV' % tuple(self.ecut_e[:] * Ha))
        p('Number of bands: {0:d}'.format(self.nbands))
        p('Coulomb cutoff:', self.truncation)
        p('Broadening: {0:g} eV'.format(self.eta * Ha))
        p()
        p('fxc mode:', self.fxc_mode)
        p('Kernel:', self.xc)
        p('Do GW too:', self.do_GW_too)
        p()

    def get_eps_and_occs(self):
        eps_skn = np.empty(self.shape)  # KS-eigenvalues
        f_skn = np.empty(self.shape)  # occupation numbers

        b1, b2 = self.bands
        for i, k in enumerate(self.kpts):
            for s in range(self.nspins):
                u = s + k * self.nspins
                kpt = self.gs.kpt_u[u]
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

        # Reset calculation
        sigmashape = (len(self.ecut_e), *self.shape)

        self.sigmas = {fxc_mode: Sigma(sigmashape)
                       for fxc_mode in self.fxc_modes}
        # Loop over q in the IBZ:
        print('Summing all q:', file=self.fd)
        pb = ProgressBar(self.fd)
        for nQ, (ie, pd0, Wdict, q_c, m2, symop, blocks1d, Q_aGii) in \
                enumerate(self.calculate_screened_potential()):

            for progress, kpt1, kpt2 in self.pair_distribution.kpt_pairs_by_q(
                    q_c, 0, m2):
                pb.update((nQ + progress) / self.qd.mynk)

                k1 = self.kd.bz2ibz_k[kpt1.K]
                i = self.kpts.index(k1)

                self.calculate_q(ie, i, kpt1, kpt2, pd0, Wdict,
                                 symop=symop,
                                 sigmas=self.sigmas,
                                 blocks1d=blocks1d,
                                 Q_aGii=Q_aGii)
        pb.finish()

        return self.postprocess(self.sigmas, loaded)

    def postprocess(self, sigmas, loaded):
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
        return self.all_results[self.fxc_mode]

    def calculate_q(self, ie, k, kpt1, kpt2, pd0, Wdict,
                    *, symop, sigmas, blocks1d, Q_aGii):
        """Calculates the contribution to the self-energy and its derivative
        for a given set of k-points, kpt1 and kpt2."""

        N_c = pd0.gd.N_c
        i_cG = symop.apply(np.unravel_index(pd0.Q_qG[0], N_c))

        q_c = self.kd.bzk_kc[kpt2.K] - self.kd.bzk_kc[kpt1.K]

        shift0_c = symop.get_shift0(q_c, pd0.kd.bzk_kc[0])
        shift_c = kpt1.shift_c - kpt2.shift_c - shift0_c

        I_G = np.ravel_multi_index(i_cG + shift_c[:, None], N_c, 'wrap')

        G_Gv = pd0.get_reciprocal_vectors()

        pos_av = np.dot(self.pair.spos_ac, pd0.gd.cell_cv)
        M_vv = symop.get_M_vv(pd0.gd.cell_cv)

        myQ_aGii = []
        for a, Q_Gii in enumerate(Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                            np.dot(M_vv, pos_av[a]))))
            U_ii = self.gs.setups[a].R_sii[symop.symno]
            Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                           U_ii.T).transpose(1, 0, 2)
            if symop.sign == -1:
                Q_Gii = Q_Gii.conj()
            myQ_aGii.append(Q_Gii)

        if debug:
            self.check(ie, i_cG, shift0_c, N_c, q_c, myQ_aGii)

        if self.ppa:
            calculate_sigma = self.calculate_sigma_ppa
        else:
            calculate_sigma = self.calculate_sigma

        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            eps1 = kpt1.eps_n[n]
            C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                      for Qa_Gii, P1_ni in zip(myQ_aGii, kpt1.P_ani)]
            n_mG = self.pair.calculate_pair_densities(
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
        pd1 = PWDescriptor(self.ecut_e[ie], self.gd, complex, qd1)
        G_I = np.empty(N_c.prod(), int)
        G_I[:] = -1
        I1_G = pd1.Q_qG[0]
        G_I[I1_G] = np.arange(len(I0_G))
        G_G = G_I[I0_G]
        assert len(I0_G) == len(I1_G)
        assert (G_G >= 0).all()
        for a, Q_Gii in enumerate(self.pair.initialize_paw_corrections(pd1)):
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

        w_m = self.wd.get_floor_index(o_m, safe=False)
        m_inb = np.where(w_m < len(self.wd) - 1)[0]
        o1_m = np.empty(len(o_m))
        o2_m = np.empty(len(o_m))
        o1_m[m_inb] = self.wd.omega_w[w_m[m_inb]]
        o2_m[m_inb] = self.wd.omega_w[w_m[m_inb] + 1]

        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)
        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):

            if w >= len(self.wd.omega_w) - 1:
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

    def calculate_screened_potential(self):
        """Calculates the screened potential for each q-point in the 1st BZ.
        Since many q-points are related by symmetry, the actual calculation is
        only done for q-points in the IBZ and the rest are obtained by symmetry
        transformations. Results are returned as a generator to that it is not
        necessary to store a huge matrix for each q-point in the memory."""
        # The decorator $timer('W') doesn't work for generators, do we will
        # have to manually start and stop the timer here:
        self.timer.start('W')
        print('\nCalculating screened Coulomb potential', file=self.fd)
        if self.truncation is not None:
            print('Using %s truncated Coloumb potential' % self.truncation,
                  file=self.fd)

        if self.ppa:
            print('Using Godby-Needs plasmon-pole approximation:',
                  file=self.fd)
            print('  Fitting energy: i*E0, E0 = %.3f Hartee' % self.E0,
                  file=self.fd)

            # use small imaginary frequency to avoid dividing by zero:
            frequencies = [1e-10j, 1j * self.E0 * Ha]

            parameters = {'eta': 0,
                          'hilbert': False,
                          'timeordered': False}
        else:
            print('Using full frequency integration:', file=self.fd)

            frequencies = self.frequencies
            parameters = {'eta': self.eta * Ha,
                          'hilbert': True,
                          'timeordered': True}

        self.fd.flush()

        from gpaw.response.chi0 import new_frequency_descriptor
        chi_context = self.context.with_txt(self.filename + '.w.txt')
        self.wd = new_frequency_descriptor(
            self.gs, self.nbands, frequencies, fd=chi_context.fd)

        chi0calc = Chi0Calculator(wd=self.wd, pair=self.pair,
                                  nbands=self.nbands,
                                  ecut=self.ecut * Ha,
                                  intraband=False,
                                  context=chi_context,
                                  **parameters)

        if self.truncation == 'wigner-seitz':
            wstc = WignerSeitzTruncatedCoulomb(
                self.gd.cell_cv,
                self.kd.N_c,
                chi0calc.fd)
        else:
            wstc = None

        # self.wd = chi0calc.wd
        self.hilbert_transform = GWHilbertTransforms(
            self.wd.omega_w, self.eta)
        print(self.wd, file=self.fd)

        # Find maximum size of chi-0 matrices:
        nGmax = max(count_reciprocal_vectors(self.ecut, self.gd, q_c)
                    for q_c in self.qd.ibzk_kc)
        nw = len(self.wd)

        size = self.blockcomm.size

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
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            if iq <= self.last_q:
                continue

            if len(self.ecut_e) > 1:
                chi0bands = chi0calc.create_chi0(q_c, extend_head=False)
            else:
                chi0bands = None

            m1 = chi0calc.nocc1
            for ie, ecut in enumerate(self.ecut_e):
                self.timer.start('W')

                # First time calculation
                if ecut == self.ecut:
                    # Nothing to cut away:
                    m2 = self.nbands
                else:
                    m2 = int(self.vol * ecut**1.5 * 2**0.5 / 3 / pi**2)
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

                for Q_c, symop in QSymmetryOp.get_symops(self.qd, iq, q_c):
                    yield (ie, pdi, Wdict, Q_c, m2, symop,
                           blocks1d, Q_aGii)

                if self.restartfile is not None:
                    self.save_restart_file(iq)

    @property
    def fxc_modes(self):
        modes = [self.fxc_mode]
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
        chi0calc.update_chi0(chi0, m1, m2, range(self.nspins))

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
            pdi, blocks1d, W_wGG = self.dyson_and_W_old(
                wstc, iq, q_c, chi0calc, chi0, ecut, Q_aGii=chi0calc.Q_aGii,
                fxc_mode=fxc_mode)

            if self.ppa:
                W_xwGG = W_wGG  # (ppa API is nonsense)
            else:
                with self.timer('Hilbert'):
                    W_xwGG = self.hilbert_transform(W_wGG)

            Wdict[fxc_mode] = W_xwGG

        return pdi, Wdict, blocks1d, chi0calc.Q_aGii

    def dyson_and_W_new(self, wstc, iq, q_c, chi0calc, chi0, ecut):
        assert not self.ppa
        assert not self.do_GW_too
        assert ecut == chi0.pd.ecut
        assert self.fxc_mode == 'GW'

        assert not np.allclose(q_c, 0)

        nW = len(self.wd)
        nG = chi0.pd.ngmax

        from gpaw.response.wgg import Grid

        WGG = (nW, nG, nG)
        WgG_grid = Grid(
            comm=self.blockcomm,
            shape=WGG,
            cpugrid=(1, self.blockcomm.size, 1))
        assert chi0.chi0_wGG.shape == WgG_grid.myshape

        my_gslice = WgG_grid.myslice[1]

        dielectric_WgG = chi0.chi0_wGG  # XXX
        for iw, chi0_GG in enumerate(chi0.chi0_wGG):
            sqrtV_G = get_coulomb_kernel(chi0.pd,  # XXX was: pdi
                                         self.kd.N_c,
                                         truncation=self.truncation,
                                         wstc=wstc)**0.5
            e_GG = np.eye(nG) - chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            e_gG = e_GG[my_gslice]

            dielectric_WgG[iw, :, :] = e_gG

        wgg_grid = Grid(comm=self.blockcomm, shape=WGG)

        dielectric_wgg = wgg_grid.zeros(dtype=complex)
        WgG_grid.redistribute(wgg_grid, dielectric_WgG, dielectric_wgg)

        assert np.allclose(dielectric_wgg, dielectric_WgG)

        wgg_grid.invert_inplace(dielectric_wgg)

        wgg_grid.redistribute(WgG_grid, dielectric_wgg, dielectric_WgG)
        inveps_WgG = dielectric_WgG

        self.timer.start('Dyson eq.')

        for iw, inveps_gG in enumerate(inveps_WgG):
            inveps_gG -= np.identity(nG)[my_gslice]
            thing_GG = sqrtV_G * sqrtV_G[:, np.newaxis]
            inveps_gG *= thing_GG[my_gslice]

        W_WgG = inveps_WgG
        Wp_wGG = W_WgG.copy()
        Wm_wGG = W_WgG.copy()
        return chi0.pd, Wm_wGG, Wp_wGG  # not Hilbert transformed yet

    def _calculate_kernel(self, nG, iq, G2G):
        return calculate_kernel(ecut=self.ecut,
                                xcflags=self.xcflags,
                                gs=self.gs, nG=nG,
                                ns=self.nspins, iq=iq,
                                cut_G=G2G, wd=self.wd,
                                Eg=self.Eg,
                                timer=self.timer, fd=self.fd)

    def dyson_and_W_old(self, wstc, iq, q_c, chi0calc, chi0,
                        ecut, Q_aGii, fxc_mode):
        nG = chi0.pd.ngmax
        blocks1d = chi0.blocks1d

        wblocks1d = Blocks1D(self.blockcomm, len(self.wd))

        # The copy() is only required when doing GW_too, since we need
        # to run this whole thin twice.
        chi0_wGG = chi0.blockdist.redistribute(chi0.chi0_wGG.copy(), chi0.nw)

        pd = chi0.pd
        chi0_wxvG = chi0.chi0_wxvG
        chi0_wvv = chi0.chi0_wvv

        if ecut == pd.ecut:
            pdi = pd
            G2G = None

        elif ecut < pd.ecut:  # construct subset chi0 matrix with lower ecut
            pdi = PWDescriptor(ecut, pd.gd, dtype=pd.dtype,
                               kd=pd.kd)
            nG = pdi.ngmax
            blocks1d = Blocks1D(self.blockcomm, nG)
            G2G = PWMapping(pdi, pd).G2_G1
            chi0_wGG = chi0_wGG.take(G2G, axis=1).take(G2G, axis=2)

            if chi0_wxvG is not None:
                chi0_wxvG = chi0_wxvG.take(G2G, axis=3)

            if Q_aGii is not None:
                for a, Q_Gii in enumerate(Q_aGii):
                    Q_aGii[a] = Q_Gii.take(G2G, axis=0)

        if self.integrate_gamma != 0:
            reduced = (self.integrate_gamma == 2)
            V0, sqrtV0 = get_integrated_kernel(pdi,
                                               self.kd.N_c,
                                               truncation=self.truncation,
                                               reduced=reduced,
                                               N=100)
        elif self.integrate_gamma == 0 and np.allclose(q_c, 0):
            bzvol = (2 * np.pi)**3 / self.vol / self.qd.nbzkpts
            Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
            V0 = 16 * np.pi**2 * Rq0 / bzvol
            sqrtV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2

        delta_GG = np.eye(nG)

        if self.ppa:
            einv_wGG = []

        if fxc_mode == 'GW':
            fv = delta_GG
        else:
            fv = self._calculate_kernel(nG, iq, G2G)

        # Generate fine grid in vicinity of gamma
        kd = self.kd
        if np.allclose(q_c, 0) and len(chi0_wGG) > 0:
            gamma_int = GammaIntegrator(truncation=self.truncation,
                                        kd=kd, pd=pd,
                                        chi0_wvv=chi0_wvv[wblocks1d.myslice],
                                        chi0_wxvG=chi0_wxvG[wblocks1d.myslice])

        self.timer.start('Dyson eq.')

        def get_sqrtV_G(N_c, q_v=None):
            return get_coulomb_kernel(
                pdi,
                N_c,
                truncation=self.truncation,
                wstc=wstc,
                q_v=q_v)**0.5

        for iw, chi0_GG in enumerate(chi0_wGG):
            if np.allclose(q_c, 0):
                einv_GG = np.zeros((nG, nG), complex)
                for iqf in range(len(gamma_int.qf_qv)):
                    chi0_GG[0, :] = gamma_int.a0_qwG[iqf, iw]
                    chi0_GG[:, 0] = gamma_int.a1_qwG[iqf, iw]
                    chi0_GG[0, 0] = gamma_int.a_wq[iw, iqf]

                    sqrtV_G = get_sqrtV_G(kd.N_c, q_v=gamma_int.qf_qv[iqf])

                    dfc = DielectricFunctionCalculator(
                        sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                    einv_GG += dfc.get_einv_GG() * gamma_int.weight_q[iqf]
            else:
                sqrtV_G = get_sqrtV_G(kd.N_c)

                dfc = DielectricFunctionCalculator(
                    sqrtV_G, chi0_GG, mode=fxc_mode, fv_GG=fv)
                einv_GG = dfc.get_einv_GG()

            if self.ppa:
                einv_wGG.append(einv_GG - delta_GG)
            else:
                W_GG = chi0_GG
                W_GG[:] = (einv_GG - delta_GG) * (sqrtV_G *
                                                  sqrtV_G[:, np.newaxis])

                if self.q0_corrector is not None and np.allclose(q_c, 0):
                    if iw == 0:
                        print_ac = True
                    else:
                        print_ac = False
                    this_w = wblocks1d.a + iw
                    self.add_q0_correction(pdi, W_GG, einv_GG,
                                           chi0_wxvG[this_w],
                                           chi0_wvv[this_w],
                                           sqrtV_G,
                                           print_ac=print_ac)
                elif np.allclose(q_c, 0) or self.integrate_gamma != 0:
                    W_GG[0, 0] = (einv_GG[0, 0] - 1.0) * V0
                    W_GG[0, 1:] = einv_GG[0, 1:] * sqrtV_G[1:] * sqrtV0
                    W_GG[1:, 0] = einv_GG[1:, 0] * sqrtV0 * sqrtV_G[1:]

        if self.ppa:
            omegat_GG = self.E0 * np.sqrt(einv_wGG[1] /
                                          (einv_wGG[0] - einv_wGG[1]))
            R_GG = -0.5 * omegat_GG * einv_wGG[0]
            W_GG = pi * R_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            if np.allclose(q_c, 0) or self.integrate_gamma != 0:
                W_GG[0, 0] = pi * R_GG[0, 0] * V0
                W_GG[0, 1:] = pi * R_GG[0, 1:] * sqrtV_G[1:] * sqrtV0
                W_GG[1:, 0] = pi * R_GG[1:, 0] * sqrtV0 * sqrtV_G[1:]

            self.timer.stop('Dyson eq.')
            return pdi, blocks1d, [W_GG, omegat_GG]

        # XXX This creates a new, large buffer.  We could perhaps
        # avoid that.  Buffer used to exist but was removed due to #456.
        W_wGG = chi0.blockdist.redistribute(chi0_wGG, chi0.nw)

        self.timer.stop('Dyson eq.')
        return pdi, blocks1d, W_wGG

    @timer('Kohn-Sham XC-contribution')
    def calculate_ks_xc_contribution(self):
        name = self.filename + '.vxc.npy'
        fd, vxc_skn = self.read_contribution(name)
        if vxc_skn is None:
            print('Calculating Kohn-Sham XC contribution', file=self.fd)
            self.fd.flush()
            vxc_skn = vxc(self.gs, self.gs.hamiltonian.xc) / Ha
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
            exx = EXX(self.gs, kpts=self.kpts, bands=self.bands,
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
        ibzk_kc = self.kd.ibzk_kc
        for s in range(self.nspins):
            for i, ik in enumerate(self.kpts):
                print('\nk-point ' +
                      '{0} ({1}): ({2:.3f}, {3:.3f}, {4:.3f})'.format(
                          i, ik, *ibzk_kc[ik]) + '                ' +
                      self.fxc_mode, file=self.fd)
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

        x = 1 / (self.qd.nbzkpts * 2 * pi * self.vol)
        return x * sigma, x * dsigma

    def save_restart_file(self, nQ):
        sigma = self.sigmas[self.fxc_mode]
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
                'integrate_gamma': self.integrate_gamma}

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
                data['integrate_gamma'] == self.integrate_gamma):
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

    def add_q0_correction(self, pd, W_GG, einv_GG, chi0_xvG, chi0_vv,
                          sqrtV_G, print_ac=False):
        self.q0_corrector.add_q0_correction(
            pd, W_GG, einv_GG, chi0_xvG, chi0_vv,
            sqrtV_G,
            fd=self.fd if print_ac else None)
