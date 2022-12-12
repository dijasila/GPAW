import functools
import os
from time import ctime

import numpy as np
from ase.units import Hartree
from scipy.special import p_roots

import gpaw.mpi as mpi
from gpaw.response import timer
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair import get_gs_and_context, PairDensityCalculator


def rpa(filename, ecut=200.0, blocks=1, extrapolate=4):
    """Calculate RPA energy.

    filename: str
        Name of restart-file.
    ecut: float
        Plane-wave cutoff.
    blocks: int
        Split polarizability matrix in this many blocks.
    extrapolate: int
        Number of cutoff energies to use for extrapolation.
    """
    name, ext = filename.rsplit('.', 1)
    assert ext == 'gpw'
    from gpaw.xc.rpa import RPACorrelation
    rpa = RPACorrelation(name, name + '-rpa.dat',
                         nblocks=blocks,
                         txt=name + '-rpa.txt')
    rpa.calculate(ecut=ecut * (1 + 0.5 * np.arange(extrapolate))**(-2 / 3))


def initialize_q_points(kd, qsym):
    bzq_qc = kd.get_bz_q_points(first=True)

    if not qsym:
        ibzq_qc = bzq_qc
        weight_q = np.ones(len(bzq_qc)) / len(bzq_qc)
    else:
        U_scc = kd.symmetry.op_scc
        ibzq_qc = kd.get_ibz_q_points(bzq_qc, U_scc)[0]
        weight_q = kd.q_weights
    return bzq_qc, ibzq_qc, weight_q


class RPACalculator:
    def __init__(self, gs, *, context, filename=None,
                 skip_gamma=False, qsym=True,
                 frequencies, weights, truncation=None,
                 nblocks=1, calculate_q=None):
        self.gs = gs
        self.context = context

        self.omega_w = frequencies / Hartree
        self.weight_w = weights / Hartree

        # TODO: We should avoid this requirement.
        assert len(self.omega_w) % nblocks == 0

        self.nblocks = nblocks

        self.truncation = CoulombKernel(truncation, gs)
        self.skip_gamma = skip_gamma

        # We should actually have a kpoint descriptor for the qpoints.
        # We are badly failing at making use of the existing tools by reducing
        # the qpoints to dumb arrays.
        self.bzq_qc, self.ibzq_qc, self.weight_q = initialize_q_points(
            gs.kd, qsym)

        self.filename = filename

        if calculate_q is None:
            calculate_q = self.calculate_q_rpa
        self.calculate_q = calculate_q

    def read(self, ecut_i, filename):
        with open(filename) as fd:
            lines = fd.readlines()[1:]

        n = 0
        energy_qi = []
        nq = len(lines) // len(ecut_i)
        for q_c in self.ibzq_qc[:nq]:
            energy_qi.append([])
            for ecut in ecut_i:
                current_inputs = np.array([*q_c, ecut * Hartree])
                numbers_from_file = [float(x) for x in lines[n].split()]
                previous_inputs = numbers_from_file[:-1]

                if not np.allclose(current_inputs, previous_inputs):
                    # Energies are not reusable since input parameters
                    # have changed
                    return []

                energy = numbers_from_file[-1]
                energy_qi[-1].append(energy / Hartree)
                n += 1

        return energy_qi

    def energies_to_string(self, energy_qi, ecut_i):
        lines = []
        app = lines.append
        app('q1 q2 q3 E_cut E_c(q)')
        for energy_i, q_c in zip(energy_qi, self.ibzq_qc):
            for energy, ecut in zip(energy_i, ecut_i):
                tokens = [repr(num) for num in
                          (*q_c, ecut * Hartree, energy * Hartree)]
                app(' '.join(tokens))

    def write(self, energy_qi, ecut_i):
        txt = self.energies_to_string(energy_qi, ecut_i)
        if self.context.world.rank == 0 and self.filename:
            with open(self.filename, 'w') as fd:
                print(txt, file=fd)

    def calculate(self, ecut, nbands=None, spin=False):
        """Calculate RPA correlation energy for one or several cutoffs.

        ecut: float or list of floats
            Plane-wave cutoff(s) in eV.
        nbands: int
            Number of bands (defaults to number of plane-waves).
        spin: bool
            Separate spin in response function.
            (Only needed for beyond RPA methods that inherit this function).
        """

        p = functools.partial(self.context.print, flush=False)

        if isinstance(ecut, (float, int)):
            ecut = ecut * (1 + 0.5 * np.arange(6))**(-2 / 3)
        ecut_i = np.asarray(np.sort(ecut)) / Hartree
        ecutmax = max(ecut_i)

        if nbands is None:
            p('Response function bands : Equal to number of plane waves')
        else:
            p('Response function bands : %s' % nbands)
        p('Plane wave cutoffs (eV) :', end='')
        for e in ecut_i:
            p(' {0:.3f}'.format(e * Hartree), end='')
        p()
        p(self.truncation.description())
        self.context.print('')

        if self.filename and os.path.isfile(self.filename):
            energy_qi = self.read(ecut_i, self.filename)
            self.context.print(
                'Read %d q-points from file: %s\n' % (len(energy_qi)))

            self.context.world.barrier()

        wd = FrequencyDescriptor(1j * self.omega_w)

        pair = PairDensityCalculator(
            self.gs,
            context=self.context.with_txt('chi0.txt'),
            nblocks=self.nblocks)

        chi0calc = Chi0Calculator(wd=wd,
                                  pair=pair,
                                  eta=0.0,
                                  intraband=False,
                                  hilbert=False,
                                  ecut=ecutmax * Hartree)

        self.blockcomm = chi0calc.blockcomm

        energy_qi = []
        nq = len(energy_qi)

        self.context.timer.start('RPA')

        for q_c in self.ibzq_qc[nq:]:
            if np.allclose(q_c, 0.0) and self.skip_gamma:
                energy_qi.append(len(ecut_i) * [0.0])
                self.write(energy_qi, ecut_i)
                p('Not calculating E_c(q) at Gamma')
                p()
                continue

            chi0_s = [chi0calc.create_chi0(q_c)]
            if spin:
                chi0_s.append(chi0calc.create_chi0(q_c))

            pd = chi0_s[0].pd
            nG = pd.ngmax

            # First not completely filled band:
            m1 = chi0calc.nocc1
            p('# %s  -  %s' % (len(energy_qi), ctime().split()[-2]))
            p('q = [%1.3f %1.3f %1.3f]' % tuple(q_c))

            energy_i = []
            for ecut in ecut_i:
                if ecut == ecutmax:
                    # Nothing to cut away:
                    cut_G = None
                    m2 = nbands or nG
                else:
                    cut_G = np.arange(nG)[pd.G2_qG[0] <= 2 * ecut]
                    m2 = len(cut_G)

                p('E_cut = %d eV / Bands = %d:' % (ecut * Hartree, m2),
                  end='\n', flush=True)

                energy = self.calculate_q(chi0calc,
                                          chi0_s,
                                          m1, m2, cut_G)

                energy_i.append(energy)
                m1 = m2

                a = 1 / chi0calc.kncomm.size
                if ecut < ecutmax and a != 1.0:
                    # Chi0 will be summed again over chicomm, so we divide
                    # by its size:
                    for chi0 in chi0_s:
                        chi0.chi0_wGG *= a
                    # if chi0_swxvG is not None:
                    #     chi0_swxvG *= a
                    #     chi0_swvv *= a

            energy_qi.append(energy_i)
            self.write(energy_qi, ecut_i)
            p()

        e_i = np.dot(self.weight_q, np.array(energy_qi))
        p('==========================================================')
        p()
        p('Total correlation energy:')
        for e_cut, e in zip(ecut_i, e_i):
            p('%6.0f:   %6.4f eV' % (e_cut * Hartree, e * Hartree))
        p()

        if len(e_i) > 1:
            self.extrapolate(e_i, ecut_i)

        p('Calculation completed at: ', ctime())
        p()

        self.context.timer.stop('RPA')
        self.context.write_timer()

        return e_i * Hartree

    @timer('chi0(q)')
    def calculate_q_rpa(self, chi0calc, chi0_s,
                        m1, m2, cut_G):
        chi0 = chi0_s[0]
        chi0calc.update_chi0(chi0,
                             m1, m2, spins='all')

        self.context.print('E_c(q) = ', end='', flush=False)

        chi0_wGG = chi0.distribute_as('wGG')

        kd = self.gs.kd
        if not chi0.pd.kd.gamma:
            e = self.calculate_energy_rpa(chi0.pd, chi0_wGG, cut_G)
            self.context.print('%.3f eV' % (e * Hartree))
        else:
            from gpaw.response.gamma_int import GammaIntegrator
            from gpaw.response.pw_parallelization import Blocks1D

            wblocks = Blocks1D(self.blockcomm, len(self.omega_w))
            gamma_int = GammaIntegrator(
                truncation=self.truncation._truncation,
                kd=kd,
                pd=chi0.pd,
                chi0_wvv=chi0.chi0_wvv[wblocks.myslice],
                chi0_wxvG=chi0.chi0_wxvG[wblocks.myslice])

            e = 0
            for iqf in range(len(gamma_int.qf_qv)):
                for iw in range(wblocks.nlocal):
                    gamma_int.set_appendages(chi0_wGG[iw], iw, iqf)
                ev = self.calculate_energy_rpa(chi0.pd, chi0_wGG, cut_G,
                                               q_v=gamma_int.qf_qv[iqf])
                e += ev * gamma_int.weight_q[iqf]
            self.context.print('%.3f eV' % (e * Hartree))

        return e

    @timer('Energy')
    def calculate_energy_rpa(self, pd, chi0_wGG, cut_G, q_v=None):
        """Evaluate correlation energy from chi0."""

        sqrtV_G = self.truncation.coulomb_kernel(pd, q_v)

        if cut_G is not None:
            sqrtV_G = sqrtV_G[cut_G]
        nG = len(sqrtV_G)

        e_w = []
        for chi0_GG in chi0_wGG:
            if cut_G is not None:
                chi0_GG = chi0_GG.take(cut_G, 0).take(cut_G, 1)

            e_GG = np.eye(nG) - chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            e = np.log(np.linalg.det(e_GG)) + nG - np.trace(e_GG)
            e_w.append(e.real)

        E_w = np.zeros_like(self.omega_w)
        self.blockcomm.all_gather(np.array(e_w), E_w)
        energy = np.dot(E_w, self.weight_w) / (2 * np.pi)
        self.E_w = E_w
        return energy

    def extrapolate(self, e_i, ecut_i):
        self.context.print('Extrapolated energies:', flush=False)
        ex_i = []
        for i in range(len(e_i) - 1):
            e1, e2 = e_i[i:i + 2]
            x1, x2 = ecut_i[i:i + 2]**-1.5
            ex = (e1 * x2 - e2 * x1) / (x2 - x1)
            ex_i.append(ex)

            self.context.print('  %4.0f -%4.0f:  %5.3f eV' %
                               (ecut_i[i] * Hartree, ecut_i[i + 1]
                                * Hartree, ex * Hartree), flush=False)
        self.context.print('')

        return e_i * Hartree


def get_gauss_legendre_points(nw=16, frequency_max=800.0, frequency_scale=2.0):
    y_w, weights_w = p_roots(nw)
    y_w = y_w.real
    ys = 0.5 - 0.5 * y_w
    ys = ys[::-1]
    w = (-np.log(1 - ys))**frequency_scale
    w *= frequency_max / w[-1]
    alpha = (-np.log(1 - ys[-1]))**frequency_scale / frequency_max
    transform = (-np.log(1 - ys))**(frequency_scale - 1) \
        / (1 - ys) * frequency_scale / alpha
    return w, weights_w * transform / 2


class RPACorrelation(RPACalculator):
    def __init__(self, calc, xc='RPA',
                 nlambda=None,
                 nfrequencies=16, frequency_max=800.0, frequency_scale=2.0,
                 frequencies=None, weights=None,
                 world=mpi.world, txt='-', **kwargs):
        """Creates the RPACorrelation object

        calc: str or calculator object
            The string should refer to the .gpw file contaning KS orbitals
        xc: str
            Exchange-correlation kernel. This is only different from RPA when
            this object is constructed from a different module - e.g. fxc.py
        filename: str
            txt output
        skip_gamme: bool
            If True, skip q = [0,0,0] from the calculation
        qsym: bool
            Use symmetry to reduce q-points
        nlambda: int
            Number of lambda points. Only used for numerical coupling
            constant integration involved when called from fxc.py
        nfrequencies: int
            Number of frequency points used in the Gauss-Legendre integration
        frequency_max: float
            Largest frequency point in Gauss-Legendre integration
        frequency_scale: float
            Determines density of frequency points at low frequencies. A slight
            increase to e.g. 2.5 or 3.0 improves convergence wth respect to
            frequency points for metals
        frequencies: list
            List of frequancies for user-specified frequency integration
        weights: list
            list of weights (integration measure) for a user specified
            frequency grid. Must be specified and have the same length as
            frequencies if frequencies is not None
        truncation: str or None
            Coulomb truncation scheme. Can be None, 'wigner-seitz', or '2D'
        world: communicator
        nblocks: int
            Number of parallelization blocks. Frequency parallelization
            can be specified by setting nblocks=nfrequencies and is useful
            for memory consuming calculations
        txt: str
            txt file for saving and loading contributions to the correlation
            energy from different q-points
        """
        gs, context = get_gs_and_context(calc=calc, txt=txt, world=world,
                                         timer=None)

        if frequencies is None:
            frequencies, weights = get_gauss_legendre_points(nfrequencies,
                                                             frequency_max,
                                                             frequency_scale)
            user_spec = False
        else:
            assert weights is not None
            user_spec = True

        super().__init__(gs=gs, context=context,
                         frequencies=frequencies, weights=weights,
                         **kwargs)

        self.print_initialization(xc, frequency_scale, nlambda, user_spec)

    def print_initialization(self, xc, frequency_scale, nlambda, user_spec):
        p = functools.partial(self.context.print, flush=False)
        p('----------------------------------------------------------')
        p('Non-self-consistent %s correlation energy' % xc)
        p('----------------------------------------------------------')
        p('Started at:  ', ctime())
        p()
        p('Atoms                          :',
          self.gs.atoms.get_chemical_formula(mode='hill'))
        p('Ground state XC functional     :', self.gs.xcname)
        p('Valence electrons              :', self.gs.nvalence)
        p('Number of bands                :', self.gs.bd.nbands)
        p('Number of spins                :', self.gs.nspins)
        p('Number of k-points             :', len(self.gs.kd.bzk_kc))
        p('Number of irreducible k-points :', len(self.gs.kd.ibzk_kc))
        p('Number of q-points             :', len(self.bzq_qc))
        p('Number of irreducible q-points :', len(self.ibzq_qc))
        p()
        for q, weight in zip(self.ibzq_qc, self.weight_q):
            p('    q: [%1.4f %1.4f %1.4f] - weight: %1.3f' %
              (q[0], q[1], q[2], weight))
        p()
        p('----------------------------------------------------------')
        p('----------------------------------------------------------')
        p()
        if nlambda is None:
            p('Analytical coupling constant integration')
        else:
            p('Numerical coupling constant integration using', nlambda,
              'Gauss-Legendre points')
        p()
        p('Frequencies')
        if not user_spec:
            p('    Gauss-Legendre integration with %s frequency points' %
              len(self.omega_w))
            p('    Transformed from [0,oo] to [0,1] using e^[-aw^(1/B)]')
            p('    Highest frequency point at %5.1f eV and B=%1.1f' %
              (self.omega_w[-1] * Hartree, frequency_scale))
        else:
            p('    User specified frequency integration with',
              len(self.omega_w), 'frequency points')
        p()
        p('Parallelization')
        p('    Total number of CPUs          : % s' % self.context.world.size)
        p('    G-vector decomposition        : % s' % self.nblocks)
        p('    K-point/band decomposition    : % s' %
          (self.context.world.size // self.nblocks))
        self.context.print('')
