import numpy as np
from ase.units import Ha
from gpaw.mpi import world
from gpaw.response import ResponseContext
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.screened_interaction import initialize_w_calculator
from gpaw.response import timer


def ibz2bz_map(qd):
    """ Maps each k in BZ to corresponding k in IBZ. """
    out_map = [[] for _ in range(qd.nibzkpts)]
    for iK in range(qd.nbzkpts):
        ik = qd.bz2ibz_k[iK]
        out_map[ik].append(iK)
    return out_map


def initialize_w_model(chi0calc, truncation=None, txt='w_model.out',
                       world=world, timer=None, integrate_gamma=0,
                       q0_correction=False):
    gs = chi0calc.gs
    wcontext = ResponseContext(txt=txt,
                               comm=world, timer=timer)
    coulomb = CoulombKernel.from_gs(gs, truncation=truncation)
    wcalc = initialize_w_calculator(chi0calc,
                                    wcontext,
                                    coulomb=coulomb,
                                    xc='RPA',
                                    integrate_gamma=integrate_gamma,
                                    q0_correction=q0_correction)
    return ModelInteraction(wcalc)


class ModelInteraction:

    def __init__(self, wcalc):
        self.wcalc = wcalc
        self.gs = wcalc.gs
        self.context = self.wcalc.context
        self.qd = self.wcalc.qd

    @timer('Calculate W in Wannier')
    def calc_in_Wannier(self, chi0calc, Uwan, bandrange, spin=0):
        """Calculates the screened interaction matrix in Wannier basis.
        NOTE: Does not work with SOC!

        W_n1,n2;n3,n4(R=0) =
        <w^*_{n1,R=0} w_{n2, R=0} | W |w^*_{n3,R=0} w_{n4, R=0} >

        w_{n R} = V/(2pi)^3 int_{BZ} dk e^{-kR} psi^w_{nk}
        psi^w_{nk} = sum_m U_nm(k) psi^{KS}_{mk}

        In this implementation we compute W_n1,n2;n3,n4(R=0) efficiently
        by expressing it in terms of the reduced Wannier density matrices

        A_{n1,n2,q} = sum_{k} rhowan^{n1,k}_{n2, k-q}

        where the Wannier density matrix is calculated from the usual
        density matrix
        rho^{m1,k}_{m2 k-q} = < psi_{m1,k} | e^{i(q+G)r} | psi_{m2, k-q} >
        and the Wannier transformation matrices U_{nm}(k) as

        rhowan^{n1, k}_{n2, k-q} =
        = sum_{m1, m5} U^*_{n1,m1}(k) U_{n2, m2}(k-q) rho^{m1,k}_{m2 k-q}
        """

        ibz2bz = ibz2bz_map(self.gs.kd)
        pair_calc = chi0calc.pair_calc
        pair_factory = chi0calc.kptpair_factory
        if isinstance(Uwan,str):  # read w90 transformation matrix from file
            Uwan, nk, nwan, nband = self.read_uwan(Uwan)
        else:
            nk = Uwan.shape[2]
            nband = Uwan.shape[1]
            nwan = Uwan.shape[0]

        assert nk == self.gs.kd.nbzkpts
        assert bandrange[1] - bandrange[0] == nband
        nfreq = len(chi0calc.chi0_body_calc.wd)

        # Variable that will store the screened interaction in Wannier basis
        Wwan_wijkl = np.zeros([nfreq, nwan, nwan, nwan, nwan], dtype=complex)

        for iq, q_c in enumerate(self.gs.kd.ibzk_kc):
            self.context.print('iq = ', iq, '/', self.gs.kd.nibzkpts)
            # Calculate chi0 and W for IBZ k-point q
            self.context.print('calculating chi0...')

            self.context.timer.start('chi0')
            chi0 = chi0calc.calculate(q_c)
            self.context.timer.stop('chi0')

            qpd = chi0.qpd
            self.context.print('calculating W_wGG...')
            W_wGG = self.wcalc.calculate_W_wGG(chi0,
                                               fxc_mode='GW',
                                               only_correlation=False)
            pawcorr = chi0calc.chi0_body_calc.pawcorr

            self.context.print('Projecting to localized Wannier basis...')
            # Loop over all equivalent k-points
            for iQ in ibz2bz[iq]:
                Q_c = self.wcalc.qd.bzk_kc[iQ]
                assert self.wcalc.qd.where_is_q(Q_c,
                                                self.wcalc.qd.bzk_kc) == iQ

                A_mnG = self.get_reduced_wannier_density_matrix(spin,
                                                                Q_c,
                                                                iq,
                                                                bandrange,
                                                                pawcorr,
                                                                qpd,
                                                                pair_calc,
                                                                pair_factory,
                                                                Uwan)
                if self.qd.time_reversal_k[iQ]:
                    # TR corresponds to complex conjugation
                    A_mnG = A_mnG.conj()

                Wwan_wijkl += np.einsum('ijk,lkm,pqm->lipjq',
                                        A_mnG.conj(),
                                        W_wGG,
                                        A_mnG,
                                        optimize='optimal')

        # factor from BZ summation and taking from Hartree to eV
        factor = Ha * self.gs.kd.nbzkpts**3
        Wwan_wijkl /= factor
        self.context.write_timer()

        return Wwan_wijkl

    @timer('get_reduced_wannier_density_matrix')
    def get_reduced_wannier_density_matrix(self, spin, Q_c, iq, bandrange,
                                           pawcorr, qpd, pair_calc,
                                           pair_factory, Uwan):
        """
        Returns sum_k sum_(m1,m2) U_{n1m1}* U_{n2m2} rho^{m1 k}_{m2 k-q}(G)
        where rho is the usual density matrix and U are wannier tranformation
        matrices.
        """
        nG = qpd.ngmax
        nwan = Uwan.shape[0]
        A_mnG = np.zeros([nwan, nwan, nG], dtype=complex)

        # Parallell sum over k-points
        for iK1 in self.myKrange():
            kpt1 = pair_factory.get_k_point(spin, iK1,
                                            bandrange[0],
                                            bandrange[1])
            iK2 = self.gs.kd.find_k_plus_q(Q_c, [kpt1.K])[0]
            kpt2 = pair_factory.get_k_point(spin, iK2,
                                            bandrange[0],
                                            bandrange[1])

            # Calculate density matrix
            rholoc, iqloc, sign = self.get_density_matrix(kpt1,
                                                          kpt2,
                                                          pawcorr,
                                                          qpd,
                                                          pair_calc)
            assert iqloc == iq

            # Rotate to Wannier basis and sum to get reduced Wannier
            # density matrix A
            A_mnG += np.einsum('ia,jb,abG->ijG',
                               Uwan[:, :, iK1].conj(),
                               Uwan[:, :, iK2],
                               rholoc)
        world.sum(A_mnG)
        return A_mnG

    def get_density_matrix(self, kpt1, kpt2, pawcorr, qpd, pair_calc):
        from gpaw.response.g0w0 import QSymmetryOp, get_nmG

        symop, iq = QSymmetryOp.get_symop_from_kpair(
            self.gs.kd, self.qd, kpt1, kpt2)
        nG = qpd.ngmax
        mypawcorr, I_G = symop.apply_symop_q(
            qpd, pawcorr, kpt1, kpt2)

        rho_mnG = np.zeros((len(kpt1.eps_n), len(kpt2.eps_n), nG),
                           complex)
        for m in range(len(rho_mnG)):
            rho_mnG[m] = get_nmG(kpt1, kpt2, mypawcorr, m, qpd, I_G, pair_calc)

        return rho_mnG, iq, symop.sign

    def read_uwan(self, seed):
        if "_u.mat" not in seed:
            seed += "_u.mat"
        self.context.print(
            "Reading Wannier transformation matrices from file " + seed)
        f = open(seed, "r")
        kd = self.gs.kd
        f.readline()  # first line is a comment
        nk, nw1, nw2 = [int(i) for i in f.readline().split()]
        assert nw1 == nw2
        assert nk == kd.nbzkpts
        uwan = np.empty([nw1, nw2, nk], dtype=complex)
        iklist = []  # list to store found iks
        for ik1 in range(nk):
            f.readline()  # empty line
            K_c = [float(rdum) for rdum in f.readline().split()]
            assert np.allclose(np.array(K_c), self.gs.kd.bzk_kc[ik1])
            ik = kd.where_is_q(K_c, kd.bzk_kc)
            iklist.append(ik)
            for ib1 in range(nw1):
                for ib2 in range(nw2):
                    rdum1, rdum2 = [float(rdum) for rdum in
                                    f.readline().split()]
                    uwan[ib1, ib2, ik] = complex(rdum1, rdum2)
        assert set(iklist) == set(range(nk))  # check that all k:s were found
        return uwan, nk, nw1, nw2

    def myKrange(self, rank=None):
        if rank is None:
            rank = world.rank
        nK = self.gs.kd.nbzkpts
        myKsize = -(-nK // world.size)
        myKrange = range(rank * myKsize,
                         min((rank + 1) * myKsize, nK))
        # myKsize = len(myKrange)
        return myKrange
