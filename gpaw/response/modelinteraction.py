import numpy as np
from ase.units import Ha
from gpaw.mpi import world

def ibz2bz_map(qd):
    """ Maps each k in BZ to corresponding k in IBZ. """
    out_map = [[] for _ in range(qd.nibzkpts)]
    for iK in range(qd.nbzkpts):
        ik = qd.bz2ibz_k[iK]
        out_map[ik].append(iK)
    return out_map


class ModelInteraction:

    def __init__(self, wcalc):
        self.wcalc = wcalc
        self.gs = wcalc.gs
        self.context = self.wcalc.context
        self.qd = self.wcalc.qd
        
    def calc_in_Wannier(self, chi0calc, Uwan, bandrange, spin = 0):
        """Calculates the screened interaction matrix in Wannier basis.
        NOTE: Does not work with SOC!

        W_n1,n2;n3,n4(R=0) =
        <w^*_{n1,R=0} w_{n2, R=0} | W |w^*_{n3,R=0} w_{n4, R=0} >

        w_{n R} = V/(2pi)^3 int_{BZ} dk e^{-kR} psi^w_{nk}
        psi^w_{nk} = sum_m U_nm(k) psi^{KS}_{mk}

        In this implementation we compute W_n1,n2;n3,n4(R=0) efficiently
        by expressing it in terms of the reduced Wannier density matrices

        A_{n1,n2,q} = \sum_{k} rhowan^{n1,k1}_{n2, k-q}

        where the Wannier density matrix is calculated from the usual density
        matrix rho^{m1,k}_{m2 k-q} = < psi_{m1,k} | e^{i(q+G)r} | psi_{m2, k-q} >
        and the Wannier transformation matrices U_{nm}(k) as

        rhowan^{n1, k}_{n2, k-q} = 
        = \sum_{m1, m5} U^*_{n1,m1}(k) U_{n2, m2}(k-q) rho^{m1,k}_{m2 k-q}

        """

        ibz2bz = ibz2bz_map(self.gs.kd)
        pair = chi0calc.pair
        if type(Uwan) == str:  # read w90 transformation matrix from file
            Uwan, nk, nwan, nband = self.read_uwan(Uwan)
        else:
            nk = Uwan.shape[2]
            nband = Uwan.shape[1]
            nwan = Uwan.shape[0]

        assert nk == self.gs.kd.nbzkpts
        assert bandrange[1] - bandrange[0] == nband
    
        nfreq = len(chi0calc.wd)

        # Variable that will store the screened interaction in Wannier basis
        Wwan_wijkl = np.zeros([nfreq, nwan, nwan, nwan, nwan], dtype=complex)
        total_k = 0


        # First calculate W in IBZ in PW basis
        # and then transform to Wannier basis
        for iq, q_c in enumerate(self.gs.kd.ibzk_kc):
            self.context.print('iq = ', iq, '/', self.gs.kd.nibzkpts)
            # Calculate chi0 and W for IBZ k-point q
            self.context.print('calculating chi0...')
            chi0 = chi0calc.calculate(q_c)
            qpd = chi0.qpd
            self.context.print('calculating W_wGG...')
            W_wGG = self.wcalc.calculate_W_wGG(chi0,
                                               fxc_mode='GW',
                                               only_correlation=False)
            pawcorr = chi0calc.pawcorr

            self.context.print('Projecting to localized Wannier basis...')
            # Loop over all equivalent k-points
            for iQ in ibz2bz[iq]:
                total_k += 1
                A_mnG = self.get_reduced_wannier_density_matrix(spin,
                                                                iQ,
                                                                iq,
                                                                bandrange,
                                                                pawcorr,
                                                                qpd,
                                                                pair,
                                                                Uwan)
                Wwan_wijkl += np.einsum('ijk,lkm,pqm->lipjq',
                                            A_mnG.conj(),
                                            W_wGG,
                                            A_mnG,
                                            optimize='optimal')

        # factor from BZ summation and taking from Hartree to eV
        factor = Ha * self.gs.kd.nbzkpts**3
        Wwan_wijkl /= factor

        return Wwan_wijkl

    def get_reduced_wannier_density_matrix(self, spin, iQ, iq, bandrange,
                                           pawcorr, qpd, pair, Uwan):
        """
        Returns sum_k sum_(m1,m2) U_{n1m1}* U_{n2m2} rho^{m1 k}_{m2 k-q}(G)
        where rho is the usual density matrix and U are wannier tranformation
        matrices.
        """
        def get_k1_k2(spin, iK1, iQ, bandrange):
            # get kpt1, kpt1-q kpoint pairs used in density matrix
            kpt1 = pair.get_k_point(spin, iK1, bandrange[0], bandrange[-1])
            # Find k2 = K1 + Q
            K2_c = self.gs.kd.bzk_kc[kpt1.K] - self.gs.kd.bzk_kc[iQ]
            iK2 = self.gs.kd.where_is_q(K2_c, self.gs.kd.bzk_kc)
            kpt2 = pair.get_k_point(spin, iK2, bandrange[0], bandrange[-1])
            return kpt1, kpt2, iK2
        nG = qpd.ngmax
        nwan = Uwan.shape[0]
        A_mnG = np.zeros([nwan, nwan, nG], dtype=complex)
        for iK1 in range(self.gs.kd.nbzkpts):
            kpt1, kpt2, iK2 = get_k1_k2(spin, iK1, iQ, bandrange)
            # Caluclate density matrix
            rholoc, iqloc = self.get_density_matrix(kpt1,
                                                    kpt2,
                                                    pawcorr,
                                                    qpd,
                                                    pair)
            assert iqloc == iq
            # Rotate to Wannier basis and sum to get reduced Wannier density matrix
            # A
            A_mnG  += np.einsum('ia,jb,abG->ijG',
                                Uwan[:, :, iK1].conj(),
                                Uwan[:, :, iK2],
                                rholoc)

        return A_mnG

    def get_density_matrix(self, kpt1, kpt2, pawcorr, qpd, pair):
        from gpaw.response.g0w0 import QSymmetryOp, get_nmG
        
        symop, iq = QSymmetryOp.get_symop_from_kpair(
            self.gs.kd, self.qd, kpt1, kpt2)
        nG = qpd.ngmax
        pawcorr, I_G = symop.apply_symop_q(
            qpd, pawcorr, kpt1, kpt2)

        rho_mnG = np.zeros((len(kpt1.eps_n), len(kpt2.eps_n), nG),
                           complex)
        for m in range(len(rho_mnG)):
            rho_mnG[m] = get_nmG(kpt1, kpt2, pawcorr, m, qpd, I_G, pair)
        return rho_mnG, iq

    def read_uwan(self, seed):
        if "_u.mat" not in seed:
            seed += "_u.mat"
        self.context.print("Reading Wannier transformation matrices from file " + seed)
              #file=self.context.fd)
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
