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
        XXX NOTE: At the moment it is assumed a single spin channel
        and no SOC!

        W_n1,n2;n3,n4(R=0) =
        <w^*_{n1,R=0} w_{n2, R=0} | W |w^*_{n3,R=0} w_{n4, R=0} >

        w_{n R} = V/(2pi)^3 int_{BZ} dk e^{-kR} psi^w_{nk}
        psi^w_{nk} = sum_n' U_nn'(k) psi^{KS}_{n'k}

        w^*_{n1,R=0} w_{n2, R=0} =
        C * int_{k,k' in BZ} psi^w*_{n1k} psi^w_{n2k'}

        psi^w*_{n1k} psi^w_{n2k'} =
        sum_{mm'} (U_{n1,m}(k) psi^{KS}_{m k} )^*
        U_{n2,m'}(k') psi^{KS}_{m' k'}
        
        First calculates W in KS-basis where we need the pair densities,
        then multiply with transformation matrices and sum over k and k'.
        Do in loop over IBZ with additional loop over equivalent k-points.
        """

        ibz2bz = ibz2bz_map(self.gs.kd)
        pair = chi0calc.pair
        if type(Uwan) == str:  # read w90 transformation matrix from file
            Uwan, nk, nwan = self.read_uwan(Uwan)
            assert nk == self.gs.kd.nbzkpts
            assert bandrange[1] - bandrange[0] == nwan
        
        nfreq = len(chi0calc.wd)
        Wwan_wijkl = np.zeros([nfreq, nwan, nwan, nwan, nwan], dtype=complex)
        total_k = 0

        # First calculate W in IBZ in PW basis
        # and transform to DFT eigenbasis
        for iq, q_c in enumerate(self.gs.kd.ibzk_kc):
            print('iq = ', iq)  # XXX make parprint
            # Calculate chi0 and W for IBZ k-point q
            chi0 = chi0calc.calculate(q_c)
            qpd = chi0.qpd
            W_wGG = self.wcalc.calculate_W_wGG(chi0,
                                               fxc_mode='GW',
                                               only_correlation=False)
            pawcorr = chi0calc.pawcorr
            
            # Loop over all equivalent k-points
            for iQ in ibz2bz[iq]:
                total_k += 1
                rho_kmnG, iKmQ = self.get_rho_list(spin, iQ, iq, bandrange,
                                                  pawcorr, qpd, pair)

                myKrange = self.get_k_range()
                # Double loop over BZ k-points
                for iK1 in myKrange: # range(self.gs.kd.nbzkpts):
                    for iK3 in range(self.gs.kd.nbzkpts):
                        # W in products of KS eigenstates
                        W_wijkl = np.einsum('ijk,lkm,pqm->lipjq',
                                            rho_kmnG[iK1].conj(),
                                            W_wGG,
                                            rho_kmnG[iK3],
                                            optimize='optimal')
                        
                        Wwan_wijkl += np.einsum('ia,jb,kc,ld,wabcd->wijkl',
                                                Uwan[:, :, iK1],
                                                Uwan[:, :, iK3].conj(),
                                                Uwan[:, :, iKmQ[iK1]].conj(),
                                                Uwan[:, :, iKmQ[iK3]],
                                                W_wijkl)
        # factor from BZ summation and taking from Hartree to eV
        world.sum(Wwan_wijkl)
        factor = Ha * self.gs.kd.nbzkpts**3
        Wwan_wijkl /= factor

        return Wwan_wijkl

    def get_rho_list(self, spin, iQ, iq, bandrange, pawcorr, qpd, pair):
        
        def get_k1_k2(spin, iK1, iQ, bandrange):
            # get kpt1, kpt1+q kpoint pairs used in density matrix
            kpt1 = pair.get_k_point(spin, iK1, bandrange[0], bandrange[-1])
            # Find k2 = K1 + Q
            K2_c = self.gs.kd.bzk_kc[kpt1.K] - self.gs.kd.bzk_kc[iQ]
            iK2 = self.gs.kd.where_is_q(K2_c, self.gs.kd.bzk_kc)
            kpt2 = pair.get_k_point(spin, iK2, bandrange[0], bandrange[-1])
            return kpt1, kpt2, iK2

        rho_kmnG = []
        iKmQ = []
        for iK1 in range(self.gs.kd.nbzkpts):
            kpt1, kpt2, iK2loc = get_k1_k2(spin, iK1, iQ, bandrange)
            rholoc, iqloc = self.get_density_matrix(kpt1,
                                                    kpt2,
                                                    pawcorr,
                                                    qpd,
                                                    pair)
            assert iqloc == iq
            rho_kmnG.append(rholoc)
            iKmQ.append(iK2loc)
        return rho_kmnG, iKmQ

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
        print("Reading Wannier transformation matrices from file " + seed,
              file=self.context.fd)
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
        return uwan, nk, nw1

    def get_k_range(self, rank=None):
        if rank is None:
            rank = world.rank
        nK = self.gs.kd.nbzkpts
        myKsize = -(-nK // world.size)
        myKrange = range(rank * myKsize,
                         min((rank + 1) * myKsize, nK))
        return myKrange
