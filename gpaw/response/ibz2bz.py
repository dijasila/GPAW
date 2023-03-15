import numpy as np


class IBZ2BZMapper:
    """Functionality to map data from k-points in the IBZ to the full BZ."""

    def __init__(self, kd, spos_ac, R_asii):
        """Construct the IBZ2BZMapper.

        Parameters
        ----------
        kd : KPointDescriptor
        spos_ac : np.array
            Scaled atomic positions
        R_asii : list
            Atomic symmetry rotations
        """
        self.kd = kd
        self.spos_ac = spos_ac
        self.R_asii = R_asii

    @classmethod
    def from_calculator(cls, calc):
        R_asii = [setup.R_sii for setup in calc.setups]
        return cls(calc.wfs.kd, calc.spos_ac, R_asii)

    def get_ik_c(self, K):
        ik = self.kd.bz2ibz_k[K]
        ik_c = self.kd.ibzk_kc[ik]
        return ik_c

    def get_rotation_matrix(self, K):
        """
        * U_cc is a rotation matrix.
        """
        s = self.kd.sym_k[K]
        U_cc = self.kd.symmetry.op_scc[s]
        return U_cc

    def get_time_reversal(self, K):
        """
        * time_reversal is a flag - if True, projections should be complex
          conjugated.
        """
        time_reversal = self.kd.time_reversal_k[K]
        return time_reversal

    def get_atomic_rotation_matrices(self, K):
        """
        * a_a is a list of symmetry related atom indices
        * U_aii is a list of rotation matrices for the PAW projections
        """
        s = self.kd.sym_k[K]
        U_cc = self.get_rotation_matrix(K)
        ik_c = self.get_ik_c(K)

        a_a = []
        U_aii = []
        for a, R_sii in enumerate(self.R_asii):
            b = self.kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
            U_ii = R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        return a_a, U_aii

    def map_kpoint(self, K):
        """
        Some documentation here! XXX
        """
        U_cc = self.get_rotation_matrix(K)
        time_reversal = self.get_time_reversal(K)

        # Apply symmetry operations to the irreducible k-point
        ik_c = self.get_ik_c(K)
        sign = 1 - 2 * time_reversal
        k_c = sign * U_cc @ ik_c

        return k_c

    def map_pseudo_wave(self, K, ut_R):
        """
        Some documentation here! XXX

        * k_c is an array of the relative k-point coordinates of the k-point to
          which the wave function is mapped. NB: This can lie outside the BZ.
        """
        U_cc = self.get_rotation_matrix(K)
        time_reversal = self.get_time_reversal(K)

        # Apply symmetry operations to the periodic part of the pseudo wave
        if not (U_cc == np.eye(3)).all():
            N_c = ut_R.shape
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            ut_R = ut_R.ravel()[i].reshape(N_c)
        if time_reversal:
            ut_R = ut_R.conj()

        return ut_R

    def map_projections(self, K, Ph):
        """Symmetrize the PAW projections. XXX

        NB: The projections of atom a1 are mapped onto a *different* atom a2
        according to the input map of atomic indices a1_a2."""
        time_reversal = self.get_time_reversal(K)
        a1_a2, U_aii = self.get_atomic_rotation_matrices(K)

        # First, we apply the symmetry operations to the projections one at a
        # time
        P_a2hi = []
        for a1, U_ii in zip(a1_a2, U_aii):
            P_hi = Ph[a1].copy(order='C')
            np.dot(P_hi, U_ii, out=P_hi)
            if time_reversal:
                np.conj(P_hi, out=P_hi)
            P_a2hi.append(P_hi)

        # Then, we store the symmetry mapped projectors in the projections
        # object
        for a2, P_hi in enumerate(P_a2hi):
            I1, I2 = Ph.map[a2]
            Ph.array[..., I1:I2] = P_hi

        return Ph
