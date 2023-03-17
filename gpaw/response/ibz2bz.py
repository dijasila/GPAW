from collections.abc import Sequence

import numpy as np


class IBZ2BZMaps(Sequence):
    """Sequence of data maps from k-points in the IBZ to the full BZ.

    The sequence is indexed by the BZ index K."""

    def __init__(self, kd, spos_ac, R_asii):
        """Construct the IBZ2BZMapper.

        Parameters
        ----------
        kd : KPointDescriptor
        spos_ac : np.array
            Scaled atomic positions
        R_asii : list
            Atomic rotations under the symmetry transformations
        """
        self.kd = kd
        self.U_kaii = self.generate_projections_rotation_matrices(
            spos_ac, R_asii)

    @classmethod
    def from_calculator(cls, calc):
        R_asii = [setup.R_sii for setup in calc.setups]
        return cls(calc.wfs.kd, calc.spos_ac, R_asii)

    def generate_projections_rotation_matrices(self, spos_ac, R_asii):
        """Rotation matrices for the PAW projections of every IBZ -> K map."""
        U_kaii = []
        for K in range(len(self)):
            s = self.kd.sym_k[K]
            ik_c = self.get_ik_c(K)
            U_cc = self.get_rotation_matrix(K)
            b_a = self.get_atomic_permutations(K)
            U_aii = []
            for a, R_sii in enumerate(R_asii):
                b = b_a[a]
                S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
                x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
                U_ii = R_sii[s].T * x
                U_aii.append(U_ii)
            U_kaii.append(U_aii)

        return U_kaii

    def __len__(self):
        return len(self.kd.bzk_kc)

    def __getitem__(self, K):
        return IBZ2BZMap(self.get_ik_c(K),
                         self.get_rotation_matrix(K),
                         self.get_atomic_permutations(K),
                         self.U_kaii[K],
                         self.get_time_reversal(K))

    def get_ik_c(self, K):
        ik = self.kd.bz2ibz_k[K]
        ik_c = self.kd.ibzk_kc[ik]
        return ik_c

    def get_rotation_matrix(self, K):
        """Coordinate rotation matrix, mapping IBZ -> K."""
        s = self.kd.sym_k[K]
        U_cc = self.kd.symmetry.op_scc[s]
        return U_cc

    def get_atomic_permutations(self, K):
        """Permutations of atomic indices in the IBZ -> K map."""
        s = self.kd.sym_k[K]
        # Atom a is mapped onto atom b.
        b_a = self.kd.symmetry.a_sa[s]
        return b_a

    def get_time_reversal(self, K):
        """Does the mapping IBZ -> K involve time reversal?"""
        time_reversal = self.kd.time_reversal_k[K]
        return time_reversal


class IBZ2BZMap:
    """
    Some documentation here! XXX
    """

    def __init__(self, ik_c, U_cc, b_a, U_aii, time_reversal):
        """Construct the IBZ2BZMap."""
        self.ik_c = ik_c

        self.U_cc = U_cc
        self.b_a = b_a
        self.U_aii = U_aii
        self.time_reversal = time_reversal

    def map_kpoint(self):
        """Get the relative k-point coordinates after the IBZ -> K mapping.

        NB: The mapped k-point can lie outside the BZ, but will always be
        related to self.kd.bzk_kc[K] by a reciprocal lattice vector.
        """
        # Apply symmetry operations to the irreducible k-point
        sign = 1 - 2 * self.time_reversal
        k_c = sign * self.U_cc @ self.ik_c

        return k_c

    def map_pseudo_wave(self, ut_R):
        """Map the periodic part of the pseudo wave from the IBZ -> K.

        The mapping takes place on the coarse real-space grid.

        NB: The k-point corresponding to the output ut_R does not necessarily
        lie within the BZ, see map_kpoint().
        """
        # Apply symmetry operations to the periodic part of the pseudo wave
        if not (self.U_cc == np.eye(3)).all():
            N_c = ut_R.shape
            i_cr = np.dot(self.U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            utout_R = ut_R.ravel()[i].reshape(N_c)
        else:
            utout_R = ut_R.copy()
        if self.time_reversal:
            utout_R = utout_R.conj()

        assert utout_R is not ut_R,\
            "We don't want the output array to point back at the input array"

        return utout_R

    def map_projections(self, projections):
        """Perform IBZ -> K mapping of the PAW projections.

        NB: The projections of atom b may be mapped onto *another* atom a.
        """
        mapped_projections = projections.new()
        for a, (b, U_ii) in enumerate(zip(self.b_a, self.U_aii)):
            # Map projections
            Pin_ni = projections[b]
            Pout_ni = Pin_ni @ U_ii
            if self.time_reversal:
                Pout_ni = np.conj(Pout_ni)

            # Store output projections
            I1, I2 = mapped_projections.map[a]
            mapped_projections.array[..., I1:I2] = Pout_ni

        return mapped_projections
