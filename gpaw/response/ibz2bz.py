from collections.abc import Sequence
from functools import lru_cache

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
            Rotations of the PAW projections under symmetry transformations
        """
        self.kd = kd
        self.spos_ac = spos_ac
        self.R_asii = R_asii

    @classmethod
    def from_calculator(cls, calc):
        R_asii = [setup.R_sii for setup in calc.setups]
        return cls(calc.wfs.kd, calc.spos_ac, R_asii)

    def __len__(self):
        return len(self.kd.bzk_kc)

    def __getitem__(self, K):
        s = self.kd.sym_k[K]
        return IBZ2BZMap(self.get_ik_c(K),
                         self.get_rotation_matrix(s),
                         self.get_atomic_permutations(s),
                         self.get_projections_rotation_matrices(K),
                         self.get_time_reversal(K))

    def get_ik_c(self, K):
        ik = self.kd.bz2ibz_k[K]
        ik_c = self.kd.ibzk_kc[ik]
        return ik_c

    def get_rotation_matrix(self, s):
        """Coordinate rotation matrix, mapping IBZ -> K."""
        U_cc = self.kd.symmetry.op_scc[s]
        return U_cc

    def get_atomic_permutations(self, s):
        """Permutations of atomic indices in the IBZ -> K map."""
        b_a = self.kd.symmetry.a_sa[s]  # Atom a is mapped onto atom b
        return b_a

    def get_projections_rotation_matrices(self, K):
        """PAW projections rotation matrices for the IBZ -> K mapping."""
        ik = self.kd.bz2ibz_k[K]
        s = self.kd.sym_k[K]
        return self._get_projections_rotation_matrices(ik, s)

    @lru_cache
    def _get_projections_rotation_matrices(self, ik, s):
        """Correct the phase of the rotations of PAW projections.

        The rotation for symmetry "s" is corrected by a phase factor depending
        on the irreducible k-point "ik", corresponding to the Bloch phase
        associated to the atomic permutations of augmentation spheres.

        Since ik and s are integers, we can easily keep a cache of the phase
        corrected rotation matrices.
        """
        ik_c = self.kd.ibzk_kc[ik]
        U_cc = self.get_rotation_matrix(s)
        b_a = self.get_atomic_permutations(s)
        U_aii = []
        for a, R_sii in enumerate(self.R_asii):
            # The symmetry transformation maps atom "a" to a position which is
            # related to atom "b" by a lattice vector (but which does not
            # necessarily lie within the unit cell)
            b = b_a[a]
            atomic_shift_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            assert np.allclose(atomic_shift_c.round(), atomic_shift_c)
            # A phase factor is added to the rotations of the projectors in
            # order to let the projections follow the atoms under the symmetry
            # transformation
            # XXX There are some serious questions to be addressed here XXX
            # * Why does the phase factor only depend on the coordinates of the
            #   irreducible k-point?
            # * Why is the phase shift mutiplied both to the diagonal and the
            #   off-diagonal of the rotation matrices?
            phase_factor = np.exp(2j * np.pi * np.dot(ik_c, atomic_shift_c))
            U_ii = R_sii[s].T * phase_factor
            U_aii.append(U_ii)

        return U_aii

    def get_time_reversal(self, K):
        """Does the mapping IBZ -> K involve time reversal?"""
        time_reversal = self.kd.time_reversal_k[K]
        return time_reversal


class IBZ2BZMap:
    """Functionality to map orbitals from the IBZ to a specific k-point K."""

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
        related to kd.bzk_kc[K] by a reciprocal lattice vector.
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

        NB: The projections of atom a are mapped onto an atom related to atom b
        by a lattice vector.
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
