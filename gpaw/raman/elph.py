import numpy as np
import ase.units as units
from ase.utils.filecache import MultiFileJSONCache
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.mpi import world
from gpaw.typing import ArrayND


class EPC(ElectronPhononCoupling):
    """Modify ElectronPhononCoupling to fit Raman stuff better.

    Primarily, always save the supercell matrix in separate files,
    so that calculation of the elph matrix can be better parallelised.
    """

    def calculate_supercell_matrix(self, calc, name='supercell', filter=None,
                                   include_pseudo=True):
        """
        Calculate elph supercell matrix.

        This is a necessary intermediary step before calculating the electron-
        phonon matrix.

        The Raman version always uses dump=2 when calling
        the ElectronPhononCoupling routine.

        Parameters
        ----------
        calc: GPAW
            Converged ground-state calculation. Same grid as before.
        name: str
            User specified name for supercell cache. Default: 'supercell'
        filter: str
            Fourier filter atomic gradients of the effective potential. The
            specified components (``normal`` or ``umklapp``) are removed
            (default: None).
        include_pseudo: bool
            Include the contribution from the psedupotential in the atomic
            gradients. If ``False``, only the gradient of the effective
            potential is included (default: True).
        """
        self.set_lcao_calculator(calc)
        ElectronPhononCoupling.calculate_supercell_matrix(self, name, filter,
                                                          include_pseudo)

    def _bloch_matrix(self, kpt, k_c, u_l, name) -> ArrayND:
        """
        This is a special q=0 version. Need to implement general version in
        ElectronPhononCoupling.
        """
        assert len(u_l.shape) == 3

        # Defining system sizes
        nmodes = u_l.shape[0]
        nbands = kpt.C_nM.shape[0]
        nao = kpt.C_nM.shape[1]
        ndisp = 3 * len(self.indices)

        # Lattice vectors
        R_cN = self.compute_lattice_vectors()
        # Number of unit cell in supercell
        N = np.prod(self.supercell)

        # Allocate array for couplings
        g_lnn = np.zeros((nmodes, nbands, nbands), dtype=complex)

        # Mass scaled polarization vectors
        u_lx = u_l.reshape(nmodes, ndisp)

        self.supercell_cache = MultiFileJSONCache(name)
        self.basis_info = self.supercell_cache['basis']

        # Multiply phase factors
        for x in range(ndisp):
            # Allocate array
            g_MM = np.zeros((nao, nao), dtype=complex)
            g_sNNMM = self.supercell_cache[str(x)]
            assert nao == g_sNNMM.shape[-1]
            for m in range(N):
                for n in range(N):
                    phase = self._get_phase_factor(R_cN, m, n, k_c,
                                                   [0., 0., 0.])
                    # Sum contributions from different cells
                    g_MM += g_sNNMM[kpt.s, m, n, :, :] * phase

            g_nn = np.dot(kpt.C_nM.conj(), np.dot(g_MM, kpt.C_nM.T))
            g_lnn += np.einsum('i,kl->ikl', u_lx[:, x], g_nn)

        return g_lnn * units.Hartree / units.Bohr  # eV / Ang

    def get_elph_matrix(self, calc, phonon, name='supercell',
                        savetofile=True) -> ArrayND:
        """Calculate the electronphonon matrix in Bloch states.

        Always uses q=0.

        Parameters
        ----------
        calc: GPAW
            Converged ground-state calculation. NOT supercell.
        phonon: ase.phonons.Phonons
            Phonon object
        name: str
            Name of supercell cache
        savetofile: bool
            Switch for saving to gsqklnn.npy file
        """
        assert calc.wfs.bd.comm.size == 1

        # Read previous phonon calculation.
        # This only looks at gamma point phonons
        phonon.read()
        frequencies, modes = phonon.band_structure([[0., 0., 0.]], modes=True)

        g_sqklnn = np.zeros([calc.wfs.nspins, 1, calc.wfs.kd.nibzkpts,
                             frequencies.shape[1], calc.wfs.bd.nbands,
                             calc.wfs.bd.nbands], dtype=complex)

        # loop over k-points
        for kpt in calc.wfs.kpt_u:
            k_c = calc.wfs.kd.ibzk_kc[kpt.k]
            # Find el-ph matrix in the LCAO basis
            g_lnn = self._bloch_matrix(kpt, k_c, modes[0], name)
            g_sqklnn[kpt.s, 0, kpt.k] += g_lnn

        calc.wfs.kd.comm.sum(g_sqklnn)

        if world.rank == 0 and savetofile:
            np.save("gsqklnn.npy", g_sqklnn)
        return g_sqklnn
