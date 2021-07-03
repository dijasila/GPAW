import numpy as np
import ase.units as units
from gpaw.elph.electronphonon import ElectronPhononCoupling
from gpaw.mpi import world
from gpaw.typing import ArrayND


class EPC(ElectronPhononCoupling):
    """Modify ElectronPhononCoupling to fit Raman stuff better.

    Primarily, always save the supercell matrix in separate files,
    so that calculation of the elph matrix can be better parallelised.
    """

    def calculate_supercell_matrix(self, calc, name=None, filter=None,
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
            User specified name of the generated pickle file(s). If not
            provided, the string in the ``name`` attribute is used.
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
        ElectronPhononCoupling.calculate_supercell_matrix(self, 2, name,
                                                          filter,
                                                          include_pseudo)

    def _bloch_matrix(self, kpt, k_c, u_l, basis=None, name=None) -> ArrayND:
        """
        This is a special q=0 version. Need to implement general version in
        ElectronPhononCoupling.
        """
        if basis is None:
            basis = ''
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

        # Multiply phase factors
        for x in range(ndisp):
            # Allocate array
            g_MM = np.zeros((nao, nao), dtype=complex)
            fname = self._set_file_name(2, basis, name, x=x)
            g_sNNMM, M_a, nao_a = self.load_supercell_matrix_x(fname)
            assert nao == g_sNNMM.shape[-1]
            if x == 0:
                self.set_basis_info(M_a, nao_a)
            for m in range(N):
                for n in range(N):
                    phase = self._get_phase_factor(R_cN, m, n, k_c,
                                                   [0., 0., 0.])
                    # Sum contributions from different cells
                    g_MM += g_sNNMM[kpt.s, m, n, :, :] * phase

            g_nn = np.dot(kpt.C_nM.conj(), np.dot(g_MM, kpt.C_nM.T))
            g_lnn += np.einsum('i,kl->ikl', u_lx[:, x], g_nn)

        return g_lnn * units.Hartree / units.Bohr  # eV / Ang

    def get_elph_matrix(self, calc, phonon, savetofile=True) -> ArrayND:
        """Calculate the electronphonon matrix in Bloch states.

        Always uses q=0.

        Parameters
        ----------
        calc: GPAW
            Converged ground-state calculation. NOT supercell.
        phonon: Phonons
            Phonon object
        savetofile: bool
            Switch for saving to gsqklnn.npy file
        """
        assert calc.wfs.bd.comm.size == 1

        # Read previous phonon calculation.
        # This only looks at gamma point phonons
        phonon.read()
        frequencies, modes = phonon.band_structure([[0., 0., 0.]], modes=True)

        # Find el-ph matrix in the LCAO basis
        if self.calc_lcao is None:
            self.set_lcao_calculator(calc)
        basis = calc.parameters['basis']
        if isinstance(basis, dict):
            basis = ""

        g_sqklnn = np.zeros([calc.wfs.nspins, 1, calc.wfs.kd.nibzkpts,
                             frequencies.shape[1], calc.wfs.bd.nbands,
                             calc.wfs.bd.nbands], dtype=complex)

        # loop over k-points
        for kpt in calc.wfs.kpt_u:
            k_c = calc.wfs.kd.ibzk_kc[kpt.k]
            g_lnn = self._bloch_matrix(kpt, k_c, modes[0], basis)
            g_sqklnn[kpt.s, 0, kpt.k] += g_lnn

        calc.wfs.kd.comm.sum(g_sqklnn)

        if world.rank == 0 and savetofile:
            np.save("gsqklnn.npy", g_sqklnn)
        return g_sqklnn
