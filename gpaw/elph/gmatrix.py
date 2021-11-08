r"""Module for calculating electron-phonon matrix.

Electron-phonon interaction::

                  __
                  \     l   +         +
        H      =   )   g   c   c   ( a   + a  ),
         el-ph    /_    ij  i   j     l     l
                 l,ij

where the electron phonon coupling is given by::

                      ______
             l       / hbar         ___
            g   =   /-------  < i | \ /  V   * e  | j > .
             ij   \/ 2 M w           'u   eff   l
                          l

Here, l denotes the vibrational mode, w_l and e_l is the frequency and
mass-scaled polarization vector, respectively, M is an effective mass, i, j are
electronic state indices and nabla_u denotes the gradient wrt atomic
displacements. The implementation supports calculations of the el-ph coupling
in both finite and periodic systems, i.e. expressed in a basis of molecular
orbitals or Bloch states.
"""
import numpy as np

from ase import Atoms
from ase.phonons import Phonons
import ase.units as units
from ase.utils.filecache import MultiFileJSONCache

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.typing import ArrayND


class ElectronPhononMatrix:
    """Class for containing the electron-phonon matrix"""
    def __init__(self, atoms: Atoms, supercell_cache: str,
                 phonon) -> None:
        """Initialize with base class args and kwargs.

        Parameters
        ----------
        atoms: Atoms
            Primitive cell object
        supercell_cache: str
            Name of JSON cache containing supercell matrix
        phonon: str, dict, Phonon
            Can be either name of phonon cache generated with
            electron-phonon DisplacementRunner or dictonary
            of arguments used in Phonon run or Phonon object.
        """
        self.atoms = atoms
        self.supercell_cache = MultiFileJSONCache(supercell_cache)

        if isinstance(phonon, Phonons):
            self.phonon = phonon
        elif isinstance(phonon, str):
            info = MultiFileJSONCache(phonon)['info']
            assert 'dr_version' in info, 'use valid cache created by elph'
            # our version of phonons
            self.phonon = Phonons(atoms, supercell=info['supercell'],
                                  name=phonon, delta=info['delta'],
                                  center_refcell=True)
        elif isinstance(phonon, dict):
            # this would need to be updated of Phonon defaults change
            supercell = phonon.get('supercell', (1, 1, 1))
            name = phonon.get('name', 'phonon')
            delta = phonon.get('delta', 0.01)
            center_refcell = phonon.get('center_refcell', False)
            self.phonon = Phonons(atoms, None, supercell, name, delta,
                                  center_refcell)
        else:
            raise ValueError

        if self.phonon.D_N is None:
            self.phonon.read(symmetrize=10)
            
    def _bloch_matrix(self, C1_nM, C2_nM, s, k_c, q_c,
                      prefactor: bool) -> ArrayND:
        """Calculates elph matrix entry for a given k and q.
        """
        omega_ql, u_ql = self.phonon.band_structure([q_c], modes=True)
        u_l = u_ql[0]
        assert len(u_l.shape) == 3

        # Defining system sizes
        nmodes = u_l.shape[0]
        nbands = C1_nM.shape[0]
        nao = C1_nM.shape[1]
        ndisp = 3 * len(self.atoms)

        # Lattice vectors
        R_cN = self.phonon.compute_lattice_vectors()
        # Number of unit cell in supercell
        N = np.prod(self.phonon.supercell)

        # Allocate array for couplings
        g_lnn = np.zeros((nmodes, nbands, nbands), dtype=complex)

        # Mass scaled polarization vectors
        u_lx = u_l.reshape(nmodes, ndisp)

        # Multiply phase factors
        for x in range(ndisp):
            # Allocate array
            g_MM = np.zeros((nao, nao), dtype=complex)
            g_sNNMM = self.supercell_cache[str(x)]
            assert nao == g_sNNMM.shape[-1]
            for m in range(N):
                for n in range(N):
                    phase = self._get_phase_factor(R_cN, m, n, k_c, q_c)
                    # Sum contributions from different cells
                    g_MM += g_sNNMM[s, m, n, :, :] * phase

            g_nn = np.dot(C2_nM.conj(), np.dot(g_MM, C1_nM.T))
            g_lnn += np.einsum('i,kl->ikl', u_lx[:, x], g_nn)

        if prefactor:
            # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
            amu = units._amu  # atomic mass unit
            me = units._me   # electron mass
            g_lnn /= np.sqrt(2 * amu / me / units.Hartree *
                             omega_ql[0, :, np.newaxis, np.newaxis])
            # Convert to eV
            return g_lnn * units.Hartree  # eV
        else:
            return g_lnn * units.Hartree / units.Bohr  # eV / Ang

    def bloch_matrix(self, calc: GPAW, k_qc: ArrayND = None,
                     savetofile: bool = True,
                     prefactor: bool = True) -> ArrayND:
        r"""Calculate el-ph coupling in the Bloch basis for the electrons.

        This function calculates the electron-phonon coupling between the
        specified Bloch states, i.e.::

                      ______
            mnl      / hbar               ^
           g    =   /-------  < m k + q | e  . grad V  | n k >
            kq    \/ 2 M w                 ql        q
                          ql

        In case the ``prefactor=False`` is given, the bare matrix
        element (in units of eV / Ang) without the sqrt prefactor is returned.

        Parameters
        ----------
        calc: GPAW
            Converged calculator object containing the LCAO wavefuntions
            (don't use point group symmetry)
        k_qc: ndarray
            q-vectors of the phonons. Must only contain values comenserate
            with k-point sampling of calculator. Default: all kpointsused.
        savetofile: bool
            If true (default), saves matrix to gsqklnn.npy
        prefactor: bool
            if false, don't multiply with sqrt prefactor (Default: True)
        """
        # assert calc.wfs.kd.comm.size == 1

        kd = calc.wfs.kd
        wfs = calc.wfs
        gwa = wfs.get_wave_function_array  # only rank 0 gets stuff
        if k_qc is None:
            k_qc = kd.get_bz_q_points(first=True)
        else:
            assert k_qc.ndim == 2

        g_sqklnn = np.zeros([wfs.nspins, k_qc.shape[0],
                             kd.nibzkpts, 3 * len(self.atoms),
                             wfs.bd.nbands, wfs.bd.nbands],
                            dtype=complex)

        for s in range(wfs.nspins):
            for q, q_c in enumerate(k_qc):
                # Find indices of k+q for the k-points
                kplusq_k = kd.find_k_plus_q(q_c)
                for k in enumerate(kd.nbzkpts):
                    k_c = kd.ibzk_kc[k]
                    kplusq_c = k_c + q_c
                    kplusq_c -= kplusq_c.round()
                    assert np.allclose(kplusq_c, kd.bzk_kc[kplusq_k[k]])
                    ck_nM = np.zeros((wfs.bd.nbands, wfs.setups.nao),
                                     dtype=complex)
                    ckplusq_nM = np.zeros((wfs.bd.nbands, wfs.setups.nao),
                                          dtype=complex)
                    for n in range(wfs.bd.nbands):
                        ck_nM[n] = gwa(n, k, s, False)
                        ckplusq_nM[n] = gwa(n, kplusq_k[k], s, False)
                    g_lnn = self._bloch_matrix(ck_nM, ckplusq_nM, s, k_c, q_c,
                                               prefactor)
                    # wfs.bd.comm.sum(g_lnn)
                    g_sqklnn[s, q, k] += g_lnn

        # kd.comm.sum(g_sqklnn)

        if world.rank == 0 and savetofile:
            np.save("gsqklnn.npy", g_sqklnn)
        return g_sqklnn

    @classmethod
    def _get_phase_factor(cls, R_cN, m, n, k_c, q_c) -> float:
        Rm_c = R_cN[:, m]
        Rn_c = R_cN[:, n]
        phase = np.exp(2.j * np.pi * (np.dot(k_c, Rm_c - Rn_c) +
                                      np.dot(q_c, Rm_c)))
        return phase

#   def lcao_matrix(self, u_l, omega_l):
#         """Calculate the el-ph coupling in the electronic LCAO basis.

#         For now, only works for Gamma-point phonons.

#         This method is not tested.

#         Parameters
#         ----------
#         u_l: ndarray
#             Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of
#             the phonons.
#         omega_l: ndarray
#             Vibrational frequencies in eV.
#         """

#         # Supercell matrix (Hartree / Bohr)
#         assert self.g_xsNNMM is not None, "Load supercell matrix."
#         assert self.g_xsNNMM.shape[2:4] == (1, 1)
#         g_xsMM = self.g_xsNNMM[:, :, 0, 0, :, :]
#         # Number of atomic orbitals
#         # nao = g_xMM.shape[-1]
#         # Number of phonon modes
#         nmodes = u_l.shape[0]

#         #
#         u_lx = u_l.reshape(nmodes, 3 * len(self.atoms))
#         # np.dot uses second to last index of second array
#         g_lsMM = np.dot(u_lx, g_xsMM.transpose(2, 0, 1, 3))

#         # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
#         amu = units._amu  # atomic mass unit
#         me = units._me   # electron mass
#         g_lsMM /= np.sqrt(2 * amu / me / units.Hartree *
#                           omega_l[:, :, np.newaxis, np.newaxis])
#         # Convert to eV
#         g_lsMM *= units.Hartree

#         return g_lsMM
