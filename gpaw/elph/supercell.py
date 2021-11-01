"""Module for electron-phonon supercell properties."""

from abc import ABC
from ase.io.trajectory import VersionTooOldError
import numpy as np

from ase import Atoms
from ase.parallel import parprint
import ase.units as units
from ase.utils.filecache import MultiFileJSONCache

from gpaw import GPAW
from gpaw.lcao.tightbinding import TightBinding
from gpaw.utilities import unpack2
from gpaw.utilities.tools import tri2full

from .filter import fourier_filter

sc_version = 1


class Supercell(ABC):
    """Class for supercell-related stuff."""

    def __init__(self, atoms: Atoms, supercell_name: str = 'supercell',
                 supercell: tuple = (1, 1, 1)) -> None:
        """Initialize supercell class.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on. Primitive cell.
        supercell_name: str
            User specified name of the generated JSON cache.
            Default is 'supercell'.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        """
        self.atoms = atoms
        self.supercell_name = supercell_name
        self.supercell = supercell

    def _calculate_supercell_entry(self, a, v, V1t_sG, dH1_asp, wfs):
        kpt_u = wfs.kpt_u
        setups = wfs.setups
        nao = setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        nspins = wfs.nspins
        indices = np.arange(len(self.atoms))

        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        dH_asp = self.cache['eq']['dH_all_asp']

        # For the contribution from the derivative of the projectors
        dP_aqvMi = wfs.manytci.P_aqMi(indices, derivative=True)
        dP_qvMi = dP_aqvMi[a]

        # Array for different k-point components
        g_sqMM = np.zeros((nspins, len(kpt_u) // nspins, nao, nao), dtype)

        # 1) Gradient of effective potential
        for kpt in kpt_u:
            # Matrix elements
            geff_MM = np.zeros((nao, nao), dtype)
            bfs.calculate_potential_matrix(V1t_sG[kpt.s], geff_MM, q=kpt.q)
            tri2full(geff_MM, 'L')
            # Insert in array
            g_sqMM[kpt.s, kpt.q] += geff_MM

        # 2) Gradient of non-local part (projectors)
        P_aqMi = wfs.P_aqMi
        # 2a) dH^a part has contributions from all other atoms
        for kpt in kpt_u:
            # Matrix elements
            gp_MM = np.zeros((nao, nao), dtype)
            for a_, dH1_sp in dH1_asp.items():
                dH1_ii = unpack2(dH1_sp[kpt.s])
                P_Mi = P_aqMi[a_][kpt.q]
                gp_MM += np.dot(P_Mi, np.dot(dH1_ii, P_Mi.T.conjugate()))
            g_sqMM[kpt.s, kpt.q] += gp_MM

        # 2b) dP^a part has only contributions from the same atoms
        dH_ii = unpack2(dH_asp[a][kpt.s])
        for kpt in kpt_u:
            # XXX Sort out the sign here; conclusion -> sign = +1 !
            P1HP_MM = +1 * np.dot(dP_qvMi[kpt.q][v], np.dot(dH_ii,
                                  P_aqMi[a][kpt.q].T.conjugate()))
            # Matrix elements
            gp_MM = P1HP_MM + P1HP_MM.T.conjugate()
            g_sqMM[kpt.s, kpt.q] += gp_MM
        return g_sqMM

    def calculate_supercell_matrix(self, calc: GPAW, fd_name: str = 'elph',
                                   filter: str = None) -> None:
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        The resulting g_xsNNMM is saved into a JSON cache.

        Parameters
        ----------
        calc: GPAW
            LCAO calculator for the calculation of the supercell matrix.
        fd_name: str
            User specified name of the finite difference JSON cache.
            Default is 'elph'.
        filter: str
            Fourier filter atomic gradients of the effective potential. The
            specified components (``normal`` or ``umklapp``) are removed
            (default: None).
        """

        assert calc.wfs.mode == 'lcao', 'LCAO mode required.'
        assert not calc.symmetry.point_group, \
            'Point group symmetry not supported'

        # JSON cache
        supercell_cache = MultiFileJSONCache(self.supercell_name)

        # Supercell atoms
        atoms_N = self.atoms * self.supercell

        # Initialize calculator if required and extract useful quantities
        if (not hasattr(calc.wfs, 'S_qMM') or
            not hasattr(calc.wfs.basis_functions, 'M_a')):
            calc.initialize(atoms_N)
            calc.initialize_positions(atoms_N)
        basis_info = self.set_basis_info()

        # Extract useful objects from the calculator
        wfs = calc.wfs
        gd = calc.wfs.gd
        kd = calc.wfs.kd
        nao = wfs.setups.nao
        nspins = wfs.nspins
        # FIXME: Domain parallelisation broken
        assert gd.comm.size == 1

        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xsG, dH1_xasp = self.calculate_gradient(fd_name)

        # Check that the grid is the same as in the calculator
        assert np.all(V1t_xsG.shape[-3:] == (gd.N_c + gd.pbc_c - 1)), \
            "Mismatch in grids."

        # Save basis information, after we checked the data is kosher
        with supercell_cache.lock('basis') as handle:
            if handle is not None:
                handle.save(basis_info)

        # Fourier filter the atomic gradients of the effective potential
        if filter is not None:
            for s in range(nspins):
                fourier_filter(self.atoms, self.supercell, V1t_xsG[:, s],
                               components=filter)

        if kd.gamma:
            print("WARNING: Gamma-point calculation. \
                   Overlap with neighboring cell cannot be removed")
        else:
            # Bloch to real-space converter
            tb = TightBinding(atoms_N, calc)

        # Calculate < i k | grad H | j k >, i.e. matrix elements in LCAO basis

        # Do each cartesian component separately
        for i, a in enumerate(np.arange(len(atoms_N))):
            for v in range(3):
                # Corresponding array index
                x = 3 * i + v

                # If exist already, don't recompute
                with self.supercell_cache.lock(str(x)) as handle:
                    if handle is None:
                        continue

                    parprint("%s-gradient of atom %u" %
                             (['x', 'y', 'z'][v], a))

                    g_sqMM = self._calculate_supercell_entry(a, v, V1t_xsG[x],
                                                             dH1_xasp[x], wfs)

                    # Extract R_c=(0, 0, 0) block by Fourier transforming
                    if kd.gamma or kd.N_c is None:
                        g_sMM = g_sqMM[:, 0]
                    else:
                        # Convert to array
                        g_sMM = []
                        for s in range(nspins):
                            g_MM = tb.bloch_to_real_space(g_sqMM[s],
                                                          R_c=(0, 0, 0))
                            g_sMM.append(g_MM[0])  # [0] because of above
                        g_sMM = np.array(g_sMM)

                    # Reshape to global unit cell indices
                    N = np.prod(self.supercell)
                    # Number of basis function in the primitive cell
                    assert (nao % N) == 0, "Alarm ...!"
                    nao_cell = nao // N
                    g_sNMNM = g_sMM.reshape((nspins, N, nao_cell, N, nao_cell))
                    g_sNNMM = g_sNMNM.swapaxes(2, 3).copy()
                    handle.save(g_sNNMM)
                if x == 0:
                    with self.supercell_cache.lock('info') as handle:
                        if handle is not None:
                            info = {'sc_version': sc_version,
                                    'natom': len(self.atoms),
                                    'supercell': self.supercell,
                                    'gshape': g_sNNMM.shape,
                                    'gtype': g_sNNMM.dtype.name}
                            handle.save(info)

    def set_basis_info(self, *args) -> dict:
        """Store LCAO basis info for atoms in reference cell in attribute.

        Parameters
        ----------
        args: tuple
            If the LCAO calculator is not available (e.g. if the supercell is
            loaded from file), the ``load_supercell_matrix`` member function
            provides the required info as arguments.

        """
        assert len(args) in (1, 2)
        if len(args) == 0:
            calc = args[0]
            setups = calc.wfs.setups
            bfs = calc.wfs.basis_functions
            nao_a = [setups[a].nao for a in range(len(self.atoms))]
            M_a = [bfs.M_a[a] for a in range(len(self.atoms))]
        else:
            M_a = args[0]
            nao_a = args[1]
        return {'M_a': M_a, 'nao_a': nao_a}

    @classmethod
    def calculate_gradient(self, cache: str):
        """Calculate gradient of effective potential and projector coefs.

        This function loads the generated json files and calculates
        finite-difference derivatives.

        Parameters
        ----------
        cache: str
            name of finite difference JSON cache
        """
        if not hasattr(cache, 'dr_version'):
            print("Cache created with old version. Use electronphonon.py")
            raise VersionTooOldError
        natom = cache['info']['natom']
        delta = cache['info']['delta']

        # Array and dict for finite difference derivatives
        V1t_xsG = []
        dH1_xasp = []

        x = 0
        for a in natom:
            for v in 'xyz':
                name = '%d%s' % (a, v)
                # Potential and atomic density matrix for atomic displacement
                Vtm_sG = cache[name + '-']['Vt_sG']
                dHm_asp = cache[name + '-']['dH_all_asp']
                Vtp_sG = cache[name + '+']['Vt_sG']
                dHp_asp = cache[name + '+']['dH_all_asp']

                # FD derivatives in Hartree / Bohr
                V1t_sG = (Vtp_sG - Vtm_sG) / (2 * delta / units.Bohr)
                V1t_xsG.append(V1t_sG)

                dH1_asp = {}
                for atom in dHm_asp.keys():
                    dH1_asp[atom] = (dHp_asp[atom] - dHm_asp[atom]) / \
                                    (2 * delta / units.Bohr)
                dH1_xasp.append(dH1_asp)
                x += 1
        return np.array(V1t_xsG), dH1_xasp

    @classmethod
    def load_supercell_matrix(self, name: str = 'supercell'):
        """Load supercell matrix from cache.

        Parameters
        ----------
        name: str
            User specified name of the cache.
        """
        supercell_cache = MultiFileJSONCache(name)
        if not hasattr(supercell_cache, 'sc_version'):
            print("Cache created with old version. Use electronphonon.py")
            raise VersionTooOldError
        shape = supercell_cache['info']['gshape']
        dtype = supercell_cache['info']['gtype']
        natom = supercell_cache['info']['natom']
        supercell = supercell_cache['info']['supercell']
        nx = natom * np.product(supercell) * 3
        g_xsNNMM = np.empty([nx, ] + list(shape), dtype=dtype)
        for x in range(nx):
            g_xsNNMM[x] = supercell_cache[str(x)]
        basis_info = supercell_cache['basis']
        return g_xsNNMM, basis_info

    # def apply_cutoff(self, cutmax=None, cutmin=None):
    #     """Zero matrix element inside/beyond the specified cutoffs.

    #     This method is not tested.
    #     This method does not respect minimum image convention.

    #     Parameters
    #     ----------
    #     cutmax: float
    #         Zero matrix elements for basis functions with a distance to the
    #         atomic gradient that is larger than the cutoff.
    #     cutmin: float
    #         Zero matrix elements where both basis functions have distances to
    #         the atomic gradient that is smaller than the cutoff.
    #     """

    #     if cutmax is not None:
    #         cutmax = float(cutmax)
    #     if cutmin is not None:
    #         cutmin = float(cutmin)

    #     # Reference to supercell matrix attribute
    #     g_xsNNMM = self.g_xsNNMM

    #     # Number of atoms and primitive cells
    #     N_atoms = len(self.indices)
    #     N = np.prod(self.supercell)
    #     nao = g_xsNNMM.shape[-1]
    #     nspins = g_xsNNMM.shape[1]

    #     # Reshape array
    #     g_avsNNMM = g_xsNNMM.reshape(N_atoms, 3, nspins, N, N, nao, nao)

    #     # Make slices for orbitals on atoms
    #     M_a = self.basis_info['M_a']
    #     nao_a = self.basis_info['nao_a']
    #     slice_a = []
    #     for a in range(len(self.atoms)):
    #         start = M_a[a]
    #         stop = start + nao_a[a]
    #         s = slice(start, stop)
    #         slice_a.append(s)

    #     # Lattice vectors
    #     R_cN = self.compute_lattice_vectors()

    #     # Unit cell vectors
    #     cell_vc = self.atoms.cell.transpose()
    #     # Atomic positions in reference cell
    #     pos_av = self.atoms.get_positions()

    #     # Create a mask array to zero the relevant matrix elements
    #     if cutmin is not None:
    #         mask_avsNNMM = np.zeros(g_avsNNMM.shape, dtype=bool)

    #     # Zero elements where one of the basis orbitals has a distance to
    #     # atoms
    #     # (atomic gradients) in the reference cell larger than the cutoff
    #     for n in range(N):
    #         # Lattice vector to cell
    #         R_v = np.dot(cell_vc, R_cN[:, n])
    #         # Atomic positions in cell
    #         posn_av = pos_av + R_v
    #         for i, a in enumerate(self.indices):
    #             # Atomic distances wrt to the position of the gradient
    #             dist_a = np.sqrt(np.sum((pos_av[a] - posn_av)**2, axis=-1))

    #             if cutmax is not None:
    #                 # Atoms indices where the distance is larger than the max
    #                 # cufoff
    #                 j_a = np.where(dist_a > cutmax)[0]
    #                 # Zero elements
    #                 for j in j_a:
    #                     g_avsNNMM[a, :, :, n, :, slice_a[j], :] = 0.0
    #                     g_avsNNMM[a, :, :, :, n, :, slice_a[j]] = 0.0

    #             if cutmin is not None:
    #                 # Atoms indices where the distance is larger than the min
    #                 # cufoff
    #                 j_a = np.where(dist_a > cutmin)[0]
    #                 # Update mask to keep elements where one LCAO is outside
    #                 # the min cutoff
    #                 for j in j_a:
    #                     mask_avsNNMM[a, :, :, n, :, slice_a[j], :] = True
    #                     mask_avsNNMM[a, :, :, :, n, :, slice_a[j]] = True

    #     # Zero elements where both LCAOs are located within the min cutoff
    #     if cutmin is not None:
    #         g_avsNNMM[~mask_avsNNMM] = 0.0
