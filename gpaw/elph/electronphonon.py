r"""Module for calculating electron-phonon couplings.

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

The implementation is based on finite-difference calculations of the the atomic
gradients of the effective potential expressed on a real-space grid. The el-ph
couplings are obtained from LCAO representations of the atomic gradients of the
effective potential and the electronic states.

In PAW the matrix elements of the derivative of the effective potential is
given by the sum of the following contributions::

                  d                  d
            < i | -- V | j > = < i | -- V | j>
                  du  eff            du

                               _
                              \        ~a     d   .       ~a
                            +  ) < i | p  >   -- /_\H   < p | j >
                              /_        i     du     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < i | -- p  >  /_\H    < p | j >
                              /_       du  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < i | p  >  /_\H    < -- p  | j >
                              /_        i        ij    du  j
                              a,ij

where the first term is the derivative of the potential (Hartree + XC) and the
last three terms originate from the PAW (pseudopotential) part of the effective
DFT Hamiltonian.

"""
from math import pi
import os.path as op
import pickle
import numpy as np

import ase.units as units
from ase.phonons import Displacement
from ase.utils.filecache import MultiFileJSONCache

from gpaw import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lcao.tightbinding import TightBinding
from gpaw.utilities import unpack2
from gpaw.utilities.timing import StepTimer, nulltimer
from gpaw.utilities.tools import tri2full


class BackwardsCompatibleDisplacement(Displacement):
    # NOTE: It's probably save to remove this, as we are not compatible with
    # old ASE anyway, thanks to breaking pckl to json change.
    def __init__(self, atoms, calc, supercell,
                 name, delta):
        if hasattr(Displacement, 'compute_lattice_vectors'):
            Displacement.__init__(self, atoms, calc, supercell,
                                  name, delta, center_refcell=True)
        else:
            Displacement.__init__(self, atoms, calc, supercell,
                                  name, delta, refcell='center')
            self.supercell = self.N_c
            self.compute_lattice_vectors = self.lattice_vectors


class ElectronPhononCoupling(BackwardsCompatibleDisplacement):
    """Class for calculating the electron-phonon coupling in an LCAO basis.

    The derivative of the effective potential wrt atomic displacements is
    obtained from a finite difference approximation to the derivative by doing
    a self-consistent calculation for atomic displacements in the +/-
    directions. These calculations are carried out in the ``run`` member
    function.

    The subsequent calculation of the coupling matrix in the basis of atomic
    orbitals (or Bloch-sums hereof for periodic systems) is handled by the
    ``calculate_matrix`` member function.

    """

    def __init__(self, atoms, calc=None, supercell=(1, 1, 1), name='elph',
                 delta=0.01, calculate_forces=False):
        """Initialize with base class args and kwargs.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on.
        calc: GPAW
            Calculator for the supercell calculation.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Name to use for files (default: 'elph').
        delta: float
            Magnitude of displacements.
        calculate_forces: bool
            If true, also calculate and store the dynamical matrix.
        """

        # Init base class and make the center cell in the supercell the
        # reference cell
        BackwardsCompatibleDisplacement.__init__(
            self, atoms, calc=calc, supercell=supercell,
            name=name, delta=delta)

        self.calculate_forces = calculate_forces
        # Log
        self.set_log()
        # LCAO calculator
        self.calc_lcao = None
        # Supercell matrix
        self.g_xsNNMM = None
        # basis
        self.basis_info = None

        # if not calculate_forces:
        #     self.phcache = MultiFileJSONCache('phonons-cache')
        self.cache = MultiFileJSONCache('elph-cache')

    def calculate(self, atoms_N, disp):
        Vt_sG, dH_all_asp, forces = self(atoms_N)
        output = {'Vt_sG': Vt_sG, 'dH_all_asp': dH_all_asp}
        if forces is not None:
            output['forces'] = forces
        return output

    def __call__(self, atoms_N):
        """Extract effective potential and projector coefficients."""

        # Do calculation
        atoms_N.get_potential_energy()

        # Calculate forces if desired
        if self.calculate_forces:
            forces = atoms_N.get_forces()
        else:
            forces = None

        # Get calculator
        calc = atoms_N.calc
        if not isinstance(calc, GPAW):
            calc = calc.dft  # unwrap DFTD3 wrapper

        # Effective potential (in Hartree) and projector coefficients
        Vt_sG = calc.hamiltonian.vt_sG
        Vt_sG = calc.wfs.gd.collect(Vt_sG, broadcast=True)
        dH_asp = calc.hamiltonian.dH_asp

        setups = calc.wfs.setups
        nspins = calc.wfs.nspins
        gd_comm = calc.wfs.gd.comm

        dH_all_asp = {}
        for a, setup in enumerate(setups):
            ni = setup.ni
            nii = ni * (ni + 1) // 2
            dH_tmp_sp = np.zeros((nspins, nii))
            if a in dH_asp:
                dH_tmp_sp[:] = dH_asp[a]
            gd_comm.sum(dH_tmp_sp)
            dH_all_asp[a] = dH_tmp_sp

        return Vt_sG, dH_all_asp, forces

    def set_lcao_calculator(self, calc):
        """Set LCAO calculator for the calculation of the supercell matrix."""

        # Add parameter checks here
        # - check that gamma
        # - check that no symmetries are used
        # - ...
        assert calc.parameters['mode'] == 'lcao', 'LCAO mode required.'
        symmetry = calc.parameters['symmetry']
        if isinstance(symmetry, dict):
            assert not symmetry.get('point_group', True),\
                'Point group symmetry not supported'
        else:
            assert symmetry == 'off', 'Point group symmetry not supported'

        self.calc_lcao = calc

    def set_basis_info(self, *args):
        """Store lcao basis info for atoms in reference cell in attribute.

        Parameters
        ----------
        args: tuple
            If the LCAO calculator is not available (e.g. if the supercell is
            loaded from file), the ``load_supercell_matrix`` member function
            provides the required info as arguments.

        """

        if len(args) == 0:
            calc = self.calc_lcao
            setups = calc.wfs.setups
            bfs = calc.wfs.basis_functions
            nao_a = [setups[a].nao for a in range(len(self.atoms))]
            M_a = [bfs.M_a[a] for a in range(len(self.atoms))]
        else:
            M_a = args[0]
            nao_a = args[1]

        self.basis_info = {'M_a': M_a,
                           'nao_a': nao_a}

    def set_log(self, log=None):
        """Set output log."""

        if log is None:
            self.timer = nulltimer
        elif log == '-':
            self.timer = StepTimer(name='elph')
        else:
            self.timer = StepTimer(name='elph', out=open(log, 'w'))

    def _set_file_name(self, dump, basis, name=None, x=None):
        """Set name of supercell file."""
        assert dump in (1, 2)
        basestr = ".supercell_matrix"
        if name is not None:
            assert isinstance(name, str)
            nname = name
        else:
            nname = self.name

        if dump == 1:
            fname = nname + basestr + '.' + basis + '.pckl'
        else:  # dump == 2
            fname = nname + basestr + '_x_' + str(x) + '.' + basis + '.pckl'
        return fname

    def calculate_supercell_matrix(self, dump=0, name=None, filter=None,
                                   include_pseudo=True):
        """Calculate matrix elements of the el-ph coupling in the LCAO basis.

        This function calculates the matrix elements between LCAOs and local
        atomic gradients of the effective potential. The matrix elements are
        calculated for the supercell used to obtain finite-difference
        approximations to the derivatives of the effective potential wrt to
        atomic displacements.

        Parameters
        ----------
        dump: int
            Dump supercell matrix to pickle file (default: 0).

            0: Supercell matrix not saved

            1: Supercell matrix saved in a single pickle file.

            2: Dump matrix for different gradients in separate files. Useful
               for large systems where the total array gets too large for a
               single pickle file. Allows restart.

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

        assert self.calc_lcao is not None, "Set LCAO calculator"

        # Supercell atoms
        atoms_N = self.atoms * self.supercell

        # Initialize calculator if required and extract useful quantities
        calc = self.calc_lcao
        if not hasattr(calc.wfs, 'S_qMM'):
            calc.initialize(atoms_N)
            calc.initialize_positions(atoms_N)
        self.set_basis_info()
        basis = calc.parameters['basis']

        # Extract useful objects from the calculator
        wfs = calc.wfs
        gd = calc.wfs.gd
        kd = calc.wfs.kd
        kpt_u = wfs.kpt_u
        setups = wfs.setups
        nao = setups.nao
        bfs = wfs.basis_functions
        dtype = wfs.dtype
        nspins = wfs.nspins

        # If gamma calculation, overlap with neighboring cell cannot be removed
        if kd.gamma:
            print("WARNING: Gamma-point calculation.")
        else:
            # Bloch to real-space converter
            tb = TightBinding(atoms_N, calc)

        self.timer.write_now("Calculating supercell matrix")

        self.timer.write_now("Calculating real-space gradients")
        # Calculate finite-difference gradients (in Hartree / Bohr)
        V1t_xsG, dH1_xasp = self.calculate_gradient()
        self.timer.write_now("Finished real-space gradients")

        # Fourier filter the atomic gradients of the effective potential
        if filter is not None:
            self.timer.write_now("Fourier filtering gradients")
            # V1_xsG = V1t_xsG.copy()
            for s in range(nspins):
                self.fourier_filter(V1t_xsG[:, s], components=filter)
            self.timer.write_now("Finished Fourier filtering")

        # For the contribution from the derivative of the projectors
        dP_aqvMi = wfs.manytci.P_aqMi(self.indices, derivative=True)
        # Equilibrium atomic Hamiltonian matrix (projector coefficients)
        Vt_sG, dH_asp, _ = self.cache['eq']  # caution, eq file is different...

        # dH_asp = pickle.load(open(self.name + '.eq.pckl', 'rb'))[1]

        # Check that the grid is the same as in the calculator
        assert np.all(V1t_xsG.shape[-3:] == (gd.N_c + gd.pbc_c - 1)), \
            "Mismatch in grids."

        # Calculate < i k | grad H | j k >, i.e. matrix elements in Bloch basis
        # List for supercell matrices;
        g_xsNNMM = []
        self.timer.write_now("Calculating gradient of PAW Hamiltonian")

        # Do each cartesian component separately
        for i, a in enumerate(self.indices):
            for v in range(3):

                # Corresponding array index
                x = 3 * i + v

                # If exist already, don't recompute
                if dump == 2:
                    fname = self._set_file_name(dump, basis, name, x=x)
                    # check whether file exists
                    if op.isfile(fname):
                        continue

                print("Atom ", i, "/", len(self.indices), " , direction ", v)

                V1t_sG = V1t_xsG[x]

                self.timer.write_now("%s-gradient of atom %u" %
                                     (['x', 'y', 'z'][v], a))

                # Array for different k-point components
                g_sqMM = np.zeros((nspins, len(kpt_u) // nspins, nao, nao),
                                  dtype)

                # 1) Gradient of effective potential
                self.timer.write_now("Starting gradient of"
                                     " effective potential")
                for kpt in kpt_u:
                    # Matrix elements
                    geff_MM = np.zeros((nao, nao), dtype)
                    bfs.calculate_potential_matrix(V1t_sG[kpt.s], geff_MM,
                                                   q=kpt.q)
                    tri2full(geff_MM, 'L')
                    # Insert in array
                    g_sqMM[kpt.s, kpt.q] += geff_MM

                self.timer.write_now("Finished gradient of "
                                     "effective potential")

                if include_pseudo:
                    self.timer.write_now("Starting gradient of pseudo part")

                    # 2) Gradient of non-local part (projectors)
                    self.timer.write_now("Starting gradient of dH^a")
                    P_aqMi = calc.wfs.P_aqMi
                    # 2a) dH^a part has contributions from all other atoms
                    for kpt in kpt_u:
                        # Matrix elements
                        gp_MM = np.zeros((nao, nao), dtype)
                        dH1_asp = dH1_xasp[x]
                        for a_, dH1_sp in dH1_asp.items():
                            dH1_ii = unpack2(dH1_sp[kpt.s])
                            gp_MM += np.dot(P_aqMi[a_][kpt.q], np.dot(dH1_ii,
                                            P_aqMi[a_][kpt.q].T.conjugate()))
                        g_sqMM[kpt.s, kpt.q] += gp_MM
                    self.timer.write_now("Finished gradient of dH^a")

                    self.timer.write_now("Starting gradient of projectors")
                    # 2b) dP^a part has only contributions from the same atoms
                    dP_qvMi = dP_aqvMi[a]
                    dH_ii = unpack2(dH_asp[str(a)][kpt.s])
                    for kpt in kpt_u:
                        # XXX Sort out the sign here; conclusion -> sign = +1 !
                        P1HP_MM = +1 * np.dot(dP_qvMi[kpt.q][v], np.dot(dH_ii,
                                              P_aqMi[a][kpt.q].T.conjugate()))
                        # Matrix elements
                        gp_MM = P1HP_MM + P1HP_MM.T.conjugate()
                        g_sqMM[kpt.s, kpt.q] += gp_MM
                    self.timer.write_now("Finished gradient of projectors")
                    self.timer.write_now("Finished gradient of pseudo part")

                # Extract R_c=(0, 0, 0) block by Fourier transforming
                if kd.gamma or kd.N_c is None:
                    g_sMM = g_sqMM[:, 0]
                else:
                    # Convert to array
                    g_sMM = []
                    for s in range(nspins):
                        g_MM = tb.bloch_to_real_space(g_sqMM[s], R_c=(0, 0, 0))
                        g_sMM.append(g_MM[0])  # [0] because of above
                    g_sMM = np.array(g_sMM)

                # Reshape to global unit cell indices
                N = np.prod(self.supercell)
                # Number of basis function in the primitive cell
                assert (nao % N) == 0, "Alarm ...!"
                nao_cell = nao // N
                g_sNMNM = g_sMM.reshape((nspins, N, nao_cell, N, nao_cell))
                g_sNNMM = g_sNMNM.swapaxes(2, 3).copy()
                self.timer.write_now("Finished supercell matrix")

                if dump != 2:
                    g_xsNNMM.append(g_sNNMM)
                else:
                    # filename should already be known
                    # fname = _set_file_name(dump, basis, name, x=x)
                    if kd.comm.rank == 0:
                        fd = open(fname, 'wb')
                        M_a = self.basis_info['M_a']
                        nao_a = self.basis_info['nao_a']
                        # backward compatibility
                        # if nspins == 1:
                        #    pickle.dump((g_sNNMM[0], M_a, nao_a), fd, 2)
                        # else:
                        pickle.dump((g_sNNMM, M_a, nao_a), fd, 2)
                        fd.close()

        self.timer.write_now("Finished gradient of PAW Hamiltonian")

        if dump != 2:
            # Collect gradients in one array
            self.g_xsNNMM = np.array(g_xsNNMM)

            # Dump to pickle file using binary mode together with basis info
            if dump and kd.comm.rank == 0:
                fname = self._set_file_name(dump, basis, name)
                fd = open(fname, 'wb')
                M_a = self.basis_info['M_a']
                nao_a = self.basis_info['nao_a']
                # backward compatibility
                # if nspins == 1:
                #     pickle.dump((self.g_xsNNMM[0], M_a, nao_a), fd, 2)
                # else:
                pickle.dump((self.g_xsNNMM, M_a, nao_a), fd, 2)
                fd.close()

    def load_supercell_matrix(self, basis=None, name=None, dump=0):
        """Load supercell matrix from pickle file.

        Parameters
        ----------
        basis: str
            String specifying the LCAO basis used to calculate the supercell
            matrix, e.g. 'dz(dzp)'.
        name: str
            User specified name of the pickle file.
        dump: int
            Dump supercell matrix to pickle file (default: 0).

            0: Supercell matrix not saved by calculate_supercell_matrix

            1: Supercell matrix was saved in a single pickle file.

            2: Dumped matrix for different gradients in separate files.
        """

        assert (basis is not None) or (name is not None), \
            "Provide basis or name."
        assert dump in (0, 1, 2)

        if dump == 0:
            # Nothing to load, just make sure everything is there
            assert self.g_xsNNMM is not None
            assert self.basis_info is not None
        elif dump == 1:
            fname = self._set_file_name(dump, basis, name)
            fd = open(fname, 'rb')
            self.g_xsNNMM, M_a, nao_a = pickle.load(fd)
            fd.close()
        else:  # dump == 2
            g_xsNNMM = []
            for x in range(len(self.indices) * 3):
                fname = self._set_file_name(dump, basis, name)
                fd = open(fname, 'rb')
                g_sNNMM, M_a, nao_a = pickle.load(fd)
                fd.close()
                g_xsNNMM.append(g_sNNMM)
            self.g_xsNNMM = np.array(g_xsNNMM)

        self.set_basis_info(M_a, nao_a)

    def apply_cutoff(self, cutmax=None, cutmin=None):
        """Zero matrix element inside/beyond the specified cutoffs.

        Parameters
        ----------
        cutmax: float
            Zero matrix elements for basis functions with a distance to the
            atomic gradient that is larger than the cutoff.
        cutmin: float
            Zero matrix elements where both basis functions have distances to
            the atomic gradient that is smaller than the cutoff.

        This method is not tested.
        """

        if cutmax is not None:
            cutmax = float(cutmax)
        if cutmin is not None:
            cutmin = float(cutmin)

        # Reference to supercell matrix attribute
        g_xsNNMM = self.g_xsNNMM

        # Number of atoms and primitive cells
        N_atoms = len(self.indices)
        N = np.prod(self.supercell)
        nao = g_xsNNMM.shape[-1]
        nspins = g_xsNNMM.shape[1]

        # Reshape array
        g_avsNNMM = g_xsNNMM.reshape(N_atoms, 3, nspins, N, N, nao, nao)

        # Make slices for orbitals on atoms
        M_a = self.basis_info['M_a']
        nao_a = self.basis_info['nao_a']
        slice_a = []
        for a in range(len(self.atoms)):
            start = M_a[a]
            stop = start + nao_a[a]
            s = slice(start, stop)
            slice_a.append(s)

        # Lattice vectors
        R_cN = self.compute_lattice_vectors()

        # Unit cell vectors
        cell_vc = self.atoms.cell.transpose()
        # Atomic positions in reference cell
        pos_av = self.atoms.get_positions()

        # Create a mask array to zero the relevant matrix elements
        if cutmin is not None:
            mask_avsNNMM = np.zeros(g_avsNNMM.shape, dtype=bool)

        # Zero elements where one of the basis orbitals has a distance to atoms
        # (atomic gradients) in the reference cell larger than the cutoff
        for n in range(N):
            # Lattice vector to cell
            R_v = np.dot(cell_vc, R_cN[:, n])
            # Atomic positions in cell
            posn_av = pos_av + R_v
            for i, a in enumerate(self.indices):
                # Atomic distances wrt to the position of the gradient
                dist_a = np.sqrt(np.sum((pos_av[a] - posn_av)**2, axis=-1))

                if cutmax is not None:
                    # Atoms indices where the distance is larger than the max
                    # cufoff
                    j_a = np.where(dist_a > cutmax)[0]
                    # Zero elements
                    for j in j_a:
                        g_avsNNMM[a, :, :, n, :, slice_a[j], :] = 0.0
                        g_avsNNMM[a, :, :, :, n, :, slice_a[j]] = 0.0

                if cutmin is not None:
                    # Atoms indices where the distance is larger than the min
                    # cufoff
                    j_a = np.where(dist_a > cutmin)[0]
                    # Update mask to keep elements where one LCAO is outside
                    # the min cutoff
                    for j in j_a:
                        mask_avsNNMM[a, :, :, n, :, slice_a[j], :] = True
                        mask_avsNNMM[a, :, :, :, n, :, slice_a[j]] = True

        # Zero elements where both LCAOs are located within the min cutoff
        if cutmin is not None:
            g_avsNNMM[~mask_avsNNMM] = 0.0

    def lcao_matrix(self, u_l, omega_l):
        """Calculate the el-ph coupling in the electronic LCAO basis.

        For now, only works for Gamma-point phonons.

        Parameters
        ----------
        u_l: ndarray
            Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of the
            phonons.
        omega_l: ndarray
            Vibrational frequencies in eV.

        This method is not tested.
        """

        # Supercell matrix (Hartree / Bohr)
        assert self.g_xsNNMM is not None, "Load supercell matrix."
        assert self.g_xsNNMM.shape[2:4] == (1, 1)
        g_xsMM = self.g_xsNNMM[:, :, 0, 0, :, :]
        # Number of atomic orbitals
        # nao = g_xMM.shape[-1]
        # Number of phonon modes
        nmodes = u_l.shape[0]

        #
        u_lx = u_l.reshape(nmodes, 3 * len(self.atoms))
        # np.dot uses second to last index of second array
        g_lsMM = np.dot(u_lx, g_xsMM.transpose(2, 0, 1, 3))

        # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
        amu = units._amu  # atomic mass unit
        me = units._me   # electron mass
        g_lsMM /= np.sqrt(2 * amu / me / units.Hartree *
                          omega_l[:, :, np.newaxis, np.newaxis])
        # Convert to eV
        g_lsMM *= units.Hartree

        return g_lsMM

    def bloch_matrix(self, kpts, qpts, c_kn, u_ql,
                     omega_ql=None, kpts_from=None, spin=0):
        r"""Calculate el-ph coupling in the Bloch basis for the electrons.

        This function calculates the electron-phonon coupling between the
        specified Bloch states, i.e.::

                      ______
            mnl      / hbar               ^
           g    =   /-------  < m k + q | e  . grad V  | n k >
            kq    \/ 2 M w                 ql        q
                          ql

        In case the ``omega_ql`` keyword argument is not given, the bare matrix
        element (in units of eV / Ang) without the sqrt prefactor is returned.

        Phonon frequencies and mode vectors must be given in
        ase units.

        Parameters
        ----------
        kpts: ndarray or tuple
            k-vectors of the Bloch states. When a tuple of integers is given, a
            Monkhorst-Pack grid with the specified number of k-points along the
            directions of the reciprocal lattice vectors is generated.
        qpts: ndarray or tuple
            q-vectors of the phonons.
        c_kn: ndarray
            Expansion coefficients for the Bloch states. The ordering must be
            the same as in the ``kpts`` argument.
        u_ql: ndarray
            Mass-scaled polarization vectors (in units of 1 / sqrt(amu)) of the
            phonons. Again, the ordering must be the same as in the
            corresponding ``qpts`` argument.
        omega_ql: ndarray
            Vibrational frequencies in eV.
        kpts_from: List[int] or int
            Calculate only the matrix element for the k-vectors specified by
            their index in the ``kpts`` argument (default: all).
        spin: int
            In case of spin-polarised system, define which spin to use
            (0 or 1).


        """

        assert self.g_xsNNMM is not None, "Load supercell matrix."
        assert len(c_kn.shape) == 3
        assert len(u_ql.shape) == 4
        if omega_ql is not None:
            assert np.all(u_ql.shape[:2] == omega_ql.shape[:2])

        # Translate k-points into 1. BZ (required by ``find_k_plus_q``` member
        # function of the ```KPointDescriptor``).
        if isinstance(kpts, np.ndarray):
            assert kpts.shape[1] == 3, "kpts_kc array must be given"
            # XXX This does not seem to cause problems!
            kpts -= kpts.round()

        # Use the KPointDescriptor to keep track of the k and q-vectors
        kd_kpts = KPointDescriptor(kpts)
        kd_qpts = KPointDescriptor(qpts)
        # Check that number of k- and q-points agree with the number of Bloch
        # functions and polarization vectors
        assert kd_kpts.nbzkpts == len(c_kn)
        assert kd_qpts.nbzkpts == len(u_ql)

        # Include all k-point per default
        if kpts_from is None:
            kpts_kc = kd_kpts.bzk_kc
            kpts_k = range(kd_kpts.nbzkpts)
        else:
            kpts_kc = kd_kpts.bzk_kc[kpts_from]
            if isinstance(kpts_from, int):
                kpts_k = list([kpts_from])
            else:
                kpts_k = list(kpts_from)

        # Supercell matrix (real matrix in Hartree / Bohr)
        g_xNNMM = self.g_xsNNMM[:, spin]

        # Number of phonon modes and electronic bands
        nmodes = u_ql.shape[1]
        nbands = c_kn.shape[1]
        # Number of atoms displacements and basis functions
        ndisp = np.prod(u_ql.shape[2:])
        assert ndisp == (3 * len(self.indices))
        nao = c_kn.shape[2]
        assert ndisp == g_xNNMM.shape[0]
        assert nao == g_xNNMM.shape[-1]

        # Lattice vectors
        R_cN = self.compute_lattice_vectors()
        # Number of unit cell in supercell
        N = np.prod(self.supercell)

        # Allocate array for couplings
        g_qklnn = np.zeros((kd_qpts.nbzkpts, len(kpts_kc), nmodes,
                            nbands, nbands), dtype=complex)

        self.timer.write_now("Calculating coupling matrix elements")
        for q, q_c in enumerate(kd_qpts.bzk_kc):

            # Find indices of k+q for the k-points
            kplusq_k = kd_kpts.find_k_plus_q(q_c, kpts_k=kpts_k)

            # Here, ``i`` is counting from 0 and ``k`` is the global index of
            # the k-point
            for i, (k, k_c) in enumerate(zip(kpts_k, kpts_kc)):

                # Check the wave vectors (adapted to the ``KPointDescriptor``
                # class)
                kplusq_c = k_c + q_c
                kplusq_c -= kplusq_c.round()
                assert np.allclose(kplusq_c, kd_kpts.bzk_kc[kplusq_k[i]]), \
                    (i, k, k_c, q_c, kd_kpts.bzk_kc[kplusq_k[i]])

                # Allocate array
                g_xMM = np.zeros((ndisp, nao, nao), dtype=complex)

                # Multiply phase factors
                for m in range(N):
                    for n in range(N):
                        Rm_c = R_cN[:, m]
                        Rn_c = R_cN[:, n]
                        phase = np.exp(2.j * pi * (np.dot(k_c, Rm_c - Rn_c)
                                                   + np.dot(q_c, Rm_c)))
                        # Sum contributions from different cells
                        g_xMM += g_xNNMM[:, m, n, :, :] * phase

                # LCAO coefficient for Bloch states
                ck_nM = c_kn[k]
                ckplusq_nM = c_kn[kplusq_k[i]]
                # Mass scaled polarization vectors
                u_lx = u_ql[q].reshape(nmodes, 3 * len(self.atoms))

                g_nxn = np.dot(ckplusq_nM.conj(), np.dot(g_xMM, ck_nM.T))
                g_lnn = np.dot(u_lx, g_nxn)

                # Insert value
                g_qklnn[q, i] = g_lnn

                # XXX Temp
                if np.all(q_c == 0.0):
                    # These should be real
                    print(g_qklnn[q].imag.min(), g_qklnn[q].imag.max())

        self.timer.write_now("Finished calculation of "
                             "coupling matrix elements")

        # Return the bare matrix element if frequencies are not given
        if omega_ql is None:
            # Convert to eV / Ang
            g_qklnn *= units.Hartree / units.Bohr
        else:
            # Multiply prefactor sqrt(hbar / 2 * M * omega) in units of Bohr
            amu = units._amu  # atomic mass unit
            me = units._me   # electron mass
            g_qklnn /= np.sqrt(2 * amu / me / units.Hartree *
                               omega_ql[:, np.newaxis, :,
                                        np.newaxis, np.newaxis])
            # Convert to eV
            g_qklnn *= units.Hartree

        # Return couplings in eV (or eV / Ang)
        return g_qklnn

    def fourier_filter(self, V1t_xG, components='normal', criteria=1):
        """Fourier filter atomic gradients of the effective potential.

        Parameters
        ----------
        V1t_xG: ndarray
            Array representation of atomic gradients of the effective potential
            in the supercell grid.
        components: str
            Fourier components to filter out (``normal`` or ``umklapp``).

        This method is not tested.
        """
        import numpy.fft as fft
        import numpy.linalg as la
        assert components in ['normal', 'umklapp']
        # Grid shape
        shape = V1t_xG.shape[-3:]

        # Primitive unit cells in Bohr/Bohr^-1
        cell_cv = self.atoms.get_cell() / units.Bohr
        reci_vc = 2 * pi * la.inv(cell_cv)
        norm_c = np.sqrt(np.sum(reci_vc**2, axis=0))
        # Periodic BC array
        pbc_c = np.array(self.atoms.get_pbc(), dtype=bool)

        # Supercell atoms and cell
        atoms_N = self.atoms * self.supercell
        supercell_cv = atoms_N.get_cell() / units.Bohr

        # q-grid in units of the grid spacing (FFT ordering)
        q_cG = np.indices(shape).reshape(3, -1)
        q_c = np.array(shape)[:, np.newaxis]
        q_cG += q_c // 2
        q_cG %= q_c
        q_cG -= q_c // 2

        # Locate q-points inside the Brillouin zone
        if criteria == 0:
            # Works for all cases
            # Grid spacing in direction of reciprocal lattice vectors
            h_c = np.sqrt(np.sum((2 * pi * la.inv(supercell_cv))**2, axis=0))
            # XXX Why does a "*=" operation on q_cG not work here ??
            q1_cG = q_cG * h_c[:, np.newaxis] / (norm_c[:, np.newaxis] / 2)
            mask_G = np.ones(np.prod(shape), dtype=bool)
            for i, pbc in enumerate(pbc_c):
                if not pbc:
                    continue
                mask_G &= (-1. < q1_cG[i]) & (q1_cG[i] <= 1.)
        else:
            # 2D hexagonal lattice
            # Projection of q points onto the periodic directions. Only in
            # these directions do normal and umklapp processees make sense.
            q_vG = np.dot(q_cG[pbc_c].T,
                          2 * pi * la.inv(supercell_cv).T[pbc_c]).T.copy()
            # Parametrize the BZ boundary in terms of the angle theta
            theta_G = np.arctan2(q_vG[1], q_vG[0]) % (pi / 3)
            phi_G = pi / 6 - np.abs(theta_G)
            qmax_G = norm_c[0] / 2 / np.cos(phi_G)
            norm_G = np.sqrt(np.sum(q_vG**2, axis=0))
            # Includes point on BZ boundary with +1e-2
            mask_G = (norm_G <= qmax_G + 1e-2)

        if components != 'normal':
            mask_G = ~mask_G

        # Reshape to grid shape
        mask_G.shape = shape

        for V1t_G in V1t_xG:
            # Fourier transform atomic gradient
            V1tq_G = fft.fftn(V1t_G)
            # Zero normal/umklapp components
            V1tq_G[mask_G] = 0.0
            # Fourier transform back
            V1t_G[:] = fft.ifftn(V1tq_G).real

    def calculate_gradient(self):
        """Calculate gradient of effective potential and projector coefs.

        This function loads the generated pickle files and calculates
        finite-difference derivatives.

        """

        # Array and dict for finite difference derivatives
        V1t_xsG = []
        dH1_xasp = []

        x = 0
        for a in self.indices:
            for v in 'xyz':
                # Note: self.name currently ignored in ase.phonon
                # name = '%s.%d%s' % (self.name, a, v)
                name = '%d%s' % (a, v)
                # Potential and atomic density matrix for atomic displacement
                Vtm_sG = self.cache[name + '-']['Vt_sG']
                dHm_asp = self.cache[name + '-']['dH_all_asp']
                Vtp_sG = self.cache[name + '+']['Vt_sG']
                dHp_asp = self.cache[name + '+']['dH_all_asp']

                # FD derivatives in Hartree / Bohr
                V1t_sG = (Vtp_sG - Vtm_sG) / (2 * self.delta / units.Bohr)
                V1t_xsG.append(V1t_sG)

                dH1_asp = {}
                for atom in dHm_asp.keys():
                    dH1_asp[atom] = (dHp_asp[atom] - dHm_asp[atom]) / \
                                    (2 * self.delta / units.Bohr)
                dH1_xasp.append(dH1_asp)
                x += 1

        return np.array(V1t_xsG), dH1_xasp
