import numpy as np
from scipy.spatial import Delaunay, cKDTree

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.bztools import get_reduced_bz, unique_rows
from _gpaw import GG_shuffle

from gpaw.response import timer


class CharacterTableBuilder:
    def __init__(self, ops_occ, verbose=True):
        print('Order of little group:', len(ops_occ))
        self.verbose = verbose
        self.ops_occ = ops_occ
        self.unknown_cc=0
        self.unknown_ir=0
        #self.ops_occ = np.einsum('odc,cv,dw->owv', ops_occ, cell_cv, np.linalg.inv(cell_cv.T))
        #self.ops_occ = np.einsum('cv,odc,dw->owv', cell_cv, ops_occ, np.linalg.inv(cell_cv.T))
        self._build_multiplication_table()
        self._find_conjugacy_classes()
        self._build_character_table()
        self._detect_conjugacy_classes()
        self._detect_irreps()
        self.print_character_table()

    def _class_id(self, classname):
        try:
            return self.names_g.index(classname)
        except ValueError:
            return None

    def _detect_irrep(self, signature):
        h = signature[ self._class_id("E") ]
        #rotations = [ self.class_id(name) for name in self.names_g if ("C" in name) ]
        rotations = self._class_id("6C2")
        if self._class_id("i"):
            ug = "g" if signature[ self._class_id("i") ] > 0 else "u"
        else:
            ug = ""
        C = "?"
        N = "?"
        if self._class_id("6C4"):
            if signature[self._class_id("6C4")]>0:
                N = "1"
            else:
                N = "2"
        if h == 1:
            C = "A"
        if h == 2:
            C = "E"
            N = ""
        if h == 3:
            C="T"

        s = str(self.unknown_ir)+" "+C+N+ug
        self.unknown_ir += 1
        return s

    def _detect_irreps(self):
        self.names_i = [self._detect_irrep(self.character_ig[i, :]) for i in range(self.character_ig.shape[0])]

    def _detect_conjugacy_class(self, ops_occ):
        det_o = np.array([np.linalg.det(op_cc) for op_cc in ops_occ])
        eigs_o = np.array([np.sort(np.linalg.eig(op_cc)[0]) for op_cc in ops_occ])
        if len(det_o) == 1 and np.all(np.isclose(eigs_o, [-1, -1, 1])):
            s="C2"
        elif len(det_o) == 1 and np.all(np.isclose(eigs_o, [-1, -1, -1])):
            s= "i"  # Inversion flips all axes, -x, -y, -z
        elif len(det_o) == 1 and np.all(np.isclose(eigs_o, [1, 1, 1])):
            s= "E"  # Identity leaves all axes intact x, y, z
        elif len(det_o) == 3 and np.all(np.isclose(eigs_o, [-1, -1, 1])):
            s= "3C2"  # C2 rotation along z is -x, -y, z
        elif len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1, -1, 1])):
            s= "6C2"
        elif len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1, -1j, 1j])):
            s= "6S4"  # -1j and 1j corresponds to 90 rotation. Determinant is -1 thus, improper.
        elif len(det_o) == 3 and np.all(np.isclose(eigs_o, [-1, 1, 1])):
            s= "3sh"  # Horizontal mirror operation flips one of the coordinates
        elif len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1, 1, 1])):
            s= "6sd"
        elif len(det_o) == 8 and np.all(np.isclose(eigs_o, [-1, np.exp(-1j * 2 * np.pi / 6), np.exp(1j * 2 * np.pi / 6)])):
            s= "8S6"
        elif len(det_o) == 6 and np.all(np.isclose(eigs_o, [-1j, 1j, 1])):
            s= "6C4"
        elif len(det_o) == 8 and np.all(np.isclose(eigs_o, [np.exp(-1j * np.pi * 2 / 3), np.exp(1j * np.pi * 2 / 3), 1])):
            s= "8C3"
        else:
            s = "?"

        if s == "?":
            s = s+str(self.unknown_cc)
            self.unknown_cc += 1
        return s


    @classmethod
    def from_PWSymmetryAnalyzer(cls, pwsym):
        """
           Symmetry operator including translation

          [ U_11 U_12 U_13  X   ] [ a ]
          [ U_21 U_22 U_23  Y   ] [ b ]
          [ U_31 U_32 U_33  Z   ] [ c ]
          [ 0    0    0     1   ] [ 1 ]

        """
        cell_cv = pwsym.qpd.gd.cell_cv

        #print(pwsym.qpd.q_c,'q_c', np.linalg.inv(cell_cv.T) @ pwsym.qpd.q_c)
        ops = [pwsym.get_symmetry_operator(pwsym.s_s[s]) 
               for s in range(pwsym.how_many_symmetries())]
        ops_ozz = []
        for op in ops:
            op_zz = np.zeros((4,4))
            #op_cc = np.einsum('cv,dc,dw->wv', cell_cv, op[1]*op[0], np.linalg.inv(cell_cv.T))
            op_cc = op[0] * op[1]
            #print('opcc', op_cc)
            #op2_cc = np.einsum('cv,dc,dw->wv', np.linalg.inv(cell_cv.T), np.linalg.inv(op[1]*op[0]), cell_cv)
            #print('op2cc', op2_cc)
            op_zz[:3, :3] = op_cc
            op_zz[3,3] = 1.0
            assert np.all(op[4] == 0)
            ops_ozz.append(op_zz[:3,:3])
        
        return cls(np.array(ops_ozz))

    def get_symmetry_operator(self, s):
        """Return symmetry operator s."""
        U_scc = self.kd.symmetry.op_scc
        ft_sc = self.kd.symmetry.op_scc

        reds = s % self.nU
        if self.timereversal(s):
            TR = np.conj
            sign = -1
        else:
            sign = 1

            def TR(x):
                return x

        return sign*U_scc[reds], sign, TR, self.shift_sc[s], ft_sc[reds]
    
    def get_op_id(self, op_cc):
        for o1, op1_cc in enumerate(self.ops_occ):
            if np.linalg.norm(op_cc - op1_cc) < 1e-8:
                return o1
        raise ValueError("Unknown operation: %s." % str(op_cc))

    def _build_multiplication_table(self):
        ops_occ = self.ops_occ
        N = len(ops_occ)
        mul_oo = np.zeros((N, N), dtype=int)
        inv_oo = np.zeros((N, N), dtype=int)
        for o1, op1_cc in enumerate(ops_occ):
            for o2, op2_cc in enumerate(ops_occ):
                #print(op1_cc,'*', op2_cc, '=', np.dot(op1_cc, op2_cc))
                o3 = self.get_op_id(np.dot(op1_cc, op2_cc))
                mul_oo[o1, o2] = o3
                inv_oo[o1, o3] = o2
        self.mul_oo, self.inv_oo = mul_oo, inv_oo

    def _find_conjugacy_classes(self):
        ops_occ = self.ops_occ
        N = len(ops_occ)
        op_pool = range(N)
        self.ops_g = []

        self.g_o = np.zeros((N,), int)  # Conjugacy class index for each op
        iteration = 0
        while True:
            iteration += 1
            if iteration > 1000:
                raise RuntimeError("Cannot find conjugacy classes: remaining "+str(op_pool))
            # Loop until all operations are assigned a class
            if len(op_pool) == 0:
                break
            # Take an element from operation pool...
            op1 = op_pool[0]
            op1_cc = ops_occ[op1]
            # ...conjugate it with all possible operations, see the result, and remove any duplicates
            conjugacy_class = np.unique([self.get_op_id(np.dot(op2_cc, np.dot(op1_cc, np.linalg.inv(op2_cc)))) for o2, op2_cc in enumerate(ops_occ)])

            # Fill g_o array that maps ops to class
            for op in conjugacy_class:
                self.g_o[op] = len(self.ops_g)
            self.ops_g.append(conjugacy_class)
            # Remove all operations already assigned a class from the pool
            op_pool = list(set(op_pool) - set(conjugacy_class))

        if self.verbose:
            print("Found %d conjugacy classes" % len(self.ops_g))
    
    def diagonalize_and_group(self, H_oo):
        eps, psi = np.linalg.eigh(H_oo)
        groups = np.unique(abs(eps[:, None] - eps[None, :]) < 1e-6, axis=0)
        if self.verbose:
            print("Found %d irreducible representations." % len(groups))

        # Add eigenvales to lists
        groups_i = [[] for i in range(len(groups))]
        for irrep, eig_idx in zip(*np.where(groups)):
            groups_i[irrep].append(psi[:, eig_idx])

        return [np.array(x) for x in groups_i]

    def _build_character_table(self):
        ops_occ = self.ops_occ
        N = len(ops_occ)

        # Diagonalize arbitrary Hamiltonian (the form 1/(1+g) is irrelevant)
        # to numerically build the character table. The degenerate eigenspaces
        # describe the irreducible representations.
        # There might be an accidental degeneracy, in which case the a representation
        # might end up being a direct product of two representations.
        H_oo = 1 / (self.g_o[self.inv_oo] + 1)
        groups_i = self.diagonalize_and_group(1 / (self.g_o[self.inv_oo] + 1.321))
        self.groups_i = groups_i
        character_ig = np.zeros((len(groups_i), len(self.ops_g)), dtype=int)

        # For each group...
        for i, psi_no in enumerate(groups_i):
            # For each conjugacy class...
            for g, ops in enumerate(self.ops_g):
                # It is not necessary to loop over all group operations. However, it will be a good sanity check.
                # Calculate the trace of this irrep under operation o
                traces_o = np.array([np.trace(np.dot(psi_no[:, self.mul_oo[:, o]], psi_no.T.conjugate())) for o in ops])
                h = len(psi_no)**0.5
                assert(np.all(abs(traces_o - traces_o[0]) < 1e-10))
                character_ig[i, g] = int(np.round(traces_o[0] / h))

        self.character_ig = character_ig
    
    def _detect_conjugacy_classes(self):
        self.names_g = [self._detect_conjugacy_class([self.ops_occ[o] for o in classops]) for classops in self.ops_g]
        #print(self.names_g)

    def class_id(self, classname):
        try:
            return self.names_g.index(classname)
        except ValueError:
            return None

    def _detect_irreps(self):
        self.names_i = [self._detect_irrep(self.character_ig[i, :]) for i in range(self.character_ig.shape[0])]

    def print_character_table(self, class_order=None, irrep_order=None):
        try:
            print("%-20s" % "irreps/classes", end="")
            if class_order is None:
                class_order = self.names_g
            if irrep_order is None:
                irrep_order = self.names_i

            for name in class_order:
                print("%-5s" % name, end="")
            print()
            for iname in irrep_order:
                i = self.names_i.index(iname)
                print("%-20s" % self.names_i[i], end="")
                for gname in class_order:
                    g = self.names_g.index(gname)
                    print("%-5s" % ("%+02d" % self.character_ig[i, g]), end="")
                print()
        except ValueError:
            print('...')


class KPointFinder:
    def __init__(self, bzk_kc):
        self.kdtree = cKDTree(self._round(bzk_kc))

    @staticmethod
    def _round(bzk_kc):
        return np.mod(np.mod(bzk_kc, 1).round(6), 1)

    def find(self, kpt_c):
        distance, k = self.kdtree.query(self._round(kpt_c))
        if distance > 1.e-6:
            raise ValueError('Requested k-point is not on the grid. '
                             'Please check that your q-points of interest '
                             'are commensurate with the k-point grid.')

        return k


class PWSymmetryAnalyzer:
    """Class for handling planewave symmetries."""

    def __init__(self, kpoints, qpd, context,
                 disable_point_group=False,
                 disable_time_reversal=False):
        print(disable_time_reversal)
        """Creates a PWSymmetryAnalyzer object.

        Determines which of the symmetries of the atomic structure
        that is compatible with the reciprocal lattice. Contains the
        necessary functions for mapping quantities between kpoints,
        and or symmetrizing arrays.

        kd: KPointDescriptor
            The kpoint descriptor containing the
            information about symmetries and kpoints.
        qpd: SingleQPWDescriptor
            Plane wave descriptor that contains the reciprocal
            lattice .
        context: ResponseContext
        disable_point_group: bool
            Switch for disabling point group symmetries.
        disable_time_reversal:
            Switch for disabling time reversal.
        """
        self.qpd = qpd
        self.kd = kd = kpoints.kd
        self.context = context

        # Settings
        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        if (kd.symmetry.has_inversion or not kd.symmetry.time_reversal) and \
           not self.disable_time_reversal:
            self.context.print('\nThe ground calculation does not support time'
                               '-reversal symmetry possibly because it has an '
                               'inversion center or that it has been manually '
                               'deactivated.\n')
            self.disable_time_reversal = True

        self.disable_symmetries = (self.disable_point_group and
                                   self.disable_time_reversal)

        # Number of symmetries
        U_scc = kd.symmetry.op_scc
        self.nU = len(U_scc)

        self.nsym = 2 * self.nU
        self.use_time_reversal = not self.disable_time_reversal

        self.kptfinder = kpoints.kptfinder
        self.initialize()

    @timer('Initialize')
    def initialize(self):
        """Initialize relevant quantities."""
        self.infostring = ''

        if self.disable_point_group:
            self.infostring += 'Point group not included. '
        else:
            self.infostring += 'Point group included. '

        if self.disable_time_reversal:
            self.infostring += 'Time reversal not included. '
        else:
            self.infostring += 'Time reversal included. '

        self.infostring += 'Disabled non-symmorphic symmetries. '

        if self.disable_symmetries:
            self.infostring += 'All symmetries have been disabled. '

        # Do the work
        self.analyze_symmetries()
        self.analyze_kpoints()
        self.initialize_G_maps()

        # Print info
        self.context.print(self.infostring)
        self.print_symmetries()

    def print_symmetries(self):
        """Handsome print function for symmetry operations."""
        isl = ['\n']
        nx = 6  # You are not allowed to use non-symmorphic syms (value 3)
        ns = len(self.s_s)
        y = 0
        for y in range((ns + nx - 1) // nx):
            for c in range(3):
                tisl = []
                for x in range(nx):
                    s = x + y * nx
                    if s == ns:
                        break
                    tmp = self.get_symmetry_operator(self.s_s[s])
                    op_cc, sign, TR, shift_c, ft_c = tmp
                    op_c = sign * op_cc[c]
                    tisl.append(f'  ({op_c[0]:2d} {op_c[1]:2d} {op_c[2]:2d})')
                tisl.append('\n')
                isl.append(''.join(tisl))
            isl.append('\n')
        self.context.print(''.join(isl))  # flush output
        print(''.join(isl))  # flush output

    @timer('Analyze')
    def analyze_kpoints(self):
        """Calculate the reduction in the number of kpoints."""
        K_gK = self.group_kpoints()
        ng = len(K_gK)
        self.infostring += f'{ng} groups of equivalent kpoints. '
        percent = (1. - (ng + 0.) / self.kd.nbzkpts) * 100
        self.infostring += f'{percent}% reduction. '

    @timer('Analyze symmetries.')
    def analyze_symmetries(self):
        r"""Determine allowed symmetries.

        An direct symmetry U must fulfill::

          U \mathbf{q} = q + \Delta

        Under time-reversal (indirect) it must fulfill::

          -U \mathbf{q} = q + \Delta

        where :math:`\Delta` is a reciprocal lattice vector.
        """
        qpd = self.qpd

        # Shortcuts
        q_c = qpd.q_c
        kd = self.kd

        U_scc = kd.symmetry.op_scc
        nU = self.nU
        nsym = self.nsym

        shift_sc = np.zeros((nsym, 3), int)
        conserveq_s = np.zeros(nsym, bool)

        newq_sc = np.dot(U_scc, q_c)

        # Direct symmetries
        dshift_sc = (newq_sc - q_c[np.newaxis]).round().astype(int)
        inds_s = np.argwhere((newq_sc == q_c[np.newaxis] + dshift_sc).all(1))
        conserveq_s[inds_s] = True

        shift_sc[:nU] = dshift_sc

        # Time reversal
        trshift_sc = (-newq_sc - q_c[np.newaxis]).round().astype(int)
        trinds_s = np.argwhere((-newq_sc == q_c[np.newaxis] +
                                trshift_sc).all(1)) + nU
        conserveq_s[trinds_s] = True
        shift_sc[nU:nsym] = trshift_sc

        # The indices of the allowed symmetries
        s_s = conserveq_s.nonzero()[0]

        # Filter out disabled symmetries
        if self.disable_point_group:
            s_s = list(filter(self.is_not_point_group, s_s))

        if self.disable_time_reversal:
            before = len(s_s)
            s_s = list(filter(self.is_not_time_reversal, s_s))
            after = len(s_s)
            print('Time reversal: Before', before, 'After', after)

        # You are not allowed to use non-symmorphic syms, sorry. So we remove
        # the option and always filter those symmetries out.
        before = len(s_s)
        s_s = list(filter(self.is_not_non_symmorphic, s_s))
        after = len(s_s)
        print('Non symmorphic: Before', before, 'After', after)

#        stmp_s = []
#        for s in s_s:
#            if self.kd.bz2bz_ks[0, s] == -1:
#                assert (self.kd.bz2bz_ks[:, s] == -1).all()
#            else:
#                stmp_s.append(s)

#        s_s = stmp_s

        self.infostring += f'Found {len(s_s)} allowed symmetries. '
        self.s_s = s_s
        self.shift_sc = shift_sc

    def is_not_point_group(self, s):
        U_scc = self.kd.symmetry.op_scc
        nU = self.nU
        return (U_scc[s % nU] == np.eye(3)).all()

    def is_not_time_reversal(self, s):
        nU = self.nU
        return not bool(s // nU)

    def is_not_non_symmorphic(self, s):
        ft_sc = self.kd.symmetry.ft_sc
        nU = self.nU
        return not bool(ft_sc[s % nU].any())

    def how_many_symmetries(self):
        """Return number of symmetries."""
        return len(self.s_s)

    @timer('Group kpoints')
    def group_kpoints(self, K_k=None):
        """Group kpoints according to the reduced symmetries"""
        if K_k is None:
            K_k = np.arange(self.kd.nbzkpts)
        s_s = self.s_s
        bz2bz_ks = self.kd.bz2bz_ks
        nk = len(bz2bz_ks)
        sbz2sbz_ks = bz2bz_ks[K_k][:, s_s]  # Reduced number of symmetries
        # Avoid -1 (see documentation in gpaw.symmetry)
        sbz2sbz_ks[sbz2sbz_ks == -1] = nk

        smallestk_k = np.sort(sbz2sbz_ks)[:, 0]
        k2g_g = np.unique(smallestk_k, return_index=True)[1]

        K_gs = sbz2sbz_ks[k2g_g]
        K_gk = [np.unique(K_s[K_s != nk]) for K_s in K_gs]

        return K_gk

    def get_BZ(self):
        # Get the little group of q
        U_scc = []
        for s in self.s_s:
            U_cc, sign, _, _, _ = self.get_symmetry_operator(s)
            U_scc.append(sign * U_cc)
        U_scc = np.array(U_scc)

        # Determine the irreducible BZ
        bzk_kc, ibzk_kc = get_reduced_bz(self.qpd.gd.cell_cv,
                                         U_scc,
                                         False)

        return bzk_kc

    def get_reduced_kd(self, *, pbc_c):
        # Get the little group of q
        U_scc = []
        for s in self.s_s:
            U_cc, sign, _, _, _ = self.get_symmetry_operator(s)
            U_scc.append(sign * U_cc)
        U_scc = np.array(U_scc)

        # Determine the irreducible BZ
        bzk_kc, ibzk_kc = get_reduced_bz(self.qpd.gd.cell_cv,
                                         U_scc,
                                         False,
                                         pbc_c=pbc_c)

        n = 3
        N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2

        # Find the irreducible kpoints
        tess = Delaunay(ibzk_kc)
        ik_kc = []
        for N_c in N_xc:
            k_kc = self.kd.bzk_kc + N_c
            k_kc = k_kc[tess.find_simplex(k_kc) >= 0]
            if not len(ik_kc) and len(k_kc):
                ik_kc = unique_rows(k_kc)
            elif len(k_kc):
                ik_kc = unique_rows(np.append(k_kc, ik_kc, axis=0))

        return KPointDescriptor(ik_kc)

    def unfold_kpoints(self, points_pv, tol=1e-8, mod=None):
        points_pc = np.dot(points_pv, self.qpd.gd.cell_cv.T) / (2 * np.pi)

        # Get the little group of q
        U_scc = []
        for s in self.s_s:
            U_cc, sign, _, _, _ = self.get_symmetry_operator(s)
            U_scc.append(sign * U_cc)
        U_scc = np.array(U_scc)

        points = np.concatenate(np.dot(points_pc, U_scc.transpose(0, 2, 1)))
        points = unique_rows(points, tol=tol, mod=mod)
        points = np.dot(points, self.qpd.gd.icell_cv) * (2 * np.pi)
        return points

    def get_kpoint_weight(self, k_c):
        K = self.kptfinder.find(k_c)
        iK = self.kd.bz2ibz_k[K]
        K_k = self.unfold_ibz_kpoint(iK)
        K_gK = self.group_kpoints(K_k)

        for K_k in K_gK:
            if K in K_k:
                return len(K_k)

    def get_kpoint_mapping(self, K1, K2):
        """Get index of symmetry for mapping between K1 and K2"""
        s_s = self.s_s
        bz2bz_ks = self.kd.bz2bz_ks
        bzk2rbz_s = bz2bz_ks[K1][s_s]
        try:
            s = np.argwhere(bzk2rbz_s == K2)[0][0]
        except IndexError:
            self.context.print(f'K = {K1} cannot be mapped into '
                               f'K = {K2}')
            raise
        return s_s[s]

    def get_shift(self, K1, K2, U_cc, sign):
        """Get shift for mapping between K1 and K2."""
        kd = self.kd
        k1_c = kd.bzk_kc[K1]
        k2_c = kd.bzk_kc[K2]

        shift_c = np.dot(U_cc, k1_c) - k2_c * sign
        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)

        return shift_c
    
    @timer('apply symop')
    def apply_symop(self, s, a_MG, ignore_TR=False):
        if len(a_MG) == 0:
            return []
        G_G, sign, shift_c = self.G_sG[self.s_s[s]]
        U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(self.s_s[s])

        if ignore_TR:
            return a_MG[..., G_G]

        return TR(a_MG[..., G_G])

    @timer('map_G')
    def map_G(self, K1, K2, a_MG):
        """Map a function of G from K1 to K2. """
        if len(a_MG) == 0:
            return []

        if K1 == K2:
            return a_MG

        G_G, sign = self.map_G_vectors(K1, K2)

        s = self.get_kpoint_mapping(K1, K2)
        U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)

        return TR(a_MG[..., G_G])

    @timer('symmetrize_wGG')
    def symmetrize_wGG(self, A_wGG):
        """Symmetrize an array in GG'."""

        for A_GG in A_wGG:
            tmp_GG = np.zeros_like(A_GG, order='C')
            # tmp2_GG = np.zeros_like(A_GG)

            for s in self.s_s:
                G_G, sign, _ = self.G_sG[s]
                GG_shuffle(G_G, sign, A_GG, tmp_GG)

                # This is the exact operation that GG_shuffle does.
                # Uncomment lines involving tmp2_GG to test the
                # implementation in action:
                #
                # if sign == 1:
                #     tmp2_GG += A_GG[G_G, :][:, G_G]
                # if sign == -1:
                #     tmp2_GG += A_GG[G_G, :][:, G_G].T

            # assert np.allclose(tmp_GG, tmp2_GG)
            A_GG[:] = tmp_GG / self.how_many_symmetries()

    # Set up complex frequency alias
    symmetrize_zGG = symmetrize_wGG

    @timer('symmetrize_wxx')
    def symmetrize_wxx(self, A_wxx, optical_limit=False):
        """Symmetrize an array in xx'."""
        tmp_wxx = np.zeros_like(A_wxx)

        A_cv = self.qpd.gd.cell_cv
        iA_cv = self.qpd.gd.icell_cv

        if self.use_time_reversal:
            AT_wxx = np.transpose(A_wxx, (0, 2, 1))

        for s in self.s_s:
            G_G, sign, shift_c = self.G_sG[s]
            if optical_limit:
                G_G = np.array(G_G) + 2
                G_G = np.insert(G_G, 0, [0, 1])
                U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)
                M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)

            if sign == 1:
                tmp = A_wxx[:, G_G, :][:, :, G_G]
                if optical_limit:
                    tmp[:, 0:3, :] = np.transpose(np.dot(M_vv.T,
                                                         tmp[:, 0:3, :]),
                                                  (1, 0, 2))
                    tmp[:, :, 0:3] = np.dot(tmp[..., 0:3], M_vv)
                tmp_wxx += tmp
            elif sign == -1:
                tmp = AT_wxx[:, G_G, :][:, :, G_G]
                if optical_limit:
                    tmp[:, 0:3, :] = np.transpose(np.dot(M_vv.T,
                                                         tmp[:, 0:3, :]),
                                                  (1, 0, 2)) * sign
                    tmp[:, :, 0:3] = np.dot(tmp[:, :, 0:3], M_vv) * sign
                tmp_wxx += tmp

        # Inplace overwriting
        A_wxx[:] = tmp_wxx / self.how_many_symmetries()

    @timer('symmetrize_wxvG')
    def symmetrize_wxvG(self, A_wxvG):
        """Symmetrize chi0_wxvG"""
        A_cv = self.qpd.gd.cell_cv
        iA_cv = self.qpd.gd.icell_cv

        if self.use_time_reversal:
            # ::-1 corresponds to transpose in wing indices
            AT_wxvG = A_wxvG[:, ::-1]

        tmp_wxvG = np.zeros_like(A_wxvG)
        for s in self.s_s:
            G_G, sign, shift_c = self.G_sG[s]
            U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)
            M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
            if sign == 1:
                tmp = sign * np.dot(M_vv.T, A_wxvG[..., G_G])
            elif sign == -1:
                tmp = sign * np.dot(M_vv.T, AT_wxvG[..., G_G])
            tmp_wxvG += np.transpose(tmp, (1, 2, 0, 3))

        # Overwrite the input
        A_wxvG[:] = tmp_wxvG / self.how_many_symmetries()

    @timer('symmetrize_wvv')
    def symmetrize_wvv(self, A_wvv):
        """Symmetrize chi_wvv."""
        A_cv = self.qpd.gd.cell_cv
        iA_cv = self.qpd.gd.icell_cv
        tmp_wvv = np.zeros_like(A_wvv)
        if self.use_time_reversal:
            AT_wvv = np.transpose(A_wvv, (0, 2, 1))

        for s in self.s_s:
            G_G, sign, shift_c = self.G_sG[s]
            U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)
            M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
            if sign == 1:
                tmp = np.dot(np.dot(M_vv.T, A_wvv), M_vv)
            elif sign == -1:
                tmp = np.dot(np.dot(M_vv.T, AT_wvv), M_vv)
            tmp_wvv += np.transpose(tmp, (1, 0, 2))

        # Overwrite the input
        A_wvv[:] = tmp_wvv / self.how_many_symmetries()

    @timer('map_v')
    def map_v(self, K1, K2, a_Mv):
        """Map a function of v (cartesian component) from K1 to K2."""

        if len(a_Mv) == 0:
            return []

        if K1 == K2:
            return a_Mv

        A_cv = self.qpd.gd.cell_cv
        iA_cv = self.qpd.gd.icell_cv

        # Get symmetry
        s = self.get_kpoint_mapping(K1, K2)
        U_cc, sign, TR, _, ft_c = self.get_symmetry_operator(s)

        # Create cartesian operator
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
        return sign * np.dot(TR(a_Mv), M_vv)

    def timereversal(self, s):
        """Is this a time-reversal symmetry?"""
        tr = bool(s // self.nU)
        return tr

    def get_symmetry_operator(self, s):
        """Return symmetry operator s."""
        U_scc = self.kd.symmetry.op_scc
        ft_sc = self.kd.symmetry.ft_sc

        reds = s % self.nU
        if self.timereversal(s):
            TR = np.conj
            sign = -1
        else:
            sign = 1

            def TR(x):
                return x

        return U_scc[reds], sign, TR, self.shift_sc[s], ft_sc[reds]

    @timer('map_G_vectors')
    def map_G_vectors(self, K1, K2):
        """Return G vector mapping."""
        s = self.get_kpoint_mapping(K1, K2)
        G_G, sign, shift_c = self.G_sG[s]

        return G_G, sign

    @timer('Initialize_G_maps')
    def initialize_G_maps(self):
        """Calculate the Gvector mappings."""
        qpd = self.qpd
        B_cv = 2.0 * np.pi * qpd.gd.icell_cv
        G_Gv = qpd.get_reciprocal_vectors(add_q=False)
        G_Gc = np.dot(G_Gv, np.linalg.inv(B_cv))
        Q_G = qpd.Q_qG[0]

        G_sG = [None] * self.nsym
        UG_sGc = [None] * self.nsym
        Q_sG = [None] * self.nsym
        for s in self.s_s:
            U_cc, sign, TR, shift_c, ft_c = self.get_symmetry_operator(s)
            iU_cc = np.linalg.inv(U_cc).T
            UG_Gc = np.dot(G_Gc - shift_c, sign * iU_cc)

            assert np.allclose(UG_Gc.round(), UG_Gc)
            UQ_G = np.ravel_multi_index(UG_Gc.round().astype(int).T,
                                        qpd.gd.N_c, 'wrap')

            G_G = len(Q_G) * [None]
            for G, UQ in enumerate(UQ_G):
                try:
                    G_G[G] = np.argwhere(Q_G == UQ)[0][0]
                except IndexError:
                    print('This should not be possible but' +
                          'a G-vector was mapped outside the sphere')
                    raise IndexError
            UG_sGc[s] = UG_Gc
            Q_sG[s] = UQ_G
            G_sG[s] = [np.array(G_G, dtype=np.int32), sign, shift_c]
        self.G_Gc = G_Gc
        self.UG_sGc = UG_sGc
        self.Q_sG = Q_sG
        self.G_sG = G_sG

    def unfold_ibz_kpoint(self, ik):
        """Return kpoints related to irreducible kpoint."""
        kd = self.kd
        K_k = np.unique(kd.bz2bz_ks[kd.ibz2bz_k[ik]])
        K_k = K_k[K_k != -1]
        return K_k
