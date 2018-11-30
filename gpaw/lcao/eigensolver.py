import numpy as np
from ase.units import Hartree
from gpaw.utilities.blas import mmm


class DirectLCAO(object):
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self, diagonalizer=None, error=np.inf):
        self.diagonalizer = diagonalizer
        # ??? why should we be able to set
        # this diagonalizer in both constructor and initialize?
        self.has_initialized = False  # XXX
        self._error = error

    def initialize(self, gd, dtype, nao, diagonalizer=None):
        self.gd = gd
        self.nao = nao
        if diagonalizer is not None:
            self.diagonalizer = diagonalizer
        assert self.diagonalizer is not None
        self.has_initialized = True  # XXX

    def reset(self):
        self._error = np.inf
        pass

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, e):
        self._error = e

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt, Vt_xMM=None,
                                     root=-1, add_kinetic=True):
        # XXX document parallel stuff, particularly root parameter
        assert self.has_initialized

        bfs = wfs.basis_functions

        # distributed_atomic_correction works with ScaLAPACK/BLACS in general.
        # If SL is not enabled, it will not work with band parallelization.
        # But no one would want that for a practical calculation anyway.
        # dH_asp = wfs.atomic_correction.redistribute(wfs, hamiltonian.dH_asp)
        # XXXXX fix atomic corrections
        dH_asp = hamiltonian.dH_asp

        if Vt_xMM is None:
            wfs.timer.start('Potential matrix')
            vt_G = hamiltonian.vt_sG[kpt.s]
            Vt_xMM = bfs.calculate_potential_matrices(vt_G)
            wfs.timer.stop('Potential matrix')

        if bfs.gamma and wfs.dtype == float:
            yy = 1.0
            H_MM = Vt_xMM[0]
        else:
            wfs.timer.start('Sum over cells')
            yy = 0.5
            k_c = wfs.kd.ibzk_qc[kpt.q]
            H_MM = (0.5 + 0.0j) * Vt_xMM[0]
            for sdisp_c, Vt_MM in zip(bfs.sdisp_xc[1:], Vt_xMM[1:]):
                H_MM += np.exp(2j * np.pi * np.dot(sdisp_c, k_c)) * Vt_MM
            wfs.timer.stop('Sum over cells')

        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        name = wfs.atomic_correction.__class__.__name__
        wfs.timer.start(name)
        wfs.atomic_correction.calculate_hamiltonian(wfs, kpt, dH_asp, H_MM, yy)
        wfs.timer.stop(name)

        wfs.timer.start('Distribute overlap matrix')
        H_MM = wfs.ksl.distribute_overlap_matrix(
            H_MM, root, add_hermitian_conjugate=(yy == 0.5))
        wfs.timer.stop('Distribute overlap matrix')

        if add_kinetic:
            H_MM += wfs.T_qMM[kpt.q]
        return H_MM

    def iterate(self, hamiltonian, wfs):
        wfs.timer.start('LCAO eigensolver')

        self._error = 0.0
        s = -1
        for kpt in wfs.kpt_u:
            if kpt.s != s:
                s = kpt.s
                wfs.timer.start('Potential matrix')
                Vt_xMM = wfs.basis_functions.calculate_potential_matrices(
                    hamiltonian.vt_sG[s])
                wfs.timer.stop('Potential matrix')
            error = self.iterate_one_k_point(hamiltonian, wfs, kpt, Vt_xMM)
            self._error += error

        self._error = wfs.kd.comm.sum(self._error)

        wfs.timer.stop('LCAO eigensolver')

    def iterate_one_k_point(self, hamiltonian, wfs, kpt, Vt_xMM):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_MM = self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt, Vt_xMM,
                                                 root=0)
        S_MM = wfs.S_qMM[kpt.q]

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(wfs.bd.mynbands)

        error = self.calculate_residual(kpt, H_MM, S_MM, wfs)

        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        self.diagonalizer.diagonalize(H_MM, kpt.C_nM, kpt.eps_n, S_MM)
        wfs.timer.stop(diagonalization_string)

        wfs.timer.start('Calculate projections')
        # P_ani are not strictly necessary as required quantities can be
        # evaluated directly using P_aMi/Paaqim.  We should perhaps get rid
        # of the places in the LCAO code using P_ani directly
        wfs.atomic_correction.calculate_projections(wfs, kpt)
        wfs.timer.stop('Calculate projections')

        return error

    def __repr__(self):
        # The "diagonalizer" must be equal to the Kohn-Sham layout
        # object presently.  That information will be printed in the
        # text output anyway so we do not need it here.
        #
        # Although maybe it may be better to print it anyway...
        return 'LCAO using direct dense diagonalizer'

    def estimate_memory(self, mem, dtype):
        pass
        # self.diagonalizer.estimate_memory(mem, dtype) #XXX enable this

    def calculate_residual_2(self, ham, wfs):

        wfs.timer.start('Residual')
        norm = []
        for kpt in wfs.kpt_u:
            nbs = 0
            if kpt.f_n is None:
                norm.append(np.inf)
                continue
            for f in kpt.f_n:
                if f > 1.0e-10:
                    nbs += 1

            C_nM = kpt.C_nM[:nbs]
            # calculate and make it matrix hermitian

            H_MM = self.calculate_hamiltonian_matrix(ham, wfs,
                                                     kpt)
            ind_l = np.tril_indices(H_MM.shape[0], -1)
            H_MM[(ind_l[1], ind_l[0])] = H_MM[ind_l].conj()

            HC_Mn = np.zeros_like(H_MM)
            mmm(1.0, H_MM.conj(), 'n', C_nM, 't', 0.0, HC_Mn)

            L = np.zeros(shape=(nbs, nbs),
                         dtype=H_MM.dtype)

            if H_MM.dtype == complex:
                mmm(1.0, C_nM.conj(), 'n', HC_Mn, 'n', 0.0, L)
            else:
                mmm(1.0, C_nM, 'n', HC_Mn, 'n', 0.0, L)
            del HC_Mn

            # gradients:
            S = wfs.S_qMM[kpt.q]
            g = C_nM @ H_MM - L.conj() @ (C_nM @ S)

            for i in range(nbs):
                norm.append(np.dot(g[i].conj(), g[i]).real * kpt.f_n[i])

        error = sum(norm) * Hartree**2 / wfs.nvalence
        self._error = wfs.kd.comm.sum(error)
        wfs.timer.stop('Residual')

    def calculate_residual(self, kpt, H_MM, S_MM, wfs):

        if kpt.C_nM is None or kpt.f_n is None:
            return np.inf
        wfs.timer.start('Residual')
        nbs = 0
        for f in kpt.f_n:
            if f > 1.0e-10:
                nbs += 1

        C_nM = kpt.C_nM[:nbs]

        ind_l = np.tril_indices(H_MM.shape[0], -1)
        H_MM[(ind_l[1], ind_l[0])] = H_MM[ind_l].conj()

        HC_Mn = np.zeros(shape=(C_nM.shape[1], C_nM.shape[0]),
                         dtype=C_nM.dtype)

        mmm(1.0, H_MM.conj(), 'N', C_nM, 'T', 0.0, HC_Mn)
        L = np.zeros(shape=(nbs, nbs), dtype=H_MM.dtype)
        mmm(1.0, C_nM.conj(), 'N', HC_Mn, 'N', 0.0, L)

        rhs = np.zeros(shape=(C_nM.shape[1], nbs),
                       dtype=C_nM.dtype)
        rhs2 = np.zeros(shape=(C_nM.shape[1], nbs),
                        dtype=C_nM.dtype)
        mmm(1.0, S_MM.conj(), 'N', C_nM[:nbs], 'T', 0.0, rhs)
        mmm(1.0, rhs, 'N', L, 'N', 0.0, rhs2)
        HC_Mn = HC_Mn[:, :nbs] - rhs2[:, :nbs]
        norm = []
        for i in range(nbs):
            norm.append(np.dot(HC_Mn[:,i].conj(),
                               HC_Mn[:,i]).real * kpt.f_n[i])

        wfs.timer.stop('Residual')

        return sum(norm) * Hartree ** 2 / wfs.nvalence
