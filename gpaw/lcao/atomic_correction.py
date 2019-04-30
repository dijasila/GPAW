import numpy as np

from gpaw.utilities.blas import gemm, mmm
from gpaw.utilities import unpack
from gpaw.utilities.partition import EvenPartitioning


def get_atomic_correction(name):
    cls = dict(dense=DenseAtomicCorrection,
               sparse=SparseAtomicCorrection)[name]
    return cls()


class BaseAtomicCorrection:
    name = 'base'
    description = 'base class for atomic corrections with LCAO'

    def calculate_hamiltonian(self, wfs, kpt, dH_asp, H_MM, yy):
        avalues = self.get_a_values()

        dH_aii = dH_asp.partition.arraydict([setup.dO_ii.shape
                                             for setup in wfs.setups],
                                            dtype=wfs.dtype)

        for a in avalues:
            dH_aii[a][:] = yy * unpack(dH_asp[a][kpt.s])

        self.calculate(wfs, kpt.q, dH_aii, H_MM)

    def add_overlap_correction(self, wfs, S_qMM):
        avalues = self.get_a_values()
        dS_aii = [wfs.setups[a].dO_ii for a in avalues]
        dS_aii = dict(zip(avalues, dS_aii))  # XXX get rid of dict

        for a in dS_aii:
            dS_aii[a] = np.asarray(dS_aii[a], wfs.dtype)

        for q, S_MM in enumerate(S_qMM):
            self.calculate(wfs, q, dS_aii, S_MM)

    def gobble_data(self, wfs):
        pass  # Prepare internal data structures for calculate().

    def calculate(self, wfs, q, dX_aii, X_MM):
        raise NotImplementedError

    def get_a_values(self):
        raise NotImplementedError

    def calculate_projections(self, wfs, kpt):
        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            mmm(1.0, kpt.C_nM, 'N', wfs.P_aqMi[a][kpt.q], 'N', 0.0, P_ni)


class DenseAtomicCorrection(BaseAtomicCorrection):
    name = 'dense'
    description = 'dense with blas'

    def gobble_data(self, wfs):
        self.initialize(wfs.P_aqMi, wfs.ksl.Mstart, wfs.ksl.Mstop)
        self.orig_partition = wfs.atom_partition
        assert set(self.orig_partition.my_indices) == set(wfs.P_aqMi)

    def initialize(self, P_aqMi, Mstart, Mstop):
        self.P_aqMi = P_aqMi
        self.Mstart = Mstart
        self.Mstop = Mstop

    def get_a_values(self):
        return self.P_aqMi.keys()

    def calculate(self, wfs, q, dX_aii, X_MM):
        dtype = X_MM.dtype

        # P_aqMi is distributed over domains (a) and bands (M).
        # Hence the correction X_MM = sum(P dX P) includes contributions
        # only from local atoms; the result must be summed over gd.comm
        # to get all 'a' contributions, and it will be locally calculated
        # only on the local slice of bands.

        for a, dX_ii in dX_aii.items():
            P_Mi = self.P_aqMi[a][q]
            assert dtype == P_Mi.dtype
            dXP_iM = np.zeros((dX_ii.shape[1], P_Mi.shape[0]), dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dX_ii, 0.0, dXP_iM, 'c')
            gemm(1.0, dXP_iM, P_Mi[self.Mstart:self.Mstop], 1.0, X_MM)


class SparseAtomicCorrection(BaseAtomicCorrection):
    name = 'sparse'
    description = 'sparse using scipy'

    def __init__(self, tolerance=1e-12):
        BaseAtomicCorrection.__init__(self)
        from scipy import sparse
        self.sparse = sparse
        # We currently don't use tolerance although we could speed things
        # up that way.
        #
        # Tolerance is for zeroing elements very close to zero, which
        # often increases sparsity somewhat, even for very small values.
        self.tolerance = tolerance

    def get_a_values(self):
        return self.orig_partition.my_indices  # XXXXXXXXXX

    def gobble_data(self, wfs):
        self.orig_partition = wfs.atom_partition
        evenpart = EvenPartitioning(self.orig_partition.comm,
                                    self.orig_partition.natoms)
        self.even_partition = evenpart.as_atom_partition()
        nq = len(wfs.kd.ibzk_qc)

        I_a = [0]
        I_a.extend(np.cumsum([setup.ni for setup in wfs.setups[:-1]]))
        nI = I_a[-1] + wfs.setups[-1].ni
        self.nI = nI
        self.I_a = I_a

        self.Psparse_qIM = wfs.P_qIM
        return

        M_a = wfs.setups.M_a
        M = M_a[-1] + wfs.setups[-1].nao
        nao_a = [setup.nao for setup in wfs.setups]
        ni_a = [setup.ni for setup in wfs.setups]

        Psparse_qIM = [self.sparse.lil_matrix((I, M), dtype=wfs.dtype)
                       for _ in range(nq)]

        for (a3, a1), P_qim in wfs.P_aaqim.items():
            if self.tolerance > 0:
                P_qim = P_qim.copy()
                approximately_zero = np.abs(P_qim) < self.tolerance
                P_qim[approximately_zero] = 0.0
            for q in range(nq):
                Psparse_qIM[q][I_a[a3]:I_a[a3] + ni_a[a3],
                               M_a[a1]:M_a[a1] + nao_a[a1]] = P_qim[q]

        self.Psparse_qIM = [x.tocsr() for x in Psparse_qIM]

    def calculate(self, wfs, q, dX_aii, X_MM):
        Mstart = wfs.ksl.Mstart
        Mstop = wfs.ksl.Mstop

        Psparse_IM = self.Psparse_qIM[q]

        dXsparse_II = self.sparse.lil_matrix((self.nI, self.nI),
                                             dtype=wfs.dtype)
        avalues = sorted(dX_aii.keys())
        for a in avalues:
            I1 = self.I_a[a]
            I2 = I1 + wfs.setups[a].ni
            dXsparse_II[I1:I2, I1:I2] = dX_aii[a]
        dXsparse_II = dXsparse_II.tocsr()

        Psparse_MI = Psparse_IM[:, Mstart:Mstop].transpose().conjugate()
        Xsparse_MM = Psparse_MI.dot(dXsparse_II.dot(Psparse_IM))
        X_MM[:, :] += Xsparse_MM.todense()

    def calculate_projections(self, wfs, kpt):
        P_In = self.Psparse_qIM[kpt.q].dot(kpt.C_nM.T.conj())
        for a in self.orig_partition.my_indices:
            I1 = self.I_a[a]
            I2 = I1 + wfs.setups[a].ni
            kpt.P_ani[a][:, :] = P_In[I1:I2, :].T.conj()
