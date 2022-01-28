from gpaw.core.matrix import Matrix
from gpaw.new.fd.builder import FDDFTComponentsBuilder
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.lcao.tci import TCIExpansions
from gpaw.utilities.timing import NullTimer


class LCAODFTComponentsBuilder(FDDFTComponentsBuilder):
    def __init__(self, atoms, params):
        super().__init__(atoms, params)
        self.basis = None

    def create_wf_description(self):
        raise NotImplementedError

    def create_basis_set(self):
        self.basis = FDDFTComponentsBuilder.create_basis_set(self)
        return self.basis

    def create_hamiltonian_operator(self):
        return LCAOHamiltonian(self.basis)

    def create_eigensolver(self, hamiltonian):
        return LCAOEigensolver(self.basis)

    def create_ibz_wave_functions(self, basis, potential):
        assert self.communicators['w'].size == 1

        ibz = self.ibz
        kpt_comm = self.communicators['k']
        band_comm = self.communicators['b']
        domain_comm = self.communicators['d']

        rank_k = ibz.ranks(kpt_comm)
        here_k = rank_k == kpt_comm.rank
        kpt_qc = ibz.kpt_kc[here_k]

        nspins = self.ncomponents % 3

        tciexpansions = TCIExpansions.new_from_setups(self.setups)
        # basis.set_matrix_distribution(self.ksl.Mstart, self.ksl.Mstop)
        manytci = tciexpansions.get_manytci_calculator(
            self.setups, self.grid._gd, self.fracpos_ac,
            kpt_qc, self.dtype, NullTimer())

        my_atom_indices = basis.my_atom_indices
        S_qMM, T_qMM = manytci.O_qMM_T_qMM(domain_comm,
                                           0, self.setups.nao,
                                           False)
        P_qIM = manytci.P_qIM(my_atom_indices)
        P_aqMi = manytci.P_aqMi(my_atom_indices)
        P_qaMi = [{a: P_aqMi[a][q] for a in my_atom_indices}
                  for q in range(len(S_qMM))]

        for a, setup in enumerate(self.setups):
            for P_Mi, S_MM in zip(P_aqMi[a], S_qMM):
                S_MM += P_Mi.conj() @ setup.dO_ii @ P_Mi.T

        # self.atomic_correction= self.atomic_correction_cls.new_from_wfs(self)
        # self.atomic_correction.add_overlap_correction(newS_qMM)

        wfs_qs = []
        for kpt_c, weight, S_MM, T_MM, P_aMi in zip(kpt_qc,
                                                    ibz.weight_k[here_k],
                                                    S_qMM,
                                                    T_qMM,
                                                    P_qaMi):
            wfs_s = []
            for s in range(nspins):
                C_nM = Matrix(self.nbands, self.setups.nao, self.dtype,
                              dist=(band_comm, band_comm.size, 1))
                wfs = LCAOWaveFunctions(kpt_c,
                                        C_nM,
                                        S_MM,
                                        T_MM,
                                        P_aMi,
                                        domain_comm,
                                        s,
                                        self.setups,
                                        self.fracpos_ac,
                                        weight,
                                        spin_degeneracy=2 // nspins)
                wfs_s.append(wfs)
            wfs_qs.append(wfs_s)

        ibzwfs = IBZWaveFunctions(ibz, rank_k, kpt_comm, wfs_qs,
                                  self.nelectrons,
                                  2 // nspins)
        return ibzwfs
