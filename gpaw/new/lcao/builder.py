from functools import partial
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

        self.tciexpansions = TCIExpansions.new_from_setups(self.setups)
        # basis.set_matrix_distribution(self.ksl.Mstart, self.ksl.Mstop)
        manytci = self.tciexpansions.get_manytci_calculator(
            self.setups, self.grid._gd, self.fracpos_ac,
            kpt_qc, self.dtype, NullTimer())

        my_atom_indices = basis.my_atom_indices
        S_qMM, T_qMM = manytci.O_qMM_T_qMM(domain_comm,
                                           0, self.setups.nao,
                                           False)
        P_aqMi = manytci.P_aqMi(my_atom_indices)
        P_qaMi = [{a: P_aqMi[a][q] for a in my_atom_indices}
                  for q in range(len(S_qMM))]

        for a, setup in enumerate(self.setups):
            for P_Mi, S_MM in zip(P_aqMi[a], S_qMM):
                S_MM += P_Mi @ setup.dO_ii @ P_Mi.T.conj()

        # self.atomic_correction= self.atomic_correction_cls.new_from_wfs(self)
        # self.atomic_correction.add_overlap_correction(newS_qMM)

        def create_wfs(spin, q, k, kpt_c, weight):
            C_nM = Matrix(self.nbands, self.setups.nao, self.dtype,
                          dist=(band_comm, band_comm.size, 1))
            return LCAOWaveFunctions(
                setups=self.setups,
                density_adder=partial(basis.construct_density, q=q),
                C_nM=C_nM,
                S_MM=S_qMM[q],
                T_MM=T_qMM[q],
                P_aMi=P_qaMi[q],
                kpt_c=kpt_c,
                domain_comm=domain_comm,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                ncomponents=self.ncomponents)

        ibzwfs = IBZWaveFunctions(ibz,
                                  self.nelectrons,
                                  self.ncomponents,
                                  create_wfs,
                                  kpt_comm)
        return ibzwfs
