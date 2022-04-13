from functools import partial

import numpy as np
from gpaw.core.matrix import Matrix
from gpaw.lcao.tci import TCIExpansions
from gpaw.new.fd.builder import FDDFTComponentsBuilder
from gpaw.new.ibzwfs import create_ibz_wave_functions as create_ibzwfs
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.lcao.hybrids import HybridLCAOEigensolver, HybridXCFunctional
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.utilities.timing import NullTimer


class LCAODFTComponentsBuilder(FDDFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 distribution=None):
        super().__init__(atoms, params)
        self.distribution = distribution
        self.basis = None

    def create_wf_description(self):
        raise NotImplementedError

    def create_xc_functional(self):
        if self.params.xc['name'] in ['HSE06', 'PBE0', 'EXX']:
            return HybridXCFunctional(self.params.xc)
        return super().create_xc_functional()

    def create_basis_set(self):
        self.basis = FDDFTComponentsBuilder.create_basis_set(self)
        return self.basis

    def create_hamiltonian_operator(self):
        return LCAOHamiltonian(self.basis)

    def create_eigensolver(self, hamiltonian):
        if self.params.xc['name'] in ['HSE06', 'PBE0', 'EXX']:
            return HybridLCAOEigensolver(self.basis,
                                         self.fracpos_ac,
                                         self.grid.cell_cv)
        return LCAOEigensolver(self.basis)

    def create_ibz_wave_functions(self, basis, potential, coefficients=None):
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
        if self.dtype == complex:
            np.negative(S_qMM.imag, S_qMM.imag)
            np.negative(T_qMM.imag, T_qMM.imag)

        P_aqMi = manytci.P_aqMi(my_atom_indices)
        P_qaMi = [{a: P_aqMi[a][q] for a in my_atom_indices}
                  for q in range(len(S_qMM))]

        for a, P_qMi in P_aqMi.items():
            dO_ii = self.setups[a].dO_ii
            for P_Mi, S_MM in zip(P_qMi, S_qMM):
                S_MM += P_Mi.conj() @ dO_ii @ P_Mi.T
        domain_comm.sum(S_qMM)

        # self.atomic_correction= self.atomic_correction_cls.new_from_wfs(self)
        # self.atomic_correction.add_overlap_correction(newS_qMM)

        def create_wfs(spin, q, k, kpt_c, weight):
            nao = self.setups.nao
            C_nM = Matrix(self.nbands, nao, self.dtype,
                          dist=(band_comm, band_comm.size, 1))
            if coefficients is not None:
                C_nM.data[:] = coefficients.proxy(spin, k)
            return LCAOWaveFunctions(
                setups=self.setups,
                density_adder=partial(basis.construct_density, q=q),
                C_nM=C_nM,
                S_MM=Matrix(nao, nao, data=S_qMM[q],
                            dist=(band_comm, band_comm.size, 1)),
                T_MM=T_qMM[q],
                P_aMi=P_qaMi[q],
                kpt_c=kpt_c,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                domain_comm=domain_comm,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                ncomponents=self.ncomponents)

        ibzwfs = create_ibzwfs(ibz,
                               self.nelectrons,
                               self.ncomponents,
                               create_wfs,
                               kpt_comm)
        return ibzwfs

    def read_ibz_wave_functions(self, reader):
        c = 1
        if reader.version >= 0 and reader.version < 4:
            c = reader.bohr**1.5

        basis = self.create_basis_set()
        potential = self.create_potential_calculator()
        if 'coefficients' in reader.wave_functions:
            coefficients = reader.wave_functions.proxy('coefficients')
            coefficients.scale = c
        else:
            coefficients = None

        ibzwfs = self.create_ibz_wave_functions(basis, potential, coefficients)

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)
        return ibzwfs
