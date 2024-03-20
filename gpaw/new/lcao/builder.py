import numpy as np
from gpaw.core.matrix import Matrix
from gpaw.lcao.tci import TCIExpansions
from gpaw.new import zips
from gpaw.new.fd.builder import FDDFTComponentsBuilder
from gpaw.new.lcao.ibzwfs import LCAOIBZWaveFunction
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.forces import TCIDerivatives
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.lcao.hybrids import HybridLCAOEigensolver, HybridXCFunctional
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.setup import Setups
from gpaw.utilities.timing import NullTimer

from typing import List, Dict
from numpy.typing import NDArray


class LCAODFTComponentsBuilder(FDDFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 *,
                 comm,
                 distribution=None):
        super().__init__(atoms, params, comm=comm)
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

        ibzwfs = self.create_ibz_wave_functions(basis, potential,
                                                coefficients=coefficients)

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)
        return ibzwfs

    def create_ibz_wave_functions(self,
                                  basis,
                                  potential,
                                  *,
                                  log=None,
                                  coefficients=None):
        ibzwfs, _ = create_lcao_ibzwfs(
            basis,
            self.ibz, self.communicators, self.setups,
            self.fracpos_ac, self.grid, self.dtype,
            self.nbands, self.ncomponents, self.atomdist, self.nelectrons,
            coefficients)
        return ibzwfs


def create_lcao_ibzwfs(basis,
                       ibz, communicators, setups,
                       fracpos_ac, grid, dtype,
                       nbands, ncomponents, atomdist, nelectrons,
                       coefficients=None):
    kpt_band_comm = communicators['D']
    kpt_comm = communicators['k']
    band_comm = communicators['b']
    domain_comm = communicators['d']

    S_qMM, T_qMM, P_qaMi, tciexpansions, tci_derivatives = tci_helper(
        basis, ibz, domain_comm, band_comm, kpt_comm,
        fracpos_ac, atomdist,
        grid, dtype, setups)

    nao = setups.nao

    def create_wfs(spin, q, k, kpt_c, weight):
        C_nM = Matrix(nbands, 2 * nao if ncomponents == 4 else nao,
                      dtype,
                      dist=(band_comm, band_comm.size, 1))
        if coefficients is not None:
            C_nM.data[:] = coefficients.proxy(spin, k)
        return LCAOWaveFunctions(
            setups=setups,
            tci_derivatives=tci_derivatives,
            basis=basis,
            C_nM=C_nM,
            S_MM=S_qMM[q],
            T_MM=T_qMM[q],
            P_aMi=P_qaMi[q],
            kpt_c=kpt_c,
            fracpos_ac=fracpos_ac,
            atomdist=atomdist,
            domain_comm=domain_comm,
            spin=spin,
            q=q,
            k=k,
            weight=weight,
            ncomponents=ncomponents)

    ibzwfs = LCAOIBZWaveFunction.create(
        ibz=ibz,
        nelectrons=nelectrons,
        ncomponents=ncomponents,
        create_wfs_func=create_wfs,
        kpt_comm=kpt_comm,
        kpt_band_comm=kpt_band_comm,
        comm=communicators['w'])
    ibzwfs.grid = grid  # The TCI-stuff needs cell and pbc from somwhere ...
    return ibzwfs, tciexpansions


def tci_helper(basis,
               ibz,
               domain_comm, band_comm, kpt_comm,
               fracpos_ac, atomdist,
               grid,
               dtype,
               setups):
    rank_k = ibz.ranks(kpt_comm)
    here_k = rank_k == kpt_comm.rank
    kpt_qc = ibz.kpt_kc[here_k]

    tciexpansions = TCIExpansions.new_from_setups(setups)
    manytci = tciexpansions.get_manytci_calculator(
        setups, grid._gd, fracpos_ac,
        kpt_qc, dtype, NullTimer())

    my_atom_indices = basis.my_atom_indices
    M1 = basis.Mstart
    M2 = basis.Mstop
    S0_qMM, T0_qMM = manytci.O_qMM_T_qMM(domain_comm, M1, M2, True)
    if dtype == complex:
        np.negative(S0_qMM.imag, S0_qMM.imag)
        np.negative(T0_qMM.imag, T0_qMM.imag)

    P_aqMi = manytci.P_aqMi(my_atom_indices)
    P_qaMi = [{a: P_aqMi[a][q] for a in my_atom_indices}
              for q in range(len(S0_qMM))]

    add_atomic_overlap_corrections(
        P_qaMi=P_qaMi,
        S0_qMM=S0_qMM,
        setups=setups)

    domain_comm.sum(S0_qMM)

    # self.atomic_correction= self.atomic_correction_cls.new_from_wfs(self)
    # self.atomic_correction.add_overlap_correction(newS_qMM)

    nao = setups.nao

    S_qMM = [Matrix(nao, nao, data=S_MM,
                    dist=(band_comm, band_comm.size, 1)) for S_MM in S0_qMM]
    T_qMM = [Matrix(nao, nao, data=T_MM,
                    dist=(band_comm, band_comm.size, 1)) for T_MM in T0_qMM]

    for S_MM in S_qMM:
        S_MM.tril2full()
    for T_MM in T_qMM:
        T_MM.tril2full()

    tci_derivatives = TCIDerivatives(manytci, atomdist, nao)

    return S_qMM, T_qMM, P_qaMi, tciexpansions, tci_derivatives


def add_atomic_overlap_corrections(
        P_qaMi: List[Dict[int, NDArray]],
        S0_qMM: List[NDArray],
        setups: Setups,
        sparse: bool = True):
    if sparse:
        from scipy import sparse
        for P_aMi, S_MM in zips(P_qaMi, S0_qMM):
            dO_II = sparse.block_diag(
                [setups[a].dO_ii for a in P_aMi.keys()],
                format='csr')
            P_MI = sparse.hstack(
                [sparse.coo_matrix(P_Mi) for P_Mi in P_aMi.values()],
                format='csr')
            S_MM += P_MI.conj().dot(dO_II.dot(P_MI.T)).todense()
    else:
        from scipy import linalg
        for P_aMi, S_MM in zips(P_qaMi, S0_qMM):
            dO_II = linalg.block_diag(
                *[setups[a].dO_ii for a in P_aMi.keys()])
            P_MI = np.hstack(
                [P_Mi for P_Mi in P_aMi.values()])
            S_MM += P_MI.conj() @ dO_II @ P_MI.T
