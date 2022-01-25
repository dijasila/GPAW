from gpaw.new.fd.builder import FDDFTComponentsBuilder
from gpaw.core.atom_centered_functions import UniformGridAtomCenteredFunctions
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.lcao.eigensolver import LCAOEigensolver


class LCAODFTComponentsBuilder(FDDFTComponentsBuilder):
    def __init__(self, atoms, params):
        super().__init__(atoms, params)

    def create_wf_description(self):
        raise NotImplementedError

    def create_hamiltonian_operator(self):
        return DummyHamiltonian()

    def create_eigensolver(self, hamiltonian):
        return LCAOEigensolver()

    def create_ibz_wave_functions(self, basis_set, potential):
        ibz = self.ibz
        kpt_comm = self.communicators['k']
        band_comm = self.communicators['b']

        rank_k = ibz.ranks(kpt_comm)

        basis_functions_a = [setup.phit_j for setup in self.setups]

        nspins = self.ncomponents % 3

        wfs_qs = []
        for kpt_c, weight, rank in zip(ibz.kpt_kc, ibz.weight_k, rank_k):
            if rank != kpt_comm.rank:
                continue
            gridk = self.grid.new(kpt=kpt_c, dtype=self.dtype)
            wfs_s = []
            for s in range(nspins):
                basis = UniformGridAtomCenteredFunctions(
                    basis_functions_a,
                    self.fracpos_ac,
                    gridk,
                    cut=True)
                wfs_s.append(LCAOWaveFunctions(basis,
                                               self.nbands,
                                               band_comm,
                                               s,
                                               self.setups,
                                               self.fracpos_ac,
                                               weight,
                                               spin_degeneracy=2 // nspins))
            wfs_qs.append(wfs_s)

        ibzwfs = IBZWaveFunctions(ibz, rank_k, kpt_comm, wfs_qs,
                                  self.nelectrons,
                                  2 // nspins)
        return ibzwfs


class DummyHamiltonian:
    apply = None
