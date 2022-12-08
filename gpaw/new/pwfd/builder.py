from types import SimpleNamespace

import numpy as np

from gpaw.band_descriptor import BandDescriptor
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.calculation import DFTState
from gpaw.new.ibzwfs import create_ibz_wave_functions as create_ibzwfs
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.potential import Potential
from gpaw.new.pwfd.davidson import Davidson
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class PWFDDFTComponentsBuilder(DFTComponentsBuilder):
    def create_eigensolver(self, hamiltonian):
        eigsolv_params = self.params.eigensolver.copy()
        name = eigsolv_params.pop('name', 'dav')
        assert name == 'dav'
        return Davidson(
            self.nbands,
            self.wf_desc,
            self.communicators['b'],
            hamiltonian.create_preconditioner,
            converge_bands=self.params.convergence.get('bands', 'occupied'),
            **eigsolv_params)

    def read_ibz_wave_functions(self, reader):
        kpt_comm, band_comm, domain_comm = (self.communicators[x]
                                            for x in 'kbd')

        def create_wfs(spin: int, q: int, k: int, kpt_c, weight: float):
            psit_nG = SimpleNamespace(
                comm=domain_comm,
                dims=(self.nbands,),
                desc=self.wf_desc.new(kpt=kpt_c),
                data=None,
                xp=np)
            wfs = PWFDWaveFunctions(
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                psit_nX=psit_nG,  # type: ignore
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents)

            return wfs

        ibzwfs = create_ibzwfs(self.ibz,
                               self.nelectrons,
                               self.ncomponents,
                               create_wfs,
                               self.communicators['k'])

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)

        return ibzwfs

    def create_ibz_wave_functions(self, basis, potential):
        from gpaw.new.lcao.builder import create_lcao_ibzwfs

        if self.params.random:
            self.log('Initializing wave functions with random numbers')
            raise NotImplementedError

        sl_default = self.params.parallel['sl_default']
        sl_lcao = self.params.parallel['sl_lcao'] or sl_default

        if 0:
            return initialize_from_lcao_old(
                self.setups,
                self.communicators,
                self.nbands,
                self.ncomponents,
                self.nelectrons,
                self.fracpos_ac,
                self.atomdist,
                self.dtype,
                self.grid,
                self.wf_desc,
                self.ibz,
                sl_lcao,
                basis,
                potential,
                self.convert_wave_functions_from_uniform_grid)

        lcao_ibzwfs, _ = create_lcao_ibzwfs(
            basis, potential,
            self.ibz, self.communicators, self.setups,
            self.fracpos_ac, self.grid, self.dtype,
            self.nbands, self.ncomponents, self.atomdist, self.nelectrons)

        state = DFTState(lcao_ibzwfs, None, potential)
        hamiltonian = LCAOHamiltonian(basis)
        LCAOEigensolver(basis).iterate(state, hamiltonian)

        def create_wfs(spin, q, k, kpt_c, weight):
            lcaowfs = lcao_ibzwfs.wfs_qs[q][spin]
            assert lcaowfs.spin == spin

            # Convert to PW-coefs in PW-mode:
            psit_nX = self.convert_wave_functions_from_uniform_grid(
                lcaowfs.C_nM, basis, kpt_c, q)
            eig_n = lcaowfs._eig_n

            nao = lcaowfs.C_nM.shape[1]
            if nao < self.nbands:
                psit_nX[nao:].randomize()
                eig_n[nao:] = np.inf

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents)
            wfs._eig_n = eig_n
            return wfs

        return create_ibzwfs(self.ibz, self.nelectrons, self.ncomponents,
                             create_wfs, self.communicators['k'])


def initialize_from_lcao_old(setups,
                             communicators,
                             nbands,
                             ncomponents,
                             nelectrons,
                             fracpos_ac,
                             atomdist,
                             dtype,
                             grid,
                             wf_desc,
                             ibz,
                             sl_lcao,
                             basis_set,
                             potential: Potential,
                             convert_wfs):
    from gpaw.utilities import pack2
    from gpaw.utilities.partition import AtomPartition
    from gpaw.utilities.timing import nulltimer

    (band_comm, domain_comm, kpt_comm, world,
     domainband_comm, kptband_comm) = (communicators[x] for x in 'bdkwKD')

    lcaonbands = min(nbands, setups.nao)
    gd = grid._gd
    nspins = ncomponents % 3

    lcaobd = BandDescriptor(lcaonbands, band_comm)
    lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                   gd, lcaobd, domainband_comm,
                                   dtype, nao=setups.nao)

    atom_partition = AtomPartition(domain_comm, atomdist.rank_a)

    lcaowfs = LCAOWaveFunctions(lcaoksl, gd, nelectrons,
                                setups, lcaobd, dtype,
                                world, basis_set.kd, kptband_comm,
                                nulltimer)
    lcaowfs.basis_functions = basis_set
    lcaowfs.set_positions(fracpos_ac, atom_partition)

    if ncomponents != 4:
        eigensolver = DirectLCAO()
    else:
        from gpaw.xc.noncollinear import NonCollinearLCAOEigensolver
        eigensolver = NonCollinearLCAOEigensolver()

    eigensolver.initialize(gd, dtype, setups.nao, lcaoksl)

    dH_asp = setups.empty_atomic_matrix(ncomponents,
                                        atom_partition,
                                        dtype=dtype)
    for a, dH_sii in potential.dH_asii.items():
        dH_asp[a][:] = [pack2(dH_ii) for dH_ii in dH_sii]
    ham = SimpleNamespace(vt_sG=potential.vt_sR.data,
                          dH_asp=dH_asp)
    eigensolver.iterate(ham, lcaowfs)

    def create_wfs(spin, q, k, kpt_c, weight):
        u = spin + nspins * q
        lcaokpt = lcaowfs.kpt_u[u]
        assert lcaokpt.s == spin
        mynbands = len(lcaokpt.C_nM)
        assert mynbands == lcaonbands

        # Convert to PW-coefs in PW-mode:
        psit_nX = convert_wfs(lcaokpt.C_nM, basis_set, kpt_c, q)

        if lcaonbands < nbands:
            psit_nX[lcaonbands:].randomize()

        return PWFDWaveFunctions(psit_nX=psit_nX,
                                 spin=spin,
                                 q=q,
                                 k=k,
                                 weight=weight,
                                 setups=setups,
                                 fracpos_ac=fracpos_ac,
                                 atomdist=atomdist,
                                 ncomponents=ncomponents)

    return create_ibzwfs(ibz, nelectrons, ncomponents, create_wfs, kpt_comm)
