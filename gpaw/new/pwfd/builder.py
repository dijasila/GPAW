from types import SimpleNamespace

import numpy as np
from gpaw.band_descriptor import BandDescriptor
from gpaw.core.atom_arrays import AtomArrays, AtomArraysLayout
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.pwfd.davidson import Davidson
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.utilities import pack2
from gpaw.utilities.partition import AtomPartition
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class PWFDDFTComponentsBuilder(DFTComponentsBuilder):
    def create_eigensolver(self, hamiltonian):
        eigsolv_params = self.params.eigensolver
        name = eigsolv_params.pop('name', 'dav')
        assert name == 'dav'
        return Davidson(self.nbands,
                        self.wf_desc,
                        self.communicators['b'],
                        hamiltonian.create_preconditioner,
                        **eigsolv_params)

    def read_ibz_wave_functions(self, reader):
        kpt_comm, band_comm, domain_comm = (self.communicators[x]
                                            for x in 'kbd')

        def create_wfs(spin: int, q: int, k: int, kpt_c, weight: float):
            psit_nG = SimpleNamespace(
                comm=domain_comm,
                dims=(self.nbands,),
                desc=SimpleNamespace(comm=domain_comm,
                                     kpt_c=kpt_c,
                                     dtype=self.dtype))
            wfs = PWFDWaveFunctions(
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                psit_nX=psit_nG,
                setups=self.setups,
                fracpos_ac=self.fracpos_ac,
                ncomponents=self.ncomponents)

            return wfs

        ibzwfs = IBZWaveFunctions(self.ibz,
                                  self.nelectrons,
                                  self.ncomponents,
                                  create_wfs,
                                  self.communicators['k'])

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)

        return ibzwfs

    def create_ibz_wave_functions(self, basis_set, potential):
        if self.params.random:
            self.log('Initializing wave functions with random numbers')
            raise NotImplementedError

        sl_default = self.params.parallel['sl_default']
        sl_lcao = self.params.parallel['sl_lcao'] or sl_default
        return initialize_from_lcao(
            self.setups,
            self.communicators,
            self.nbands,
            self.ncomponents,
            self.nelectrons,
            self.fracpos_ac,
            self.dtype,
            self.grid,
            self.wf_desc,
            self.ibz,
            sl_lcao,
            basis_set,
            potential,
            self.convert_wave_functions_from_uniform_grid)


def initialize_from_lcao(setups,
                         communicators,
                         nbands,
                         ncomponents,
                         nelectrons,
                         fracpos_ac,
                         dtype,
                         grid,
                         wf_desc,
                         ibz,
                         sl_lcao,
                         basis_set,
                         potential: Potential,
                         convert_wfs) -> None:
    (band_comm, domain_comm, kpt_comm, world,
     domainband_comm, kptband_comm) = (communicators[x] for x in 'bdkwKD')

    lcaonbands = min(nbands, setups.nao)
    gd = grid._gd
    nspins = ncomponents % 3

    lcaobd = BandDescriptor(lcaonbands, band_comm)
    lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                   gd, lcaobd, domainband_comm,
                                   dtype, nao=setups.nao)

    atom_partition = AtomPartition(
        domain_comm,
        np.array([sphere.rank for sphere in basis_set.sphere_a]))

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

    dH_asp = setups.empty_atomic_matrix(ncomponents, atom_partition,
                                        dtype=dtype)
    for a, dH_sii in potential.dH_asii.items():
        dH_asp[a][:] = [pack2(dH_ii) for dH_ii in dH_sii]
    ham = SimpleNamespace(vt_sG=potential.vt_sR.data,
                          dH_asp=dH_asp)
    eigensolver.iterate(ham, lcaowfs)

    def create_wfs(spin, q, k, kpt_c, weight):
        gridk = grid.new(kpt=kpt_c, dtype=dtype)
        u = spin + nspins * q
        lcaokpt = lcaowfs.kpt_u[u]
        assert lcaokpt.s == spin
        psit_nR = gridk.zeros(nbands, band_comm)
        mynbands = len(lcaokpt.C_nM)
        assert mynbands == lcaonbands
        basis_set.lcao_to_grid(lcaokpt.C_nM,
                               psit_nR.data[:mynbands], lcaokpt.q)

        if lcaonbands < nbands:
            psit_nR[lcaonbands:].randomize()

        # Convert to PW-coefs in PW-mode:
        psit_nX = convert_wfs(psit_nR)

        return PWFDWaveFunctions(psit_nX=psit_nX,
                                 spin=spin,
                                 q=q,
                                 k=k,
                                 weight=weight,
                                 setups=setups,
                                 fracpos_ac=fracpos_ac,
                                 ncomponents=ncomponents)

    return IBZWaveFunctions(ibz, nelectrons, ncomponents, create_wfs, kpt_comm)
