import numpy as np
from gpaw.utilities import pack2
from gpaw.band_descriptor import BandDescriptor
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.new.wave_functions import IBZWaveFunctions, WaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions
from types import SimpleNamespace
from gpaw.utilities.timing import nulltimer
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.new.potential import Potential
from gpaw.utilities.partition import AtomPartition
from gpaw.core import PlaneWaves


def create_lcao_ibz_wave_functions(setups,
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
                                   potential: Potential) -> IBZWaveFunctions:
    band_comm = communicators['b']
    domain_comm = communicators['d']
    domainband_comm = communicators['K']
    kptband_comm = communicators['D']
    kpt_comm = communicators['k']
    world = communicators['w']
    lcaonbands = min(nbands, setups.nao)
    gd = grid._gd
    ibz = ibz

    lcaobd = BandDescriptor(lcaonbands, band_comm)
    lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                   gd, lcaobd, domainband_comm,
                                   dtype, nao=setups.nao)

    kd = KPointDescriptor(ibz.bz.points, ncomponents % 3)
    kd.set_symmetry(SimpleNamespace(pbc=grid.pbc),
                    ibz.symmetry.symmetry,
                    comm=world)

    atom_partition = AtomPartition(
        domain_comm,
        np.array([sphere.rank for sphere in basis_set.sphere_a]))

    lcaowfs = LCAOWaveFunctions(lcaoksl, gd, nelectrons,
                                setups, lcaobd, dtype,
                                world, kd, kptband_comm,
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

    assert len(ibz) == 1
    ranks = [0]

    u = 0
    mykpts = []
    for kpt, weight, rank in zip(ibz.points, ibz.weights, ranks):
        if rank != kpt_comm.rank:
            continue
        gridk = grid.new(kpt=kpt)  # dtype?
        lcaokpt = lcaowfs.kpt_u[u]
        assert (ibz.bz.points[lcaokpt.k] == kpt).all()
        psit_nR = gridk.zeros(nbands, band_comm)
        mynbands = len(lcaokpt.C_nM)
        basis_set.lcao_to_grid(lcaokpt.C_nM,
                               psit_nR.data[:mynbands], lcaokpt.q)
        if isinstance(wf_desc, PlaneWaves):
            pw = wf_desc.new(kpt=kpt)
            psit_nG = pw.empty(nbands, band_comm)
            for psit_R, psit_G in zip(psit_nR, psit_nG):
                psit_R.fft(out=psit_G)
            psit_nX = psit_nG
        else:
            psit_nX = psit_nR
        mykpts.append(WaveFunctions(psit_nX, lcaokpt.s, setups,
                                    fracpos_ac))
        assert mynbands == nbands

    ibzwfs = IBZWaveFunctions(ibz, ranks, kpt_comm, mykpts, nelectrons)
    return ibzwfs
