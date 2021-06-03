import numpy as np

from ase.parallel import world
from gpaw.fd_operators import Gradient

# TODO: This file is work in progress. Aim is to get the k-parallelisation
#       working.


# NOTE: This routine is not specific to Raman per se. Maybe it should go
#       somewhere else?
def get_dipole_transitions(atoms, calc, savetofile=True, realdipole=False):
    r"""
    Finds the dipole matrix elements:
    <\psi_n|\nabla|\psi_m> = <u_n|nabla|u_m> + ik<u_n|u_m>
    where psi_n = u_n(r)*exp(ikr).
    ik<u_n|u_m> is supposed to be zero off diagonal and not calculated as
    we are not intereste in diagonal terms.

    NOTE: Function name seems to be a misnomer. The routine is supposed to
    calculate <psi_n|p|psi_m>, which is not quite the normal dipole moment.
    Use realdipole=True to return proper dipole.

    Input:
        atoms           Relevant ASE atoms object
        calc            GPAW calculator object.
    Output:
        dip_svknm.npy    Array with dipole matrix elements
    """
    assert calc.wfs.bd.comm.size == 1
    assert calc.wfs.mode == 'lcao'
    nbands = calc.wfs.bd.nbands
    nspins = calc.wfs.nspins
    nk = calc.wfs.kd.nibzkpts
    gd = calc.wfs.gd
    dtype = calc.wfs.dtype

    # Why?
    calc.wfs.set_positions
    calc.initialize_positions(atoms)

    nabla_v = [Gradient(gd, v, 1.0, 2, dtype).apply for v in range(3)]
    dip_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=dtype)

    for kpt in calc.wfs.kpt_u:
        dipe_vnm = np.zeros((3, nbands, nbands), dtype=dtype)
        dipa_vnm = np.zeros((3, nbands, nbands), dtype=dtype)
        wf = []
        for n in range(nbands):
            C_M = kpt.C_nM[n]
            psit_G = gd.zeros(dtype=dtype)
            calc.wfs.basis_functions.lcao_to_grid(C_M, psit_G, kpt.q)
            if dtype == complex:
                k_c = calc.wfs.kd.ibzk_kc[kpt.k]
                psit_G *= gd.plane_wave(-k_c)
            wf.append(psit_G)
        wf = np.array(wf)

        grad_nv = gd.zeros((nbands, 3), dtype=dtype)

        # Calculate <phit|nabla|phit> for the pseudo wavefunction
        # Parellisation note: Every rank has same result???
        for v in range(3):
            for n in range(nbands):
                # NOTE: It's unclear to me, whether or nor to use phase_cd
                # nabla_v[v](wf[n], grad_nv[n, v],  np.ones((3, 2)))
                nabla_v[v](wf[n], grad_nv[n, v], kpt.phase_cd)
                dipe_vnm[v] = gd.integrate(wf, grad_nv[:, v],
                                           global_integral=False)
                # dipe_vnm[v] = gd.integrate(wf.conj() * grad_nv[:, v])

        # augmentation part
        # Parallelisatin note: Need to sum???
        for a, P_ni in kpt.P_ani.items():
            nabla_iiv = calc.wfs.setups[a].nabla_iiv
            dipa_vnm += np.einsum('ni,ijv,mj->vnm',
                                  P_ni.conj(), nabla_iiv, P_ni)
        # dip_vnm = dipe_vnm
        # dip_vnm = dipa_vnm
        dip_vnm = dipe_vnm + dipa_vnm
        if realdipole:  # need this for testing against other dipole routines
            deltaE = abs(kpt.eps_n[:, None] - kpt.eps_n[None, :])
            np.fill_diagonal(deltaE, np.inf)
            dip_vnm[:] /= deltaE

        dip_skvnm[kpt.s, kpt.k] = dip_vnm

    calc.wfs.world.sum(dip_skvnm)
    if world.rank == 0 and savetofile:
        np.save('dip_skvnm.npy', dip_skvnm)
    return dip_skvnm
