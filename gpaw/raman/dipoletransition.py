import numpy as np

from ase.parallel import world


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

    # print(calc.wfs.kd.comm.size, calc.wfs.gd.comm.size,
    # calc.wfs.bd.comm.size)

    # Why?
    calc.wfs.set_positions
    calc.initialize_positions(atoms)

    dip_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=dtype)

    ksl = calc.wfs.ksl
    dThetadR_qvMM, dTdR_qvMM = calc.wfs.manytci.O_qMM_T_qMM(gd.comm,
                                                            ksl.Mstart,
                                                            ksl.Mstop,
                                                            False,
                                                            derivative=True)

    dipe_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=dtype)
    dipa_skvnm = np.zeros((nspins, nk, 3, nbands, nbands), dtype=dtype)

    for kpt in calc.wfs.kpt_u:
        if realdipole:  # need this for testing against other dipole routines
            deltaE = abs(kpt.eps_n[:, None] - kpt.eps_n[None, :])
            np.fill_diagonal(deltaE, np.inf)
        else:
            deltaE = 1.

        C_nM = kpt.C_nM
        for v in range(3):
            dThetadRv_MM = dThetadR_qvMM[kpt.q, v]
            nabla_nn = -(C_nM.conj() @ dThetadRv_MM.conj() @ C_nM.T)
            gd.comm.sum(nabla_nn)
            dipe_skvnm[kpt.s, kpt.k, v] = nabla_nn / deltaE

        # augmentation part
        # Parallelisatin note: Need to sum???
        dipa_vnm = np.zeros((3, nbands, nbands), dtype=dtype)
        for a, P_ni in kpt.P_ani.items():
            nabla_iiv = calc.wfs.setups[a].nabla_iiv
            dipa_vnm += np.einsum('ni,ijv,mj->vnm',
                                  P_ni.conj(), nabla_iiv, P_ni)
        gd.comm.sum(dipa_vnm)
        dipa_skvnm[kpt.s, kpt.k] = dipa_vnm / deltaE

    dip_skvnm = dipe_skvnm + dipa_skvnm
    calc.wfs.kd.comm.sum(dip_skvnm)

    # calc.wfs.world.sum(dip_skvnm)
    if world.rank == 0 and savetofile:
        np.save('dip_skvnm.npy', dip_skvnm)
    return dip_skvnm
