import numpy as np

from ase.parallel import world  # , parprint
from gpaw.fd_operators import Gradient


# NOTE: This routine is not specific to Raman per se. Maybe it should go
#       somewhere else?
def get_dipole_transitions(atoms, calc, savetofile=True, realdipole=False):
    r"""
    Finds the dipole matrix elements:
    <\psi_n|\nabla|\psi_m> = <u_n|nabla|u_m> + ik<u_n|u_m>
    where psi_n = u_n(r)*exp(ikr).
    ik<u_n|u_m> is supposed to be zero off diagonal and not calculated as
    we are not interested in diagonal terms.

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
    assert calc.wfs.kd.comm.size == 1
    assert calc.wfs.mode == 'lcao'
    n = calc.wfs.bd.nbands
    nk = calc.wfs.kd.nibzkpts
    gd = calc.wfs.gd

    # Why?
    calc.wfs.set_positions
    calc.initialize_positions(atoms)

    nabla_v = [Gradient(gd, v, 1.0, 2, calc.wfs.dtype).apply for v in range(3)]

    dip_skvnm = []
    for s in range(calc.wfs.nspins):
        dipe_kvnm = np.zeros((nk, 3, n, n), dtype=complex)
        dipa_kvnm = np.zeros((nk, 3, n, n), dtype=complex)

        for k in range(nk):
            # parprint("Distributing wavefunctions.")
            # Collects the wavefunctions and the projections to rank 0.
            gwfa = calc.wfs.get_wave_function_array
            wf = []
            for i in range(n):
                wfi = gwfa(n=i, k=k, s=s, realspace=True, periodic=True)
                if calc.wfs.world.rank != 0:
                    wfi = gd.empty(dtype=calc.wfs.dtype, global_array=True)
                wfi = np.ascontiguousarray(wfi)
                calc.wfs.world.broadcast(wfi, 0)
                wfd = gd.empty(dtype=calc.wfs.dtype, global_array=False)
                wfd = gd.distribute(wfi)
                wf.append(wfd)
            wf = np.array(wf)
            kpt = calc.wfs.kpt_qs[k][s]

            # parprint("Evaluating dipole transition matrix elements.")
            grad_nv = gd.zeros((n, 3), dtype=calc.wfs.dtype)

            # Calculate <phit|nabla|phit> for the pseudo wavefunction
            # Parellisation note: Every rank has same result

            for v in range(3):
                for i in range(n):
                    # NOTE: It's unclear to me, whether or nor to use phase_cd
                    # nabla_v[v](wf[i], grad_nv[i, v], np.ones((3, 2)))
                    nabla_v[v](wf[i], grad_nv[i, v], kpt.phase_cd)
                dipe_kvnm[k, v] = gd.integrate(wf, grad_nv[:, v])

            # augmentation part
            # Parallelisatin note: Need to sum
            for a, P_ni in kpt.P_ani.items():
                nabla_iiv = calc.wfs.setups[a].nabla_iiv
                dipa_kvnm[k] += np.einsum('ni,ijv,mj->vnm',
                                          P_ni.conj(), nabla_iiv, P_ni)
        gd.comm.sum(dipa_kvnm)

        dip_kvnm = dipe_kvnm + dipa_kvnm
        # dip_kvnm = dipe_kvnm
        # dip_kvnm = dipa_kvnm

        if realdipole:  # need this for testing against other dipole routines
            for k in range(nk):
                kpt = calc.wfs.kpt_qs[k][s]
                deltaE = abs(kpt.eps_n[:, None] - kpt.eps_n[None, :])
                np.fill_diagonal(deltaE, np.inf)
                dip_kvnm[k, :] /= deltaE

        dip_skvnm.append(dip_kvnm)

    if world.rank == 0 and savetofile:
        np.save('dip_skvnm.npy', np.array(dip_skvnm))
    return np.array(dip_skvnm)
