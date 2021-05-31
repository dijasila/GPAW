import numpy as np

from ase.parallel import world, parprint
from gpaw.fd_operators import Gradient


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

    Input:
        atoms           Relevant ASE atoms object
        calc            GPAW calculator object.
    Output:
        dip_svknm.npy    Array with dipole matrix elements
    """
    assert calc.wfs.bd.comm.size == 1
    bzk_kc = calc.get_ibz_k_points()
    n = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]
    gd = calc.wfs.gd

    # Why?
    calc.wfs.set_positions
    calc.initialize_positions(atoms)

    nabla_v = [Gradient(gd, v, 1.0, 2, calc.wfs.dtype).apply for v in range(3)]

    dip_svknm = []
    for s in range(calc.wfs.nspins):
        dip_vknm = np.zeros((3, nk, n, n), dtype=complex)
        dip1_vknm = np.zeros((3, nk, n, n), dtype=complex)
        # dip2_vknm = np.zeros((3, nk, n, n), dtype=complex)
        dip3_vknm = np.zeros((3, nk, n, n), dtype=complex)



        for k in range(nk):
            parprint("Distributing wavefunctions.")
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

            parprint("Evaluating dipole transition matrix elements.")
            grad_nv = gd.zeros((n, 3), dtype=calc.wfs.dtype)

            # Calculate <phit|nabla|phit> for the pseudo wavefunction
            # Parellisation note: Every rank has same result
            for v in range(3):
                for i in range(n):
                    nabla_v[v](wf[i], grad_nv[i, v], kpt.phase_cd)
                dip1_vknm[v, k] = gd.integrate(wf, grad_nv[:, v])
                # dip1_vknm[v, k] = gd.integrate(wf.conj() * grad_nv[:, v])

            # augmentation part
            # Parallelisatin note: Need to sum
            for a, P_ni in kpt.P_ani.items():
                nabla_iiv = calc.wfs.setups[a].nabla_iiv
                dip3_vknm[:, k, :, :] += np.einsum('ni,ijv,mj->vnm',
                                                   P_ni.conj(), nabla_iiv,
                                                   P_ni)
        gd.comm.sum(dip3_vknm)
        dip_vknm = dip1_vknm + dip3_vknm

        if realdipole:  # need this for testing against other dipole routines
            for k in range(nk):
                kpt = calc.wfs.kpt_qs[k][s]
                deltaE = abs(kpt.eps_n[:, None] - kpt.eps_n[None, :])
                dip_vknm[:, k] /= (deltaE + 1j * 1e-8)

        dip_svknm.append(dip_vknm)

    if world.rank == 0 and savetofile:
        np.save('dip_svknm.npy', np.array(dip_svknm))
    return np.array(dip_svknm)
