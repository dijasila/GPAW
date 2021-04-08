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

    NOTE: This seems to be a misnomer. The routine is supposed to calculate
    <psi_n|p|psi_m>, which is not quite the normal dipole moment

    Input:
        atoms           Relevant ASE atoms object
        calc            GPAW calculator object.
    Output:
        dip_svknm.npy    Array with dipole matrix elements
    """
    bzk_kc = calc.get_ibz_k_points()
    n = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]
    nspins = calc.wfs.nspins
    gd = calc.wfs.gd

    # Why?
    calc.wfs.set_positions
    calc.initialize_positions(atoms)

    # NOTE: This currently will give wrong resulst in parallel

    dip_svknm = []
    for s in range(nspins):
        dip_vknm = np.zeros((3, nk, n, n), dtype=complex)
        nabla_v = [Gradient(gd, v, 1.0, 2, calc.wfs.dtype
                            ).apply for v in range(3)]

        for k in range(nk):
            parprint("Distributing wavefunctions.")
            # Collects the wavefunctions and the projections to rank 0.
            gwfa = calc.wfs.get_wave_function_array
            wf = []
            for i in range(n):
                wfi = gwfa(i, k, s, realspace=True, periodic=True)
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
            for v in range(3):
                for i in range(n):
                    nabla_v[v](wf[i], grad_nv[i, v], kpt.phase_cd)
                dip_vknm[v, k] = gd.integrate(wf, grad_nv[:, v])
                # dip_vknm[v, k] = gd.integrate(wf.conj() * grad_nv[:, v])

            # NOTE: this part is not tested in parallel. may or may not work.
            # not even sure this term is needed
            overlap_nm = np.zeros((n, n), dtype=calc.wfs.dtype)
            overlap_nm[:, :] = [gd.integrate(wf[i], wf) for i in range(n)]
            k_v = 2 * np.pi * np.dot(calc.wfs.kd.bzk_kc[k], gd.icell_cv)
            for v in range(3):
                dip_vknm[v, k] += 1j * k_v[v] * overlap_nm

            # augmentation part (tested in parallel)
            for a, P_ni in kpt.P_ani.items():
                nabla_iiv = calc.wfs.setups[a].nabla_iiv
                dip_vknm[:, k, :, :] += np.einsum('ni,ijv,mj->vnm',
                                                  P_ni.conj(), nabla_iiv,
                                                  P_ni)

        gd.comm.sum(dip_vknm)
        if realdipole:  # need this for testing
            for k in range(nk):
                kpt = calc.wfs.kpt_qs[k][s]
                deltaE = abs(kpt.eps_n[:, None] - kpt.eps_n[None, :])
                print(deltaE)
                dip_vknm[:, k] /= deltaE

        dip_svknm.append(dip_vknm)

    if world.rank == 0 and savetofile:
        np.save('dip_svknm.npy', np.array(dip_svknm))
    return np.array(dip_svknm)
