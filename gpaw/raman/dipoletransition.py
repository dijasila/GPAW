import numpy as np

from ase.parallel import world, parprint
from gpaw.fd_operators import Gradient


# NOTE: This routine is not specific to Raman per se. Maybe it should go
#       somewhere else?
def get_dipole_transitions(atoms, calc, savetofile=True):
    r"""
    Finds the dipole matrix elements:
    <\psi_n|\nabla|\psi_m> = <u_n|nabla|u_m> + ik<u_n|u_m>
    where psi_n = u_n(r)*exp(ikr).

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

    # Why?
    calc.wfs.set_positions
    calc.initialize_positions(atoms)

    parprint("Distributing wavefunctions.")

    dip_svknm = []
    # XXX Should be changed so the information is distributed directly.
    for s in range(nspins):
        wfs = {}
        for k in range(nk):
            # Collects the wavefunctions and the projections to rank 0.
            gwfa = calc.wfs.get_wave_function_array
            wf = np.array([gwfa(i, k, s, realspace=True, periodic=True
                                ) for i in range(n)], dtype=complex)
            P_nI = calc.wfs.collect_projections(k, s)

        # Distributes the information to rank k % size.
        if world.rank == 0:
            if k % world.size == world.rank:
                wfs[k] = wf, P_nI
            else:
                world.comm.Send(P_nI, dest=k % world.size, tag=nk + k)
                world.comm.Send(wf, dest=k % world.size, tag=k)
        else:
            if k % world.size == world.rank:
                nproj = sum(setup.ni for setup in calc.wfs.setups)
                if not calc.wfs.collinear:
                    nproj *= 2
                P_nI = np.empty((calc.wfs.bd.nbands, nproj), calc.wfs.dtype)
                shape = () if calc.wfs.collinear else(2,)
                wf = np.tile(calc.wfs.empty(shape, global_array=True,
                                            realspace=True), (n, 1, 1, 1))

                world.comm.Recv(P_nI, source=0, tag=nk + k)
                world.comm.Recv(wf, source=0, tag=k)

                wfs[k] = wf, P_nI

        parprint("Evaluating dipole transition matrix elements.")

        dip_vknm = np.zeros((3, nk, n, n), dtype=complex)
        overlap_knm = np.zeros((nk, n, n), dtype=complex)

        nabla_v = [Gradient(calc.wfs.gd, v, 1.0, 4, complex
                            ).apply for v in range(3)]
        phases = np.ones((3, 2), dtype=complex)
        grad_nv = calc.wfs.gd.zeros((n, 3), complex)

        for k, (wf, P_nI) in wfs.items():
            # Calculate <phit|nabla|phit> for the pseudo wavefunction
            for v in range(3):
                for i in range(n):
                    nabla_v[v](wf[i], grad_nv[i, v], phases)

            dip_vknm[:, k] = np.transpose(calc.wfs.gd.integrate(wf, grad_nv),
                                          (2, 0, 1))

            overlap_knm[k] = [calc.wfs.gd.integrate(wf[i], wf
                                                    ) for i in range(n)]
            k_v = 2 * np.pi * np.dot(calc.wfs.kd.ibzk_kc[k],
                                     calc.wfs.gd.icell_cv)
            dip_vknm[:, k] += (1j * k_v[:, None, None] *
                               overlap_knm[None, k, :, :])

            # The PAW corrections are added - see
            # https://wiki.fysik.dtu.dk/gpaw/documentation/tddft/
            # dielectric_response.html
            I1 = 0
            # np.einsum is slow but very memory efficient.
            for a, setup in enumerate(calc.wfs.setups):
                I2 = I1 + setup.ni
                P_ni = P_nI[:, I1:I2]
                dip_vknm[:, k, :, :] += np.einsum('ni,ijv,mj->vnm',
                                                  P_ni.conj(), setup.nabla_iiv,
                                                  P_ni)
                I1 = I2

        world.sum(dip_vknm)
        dip_svknm.append(dip_vknm)

    if world.rank == 0 and savetofile:
        np.save('dip_svknm.npy', np.array(dip_svknm))
    return np.array(dip_svknm)
