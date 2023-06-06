import numpy as np
from gpaw.mpi import world
from gpaw.spinorbit import get_L_vlmm

L_vlii = get_L_vlmm()


def get_om_from_calc(calc):
    if not calc.density.ncomponents == 4:
        raise AssertionError('Collinear calculations require spin-orbit' +
                             'coupling for nonzero orbital magnetization.')

    om_av = np.zeros([len(calc.atoms), 3])

    for wfs in calc.wfs.kpt_u:
        f_n = wfs.f_n
        for (a, P_nsi), setup in zip(wfs.P_ani.items(), calc.setups):
            for v in range(3):
                L_lii = L_vlii[v]
                Ni = 0
                for l in setup.l_j:
                    Nl = 2 * l + 1

                    om_av[a, v] += np.einsum('nsi,nsj,n,ij->',
                                             P_nsi[:, :, Ni:Ni+Nl].conj(),
                                             P_nsi[:, :, Ni:Ni+Nl],
                                             f_n, L_lii[l]).real

                    #A0_nn = P_nsi[:, 0, Ni:Ni+Nl].conj().dot(L_ii).dot(P_nsi[:, 0, Ni:Ni+Nl].T)
                    #A1_nn = P_nsi[:, 1, Ni:Ni+Nl].conj().dot(L_ii).dot(P_nsi[:, 1, Ni:Ni+Nl].T)

                    #om_av[a, v] += np.dot(f_n, np.diag(A0_nn) + np.diag(A1_nn)).real

                    Ni += Nl

    world.sum(om_av)

    return om_av


def get_om_from_soc_eigs(soc):
    
    l_aj = soc.l_aj
    om_av = np.zeros([len(l_aj), 3])
    
    for wfs, weight in zip(soc.wfs.values(), soc.weights()):
        f_n = wfs.f_m * weight
        for a, l_j in l_aj.items():
            P_nsi = wfs.projections[a]
            for v in range(3):
                L_lii = L_vlii[v]
                Ni = 0
                for l in l_j:
                    Nl = 2 * l + 1

                    om_av[a, v] += np.einsum('nsi,nsj,n,ij->',
                                             P_nsi[:, :, Ni:Ni+Nl].conj(),
                                             P_nsi[:, :, Ni:Ni+Nl],
                                             f_n, L_lii[l]).real

                    Ni += Nl

    world.sum(om_av)

    return om_av
