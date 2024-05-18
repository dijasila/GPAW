import numpy as np
from gpaw.response.df import DielectricFunction

df = DielectricFunction('gs_MoS2.gpw',
                        frequencies=[0.5],
                        txt='eps_GG.txt',
                        hilbert=False,
                        ecut=50,
                        nbands=50)
df_t = DielectricFunction('gs_MoS2.gpw',
                          frequencies=[0.5],
                          txt='eps_GG.txt',
                          hilbert=False,
                          ecut=50,
                          nbands=50,
                          truncation='2D')

for iq in range(22):
    eps = df.get_dielectric_matrix(q_c=[iq / 42, iq / 42, 0])

    # Periodic degrees of freedom
    Gvec_Gv = eps.qpd.get_reciprocal_vectors(add_q=False)
    z0 = eps.qpd.gd.cell_cv[2, 2] / 2  # Center of layer

    epsinv_GG = np.linalg.inv(eps.eps_wGG[0])
    eps_t = df_t.get_dielectric_matrix(q_c=[iq / 42, iq / 42, 0])
    epsinv_t_GG = np.linalg.inv(eps_t.eps_wGG[0])

    epsinv = 0.0
    epsinv_t = 0.0

    for iG in range(len(Gvec_Gv)):
        if np.allclose(Gvec_Gv[iG, :2], 0.0):
            Gz = Gvec_Gv[iG, 2]
            epsinv += np.exp(1.0j * Gz * z0) * epsinv_GG[iG, 0]
            epsinv_t += np.exp(1.0j * Gz * z0) * epsinv_t_GG[iG, 0]

    with open('2d_eps.dat', 'a') as f:
        print(iq, (1.0 / epsinv).real, (1.0 / epsinv_t).real, file=f)
